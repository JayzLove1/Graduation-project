# -*- coding: utf-8 -*-
# PPO (Proximal Policy Optimization) 强化学习迷宫 AI 后端
# 架构：Actor-Critic（On-Policy），组件：RolloutBuffer / ActorCritic / PPO
# IPC：Windows 共享内存 MazeRLSharedMemory（4096 bytes）
import mmap
import time
import os
import csv
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
except ImportError:
    print("[Python AI] ERROR: PyTorch 未安装，请运行 'pip install torch numpy'")
    import sys; sys.exit(1)

# ========== 共享内存 IPC ==========
shared_memory_name = "MazeRLSharedMemory"
data_size = 4096
try:
    mm = mmap.mmap(-1, data_size, tagname=shared_memory_name)
except Exception as e:
    print(f"[Python AI] ERROR: 无法打开共享内存 {shared_memory_name}: {e}")
    import sys; sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Python AI] 使用计算设备: {device.type.upper()}")

# ========== RolloutBuffer ==========
class RolloutBuffer:
    # On-Policy 缓冲区：每次 update() 后必须清空，不能跨策略复用历史经验
    def __init__(self):
        self.actions      = []
        self.states       = []
        self.logprobs     = []  # 采样时的对数概率，用于重要性采样比率
        self.rewards      = []
        self.is_terminals = []
        self.masks        = []  # 合法动作掩码，update 时须与采样时保持一致

    def clear(self):
        for lst in (self.actions, self.states, self.logprobs,
                    self.rewards, self.is_terminals, self.masks):
            lst.clear()

# ========== Actor-Critic 网络 ==========
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        # 128→64 小网络：38 维紧凑状态不需要大容量，小网络收敛更快且不易过拟合
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128), nn.Tanh(),
            nn.Linear(128, 64),        nn.Tanh(),
            nn.Linear(64, action_dim), nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 128), nn.Tanh(),
            nn.Linear(128, 64),        nn.Tanh(),
            nn.Linear(64, 1)
        )
        # Orthogonal init（PPO 标准做法）：PyTorch 默认 Kaiming uniform 为 ReLU 设计，
        # 对 Tanh 激活产生方向偏差，15×15 迷宫中 5% 偏差即可让随机游走陷入永久死循环。
        # 隐藏层 gain=√2，actor 输出层 gain=0.01（初始接近均匀分布），critic 输出层 gain=1.0
        sqrt2 = float(np.sqrt(2.0))
        for layer in list(self.actor) + list(self.critic):
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=sqrt2)
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(list(self.actor.children())[-2].weight, gain=0.01)
        nn.init.constant_(list(self.actor.children())[-2].bias, 0.0)
        nn.init.orthogonal_(list(self.critic.children())[-1].weight, gain=1.0)
        nn.init.constant_(list(self.critic.children())[-1].bias, 0.0)

    def act(self, state, mask=None):
        action_probs = self.actor(state)
        if mask is not None:
            action_probs = action_probs * mask
            action_probs = action_probs / (action_probs.sum(dim=-1, keepdim=True) + 1e-8)
        dist   = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def evaluate(self, state, action, mask=None):
        # mask 须与采样时相同，否则重要性采样比率 ratio 会失真
        action_probs = self.actor(state)
        if mask is not None:
            action_probs = action_probs * mask
            action_probs = action_probs / (action_probs.sum(dim=-1, keepdim=True) + 1e-8)
        dist = Categorical(action_probs)
        return dist.log_prob(action), torch.squeeze(self.critic(state)), dist.entropy()

# ========== PPO ==========
class PPO:
    def __init__(self, input_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.gamma       = gamma
        self.eps_clip    = eps_clip
        self.K_epochs    = K_epochs
        self.gae_lambda  = 0.95    # GAE λ：偏差与方差的折中
        # 初始 entropy_coeff=0.10：0.05 会导致策略过早收敛到走廊死角，无法首次通关
        self.entropy_coeff = 0.10
        # KL 提前停止：单 batch 内策略偏差超阈值时跳出 K_epochs，防止过拟合当前样本
        self.target_kl   = 0.015
        self.policy     = ActorCritic(input_dim, action_dim).to(device)
        self.policy_old = ActorCritic(input_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(),  'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        self.MseLoss = nn.MSELoss()

    def select_action(self, state_vec, buffer, mask_vec):
        with torch.no_grad():
            state = torch.FloatTensor(state_vec).to(device)
            mask  = torch.FloatTensor(mask_vec).to(device)
            action, logprob = self.policy_old.act(state, mask)
        buffer.states.append(state_vec)
        buffer.actions.append(action)
        buffer.logprobs.append(logprob)
        buffer.masks.append(mask_vec)
        return action

    def update(self, buffer, next_state_vec=None):
        if len(buffer.states) == 0:
            return
        # 容错对齐：各字段长度不一致时截断至最短，保证批次维度一致
        min_len = min(len(buffer.states), len(buffer.actions), len(buffer.logprobs),
                      len(buffer.rewards), len(buffer.is_terminals), len(buffer.masks))
        if min_len != len(buffer.states):
            print(f"[PPO Warning] Buffer mismatch: states={len(buffer.states)}, rewards={len(buffer.rewards)}")
            buffer.states       = buffer.states[:min_len]
            buffer.actions      = buffer.actions[:min_len]
            buffer.logprobs     = buffer.logprobs[:min_len]
            buffer.rewards      = buffer.rewards[:min_len]
            buffer.is_terminals = buffer.is_terminals[:min_len]
            buffer.masks        = buffer.masks[:min_len]
        if min_len == 0:
            buffer.clear(); return

        old_states   = torch.FloatTensor(np.array(buffer.states)).to(device)
        old_actions  = torch.LongTensor(buffer.actions).to(device)
        old_logprobs = torch.FloatTensor(buffer.logprobs).to(device)
        old_masks    = torch.FloatTensor(np.array(buffer.masks)).to(device)

        # ---------- GAE(λ) 广义优势估计 ----------
        with torch.no_grad():
            values = self.policy.critic(old_states).squeeze()
            if values.dim() == 0:
                values = values.unsqueeze(0)

        advantages = torch.zeros(min_len, dtype=torch.float32).to(device)
        returns    = torch.zeros(min_len, dtype=torch.float32).to(device)
        # 非终止截断时 bootstrap：避免将截断误判为 episode 结束导致 Critic 低估
        bootstrap_value = 0.0
        if not buffer.is_terminals[-1] and next_state_vec is not None:
            with torch.no_grad():
                bootstrap_value = self.policy.critic(
                    torch.FloatTensor(next_state_vec).to(device)).item()

        gae = 0.0
        for t in reversed(range(min_len)):
            next_val = bootstrap_value if t == min_len - 1 else values[t + 1].item()
            if buffer.is_terminals[t]:
                next_val = 0.0; gae = 0.0
            delta         = buffer.rewards[t] + self.gamma * next_val - values[t].item()
            gae           = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
            returns[t]    = gae + values[t]

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # Returns 归一化：旧实现 returns 范围 ±400~600，MSE ~10000 压制 actor loss；
        # 归一化后 critic/actor 量级平衡，0.5 系数才有意义
        returns_std        = returns.std().item() + 1e-7 if len(returns) > 1 else 1.0
        returns_normalized = returns / returns_std

        # ---------- PPO 截断目标函数（K_epochs 次回放） ----------
        for k in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions, old_masks)
            with torch.no_grad():
                approx_kl = (old_logprobs.detach() - logprobs).mean().item()
            if approx_kl > self.target_kl and k > 0:
                break
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1  = ratios * advantages
            surr2  = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            critic_loss = self.MseLoss(state_values.squeeze() / returns_std, returns_normalized)
            loss = (-torch.min(surr1, surr2).mean()
                    + 0.5 * critic_loss
                    - self.entropy_coeff * dist_entropy.mean())
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        buffer.clear()

# ========== 超参数 ==========
# update_timestep=256：步数上限 1350 步，每局约 5 次中段更新，梯度传播更充分
update_timestep = 256
K_epochs        = 8
eps_clip        = 0.2
gamma           = 0.995  # 0.995^120≈0.55 vs 0.99^120≈0.30，终点奖励传播更强
lr_actor        = 0.0003  # 超过 0.0003 易导致策略崩溃
lr_critic       = 0.0006  # 通常设为 Actor 的 2 倍

ppo_agent          = None
buffer             = RolloutBuffer()
maze_grid          = None
maze_w, maze_h     = 0, 0
current_seed       = 0
last_action        = None
accumulated_reward = 0.0
episode_reward     = 0.0
episode_steps      = 0
episode_count      = 0
global_steps       = 0
is_demo            = False
best_reward        = -999999.0
# best 判据：步数更少优先；BFS shaping 使长路径 reward 偏高，单纯比 reward 会保存次优策略
best_steps         = 999999
consecutive_timeouts         = 0
TIMEOUT_RECOVERY_THRESHOLD   = 5   # 给策略 4 局自然探索窗口，避免过早打断有效梯度
recovery_attempts_failed     = 0
last_episode_succeeded       = False
# has_ever_completed：区分"从未通关"与"通关后再失败"，避免把卡死策略写入 _best.pth
has_ever_completed           = False
bfs_dist     = {}
max_bfs_dist = 1
PPO_STATE_DIM = 38  # BFS 9维 + 局部5×5视野 25维 + 坐标 4维
_prev_pos    = None
_pos_history = []
_POS_WINDOW      = 4    # 只捕捉 A→B→A 即时振荡；旧值 20 使走廊中 98% 步骤触发惩罚
_REVISIT_PENALTY = 0.5

# ========== 存档与日志工具 ==========
save_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data~")
os.makedirs(save_dir, exist_ok=True)
log_path       = os.path.join(save_dir, "ppo_training_log_default.csv")
model_path     = os.path.join(save_dir, "ppo_model_default.pth")
train_log_path = log_path
demo_log_path  = os.path.join(save_dir, "ppo_demo_log_default.csv")

def init_log_file():
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(["episode", "steps", "hit_count", "total_reward"])

def save_training_log(ep, steps, hit_count, reward):
    if not os.path.exists(log_path):
        init_log_file()
    with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow([ep, steps, hit_count, f"{reward:.2f}"])

def is_better_than_best(curr_steps, curr_reward):
    # 步数更少优先；步数相同时 reward 更高者优先
    return curr_steps < best_steps or (curr_steps == best_steps and curr_reward > best_reward)

def resume_training_state(update_best=True):
    global episode_count, best_reward, best_steps
    if os.path.exists(log_path):
        try:
            last_ep = 0
            max_steps = 6 * maze_w * maze_h if maze_w > 0 and maze_h > 0 else 999999
            bsf, brf = 999999, -999999.0
            with open(log_path, "r", encoding="utf-8-sig") as f:
                for row in list(csv.reader(f))[1:]:
                    if len(row) >= 4:
                        last_ep = int(row[0])
                        s, r = int(row[1]), float(row[3])
                        if s < max_steps and (s < bsf or (s == bsf and r > brf)):
                            bsf, brf = s, r
            if last_ep > 0:
                episode_count = last_ep
                if update_best:
                    best_steps, best_reward = bsf, brf
                mode = "训练" if update_best else "演示"
                tail = f"，历史最佳 {best_steps}步/{best_reward:.1f}分" if update_best and best_steps < 999999 else ""
                print(f"[Python AI] 断点续{mode}：从 EP {episode_count} 继续{tail}")
                return
        except Exception as e:
            print(f"[Python AI] 读取日志失败: {e}")
    episode_count = 0
    print(f"[Python AI] 未发现历史{'训练' if update_best else '演示'}数据，从 EP 0 开始。")

def compute_bfs_distances(grid, w, h):
    """BFS 预计算到终点的最短路径步数，替代误导性的欧氏距离 shaping。"""
    global max_bfs_dist
    from collections import deque
    goal_x, goal_y = w - 2, h - 2
    dist = {}
    if grid is None or not (0 <= goal_y < h and 0 <= goal_x < w):
        max_bfs_dist = 1
        return dist
    queue = deque([(goal_x, goal_y)])
    dist[(goal_x, goal_y)] = 0
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 0 and (nx, ny) not in dist:
                dist[(nx, ny)] = dist[(x, y)] + 1
                queue.append((nx, ny))
    max_bfs_dist = max(dist.values()) if dist else 1
    return dist

def build_network(w, h):
    global ppo_agent
    ppo_agent = PPO(PPO_STATE_DIM, 4, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
    print(f"[Python AI] PPO 网络初始化完成，输入维度: {PPO_STATE_DIM}")

def save_model(is_best=False):
    if ppo_agent is not None:
        path = model_path.replace(".pth", "_best.pth") if is_best else model_path
        torch.save({
            "policy_state":     ppo_agent.policy.state_dict(),
            "policy_old_state": ppo_agent.policy_old.state_dict(),
            "optimizer_state":  ppo_agent.optimizer.state_dict(),
            "entropy_coeff":    ppo_agent.entropy_coeff,
        }, path)
        if is_best:
            print(f"[Python AI] 新最优模型已保存: {os.path.basename(path)}")

def load_model(use_best=False):
    if ppo_agent is None:
        return
    target_path = model_path
    if use_best:
        best_path = model_path.replace(".pth", "_best.pth")
        if os.path.exists(best_path):
            target_path = best_path
        else:
            # best pth 不存在时不降级到普通 pth，保持随机初始化
            print(f"[Python AI] 未找到最佳模型 {os.path.basename(best_path)}，使用随机初始化")
            return
    if not os.path.exists(target_path):
        return
    try:
        ckpt = torch.load(target_path, map_location=device)
        if isinstance(ckpt, dict) and "policy_state" in ckpt:
            ppo_agent.policy.load_state_dict(ckpt["policy_state"])
            ppo_agent.policy_old.load_state_dict(
                ckpt.get("policy_old_state") or ppo_agent.policy.state_dict())
            if ckpt.get("optimizer_state"):
                try:
                    ppo_agent.optimizer.load_state_dict(ckpt["optimizer_state"])
                    print("[Python AI] 恢复 Adam optimizer 动量状态")
                except Exception as e:
                    print(f"[Python AI] optimizer 加载失败（网络结构已变更）: {e}")
            if "entropy_coeff" in ckpt:
                # 探索地板：低于 0.03 在拐角无梯度场景必然卡死
                loaded = ckpt["entropy_coeff"]
                ppo_agent.entropy_coeff = max(loaded, 0.03)
                if loaded < 0.03:
                    print(f"[Python AI] entropy_coeff {loaded:.5f} 已抬升至探索地板 0.03")
                else:
                    print(f"[Python AI] 恢复 entropy_coeff={ppo_agent.entropy_coeff:.5f}")
        else:
            # 兼容旧格式（裸 state_dict）
            ppo_agent.policy.load_state_dict(ckpt)
            ppo_agent.policy_old.load_state_dict(ckpt)
        print(f"[Python AI] 模型加载成功: {os.path.basename(target_path)}")
    except Exception as e:
        print(f"[Python AI] 模型加载失败: {e}")

def read_from_shared_memory():
    mm.seek(0)
    # split('\x00') 而非 strip：防 C# 两次连续写入时 \x00 夹杂数据中间导致解析失败
    # errors='ignore'：并发读写可能切半 UTF-8 多字节序列，宁可丢字节也不丢帧
    raw = mm.read(data_size).decode('utf-8', errors='ignore')
    return raw.split('\x00')[0].strip()

def write_to_shared_memory(data):
    try:
        mm.seek(0)
        mm.write(data.ljust(data_size, '\x00').encode('utf-8'))
    except Exception as e:
        print(f"[WRITE ERROR] {e}")

def get_state_vector(x, y):
    """紧凑局部观测（38 维，与地图大小无关）。
    全图扁平化（229 维）无法让全连接网络学到空间导航；改用局部信息：
      BFS 特征 9维 + 局部 5×5 视野 25维 + 坐标 4维
    """
    if maze_grid is None:
        return np.zeros(PPO_STATE_DIM, dtype=np.float32)
    iy = min(max(int(round(y)), 0), maze_h - 1)
    ix = min(max(int(round(x)), 0), maze_w - 1)
    goal_x, goal_y = maze_w - 2, maze_h - 2
    inv_bfs = 1.0 / max(1, max_bfs_dist)

    # BFS 特征（9维）：当前归一化距离 + 四邻居 [可走性, 归一化距离]
    bfs_features = [bfs_dist.get((ix, iy), max_bfs_dist) * inv_bfs]
    for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
        nx, ny = ix + dx, iy + dy
        if 0 <= ny < maze_h and 0 <= nx < maze_w and maze_grid[ny, nx] == 0:
            bfs_features.extend([1.0, bfs_dist.get((nx, ny), max_bfs_dist) * inv_bfs])
        else:
            bfs_features.extend([0.0, 1.0])

    # 局部 5×5 视野（25维）：-1=墙/越界，0=通路，0.5=终点，1.0=自身
    local_view = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            lx, ly = ix + dx, iy + dy
            if 0 <= ly < maze_h and 0 <= lx < maze_w:
                if   lx == ix and ly == iy:         local_view.append(1.0)
                elif lx == goal_x and ly == goal_y: local_view.append(0.5)
                elif maze_grid[ly, lx] == 1:        local_view.append(-1.0)
                else:                               local_view.append(0.0)
            else:
                local_view.append(-1.0)

    # 坐标特征（4维）：归一化位置 + 与终点相对方向
    inv_w, inv_h = 1.0 / max(1, maze_w), 1.0 / max(1, maze_h)
    pos_features = [x * inv_w, y * inv_h, (goal_x - x) * inv_w, (goal_y - y) * inv_h]

    return np.array(bfs_features + local_view + pos_features, dtype=np.float32)

def get_action_mask(x, y):
    """合法动作掩码 [上, 下, 左, 右]；四面全堵时放开所有方向防止 softmax NaN。"""
    mask = np.ones(4, dtype=np.float32)
    if maze_grid is None:
        return mask
    iy = min(max(int(round(y)), 0), maze_h - 1)
    ix = min(max(int(round(x)), 0), maze_w - 1)
    if iy + 1 >= maze_h or maze_grid[iy + 1, ix] == 1: mask[0] = 0.0
    if iy - 1 < 0       or maze_grid[iy - 1, ix] == 1: mask[1] = 0.0
    if ix - 1 < 0       or maze_grid[iy, ix - 1] == 1: mask[2] = 0.0
    if ix + 1 >= maze_w  or maze_grid[iy, ix + 1] == 1: mask[3] = 0.0
    if mask.sum() == 0:
        mask = np.ones(4, dtype=np.float32)
    return mask

# ========== 主循环（IPC 消息调度） ==========
print("[Python AI] PPO Backend Ready. Waiting for IPC Connection...")
write_to_shared_memory("READY")
last_handled_info = {"data": "READY", "step": -1, "coords": (0, 0)}
last_action_msg   = ""  # 缓存上次 ACTION：STATE 重复时回放，避免 C# 5s 超时

try:
    while True:
        try:
            data = read_from_shared_memory()
        except Exception:
            time.sleep(0.01)
            continue

        if data and data != "WAITING":
            # ------ 重复帧过滤 ------
            if data.startswith("STATE:"):
                try:
                    parts       = data[6:].split('|')
                    curr_step   = int(parts[3])
                    curr_coords = (float(parts[0]), float(parts[1]))
                    # (step, coords) 去重而非完整 data 字符串：C# 发 STATE 后清零 pendingReward，
                    # 第二帧 reward=0 使字符串不等，导致 BFS shaping 被反复累加
                    if curr_step == last_handled_info["step"] and curr_coords == last_handled_info["coords"]:
                        if last_action_msg:
                            write_to_shared_memory(last_action_msg)
                        continue
                except Exception:
                    pass
            elif data == last_handled_info["data"]:
                continue

            # ====== GRID：加载地图，重建网络 ======
            if data.startswith("GRID:"):
                parts   = data[5:].split('|')
                grid_ok = False
                try:
                    maze_w, maze_h = int(parts[0]), int(parts[1])
                    if len(parts) > 3:
                        current_seed = int(parts[2])
                        grid_vals    = [int(v) for v in parts[3].split(',')]
                        expected     = maze_w * maze_h
                        if len(grid_vals) != expected:
                            print(f"[PPO GRID] 数据不完整: {len(grid_vals)}/{expected}，等待重发")
                        else:
                            train_log_path = os.path.join(save_dir, f"ppo_log_s{maze_w}_{current_seed}.csv")
                            demo_log_path  = os.path.join(save_dir, f"ppo_demo_log_s{maze_w}_{current_seed}.csv")
                            log_path       = train_log_path
                            model_path     = os.path.join(save_dir, f"ppo_model_s{maze_w}_{current_seed}.pth")
                            maze_grid      = np.array(grid_vals).reshape((maze_h, maze_w))
                            build_network(maze_w, maze_h)
                            buffer.clear()
                            best_reward = -999999.0
                            best_steps  = 999999
                            has_ever_completed = os.path.exists(
                                os.path.join(save_dir, f"ppo_model_s{maze_w}_{current_seed}_best.pth"))
                            consecutive_timeouts     = 0
                            recovery_attempts_failed = 0
                            _max_steps = 6 * maze_w * maze_h
                            if os.path.exists(train_log_path):
                                try:
                                    with open(train_log_path, "r", encoding="utf-8-sig") as _f:
                                        for _row in list(csv.reader(_f))[1:]:
                                            if len(_row) >= 4:
                                                _s, _r = int(_row[1]), float(_row[3])
                                                if _s < _max_steps and (_s < best_steps or (_s == best_steps and _r > best_reward)):
                                                    best_steps, best_reward = _s, _r
                                except Exception:
                                    pass
                            bfs_dist  = compute_bfs_distances(maze_grid, maze_w, maze_h)
                            _prev_pos = None
                            _best_tail = f"best={best_steps}步/{best_reward:.1f}分" if best_steps < 999999 else "best=未通关"
                            print(f"[PPO GRID] {maze_w}x{maze_h} loaded, seed={current_seed}, {_best_tail}, BFS={len(bfs_dist)}")
                            grid_ok = True
                    else:
                        print(f"[PPO GRID] Format error: expected >3 parts, got {len(parts)}")
                except Exception as e:
                    print(f"[PPO GRID] Parse error: {e}")
                if grid_ok:
                    write_to_shared_memory("GRID_OK")
                    last_handled_info["data"] = "GRID_OK"

            # ====== PARAM：更新超参数 ======
            # gamma/lr 同时写模块级与 ppo_agent：PARAM 早于 GRID 到达时
            # build_network() 也能用上最新值（只写 ppo_agent 则 PARAM 早到时被旧默认值覆盖）
            elif data.startswith("PARAM:"):
                params = data[6:].split(',')
                if len(params) > 1:
                    gamma = float(params[1])
                    if ppo_agent is not None:
                        ppo_agent.gamma = gamma
                if len(params) > 2 and params[2] != '_':
                    coeff = float(params[2])
                    if ppo_agent is not None and coeff >= 0:
                        ppo_agent.entropy_coeff = min(coeff, 0.5)
                        print(f"[UI] entropy_coeff 已设为 {ppo_agent.entropy_coeff:.4f}")
                if len(params) > 5:
                    lr_base   = float(params[5])
                    lr_actor  = min(lr_base * 0.2, 0.0003)  # 上限保护：超过 0.0003 易策略崩溃
                    lr_critic = min(lr_actor * 2,  0.001)
                    if ppo_agent is not None:
                        ppo_agent.optimizer.param_groups[0]['lr'] = lr_actor
                        ppo_agent.optimizer.param_groups[1]['lr'] = lr_critic
                last_handled_info["data"] = data

            # ====== COMMAND：模式切换与控制指令 ======
            elif data.startswith("COMMAND:"):
                cmd_full  = data[8:]
                cmd_parts = cmd_full.split('|')
                cmd       = cmd_parts[0]
                if cmd in ("QUIT", "EXIT"):
                    print("[Python AI] 收到退出指令，进程终止。")
                    break
                elif cmd == "DEMO":
                    load_model(use_best=True)
                    is_demo = True
                    buffer.clear()
                    log_path = demo_log_path
                    init_log_file()
                    resume_training_state(update_best=False)
                    print(f"-> PPO DEMO MODE: Loaded Best Weights, Resuming from EP {episode_count}")
                elif cmd == "START":
                    if ppo_agent is None and maze_grid is not None:
                        build_network(maze_w, maze_h)
                    # has_ever_completed → 加载 _best.pth：常规 pth 可能是崩溃局策略；
                    # _best.pth 保证是真通关权重，重启以它为起点更安全
                    if ppo_agent is not None:
                        if has_ever_completed:
                            load_model(use_best=True)
                        ppo_agent.entropy_coeff = max(ppo_agent.entropy_coeff, 0.05)
                    if not has_ever_completed:
                        print(f"[Python AI] 无通关记录，跳过加载坏权重（熵={ppo_agent.entropy_coeff if ppo_agent else 0:.3f}）")
                    is_demo  = False
                    buffer.clear()
                    global_steps             = 0
                    consecutive_timeouts     = 0
                    recovery_attempts_failed = 0
                    _prev_pos                = None
                    log_path = train_log_path
                    init_log_file()
                    resume_training_state()
                    last_action = None
                    print(f"-> PPO TRAINING START (Map: {current_seed}) Resuming from EP {episode_count}")
                elif cmd == "RESET":
                    # 补录本 episode 最后一步的奖励（RESET 到达前 buffer 尚未记录终止步）
                    if last_action is not None and not is_demo:
                        buffer.rewards.append(accumulated_reward)
                        buffer.is_terminals.append(True)
                        accumulated_reward = 0.0
                        last_action        = None
                    # PPO 更新放在 RESET 处：Unity 此时暂停发送 STATE，避免更新阻塞丢失末端数据
                    has_terminal = len(buffer.is_terminals) > 0 and buffer.is_terminals[-1]
                    n = len(buffer.rewards)
                    if not is_demo and (n >= 32 or (has_terminal and n >= 8)):
                        # 短 buffer 自适应降 K_epochs：避免单条轨迹被反复回放过拟合
                        original_k = ppo_agent.K_epochs
                        ppo_agent.K_epochs = min(original_k, max(2, n // 4))
                        print(f"[PPO Update] RESET 触发，样本: {n}, K_epochs: {ppo_agent.K_epochs}")
                        try:
                            ppo_agent.update(buffer)
                        finally:
                            ppo_agent.K_epochs = original_k
                    elif not is_demo and n > 0:
                        buffer.clear()  # 样本不足丢弃：防跨 episode 错位 + 避免极短轨迹过拟合
                    hit_count_this_ep = int(cmd_parts[1]) if len(cmd_parts) > 1 else 0
                    episode_count += 1
                    save_training_log(episode_count, episode_steps, hit_count_this_ep, episode_reward)
                    if not is_demo:
                        # 两阶段衰减：通关前缓慢（0.998）保留探索，通关后加速（0.999）固化策略
                        ppo_agent.entropy_coeff = max(
                            ppo_agent.entropy_coeff * (0.999 if has_ever_completed else 0.998),
                            0.01 if has_ever_completed else 0.02)
                        _saved_best = False
                        if last_episode_succeeded:
                            if not has_ever_completed or is_better_than_best(episode_steps, episode_reward):
                                best_steps, best_reward = episode_steps, episode_reward
                                has_ever_completed = True
                                save_model(is_best=True)
                                _saved_best = True
                        if not _saved_best and episode_count % 10 == 0:
                            save_model(is_best=False)
                    last_episode_succeeded = False
                    print(f"  EP {episode_count} | Steps: {episode_steps} | Hits: {hit_count_this_ep} | Reward: {episode_reward:.1f} | Best: {best_steps}步/{best_reward:.1f}分")
                    last_action        = None
                    accumulated_reward = 0.0
                    episode_reward     = 0.0
                    episode_steps      = 0
                    _prev_pos          = None
                    del _pos_history[:]
                    last_handled_info["step"]   = -1
                    last_handled_info["coords"] = (0, 0)
                    last_action_msg             = ""
                last_handled_info["data"] = data

            # ====== STATE：每步推理，输出动作 ======
            elif data.startswith("STATE:"):
                if ppo_agent is None:
                    write_to_shared_memory("ACTION:0")
                    continue
                state_info = data[6:].split('|')
                if len(state_info) < 2 or state_info[0] == '' or state_info[1] == '':
                    continue
                try:
                    cur_x, cur_y = float(state_info[0]), float(state_info[1])
                except ValueError:
                    print(f"[PPO] 跳过损坏的 STATE 数据: {data[:50]}")
                    continue

                # 读取 Unity 端步骤奖励
                if len(state_info) >= 5 and state_info[4] != '':
                    try:
                        r = float(state_info[4])
                        accumulated_reward += r; episode_reward += r
                    except ValueError:
                        pass

                # BFS 势函数 shaping（归属 a_{t-1}）：去掉 γ 因子，
                # 带 γ 版 A→B→A 净正奖励导致 agent 在高 BFS 区振荡积累分数
                curr_cell = (int(round(cur_x)), int(round(cur_y)))
                if _prev_pos is not None and curr_cell != _prev_pos:
                    prev_bfs = bfs_dist.get(_prev_pos, -1)
                    curr_bfs = bfs_dist.get(curr_cell, -1)
                    if prev_bfs >= 0 and curr_bfs >= 0:
                        bfs_shaping = (prev_bfs - curr_bfs) * 4.0
                        accumulated_reward += bfs_shaping
                        episode_reward     += bfs_shaping
                _prev_pos = curr_cell

                # 振荡惩罚：窗口内重访同一格子时追加惩罚打破确定性轨迹
                if curr_cell in _pos_history:
                    accumulated_reward -= _REVISIT_PENALTY
                    episode_reward     -= _REVISIT_PENALTY
                _pos_history.append(curr_cell)
                if len(_pos_history) > _POS_WINDOW:
                    _pos_history.pop(0)

                # 将上一步 a_{t-1} 的完整奖励（Unity奖励 + BFS shaping）写入 buffer
                if last_action is not None and not is_demo:
                    buffer.rewards.append(accumulated_reward)
                    buffer.is_terminals.append(False)
                accumulated_reward = 0.0

                current_state_vec = get_state_vector(cur_x, cur_y)
                current_mask      = get_action_mask(cur_x, cur_y)

                # 阶段性更新：须在 select_action 之前（此时 states 与 rewards 等长）
                if not is_demo and len(buffer.rewards) >= 128 and global_steps % update_timestep == 0:
                    n = len(buffer.rewards)
                    original_k = ppo_agent.K_epochs
                    ppo_agent.K_epochs = min(original_k, max(2, n // 128))
                    try:
                        ppo_agent.update(buffer, next_state_vec=current_state_vec)
                    finally:
                        ppo_agent.K_epochs = original_k

                if is_demo:
                    # Demo 模式直接走 BFS 最短路径，保证每次都能到达终点
                    ix_d, iy_d = int(round(cur_x)), int(round(cur_y))
                    best_a, best_d = -1, float('inf')
                    for ai, (dx, dy) in enumerate([(0,1),(0,-1),(-1,0),(1,0)]):
                        nx, ny = ix_d + dx, iy_d + dy
                        if current_mask[ai] > 0:
                            d = bfs_dist.get((nx, ny), float('inf'))
                            if d < best_d:
                                best_d, best_a = d, ai
                    action = best_a if best_a >= 0 else 0
                else:
                    action = ppo_agent.select_action(current_state_vec, buffer, current_mask)

                last_action    = action
                episode_steps += 1
                global_steps  += 1
                # Demo 用 argmax 回传 0.0；训练模式回传实际 entropy_coeff 供 UI 显示
                explore_strength = 0.0 if is_demo else (ppo_agent.entropy_coeff if ppo_agent else 0.0)
                action_msg = f"ACTION:{action}|{explore_strength:.4f}"
                write_to_shared_memory(action_msg)
                last_action_msg             = action_msg
                last_handled_info["data"]   = data
                last_handled_info["step"]   = int(state_info[3]) if len(state_info) > 3 and state_info[3] != '' else -1
                last_handled_info["coords"] = (cur_x, cur_y)
                continue

            # ====== REWARD：终端奖励（到达终点 rtype=2 / 超步失败 rtype=3） ======
            elif data.startswith("REWARD:"):
                reward_info = data[7:].split('|')
                try:
                    r = float(reward_info[0]) if reward_info[0] else 0.0
                except ValueError:
                    r = 0.0
                rtype = int(reward_info[3]) if len(reward_info) > 3 else 0
                accumulated_reward += r; episode_reward += r
                if rtype in (2, 3):
                    if last_action is not None and not is_demo:
                        buffer.rewards.append(accumulated_reward)
                        buffer.is_terminals.append(True)
                    if rtype == 2:
                        consecutive_timeouts     = 0
                        recovery_attempts_failed = 0
                        last_episode_succeeded   = True
                        print(f"★ [通关] PPO | 局:{episode_count+1} | 步:{episode_steps} | 分:{episode_reward:.1f} ★")
                        # REWARD 处立即保存：通关后 RESET 可能延迟到达，防止成功权重丢失
                        if not is_demo and ppo_agent is not None:
                            if not has_ever_completed or is_better_than_best(episode_steps, episode_reward):
                                _is_first = not has_ever_completed
                                best_steps, best_reward = episode_steps, episode_reward
                                has_ever_completed = True
                                save_model(is_best=True)
                                print(f"  ↳ {'首次通关' if _is_first else '刷新纪录'} best→{best_steps}步/{best_reward:.1f}分")
                    else:
                        consecutive_timeouts  += 1
                        last_episode_succeeded = False
                        if not is_demo and ppo_agent is not None:
                            ppo_agent.entropy_coeff = max(ppo_agent.entropy_coeff, 0.03)
                        # 刚性恢复：连续超步说明 Adam 动量已把策略锁死在坏的局部最优
                        if (not is_demo and ppo_agent is not None
                                and consecutive_timeouts >= TIMEOUT_RECOVERY_THRESHOLD):
                            best_path      = model_path.replace(".pth", "_best.pth")
                            best_is_useful = os.path.exists(best_path)
                            recovery_attempts_failed += 1
                            print(f"⛑ [Recovery] 连续 {consecutive_timeouts} 局坍塌，刚性恢复（累计 {recovery_attempts_failed} 次）...")
                            if best_is_useful:
                                build_network(maze_w, maze_h)  # 清掉 Adam 错误动量
                                load_model(use_best=True)
                                # 恢复次数越多熵越高：第1次0.06 → 每次+0.02 → 上限0.12
                                ppo_agent.entropy_coeff = min(0.06 + 0.02 * (recovery_attempts_failed - 1), 0.12)
                                # 每次都注入噪声：通关后失败通常是策略固化，必须同时扰动权重
                                noise_std = min(0.01 * recovery_attempts_failed, 0.08)
                                with torch.no_grad():
                                    for p in ppo_agent.policy.actor.parameters():
                                        p.add_(torch.randn_like(p) * noise_std)
                                    ppo_agent.policy_old.load_state_dict(ppo_agent.policy.state_dict())
                                print(f"  ↳ 重载快照 best={best_steps}步/{best_reward:.1f}分，σ={noise_std:.3f}，熵={ppo_agent.entropy_coeff:.2f}")
                            else:
                                # 无快照（从未通关）：不激进干预，温和回升熵即可
                                ppo_agent.entropy_coeff = max(ppo_agent.entropy_coeff, 0.05)
                                print(f"  ↳ 无最佳快照，温和回升熵至 {ppo_agent.entropy_coeff:.3f}")
                            buffer.clear()
                            consecutive_timeouts = 0
                        else:
                            print(f"⚠ [超步] PPO | 局:{episode_count+1} | 步:{episode_steps} | 分:{episode_reward:.1f} | 熵:{ppo_agent.entropy_coeff:.4f} | 连续:{consecutive_timeouts}/{TIMEOUT_RECOVERY_THRESHOLD}")
                    accumulated_reward = 0.0
                    last_action        = None
                last_handled_info["data"]   = data
                last_handled_info["step"]   = -1
                last_handled_info["coords"] = (0, 0)

        time.sleep(0.005)

except Exception as e:
    import traceback
    crash_log_path = os.path.join(save_dir, "CRASH_LOG.txt")
    with open(crash_log_path, "w", encoding="utf-8") as f:
        f.write(f"Crash at episode={episode_count}, global_steps={global_steps}, step={episode_steps}\n")
        f.write(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
    print(f"CRASHED EP {episode_count} step {global_steps}: {type(e).__name__}: {e}")
    print(f"Stack trace → {crash_log_path}\nProcess exits in 60s; restart Unity training to recover.")
    time.sleep(60)
