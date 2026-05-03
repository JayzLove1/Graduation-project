# -*- coding: utf-8 -*-
# DQN (Dueling Double Deep Q-Network) 强化学习迷宫 AI 后端
# 特性：Dueling Architecture / Double DQN / Experience Replay
# IPC：Windows 共享内存 MazeRLSharedMemory（4096 bytes）
import mmap
import time
import random
import os
import csv
import pickle
import numpy as np
from collections import deque
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
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

# ========== Dueling DQN 网络结构 ==========
# Q(s,a) = V(s) + (A(s,a) - mean(A))
# 减去优势均值确保 V(s) 与 A(s,a) 的分解唯一，增强训练稳定性
class DQNLayer(nn.Module):
    def __init__(self, input_size, num_actions=4):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(),
            nn.Linear(256, 128),        nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_actions)
        )

    def forward(self, x):
        x         = self.feature(x)
        value     = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

# ========== 超参数 ==========
lr             = 0.001
gamma          = 0.95
epsilon        = 1.0
epsilon_decay  = 0.995
min_epsilon    = 0.05
batch_size     = 64
target_update_freq = 100  # 每 100 步硬拷贝 policy_net → target_net，延迟更新防振荡

policy_net    = None
target_net    = None
optimizer     = None
replay_buffer = deque(maxlen=20000)

# ========== 状态追踪 ==========
maze_grid       = None
maze_w, maze_h  = 0, 0
current_seed    = 0
last_state_vec     = None
last_action        = None
accumulated_reward = 0.0
episode_reward     = 0.0
episode_steps      = 0
episode_count      = 0
global_steps       = 0
is_demo            = False
best_reward        = -999999.0
position_history   = deque(maxlen=50)
visit_count_map    = {}
max_episode_steps  = 10000  # 收到 GRID 后按地图尺寸动态调整

# ========== 存档路径 ==========
save_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data~")
os.makedirs(save_dir, exist_ok=True)
log_path       = os.path.join(save_dir, "dqn_training_log_default.csv")
model_path     = os.path.join(save_dir, "dqn_model_default.pth")
buffer_path    = os.path.join(save_dir, "dqn_buffer_default.pkl")
train_log_path = log_path
demo_log_path  = os.path.join(save_dir, "dqn_demo_log_default.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Python AI] 使用计算设备: {device.type.upper()}")

# ========== 日志与存档工具 ==========
def init_log_file():
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="", encoding='utf-8-sig') as f:
            csv.writer(f).writerow(["episode", "steps", "hit_count", "total_reward", "epsilon"])

def save_training_log(ep, steps, hit_count, reward, eps):
    if not os.path.exists(log_path):
        init_log_file()
    with open(log_path, "a", newline="", encoding='utf-8-sig') as f:
        csv.writer(f).writerow([ep, steps, hit_count, f"{reward:.2f}", f"{eps:.4f}"])

def resume_training_state(update_best=True):
    """断点续训/续演示。update_best=False 时仅续号，避免 demo CSV 污染训练状态。"""
    global episode_count, epsilon, best_reward
    if os.path.exists(log_path):
        try:
            last_ep     = 0
            last_eps    = epsilon
            best_so_far = -999999.0
            with open(log_path, "r", encoding="utf-8-sig") as f:
                for row in list(csv.reader(f))[1:]:
                    if len(row) >= 5:
                        last_ep     = int(row[0])
                        last_eps    = float(row[4])
                        r           = float(row[3])
                        if r > best_so_far:
                            best_so_far = r
            if last_ep > 0:
                episode_count = last_ep
                if update_best:
                    epsilon     = last_eps
                    best_reward = best_so_far
                mode = "训练" if update_best else "演示"
                msg  = f"[Python AI] 断点续{mode}：从 EP {episode_count} 继续"
                if update_best:
                    msg += f"，历史最高分 {best_reward:.1f}，Epsilon={epsilon:.4f}"
                print(msg)
                return
        except Exception as e:
            print(f"[Python AI] 读取训练日志失败: {e}")
    episode_count = 0
    # epsilon 仅在训练模式重置：演示模式保留 0.0，
    # 否则 demo CSV 为空时兜底分支会把 epsilon 拉回 1.0，导致整局变成纯随机游走
    if update_best:
        epsilon = 1.0
    print("[Python AI] 未发现历史训练数据，从头开始。")

def build_network(w, h):
    global policy_net, target_net, optimizer
    input_size = w * h + 4
    policy_net = DQNLayer(input_size, 4).to(device)
    target_net = DQNLayer(input_size, 4).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer  = optim.Adam(policy_net.parameters(), lr=lr)
    print(f"[Python AI] Dueling DQN 初始化完成，输入维度: {input_size} ({w}x{h})")

def save_model(is_best=False):
    if policy_net is not None:
        path = model_path.replace(".pth", "_best.pth") if is_best else model_path
        torch.save({
            "policy_net_state": policy_net.state_dict(),
            "optimizer_state":  optimizer.state_dict() if optimizer is not None else None,
        }, path)
        if is_best:
            print(f"[Python AI] 新最优模型已保存: {os.path.basename(path)}")

def _migrate_state_dict(saved_state: dict, current_input_size: int) -> dict:
    """旧输入层（w*h+2）权重迁移至新格式（w*h+4），新增列零填充不影响已学特征。"""
    import copy
    new_state = copy.deepcopy(saved_state)
    if "fc1.weight" not in new_state:
        return new_state
    old_w     = new_state["fc1.weight"]
    old_input = old_w.shape[1]
    if old_input == current_input_size:
        return new_state
    if old_input > current_input_size:
        new_state["fc1.weight"] = old_w[:, :current_input_size]
        print(f"[Python AI] 输入层从 {old_input} 截断为 {current_input_size}")
    else:
        pad_cols = current_input_size - old_input
        padding  = torch.zeros(old_w.shape[0], pad_cols, dtype=old_w.dtype, device=old_w.device)
        new_state["fc1.weight"] = torch.cat([old_w, padding], dim=1)
        print(f"[Python AI] 输入层从 {old_input} 扩展为 {current_input_size}（零填充 {pad_cols} 列）")
    return new_state

def load_model(use_best=False):
    """加载模型，兼容新格式（含 optimizer）和旧格式（裸 state_dict）。"""
    if policy_net is None:
        return
    target_path = model_path
    if use_best:
        best_path = model_path.replace(".pth", "_best.pth")
        if os.path.exists(best_path):
            target_path = best_path
        else:
            print(f"[Python AI] 未找到最佳模型 {os.path.basename(best_path)}，使用随机初始化")
            return
    if not os.path.exists(target_path):
        print(f"[Python AI] 未找到模型文件 {os.path.basename(target_path)}，使用随机初始化")
        return
    try:
        checkpoint         = torch.load(target_path, map_location=device)
        is_new_format      = isinstance(checkpoint, dict) and "policy_net_state" in checkpoint
        state_dict         = checkpoint["policy_net_state"] if is_new_format else checkpoint
        current_input_size = list(policy_net.parameters())[0].shape[1]
        saved_input_size   = state_dict["fc1.weight"].shape[1] if "fc1.weight" in state_dict else current_input_size
        state_dict = _migrate_state_dict(state_dict, current_input_size)
        policy_net.load_state_dict(state_dict)
        target_net.load_state_dict(policy_net.state_dict())
        if is_new_format and checkpoint.get("optimizer_state") is not None:
            if saved_input_size == current_input_size and optimizer is not None:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state"])
                    print("[Python AI] 恢复 Adam optimizer 动量状态")
                except Exception as e:
                    print(f"[Python AI] optimizer 状态加载失败: {e}")
            else:
                print("[Python AI] 输入层已迁移，optimizer 动量从零重建")
        print(f"[Python AI] 模型加载成功: {os.path.basename(target_path)}")
    except Exception as e:
        print(f"[Python AI] 模型加载失败: {e}")

def save_replay_buffer():
    """序列化至磁盘，最多保留最新 10000 条（约 10~30 MB）。"""
    try:
        with open(buffer_path, "wb") as f:
            pickle.dump(list(replay_buffer)[-10000:], f)
    except Exception as e:
        print(f"[Python AI] replay buffer 保存失败: {e}")

def load_replay_buffer():
    """从磁盘恢复 replay buffer，自动过滤维度不匹配的旧样本。"""
    if not os.path.exists(buffer_path):
        return
    try:
        with open(buffer_path, "rb") as f:
            samples = pickle.load(f)
        if policy_net is not None and len(samples) > 0:
            expected_dim = list(policy_net.parameters())[0].shape[1]
            valid        = [s for s in samples if hasattr(s[0], '__len__') and len(s[0]) == expected_dim]
            skipped      = len(samples) - len(valid)
            if skipped > 0:
                print(f"[Python AI] replay buffer 过滤 {skipped} 条旧样本，保留 {len(valid)} 条")
            samples = valid
        replay_buffer.extend(samples)
        print(f"[Python AI] 恢复 replay buffer: {len(replay_buffer)} 条经验")
    except Exception as e:
        print(f"[Python AI] replay buffer 加载失败: {e}")

def read_from_shared_memory():
    mm.seek(0)
    # split('\x00') 而非 strip：防 C# 两次连续写入时 \x00 夹杂数据中间导致解析失败
    raw = mm.read(data_size).decode('utf-8')
    return raw.split('\x00')[0].strip()

def write_to_shared_memory(data):
    try:
        mm.seek(0)
        mm.write(data.ljust(data_size, '\x00').encode('utf-8'))
    except Exception as e:
        print(f"[WRITE ERROR] {e}")

def get_state_vector(x, y):
    """状态编码：墙壁=-1，空地=0，终点=0.5，自身=1.0；追加 4 维坐标特征。"""
    if maze_grid is None:
        return np.zeros(1, dtype=np.float32)
    state      = np.copy(maze_grid).astype(np.float32)
    state[state == 1] = -1.0
    iy         = min(max(int(y), 0), maze_h - 1)
    ix         = min(max(int(x), 0), maze_w - 1)
    goal_x, goal_y = maze_w - 2, maze_h - 2
    if 0 <= goal_y < maze_h and 0 <= goal_x < maze_w:
        state[goal_y, goal_x] = 0.5
    state[iy, ix] = 1.0  # 自身位置覆盖写入
    inv_w, inv_h  = 1.0 / float(max(1, maze_w)), 1.0 / float(max(1, maze_h))
    coord_info = np.array([
        x * inv_w, y * inv_h,
        (goal_x - x) * inv_w, (goal_y - y) * inv_h,
    ], dtype=np.float32)
    return np.concatenate((state.flatten(), coord_info))

def decide_action(state_vec):
    """Epsilon-Greedy。演示模式强制走贪婪分支，防止任何错误抬升的 epsilon 污染演示。"""
    if not is_demo and random.random() < epsilon:
        choices = [0, 1, 2, 3]
        # 80% 概率排除上步反向动作，抑制往返循环
        if last_action is not None and random.random() < 0.8:
            rev = {0: 1, 1: 0, 2: 3, 3: 2}.get(last_action, -1)
            if rev in choices:
                choices.remove(rev)
        return random.choice(choices)
    with torch.no_grad():
        t_state = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
        return policy_net(t_state).argmax().item()

def learn(batch_size):
    """Double DQN：policy_net 选动作，target_net 评估价值，消除 Q 值高估。"""
    if len(replay_buffer) < batch_size or policy_net is None:
        return
    batch                              = random.sample(replay_buffer, batch_size)
    states, actions, rewards, nexts, dones = zip(*batch)
    states      = torch.FloatTensor(np.array(states)).to(device)
    actions     = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(np.array(nexts)).to(device)
    dones       = torch.FloatTensor(dones).unsqueeze(1).to(device)
    q_values    = policy_net(states).gather(1, actions)
    with torch.no_grad():
        next_actions    = policy_net(next_states).max(1)[1].unsqueeze(1)
        next_q_values   = target_net(next_states).gather(1, next_actions)
        target_q_values = rewards + gamma * next_q_values * (1 - dones)
    loss = F.mse_loss(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ========== 主循环（IPC 消息调度） ==========
print("[Python AI] DQN Backend Ready. Waiting for IPC Connection...")
write_to_shared_memory("READY")
last_handled_info = {"data": "READY", "step": -1, "coords": (0, 0)}

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
                    if (data == last_handled_info["data"]
                            and curr_step   == last_handled_info["step"]
                            and curr_coords == last_handled_info["coords"]):
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
                        current_seed   = int(parts[2])
                        grid_vals      = [int(v) for v in parts[3].split(',')]
                        expected       = maze_w * maze_h
                        if len(grid_vals) != expected:
                            # 共享内存写入未完成时被读取导致数据截断，等待 C# 重发
                            print(f"[DQN GRID] 数据不完整: {len(grid_vals)}/{expected}，等待重发")
                        else:
                            train_log_path = os.path.join(save_dir, f"dqn_log_s{maze_w}_{current_seed}.csv")
                            demo_log_path  = os.path.join(save_dir, f"dqn_demo_log_s{maze_w}_{current_seed}.csv")
                            log_path       = train_log_path
                            model_path     = os.path.join(save_dir, f"dqn_model_s{maze_w}_{current_seed}.pth")
                            buffer_path    = os.path.join(save_dir, f"dqn_buffer_s{maze_w}_{current_seed}.pkl")
                            maze_grid      = np.array(grid_vals).reshape((maze_h, maze_w))
                            build_network(maze_w, maze_h)
                            replay_buffer.clear()
                            load_replay_buffer()
                            best_reward = -999999.0
                            # best_reward 永远从训练 CSV 计算，避免演示成绩误导模型评估
                            if os.path.exists(train_log_path):
                                try:
                                    with open(train_log_path, "r", encoding="utf-8-sig") as _f:
                                        for _row in list(csv.reader(_f))[1:]:
                                            if len(_row) >= 4:
                                                _r = float(_row[3])
                                                if _r > best_reward:
                                                    best_reward = _r
                                except Exception:
                                    pass
                            max_episode_steps = maze_w * maze_h * 50
                            epsilon_decay     = 0.990 if maze_w >= 15 else 0.995
                            print(f"[DQN GRID] {maze_w}x{maze_h} loaded, seed={current_seed}, max_steps={max_episode_steps}, decay={epsilon_decay}")
                            grid_ok = True
                    else:
                        print(f"[DQN GRID] Format error: expected >3 parts, got {len(parts)}")
                except Exception as e:
                    print(f"[DQN GRID] Parse error: {e}")
                if grid_ok:
                    write_to_shared_memory("GRID_OK")
                    last_handled_info["data"] = "GRID_OK"
                last_handled_info["step"]   = -1
                last_handled_info["coords"] = (0, 0)

            # ====== PARAM：更新超参数 ======
            # params[3] (epsilon_decay) 由 Python 端按地图尺寸自适应，C# 传 "_"，此处不覆盖
            elif data.startswith("PARAM:"):
                params     = data[6:].split(',')
                gamma      = float(params[1])
                epsilon    = float(params[2])
                batch_size = int(params[4])
                if len(params) > 5:
                    lr = float(params[5])
                    if optimizer is not None:
                        for pg in optimizer.param_groups:
                            pg['lr'] = lr
                last_handled_info["data"]   = data
                last_handled_info["step"]   = -1
                last_handled_info["coords"] = (0, 0)

            # ====== COMMAND：模式切换与控制指令 ======
            elif data.startswith("COMMAND:"):
                cmd_full  = data[8:]
                cmd_parts = cmd_full.split('|')
                cmd       = cmd_parts[0]
                if cmd in ("QUIT", "EXIT"):
                    print("[Python AI] 收到退出指令，进程终止。")
                    break
                elif cmd == "DEMO":
                    if policy_net is None:
                        if maze_grid is not None:
                            build_network(maze_w, maze_h)
                        else:
                            print("[DQN] 警告: DEMO 模式下尚未收到地图数据 (GRID)")
                    if policy_net is not None:
                        load_model(use_best=True)
                        is_demo = True
                        epsilon = 0.0
                        replay_buffer.clear()
                        policy_net.eval()
                        log_path = demo_log_path
                        init_log_file()
                        resume_training_state(update_best=False)
                        print(f"-> DEMO MODE: Loaded Best DQN Weights, Resuming from EP {episode_count}")
                elif cmd == "START":
                    if policy_net is None:
                        if maze_grid is not None:
                            build_network(maze_w, maze_h)
                        else:
                            print("[DQN] 警告: 训练启动时尚未收到地图数据 (GRID)")
                    if policy_net is not None:
                        # 只在有通关记录时才加载权重：无通关历史说明已有 pth 大概率是坏权重
                        if best_reward >= 50.0:
                            load_model(use_best=False)
                        else:
                            print(f"[DQN] 无通关记录（best={best_reward:.1f}），跳过加载，使用随机初始化")
                        is_demo    = False
                        global_steps = 0
                        policy_net.train()
                        log_path = train_log_path
                        init_log_file()
                        resume_training_state()
                        last_state_vec = None
                        last_action    = None
                        print(f"-> DQN TRAINING START (Map: {current_seed}) Resuming from EP {episode_count}, Eps={epsilon:.4f}")
                elif cmd == "RESET":
                    hit_count_this_ep = int(cmd_parts[1]) if len(cmd_parts) > 1 else 0
                    episode_count += 1
                    # 演示模式不衰减 epsilon：保持 0.0 完全用 best 策略
                    if not is_demo and epsilon > min_epsilon:
                        epsilon *= epsilon_decay
                    save_training_log(episode_count, episode_steps, hit_count_this_ep, episode_reward, epsilon)
                    if not is_demo:
                        if episode_reward > best_reward:
                            best_reward = episode_reward
                            save_model(is_best=True)
                        elif episode_count % 10 == 0:
                            save_model(is_best=False)
                        if episode_count % 50 == 0:
                            save_replay_buffer()  # buffer 写入较慢，每 50 局一次
                    print(f"  EP {episode_count} | Steps: {episode_steps} | Hits: {hit_count_this_ep} | Reward: {episode_reward:.1f} | Eps: {epsilon:.3f} | Best: {best_reward:.1f}")
                    last_state_vec     = None
                    last_action        = None
                    accumulated_reward = 0.0
                    episode_reward     = 0.0
                    episode_steps      = 0
                    position_history.clear()
                    visit_count_map.clear()
                last_handled_info["data"]   = data
                last_handled_info["step"]   = -1
                last_handled_info["coords"] = (0, 0)

            # ====== STATE：每步推理，输出动作 ======
            elif data.startswith("STATE:"):
                if policy_net is None:
                    write_to_shared_memory("ACTION:0")
                    continue
                state_info = data[6:].split('|')
                if len(state_info) < 2 or state_info[0] == '' or state_info[1] == '':
                    continue
                try:
                    cur_x, cur_y = float(state_info[0]), float(state_info[1])
                except ValueError:
                    print(f"[DQN] 跳过损坏的 STATE 数据: {data[:50]}")
                    continue

                if len(state_info) >= 5 and state_info[4] != '':
                    try:
                        r = float(state_info[4])
                        accumulated_reward += r; episode_reward += r
                    except ValueError:
                        pass

                current_state_vec = get_state_vector(cur_x, cur_y)
                if last_state_vec is not None and last_action is not None:
                    replay_buffer.append((last_state_vec, last_action, accumulated_reward, current_state_vec, 0.0))
                if not is_demo and len(replay_buffer) > batch_size:
                    learn(batch_size)
                global_steps += 1
                if global_steps % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                accumulated_reward = 0.0

                # 循环检测：短窗口内同一格子多次出现则施加递增惩罚
                grid_pos = (int(cur_x), int(cur_y))
                visit_count_map[grid_pos] = visit_count_map.get(grid_pos, 0) + 1
                position_history.append(grid_pos)
                short_visits = sum(1 for p in position_history if p == grid_pos)
                if short_visits >= 3 and not is_demo:
                    loop_penalty        = -0.3 * (short_visits - 2)
                    accumulated_reward += loop_penalty
                    episode_reward     += loop_penalty

                # 步数超限强制截断
                if episode_steps >= max_episode_steps and not is_demo:
                    if last_state_vec is not None and last_action is not None:
                        timeout_penalty    = -10.0
                        accumulated_reward += timeout_penalty
                        episode_reward     += timeout_penalty
                        replay_buffer.append((last_state_vec, last_action, accumulated_reward, current_state_vec, 1.0))
                    print(f"  [DQN] EP {episode_count+1} 超过步数上限 ({max_episode_steps})，强制截断")
                    accumulated_reward = 0.0
                    last_state_vec     = None
                    last_action        = None
                    position_history.clear()
                    visit_count_map.clear()
                    write_to_shared_memory(f"ACTION:0|{epsilon:.4f}")
                    last_handled_info["data"]   = data
                    last_handled_info["step"]   = int(state_info[3]) if len(state_info) > 3 and state_info[3] != '' else -1
                    last_handled_info["coords"] = (cur_x, cur_y)
                    continue

                action         = decide_action(current_state_vec)
                last_state_vec = current_state_vec
                last_action    = action
                episode_steps += 1
                write_to_shared_memory(f"ACTION:{action}|{epsilon:.4f}")
                last_handled_info["data"]   = data
                last_handled_info["step"]   = int(state_info[3]) if len(state_info) > 3 and state_info[3] != '' else -1
                last_handled_info["coords"] = (cur_x, cur_y)
                continue

            # ====== REWARD：终端奖励（到达终点） ======
            elif data.startswith("REWARD:"):
                reward_info = data[7:].split('|')
                try:
                    r = float(reward_info[0]) if reward_info[0] else 0.0
                except ValueError:
                    r = 0.0
                rtype = int(reward_info[3]) if len(reward_info) > 3 else 0
                accumulated_reward += r; episode_reward += r
                if rtype == 2:  # 到达终点：写入终止样本并强化学习 5 次
                    if last_state_vec is not None and last_action is not None:
                        current_state_vec = get_state_vector(maze_w - 2, maze_h - 2)
                        replay_buffer.append((last_state_vec, last_action, accumulated_reward, current_state_vec, 1.0))
                        for _ in range(5):
                            learn(batch_size)
                    print(f"★ [通关] DQN | 局:{episode_count+1} | 步:{episode_steps} | 分:{episode_reward:.1f} ★")
                    # 清零防止下一局首帧将"终点→起点"写入 buffer
                    last_state_vec     = None
                    last_action        = None
                    accumulated_reward = 0.0
                last_handled_info["data"]   = data
                last_handled_info["step"]   = -1
                last_handled_info["coords"] = (0, 0)

        time.sleep(0.005)

except Exception as e:
    import traceback
    crash_log_path = os.path.join(save_dir, "CRASH_LOG.txt")
    with open(crash_log_path, "w") as f:
        f.write(traceback.format_exc())
    print(f"CRASHED! Wrote log to {crash_log_path}")
    time.sleep(10)
