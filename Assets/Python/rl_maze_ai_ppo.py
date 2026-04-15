# -*- coding: utf-8 -*-
"""
PPO (Proximal Policy Optimization) 强化学习迷宫 AI 后端
=========================================================
该文件使用了近端策略优化算法，这是目前大模型训练对齐甚至高级游戏AI最主流最稳定的一种算法。
不同于 DQN 的值估计法，PPO 这种 Actor-Critic (演员-评委模式) 的直接策略优化法具有更稳定平滑的学习曲线。
本页涵盖：
1. RollOutBuffer: 保存和整顿走一小段期间内的一串连续状态链。
2. Actor-Critic: 包含生成四行动可能性的分布 Actor 层和估算未来总成绩的 Critic 层。
3. K_epochs: 通过在旧有缓冲带多次重复回放榨取样本价值，通过裁剪幅度（Clipping）防止训脱轨。
"""

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
    print("[Python AI] ERROR: PyTorch 尚未安装。请运行 'pip install torch numpy' 以支持 PPO 算法。")
    import sys
    sys.exit(1)

# ========== 共享内存设置 (IPC通信机制) ==========
shared_memory_name = "MazeRLSharedMemory"
data_size = 4096
try:
    mm = mmap.mmap(-1, data_size, tagname=shared_memory_name)
except Exception as e:
    print(f"[Python AI] ERROR: 无法打开共享内存 {shared_memory_name}: {e}")
    import sys
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Python AI] 使用计算设备: {device.type.upper()}")


# ========== PPO 经验记忆体 ==========
class RolloutBuffer:
    """
    PPO 基于同策略 (On-Policy)，不支持将八百年前跟历史策略杂交产生的经验捞过来用(像DQN那样)。
    必须要求：我最新的兵力亲自下场探雷跑回来的数据立刻作为复盘结算带！算完必须清空，然后拿着新策略再次上阵重头攒经验。
    """
    def __init__(self):
        self.actions = []     # 从分布中随机抽出来的具体决策记录
        self.states = []      # 当时的所见所闻画面
        self.logprobs = []    # 那一刹那产生该项动作的策略对数概率 (用来和未来更好的策略比值算 Ratio用的)
        self.rewards = []     # 环境当时给的反馈激励大饼
        self.is_terminals = []# 标记这儿是不是摔死了无底洞还是到了胜利彼岸
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

# ========== PPO 网络结构 (Actor-Critic) ==========
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # 【演员网络 (Actor)】：专门负责给出你“下一步应该干嘛”的建议。
        # 会给每个动作评委派一个选择"可能性(概率)"。用 Softmax 规范在 100% 概率和内。
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),  # Tanh 会由于对称性让动作概率收敛变得圆滑无强烈偏移特征。
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # 【评委网络 (Critic / Value Net)】：专门负责预判此时此刻此情此地能有多高的分出路。
        # 无需在意能走哪个行动，单纯估价你的生存处境。输出一个干巴巴浮点数值（State Value）。
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        """为环境中的当下探索给出一个符合概率分布随机丢骰子出来的抉择"""
        if state.dim() == 1:
            state = state.unsqueeze(0) # 修正出 [1, ...] 的批处理流外壳
            
        action_probs = self.actor(state)
        # 用这个动作的多项分布概率建一个筛子结构（类别分布器）
        dist = Categorical(action_probs)
        # 丢骰子摇一个（如果上=0.9, 下=0.1，大概率会丢出来"上"，但并非绝对）
        action = dist.sample()
        # 顺便提炼出发生这个巧合的其原始真实对数概率封存进缓冲区以备后验
        action_logprob = dist.log_prob(action)
        
        return action.item(), action_logprob.item()

    def evaluate(self, state, action):
        """后见之明审判阶段：拿着以前那一批干的一摞子勾当的数据，用现在的目光给当时审判一番以做纠正基底"""
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        # 信息熵 (Entropy)：这代表当时你做这事是属于成竹在胸、破釜沉舟，还只是迷茫均匀分发概率全看蒙的浑水摸鱼？(Entropy 很大代表什么都想试但全不专精)
        dist_entropy = dist.entropy()
        
        # 评委来估量此状态当时的得分空间
        state_values = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_values), dist_entropy


class PPO:
    """PPO 包装内核组件"""
    def __init__(self, input_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.gamma = gamma          # 未来折旧率
        self.eps_clip = eps_clip    # 信任域裁剪范围比值。超过0.2等被认为是跨出安全阈值的危险激进改动，斩！
        self.K_epochs = K_epochs    # “榨汁机轮次”。一批新的走完后，重播倒带在这批数据上反复咀嚼的次数。
        
        self.policy = ActorCritic(input_dim, action_dim).to(device)
        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        
        # 保存稳定的老策略，用于计算 PPO 的概率修正截断比 (Ratio) 以不至于新出炉的政策跳变偏离太多
        self.policy_old = ActorCritic(input_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state_vec, buffer):
        """走一个常规步骤被调用的选子入口"""
        with torch.no_grad():
            state = torch.FloatTensor(state_vec).to(device)
            # 老手政策网络给出实操指示与概率
            action, action_logprob = self.policy_old.act(state)
        
        # 把该存的录像留证封包
        buffer.states.append(state_vec)
        buffer.actions.append(action)
        buffer.logprobs.append(action_logprob)
        return action

    def update(self, buffer):
        """神经网络的集中大清算、大回查、与大更新时代。在此处真正改变它的脑子。"""
        if len(buffer.states) == 0:
            return
            
        # ---------- 1. Monte Carlo 估算法算出整条时光链上的预期总收益 ----------
        rewards = []
        discounted_reward = 0
        # 时光倒流 (Reversed)，从结局终点往往前回推估值，倒一倒就能把后面那一步通关奖励光速发放到整个前期入口路径之上。
        for reward, is_terminal in zip(reversed(buffer.rewards), reversed(buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # 容错降级对齐保证 (如果共享内存通信由于断层丢了信息，直接暴力斩掉这截留存信息防引发崩溃)
        if len(rewards) != len(buffer.states):
            print(f"[PPO Warning] Buffer size mismatch! states={len(buffer.states)}, rewards={len(rewards)}")
            min_len = min(len(rewards), len(buffer.states), len(buffer.actions), len(buffer.logprobs), len(buffer.is_terminals))
            buffer.states = buffer.states[:min_len]
            buffer.actions = buffer.actions[:min_len]
            buffer.logprobs = buffer.logprobs[:min_len]
            buffer.is_terminals = buffer.is_terminals[:min_len]
            rewards = rewards[:min_len]
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        old_states = torch.FloatTensor(np.array(buffer.states)).to(device)
        old_actions = torch.LongTensor(buffer.actions).to(device)
        old_logprobs = torch.FloatTensor(buffer.logprobs).to(device)
        
        # ---------- 2. 【核心数学修复】：提前算好广义优势(Advantage) ----------
        # 优势函数 = 那时候根据时间倒流算出的【实际最终到手奖励】 - 当时本方大评委瞎蒙的【估算本该有多少价值】
        # 优势就是衡量“哇，这件事比我以前预估的要好太多啦！”。如果为正，说明此事超预期，应该要大力推崇加概率！
        with torch.no_grad():
            old_state_values = self.policy.critic(old_states).squeeze()
            if old_state_values.dim() == 0:
                old_state_values = old_state_values.unsqueeze(0)
                
        advantages = rewards - old_state_values
        
        # 标准化 Advantage 以约束极其狂暴或极为负面的异常奖励震荡
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
            
        # ---------- 3. PPO 核心截断大盘优化 (榨取汁液期) ----------
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # 算出与以前这个决定老网络之间的比率 (Ratio)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # PPO 的截断优势目标函数 (Clipping Objective): TRPO近端改进法的神来之笔。
            # 如果事情搞砸了但网往反向超出了限幅(变了心)，用 clamp(截断) 无情将大过那条限度的梯度作废掉，不准瞎变太歪。
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # ===== 综合裁判系统扣减发功 =====
            # 演员损失 (尽可能最大化优势期望比)：所以前面加个负号 - 取 min
            # 评委估价损失 (MSE:要求尽可能接近事实发放的奖励数目使得眼睛变雪亮)
            # 信息熵增强红利：拿走一点扣去，当成红包给它，激励它就算知道去哪也多去探几脚未知盲区免成死循环傻子。
            loss = -torch.min(surr1, surr2).mean() + 0.5 * self.MseLoss(state_values.squeeze(), rewards) - 0.05 * dist_entropy.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            # 【关键护符防魔化】：防止因为深层复杂网络和乘积比率产生的爆炸梯度梯度，超过 0.5 的全压下去！
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            
        # 更新终于完事，令现在的决策作为未来前去老策略版本存根
        self.policy_old.load_state_dict(self.policy.state_dict())
        buffer.clear()

# ========== 核心控制器超参数 ==========
# 由于是 On-Policy 策略方法，不需要一上来就把 100% 当瞎眼摸象。但 PPO 对参数的配合极其吹毛求疵。
# 以下是根据迷宫场景微调过的安全界参数
update_timestep = 500       # 积攒 500 次环境交互样本之后再放行去“消化”学习
K_epochs = 5                # 【稳定性把控】：一次批样本咀嚼 5 遍结束了拉倒，防止它只记住了这批碰壁倒霉经历然后不敢动弹(即"过度拟合导致的自闭不走路现象")
eps_clip = 0.2              # 信任域半径 20%
gamma = 0.99                # 折扣率：高度重视长远眼光(到终点的路有时需要兜很久大弯，如果不拉长到 0.99 会因为沿路每步-0.01被扣怕了而在墙角画圈)
lr_actor = 0.0003           # PPO 行规 黄金老演员学习率 (如果敢改大到 0.01 它立刻就飞升死在地图角落)
lr_critic = 0.0006          # 评委学习率放长线

ppo_agent = None
buffer = RolloutBuffer()

maze_grid = None
maze_w, maze_h = 0, 0
current_seed = 0
last_action = None

accumulated_reward = 0.0
episode_reward = 0.0
episode_steps = 0
episode_count = 0
global_steps = 0
episode_hit_count = 0

is_demo = False
best_reward = -999999.0

# ========== 存档与工具 ==========
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data~")
os.makedirs(save_dir, exist_ok=True)
log_path = os.path.join(save_dir, "ppo_training_log_default.csv")
model_path = os.path.join(save_dir, "ppo_model_default.pth")

def init_log_file():
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "steps", "hit_count", "total_reward"])

def save_training_log(ep, steps, hit_count, reward):
    if not os.path.exists(log_path):
        init_log_file()
    with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([ep, steps, hit_count, f"{reward:.2f}"])

def resume_training_state():
    """断点续训"""
    global episode_count, best_reward
    if os.path.exists(log_path):
        try:
            last_ep = 0
            best_so_far = -999999.0
            with open(log_path, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                next(reader) 
                for row in reader:
                    if len(row) >= 4:
                        last_ep = int(row[0])
                        r = float(row[3])
                        if r > best_so_far:
                            best_so_far = r
            if last_ep > 0:
                episode_count = last_ep
                best_reward = best_so_far
                print(f"[Python AI] 断点续训：从 EP {episode_count} 继续，当前历史最高分 {best_reward:.1f}")
                return
        except Exception as e:
            print(f"[Python AI] 读取训练日志失败: {e}")
    episode_count = 0
    print("[Python AI] 未发现历史训练数据，从头开始训练。")

def build_network(w, h):
    """初始化装载 Actor-Critic 构造基体核心双模块网络"""
    global ppo_agent
    input_size = w * h + 2 # 环境打扁格数特征量 + 高级相对X/Y位置双通道
    ppo_agent = PPO(input_size, 4, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
    print(f"[Python AI] 已初始化 PPO (Actor-Critic) 双通道网络, 输入维度: {input_size}")

def save_model(is_best=False):
    """将参数落盘，留存最佳"""
    if ppo_agent is not None:
        path = model_path
        if is_best:
            path = model_path.replace(".pth", "_best.pth")
        torch.save(ppo_agent.policy_old.state_dict(), path)
        if is_best:
            print(f"[Python AI] !!! 发现 PPO 最佳表现，模型已锁定: {os.path.basename(path)}")

def load_model(use_best=False):
    """从档案提取历史记忆附着网络"""
    if ppo_agent is not None:
        target_path = model_path
        if use_best:
            best_path = model_path.replace(".pth", "_best.pth")
            if os.path.exists(best_path):
                target_path = best_path
        if os.path.exists(target_path):
            try:
                ppo_agent.policy_old.load_state_dict(torch.load(target_path, map_location=device))
                ppo_agent.policy.load_state_dict(ppo_agent.policy_old.state_dict())
                print(f"[Python AI] 成功加载 PPO 模型: {os.path.basename(target_path)}")
            except Exception as e:
                print(f"[Python AI] 模型加载失败: {e}")


def read_from_shared_memory():
    mm.seek(0)
    return mm.read(data_size).decode('utf-8').strip('\x00').strip()

def write_to_shared_memory(data):
    try:
        mm.seek(0)
        padded_data = data.ljust(data_size, '\x00')
        mm.write(padded_data.encode('utf-8'))
    except Exception as e:
        print(f"[WRITE ERROR] Failed to write to shared memory: {e}")

def get_state_vector(x, y):
    """状态环境融合器"""
    if maze_grid is None:
        return np.zeros(1, dtype=np.float32)
    state = np.copy(maze_grid).astype(np.float32)
    state[state == 1] = -1.0  # 定义厚实墙壁为不可逾越且不喜吸引之物 -1
    
    iy = min(max(int(y), 0), maze_h - 1)
    ix = min(max(int(x), 0), maze_w - 1)
    state[iy, ix] = 1.0       # 定义目前我在所处位置的高亮点，亮如白夜以指引前路 1
    
    flat_state = state.flatten()
    coord_info = np.array([x / float(max(1, maze_w)), y / float(max(1, maze_h))], dtype=np.float32)
    
    return np.concatenate((flat_state, coord_info))

# ========== PPO 训练大循环 (消息调度拦截) ==========
print("[Python AI] PPO Backend Ready. Waiting for IPC Connection...")
write_to_shared_memory("READY")
last_handled_info = {"data": "READY", "step": -1, "coords": (0, 0)}

try:
    while True:
        try:
            data = read_from_shared_memory()
        except:
            time.sleep(0.01)
            continue
            
        if data and data != "WAITING":
            # ====== 防卡墙冗余状态过滤 ======
            if data.startswith("STATE:"):
                try:
                    state_parts = data[6:].split('|')
                    curr_step = int(state_parts[3])
                    curr_coords = (float(state_parts[0]), float(state_parts[1]))
                    if data == last_handled_info["data"] and curr_step == last_handled_info["step"] and curr_coords == last_handled_info["coords"]:
                        continue # 被拦截下了，此状态无需让其复读产生垃圾冗复包。
                except:
                    pass
            elif data == last_handled_info["data"]:
                continue 
                
            # ====== GRID 地图挂载分发 ======
            if data.startswith("GRID:"):
                parts = data[5:].split('|')
                try:
                    maze_w, maze_h = int(parts[0]), int(parts[1])
                    if len(parts) > 3:
                        current_seed = int(parts[2])
                        
                        # 按尺寸隔离模型。PPO若接受了长度不一的状态数组会当场触发矩阵乘法灾难奔溃。必须做以物理区分。
                        log_path = os.path.join(save_dir, f"ppo_log_s{maze_w}_{current_seed}.csv")
                        model_path = os.path.join(save_dir, f"ppo_model_s{maze_w}_{current_seed}.pth")
                        
                        grid_vals = [int(x) for x in parts[3].split(',')]
                        grid_array = np.array(grid_vals)
                        maze_grid = grid_array.reshape((maze_h, maze_w))
                        
                        build_network(maze_w, maze_h)
                        buffer.clear()
                        best_reward = -999999.0 # 换图大洗牌不认历史成绩记录
                        
                        print(f"[PPO GRID] {maze_w}x{maze_h} loaded, seed={current_seed}")
                    else:
                        print(f"[PPO GRID] Format error")
                except Exception as e:
                    print(f"[PPO GRID] Parse error: {e}")
                    
                write_to_shared_memory("GRID_OK")
                last_handled_info["data"] = "GRID_OK"
                
            # ====== PARAM C#传略参数干涉 ======
            elif data.startswith("PARAM:"):
                params = data[6:].split(',')
                gamma = float(params[1])
                if len(params) > 5 and ppo_agent is not None:
                    # 【稳定性抢修线】：严加约束由于用户或者C#在界面把学习率拉太高而毁了整个策略网络的恶劣影响机制
                    lr_base = float(params[5])
                    lr_actor = min(lr_base * 0.2, 0.0003)    # 锁死最高不能超过 0.0003 (PPO黄金敏感收敛线)否则出局死跑
                    lr_critic = min(lr_actor * 2, 0.001)     # 评委可以大胆一些激进给分
                    for param_group in ppo_agent.optimizer.param_groups[0:1]: param_group['lr'] = lr_actor
                    for param_group in ppo_agent.optimizer.param_groups[1:2]: param_group['lr'] = lr_critic
                last_handled_info["data"] = data
                
            # ====== COMMAND 后门与模式切换主命令 ======
            elif data.startswith("COMMAND:"):
                cmd_full = data[8:]
                cmd_parts = cmd_full.split('|')
                cmd = cmd_parts[0]
                
                if cmd == "QUIT" or cmd == "EXIT":
                    print("[Python AI] 终端下线，进程自裁。")
                    break
                    
                elif cmd == "DEMO":
                    load_model(use_best=True)  # 请出本组最好选手
                    is_demo = True
                    buffer.clear()             # 它的行动不再录入经验作为日后教导
                    episode_count = 0
                    print(f"-> PPO DEMO MODE: Loaded Best Weights")
                    
                elif cmd == "START":
                    if ppo_agent is None and maze_grid is not None:
                        build_network(maze_w, maze_h)
                    load_model(use_best=False) # 接着苦练爬阶
                    is_demo = False
                    buffer.clear()
                    global_steps = 0
                    init_log_file()
                    resume_training_state()
                    last_action = None
                    print(f"-> PPO TRAINING START (Map: {current_seed}) Resuming from EP {episode_count}")
                    
                elif cmd == "RESET":
                    # 【核心补救措施1】处理 C# 端由于死亡/碰壁到达重下发 RESET 前还没来得及把当时的死后奖惩并入缓冲池的问题
                    if last_action is not None and not is_demo:
                        buffer.rewards.append(accumulated_reward)
                        buffer.is_terminals.append(False) # 中途死亡也强忍不算作为强行终止来惩罚它，让它继续有活念！
                        accumulated_reward = 0.0
                        last_action = None

                    # 【核心架构重定位】：
                    # DQN 可以做到一边获取事件一步立马更新。PPO 的 On-Policy 更新极其厚重(近一秒耗费计算)。
                    # 若放在 STATE/REWARD 获取触发条件了就直接原地阻塞死板的计算，那就完了。
                    # 等这段时间算完，由于 Unity 没歇着，他发现没发 ACTION 但小球超死亡生命上限了自动洗成了出生地坐标
                    # 直接把上一回合的记录内存生生覆盖，导致 Python 直接丢了一整局最后的数据从而把评委带坑里去。
                    # 因此：它必须在这 C# 等它出结果(也就是等其下发指令的 RESET 站岗时期)来进行一次更新倾倒以换取绝对安全不丢帧。
                    if not is_demo and len(buffer.rewards) >= min(update_timestep // 5, 20):
                        print(f"[PPO Update] 步长周期积累完结且收到截流冲洗 (RESET触发)，使用 {len(buffer.rewards)} 个印记开算神经网络并升级脑回路！")
                        ppo_agent.update(buffer)
                        
                    hit_count_this_ep = int(cmd_parts[1]) if len(cmd_parts) > 1 else 0
                    episode_count += 1
                    
                    if not is_demo:
                        save_training_log(episode_count, episode_steps, hit_count_this_ep, episode_reward)
                        if episode_reward > best_reward: # 功碑系统判断留用最佳模型文件
                            best_reward = episode_reward
                            save_model(is_best=True)
                        else:
                            save_model(is_best=False)
                            
                    print(f"  EP {episode_count} | Steps: {episode_steps} | Hits: {hit_count_this_ep} | Reward: {episode_reward:.1f} | Best: {best_reward:.1f}")
                    last_action = None
                    accumulated_reward = 0.0
                    episode_reward = 0.0
                    episode_steps = 0
                    
                last_handled_info["data"] = data
                
            # ====== STATE PPO推导核心运转期 ======
            elif data.startswith("STATE:"):
                if ppo_agent is None:
                    write_to_shared_memory("ACTION:0")
                    continue
                    
                state_info = data[6:].split('|')
                # 破损数据废除补截处理措施
                if len(state_info) < 2 or state_info[0] == '' or state_info[1] == '':
                    continue
                try:
                    cur_x, cur_y = float(state_info[0]), float(state_info[1])
                except ValueError:
                    print(f"[PPO] 跳过损坏的 STATE 数据: {data[:50]}")
                    continue
                    
                # 【汇总奖品】
                if len(state_info) >= 5 and state_info[4] != '':
                    try:
                        r = float(state_info[4])
                        accumulated_reward += r
                        episode_reward += r
                    except ValueError:
                        pass
                        
                # ==== 录入上一帧结果进行历史存档封装 ====
                if last_action is not None and not is_demo:
                    buffer.rewards.append(accumulated_reward)
                    buffer.is_terminals.append(False)
                    
                accumulated_reward = 0.0
                
                # ==== 倒车入库前的例行检查。看手上的料在跑的路段中是否够一次开火标准去升级 ====
                if not is_demo and len(buffer.rewards) > 0 and global_steps % update_timestep == 0:
                    print(f"[PPO Update] 漫步已满 {global_steps} 触发此段落阶段性脑中风更新策略修正！")
                    ppo_agent.update(buffer)
                    
                # ==== 提取特征 ====
                current_state_vec = get_state_vector(cur_x, cur_y)
                
                # ==== 根据身份推选行动走向 ====
                if is_demo:
                    # 如果这只是单纯的一场面向观众的展示：那么不需要抛点子选概率带信息熵搞试错！直接取其输出里面最稳最高最大的即可！以防丢丑！
                    with torch.no_grad():
                        state = torch.FloatTensor(current_state_vec).unsqueeze(0).to(device)
                        action_probs = ppo_agent.policy_old.actor(state)
                        action = torch.argmax(action_probs).item()
                else:
                    # 如果是在艰苦训练：进入抛点子函数以按照当前网络出具的四大路向的胜券分布概率，盲抽产生一条出路。
                    action = ppo_agent.select_action(current_state_vec, buffer)
                    
                last_action = action
                episode_steps += 1
                global_steps += 1
                
                write_to_shared_memory(f"ACTION:{action}")
                
                last_handled_info["data"] = data
                last_handled_info["step"] = int(state_info[3]) if len(state_info) > 3 and state_info[3] != '' else -1
                last_handled_info["coords"] = (cur_x, cur_y)
                continue
                
            # ====== REWARD 极高纯度终端奖项判定区 ======
            elif data.startswith("REWARD:"):
                reward_info = data[7:].split('|')
                try:
                    r = float(reward_info[0]) if len(reward_info) > 0 and reward_info[0] != '' else 0.0
                except ValueError:
                    r = 0.0
                rtype = int(reward_info[3]) if len(reward_info) > 3 else 0
                accumulated_reward += r
                episode_reward += r
                
                if rtype == 2: # 它已登顶：这代表一次不可忽视的终端完结传奇
                    if last_action is not None and not is_demo: 
                        buffer.rewards.append(accumulated_reward)
                        buffer.is_terminals.append(True) # True打断标记表示此路已抵达辉煌顶点，不要指望再往后面有牵扯的后续路数折算了。
                    accumulated_reward = 0.0
                    last_action = None
                    
                last_handled_info["data"] = data
                last_handled_info["step"] = -1
                last_handled_info["coords"] = (0, 0)
                
        time.sleep(0.005)

except Exception as e:
    import traceback
    crash_log_path = os.path.join(save_dir, "CRASH_LOG.txt")
    with open(crash_log_path, "w") as f:
        f.write(traceback.format_exc())
    print(f"CRASHED! Wrote log to {crash_log_path}")
    time.sleep(10)