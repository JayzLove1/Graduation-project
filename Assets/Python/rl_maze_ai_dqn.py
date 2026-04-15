# -*- coding: utf-8 -*-
"""
DQN (Dueling Double Deep Q-Network) 强化学习迷宫 AI 后端
=========================================================
本文件是基于深度学习的 Q-learning 扩展。
使用 PyTorch 搭建神经网络来逼近 Q 值函数，从而解决高维状态空间问题。
主要特性涵盖：
1. Dueling Architecture (竞争架构网络)：将 Q值 拆分为 状态价值(V) 和 动作优势(A)。
2. Double DQN：使用两个网络 (Policy & Target) 分离选择动作和评估动作，防止过度估计。
3. Experience Replay (经验回放池)：打乱记忆的时间顺序，打破长期马尔可夫耦合性，提高样本利用率。
"""

import mmap
import time
import random
import os
import csv
import numpy as np
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except ImportError:
    print("[Python AI] ERROR: PyTorch 尚未安装。请打开终端运行 'pip install torch numpy' 以支持 DQN 算法。")
    import sys
    sys.exit(1)

# ========== 共享内存设置 (IPC) ==========
shared_memory_name = "MazeRLSharedMemory"
data_size = 4096
try:
    mm = mmap.mmap(-1, data_size, tagname=shared_memory_name)
except Exception as e:
    print(f"[Python AI] ERROR: 无法打开共享内存 {shared_memory_name}: {e}")
    import sys
    sys.exit(1)

# ========== 网络结构 (Dueling DQN) ==========
# 传统 DQN 输出整合成一个Q值列表。
# Dueling DQN 将网络在中途切分成两支，再在最后合并：
#  - V层 (Value Stream): 当前这个格子好不好 (比如这格子离终点近不近)
#  - A层 (Advantage Stream): 在这个格子上走向各个方向的好处差了多少 (往墙撞跟往空地走区别多大)
class DQNLayer(nn.Module):
    def __init__(self, input_size, num_actions=4):
        super(DQNLayer, self).__init__()
        # 公共特征抽取层 (全连接多层感知机)
        self.feature = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 状态价值头 (Value Stream): 标量输出，判断所处环境的综合价值
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 动作优势头 (Advantage Stream): 向量输出，判断做每个动作的独家优势
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        # 提取共同特征
        x = self.feature(x)
        # 双轨分支
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # 数学合并层设计：Q(s, a) = V(s) + ( A(s, a) - 1/|A| * sum(A(s, a')) )
        # 减去 advantage 的均值，是为了确保 V(s) 的绝对地位不被偏置掩盖，增强训练稳定性。
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

# ========== 核心超参数配置 ==========
lr = 0.001           # 神经网络初始学习率 (优化器步长)
gamma = 0.95         # 贝尔曼折旧因子 (越接近1，网络越"目光长远") -- 0.95能让终点信号传播更远
epsilon = 1.0        # Epsilon-Greedy 的探索随机率
epsilon_decay = 0.995 # 随机率退火衰减乘值
min_epsilon = 0.05   # 全局保留的最小鲁棒性探索率
batch_size = 64      # 每次做梦(学习)从经验池重抽取切片的批量大小 (Batch Size)

target_update_freq = 100 # C 步频：每隔100步，将政策网络硬拷贝一份给目标网络，制造延迟，防振荡反馈循环。

# 网络双核架构声明 (Double Q-Learning 的防高估机制)
policy_net = None # Q_eval，负责在游戏里直接指引动作的主网络
target_net = None # Q_next，只在学习更新时使用，像影子一样慢半拍，用来评估新状态价值
optimizer = None

# DQN 的记忆中枢：高达20000条记录的回放池
replay_buffer = deque(maxlen=20000)

# ========== 状态环境追踪缓存 ==========
maze_grid = None
maze_w, maze_h = 0, 0
current_seed = 0

last_state_vec = None      # 上一帧送入深度网络的张量(Tensor)矩阵向量
last_action = None
accumulated_reward = 0.0
episode_reward = 0.0
episode_steps = 0
episode_count = 0
global_steps = 0           # 历史总步数跟踪器(用于决定何时刷新 Target 网络)
episode_hit_count = 0
is_demo = False            # 是否处于给人类展示的"闭卷表现(Demo)"模式
best_reward = -999999.0

# ========== 循环检测与防原地踏步 ==========
position_history = deque(maxlen=50)  # 最近50步的位置记录
visit_count_map = {}                 # 全局计数: 每个格子被踩多少次
max_episode_steps = 10000            # 单局步数上限, 收到GRID后动态调整

# ========== 持久化存储与日志目录 ==========
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data~")
os.makedirs(save_dir, exist_ok=True)
log_path = os.path.join(save_dir, "dqn_training_log_default.csv")
model_path = os.path.join(save_dir, "dqn_model_default.pth")

# 启用 Nvidia GPU 并行加速，否则使用 CPU fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Python AI] 使用计算设备: {device.type.upper()}")

# ========== 日志与数据恢复工程 ==========
def init_log_file():
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="", encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "steps", "hit_count", "total_reward", "epsilon"])

def save_training_log(ep, steps, hit_count, reward, eps):
    if not os.path.exists(log_path):
        init_log_file()
    with open(log_path, "a", newline="", encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([ep, steps, hit_count, f"{reward:.2f}", f"{eps:.4f}"])

def resume_training_state():
    """断点续训：防止奔溃导致丢失从几万局之后开始训练的机会，直接衔接上次的回合计数与衰减进度"""
    global episode_count, epsilon, best_reward
    if os.path.exists(log_path):
        try:
            last_ep = 0
            last_eps = epsilon
            best_so_far = -999999.0
            with open(log_path, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                next(reader)  # 跳过表头
                for row in reader:
                    if len(row) >= 5:
                        last_ep = int(row[0])
                        r = float(row[3])
                        last_eps = float(row[4])
                        if r > best_so_far:
                            best_so_far = r
            if last_ep > 0:
                episode_count = last_ep
                epsilon = last_eps
                best_reward = best_so_far
                print(f"[Python AI] 断点续训：从 EP {episode_count} 继续，历史最高分 {best_reward:.1f}，当前 Epsilon={epsilon:.4f}")
                return
        except Exception as e:
            print(f"[Python AI] 读取训练日志失败: {e}")
    episode_count = 0
    epsilon = 1.0
    print("[Python AI] 未发现历史训练数据，从头开始训练。")

# ========== PyTorch 网络构建与实例化 ==========
def build_network(w, h):
    """根据给定的地图尺寸建立全连接图视觉拓扑层"""
    global policy_net, target_net, optimizer
    # 输入维度：包含完整的迷宫特征摊平后的阵列大小 + 附加提供的两个归一化相对角色位置参数 (x, y)
    input_size = w * h + 2
    policy_net = DQNLayer(input_size, 4).to(device)
    target_net = DQNLayer(input_size, 4).to(device)
    
    # Target(影子基准评估网络) 出具时复刻主核心参数，然后永远死扛在 eval 模式（不允许自动更新反向传播梯度）
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # 启用强大的自适应矩估计器负责梯度下降
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    print(f"[Python AI] 已初始化 Dueling DQN 网络, 输入维度: {input_size} ({w}x{h})")

def save_model(is_best=False):
    """将整个权重 Tensor 落盘到 .pth 实体文件保存"""
    if policy_net is not None:
        path = model_path
        if is_best:
            path = model_path.replace(".pth", "_best.pth")
        torch.save(policy_net.state_dict(), path)
        if is_best:
            print(f"[Python AI] !!! 发现历史最高分，已更新最佳模型: {os.path.basename(path)}")

def load_model(use_best=False):
    if policy_net is not None:
        target_path = model_path
        if use_best:
            best_path = model_path.replace(".pth", "_best.pth")
            if os.path.exists(best_path):
                target_path = best_path
        if os.path.exists(target_path):
            try:
                # 以跨卡兼容方式 map_location=device 加载张量
                policy_net.load_state_dict(torch.load(target_path, map_location=device))
                target_net.load_state_dict(policy_net.state_dict())
                print(f"[Python AI] 已成功加载模型权重: {os.path.basename(target_path)}")
            except Exception as e:
                print(f"[Python AI] 模型加载失败: {e}")
        else:
            print(f"[Python AI] 未找到匹配的模型缓存 ({os.path.basename(target_path)})，采用随机初始化结构")

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

# ========== 核心前传：视野环境建模 ==========
def get_state_vector(x, y):
    """
    状态转录器：把迷宫在内存中的结构和角色的当前坐标"揉"在一起，制作成一根提供给神经网络吞进去的特征长面条。
    """
    if maze_grid is None:
        return np.zeros(1, dtype=np.float32)
    # 将包含 0(空地)/1(障碍物) 的图缩印至 state 中。    
    state = np.copy(maze_grid).astype(np.float32)
    state[state == 1] = -1.0 # 墙壁映射为 -1 防止其特征值正数化被当成奖励激活元
    
    # 放置角色的"当前落脚点"特征（防越界兜底）
    iy = min(max(int(y), 0), maze_h - 1)
    ix = min(max(int(x), 0), maze_w - 1)
    state[iy, ix] = 1.0
    
    # 展平矩阵，并补增精密的当前(X,Y)相对坐标系统。这步可以极大化改善全连接层"路痴症"的隐患。
    flat_state = state.flatten()
    coord_info = np.array([x / float(max(1, maze_w)), y / float(max(1, maze_h))], dtype=np.float32)
    return np.concatenate((flat_state, coord_info))

def decide_action(state_vec):
    """推断层：负责使用最新的策略网络或通过蒙特卡洛抽彩选动作。"""
    if random.random() < epsilon:
        # ------- Epsilon 随机盲目试错 -------
        choices = [0, 1, 2, 3]
        # 防呆机制：瞎蒙时候也防一手倒着往后跑的鬼畜复读循环问题，禁止走上一步的原方向
        if last_action is not None and random.random() < 0.8:
            rev_map = {0:1, 1:0, 2:3, 3:2}
            rev = rev_map.get(last_action, -1)
            if rev in choices: choices.remove(rev)
        return random.choice(choices)
    else:
        # ------- Q网络 睿智的推断（剥削） -------
        with torch.no_grad(): # 切记挂载该标志，不参与反向求导计算记录防止显存放崩内存！
            policy_net.eval()
            # 从 numpy 单管转换为 Pytorch 可吃的含 batch=1 维度的张量 [1 , 全特征规模]
            t_state = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
            # 送给它，得到他给那四个路口方向作出的四股评价分：
            q_vals = policy_net(t_state)
            return q_vals.argmax().item()

# ========== 决策训练模块：贝尔曼错觉反馈 =========
def learn(batch_size):
    """
    大名鼎鼎的 Double DQN + 回放池机制网络优化算法核心。
    """
    # 池子小过我们一勺大小了(样本太单一缺乏泛华代表性)，退回等一会
    if len(replay_buffer) < batch_size or policy_net is None:
        return
        
    # 从过去经历的时间长河里随意抽出相互不关联的 64 个时刻碎片：
    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    # 强制转移成能使用算力的张量矩阵集群：
    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones = torch.FloatTensor(dones).unsqueeze(1).to(device) # dones 标志此景是否已入绝境终局！ 1为是。
    
    # Policy_Net 算出当初在这个时候，采取这一具体行动能带来的当前模型估算价值 (这就是我们需要训练它矫正的被修者：Current Q)
    q_values = policy_net(states).gather(1, actions)
    
    # ========== [Double DQN 的精髓机制] ==========
    # 最初的原版 DQN 容易过度膨胀估值导致模型高估崩盘。
    # Double DQN 解决方法把动作的"选拔"和"计价"分开：
    with torch.no_grad():
        # 1. 甄选权交向比较活跃的主推网络 Policy_Net：在抵达 next_states 之后选出一个在它看来的最强前途行动。
        next_actions = policy_net(next_states).max(1)[1].unsqueeze(1)
        
        # 2. 质询权交向被冻结较慢跟在屁股后头防骄傲的 Target_Net：给这个选出来的举动当"纪检委老手评估者"算个相对客观稳定的价格。
        next_q_values = target_net(next_states).gather(1, next_actions)
        
        # 最终依靠强大的贝尔曼公式组合出 Target Q (正确标答引导标枪)：
        # 如果达到了终点（Done = 1），则下周期的妄想全部切断，因为它是此宇宙回合结局没有任何"如果还有明天"的收益了。
        target_q_values = rewards + gamma * next_q_values * (1 - dones)
        
    # 计算实际预测与基准标答间的差平方距离（均方误差）
    loss = F.mse_loss(q_values, target_q_values)
    
    # 下发给 Adam，从图上溯源找到全连接层把梯度的锅全部分掉更新下去，使得这个预测和标答间的损失逐渐被吞噬消融
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ========== 引擎交互伺候大循环轮询区 ==========
print("[Python AI] DQN Backend Ready. Waiting for IPC Connection...")
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
            # 性能屏障：仅拦截被重发或没营养的僵持卡顿包
            if data.startswith("STATE:"):
                try:
                    state_parts = data[6:].split('|')
                    curr_step = int(state_parts[3])
                    curr_coords = (float(state_parts[0]), float(state_parts[1]))
                    if data == last_handled_info["data"] and curr_step == last_handled_info["step"] and curr_coords == last_handled_info["coords"]:
                        continue
                except:
                    pass
            elif data == last_handled_info["data"]:
                continue
                
            # ---------------- 1. 初始化战场环境 (GRID) ----------------
            if data.startswith("GRID:"):
                parts = data[5:].split('|')
                try:
                    maze_w, maze_h = int(parts[0]), int(parts[1])
                    if len(parts) > 3:
                        current_seed = int(parts[2])
                        log_path = os.path.join(save_dir, f"dqn_log_s{maze_w}_{current_seed}.csv")
                        model_path = os.path.join(save_dir, f"dqn_model_s{maze_w}_{current_seed}.pth")
                        grid_vals = [int(x) for x in parts[3].split(',')]
                        grid_array = np.array(grid_vals)
                        maze_grid = grid_array.reshape((maze_h, maze_w))
                        
                        build_network(maze_w, maze_h)
                        replay_buffer.clear()
                        best_reward = -999999.0
                        # 动态调整步数上限和衰减率: 大迷宫需要更激进的衰减
                        max_episode_steps = maze_w * maze_h * 50
                        if maze_w >= 15:
                            epsilon_decay = 0.990  # 大迷宫加速衰减
                        else:
                            epsilon_decay = 0.995  # 小迷宫保持原速
                        print(f"[DQN GRID] {maze_w}x{maze_h} loaded, seed={current_seed}, max_steps={max_episode_steps}, decay={epsilon_decay}")
                    else:
                        print(f"[DQN GRID] Format error: expected >3 parts")
                except Exception as e:
                    print(f"[DQN GRID] Parse error: {e}")
                    
                write_to_shared_memory("GRID_OK")
                last_handled_info["data"] = "GRID_OK"
                last_handled_info["step"] = -1
                last_handled_info["coords"] = (0, 0)
                
            # ---------------- 2. 收到强降的改变学习参数指令 ----------------
            elif data.startswith("PARAM:"):
                params = data[6:].split(',')
                gamma = float(params[1])
                epsilon = float(params[2])
                epsilon_decay = float(params[3])
                batch_size = int(params[4])
                if len(params) > 5:
                    lr = float(params[5])
                    if optimizer is not None:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                last_handled_info["data"] = data
                last_handled_info["step"] = -1
                last_handled_info["coords"] = (0, 0)
                
            # ---------------- 3. 后门指令：游戏状态接管 ----------------
            elif data.startswith("COMMAND:"):
                cmd_full = data[8:]
                cmd_parts = cmd_full.split('|')
                cmd = cmd_parts[0]
                
                if cmd == "QUIT" or cmd == "EXIT":
                    print("[Python AI] 收到指令: 宿主引擎下线，算法端执行自裁！")
                    break
                    
                elif cmd == "DEMO":
                    if policy_net is None:
                        if maze_grid is not None:
                            build_network(maze_w, maze_h)
                        else:
                            print("[DQN] 警告: 尝试进入 DEMO 模式但尚未收到地图数据 (GRID)！")
                    
                    if policy_net is not None:
                        # 【只读演示期】：载入这个环境历经测试里目前保存最高记录的模型存档位来运行，以彰显算法雄风不掉链子！
                        load_model(use_best=True)
                        is_demo = True
                        epsilon = 0.0          # 卡死一切投机瞎走的概率
                        episode_count = 0
                        replay_buffer.clear()
                        policy_net.eval()
                        print(f"-> DEMO MODE: Loaded Best DQN Weights")
                        
                elif cmd == "START":
                    if policy_net is None:
                        if maze_grid is not None:
                            build_network(maze_w, maze_h)
                        else:
                            print("[DQN] 警告: 尝试启动训练但尚未收到地图数据 (GRID)！")
                    
                    if policy_net is not None:
                        load_model(use_best=False) # 继续被虐，加载的是带遗憾的断点日常版而不是最好的，因为还要接着摸黑爬坑找新高塔。
                        is_demo = False
                        episode_hit_count = 0
                        global_steps = 0
                        policy_net.train()
                        init_log_file()
                        resume_training_state()
                        last_state_vec = None
                        last_action = None
                        print(f"-> DQN TRAINING START (Map: {current_seed}) Resuming from EP {episode_count}, Eps={epsilon:.4f}")
                        
                elif cmd == "RESET":
                    # Unity端强制拉黑/过关后的小球回收到发源位点重启！
                    hit_count_this_ep = int(cmd_parts[1]) if len(cmd_parts) > 1 else 0
                    episode_count += 1
                    
                    if epsilon > min_epsilon:
                        epsilon *= epsilon_decay
                        
                    save_training_log(episode_count, episode_steps, hit_count_this_ep, episode_reward, epsilon)
                    
                    if not is_demo:
                        # 若超越了该图往期辉煌里程碑：
                        if episode_reward > best_reward:
                            best_reward = episode_reward
                            save_model(is_best=True) # 造神加冕！
                        else:
                            save_model(is_best=False)
                            
                    print(f"  EP {episode_count} | Steps: {episode_steps} | Hits: {hit_count_this_ep} | Reward: {episode_reward:.1f} | Eps: {epsilon:.3f} | Best: {best_reward:.1f}")
                    
                    last_state_vec = None
                    last_action = None
                    accumulated_reward = 0.0
                    episode_reward = 0.0
                    episode_steps = 0
                    position_history.clear()
                    visit_count_map.clear()
                    
                last_handled_info["data"] = data
                last_handled_info["step"] = -1
                last_handled_info["coords"] = (0, 0)
                
            # ---------------- 4. 网络训练常规事件状态管道 (STATE) ----------------
            elif data.startswith("STATE:"):
                if policy_net is None:
                    write_to_shared_memory("ACTION:0")
                    continue
                    
                state_info = data[6:].split('|')
                # 防御性解析数据错位丢包崩溃
                if len(state_info) < 2 or state_info[0] == '' or state_info[1] == '':
                    continue
                try:
                    cur_x, cur_y = float(state_info[0]), float(state_info[1])
                except ValueError:
                    print(f"[DQN] 跳过损坏的 STATE 数据: {data[:50]}")
                    continue
                    
                # 这一步把刚刚这几纳秒期间 Unity 硬塞发来的碰撞等琐碎的奖惩合并结案：
                if len(state_info) >= 5 and state_info[4] != '':
                    try:
                        r = float(state_info[4])
                        accumulated_reward += r
                        episode_reward += r
                    except ValueError:
                        pass
                        
                current_state_vec = get_state_vector(cur_x, cur_y)
                
                # ==== 梦境池打包：形成四元学习元并抛入记忆轮滑带中 ==== 
                # (老状态, 当时做的动作, 因为这个动作拿了什么好果子, 走到了哪个现在的状态, 不是通关终点)
                if last_state_vec is not None and last_action is not None:
                    replay_buffer.append((last_state_vec, last_action, accumulated_reward, current_state_vec, 0.0))
                    
                # 真正开始使用经验反推学习损失并覆盖重写全连接脑沟壑参数：
                if not is_demo and len(replay_buffer) > batch_size:
                    policy_net.train()
                    learn(batch_size)
                    policy_net.eval()
                    
                global_steps += 1
                
                # 【目标影子网络迟延迭代法】：如果直接把老将顶在一线，大家就会在自己做的决策里自我高估相互包庇跌入极点反馈无法自拔。用落后100个步伐的老网来克制它。
                if global_steps % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                    
                accumulated_reward = 0.0
                
                # ========== 循环检测机制 ==========
                grid_pos = (int(cur_x), int(cur_y))
                if grid_pos not in visit_count_map:
                    visit_count_map[grid_pos] = 0
                visit_count_map[grid_pos] += 1
                vc = visit_count_map[grid_pos]
                # 在短窗口内检测循环: 如果最近50步内同一格子出现>=3次, 给额外惩罚
                position_history.append(grid_pos)
                short_visits = sum(1 for p in position_history if p == grid_pos)
                if short_visits >= 3 and not is_demo:
                    loop_penalty = -0.3 * (short_visits - 2)
                    accumulated_reward += loop_penalty
                    episode_reward += loop_penalty
                
                # ========== 步数超限强制截断 ==========
                if episode_steps >= max_episode_steps and not is_demo:
                    if last_state_vec is not None and last_action is not None:
                        timeout_penalty = -10.0
                        accumulated_reward += timeout_penalty
                        episode_reward += timeout_penalty
                        replay_buffer.append((last_state_vec, last_action, accumulated_reward, current_state_vec, 1.0))
                    print(f"  [DQN] EP {episode_count+1} steps exceeded limit ({max_episode_steps}), force truncating")
                    accumulated_reward = 0.0
                    last_state_vec = current_state_vec
                    last_action = None
                    write_to_shared_memory(f"ACTION:0|{epsilon:.4f}")
                    last_handled_info["data"] = data
                    last_handled_info["step"] = int(state_info[3]) if len(state_info) > 3 and state_info[3] != '' else -1
                    last_handled_info["coords"] = (cur_x, cur_y)
                    continue
                
                # 为下个时钟周期生成走位定夺：
                action = decide_action(current_state_vec)
                
                last_state_vec = current_state_vec
                last_action = action
                episode_steps += 1
                
                write_to_shared_memory(f"ACTION:{action}|{epsilon:.4f}")
                
                last_handled_info["data"] = data
                last_handled_info["step"] = int(state_info[3]) if len(state_info) > 3 and state_info[3] != '' else -1
                last_handled_info["coords"] = (cur_x, cur_y)
                continue
                
            # ---------------- 5. 奖励分配特殊通道拦截 (针对通关死亡终端) (REWARD) ----------------
            elif data.startswith("REWARD:"):
                reward_info = data[7:].split('|')
                try:
                    r = float(reward_info[0]) if len(reward_info) > 0 and reward_info[0] != '' else 0.0
                except ValueError:
                    r = 0.0
                    
                rtype = int(reward_info[3]) if len(reward_info) > 3 else 0
                accumulated_reward += r
                episode_reward += r
                
                # 当 rtype == 2 代表抵达到这世间的至高理想终点，本局宣布直接终止（Terminal Block）。
                if rtype == 2:
                    if last_state_vec is not None and last_action is not None:
                        current_state_vec = get_state_vector(maze_w - 2, maze_h - 2)
                        
                        # !!!!!!!!! 注意这里末尾标志为 1.0 (True) !!!!!!!! #
                        # DQN 神经网络在接受训练时候，看到这个 1.0 标记位，会把它未来可能延伸的价值强制砍为绝对的0去计算。只给它保留积累的这些金币算入。防止把这种末日后没有明天的情况带入模型干扰收敛循环。
                        replay_buffer.append((last_state_vec, last_action, accumulated_reward, current_state_vec, 1.0))
                        
                        # 极端难获得的稀缺终点样本出现！不要只学一次，给它特批重推敲重复五遍连学，吃死终点的环境周边红利！
                        for _ in range(5):
                            learn(batch_size)
                            
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