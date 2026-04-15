# -*- coding: utf-8 -*-
"""
Q-Learning 强化学习迷宫 AI 后端 (带 Dyna-Q 经验回放机制)
=========================================================
该文件作为服务端运行，通过共享内存与 Unity 游戏引擎进行跨进程通信(IPC)。
主要实现原教旨 Q-Learning 算法，以及带经验池回顾的 Dyna-Q 变体，来加速迷宫寻路学习。
"""
import mmap
import time
import random
import os
from collections import deque
import csv
# ========== 共享内存(IPC)配置 ==========
# 共享内存名称，必须与 Unity 端的配置完全一致
shared_memory_name = "MazeRLSharedMemory"
# 分配 4KB (4096 bytes) 的连续内存空间，足够高频极小的字符串指令通信使用
data_size = 4096
# -1 表示分配匿名/页文件后备的共享内存 (Windows系统级实现)
mm = mmap.mmap(-1, data_size, tagname=shared_memory_name)
# ========== Q-Learning 核心超参数 ==========
current_algorithm = 0
lr = 0.1             # 学习率 (Learning Rate, α)：0~1之间，控制新旧知识的覆盖比例，0.1表示每次用10%的新学知识迭代旧知识。
gamma = 0.95         # 折扣因子 (Discount Factor, γ)：0.95能让终点奖励信号传播更远，适合大迷宫场景。
epsilon = 1.0        # 探索率 (Epsilon, ε)：决定了AI随机乱走的概率，初始为1.0即100%瞎蒙，确保充分探索未知的整个地图。
epsilon_decay = 0.995 # 探索衰减率：每回合后探索率按此乘积缩小，使得行为模式逐步从"瞎蒙阶段"(探索)过渡到"利用已知"(开发)。
min_epsilon = 0.05   # 最小探索率：即使是训练很成熟的后期，也保留5%的可能走错或探索新路线，防止陷入局部死循环。
# Q-Table(Q值表)：字典形式。Key 是包含(x,y)坐标的元组，Value 是长度为4的列表，对应[上, 下, 左, 右]四个动作的期望积累奖励(Q值)。
q_table = {}
# ========== 训练状态与环境追踪 ==========
last_state = None          # 记录上一次所在的坐标点 (x, y)
last_action = None         # 记录上一次采取的动作标号 (0:上, 1:下, 2:左, 3:右)
accumulated_reward = 0.0   # 累计未被结算的即时奖励 (由于IPC异步原因，引擎频繁发送的惩罚或小奖励先汇聚于此，直到下一次Action做出时结算)
episode_reward = 0.0       # 本回合(Episode)累计获得的总奖励，用来画图或综合衡量本次Agent表现好坏。
episode_steps = 0          # 本回合存活中已总共走出的步数
episode_count = 0          # 全局计数器，当前是第几个训练回合
maze_grid = []             # (预留变量) 可用于接收完整地图结构
maze_w, maze_h = 0, 0      # 地图的宽度和高度尺寸，通信中由上层(Unity)派发过来
# 经验池 (Dyna-Q 算法的核心组件)：
# 使用双端队列(deque)，当记录数超过 maxlen=5000 时，新的记忆加入会自动挤掉(淘汰)最左侧最老的记忆，实现循环队列。
experience_buffer = deque(maxlen=5000)
episode_hit_count = 0      # 本回合不幸撞到墙壁被反弹的次数统计
best_reward = -999999.0    # 记录当前迷宫随机种子的最高分界碑
# ========== 循环检测与防原地踏步 ==========
position_history = deque(maxlen=50)  # 最近50步的位置记录
visit_count_map = {}                 # 全局计数: 每个格子被踩多少次
max_episode_steps = 10000            # 单局步数上限, 收到GRID后动态调整
# ========== 数据持久化、保存与日志配置 ==========
# 确定数据存储目录 (当前代码文件所在目录下的 training_data~ 文件夹，通常这会被 Git 或 Unity 静态剔除忽略包)
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data~")
os.makedirs(save_dir, exist_ok=True) # 使用 os.makedirs，如果路径和对应文件夹均不存在则逐层创建
current_seed = 0           # 当前所挑战的迷宫在游戏内的生成随机种子(用以区分训练集)
# 缺省的回退日志名称和模型记录名称，收到 Unity 发来的确切 GRID 尺寸与 Seed 后会将其修正覆写
log_path = os.path.join(save_dir, "training_log_default.csv")
qtable_path = os.path.join(save_dir, "q_table_default.csv")
def init_log_file():
    """初始化/创建训练过程日志记录表。如果未创建则自动生成并在第一行写入 CSV 表头(Columns)"""
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "steps", "hit_count", "total_reward", "epsilon"])
def save_training_log(ep, steps, hit_count, reward, eps):
    """记录每一次训练回合的各项数据结单，用于在 Unity 端读取并绘制损失、奖励下降折线图。"""
    if not os.path.exists(log_path):
        init_log_file()
    # "a" 意为追加模式 (Append) 写入一行，避免覆盖旧数据影响连贯绘图
    with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([ep, steps, hit_count, f"{reward:.2f}", f"{eps:.4f}"])
def resume_training_state():
    """断点续训 (Resume Training) 机制。如果在同一个种子地图上之前有残余数据，从中抽取最后的时间点开始继承。"""
    global episode_count, epsilon, best_reward
    if os.path.exists(log_path):
        try:
            last_ep = 0
            last_eps = epsilon
            best_so_far = -999999.0
            with open(log_path, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    # 读取整份旧日志以获得最大的回合索引与收敛到的 epsilon
                    if len(row) >= 5:
                        last_ep = int(row[0])
                        r = float(row[3])
                        last_eps = float(row[4])
                        # 沿途搜集出最高分奖励纪录，防止最高分被后来的断点覆盖导致无法刷新 best model
                        if r > best_so_far:
                            best_so_far = r
            if last_ep > 0: # 成功抽取出断点数据
                episode_count = last_ep
                epsilon = last_eps
                best_reward = best_so_far
                print(f"[Python AI] 断点续训：从 EP {episode_count} 继续，历史最高分 {best_reward:.1f}")
                return
        except Exception as e:
            print(f"[Python AI] 读取训练日志失败: {e}")
    # 全新训练环境则重置
    episode_count = 0
    epsilon = 1.0
    print("[Python AI] 未发现历史训练数据，从头开始训练。")
def save_q_table(is_best=False):
    """将内存中作为大脑存在的 q_table 字典，执行键位解析和持久化结构落盘 CSV 文件。"""
    path = qtable_path
    if is_best:
        # 当被标记为创纪录的一场时，不仅存正常位，还要复制留存为专门的 _best 名的模型记录
        path = qtable_path.replace(".csv", "_best.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "Q_up", "Q_down", "Q_left", "Q_right"])
        for (x, y), values in q_table.items():
            # (x,y)转为两列，values是个拥有4元素的数组转为四列
            writer.writerow([x, y] + [f"{v:.4f}" for v in values])
    if is_best:
        print(f"[Python AI] !!! 记录刷新，已保存 Q-Table 最佳模型: {os.path.basename(path)}")
def load_q_table():
    """解析已有的 CSV 模型大脑存盘，还原转化为纯正的以 (x,y) 为键，长度4数组为值的大型记忆字典 q_table"""
    if os.path.exists(qtable_path):
        try:
            with open(qtable_path, "r", newline="", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                next(reader) # 跳过毫无意义的列名表头 skip header
                for row in reader:
                    if len(row) == 6:
                        x, y = int(row[0]), int(row[1])
                        q_values = [float(v) for v in row[2:]]
                        q_table[(x, y)] = q_values
            print(f"[Python AI] Loaded previous Q-Table with {len(q_table)} states.")
        except Exception as e:
            print(f"[Python AI] Error loading Q-Table: {e}")
    else:
        print("[Python AI] No previous Q-Table found, starting fresh.")
# ========== IPC 共享内存读写工具 ==========
def read_from_shared_memory():
    """归零指针，读取内存整段 4KB 的字符，并剔掉多带上的空白二进制填充符号。"""
    mm.seek(0) # 将读写指针移致句柄开头
    return mm.read(data_size).decode('utf-8').strip('\x00').strip()
def write_to_shared_memory(data):
    """带空位填充机制的高精度内存写入写入操作"""
    try:
        mm.seek(0)
        # 用空字符将传入的命令补全到 4096 字节。否则如果刚发 "ACTION:1" (短) 后覆盖原先的 "COMMAND:XXXX" (长)，后半部分幽灵缓存会遗留下导致截断解析bug
        padded_data = data.ljust(data_size, '\x00')
        mm.write(padded_data.encode('utf-8'))
        mm.flush()  # 迫使系统不计代价从缓冲中下落底层页，即刻可用
    except Exception as e:
        print(f"[WRITE ERROR] Failed to write to shared memory: {e}")
# ========== RL 决策系统与算法基石 ==========
def get_q_values(x, y):
    """如果探索到了未曾去过的全新坐标网格区域，则原位扩展 Q表 ，赋初识权重 [0,0,0,0]"""
    state_key = (x, y)
    if state_key not in q_table:
        q_table[state_key] = [0.0, 0.0, 0.0, 0.0]
    return q_table[state_key]
def get_reverse_action(action):
    """
    提供方向反转映射功能。
    作用是在部分强向导探索环境中，刻意避免特工无脑来回"往返跑"浪费随机概率（例如 走上->走下->走上->走下）。
    """
    if action == 0: return 1
    if action == 1: return 0
    if action == 2: return 3
    if action == 3: return 2
    return -1
def decide_action(x, y):
    """
    贪婪探索机制 (Epsilon-Greedy Strategy)。RL的心脏：用来选择下个动作去开发固有收益还是一场未知的豪赌。
    """
    q_values = get_q_values(x, y)
    # 获取随机数判断是否落入目前 Epsilon 区间以进入乱走模式
    if random.random() < epsilon:
        choices = [0, 1, 2, 3] # 0上 1下 2左 3右
        # 【前向探索启发机制优化】：
        # 为了防备纯随机中极大可能产生的在宽敞回廊里"反复横跳"，这里引入：
        # 假如它上一步往左去了，既然还在探索中，我们80%约束它现在不要倒着往右跑回去当缩头乌龟，而是鼓励前向继续探迷宫腹地。
        if last_action is not None and random.random() < 0.8:
            rev = get_reverse_action(last_action)
            if rev in choices:
                choices.remove(rev)
        return random.choice(choices)
    else:
        # 脱离乱走区间后，进入剥削阶段 (Exploitation)：严格选取四个方向里的最大期望方向。
        max_q = max(q_values)
        # 如果大家初始都是0并列第一，或者有同等最佳的两条岔道，那在这平局最佳列表里面随便盲拆一条。
        best_actions = [i for i, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)
def update_q_table(cur_x, cur_y):
    """
    核心强化学习值迭代结算。
    将时差错开的一对 [上一个动作环境产生的分数] 结清，算给对应的上一个动作行为。
    并执行基于缓冲区的 Dyna-Q 经验梦境反刍。
    """
    global last_state, last_action, accumulated_reward, experience_buffer
    if last_state is not None and last_action is not None:
        # ------- 1. 常规的当下学习模块 (TD Error 贝尔曼方程) -------
        # 获取当时作出决断前的路口的 Q 分数记忆数组
        q_values_old = get_q_values(last_state[0], last_state[1])
        # 获取到了这个点以后，此地点具备的最高脱身希望数组
        q_values_new = get_q_values(cur_x, cur_y)
        # max_next_q: 代表了新的节点对于未来的最大预估分数。
        max_next_q = max(q_values_new)
        # Target目标 = 当下的立刻获得扣分/奖分 + 未来期望折扣，即 V(s) = R + γ * max Q(s', a')
        target = accumulated_reward + gamma * max_next_q
        # 按学习率迭代，将原本对此节点此动作的分数缓慢拉扯向真实的 Target，即 Q(s,a) = Q(s,a) + α(Target - Q(s,a))
        q_values_old[last_action] += lr * (target - q_values_old[last_action])
        # ------- 2. 将这次宝贵的经历存入大脑记忆池区 -------
        # (S_t, A_t, R_{t+1}, S_{t+1}) 这是无模型方法下的最精髓信息四元组
        experience_buffer.append((last_state, last_action, accumulated_reward, (cur_x, cur_y)))
        # ------- 3. 大脑在后台飞速“睡梦反刍”（Dyna-Q 规划算法核心步骤） -------
        # Q-learning 在迷宫这种极少产生终点奖励的稀疏报酬下，学习收敛极其缓慢。
        # 加入这一步的原理是：每次有空余的时候，它在记忆大海中随机打捞 10 块记忆碎片！
        # 让成功的一条路在十几次不断的离线重新模拟连接当中，快速像多条闪电链接一样把 "到达终点的100分高分价值" 从终点极速顺藤摸瓜回溯推演至入口方向。
        if len(experience_buffer) > 10:
            samples = random.sample(experience_buffer, 10)
            for s, a, r, next_s in samples:
                qs_old = get_q_values(s[0], s[1])
                # 特殊陷阱处理：如果这是一段死亡或直接被转移到结点的最后时光片段 (即 Terminal State，next_s 传的不存在)
                # 那么它不存在任何未来出路，最大期待值为绝对的 0
                if next_s is None:
                    mq_next = 0.0
                else:
                    qs_next = get_q_values(next_s[0], next_s[1])
                    mq_next = max(qs_next)
                # 反刍，重新巩固一下旧有 Q 值
                t = r + gamma * mq_next
                qs_old[a] += lr * (t - qs_old[a])
# ========== 主工作轮询进程死循环 (Main IPC Loop) ==========
print("[Python AI] Q-Learning Backend Ready.")
write_to_shared_memory("READY")
last_handled_data = "READY"
while True:
    data = read_from_shared_memory()
    # 由于非阻塞，极大概率这一纳秒读到的跟上一纳秒发过来的全无变化，加锁跳过以减负
    if data and data != last_handled_data and data != "WAITING":
        # ===== [GRID] 地图分配通信块 =====
        if data.startswith("GRID:"):
            # 协议拆解 => "GRID:宽度|高度|随机产生种子数"
            parts = data[5:].split('|')
            maze_w, maze_h = int(parts[0]), int(parts[1])
            if len(parts) > 3:
                current_seed = int(parts[2])
                # 针对不同种子的迷宫设定独有的存盘文件名，防止地图结构被大脑串味覆盖
                log_path = os.path.join(save_dir, f"training_log_QL_s{maze_w}_{current_seed}.csv")
                qtable_path = os.path.join(save_dir, f"q_table_QL_s{maze_w}_{current_seed}.csv")
                q_table.clear() # 重置洗脑
                load_q_table()  # 尝试从新获取的名字文件唤起专属于此随机生成的旧有记忆
                best_reward = -999999.0
                experience_buffer.clear() # 新图必须清理经验池
                # 动态调整步数上限和衰减率
                max_episode_steps = maze_w * maze_h * 50
                if maze_w >= 15:
                    epsilon_decay = 0.990  # 大迷宫加速衰减
                else:
                    epsilon_decay = 0.995  # 小迷宫保持原速
            print(f"[QL GRID] {maze_w}x{maze_h} loaded, seed={current_seed}, max_steps={max_episode_steps}, decay={epsilon_decay}")
            # 【双向握手】：向 Unity 报告本地目录已妥可以放行进入准备阶段了。
            write_to_shared_memory("GRID_OK")
            last_handled_data = "GRID_OK"
        # ===== [ALGO] 模式选填通信块 =====
        elif data.startswith("ALGO:"):
            current_algorithm = int(data[5:])
            last_handled_data = data
        # ===== [PARAM] 同步界面调教值通信块 =====
        elif data.startswith("PARAM:"):
            # 热加载功能：Unity界面的UI拖动数值一旦变化可即刻应用修改当前此模型变量，免重启改调参。
            params = data[6:].split(',')
            lr = float(params[0])
            gamma = float(params[1])
            epsilon = float(params[2])
            epsilon_decay = float(params[3])
            last_handled_data = data
        # ===== [STATE] 物理推演帧事件通信块 =====
        elif data.startswith("STATE:"):
            state_info = data[6:].split('|')
            # 规避读取碰撞包或网络堵塞截断引发越界访问直接崩溃的情况
            if len(state_info) < 2 or state_info[0] == '' or state_info[1] == '':
                continue
            try:
                cur_x, cur_y = float(state_info[0]), float(state_info[1])
            except ValueError:
                continue
            ix, iy = int(cur_x), int(cur_y)
            # 【重要】接收引擎跨进程下发的碰撞奖惩，比如每次触墙的 -0.2。
            # 这里是把当前状态和之前的动作算进去累加。防止太快没有出 STATE 就遗漏。
            if len(state_info) >= 5 and state_info[4] != '':
                try:
                    r = float(state_info[4])
                    accumulated_reward += r
                    episode_reward += r
                except ValueError:
                    pass
            # 总结上面刚获得的所有累加痛苦和收益，真正结算给前一座标格子的决策分数里区：
            update_q_table(ix, iy)
            # 随后立刻置零，切断连带，为接下来这步走完预备奖池
            accumulated_reward = 0.0
            
            # ========== 循环检测机制 ==========
            grid_pos = (ix, iy)
            if grid_pos not in visit_count_map:
                visit_count_map[grid_pos] = 0
            visit_count_map[grid_pos] += 1
            # 在短窗口内检测循环: 如果最近50步内同一格子出现>=3次, 给额外惩罚
            position_history.append(grid_pos)
            short_visits = sum(1 for p in position_history if p == grid_pos)
            if short_visits >= 3:
                loop_penalty = -0.3 * (short_visits - 2)
                accumulated_reward += loop_penalty
                episode_reward += loop_penalty
            
            # ========== 步数超限强制截断 ==========
            if episode_steps >= max_episode_steps:
                # 强制截断: 将当前状态作为终止状态
                if last_state is not None and last_action is not None:
                    q_values_old = get_q_values(last_state[0], last_state[1])
                    timeout_penalty = -10.0
                    accumulated_reward += timeout_penalty
                    episode_reward += timeout_penalty
                    target = accumulated_reward  # 终止状态无未来价值
                    q_values_old[last_action] += lr * (target - q_values_old[last_action])
                    experience_buffer.append((last_state, last_action, accumulated_reward, None))
                print(f"  [QL] EP {episode_count+1} steps exceeded limit ({max_episode_steps}), force truncating")
                accumulated_reward = 0.0
                last_state = None
                last_action = None
                write_to_shared_memory(f"ACTION:0|{epsilon:.4f}")
                continue
            
            # 算出它该去哪里：
            action = decide_action(ix, iy)
            # 保存坐标遗书：
            last_state = (ix, iy)
            last_action = action
            episode_steps += 1
            # 封包发送指令，同时将当前它发神经盲目跑的倾向性百分比告知 Unity 的实时可视化面板图表。
            write_to_shared_memory(f"ACTION:{action}|{epsilon:.4f}")
            # STATE事件可能接纳连续多次而不被阻挡跳过，不设到last_handled_data里去。
            continue
        # ===== [REWARD] 单纯获取奖励结算通信块 =====
        elif data.startswith("REWARD:"):
            reward_info = data[7:].split('|')
            try:
                r = float(reward_info[0]) if len(reward_info) > 0 and reward_info[0] != '' else 0.0
            except ValueError:
                r = 0.0
            rtype = int(reward_info[3]) if len(reward_info) > 3 else 0
            accumulated_reward += r
            episode_reward += r
            # 【核心逻辑坑点修复：终点结算闭环法则】
            # 当类型为 2 (即 ReachEnd 到达终点状态)，Unity 此时的游戏机制将会强制清除所有物理特性瞬间将此代理人放回复活点重新下一局开始。
            # 如果等复活点那一句的新 STATE 来了再来总结刚才抵达终点的行动奖惩，那就会把"站在复活点的前一秒的到达高额一百分奖励"作为目标算给了刚出生点的地方！
            # 因此，对于【终点】这唯一一处的 Terminal State，直接进行就地截断并强制计算其巨大悬赏得分：
            if rtype == 2:
                if last_state is not None and last_action is not None:
                    q_values_old = get_q_values(last_state[0], last_state[1])
                    target = accumulated_reward # 万物归宗，它的未来可能已是0，最高也就是其当下带上的满载赏金
                    q_values_old[last_action] += lr * (target - q_values_old[last_action])
                    # 放入特级档案记忆：终点前一步的记录不带 next state ("None")，以作绝后的凭签阻止死循环污染反刍池
                    experience_buffer.append((last_state, last_action, accumulated_reward, None))
            last_handled_data = data
        # ===== [COMMAND] 引擎直发全局指令通信块 =====
        elif data.startswith("COMMAND:"):
            cmd_full = data[8:]
            cmd_parts = cmd_full.split('|')
            cmd = cmd_parts[0]
            if cmd == "QUIT" or cmd == "EXIT":
                print("[Python AI] 收到指令: 宿主引擎下线，算法端执行自裁！")
                break # 直接终止内部跑动循环以结束这个 Python 开销控制台外挂进程
            elif cmd == "DEMO":
                # DEMO 指令是要求给老板/别人纯纯的播放模型收敛效果阶段而不带训练成分。
                # 【防并发竞态关键修正】：由于只有 4KB 缓冲共享槽道，如果在按 C# 按键和通信途中可能挤丢了准确获取最好版本的记忆时间窗口。
                # 为防此情况，切换为表现层的时候一锤定音再强制从对应这个生成的随机种子号里面读出记录（避免走秀垮台）
                q_table.clear()
                load_q_table()
                epsilon = 0.0  # 强制彻底锁死乱走概率参数=0，一切抉择必须按最高理论指使最大Q值所办到不能冒险出戏
                episode_count = 0
                print(f"-> DEMO MODE: Loaded Q-Table for seed {current_seed} ({len(q_table)} states). Epsilon=0.")
            elif cmd == "START":
                # 点击了【开始训练】按钮
                q_table.clear()
                load_q_table()
                episode_hit_count = 0
                init_log_file()
                # 尝试从 CSV 寻找本子本子中上次可能训练过一般的中断进度并衔接：
                resume_training_state()
                last_state = None
                last_action = None
                print(f"-> START MODE: Loaded model for seed {current_seed}. Resuming from EP {episode_count}, Eps={epsilon:.4f}")
            elif cmd == "RESET":
                # Unity 一旦决定要洗牌重启小球 (诸如已经到达终点或被墙壁磨损坏死超过设定的最高存活步数)：
                hit_count_this_ep = int(cmd_parts[1]) if len(cmd_parts) > 1 else 0
                episode_count += 1
                # 进行衰减机制：把高随机性在打败一局后乘除缩短。逐渐让脑子变得清纯听话。
                if epsilon > 0.0:
                    epsilon = max(min_epsilon, epsilon * epsilon_decay)
                # 【每局落锤】把折腾这一集的结果打一张表放入长期历史日志监控。
                save_training_log(episode_count, episode_steps, hit_count_this_ep, episode_reward, epsilon)
                # 【优胜劣汰保存法则】
                if episode_reward > best_reward:
                    # 分数比以往都高，那么这个模型有可能是这道迷宫目前的全局最优解网络参数字典！
                    best_reward = episode_reward
                    save_q_table(is_best=True) # 存一份带 _best 后缀的荣誉档案！
                else:
                    # 分数平平，那常规保存以覆盖正常旧记录防丢失
                    save_q_table(is_best=False)
                # 控制台终端上打出每一集的结盘通告
                print(f"  EP {episode_count} | Steps: {episode_steps} | Hits: {hit_count_this_ep} | Reward: {episode_reward:.1f} | Eps: {epsilon:.3f} | Best: {best_reward:.1f}")
                # 格式化其在本次模拟周期的局部脑子，使之以纯白的临时心态进入重新回到起点的下一个 Episode 迭代。
                last_state = None
                last_action = None
                accumulated_reward = 0.0
                episode_reward = 0.0
                episode_steps = 0
                position_history.clear()
                visit_count_map.clear()
            last_handled_data = data
    # 【时间片出让操作】：极其重要，如果不 sleep 出让给系统分配片，死循环会死锁占据一个 CPU 核心100% 从而反而饿死 Unity 端收发延迟导致丢帧。0.01 秒(10ms) 是一个通信的折中点。
    time.sleep(0.01)