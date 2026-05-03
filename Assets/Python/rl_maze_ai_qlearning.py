# -*- coding: utf-8 -*-
# Q-Learning 强化学习迷宫 AI 后端（含 Dyna-Q 经验回放）
# 算法：表格式 Q-Learning + Dyna-Q 规划（每步随机回放 10 条经验，加速 Q 值传播）
# IPC：Windows 共享内存 MazeRLSharedMemory（4096 bytes）
import mmap
import time
import random
import os
import csv
from collections import deque

# ========== 共享内存 IPC ==========
shared_memory_name = "MazeRLSharedMemory"
data_size = 4096
mm = mmap.mmap(-1, data_size, tagname=shared_memory_name)

# ========== 超参数 ==========
lr            = 0.1
gamma         = 0.95
epsilon       = 1.0
epsilon_decay = 0.995
min_epsilon   = 0.05
q_table = {}  # {(x, y): [Q_up, Q_down, Q_left, Q_right]}

# ========== 状态追踪 ==========
last_state         = None
last_action        = None
accumulated_reward = 0.0
episode_reward     = 0.0
episode_steps      = 0
episode_count      = 0
maze_grid          = []
maze_w, maze_h     = 0, 0
experience_buffer  = deque(maxlen=5000)
is_demo            = False
best_reward        = -999999.0
position_history   = deque(maxlen=50)
visit_count_map    = {}
max_episode_steps  = 10000  # 收到 GRID 后按地图尺寸动态调整

# ========== 存档路径 ==========
save_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data~")
os.makedirs(save_dir, exist_ok=True)
current_seed   = 0
log_path       = os.path.join(save_dir, "training_log_default.csv")
qtable_path    = os.path.join(save_dir, "q_table_default.csv")
train_log_path = log_path
demo_log_path  = os.path.join(save_dir, "training_log_QL_demo_default.csv")

# ========== 日志与存档工具 ==========
def init_log_file():
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(["episode", "steps", "hit_count", "total_reward", "epsilon"])

def save_training_log(ep, steps, hit_count, reward, eps):
    if not os.path.exists(log_path):
        init_log_file()
    with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow([ep, steps, hit_count, f"{reward:.2f}", f"{eps:.4f}"])

def resume_training_state(update_best=True):
    """断点续训/续演示。update_best=False 时仅续号，避免 demo CSV 污染训练阶段的状态。"""
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
                    epsilon, best_reward = last_eps, best_so_far
                mode = "训练" if update_best else "演示"
                msg  = f"[Python AI] 断点续{mode}：从 EP {episode_count} 继续"
                if update_best:
                    msg += f"，历史最高分 {best_reward:.1f}"
                print(msg)
                return
        except Exception as e:
            print(f"[Python AI] 读取日志失败: {e}")
    episode_count = 0
    if update_best:
        epsilon = 1.0
    print(f"[Python AI] 未发现历史{'训练' if update_best else '演示'}数据，从 EP 0 开始。")

def save_q_table(is_best=False):
    path = qtable_path.replace(".csv", "_best.csv") if is_best else qtable_path
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "Q_up", "Q_down", "Q_left", "Q_right"])
        for (x, y), values in q_table.items():
            writer.writerow([x, y] + [f"{v:.4f}" for v in values])
    if is_best:
        print(f"[Python AI] 新最优 Q-Table 已保存: {os.path.basename(path)}")

def load_q_table(use_best=False):
    """use_best=True 时优先加载 _best.csv（Demo 模式专用）。"""
    target_path = qtable_path
    if use_best:
        best_path = qtable_path.replace(".csv", "_best.csv")
        if os.path.exists(best_path):
            target_path = best_path
    if not os.path.exists(target_path):
        print("[Python AI] 未找到 Q-Table 存档，从空表开始。")
        return
    try:
        with open(target_path, "r", newline="", encoding="utf-8-sig") as f:
            for row in list(csv.reader(f))[1:]:
                if len(row) == 6:
                    q_table[(int(row[0]), int(row[1]))] = [float(v) for v in row[2:]]
        print(f"[Python AI] Q-Table 加载完成，共 {len(q_table)} 个状态: {os.path.basename(target_path)}")
    except Exception as e:
        print(f"[Python AI] Q-Table 加载失败: {e}")

# ========== IPC 读写 ==========
def read_from_shared_memory():
    mm.seek(0)
    # split('\x00') 而非 strip：防 C# 两次连续写入时 \x00 夹杂数据中间导致解析失败
    raw = mm.read(data_size).decode('utf-8')
    return raw.split('\x00')[0].strip()

def write_to_shared_memory(data):
    """补全至 4096 字节，防止短消息覆盖长消息后残留旧数据导致解析错误。"""
    try:
        mm.seek(0)
        mm.write(data.ljust(data_size, '\x00').encode('utf-8'))
        mm.flush()
    except Exception as e:
        print(f"[WRITE ERROR] {e}")

# ========== Q-Learning 核心逻辑 ==========
def get_q_values(x, y):
    key = (x, y)
    if key not in q_table:
        q_table[key] = [0.0, 0.0, 0.0, 0.0]
    return q_table[key]

def decide_action(x, y):
    """Epsilon-Greedy；探索时 80% 概率排除上步反向动作，抑制往返循环。"""
    q_values = get_q_values(x, y)
    if random.random() < epsilon:
        choices = [0, 1, 2, 3]
        if last_action is not None and random.random() < 0.8:
            rev = {0: 1, 1: 0, 2: 3, 3: 2}.get(last_action, -1)
            if rev in choices:
                choices.remove(rev)
        return random.choice(choices)
    max_q        = max(q_values)
    best_actions = [i for i, q in enumerate(q_values) if q == max_q]
    return random.choice(best_actions)  # 同值时随机打破平局

def update_q_table(cur_x, cur_y):
    """TD 更新 + Dyna-Q 规划（每步采样 10 条经验离线回放，加速稀疏奖励下的 Q 值传播）。"""
    global last_state, last_action, accumulated_reward
    if last_state is None or last_action is None:
        return
    q_old    = get_q_values(last_state[0], last_state[1])
    max_next = max(get_q_values(cur_x, cur_y))
    target   = accumulated_reward + gamma * max_next
    q_old[last_action] += lr * (target - q_old[last_action])
    experience_buffer.append((last_state, last_action, accumulated_reward, (cur_x, cur_y)))
    if len(experience_buffer) > 10:
        for s, a, r, next_s in random.sample(experience_buffer, 10):
            qs   = get_q_values(s[0], s[1])
            mq   = 0.0 if next_s is None else max(get_q_values(next_s[0], next_s[1]))
            qs[a] += lr * (r + gamma * mq - qs[a])

# ========== 主循环（IPC 消息调度） ==========
print("[Python AI] Q-Learning Backend Ready.")
write_to_shared_memory("READY")
last_handled_data = "READY"

while True:
    try:
        data = read_from_shared_memory()
    except Exception as _read_err:
        print(f"[QL] 共享内存读取异常: {_read_err}")
        time.sleep(0.05)
        continue

    if data and data != last_handled_data and data != "WAITING":

        # ====== GRID：加载地图，重置 Q 表 ======
        if data.startswith("GRID:"):
            parts = data[5:].split('|')
            try:
                maze_w, maze_h = int(parts[0]), int(parts[1])
            except (ValueError, IndexError) as e:
                print(f"[QL GRID] 包头解析失败: {e}")
                time.sleep(0.01)
                continue
            if len(parts) > 3:
                current_seed   = int(parts[2])
                # 按地图种子隔离存档，防止不同地图的 Q 表相互污染
                train_log_path = os.path.join(save_dir, f"training_log_QL_s{maze_w}_{current_seed}.csv")
                demo_log_path  = os.path.join(save_dir, f"training_log_QL_demo_s{maze_w}_{current_seed}.csv")
                log_path       = train_log_path
                qtable_path    = os.path.join(save_dir, f"q_table_QL_s{maze_w}_{current_seed}.csv")
                q_table.clear()
                load_q_table()
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
                experience_buffer.clear()
                max_episode_steps = maze_w * maze_h * 50
                epsilon_decay     = 0.990 if maze_w >= 15 else 0.995
            print(f"[QL GRID] {maze_w}x{maze_h} loaded, seed={current_seed}, max_steps={max_episode_steps}, decay={epsilon_decay}")
            write_to_shared_memory("GRID_OK")
            last_handled_data = "GRID_OK"

        # ====== ALGO：切换算法变体 ======
        elif data.startswith("ALGO:"):
            last_handled_data = data

        # ====== PARAM：更新超参数 ======
        # params[3] (epsilon_decay) 由 Python 端按迷宫大小自适应，C# 传 "_"，此处不覆盖
        elif data.startswith("PARAM:"):
            params    = data[6:].split(',')
            lr        = float(params[0])
            gamma     = float(params[1])
            epsilon   = float(params[2])
            last_handled_data = data

        # ====== STATE：每步推理，输出动作 ======
        elif data.startswith("STATE:"):
            state_info = data[6:].split('|')
            if len(state_info) < 2 or state_info[0] == '' or state_info[1] == '':
                continue
            try:
                cur_x, cur_y = float(state_info[0]), float(state_info[1])
            except ValueError:
                continue
            ix, iy = int(cur_x), int(cur_y)

            if len(state_info) >= 5 and state_info[4] != '':
                try:
                    r = float(state_info[4])
                    accumulated_reward += r; episode_reward += r
                except ValueError:
                    pass

            update_q_table(ix, iy)
            accumulated_reward = 0.0

            # 循环检测：短窗口内同一格子多次出现则施加递增惩罚
            grid_pos = (ix, iy)
            visit_count_map[grid_pos] = visit_count_map.get(grid_pos, 0) + 1
            position_history.append(grid_pos)
            short_visits = sum(1 for p in position_history if p == grid_pos)
            if short_visits >= 3:
                loop_penalty        = -0.3 * (short_visits - 2)
                accumulated_reward += loop_penalty
                episode_reward     += loop_penalty

            # 步数超限强制截断
            if episode_steps >= max_episode_steps:
                if last_state is not None and last_action is not None:
                    q_old              = get_q_values(last_state[0], last_state[1])
                    timeout_penalty    = -10.0
                    accumulated_reward += timeout_penalty
                    episode_reward     += timeout_penalty
                    q_old[last_action] += lr * (accumulated_reward - q_old[last_action])
                    experience_buffer.append((last_state, last_action, accumulated_reward, None))
                print(f"  [QL] EP {episode_count+1} 超过步数上限 ({max_episode_steps})，强制截断")
                accumulated_reward = 0.0
                last_state         = None
                last_action        = None
                position_history.clear()
                visit_count_map.clear()
                write_to_shared_memory(f"ACTION:0|{epsilon:.4f}")
                continue

            action      = decide_action(ix, iy)
            last_state  = (ix, iy)
            last_action = action
            episode_steps += 1
            write_to_shared_memory(f"ACTION:{action}|{epsilon:.4f}")
            # STATE 允许连续多帧处理，不更新 last_handled_data
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
            if rtype == 2:  # 到达终点：立即结算，防止下一局首帧将终点奖励归入起点
                if last_state is not None and last_action is not None:
                    q_old = get_q_values(last_state[0], last_state[1])
                    q_old[last_action] += lr * (accumulated_reward - q_old[last_action])
                    experience_buffer.append((last_state, last_action, accumulated_reward, None))
                print(f"★ [通关] Q-Learning | 局:{episode_count+1} | 步:{episode_steps} | 分:{episode_reward:.1f} ★")
                last_state         = None
                last_action        = None
                accumulated_reward = 0.0
            last_handled_data = data

        # ====== COMMAND：模式切换与控制指令 ======
        elif data.startswith("COMMAND:"):
            cmd_full  = data[8:]
            cmd_parts = cmd_full.split('|')
            cmd       = cmd_parts[0]
            if cmd in ("QUIT", "EXIT"):
                print("[Python AI] 收到退出指令，进程终止。")
                break
            elif cmd == "DEMO":
                q_table.clear()
                load_q_table(use_best=True)
                is_demo = True
                epsilon = 0.0  # Demo 完全禁止随机探索
                log_path = demo_log_path
                init_log_file()
                resume_training_state(update_best=False)
                print(f"-> DEMO MODE: Loaded Best Q-Table for seed {current_seed} ({len(q_table)} states). Resuming from EP {episode_count}.")
            elif cmd == "START":
                q_table.clear()
                load_q_table()
                is_demo = False
                log_path = train_log_path
                init_log_file()
                resume_training_state()
                last_state  = None
                last_action = None
                print(f"-> START MODE: Loaded model for seed {current_seed}. Resuming from EP {episode_count}, Eps={epsilon:.4f}")
            elif cmd == "RESET":
                hit_count_this_ep = int(cmd_parts[1]) if len(cmd_parts) > 1 else 0
                episode_count += 1
                if not is_demo and epsilon > 0.0:
                    epsilon = max(min_epsilon, epsilon * epsilon_decay)
                save_training_log(episode_count, episode_steps, hit_count_this_ep, episode_reward, epsilon)
                # DEMO 不写 Q 表：旧版无此 gate，演示数据会反向覆盖训练成果
                if not is_demo:
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                        save_q_table(is_best=True)
                    else:
                        save_q_table(is_best=False)
                print(f"  EP {episode_count} | Steps: {episode_steps} | Hits: {hit_count_this_ep} | Reward: {episode_reward:.1f} | Eps: {epsilon:.3f} | Best: {best_reward:.1f}")
                last_state         = None
                last_action        = None
                accumulated_reward = 0.0
                episode_reward     = 0.0
                episode_steps      = 0
                position_history.clear()
                visit_count_map.clear()
            last_handled_data = data

    # 出让 CPU 时间片，防止死循环独占核心导致 Unity 端通信延迟
    time.sleep(0.01)
