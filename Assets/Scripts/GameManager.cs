using System;
using System.Collections;
using System.Diagnostics;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Text;
using UnityEngine;
using Debug = UnityEngine.Debug;
// 游戏核心大管家 (GameManager)
// 这个脚本就是咱们这边的老大，主要管这几件事情：
// 1. 管一下背后的 Python什么时候开始什么时候结束。
// 2. 管“共享内存管道”，好让 C# 和 Python交互通信。
// 3. 记账大师，算算走了几步走错了算几下，算算到底过了多少局。
// 4. 把游戏里的情报送给AI，然后把AI想出来的步子指挥方块。
public class GameManager : MonoBehaviour
{
    public static GameManager instance;
    [Header("AI角色")]
    [Tooltip("场景中实际移动的智能体对象")]
    public PlayerController aiPlayer;
    [Header("奖励设置")]
    [Tooltip("全局奖励打分板定义")]
    public RewardConfig rewardConfig;
    [Header("UI 界面")]
    [Tooltip("实时绘制训练曲线的图表组件")]
    public MazeAI.UI.TrainingChartUI trainingChart;
    [Header("Python脚本路径")]
    [Tooltip("指向本地 rl_maze_ai_qlearning.py 的绝对或相对路径")]
    public string pythonScriptPath = Application.dataPath + "/Python/rl_maze_ai_qlearning.py";
    // 运行模式：0=手动测试, 1=算法自动训练, 2=模型加载演示
    public int runMode;
    // 选择算法：0=Q-Learning, 1=DQN, 2=PPO
    public int selectAlgorithm;
    // 自动训练用的连续测试索引
    public int currentSeedIndex = 0;
    // 自动训练用的探索率回合记录
    [HideInInspector] public System.Collections.Generic.List<float> episodeEpsilonHistory = new System.Collections.Generic.List<float>();
    // PPO 自动训练奖励收敛检测：当前地图的逐局奖励记录（切图时清空）
    private System.Collections.Generic.List<float> _ppoMapRewardHistory = new System.Collections.Generic.List<float>();
    // 训练超参数面板映射
    public TrainParam trainParam;
    // 当前回合内的撞墙累计次数
    public int hitCount;
    // ========== 训练指标统计 ==========
    [HideInInspector] public int episodeCount = 0;        // 历史已执行的回合总数
    [HideInInspector] public float episodeTotalReward = 0; // 本回合累计获得的奖励(Reward)合计数
    [HideInInspector] public int episodeStepCount = 0;     // 本回合已经走过的格数
    [HideInInspector] public int totalSuccessCount = 0;    // 历史成功抵达终点的次数（用于计算胜率）
    // 由于移动和撞墙是连续高速发生的，为防止单帧内多次通信产生读写冲突或奖励漏算，
    // 将日常奖励缓存至 pendingReward，并在下一次发送 STATE 时一并打包传给 Python。
    private float pendingReward = 0;
    private bool isResettingEpisode = false; // 防止终点双重触发导致 CreateMaze 逻辑重入崩盘
    // ============ [第一步：奖励重塑相关变量] ============
    private float _lastDistanceToGoal = -1; // 上一步离终点的距离
    public Vector2 goalPos;                // 目标点（终点）坐标 (由 MazeGenerator 告知)
    private System.Collections.Generic.HashSet<Vector2Int> _visitedCells = new System.Collections.Generic.HashSet<Vector2Int>();
    // ============ [循环检测与步数限制] ============
    private System.Collections.Generic.Dictionary<Vector2Int, int> _cellVisitCount = new System.Collections.Generic.Dictionary<Vector2Int, int>(); // 每格被踩的次数
    // ========== 实时图表持久化数据 ==========
    [HideInInspector] public System.Collections.Generic.List<float> rewardHistory = new System.Collections.Generic.List<float>();
    [HideInInspector] public System.Collections.Generic.List<int> stepHistory = new System.Collections.Generic.List<int>();
    [HideInInspector] public System.Collections.Generic.List<float> epsilonHistory = new System.Collections.Generic.List<float>();
    // IPC 进程间通信组件
    private MemoryMappedFile mmf;
    private MemoryMappedViewAccessor view;
    private Process pythonProcess;
    private bool isPythonProcessRunning;
    private bool hasReceivedReady;
    // 共享内存大小（字节），C#与Python必须严格保证一致大小
    private const int DATA_SIZE = 4096;
    public enum RewardType
    {
        NormalMove, // 走到合法空白格
        HitWall,    // 撞到墙壁被阻挡
        ReachEnd,   // 顺利抵达终点
    }
    [Serializable]
    public class RewardConfig
    {
        public float normalMoveReward = -0.05f;  // 削弱闲逛惩罚，防止AI不敢动弹
        public float hitWallReward = -0.5f;      // 恢复其他算法的默认设定
        public float reachEndReward = 100f;      // 恢复其他算法的默认设定
    }
    [Serializable]
    public class TrainParam
    {
        public float lr = 0.1f;           // 学习率 (Alpha)
        public float gamma = 0.95f;       // 衰减因子 (Gamma) — 提高到0.95让终点信号传播更远
        public float epsilon = 0.9f;      // 探索率 (Epsilon)
        public float epsilonDecay = 0.995f; // 探索衰减率 (Python端会根据迷宫大小动态调整)
        public int batchSize = 32;        // 经验回放批次大小
        public float algoLr = 0.001f;     // 神经网络学习率 (DQN/PPO预留)
    }
    private void Awake()
    {
        if (instance == null)
            instance = this;
        else
            Destroy(gameObject);
        // 保证 GameManager 跨场景不被摧毁，因为包含底层 IPC 句柄
        DontDestroyOnLoad(gameObject);
    }
    private void OnSceneLoaded(UnityEngine.SceneManagement.Scene scene, UnityEngine.SceneManagement.LoadSceneMode mode)
    {
        if (scene.name == "GameScene")
        {
            // 在游玩关卡场景载入后，重新绑定游玩角色
            GameObject playerObj = GameObject.FindGameObjectWithTag("Player");
            if (playerObj != null)
            {
                aiPlayer = playerObj.GetComponent<PlayerController>();
            }
            InitAfterSceneLoaded();
        }
    }
    private void Start()
    {
        runMode = 0; // 默认采用手动测试模式
        UnityEngine.SceneManagement.SceneManager.sceneLoaded += OnSceneLoaded;
        // 一次性迁移：将旧版 PlayerPrefs key 转换为包含迷宫尺寸的新格式
        MigrateOldSeedKeys();
    }
    /// <summary>
    /// 将旧格式的种子 key (MazeRandomSeed_{algo}) 迁移到新格式 (MazeRandomSeed_{algo}_{size})
    /// 只执行一次，迁移完成后写入标记防止重复执行
    /// </summary>
    private void MigrateOldSeedKeys()
    {
        if (PlayerPrefs.GetInt("SeedKeyMigrated_v2", 0) == 1) return; // 已迁移过
        for (int algo = 0; algo <= 2; algo++)
        {
            string oldSeedKey = "MazeRandomSeed_" + algo;
            string oldWidthKey = "MazeSavedWidth_" + algo;
            string oldHeightKey = "MazeSavedHeight_" + algo;
            if (PlayerPrefs.HasKey(oldSeedKey))
            {
                int oldSeed = PlayerPrefs.GetInt(oldSeedKey);
                int oldW = PlayerPrefs.GetInt(oldWidthKey, 11); // 默认按简单模式
                // 写入新格式 key
                string newKey = $"MazeRandomSeed_{algo}_{oldW}";
                PlayerPrefs.SetInt(newKey, oldSeed);
                Debug.Log($"<color=green>[数据迁移] 算法{algo} 尺寸{oldW} 种子{oldSeed} -> {newKey}</color>");
                // 清理旧 key
                PlayerPrefs.DeleteKey(oldSeedKey);
                PlayerPrefs.DeleteKey(oldWidthKey);
                PlayerPrefs.DeleteKey(oldHeightKey);
            }
        }
        PlayerPrefs.SetInt("SeedKeyMigrated_v2", 1);
        PlayerPrefs.Save();
        Debug.Log("[数据迁移] 旧版种子 Key 迁移完成！");
    }
    private void OnDestroy()
    {
        // 程序退出前务必安全卸载 Python 进程和内存句柄，防止形成僵尸进程
        StopPythonProcess();
        CloseSharedMemory();
    }
    #region 模式 & 算法切换
    public void SwitchRunMode(int mode)
    {
        runMode = mode;
    }
    // 大厅选择完模式后，进入正式场景第一时间叫它。
    public void InitAfterSceneLoaded()
    {
        if (aiPlayer == null)
            return;
        aiPlayer.canMove = true;
        aiPlayer.ResetPlayer();
        hitCount = 0;
        if (runMode == 0)
        {
            StopPythonProcess(); // 手动模式确保关闭 Python
        }
        else
        {
            StartPythonProcess();
            InitSharedMemory();
            // 在训练/演示模式下，主动发出第一个起步信号
            StartTrain();
        }
    }
    // 收到大厅的选择后，把算法要用的脑补参数发给后面的大黑盒
    public void SwitchAlgorithm(int algo)
    {
        selectAlgorithm = algo;
        // 首次初始化参数
        UpdateRealtimeParams();
        // 算法切换后，立刻重新加载该算法在该难度下的历史战绩
        LoadHistoryData();
    }
    /// <summary>
    /// 将最新的超参数实时同步给 Python 后端，让 AI 立即执行新的学习策略
    /// </summary>
    public void UpdateRealtimeParams()
    {
        if (view == null) return;
        SendDataToPython($"ALGO:{selectAlgorithm}");
        SendDataToPython($"PARAM:{trainParam.lr},{trainParam.gamma},{trainParam.epsilon},{trainParam.epsilonDecay},{trainParam.batchSize},{trainParam.algoLr}");
        Debug.Log($"<color=yellow>[Param Update] 参数同步完成: LR:{trainParam.lr} | Gamma:{trainParam.gamma} | Eps:{trainParam.epsilon}</color>");
    }
    // 获取当前训练组合的唯一标识符（算法ID+迷宫尺寸）
    private string GetHistoryKey()
    {
        // 格式如：HistoryData_0_11 (表示 Q-Learning, 11x11 难度)
        return $"HistoryData_{selectAlgorithm}_{GameData.mazeWidth}";
    }
    private void SaveHistoryData()
    {
        string key = GetHistoryKey();
        // 将 List 转为逗号分隔的字符串存储
        string rewardStr = string.Join(",", rewardHistory);
        string stepStr = string.Join(",", stepHistory);
        PlayerPrefs.SetString(key + "_Reward", rewardStr);
        PlayerPrefs.SetString(key + "_Step", stepStr);
        PlayerPrefs.Save();
    }
    private void LoadHistoryData()
    {
        rewardHistory.Clear();
        stepHistory.Clear();
        string key = GetHistoryKey();
        if (PlayerPrefs.HasKey(key + "_Reward"))
        {
            string[] rewards = PlayerPrefs.GetString(key + "_Reward").Split(',');
            foreach (var r in rewards) { if (float.TryParse(r, out float val)) rewardHistory.Add(val); }
        }
        if (PlayerPrefs.HasKey(key + "_Step"))
        {
            string[] steps = PlayerPrefs.GetString(key + "_Step").Split(',');
            foreach (var s in steps) { if (int.TryParse(s, out int val)) stepHistory.Add(val); }
        }
        // 加载完数据后，通知 UI 刷新显示
        if (trainingChart != null)
        {
            trainingChart.UpdateChartTitle();
            trainingChart.AddDataPoint(0, 0); // 触发重绘
        }
    }
    public string GetAlgorithmName()
    {
        return selectAlgorithm switch
        {
            0 => "Q-Learning",
            1 => "DQN",
            2 => "PPO",
            _ => "Unknown"
        };
    }
    #endregion
    #region 奖励发放系统
    // 小球碰到了好事或者坏事（走路、撞墙、通关），就会找这就行结算登记
    public void TriggerReward(RewardType type)
    {
        float reward = type switch
        {
            RewardType.NormalMove => rewardConfig.normalMoveReward,
            RewardType.HitWall => (selectAlgorithm == 2) ? -1.0f : rewardConfig.hitWallReward, // PPO 专属撞墙巨罚
            RewardType.ReachEnd => (selectAlgorithm == 2) ? 50f : rewardConfig.reachEndReward, // PPO 专属终点平滑分
            _ => 0,
        };
        // ============ [奖励重塑逻辑：距离启发式 & 探索奖励] ============
        if (type == RewardType.NormalMove && aiPlayer != null)
        {
            Vector2 currentPos = aiPlayer.transform.position;
            float currentDist = Vector2.Distance(currentPos, goalPos);
            // 1. 距离奖励 (Potential Reward): 改为直接使用连续的差值，绝对保证前后移动积分抵消，防止原地横跳刷分漏洞
            if (_lastDistanceToGoal > 0)
            {
                float diff = _lastDistanceToGoal - currentDist;
                // 【核心修复】：根据你的要求，修改仅限 PPO 算法
                // 其他算法依然保持 diff * 1.0f 的严苛机制
                float distanceMultiplier = (selectAlgorithm == 2) ? 0.1f : 1.0f;
                reward += diff * distanceMultiplier;
            }
            _lastDistanceToGoal = currentDist;
            // 2. 探索奖励 (Exploration Reward): 第一次走到的格子额外加分，重复走的扣分
            Vector2Int gridPos = new Vector2Int(Mathf.RoundToInt(currentPos.x), Mathf.RoundToInt(currentPos.y));
            if (!_visitedCells.Contains(gridPos))
            {
                reward += 0.2f; // 适当降低发现新格子过高的奖励，防止它沉迷走路忘了终极目标
                _visitedCells.Add(gridPos);
                _cellVisitCount[gridPos] = 1;
            }
            else
            {
                // 递增式重复惩罚：踩得越多罚得越狠，有效打破循环陷阱
                if (!_cellVisitCount.ContainsKey(gridPos)) _cellVisitCount[gridPos] = 1;
                _cellVisitCount[gridPos]++;
                int visitNum = _cellVisitCount[gridPos];
                // 基础 -0.1，每多踩一次额外 -0.05，上限 -1.0
                float loopPenalty = Mathf.Min(0.1f + (visitNum - 2) * 0.05f, 1.0f);
                reward -= loopPenalty;
            }
        }
        // 统计面板数据累加
        episodeTotalReward += reward;
        episodeStepCount++;
        if (type == RewardType.HitWall)
            hitCount++;
        // 全算法步数上限截断控制：防止 agent 在大迷宫中无限循环
        // 根据迷宫面积动态计算：小迷宫(11x11)=6050步, 大迷宫(15x15)=11250步, PPO保持30000
        int maxSteps;
        if (selectAlgorithm == 2)
          maxSteps = GameData.mazeWidth * GameData.mazeHeight * 50;
        else
            maxSteps = GameData.mazeWidth * GameData.mazeHeight * 50; // DQN/QL: 面积×50
        bool isTimeout = (episodeStepCount >= maxSteps);
        if (type == RewardType.ReachEnd || isTimeout)
        {
            if (isTimeout)
            {
                // 超时惩罚，这把算是废了
                float timeoutPenalty = (selectAlgorithm == 2) ? 5.0f : 10.0f; // DQN/QL 给更重的超时罚分
                reward -= timeoutPenalty;
                episodeTotalReward -= timeoutPenalty;
            }
            // 实时更新 UI 数据报表
            if (trainingChart != null)
            {
                trainingChart.AddDataPoint(episodeTotalReward, episodeStepCount);
            }
            // 终点或者超时，不再需要 Python 返回后续行动，发送强力经验包
            // rtype: 2=通关, 1=非通关被截断
            int rtype = (type == RewardType.ReachEnd) ? 2 : 1;
            SendDataToPython($"REWARD:{reward}|0|0|{rtype}");
            if (runMode == 1 || runMode == 2 || runMode == 3)
            {
                if (type == RewardType.ReachEnd)
                    totalSuccessCount++;
                episodeCount++;
                // 自动训练：按回合记录探索率用于判断是否收敛
                episodeEpsilonHistory.Add(trainParam.epsilon);
                if (episodeEpsilonHistory.Count > 50) episodeEpsilonHistory.RemoveAt(0);
                // 实时压入持久化曲线数据
                rewardHistory.Add(episodeTotalReward);
                // PPO 自动挂机训练：额外记录当前地图的奖励数据用于收敛判定
                if (selectAlgorithm == 2 && runMode == 3)
                    _ppoMapRewardHistory.Add(episodeTotalReward);
                stepHistory.Add(episodeStepCount);
                if (rewardHistory.Count > 100) rewardHistory.RemoveAt(0);
                if (stepHistory.Count > 100) stepHistory.RemoveAt(0);
                // 同步记录 Epsilon 历史用于绘图（每局记录一次，与 reward/step 保持同频）
                epsilonHistory.Add(trainParam.epsilon);
                if (epsilonHistory.Count > 100) epsilonHistory.RemoveAt(0);
                // 立即持久化到本地硬盘，防止程序崩溃丢失数据
                SaveHistoryData();
                Debug.Log($"[系统统计] 第 {episodeCount} 回合结束! 步数:{episodeStepCount} 撞墙:{hitCount} 状态:{(rtype==2?"通关":"超时重置")} 累计奖励:{episodeTotalReward:F1} 历史通关率:{(float)totalSuccessCount / episodeCount * 100:F1}%");
                // 数据归位
                episodeTotalReward = 0;
                episodeStepCount = 0;
                // 统一交给协同程序处理下一回合或切换地图逻辑
                StartCoroutine(AutoNextEpisode());
            }
            else
            {
                // 手动模式
                if (type == RewardType.ReachEnd)
                {
                    if (aiPlayer != null) aiPlayer.LockMovement();
                    UIManager.instance?.ShowPanel("EndPanel");
                }
                else
                {
                    if (aiPlayer != null) aiPlayer.ResetPlayer();
                    episodeStepCount = 0;
                }
            }
        }
        else
        {
            // 对于高频度发生的基础移动和撞墙行为，将其放进账本（pendingReward），
            // 在下一个 OnAIMoveComplete 时由统一管道发出，提升通信吞吐稳定性并防止数据覆盖
            pendingReward += reward;
        }
    }
    // 一局打完了，负责清理然后再开下一把
    private IEnumerator AutoNextEpisode()
    {
        // 训练模式用极短等待加速迭代，其他模式保持安全间隔
        float gap = (runMode == 1 || runMode == 3) ? 0.02f : 0.1f;
        if (aiPlayer != null) aiPlayer.LockMovement();
        // 停留在终点一瞬间，让你能看清
        yield return new WaitForSeconds(gap);
        // 重置环境，发送 COMMAND:RESET 给 Python
        ResetTrain();
        // 发送完 RESET 后稍等，以便 Python 处理存盘
        yield return new WaitForSeconds(gap);
        // 如果是自动化挂机训练模式，就验证一下是不是需要自动切图了
        if (runMode == 3)
        {
            if (CheckAutoTrainConvergence())
            {
                // 如果已经触发切图/切算法，就不必执行这局后续开头的运算了，中断携程！
                yield break;
            }
        }
        if (aiPlayer != null)
            aiPlayer.canMove = true;
        // AI 已经重生至起点，让其开始推算新的一局
        OnAIMoveComplete();
    }
    #endregion
    #region AI控制与通信
    // 只要小球走完一步了，立马问 Python 下一步往哪里走。
    public void OnAIMoveComplete()
    {
        if (!isPythonProcessRunning || runMode == 0 || aiPlayer == null)
            return;
        // 必须开携程，因为物理状态改变和 IPC读取不能阻塞 Unity 的主渲染线程
        StartCoroutine(RequestActionCoroutine());
    }
    private IEnumerator RequestActionCoroutine()
    {
        // 故意让出当前帧执行权限，等待上一帧由于角色位移带来的底层碰撞与逻辑收尾完毕
        yield return null;
        // 确保共享内存已被清除，避免Python读到旧数据
        // 训练模式缩短等待以提升迭代速度，其他模式保持安全间隔
        yield return new WaitForSeconds((runMode == 1) ? 0.02f : 0.1f);
        // 1. 发射包含角色实时空间坐标、步数以及所有沿途结算奖励的数据包给 Python。
        string state = $"STATE:{aiPlayer.transform.position.x}|{aiPlayer.transform.position.y}|{aiPlayer.canMove}|{aiPlayer.stepCount}|{pendingReward}";
        SendDataToPython(state);
        pendingReward = 0; // 发射后立刻清零本地账本
        string actionData = "";
        float timeout = 5.0f; // 死锁防卫：增加超时时间到5秒，给深度学习计算更充裕的时间
        float timePassed = 0f;
        // 2. 将 C# 的当前携程暂停，去 IPC 隧道里紧密轮询探查 Python 的 "ACTION" 指令答复
        while (timePassed < timeout)
        {
            actionData = ReadDataFromPython();
            if (actionData.StartsWith("ACTION:"))
            {
                break; // 接头成功,跳出等待
            }
            yield return null;
            timePassed += Time.deltaTime;
        }
        // 3. 执行获取到的预言家(AI)行动决策
        if (actionData.StartsWith("ACTION:"))
        {
            string[] parts = actionData.Split(':')[1].Split('|');
            int dir = int.Parse(parts[0]);
            // 如果后端传回了 Epsilon，同步更新到本地，让 UI 滑条能自动跟着变（尤其是自动衰减时）
            if (parts.Length > 1)
            {
                if (float.TryParse(parts[1], out float eps))
                {
                    trainParam.epsilon = eps;
                }
            }
            // 下发 WAIT 占位符防多条指令黏连 (粘包防护)
            SendDataToPython("WAITING");
            ExecuteAIMove(dir);
        }
        else
        {
            Debug.LogWarning($"不好，等那个 Python 小弟等太久了！(要么死机要么罢工了) - Last read: {(string.IsNullOrEmpty(actionData) ? "EMPTY" : actionData.Substring(0, Math.Min(30, actionData.Length)))}");
        }
    }
    // 把 Python 报的 "0, 1, 2, 3" 翻译成真正操作的上下左右
    public void ExecuteAIMove(int dir)
    {
        if (aiPlayer == null || !aiPlayer.canMove || aiPlayer.isMoving)
            return;
        Vector2 moveDir = dir switch
        {
            0 => Vector2.up,
            1 => Vector2.down,
            2 => Vector2.left,
            3 => Vector2.right,
            _ => Vector2.zero,
        };
        aiPlayer.TryMove(moveDir);
    }
    #endregion
    #region Python 生命周期及 IPC 管理
    private void StartPythonProcess()
    {
        if (isPythonProcessRunning)
            return;
        try
        {
            // 根据切换的算法类型判定究竟应当拉起后台哪套 Python 程序作为驱动网络
            string scriptName = "rl_maze_ai_qlearning.py"; // 默认/保底值
            if (selectAlgorithm == 0) scriptName = "rl_maze_ai_qlearning.py";     // Q-Learning
            if (selectAlgorithm == 1) scriptName = "rl_maze_ai_dqn.py"; // DQN
            if (selectAlgorithm == 2) scriptName = "rl_maze_ai_ppo.py"; // PPO
            string realPath = Application.dataPath + "/Python/" + scriptName;
            Debug.Log($"[IPC] 即将唤起本地算法端守护进程 ({scriptName})。 所在路径: {realPath}");
            ProcessStartInfo psi = new ProcessStartInfo
            {
                FileName = "python",
                Arguments = $"-u \"{realPath}\"",
                UseShellExecute = true,  // true 代表它会弹出一个黑色的终端窗口，极大的方便观测 Python 的崩溃报错日志
            };
            pythonProcess = Process.Start(psi);
            isPythonProcessRunning = true;
            Debug.Log("[IPC] Python 进程成功启动, 进程号 PID=" + pythonProcess.Id);
        }
        catch (Exception e)
        {
            Debug.LogError("[IPC] 启动 Python 算法端环境失败，请核查系统是否安装 python 且添加了环境变量: " + e.Message);
        }
    }
    private void StopPythonProcess()
    {
        if (isPythonProcessRunning)
        {
            SendDataToPython("COMMAND:QUIT"); // 让后台死命执行中的 Python 进程听话自裁
            // 稍等一会儿，给 Python 发送并响应 QUIT 的时间，防止野蛮 Kill 导致僵尸进程
            System.Threading.Thread.Sleep(200);
        }
        if (pythonProcess != null && !pythonProcess.HasExited)
        {
            try
            {
                pythonProcess.Kill();
                pythonProcess.WaitForExit(500); // 确保真的死了
            }
            catch { }
        }
        isPythonProcessRunning = false;
        hasReceivedReady = false;
    }
    // 创建 “共享内存文件夹” 供 C# 和 Python 之间交互通信
    private void InitSharedMemory()
    {
        try
        {
            // CreateOrOpen 表示如果有旧的没关干净就直接挂载（能避免大量 Restart 时的冲突）
            mmf = MemoryMappedFile.CreateOrOpen("MazeRLSharedMemory", DATA_SIZE);
            view = mmf.CreateViewAccessor();
            // 必须在挂载后立即清空内存，防止上一次运行残留的 "READY" 骗过 C# 提前发送 GRID 数据
            byte[] emptyData = new byte[DATA_SIZE];
            view.WriteArray(0, emptyData, 0, DATA_SIZE);
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogError("[IPC] 初始化共享内存映射表失败: " + e.Message);
        }
    }
    private void CloseSharedMemory()
    {
        view?.Dispose();
        mmf?.Dispose();
    }
    /// <summary>
    /// 通用发包接口，将字符串以 UTF-8 转换为固定长度的数据包后推断进管道。
    /// </summary>
    private void SendDataToPython(string data)
    {
        if (view == null)
            return;
        // PadRight 是必不可少的，空余的字节必须用 '\0' 填满，避免脏数据残留导致 Python 解析异常。
        byte[] bytes = Encoding.UTF8.GetBytes(data.PadRight(DATA_SIZE, '\0'));
        view.WriteArray(0, bytes, 0, bytes.Length);
    }
    private string ReadDataFromPython()
    {
        if (view == null)
            return "";
        byte[] bytes = new byte[DATA_SIZE];
        view.ReadArray(0, bytes, 0, bytes.Length);
        return Encoding.UTF8.GetString(bytes).Trim('\0');
    }
    // 把迷宫的地图整个背下来发给 Python，因为 Python 自己看不到画面，
    public void SyncGridToPython()
    {
        MazeGenerator generator = FindAnyObjectByType<MazeGenerator>();
        if (generator != null)
        {
            // 通过传输关联的种子(seed)，Python可以按种分类存档Q表日志！
            // 必须与 MazeGenerator 使用同一套按算法+尺寸分开的 key
            string seedKey = $"MazeRandomSeed_{selectAlgorithm}_{GameData.mazeWidth}";
            int seed = PlayerPrefs.GetInt(seedKey, 0);
            string gridStr = generator.GetGridString();
            SendDataToPython($"GRID:{generator.Width}|{generator.Height}|{seed}|{gridStr}");
        }
    }
    #endregion
    #region 训练指挥调度
    /// <summary>
    /// 从 Python 训练日志 CSV 中恢复上次的回合数，实现 Unity 端的断点续训显示
    /// </summary>
    private void ResumeEpisodeCount()
    {
        try
        {
            string seedKey = $"MazeRandomSeed_{selectAlgorithm}_{GameData.mazeWidth}";
            int seed = PlayerPrefs.GetInt(seedKey, 0);
            int mazeW = GameData.mazeWidth;
            // 根据算法类型拼出对应的 CSV 文件名（必须与 Python 端完全一致）
            string csvName = selectAlgorithm switch
            {
                0 => $"training_log_QL_s{mazeW}_{seed}.csv",
                1 => $"dqn_log_s{mazeW}_{seed}.csv",
                2 => $"ppo_log_s{mazeW}_{seed}.csv",
                _ => ""
            };
            if (string.IsNullOrEmpty(csvName)) return;
            string csvPath = Path.Combine(Application.dataPath, "Python", "training_data~", csvName);
            if (!File.Exists(csvPath))
            {
                episodeCount = 0;
                totalSuccessCount = 0;
                Debug.Log("[系统] 未找到历史训练日志，回合数从 0 开始。");
                return;
            }
            // 读取 CSV 最后一行的 episode 数
            string[] lines = File.ReadAllLines(csvPath);
            if (lines.Length > 1)
            {
                // 从尾部找到最后一行有效数据
                for (int i = lines.Length - 1; i >= 1; i--)
                {
                    string line = lines[i].Trim();
                    if (string.IsNullOrEmpty(line)) continue;
                    string[] cols = line.Split(',');
                    if (cols.Length >= 1 && int.TryParse(cols[0], out int lastEp))
                    {
                        episodeCount = lastEp;
                        totalSuccessCount = lastEp; // 每回合都到达了终点才会记录
                        Debug.Log($"<color=cyan>[断点续训] 从历史日志恢复：已完成 {episodeCount} 回合</color>");
                        return;
                    }
                }
            }
            episodeCount = 0;
            totalSuccessCount = 0;
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[系统] 读取历史训练日志失败: {e.Message}");
            episodeCount = 0;
            totalSuccessCount = 0;
        }
    }
    /// <summary>
    /// 发射第一次心跳包
    /// </summary>
    public void StartTrain()
    {
        if (runMode == 0 || aiPlayer == null)
            return;
        StartCoroutine(StartTrainRoutine());
    }
    private IEnumerator StartTrainRoutine()
    {
        // 因为 PyTorch 引入导致 Python 解释器首次启动通常需要 3~5 秒
        // 我们必须轮询等待 Python 写入 "READY" 信号，否则直接发 GRID 极容易被后续命令覆盖导致 AI 失明！
        float waitTime = 0;
        string data = "";
        // 如果 Python 已经处于运行状态且曾经握手成功，则跳过 READY 等待
        // 因为当重新生成地图时，Python 早就进入了主循环（它是肯定不会再发一次 READY 的）。
        // 增加 PyTorch 启动宽容度到 60 秒，因为一些 CPU 初始化深度学习环境非常慢！
        while (!hasReceivedReady && !data.StartsWith("READY") && waitTime < 60.0f)
        {
            yield return new WaitForSeconds(0.1f);
            waitTime += 0.1f;
            data = ReadDataFromPython();
        }
        if (!hasReceivedReady)
        {
            if (waitTime >= 60.0f)
                Debug.LogError($"[IPC - 严重超时] Python未在 60 秒内回复 READY 信号，管道建立疑似失败！");
            else
            {
                Debug.Log($"[IPC] Python 已于 {waitTime:F1}s 后完全加载完毕并成功发送 READY 接头信号。");
                hasReceivedReady = true;
            }
        }
        aiPlayer.ResetPlayer();
        hitCount = 0;
        // 发送环境图信息
        SyncGridToPython();
        // 【关键修复】等待 Python 确认收到 GRID，防止被后续指令覆盖
        float gridWait = 0;
        while (gridWait < 10.0f)
        {
            yield return new WaitForSeconds(0.1f);
            gridWait += 0.1f;
            string response = ReadDataFromPython();
            if (response.StartsWith("GRID_OK"))
            {
                Debug.Log($"[IPC] Python 已确认收到 GRID 数据 ({gridWait:F1}s)");
                break;
            }
        }
        if (gridWait >= 10.0f)
            Debug.LogWarning("[IPC] Python 未确认 GRID 收讫，继续执行（可能导致数据错乱）");
        yield return new WaitForSeconds(0.1f);
        // 指针指令：如果是 DEMO(演示)，那么 Python 会强制关停 epsilon 让其实验 100% 收敛走势。
        if (runMode == 2)
            SendDataToPython("COMMAND:DEMO");
        else
        {
            SendDataToPython("COMMAND:START");
            // 同步恢复 Unity 端的回合计数，与 Python 端保持一致
            ResumeEpisodeCount();
            // PPO 自动训练开始新地图时清空奖励收敛记录
            if (selectAlgorithm == 2 && runMode == 3)
                _ppoMapRewardHistory.Clear();
        }
        yield return new WaitForSeconds(0.2f);
        // 激活 AI 行动环路
        OnAIMoveComplete();
    }
    public void StopTrain()
    {
        SendDataToPython("COMMAND:STOP");
    }
    public void PauseTrain()
    {
        SendDataToPython("COMMAND:PAUSE");
        if (aiPlayer != null)
            aiPlayer.canMove = false;
    }
    public void ResumeTrain()
    {
        SendDataToPython("COMMAND:RESUME");
        if (aiPlayer != null)
            aiPlayer.canMove = true;
        OnAIMoveComplete();
    }
    public void ResetTrain()
    {
        if (aiPlayer != null)
            aiPlayer.ResetPlayer();
        // 重置奖励启发式状态
        _lastDistanceToGoal = -1;
        _visitedCells.Clear();
        _cellVisitCount.Clear();
        // 将旧的统计数字一并通过 RESET 告知算法模型，算法需要重置一些 Episode-Level 的数据
        SendDataToPython($"COMMAND:RESET|{hitCount}");
        // 确保上一个 hitCount 被安全发出后才擦除
        hitCount = 0;
    }
    #region 自动化批量训练挂机管家
    private bool CheckAutoTrainConvergence()
    {
        // PPO 不使用 epsilon-greedy 探索策略，需要基于奖励稳定性来判定收敛
        if (selectAlgorithm == 2)
        {
            return CheckPPORewardConvergence();
        }
        // Q-Learning / DQN: 原有的 epsilon 稳定性判断逻辑
        int checkEpsCount = 10;
        if (episodeEpsilonHistory.Count < checkEpsCount) return false;
        float firstEps = episodeEpsilonHistory[episodeEpsilonHistory.Count - checkEpsCount];
        bool isConverged = true;
        for (int i = episodeEpsilonHistory.Count - checkEpsCount + 1; i < episodeEpsilonHistory.Count; i++)
        {
            if (Mathf.Abs(episodeEpsilonHistory[i] - firstEps) > 0.0001f)
            {
                isConverged = false;
                break;
            }
        }
        if (isConverged)
        {
            Debug.Log($"<color=green>[Auto Train] 当前地图 (算法: {GetAlgorithmName()}, 种子索引: {currentSeedIndex}) 探索率已连续 {checkEpsCount} 局保持 {firstEps:F4} 不变，判定收敛！准备切换下一地图。</color>");
            episodeEpsilonHistory.Clear();
            SwitchToNextMapOrAlgorithm();
            return true;
        }
        return false;
    }
    /// <summary>
    /// PPO 专用收敛检测：基于最近 N 局奖励的「均值 + 变异系数」双重指标判断
    /// PPO 是策略梯度算法，探索通过网络输出的概率分布自然实现，
    /// 不存在 epsilon 衰减机制，因此改用奖励稳定性来判定收敛：
    ///   条件1: 最近窗口期内的平均奖励 > 阈值 → 确认 agent 确实在通关
    ///   条件2: 奖励的变异系数(CV = 标准差/均值) < 30% → 确认表现已趋于稳定
    ///   安全阀: 单张地图超过 200 局仍未收敛则强制切换，防止无限挂机
    /// </summary>
    private bool CheckPPORewardConvergence()
    {
        int minEpisodes = 30;        // 最少需要积累 30 局数据才开始判定
        int windowSize = 20;         // 取最近 20 局作为分析窗口
        float minMeanReward = 5.0f;  // 平均奖励最低门槛（正值说明 agent 在通关，而非持续超时）
        float maxCV = 0.3f;          // 变异系数(CV)最大容许值（30% 以内视为稳定）
        int maxEpisodesPerMap = 200;  // 安全阀：单张地图最大训练局数
        if (_ppoMapRewardHistory.Count < minEpisodes) return false;
        // 安全阀：超过最大局数强制切换，防止永远达不到收敛却无限挂机
        if (_ppoMapRewardHistory.Count >= maxEpisodesPerMap)
        {
            Debug.Log($"<color=yellow>[Auto Train] PPO 已训练 {_ppoMapRewardHistory.Count} 局仍未达收敛标准，安全阀触发 → 强制切换下一地图</color>");
            episodeEpsilonHistory.Clear();
            SwitchToNextMapOrAlgorithm();
            return true;
        }
        // 计算最近 windowSize 局的均值
        float sum = 0f;
        int startIdx = _ppoMapRewardHistory.Count - windowSize;
        for (int i = startIdx; i < _ppoMapRewardHistory.Count; i++)
            sum += _ppoMapRewardHistory[i];
        float mean = sum / windowSize;
        // 计算标准差
        float sumSqDiff = 0f;
        for (int i = startIdx; i < _ppoMapRewardHistory.Count; i++)
        {
            float diff = _ppoMapRewardHistory[i] - mean;
            sumSqDiff += diff * diff;
        }
        float stdDev = Mathf.Sqrt(sumSqDiff / windowSize);
        // 计算变异系数 (Coefficient of Variation = 标准差 / 均值)
        float cv = (Mathf.Abs(mean) > 0.01f) ? (stdDev / Mathf.Abs(mean)) : 999f;
        // 收敛判定：奖励足够高 且 波动足够小
        if (mean >= minMeanReward && cv < maxCV)
        {
            Debug.Log($"<color=green>[Auto Train] PPO 奖励收敛判定通过！最近 {windowSize} 局：均值={mean:F1}, 标准差={stdDev:F1}, 变异系数={cv:F3} (< {maxCV}) → 切换下一地图</color>");
            episodeEpsilonHistory.Clear();
            SwitchToNextMapOrAlgorithm();
            return true;
        }
        // 每 10 局输出一次监控进度日志，方便观察训练趋势
        if (_ppoMapRewardHistory.Count % 10 == 0)
        {
            string status = mean < minMeanReward ? $"均值不足(需>{minMeanReward:F1})" : $"波动过大(CV={cv:F3}>{maxCV})";
            Debug.Log($"<color=cyan>[Auto Train] PPO 收敛监控 (第{_ppoMapRewardHistory.Count}局)：近{windowSize}局均值={mean:F1}, 标准差={stdDev:F1}, CV={cv:F3} | 未收敛原因: {status}</color>");
        }
        return false;
    }
    private void SwitchToNextMapOrAlgorithm()
    {
        _ppoMapRewardHistory.Clear(); // 切图/切算法时清空 PPO 奖励收敛记录
        MazeGenerator generator = FindAnyObjectByType<MazeGenerator>();
        currentSeedIndex++;
        int maxMapCount = generator != null ? generator.experimentSeeds.Length : 5;
        if (currentSeedIndex >= maxMapCount)
        {
            // 所有图都切完了，换算法！
            currentSeedIndex = 0;
            int nextAlgo = selectAlgorithm + 1;
            // 如果连PPO都跑完了，挂机结束
            if (nextAlgo > 2)
            {
                Debug.Log("<color=red>★★ [Auto Train] 所有算法的全部地图均已完美收敛！全线挂机大捷！★★</color>");
                runMode = 0;
                StopTrain();
                return;
            }
            Debug.Log($"<color=yellow>[Auto Train] 切换至下一个算法并从头开始地图循环: {nextAlgo}</color>");
            StopPythonProcess(); // 在切算法前必须先安全停机，不要让新选算法的初始指令发进空气里
            SwitchAlgorithm(nextAlgo);
            // 在 Unity 这边重新启动！
            StartCoroutine(DelayedRestartPythonRoutine());
        }
        else
        {
            Debug.Log($"<color=yellow>[Auto Train] 切换至当前算法的下一个地图索引: {currentSeedIndex}</color>");
            StartCoroutine(DelayedSwapMapRoutine());
        }
    }
    private IEnumerator DelayedSwapMapRoutine()
    {
        // 防止和上次遗留的数据管道造成串扰或争抢
        yield return new WaitForSeconds(0.2f);
        MazeGenerator generator = FindAnyObjectByType<MazeGenerator>();
        if (generator != null)
        {
            generator.CreateMaze();
        }
        yield return new WaitForSeconds(0.2f);
        StartCoroutine(StartTrainRoutine());
    }
    private IEnumerator DelayedRestartPythonRoutine()
    {
        // 留时间释放文件占用和内存映射
        yield return new WaitForSeconds(1.0f);
        StartPythonProcess();
        InitSharedMemory();
        StartTrain();
    }
    #endregion
    #endregion
}