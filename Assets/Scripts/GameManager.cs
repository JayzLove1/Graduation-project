// Unity 与 Python RL 后端的核心调度桥梁
// 负责 IPC 通信（Windows 共享内存）、奖励重塑、训练调度和断点续训
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Text;
using UnityEngine;
using Debug = UnityEngine.Debug;
public class GameManager : MonoBehaviour
{
    public static GameManager instance;
    #region ========== 配置与状态 ==========
    [Header("AI 角色与环境")]
    public PlayerController aiPlayer;
    [Header("奖励配置")]
    public RewardConfig rewardConfig;
    [Header("训练图表")]
    public MazeAI.UI.TrainingChartUI trainingChart;
    [Header("运行模式")]
    public int runMode;
    public int selectAlgorithm;
    public int currentSeedIndex;
    [HideInInspector] public int episodeCount;
    [HideInInspector] public float episodeTotalReward;
    [HideInInspector] public int episodeStepCount;
    [HideInInspector] public int totalSuccessCount;
    [HideInInspector] public int hitCount;
    private float pendingReward;
    private bool isResettingEpisode;
    // 防重入锁：OnSceneLoaded 和 GameInit.Start 都会调用 InitAfterSceneLoaded，
    // 第二次进入会重置共享内存并启动重复握手协程
    private bool _isTrainInitializing;
    private float _lastDistanceToGoal = -1;
    public Vector2 goalPos;
    private readonly HashSet<Vector2Int> _visitedCells = new();
    private readonly Dictionary<Vector2Int, int> _cellVisitCount = new();
    // 步数上限由 SyncGridData 按地图大小计算，0=未初始化（不启用）
    private int _maxStepsPerEpisode;
    [HideInInspector] public List<float> rewardHistory = new();
    [HideInInspector] public List<int> stepHistory = new();
    // epsilonHistory：QL/DQN 记录 epsilon，PPO 记录 entropy_coeff，供图表绘制
    [HideInInspector] public List<float> epsilonHistory = new();
    private List<float> _ppoMapRewardHistory = new();
    // 与 _ppoMapRewardHistory 一一对应，true = ReachEnd 真通关
    // 用真实信号而非 reward 阈值，避免 BFS shaping 累积后未通关局也能超阈值
    private List<bool> _ppoMapSuccessHistory = new();
    private bool _lastEpisodeWasSuccess;
    public TrainParam trainParam;
    private MemoryMappedFile mmf;
    private MemoryMappedViewAccessor view;
    private Process pythonProcess;
    private bool isPythonProcessRunning;
    private bool hasReceivedReady;
    private const int DATA_SIZE = 4096;
    public enum RewardType { NormalMove, HitWall, ReachEnd }
    [Serializable]
    public class RewardConfig
    {
        public float normalMoveReward = -0.05f;
        public float hitWallReward = -0.5f;
        public float reachEndReward = 100f;
    }
    [Serializable]
    public class TrainParam
    {
        public float lr = 0.1f;
        public float gamma = 0.99f;
        public float epsilon = 1.0f;
        public float epsilonDecay = 0.995f;
        public int batchSize = 32;
        public float algoLr = 0.001f;
    }
    #endregion
    #region ========== 生命周期 ==========
    private void Awake()
    {
        if (instance == null) { instance = this; DontDestroyOnLoad(gameObject); }
        else Destroy(gameObject);
    }
    private void Start()
    {
        runMode = 0;
        UnityEngine.SceneManagement.SceneManager.sceneLoaded += OnSceneLoaded;
        MigrateOldSeedKeys();
    }
    private void OnSceneLoaded(UnityEngine.SceneManagement.Scene scene, UnityEngine.SceneManagement.LoadSceneMode mode)
    {
        if (scene.name != "GameScene") return;
        var playerObj = GameObject.FindGameObjectWithTag("Player");
        if (playerObj != null) aiPlayer = playerObj.GetComponent<PlayerController>();
        InitAfterSceneLoaded();
    }
    private void OnDestroy()
    {
        StopPythonProcess();
        CloseSharedMemory();
        PlayerPrefs.Save(); // 兜底刷盘：防止最近几局数据未达刷盘阈值而丢失
    }
    #endregion
    #region ========== 初始化 ==========
    public void InitAfterSceneLoaded()
    {
        if (aiPlayer == null || _isTrainInitializing) return;
        aiPlayer.canMove = true;
        aiPlayer.ResetPlayer();
        hitCount = 0;
        if (runMode == 0) { StopPythonProcess(); return; }
        _isTrainInitializing = true;
        StartPythonProcess();
        InitSharedMemory();
        // sceneLoaded 在所有对象 Start() 之前触发，此时 MazeGenerator.maze 还是 null，
        // 延迟一帧让 CreateMaze() 完成后再发 GRID
        StartCoroutine(DelayedStartTrain());
    }
    private IEnumerator DelayedStartTrain()
    {
        yield return null;
        StartTrain();
        _isTrainInitializing = false;
    }
    public void SwitchRunMode(int mode) => runMode = mode;
    public void SwitchAlgorithm(int algo)
    {
        selectAlgorithm = algo;
        UpdateRealtimeParams();
        LoadHistoryData();
    }
    public void UpdateRealtimeParams()
    {
        if (view == null) return;
        SendDataToPython($"ALGO:{selectAlgorithm}");
        // epsilon_decay 用 "_" 占位：Python 端按迷宫大小自适应，避免 UI 默认值覆盖
        SendDataToPython($"PARAM:{trainParam.lr},{trainParam.gamma},{trainParam.epsilon},_,{trainParam.batchSize},{trainParam.algoLr}");
        Debug.Log($"<color=yellow>[IPC] 超参数同步: {GetAlgorithmName()} | lr={trainParam.lr:F4} | γ={trainParam.gamma:F2} | ε={trainParam.epsilon:F4} | batch={trainParam.batchSize} | algoLr={trainParam.algoLr:F4}</color>");
    }
    public string GetAlgorithmName() => selectAlgorithm switch
    {
        0 => "Q-Learning",
        1 => "DQN",
        2 => "PPO",
        _ => "AI_Backend"
    };
    #endregion
    #region ========== 奖励重塑 ==========
    public void TriggerReward(RewardType type)
    {
        float reward = type switch
        {
            // PPO 步惩罚 -0.15（DQN/QL -0.05）：让路径长度影响总分，
            // 防止 +500 通关奖励过大导致智能体通关后不再优化路径
            RewardType.NormalMove => (selectAlgorithm == 2) ? -0.15f : rewardConfig.normalMoveReward,
            RewardType.HitWall => (selectAlgorithm == 2) ? -0.3f : rewardConfig.hitWallReward,
            // PPO 通关奖励 +500：on-policy batch 中约 98% 为超时样本，
            // 通关轨迹 advantage 必须远大于超时轨迹才不会被淹没
            RewardType.ReachEnd => (selectAlgorithm == 2) ? 500f : rewardConfig.reachEndReward,
            _ => 0,
        };
        if (type == RewardType.NormalMove && aiPlayer != null)
        {
            Vector2 curPos = aiPlayer.transform.position;
            float curDist = Vector2.Distance(curPos, goalPos);
            if (_lastDistanceToGoal > 0)
            {
                float diff = _lastDistanceToGoal - curDist;
                // PPO 的距离 shaping 由 Python 端 BFS 替代：欧氏距离在迷宫拐角处
                // 会给唯一正确方向负奖励，导致 PPO 学会"避开出口"
                float multiplier = (selectAlgorithm == 2) ? 0f : 1f;
                reward += diff * multiplier;
            }
            _lastDistanceToGoal = curDist;
            var gridPos = new Vector2Int(Mathf.RoundToInt(curPos.x), Mathf.RoundToInt(curPos.y));
            if (!_visitedCells.Contains(gridPos))
            {
                // PPO 探索奖励 0.3（DQN/QL 0.2）
                float bonus = (selectAlgorithm == 2) ? 0.3f : 0.2f;
                reward += bonus;
                _visitedCells.Add(gridPos);
                _cellVisitCount[gridPos] = 1;
            }
            else
            {
                _cellVisitCount.TryGetValue(gridPos, out int cnt);
                _cellVisitCount[gridPos] = ++cnt;
                // PPO 重复访问惩罚 -0.03
                // 叠加超时 -500 后 critic 对所有状态估值极低，advantage 信噪比差，
                // BFS 方向梯度被淹没。-0.03 仍能打破短期振荡但不压制方向信号
                float perVisit = (selectAlgorithm == 2) ? -0.03f : -0.05f;
                reward += perVisit * Mathf.Min(cnt - 1, 5); // 封顶 5 次防梯度爆炸
            }
        }
        pendingReward += reward;
        episodeTotalReward += reward;
        if (type == RewardType.HitWall) { hitCount++; return; }
        if (type == RewardType.ReachEnd)
        {
            if (isResettingEpisode) return;
            isResettingEpisode = true;
            _lastEpisodeWasSuccess = true;
            totalSuccessCount++;
            Debug.Log($"<color=green>★ [通关] {GetAlgorithmName()} | 局:{episodeCount + 1} | 步:{aiPlayer.stepCount} | 撞:{hitCount} | 分:{episodeTotalReward:F1}</color>");
            SendDataToPython($"REWARD:{reward}|{aiPlayer.stepCount}|1|2");
            StartCoroutine(ResetEpisodeRoutine());
        }
    }
    #endregion
    #region ========== IPC 共享内存 ==========
    private void InitSharedMemory()
    {
        try
        {
            mmf = MemoryMappedFile.CreateOrOpen("MazeRLSharedMemory", DATA_SIZE);
            view = mmf.CreateViewAccessor();
            hasReceivedReady = false;
        }
        catch (Exception e) { Debug.LogError($"[IPC] 内存映射分配失败: {e.Message}"); }
    }
    private void CloseSharedMemory()
    {
        view?.Dispose(); mmf?.Dispose();
        view = null; mmf = null;
    }
    public void SendDataToPython(string data)
    {
        if (view == null) return;
        byte[] bytes = Encoding.UTF8.GetBytes(data.PadRight(DATA_SIZE, '\0'));
        view.WriteArray(0, bytes, 0, DATA_SIZE);
    }
    public string ReadDataFromPython()
    {
        if (view == null) return "";
        byte[] bytes = new byte[DATA_SIZE];
        view.ReadArray(0, bytes, 0, DATA_SIZE);
        return Encoding.UTF8.GetString(bytes).TrimEnd('\0').Trim();
    }
    #endregion
    #region ========== Python 进程管控 ==========
    private void StartPythonProcess()
    {
        if (isPythonProcessRunning) return;
        try
        {
            string script = selectAlgorithm switch
            {
                0 => "rl_maze_ai_qlearning.py",
                1 => "rl_maze_ai_dqn.py",
                _ => "rl_maze_ai_ppo.py"
            };
            var psi = new ProcessStartInfo
            {
                FileName = "python",
                Arguments = $"-u \"{Path.Combine(Application.dataPath, "Python", script)}\"",
                UseShellExecute = true,
                CreateNoWindow = false
            };
            pythonProcess = Process.Start(psi);
            isPythonProcessRunning = true;
            Debug.Log($"[Backend] Python 启动 (PID:{pythonProcess.Id}, Script:{script})");
        }
        catch (Exception e) { Debug.LogError($"[Backend] 启动失败，请检查 Python 路径: {e.Message}"); }
    }
    public void StopPythonProcess()
    {
        if (pythonProcess != null && !pythonProcess.HasExited)
        {
            SendDataToPython("COMMAND:QUIT");
            System.Threading.Thread.Sleep(200);
            try { pythonProcess.Kill(); } catch { }
            pythonProcess.Dispose();
        }
        pythonProcess = null;
        isPythonProcessRunning = false;
        hasReceivedReady = false;
    }
    #endregion
    #region ========== 训练调度 ==========
    public void StartTrain()
    {
        if (runMode == 0 || aiPlayer == null) return;
        StartCoroutine(StartTrainRoutine());
    }
    private IEnumerator StartTrainRoutine()
    {
        // 阶段 1：等待 Python 后端发送 READY（超时 30s）
        float timeout = 0;
        while (!hasReceivedReady && timeout < 30f)
        {
            if (ReadDataFromPython() == "READY") { hasReceivedReady = true; break; }
            yield return new WaitForSeconds(0.2f);
            timeout += 0.2f;
        }
        if (!hasReceivedReady)
        {
            Debug.LogError("[IPC] 握手失败：Python 未响应 READY"); yield break;
        }
        // 阶段 2：发送 GRID 并等待 GRID_OK
        // DQN/PPO 必须收到 GRID_OK，否则网络未初始化，全程返回 ACTION:0（静默跑空模型）
        // Q-Learning 不回复 GRID_OK，超时直接继续
        bool gridConfirmed = false;
        int maxAttempts = (selectAlgorithm == 0) ? 1 : 3;
        for (int attempt = 0; attempt < maxAttempts && !gridConfirmed; attempt++)
        {
            if (attempt > 0)
                Debug.LogWarning($"[IPC] 未收到 GRID_OK (尝试 {attempt}/{maxAttempts})，重发 GRID...");
            SyncGridData();
            float elapsed = 0f;
            while (elapsed < 5f)
            {
                if (ReadDataFromPython() == "GRID_OK") { gridConfirmed = true; break; }
                yield return new WaitForSeconds(0.1f);
                elapsed += 0.1f;
            }
        }
        if (!gridConfirmed && selectAlgorithm != 0)
        {
            Debug.LogError($"[IPC FATAL] {GetAlgorithmName()} 网络初始化失败：{maxAttempts} 次 GRID 均无 GRID_OK，训练中止");
            StopPythonProcess(); yield break;
        }
        // 阶段 3：推送超参数 → 发送训练指令 → 启动动作循环
        UpdateRealtimeParams();
        ResumeEpisodeCount();
        SendDataToPython(runMode == 2 ? "COMMAND:DEMO" : "COMMAND:START");
        aiPlayer.canMove = true;
        yield return new WaitForSeconds(0.3f);
        OnAIMoveComplete(); // 触发首帧 STATE，之后由 PlayerController 回调驱动闭环
    }
    private void SyncGridData()
    {
        var gen = FindAnyObjectByType<MazeGenerator>();
        if (gen == null) { Debug.LogError("[IPC] SyncGridData 失败：找不到 MazeGenerator"); return; }
        if (gen.maze == null) { Debug.LogError("[IPC] SyncGridData 失败：maze 为 null，DelayedStartTrain 未等待一帧"); return; }
        int w = gen.Width, h = gen.Height;
        // 步数上限 6×w×h（15×15=1350）：随机游走混合时间约 O(n²)≈11000，
        // 675 步时初始随机策略几乎走不到终点，1350 步显著提升首次通关概率
        _maxStepsPerEpisode = 6 * w * h;
        var sb = new StringBuilder($"GRID:{w}|{h}|{gen.currentSeed}|");
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                sb.Append(gen.maze[x, y] == 1 ? "1" : "0");
                if (x < w - 1 || y < h - 1) sb.Append(',');
            }
        SendDataToPython(sb.ToString());
        Debug.Log($"[IPC] GRID 发送：{w}×{h}, seed={gen.currentSeed}");
    }
    public void OnAIMoveComplete()
    {
        if (!isPythonProcessRunning || runMode == 0) return;
        // 步数超限：强制以失败结束 episode，阻止错误轨迹持续污染 on-policy buffer
        if (_maxStepsPerEpisode > 0 && aiPlayer != null
            && aiPlayer.stepCount >= _maxStepsPerEpisode && !isResettingEpisode)
        {
            TriggerTimeoutFailure(); return;
        }
        StartCoroutine(RequestActionCoroutine());
    }
    private void TriggerTimeoutFailure()
    {
        if (isResettingEpisode) return;
        isResettingEpisode = true;
        _lastEpisodeWasSuccess = false;
        // 超时惩罚 -500：与通关 +500 对称，让"超时=坏"产生等量级梯度
        float penalty = (selectAlgorithm == 2) ? -500f : -10f;
        pendingReward += penalty;
        episodeTotalReward += penalty;
        // rtype=3：Python 端据此写终止标记并临时回升 entropy_coeff
        SendDataToPython($"REWARD:{penalty}|{aiPlayer.stepCount}|0|3");
        Debug.Log($"<color=orange>⚠ [超时] {GetAlgorithmName()} | 局:{episodeCount + 1} | 步:{aiPlayer.stepCount} | 分:{episodeTotalReward:F1}</color>");
        StartCoroutine(ResetEpisodeRoutine());
    }
    private int _consecutiveActionTimeouts;
    private const int MAX_ACTION_TIMEOUTS = 3;
    private IEnumerator RequestActionCoroutine()
    {
        yield return null;
        yield return new WaitForSeconds((runMode == 1 || runMode == 3) ? 0.01f : 0.1f);
        SendDataToPython($"STATE:{aiPlayer.transform.position.x}|{aiPlayer.transform.position.y}|{aiPlayer.canMove}|{aiPlayer.stepCount}|{pendingReward}");
        pendingReward = 0;
        string actionData = "";
        float wait = 0;
        while (wait < 5f)
        {
            actionData = ReadDataFromPython();
            if (actionData.StartsWith("ACTION:")) break;
            yield return null;
            wait += Time.deltaTime;
        }
        if (actionData.StartsWith("ACTION:"))
        {
            _consecutiveActionTimeouts = 0;
            string[] parts = actionData.Split(':')[1].Split('|');
            int dir = int.Parse(parts[0]);
            if (parts.Length > 1 && float.TryParse(parts[1], out float e)) trainParam.epsilon = e;
            SendDataToPython("WAITING");
            ExecuteAIMove(dir);
        }
        else
        {
            // ACTION 超时：连续 N 次失败后触发 episode 终止，防止动作链永久冻结
            _consecutiveActionTimeouts++;
            Debug.LogWarning($"[IPC] ACTION 超时 ({_consecutiveActionTimeouts}/{MAX_ACTION_TIMEOUTS})");
            if (_consecutiveActionTimeouts >= MAX_ACTION_TIMEOUTS)
            {
                Debug.LogError("[IPC] 连续 ACTION 超时上限，触发失败终止");
                _consecutiveActionTimeouts = 0;
                if (!isResettingEpisode) TriggerTimeoutFailure();
            }
            else
            {
                yield return new WaitForSeconds(0.2f);
                OnAIMoveComplete();
            }
        }
    }
    public void ExecuteAIMove(int dir)
    {
        if (aiPlayer == null || !aiPlayer.canMove) return;
        Vector2 v = dir switch { 0 => Vector2.up, 1 => Vector2.down, 2 => Vector2.left, 3 => Vector2.right, _ => Vector2.zero };
        aiPlayer.TryMove(v);
    }
    private IEnumerator ResetEpisodeRoutine()
    {
        // yield 一帧再发 RESET：让 Python 有至少 3 次（×5ms）轮询窗口稳定读到 REWARD。
        // 若同帧发 RESET，REWARD 字符串会在微秒内被覆盖，终端 +500 永远进不了 buffer。
        yield return null;
        SendDataToPython($"COMMAND:RESET|{hitCount}");
        rewardHistory.Add(episodeTotalReward);
        stepHistory.Add(aiPlayer.stepCount);
        epsilonHistory.Add(trainParam.epsilon);
        if (selectAlgorithm == 2)
        {
            _ppoMapRewardHistory.Add(episodeTotalReward);
            _ppoMapSuccessHistory.Add(_lastEpisodeWasSuccess);
        }
        _lastEpisodeWasSuccess = false;
        if (trainingChart != null) trainingChart.AddDataPoint(episodeTotalReward, aiPlayer.stepCount);
        SaveHistoryData();
        yield return new WaitForSeconds(0.2f);
        bool switchingMap = CheckAutoTrainConvergence();
        episodeCount++;
        hitCount = 0;
        episodeTotalReward = 0;
        // pendingReward 清零：终局的 ±500 若不在此清零，会随下一局首帧 STATE 发给 Python，
        // 导致新局 episode_reward 凭空含上一局终端奖励
        pendingReward = 0;
        _lastDistanceToGoal = -1;
        _visitedCells.Clear();
        _cellVisitCount.Clear();
        aiPlayer.ResetPlayer();
        isResettingEpisode = false;
        if (!switchingMap && runMode != 0)
        {
            yield return new WaitForSeconds(0.1f);
            OnAIMoveComplete();
        }
    }
    #endregion
    #region ========== 收敛判定与批量调度 ==========
    private float _lastEpsilon = -1f;
    private int _epsilonStableCount;
    private const int EPSILON_STABLE_THRESHOLD = 5;
    private bool CheckAutoTrainConvergence()
    {
        if (runMode != 3) return false;
        return selectAlgorithm <= 1 ? CheckEpsilonConvergence() : CheckPPORewardConvergence();
    }
    // QL/DQN：epsilon 连续 N 局不再衰减（已到 min_epsilon）时视为收敛
    private bool CheckEpsilonConvergence()
    {
        float eps = trainParam.epsilon;
        _epsilonStableCount = (_lastEpsilon >= 0 && Mathf.Approximately(eps, _lastEpsilon))
            ? _epsilonStableCount + 1 : 0;
        _lastEpsilon = eps;
        if (_epsilonStableCount < EPSILON_STABLE_THRESHOLD) return false;
        Debug.Log($"<color=green>[Batch] {GetAlgorithmName()} epsilon 稳定 ({eps:F4})，连续 {_epsilonStableCount} 局不变，切换地图</color>");
        _epsilonStableCount = 0; _lastEpsilon = -1f;
        SwitchToNextMapOrAlgorithm();
        return true;
    }
    // PPO 三维收敛判定：
    //   主窗口 200 局：通关率 ≥80% 且均值 ≥100
    //   抗回退窗口 30 局：通关率 ≥70%（防策略坍塌后历史均值蒙混过关）
    //   兜底超时：400 局强制切换
    private bool CheckPPORewardConvergence()
    {
        const int window = 200, recentWindow = 30;
        int total = _ppoMapRewardHistory.Count;
        if (total < window) return false;
        float sum = 0; int successes = 0;
        for (int i = total - window; i < total; i++)
        {
            sum += _ppoMapRewardHistory[i];
            if (_ppoMapSuccessHistory[i]) successes++;
        }
        float mean = sum / window;
        float successRate = (float)successes / window;
        int recentOk = 0;
        for (int i = total - recentWindow; i < total; i++)
            if (_ppoMapSuccessHistory[i]) recentOk++;
        float recentRate = (float)recentOk / recentWindow;
        bool converged = successRate >= 0.8f && mean >= 100f && recentRate >= 0.7f;
        bool timeout = total > 400;
        if (!converged && !timeout) return false;
        string reason = converged
            ? $"收敛 (通关率:{successRate:P0}, 近30局:{recentRate:P0}, Mean:{mean:F1})"
            : $"超时强切 ({total}局, 通关率:{successRate:P0}, Mean:{mean:F1})";
        Debug.Log($"<color=green>[Batch] PPO {reason}，切换地图</color>");
        _ppoMapRewardHistory.Clear();
        _ppoMapSuccessHistory.Clear();
        SwitchToNextMapOrAlgorithm();
        return true;
    }
    private void SwitchToNextMapOrAlgorithm()
    {
        currentSeedIndex++;
        var mazeGenForSeeds = FindAnyObjectByType<MazeGenerator>();
        int maxSeeds = mazeGenForSeeds != null ? mazeGenForSeeds.experimentSeeds.Length : 5;
        if (currentSeedIndex < maxSeeds)
        {
            StartCoroutine(DelayedSwapMapRoutine()); return;
        }
        currentSeedIndex = 0;
        int nextAlgo = selectAlgorithm + 1;
        if (nextAlgo > 2)
        {
            Debug.Log("<color=red>★★ [BATCH COMPLETE] 所有实验序列完成 ★★</color>");
            runMode = 0; StopPythonProcess(); return;
        }
        StopPythonProcess();
        SwitchAlgorithm(nextAlgo);
        StartCoroutine(DelayedRestartPythonRoutine());
    }
    private IEnumerator DelayedSwapMapRoutine()
    {
        yield return new WaitForSeconds(0.2f);
        var swapGen = FindAnyObjectByType<MazeGenerator>();
        if (swapGen != null) swapGen.CreateMaze();
        yield return new WaitForSeconds(0.2f);
        ResetCountersForNewMap();
        StartTrain();
    }
    private IEnumerator DelayedRestartPythonRoutine()
    {
        yield return new WaitForSeconds(1.2f);
        // 必须重新生成迷宫：否则 MazeGenerator.currentSeed 仍是上个算法最后一张图的种子，
        // 新算法 Python 进程会找不到对应的 .pth 文件
        var restartGen = FindAnyObjectByType<MazeGenerator>();
        if (restartGen != null) restartGen.CreateMaze();
        yield return new WaitForSeconds(0.2f);
        StartPythonProcess();
        InitSharedMemory();
        ResetCountersForNewMap();
        StartTrain();
    }
    private void ResetCountersForNewMap()
    {
        episodeCount = 0; episodeTotalReward = 0f; hitCount = 0; totalSuccessCount = 0;
        _lastDistanceToGoal = -1;
        _lastEpsilon = -1f;
        _epsilonStableCount = 0;
        _visitedCells.Clear(); _cellVisitCount.Clear();
        _ppoMapRewardHistory.Clear(); _ppoMapSuccessHistory.Clear();
        rewardHistory.Clear(); stepHistory.Clear(); epsilonHistory.Clear();
    }
    #endregion
    #region ========== 数据持久化 ==========
    private void MigrateOldSeedKeys()
    {
        if (PlayerPrefs.GetInt("SeedKeyMigrated", 0) == 1) return;
        for (int algo = 0; algo <= 2; algo++)
        {
            string oldKey = "MazeRandomSeed_" + algo;
            if (!PlayerPrefs.HasKey(oldKey)) continue;
            PlayerPrefs.SetInt($"MazeRandomSeed_{algo}_{GameData.mazeWidth}", PlayerPrefs.GetInt(oldKey));
            PlayerPrefs.DeleteKey(oldKey);
        }
        PlayerPrefs.SetInt("SeedKeyMigrated", 1);
        PlayerPrefs.Save();
    }
    // 训练与演示历史分开存储，防止 demo 数据污染训练曲线
    private string GetHistoryKey()
    {
        string suffix = (runMode == 2) ? "_demo" : "";
        return $"HistoryData_{selectAlgorithm}_{GameData.mazeWidth}{suffix}";
    }
    private int _saveFlushTickCounter;
    private const int SAVE_FLUSH_INTERVAL = 10; // 每 10 局刷盘一次，降低注册表写入频率
    private void SaveHistoryData()
    {
        string key = GetHistoryKey();
        PlayerPrefs.SetString(key + "_Reward", string.Join(",", rewardHistory));
        PlayerPrefs.SetString(key + "_Step", string.Join(",", stepHistory));
        if (++_saveFlushTickCounter >= SAVE_FLUSH_INTERVAL)
        {
            PlayerPrefs.Save();
            _saveFlushTickCounter = 0;
        }
    }
    private void LoadHistoryData()
    {
        rewardHistory.Clear(); stepHistory.Clear(); epsilonHistory.Clear();
        string key = GetHistoryKey();
        if (PlayerPrefs.HasKey(key + "_Reward"))
            foreach (var r in PlayerPrefs.GetString(key + "_Reward").Split(','))
                if (float.TryParse(r, out float v)) rewardHistory.Add(v);
        if (PlayerPrefs.HasKey(key + "_Step"))
            foreach (var s in PlayerPrefs.GetString(key + "_Step").Split(','))
                if (int.TryParse(s, out int v)) stepHistory.Add(v);
    }
    private void ResumeEpisodeCount()
    {
        try
        {
            string seedKey = $"MazeRandomSeed_{selectAlgorithm}_{GameData.mazeWidth}";
            int seed = PlayerPrefs.GetInt(seedKey, 0);
            var resumeGen = FindAnyObjectByType<MazeGenerator>();
            int mazeW = resumeGen != null ? resumeGen.Width : GameData.mazeWidth;
            string demo = (runMode == 2) ? "_demo" : "";
            string csvName = selectAlgorithm switch
            {
                0 => $"training_log_QL{demo}_s{mazeW}_{seed}.csv",
                1 => $"dqn{demo}_log_s{mazeW}_{seed}.csv",
                2 => $"ppo{demo}_log_s{mazeW}_{seed}.csv",
                _ => ""
            };
            if (string.IsNullOrEmpty(csvName)) return;
            string csvPath = Path.Combine(Application.dataPath, "Python", "training_data~", csvName);
            if (!File.Exists(csvPath))
            {
                episodeCount = totalSuccessCount = 0;
                Debug.Log($"<color=cyan>[Resume] 新地图，从第 0 局开始 (seed={seed})</color>");
                return;
            }
            string[] lines = File.ReadAllLines(csvPath);
            int lastEp = 0, successes = 0;
            for (int i = 1; i < lines.Length; i++)
            {
                string[] cols = lines[i].Split(',');
                if (cols.Length >= 4 && int.TryParse(cols[0], out int ep))
                {
                    lastEp = ep;
                    if (float.TryParse(cols[3], out float r) && r > 50f) successes++;
                }
            }
            episodeCount = lastEp;
            totalSuccessCount = successes;
            Debug.Log($"<color=cyan>[Resume] 断点续训: Episode={episodeCount} | Success={totalSuccessCount} | seed={seed}</color>");
        }
        catch (Exception e)
        {
            // 不静默吞掉：CSV 损坏后用户需要知道续训失败，而不是局数默默归零
            Debug.LogWarning($"[Resume] CSV 解析失败，从第 0 局开始: {e.Message}");
            episodeCount = totalSuccessCount = 0;
        }
    }
    #endregion
}