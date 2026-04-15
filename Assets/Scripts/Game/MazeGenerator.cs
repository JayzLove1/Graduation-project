using System.Collections.Generic;
using UnityEngine;
// 迷宫生成器 (MazeGenerator)
// 使用深度优先搜索（DFS）算法来随机生成迷宫。
// 本脚本主要满足以下需求：
// 1. 训练模式：固定随机种子，保证每次生成的迷宫长得一样，方便AI学习认路。
// 2. 演示模式：读取训练时的那个种子，一模一样地还原迷宫，看AI是怎么走的。
// 3. 通信功能：把生成的迷宫变成一连串的0和1（1是墙，0是路）发给后端的Python程序。
public class MazeGenerator : MonoBehaviour
{
    [Header("迷宫方块设置")]
    public GameObject wall;         // 表示墙壁的方块
    public GameObject floor;        // 表示地板的方块
    public GameObject startPoint;   // AI出生的起点
    public GameObject endPoint;     // 目标终点
    private int[,] maze;            // 一个大表格，用来登记迷宫每一个格子的状态（1是墙，0是路）
    private int w, h;               // 迷宫最终的宽度和高度（生成算法要求长宽必须是奇数）
    // 把宽和高的数值开放出去给别人读取
    public int Width => w;
    public int Height => h;
    [Header("固定种子实验设置")]
    [Tooltip("开启后，训练模式将从下方的五个种子中随机选取，而不是完全随机生成")]
    public bool useFixedSeeds = true;
    public int[] experimentSeeds = { 123456, 234567, 345678, 456789, 567890 };
    private void Start() => CreateMaze();
    // 把输入的偶数变成奇数（加1）
    // 因为这套算法如果长宽不是奇数，墙壁就没法正常地隔开道路。
    private int Odd(int v) => v % 2 == 0 ? v + 1 : v;
    // 创建迷宫的核心函数
    // 主要是根据当前选择的模式，看看是该随机生新图，还是该读取旧图。
    public void CreateMaze()
    {
        // === 设置随机种子，保证每次的随机情况都可以控制 ===
        if (GameManager.instance != null && (GameManager.instance.runMode == 1 || GameManager.instance.runMode == 3))
        {
            // [训练模式 / 自动挂机模式]
            int newSeed;
            if (useFixedSeeds)
            {
                // 从五个实验种子中选，如果是自动挂机模式，严格按顺序选！
                int targetIndex = 0;
                if (GameManager.instance.runMode == 3)
                {
                    targetIndex = GameManager.instance.currentSeedIndex;
                    if (targetIndex >= experimentSeeds.Length) targetIndex = 0;
                }
                else
                {
                    targetIndex = new System.Random(System.Environment.TickCount).Next(0, experimentSeeds.Length);
                }
                newSeed = experimentSeeds[targetIndex];
                Debug.Log($"<color=cyan>[MazeGenerator] 实验关卡模式：加载第 {targetIndex + 1} 个固定种子: {newSeed}</color>");
            }
            else
            {
                // 彻底随机生成
                newSeed = System.Math.Abs(System.Guid.NewGuid().GetHashCode() % 900000 + 100000);
            }
            // 为了防止不同算法、不同难度的迷宫混在一起，存的时候要带上算法+尺寸
            string seedKey = $"MazeRandomSeed_{GameManager.instance.selectAlgorithm}_{GameData.mazeWidth}";
            // 把当前定好的种子保存到电脑里
            PlayerPrefs.SetInt(seedKey, newSeed);
            PlayerPrefs.Save();
            // 给随机函数定好这个种子，接下来的“随机”就全按这个剧本走了
            Random.InitState(newSeed);
            Debug.Log($"[MazeGenerator] 训练准备就绪，当前种子: {newSeed}，尺寸: {GameData.mazeWidth}x{GameData.mazeHeight}");
        }
        else if (GameManager.instance != null && GameManager.instance.runMode == 2)
        {
            // [演示模式]
            // key 包含算法+尺寸，确保加载的是当前难度的训练种子，不会串到其他难度
            string seedKey = $"MazeRandomSeed_{GameManager.instance.selectAlgorithm}_{GameData.mazeWidth}";
            // 把之前训练时存下来的种子翻出来
            int savedSeed = PlayerPrefs.GetInt(seedKey, 0);
            // 用旧种子初始化，确保生成的地图跟之前的一模一样
            Random.InitState(savedSeed);
            Debug.Log($"[MazeGenerator] 演示模式启动，种子: {savedSeed}，尺寸: {GameData.mazeWidth}x{GameData.mazeHeight}（Key={seedKey}）");
        }
        else
        {
            // [玩家手动测试模式] 每次就单纯看当前时间随缘生成即可
            Random.InitState(System.DateTime.Now.Millisecond);
        }
        w = Odd(GameData.mazeWidth);
        h = Odd(GameData.mazeHeight);
        // 为了防止下一把和上一把的地图叠在一起，先把场上以前的方块全删了
        for (int i = transform.childCount - 1; i >= 0; i--)
            DestroyImmediate(transform.GetChild(i).gameObject);
        // 默认让整个地图全是死路（填满 1）
        maze = new int[w, h];
        for (int x = 0; x < w; x++)
            for (int y = 0; y < h; y++)
                maze[x, y] = 1;
        // DFS 算法需要用到一个“栈”（先进后出）来记录挖掘路线
        Stack<Vector2> stack = new Stack<Vector2>();
        Vector2 current = new Vector2(1, 1);
        maze[1, 1] = 0; // 起点必须是空地才能出发
        stack.Push(current);
        // 代表往上、下、左、右四个方向的箭头
        Vector2[] dirs = { Vector2.right, Vector2.left, Vector2.up, Vector2.down };
        // 像挖地道一样把一条条可以走的路给“凿”出来
        while (stack.Count > 0)
        {
            current = stack.Pop();
            List<Vector2> neighbors = new List<Vector2>();
            foreach (var d in dirs)
            {
                // 一次跨两步，这样挖路的时候周围就会留下一堵墙
                int nx = (int)(current.x + d.x * 2);
                int ny = (int)(current.y + d.y * 2);
                // 看看目标远方这一格没超出边界，并且还是一堵没开垦的墙，就当成了备选项加入列表
                if (nx > 0 && nx < w - 1 && ny > 0 && ny < h - 1 && maze[nx, ny] == 1)
                    neighbors.Add(new Vector2(nx, ny));
            }
            if (neighbors.Count > 0)
            {
                stack.Push(current); // 这边还有得挖，先把当前的保留存好
                // 随便选一个接下来的挖掘方向
                Vector2 next = neighbors[Random.Range(0, neighbors.Count)];
                // 把跳跃的那一格，和中间相隔的那一格全部打通变成路 (0)
                maze[(int)next.x, (int)next.y] = 0;
                maze[(int)(current.x + next.x) / 2, (int)(current.y + next.y) / 2] = 0;
                stack.Push(next);
            }
        }
        // 到了这一步，二维表格里已经画好地图了，接下来就是在场景里真真切切地摆放方块
        for (int x = 0; x < w; x++)
            for (int y = 0; y < h; y++)
                Instantiate(
                    maze[x, y] == 1 ? wall : floor, // 看到 1 就放墙壁，看到 0 就放地板
                    new Vector2(x, y),
                    Quaternion.identity,
                    transform
                );
        // 在特定地点把起点和终点的指示牌挂上去
        Instantiate(startPoint, new Vector2(1, 1), Quaternion.identity, transform);
        Instantiate(endPoint, new Vector2(w - 2, h - 2), Quaternion.identity, transform);
        // 告知 GameManager 终点坐标，用于计算奖励
        if (GameManager.instance != null)
        {
            GameManager.instance.goalPos = new Vector2(w - 2, h - 2);
        }
        // 修好迷宫后，强制让扮演 AI 的小球回到（1, 1）这个位置重新开始
        var player = FindAnyObjectByType<PlayerController>();
        if (player != null)
        {
            player.ResetPlayer();
        }
    }
    // 这个函数的任务就是把地图表格拍平成一根长长的面条，方便发送。
    // 因为 Python 端的 AI 看不懂画面，只能接收比如 "1,1,1,1,0,0,..." 这样的文字。
    public string GetGridString()
    {
        if (maze == null) return "";
        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                // 把格子里的数字拼起来，加上逗号
                sb.Append(maze[x, y]).Append(",");
            }
        }
        // 把最后不小心多加的那个逗号删掉
        if (sb.Length > 0) sb.Length--;
        return sb.ToString();
    }
}