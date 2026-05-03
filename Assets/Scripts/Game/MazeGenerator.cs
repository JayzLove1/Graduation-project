using System.Collections.Generic;
using UnityEngine;

// DFS 回溯法迷宫生成器
// 支持固定种子实验（可复现）和随机种子自由模式
public class MazeGenerator : MonoBehaviour
{
    // ========== 预制体引用 ==========
    [Header("建筑组件")]
    public GameObject wall;
    public GameObject floor;
    public GameObject startPoint;
    public GameObject endPoint;

    // ========== 生成状态 ==========
    public int[,] maze;
    public int currentSeed;
    private int w, h;
    public int Width  => w;
    public int Height => h;

    // ========== 实验种子配置 ==========
    [Header("实验一致性")]
    [Tooltip("启用后训练模式循环使用固定种子，保证不同算法在同一拓扑下对比")]
    public bool useFixedSeeds = true;
    public int[] experimentSeeds = { 123456, 234567, 345678, 456789, 567890 };

    // ========== 生命周期 ==========
    private void Start() => CreateMaze();

    // 强制尺寸为奇数：DFS 雕墙算法要求偶数格为墙、奇数格为通道
    private int Odd(int v) => v % 2 == 0 ? v + 1 : v;

    // ========== 迷宫生成 ==========
    public void CreateMaze()
    {
        AssignSeed();
        BuildGrid();
        CarvePaths();
        InstantiateScene();
    }

    // 根据运行模式分配随机种子
    private void AssignSeed()
    {
        var gm = GameManager.instance;
        if (gm != null && (gm.runMode == 1 || gm.runMode == 3))
        {
            // 训练/自动模式：固定或随机种子，持久化供演示模式回溯
            int newSeed = useFixedSeeds
                ? experimentSeeds[(gm.runMode == 3 ? gm.currentSeedIndex : new System.Random().Next()) % experimentSeeds.Length]
                : System.Math.Abs(System.Guid.NewGuid().GetHashCode() % 900000 + 100000);

            string key = $"MazeRandomSeed_{gm.selectAlgorithm}_{GameData.mazeWidth}";
            PlayerPrefs.SetInt(key, newSeed);
            PlayerPrefs.Save();
            currentSeed = newSeed;
        }
        else if (gm != null && gm.runMode == 2)
        {
            // 演示模式：加载训练时保存的种子，确保与训练迷宫拓扑一致
            string key = $"MazeRandomSeed_{gm.selectAlgorithm}_{GameData.mazeWidth}";
            currentSeed = PlayerPrefs.GetInt(key, 0);
        }
        else
        {
            // 手动/自由模式：毫秒时间戳作为随机熵
            currentSeed = System.DateTime.Now.Millisecond;
        }
        Random.InitState(currentSeed);
    }

    // 初始化全填充为墙的网格
    private void BuildGrid()
    {
        w = Odd(GameData.mazeWidth);
        h = Odd(GameData.mazeHeight);
        for (int i = transform.childCount - 1; i >= 0; i--)
            DestroyImmediate(transform.GetChild(i).gameObject);
        maze = new int[w, h];
        for (int x = 0; x < w; x++)
            for (int y = 0; y < h; y++)
                maze[x, y] = 1;
    }

    // DFS 栈式雕墙：从 (1,1) 出发，每次随机选未访问邻格并打通中间墙
    private void CarvePaths()
    {
        var stack = new Stack<Vector2>();
        maze[1, 1] = 0;
        stack.Push(new Vector2(1, 1));
        Vector2[] dirs = { Vector2.right, Vector2.left, Vector2.up, Vector2.down };

        while (stack.Count > 0)
        {
            var cur = stack.Pop();
            var neighbors = new List<Vector2>();
            foreach (var d in dirs)
            {
                int nx = (int)(cur.x + d.x * 2);
                int ny = (int)(cur.y + d.y * 2);
                if (nx > 0 && nx < w - 1 && ny > 0 && ny < h - 1 && maze[nx, ny] == 1)
                    neighbors.Add(new Vector2(nx, ny));
            }
            if (neighbors.Count == 0) continue;
            stack.Push(cur);
            var next = neighbors[Random.Range(0, neighbors.Count)];
            maze[(int)next.x, (int)next.y] = 0;
            maze[(int)(cur.x + next.x) / 2, (int)(cur.y + next.y) / 2] = 0;
            stack.Push(next);
        }
    }

    // 将网格数组实例化为场景对象，并放置起点终点标识
    private void InstantiateScene()
    {
        for (int x = 0; x < w; x++)
            for (int y = 0; y < h; y++)
                UnityEngine.Object.Instantiate(maze[x, y] == 1 ? wall : floor, new Vector2(x, y), Quaternion.identity, transform);

        UnityEngine.Object.Instantiate(startPoint, new Vector2(1, 1),         Quaternion.identity, transform);
        UnityEngine.Object.Instantiate(endPoint,   new Vector2(w - 2, h - 2), Quaternion.identity, transform);

        if (GameManager.instance != null) GameManager.instance.goalPos = new Vector2(w - 2, h - 2);
        FindAnyObjectByType<PlayerController>()?.ResetPlayer();
    }

    // ========== 数据导出 ==========
    // 将迷宫拓扑序列化为逗号分隔字符串，通过 IPC 发送给 Python 端做状态编码
    public string GetGridString()
    {
        if (maze == null) return "";
        var sb = new System.Text.StringBuilder();
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                sb.Append(maze[x, y]).Append(',');
        if (sb.Length > 0) sb.Length--;
        return sb.ToString();
    }
}
