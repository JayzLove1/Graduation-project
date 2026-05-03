// 全局迷宫尺寸配置，供 Camera、MazeGenerator 及 Python 后端共享
// 必须为奇数：DFS 雕墙算法要求偶数格为墙、奇数格为通道
public static class GameData
{
    // ========== 迷宫尺寸 ==========
    public static int mazeWidth  = 21;
    public static int mazeHeight = 21;
}
