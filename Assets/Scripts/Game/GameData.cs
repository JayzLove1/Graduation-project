// 存全局数据的类
// 主要是让各个地方不同的脚本都能拿到一些大家都需要的基础配置。
// 在这个项目里，主要是用来记住我们在主菜单界面设置的迷宫的宽度和高度。
public static class GameData
{
    // 默认迷宫的宽度，横着有21个格子
    public static int mazeWidth = 21;
    // 默认迷宫的高度，竖着有21个格子
    public static int mazeHeight = 21;
}
