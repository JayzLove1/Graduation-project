using UnityEngine;
using UnityEngine.UI;
// 开始界面的脚本 (StartPanelUI)
// 就是管理游戏一打开第一屏那两个按钮：一个是去设置难度，另一个是退出游戏。
public class StartPanelUI : MonoBehaviour
{
    [Header("界面上的按钮")]
    public Button btnStartGame; // 点击这个就去选难度和参数的界面
    public Button btnExit; // 点击这个直接退出整个游戏啦
    private void Start()
    {
        // 给这两个按钮装上功能，点一下就执行对应的代码
        btnStartGame.onClick.AddListener(OpenDifficulty);
        btnExit.onClick.AddListener(() => Application.Quit());
    }
    // 呼叫 UIManager 帮忙，把选择难度和算法设置的那个面板弹出来
    private void OpenDifficulty()
    {
        if (UIManager.instance != null)
        {
            UIManager.instance.ShowPanel("DifficultyPanel");
        }
    }
}
