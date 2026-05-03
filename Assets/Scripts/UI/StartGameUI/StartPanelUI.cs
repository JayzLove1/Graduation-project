using UnityEngine;
using UnityEngine.UI;

// 主菜单入口：开始游戏跳转配置面板，退出直接关闭应用
public class StartPanelUI : MonoBehaviour
{
    // ========== UI 组件 ==========
    [Header("导航按钮")]
    public Button btnStartGame;
    public Button btnExit;

    // ========== 初始化 ==========
    private void Start()
    {
        btnStartGame.onClick.AddListener(() => UIManager.instance?.ShowPanel("DifficultyPanel"));
        btnExit.onClick.AddListener(() => Application.Quit());
    }
}
