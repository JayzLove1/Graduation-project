using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

// 通关结算面板：展示本局评估数据，提供重玩/返回菜单/退出操作
public class EndPanelUI : MonoBehaviour
{
    // ========== UI 组件 ==========
    [Header("统计显示")]
    public Text resultText;
    public Text aiTrainText;

    [Header("操作按钮")]
    public Button btnRestart;
    public Button btnBackMenu;
    public Button btnExit;

    // ========== 生命周期 ==========
    // 每次面板激活时刷新数据，避免显示上一局的残留结果
    private void OnEnable()
    {
        ShowFinalResult();
        BindButtonEvents();
    }

    // ========== 业务逻辑 ==========
    private void ShowFinalResult()
    {
        int   steps   = FindObjectOfType<PlayerController>().stepCount;
        int   hits    = GameManager.instance.hitCount;
        int   episode = GameManager.instance.episodeCount + 1;
        float reward  = GameManager.instance.episodeTotalReward;

        resultText.text  = $"★ 通关成功 ★\n局数：{episode}\n总步数：{steps}\n撞墙次数：{hits}\n得分：{reward:F1}";
        aiTrainText.text = $"算法：{GameManager.instance.GetAlgorithmName()}";
    }

    private void BindButtonEvents()
    {
        // RemoveAllListeners 防止 OnEnable 重复触发时监听器叠加
        btnRestart.onClick.RemoveAllListeners();
        btnRestart.onClick.AddListener(() =>
        {
            FindObjectOfType<PlayerController>()?.ResetPlayer();
            UIManager.instance?.ShowPanel("DifficultyPanel");
        });

        btnBackMenu.onClick.RemoveAllListeners();
        btnBackMenu.onClick.AddListener(() => SceneManager.LoadScene("StartScene"));

        btnExit.onClick.RemoveAllListeners();
        btnExit.onClick.AddListener(() => Application.Quit());
    }
}
