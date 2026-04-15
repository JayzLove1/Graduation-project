using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
// 游戏结束总结界面 (EndPanelUI)
// 只要玩家或者AI走到终点，就会弹出这个框。
// 会告诉你走了多少步，撞了几次墙，得多少分数，并且问你要不要重新来一把或者退出。
public class EndPanelUI : MonoBehaviour
{
    [Header("显示分数的文字区域")]
    public Text resultText; // 放通关步数和分数的文字框
    public Text aiTrainText; // 显示这是哪个AI的文字框
    [Header("按钮配置")]
    public Button btnRestart;   // 重新来一局（回选择难度界面）
    public Button btnBackMenu;  // 返回最外面的主菜单
    public Button btnExit;      // 彻底退出游戏
    private void OnEnable()
    {
        // 这个界面每次弹出来的时候，重新查一下分数和数据
        ShowFinalResult();
        BindButtonEvents();
    }
    // 计算得分
    private void ShowFinalResult()
    {
        // 去问 PlayerController 我们一共走了多少步
        int steps = FindObjectOfType<PlayerController>().stepCount;
        // 去问 GameManager 我们一共撞了几次墙
        int hitCount = GameManager.instance.hitCount;
        // 给个基础分 100，每走一步扣一分，撞一次墙扣 5 分
        float baseScore = 100;
        float total = baseScore + steps * (-1f) + hitCount * (-5f);
        // 翻译算法选择
        string algoName = GameManager.instance.selectAlgorithm switch
        {
            0 => "Q-Learning",
            1 => "DQN",
            2 => "PPO",
            _ => "未知",
        };
        // 把数据填到界面里显示出来
        resultText.text = $"最终总步数：{steps}\n发生磕碰次数：{hitCount}\n总评估得分：{Mathf.RoundToInt(total)}";
        aiTrainText.text = $"核心驱动网络：{algoName}";
    }
private void BindButtonEvents()
    {
        // 1. 再来一局 (回到选择难度的界面)
        btnRestart.onClick.AddListener(() =>
        {
            PlayerController player = FindObjectOfType<PlayerController>();
            if (player != null)
                player.ResetPlayer();
            // 让 UIManager 把选难度的界面再端上来
            UIManager.instance.ShowPanel("DifficultyPanel");
        });
        // 2. 返回主菜单，一切重新开始
        btnBackMenu.onClick.AddListener(() =>
        {
            SceneManager.LoadScene("StartScene");
        });
        // 3. 退出游戏
        btnExit.onClick.AddListener(() =>
        {
            Application.Quit();
        });
    }
}
