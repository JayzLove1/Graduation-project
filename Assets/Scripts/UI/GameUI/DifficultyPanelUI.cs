using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
// 难度选择界面脚本 (DifficultyPanelUI)
// 就是那个让玩家选迷宫要多大、用什么AI模型，以及是怎么玩的那个界面。
public class DifficultyPanelUI : MonoBehaviour
{
    [Header("迷宫规格配置")]
    public Button btnEasy; // 简单模式：生成 11x11 大小的迷宫
    public Button btnHard; // 困难模式：生成 15x15 的迷宫
    [Header("AI参数配置")]
    public Dropdown algorithmDropdown; // 选要用哪个算法的下拉框
    // 选是要手动玩还是让AI训练的下拉框
    public Dropdown modeDropdown;
    // 记下玩家在下拉框选了什么，点开始的时候一起交上去
    private int selectedAlgorithmIndex = 0;
    private int selectedModeIndex = 0;
    private void Start()
    {
        // 给按钮绑定功能，点了之后生成对应大小的迷宫
        btnEasy.onClick.AddListener(() => StartGame(11));
        btnHard.onClick.AddListener(() => StartGame(15));
        // 初始化算法选择下拉框
        algorithmDropdown.ClearOptions();
        algorithmDropdown.AddOptions(
            new System.Collections.Generic.List<string> { "Q-Learning", "DQN", "PPO" }
        );
        algorithmDropdown.onValueChanged.AddListener(index =>
        {
            selectedAlgorithmIndex = index;
        });
        // 初始化模式选择下拉框
        modeDropdown.ClearOptions();
        modeDropdown.AddOptions(
            new System.Collections.Generic.List<string> { "手动测试模式", "AI训练模式", "AI演示模式", "自动化批量训练" }
        );
        modeDropdown.onValueChanged.AddListener(index =>
        {
            selectedModeIndex = index;
        });
    }
    // 把刚才选的配置全部发给游戏大管家，并且切换到真正的游玩场景
    private void StartGame(int mazeSize)
    {
        // 1. 记下迷宫有多大
        GameData.mazeWidth = mazeSize;
        GameData.mazeHeight = mazeSize;
        // 2. 告诉 GameManager 咱们选了啥算法和啥模式
        if (GameManager.instance != null)
        {
            GameManager.instance.SwitchAlgorithm(selectedAlgorithmIndex);
            GameManager.instance.SwitchRunMode(selectedModeIndex);
        }
        // 3. 回到主场景
        SceneManager.LoadScene("GameScene");
    }
}
