using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

// 实验配置面板：选择迷宫大小、算法和运行模式后进入 GameScene
public class DifficultyPanelUI : MonoBehaviour
{
    // ========== UI 组件 ==========
    [Header("迷宫规格")]
    [Tooltip("11×11 简单迷宫")]
    public Button btnEasy;
    [Tooltip("15×15 困难迷宫")]
    public Button btnHard;

    [Header("AI 参数")]
    [Tooltip("算法：Q-Learning / DQN / PPO")]
    public Dropdown algorithmDropdown;
    [Tooltip("模式：手动 / 训练 / 演示 / 批量")]
    public Dropdown modeDropdown;

    // ========== 内部状态 ==========
    private int selectedAlgorithmIndex;
    private int selectedModeIndex;

    // ========== 初始化 ==========
    private void Start()
    {
        btnEasy.onClick.AddListener(() => StartGame(11));
        btnHard.onClick.AddListener(() => StartGame(15));

        algorithmDropdown.ClearOptions();
        algorithmDropdown.AddOptions(new List<string> { "Q-Learning", "DQN", "PPO" });
        algorithmDropdown.onValueChanged.AddListener(i => selectedAlgorithmIndex = i);

        modeDropdown.ClearOptions();
        modeDropdown.AddOptions(new List<string> { "手动测试模式", "AI训练模式", "AI演示模式", "自动化批量训练" });
        modeDropdown.onValueChanged.AddListener(i => selectedModeIndex = i);
    }

    // ========== 游戏启动 ==========
    private void StartGame(int mazeSize)
    {
        GameData.mazeWidth  = mazeSize;
        GameData.mazeHeight = mazeSize;
        GameManager.instance?.SwitchAlgorithm(selectedAlgorithmIndex);
        GameManager.instance?.SwitchRunMode(selectedModeIndex);
        SceneManager.LoadScene("GameScene");
    }
}
