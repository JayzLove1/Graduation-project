using System.Collections;
using UnityEngine;

// 全局面板管理器：同一时刻只显示一个 UIPanel，切换时自动隐藏其余面板
public class UIManager : MonoBehaviour
{
    public static UIManager instance;

    // isSwitchingPanel：0.5s 冷却防止快速连点导致面板叠显
    private bool isSwitchingPanel;

    // ========== 生命周期 ==========
    private void Awake()
    {
        if (instance == null) { instance = this; DontDestroyOnLoad(gameObject); }
        else Destroy(gameObject);
    }

    // ========== 面板导航 ==========
    public void ShowPanel(string panelName)
    {
        if (isSwitchingPanel) return;
        isSwitchingPanel = true;

        // 隐藏所有标记为 UIPanel 的面板
        foreach (var p in GameObject.FindGameObjectsWithTag("UIPanel"))
            p.SetActive(false);

        // 激活目标面板
        GameObject.Find("Canvas")?.transform.Find(panelName)?.gameObject.SetActive(true);

        StartCoroutine(UnlockAfterDelay());
    }

    private IEnumerator UnlockAfterDelay()
    {
        yield return new WaitForSeconds(0.5f);
        isSwitchingPanel = false;
    }
}
