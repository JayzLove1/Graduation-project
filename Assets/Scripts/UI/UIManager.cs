using System.Collections;
using UnityEngine;
// UI界面管理器
// 用来管理所有的全屏界面（比如主菜单、难度选择、通关界面）的切换。
// 还能防止玩家手滑一直点，导致界面全叠在一起。
public class UIManager : MonoBehaviour
{
    public static UIManager instance;
    private void Awake()
    {
        if (instance == null)
        {
            instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }
    // 防连点锁：防止玩家手速太快连续按按钮
    private bool isSwitchingPanel = false;
    // 显示指定的面板，同时把其他的关掉
    public void ShowPanel(string panelName)
    {
        if (isSwitchingPanel)
            return;
        isSwitchingPanel = true;
        // 1. 把所有打上了 "UIPanel" 标签的界面先全关掉
        GameObject[] allPanels = GameObject.FindGameObjectsWithTag("UIPanel");
        foreach (var p in allPanels)
            p.SetActive(false);
        // 2. 根据名字把我们要的那个界面打开
        Transform target = GameObject.Find("Canvas").transform.Find(panelName);
        if (target != null)
        {
            target.gameObject.SetActive(true);
        }
        // 3. 开始一个计时器，过一会才允许点击下一个界面
        StartCoroutine(AllowPanelSwitch());
    }
    private IEnumerator AllowPanelSwitch()
    {
        // 等待个半秒钟，不准乱按
        yield return new WaitForSeconds(0.5f);
        isSwitchingPanel = false;
    }
}
