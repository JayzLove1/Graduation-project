using UnityEngine;

// 游戏开始初始化的脚本
// 它的作用就是：等真正的游戏场景（GameScene）加载完之后，马上告诉主管一切的 GameManager。
public class GameInit : MonoBehaviour
{
    private void Start()
    {
        // 场景一加载好，就让 GameManager 去看一下现在是什么模式。
        // 然后 GameManager 就会决定现在的模式是自己玩、还是AI在训练、或者AI在表演，最后决定要不要启动后端的 Python 代码。
        if (GameManager.instance != null)
        {
            GameManager.instance.InitAfterSceneLoaded();
        }
    }
}
