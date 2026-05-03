using UnityEngine;

// [已废弃] 此脚本不再承担任何初始化逻辑。
// 场景初始化已由 GameManager.OnSceneLoaded 事件统一处理；保留空类只是为了避免
// 删除文件后场景中残留 Missing Script 引用。
//
// 安全删除步骤：
//   1. 在 GameScene 中找到挂载本组件的 GameObject，移除 GameInit 组件；
//   2. 保存场景；
//   3. 删除 GameInit.cs 与 GameInit.cs.meta。
[System.Obsolete("GameInit 不再使用，请按文件头注释步骤从场景中移除组件后删除该文件。")]
public class GameInit : MonoBehaviour
{
}
