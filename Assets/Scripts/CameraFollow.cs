// 正交摄像机：锁定在迷宫几何中心，Z=-10 避免被地图精灵裁剪
using UnityEngine;
public class CameraFollow : MonoBehaviour
{
    // ========== 配置参数 ==========
    [Header("上帝视角")]
    [Tooltip("追踪目标（可选，当前以地图中心为准）")]
    public Transform target;
    [Tooltip("正交视野大小，需随迷宫尺寸手动调整")]
    public float fixedSize = 8f;

    // ========== 渲染更新 ==========
    private void LateUpdate()
    {
        GetComponent<Camera>().orthographicSize = fixedSize;
        transform.position = new Vector3(GameData.mazeWidth / 2f, GameData.mazeHeight / 2f, -10f);
    }
}
