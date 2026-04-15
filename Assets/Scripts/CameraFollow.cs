using UnityEngine;
// 摄像机控制脚本
// 这个脚本就是用来固定摄像机的视角，让它能正好框住整个迷宫，而且一直在最中间。
// 这样我们在看AI自己跑的时候，就能像上帝视角一样清楚地看到它在干嘛。
public class CameraFollow : MonoBehaviour
{
    [Header("默认上帝视角")]
    public Transform target;
    [Header("视野范围")]
    public float fixedSize = 8f; // 能改摄像机的大小，要是迷宫变大了这个也得改大
    private void LateUpdate()
    {
        // 1. 把摄像机的视野大小定死，保证能看全图
        GetComponent<Camera>().orthographicSize = fixedSize;
        // 2. 算一下迷宫的中心点在哪里，就是长和宽的一半
        float centerX = GameData.mazeWidth / 2f;
        float centerY = GameData.mazeHeight / 2f;
        // 3. 把摄像机移动到这个中心点，Z轴拉到-10免得跟迷宫贴在一起看不见
        transform.position = new Vector3(centerX, centerY, -10f);
    }
}
