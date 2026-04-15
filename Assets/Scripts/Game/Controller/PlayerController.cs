using System.Collections;
using UnityEngine;
// 玩家小球控制脚本 (PlayerController)
// 这个脚本就是用来管迷宫里那个小球怎么走的，撞到墙怎么办，还有就是记下了自己在这一局走了几步。
// 如果是AI在训练，它还会负责听AI（Python那边）的话，它说往左走就往左走。
public class PlayerController : MonoBehaviour
{
    [Header("玩家(AI)参数")]
    public float moveSpeed = 5f;       // 手动/演示模式移动速度（每秒格数）
    public float trainMoveSpeed = 30f; // 训练模式专用速度（仅 runMode==1 时自动切换）
    private Vector2 targetPos;    // 马上要去的位置坐标
    public bool isMoving;         // 记录当前是不是还在半路上（如果还在移动就不听指挥）
    public int stepCount;         // 记录这一把一共走了几步
    public bool canMove = true;   // 全局的移动开关（如果通关了或者被暂停了就禁止移动了）
    // 防止AI一直往同一面墙上撞
    private bool _hasTriggeredHitWall = false;
    private void Start() => ResetPlayer();
    private void Update()
    {
        // 如果开关关了，或者正在移动的过程中，就别接受新的上下左右了
        if (!canMove || isMoving)
            return;
        // 【手动测试模式】 runMode == 0 时，可以用键盘上下左右或者WASD走
        if (GameManager.instance.runMode == 0)
        {
            float x = Input.GetAxisRaw("Horizontal");
            float y = Input.GetAxisRaw("Vertical");
            if (Mathf.Abs(x) > 0)
                TryMove(new Vector2(x, 0));
            else if (Mathf.Abs(y) > 0)
                TryMove(new Vector2(0, y));
        }
    }
    // 尝试往某个方向走一步
    // 无论是玩家键盘按的，还是AI在后面算出来的结果，最后都要在这个函数里排队。
    public void TryMove(Vector2 dir)
    {
        if (!canMove || isMoving) return;
        // 【给AI专用的防卡墙设置】
        // 因为AI每下一次命令就算重新走一步。必须要把它复位一下。
        if (GameManager.instance.runMode == 1 || GameManager.instance.runMode == 2 || GameManager.instance.runMode == 3)
            _hasTriggeredHitWall = false;
        Vector2 next = targetPos + dir; // 用口算一下下一步去哪里
        // 碰壁预检：看看下一步的格子上是不是贴了 "Wall" 的标签
        if (Physics2D.OverlapCircle(next, 0.1f, LayerMask.GetMask("Wall")))
        {
            // 如果撞墙了：
            if (!_hasTriggeredHitWall)
            {
                // 1. 赶紧找 GameManager 投诉，说自己撞墙了，看该扣分就扣分
                GameManager.instance.TriggerReward(GameManager.RewardType.HitWall);
                _hasTriggeredHitWall = true;
                // 2. 如果是 AI 玩的，既然都撞头了，也得算它这一步结束了，必须回句话给AI，让他想接下来的办法，不然它会干等。
                if (GameManager.instance.runMode == 1 || GameManager.instance.runMode == 2 || GameManager.instance.runMode == 3)
                {
                    stepCount++;
                    GameManager.instance.OnAIMoveComplete();
                }
            }
            return; // 终止本次尝试，不实际修改位置
        }
        // 没碰到墙，标记好新的落脚点，启动平滑移动协程
        _hasTriggeredHitWall = false;
        targetPos = next;
        isMoving = true;
        StartCoroutine(SmoothMoveCoroutine(transform.position, targetPos));
    }
    /// <summary>
    /// 使用 SmoothStep 缓动曲线平滑移动到目标位置（先加速后减速，视觉上更自然）
    /// 训练模式下 duration 极短因此速度更快，但曲线形状不变
    /// </summary>
    private IEnumerator SmoothMoveCoroutine(Vector2 from, Vector2 to)
    {
        bool isTraining = GameManager.instance != null && (GameManager.instance.runMode == 1 || GameManager.instance.runMode == 3);
        // 根据速度计算本次移动耗时（格子距离固定为1）
        float duration = 1f / (isTraining ? trainMoveSpeed : moveSpeed);
        float elapsed = 0f;
        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            // SmoothStep：t² × (3 - 2t)，产生先加速后减速的缓动效果
            float t = Mathf.Clamp01(elapsed / duration);
            float smoothT = t * t * (3f - 2f * t);
            transform.position = Vector2.LerpUnclamped(from, to, smoothT);
            yield return null;
        }
        // 确保最终精确落在目标格子中心
        transform.position = to;
        isMoving = false;
        stepCount++;
        _hasTriggeredHitWall = false;
        // 通知 GameManager 记录本步奖励并让 AI 继续决策
        GameManager.instance.TriggerReward(GameManager.RewardType.NormalMove);
        GameManager.instance.OnAIMoveComplete();
    }
    /// <summary>
    /// 重置状态回到 (1,1) 起点
    /// 在每次 Episode (回合) 重开或者重新载入关卡时调用
    /// </summary>
    public void ResetPlayer()
    {
        StopAllCoroutines(); // 终止正在进行的移动，防止旧协程干扰新一局
        targetPos = new Vector2(1, 1); // 我们游戏里起点永远是第二格 (1,1)
        transform.position = targetPos;
        stepCount = 0;
        isMoving = false;
        canMove = true;
        _hasTriggeredHitWall = false;
    }
    // 锁定角色输入（当碰到终点获胜后调用）
    public void LockMovement()
    {
        StopAllCoroutines(); // 同时停止移动协程
        canMove = false;
        isMoving = false;
    }
}