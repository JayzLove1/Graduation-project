using System.Collections;
using UnityEngine;

// Agent 物理控制器：执行移动指令、碰撞预检、SmoothStep 缓动，并维护步进计数
public class PlayerController : MonoBehaviour
{
    // ========== 运动参数 ==========
    [Header("移动速率")]
    [Tooltip("手动/演示模式的移动速度")]
    public float moveSpeed = 5f;
    [Tooltip("训练模式加速，缩短单步耗时以提升训练效率")]
    public float trainMoveSpeed = 30f;

    // ========== 内部状态 ==========
    private Vector2 targetPos;
    public bool isMoving;
    public int  stepCount;
    public bool canMove = true;
    private bool _hasTriggeredHitWall;

    // ========== 生命周期 ==========
    private void Start() => ResetPlayer();

    private void Update()
    {
        if (!canMove || isMoving) return;
        // 手动模式：读取键盘输入
        if (GameManager.instance.runMode == 0)
        {
            float x = Input.GetAxisRaw("Horizontal");
            float y = Input.GetAxisRaw("Vertical");
            if (Mathf.Abs(x) > 0)      TryMove(new Vector2(x, 0));
            else if (Mathf.Abs(y) > 0) TryMove(new Vector2(0, y));
        }
    }

    // ========== 动作执行 ==========
    public void TryMove(Vector2 dir)
    {
        if (!canMove || isMoving) return;

        // 每步前重置碰墙标记，避免同一格子碰撞被重复触发
        if (GameManager.instance.runMode != 0)
            _hasTriggeredHitWall = false;

        Vector2 next = targetPos + dir;

        // 碰墙检测：目标格有 Wall 层碰撞体时触发惩罚并跳过移动
        if (Physics2D.OverlapCircle(next, 0.1f, LayerMask.GetMask("Wall")))
        {
            if (!_hasTriggeredHitWall)
            {
                GameManager.instance.TriggerReward(GameManager.RewardType.HitWall);
                _hasTriggeredHitWall = true;
                if (GameManager.instance.runMode != 0)
                {
                    stepCount++;
                    GameManager.instance.OnAIMoveComplete();
                }
            }
            return;
        }

        _hasTriggeredHitWall = false;
        targetPos = next;
        isMoving  = true;
        StartCoroutine(SmoothMoveCoroutine(transform.position, targetPos));
    }

    // SmoothStep 缓动：t² × (3 - 2t)，使位移在起止端均匀减速
    private IEnumerator SmoothMoveCoroutine(Vector2 from, Vector2 to)
    {
        bool isTraining = GameManager.instance != null &&
                          (GameManager.instance.runMode == 1 || GameManager.instance.runMode == 3);
        float duration = 1f / (isTraining ? trainMoveSpeed : moveSpeed);
        float elapsed  = 0f;

        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            float t       = Mathf.Clamp01(elapsed / duration);
            float smoothT = t * t * (3f - 2f * t);
            transform.position = Vector2.LerpUnclamped(from, to, smoothT);
            yield return null;
        }

        transform.position    = to;
        isMoving              = false;
        stepCount++;
        _hasTriggeredHitWall  = false;
        GameManager.instance.TriggerReward(GameManager.RewardType.NormalMove);
        GameManager.instance.OnAIMoveComplete();
    }

    // ========== 状态控制 ==========
    // 将 Agent 归位至起点 (1,1) 并清空所有运行时状态
    public void ResetPlayer()
    {
        StopAllCoroutines();
        targetPos            = new Vector2(1, 1);
        transform.position   = targetPos;
        stepCount            = 0;
        isMoving             = false;
        canMove              = true;
        _hasTriggeredHitWall = false;
    }

    // 通关/暂停时锁定输入，终止所有进行中的移动协程
    public void LockMovement()
    {
        StopAllCoroutines();
        canMove  = false;
        isMoving = false;
    }
}
