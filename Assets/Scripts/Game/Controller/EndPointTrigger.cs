using UnityEngine;

// 终点触发区：Agent 进入后锁定移动并通知 GameManager 结算奖励
public class EndPointTrigger : MonoBehaviour
{
    // ========== 碰撞响应 ==========
    private void OnTriggerEnter2D(Collider2D other)
    {
        if (!other.CompareTag("Player")) return;
        // 先锁移动，防止通关后残留动作指令将 Agent 移出终点区域
        other.GetComponent<PlayerController>()?.LockMovement();
        GameManager.instance.TriggerReward(GameManager.RewardType.ReachEnd);
    }
}
