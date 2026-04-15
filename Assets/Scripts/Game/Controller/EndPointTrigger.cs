using UnityEngine;
// 终点触发器脚本
// 只要碰到终点了，就会发个通报给 GameManager 说通关了，然后要奖励。
public class EndPointTrigger : MonoBehaviour
{
    private void OnTriggerEnter2D(Collider2D other)
    {
        // 如果碰过来的是“玩家”（也就是我们的AI或者手动控制的角色）
        if (other.CompareTag("Player"))
        {
            // 就直接告诉主管的 GameManager 给奖励。
            // （如果是训练它就会重开一局，如果是手动玩它就会弹出通关界面）
            GameManager.instance.TriggerReward(GameManager.RewardType.ReachEnd);
        }
    }
}