using UnityEngine;
using UnityEngine.UI;

namespace MazeAI.UI
{
    // 超参数实时控制面板：UI 输入 → GameManager → Python IPC
    // 防抖：0.5s 内多次修改只推送一次，避免高频 IPC 阻塞
    public class TrainingParamUI : MonoBehaviour
    {
        // ========== UI 控件 ==========
        [Header("超参数输入框")]
        public InputField lrInput;        // 学习率
        public InputField neuralLrInput;  // 神经网络学习率 (Adam LR)
        public InputField epsilonInput;   // 探索率 / PPO 熵系数
        public InputField gammaInput;     // 折扣因子
        public InputField batchSizeInput; // 批次大小（整数）

        // ========== 防抖状态 ==========
        private float _lastUpdateTime;
        private bool  _hasPendingUpdate;
        private const float MinUpdateInterval = 0.5f;

        // ========== 初始化 ==========
        private void Start()
        {
            // 从 GameManager 恢复上次的参数到 UI
            if (GameManager.instance != null)
            {
                var tp = GameManager.instance.trainParam;
                UpdateInputTexts(tp.lr, tp.algoLr, tp.epsilon, tp.gamma, tp.batchSize);
            }

            // OnEndEdit 模式：用户确认输入后才触发，避免中途字符触发无效更新
            lrInput.onEndEdit.AddListener(val        => { if (float.TryParse(val, out float f))           { GameManager.instance.trainParam.lr        = f; MarkForUpdate(); } });
            neuralLrInput.onEndEdit.AddListener(val  => { if (float.TryParse(val, out float f))           { GameManager.instance.trainParam.algoLr    = f; MarkForUpdate(); } });
            epsilonInput.onEndEdit.AddListener(val   => { if (float.TryParse(val, out float f))           { GameManager.instance.trainParam.epsilon   = f; MarkForUpdate(); } });
            gammaInput.onEndEdit.AddListener(val     => { if (float.TryParse(val, out float f))           { GameManager.instance.trainParam.gamma     = f; MarkForUpdate(); } });
            // batchSize 用 int.TryParse，防止输入小数点时误更新为 0
            batchSizeInput?.onEndEdit.AddListener(val => { if (int.TryParse(val, out int n) && n > 0)     { GameManager.instance.trainParam.batchSize = n; MarkForUpdate(); } });
        }

        // ========== 更新循环 ==========
        private void Update()
        {
            // Python 端（如 PPO 熵衰减）会回传 epsilon，未处于编辑中时同步到 UI
            if (GameManager.instance != null && !_hasPendingUpdate)
            {
                var tp = GameManager.instance.trainParam;
                UpdateInputTexts(tp.lr, tp.algoLr, tp.epsilon, tp.gamma, tp.batchSize);
            }

            // 防抖到期后批量推送至 Python
            if (_hasPendingUpdate && Time.time - _lastUpdateTime > MinUpdateInterval)
            {
                GameManager.instance?.UpdateRealtimeParams();
                _lastUpdateTime   = Time.time;
                _hasPendingUpdate = false;
            }
        }

        // ========== 私有工具 ==========
        private void MarkForUpdate() => _hasPendingUpdate = true;

        private void UpdateInputTexts(float lr, float algoLr, float eps, float gamma, int batch)
        {
            if (!lrInput.isFocused)        lrInput.text        = lr.ToString("F4");
            if (!neuralLrInput.isFocused)  neuralLrInput.text  = algoLr.ToString("F4");
            if (!epsilonInput.isFocused)   epsilonInput.text   = eps.ToString("F4");
            if (!gammaInput.isFocused)     gammaInput.text     = gamma.ToString("F2");
            if (batchSizeInput != null && !batchSizeInput.isFocused)
                batchSizeInput.text = batch.ToString();
        }
    }
}
