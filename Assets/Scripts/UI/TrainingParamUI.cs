using UnityEngine;
using UnityEngine.UI;
namespace MazeAI.UI
{
    /// <summary>
    /// 训练超参数实时调节面板
    /// 这里的改动会直接写进 GameManager 的 trainParam，并同步给 Python 进程
    /// </summary>
    public class TrainingParamUI : MonoBehaviour
    {
        [Header("UI 控件引用")]
        public InputField lrInput;
        public InputField neuralLrInput;
        public InputField epsilonInput;
        public InputField gammaInput;
        public InputField batchSizeInput;
        [Header("更新间隔设置")]
        // 防抖：不用每帧都更新给 Python，减少 IPC 通信开销
        private float _lastUpdateTime = 0;
        private const float MinUpdateInterval = 0.5f;
        private bool _hasPendingUpdate = false;
        private void Start()
        {
            // 1. 初始化输入框的默认值（从 GameManager 同步数据）
            if (GameManager.instance != null)
            {
                var tp = GameManager.instance.trainParam;
                UpdateInputTexts(tp.lr, tp.algoLr, tp.epsilon, tp.gamma, tp.batchSize);
            }
            // 2. 绑定事件监听器 (使用 OnEndEdit：当玩家输完数值按回车或点开时触发同步)
            lrInput.onEndEdit.AddListener(val => { if (float.TryParse(val, out float f)) { GameManager.instance.trainParam.lr = f; MarkForUpdate(); } });
            neuralLrInput.onEndEdit.AddListener(val => { if (float.TryParse(val, out float f)) { GameManager.instance.trainParam.algoLr = f; MarkForUpdate(); } });
            epsilonInput.onEndEdit.AddListener(val => { if (float.TryParse(val, out float f)) { GameManager.instance.trainParam.epsilon = f; MarkForUpdate(); } });
            gammaInput.onEndEdit.AddListener(val => { if (float.TryParse(val, out float f)) { GameManager.instance.trainParam.gamma = f; MarkForUpdate(); } });
            batchSizeInput.onEndEdit.AddListener(val => { if (int.TryParse(val, out int i)) { GameManager.instance.trainParam.batchSize = i; MarkForUpdate(); } });
        }
        private void MarkForUpdate()
        {
            _hasPendingUpdate = true;
        }
        private void UpdateInputTexts(float lr, float algoLr, float eps, float gamma, int batch)
        {
            // 格式化输出到输入框，防止长浮点数看着乱
            if (!lrInput.isFocused) lrInput.text = lr.ToString("F4");
            if (!neuralLrInput.isFocused) neuralLrInput.text = algoLr.ToString("F4");
            if (!epsilonInput.isFocused) epsilonInput.text = eps.ToString("F4");
            if (!gammaInput.isFocused) gammaInput.text = gamma.ToString("F2");
            if (!batchSizeInput.isFocused) batchSizeInput.text = batch.ToString();
        }
        private void Update()
        {
            // 实时反向同步：如果 Python 端更新了 Epsilon (比如自动衰减)，我们要反向显示在输入框里
            if (GameManager.instance != null && !_hasPendingUpdate)
            {
                var tp = GameManager.instance.trainParam;
                UpdateInputTexts(tp.lr, tp.algoLr, tp.epsilon, tp.gamma, tp.batchSize);
            }
            // 防抖逻辑：只有改变了参数，且距离上一次同步超过一定时间，才会发送给 Python
            if (_hasPendingUpdate && Time.time - _lastUpdateTime > MinUpdateInterval)
            {
                if (GameManager.instance != null)
                {
                    GameManager.instance.UpdateRealtimeParams();
                }
                _lastUpdateTime = Time.time;
                _hasPendingUpdate = false;
            }
        }
    }
}