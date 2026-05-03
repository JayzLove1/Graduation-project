using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace MazeAI.UI
{
    // 训练实时图表：用 UI Image 绘制奖励曲线和探索率曲线，数据来源于 GameManager
    public class TrainingChartUI : MonoBehaviour
    {
        // ========== UI 引用 ==========
        [Header("容器与文本")]
        public RectTransform chartContainer;
        public Text titleText;
        public Text infoText;

        // ========== 视觉配置 ==========
        [Header("曲线颜色")]
        public Color rewardLineColor  = new Color(0.2f, 1f,   0.4f, 1f);   // 奖励：亮绿
        public Color epsilonLineColor = new Color(1f,   0.9f, 0.2f, 1f);   // 探索率：金黄
        public Color gridLineColor    = new Color(1f,   1f,   1f,   0.1f); // 网格：极淡白

        [Header("线条参数")]
        public float lineThickness = 2f;
        public float dotSize       = 4f;
        public int   maxDataPoints = 100;

        // ========== 内部状态 ==========
        private List<GameObject> _visualElements = new List<GameObject>();

        // ========== 生命周期 ==========
        private void Start()
        {
            if (chartContainer == null) chartContainer = GetComponent<RectTransform>();
            // 跨场景引用丢失后重新注册，确保 GameManager 能直接推送数据
            if (GameManager.instance != null) GameManager.instance.trainingChart = this;
            UpdateChartTitle();
            UpdateInfoText();
        }

        // ========== 公开接口 ==========
        public void AddDataPoint(float reward, int steps)
        {
            UpdateChart();
            UpdateInfoText();
        }

        public void Clear() => UpdateChart();

        public void UpdateChartTitle()
        {
            if (titleText != null && GameManager.instance != null)
                titleText.text = $"{GameManager.instance.GetAlgorithmName()} Training ({GameData.mazeWidth}x{GameData.mazeHeight})";
        }

        // ========== 私有绘制 ==========
        private void UpdateInfoText()
        {
            if (infoText == null || GameManager.instance == null) return;
            string diff = GameData.mazeWidth <= 11 ? "Easy" : "Hard";
            infoText.text = $"[{GameManager.instance.GetAlgorithmName()} | {diff}] " +
                            $"Ep: {GameManager.instance.episodeCount} | " +
                            $"Hits: {GameManager.instance.hitCount} | " +
                            $"Success: {GameManager.instance.totalSuccessCount}";
        }

        private void UpdateChart()
        {
            foreach (var go in _visualElements)
                if (go != null) Destroy(go);
            _visualElements.Clear();

            if (GameManager.instance == null || GameManager.instance.rewardHistory.Count < 1) return;

            DrawGrid();

            if (GameManager.instance.rewardHistory.Count >= 2)
                DrawLine(GameManager.instance.rewardHistory, rewardLineColor);

            if (GameManager.instance.epsilonHistory.Count >= 2)
                DrawLine(GameManager.instance.epsilonHistory, epsilonLineColor, 0f, 1f);
        }

        private void DrawGrid()
        {
            float w = chartContainer.sizeDelta.x;
            float h = chartContainer.sizeDelta.y;
            for (int i = 1; i < 5;  i++) CreateLine(new Vector2(0, h / 5f * i),  new Vector2(w, h / 5f * i),  gridLineColor, 1f);
            for (int i = 1; i < 10; i++) CreateLine(new Vector2(w / 10f * i, 0), new Vector2(w / 10f * i, h), gridLineColor, 1f);
        }

        private void DrawLine(List<float> data, Color color, float? forcedMin = null, float? forcedMax = null)
        {
            float w = chartContainer.sizeDelta.x;
            float h = chartContainer.sizeDelta.y;

            float min = forcedMin ?? float.MaxValue;
            float max = forcedMax ?? float.MinValue;
            if (!forcedMin.HasValue || !forcedMax.HasValue)
                foreach (var v in data)
                {
                    if (!forcedMin.HasValue  && v < min) min = v;
                    if (!forcedMax.HasValue && v > max) max = v;
                }

            // 防止极差为 0 时 InverseLerp 除零
            if (Mathf.Approximately(max, min)) { max = min + 1f; min -= 1f; }

            float xStep = w / (maxDataPoints - 1);
            int   last  = data.Count - 1;
            for (int i = 0; i < last; i++)
            {
                Vector2 a = new(i * xStep,       Mathf.InverseLerp(min, max, data[i])     * h);
                Vector2 b = new((i + 1) * xStep, Mathf.InverseLerp(min, max, data[i + 1]) * h);
                CreateLine(a, b, color, lineThickness);
                CreateDot(a, color);
            }
            if (data.Count > 0)
                CreateDot(new(last * xStep, Mathf.InverseLerp(min, max, data[^1]) * h), color);
        }

        private void CreateLine(Vector2 a, Vector2 b, Color color, float thickness)
        {
            var go = new GameObject("chartLine", typeof(Image));
            go.transform.SetParent(chartContainer, false);
            go.GetComponent<Image>().color = color;
            _visualElements.Add(go);

            var rt  = go.GetComponent<RectTransform>();
            var dir = (b - a).normalized;
            rt.anchorMin         = Vector2.zero;
            rt.anchorMax         = Vector2.zero;
            float dist           = Vector2.Distance(a, b);
            rt.sizeDelta         = new Vector2(dist, thickness);
            rt.anchoredPosition  = a + dir * (dist * 0.5f);
            rt.localEulerAngles  = new Vector3(0, 0, Mathf.Atan2(dir.y, dir.x) * Mathf.Rad2Deg);
        }

        private void CreateDot(Vector2 position, Color color)
        {
            var go = new GameObject("chartDot", typeof(Image));
            go.transform.SetParent(chartContainer, false);
            go.GetComponent<Image>().color = color;
            _visualElements.Add(go);

            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin        = Vector2.zero;
            rt.anchorMax        = Vector2.zero;
            rt.sizeDelta        = new Vector2(dotSize, dotSize);
            rt.anchoredPosition = position;
        }
    }
}
