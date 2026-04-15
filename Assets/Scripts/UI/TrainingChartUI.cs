using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
namespace MazeAI.UI
{
    /// <summary>
    /// 实时训练折线图工具
    /// 用于在 Unity 界面中动态绘制强化学习的训练曲线（如奖励值、步数等）
    /// </summary>
    public class TrainingChartUI : MonoBehaviour
    {
        [Header("UI 引用")]
        public RectTransform chartContainer; // 图表绘制区域
        public Text titleText;
        public Text infoText;
        [Header("设计设置")]
        public Color rewardLineColor = new Color(0.2f, 1f, 0.4f, 1f); // 绿色
        public Color stepLineColor = new Color(0.2f, 0.6f, 1f, 1f);   // 蓝色
        public Color epsilonLineColor = new Color(1f, 0.9f, 0.2f, 1f); // 黄色(Epsilon)
        public Color gridLineColor = new Color(1f, 1f, 1f, 0.1f);     // 极淡的白色格子
        public float lineThickness = 2f;
        public float dotSize = 4f;     // 圆点大小
        public int maxDataPoints = 100; // 增加显示点数到 100
        private List<float> _rewardHistory = new List<float>();
        private List<float> _stepHistory = new List<float>();
        private List<GameObject> _visualElements = new List<GameObject>();
        private void Start()
        {
            if (chartContainer == null) chartContainer = GetComponent<RectTransform>();
            // 重要：注册到 GameManager，解决跨场景引用丢失问题
            if (GameManager.instance != null)
            {
                GameManager.instance.trainingChart = this;
            }
            UpdateChartTitle();
            UpdateInfoText();
        }
        /// <summary>
        /// 添加一个新的数据点并重绘
        /// </summary>
        public void AddDataPoint(float reward, int steps)
        {
            // 数据现在持久化存储在 GameManager 中，这里只需调用重绘
            UpdateChart();
            UpdateInfoText();
        }
        private void UpdateInfoText()
        {
            if (infoText != null && GameManager.instance != null)
            {
                string algoName = GameManager.instance.GetAlgorithmName();
                string diff = GameData.mazeWidth <= 11 ? "Easy" : "Hard";
                infoText.text = $"[{algoName} | {diff}] Ep: {GameManager.instance.episodeCount} | Hits: {GameManager.instance.hitCount} | Success: {GameManager.instance.totalSuccessCount}";
            }
        }
        public void UpdateChartTitle()
        {
            if (titleText != null && GameManager.instance != null)
            {
                string algoName = GameManager.instance.GetAlgorithmName();
                string size = $"{GameData.mazeWidth}x{GameData.mazeWidth}";
                titleText.text = $"{algoName} Training Performance ({size})";
            }
        }
        private void UpdateChart()
        {
            // 清理旧的线段
            foreach (var go in _visualElements)
            {
                if (go != null) Destroy(go);
            }
            _visualElements.Clear();
            if (GameManager.instance == null || GameManager.instance.rewardHistory.Count < 1) return;
            // 1. 绘制背景网格 (Grid) 让图表看起来更专业
            DrawGrid();
            // 2. 绘制奖励曲线 (绿色)
            if (GameManager.instance.rewardHistory.Count >= 2)
            {
                DrawLine(GameManager.instance.rewardHistory, rewardLineColor);
            }
            // 3. 绘制 Epsilon 曲线 (黄色 - 它的范围固定在 0-1)
            if (GameManager.instance.epsilonHistory.Count >= 2)
            {
                DrawLine(GameManager.instance.epsilonHistory, epsilonLineColor, 0f, 1f);
            }
        }
        private void DrawGrid()
        {
            float width = chartContainer.sizeDelta.x;
            float height = chartContainer.sizeDelta.y;
            int hDivisions = 5; // 横线数
            int vDivisions = 10; // 纵线数
            for (int i = 1; i < hDivisions; i++)
            {
                float y = (height / hDivisions) * i;
                CreateLine(new Vector2(0, y), new Vector2(width, y), gridLineColor, 1f);
            }
            for (int i = 1; i < vDivisions; i++)
            {
                float x = (width / vDivisions) * i;
                CreateLine(new Vector2(x, 0), new Vector2(x, height), gridLineColor, 1f);
            }
        }
        private void DrawLine(List<float> data, Color color, float? forcedMin = null, float? forcedMax = null)
        {
            float width = chartContainer.sizeDelta.x;
            float height = chartContainer.sizeDelta.y;
            // 计算最大最小值以便缩放
            float min = forcedMin ?? float.MaxValue;
            float max = forcedMax ?? float.MinValue;
            if (!forcedMin.HasValue || !forcedMax.HasValue)
            {
                foreach (var v in data)
                {
                    if (!forcedMin.HasValue && v < min) min = v;
                    if (!forcedMax.HasValue && v > max) max = v;
                }
            }
            // 防止除零
            if (Mathf.Approximately(max, min))
            {
                max = min + 1f;
                min = min - 1f;
            }
            float xPadding = width / (maxDataPoints - 1);
            for (int i = 0; i < data.Count - 1; i++)
            {
                Vector2 posA = new Vector2(i * xPadding, Mathf.InverseLerp(min, max, data[i]) * height);
                Vector2 posB = new Vector2((i + 1) * xPadding, Mathf.InverseLerp(min, max, data[i + 1]) * height);
                // 画折线
                CreateLine(posA, posB, color, lineThickness);
                // 画数据点
                CreateDot(posA, color);
            }
            // 画最后一个点
            if (data.Count > 0)
            {
                Vector2 lastPos = new Vector2((data.Count - 1) * xPadding, Mathf.InverseLerp(min, max, data[data.Count - 1]) * height);
                CreateDot(lastPos, color);
            }
        }
        private void CreateLine(Vector2 dotPositionA, Vector2 dotPositionB, Color color, float thickness)
        {
            GameObject gameObject = new GameObject("chartLine", typeof(Image));
            gameObject.transform.SetParent(chartContainer, false);
            gameObject.GetComponent<Image>().color = color;
            _visualElements.Add(gameObject);
            RectTransform rectTransform = gameObject.GetComponent<RectTransform>();
            Vector2 dir = (dotPositionB - dotPositionA).normalized;
            float distance = Vector2.Distance(dotPositionA, dotPositionB);
            rectTransform.anchorMin = Vector2.zero;
            rectTransform.anchorMax = Vector2.zero;
            rectTransform.sizeDelta = new Vector2(distance, thickness);
            rectTransform.anchoredPosition = dotPositionA + dir * distance * 0.5f;
            rectTransform.localEulerAngles = new Vector3(0, 0, Mathf.Atan2(dir.y, dir.x) * Mathf.Rad2Deg);
        }
        private void CreateDot(Vector2 position, Color color)
        {
            GameObject gameObject = new GameObject("chartDot", typeof(Image));
            gameObject.transform.SetParent(chartContainer, false);
            // 如果你想让圆点更好看，可以在这里动态加载一个圆形 Sprite
            gameObject.GetComponent<Image>().color = color;
            _visualElements.Add(gameObject);
            RectTransform rectTransform = gameObject.GetComponent<RectTransform>();
            rectTransform.anchorMin = Vector2.zero;
            rectTransform.anchorMax = Vector2.zero;
            rectTransform.sizeDelta = new Vector2(dotSize, dotSize);
            rectTransform.anchoredPosition = position;
        }
        public void Clear()
        {
            _rewardHistory.Clear();
            _stepHistory.Clear();
            UpdateChart();
        }
    }
}