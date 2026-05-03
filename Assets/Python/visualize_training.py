# -*- coding: utf-8 -*-
"""
训练数据可视化工具 (visualize_training.py)
扫描 training_data~ 目录，读取各算法的 CSV 日志，
绘制 奖励 / 步数 / 撞墙数 / 探索率 曲线并保存为 PNG。

使用方法:
    python visualize_training.py
依赖环境:
    pip install matplotlib pandas
"""
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # 必须在 import pyplot 之前

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

# ==========================================================================
# 中文字体配置
# 核心策略：直接用 TTF/TTC 文件路径创建 FontProperties
# 完全绕过 matplotlib 字体名称缓存（缓存是中文乱码最常见根源）
# ==========================================================================
_CANDIDATE_FONTS = [
    r"C:\Windows\Fonts\simhei.ttf",      # 黑体（几乎所有 Windows 都有）
    r"C:\Windows\Fonts\msyh.ttc",        # 微软雅黑
    r"C:\Windows\Fonts\msyhbd.ttc",      # 微软雅黑 Bold
    r"C:\Windows\Fonts\simsun.ttc",      # 宋体
    r"C:\Windows\Fonts\simkai.ttf",      # 楷体
    r"C:\Windows\Fonts\simfang.ttf",     # 仿宋
]

ZH_FONT: FontProperties | None = None

for _fp in _CANDIDATE_FONTS:
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)          # 注册到字体管理器
        ZH_FONT = FontProperties(fname=_fp)            # 直接绑定文件
        plt.rcParams["font.sans-serif"] = [ZH_FONT.get_name()]
        plt.rcParams["axes.unicode_minus"] = False     # 修复负号变方框
        print(f"[Viz] 中文字体已加载: {_fp}")
        break
else:
    plt.rcParams["axes.unicode_minus"] = False
    print("[Viz] 警告: 未找到系统中文字体，将以英文显示标签。")

# ---------- 便捷封装（每个文字元素都显式传入 fontproperties） ----------

def _fp_kw() -> dict:
    """返回 fontproperties 关键字参数（若字体可用）。"""
    return {"fontproperties": ZH_FONT} if ZH_FONT else {}

def _suptitle(fig, text: str, **extra):
    fig.suptitle(text, **_fp_kw(), **extra)

def _set_title(ax, text: str, pad=10, **extra):
    ax.set_title(text, pad=pad, **_fp_kw(), **extra)

def _set_xlabel(ax, text: str):
    ax.set_xlabel(text, **_fp_kw())

def _set_ylabel(ax, text: str):
    ax.set_ylabel(text, **_fp_kw())

def _add_legend(ax, **kw):
    leg = ax.legend(**kw)
    if ZH_FONT and leg:
        for t in leg.get_texts():
            t.set_fontproperties(ZH_FONT)

# ==========================================================================
# 主题配置（Dark Premium）
# ==========================================================================
plt.rcParams.update({
    "figure.facecolor": "#111111",
    "axes.facecolor":   "#1A1A2E",
    "axes.edgecolor":   "#2A2A4A",
    "axes.labelcolor":  "#DDDDDD",
    "axes.titlecolor":  "#FFFFFF",
    "xtick.color":      "#888899",
    "ytick.color":      "#888899",
    "text.color":       "#CCCCCC",
    "grid.color":       "#2A2A4A",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "legend.facecolor": "#1A1A2E",
    "legend.edgecolor": "#2A2A4A",
    "font.size":        10,
})

# ==========================================================================
# 路径 & 算法配置
# ==========================================================================
BASE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data~")
OUTPUT_DIR = BASE_DIR

ALGO_CONFIG = [
    {"prefix": "training_log_QL_", "label": "Q-Learning", "color": "#4FC3F7", "has_epsilon": True},
    {"prefix": "dqn_log_",         "label": "DQN",        "color": "#81C784", "has_epsilon": True},
    {"prefix": "ppo_log_",         "label": "PPO",        "color": "#FFB74D", "has_epsilon": False},
]

# ==========================================================================
# 工具函数
# ==========================================================================
def smooth(values, window=7):
    if len(values) < window:
        return values
    out = []
    for i in range(len(values)):
        s = max(0, i - window // 2)
        e = min(len(values), i + window // 2 + 1)
        out.append(sum(values[s:e]) / (e - s))
    return out

def _is_valid(path):
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            df = pd.read_csv(path, encoding=enc)
            return not df.empty
        except Exception:
            continue
    return False

def _read_csv(path):
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"无法读取文件: {path}")

# ==========================================================================
# 数据加载
# ==========================================================================
def load_all_logs():
    algo_data = {}
    if not os.path.exists(BASE_DIR):
        return algo_data
    for cfg in ALGO_CONFIG:
        pattern = os.path.join(BASE_DIR, cfg["prefix"] + "*.csv")
        files   = [f for f in sorted(glob.glob(pattern)) if _is_valid(f)]
        frames  = []
        for f in files:
            try:
                df = _read_csv(f)
                seed = os.path.basename(f).replace(cfg["prefix"], "").replace(".csv", "")
                df["seed"] = seed
                frames.append(df)
            except Exception as ex:
                print(f"[警告] 跳过 {f}: {ex}")
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            algo_data[cfg["label"]] = {
                "df":          combined,
                "color":       cfg["color"],
                "has_epsilon": cfg["has_epsilon"],
                "n_seeds":     len(files),
            }
    return algo_data

# ==========================================================================
# 绘图：单算法
# ==========================================================================
def plot_algo(label, info, output_path):
    df      = info["df"]
    color   = info["color"]
    has_eps = info["has_epsilon"]
    n_plots = 4 if has_eps else 3

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    _suptitle(
        fig,
        f"{label}  训练收敛趋势   ({len(df)} 回合 / {info['n_seeds']} 种地图)",
        fontsize=14, fontweight="bold", y=1.05
    )

    episodes = df["episode"].tolist()

    def _panel(ax, col, title_cn, title_en, ylabel_cn, ylabel_en, w=7):
        try:
            raw = [float(v) for v in df[col].tolist()]
            smo = smooth(raw, w)
            ax.plot(episodes, raw, color=color, alpha=0.18, linewidth=1,
                    label="原始数据" if ZH_FONT else "Raw")
            ax.plot(episodes, smo, color=color, linewidth=2.5,
                    label=f"平滑(w={w})" if ZH_FONT else f"Smooth(w={w})")
            _set_title(ax, title_cn if ZH_FONT else title_en, fontsize=11)
            _set_xlabel(ax, "回合数" if ZH_FONT else "Episode")
            _set_ylabel(ax, ylabel_cn if ZH_FONT else ylabel_en)
            ax.grid(True)
            _add_legend(ax, fontsize=8, loc="upper right")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
        except Exception as e:
            print(f"[绘制错误] {title_cn}: {e}")

    _panel(axes[0], "total_reward",
           "回合累计奖励", "Total Reward", "奖励", "Reward")
    _panel(axes[1], "steps",
           "每回合步数",   "Steps / Episode", "步数", "Steps")
    _panel(axes[2], "hit_count",
           "每回合撞墙数", "Wall Hits",      "撞墙次数", "Hits")

    if has_eps:
        ax_e = axes[3]
        ax_e.plot(episodes, df["epsilon"].tolist(), color="#FF4081", linewidth=2.5)
        _set_title(ax_e, "探索率衰减" if ZH_FONT else "Epsilon Decay", fontsize=11)
        _set_xlabel(ax_e, "回合数" if ZH_FONT else "Episode")
        _set_ylabel(ax_e, "探索率" if ZH_FONT else "Epsilon")
        ax_e.grid(True)
        ax_e.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] Saved: {output_path}")

# ==========================================================================
# 绘图：多算法对比
# ==========================================================================
def plot_comparison(algo_data, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    _suptitle(fig,
              "算法横向对比: 奖励 & 效率" if ZH_FONT else "Algorithm Comparison: Reward & Steps",
              fontsize=15, fontweight="bold", y=1.05)

    for label, info in algo_data.items():
        df, color = info["df"], info["color"]
        eps = df["episode"].tolist()
        try:
            rews = smooth([float(v) for v in df["total_reward"].tolist()], 9)
            stps = smooth([float(v) for v in df["steps"].tolist()], 9)
            axes[0].plot(eps, rews, color=color, linewidth=3, label=label, alpha=0.9)
            axes[1].plot(eps, stps, color=color, linewidth=3, label=label, alpha=0.9)
        except Exception as e:
            print(f"[警告] {label} 对比数据解析失败: {e}")

    for ax, title_cn, title_en, ylabel_cn, ylabel_en in [
        (axes[0], "累计奖励对比(平滑)", "Reward Comparison (Smoothed)", "平均奖励", "Reward"),
        (axes[1], "通关效率对比(平滑)", "Steps Comparison (Smoothed)",  "平均步数", "Steps"),
    ]:
        _set_title(ax, title_cn if ZH_FONT else title_en, fontsize=12)
        _set_xlabel(ax, "回合数" if ZH_FONT else "Episode")
        _set_ylabel(ax, ylabel_cn if ZH_FONT else ylabel_en)
        ax.grid(True)
        _add_legend(ax, frameon=True)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] Saved: {output_path}")

# ==========================================================================
# 主入口
# ==========================================================================
if __name__ == "__main__":
    print(f"[Viz] 扫描目录: {BASE_DIR}")

    algo_data = load_all_logs()
    if not algo_data:
        print("[错误] 未找到训练日志，请先在 Unity 中完成训练。")
        raise SystemExit(1)

    print(f"[Viz] 发现算法: {list(algo_data.keys())}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for label, info in algo_data.items():
        slug = label.lower().replace("-", "").replace(" ", "")
        out  = os.path.join(OUTPUT_DIR, f"chart_{slug}.png")
        print(f"\n[Plot] {label} ({len(info['df'])} episodes)...")
        plot_algo(label, info, out)

    if len(algo_data) > 1:
        out_cmp = os.path.join(OUTPUT_DIR, "chart_comparison.png")
        print(f"\n[Plot] 多算法对比图...")
        plot_comparison(algo_data, out_cmp)

    print(f"\n[完成] 图表已保存至: {OUTPUT_DIR}")
