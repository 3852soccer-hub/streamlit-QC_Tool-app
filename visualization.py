from __future__ import annotations

import io
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

# デプロイ先(Linux/コンテナ/Streamlit Cloud等)では日本語フォントが入っていないことが多く、
# その場合に図の日本語が豆腐/空白になります。
# japanize-matplotlib は日本語フォント設定を自動化できるため、存在すれば優先して適用します。
try:
    import japanize_matplotlib  # noqa: F401
except Exception:
    japanize_matplotlib = None  # type: ignore[assignment]

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import capability_engine

sns.set_theme()

# 日本語ラベルが豆腐/空白になる環境があるため、日本語フォント候補を広めに指定
# (OSごとに利用可能なフォントが異なる)
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    # Windows
    "Yu Gothic",
    "Meiryo",
    "MS Gothic",
    "MS PGothic",
    "Arial Unicode MS",
    # macOS
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    # Linux (よく入っている/入れやすい)
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "IPAPGothic",
    "IPAexGothic",
    "TakaoGothic",
    "VL Gothic",
    # 最後の砦
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class PngFigure:
    png_bytes: bytes
    fig: plt.Figure


def fig_to_png(fig: plt.Figure, dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def pareto_chart(df: pd.DataFrame, title: str = "パレート図", lang: str = "ja") -> PngFigure:
    """df: columns=['category','count']"""
    work = df.copy()
    work["count"] = pd.to_numeric(work["count"], errors="coerce").fillna(0)
    work = work.groupby("category", as_index=False)["count"].sum()
    work = work.sort_values("count", ascending=False).reset_index(drop=True)

    total = float(work["count"].sum())
    if total <= 0:
        raise ValueError("件数の合計が0です。")

    work["cum_ratio"] = work["count"].cumsum() / total

    if lang == "en" and title == "パレート図":
        title = "Pareto chart"

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(work["category"].astype(str), work["count"], color=sns.color_palette()[0])
    ax1.set_ylabel("Count" if lang == "en" else "件数")
    ax1.set_title(title)
    ax1.tick_params(axis="x", rotation=45)
    plt.setp(ax1.get_xticklabels(), ha="right")

    ax2 = ax1.twinx()
    ax2.plot(work["category"].astype(str), work["cum_ratio"], color=sns.color_palette()[3], marker="o")
    ax2.axhline(0.8, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("Cumulative ratio" if lang == "en" else "累積比率")
    ax2.set_ylim(0, 1.05)
    # 0〜1 を 0%〜100% 表示にする
    ax2.set_yticks(np.linspace(0, 1.0, 6))
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    png = fig_to_png(fig)
    return PngFigure(png_bytes=png, fig=fig)


def histogram_with_normal(series: pd.Series, bins: int = 20, title: str = "ヒストグラム", lang: str = "ja") -> PngFigure:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        raise ValueError("有効な数値がありません。")

    mu = float(s.mean())
    sigma = float(s.std(ddof=1)) if len(s) >= 2 else 0.0

    if lang == "en" and title == "ヒストグラム":
        title = "Histogram"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(s, bins=bins, density=True, alpha=0.7, color=sns.color_palette()[0], edgecolor="white")

    x = np.linspace(float(s.min()), float(s.max()), 200)
    if sigma > 0:
        y = stats.norm.pdf(x, loc=mu, scale=sigma)
        ax.plot(
            x,
            y,
            color=sns.color_palette()[3],
            linewidth=2,
            label="Normal (estimated)" if lang == "en" else "正規分布(推定)",
        )

    ax.set_title(title)
    ax.set_xlabel("Value" if lang == "en" else "値")
    ax.set_ylabel("Density" if lang == "en" else "密度")
    if sigma > 0:
        ax.legend()

    png = fig_to_png(fig)
    return PngFigure(png_bytes=png, fig=fig)


def before_after_distribution_plot(
    before: pd.Series,
    after: pd.Series,
    bins: int = 20,
    title: str = "改善前後の分布比較",
    lang: str = "ja",
) -> PngFigure:
    """改善前後の分布を比較する（対応なし前提）。

    - 重ねヒストグラム（密度）
    - 平均の縦線
    """
    b = pd.to_numeric(before, errors="coerce").dropna()
    a = pd.to_numeric(after, errors="coerce").dropna()
    if b.empty or a.empty:
        raise ValueError("改善前・改善後ともに有効な数値が必要です。")

    if lang == "en" and title == "改善前後の分布比較":
        title = "Before/After distribution"

    fig, ax = plt.subplots(figsize=(10, 4.8))

    # densityで比較
    label_before = f"Before (n={len(b)})" if lang == "en" else f"改善前 (n={len(b)})"
    label_after = f"After (n={len(a)})" if lang == "en" else f"改善後 (n={len(a)})"
    ax.hist(b, bins=bins, density=True, alpha=0.55, color=sns.color_palette()[0], label=label_before)
    ax.hist(a, bins=bins, density=True, alpha=0.55, color=sns.color_palette()[3], label=label_after)

    # 平均線
    ax.axvline(float(b.mean()), color=sns.color_palette()[0], linestyle="--", linewidth=1.5)
    ax.axvline(float(a.mean()), color=sns.color_palette()[3], linestyle="--", linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel("Value" if lang == "en" else "値")
    ax.set_ylabel("Density" if lang == "en" else "密度")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8)
    ax.legend(loc="upper right")

    png = fig_to_png(fig)
    return PngFigure(png_bytes=png, fig=fig)


def doe_main_effects_plot(main_effects_long: pd.DataFrame, title: str = "DOE 主効果プロット", lang: str = "ja") -> PngFigure:
    """main_effects_long: columns=['factor','level','mean','n']"""
    req = {"factor", "level", "mean", "n"}
    if not req.issubset(set(main_effects_long.columns)):
        raise ValueError("主効果プロット用の列が不足しています。")

    factors = list(pd.unique(main_effects_long["factor"]))
    if not factors:
        raise ValueError("主効果データが空です。")

    if lang == "en" and title == "DOE 主効果プロット":
        title = "DOE main effects"

    ncols = 2
    nrows = int(np.ceil(len(factors) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for idx, f in enumerate(factors):
        ax = axes[idx]
        g = main_effects_long[main_effects_long["factor"] == f].copy()

        # levelのソート（数値っぽければ数値順）
        try:
            level_num = pd.to_numeric(g["level"], errors="coerce")
            if level_num.notna().all():
                g = g.assign(_level_num=level_num).sort_values("_level_num")
            else:
                g = g.sort_values("level")
        except Exception:
            g = g.sort_values("level")

        ax.plot(g["level"].astype(str), g["mean"], marker="o", color=sns.color_palette()[0])
        ax.set_title(str(f))
        ax.set_xlabel("Level" if lang == "en" else "水準")
        ax.set_ylabel("Mean (response)" if lang == "en" else "平均(応答)")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.8)

    # 余ったaxesを消す
    for j in range(len(factors), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    png = fig_to_png(fig)
    return PngFigure(png_bytes=png, fig=fig)


def xr_control_chart_plot(xr: capability_engine.XRChartResult, title: str = "X-R管理図", lang: str = "ja") -> PngFigure:
    """Xbar管理図とR管理図を2段で描画する。"""
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 7), sharex=True)

    idx = np.arange(1, len(xr.xbar) + 1)

    # Xbar chart
    ax1.plot(idx, xr.xbar.values, marker="o", color=sns.color_palette()[0], linewidth=1.5)
    ax1.axhline(xr.cl_x, color="black", linewidth=1, label="CL")
    ax1.axhline(xr.ucl_x, color="red", linestyle="--", linewidth=1, label="UCL")
    ax1.axhline(xr.lcl_x, color="red", linestyle="--", linewidth=1, label="LCL")
    ax1.set_ylabel("X\u0305 (Mean)" if lang == "en" else "\u0305X")
    if lang == "en" and title == "X-R管理図":
        title = "X-R control chart"
    subtitle = f" (subgroup n={xr.subgroup_size})" if lang == "en" else f"（サブグループn={xr.subgroup_size}）"
    ax1.set_title(f"{title}{subtitle}")
    ax1.grid(True, axis="y", linestyle=":", linewidth=0.8)
    ax1.legend(loc="upper right")

    # 管理限界値の数値注記（図示要件）
    ax1.text(
        0.995,
        0.02,
        f"UCL={xr.ucl_x:.4g}\nCL={xr.cl_x:.4g}\nLCL={xr.lcl_x:.4g}",
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )

    # R chart
    ax2.plot(idx, xr.r.values, marker="o", color=sns.color_palette()[3], linewidth=1.5)
    ax2.axhline(xr.cl_r, color="black", linewidth=1, label="CL")
    ax2.axhline(xr.ucl_r, color="red", linestyle="--", linewidth=1, label="UCL")
    ax2.axhline(xr.lcl_r, color="red", linestyle="--", linewidth=1, label="LCL")
    ax2.set_xlabel("Subgroup #" if lang == "en" else "サブグループ番号")
    ax2.set_ylabel("R")
    ax2.grid(True, axis="y", linestyle=":", linewidth=0.8)
    ax2.legend(loc="upper right")

    ax2.text(
        0.995,
        0.02,
        f"UCL={xr.ucl_r:.4g}\nCL={xr.cl_r:.4g}\nLCL={xr.lcl_r:.4g}",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )

    fig.tight_layout()
    png = fig_to_png(fig)
    return PngFigure(png_bytes=png, fig=fig)


def ppk_distribution_plot(
    series: pd.Series,
    lsl: float,
    usl: float,
    mean: float,
    sigma_overall: float,
    bins: int = 20,
    title: str = "Ppk用 分布図（overall）",
    lang: str = "ja",
) -> PngFigure:
    """Ppk(長期)評価向けに、ヒストグラム+正規曲線+規格線を表示する。"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        raise ValueError("有効な数値がありません。")

    if lang == "en" and title == "Ppk用 分布図（overall）":
        title = "Ppk distribution (overall)"

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.hist(s, bins=bins, density=True, alpha=0.7, color=sns.color_palette()[0], edgecolor="white")

    x_min = float(min(s.min(), lsl))
    x_max = float(max(s.max(), usl))
    x = np.linspace(x_min, x_max, 300)
    if sigma_overall > 0:
        y = stats.norm.pdf(x, loc=mean, scale=sigma_overall)
        ax.plot(
            x,
            y,
            color=sns.color_palette()[3],
            linewidth=2,
            label="Normal (σ_overall)" if lang == "en" else "正規分布(σ_overall)",
        )

    ax.axvline(lsl, color="red", linestyle="--", linewidth=1.5, label="LSL")
    ax.axvline(usl, color="red", linestyle="--", linewidth=1.5, label="USL")
    ax.axvline(mean, color="black", linewidth=1.2, label="Mean" if lang == "en" else "平均")

    ax.set_title(title)
    ax.set_xlabel("Value" if lang == "en" else "値")
    ax.set_ylabel("Density" if lang == "en" else "密度")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8)
    ax.legend(loc="upper right")

    png = fig_to_png(fig)
    return PngFigure(png_bytes=png, fig=fig)
