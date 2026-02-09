from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# X-R管理図で用いる定数 d2（サブグループサイズ n に依存）
# 出典: 標準的なSPC管理図定数表（AIAG/ISO系で一般的に使われる値）
# 注: d2 は「サブグループ内のレンジ R の期待値」と σ の関係 E[R] = d2 * σ を与える。
_D2_TABLE: dict[int, float] = {
    2: 1.128,
    3: 1.693,
    4: 2.059,
    5: 2.326,
    6: 2.534,
    7: 2.704,
    8: 2.847,
    9: 2.970,
    10: 3.078,
    11: 3.173,
    12: 3.258,
    13: 3.336,
    14: 3.407,
    15: 3.472,
    16: 3.532,
    17: 3.588,
    18: 3.640,
    19: 3.689,
    20: 3.735,
    21: 3.778,
    22: 3.819,
    23: 3.858,
    24: 3.895,
    25: 3.931,
}

# X-R管理図の管理限界に用いる定数（サブグループサイズ n に依存）
# - Xbar 管理図: UCL/LCL = Xbarbar ± A2 * Rbar
# - R 管理図: UCL = D4 * Rbar, LCL = D3 * Rbar
_A2_TABLE: dict[int, float] = {
    2: 1.880,
    3: 1.023,
    4: 0.729,
    5: 0.577,
    6: 0.483,
    7: 0.419,
    8: 0.373,
    9: 0.337,
    10: 0.308,
    11: 0.285,
    12: 0.266,
    13: 0.249,
    14: 0.235,
    15: 0.223,
    16: 0.212,
    17: 0.203,
    18: 0.194,
    19: 0.187,
    20: 0.180,
    21: 0.173,
    22: 0.167,
    23: 0.162,
    24: 0.157,
    25: 0.153,
}

_D3_TABLE: dict[int, float] = {
    2: 0.000,
    3: 0.000,
    4: 0.000,
    5: 0.000,
    6: 0.000,
    7: 0.076,
    8: 0.136,
    9: 0.184,
    10: 0.223,
    11: 0.256,
    12: 0.283,
    13: 0.307,
    14: 0.328,
    15: 0.347,
    16: 0.363,
    17: 0.378,
    18: 0.391,
    19: 0.403,
    20: 0.415,
    21: 0.425,
    22: 0.434,
    23: 0.443,
    24: 0.451,
    25: 0.459,
}

_D4_TABLE: dict[int, float] = {
    2: 3.267,
    3: 2.574,
    4: 2.282,
    5: 2.114,
    6: 2.004,
    7: 1.924,
    8: 1.864,
    9: 1.816,
    10: 1.777,
    11: 1.744,
    12: 1.717,
    13: 1.693,
    14: 1.672,
    15: 1.653,
    16: 1.637,
    17: 1.622,
    18: 1.608,
    19: 1.597,
    20: 1.585,
    21: 1.575,
    22: 1.566,
    23: 1.557,
    24: 1.548,
    25: 1.541,
}


@dataclass(frozen=True)
class CapabilityResult:
    n: int
    mean: float
    # overall（長期）: 観測全体のばらつき
    std_overall_s: float
    sigma_overall: float
    sigma_overall_ci_low: Optional[float]
    sigma_overall_ci_high: Optional[float]
    # within（短期）: サブグループ内ばらつき（X-R管理図の考え方）
    subgroup_size: Optional[int]
    rbar: Optional[float]
    sigma_within: float
    cp: Optional[float]
    cpk: Optional[float]
    pp: Optional[float]
    ppk: Optional[float]
    # SPC（管理状態）
    out_of_control: Optional[bool]
    status: str
    comment: str
    level: str


@dataclass(frozen=True)
class XRChartResult:
    """X-R管理図用の統計量と管理限界。"""

    subgroup_size: int
    xbar: pd.Series
    r: pd.Series
    xbarbar: float
    rbar: float
    d2: float
    a2: float
    d3: float
    d4: float
    ucl_x: float
    cl_x: float
    lcl_x: float
    ucl_r: float
    cl_r: float
    lcl_r: float


def capability_indices(
    series: pd.Series,
    lsl: float,
    usl: float,
    *,
    alpha: float = 0.05,
    subgroup_size: Optional[int] = None,
    subgroup_df: Optional[pd.DataFrame] = None,
) -> CapabilityResult:
    """工程能力指数を算出する。

    概念の分離（重要）:
    - Cpk/Cp: 短期（within-process）= サブグループ内変動。X-R管理図の σ_within = R̄ / d2 で推定。
    - Ppk/Pp: 長期（overall）= 観測全体の変動。χ²分布に基づく σ_overall を推定（点推定 + CI）。

    subgroup_size が未指定/不適切な場合は σ_within をサンプル標準偏差でフォールバックする。
    （この場合でも σ_overall とは別計算として扱い、同一σを使い回さない。）
    """
    if usl <= lsl:
        raise ValueError("USLはLSLより大きい必要があります。")

    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n = len(s)
    if n < 2:
        raise ValueError("工程能力指数の算出には2点以上のデータが必要です。")

    mean = float(s.mean())
    std_overall_s = float(s.std(ddof=1))

    # まず overall（長期）σを計算（χ²ベース）
    sigma_overall, sigma_overall_ci_low, sigma_overall_ci_high = _estimate_sigma_overall_chi_square(
        std_s=std_overall_s,
        n=n,
        alpha=alpha,
    )

    # within（短期）σ: サブグループがあれば R̄/d2、なければフォールバック
    sigma_within, rbar, subgroup_used, within_note = _estimate_sigma_within_xr(
        s,
        subgroup_size=subgroup_size,
        subgroup_df=subgroup_df,
    )

    # 管理状態（X-R管理図の管理限界逸脱の有無）
    out_of_control: Optional[bool] = None
    if subgroup_df is not None and subgroup_size is not None:
        try:
            xr = xr_chart_from_subgroups(subgroup_df, subgroup_size=int(subgroup_size))
            out_of_control = detect_out_of_control(xr)
        except Exception:
            out_of_control = None

    if sigma_within <= 0 or sigma_overall <= 0:
        # ばらつきゼロ/不安定
        cp = float("inf") if sigma_within <= 0 else (usl - lsl) / (6 * sigma_within)
        cpk = float("inf") if (sigma_within <= 0 and (lsl <= mean <= usl)) else 0.0
        pp = float("inf") if sigma_overall <= 0 else (usl - lsl) / (6 * sigma_overall)
        ppk = float("inf") if (sigma_overall <= 0 and (lsl <= mean <= usl)) else 0.0
        level = "判定不可"
        comment = (
            "ばらつき推定が0または不安定なため、指数が無限大/不安定になります。"
            "測定分解能・データ収集方法・サブグループの切り方を確認してください。"
        )
        return CapabilityResult(
            n=n,
            mean=mean,
            std_overall_s=std_overall_s,
            sigma_overall=sigma_overall,
            sigma_overall_ci_low=sigma_overall_ci_low,
            sigma_overall_ci_high=sigma_overall_ci_high,
            subgroup_size=subgroup_used,
            rbar=rbar,
            sigma_within=sigma_within,
            cp=cp,
            cpk=cpk,
            pp=pp,
            ppk=ppk,
            out_of_control=out_of_control,
            status="判定不可",
            comment=comment,
            level=level,
        )

    # Cp/Cpk: within(短期)
    cp = (usl - lsl) / (6 * sigma_within)
    cpk = min((usl - mean) / (3 * sigma_within), (mean - lsl) / (3 * sigma_within))

    # Pp/Ppk: overall(長期)
    pp = (usl - lsl) / (6 * sigma_overall)
    ppk = min((usl - mean) / (3 * sigma_overall), (mean - lsl) / (3 * sigma_overall))

    interp = interpretation_engine(out_of_control=out_of_control, cpk=float(cpk), ppk=float(ppk))
    comment = (
        interp.message
        + "\n\n"
        + "指標の意味: Cpkは短期（サブグループ内）ばらつき、Ppkは長期（全体）ばらつきで評価します。"
        + ("\n" + within_note if within_note else "")
    )

    logger.info(
        "capability: n=%s mean=%.6g sigma_within=%.6g sigma_overall=%.6g cpk=%.6g ppk=%.6g",
        n,
        mean,
        sigma_within,
        sigma_overall,
        cpk,
        ppk,
    )

    return CapabilityResult(
        n=n,
        mean=mean,
        std_overall_s=std_overall_s,
        sigma_overall=sigma_overall,
        sigma_overall_ci_low=sigma_overall_ci_low,
        sigma_overall_ci_high=sigma_overall_ci_high,
        subgroup_size=subgroup_used,
        rbar=rbar,
        sigma_within=sigma_within,
        cp=float(cp),
        cpk=float(cpk),
        pp=float(pp),
        ppk=float(ppk),
        out_of_control=out_of_control,
        status=interp.status,
        comment=comment,
        level=interp.level,
    )


def _estimate_sigma_within_xr(series: pd.Series, subgroup_size: Optional[int]) -> tuple[float, Optional[float], Optional[int], str]:
    """X-R管理図の考え方で短期σを推定する。

    - 入力データを時系列順にサブグループに分割（連続ブロック）
    - 各サブグループの R = max-min を計算
    - R̄ を求め、σ_within = R̄ / d2

    subgroup_size が未指定/テーブル外/サブグループ数不足の場合はフォールバック。
    """
    raise NotImplementedError


def _estimate_sigma_within_xr(
    series: pd.Series,
    subgroup_size: Optional[int],
    subgroup_df: Optional[pd.DataFrame],
) -> tuple[float, Optional[float], Optional[int], str]:
    """X-R管理図の考え方で短期σを推定する。

    優先順位:
    1) subgroup_df が与えられれば、それをサブグループとしてR̄/d2を計算（推奨）。
    2) それ以外は入力系列を時系列順に n 個ずつ連続ブロック分割してR̄/d2を計算。
    3) サブグループ条件を満たせない場合は標準偏差でフォールバック。
    """
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return 0.0, None, None, ""

    # まずは、ユーザー入力のサブグループ（wide形式）を使う
    if subgroup_df is not None and not subgroup_df.empty:
        m = int(subgroup_size) if subgroup_size is not None else None
        if m is None:
            # df列数から推定
            m = int(subgroup_df.shape[1])

        try:
            xr = xr_chart_from_subgroups(subgroup_df, subgroup_size=m)
            d2 = _D2_TABLE[m]
            sigma_within = float(xr.rbar / d2) if d2 > 0 else 0.0
            return sigma_within, float(xr.rbar), int(m), ""
        except Exception as e:
            sigma_fb = float(s.std(ddof=1))
            note = f"サブグループデータからσ_within(R̄/d2)を算出できなかったため、標準偏差で近似しました。理由: {e}"
            return sigma_fb, None, m, note

    # サブグループサイズが未指定ならフォールバック
    if subgroup_size is None:
        sigma_fb = float(s.std(ddof=1))
        note = "サブグループが未指定のため、Cpkのσ_withinはサンプル標準偏差で近似しました（X-R管理図ベース推定にはサブグループが必要）。"
        return sigma_fb, None, None, note

    m = int(subgroup_size)
    if m < 2:
        sigma_fb = float(s.std(ddof=1))
        note = "サブグループサイズが2未満のため、Cpkのσ_withinはサンプル標準偏差で近似しました。"
        return sigma_fb, None, None, note

    if m not in _D2_TABLE:
        sigma_fb = float(s.std(ddof=1))
        note = f"サブグループサイズ n={m} はd2テーブル未対応のため、Cpkのσ_withinはサンプル標準偏差で近似しました（対応範囲: {min(_D2_TABLE)}..{max(_D2_TABLE)}）。"
        return sigma_fb, None, None, note

    # 連続ブロックで分割し、最後の端数は捨てる（SPCではサブグループサイズ固定が前提）
    k = int(len(s) // m)
    if k < 2:
        sigma_fb = float(s.std(ddof=1))
        note = "サブグループ数が2未満のため、Cpkのσ_withinはサンプル標準偏差で近似しました（最低2サブグループ推奨）。"
        return sigma_fb, None, m, note

    trimmed = s.iloc[: k * m]
    values = trimmed.to_numpy().reshape(k, m)
    ranges = values.max(axis=1) - values.min(axis=1)
    rbar = float(np.mean(ranges))
    d2 = _D2_TABLE[m]
    sigma_within = float(rbar / d2) if d2 > 0 else 0.0

    return sigma_within, rbar, m, ""


def xr_chart_from_subgroups(subgroup_df: pd.DataFrame, subgroup_size: int) -> XRChartResult:
    """wide形式のサブグループデータ（行=サブグループ、列=サンプル）からX-R管理図統計量を計算する。"""
    m = int(subgroup_size)
    if m < 2:
        raise ValueError("サブグループサイズは2以上が必要です。")
    if m not in _D2_TABLE or m not in _A2_TABLE or m not in _D3_TABLE or m not in _D4_TABLE:
        raise ValueError(f"サブグループサイズ n={m} の管理図定数が未対応です（対応範囲: {min(_D2_TABLE)}..{max(_D2_TABLE)}）。")

    if subgroup_df.shape[1] != m:
        raise ValueError(f"サブグループ列数({subgroup_df.shape[1]})と指定n({m})が一致しません。")

    data = subgroup_df.apply(pd.to_numeric, errors="coerce")
    # すべての行がm個の値を持つこと（管理図定数の前提）
    per_row_n = data.notna().sum(axis=1)
    if (per_row_n != m).any():
        bad = int((per_row_n != m).sum())
        raise ValueError(f"欠損があるサブグループが {bad} 行あります。各行に{m}個すべて入力してください。")

    if data.shape[0] < 2:
        raise ValueError("X-R管理図には2サブグループ以上が推奨です。")

    xbar = data.mean(axis=1)
    r = data.max(axis=1) - data.min(axis=1)
    xbarbar = float(xbar.mean())
    rbar = float(r.mean())

    d2 = _D2_TABLE[m]
    a2 = _A2_TABLE[m]
    d3 = _D3_TABLE[m]
    d4 = _D4_TABLE[m]

    ucl_x = xbarbar + a2 * rbar
    lcl_x = xbarbar - a2 * rbar
    ucl_r = d4 * rbar
    lcl_r = d3 * rbar

    return XRChartResult(
        subgroup_size=m,
        xbar=xbar,
        r=r,
        xbarbar=xbarbar,
        rbar=rbar,
        d2=float(d2),
        a2=a2,
        d3=d3,
        d4=d4,
        ucl_x=float(ucl_x),
        cl_x=float(xbarbar),
        lcl_x=float(lcl_x),
        ucl_r=float(ucl_r),
        cl_r=float(rbar),
        lcl_r=float(lcl_r),
    )


def _estimate_sigma_overall_chi_square(std_s: float, n: int, alpha: float) -> tuple[float, Optional[float], Optional[float]]:
    """χ²分布に基づいて長期（overall）σを推定。

    理論:
    - (n-1) * s^2 / σ^2 ~ χ²(df=n-1)
    - これを用いて σ の信頼区間（および代表値）を得る。

    実務上の「σ_overall」の点推定は s を使うことが多いが、
    本実装ではχ²分布を明示的に使うため、中央値χ²を用いた代表値を返す。
    """
    if n < 2 or std_s <= 0:
        return 0.0, None, None

    df = n - 1
    s2 = std_s**2

    # 代表値: χ²の中央値を用いて σ を推定（χ²を使うという要件に合わせる）
    chi2_med = stats.chi2.ppf(0.5, df)
    sigma_overall = float(np.sqrt(df * s2 / chi2_med)) if chi2_med > 0 else float(std_s)

    # CI: 既存同様
    chi2_low = stats.chi2.ppf(alpha / 2, df)
    chi2_high = stats.chi2.ppf(1 - alpha / 2, df)
    sigma_ci_low = float(np.sqrt(df * s2 / chi2_high)) if chi2_high > 0 else None
    sigma_ci_high = float(np.sqrt(df * s2 / chi2_low)) if chi2_low > 0 else None

    return sigma_overall, sigma_ci_low, sigma_ci_high


def _judge_cpk(cpk: float) -> tuple[str, str]:
    if cpk >= 2.0:
        return "シックスシグマ水準", "Cpkが2.0以上で、非常に高い工程能力が期待できます。管理の維持と異常検知の仕組み化が有効です。"
    if cpk >= 1.67:
        return "優秀", "Cpkが1.67以上で、工程能力は優秀です。現状の条件を標準化し、変動要因を固定化してください。"
    if cpk >= 1.33:
        return "良好", "Cpkが1.33以上で、工程能力は良好です。更なるばらつき低減や中心合わせで余裕度を増やせます。"
    if cpk >= 1.0:
        return "要改善", "Cpkが1.0以上ですが余裕が小さく、不良が発生しやすい可能性があります。中心合わせ・分散低減を検討してください。"
    return "不十分", "Cpkが1.0未満で、規格外が発生しやすい状態です。工程の見直し（条件最適化/設備/材料/測定系）を優先してください。"


@dataclass(frozen=True)
class CapabilityInterpretation:
    """SPC + Six Sigmaの考え方に基づく、ルールベースの解釈結果。"""

    status: str
    level: str
    message: str


def interpretation_engine(*, out_of_control: Optional[bool], cpk: float, ppk: float) -> CapabilityInterpretation:
    """解釈の優先順位（必須）

    1. 管理状態（SPC）: out_of_control が True なら能力評価は無効
    2. Ppk（長期性能）: Ppk < 1.33 なら長期性能不足（Cpkが高くても優秀とはしない）
    3. Cpk（短期能力）: 安定かつPpkが閾値以上のときのみ、高能力を評価
    """

    # 1) 管理図優先
    if out_of_control is True:
        return CapabilityInterpretation(
            status="UNSTABLE PROCESS",
            level="統計的不安定",
            message=(
                "管理図上で管理限界線逸脱が検出されており、工程は統計的に管理状態ではありません。\n"
                "この状態ではCpk/Ppkによる工程能力評価は無効です。\n"
                "まず工程の安定化（異常原因の特定・除去・標準化）を優先してください。"
            ),
        )

    # out_of_control が不明（サブグループなし等）の場合は、あくまで参考である旨を残しつつ判定
    stability_note = "" if out_of_control is False else "（注: サブグループがないため管理状態の確認は未実施です。結果は参考値です。）\n"

    # 2) 長期性能優先
    if ppk < 1.33:
        return CapabilityInterpretation(
            status="LOW LONG-TERM PERFORMANCE",
            level="長期性能不足",
            message=(
                stability_note
                + "長期性能指標（Ppk）が基準を下回っており、工程の長期的な品質保証能力が不足しています。\n"
                "短期能力（Cpk）が高くても、工程の再現性・安定性に問題があります。"
            ),
        )

    # 3) 高能力（安定 + Ppk閾値クリアが前提）
    if cpk >= 1.67 and ppk >= 1.33:
        return CapabilityInterpretation(
            status="HIGH CAPABILITY PROCESS",
            level="高能力（有効）",
            message=(
                stability_note
                + "工程は管理状態にあり、短期能力（Cpk）・長期性能（Ppk）ともに良好です。\n"
                "現条件の標準化と維持管理を推奨します。"
            ),
        )

    return CapabilityInterpretation(
        status="LIMITED CAPABILITY",
        level="能力限定",
        message=(
            stability_note
            + "工程能力/性能指標は一定の水準にありますが、改善余地が残ります。\n"
            "ばらつき低減（σ低減）や中心合わせ（平均の規格中心化）を検討してください。"
        ),
    )


def detect_out_of_control(xr: XRChartResult) -> bool:
    """Xbar/Rのいずれかで管理限界線逸脱が1点でもあれば out-of-control とする。"""
    x = xr.xbar
    r = xr.r
    vio_x = ((x > xr.ucl_x) | (x < xr.lcl_x)).any()
    vio_r = ((r > xr.ucl_r) | (r < xr.lcl_r)).any()
    return bool(vio_x or vio_r)
