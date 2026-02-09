from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NormalityResult:
    shapiro_p: Optional[float]
    ks_p: Optional[float]
    decision: str
    comment: str


def normality_tests(series: pd.Series, alpha: float = 0.05) -> NormalityResult:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n = len(s)
    if n < 3:
        return NormalityResult(
            shapiro_p=None,
            ks_p=None,
            decision="判定不可",
            comment="データ数が少ないため正規性検定は判定できません（目安: n>=3）。",
        )

    shapiro_p: Optional[float]
    try:
        shapiro_p = float(stats.shapiro(s)[1]) if n <= 5000 else None
    except Exception:
        shapiro_p = None

    # KS検定は母平均・母分散既知が前提なので、標準化してN(0,1)に対して実施
    ks_p: Optional[float]
    try:
        z = (s - s.mean()) / s.std(ddof=1)
        ks_p = float(stats.kstest(z, "norm")[1]) if n >= 3 else None
    except Exception:
        ks_p = None

    # どちらかが有意(=p<alpha)なら「非正規寄り」とする保守的判定
    pvals = [p for p in (shapiro_p, ks_p) if p is not None]
    if not pvals:
        decision = "判定不可"
        comment = "正規性検定を計算できませんでした。"
        return NormalityResult(shapiro_p, ks_p, decision, comment)

    is_normal = all(p >= alpha for p in pvals)
    decision = "正規分布とみなせる" if is_normal else "正規分布とみなせない"

    if is_normal:
        comment = "検定の結果、統計的に有意な非正規性は検出されませんでした（p>=有意水準）。"
    else:
        comment = "検定の結果、正規分布からの逸脱が示唆されます（p<有意水準）。ノンパラメトリック検定も検討してください。"

    return NormalityResult(shapiro_p, ks_p, decision, comment)


@dataclass(frozen=True)
class CompareResult:
    test_name: str
    method: str
    alpha: float
    decision: str
    judgement: str
    # t検定系のみ
    t_stat: Optional[float]
    df: Optional[float]
    t_crit: Optional[float]
    tail: Optional[Literal["two-sided", "one-sided-improvement"]]
    # 非t検定系の参考情報
    statistic: Optional[float]
    p_value: Optional[float]
    details: Dict[str, float]
    comment: str


@dataclass(frozen=True)
class TTestDecision:
    t_stat: float
    df: float
    t_crit: float
    alpha: float
    tail: Literal["two-sided", "one-sided-improvement"]
    decision: str
    judgement: str


def _ttest_equal_var_t_stat_df(before: pd.Series, after: pd.Series) -> tuple[float, float]:
    b = pd.to_numeric(before, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    a = pd.to_numeric(after, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    nb = len(b)
    na = len(a)
    if nb < 2 or na < 2:
        raise ValueError("t検定には各群2点以上のデータが必要です。")

    vb = float(np.var(b, ddof=1))
    va = float(np.var(a, ddof=1))
    df = float(na + nb - 2)
    pooled = ((na - 1) * va + (nb - 1) * vb) / df
    if pooled <= 0:
        raise ValueError("分散が0のためt統計量を算出できません（全データが同一値の可能性）。")

    se = float(np.sqrt(pooled) * np.sqrt(1 / na + 1 / nb))
    if se == 0:
        raise ValueError("標準誤差が0のためt統計量を算出できません。")

    t_stat = (float(a.mean()) - float(b.mean())) / se
    return float(t_stat), float(df)


def _ttest_welch_t_stat_df(before: pd.Series, after: pd.Series) -> tuple[float, float]:
    b = pd.to_numeric(before, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    a = pd.to_numeric(after, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    nb = len(b)
    na = len(a)
    if nb < 2 or na < 2:
        raise ValueError("Welchのt検定には各群2点以上のデータが必要です。")

    vb = float(np.var(b, ddof=1))
    va = float(np.var(a, ddof=1))

    term_a = va / na
    term_b = vb / nb
    se2 = term_a + term_b
    if se2 <= 0:
        raise ValueError("分散が0のためt統計量を算出できません（全データが同一値の可能性）。")

    se = float(np.sqrt(se2))
    t_stat = (float(a.mean()) - float(b.mean())) / se

    denom = (term_a**2) / (na - 1) + (term_b**2) / (nb - 1)
    if denom == 0:
        raise ValueError("自由度(df)を算出できません（分散が0の可能性）。")
    df = (se2**2) / denom
    return float(t_stat), float(df)


def decide_by_t_critical(
    *,
    t_stat: float,
    df: float,
    alpha: float = 0.05,
    tail: Literal["two-sided", "one-sided-improvement"],
) -> TTestDecision:
    """QC現場運用向け: p値ではなく、t値と臨界値の比較で判定する。"""
    if not (df > 0):
        raise ValueError("自由度(df)が不正です。")
    if not (0 < alpha < 1):
        raise ValueError("有意水準alphaは0〜1の範囲で指定してください。")

    if tail == "two-sided":
        t_crit = float(stats.t.ppf(1 - alpha / 2, df))
        significant = abs(float(t_stat)) > t_crit
        decision = "有意差あり（工程変化あり）" if significant else "有意差なし"
        judgement = (
            "自由度: {df:.0f}\n"
            "両側検定（変化検出）\n"
            "臨界値 t({a:.3g},{df:.0f}) = {tcrit:.3g}\n"
            "t算出値 = {tstat:.3g}\n\n"
            "判定:\n"
            "|t算出値| {op} {tcrit:.3g} → {desc}"
        ).format(
            df=float(df),
            a=float(alpha / 2),
            tcrit=float(t_crit),
            tstat=float(t_stat),
            op=">" if significant else "≤",
            desc=decision,
        )
        return TTestDecision(
            t_stat=float(t_stat),
            df=float(df),
            t_crit=float(t_crit),
            alpha=float(alpha),
            tail=tail,
            decision=decision,
            judgement=judgement,
        )

    # one-sided improvement
    t_crit = float(stats.t.ppf(1 - alpha, df))
    improved = float(t_stat) > t_crit
    decision = "統計的に有意な改善" if improved else "改善とは言えない"
    judgement = (
        "自由度: {df:.0f}\n"
        "片側検定（改善方向）\n"
        "臨界値 t({a:.3g},{df:.0f}) = {tcrit:.3g}\n"
        "t算出値 = {tstat:.3g}\n\n"
        "判定:\n"
        "t算出値 {op} {tcrit:.3g} → {desc}"
    ).format(
        df=float(df),
        a=float(alpha),
        tcrit=float(t_crit),
        tstat=float(t_stat),
        op=">" if improved else "≤",
        desc=decision,
    )
    return TTestDecision(
        t_stat=float(t_stat),
        df=float(df),
        t_crit=float(t_crit),
        alpha=float(alpha),
        tail=tail,
        decision=decision,
        judgement=judgement,
    )


def f_test_equal_variance(x: pd.Series, y: pd.Series) -> float:
    """2群の分散が等しいかのF検定（両側）。戻り値: p値"""
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2:
        return float("nan")

    vx = float(np.var(x, ddof=1))
    vy = float(np.var(y, ddof=1))
    if vx == 0 and vy == 0:
        return 1.0
    if vy == 0:
        return 0.0

    f = vx / vy
    dfn = len(x) - 1
    dfd = len(y) - 1
    cdf = stats.f.cdf(f, dfn, dfd)
    p = 2 * min(cdf, 1 - cdf)
    return float(max(min(p, 1.0), 0.0))


def compare_before_after(
    before: pd.Series,
    after: pd.Series,
    alpha: float = 0.05,
    allow_ztest: bool = False,
    known_sigma: Optional[float] = None,
    test_kind: Literal["two-sided", "one-sided-improvement"] = "two-sided",
) -> CompareResult:
    """改善前後の2群比較（対応なし）。条件に応じて検定法を自動選択。"""
    b = pd.to_numeric(before, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    a = pd.to_numeric(after, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    if len(b) < 2 or len(a) < 2:
        raise ValueError("比較には各群2点以上のデータが必要です。")

    # Z検定（母分散既知、n>=30）
    if allow_ztest and known_sigma is not None and known_sigma > 0 and len(b) >= 30 and len(a) >= 30:
        se = known_sigma * np.sqrt(1 / len(b) + 1 / len(a))
        z = (float(a.mean()) - float(b.mean())) / se
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        decision = "統計的に有意" if p < alpha else "有意差なし"
        comment = _interpret_compare(decision, float(b.mean()), float(a.mean()))
        return CompareResult(
            test_name="Z検定（母分散既知・2群）",
            method="ztest",
            alpha=alpha,
            decision=decision,
            judgement="",
            t_stat=None,
            df=None,
            t_crit=None,
            tail=None,
            statistic=float(z),
            p_value=float(p),
            details={"before_mean": float(b.mean()), "after_mean": float(a.mean()), "known_sigma": float(known_sigma)},
            comment=comment,
        )

    norm_b = normality_tests(b, alpha=alpha)
    norm_a = normality_tests(a, alpha=alpha)
    normal_ok = (norm_b.decision == "正規分布とみなせる") and (norm_a.decision == "正規分布とみなせる")

    f_p = f_test_equal_variance(b, a) if normal_ok else float("nan")
    equal_var = bool((not np.isnan(f_p)) and (f_p >= alpha)) if normal_ok else False

    if normal_ok and equal_var:
        t_stat, df = _ttest_equal_var_t_stat_df(b, a)
        d = decide_by_t_critical(t_stat=t_stat, df=df, alpha=float(alpha), tail=test_kind)
        test_name = "t検定（等分散・対応なし）"
        method = "t_equal_var"
        decision = d.decision
        judgement = d.judgement
        comment = _interpret_compare(decision, float(b.mean()), float(a.mean()))
        t_fields = {"t_stat": d.t_stat, "df": d.df, "t_crit": d.t_crit, "tail": d.tail}
        stat = None
        p = None
    elif normal_ok and (not equal_var):
        t_stat, df = _ttest_welch_t_stat_df(b, a)
        d = decide_by_t_critical(t_stat=t_stat, df=df, alpha=float(alpha), tail=test_kind)
        test_name = "Welchのt検定（不等分散・対応なし）"
        method = "t_welch"
        decision = d.decision
        judgement = d.judgement
        comment = _interpret_compare(decision, float(b.mean()), float(a.mean()))
        t_fields = {"t_stat": d.t_stat, "df": d.df, "t_crit": d.t_crit, "tail": d.tail}
        stat = None
        p = None
    else:
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        test_name = "Mann-WhitneyのU検定（ノンパラメトリック）"
        method = "mwu"
        p = float(p)
        stat = float(stat)
        decision = "統計的に有意" if p < alpha else "有意差なし"
        judgement = ""
        comment = _interpret_compare(decision, float(b.mean()), float(a.mean()))
        t_fields = {"t_stat": None, "df": None, "t_crit": None, "tail": None}

    details = {
        "before_n": float(len(b)),
        "after_n": float(len(a)),
        "before_mean": float(b.mean()),
        "after_mean": float(a.mean()),
    }
    if normal_ok:
        details["f_test_p"] = float(f_p) if not np.isnan(f_p) else float("nan")

    # t検定系の付帯情報
    if t_fields["t_stat"] is not None:
        details["t_stat"] = float(t_fields["t_stat"])  # type: ignore[arg-type]
    if t_fields["df"] is not None:
        details["df"] = float(t_fields["df"])  # type: ignore[arg-type]
    if t_fields["t_crit"] is not None:
        details["t_crit"] = float(t_fields["t_crit"])  # type: ignore[arg-type]

    if method.startswith("t_"):
        logger.info(
            "compare: test=%s method=%s t=%.6g df=%.6g tcrit=%.6g decision=%s",
            test_name,
            method,
            float(t_fields["t_stat"]),
            float(t_fields["df"]),
            float(t_fields["t_crit"]),
            decision,
        )
    else:
        logger.info("compare: test=%s method=%s p=%.6g decision=%s", test_name, method, float(p), decision)

    return CompareResult(
        test_name=test_name,
        method=method,
        alpha=alpha,
        decision=decision,
        judgement=judgement,
        t_stat=t_fields["t_stat"],
        df=t_fields["df"],
        t_crit=t_fields["t_crit"],
        tail=t_fields["tail"],
        statistic=stat,
        p_value=p,
        details=details,
        comment=comment,
    )


def _interpret_compare(decision: str, before_mean: float, after_mean: float) -> str:
    direction = "改善後が大きい" if after_mean > before_mean else "改善後が小さい"
    significant = decision in {"統計的に有意", "有意差あり（工程変化あり）", "統計的に有意な改善"}
    if significant:
        return (
            f"{direction}（平均: 改善前={before_mean:.4g}, 改善後={after_mean:.4g}）傾向が、偶然だけでは説明しにくい結果です。"
            "品質改善の施策が効果を持った可能性が高いので、工程条件・要因の変化点を整理し、再現性確認（追加データ/再実験）を推奨します。"
        )
    return (
        f"{direction}（平均: 改善前={before_mean:.4g}, 改善後={after_mean:.4g}）ですが、統計的には有意差が確認できませんでした。"
        "効果が小さい/ばらつきが大きい/サンプル数不足の可能性があるため、追加サンプル取得や分散低減の検討を推奨します。"
    )
