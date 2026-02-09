from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class DoeResult:
    method: str
    design: pd.DataFrame
    runs: int
    full_factorial_runs: int
    efficiency_ratio: float
    reduction_ratio: float
    comment: str


@dataclass(frozen=True)
class DoeAnalysisResult:
    response_col: str
    objective: Literal["maximize", "minimize"]
    cleaned_n: int
    main_effects_long: pd.DataFrame
    factor_summary: pd.DataFrame
    recommendation: str


def full_factorial(factor_levels: List[int]) -> DoeResult:
    factors = [f"F{i+1}" for i in range(len(factor_levels))]
    levels = [list(range(1, int(lv) + 1)) for lv in factor_levels]
    rows = list(itertools.product(*levels))
    df = pd.DataFrame(rows, columns=factors)
    full_runs = len(rows)
    return DoeResult(
        method="フルファクタリアル",
        design=df,
        runs=full_runs,
        full_factorial_runs=full_runs,
        efficiency_ratio=1.0,
        reduction_ratio=0.0,
        comment="全組合せを網羅するため解析しやすい一方、因子/水準が増えると実験回数が急増します。",
    )


def taguchi_orthogonal_array(factor_levels: List[int]) -> DoeResult:
    """簡易タグチ：代表的な直交表(L4/L8/L9/L16)の範囲で自動選択。

    - 2水準: L4(2^3), L8(2^7), L16(2^15)
    - 3水準: L9(3^4)

    それ以外はフルファクタリアルへフォールバック。
    """
    if not factor_levels:
        raise ValueError("因子数が0です。")

    k = len(factor_levels)
    if any(lv < 2 for lv in factor_levels):
        raise ValueError("水準数は2以上にしてください。")

    all_two = all(lv == 2 for lv in factor_levels)
    all_three = all(lv == 3 for lv in factor_levels)

    factors = [f"F{i+1}" for i in range(k)]

    if all_two and k <= 3:
        # L4: 4 runs, 3 cols
        base = [
            [1, 1, 1],
            [1, 2, 2],
            [2, 1, 2],
            [2, 2, 1],
        ]
        df = pd.DataFrame([row[:k] for row in base], columns=factors)
        return _wrap("タグチ(L4)", df, factor_levels)

    if all_two and k <= 7:
        # L8: 8 runs, 7 cols (標準的な符号表)
        base = [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 2, 2, 2, 2],
            [1, 2, 2, 1, 1, 2, 2],
            [1, 2, 2, 2, 2, 1, 1],
            [2, 1, 2, 1, 2, 1, 2],
            [2, 1, 2, 2, 1, 2, 1],
            [2, 2, 1, 1, 2, 2, 1],
            [2, 2, 1, 2, 1, 1, 2],
        ]
        df = pd.DataFrame([row[:k] for row in base], columns=factors)
        return _wrap("タグチ(L8)", df, factor_levels)

    if all_three and k <= 4:
        # L9: 9 runs, 4 cols
        base = [
            [1, 1, 1, 1],
            [1, 2, 2, 2],
            [1, 3, 3, 3],
            [2, 1, 2, 3],
            [2, 2, 3, 1],
            [2, 3, 1, 2],
            [3, 1, 3, 2],
            [3, 2, 1, 3],
            [3, 3, 2, 1],
        ]
        df = pd.DataFrame([row[:k] for row in base], columns=factors)
        return _wrap("タグチ(L9)", df, factor_levels)

    if all_two and k <= 15:
        # L16: 16 runs, 15 cols（簡易：生成的に2水準の直交配列を作る）
        # ここでは 4因子のフル(16)を基底に、列を拡張する最小実装。
        cols = []
        for bits in itertools.product([1, 2], repeat=4):
            cols.append(list(bits))
        core = pd.DataFrame(cols, columns=["A", "B", "C", "D"])  # 16x4
        # 15列を作る（A,B,C,D とその交互作用相当を近似）
        design = pd.DataFrame({
            "c1": core["A"],
            "c2": core["B"],
            "c3": core["C"],
            "c4": core["D"],
            "c5": _xor2(core["A"], core["B"]),
            "c6": _xor2(core["A"], core["C"]),
            "c7": _xor2(core["A"], core["D"]),
            "c8": _xor2(core["B"], core["C"]),
            "c9": _xor2(core["B"], core["D"]),
            "c10": _xor2(core["C"], core["D"]),
            "c11": _xor3(core["A"], core["B"], core["C"]),
            "c12": _xor3(core["A"], core["B"], core["D"]),
            "c13": _xor3(core["A"], core["C"], core["D"]),
            "c14": _xor3(core["B"], core["C"], core["D"]),
            "c15": _xor4(core["A"], core["B"], core["C"], core["D"]),
        })
        df = design.iloc[:, :k].copy()
        df.columns = factors
        return _wrap("タグチ(L16相当)", df, factor_levels)

    # フォールバック
    ff = full_factorial(factor_levels)
    return DoeResult(
        method=ff.method,
        design=ff.design,
        runs=ff.runs,
        full_factorial_runs=ff.full_factorial_runs,
        efficiency_ratio=ff.efficiency_ratio,
        reduction_ratio=ff.reduction_ratio,
        comment="指定された因子/水準では代表的な直交表を自動適用できないため、フルファクタリアル設計を生成しました。",
    )


def _wrap(method: str, df: pd.DataFrame, factor_levels: List[int]) -> DoeResult:
    full = 1
    for lv in factor_levels:
        full *= int(lv)

    runs = len(df)
    eff = full / runs if runs else float("inf")
    red = 1 - (runs / full) if full else 0.0
    comment = "直交表により実験回数を削減しつつ、主効果の推定に向いた設計です（交互作用は別途検討が必要な場合があります）。"

    return DoeResult(
        method=method,
        design=df,
        runs=runs,
        full_factorial_runs=full,
        efficiency_ratio=float(eff),
        reduction_ratio=float(red),
        comment=comment,
    )


def _xor2(a, b):
    # 2水準(1/2)をXOR的に合成
    return ((a == b).map({True: 1, False: 2})).astype(int)


def _xor3(a, b, c):
    return _xor2(_xor2(a, b), c)


def _xor4(a, b, c, d):
    return _xor2(_xor3(a, b, c), d)


def analyze_doe_results(
    df: pd.DataFrame,
    response_col: str,
    objective: Literal["maximize", "minimize"] = "maximize",
) -> DoeAnalysisResult:
    """実験計画表 + 応答(実験値)から、主効果の要約を行う（最小実装）。

    - 要求列: 因子列（F1..など） + 応答列
    - 出力: 各因子×水準の平均、因子ごとの効果幅、推奨水準

    注意: 交互作用や回帰モデルは扱わない（直交表の主効果用途を優先）。
    """
    if df is None or df.empty:
        raise ValueError("DOE結果のデータが空です。")
    if response_col not in df.columns:
        raise ValueError(f"応答列 '{response_col}' が見つかりません。")
    if objective not in ("maximize", "minimize"):
        raise ValueError("objective は maximize / minimize のいずれかです。")

    # 因子列: 応答以外で、数値/カテゴリどちらも許容
    factor_cols = [c for c in df.columns if c != response_col]
    if not factor_cols:
        raise ValueError("因子列が見つかりません（応答列以外の列が必要です）。")

    work = df.copy()
    work[response_col] = pd.to_numeric(work[response_col], errors="coerce")
    work = work.dropna(subset=[response_col])
    if work.empty:
        raise ValueError("応答列がすべて欠損/非数値です。")

    # 主効果（long）: factor, level, mean, n
    rows = []
    for f in factor_cols:
        g = (
            work.groupby(f, dropna=False)[response_col]
            .agg([("mean", "mean"), ("n", "count")])
            .reset_index()
            .rename(columns={f: "level"})
        )
        g.insert(0, "factor", f)
        rows.append(g)
    main_effects_long = pd.concat(rows, ignore_index=True)

    # 因子サマリ: 効果幅と推奨水準
    summaries = []
    for f in factor_cols:
        g = main_effects_long[main_effects_long["factor"] == f].copy()
        if g.empty:
            continue

        best_row = g.loc[g["mean"].idxmax()] if objective == "maximize" else g.loc[g["mean"].idxmin()]
        effect_range = float(g["mean"].max() - g["mean"].min())
        summaries.append(
            {
                "factor": f,
                "best_level": best_row["level"],
                "best_mean": float(best_row["mean"]),
                "effect_range": effect_range,
                "levels": int(g.shape[0]),
            }
        )

    factor_summary = pd.DataFrame(summaries)
    if factor_summary.empty:
        raise ValueError("因子の主効果を計算できませんでした。")
    factor_summary = factor_summary.sort_values("effect_range", ascending=False).reset_index(drop=True)

    direction = "大きいほど良い" if objective == "maximize" else "小さいほど良い"
    top = factor_summary.iloc[0]
    recommendation = (
        f"目的（{direction}）に対して、主効果の効果幅が最も大きいのは '{top['factor']}' でした。"
        "まずは効果幅の大きい因子から条件の固定化/最適化を検討してください。"
        "なお、交互作用が強い場合はこの主効果だけでは結論が変わる可能性があります。"
    )

    return DoeAnalysisResult(
        response_col=response_col,
        objective=objective,
        cleaned_n=int(work.shape[0]),
        main_effects_long=main_effects_long,
        factor_summary=factor_summary,
        recommendation=recommendation,
    )
