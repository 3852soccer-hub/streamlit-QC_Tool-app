from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import streamlit as st

import capability_engine
import data_loader
import doe_engine
import stats_engine
import visualization


TEXT = {
    "ja": {
        "language_label": "表示言語",
        "language_ja": "日本語",
        "language_en": "英語",
        "app_title": "品質改善アプリ",
        "app_caption": "品質データの入力 → 可視化 → 統計判定 → 解釈までをワンストップで実行します。",
        "menu": "メニュー",
        "feature_select": "機能選択",
        "feature_pareto": "① パレート図作成",
        "feature_hist": "② ヒストグラム + 正規分布判定",
        "feature_compare": "③ 改善前後データの統計比較（対応なし）",
        "feature_doe": "④ 実験計画法（DOE）",
        "feature_capability": "⑤ 工程能力指数（Cp, Cpk, Pp, Ppk）",
        "common_settings": "共通設定",
        "missing_policy": "欠損値処理",
        "missing_drop": "削除（drop）",
        "missing_mean": "平均で補完（mean）",
        "missing_median": "中央値で補完（median）",
        "iqr_checkbox": "IQR法で外れ値除去（任意）",
        "pareto_subheader": "① パレート図作成",
        "template_csv_header": "テンプレCSV（入力用）",
        "pareto_template_caption": "カテゴリ名と件数を入力してアップロードしてください（列: category, count）。",
        "template_rows": "テンプレ行数",
        "pareto_template_download": "テンプレCSVをダウンロード（category, count）",
        "pareto_upload": "入力済みCSVアップロード（列: category, count）",
        "run": "実行",
        "pareto_need_upload": "入力済みCSVをアップロードしてください（テンプレを使用できます）。",
        "pareto_missing_cols": "CSVには 'category' と 'count' 列が必要です（テンプレを使用してください）。",
        "pareto_chart_caption": "パレート図",
        "pareto_top_header": "上位重要項目（累積80%まで）",
        "pareto_top_col": "重要カテゴリ",
        "pareto_rank": "順位",
        "pareto_info": "上位カテゴリに対して、発生メカニズムの深掘り（現場観察/なぜなぜ分析）を優先すると効果が出やすいです。",
        "hist_subheader": "② ヒストグラム作成 + 正規分布判定",
        "hist_template_caption": "数値を1列に入力してアップロードしてください（列名: value 推奨。列名は任意でも1列目を使用します）。",
        "hist_template_download": "テンプレCSVをダウンロード（value）",
        "hist_upload": "入力済みCSVアップロード（1列目を使用）",
        "hist_bins": "ビン数",
        "hist_alpha": "有意水準",
        "hist_need_upload": "入力済みCSVをアップロードしてください（テンプレを使用できます）。",
        "hist_no_columns": "CSVに列がありません。",
        "hist_chart_caption": "ヒストグラム + 正規分布",
        "normality_header": "正規性検定結果",
        "shapiro_p": "Shapiro-Wilk p値",
        "ks_p": "KS p値",
        "decision": "判定",
        "iqr_removed": "IQR外れ値除去: removed={removed}",
        "interpretation": "解釈",
        "compare_subheader": "③ 改善前後データの統計比較（対応なし）",
        "compare_template_caption": "改善前/改善後をそれぞれ列に入力してアップロードできます（行数は揃っていなくてもOK。空欄は欠損として扱います）。",
        "compare_template_download": "テンプレCSVをダウンロード（before, after）",
        "compare_upload": "入力済みCSVアップロード（列: before, after）",
        "alpha": "有意水準（デフォルト0.05）",
        "test_kind_header": "検定種別（t値と臨界値で判定）",
        "test_kind": "判定方式",
        "two_sided": "両側検定（変化検出）",
        "one_sided": "片側検定（改善方向）",
        "ztest_header": "Z検定（条件: n>=30 かつ 母標準偏差既知）",
        "ztest_allow": "Z検定を選択肢に含める",
        "ztest_sigma": "母標準偏差 σ（既知）",
        "compare_need_upload": "入力済みCSVをアップロードしてください（テンプレを使用できます）。",
        "compare_missing_cols": "アップロードCSVには 'before' と 'after' 列が必要です（テンプレを使用してください）。",
        "compare_used_test": "使用検定法",
        "compare_dist_header": "分布の比較",
        "compare_dist_caption": "改善前後の分布比較",
        "compare_numbers": "数値結果",
        "t_test_type": "検定種別",
        "t_two_sided": "両側",
        "t_one_sided": "片側（改善方向）",
        "t_stat": "t算出値（t_stat）",
        "t_df": "自由度（df）",
        "t_crit": "臨界値（t_crit）",
        "alpha_simple": "有意水準",
        "qc_judgement": "判定（QC文脈）",
        "statistic": "統計量",
        "p_value": "p値",
        "extra_info": "付帯情報",
        "compare_interpretation": "解釈（品質改善の観点）",
        "doe_subheader": "④ 実験計画法（DOE）",
        "doe_k": "因子数",
        "doe_levels_caption": "各因子の水準数を入力してください（例: 2,2,2）。",
        "doe_levels": "水準数（カンマ区切り）",
        "doe_method": "設計方法",
        "doe_method_taguchi": "直交表（タグチ簡易）",
        "doe_method_full": "フルファクタリアル",
        "generate": "生成",
        "doe_levels_mismatch": "因子数と水準数の個数が一致していません。",
        "doe_design_header": "実験計画表",
        "doe_metrics": "指標",
        "doe_method_col": "設計法",
        "doe_runs": "実験回数",
        "doe_full_runs": "フルファクタリアル回数",
        "doe_efficiency": "実験効率指標（full/runs）",
        "doe_reduction": "削減率",
        "download_csv": "CSVダウンロード",
        "doe_info": "ダウンロードしたCSVに、実験結果（応答）列を追加して保存してください。例: 列名を 'Y' として各行に測定値を入力 → 下の『結果CSVをインポートして分析』でアップロード。",
        "doe_result_header": "結果CSVをインポートして分析",
        "doe_result_upload": "実験結果CSV（因子列 + 応答列）",
        "doe_loaded": "読み込み: {rows}行 × {cols}列",
        "doe_need_cols": "因子列と応答列を含むCSVが必要です（列が足りません）。",
        "doe_response_col": "応答列（実験値）",
        "doe_objective": "目的",
        "doe_objective_max": "大きいほど良い（最大化）",
        "doe_objective_min": "小さいほど良い（最小化）",
        "analyze": "分析",
        "doe_factor_summary": "因子サマリ（効果幅が大きい順）",
        "doe_main_effects": "主効果（因子×水準の平均）",
        "doe_main_effects_caption": "DOE 主効果プロット",
        "doe_interpretation": "解釈",
        "capability_subheader": "⑤ 工程能力指数算出（Cp, Cpk, Pp, Ppk）",
        "cap_csv_upload": "CSVアップロード（1列目を使用）",
        "lsl": "規格下限 LSL",
        "usl": "規格上限 USL",
        "subgroup_header": "Cpk（短期: within）用サブグループ設定",
        "subgroup_size": "サブグループサイズ n（例: 5）",
        "subgroup_help": "CpkはX-R管理図の考え方で σ_within = R̄/d2 を用います。サブグループ（同条件で短時間に採取したn個）を作るのが前提です。",
        "subgroup_template_header": "サブグループ入力用CSV（テンプレ）",
        "subgroup_count": "サブグループ数（行数）",
        "subgroup_count_help": "サブグループサイズnと合わせて、入力用CSVテンプレを生成します（例: n=5、サブグループ数=10）。",
        "subgroup_template_download": "テンプレCSVをダウンロード",
        "subgroup_template_help": "ダウンロード後、各行（サブグループ）にx1..xnの測定値を入力して保存し、下でインポートしてください。",
        "cap_analysis_header": "解析（入力済みCSVをインポートして実行）",
        "cap_note": "※入力済みCSVをインポートして解析します。",
        "cap_input_method": "入力方法",
        "cap_mode_subgroup": "サブグループCSV（推奨: X-R管理図でCpk算出）",
        "cap_mode_single": "単票データ（簡易）",
        "cap_subgroup_upload": "入力済みサブグループCSV（列: subgroup, x1..xn）",
        "calculate": "算出",
        "cap_need_subgroup": "サブグループCSVをアップロードしてください。",
        "cap_need_upload": "入力済みCSVをアップロードしてください。",
        "cap_need_n": "サブグループサイズ n を指定してください（例: 5）。",
        "cap_missing_cols": "サブグループCSVの列が不足しています: {cols}",
        "cap_judgement_header": "判定（優先順位: 管理状態 → Ppk → Cpk）",
        "cap_status": "ステータス",
        "cap_level": "判定",
        "cap_out_of_control": "管理限界逸脱(out_of_control)",
        "cap_n": "n",
        "cap_mean": "平均",
        "cap_warning": "管理図で逸脱があるため工程は不安定です。Cpk/Ppkの解釈は無効として、安定化を優先してください。",
        "cap_cpk_header": "Cpk（短期: within / X-R管理図）",
        "cap_rbar": "R̄",
        "cap_sigma_within": "σ_within（R̄/d2）",
        "cap_cp": "Cp（within）",
        "cap_cpk": "Cpk（within）",
        "cap_ppk_header": "Ppk（長期: overall / 分布ベース）",
        "cap_std_overall": "標準偏差(s) [overall]",
        "cap_sigma_overall": "σ_overall（χ²ベース）",
        "cap_sigma_ci_low": "σ_overall CI下限",
        "cap_sigma_ci_high": "σ_overall CI上限",
        "cap_pp": "Pp（overall）",
        "cap_ppk": "Ppk（overall）",
        "cap_pp_high": "Pp（σ上限参考）",
        "cap_ppk_high": "Ppk（σ上限参考）",
        "cap_fig_header": "図示",
        "cap_xr_caption": "X-R管理図",
        "cap_limits_header": "管理限界値（算出結果）",
        "cap_const": "定数",
        "cap_value": "値",
        "cap_chart": "管理図",
        "cap_type": "種別",
        "cap_value_col": "値",
        "cap_xr_info": "X-R管理図はサブグループCSV入力時のみ表示します。",
        "cap_ppk_caption": "Ppk用 分布図（overall）",
        "cap_comment_header": "工程能力評価コメント",
        "error_no_columns": "CSVに列がありません。",
    },
    "en": {
        "language_label": "Language",
        "language_ja": "Japanese",
        "language_en": "English",
        "app_title": "Quality Improvement App",
        "app_caption": "Input quality data → visualize → statistical judgment → interpretation in one stop.",
        "menu": "Menu",
        "feature_select": "Select feature",
        "feature_pareto": "① Pareto chart",
        "feature_hist": "② Histogram + normality test",
        "feature_compare": "③ Before/After statistical comparison (unpaired)",
        "feature_doe": "④ Design of Experiments (DOE)",
        "feature_capability": "⑤ Process capability (Cp, Cpk, Pp, Ppk)",
        "common_settings": "Common settings",
        "missing_policy": "Missing value handling",
        "missing_drop": "Drop",
        "missing_mean": "Impute with mean",
        "missing_median": "Impute with median",
        "iqr_checkbox": "IQR outlier removal (optional)",
        "pareto_subheader": "① Pareto chart",
        "template_csv_header": "Template CSV",
        "pareto_template_caption": "Enter category and count and upload (columns: category, count).",
        "template_rows": "Template rows",
        "pareto_template_download": "Download template CSV (category, count)",
        "pareto_upload": "Upload filled CSV (columns: category, count)",
        "run": "Run",
        "pareto_need_upload": "Please upload a filled CSV (you can use the template).",
        "pareto_missing_cols": "CSV must contain 'category' and 'count' columns (use the template).",
        "pareto_chart_caption": "Pareto chart",
        "pareto_top_header": "Top categories (up to cumulative 80%)",
        "pareto_top_col": "Key category",
        "pareto_rank": "Rank",
        "pareto_info": "Focus on top categories and dig into root causes for best impact.",
        "hist_subheader": "② Histogram + normality test",
        "hist_template_caption": "Enter numbers in one column and upload (column name 'value' recommended; first column is used).",
        "hist_template_download": "Download template CSV (value)",
        "hist_upload": "Upload filled CSV (use first column)",
        "hist_bins": "Bins",
        "hist_alpha": "Significance level",
        "hist_need_upload": "Please upload a filled CSV (you can use the template).",
        "hist_no_columns": "CSV has no columns.",
        "hist_chart_caption": "Histogram + normal curve",
        "normality_header": "Normality test results",
        "shapiro_p": "Shapiro-Wilk p-value",
        "ks_p": "KS p-value",
        "decision": "Decision",
        "iqr_removed": "IQR outlier removal: removed={removed}",
        "interpretation": "Interpretation",
        "compare_subheader": "③ Before/After statistical comparison (unpaired)",
        "compare_template_caption": "Enter before/after in separate columns and upload (row counts can differ; blanks are treated as missing).",
        "compare_template_download": "Download template CSV (before, after)",
        "compare_upload": "Upload filled CSV (columns: before, after)",
        "alpha": "Significance level (default 0.05)",
        "test_kind_header": "Test type (decision by t-statistic and critical value)",
        "test_kind": "Decision method",
        "two_sided": "Two-sided (detect change)",
        "one_sided": "One-sided (improvement)",
        "ztest_header": "Z-test (conditions: n>=30 and known population σ)",
        "ztest_allow": "Include Z-test as an option",
        "ztest_sigma": "Population std dev σ (known)",
        "compare_need_upload": "Please upload a filled CSV (you can use the template).",
        "compare_missing_cols": "Uploaded CSV must contain 'before' and 'after' columns (use the template).",
        "compare_used_test": "Test used",
        "compare_dist_header": "Distribution comparison",
        "compare_dist_caption": "Before/After distribution",
        "compare_numbers": "Numeric results",
        "t_test_type": "Test type",
        "t_two_sided": "Two-sided",
        "t_one_sided": "One-sided (improvement)",
        "t_stat": "t-statistic",
        "t_df": "df",
        "t_crit": "t-critical",
        "alpha_simple": "Significance level",
        "qc_judgement": "Judgment (QC context)",
        "statistic": "Statistic",
        "p_value": "p-value",
        "extra_info": "Additional info",
        "compare_interpretation": "Interpretation (quality improvement)",
        "doe_subheader": "④ Design of Experiments (DOE)",
        "doe_k": "Number of factors",
        "doe_levels_caption": "Enter the number of levels for each factor (e.g., 2,2,2).",
        "doe_levels": "Levels (comma-separated)",
        "doe_method": "Design method",
        "doe_method_taguchi": "Orthogonal array (Taguchi - simple)",
        "doe_method_full": "Full factorial",
        "generate": "Generate",
        "doe_levels_mismatch": "Number of factors does not match number of level entries.",
        "doe_design_header": "Experiment design table",
        "doe_metrics": "Metrics",
        "doe_method_col": "Method",
        "doe_runs": "Runs",
        "doe_full_runs": "Full factorial runs",
        "doe_efficiency": "Efficiency (full/runs)",
        "doe_reduction": "Reduction ratio",
        "download_csv": "Download CSV",
        "doe_info": "Add the response column to the downloaded CSV and save it. Example: name the response column 'Y', enter measurements, then upload below.",
        "doe_result_header": "Import result CSV and analyze",
        "doe_result_upload": "Result CSV (factor columns + response column)",
        "doe_loaded": "Loaded: {rows} rows × {cols} columns",
        "doe_need_cols": "CSV must include factor and response columns (not enough columns).",
        "doe_response_col": "Response column",
        "doe_objective": "Objective",
        "doe_objective_max": "Larger is better (maximize)",
        "doe_objective_min": "Smaller is better (minimize)",
        "analyze": "Analyze",
        "doe_factor_summary": "Factor summary (sorted by effect size)",
        "doe_main_effects": "Main effects (factor × level mean)",
        "doe_main_effects_caption": "DOE main effects plot",
        "doe_interpretation": "Interpretation",
        "capability_subheader": "⑤ Process capability (Cp, Cpk, Pp, Ppk)",
        "cap_csv_upload": "Upload CSV (use first column)",
        "lsl": "LSL",
        "usl": "USL",
        "subgroup_header": "Subgroup settings for Cpk (short-term: within)",
        "subgroup_size": "Subgroup size n (e.g., 5)",
        "subgroup_help": "Cpk uses X-R chart concept and σ_within = R̄/d2. Subgroups are required (n samples taken under same conditions).",
        "subgroup_template_header": "Subgroup CSV template",
        "subgroup_count": "Number of subgroups (rows)",
        "subgroup_count_help": "Generate template CSV with subgroup size n (e.g., n=5, subgroups=10).",
        "subgroup_template_download": "Download template CSV",
        "subgroup_template_help": "Fill x1..xn for each subgroup row, save, and import below.",
        "cap_analysis_header": "Analysis (import filled CSV)",
        "cap_note": "*Analysis is performed by importing the filled CSV.",
        "cap_input_method": "Input method",
        "cap_mode_subgroup": "Subgroup CSV (recommended: Cpk via X-R chart)",
        "cap_mode_single": "Single data (simplified)",
        "cap_subgroup_upload": "Filled subgroup CSV (columns: subgroup, x1..xn)",
        "calculate": "Calculate",
        "cap_need_subgroup": "Please upload the subgroup CSV.",
        "cap_need_upload": "Please upload the filled CSV.",
        "cap_need_n": "Please specify subgroup size n (e.g., 5).",
        "cap_missing_cols": "Subgroup CSV is missing columns: {cols}",
        "cap_judgement_header": "Judgment (priority: control status → Ppk → Cpk)",
        "cap_status": "Status",
        "cap_level": "Judgment",
        "cap_out_of_control": "Out of control",
        "cap_n": "n",
        "cap_mean": "Mean",
        "cap_warning": "The process is unstable due to control chart violations. Interpret Cpk/Ppk as invalid and stabilize first.",
        "cap_cpk_header": "Cpk (short-term: within / X-R chart)",
        "cap_rbar": "R̄",
        "cap_sigma_within": "σ_within (R̄/d2)",
        "cap_cp": "Cp (within)",
        "cap_cpk": "Cpk (within)",
        "cap_ppk_header": "Ppk (long-term: overall / distribution-based)",
        "cap_std_overall": "Std dev (s) [overall]",
        "cap_sigma_overall": "σ_overall (chi-square)",
        "cap_sigma_ci_low": "σ_overall CI low",
        "cap_sigma_ci_high": "σ_overall CI high",
        "cap_pp": "Pp (overall)",
        "cap_ppk": "Ppk (overall)",
        "cap_pp_high": "Pp (using σ high)",
        "cap_ppk_high": "Ppk (using σ high)",
        "cap_fig_header": "Plots",
        "cap_xr_caption": "X-R control chart",
        "cap_limits_header": "Control limits (calculated)",
        "cap_const": "Constant",
        "cap_value": "Value",
        "cap_chart": "Chart",
        "cap_type": "Type",
        "cap_value_col": "Value",
        "cap_xr_info": "X-R chart is shown only when subgroup CSV is provided.",
        "cap_ppk_caption": "Ppk distribution (overall)",
        "cap_comment_header": "Process capability comment",
        "error_no_columns": "CSV has no columns.",
    },
}


def t(lang: str, key: str, **kwargs: object) -> str:
    text = TEXT.get(lang, TEXT["ja"]).get(key, key)
    return text.format(**kwargs)


def _translate_normality(lang: str, decision: str, comment: str) -> tuple[str, str]:
    if lang != "en":
        return decision, comment
    decision_map = {
        "正規分布とみなせる": "Normal",
        "正規分布とみなせない": "Non-normal",
        "判定不可": "Not determined",
    }
    decision_en = decision_map.get(decision, decision)
    if decision == "正規分布とみなせる":
        comment_en = "No statistically significant non-normality was detected (p ≥ alpha)."
    elif decision == "正規分布とみなせない":
        comment_en = "Departure from normality is suggested (p < alpha). Consider nonparametric tests."
    else:
        comment_en = "Normality could not be determined (insufficient data or calculation failed)."
    return decision_en, comment_en


def _translate_compare_test_name(lang: str, test_name: str) -> str:
    if lang != "en":
        return test_name
    mapping = {
        "Z検定（母分散既知・2群）": "Z-test (known σ, two groups)",
        "t検定（等分散・対応なし）": "t-test (equal variance, unpaired)",
        "Welchのt検定（不等分散・対応なし）": "Welch's t-test (unequal variance, unpaired)",
        "Mann-WhitneyのU検定（ノンパラメトリック）": "Mann–Whitney U test (nonparametric)",
    }
    return mapping.get(test_name, test_name)


def _translate_compare_decision(lang: str, decision: str) -> str:
    if lang != "en":
        return decision
    mapping = {
        "有意差あり（工程変化あり）": "Significant difference (process change)",
        "有意差なし": "No significant difference",
        "統計的に有意": "Statistically significant",
        "統計的に有意な改善": "Statistically significant improvement",
        "改善とは言えない": "Improvement not supported",
        "判定不可": "Not determined",
    }
    return mapping.get(decision, decision)


def _compare_judgement_en(result: stats_engine.CompareResult, decision_en: str) -> str:
    if result.t_stat is None or result.df is None or result.t_crit is None or result.tail is None:
        return ""
    if result.tail == "two-sided":
        significant = abs(float(result.t_stat)) > float(result.t_crit)
        op = ">" if significant else "≤"
        return (
            f"df: {result.df:.0f}\n"
            "Two-sided test (detect change)\n"
            f"Critical value t({(result.alpha/2):.3g},{result.df:.0f}) = {result.t_crit:.3g}\n"
            f"t-statistic = {result.t_stat:.3g}\n\n"
            "Decision:\n"
            f"|t| {op} {result.t_crit:.3g} → {decision_en}"
        )
    significant = float(result.t_stat) > float(result.t_crit)
    op = ">" if significant else "≤"
    return (
        f"df: {result.df:.0f}\n"
        "One-sided test (improvement)\n"
        f"Critical value t({result.alpha:.3g},{result.df:.0f}) = {result.t_crit:.3g}\n"
        f"t-statistic = {result.t_stat:.3g}\n\n"
        "Decision:\n"
        f"t-statistic {op} {result.t_crit:.3g} → {decision_en}"
    )


def _compare_comment_en(decision: str, before_mean: float, after_mean: float) -> str:
    direction = "After is higher" if after_mean > before_mean else "After is lower"
    significant = decision in {"統計的に有意", "有意差あり（工程変化あり）", "統計的に有意な改善"}
    if significant:
        return (
            f"{direction} (mean: before={before_mean:.4g}, after={after_mean:.4g}), which is unlikely due to chance. "
            "The improvement actions likely had an effect; confirm reproducibility with additional data or reruns."
        )
    return (
        f"{direction} (mean: before={before_mean:.4g}, after={after_mean:.4g}), but not statistically significant. "
        "The effect may be small, variability high, or sample size insufficient; consider more samples or variance reduction."
    )


def _capability_labels_en(res: capability_engine.CapabilityResult) -> tuple[str, str, str]:
    status_map = {"判定不可": "Not determined"}
    level_map = {
        "統計的不安定": "Statistically unstable",
        "長期性能不足": "Low long-term performance",
        "高能力（有効）": "High capability (effective)",
        "能力限定": "Limited capability",
        "シックスシグマ水準": "Six Sigma level",
        "優秀": "Excellent",
        "良好": "Good",
        "要改善": "Needs improvement",
        "不十分": "Insufficient",
        "判定不可": "Not determined",
    }
    status = status_map.get(res.status, res.status)
    level = level_map.get(res.level, res.level)

    if res.out_of_control is True:
        comment = (
            "Control chart violations were detected, so the process is statistically unstable. "
            "Cpk/Ppk interpretations are invalid until the process is stabilized."
        )
    elif res.ppk is not None and res.ppk < 1.33:
        comment = (
            "Long-term performance (Ppk) is below the threshold, indicating insufficient long-term capability. "
            "Even if short-term capability (Cpk) is high, reproducibility and stability need improvement."
        )
    elif res.cpk is not None and res.ppk is not None and res.cpk >= 1.67 and res.ppk >= 1.33:
        comment = (
            "The process appears stable with good short-term capability (Cpk) and long-term performance (Ppk). "
            "Standardize and maintain current conditions."
        )
    else:
        comment = (
            "Capability is moderate; improvement opportunities remain. "
            "Consider reducing variation (σ) and centering the mean."
        )

    if res.out_of_control is None:
        comment = "Note: control status was not evaluated (no subgroup data). " + comment

    comment += "\n\nMeaning: Cpk evaluates within-subgroup variation; Ppk evaluates overall variation."
    return status, level, comment


def _doe_recommendation_en(res: doe_engine.DoeAnalysisResult) -> str:
    direction = "Larger is better" if res.objective == "maximize" else "Smaller is better"
    top = res.factor_summary.iloc[0]
    return (
        f"Objective ({direction}): the factor with the largest main effect range is '{top['factor']}'. "
        "Start by standardizing/optimizing the factor(s) with the largest effect. "
        "If interactions are strong, conclusions based only on main effects may change."
    )


def setup_logging() -> None:
    # Streamlitの再実行でも重複ハンドラを作りにくい設定
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        fh = logging.FileHandler("qc_app.log", encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        fh.setFormatter(fmt)
        root.addHandler(fh)


def main() -> None:
    setup_logging()

    st.set_page_config(page_title=t("ja", "app_title"), layout="wide")

    with st.sidebar:
        lang_label = t("ja", "language_label")
        lang = st.selectbox(lang_label, ["ja", "en"], format_func=lambda k: t(k, f"language_{k}"))

    st.title(t(lang, "app_title"))
    st.caption(t(lang, "app_caption"))

    with st.sidebar:
        st.header(t(lang, "menu"))
        feature_labels = {
            "pareto": t(lang, "feature_pareto"),
            "hist": t(lang, "feature_hist"),
            "compare": t(lang, "feature_compare"),
            "doe": t(lang, "feature_doe"),
            "capability": t(lang, "feature_capability"),
        }
        feature = st.selectbox(
            t(lang, "feature_select"),
            list(feature_labels.keys()),
            format_func=lambda k: feature_labels[k],
        )

        st.divider()
        st.subheader(t(lang, "common_settings"))
        missing_policy = st.selectbox(
            t(lang, "missing_policy"),
            ["drop", "mean", "median"],
            index=0,
            format_func=lambda k: t(lang, f"missing_{k}"),
        )
        use_iqr = st.checkbox(t(lang, "iqr_checkbox"), value=False)

    if feature == "pareto":
        ui_pareto(missing_policy, use_iqr, lang)
    elif feature == "hist":
        ui_hist_normality(missing_policy, use_iqr, lang)
    elif feature == "compare":
        ui_compare(missing_policy, use_iqr, lang)
    elif feature == "doe":
        ui_doe(lang)
    elif feature == "capability":
        ui_capability(missing_policy, use_iqr, lang)


def ui_pareto(missing_policy: str, use_iqr: bool, lang: str) -> None:
    st.subheader(t(lang, "pareto_subheader"))

    st.markdown(f"##### {t(lang, 'template_csv_header')}")
    st.caption(t(lang, "pareto_template_caption"))
    template_rows = st.number_input(t(lang, "template_rows"), min_value=0, value=30, step=5, key="pareto_template_rows")
    if int(template_rows) > 0:
        template = pd.DataFrame({"category": [""] * int(template_rows), "count": [""] * int(template_rows)})
        template_bytes = template.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            t(lang, "pareto_template_download"),
            data=template_bytes,
            file_name="pareto_template.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader(t(lang, "pareto_upload"), type=["csv"], key="pareto_csv")

    if st.button(t(lang, "run"), key="pareto_run"):
        try:
            if uploaded is None:
                raise ValueError(t(lang, "pareto_need_upload"))

            df = data_loader.read_csv(uploaded)
            if not {"category", "count"}.issubset(set(df.columns)):
                raise ValueError(t(lang, "pareto_missing_cols"))
            df = df[["category", "count"]].copy()

            df["count"] = pd.to_numeric(df["count"], errors="coerce")
            df = df.dropna(subset=["category", "count"])
            df = df[df["count"] >= 0]

            fig = visualization.pareto_chart(df, lang=lang)
            st.image(fig.png_bytes, caption=t(lang, "pareto_chart_caption"), output_format="PNG")

            work = df.groupby("category", as_index=False)["count"].sum().sort_values("count", ascending=False)
            total = float(work["count"].sum())
            work["cum_ratio"] = work["count"].cumsum() / total
            important = work[work["cum_ratio"] <= 0.8]["category"].astype(str).tolist()
            if not important:
                # 最低1項目
                important = [str(work.iloc[0]["category"])]

            st.markdown(f"#### {t(lang, 'pareto_top_header')}")
            important_df = pd.DataFrame({t(lang, "pareto_top_col"): important})
            important_df.index = range(1, len(important_df) + 1)
            important_df.index.name = t(lang, "pareto_rank")
            st.table(important_df)
            st.info(t(lang, "pareto_info"))

        except Exception as e:
            st.error(str(e))


def ui_hist_normality(missing_policy: str, use_iqr: bool, lang: str) -> None:
    st.subheader(t(lang, "hist_subheader"))

    st.markdown(f"##### {t(lang, 'template_csv_header')}")
    st.caption(t(lang, "hist_template_caption"))
    template_rows = st.number_input(t(lang, "template_rows"), min_value=0, value=30, step=5, key="hist_template_rows")
    if int(template_rows) > 0:
        template = pd.DataFrame({"value": [""] * int(template_rows)})
        template_bytes = template.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            t(lang, "hist_template_download"),
            data=template_bytes,
            file_name="hist_template.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader(t(lang, "hist_upload"), type=["csv"], key="hist_csv")

    bins = st.slider(t(lang, "hist_bins"), min_value=5, max_value=60, value=20)
    alpha = st.number_input(t(lang, "hist_alpha"), min_value=0.001, max_value=0.2, value=0.05, step=0.01, format="%.3f")

    if st.button(t(lang, "run"), key="hist_run"):
        try:
            if uploaded is None:
                raise ValueError(t(lang, "hist_need_upload"))
            df = data_loader.read_csv(uploaded)
            if df.shape[1] < 1:
                raise ValueError(t(lang, "hist_no_columns"))
            s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
            s = data_loader.apply_missing_policy(s, missing_policy)
            s, out = data_loader.iqr_outlier_filter(s, enabled=use_iqr)

            fig = visualization.histogram_with_normal(s, bins=bins, lang=lang)
            st.image(fig.png_bytes, caption=t(lang, "hist_chart_caption"), output_format="PNG")

            res = stats_engine.normality_tests(s, alpha=alpha)
            decision_disp, comment_disp = _translate_normality(lang, res.decision, res.comment)
            st.markdown(f"#### {t(lang, 'normality_header')}")
            st.table(
                pd.DataFrame(
                    {
                        t(lang, "shapiro_p"): [res.shapiro_p],
                        t(lang, "ks_p"): [res.ks_p],
                        t(lang, "decision"): [decision_disp],
                    }
                )
            )

            if out.enabled:
                st.caption(t(lang, "iqr_removed", removed=out.removed_count))

            st.markdown(f"#### {t(lang, 'interpretation')}")
            st.write(comment_disp)

        except Exception as e:
            st.error(str(e))


def ui_compare(missing_policy: str, use_iqr: bool, lang: str) -> None:
    st.subheader(t(lang, "compare_subheader"))

    st.markdown(f"##### {t(lang, 'template_csv_header')}")
    st.caption(t(lang, "compare_template_caption"))
    template_rows = st.number_input(t(lang, "template_rows"), min_value=0, value=30, step=5)
    if int(template_rows) > 0:
        template = pd.DataFrame({"before": [""] * int(template_rows), "after": [""] * int(template_rows)})
        template_bytes = template.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            t(lang, "compare_template_download"),
            data=template_bytes,
            file_name="before_after_template.csv",
            mime="text/csv",
        )

    uploaded_pair = st.file_uploader(t(lang, "compare_upload"), type=["csv"], key="pair_csv")

    alpha = st.number_input(t(lang, "alpha"), min_value=0.001, max_value=0.2, value=0.05, step=0.01, format="%.3f")

    st.markdown(f"##### {t(lang, 'test_kind_header')}")
    test_kind_label = st.radio(
        t(lang, "test_kind"),
        [t(lang, "two_sided"), t(lang, "one_sided")],
        index=0,
        horizontal=True,
    )
    test_kind = "two-sided" if test_kind_label == t(lang, "two_sided") else "one-sided-improvement"

    st.markdown(f"##### {t(lang, 'ztest_header')}")
    allow_z = st.checkbox(t(lang, "ztest_allow"), value=False)
    known_sigma: Optional[float] = None
    if allow_z:
        known_sigma = st.number_input(t(lang, "ztest_sigma"), min_value=0.0, value=0.0, step=0.1)
        if known_sigma == 0.0:
            known_sigma = None

    if st.button(t(lang, "run"), key="compare_run"):
        try:
            if uploaded_pair is None:
                raise ValueError(t(lang, "compare_need_upload"))

            df = data_loader.read_csv(uploaded_pair)
            if not {"before", "after"}.issubset(set(df.columns)):
                raise ValueError(t(lang, "compare_missing_cols"))
            b = pd.to_numeric(df["before"], errors="coerce")
            a = pd.to_numeric(df["after"], errors="coerce")

            b = data_loader.apply_missing_policy(b, missing_policy)
            a = data_loader.apply_missing_policy(a, missing_policy)

            b, out_b = data_loader.iqr_outlier_filter(b, enabled=use_iqr)
            a, out_a = data_loader.iqr_outlier_filter(a, enabled=use_iqr)

            result = stats_engine.compare_before_after(
                before=b,
                after=a,
                alpha=float(alpha),
                allow_ztest=allow_z,
                known_sigma=known_sigma,
                test_kind=test_kind,
            )

            test_name_disp = _translate_compare_test_name(lang, result.test_name)
            decision_disp = _translate_compare_decision(lang, result.decision)
            judgement_disp = result.judgement
            comment_disp = result.comment
            if lang == "en":
                judgement_disp = _compare_judgement_en(result, decision_disp)
                comment_disp = _compare_comment_en(result.decision, float(b.mean()), float(a.mean()))

            st.markdown(f"#### {t(lang, 'compare_used_test')}")
            st.write(test_name_disp)

            st.markdown(f"#### {t(lang, 'compare_dist_header')}")
            dist = visualization.before_after_distribution_plot(b, a, bins=20, lang=lang)
            st.image(dist.png_bytes, caption=t(lang, "compare_dist_caption"), output_format="PNG")

            st.markdown(f"#### {t(lang, 'compare_numbers')}")

            if result.method.startswith("t_"):
                st.table(
                    pd.DataFrame(
                        {
                            t(lang, "t_test_type"): [
                                t(lang, "t_two_sided") if result.tail == "two-sided" else t(lang, "t_one_sided")
                            ],
                            t(lang, "t_stat"): [result.t_stat],
                            t(lang, "t_df"): [result.df],
                            t(lang, "t_crit"): [result.t_crit],
                            t(lang, "alpha_simple"): [result.alpha],
                            t(lang, "decision"): [decision_disp],
                        }
                    )
                )
                st.markdown(f"#### {t(lang, 'qc_judgement')}")
                st.text(judgement_disp)
            else:
                st.table(
                    pd.DataFrame(
                        {
                            t(lang, "statistic"): [result.statistic],
                            t(lang, "p_value"): [result.p_value],
                            t(lang, "alpha_simple"): [result.alpha],
                            t(lang, "decision"): [decision_disp],
                        }
                    )
                )

            st.markdown(f"#### {t(lang, 'extra_info')}")
            st.table(pd.DataFrame([result.details]))

            if use_iqr:
                st.caption(
                    t(
                        lang,
                        "iqr_removed",
                        removed=f"before {out_b.removed_count}, after {out_a.removed_count}",
                    )
                )

            st.markdown(f"#### {t(lang, 'compare_interpretation')}")
            st.write(comment_disp)

        except Exception as e:
            st.error(str(e))


def ui_doe(lang: str) -> None:
    st.subheader(t(lang, "doe_subheader"))

    k = st.number_input(t(lang, "doe_k"), min_value=1, max_value=20, value=3, step=1)

    st.caption(t(lang, "doe_levels_caption"))
    levels_text = st.text_input(t(lang, "doe_levels"), value=",".join(["2"] * int(k)))

    method = st.selectbox(
        t(lang, "doe_method"),
        [t(lang, "doe_method_taguchi"), t(lang, "doe_method_full")],
        index=0,
    )

    if st.button(t(lang, "generate"), key="doe_run"):
        try:
            factor_levels = [int(x.strip()) for x in levels_text.split(",") if x.strip()]
            if len(factor_levels) != int(k):
                raise ValueError(t(lang, "doe_levels_mismatch"))

            if method == t(lang, "doe_method_taguchi"):
                res = doe_engine.taguchi_orthogonal_array(factor_levels)
            else:
                res = doe_engine.full_factorial(factor_levels)

            st.markdown(f"#### {t(lang, 'doe_design_header')}")
            st.dataframe(res.design, use_container_width=True)

            st.markdown(f"#### {t(lang, 'doe_metrics')}")
            st.table(
                pd.DataFrame(
                    {
                        t(lang, "doe_method_col"): [res.method],
                        t(lang, "doe_runs"): [res.runs],
                        t(lang, "doe_full_runs"): [res.full_factorial_runs],
                        t(lang, "doe_efficiency"): [res.efficiency_ratio],
                        t(lang, "doe_reduction"): [res.reduction_ratio],
                    }
                )
            )
            st.write(res.comment)

            csv_bytes = res.design.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                t(lang, "download_csv"),
                data=csv_bytes,
                file_name="doe_design.csv",
                mime="text/csv",
            )

            st.info(t(lang, "doe_info"))

        except Exception as e:
            st.error(str(e))

    st.divider()
    st.markdown(f"#### {t(lang, 'doe_result_header')}")
    uploaded = st.file_uploader(t(lang, "doe_result_upload"), type=["csv"], key="doe_result_csv")

    if uploaded is not None:
        try:
            df = data_loader.read_csv(uploaded)
            st.caption(t(lang, "doe_loaded", rows=df.shape[0], cols=df.shape[1]))
            st.dataframe(df.head(20), use_container_width=True)

            if df.shape[1] < 2:
                raise ValueError(t(lang, "doe_need_cols"))

            # 応答列候補: 数値に変換して1つでも値が入る列
            numeric_candidates = []
            for c in df.columns:
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().any():
                    numeric_candidates.append(c)

            default_col = "Y" if "Y" in df.columns else (numeric_candidates[0] if numeric_candidates else df.columns[-1])
            response_col = st.selectbox(
                t(lang, "doe_response_col"),
                options=list(df.columns),
                index=list(df.columns).index(default_col),
            )
            objective_label = st.selectbox(
                t(lang, "doe_objective"),
                [t(lang, "doe_objective_max"), t(lang, "doe_objective_min")],
                index=0,
            )
            objective = "maximize" if objective_label == t(lang, "doe_objective_max") else "minimize"

            if st.button(t(lang, "analyze"), key="doe_analyze"):
                res = doe_engine.analyze_doe_results(df, response_col=response_col, objective=objective)

                st.markdown(f"##### {t(lang, 'doe_factor_summary')}")
                st.table(res.factor_summary)

                st.markdown(f"##### {t(lang, 'doe_main_effects')}")
                st.dataframe(res.main_effects_long, use_container_width=True)

                fig = visualization.doe_main_effects_plot(res.main_effects_long, lang=lang)
                st.image(fig.png_bytes, caption=t(lang, "doe_main_effects_caption"), output_format="PNG")

                st.markdown(f"##### {t(lang, 'doe_interpretation')}")
                recommendation = res.recommendation if lang != "en" else _doe_recommendation_en(res)
                st.write(recommendation)

        except Exception as e:
            st.error(str(e))


def ui_capability(missing_policy: str, use_iqr: bool, lang: str) -> None:
    st.subheader(t(lang, "capability_subheader"))

    uploaded = st.file_uploader(t(lang, "cap_csv_upload"), type=["csv"], key="cap_csv")

    lsl = st.number_input(t(lang, "lsl"), value=0.0, step=0.1, format="%.6g")
    usl = st.number_input(t(lang, "usl"), value=1.0, step=0.1, format="%.6g")

    st.markdown(f"##### {t(lang, 'subgroup_header')}")
    subgroup_size = st.number_input(
        t(lang, "subgroup_size"),
        min_value=0,
        value=0,
        step=1,
        help=t(lang, "subgroup_help"),
    )

    st.markdown(f"##### {t(lang, 'subgroup_template_header')}")
    subgroup_count = st.number_input(
        t(lang, "subgroup_count"),
        min_value=0,
        value=0,
        step=1,
        help=t(lang, "subgroup_count_help"),
    )

    if int(subgroup_size) > 0 and int(subgroup_count) > 0:
        n = int(subgroup_size)
        k = int(subgroup_count)
        cols = [f"x{i+1}" for i in range(n)]
        template = pd.DataFrame({"subgroup": list(range(1, k + 1))})
        for c in cols:
            template[c] = ""
        template_bytes = template.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            t(lang, "subgroup_template_download"),
            data=template_bytes,
            file_name=f"capability_subgroup_template_n{n}_k{k}.csv",
            mime="text/csv",
            help=t(lang, "subgroup_template_help"),
        )

    st.divider()
    st.markdown(f"#### {t(lang, 'cap_analysis_header')}")
    st.caption(t(lang, "cap_note"))
    cap_mode = st.radio(
        t(lang, "cap_input_method"),
        [t(lang, "cap_mode_subgroup"), t(lang, "cap_mode_single")],
        index=0,
        horizontal=False,
    )

    subgroup_csv = None
    if cap_mode == t(lang, "cap_mode_subgroup"):
        subgroup_csv = st.file_uploader(
            t(lang, "cap_subgroup_upload"),
            type=["csv"],
            key="cap_subgroup_csv",
        )

    if st.button(t(lang, "calculate"), key="cap_run"):
        try:
            subgroup_df = None
            out = None

            if cap_mode == t(lang, "cap_mode_subgroup"):
                if subgroup_csv is None:
                    raise ValueError(t(lang, "cap_need_subgroup"))
                if int(subgroup_size) <= 0:
                    raise ValueError(t(lang, "cap_need_n"))

                df = data_loader.read_csv(subgroup_csv)
                n = int(subgroup_size)
                expected_cols = ["subgroup"] + [f"x{i+1}" for i in range(n)]
                missing = [c for c in expected_cols if c not in df.columns]
                if missing:
                    raise ValueError(t(lang, "cap_missing_cols", cols=missing))

                subgroup_df = df[[f"x{i+1}" for i in range(n)]].copy()
                # overall用に全データをフラット化
                flat = pd.to_numeric(subgroup_df.stack(), errors="coerce")
                s = flat
            else:
                if uploaded is None:
                    raise ValueError(t(lang, "cap_need_upload"))
                df = data_loader.read_csv(uploaded)
                if df.shape[1] < 1:
                    raise ValueError(t(lang, "error_no_columns"))
                s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
                s = data_loader.apply_missing_policy(s, missing_policy)
                s, out = data_loader.iqr_outlier_filter(s, enabled=use_iqr)

            # 共通: overall（Ppk）と within（Cpk）を計算
            res = capability_engine.capability_indices(
                s,
                lsl=float(lsl),
                usl=float(usl),
                alpha=0.05,
                subgroup_size=int(subgroup_size) if int(subgroup_size) > 0 else None,
                subgroup_df=subgroup_df,
            )

            status_disp = res.status
            level_disp = res.level
            comment_disp = res.comment
            if lang == "en":
                status_disp, level_disp, comment_disp = _capability_labels_en(res)

            st.markdown(f"#### {t(lang, 'cap_judgement_header')}")
            st.table(
                pd.DataFrame(
                    {
                        t(lang, "cap_status"): [status_disp],
                        t(lang, "cap_level"): [level_disp],
                        t(lang, "cap_out_of_control"): [res.out_of_control],
                        t(lang, "cap_n"): [res.n],
                        t(lang, "cap_mean"): [res.mean],
                    }
                )
            )

            if res.out_of_control is True:
                st.warning(t(lang, "cap_warning"))

            st.markdown(f"#### {t(lang, 'cap_cpk_header')}")
            st.table(
                pd.DataFrame(
                    {
                        t(lang, "cap_n"): [res.subgroup_size],
                        t(lang, "cap_rbar"): [res.rbar],
                        t(lang, "cap_sigma_within"): [res.sigma_within],
                        t(lang, "cap_cp"): [res.cp],
                        t(lang, "cap_cpk"): [res.cpk],
                    }
                )
            )

            st.markdown(f"#### {t(lang, 'cap_ppk_header')}")
            # σの上限（CI上限）で計算した参考値（保守的に小さくなる）
            ppk_sigma_high = None
            pp_sigma_high = None
            if res.sigma_overall_ci_high is not None and res.sigma_overall_ci_high > 0:
                sigma_h = float(res.sigma_overall_ci_high)
                pp_sigma_high = (float(usl) - float(lsl)) / (6 * sigma_h)
                ppk_sigma_high = min((float(usl) - float(res.mean)) / (3 * sigma_h), (float(res.mean) - float(lsl)) / (3 * sigma_h))

            st.table(
                pd.DataFrame(
                    {
                        t(lang, "cap_std_overall"): [res.std_overall_s],
                        t(lang, "cap_sigma_overall"): [res.sigma_overall],
                        t(lang, "cap_sigma_ci_low"): [res.sigma_overall_ci_low],
                        t(lang, "cap_sigma_ci_high"): [res.sigma_overall_ci_high],
                        t(lang, "cap_pp"): [res.pp],
                        t(lang, "cap_ppk"): [res.ppk],
                        t(lang, "cap_pp_high"): [pp_sigma_high],
                        t(lang, "cap_ppk_high"): [ppk_sigma_high],
                    }
                )
            )

            if out is not None and out.enabled:
                st.caption(t(lang, "iqr_removed", removed=out.removed_count))

            st.markdown(f"#### {t(lang, 'cap_fig_header')}")
            # X-R管理図（サブグループCSVがある場合のみ）
            if subgroup_df is not None:
                xr = capability_engine.xr_chart_from_subgroups(subgroup_df, subgroup_size=int(subgroup_size))
                xr_fig = visualization.xr_control_chart_plot(xr, lang=lang)
                st.image(xr_fig.png_bytes, caption=t(lang, "cap_xr_caption"), output_format="PNG")

                st.markdown(f"##### {t(lang, 'cap_limits_header')}")
                st.table(
                    pd.DataFrame(
                        {
                            t(lang, "cap_const"): ["d2", "A2", "D3", "D4"],
                            t(lang, "cap_value"): [xr.d2, xr.a2, xr.d3, xr.d4],
                        }
                    )
                )
                st.table(
                    pd.DataFrame(
                        {
                            t(lang, "cap_chart"): ["X\u0305", "X\u0305", "X\u0305", "R", "R", "R"],
                            t(lang, "cap_type"): ["UCL", "CL", "LCL", "UCL", "CL", "LCL"],
                            t(lang, "cap_value_col"): [xr.ucl_x, xr.cl_x, xr.lcl_x, xr.ucl_r, xr.cl_r, xr.lcl_r],
                        }
                    )
                )
            else:
                st.info(t(lang, "cap_xr_info"))

            # Ppk用 分布図（overall）
            dist_fig = visualization.ppk_distribution_plot(
                series=s,
                lsl=float(lsl),
                usl=float(usl),
                mean=float(res.mean),
                sigma_overall=float(res.sigma_overall),
                bins=20,
                lang=lang,
            )
            st.image(dist_fig.png_bytes, caption=t(lang, "cap_ppk_caption"), output_format="PNG")

            st.markdown(f"#### {t(lang, 'cap_comment_header')}")
            st.write(comment_disp)

        except Exception as e:
            st.error(str(e))


if __name__ == "__main__":
    main()
