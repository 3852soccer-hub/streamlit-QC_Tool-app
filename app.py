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

    st.set_page_config(page_title="品質改善アプリ", layout="wide")
    st.title("品質改善アプリ（Quality Improvement App）")
    st.caption("品質データの入力 → 可視化 → 統計判定 → 解釈までをワンストップで実行します。")

    with st.sidebar:
        st.header("メニュー")
        feature = st.selectbox(
            "機能選択",
            [
                "① パレート図作成",
                "② ヒストグラム + 正規分布判定",
                "③ 改善前後データの統計比較",
                "④ 実験計画法（DOE）",
                "⑤ 工程能力指数（Cp, Cpk, Pp, Ppk）",
            ],
        )

        st.divider()
        st.subheader("共通設定")
        missing_policy = st.selectbox("欠損値処理", ["drop", "mean", "median"], index=0)
        use_iqr = st.checkbox("IQR法で外れ値除去（任意）", value=False)

    if feature.startswith("①"):
        ui_pareto(missing_policy, use_iqr)
    elif feature.startswith("②"):
        ui_hist_normality(missing_policy, use_iqr)
    elif feature.startswith("③"):
        ui_compare(missing_policy, use_iqr)
    elif feature.startswith("④"):
        ui_doe()
    elif feature.startswith("⑤"):
        ui_capability(missing_policy, use_iqr)


def ui_pareto(missing_policy: str, use_iqr: bool) -> None:
    st.subheader("① パレート図作成")

    st.markdown("##### テンプレCSV（入力用）")
    st.caption("カテゴリ名と件数を入力してアップロードしてください（列: category, count）。")
    template_rows = st.number_input("テンプレ行数", min_value=0, value=30, step=5, key="pareto_template_rows")
    if int(template_rows) > 0:
        template = pd.DataFrame({"category": [""] * int(template_rows), "count": [""] * int(template_rows)})
        template_bytes = template.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "テンプレCSVをダウンロード（category, count）",
            data=template_bytes,
            file_name="pareto_template.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("入力済みCSVアップロード（列: category, count）", type=["csv"], key="pareto_csv")

    if st.button("実行", key="pareto_run"):
        try:
            if uploaded is None:
                raise ValueError("入力済みCSVをアップロードしてください（テンプレを使用できます）。")

            df = data_loader.read_csv(uploaded)
            if not {"category", "count"}.issubset(set(df.columns)):
                raise ValueError("CSVには 'category' と 'count' 列が必要です（テンプレを使用してください）。")
            df = df[["category", "count"]].copy()

            df["count"] = pd.to_numeric(df["count"], errors="coerce")
            df = df.dropna(subset=["category", "count"])
            df = df[df["count"] >= 0]

            fig = visualization.pareto_chart(df)
            st.image(fig.png_bytes, caption="パレート図", output_format="PNG")

            work = df.groupby("category", as_index=False)["count"].sum().sort_values("count", ascending=False)
            total = float(work["count"].sum())
            work["cum_ratio"] = work["count"].cumsum() / total
            important = work[work["cum_ratio"] <= 0.8]["category"].astype(str).tolist()
            if not important:
                # 最低1項目
                important = [str(work.iloc[0]["category"])]

            st.markdown("#### 上位重要項目（累積80%まで）")
            important_df = pd.DataFrame({"重要カテゴリ": important})
            important_df.index = range(1, len(important_df) + 1)
            important_df.index.name = "順位"
            st.table(important_df)
            st.info("上位カテゴリに対して、発生メカニズムの深掘り（現場観察/なぜなぜ分析）を優先すると効果が出やすいです。")

        except Exception as e:
            st.error(str(e))


def ui_hist_normality(missing_policy: str, use_iqr: bool) -> None:
    st.subheader("② ヒストグラム作成 + 正規分布判定")

    st.markdown("##### テンプレCSV（入力用）")
    st.caption("数値を1列に入力してアップロードしてください（列名: value 推奨。列名は任意でも1列目を使用します）。")
    template_rows = st.number_input("テンプレ行数", min_value=0, value=30, step=5, key="hist_template_rows")
    if int(template_rows) > 0:
        template = pd.DataFrame({"value": [""] * int(template_rows)})
        template_bytes = template.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "テンプレCSVをダウンロード（value）",
            data=template_bytes,
            file_name="hist_template.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("入力済みCSVアップロード（1列目を使用）", type=["csv"], key="hist_csv")

    bins = st.slider("ビン数", min_value=5, max_value=60, value=20)
    alpha = st.number_input("有意水準", min_value=0.001, max_value=0.2, value=0.05, step=0.01, format="%.3f")

    if st.button("実行", key="hist_run"):
        try:
            if uploaded is None:
                raise ValueError("入力済みCSVをアップロードしてください（テンプレを使用できます）。")
            df = data_loader.read_csv(uploaded)
            if df.shape[1] < 1:
                raise ValueError("CSVに列がありません。")
            s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
            s = data_loader.apply_missing_policy(s, missing_policy)
            s, out = data_loader.iqr_outlier_filter(s, enabled=use_iqr)

            fig = visualization.histogram_with_normal(s, bins=bins)
            st.image(fig.png_bytes, caption="ヒストグラム + 正規分布", output_format="PNG")

            res = stats_engine.normality_tests(s, alpha=alpha)
            st.markdown("#### 正規性検定結果")
            st.table(
                pd.DataFrame(
                    {
                        "Shapiro-Wilk p値": [res.shapiro_p],
                        "KS p値": [res.ks_p],
                        "判定": [res.decision],
                    }
                )
            )

            if out.enabled:
                st.caption(f"IQR外れ値除去: removed={out.removed_count}")

            st.markdown("#### 解釈")
            st.write(res.comment)

        except Exception as e:
            st.error(str(e))


def ui_compare(missing_policy: str, use_iqr: bool) -> None:
    st.subheader("③ 改善前後データの統計比較（対応なし）")

    st.markdown("##### テンプレCSV（入力用）")
    st.caption("改善前/改善後をそれぞれ列に入力してアップロードできます（行数は揃っていなくてもOK。空欄は欠損として扱います）。")
    template_rows = st.number_input("テンプレ行数", min_value=0, value=30, step=5)
    if int(template_rows) > 0:
        template = pd.DataFrame({"before": [""] * int(template_rows), "after": [""] * int(template_rows)})
        template_bytes = template.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "テンプレCSVをダウンロード（before, after）",
            data=template_bytes,
            file_name="before_after_template.csv",
            mime="text/csv",
        )

    uploaded_pair = st.file_uploader("入力済みCSVアップロード（列: before, after）", type=["csv"], key="pair_csv")

    col1, col2 = st.columns(2)
    with col1:
        before_csv = st.file_uploader("改善前 CSV（1列目を使用）", type=["csv"], key="before_csv")
        before_text = st.text_area("改善前 手入力", height=160, key="before_text")
    with col2:
        after_csv = st.file_uploader("改善後 CSV（1列目を使用）", type=["csv"], key="after_csv")
        after_text = st.text_area("改善後 手入力", height=160, key="after_text")

    alpha = st.number_input("有意水準（デフォルト0.05）", min_value=0.001, max_value=0.2, value=0.05, step=0.01, format="%.3f")

    st.markdown("##### 検定種別（t値と臨界値で判定）")
    test_kind_label = st.radio(
        "判定方式",
        ["両側検定（変化検出）", "片側検定（改善方向）"],
        index=0,
        horizontal=True,
    )
    test_kind = "two-sided" if test_kind_label.startswith("両側") else "one-sided-improvement"

    st.markdown("##### Z検定（条件: n>=30 かつ 母標準偏差既知）")
    allow_z = st.checkbox("Z検定を選択肢に含める", value=False)
    known_sigma: Optional[float] = None
    if allow_z:
        known_sigma = st.number_input("母標準偏差 σ（既知）", min_value=0.0, value=0.0, step=0.1)
        if known_sigma == 0.0:
            known_sigma = None

    if st.button("実行", key="compare_run"):
        try:
            if uploaded_pair is not None:
                df = data_loader.read_csv(uploaded_pair)
                if not {"before", "after"}.issubset(set(df.columns)):
                    raise ValueError("アップロードCSVには 'before' と 'after' 列が必要です（テンプレを使用してください）。")
                b = pd.to_numeric(df["before"], errors="coerce")
                a = pd.to_numeric(df["after"], errors="coerce")
            else:
                b = _load_numeric(before_csv, before_text)
                a = _load_numeric(after_csv, after_text)

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

            st.markdown("#### 使用検定法")
            st.write(result.test_name)

            st.markdown("#### 分布の比較")
            dist = visualization.before_after_distribution_plot(b, a, bins=20)
            st.image(dist.png_bytes, caption="改善前後の分布比較", output_format="PNG")

            st.markdown("#### 数値結果")

            if result.method.startswith("t_"):
                st.table(
                    pd.DataFrame(
                        {
                            "検定種別": ["両側" if result.tail == "two-sided" else "片側（改善方向）"],
                            "t算出値（t_stat）": [result.t_stat],
                            "自由度（df）": [result.df],
                            "臨界値（t_crit）": [result.t_crit],
                            "有意水準": [result.alpha],
                            "判定": [result.decision],
                        }
                    )
                )
                st.markdown("#### 判定（QC文脈）")
                st.text(result.judgement)
            else:
                st.table(
                    pd.DataFrame(
                        {
                            "統計量": [result.statistic],
                            "p値": [result.p_value],
                            "有意水準": [result.alpha],
                            "判定": [result.decision],
                        }
                    )
                )

            st.markdown("#### 付帯情報")
            st.table(pd.DataFrame([result.details]))

            if use_iqr:
                st.caption(f"IQR外れ値除去: before removed={out_b.removed_count}, after removed={out_a.removed_count}")

            st.markdown("#### 解釈（品質改善の観点）")
            st.write(result.comment)

        except Exception as e:
            st.error(str(e))


def ui_doe() -> None:
    st.subheader("④ 実験計画法（DOE）")

    k = st.number_input("因子数", min_value=1, max_value=20, value=3, step=1)

    st.caption("各因子の水準数を入力してください（例: 2,2,2）。")
    levels_text = st.text_input("水準数（カンマ区切り）", value=",".join(["2"] * int(k)))

    method = st.selectbox("設計方法", ["直交表（タグチ簡易）", "フルファクタリアル"], index=0)

    if st.button("生成", key="doe_run"):
        try:
            factor_levels = [int(x.strip()) for x in levels_text.split(",") if x.strip()]
            if len(factor_levels) != int(k):
                raise ValueError("因子数と水準数の個数が一致していません。")

            if method.startswith("直交表"):
                res = doe_engine.taguchi_orthogonal_array(factor_levels)
            else:
                res = doe_engine.full_factorial(factor_levels)

            st.markdown("#### 実験計画表")
            st.dataframe(res.design, use_container_width=True)

            st.markdown("#### 指標")
            st.table(
                pd.DataFrame(
                    {
                        "設計法": [res.method],
                        "実験回数": [res.runs],
                        "フルファクタリアル回数": [res.full_factorial_runs],
                        "実験効率指標（full/runs）": [res.efficiency_ratio],
                        "削減率": [res.reduction_ratio],
                    }
                )
            )
            st.write(res.comment)

            csv_bytes = res.design.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "CSVダウンロード",
                data=csv_bytes,
                file_name="doe_design.csv",
                mime="text/csv",
            )

            st.info(
                "ダウンロードしたCSVに、実験結果（応答）列を追加して保存してください。"
                "例: 列名を 'Y' として各行に測定値を入力 → 下の『結果CSVをインポートして分析』でアップロード。"
            )

        except Exception as e:
            st.error(str(e))

    st.divider()
    st.markdown("#### 結果CSVをインポートして分析")
    uploaded = st.file_uploader("実験結果CSV（因子列 + 応答列）", type=["csv"], key="doe_result_csv")

    if uploaded is not None:
        try:
            df = data_loader.read_csv(uploaded)
            st.caption(f"読み込み: {df.shape[0]}行 × {df.shape[1]}列")
            st.dataframe(df.head(20), use_container_width=True)

            if df.shape[1] < 2:
                raise ValueError("因子列と応答列を含むCSVが必要です（列が足りません）。")

            # 応答列候補: 数値に変換して1つでも値が入る列
            numeric_candidates = []
            for c in df.columns:
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().any():
                    numeric_candidates.append(c)

            default_col = "Y" if "Y" in df.columns else (numeric_candidates[0] if numeric_candidates else df.columns[-1])
            response_col = st.selectbox("応答列（実験値）", options=list(df.columns), index=list(df.columns).index(default_col))
            objective_label = st.selectbox("目的", ["大きいほど良い（最大化）", "小さいほど良い（最小化）"], index=0)
            objective = "maximize" if objective_label.startswith("大きい") else "minimize"

            if st.button("分析", key="doe_analyze"):
                res = doe_engine.analyze_doe_results(df, response_col=response_col, objective=objective)

                st.markdown("##### 因子サマリ（効果幅が大きい順）")
                st.table(res.factor_summary)

                st.markdown("##### 主効果（因子×水準の平均）")
                st.dataframe(res.main_effects_long, use_container_width=True)

                fig = visualization.doe_main_effects_plot(res.main_effects_long)
                st.image(fig.png_bytes, caption="DOE 主効果プロット", output_format="PNG")

                st.markdown("##### 解釈")
                st.write(res.recommendation)

        except Exception as e:
            st.error(str(e))


def ui_capability(missing_policy: str, use_iqr: bool) -> None:
    st.subheader("⑤ 工程能力指数算出（Cp, Cpk, Pp, Ppk）")

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("CSVアップロード（1列目を使用）", type=["csv"], key="cap_csv")
    with col2:
        text = st.text_area("手入力（数値を改行/カンマ区切り）", height=180, key="cap_text")

    lsl = st.number_input("規格下限 LSL", value=0.0, step=0.1, format="%.6g")
    usl = st.number_input("規格上限 USL", value=1.0, step=0.1, format="%.6g")

    st.markdown("##### Cpk（短期: within）用サブグループ設定")
    subgroup_size = st.number_input(
        "サブグループサイズ n（例: 5）",
        min_value=0,
        value=0,
        step=1,
        help="CpkはX-R管理図の考え方で σ_within = R̄/d2 を用います。サブグループ（同条件で短時間に採取したn個）を作るのが前提です。",
    )

    st.markdown("##### サブグループ入力用CSV（テンプレ）")
    subgroup_count = st.number_input(
        "サブグループ数（行数）",
        min_value=0,
        value=0,
        step=1,
        help="サブグループサイズnと合わせて、入力用CSVテンプレを生成します（例: n=5、サブグループ数=10）。",
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
            "テンプレCSVをダウンロード",
            data=template_bytes,
            file_name=f"capability_subgroup_template_n{n}_k{k}.csv",
            mime="text/csv",
            help="ダウンロード後、各行（サブグループ）にx1..xnの測定値を入力して保存し、下でインポートしてください。",
        )

    st.divider()
    st.markdown("#### 解析（入力済みCSVをインポートして実行）")
    cap_mode = st.radio(
        "入力方法",
        ["サブグループCSV（推奨: X-R管理図でCpk算出）", "単票データ（簡易）"],
        index=0,
        horizontal=False,
    )

    subgroup_csv = None
    if cap_mode.startswith("サブグループ"):
        subgroup_csv = st.file_uploader(
            "入力済みサブグループCSV（列: subgroup, x1..xn）",
            type=["csv"],
            key="cap_subgroup_csv",
        )

    if st.button("算出", key="cap_run"):
        try:
            subgroup_df = None
            out = None

            if cap_mode.startswith("サブグループ"):
                if subgroup_csv is None:
                    raise ValueError("サブグループCSVをアップロードしてください。")
                if int(subgroup_size) <= 0:
                    raise ValueError("サブグループサイズ n を指定してください（例: 5）。")

                df = data_loader.read_csv(subgroup_csv)
                n = int(subgroup_size)
                expected_cols = ["subgroup"] + [f"x{i+1}" for i in range(n)]
                missing = [c for c in expected_cols if c not in df.columns]
                if missing:
                    raise ValueError(f"サブグループCSVの列が不足しています: {missing}")

                subgroup_df = df[[f"x{i+1}" for i in range(n)]].copy()
                # overall用に全データをフラット化
                flat = pd.to_numeric(subgroup_df.stack(), errors="coerce")
                s = flat
            else:
                s = _load_numeric(uploaded, text)
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

            st.markdown("#### 判定（優先順位: 管理状態 → Ppk → Cpk）")
            st.table(
                pd.DataFrame(
                    {
                        "ステータス": [res.status],
                        "判定": [res.level],
                        "管理限界逸脱(out_of_control)": [res.out_of_control],
                        "n": [res.n],
                        "平均": [res.mean],
                    }
                )
            )

            if res.out_of_control is True:
                st.warning(
                    "管理図で逸脱があるため工程は不安定です。Cpk/Ppkの解釈は無効として、安定化を優先してください。"
                )

            st.markdown("#### Cpk（短期: within / X-R管理図）")
            st.table(
                pd.DataFrame(
                    {
                        "サブグループn": [res.subgroup_size],
                        "R̄": [res.rbar],
                        "σ_within（R̄/d2）": [res.sigma_within],
                        "Cp（within）": [res.cp],
                        "Cpk（within）": [res.cpk],
                    }
                )
            )

            st.markdown("#### Ppk（長期: overall / 分布ベース）")
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
                        "標準偏差(s) [overall]": [res.std_overall_s],
                        "σ_overall（χ²ベース）": [res.sigma_overall],
                        "σ_overall CI下限": [res.sigma_overall_ci_low],
                        "σ_overall CI上限": [res.sigma_overall_ci_high],
                        "Pp（overall）": [res.pp],
                        "Ppk（overall）": [res.ppk],
                        "Pp（σ上限参考）": [pp_sigma_high],
                        "Ppk（σ上限参考）": [ppk_sigma_high],
                    }
                )
            )

            if out is not None and out.enabled:
                st.caption(f"IQR外れ値除去: removed={out.removed_count}")

            st.markdown("#### 図示")
            # X-R管理図（サブグループCSVがある場合のみ）
            if subgroup_df is not None:
                xr = capability_engine.xr_chart_from_subgroups(subgroup_df, subgroup_size=int(subgroup_size))
                xr_fig = visualization.xr_control_chart_plot(xr)
                st.image(xr_fig.png_bytes, caption="X-R管理図", output_format="PNG")

                st.markdown("##### 管理限界値（算出結果）")
                st.table(
                    pd.DataFrame(
                        {
                            "定数": ["d2", "A2", "D3", "D4"],
                            "値": [xr.d2, xr.a2, xr.d3, xr.d4],
                        }
                    )
                )
                st.table(
                    pd.DataFrame(
                        {
                            "管理図": ["X\u0305", "X\u0305", "X\u0305", "R", "R", "R"],
                            "種別": ["UCL", "CL", "LCL", "UCL", "CL", "LCL"],
                            "値": [xr.ucl_x, xr.cl_x, xr.lcl_x, xr.ucl_r, xr.cl_r, xr.lcl_r],
                        }
                    )
                )
            else:
                st.info("X-R管理図はサブグループCSV入力時のみ表示します。")

            # Ppk用 分布図（overall）
            dist_fig = visualization.ppk_distribution_plot(
                series=s,
                lsl=float(lsl),
                usl=float(usl),
                mean=float(res.mean),
                sigma_overall=float(res.sigma_overall),
                bins=20,
            )
            st.image(dist_fig.png_bytes, caption="Ppk用 分布図（overall）", output_format="PNG")

            st.markdown("#### 工程能力評価コメント")
            st.write(res.comment)

        except Exception as e:
            st.error(str(e))


def _load_numeric(uploaded, text: str) -> pd.Series:
    if uploaded is not None:
        df = data_loader.read_csv(uploaded)
        if df.shape[1] < 1:
            raise ValueError("CSVに列がありません。")
        s = df.iloc[:, 0]
        return pd.to_numeric(s, errors="coerce")
    return data_loader.parse_numeric_series(text)


if __name__ == "__main__":
    main()
