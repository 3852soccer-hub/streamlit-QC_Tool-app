from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutlierReport:
    enabled: bool
    removed_count: int
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


def read_csv(uploaded_file) -> pd.DataFrame:
    """StreamlitのUploadedFileからDataFrameを読む（UTF-8/Shift-JIS等を簡易対応）。"""
    if uploaded_file is None:
        raise ValueError("CSVファイルが選択されていません。")

    raw = uploaded_file.getvalue()
    if not raw:
        raise ValueError("CSVファイルが空です。")

    # ざっくりエンコーディング試行
    for enc in ("utf-8", "utf-8-sig", "cp932"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception:
            continue
    # 最後はpandasに任せる
    return pd.read_csv(io.BytesIO(raw))


_NUM_SPLIT_RE = re.compile(r"[\s,;\t]+")


def parse_numeric_series(text: str) -> pd.Series:
    """テキストから数値列を抽出。区切り：改行/空白/カンマ/セミコロン/タブ。"""
    if text is None or not str(text).strip():
        raise ValueError("数値データが空です。")

    parts = [p for p in _NUM_SPLIT_RE.split(str(text).strip()) if p]
    values: list[float] = []
    bad: list[str] = []
    for p in parts:
        try:
            values.append(float(p))
        except Exception:
            bad.append(p)

    if bad:
        raise ValueError(f"数値として解釈できない値があります: {bad[:5]}{'…' if len(bad) > 5 else ''}")

    s = pd.Series(values, dtype="float64")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        raise ValueError("有効な数値が見つかりませんでした。")
    return s


def parse_category_count(text: str) -> pd.DataFrame:
    """カテゴリ+件数の入力をDataFrameにする。

    1行=1カテゴリ。形式例：
      A,10
      B,5
    区切りはカンマ/タブ/複数空白を許容。
    """
    if text is None or not str(text).strip():
        raise ValueError("カテゴリ+件数の入力が空です。")

    rows: list[Tuple[str, float]] = []
    for i, line in enumerate(str(text).splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        # 先頭から2要素だけ取る
        parts = re.split(r"[\t,]+|\s{2,}", line)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) < 2:
            raise ValueError(f"{i}行目: 'カテゴリ,件数' の形式で入力してください: {line}")
        category = parts[0]
        try:
            count = float(parts[1])
        except Exception as e:
            raise ValueError(f"{i}行目: 件数が数値ではありません: {parts[1]}") from e
        if count < 0:
            raise ValueError(f"{i}行目: 件数が負です: {count}")
        rows.append((category, count))

    if not rows:
        raise ValueError("有効な行がありません。")

    df = pd.DataFrame(rows, columns=["category", "count"])
    return df


def apply_missing_policy(series: pd.Series, policy: str) -> pd.Series:
    """欠損値の扱い。policy: drop / mean / median"""
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if policy == "drop":
        s = s.dropna()
    elif policy == "mean":
        s = s.fillna(s.mean())
    elif policy == "median":
        s = s.fillna(s.median())
    else:
        raise ValueError("欠損値処理の指定が不正です。")
    return s


def iqr_outlier_filter(series: pd.Series, enabled: bool) -> tuple[pd.Series, OutlierReport]:
    """IQR法による外れ値除去（任意）。"""
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if not enabled:
        return s, OutlierReport(enabled=False, removed_count=0)

    if len(s) < 4:
        # IQRが安定しないため何もしない
        return s, OutlierReport(enabled=True, removed_count=0)

    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0:
        return s, OutlierReport(enabled=True, removed_count=0)

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (s >= lower) & (s <= upper)
    removed = int((~mask).sum())
    filtered = s.loc[mask]

    logger.info("IQR外れ値除去: removed=%s, lower=%.6g, upper=%.6g", removed, lower, upper)
    return filtered, OutlierReport(enabled=True, removed_count=removed, lower_bound=lower, upper_bound=upper)
