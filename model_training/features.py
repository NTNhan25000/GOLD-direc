"""Kỹ nghệ đặc trưng cho dự báo giá đóng cửa GC=F phiên kế tiếp.

Mỗi dòng X(t) chỉ dùng thông tin đã biết đến cuối phiên t;
mục tiêu y(t) = Close(t+1). Tuân thủ chống rò rỉ dữ liệu (no leakage).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "Gold_Close"
MACRO = ["DXY_Close", "US10Y_Close", "SP500_Close", "WTI_Close", "Silver_Close"]
LAGS = (0, 1, 2, 3, 5, 10)
WINDOWS = (5, 10, 20)
RSI_WINDOW = 14

# Mốc chia tập theo NGÀY CỦA GIÁ MỤC TIÊU (phiên t+1), không shuffle.
TRAIN_END = pd.Timestamp("2021-12-31")
VAL_END = pd.Timestamp("2023-12-31")

FEATURE_COLUMNS: list[str] = []


def make_features(
    raw: pd.DataFrame, drop_unlabeled: bool = True
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Trả về (X, y, target_dates).

    - X: đặc trưng tại phiên t (chỉ dùng dữ liệu <= t).
    - y: giá đóng cửa phiên kế tiếp (t+1).
    - target_dates: ngày của giá mục tiêu, dùng để chia tập theo thời gian.

    drop_unlabeled=False: giữ phiên CUỐI CÙNG (chưa có nhãn t+1) — dùng khi
    suy luận: đặc trưng cuối chính là những gì biết được đến hết phiên hiện tại.
    Khi đó y/target_dates của hàng cuối là NaN.
    """
    d = raw.sort_index().copy()
    close = d[TARGET]
    x = pd.DataFrame(index=d.index)

    for lag in LAGS:
        x[f"close_lag_{lag}"] = close.shift(lag)
    for w in WINDOWS:
        x[f"sma_{w}"] = close.rolling(w).mean()
        x[f"volatility_{w}"] = close.pct_change().rolling(w).std()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(RSI_WINDOW).mean()
    loss = -delta.clip(upper=0).rolling(RSI_WINDOW).mean()
    x["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    x["gold_return_1d"] = close.pct_change()
    x["gold_range"] = (d["Gold_High"] - d["Gold_Low"]) / close

    for col in MACRO:
        if col in d.columns:
            x[col] = d[col]
            x[f"{col}_return_1d"] = d[col].pct_change()

    y = close.shift(-1).rename("target_next_close")
    target_dates = pd.Series(d.index, index=d.index).shift(-1).rename("target_date")

    if drop_unlabeled:
        valid = y.notna()
        x, y, target_dates = x.loc[valid], y.loc[valid], target_dates.loc[valid]
        # Loại hàng đầu chuỗi do rolling/RSI cần lịch sử.
        ready = x.notna().all(axis=1)
        x, y, target_dates = x.loc[ready], y.loc[ready], target_dates.loc[ready]
    # drop_unlabeled=False: giữ nguyên mọi hàng (hàng đầu có thể còn NaN do
    # rolling — tầng imputer/median sẽ xử lý khi suy luận).

    global FEATURE_COLUMNS
    FEATURE_COLUMNS = list(x.columns)
    return x, y, target_dates


def split_masks(target_dates: pd.Series) -> dict[str, pd.Series]:
    """Chia tập theo thứ tự thời gian, căn theo ngày giá mục tiêu."""
    return {
        "train": target_dates <= TRAIN_END,
        "val": (target_dates > TRAIN_END) & (target_dates <= VAL_END),
        "test": target_dates > VAL_END,
    }


def scores(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE, MAPE (%), R2 — cùng định nghĩa với báo cáo."""
    from sklearn.metrics import mean_absolute_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "MAPE_pct": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "R2": float(r2_score(y_true, y_pred)),
    }


DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "gold_forecast_dataset.csv"


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Đọc CSV thô, index theo ngày, sắp tăng dần (cùng định dạng với lúc huấn luyện)."""
    p = Path(path) if path else DEFAULT_DATA
    return pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()


def latest_features(raw: pd.DataFrame, close_override: float | None = None) -> pd.DataFrame:
    """Hàng đặc trưng của PHIÊN GẦN NHẤT để dự báo phiên kế tiếp.

    Phải dùng drop_unlabeled=False: nếu dùng drop_unlabeled=True thì phiên cuối cùng
    bị loại (vì chưa có nhãn t+1) và ta sẽ dự báo lùi mất một phiên — kết quả lệch
    khoảng 15–20 USD so với đúng.

    close_override: thay giá đóng cửa của phiên cuối bằng số nhập tay; High/Low được
    nới ra để bao lấy giá mới (nếu không gold_range sẽ ra số vô nghĩa).
    """
    base = raw.copy()
    if close_override is not None:
        last = base.index[-1]
        base.at[last, TARGET] = float(close_override)
        base.at[last, "Gold_High"] = max(float(base.at[last, "Gold_High"]), close_override)
        base.at[last, "Gold_Low"] = min(float(base.at[last, "Gold_Low"]), close_override)
    X, _, _ = make_features(base, drop_unlabeled=False)
    return X.iloc[[-1]]
