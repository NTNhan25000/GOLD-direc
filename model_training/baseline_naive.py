"""MÔ HÌNH 1/4 — Naive Baseline (mốc cơ sở bắt buộc).

    ŷ(t+1) = Close(t)

Không huấn luyện, không tham số. Với dự báo 1 phiên, giá hai phiên liên tiếp rất gần
nhau nên đây là **baseline rất mạnh** — mọi mô hình học máy phải vượt được nó mới chứng
minh có giá trị. Trên Test 2024 nó thắng cả 3 mô hình còn lại.

Chạy:
    python model_training/baseline_naive.py            # in bảng Test + dự báo phiên kế tiếp
    python model_training/baseline_naive.py --csv      # kèm dự báo từng phiên ra CSV

Dùng như thư viện:
    from baseline_naive import NaiveBaseline
    m = NaiveBaseline()
    m.fit(X_train, y_train)         # no-op, giữ cho API giống 3 mô hình kia
    m.predict(X_test)               # = X_test["close_lag_0"]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from features import (  # noqa: E402
    latest_features, load_raw, make_features, scores, split_masks,
)

NAME = "Naive"
LABEL = "Naive Baseline"
LAG_COLUMN = "close_lag_0"      # đặc trưng duy nhất mô hình này dùng


class NaiveBaseline:
    """ŷ(t+1) = Close(t). Giữ API fit/predict để so sánh công bằng với 3 mô hình kia."""

    name = NAME
    uses_features = [LAG_COLUMN]

    def fit(self, X, y=None):                     # noqa: ARG002 - không học gì
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(X[self.uses_features[0]], dtype=float)

    @property
    def params(self) -> dict:
        return {"strategy": "giữ nguyên giá hiện tại", "n_params": 0, "lags": [0]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Naive Baseline — ŷ(t+1) = Close(t)")
    ap.add_argument("--data", default=None, help="CSV thô (mặc định data/gold_forecast_dataset.csv)")
    ap.add_argument("--csv", action="store_true", help="Xuất dự báo từng phiên Test ra CSV")
    ap.add_argument("--out", default=str(HERE / "outputs"))
    args = ap.parse_args()

    raw = load_raw(args.data)
    X, y, target_dates = make_features(raw)
    masks = split_masks(target_dates)

    model = NaiveBaseline().fit(X.loc[masks["train"]], y.loc[masks["train"]])
    ytest = y.loc[masks["test"]]
    pred = model.predict(X.loc[masks["test"]])
    s = scores(ytest, pred)

    print("=" * 74)
    print(f" {LABEL}  —  {model.params['strategy']}  (0 tham số, không huấn luyện)")
    print("=" * 74)
    print(f" Train {int(masks['train'].sum())} · Val {int(masks['val'].sum())} · "
          f"Test {int(masks['test'].sum())} mẫu")
    print(f"\n Test 2024:  MAE {s['MAE']:.3f} · RMSE {s['RMSE']:.3f} · "
          f"MAPE {s['MAPE_pct']:.3f}% · R² {s['R2']:.4f}")

    last = latest_features(raw)          # phiên cuối, chưa có nhãn t+1
    print(f"\n Dự báo phiên sau {raw.index[-1].date()}: "
          f"{model.predict(last)[0]:,.2f} USD/oz  (giá hiện tại {raw['Gold_Close'].iloc[-1]:,.2f})")

    if args.csv:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "ForecastOrigin": X.index[masks["test"]],
            "Date": target_dates.loc[masks["test"]].dt.strftime("%Y-%m-%d"),
            "Actual": ytest.to_numpy(),
            "Naive": pred,
        }).to_csv(out / "predictions_naive.csv", index=False)
        print(f"\n [csv] {out / 'predictions_naive.csv'}")


if __name__ == "__main__":
    main()
