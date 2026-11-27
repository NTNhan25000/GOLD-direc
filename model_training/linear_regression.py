"""MÔ HÌNH 2/4 — Linear Regression (hồi quy tuyến tính, mốc diễn giải được).

    ŷ(t+1) = β₀ + Σ βᵢ · xᵢ(t)

Pipeline: StandardScaler → LinearRegression. **Scaler chỉ fit trên Train** (chống rò rỉ),
và nằm BÊN TRONG Pipeline nên khi dự báo không được chuẩn hoá thêm lần nữa.

Không quét K (không có siêu tham số số cây). Fit trên Train, đánh giá trên Test.

Chạy:
    python model_training/linear_regression.py           # bảng Test + 10 hệ số lớn nhất
    python model_training/linear_regression.py --csv     # kèm dự báo từng phiên
    python model_training/linear_regression.py --save    # lưu artifacts/linear_regression.joblib

Dùng như thư viện:
    from linear_regression import LinearRegressionModel
    m = LinearRegressionModel().fit(X_train, y_train)
    m.predict(X_test)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from features import (  # noqa: E402
    latest_features, load_raw, make_features, scores, split_masks,
)

from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

NAME = "LinearRegression"
LABEL = "Linear Regression"
# Siêu tham số: dùng mặc định của sklearn, không tinh chỉnh (để so sánh công bằng).
PARAMS = dict(fit_intercept=True)


class LinearRegressionModel:
    """Bọc sklearn trong Pipeline(scaler → OLS) để không ai chuẩn hoá thiếu/thừa."""

    name = NAME
    params = {**PARAMS, "scaler": "StandardScaler (fit trên Train)"}

    def __init__(self, **overrides):
        self.model = Pipeline([
            ("standardscaler", StandardScaler()),
            ("linearregression", LinearRegression(**{**PARAMS, **overrides})),
        ])
        self.feature_names: list[str] | None = None

    def fit(self, X: pd.DataFrame, y) -> "LinearRegressionModel":
        self.feature_names = list(X.columns)
        self.model.fit(X, y)          # scaler.fit_transform bên trong, chỉ thấy Train
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # KHÔNG transform thủ công: scaler đã ở trong Pipeline.
        return self.model.predict(X[self.feature_names])

    # ---------------------------------------------------------------- diễn giải
    @property
    def coef(self) -> pd.Series:
        """Hệ số trên thang đã chuẩn hoá → so sánh được độ lớn giữa các biến."""
        return pd.Series(self.model.named_steps["linearregression"].coef_,
                         index=self.feature_names).sort_values(key=np.abs, ascending=False)

    @property
    def intercept(self) -> float:
        return float(self.model.named_steps["linearregression"].intercept_)


def main() -> None:
    ap = argparse.ArgumentParser(description="Linear Regression (OLS có chuẩn hoá)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--save", action="store_true", help="Lưu artifacts/linear_regression.joblib")
    ap.add_argument("--out", default=str(HERE / "outputs"))
    args = ap.parse_args()

    raw = load_raw(args.data)
    X, y, target_dates = make_features(raw)
    masks = split_masks(target_dates)

    model = LinearRegressionModel().fit(X.loc[masks["train"]], y.loc[masks["train"]])
    ytest = y.loc[masks["test"]]
    pred = model.predict(X.loc[masks["test"]])
    s = scores(ytest, pred)

    print("=" * 74)
    print(f" {LABEL}  —  Pipeline(StandardScaler → OLS), 25 hệ số + intercept")
    print("=" * 74)
    print(f" Siêu tham số: fit_intercept=True, không tinh chỉnh")
    print(f" Train {int(masks['train'].sum())} · Test {int(masks['test'].sum())} mẫu")
    print(f"\n Test 2024:  MAE {s['MAE']:.3f} · RMSE {s['RMSE']:.3f} · "
          f"MAPE {s['MAPE_pct']:.3f}% · R² {s['R2']:.4f}")
    print(f" Intercept: {model.intercept:,.3f}")

    print("\n 10 hệ số lớn nhất (thang đã chuẩn hoá):")
    for feat, c in model.coef.head(10).items():
        print(f"   {feat:26s} {c:>10.3f}")

    last = latest_features(raw)
    print(f"\n Dự báo phiên sau {raw.index[-1].date()}: "
          f"{model.predict(last)[0]:,.2f} USD/oz")

    if args.csv:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "ForecastOrigin": X.index[masks["test"]],
            "Date": target_dates.loc[masks["test"]].dt.strftime("%Y-%m-%d"),
            "Actual": ytest.to_numpy(),
            "LinearRegression": pred,
        }).to_csv(out / "predictions_linear.csv", index=False)
        print(f"\n [csv] {out / 'predictions_linear.csv'}")

    if args.save:
        p = HERE / "artifacts" / "linear_regression.joblib"
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model.model, p)        # lưu Pipeline, không lưu wrapper
        print(f" [joblib] {p}")


if __name__ == "__main__":
    main()
