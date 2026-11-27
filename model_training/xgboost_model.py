"""MÔ HÌNH 4/4 — XGBoost (gradient boosting, giảm độ chệch).

Siêu tham số cố định (không tinh chỉnh trên Test):
    max_depth = 3 · learning_rate = 0.03 · subsample = 0.8
    colsample_bytree = 0.8 · reg_lambda = 1 · objective = reg:squarederror
    random_state = 42
Chỉ **số vòng boosting K được quét 1..500 trên Validation** → K = 165.
(Cần tới 165 vòng vì learning rate thấp: tại K = 100 RMSE còn 49,78 USD, chưa hội tụ.)

Fit cuối cùng trên **Train + Validation**.

Chạy:
    python model_training/xgboost_model.py            # quét K 1..500 (~3 giây)
    python model_training/xgboost_model.py --fast     # dùng K = 165
    python model_training/xgboost_model.py --save     # lưu artifacts/xgboost.joblib

Dùng như thư viện:
    from xgboost_model import XGBoostModel
    m = XGBoostModel(k=165).fit(X_train, y_train)
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

from xgboost import XGBRegressor  # noqa: E402

NAME = "XGBoost"
LABEL = "XGBoost"
PRESET_K = 165

XGB_PARAMS = dict(max_depth=3, learning_rate=0.03, subsample=0.8,
                  colsample_bytree=0.8, reg_lambda=1,
                  objective="reg:squarederror", random_state=42, n_jobs=-1)


class XGBoostModel:
    """Gradient boosting với siêu tham số của báo cáo; chỉ K (số vòng) biến thiên."""

    name = NAME

    def __init__(self, k: int = PRESET_K, **overrides):
        self.k = int(k)
        self.params = {**XGB_PARAMS, **overrides, "n_estimators": self.k}
        self.model = XGBRegressor(**self.params)

    def fit(self, X, y) -> "XGBoostModel":
        self.model.fit(X, y, verbose=False)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    @property
    def importances(self) -> pd.Series:
        """Gain % đọc thẳng từ booster — cùng thang với xgb_feature_importance.csv."""
        booster = self.model.get_booster()
        gain = booster.get_score(importance_type="gain")
        total = sum(gain.values()) or 1.0
        names = booster.feature_names or list(self.model.feature_names_in_)
        return (pd.Series({f: gain.get(f, 0.0) / total * 100 for f in names})
                .sort_values(ascending=False))


def scan_k(X, y, target_dates, k_max: int = 500) -> pd.DataFrame:
    """Quét số vòng boosting 1..k_max trên Validation.

    Fit một booster k_max vòng rồi dùng `iteration_range=(0, k)` — cho đúng kết quả
    của mô hình fit k vòng, nên 500 giá trị K chỉ tốn một lần fit.
    """
    import xgboost as xgb

    masks = split_masks(target_dates)
    Xv = X.loc[masks["val"]]
    yv = y.loc[masks["val"]]

    booster = XGBRegressor(n_estimators=k_max, **XGB_PARAMS).fit(
        X.loc[masks["train"]], y.loc[masks["train"]], verbose=False).get_booster()
    dval = xgb.DMatrix(Xv, feature_names=list(X.columns))

    rows = [{"K": k, **scores(yv, booster.predict(dval, iteration_range=(0, k)))}
            for k in range(1, k_max + 1)]
    return pd.DataFrame(rows)


def best_k_from_scan(scan: pd.DataFrame) -> tuple[int, float]:
    i = scan["RMSE"].idxmin()
    return int(scan.loc[i, "K"]), float(scan.loc[i, "RMSE"])


def main() -> None:
    ap = argparse.ArgumentParser(description="XGBoost (gradient boosting)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--fast", action="store_true", help="Dùng K preset (165) thay vì quét")
    ap.add_argument("--k", type=int, default=None, help="Ép một giá trị K cụ thể")
    ap.add_argument("--k-max", type=int, default=500)
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--out", default=str(HERE / "outputs"))
    args = ap.parse_args()

    raw = load_raw(args.data)
    X, y, target_dates = make_features(raw)
    masks = split_masks(target_dates)

    if args.k is not None:
        k, val_rmse = args.k, float("nan")
        print(f"[K] ép K = {k}")
    elif args.fast:
        k, val_rmse = PRESET_K, float("nan")
        print(f"[K] --fast: dùng K = {PRESET_K}")
    else:
        print(f"[K] quét K = 1..{args.k_max} vòng boosting trên Validation...")
        scan = scan_k(X, y, target_dates, args.k_max)
        k, val_rmse = best_k_from_scan(scan)
        print(f"[K] K tốt nhất = {k} (Val RMSE {val_rmse:.3f})")
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        scan.to_csv(out / "k_scan_xgboost.csv", index=False)

    Xtv = pd.concat([X.loc[masks["train"]], X.loc[masks["val"]]])
    ytv = pd.concat([y.loc[masks["train"]], y.loc[masks["val"]]])
    model = XGBoostModel(k=k).fit(Xtv, ytv)

    ytest = y.loc[masks["test"]]
    pred = model.predict(X.loc[masks["test"]])
    s = scores(ytest, pred)

    print("=" * 74)
    print(f" {LABEL}  —  {k} vòng · depth {XGB_PARAMS['max_depth']} · "
          f"lr {XGB_PARAMS['learning_rate']} · subsample {XGB_PARAMS['subsample']}")
    print("=" * 74)
    print(f" Fit trên Train+Val ({len(Xtv)} mẫu) · Test {int(masks['test'].sum())} mẫu")
    print(f"\n Test 2024:  MAE {s['MAE']:.3f} · RMSE {s['RMSE']:.3f} · "
          f"MAPE {s['MAPE_pct']:.3f}% · R² {s['R2']:.4f}")
    print(f" Dự báo cao nhất trên Test: {pred.max():,.2f} USD "
          f"(giá thực cao nhất {ytest.max():,.2f}) ← dấu hiệu chặn ngoại suy")

    print("\n Top 5 đặc trưng theo Gain %:")
    for feat, v in model.importances.head(5).items():
        print(f"   {feat:26s} {v:>6.1f}%")

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
            "XGBoost": pred,
        }).to_csv(out / "predictions_xgboost.csv", index=False)
        print(f"\n [csv] {out / 'predictions_xgboost.csv'}")

    if args.save:
        p = HERE / "artifacts" / "xgboost.joblib"
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model.model, p)
        print(f" [joblib] {p}")


if __name__ == "__main__":
    main()
