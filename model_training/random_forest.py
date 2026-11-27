"""MÔ HÌNH 3/4 — Random Forest (bagging, giảm phương sai).

Siêu tham số cố định (không tinh chỉnh trên Test):
    max_features = 0.8 · min_samples_leaf = 2 · random_state = 42 · n_jobs = -1
Chỉ **số cây K được quét 1..500 trên Validation**, chọn theo RMSE thấp nhất → K = 11.

Fit cuối cùng trên **Train + Validation** (vì K đã được chọn trên Val).

Chạy:
    python model_training/random_forest.py             # quét K 1..500 (~3 giây)
    python model_training/random_forest.py --fast      # dùng K = 11, khỏi quét
    python model_training/random_forest.py --save      # lưu artifacts/randomforest.joblib

Dùng như thư viện:
    from random_forest import RandomForestModel
    m = RandomForestModel(k=11).fit(X_train, y_train)
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

from sklearn.ensemble import RandomForestRegressor  # noqa: E402

NAME = "RandomForest"
LABEL = "Random Forest"
PRESET_K = 11

# Siêu tham số cố định trong báo cáo — đổi ở đây là đổi toàn bộ kết luận.
RF_PARAMS = dict(max_features=0.8, min_samples_leaf=2, random_state=42, n_jobs=-1)


class RandomForestModel:
    """Rừng ngẫu nhiên với siêu tham số của báo cáo; chỉ K là biến thiên."""

    name = NAME

    def __init__(self, k: int = PRESET_K, **overrides):
        self.k = int(k)
        self.params = {**RF_PARAMS, **overrides, "n_estimators": self.k}
        self.model = RandomForestRegressor(**self.params)

    def fit(self, X, y) -> "RandomForestModel":
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    @property
    def importances(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_,
                         index=self.model.feature_names_in_).sort_values(ascending=False)


def scan_k(X, y, target_dates, k_max: int = 500) -> pd.DataFrame:
    """Quét K = 1..k_max trên Validation, không fit lại k_max lần.

    Cây thứ 1..k của RandomForest(n_estimators=k_max, random_state=42) **chính là**
    rừng k cây fit riêng biệt (sai khác < 1e-13, có test riêng). Vì vậy chỉ cần fit
    một lần rồi lấy trung bình tích lũy trên estimators_ → 500 K chỉ mất vài giây.
    """
    masks = split_masks(target_dates)
    # Dùng numpy (không phải DataFrame) khi quét: từng cây con bên trong rừng không
    # lưu feature names, truyền DataFrame vào sẽ sinh UserWarning cho mỗi cây.
    Xt = X.loc[masks["train"]].to_numpy()
    Xv = X.loc[masks["val"]].to_numpy()
    yv = y.loc[masks["val"]]

    scan = RandomForestRegressor(n_estimators=k_max, **RF_PARAMS).fit(
        Xt, y.loc[masks["train"]].to_numpy())
    cum = (np.cumsum([est.predict(Xv) for est in scan.estimators_], axis=0)
           / np.arange(1, k_max + 1)[:, None])
    return pd.DataFrame([
        {"K": k, **scores(yv, cum[k - 1])} for k in range(1, k_max + 1)
    ])


def best_k_from_scan(scan: pd.DataFrame) -> tuple[int, float]:
    i = scan["RMSE"].idxmin()
    return int(scan.loc[i, "K"]), float(scan.loc[i, "RMSE"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Random Forest (bagging)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--fast", action="store_true", help="Dùng K preset (11) thay vì quét")
    ap.add_argument("--k", type=int, default=None, help="Ép một giá trị K cụ thể")
    ap.add_argument("--k-max", type=int, default=500)
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--out", default=str(HERE / "outputs"))
    args = ap.parse_args()

    raw = load_raw(args.data)
    X, y, target_dates = make_features(raw)
    masks = split_masks(target_dates)

    # ---------------------------------------------------------------- chọn K
    if args.k is not None:
        k, val_rmse = args.k, float("nan")
        print(f"[K] ép K = {k}")
    elif args.fast:
        k, val_rmse = PRESET_K, float("nan")
        print(f"[K] --fast: dùng K = {PRESET_K}")
    else:
        print(f"[K] quét K = 1..{args.k_max} trên Validation...")
        scan = scan_k(X, y, target_dates, args.k_max)
        k, val_rmse = best_k_from_scan(scan)
        print(f"[K] K tốt nhất = {k} (Val RMSE {val_rmse:.3f})")
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        scan.to_csv(out / "k_scan_randomforest.csv", index=False)

    # ------------------------------------------- fit trên Train+Val, đánh giá Test
    Xtv = pd.concat([X.loc[masks["train"]], X.loc[masks["val"]]])
    ytv = pd.concat([y.loc[masks["train"]], y.loc[masks["val"]]])
    model = RandomForestModel(k=k).fit(Xtv, ytv)

    ytest = y.loc[masks["test"]]
    pred = model.predict(X.loc[masks["test"]])
    s = scores(ytest, pred)

    print("=" * 74)
    print(f" {LABEL}  —  {k} cây · max_features {RF_PARAMS['max_features']} · "
          f"min_samples_leaf {RF_PARAMS['min_samples_leaf']}")
    print("=" * 74)
    print(f" Fit trên Train+Val ({len(Xtv)} mẫu) · Test {int(masks['test'].sum())} mẫu")
    print(f"\n Test 2024:  MAE {s['MAE']:.3f} · RMSE {s['RMSE']:.3f} · "
          f"MAPE {s['MAPE_pct']:.3f}% · R² {s['R2']:.4f}")
    print(f" Dự báo cao nhất trên Test: {pred.max():,.2f} USD "
          f"(giá thực cao nhất {ytest.max():,.2f}) ← dấu hiệu chặn ngoại suy")

    print("\n Top 5 đặc trưng theo MDI:")
    for feat, v in model.importances.head(5).items():
        print(f"   {feat:26s} {v * 100:>6.1f}%")

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
            "RandomForest": pred,
        }).to_csv(out / "predictions_randomforest.csv", index=False)
        print(f"\n [csv] {out / 'predictions_randomforest.csv'}")

    if args.save:
        p = HERE / "artifacts" / "randomforest.joblib"
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model.model, p)
        print(f" [joblib] {p}")


if __name__ == "__main__":
    main()
