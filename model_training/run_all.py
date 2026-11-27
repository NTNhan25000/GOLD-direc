"""Chạy cả 4 mô hình trong một bảng so sánh — dùng để đối chiếu chéo.

Mỗi mô hình vẫn nằm trong file riêng (baseline_naive.py, linear_regression.py,
random_forest.py, xgboost_model.py); file này chỉ gọi lại chúng cho tiện so sánh.

Chạy:
    python model_training/run_all.py              # quét K đầy đủ
    python model_training/run_all.py --fast       # dùng K preset RF=11, XGB=165
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from baseline_naive import NaiveBaseline  # noqa: E402
from features import latest_features, load_raw, make_features, scores, split_masks  # noqa: E402
from linear_regression import LinearRegressionModel  # noqa: E402
from random_forest import best_k_from_scan as rf_best_k, scan_k as rf_scan  # noqa: E402
from random_forest import RandomForestModel, PRESET_K as RF_PRESET  # noqa: E402
from xgboost_model import XGBoostModel  # noqa: E402
from xgboost_model import best_k_from_scan as xgb_best_k, scan_k as xgb_scan  # noqa: E402
from xgboost_model import PRESET_K as XGB_PRESET  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="So sánh 4 mô hình")
    ap.add_argument("--data", default=None)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--k-max", type=int, default=500)
    args = ap.parse_args()

    t0 = time.time()
    raw = load_raw(args.data)
    X, y, target_dates = make_features(raw)
    m = split_masks(target_dates)
    print(f"[data] {len(raw)} phiên · {X.shape[1]} đặc trưng · "
          f"train/val/test = {int(m['train'].sum())}/{int(m['val'].sum())}/{int(m['test'].sum())}")

    # ---------------------------------------------------------------- chọn K
    if args.fast:
        k_rf, k_xgb = RF_PRESET, XGB_PRESET
        print(f"[K] --fast: RF={k_rf}, XGB={k_xgb}")
    else:
        print("[K] quét trên Validation...")
        k_rf, rmse_rf = rf_best_k(rf_scan(X, y, target_dates, args.k_max))
        k_xgb, rmse_xgb = xgb_best_k(xgb_scan(X, y, target_dates, args.k_max))
        print(f"[K] RF = {k_rf} (Val RMSE {rmse_rf:.3f}) · XGB = {k_xgb} (Val RMSE {rmse_xgb:.3f})")

    # ---------------------------------------------------------------- fit 4 mô hình
    Xtr, ytr = X.loc[m["train"]], y.loc[m["train"]]
    Xtv = pd.concat([X.loc[m["train"]], X.loc[m["val"]]])
    ytv = pd.concat([y.loc[m["train"]], y.loc[m["val"]]])
    Xte, yte = X.loc[m["test"]], y.loc[m["test"]]

    models = {
        "Naive": NaiveBaseline().fit(Xtr, ytr),
        # LR chỉ dùng Train; RF/XGB fit lại trên Train+Val vì K đã chọn trên Val.
        "LinearRegression": LinearRegressionModel().fit(Xtr, ytr),
        "RandomForest": RandomForestModel(k=k_rf).fit(Xtv, ytv),
        "XGBoost": XGBoostModel(k=k_xgb).fit(Xtv, ytv),
    }
    fitted_on = {"Naive": "không học", "LinearRegression": "Train",
                 "RandomForest": "Train+Val", "XGBoost": "Train+Val"}
    k_col = {"Naive": "—", "LinearRegression": "—",
             "RandomForest": str(k_rf), "XGBoost": str(k_xgb)}

    rows, next_rows = [], []
    last = latest_features(raw)
    for name, model in models.items():
        s = scores(yte, model.predict(Xte))
        rows.append({"model": name, "K": k_col[name], "fit_on": fitted_on[name], **s})
        next_rows.append({"model": name, "next_close": float(model.predict(last)[0])})

    tbl = pd.DataFrame(rows)
    print("\n=== Test 2024 ===")
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:10.4f}"))

    nxt = pd.DataFrame(next_rows)
    print(f"\n=== Dự báo phiên sau {raw.index[-1].date()} ===")
    print(nxt.to_string(index=False, float_format=lambda v: f"{v:10.2f}"))
    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
