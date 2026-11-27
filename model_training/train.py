"""Huấn luyện 4 mô hình dự báo giá vàng GC=F phiên kế tiếp — bản copy độc lập.

Folder này tự chạy được một mình: chỉ cần `data/gold_forecast_dataset.csv` và
`features.py` nằm cùng thư mục. Không import gì từ `gold_price_fullstack/`.

Nguồn gốc: chép từ `gold_price_fullstack/ml/train.py` (cùng siêu tham số, cùng
cách chia tập, cùng K preset) để kết quả tái lập đúng từng con số.

Chạy:
    python model_training/train.py               # quét K đầy đủ 1..500 trên Validation (chậm hơn)
    python model_training/train.py --fast        # dùng K preset RF=11, XGB=165 (~6 giây)
    python model_training/train.py --out /tmp/run

Đầu ra (mặc định trong model_training/outputs/):
    artifacts/*.joblib          imputer, linear_regression, randomforest, xgboost
    artifacts/metadata.json     danh sách đặc trưng, mốc chia tập, metric Test, K tốt nhất
    outputs/test_metrics.csv    MAE / RMSE / MAPE / R² trên Test 2024
    outputs/splits.csv          số mẫu từng tập để đối chiếu
    outputs/k_validation_scan.csv   (chỉ khi quét K đầy đủ)
    outputs/predictions_test.csv    dự báo từng phiên Test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from features import MACRO, TARGET, make_features, scores, split_masks  # noqa: E402

from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# K tốt nhất theo Validation trong báo cáo — dùng cho --fast.
PRESET_K = {"RandomForest": 11, "XGBoost": 165}

RF_PARAMS = dict(max_features=0.8, min_samples_leaf=2, random_state=42, n_jobs=-1)
XGB_PARAMS = dict(max_depth=3, learning_rate=0.03, subsample=0.8,
                  colsample_bytree=0.8, reg_lambda=1,
                  objective="reg:squarederror", random_state=42, n_jobs=-1)


def parse_args():
    ap = argparse.ArgumentParser(description="Huấn luyện mô hình dự báo giá vàng GC=F")
    ap.add_argument("--data", default=str(HERE / "data" / "gold_forecast_dataset.csv"))
    ap.add_argument("--out", default=str(HERE), help="Thư mục gốc để ghi artifacts/ và outputs/")
    ap.add_argument("--fast", action="store_true",
                    help="Bỏ quét K, dùng K preset từ báo cáo (RF=11, XGB=165)")
    ap.add_argument("--k-max", type=int, default=500, help="K lớn nhất khi quét")
    return ap.parse_args()


def fit_final(model_name: str, k: int, X, y):
    """Fit mô hình cây cuối cùng trên Train+Val với đúng K đã chọn."""
    if model_name == "RandomForest":
        return RandomForestRegressor(n_estimators=k, **RF_PARAMS).fit(X, y)
    if model_name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=k, **XGB_PARAMS).fit(X, y, verbose=False)
    raise ValueError(model_name)


def main() -> None:
    args = parse_args()
    t0 = time.time()
    out = Path(args.out)
    artifacts_dir = out / "artifacts"
    outputs_dir = out / "outputs"
    for d in (artifacts_dir, outputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ Dữ liệu
    raw = pd.read_csv(args.data, parse_dates=["Date"]).set_index("Date").sort_index()
    print(f"[data] {args.data}\n       {len(raw)} phiên, "
          f"{raw.index.min().date()} -> {raw.index.max().date()}")

    X, y, target_dates = make_features(raw)
    masks = split_masks(target_dates)
    print(f"[features] {X.shape[1]} đặc trưng, {len(X)} mẫu có nhãn")
    for name, mask in masks.items():
        print(f"[split] {name:5s}: {int(mask.sum()):4d} mẫu")

    train_m, val_m, test_m = masks["train"], masks["val"], masks["test"]
    if not (train_m.any() and val_m.any() and test_m.any()):
        raise ValueError("Không đủ dữ liệu cho train/validation/test.")

    pd.DataFrame([
        {"split": k, "n_samples": int(v.sum()),
         "target_from": target_dates[v].min().date().isoformat(),
         "target_to": target_dates[v].max().date().isoformat()}
        for k, v in masks.items()
    ]).to_csv(outputs_dir / "splits.csv", index=False)

    # Imputer CHỈ fit trên Train — chống rò rỉ dữ liệu Val/Test.
    imputer = SimpleImputer(strategy="median")
    Xt = imputer.fit_transform(X.loc[train_m])
    Xv = imputer.transform(X.loc[val_m])
    Xtest = imputer.transform(X.loc[test_m])
    yt, yv, ytest = y.loc[train_m], y.loc[val_m], y.loc[test_m]

    result: list[dict] = []

    # ------------------------------------------------------------------ 1. Naive
    naive_pred = X.loc[test_m, "close_lag_0"].to_numpy()
    result.append({"model": "Naive", "K": None, **scores(ytest, naive_pred)})

    # -------------------------------------------------- 2. Linear Regression (OLS)
    linear = make_pipeline(StandardScaler(), LinearRegression()).fit(Xt, yt)
    joblib.dump(linear, artifacts_dir / "linear_regression.joblib")
    lr_pred = linear.predict(Xtest)
    result.append({"model": "LinearRegression", "K": None, **scores(ytest, lr_pred)})

    # ---------------------------------------- 3+4. Quét K trên Validation
    k_rows: list[dict] = []
    best_k: dict[str, int] = {}

    if args.fast:
        best_k = dict(PRESET_K)
        print(f"[K] --fast: dùng K preset {best_k}")
    else:
        print(f"[K] quét K = 1..{args.k_max} trên Validation (chọn theo RMSE thấp nhất)")

        # Tăng tốc: k cây đầu của RandomForest(n_estimators=k_max, seed 42) chính là
        # rừng k cây fit riêng → lấy trung bình tích lũy trên estimators_.
        rf_scan = RandomForestRegressor(n_estimators=args.k_max, **RF_PARAMS).fit(Xt, yt)
        rf_cum = (np.cumsum([est.predict(Xv) for est in rf_scan.estimators_], axis=0)
                  / np.arange(1, args.k_max + 1)[:, None])
        rf_best = min(((scores(yv, rf_cum[k - 1])["RMSE"], k)
                       for k in range(1, args.k_max + 1)), key=lambda t: t[0])
        for k in range(1, args.k_max + 1):
            k_rows.append({"model": "RandomForest", "K": k,
                           **scores(yv, rf_cum[k - 1])})
        best_k["RandomForest"] = rf_best[1]
        print(f"  RandomForest: K = {rf_best[1]} (val RMSE {rf_best[0]:.3f})")

        # XGBoost: cùng ý tưởng với iteration_range=(0, k) trên booster fit k_max vòng.
        import xgboost as xgb
        from xgboost import XGBRegressor

        xb_scan = XGBRegressor(n_estimators=args.k_max, **XGB_PARAMS).fit(Xt, yt, verbose=False)
        booster = xb_scan.get_booster()
        dval = xgb.DMatrix(Xv, feature_names=list(X.columns))
        xb_best = None
        for k in range(1, args.k_max + 1):
            s = scores(yv, booster.predict(dval, iteration_range=(0, k)))
            k_rows.append({"model": "XGBoost", "K": k, **s})
            if xb_best is None or s["RMSE"] < xb_best[0]:
                xb_best = (s["RMSE"], k)
        best_k["XGBoost"] = xb_best[1]
        print(f"  XGBoost: K = {xb_best[1]} (val RMSE {xb_best[0]:.3f})")

        pd.DataFrame(k_rows).to_csv(outputs_dir / "k_validation_scan.csv", index=False)

    # --------------------------- Fit 2 mô hình cây trên Train+Val, đánh giá trên Test
    Xtv = np.vstack([Xt, Xv])
    ytv = pd.concat([yt, yv])
    fitted: dict[str, object] = {}
    for model_name, k in best_k.items():
        fitted[model_name] = fit_final(model_name, k, Xtv, ytv)
        joblib.dump(fitted[model_name], artifacts_dir / f"{model_name.lower()}.joblib")
        result.append({"model": model_name, "K": k,
                       **scores(ytest, fitted[model_name].predict(Xtest))})

    metrics_df = pd.DataFrame(result)
    metrics_df.to_csv(outputs_dir / "test_metrics.csv", index=False)
    print("\n=== Kết quả trên Test 2024 ===")
    print(metrics_df.to_string(index=False))

    # ------------------------------------------------- Dự báo chi tiết từng phiên
    preds = pd.DataFrame({
        "ForecastOrigin": X.index[test_m],
        "Date": target_dates.loc[test_m].dt.strftime("%Y-%m-%d").to_numpy(),
        "Actual": ytest.to_numpy(),
        "Naive": naive_pred,
        "LinearRegression": lr_pred,
    })
    for model_name, m in fitted.items():
        preds[model_name] = m.predict(Xtest)
    preds.to_csv(outputs_dir / "predictions_test.csv", index=False)

    # -------------------------------------------------- Độ quan trọng đặc trưng
    if "RandomForest" in fitted:
        pd.DataFrame({
            "feature": list(X.columns),
            "MDI_pct": fitted["RandomForest"].feature_importances_ * 100.0,
        }).sort_values("MDI_pct", ascending=False).to_csv(
            outputs_dir / "feature_importance_rf.csv", index=False)
    if "XGBoost" in fitted:
        import xgboost as xgb
        bst = fitted["XGBoost"].get_booster()
        bst.feature_names = list(X.columns)
        gain = bst.get_score(importance_type="gain")
        dtest = xgb.DMatrix(Xtest, feature_names=list(X.columns))
        shap = bst.predict(dtest, pred_contribs=True)[:, :-1]
        imp = pd.DataFrame({
            "feature": list(X.columns),
            "gain": [gain.get(f, 0.0) for f in X.columns],
            "mean_abs_shap_test_USD": np.abs(shap).mean(axis=0),
        })
        imp["gain_pct"] = 100.0 * imp["gain"] / imp["gain"].sum()
        imp.sort_values("gain", ascending=False).to_csv(
            outputs_dir / "feature_importance_xgb.csv", index=False)

    # ------------------------------------------------------ Lưu imputer + metadata
    joblib.dump(imputer, artifacts_dir / "imputer.joblib")
    metadata = {
        "target": TARGET,
        "features": list(X.columns),
        "macro_columns": MACRO,
        "split": {"train_end": "2021-12-31", "val_end": "2023-12-31",
                  "test": "2024-01-01 onward"},
        "n_samples": {"train": int(train_m.sum()), "val": int(val_m.sum()),
                      "test": int(test_m.sum())},
        "n_sessions": int(len(raw)),
        "data_range": {"start": raw.index.min().strftime("%Y-%m-%d"),
                       "end": raw.index.max().strftime("%Y-%m-%d")},
        "best_k": best_k,
        "test_metrics": {r["model"]: {c: r[c] for c in ("MAE", "RMSE", "MAPE_pct", "R2")}
                         for r in result},
        "data_file": str(args.data),
        "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_seconds": round(time.time() - t0, 1),
    }
    (artifacts_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[done] {time.time() - t0:.1f}s · model trong {artifacts_dir}/ · "
          f"kết quả trong {outputs_dir}/")


if __name__ == "__main__":
    main()
