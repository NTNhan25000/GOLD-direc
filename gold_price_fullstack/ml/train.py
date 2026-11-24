"""Huấn luyện và đánh giá 4 mô hình dự báo giá vàng phiên kế tiếp.

Chạy từ thư mục gold_price_fullstack:
    python ml/train.py                     # quét K đầy đủ (1..500) trên validation
    python ml/train.py --fast              # dùng K đã chọn sẵn từ báo cáo (RF=11, XGB=165)
    python ml/train.py --data data/raw/gold_forecast_dataset.csv

Đầu ra:
    data/processed/{train,val,test}.csv    # tập dữ liệu đã chia theo thời gian
    models/*.joblib                        # mô hình + scaler + imputer
    models/metadata.json                   # siêu thông tin mô hình
    outputs/test_metrics.csv               # MAE/RMSE/MAPE/R2 trên test
    outputs/test_predictions.csv           # dự báo chi tiết từng phiên test
    outputs/k_validation_scan.csv          # kết quả quét K trên validation
    outputs/*.png                          # biểu đồ
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.features import MACRO, TARGET, make_features, scores, split_masks

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PRESET_K = {"RandomForest": 11, "XGBoost": 165}  # K tốt nhất theo validation trong báo cáo
RF_PARAMS = dict(max_features=0.8, min_samples_leaf=2, random_state=42, n_jobs=-1)
XGB_PARAMS = dict(max_depth=3, learning_rate=0.03, subsample=0.8,
                  colsample_bytree=0.8, reg_lambda=1,
                  objective="reg:squarederror", random_state=42, n_jobs=-1)


def parse_args():
    ap = argparse.ArgumentParser(description="Huấn luyện mô hình dự báo giá vàng GC=F")
    ap.add_argument("--data", default=str(ROOT / "data" / "raw" / "gold_forecast_dataset.csv"))
    ap.add_argument("--out", default=str(ROOT))
    ap.add_argument("--fast", action="store_true",
                    help="Bỏ qua quét K, dùng K preset từ báo cáo (RF=11, XGB=165)")
    ap.add_argument("--k-max", type=int, default=500, help="Giá trị K lớn nhất khi quét")
    return ap.parse_args()


def fit_final(model_name: str, k: int, X, y):
    if model_name == "RandomForest":
        return RandomForestRegressor(n_estimators=k, **RF_PARAMS).fit(X, y)
    if model_name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=k, **XGB_PARAMS).fit(X, y, verbose=False)
    raise ValueError(model_name)


def main():
    args = parse_args()
    t0 = time.time()
    out = Path(args.out)
    data_dir = out / "data"
    proc_dir = data_dir / "processed"
    models_dir = out / "models"
    outputs_dir = out / "outputs"
    for d in (proc_dir, models_dir, outputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ Dữ liệu
    raw = pd.read_csv(args.data, parse_dates=["Date"]).set_index("Date").sort_index()
    print(f"[data] {args.data}: {len(raw)} phiên, {raw.index.min().date()} -> {raw.index.max().date()}")

    X, y, target_dates = make_features(raw)
    masks = split_masks(target_dates)
    print(f"[features] {X.shape[1]} đặc trưng, {len(X)} mẫu có nhãn")

    # Lưu các tập đã chia (theo ngày giá mục tiêu) để dùng lại / kiểm chứng.
    for name, mask in masks.items():
        df = X.loc[mask].copy()
        df["target_next_close"] = y.loc[mask]
        df["target_date"] = target_dates.loc[mask].dt.strftime("%Y-%m-%d")
        df.index.name = "Date"
        df.to_csv(proc_dir / f"{name}.csv")
        print(f"[split] {name}: {int(mask.sum())} mẫu  "
              f"({target_dates[mask].min().date()} -> {target_dates[mask].max().date()})")

    train_m, val_m, test_m = masks["train"], masks["val"], masks["test"]
    if not (train_m.any() and val_m.any() and test_m.any()):
        raise ValueError("Không đủ dữ liệu cho train/validation/test.")

    # Imputer chỉ fit trên Train (chống rò rỉ).
    imputer = SimpleImputer(strategy="median")
    Xt = imputer.fit_transform(X.loc[train_m])
    Xv = imputer.transform(X.loc[val_m])
    Xtest = imputer.transform(X.loc[test_m])
    yt, yv, ytest = y.loc[train_m], y.loc[val_m], y.loc[test_m]

    # ------------------------------------------------------------------ Naive
    result = [{"model": "Naive", "K": None,
               **scores(ytest, X.loc[test_m, "close_lag_0"].to_numpy())}]

    # ---------------------------------------------------------- Linear Regression
    linear = make_pipeline(StandardScaler(), LinearRegression()).fit(Xt, yt)
    result.append({"model": "LinearRegression", "K": None,
                   **scores(ytest, linear.predict(Xtest))})
    joblib.dump(linear, models_dir / "linear_regression.joblib")

    # ------------------------------------------------- Quét / chọn K trên Validation
    k_rows: list[dict] = []
    best_k: dict[str, int] = {}

    if args.fast:
        best_k = dict(PRESET_K)
        print(f"[K] --fast: dùng K preset {best_k}")
    else:
        print(f"[K] quét K = 1..{args.k_max} trên validation (chọn theo RMSE thấp nhất)...")
        for model_name in ("RandomForest", "XGBoost"):
            local_best = None
            for k in range(1, args.k_max + 1):
                m = fit_final(model_name, k, Xt, yt)
                s = scores(yv, m.predict(Xv))
                k_rows.append({"model": model_name, "K": k, **s})
                if local_best is None or s["RMSE"] < local_best[0]:
                    local_best = (s["RMSE"], k)
            best_k[model_name] = local_best[1]
            print(f"  {model_name}: K tốt nhất = {local_best[1]} "
                  f"(val RMSE = {local_best[0]:.3f})")
        pd.DataFrame(k_rows).to_csv(outputs_dir / "k_validation_scan.csv", index=False)

    # ------------------------------------- Fit mô hình cây trên Train+Val với K đã chọn
    Xtv = np.vstack([Xt, Xv])
    ytv = pd.concat([yt, yv])
    fitted: dict[str, object] = {}
    for model_name, k in best_k.items():
        fitted[model_name] = fit_final(model_name, k, Xtv, ytv)
        joblib.dump(fitted[model_name], models_dir / f"{model_name.lower()}.joblib")
        result.append({"model": model_name, "K": k,
                       **scores(ytest, fitted[model_name].predict(Xtest))})

    # ------------------------------------------------------------------ Kết quả
    metrics_df = pd.DataFrame(result)
    metrics_df.to_csv(outputs_dir / "test_metrics.csv", index=False)
    print("\n=== Kết quả trên Test ===")
    print(metrics_df.to_string(index=False))

    # Dự báo chi tiết từng phiên Test của cả 4 mô hình.
    preds = pd.DataFrame({
        "ForecastOrigin": X.index[test_m],
        "Date": target_dates.loc[test_m].dt.strftime("%Y-%m-%d").to_numpy(),
        "Actual": ytest.to_numpy(),
        "Naive": X.loc[test_m, "close_lag_0"].to_numpy(),
        "LinearRegression": linear.predict(Xtest),
    })
    for model_name, m in fitted.items():
        preds[model_name] = m.predict(Xtest)
    preds.to_csv(outputs_dir / "test_predictions.csv", index=False)

    # Feature importance (mô hình cây) — top 15.
    if "RandomForest" in fitted:
        imp = pd.Series(fitted["RandomForest"].feature_importances_, index=X.columns)
        imp.sort_values(ascending=False).head(15).to_csv(
            outputs_dir / "feature_importance_rf.csv", header=["importance"])

    # ------------------------------------------------------------------ Lưu models
    joblib.dump(imputer, models_dir / "imputer.joblib")
    metadata = {
        "target": TARGET,
        "features": list(X.columns),
        "macro_columns": MACRO,
        "split": {"train_end": "2021-12-31", "val_end": "2023-12-31",
                  "test": "2024-01-01 onward"},
        "n_samples": {"train": int(train_m.sum()), "val": int(val_m.sum()),
                      "test": int(test_m.sum())},
        "best_k": best_k,
        "test_metrics": {r["model"]: {k: r[k] for k in ("MAE", "RMSE", "MAPE_pct", "R2")}
                         for r in result},
        "data_file": str(args.data),
        "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_seconds": round(time.time() - t0, 1),
    }
    (models_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------ Biểu đồ
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(preds["Date"], preds["Actual"], label="Thực tế", lw=1.6, color="#1f77b4")
        ax.plot(preds["Date"], preds["Naive"], label="Naive", lw=1.0, alpha=0.8)
        ax.plot(preds["Date"], preds["LinearRegression"], label="Linear Regression", lw=1.0, alpha=0.8)
        if "RandomForest" in preds:
            ax.plot(preds["Date"], preds["RandomForest"], label="Random Forest", lw=1.0, alpha=0.8)
        if "XGBoost" in preds:
            ax.plot(preds["Date"], preds["XGBoost"], label="XGBoost", lw=1.0, alpha=0.8)
        ax.set_title("Giá thực tế vs. dự báo — tập Test 2024 (USD/troy ounce)")
        ax.set_xlabel("Ngày của giá mục tiêu")
        ax.set_ylabel("USD/oz")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(outputs_dir / "test_predictions_plot.png", dpi=150)
        plt.close(fig)
        print(f"\n[plot] đã lưu {outputs_dir / 'test_predictions_plot.png'}")
    except Exception as e:  # matplotlib optional
        print(f"[plot] bỏ qua biểu đồ: {e}")

    print(f"\n[done] toàn bộ pipeline mất {time.time() - t0:.1f}s")
    print(f"[done] mô hình trong {models_dir}/, kết quả trong {outputs_dir}/")


if __name__ == "__main__":
    main()
