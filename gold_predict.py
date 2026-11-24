"""Dự báo giá đóng cửa GC=F phiên kế tiếp theo quy trình trong báo cáo.

Chạy:
    python gold_predict.py [--data gold_forecast_dataset_yfinance.csv] [--fast]
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "Gold_Close"
MACRO = ["DXY_Close", "US10Y_Close", "SP500_Close", "WTI_Close", "Silver_Close"]
PRESET_K = {"RandomForest": 11, "XGBoost": 165}


def make_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Mỗi dòng X(t) chỉ dùng quan sát đã biết cuối phiên t; y(t)=Close(t+1)."""
    d = raw.sort_index().copy()
    close = d[TARGET]
    x = pd.DataFrame(index=d.index)
    for lag in (0, 1, 2, 3, 5, 10):
        x[f"close_lag_{lag}"] = close.shift(lag)
    for window in (5, 10, 20):
        x[f"sma_{window}"] = close.rolling(window).mean()
        x[f"volatility_{window}"] = close.pct_change().rolling(window).std()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    x["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    x["gold_return_1d"] = close.pct_change()
    x["gold_range"] = (d["Gold_High"] - d["Gold_Low"]) / close
    for col in MACRO:
        if col in d.columns:
            x[col] = d[col]
            x[f"{col}_return_1d"] = d[col].pct_change()
    y = close.shift(-1).rename("target_next_close")
    valid = y.notna()
    target_dates = pd.Series(d.index, index=d.index).shift(-1).loc[valid]
    return x.loc[valid], y.loc[valid], target_dates


def scores(y, pred) -> dict[str, float]:
    y_arr = np.asarray(y, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_arr, pred_arr)),
        "RMSE": float(np.sqrt(mean_squared_error(y_arr, pred_arr))),
        "MAPE_pct": float(np.mean(np.abs((y_arr - pred_arr) / y_arr)) * 100),
        "R2": float(r2_score(y_arr, pred_arr)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline dự báo giá vàng GC=F phiên kế tiếp")
    ap.add_argument("--data", default=str(REPO_ROOT / "gold_forecast_dataset_yfinance.csv"))
    ap.add_argument("--out", default=str(REPO_ROOT / "gold_model_output"))
    ap.add_argument("--fast", action="store_true",
                    help="Dùng K đã chọn sẵn trên Validation (RF=11, XGB=165) thay vì quét 1..500")
    args = ap.parse_args()

    raw = pd.read_csv(args.data, parse_dates=["Date"]).set_index("Date").sort_index()
    X, y, target_dates = make_features(raw)

    # Chia theo ngày đích thực tế (phiên giao dịch tiếp theo), không giả định BDay.
    train = target_dates <= pd.Timestamp("2021-12-31")
    val = (target_dates > pd.Timestamp("2021-12-31")) & (target_dates <= pd.Timestamp("2023-12-31"))
    test = target_dates > pd.Timestamp("2023-12-31")

    # Loại bỏ đầu chuỗi do rolling features và ngày thiếu.
    ready = X.notna().all(axis=1) & y.notna()
    train &= ready
    val &= ready
    test &= ready
    if not train.any() or not val.any() or not test.any():
        raise ValueError("Không đủ dữ liệu cho train/validation/test.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    imputer = SimpleImputer(strategy="median")
    Xt = imputer.fit_transform(X.loc[train])
    Xv = imputer.transform(X.loc[val])
    Xtest = imputer.transform(X.loc[test])
    yt, yv, ytest = y.loc[train], y.loc[val], y.loc[test]
    Xtv = np.vstack([Xt, Xv])
    ytv = pd.concat([yt, yv])

    result: list[dict] = []
    k_rows: list[dict] = []
    test_preds: dict[str, np.ndarray] = {}

    # 1. Naive baseline: close(t) dự báo close(t+1)
    naive_pred = X.loc[test, "close_lag_0"].to_numpy()
    test_preds["Naive"] = naive_pred
    result.append({"model": "Naive", **scores(ytest, naive_pred), "K_trees": np.nan})

    # 2. OLS: scaler chỉ fit trên Train, rồi đánh giá trên Test.
    linear = make_pipeline(StandardScaler(), LinearRegression()).fit(Xt, yt)
    lr_pred = linear.predict(Xtest)
    test_preds["LinearRegression"] = lr_pred
    result.append({"model": "LinearRegression", **scores(ytest, lr_pred), "K_trees": np.nan})

    # 3. Random Forest: quét K = 1..500 trên Validation, chọn RMSE thấp nhất.
    # Tiền tố k cây đầu của RandomForest(n_estimators=500, random_state=42) trùng khớp
    # tuyệt đối với mô hình huấn luyện riêng k cây, giúp tăng tốc quét ~180 lần.
    if args.fast:
        rf_best_k = PRESET_K["RandomForest"]
    else:
        rf_scan = RandomForestRegressor(
            n_estimators=500, max_features=0.8, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        ).fit(Xt, yt)
        tree_preds = np.array([est.predict(Xv) for est in rf_scan.estimators_])
        cum_preds = np.cumsum(tree_preds, axis=0) / np.arange(1, 501)[:, None]
        rf_best = None
        for k in range(1, 501):
            s = scores(yv, cum_preds[k - 1])
            k_rows.append({"model": "RandomForest", "K_trees": k, **s})
            if rf_best is None or s["RMSE"] < rf_best[0]:
                rf_best = (s["RMSE"], k)
        rf_best_k = rf_best[1]

    rf = RandomForestRegressor(
        n_estimators=rf_best_k, max_features=0.8, min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    ).fit(Xtv, ytv)
    rf_pred = rf.predict(Xtest)
    test_preds["RandomForest"] = rf_pred
    result.append({"model": "RandomForest", **scores(ytest, rf_pred), "K_trees": float(rf_best_k)})

    # 4. XGBoost: quét K = 1..500 vòng boosting trên Validation, chọn RMSE thấp nhất.
    try:
        import xgboost as xgb
        from xgboost import XGBRegressor

        if args.fast:
            xb_best_k = PRESET_K["XGBoost"]
        else:
            xb_scan = XGBRegressor(
                n_estimators=500, max_depth=3, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1,
                objective="reg:squarederror", random_state=42, n_jobs=-1,
            ).fit(Xt, yt, verbose=False)
            booster = xb_scan.get_booster()
            dval = xgb.DMatrix(Xv)
            xb_best = None
            for k in range(1, 501):
                s = scores(yv, booster.predict(dval, iteration_range=(0, k)))
                k_rows.append({"model": "XGBoost", "K_trees": k, **s})
                if xb_best is None or s["RMSE"] < xb_best[0]:
                    xb_best = (s["RMSE"], k)
            xb_best_k = xb_best[1]

        xb = XGBRegressor(
            n_estimators=xb_best_k, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1,
            objective="reg:squarederror", random_state=42, n_jobs=-1,
        ).fit(Xtv, ytv, verbose=False)
        xb_pred = xb.predict(Xtest)
        test_preds["XGBoost"] = xb_pred
        result.append({"model": "XGBoost", **scores(ytest, xb_pred), "K_trees": float(xb_best_k)})
    except ImportError:
        print("Bỏ qua XGBoost: cài bằng pip install xgboost")

    # Lưu kết quả đánh giá
    metrics_df = pd.DataFrame(result)[["model", "MAE", "RMSE", "MAPE_pct", "R2", "K_trees"]]
    metrics_df.to_csv(out / "test_metrics.csv", index=False)
    if k_rows:
        pd.DataFrame(k_rows).to_csv(out / "k_validation_test.csv", index=False)

    # Dự báo chi tiết từng phiên Test
    target_dt = pd.to_datetime(target_dates.loc[test])
    all_preds_df = pd.DataFrame({
        "ForecastOrigin": X.index[test].strftime("%Y-%m-%d"),
        "Date": target_dt.dt.strftime("%Y-%m-%d").to_numpy(),
        "Actual": ytest.to_numpy(),
        **test_preds,
    })
    all_preds_df.to_csv(out / "test_predictions.csv", index=False)

    all_with_year = all_preds_df.copy()
    all_with_year["Year"] = target_dt.dt.year.to_numpy()
    all_with_year.to_csv(out / "all_model_test_predictions.csv", index=False)

    # Bảng chỉ số theo năm của ngày giá mục tiêu
    annual_rows = []
    for yr, grp in all_with_year.groupby("Year"):
        for m_name in test_preds:
            sc = scores(grp["Actual"], grp[m_name])
            annual_rows.append({
                "Year": int(yr),
                "Model": m_name,
                "N": len(grp),
                "MAE": sc["MAE"],
                "RMSE": sc["RMSE"],
                "MAPE": sc["MAPE_pct"],
                "R2": sc["R2"],
            })
    pd.DataFrame(annual_rows).to_csv(out / "annual_test_metrics.csv", index=False)

    pd.DataFrame([
        {"method": "Naive baseline", "K": "N/A (không huấn luyện cây)"},
        {"method": "Linear Regression", "K": "N/A (không có siêu tham số số cây)"},
        {"method": "Random Forest", "K": "1-500 trees; chọn theo validation RMSE"},
        {"method": "XGBoost", "K": "1-500 boosting rounds; chọn theo validation RMSE"},
    ]).to_csv(out / "K_test_plan.csv", index=False)

    print(metrics_df.to_string(index=False))
    print(f"\nĐã lưu báo cáo và dự báo trong {out}/")


if __name__ == "__main__":
    main()
