"""Dự báo giá đóng cửa GC=F phiên kế tiếp theo quy trình trong báo cáo.
Chạy: python gold_predict.py [--data gold_forecast_dataset_yfinance.csv]
"""
from __future__ import annotations
import argparse
from pathlib import Path
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


def make_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Mỗi dòng X(t) chỉ dùng quan sát đã biết cuối phiên t; y(t)=Close(t+1)."""
    d = raw.copy()
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
        if col in d:
            x[col] = d[col]
            x[f"{col}_return_1d"] = d[col].pct_change()
    y = close.shift(-1).rename("target_next_close")
    valid = y.notna()
    target_dates = pd.Series(raw.index, index=raw.index).shift(-1).loc[valid]
    return x.loc[valid], y.loc[valid], target_dates


def scores(y, pred):
    return {"MAE": mean_absolute_error(y, pred),
            "RMSE": np.sqrt(mean_squared_error(y, pred)),
            "MAPE_pct": np.mean(np.abs((np.asarray(y)-pred)/np.asarray(y))) * 100,
            "R2": r2_score(y, pred)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="gold_forecast_dataset_yfinance.csv")
    ap.add_argument("--out", default="gold_model_output")
    args = ap.parse_args()
    raw = pd.read_csv(args.data, parse_dates=["Date"]).set_index("Date").sort_index()
    X, y, target_dates = make_features(raw)
    # Chia theo ngày đích thực tế (phiên giao dịch tiếp theo), không giả định BDay
    # vì ngày lễ và lịch giao dịch không phải lúc nào cũng là ngày làm việc chuẩn.
    train = target_dates <= pd.Timestamp("2021-12-31")
    val = (target_dates > pd.Timestamp("2021-12-31")) & (target_dates <= pd.Timestamp("2023-12-31"))
    test = target_dates > pd.Timestamp("2023-12-31")
    # Loại bỏ đầu chuỗi do rolling features và ngày thiếu.
    ready = X.notna().all(axis=1) & y.notna()
    train &= ready; val &= ready; test &= ready
    if not train.any() or not val.any() or not test.any():
        raise ValueError("Không đủ dữ liệu cho train/validation/test.")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    imputer = SimpleImputer(strategy="median")
    Xt = imputer.fit_transform(X.loc[train]); Xv = imputer.transform(X.loc[val]); Xtest = imputer.transform(X.loc[test])
    yt, yv, ytest = y.loc[train], y.loc[val], y.loc[test]
    result, k_rows = [], []
    # Naive baseline: close(t) dự báo close(t+1)
    result.append({"model":"Naive", **scores(ytest, X.loc[test,"close_lag_0"].to_numpy())})
    # OLS: scaler chỉ fit trên train, rồi đánh giá trên test.
    linear = make_pipeline(StandardScaler(), LinearRegression()).fit(Xt, yt)
    result.append({"model":"LinearRegression", **scores(ytest, linear.predict(Xtest))})
    # Quét K cây trên Validation; lựa chọn K nhỏ nhất khi đồng hạng.
    rf_best = None
    for k in range(1, 501):
        m = RandomForestRegressor(n_estimators=k, max_features=0.8, min_samples_leaf=2,
                                  random_state=42, n_jobs=-1).fit(Xt, yt)
        s = scores(yv, m.predict(Xv)); k_rows.append({"model":"RandomForest", "K_trees":k, **s})
        if rf_best is None or s["RMSE"] < rf_best[0]: rf_best = (s["RMSE"], k)
    rf = RandomForestRegressor(n_estimators=rf_best[1], max_features=.8, min_samples_leaf=2,
                               random_state=42, n_jobs=-1).fit(np.vstack([Xt,Xv]), pd.concat([yt,yv]))
    result.append({"model":"RandomForest", "K_trees":rf_best[1], **scores(ytest,rf.predict(Xtest))})
    try:
        from xgboost import XGBRegressor
        xb_best = None
        for k in range(1, 501):
            m = XGBRegressor(n_estimators=k, max_depth=3, learning_rate=.03,
                 subsample=.8, colsample_bytree=.8, reg_lambda=1, objective="reg:squarederror",
                 random_state=42, n_jobs=-1).fit(Xt, yt, verbose=False)
            s = scores(yv,m.predict(Xv)); k_rows.append({"model":"XGBoost", "K_trees":k, **s})
            if xb_best is None or s["RMSE"] < xb_best[0]: xb_best=(s["RMSE"],k)
        xb = XGBRegressor(n_estimators=xb_best[1], max_depth=3, learning_rate=.03,
             subsample=.8, colsample_bytree=.8, reg_lambda=1, objective="reg:squarederror",
             random_state=42, n_jobs=-1).fit(np.vstack([Xt,Xv]),pd.concat([yt,yv]),verbose=False)
        result.append({"model":"XGBoost", "K_trees":xb_best[1], **scores(ytest,xb.predict(Xtest))})
    except ImportError:
        print("Bỏ qua XGBoost: cài bằng pip install xgboost")
    pd.DataFrame(result).to_csv(out/"test_metrics.csv",index=False)
    pd.DataFrame(k_rows).to_csv(out/"k_validation_test.csv",index=False)
    final = RandomForestRegressor(n_estimators=rf_best[1], max_features=.8, min_samples_leaf=2,
                                  random_state=42, n_jobs=-1).fit(np.vstack([Xt,Xv]),pd.concat([yt,yv]))
    pred = final.predict(Xtest)
    pd.DataFrame({"ForecastOrigin":X.index[test],"Date":target_dates.loc[test].to_numpy(),"Actual":ytest.to_numpy(),"Naive":X.loc[test,"close_lag_0"].to_numpy(),"RandomForest":pred}).to_csv(out/"test_predictions.csv",index=False)
    # K là số vòng/cây boosting trong báo cáo; LR không có K cây nên được ghi rõ là N/A.
    pd.DataFrame([{"method":"Naive baseline","K":"N/A (không huấn luyện cây)"},
                  {"method":"Linear Regression","K":"N/A (không có siêu tham số số cây)"},
                  {"method":"Random Forest","K":"1–500 trees; chọn theo validation RMSE"},
                  {"method":"XGBoost","K":"1–500 boosting rounds; chọn theo validation RMSE"}]).to_csv(out/"K_test_plan.csv",index=False)
    print(pd.DataFrame(result).to_string(index=False))
    print(f"\nĐã lưu báo cáo và dự báo trong {out}/")

if __name__ == "__main__": main()
