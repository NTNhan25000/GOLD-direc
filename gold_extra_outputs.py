"""Xuất đầy đủ các phân tích thống kê, giải thích mô hình và thực nghiệm bổ sung.

Script này tái lập mọi bảng phân tích chuyên sâu và bổ sung các đầu ra mở rộng:
  1. ols_diagnostics.csv          – Hệ số β, SE, t-stat, p-value, CI 95%, VIF, Durbin–Watson
  2. quarterly_test_metrics.csv   – Sai số MAE/RMSE theo từng quý Q1–Q4 năm 2024
  3. rf_feature_importance.csv    – MDI (%) + Permutation Importance trên Validation & Test + OOB
  4. xgb_learning_curve.csv       – Đường cong RMSE Train vs Validation của XGBoost (K=1..500)
  5. xgb_feature_importance.csv   – Gain, Weight, Cover + TreeSHAP mean(|SHAP|) trên Val & Test
  6. model_detailed_summary.csv   – Tổng hợp Val/Test, sai số có dấu, max error, tỷ lệ thắng Naive, đúng chiều
  7. ablation_macro_study.csv     – Thực nghiệm Ablation: Chỉ kỹ thuật (15 biến) vs Kỹ thuật + Vĩ mô (25 biến)
  8. delta_target_experiment.csv  – Thực nghiệm dự báo mức thay đổi giá ΔClose(t+1) để khắc phục lỗi ngoại suy
  9. Biểu đồ chuẩn báo cáo        – k_validation_curve.png, test_forecast_comparison.png,
                                    xgb_learning_curve.png, xgb_shap_importance.png,
                                    delta_vs_level_comparison.png

Chạy:
    python gold_extra_outputs.py [--data gold_forecast_dataset_yfinance.csv] [--out gold_model_output]
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gold_predict import MACRO, REPO_ROOT, TARGET, make_features, scores

# K = 11 cây nên một số ít mẫu Train+Validation không có dự báo OOB (đã xử lý riêng ở mục 4)
warnings.filterwarnings("ignore", message="Some inputs do not have OOB scores.*")

RF_BEST_K = 11
XGB_BEST_K = 165
RF_PARAMS = dict(max_features=0.8, min_samples_leaf=2, random_state=42, n_jobs=-1)
XGB_PARAMS = dict(
    max_depth=3, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=1,
    objective="reg:squarederror", random_state=42, n_jobs=-1,
)


def durbin_watson(residuals: np.ndarray) -> float:
    diff = np.diff(residuals)
    return float(np.sum(diff ** 2) / np.sum(residuals ** 2))


def compute_ols_diagnostics(
    Xt_raw: np.ndarray, yt: np.ndarray, Xtest_raw: np.ndarray, ytest: np.ndarray, feature_names: list[str]
) -> tuple[pd.DataFrame, dict[str, float]]:
    scaler = StandardScaler().fit(Xt_raw)
    Xs = scaler.transform(Xt_raw)
    Xs_test = scaler.transform(Xtest_raw)

    n, p = Xs.shape
    X_design = np.column_stack([np.ones(n), Xs])
    X_test_design = np.column_stack([np.ones(len(Xs_test)), Xs_test])

    # Nghiệm OLS chính xác và ma trận hiệp phương sai
    beta, _, _, _ = np.linalg.lstsq(X_design, yt, rcond=None)
    yhat_train = X_design @ beta
    yhat_test = X_test_design @ beta

    resid_train = yt - yhat_train
    resid_test = ytest - yhat_test
    dof = n - p - 1
    sigma2 = float((resid_train @ resid_train) / dof)
    cov_beta = sigma2 * np.linalg.inv(X_design.T @ X_design)
    se = np.sqrt(np.diag(cov_beta))
    t_stats = beta / se
    p_vals = 2.0 * stats.t.sf(np.abs(t_stats), df=dof)
    t_crit = stats.t.ppf(0.975, df=dof)
    ci_low = beta - t_crit * se
    ci_high = beta + t_crit * se

    # Hệ số phóng đại phương sai VIF = phần tử đường chéo của R^{-1}
    corr = np.corrcoef(Xs, rowvar=False)
    vifs = np.diag(np.linalg.inv(corr))

    rows = [{
        "feature": "intercept",
        "coef": float(beta[0]),
        "std_err": float(se[0]),
        "t_stat": float(t_stats[0]),
        "p_value": float(p_vals[0]),
        "ci95_low": float(ci_low[0]),
        "ci95_high": float(ci_high[0]),
        "VIF": np.nan,
    }]
    for idx, fname in enumerate(feature_names, start=1):
        rows.append({
            "feature": fname,
            "coef": float(beta[idx]),
            "std_err": float(se[idx]),
            "t_stat": float(t_stats[idx]),
            "p_value": float(p_vals[idx]),
            "ci95_low": float(ci_low[idx]),
            "ci95_high": float(ci_high[idx]),
            "VIF": float(vifs[idx - 1]),
        })

    meta = {
        "train_r2": float(1.0 - (resid_train @ resid_train) / np.sum((yt - np.mean(yt)) ** 2)),
        "durbin_watson_train": durbin_watson(resid_train),
        "durbin_watson_test": durbin_watson(resid_test),
        "n_vif_gt_10": int(np.sum(vifs > 10.0)),
        "max_vif": float(np.max(vifs)),
        "n_significant_p005": int(np.sum(p_vals[1:] < 0.05)),
    }
    return pd.DataFrame(rows), meta


def make_relative_features_for_delta(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Đặc trưng tương đối (stationary-friendly) để dự báo ΔClose(t+1) = Close(t+1) - Close(t)."""
    d = raw.sort_index().copy()
    close = d[TARGET]
    x = pd.DataFrame(index=d.index)

    for lag in (0, 1, 2, 3, 5, 10):
        x[f"return_1d_lag_{lag}"] = close.pct_change().shift(lag)
        x[f"diff_1d_lag_{lag}"] = close.diff().shift(lag)
    for w in (5, 10, 20):
        sma = close.rolling(w).mean()
        x[f"close_to_sma_{w}"] = close / sma - 1.0
        x[f"volatility_{w}"] = close.pct_change().rolling(w).std()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    x["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    x["gold_range"] = (d["Gold_High"] - d["Gold_Low"]) / close

    for col in MACRO:
        if col in d.columns:
            x[f"{col}_return_1d"] = d[col].pct_change()
            x[f"{col}_return_5d"] = d[col].pct_change(5)

    y_delta = (close.shift(-1) - close).rename("target_delta")
    valid = y_delta.notna()
    target_dates = pd.Series(d.index, index=d.index).shift(-1).loc[valid]
    return x.loc[valid], y_delta.loc[valid], close.loc[valid], target_dates


def run_ablation_subset(
    X_sub: pd.DataFrame, y: pd.Series, train: pd.Series, val: pd.Series, test: pd.Series, label: str
) -> list[dict]:
    imp = SimpleImputer(strategy="median")
    Xt = imp.fit_transform(X_sub.loc[train])
    Xv = imp.transform(X_sub.loc[val])
    Xte = imp.transform(X_sub.loc[test])
    yt, yv, yte = y.loc[train], y.loc[val], y.loc[test]
    Xtv = np.vstack([Xt, Xv])
    ytv = pd.concat([yt, yv])

    rows = []
    # Linear Regression
    lr = make_pipeline(StandardScaler(), LinearRegression()).fit(Xt, yt)
    rows.append({
        "feature_set": label,
        "n_features": X_sub.shape[1],
        "model": "LinearRegression",
        "K": np.nan,
        **{f"val_{k}": v for k, v in scores(yv, lr.predict(Xv)).items()},
        **{f"test_{k}": v for k, v in scores(yte, lr.predict(Xte)).items()},
    })

    # Random Forest: quét toàn bộ K = 1..500 trên Validation bằng tiền tố cây tích lũy
    rf_scan = RandomForestRegressor(n_estimators=500, **RF_PARAMS).fit(Xt, yt)
    rf_tree_preds = np.array([est.predict(Xv) for est in rf_scan.estimators_])
    rf_cum_preds = np.cumsum(rf_tree_preds, axis=0) / np.arange(1, 501)[:, None]
    rf_best = min(
        ((scores(yv, rf_cum_preds[k - 1])["RMSE"], k) for k in range(1, 501)),
        key=lambda t: t[0],
    )
    rf_val_pred = rf_cum_preds[rf_best[1] - 1]
    rf_final = RandomForestRegressor(n_estimators=rf_best[1], **RF_PARAMS).fit(Xtv, ytv)
    rows.append({
        "feature_set": label,
        "n_features": X_sub.shape[1],
        "model": "RandomForest",
        "K": rf_best[1],
        **{f"val_{k}": v for k, v in scores(yv, rf_val_pred).items()},
        **{f"test_{k}": v for k, v in scores(yte, rf_final.predict(Xte)).items()},
    })

    # XGBoost: dùng staged boosting hiệu quả với n_estimators=500 rồi cắt đúng vòng K tốt nhất
    xb_full = xgb.XGBRegressor(n_estimators=500, **XGB_PARAMS).fit(Xt, yt, verbose=False)
    booster = xb_full.get_booster()
    dval = xgb.DMatrix(Xv)
    xb_best = min(
        ((scores(yv, booster.predict(dval, iteration_range=(0, k)))["RMSE"], k) for k in range(1, 501)),
        key=lambda t: t[0],
    )
    val_pred_xb = booster.predict(dval, iteration_range=(0, xb_best[1]))
    xb_final = xgb.XGBRegressor(n_estimators=xb_best[1], **XGB_PARAMS).fit(Xtv, ytv, verbose=False)
    rows.append({
        "feature_set": label,
        "n_features": X_sub.shape[1],
        "model": "XGBoost",
        "K": xb_best[1],
        **{f"val_{k}": v for k, v in scores(yv, val_pred_xb).items()},
        **{f"test_{k}": v for k, v in scores(yte, xb_final.predict(Xte)).items()},
    })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Xuất phân tích thống kê & thực nghiệm bổ sung")
    ap.add_argument("--data", default=str(REPO_ROOT / "gold_forecast_dataset_yfinance.csv"))
    ap.add_argument("--out", default=str(REPO_ROOT / "gold_model_output"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.data, parse_dates=["Date"]).set_index("Date").sort_index()
    X, y, target_dates = make_features(raw)

    ready = X.notna().all(axis=1) & y.notna()
    train = (target_dates <= pd.Timestamp("2021-12-31")) & ready
    val = (target_dates > pd.Timestamp("2021-12-31")) & (target_dates <= pd.Timestamp("2023-12-31")) & ready
    test = (target_dates > pd.Timestamp("2023-12-31")) & ready

    feature_names = list(X.columns)
    imputer = SimpleImputer(strategy="median")
    Xt = imputer.fit_transform(X.loc[train])
    Xv = imputer.transform(X.loc[val])
    Xte = imputer.transform(X.loc[test])
    yt = y.loc[train].to_numpy()
    yv = y.loc[val].to_numpy()
    yte = y.loc[test].to_numpy()
    Xtv = np.vstack([Xt, Xv])
    ytv = np.concatenate([yt, yv])

    # ------------------------------------------------------------------ 1. OLS Diagnostics
    ols_df, ols_meta = compute_ols_diagnostics(Xt, yt, Xte, yte, feature_names)
    ols_df.to_csv(out / "ols_diagnostics.csv", index=False)

    # ------------------------------------------------------------------ 2. Huấn luyện 4 mô hình chuẩn
    naive_val = X.loc[val, "close_lag_0"].to_numpy()
    naive_test = X.loc[test, "close_lag_0"].to_numpy()

    lr = make_pipeline(StandardScaler(), LinearRegression()).fit(Xt, yt)
    lr_val = lr.predict(Xv)
    lr_test = lr.predict(Xte)

    rf_val_model = RandomForestRegressor(n_estimators=RF_BEST_K, **RF_PARAMS).fit(Xt, yt)
    rf_val = rf_val_model.predict(Xv)
    rf_final = RandomForestRegressor(n_estimators=RF_BEST_K, oob_score=True, **RF_PARAMS).fit(Xtv, ytv)
    rf_test = rf_final.predict(Xte)

    # XGBoost learning curve K = 1..500 trên Train & Validation
    xb_scan_model = xgb.XGBRegressor(n_estimators=500, **XGB_PARAMS).fit(Xt, yt, verbose=False)
    booster_scan = xb_scan_model.get_booster()
    dtrain = xgb.DMatrix(Xt, feature_names=feature_names)
    dval = xgb.DMatrix(Xv, feature_names=feature_names)
    lc_rows = []
    for k in range(1, 501):
        p_tr = booster_scan.predict(dtrain, iteration_range=(0, k))
        p_va = booster_scan.predict(dval, iteration_range=(0, k))
        lc_rows.append({
            "K": k,
            "train_RMSE": scores(yt, p_tr)["RMSE"],
            "val_RMSE": scores(yv, p_va)["RMSE"],
            "train_MAE": scores(yt, p_tr)["MAE"],
            "val_MAE": scores(yv, p_va)["MAE"],
        })
    lc_df = pd.DataFrame(lc_rows)
    lc_df.to_csv(out / "xgb_learning_curve.csv", index=False)
    xb_val = booster_scan.predict(dval, iteration_range=(0, XGB_BEST_K))

    xb_final = xgb.XGBRegressor(n_estimators=XGB_BEST_K, **XGB_PARAMS).fit(Xtv, ytv, verbose=False)
    xb_test = xb_final.predict(Xte)

    # ------------------------------------------------------------------ 3. Quarterly Test Metrics
    target_dt_test = pd.to_datetime(target_dates.loc[test])
    q_df = pd.DataFrame({
        "Date": target_dt_test.to_numpy(),
        "Quarter": target_dt_test.dt.quarter.to_numpy(),
        "Actual": yte,
        "Naive": naive_test,
        "LinearRegression": lr_test,
        "RandomForest": rf_test,
        "XGBoost": xb_test,
    })
    q_rows = []
    for q, grp in q_df.groupby("Quarter"):
        row = {
            "Quarter": f"Q{int(q)}",
            "N_sessions": len(grp),
            "Mean_Actual_USD": float(grp["Actual"].mean()),
        }
        for m_name in ("Naive", "LinearRegression", "RandomForest", "XGBoost"):
            sc = scores(grp["Actual"], grp[m_name])
            row[f"MAE_{m_name}"] = sc["MAE"]
            row[f"RMSE_{m_name}"] = sc["RMSE"]
        q_rows.append(row)
    pd.DataFrame(q_rows).to_csv(out / "quarterly_test_metrics.csv", index=False)

    # ------------------------------------------------------------------ 4. RF Importance & OOB
    def neg_rmse(estimator, X_eval, y_eval):
        return -scores(y_eval, estimator.predict(X_eval))["RMSE"]

    perm_val = permutation_importance(rf_val_model, Xv, yv, scoring=neg_rmse, n_repeats=20, random_state=42)
    perm_test = permutation_importance(rf_final, Xte, yte, scoring=neg_rmse, n_repeats=20, random_state=42)

    rf_imp_df = pd.DataFrame({
        "feature": feature_names,
        "MDI_pct": rf_final.feature_importances_ * 100.0,
        "perm_RMSE_increase_val_USD": perm_val.importances_mean,
        "perm_RMSE_increase_test_USD": perm_test.importances_mean,
    }).sort_values("MDI_pct", ascending=False)
    rf_imp_df.to_csv(out / "rf_feature_importance.csv", index=False)

    # Với K=11 cây, hiếm hoi có mẫu không rơi vào tập OOB của bất kỳ cây nào →
    # sklearn gán giá trị 0.0 và phát cảnh báo; ta loại các mẫu đó khỏi thống kê OOB.
    oob_pred = rf_final.oob_prediction_
    oob_valid = np.isfinite(oob_pred) & (oob_pred > 0)
    oob_metrics = scores(ytv[oob_valid], oob_pred[oob_valid])

    # ------------------------------------------------------------------ 5. XGBoost Importance & TreeSHAP
    booster_final = xb_final.get_booster()
    booster_final.feature_names = feature_names
    gain_map = booster_final.get_score(importance_type="gain")
    weight_map = booster_final.get_score(importance_type="weight")
    cover_map = booster_final.get_score(importance_type="cover")

    # TreeSHAP chính xác từ XGBoost Booster (cột cuối là bias term)
    dval_final = xgb.DMatrix(Xv, feature_names=feature_names)
    dtest_final = xgb.DMatrix(Xte, feature_names=feature_names)
    shap_val = booster_scan.predict(dval_final, iteration_range=(0, XGB_BEST_K), pred_contribs=True)[:, :-1]
    shap_test = booster_final.predict(dtest_final, pred_contribs=True)[:, :-1]

    total_gain = sum(gain_map.values()) or 1.0
    xgb_imp_df = pd.DataFrame({
        "feature": feature_names,
        "gain_pct": [gain_map.get(f, 0.0) / total_gain * 100.0 for f in feature_names],
        "split_count": [int(weight_map.get(f, 0.0)) for f in feature_names],
        "cover": [float(cover_map.get(f, 0.0)) for f in feature_names],
        "mean_abs_SHAP_val_USD": np.mean(np.abs(shap_val), axis=0),
        "mean_abs_SHAP_test_USD": np.mean(np.abs(shap_test), axis=0),
    }).sort_values("mean_abs_SHAP_val_USD", ascending=False)
    xgb_imp_df.to_csv(out / "xgb_feature_importance.csv", index=False)

    # ------------------------------------------------------------------ 6. Detailed Model Summary
    val_preds = {"Naive": naive_val, "LinearRegression": lr_val, "RandomForest": rf_val, "XGBoost": xb_val}
    test_preds = {"Naive": naive_test, "LinearRegression": lr_test, "RandomForest": rf_test, "XGBoost": xb_test}
    actual_change = yte - naive_test
    naive_abs_err = np.abs(yte - naive_test)

    summary_rows = []
    for m_name in ("Naive", "LinearRegression", "RandomForest", "XGBoost"):
        sv = scores(yv, val_preds[m_name])
        st = scores(yte, test_preds[m_name])
        err_signed = yte - test_preds[m_name]
        abs_err = np.abs(err_signed)
        max_idx = int(np.argmax(abs_err))
        pred_change = test_preds[m_name] - naive_test
        dir_acc = (
            np.nan if m_name == "Naive"
            else float(np.mean(np.sign(pred_change) == np.sign(actual_change)) * 100.0)
        )
        win_vs_naive = (
            np.nan if m_name == "Naive"
            else float(np.mean(abs_err < naive_abs_err) * 100.0)
        )
        summary_rows.append({
            "model": m_name,
            "val_MAE": sv["MAE"],
            "val_RMSE": sv["RMSE"],
            "val_MAPE_pct": sv["MAPE_pct"],
            "val_R2": sv["R2"],
            "test_MAE": st["MAE"],
            "test_RMSE": st["RMSE"],
            "test_MAPE_pct": st["MAPE_pct"],
            "test_R2": st["R2"],
            "signed_mean_error_USD": float(np.mean(err_signed)),
            "max_abs_error_USD": float(abs_err[max_idx]),
            "max_error_date": target_dt_test.iloc[max_idx].strftime("%Y-%m-%d"),
            "win_rate_vs_naive_pct": win_vs_naive,
            "directional_accuracy_pct": dir_acc,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out / "model_detailed_summary.csv", index=False)

    # ------------------------------------------------------------------ 7. Ablation Study (Macro vs Tech-only)
    tech_cols = [c for c in feature_names if not any(c.startswith(m) for m in MACRO)]
    ablation_rows = [
        {
            "feature_set": "Baseline (Close_t only)",
            "n_features": 1,
            "model": "Naive",
            "K": np.nan,
            **{f"val_{k}": v for k, v in scores(yv, naive_val).items()},
            **{f"test_{k}": v for k, v in scores(yte, naive_test).items()},
        }
    ]
    ablation_rows.extend(run_ablation_subset(X[tech_cols], y, train, val, test, "Technical_Only"))
    ablation_rows.extend(run_ablation_subset(X, y, train, val, test, "Technical_Plus_Macro"))
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(out / "ablation_macro_study.csv", index=False)

    # ------------------------------------------------------------------ 8. Thực nghiệm mục tiêu ΔClose(t+1)
    Xd, yd, close_t, td_d = make_relative_features_for_delta(raw)
    ready_d = Xd.notna().all(axis=1) & yd.notna()
    tr_d = (td_d <= pd.Timestamp("2021-12-31")) & ready_d
    va_d = (td_d > pd.Timestamp("2021-12-31")) & (td_d <= pd.Timestamp("2023-12-31")) & ready_d
    te_d = (td_d > pd.Timestamp("2023-12-31")) & ready_d

    imp_d = SimpleImputer(strategy="median")
    Xt_d = imp_d.fit_transform(Xd.loc[tr_d])
    Xv_d = imp_d.transform(Xd.loc[va_d])
    Xte_d = imp_d.transform(Xd.loc[te_d])
    yt_d = yd.loc[tr_d].to_numpy()
    yv_d = yd.loc[va_d].to_numpy()
    Xtv_d = np.vstack([Xt_d, Xv_d])
    ytv_d = np.concatenate([yt_d, yv_d])

    c_val = close_t.loc[va_d].to_numpy()
    c_test = close_t.loc[te_d].to_numpy()
    actual_val_price = c_val + yv_d
    actual_test_price = c_test + yd.loc[te_d].to_numpy()

    lr_d = make_pipeline(StandardScaler(), LinearRegression()).fit(Xt_d, yt_d)
    rf_d_scan = RandomForestRegressor(n_estimators=500, **RF_PARAMS).fit(Xt_d, yt_d)
    rf_d_cum = np.cumsum([est.predict(Xv_d) for est in rf_d_scan.estimators_], axis=0) / np.arange(1, 501)[:, None]
    rf_best_d = min(
        ((scores(actual_val_price, c_val + rf_d_cum[k - 1])["RMSE"], k) for k in range(1, 501)),
        key=lambda t: t[0],
    )
    rf_d = RandomForestRegressor(n_estimators=rf_best_d[1], **RF_PARAMS).fit(Xtv_d, ytv_d)

    xb_d_scan = xgb.XGBRegressor(n_estimators=500, **XGB_PARAMS).fit(Xt_d, yt_d, verbose=False)
    b_d = xb_d_scan.get_booster()
    dval_d = xgb.DMatrix(Xv_d)
    xb_best_d = min(
        ((scores(actual_val_price, c_val + b_d.predict(dval_d, iteration_range=(0, k)))["RMSE"], k)
         for k in range(1, 501)),
        key=lambda t: t[0],
    )
    xb_d = xgb.XGBRegressor(n_estimators=xb_best_d[1], **XGB_PARAMS).fit(Xtv_d, ytv_d, verbose=False)

    delta_preds_price = {
        "Naive (Δ=0)": c_test,
        "LinearRegression (ΔClose)": c_test + lr_d.predict(Xte_d),
        f"RandomForest (ΔClose, K={rf_best_d[1]})": c_test + rf_d.predict(Xte_d),
        f"XGBoost (ΔClose, K={xb_best_d[1]})": c_test + xb_d.predict(Xte_d),
    }
    delta_rows = []
    for name, p_price in delta_preds_price.items():
        sc = scores(actual_test_price, p_price)
        dir_acc = (
            np.nan if name.startswith("Naive")
            else float(np.mean(np.sign(p_price - c_test) == np.sign(actual_test_price - c_test)) * 100.0)
        )
        delta_rows.append({"approach": name, **sc, "directional_accuracy_pct": dir_acc})
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out / "delta_target_experiment.csv", index=False)

    # Lưu metadata chẩn đoán tổng hợp (JSON)
    diagnostics_json = {
        "ols": ols_meta,
        "rf_oob": {**oob_metrics, "n_oob_valid": int(np.sum(oob_valid)), "n_total": len(ytv)},
        "rf_extrapolation": {
            "max_pred_USD": float(np.max(rf_test)),
            "max_actual_USD": float(np.max(yte)),
            "n_sessions_above_max_pred": int(np.sum(yte > np.max(rf_test))),
        },
        "xgb_extrapolation": {
            "max_pred_USD": float(np.max(xb_test)),
            "max_actual_USD": float(np.max(yte)),
            "n_sessions_above_max_pred": int(np.sum(yte > np.max(xb_test))),
        },
    }
    (out / "extra_diagnostics.json").write_text(
        json.dumps(diagnostics_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ------------------------------------------------------------------ 9. Biểu đồ chuẩn báo cáo
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    # (a) Đường cong quét K trên Validation (RF vs XGBoost)
    k_scan_path = out / "k_validation_test.csv"
    if k_scan_path.exists():
        k_scan = pd.read_csv(k_scan_path)
        rf_curve = k_scan[k_scan["model"] == "RandomForest"]
        xgb_curve = k_scan[k_scan["model"] == "XGBoost"]
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        ax.plot(rf_curve["K_trees"], rf_curve["RMSE"], label="Random Forest", color="#1b4f72", lw=2)
        ax.plot(xgb_curve["K_trees"], xgb_curve["RMSE"], label="XGBoost", color="#ca5b16", lw=2, ls="--")
        ax.set_yscale("log")
        ax.scatter([RF_BEST_K], [rf_curve.loc[rf_curve["K_trees"] == RF_BEST_K, "RMSE"].iloc[0]], color="#1b4f72", zorder=5)
        ax.scatter([XGB_BEST_K], [xgb_curve.loc[xgb_curve["K_trees"] == XGB_BEST_K, "RMSE"].iloc[0]], color="#ca5b16", zorder=5)
        ax.set_title("RMSE Validation theo số cây / số vòng boosting K (thang log)", fontweight="bold")
        ax.set_xlabel("Số cây / số vòng boosting K")
        ax.set_ylabel("RMSE Validation (USD/oz)")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "k_validation_curve.png", dpi=200)
        plt.close(fig)

    # (b) Giá thực tế vs Dự báo 4 mô hình trên Test 2024
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.plot(target_dt_test, yte, label="Giá thực tế", color="#111111", lw=2.0)
    ax.plot(target_dt_test, naive_test, label="Naive Baseline", color="#6c757d", lw=1.2, ls=":")
    ax.plot(target_dt_test, lr_test, label="Hồi quy tuyến tính", color="#1f77b4", lw=1.3, ls="-.")
    ax.plot(target_dt_test, rf_test, label=f"Random Forest (K={RF_BEST_K})", color="#2ca02c", lw=1.4, ls="--")
    ax.plot(target_dt_test, xb_test, label=f"XGBoost (K={XGB_BEST_K})", color="#d95f02", lw=1.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.set_title("Giá thực tế và dự báo của bốn mô hình trên tập Test năm 2024", fontweight="bold")
    ax.set_xlabel("Ngày dự báo (năm 2024)")
    ax.set_ylabel("Giá đóng cửa GC=F (USD/oz)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "test_forecast_comparison.png", dpi=200)
    plt.close(fig)

    # (c) Đường cong học Train vs Validation của XGBoost
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(lc_df["K"], lc_df["train_RMSE"], label="Train RMSE", color="#1f77b4", lw=2)
    ax.plot(lc_df["K"], lc_df["val_RMSE"], label="Validation RMSE", color="#d95f02", lw=2, ls="--")
    ax.axvline(XGB_BEST_K, color="#2ca02c", lw=1.2, ls=":", label=f"K tối ưu = {XGB_BEST_K}")
    ax.set_yscale("log")
    ax.set_title("Đường cong học Train vs. Validation của XGBoost (K = 1..500)", fontweight="bold")
    ax.set_xlabel("Số vòng boosting K")
    ax.set_ylabel("RMSE (USD/oz, thang log)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "xgb_learning_curve.png", dpi=200)
    plt.close(fig)

    # (d) Top 12 đặc trưng theo TreeSHAP của XGBoost
    top_shap = xgb_imp_df.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.barh(top_shap["feature"], top_shap["mean_abs_SHAP_val_USD"], color="#ca5b16", alpha=0.88)
    ax.set_title("Top 12 đặc trưng đóng góp lớn nhất theo TreeSHAP — XGBoost (K = 165)", fontweight="bold")
    ax.set_xlabel("Trung bình |SHAP value| trên Validation (USD/oz)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "xgb_shap_importance.png", dpi=200)
    plt.close(fig)

    # (e) So sánh dự báo mức giá trực tiếp vs. dự báo ΔClose(t+1) cho mô hình cây
    dt_delta_test = pd.to_datetime(td_d.loc[te_d])
    rf_delta_key = f"RandomForest (ΔClose, K={rf_best_d[1]})"
    xgb_delta_key = f"XGBoost (ΔClose, K={xb_best_d[1]})"
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.plot(dt_delta_test, actual_test_price, label="Giá thực tế", color="#111111", lw=2.0)
    ax.plot(target_dt_test, rf_test, label="RF dự báo trực tiếp mức giá (bị chặn ngoại suy)", color="#adb5bd", lw=1.4, ls="--")
    ax.plot(dt_delta_test, delta_preds_price[rf_delta_key], label=rf_delta_key, color="#2ca02c", lw=1.3)
    ax.plot(dt_delta_test, delta_preds_price[xgb_delta_key], label=xgb_delta_key, color="#d95f02", lw=1.3, ls="-.")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.set_title("Khắc phục giới hạn ngoại suy bằng mục tiêu ΔClose(t+1) — Test 2024", fontweight="bold")
    ax.set_xlabel("Ngày dự báo (năm 2024)")
    ax.set_ylabel("USD/troy ounce")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "delta_vs_level_comparison.png", dpi=200)
    plt.close(fig)

    print("=== HOÀN TẤT XUẤT PHÂN TÍCH MỞ RỘNG ===")
    print("\n[1] Tóm tắt chi tiết 4 mô hình (model_detailed_summary.csv):")
    print(summary_df[["model", "val_RMSE", "test_MAE", "test_RMSE", "test_R2", "directional_accuracy_pct"]].round(3).to_string(index=False))
    print("\n[2] Thực nghiệm Ablation biến vĩ mô (ablation_macro_study.csv):")
    print(ablation_df[["feature_set", "n_features", "model", "K", "val_RMSE", "test_MAE", "test_RMSE"]].round(3).to_string(index=False))
    print("\n[3] Thực nghiệm đổi mục tiêu sang ΔClose(t+1) (delta_target_experiment.csv):")
    print(delta_df.round(4).to_string(index=False))
    print(f"\nĐã lưu toàn bộ bảng CSV, JSON và biểu đồ PNG vào thư mục: {out}/")


if __name__ == "__main__":
    main()
