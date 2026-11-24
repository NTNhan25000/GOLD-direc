"""Bộ kiểm thử tự động cho pipeline ML và REST API của gold_price_fullstack.

Chạy từ thư mục gốc repository:
    .venv/bin/python -m pytest gold_price_fullstack/tests -q

Các nhóm kiểm thử:
  1. Feature engineering: không rò rỉ dữ liệu, số đặc trưng, đúng công thức.
  2. Pipeline huấn luyện: 4 mô hình cho đúng chỉ số trong báo cáo; imputer chỉ fit trên Train.
  3. REST API: /api/health, /api/metrics, /api/importance?model=rf|xgb, /api/predict.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]          # .../gold_price_fullstack
REPO_ROOT = ROOT.parent


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    path = ROOT / "data" / "raw" / "gold_forecast_dataset.csv"
    return pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()


@pytest.fixture(scope="session")
def features(raw):
    from ml.features import make_features, split_masks

    X, y, target_dates = make_features(raw)
    return X, y, target_dates, split_masks(target_dates)


# --------------------------------------------------------------------- features
def test_feature_count_and_no_missing(features):
    """25 đặc trưng, không NaN sau khi drop_unlabeled, đủ 2493 mẫu có nhãn."""
    X, y, target_dates, _ = features
    assert X.shape[1] == 25
    assert len(X) == 2493
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert not target_dates.isna().any()


def test_no_leakage_last_row_features(features, raw):
    """Hàng X(t) chỉ dùng dữ liệu <= t; y(t) = Close(t+1)."""
    X, y, _, _ = features
    for origin in ("2024-03-15", "2022-07-01", "2019-11-20"):
        t = pd.Timestamp(origin)
        row = X.loc[t]
        assert row["close_lag_0"] == pytest.approx(raw.loc[t, "Gold_Close"])
        assert row["close_lag_1"] == pytest.approx(raw.loc[:t].iloc[-2]["Gold_Close"])
        nxt = raw.index[raw.index.get_loc(t) + 1]
        assert y.loc[t] == pytest.approx(raw.loc[nxt, "Gold_Close"])


def test_make_features_inference_keeps_last_row(features, raw):
    """drop_unlabeled=False phải giữ phiên cuối cùng để suy luận t+1."""
    from ml.features import make_features

    X_inf, y_inf, _ = make_features(raw, drop_unlabeled=False)
    assert X_inf.index[-1] == raw.index[-1]
    assert np.isnan(y_inf.loc[raw.index[-1]])
    assert X_inf.loc[raw.index[-1], "close_lag_0"] == pytest.approx(
        raw.loc[raw.index[-1], "Gold_Close"])


def test_split_is_chronological_no_overlap(features):
    X, _, target_dates, masks = features
    assert not (masks["train"] & masks["val"]).any()
    assert not (masks["val"] & masks["test"]).any()
    assert target_dates[masks["train"]].max() <= pd.Timestamp("2021-12-31")
    assert target_dates[masks["val"]].min() > pd.Timestamp("2021-12-31")
    assert target_dates[masks["val"]].max() <= pd.Timestamp("2023-12-31")
    assert target_dates[masks["test"]].min() > pd.Timestamp("2023-12-31")
    assert masks["test"].sum() == 252


# --------------------------------------------------------------------- pipeline
def test_report_metrics_match(features):
    """Cổng hồi quy: MAE Test của 4 mô hình phải khớp báo cáo tới 1e-6."""
    from ml.features import scores
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    X, y, _, masks = features
    imputer = SimpleImputer(strategy="median").fit(X.loc[masks["train"]])
    Xt, Xv, Xte = (imputer.transform(X.loc[m]) for m in
                   (masks["train"], masks["val"], masks["test"]))
    yt, yv, yte = y.loc[masks["train"]], y.loc[masks["val"]], y.loc[masks["test"]]
    Xtv = np.vstack([Xt, Xv])
    ytv = pd.concat([yt, yv])
    rng = dict(max_features=0.8, min_samples_leaf=2, random_state=42, n_jobs=-1)
    xgb_pars = dict(max_depth=3, learning_rate=0.03, subsample=0.8,
                    colsample_bytree=0.8, reg_lambda=1,
                    objective="reg:squarederror", random_state=42, n_jobs=-1)

    naive = scores(yte, X.loc[masks["test"], "close_lag_0"].to_numpy())
    assert naive["MAE"] == pytest.approx(18.038101438492063, abs=1e-9)

    # OLS chỉ fit trên Train (scaler không thấy Val/Test).
    lr = make_pipeline(StandardScaler(), LinearRegression()).fit(Xt, yt)
    assert scores(yte, lr.predict(Xte))["MAE"] == pytest.approx(22.05432709103347, abs=1e-9)

    # Hai mô hình cây: chọn K trên Val rồi fit lại trên Train+Val (đúng quy trình báo cáo).
    rf = RandomForestRegressor(n_estimators=11, **rng).fit(Xtv, ytv)
    assert scores(yte, rf.predict(Xte))["MAE"] == pytest.approx(364.52873584881263, abs=1e-6)

    xb = XGBRegressor(n_estimators=165, **xgb_pars).fit(Xtv, ytv, verbose=False)
    assert scores(yte, xb.predict(Xte))["MAE"] == pytest.approx(369.7295866466704, abs=1e-6)


def test_cumulative_rf_scan_is_exact(features):
    """Quét K bằng trung bình tích lũy estimators_ phải trùng khớp mô hình fit riêng."""
    from ml.features import scores
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer

    X, y, _, masks = features
    imputer = SimpleImputer(strategy="median").fit(X.loc[masks["train"]])
    Xt = imputer.transform(X.loc[masks["train"]])
    Xv = imputer.transform(X.loc[masks["val"]])
    yt, yv = y.loc[masks["train"]], y.loc[masks["val"]]
    pars = dict(max_features=0.8, min_samples_leaf=2, random_state=42, n_jobs=-1)

    rf_k11 = RandomForestRegressor(n_estimators=11, **pars).fit(Xt, yt)
    rf500 = RandomForestRegressor(n_estimators=500, **pars).fit(Xt, yt)
    cum = np.mean([est.predict(Xv) for est in rf500.estimators_[:11]], axis=0)
    assert np.max(np.abs(cum - rf_k11.predict(Xv))) < 1e-9

    best = min((scores(yv, cum2)["RMSE"], k) for k, cum2 in
               ((k, np.mean([e.predict(Xv) for e in rf500.estimators_[:k]], axis=0))
                for k in range(1, 501)))
    assert best[1] == 11
    assert best[0] == pytest.approx(28.413333063056417, abs=1e-9)


def test_xgb_iteration_range_scan(features):
    """iteration_range=(0, k) cho kết quả giống hệt mô hình fit k vòng boosting."""
    from xgboost import XGBRegressor
    from sklearn.impute import SimpleImputer

    X, y, _, masks = features
    imputer = SimpleImputer(strategy="median").fit(X.loc[masks["train"]])
    Xt = imputer.transform(X.loc[masks["train"]])
    Xv = imputer.transform(X.loc[masks["val"]])
    yt = y.loc[masks["train"]]
    pars = dict(max_depth=3, learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=1,
                objective="reg:squarederror", random_state=42, n_jobs=-1)

    m165 = XGBRegressor(n_estimators=165, **pars).fit(Xt, yt, verbose=False)
    m500 = XGBRegressor(n_estimators=500, **pars).fit(Xt, yt, verbose=False)
    import xgboost as xgb

    dval = xgb.DMatrix(Xv, feature_names=list(X.columns))
    p500 = m500.get_booster().predict(dval, iteration_range=(0, 165))
    assert np.max(np.abs(p500 - m165.predict(Xv))) < 1e-6


def test_artifacts_exist_and_reload():
    """Mô hình đã lưu phải nạp lại được và cho dự báo hợp lý."""
    import joblib

    models = ROOT / "models"
    for name in ("imputer", "linear_regression", "randomforest", "xgboost"):
        assert (models / f"{name}.joblib").exists(), f"thiếu {name}.joblib"
    rf = joblib.load(models / "randomforest.joblib")
    assert rf.n_estimators == 11


# -------------------------------------------------------------------------- API
@pytest.fixture(scope="session")
def client():
    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient

    from backend.app import app

    with TestClient(app) as c:
        yield c


def test_api_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body["models"]) == {"linear", "randomforest", "xgboost"}
    assert body["n_features"] == 25


def test_api_metadata_and_metrics(client):
    meta = client.get("/api/metadata").json()
    assert len(meta["features"]) == 25
    assert meta["best_k"] == {"RandomForest": 11, "XGBoost": 165}
    assert meta["n_samples"] == {"train": 1739, "val": 502, "test": 252}

    metrics = {row["model"]: row for row in client.get("/api/metrics").json()}
    assert metrics["Naive"]["MAE"] == pytest.approx(18.0381, abs=1e-4)
    assert metrics["RandomForest"]["K"] == 11


def test_api_importance_both_models(client):
    for model, key in (("rf", "MDI"), ("xgb", "Gain")):
        r = client.get(f"/api/importance?model={model}")
        assert r.status_code == 200
        body = r.json()
        assert body["n_features"] == 25
        assert len(body["features"]) == 25
        assert key in body["metric"]
        imp = [f["importance"] for f in body["features"]]
        assert imp == sorted(imp, reverse=True)
        # Nhóm đặc trưng giá trễ / SMA phải nằm trong top 3 (đặc trưng quyết định).
        top3 = {f["feature"] for f in body["features"][:3]}
        assert top3 & {"close_lag_0", "close_lag_1", "sma_20"}, top3

    assert client.get("/api/importance?model=abc").status_code == 422


def test_api_predict_close_only(client):
    """Chỉ gửi gold_close vẫn phải dự báo được (OHLC tự suy ra)."""
    r = client.post("/api/predict", json={"gold_close": 2650.0})
    assert r.status_code == 200, r.text
    body = r.json()
    preds = body["forecast_next_session"]
    assert set(preds) == {"Naive", "LinearRegression", "RandomForest", "XGBoost"}
    assert preds["Naive"] == pytest.approx(2650.0, abs=0.01)
    for v in preds.values():
        assert 1500 < v < 4000


def test_api_predict_full_payload(client):
    payload = {
        "date": "2024-12-31", "gold_close": 2625.0, "gold_open": 2640.0,
        "gold_high": 2655.0, "gold_low": 2618.0,
        "dxy_close": 108.5, "us10y_close": 4.55, "sp500_close": 5881.6,
        "wti_close": 71.7, "silver_close": 29.0,
    }
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["origin_date"] == "2024-12-31"


def test_api_predict_rejects_bad_ohlc(client):
    bad = {"gold_close": 2650.0, "gold_open": 2600.0, "gold_high": 2610.0}
    assert client.post("/api/predict", json=bad).status_code == 422
    assert client.post("/api/predict", json={"gold_close": -5}).status_code == 422
    assert client.post("/api/predict", json={}).status_code == 422


def test_api_predict_round_trip_matches_last_session(client):
    """Dự báo lại đúng phiên cuối dataset phải trùng /api/last-session."""
    last = client.get("/api/last-session").json()
    sess = last["session"]
    payload = {
        "gold_close": sess["Gold_Close"], "gold_open": sess["Gold_Open"],
        "gold_high": sess["Gold_High"], "gold_low": sess["Gold_Low"],
        "dxy_close": sess["DXY_Close"], "us10y_close": sess["US10Y_Close"],
        "sp500_close": sess["SP500_Close"], "wti_close": sess["WTI_Close"],
        "silver_close": sess["Silver_Close"],
    }
    again = client.post("/api/predict", json=payload).json()["forecast_next_session"]
    for k, v in last["forecast_next_session"].items():
        assert again[k] == pytest.approx(v, abs=0.02)


def test_api_static_frontend(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Dự đoán Giá Vàng" in r.text
