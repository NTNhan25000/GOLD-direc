"""Nạp và dùng các file .joblib đã huấn luyện — bản độc lập, không phụ thuộc app.

Folder này tự chạy được một mình: chỉ cần `artifacts/*.joblib`, `artifacts/metadata.json`
và `data/gold_forecast_dataset.csv`. Không import gì từ `gold_price_fullstack`.

Nội dung folder
---------------
    model_joblib/
    ├── artifacts/                     ← 4 file .joblib + metadata.json
    │   ├── imputer.joblib             SimpleImputer (median, 25 đặc trưng)
    │   ├── linear_regression.joblib   Pipeline: StandardScaler → LinearRegression
    │   ├── randomforest.joblib        RandomForestRegressor (K = 11 cây)
    │   ├── xgboost.joblib             XGBRegressor (K = 165 vòng)
    │   └── metadata.json              danh sách 25 đặc trưng, mốc chia tập, metric Test
    ├── data/
    │   └── gold_forecast_dataset.csv  2.514 phiên 2015-01-02 → 2024-12-31
    ├── joblib_loader.py               code nạp model (file này)
    └── README.md

Đường dẫn dự báo (thứ tự bắt buộc — sai thứ tự là sai kết quả)
---------------------------------------------------------------
    dữ liệu thô → make_features() → 25 đặc trưng
    → imputer.transform()  → điền NaN bằng median của Train
    → model.predict()      → Linear Regression tự chuẩn hoá trong Pipeline,
                             RF/XGB nhận thẳng ma trận 25 cột

Chạy thử:
    python model_joblib/joblib_loader.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
DATA = HERE / "data" / "gold_forecast_dataset.csv"

TARGET = "Gold_Close"
MACRO = ["DXY_Close", "US10Y_Close", "SP500_Close", "WTI_Close", "Silver_Close"]

MODEL_FILES = {
    "LinearRegression": "linear_regression.joblib",
    "RandomForest": "randomforest.joblib",
    "XGBoost": "xgboost.joblib",
}

FILE_LABELS = {
    "LinearRegression": "Linear Regression",
    "RandomForest": "Random Forest (K=11)",
    "XGBoost": "XGBoost (K=165)",
}


# --------------------------------------------------------------- nạp artifacts
def load_models(artifacts_dir: Path = ARTIFACTS) -> dict[str, Any]:
    """Nạp toàn bộ .joblib + metadata. Trả về dict dùng lại được nhiều lần."""
    meta_path = artifacts_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Thiếu {meta_path} — copy lại từ gold_price_fullstack/models/")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    imputer_path = artifacts_dir / "imputer.joblib"
    if not imputer_path.exists():
        raise FileNotFoundError(f"Thiếu {imputer_path}")

    models: dict[str, Any] = {
        "metadata": metadata,
        "features": metadata["features"],
        "imputer": joblib.load(imputer_path),
    }
    for key, fname in MODEL_FILES.items():
        p = artifacts_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Thiếu {p}")
        models[key] = joblib.load(p)
    return models


# ------------------------------------------------------------ kỹ nghệ đặc trưng
def make_features(raw: pd.DataFrame, drop_unlabeled: bool = True):
    """Sinh 25 đặc trưng y hệt lúc huấn luyện.

    Mỗi dòng X(t) chỉ dùng dữ liệu ≤ t (chống rò rỉ). Hàm này chép lại nguyên
    văn logic của gold_price_fullstack/ml/features.py để folder này độc lập —
    nếu bên đó đổi công thức thì phải sửa cả ở đây.

    Trả về (X, y, target_dates). drop_unlabeled=False giữ phiên cuối chưa có nhãn
    để suy luận cho phiên kế tiếp.
    """
    d = raw.sort_index().copy()
    close = d[TARGET]
    x = pd.DataFrame(index=d.index)

    for lag in (0, 1, 2, 3, 5, 10):
        x[f"close_lag_{lag}"] = close.shift(lag)
    for w in (5, 10, 20):
        x[f"sma_{w}"] = close.rolling(w).mean()
        x[f"volatility_{w}"] = close.pct_change().rolling(w).std()

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
    target_dates = pd.Series(d.index, index=d.index).shift(-1).rename("target_date")

    if drop_unlabeled:
        valid = y.notna()
        x, y, target_dates = x.loc[valid], y.loc[valid], target_dates.loc[valid]
        ready = x.notna().all(axis=1)
        x, y, target_dates = x.loc[ready], y.loc[ready], target_dates.loc[ready]
    return x, y, target_dates


def load_raw(data_path: Path = DATA) -> pd.DataFrame:
    return pd.read_csv(data_path, parse_dates=["Date"]).set_index("Date").sort_index()


# ------------------------------------------------------------------ dự báo
def predict(models: dict[str, Any], X: pd.DataFrame) -> pd.DataFrame:
    """Dự báo mức giá t+1 cho ma trận đặc trưng X.

    LƯU Ý: imputer chỉ áp dụng cho X, và phải áp dụng TRƯỚC mọi model.
    Linear Regression đã có StandardScaler bên trong Pipeline nên không
    chuẩn hoá thêm ở ngoài (chuẩn hoá 2 lần sẽ ra số sai).
    """
    X = X[models["features"]]                     # cố định đúng thứ tự 25 cột
    Xi = models["imputer"].transform(X)

    out = pd.DataFrame(index=X.index)
    out["Naive"] = X["close_lag_0"].to_numpy()     # baseline: giữ nguyên giá hiện tại
    for key in MODEL_FILES:
        if key in models:
            out[key] = np.asarray(models[key].predict(Xi), dtype=float)
    return out


def predict_next_session(models: dict[str, Any], raw: pd.DataFrame | None = None,
                         origin=None, close_override: float | None = None) -> pd.Series:
    """Dự báo phiên kế tiếp sau `origin` (mặc định: phiên cuối có trong dữ liệu)."""
    raw = load_raw() if raw is None else raw
    origin = raw.index[-1] if origin is None else pd.Timestamp(origin)
    base = raw.loc[:origin].copy()
    if base.empty:
        raise ValueError(f"Không có dữ liệu đến {origin.date()}")

    if close_override is not None:
        # Giá nhập tay thay cho phiên gần nhất; High/Low phải bao lấy giá mới,
        # nếu không gold_range = (High − Low)/Close sẽ ra số vô nghĩa.
        last = base.index[-1]
        base.at[last, "Gold_Close"] = float(close_override)
        base.at[last, "Gold_High"] = max(float(base.at[last, "Gold_High"]), close_override)
        base.at[last, "Gold_Low"] = min(float(base.at[last, "Gold_Low"]), close_override)

    X, _, _ = make_features(base, drop_unlabeled=False)
    return predict(models, X.iloc[[-1]]).iloc[0]


def metrics(y_true, y_pred) -> dict[str, float]:
    """MAE / RMSE / MAPE (%) / R² — cùng định nghĩa với báo cáo."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_pct": float(np.mean(np.abs(err / y_true)) * 100),
        "R2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
    }


def score_test_split(models: dict[str, Any], raw: pd.DataFrame | None = None
                     ) -> pd.DataFrame:
    """Chạy cả 4 mô hình trên Test 2024 và trả bảng metric (đối chiếu báo cáo)."""
    raw = load_raw() if raw is None else raw
    X, y, target_dates = make_features(raw, drop_unlabeled=True)
    mask = target_dates > pd.Timestamp("2023-12-31")
    Xt, yt = X[mask], y[mask]
    preds = predict(models, Xt)
    return pd.DataFrame({c: metrics(yt, preds[c]) for c in preds.columns}).T


def describe(models: dict[str, Any]) -> str:
    """Tóm tắt nhanh để kiểm tra phiên bản model / sklearn / xgboost."""
    import sklearn
    import xgboost

    m = models["metadata"]
    lines = [
        f"metadata.trained_at : {m.get('trained_at')}",
        f"sklearn / xgboost   : {sklearn.__version__} / {xgboost.__version__}",
        f"số đặc trưng        : {len(m['features'])}",
        f"K tốt nhất          : RF={m['best_k']['RandomForest']}, "
        f"XGB={m['best_k']['XGBoost']}",
        f"mẫu train/val/test  : {m['n_samples']['train']}/"
        f"{m['n_samples']['val']}/{m['n_samples']['test']}",
    ]
    for key, fname in MODEL_FILES.items():
        obj = models[key]
        if key == "LinearRegression":
            est = obj.steps[-1][1]
            lines.append(f"{fname:28s} : Pipeline{ [s[0] for s in obj.steps] } "
                         f"{type(est).__name__}")
        elif key == "RandomForest":
            lines.append(f"{fname:28s} : {type(obj).__name__} "
                         f"n_estimators={obj.n_estimators} "
                         f"max_features={obj.max_features}")
        else:
            lines.append(f"{fname:28s} : {type(obj).__name__} "
                         f"n_estimators={obj.n_estimators} max_depth={obj.max_depth} "
                         f"learning_rate={obj.learning_rate}")
    return "\n".join(lines)


# ------------------------------------------------------------------ CLI tự kiểm
def _main() -> None:
    models = load_models()
    print("=" * 78)
    print(" MODEL JOBLIB — nạp bằng joblib.load() từ model_joblib/artifacts/")
    print("=" * 78)
    print(describe(models))

    raw = load_raw()
    print(f"\nDữ liệu: {len(raw)} phiên · {raw.index.min().date()} → {raw.index.max().date()}")

    print("\n" + "-" * 78)
    print(" 1) Dự báo phiên kế tiếp sau phiên cuối cùng của dữ liệu")
    print("-" * 78)
    nxt = predict_next_session(models, raw)
    for k, v in nxt.items():
        print(f"   {k:22s} {v:10,.2f} USD/oz")

    print("\n" + "-" * 78)
    print(" 2) Tái lập bảng Test 2024 từ chính các file .joblib này")
    print("-" * 78)
    tbl = score_test_split(models, raw)
    print(f"   {'Mô hình':<22}{'MAE':>10}{'RMSE':>10}{'MAPE %':>10}{'R²':>10}")
    for name, r in tbl.iterrows():
        label = {"Naive": "Naive Baseline"}.get(name, FILE_LABELS.get(name, name))
        print(f"   {label:<22}{r.MAE:>10.3f}{r.RMSE:>10.3f}"
              f"{r.MAPE_pct:>10.3f}{r.R2:>10.4f}")

    ref = models["metadata"]["test_metrics"]
    worst = max(abs(tbl.loc[k, "MAE"] - ref[k]["MAE"])
                for k in ref if k in tbl.index)
    print(f"\n   Lệch MAE lớn nhất so với metadata.json: {worst:.2e} "
          f"→ {'KHỚP' if worst < 1e-6 else 'KHÔNG KHỚP, kiểm tra phiên bản sklearn/xgboost'}")


if __name__ == "__main__":
    _main()
