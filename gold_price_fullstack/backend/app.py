"""Backend FastAPI — dự báo giá vàng phiên kế tiếp.

Chạy từ thư mục gold_price_fullstack:
    uvicorn backend.app:app --host 0.0.0.0 --port 8000

API:
    GET  /api/health          – trạng thái
    GET  /api/metadata        – siêu thông tin mô hình & chia tập
    GET  /api/history         – chuỗi giá lịch sử (Date, Gold_Close, ...)
    GET  /api/metrics         – bảng MAE/RMSE/MAPE/R2 trên Test
    GET  /api/predictions     – dự báo chi tiết từng phiên Test
    GET  /api/importance      – feature importance (RF)
    GET  /api/last-session    – phiên gần nhất + dự báo cho phiên kế tiếp
    POST /api/predict         – dự báo với dữ liệu phiên do người dùng cung cấp
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- Khởi tạo
app = FastAPI(
    title="Gold Price Prediction API",
    description="Dự báo giá đóng cửa hợp đồng tương lai vàng COMEX (GC=F) phiên kế tiếp",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_artifacts: dict[str, Any] = {}


def load_artifacts() -> None:
    if _artifacts:
        return
    models_dir = ROOT / "models"
    meta_path = models_dir / "metadata.json"
    if not meta_path.exists():
        raise RuntimeError(
            "Chưa có mô hình. Chạy `python ml/train.py` trước để huấn luyện."
        )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    artifacts = {
        "metadata": metadata,
        "features": metadata["features"],
        "imputer": joblib.load(models_dir / "imputer.joblib"),
        "linear": joblib.load(models_dir / "linear_regression.joblib"),
        "naive": None,  # naive = giá đóng cửa hiện tại, không cần model
    }
    for name in ("randomforest", "xgboost"):
        p = models_dir / f"{name}.joblib"
        if p.exists():
            artifacts[name] = joblib.load(p)
    # Dữ liệu lịch sử
    data_path = ROOT / "data" / "raw" / "gold_forecast_dataset.csv"
    artifacts["raw"] = pd.read_csv(data_path, parse_dates=["Date"]).set_index("Date").sort_index()
    _artifacts.update(artifacts)


@app.on_event("startup")
def _startup() -> None:
    load_artifacts()


# --------------------------------------------------------------------------- Schema
class PredictRequest(BaseModel):
    """Phiên giao dịch cần dự báo (thông tin đã biết đến cuối phiên)."""
    date: Optional[str] = Field(None, description="Ngày phiên, yyyy-mm-dd (mặc định: phiên cuối dataset)")
    gold_open: float
    gold_high: float
    gold_low: float
    gold_close: float
    dxy_close: Optional[float] = None
    us10y_close: Optional[float] = None
    sp500_close: Optional[float] = None
    wti_close: Optional[float] = None
    silver_close: Optional[float] = None


# --------------------------------------------------------------------------- Helpers
def _predict_from_row(row: pd.Series, origin: pd.Timestamp) -> dict[str, float]:
    """Dự báo phiên kế tiếp từ một dòng dữ liệu phiên t (đã biết hết đến t)."""
    a = _artifacts
    # Ghép dòng mới vào cuối chuỗi lịch sử để tính lag/SMA/RSI nhất quán.
    hist = a["raw"]
    if origin <= hist.index[-1]:
        base = hist.loc[:origin].copy()
        base.iloc[-1] = row.reindex(hist.columns)
    else:
        new = pd.DataFrame([row.reindex(hist.columns)], index=[origin])
        base = pd.concat([hist, new])
    base = base.sort_index()

    from ml.features import make_features
    # drop_unlabeled=False: giữ phiên hiện tại (t) làm dòng cuối để dự báo t+1.
    X, _, _ = make_features(base, drop_unlabeled=False)
    if X.empty:
        raise HTTPException(status_code=422, detail="Không đủ lịch sử để tính đặc trưng.")
    x_last = X.iloc[[-1]]
    Xi = a["imputer"].transform(x_last)

    out: dict[str, float] = {"Naive": float(x_last["close_lag_0"].iloc[0])}
    out["LinearRegression"] = float(a["linear"].predict(Xi)[0])
    if "randomforest" in a:
        out["RandomForest"] = float(a["randomforest"].predict(Xi)[0])
    if "xgboost" in a:
        out["XGBoost"] = float(a["xgboost"].predict(Xi)[0])
    return out


# --------------------------------------------------------------------------- Routes
@app.get("/api/health")
def health() -> dict:
    load_artifacts()
    return {"status": "ok", "models": [k for k in _artifacts if k in
                                       ("linear", "randomforest", "xgboost")]}


@app.get("/api/metadata")
def metadata() -> dict:
    load_artifacts()
    return _artifacts["metadata"]


@app.get("/api/history")
def history(start: Optional[str] = None, end: Optional[str] = None) -> list[dict]:
    load_artifacts()
    df = _artifacts["raw"].reset_index()
    if start:
        df = df[df["Date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["Date"] <= pd.Timestamp(end)]
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df.to_dict(orient="records")


@app.get("/api/metrics")
def metrics() -> list[dict]:
    load_artifacts()
    path = ROOT / "outputs" / "test_metrics.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Chưa có kết quả. Chạy ml/train.py.")
    return pd.read_csv(path).to_dict(orient="records")


@app.get("/api/predictions")
def predictions() -> list[dict]:
    load_artifacts()
    path = ROOT / "outputs" / "test_predictions.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Chưa có dự báo. Chạy ml/train.py.")
    return pd.read_csv(path).to_dict(orient="records")


@app.get("/api/importance")
def importance() -> list[dict]:
    load_artifacts()
    path = ROOT / "outputs" / "feature_importance_rf.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Không có feature importance.")
    df = pd.read_csv(path, index_col=0).rename_axis("feature").reset_index()
    df.columns = ["feature", "importance"]
    return df.to_dict(orient="records")


@app.get("/api/last-session")
def last_session() -> dict:
    """Phiên gần nhất trong dataset + dự báo cho phiên kế tiếp."""
    load_artifacts()
    hist = _artifacts["raw"]
    last = hist.iloc[-1]
    origin = hist.index[-1]
    preds = _predict_from_row(last, origin)
    rec = {"date": origin.strftime("%Y-%m-%d"),
           **{c: float(last[c]) for c in hist.columns}}
    return {"session": rec, "forecast_next_session": preds,
            "target_date_hint": "phiên giao dịch COMEX kế tiếp sau " + origin.strftime("%Y-%m-%d")}


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict:
    load_artifacts()
    hist = _artifacts["raw"]
    origin = pd.Timestamp(req.date) if req.date else hist.index[-1]
    row = pd.Series({
        "Gold_Open": req.gold_open,
        "Gold_High": req.gold_high,
        "Gold_Low": req.gold_low,
        "Gold_Close": req.gold_close,
        "Gold_Volume": float(hist["Gold_Volume"].iloc[-1]),
        "DXY_Close": req.dxy_close if req.dxy_close is not None else float(hist["DXY_Close"].iloc[-1]),
        "US10Y_Close": req.us10y_close if req.us10y_close is not None else float(hist["US10Y_Close"].iloc[-1]),
        "SP500_Close": req.sp500_close if req.sp500_close is not None else float(hist["SP500_Close"].iloc[-1]),
        "WTI_Close": req.wti_close if req.wti_close is not None else float(hist["WTI_Close"].iloc[-1]),
        "Silver_Close": req.silver_close if req.silver_close is not None else float(hist["Silver_Close"].iloc[-1]),
    })
    try:
        preds = _predict_from_row(row, origin)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Lỗi dự báo: {e}")
    return {
        "origin_date": origin.strftime("%Y-%m-%d"),
        "input_close": req.gold_close,
        "forecast_next_session": preds,
        "note": "Đơn vị USD/troy ounce. Naive = giá đóng cửa phiên hiện tại. "
                "Đây là mô hình nghiên cứu, không phải khuyến nghị đầu tư.",
    }


# --------------------------------------------------------------------------- Static frontend
frontend_dir = ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
