"""Backend FastAPI — dự báo giá vàng phiên kế tiếp.

Chạy từ thư mục gold_price_fullstack:
    uvicorn backend.app:app --host 0.0.0.0 --port 8000

API:
    GET  /api/health          – trạng thái + danh sách mô hình đã nạp
    GET  /api/metadata        – siêu thông tin mô hình & chia tập
    GET  /api/history         – chuỗi giá lịch sử (Date, Gold_Close, ...)
    GET  /api/metrics         – bảng MAE/RMSE/MAPE/R2 trên Test
    GET  /api/predictions     – dự báo chi tiết từng phiên Test
    GET  /api/importance      – feature importance, hỗ trợ ?model=rf|xgb
    GET  /api/quarterly       – sai số theo từng quý trên Test
    GET  /api/annual          – sai số theo từng năm trên Test
    GET  /api/diagnostics     – OLS / OOB / giới hạn ngoại suy (nếu đã chạy gold_extra_outputs.py)
    GET  /api/last-session    – phiên gần nhất + dự báo cho phiên kế tiếp
    POST /api/predict         – dự báo với dữ liệu phiên do người dùng cung cấp
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]          # .../gold_price_fullstack
REPO_ROOT = ROOT.parent                              # thư mục gốc repository

_artifacts: dict[str, Any] = {}


def load_artifacts() -> None:
    """Nạp mô hình + dữ liệu lịch sử (idempotent)."""
    if _artifacts:
        return
    models_dir = ROOT / "models"
    meta_path = models_dir / "metadata.json"
    if not meta_path.exists():
        raise RuntimeError(
            "Chưa có mô hình. Chạy `python ml/train.py` trước để huấn luyện."
        )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    artifacts: dict[str, Any] = {
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
    data_path = ROOT / "data" / "raw" / "gold_forecast_dataset.csv"
    artifacts["raw"] = (
        pd.read_csv(data_path, parse_dates=["Date"]).set_index("Date").sort_index()
    )
    _artifacts.update(artifacts)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Nạp mô hình khi khởi động (thay cho @app.on_event deprecated)."""
    load_artifacts()
    yield
    _artifacts.clear()


app = FastAPI(
    title="Gold Price Prediction API",
    description="Dự báo giá đóng cửa hợp đồng tương lai vàng COMEX (GC=F) phiên kế tiếp",
    version="1.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- Schema
class PredictRequest(BaseModel):
    """Phiên giao dịch cần dự báo (thông tin đã biết đến cuối phiên).

    Chỉ `gold_close` là bắt buộc; `gold_open`, `gold_high`, `gold_low` có thể bỏ trống
    và sẽ được suy ra từ giá đóng cửa (open = high = low = close) — hữu ích khi
    người dùng chỉ biết giá chốt phiên.
    """
    date: Optional[str] = Field(None, description="Ngày phiên, yyyy-mm-dd (mặc định: phiên cuối dataset)")
    gold_close: float = Field(..., gt=0, description="Giá đóng cửa phiên hiện tại (USD/oz)")
    gold_open: Optional[float] = Field(None, gt=0)
    gold_high: Optional[float] = Field(None, gt=0)
    gold_low: Optional[float] = Field(None, gt=0)
    gold_volume: Optional[float] = Field(None, ge=0)
    dxy_close: Optional[float] = None
    us10y_close: Optional[float] = None
    sp500_close: Optional[float] = None
    wti_close: Optional[float] = None
    silver_close: Optional[float] = None

    @model_validator(mode="after")
    def _fill_and_check_ohlc(self) -> "PredictRequest":
        c = self.gold_close
        if self.gold_open is None:
            self.gold_open = c
        if self.gold_high is None:
            self.gold_high = max(self.gold_open, c)
        if self.gold_low is None:
            self.gold_low = min(self.gold_open, c)
        tol = 1e-6
        if self.gold_high < max(self.gold_open, c) - tol:
            raise ValueError("gold_high phải >= max(gold_open, gold_close)")
        if self.gold_low > min(self.gold_open, c) + tol:
            raise ValueError("gold_low phải <= min(gold_open, gold_close)")
        return self


# --------------------------------------------------------------------------- Helpers
def _predict_from_row(row: pd.Series, origin: pd.Timestamp) -> dict[str, float]:
    """Dự báo phiên kế tiếp từ một dòng dữ liệu phiên t (đã biết hết đến t)."""
    a = _artifacts
    hist = a["raw"]
    base = hist.astype("float64")
    if origin in base.index:
        base.loc[origin] = row.reindex(hist.columns).to_numpy(dtype="float64")
        base = base.loc[:origin]
    else:
        new = pd.DataFrame([row.reindex(hist.columns).to_numpy(dtype="float64")],
                           index=[origin], columns=hist.columns)
        base = pd.concat([hist.astype("float64"), new]).sort_index()

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
    return {k: round(v, 2) for k, v in out.items()}


def _read_output(name: str, detail: str) -> pd.DataFrame:
    path = ROOT / "outputs" / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=detail)
    return pd.read_csv(path)


def _read_repo_output(name: str) -> Optional[pd.DataFrame]:
    path = REPO_ROOT / "gold_model_output" / name
    return pd.read_csv(path) if path.exists() else None


# --------------------------------------------------------------------------- Routes
@app.get("/api/health")
def health() -> dict:
    load_artifacts()
    return {
        "status": "ok",
        "models": [k for k in _artifacts if k in ("linear", "randomforest", "xgboost")],
        "n_features": len(_artifacts["features"]),
        "trained_at": _artifacts["metadata"].get("trained_at"),
    }


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
    return _read_output("test_metrics.csv", "Chưa có kết quả. Chạy ml/train.py.").to_dict(orient="records")


@app.get("/api/predictions")
def predictions() -> list[dict]:
    load_artifacts()
    return _read_output("test_predictions.csv", "Chưa có dự báo. Chạy ml/train.py.").to_dict(orient="records")


@app.get("/api/importance")
def importance(model: Literal["rf", "xgb"] = Query("rf", description="rf = RandomForest (MDI), xgb = XGBoost (gain)")) -> dict:
    """Feature importance đầy đủ 25 đặc trưng của mô hình cây được chọn."""
    load_artifacts()
    features = _artifacts["features"]
    if model == "rf":
        df = _read_output("feature_importance_rf.csv", "Không có feature importance RF. Chạy ml/train.py.")
        records = [
            {"feature": r["feature"], "importance": round(float(r["MDI_pct"]), 4)}
            for _, r in df.iterrows()
        ]
        info = {"model": "RandomForest", "metric": "MDI (%)"}
    else:
        df = _read_output("feature_importance_xgb.csv", "Không có feature importance XGB. Chạy ml/train.py.")
        df = df.sort_values("gain", ascending=False)
        records = [
            {
                "feature": r["feature"],
                "importance": round(float(r["gain_pct"]), 4),
                "gain": round(float(r["gain"]), 4),
                "weight": int(r["weight"]),
                "mean_abs_shap_test_USD": round(float(r["mean_abs_shap_test_USD"]), 4),
            }
            for _, r in df.iterrows()
        ]
        info = {"model": "XGBoost", "metric": "Gain (%)"}
    records.sort(key=lambda d: d["importance"], reverse=True)
    return {**info, "n_features": len(features), "features": records}


@app.get("/api/quarterly")
def quarterly() -> list[dict]:
    df = _read_repo_output("quarterly_test_metrics.csv")
    if df is None:
        raise HTTPException(status_code=404, detail="Chưa có phân tích theo quý. Chạy gold_extra_outputs.py.")
    return df.to_dict(orient="records")


@app.get("/api/annual")
def annual() -> list[dict]:
    df = _read_repo_output("annual_test_metrics.csv")
    if df is None:
        raise HTTPException(status_code=404, detail="Chưa có phân tích theo năm. Chạy gold_predict.py.")
    return df.to_dict(orient="records")


@app.get("/api/diagnostics")
def diagnostics() -> dict:
    path = REPO_ROOT / "gold_model_output" / "extra_diagnostics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Chưa có chẩn đoán mở rộng. Chạy gold_extra_outputs.py.")
    return json.loads(path.read_text(encoding="utf-8"))


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
    origin = hist.index[-1]
    if req.date:
        try:
            origin = pd.Timestamp(req.date)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Ngày không hợp lệ: {e}")
    row = pd.Series({
        "Gold_Open": req.gold_open,
        "Gold_High": req.gold_high,
        "Gold_Low": req.gold_low,
        "Gold_Close": req.gold_close,
        "Gold_Volume": req.gold_volume if req.gold_volume is not None
        else (float(hist["Gold_Volume"].iloc[-1]) if "Gold_Volume" in hist else 0.0),
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
