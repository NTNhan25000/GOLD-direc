"""Kiểm tra trước giờ bảo vệ: mọi thứ sẵn sàng để demo chưa?

Chạy từ thư mục gốc repository:
    .venv/bin/python demo_preflight.py          # (hoặc python demo_preflight.py)

Script kiểm tra 6 nhóm, in PASS/FAIL cho từng mục, và in luôn 3 lệnh demo
để dán vào terminal. Không sửa gì trên đĩa — chỉ đọc.
"""
from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FS = ROOT / "gold_price_fullstack"
PY = sys.executable

OK, FAIL, WARN = "✅ PASS", "❌ FAIL", "⚠️  WARN"
results: list[tuple[str, str, str]] = []


class Warning_(Exception):
    """Vấn đề không chặn demo nhưng giám thị nên biết."""


def check(group: str, name: str, fn) -> None:
    try:
        detail = fn()
        results.append((group, name, f"{OK}  {detail or ''}".rstrip()))
    except Warning_ as e:
        results.append((group, name, f"{WARN}  {e}"))
    except Exception as e:  # noqa: BLE001
        results.append((group, name, f"{FAIL}  {e}"))


# ------------------------------------------------------------------ 1. Môi trường
def _deps() -> str:
    need = ["pandas", "numpy", "sklearn", "xgboost", "joblib", "matplotlib",
            "seaborn", "scipy", "fastapi", "uvicorn", "pydantic", "httpx", "pytest"]
    missing = []
    for m in need:
        try:
            importlib.import_module(m)
        except ImportError:
            missing.append(m)
    if missing:
        raise RuntimeError("thiếu: " + ", ".join(missing) +
                           "\n     → pip install -r requirements.txt")
    import sklearn, xgboost, pandas
    return f"Python {sys.version.split()[0]} · sklearn {sklearn.__version__} · xgboost {xgboost.__version__} · pandas {pandas.__version__}"


# -------------------------------------------------------------------- 2. Dữ liệu
def _data() -> str:
    import pandas as pd
    raw = pd.read_csv(ROOT / "gold_forecast_dataset_yfinance.csv", parse_dates=["Date"])
    assert len(raw) == 2514, f"kỳ vọng 2514 phiên, thấy {len(raw)}"
    assert raw["Date"].min().strftime("%Y-%m-%d") == "2015-01-02"
    assert raw["Date"].max().strftime("%Y-%m-%d") == "2024-12-31"
    for name in ("train", "val", "test"):
        assert (ROOT / f"gold_{name}_yfinance.csv").exists(), f"thiếu gold_{name}_yfinance.csv"
    return "2.514 phiên · 2015-01-02 → 2024-12-31 · 11 cột"


# --------------------------------------------------------------------- 3. Model
def _models() -> str:
    import joblib
    needed = ["imputer", "linear_regression", "randomforest", "xgboost"]
    missing = [n for n in needed if not (FS / "models" / f"{n}.joblib").exists()]
    if missing:
        raise RuntimeError(f"thiếu models/{missing} → cd gold_price_fullstack && "
                           f"python ml/train.py")
    rf = joblib.load(FS / "models" / "randomforest.joblib")
    xb = joblib.load(FS / "models" / "xgboost.joblib")
    meta = json.loads((FS / "models" / "metadata.json").read_text(encoding="utf-8"))
    assert rf.n_estimators == 11, f"RF phải có 11 cây, thấy {rf.n_estimators}"
    assert xb.n_estimators == 165, f"XGB phải có 165 vòng, thấy {xb.n_estimators}"
    return f"RF K=11 · XGB K=165 · 25 đặc trưng · train {meta['n_samples']['train']}/val {meta['n_samples']['val']}/test {meta['n_samples']['test']}"


def _metrics() -> str:
    import pandas as pd
    m = pd.read_csv(FS / "outputs" / "test_metrics.csv").set_index("model")
    expect = {"Naive": 18.038101, "LinearRegression": 22.054327,
              "RandomForest": 364.528736, "XGBoost": 369.729587}
    for model, mae in expect.items():
        got = float(m.loc[model, "MAE"])
        assert abs(got - mae) < 1e-4, f"{model} MAE = {got:.6f}, kỳ vọng {mae:.6f}"
    return "4 mô hình khớp báo cáo tới 6 chữ số thập phân"


# ------------------------------------------------------------------- 4. Đầu ra
def _root_outputs() -> str:
    out = ROOT / "gold_model_output"
    need = ["test_metrics.csv", "k_validation_test.csv", "ols_diagnostics.csv",
            "rf_feature_importance.csv", "xgb_feature_importance.csv",
            "ablation_macro_study.csv", "delta_target_experiment.csv",
            "quarterly_test_metrics.csv", "annual_test_metrics.csv",
            "extra_diagnostics.json", "k_validation_curve.png",
            "test_forecast_comparison.png", "xgb_learning_curve.png",
            "xgb_shap_importance.png", "delta_vs_level_comparison.png"]
    missing = [n for n in need if not (out / n).exists()]
    if missing:
        raise RuntimeError(f"thiếu {len(missing)} file → python gold_predict.py && "
                           f"python gold_extra_outputs.py\n     thiếu: {missing}")
    return f"{len(need)} bảng/biểu đồ phân tích trong gold_model_output/"


def _plots() -> str:
    need = ["gold_train_plot.png", "gold_val_plot.png", "gold_test_plot.png",
            "gold_comprehensive_analysis.png"]
    missing = [n for n in need if not (ROOT / n).exists()]
    if missing:
        raise RuntimeError(f"thiếu {missing} → python plot_splits.py")
    return "4 biểu đồ khảo sát dữ liệu ở thư mục gốc"


def _fullstack_outputs() -> str:
    out = FS / "outputs"
    need = ["test_metrics.csv", "test_predictions.csv", "k_validation_scan.csv",
            "feature_importance_rf.csv", "feature_importance_xgb.csv",
            "test_predictions_plot.png", "feature_importance_plot.png"]
    missing = [n for n in need if not (out / n).exists()]
    if missing:
        raise RuntimeError(f"thiếu {missing} → cd gold_price_fullstack && python ml/train.py")
    return "7 file trong gold_price_fullstack/outputs/"


# ------------------------------------------------------------------- 5. Demo CLI
def _demo_cli() -> str:
    r = subprocess.run([PY, str(FS / "demo" / "demo.py"), "--close", "2650", "--skip-test"],
                       capture_output=True, text=True, timeout=180, cwd=FS)
    assert r.returncode == 0, r.stderr[-400:]
    assert "XGBoost" in r.stdout and "Naive Baseline" in r.stdout
    return "demo/demo.py chạy được, in đủ 4 mô hình"


# --------------------------------------------------------------------- 6. API
def _api() -> str:
    sys.path.insert(0, str(FS))
    from fastapi.testclient import TestClient

    from backend.app import app

    with TestClient(app) as c:
        h = c.get("/api/health")
        assert h.status_code == 200, h.text
        p = c.post("/api/predict", json={"gold_close": 2650})
        assert p.status_code == 200, p.text
        preds = p.json()["forecast_next_session"]
        assert set(preds) == {"Naive", "LinearRegression", "RandomForest", "XGBoost"}
        i = c.get("/api/importance?model=xgb")
        assert i.status_code == 200 and i.json()["n_features"] == 25
    return f"health + predict + importance OK (Naive={preds['Naive']:.2f})"


def _frontend_assets() -> str:
    f = FS / "frontend"
    need = ["index.html", "app.js", "style.css", "chart.umd.js"]
    missing = [n for n in need if not (f / n).exists()]
    if missing:
        raise RuntimeError(f"thiếu {missing}")
    # Chart.js phải là bản local → demo không cần mạng
    html = (f / "index.html").read_text(encoding="utf-8")
    assert "http://" not in html and "https://" not in html, \
        "index.html còn tham chiếu mạng ngoài → demo sẽ hỏng khi mất wifi"
    return "4 asset tự chứa (Chart.js local) → demo chạy offline"


def _port() -> str:
    import socket
    with socket.socket() as s:
        free = s.connect_ex(("127.0.0.1", 8000)) != 0
    if not free:
        raise Warning_("cổng 8000 đang bận — nếu là server cũ thì tắt bằng Ctrl+C trước khi demo")
    return "cổng 8000 trống, sẵn sàng"


def _node() -> str:
    if shutil.which("node"):
        return "có node (chỉ dùng để tự kiểm tra cú pháp JS, không bắt buộc)"
    return "không có node — không sao, trình duyệt mới cần"


# ------------------------------------------------------------------------ Chạy
check("1. Môi trường", "Thư viện Python", _deps)
check("1. Môi trường", "Node.js (tuỳ chọn)", _node)
check("1. Môi trường", "Cổng 8000", _port)
check("2. Dữ liệu", "Dataset gốc + 3 tập chia", _data)
check("3. Mô hình", "models/*.joblib", _models)
check("3. Mô hình", "Metric khớp báo cáo", _metrics)
check("4. Đầu ra", "gold_model_output/", _root_outputs)
check("4. Đầu ra", "Biểu đồ khảo sát gốc", _plots)
check("4. Đầu ra", "Fullstack outputs/", _fullstack_outputs)
check("5. Demo CLI", "demo/demo.py", _demo_cli)
check("6. Ứng dụng web", "REST API (TestClient)", _api)
check("6. Ứng dụng web", "Asset frontend offline", _frontend_assets)

width = max(len(r[1]) for r in results) + 2
group_now = None
for group, name, status in results:
    if group != group_now:
        print(f"\n{group}")
        group_now = group
    print(f"  {name:<{width}} {status}")

fails = sum(1 for _, _, s in results if s.startswith(FAIL))
warns = sum(1 for _, _, s in results if s.startswith(WARN))
print("\n" + "=" * 78)
if fails:
    print(f"❌ CÓ {fails} MỤC CHƯA SẴN SÀNG — sửa theo gợi ý ở trên rồi chạy lại.")
    sys.exit(1)
if warns:
    print(f"⚠️  {warns} cảnh báo nhỏ (không chặn demo) — xem dòng WARN ở trên.\n")

print("🎉 TẤT CẢ ĐÃ SẴN SÀNG. Ba lệnh để demo:\n")
print("  # (1) Bảng số liệu + biểu đồ bằng CLI — không cần server")
print("  .venv/bin/python gold_price_fullstack/demo/demo.py\n")
print("  # (2) Dashboard web")
print("  cd gold_price_fullstack && ../.venv/bin/python -m uvicorn backend.app:app \\")
print("      --host 0.0.0.0 --port 8000")
print("  #     rồi mở http://localhost:8000\n")
print("  # (3) Chạy lại toàn bộ pipeline để chứng minh kết quả tái lập được")
print("  .venv/bin/python gold_predict.py && .venv/bin/python gold_extra_outputs.py\n")
print("Chi tiết kịch bản từng phút: HUONG_DAN_DEMO.md")
print("=" * 78)
