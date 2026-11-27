"""Test cho folder model_training — kiểm tra code huấn luyện tái lập đúng báo cáo.

    .venv/bin/python -m pytest model_training/test_train.py -q
    .venv/bin/python model_training/test_train.py          # không cần pytest

Test chạy `train.py --fast` vào thư mục tạm (~1 giây) rồi so từng con số với
`gold_model_output/test_metrics.csv` của repo. Nếu ai sửa công thức đặc trưng,
siêu tham số hay cách chia tập, test sẽ fail ngay.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
GOLD = REPO / "gold_model_output" / "test_metrics.csv"
PY = sys.executable

TMP = Path(tempfile.mkdtemp(prefix="mt_test_"))
subprocess.run([PY, str(HERE / "train.py"), "--fast", "--out", str(TMP)],
               check=True, capture_output=True, text=True)

TRAINED = pd.read_csv(TMP / "outputs" / "test_metrics.csv").set_index("model")
REF = pd.read_csv(GOLD).set_index("model") if GOLD.exists() else None
META = json.loads((TMP / "artifacts" / "metadata.json").read_text(encoding="utf-8"))


def test_artifacts_written():
    for name in ("imputer", "linear_regression", "randomforest", "xgboost"):
        assert (TMP / "artifacts" / f"{name}.joblib").exists(), f"thiếu {name}.joblib"
    assert (TMP / "outputs" / "predictions_test.csv").exists()
    assert (TMP / "outputs" / "splits.csv").exists()


def test_preset_k():
    assert META["best_k"] == {"RandomForest": 11, "XGBoost": 165}


def test_feature_contract():
    assert len(META["features"]) == 25
    assert META["n_samples"] == {"train": 1739, "val": 502, "test": 252}
    assert META["n_sessions"] == 2514


def test_metrics_match_report():
    if REF is None:
        print("bỏ qua: không thấy gold_model_output/test_metrics.csv")
        return
    for model in ("Naive", "LinearRegression", "RandomForest", "XGBoost"):
        for col in ("MAE", "RMSE", "MAPE_pct", "R2"):
            a, b = TRAINED.loc[model, col], REF.loc[model, col]
            assert abs(a - b) < 1e-9, f"{model}.{col}: {a} vs {b}"


def test_predictions_columns():
    p = pd.read_csv(TMP / "outputs" / "predictions_test.csv")
    assert len(p) == 252
    for col in ("Actual", "Naive", "LinearRegression", "RandomForest", "XGBoost"):
        assert col in p.columns, f"thiếu cột {col}"
    assert p["Actual"].notna().all()


def test_artifacts_load_and_predict():
    """Artifact do train.py sinh ra phải nạp được bằng model_joblib/joblib_loader."""
    loader_dir = REPO / "model_joblib"
    if not (loader_dir / "joblib_loader.py").exists():
        print("bỏ qua: không thấy model_joblib/")
        return
    sys.path.insert(0, str(loader_dir))
    from joblib_loader import load_models, load_raw, score_test_split

    models = load_models(TMP / "artifacts")
    tbl = score_test_split(models, load_raw())
    for model in ("Naive", "LinearRegression", "RandomForest", "XGBoost"):
        assert abs(tbl.loc[model, "MAE"] - TRAINED.loc[model, "MAE"]) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} test PASS")
    sys.exit(1 if failed else 0)
