"""Test cho folder model_joblib — chạy độc lập, không cần gold_price_fullstack.

    .venv/bin/python -m pytest model_joblib/test_joblib_loader.py -q
    .venv/bin/python model_joblib/test_joblib_loader.py      # không cần pytest

Kiểm tra 4 điều:
  1. Nạp đủ 4 file .joblib + metadata.
  2. Đường dự báo sinh đúng 25 đặc trưng, đúng thứ tự trong metadata.
  3. Tái lập bảng Test 2024 khớp metadata tới 1e-9 (chống lệch do đổi phiên bản).
  4. Kết quả trùng khớp với demo gốc của repo (nếu có).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from joblib_loader import (  # noqa: E402
    MODEL_FILES, load_models, load_raw, make_features, predict,
    predict_next_session, score_test_split,
)

MODELS = load_models()
RAW = load_raw()


def test_artifacts_loaded():
    for fname in list(MODEL_FILES.values()) + ["imputer.joblib"]:
        assert (HERE / "artifacts" / fname).exists(), f"thiếu {fname}"
    assert len(MODELS["features"]) == 25
    assert MODELS["metadata"]["best_k"] == {"RandomForest": 11, "XGBoost": 165}


def test_feature_contract():
    X, y, target_dates = make_features(RAW, drop_unlabeled=True)
    assert list(X.columns) == MODELS["features"], "thứ tự cột lệch metadata"
    assert len(X) == len(y) == len(target_dates)
    assert X.notna().all(axis=None), "make_features(drop_unlabeled=True) còn NaN"
    # chống rò rỉ: đặc trưng tại t không được chứa close(t+1)
    assert (X["close_lag_0"].to_numpy() == RAW["Gold_Close"].loc[X.index].to_numpy()).all()


def test_test_split_matches_metadata():
    ref = MODELS["metadata"]["test_metrics"]
    tbl = score_test_split(MODELS, RAW)
    for name in ("Naive", "LinearRegression", "RandomForest", "XGBoost"):
        for col in ("MAE", "RMSE", "MAPE_pct", "R2"):
            assert abs(tbl.loc[name, col] - ref[name][col]) < 1e-9, \
                f"{name}.{col} lệch: {tbl.loc[name, col]} vs {ref[name][col]}"


def test_predict_next_session():
    s = predict_next_session(MODELS, RAW)
    assert set(s.index) == {"Naive", "LinearRegression", "RandomForest", "XGBoost"}
    # Naive = đúng giá đóng cửa phiên cuối
    assert abs(s["Naive"] - float(RAW["Gold_Close"].iloc[-1])) < 1e-9
    # mô hình cây bị chặn ngoại suy nên dự báo thấp hơn nhiều so với giá thực 2024
    assert s["RandomForest"] < 2_200, "RF lẽ ra bị chặn quanh 2.040"


def test_close_override():
    s = predict_next_session(MODELS, RAW, close_override=2650.0)
    assert abs(s["Naive"] - 2650.0) < 1e-9
    base = predict_next_session(MODELS, RAW)
    assert not abs(s["LinearRegression"] - base["LinearRegression"]) < 1e-9


def test_matches_repo_demo():
    """Nếu repo gốc có mặt, kết quả phải trùng tuyệt đối với demo.load_bundle."""
    fs = HERE.parent / "gold_price_fullstack"
    if not (fs / "demo" / "demo.py").exists():
        print("bỏ qua: không thấy gold_price_fullstack/demo/demo.py")
        return
    sys.path.insert(0, str(fs))
    from demo import demo as repo_demo  # noqa: E402

    bundle, raw2, _ = repo_demo.load_bundle()
    for close in (None, 2650.0):
        mine = predict_next_session(MODELS, RAW, close_override=close)
        theirs, _ = repo_demo.predict_all(bundle, raw2, raw2.index[-1], close_override=close)
        for k in theirs:
            assert abs(float(mine[k]) - float(theirs[k])) < 1e-9, f"{k} lệch khi close={close}"


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
