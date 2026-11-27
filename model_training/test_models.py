"""Test cho 4 file mô hình rời — canh chúng khớp nhau VÀ khớp báo cáo.

    .venv/bin/python -m pytest model_training/test_models.py -q
    .venv/bin/python model_training/test_models.py

Điều đáng canh nhất: 4 file mô hình rời, file `train.py` gộp, và `model_joblib/`
phải cho ra **cùng một bộ số**. Nếu ai sửa siêu tham số ở một file mà quên file khác,
test này bắt được.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from baseline_naive import NaiveBaseline  # noqa: E402
from features import latest_features, load_raw, make_features, scores, split_masks  # noqa: E402
from linear_regression import LinearRegressionModel  # noqa: E402
from random_forest import PRESET_K as RF_K, RandomForestModel  # noqa: E402
from xgboost_model import PRESET_K as XGB_K, XGBoostModel  # noqa: E402

RAW = load_raw()
X, Y, DATES = make_features(RAW)
M = split_masks(DATES)

XTR, YTR = X.loc[M["train"]], Y.loc[M["train"]]
XTV = pd.concat([X.loc[M["train"]], X.loc[M["val"]]])
YTV = pd.concat([Y.loc[M["train"]], Y.loc[M["val"]]])
XTE, YTE = X.loc[M["test"]], Y.loc[M["test"]]

# Fit một lần cho cả module (đủ nhanh: RF 11 cây + XGB 165 vòng).
MODELS = {
    "Naive": NaiveBaseline().fit(XTR, YTR),
    "LinearRegression": LinearRegressionModel().fit(XTR, YTR),
    "RandomForest": RandomForestModel(k=RF_K).fit(XTV, YTV),
    "XGBoost": XGBoostModel(k=XGB_K).fit(XTV, YTV),
}
PRED = {name: m.predict(XTE) for name, m in MODELS.items()}


def test_splits():
    assert (int(M["train"].sum()), int(M["val"].sum()), int(M["test"].sum())) == (1739, 502, 252)
    assert len(X.columns) == 25


def test_preset_k():
    assert (RF_K, XGB_K) == (11, 165)


def test_each_model_has_own_params():
    """Mỗi file mô hình phải mang siêu tham số riêng, không đọc ké của file khác."""
    rf = MODELS["RandomForest"]
    assert rf.model.n_estimators == 11
    assert rf.model.max_features == 0.8
    assert rf.model.min_samples_leaf == 2
    xg = MODELS["XGBoost"]
    assert xg.model.n_estimators == 165
    assert xg.model.max_depth == 3
    assert abs(xg.model.learning_rate - 0.03) < 1e-12
    lr = MODELS["LinearRegression"]
    assert [s[0] for s in lr.model.steps] == ["standardscaler", "linearregression"]


def test_metrics_match_report():
    ref = pd.read_csv(REPO / "gold_model_output" / "test_metrics.csv").set_index("model")
    for name, pred in PRED.items():
        assert abs(ref.loc[name, "MAE"] - scores(YTE, pred)["MAE"]) < 1e-9, name
        assert abs(ref.loc[name, "RMSE"] - scores(YTE, pred)["RMSE"]) < 1e-9, name


def test_predictions_agree_with_joblib_loader():
    """4 file mô hình rời phải trùng model_joblib/ (nguồn đang dùng cho dashboard)."""
    loader_dir = REPO / "model_joblib"
    if not (loader_dir / "joblib_loader.py").exists():
        print("bỏ qua: không thấy model_joblib/")
        return
    sys.path.insert(0, str(loader_dir))
    from joblib_loader import load_models, predict

    ref = predict(load_models(), XTE)
    for name, pred in PRED.items():
        assert np.max(np.abs(ref[name].to_numpy() - pred)) < 1e-9, f"{name} lệch"


def test_next_session_uses_last_row():
    """Bẫy đã từng mắc: dùng drop_unlabeled=True thì origin lùi một phiên."""
    last = latest_features(RAW)
    assert len(last) == 1
    close = float(last["close_lag_0"].iloc[0])
    assert abs(close - float(RAW["Gold_Close"].iloc[-1])) < 1e-9, \
        f"origin sai: {close} vs {RAW['Gold_Close'].iloc[-1]}"
    # NaN không được lọt vào hàng suy luận (rolling 20 cần lịch sử, đã đủ)
    assert int(last.isna().sum().sum()) == 0


def test_naive_is_identity():
    assert np.allclose(PRED["Naive"], XTE["close_lag_0"].to_numpy())


def test_tree_predictions_are_capped():
    """Đặc trưng cốt lõi của bài: cây bị chặn trên ở đỉnh Train (2.069,40)."""
    for name in ("RandomForest", "XGBoost"):
        assert PRED[name].max() < 2_200, f"{name} dự báo {PRED[name].max():.0f} — lẽ ra bị chặn"
    assert YTE.max() > 2_700, "giá thực Test phải vượt xa trần dự báo"


def test_close_override():
    """Giá nhập tay phải thay được phiên cuối, và High/Low phải nới ra bao lấy nó."""
    last = latest_features(RAW, close_override=2650.0)
    assert abs(float(last["close_lag_0"].iloc[0]) - 2650.0) < 1e-9

    # latest_features trả ma trận 25 đặc trưng, không có OHLC → kiểm tra qua gold_range.
    # gold_range = (High − Low)/Close; nếu High/Low không nới theo giá mới thì
    # đại lượng này sẽ âm hoặc vô lý.
    rng = float(last["gold_range"].iloc[0])
    assert rng >= 0, f"gold_range âm ({rng}) — High/Low không bao lấy giá mới"
    base_rng = float(latest_features(RAW)["gold_range"].iloc[0])
    assert rng >= base_rng - 1e-9, "biên độ phải rộng ra khi giá mới nằm ngoài High/Low cũ"


def test_close_override_reaches_models():
    """Giá nhập tay phải lan tới dự báo của cả 4 mô hình, không bị bỏ qua âm thầm."""
    last = latest_features(RAW, close_override=2650.0)
    preds = {name: float(m.predict(last)[0]) for name, m in MODELS.items()}
    assert abs(preds["Naive"] - 2650.0) < 1e-9
    # LR bám giá hiện tại nên phải gần 2650 hơn hẳn so với khi dùng giá gốc 2641
    default_lr = float(MODELS["LinearRegression"].predict(latest_features(RAW))[0])
    assert preds["LinearRegression"] > default_lr


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
