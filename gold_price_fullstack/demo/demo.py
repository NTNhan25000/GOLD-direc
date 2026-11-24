"""DEMO dự đoán giá vàng — 4 mô hình: Naive, Linear Regression, Random Forest, XGBoost.

Chạy từ thư mục gold_price_fullstack (hoặc bất kỳ đâu, script tự tìm đường dẫn):

    python demo/demo.py                  # dự báo phiên kế tiếp + 5 phiên Test gần nhất
    python demo/demo.py --last 10        # số phiên Test gần nhất cần hiển thị
    python demo/demo.py --date 2024-06-14  # dự báo bắt đầu từ phiên cụ thể
    python demo/demo.py --close 2650     # tự nhập giá đóng cửa (không cần --date)

Không cần server API — script tự tải models/*.joblib đã huấn luyện.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.features import make_features  # noqa: E402

MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "raw" / "gold_forecast_dataset.csv"

MODEL_LABELS = {
    "Naive": "Naive Baseline",
    "LinearRegression": "Linear Regression",
    "RandomForest": "Random Forest",
    "XGBoost": "XGBoost",
}


def load_bundle():
    if not (MODELS_DIR / "metadata.json").exists():
        sys.exit("Chưa có mô hình. Chạy trước:  python ml/train.py")
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
    bundle = {
        "metadata": metadata,
        "imputer": joblib.load(MODELS_DIR / "imputer.joblib"),
        "linear": joblib.load(MODELS_DIR / "linear_regression.joblib"),
    }
    for name in ("randomforest", "xgboost"):
        p = MODELS_DIR / f"{name}.joblib"
        if p.exists():
            bundle[name] = joblib.load(p)
    raw = pd.read_csv(DATA_PATH, parse_dates=["Date"]).set_index("Date").sort_index()
    return bundle, raw, metadata


def predict_all(bundle: dict, raw: pd.DataFrame, origin: pd.Timestamp,
                close_override: float | None = None) -> tuple[dict[str, float], pd.DataFrame]:
    """Dự báo 4 mô hình cho phiên sau `origin`.

    Trả về (dict dự báo, DataFrame đặc trưng tại origin).
    """
    base = raw.loc[:origin].copy()
    if base.empty:
        raise SystemExit(f"Không có dữ liệu đến ngày {origin.date()}.")
    if close_override is not None:
        base.iloc[-1, base.columns.get_loc("Gold_Close")] = close_override
        base.iloc[-1, base.columns.get_loc("Gold_High")] = max(
            float(base.iloc[-1]["Gold_High"]), close_override)
        base.iloc[-1, base.columns.get_loc("Gold_Low")] = min(
            float(base.iloc[-1]["Gold_Low"]), close_override)

    # drop_unlabeled=False: giữ phiên origin làm dòng cuối để dự báo t+1.
    X, _, _ = make_features(base, drop_unlabeled=False)
    x_last = X.iloc[[-1]]
    Xi = bundle["imputer"].transform(x_last)

    preds = {"Naive": float(x_last["close_lag_0"].iloc[0])}
    preds["LinearRegression"] = float(bundle["linear"].predict(Xi)[0])
    if "randomforest" in bundle:
        preds["RandomForest"] = float(bundle["randomforest"].predict(Xi)[0])
    if "xgboost" in bundle:
        preds["XGBoost"] = float(bundle["xgboost"].predict(Xi)[0])
    return preds, x_last


def fmt(v: float) -> str:
    return f"{v:>10,.2f}"


def demo_next_session(bundle, raw, metadata, close_override=None, origin=None):
    origin = origin or raw.index[-1]
    preds, _ = predict_all(bundle, raw, origin, close_override)
    actual_close = float(raw.loc[origin, "Gold_Close"])

    print("=" * 78)
    print(f" DEMO DỰ ĐOÁN GIÁ VÀNG GC=F — 4 MÔ HÌNH")
    print(f" Phiên phát dự báo (t): {origin.date()}"
          + (f"  ·  giá đóng cửa nhập: {close_override:,.2f}" if close_override else ""))
    print(f" Mục tiêu: giá đóng cửa phiên kế tiếp (t+1), USD/troy ounce")
    print("=" * 78)
    print(f"{'Mô hình':<22}{'Dự báo t+1':>14}{'Chênh vs Naive':>18}{'Ghi chú':>20}")
    print("-" * 78)
    base = preds["Naive"]
    notes = {
        "Naive": "mốc cơ sở",
        "LinearRegression": "học được",
        "RandomForest": "K=11 cây",
        "XGBoost": "K=165 vòng",
    }
    for name, val in preds.items():
        delta = val - base
        print(f"{MODEL_LABELS[name]:<22}{val:>14,.2f}{delta:>+18,.2f}{notes[name]:>20}")
    print("-" * 78)
    if close_override is None:
        print(f" Giá đóng cửa phiên t thực tế: {actual_close:,.2f} USD/oz "
              f"(Naive = giữ nguyên giá này)")
    return preds


def demo_test_tail(bundle, raw, metadata, n=5, start=None):
    """Dự báo n phiên cuối (hoặc từ --date) trên tập Test và đối chiếu thực tế."""
    X, y, target_dates = make_features(raw, drop_unlabeled=True)
    test_mask = target_dates > pd.Timestamp("2023-12-31")
    X_t, y_t, td_t = X[test_mask], y[test_mask], target_dates[test_mask]

    if start is not None:
        sel = td_t >= pd.Timestamp(start)
    else:
        sel = pd.Series(False, index=td_t.index)
        sel.iloc[-n:] = True
    idx = td_t.index[sel]
    if len(idx) == 0:
        raise SystemExit("Không có phiên Test nào khớp điều kiện.")

    print("=" * 100)
    print(f" DEMO TRÊN TẬP TEST — đối chiếu dự báo với GIÁ THỰC TẾ ({len(idx)} phiên)")
    print("=" * 100)
    header = (f"{'Ngày t+1':<12}{'Thực tế':>10}{'Naive':>10}{'Linear':>10}"
              f"{'RF':>10}{'XGB':>10}   |  |sai số| nhỏ nhất")
    print(header)
    print("-" * 100)

    errs = {k: [] for k in ("Naive", "LinearRegression", "RandomForest", "XGBoost")}
    for i in idx:
        origin = i  # phiên phát dự báo = ngày gốc của hàng đặc trưng
        target_date = target_dates.loc[i]
        row_raw = raw.loc[:origin]
        preds, _ = predict_all(bundle, raw, origin)
        actual = float(y.loc[i])
        best = min(errs, key=lambda k: abs(preds[k] - actual))
        errs["Naive"].append(abs(preds["Naive"] - actual))
        errs["LinearRegression"].append(abs(preds["LinearRegression"] - actual))
        errs["RandomForest"].append(abs(preds["RandomForest"] - actual))
        errs["XGBoost"].append(abs(preds["XGBoost"] - actual))
        print(f"{target_date.strftime('%Y-%m-%d'):<12}{actual:>10,.2f}"
              f"{preds['Naive']:>10,.2f}{preds['LinearRegression']:>10,.2f}"
              f"{preds['RandomForest']:>10,.2f}{preds['XGBoost']:>10,.2f}"
              f"   |  {MODEL_LABELS[best]}")
    print("-" * 100)
    print("MAE trung bình trên các phiên demo:")
    for k, v in errs.items():
        print(f"  {MODEL_LABELS[k]:<22}{np.mean(v):>10,.2f} USD/oz")
    print("=" * 100)


def main():
    ap = argparse.ArgumentParser(description="Demo dự đoán giá vàng 4 mô hình")
    ap.add_argument("--last", type=int, default=5, help="số phiên Test gần nhất (mặc định 5)")
    ap.add_argument("--date", default=None, help="phiên phát dự báo, yyyy-mm-dd (bỏ qua nếu có --close)")
    ap.add_argument("--close", type=float, default=None, help="giá đóng cửa tự nhập cho phiên t")
    ap.add_argument("--skip-test", action="store_true", help="chỉ chạy dự báo phiên kế tiếp")
    args = ap.parse_args()

    bundle, raw, metadata = load_bundle()
    origin = pd.Timestamp(args.date) if args.date else None

    demo_next_session(bundle, raw, metadata,
                      close_override=args.close, origin=origin)
    print()

    if not args.skip_test:
        demo_test_tail(bundle, raw, metadata, n=args.last,
                       start=None if args.close else (origin if args.date else None))

    print("\n(Nhận xét: trên Test 2024 Naive thường thắng — giá vàng tăng ra ngoài")
    print(" miền mô hình cây đã học. Đây là demo học thuật, không phải khuyến nghị đầu tư.)")


if __name__ == "__main__":
    main()
