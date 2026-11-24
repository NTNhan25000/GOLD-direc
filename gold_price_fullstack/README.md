# Dự đoán Giá Vàng — Fullstack (GC=F)

Dự án **fullstack dự đoán giá vàng COMEX (`GC=F`, USD/troy ounce)** cho phiên giao dịch
kế tiếp, xây dựng theo báo cáo môn Trí tuệ Nhân tạo — *Nhóm 8*.

- **Backend:** FastAPI (REST API + phục vụ frontend tĩnh)
- **ML:** Naive Baseline, Linear Regression, Random Forest, XGBoost
- **Frontend:** Dashboard HTML/CSS/JS + Chart.js
- **Dữ liệu:** 2.514 phiên 2015–2024 (Yahoo Finance), chia theo thời gian
  Train ≤ 2021 · Validation 2022–2023 · Test 2024 — không shuffle, không rò rỉ dữ liệu.

## Cấu trúc

```
gold_price_fullstack/
├── data/
│   ├── raw/gold_forecast_dataset.csv      # dữ liệu gốc 2015–2024
│   └── processed/{train,val,test}.csv      # tập đã chia + đặc trưng (25 cột)
├── ml/
│   ├── features.py                         # kỹ thuật đặc trưng (lag, SMA, RSI, vĩ mô…)
│   └── train.py                            # pipeline huấn luyện & đánh giá
├── backend/app.py                          # FastAPI — REST API + static frontend
├── frontend/                               # dashboard (index.html, style.css, app.js)
├── demo/
│   ├── demo.py                             # DEMO dự đoán 4 mô hình (CLI, không cần server)
│   └── README.md                           # hướng dẫn chạy demo
├── models/                                 # mô hình đã lưu (.joblib) + metadata.json
├── outputs/                                # test_metrics, test_predictions, importance, biểu đồ
├── tests/test_pipeline_and_api.py          # test: chống rò rỉ, metric, toàn bộ API
├── requirements.txt
└── README.md
```

## Cài đặt & chạy

```bash
# 1. Cài dependencies (ở repo root đã có .venv, hoặc tự tạo venv)
python -m pip install -r gold_price_fullstack/requirements.txt

# 2. Huấn luyện (quét đầy đủ K = 1..500 trên validation — chỉ ~6 giây)
cd gold_price_fullstack
python ml/train.py            # hoặc: python ml/train.py --fast (K preset RF=11, XGB=165)

# 3. Chạy server (backend + frontend cùng cổng)
uvicorn backend.app:app --host 0.0.0.0 --port 8000

# 4. Kiểm thử tự động
python -m pytest tests -q
```

Mở `http://localhost:8000` để xem dashboard.

Trước buổi bảo vệ, chạy `python demo_preflight.py` (ở thư mục gốc repo) để kiểm tra toàn bộ
12 mục sẵn sàng demo, và xem [`../HUONG_DAN_DEMO.md`](../HUONG_DAN_DEMO.md) cho kịch bản
từng phút cùng bộ câu hỏi phản biện.

> **Quét K tăng tốc:** thay vì fit 500 mô hình riêng lẻ, `ml/train.py` fit
> `RandomForest(n_estimators=500, random_state=42)` **một lần** rồi lấy trung bình tích lũy
> trên `estimators_` (cây 1..k trùng khớp tuyệt đối mô hình fit riêng k cây, sai khác < 1e-13),
> và với XGBoost dùng `booster.predict(..., iteration_range=(0, k))`. Nhờ vậy toàn bộ pipeline
> chạy dưới 6 giây thay vì vài phút, cho **cùng kết quả chính xác**. `tests/` có test khẳng định
> tính tương đương này.

## Dashboard có gì

| Khối | Nội dung |
|---|---|
| Thẻ chỉ số | Giá đóng cửa gần nhất + thay đổi, sai số Naive, mô hình học tốt nhất, số phiên từng tập (lấy từ `/api/metadata`) |
| Dự báo phiên kế tiếp | 4 mô hình + chênh lệch so với Naive |
| Biểu đồ lịch sử | Toàn bộ 2015–2024, hoặc lọc theo Train/Validation/Test |
| Thực tế vs. dự báo | 4 đường dự báo trên Test 2024 + bảng MAE/RMSE/MAPE/R² |
| **Feature importance** | Biểu đồ thanh ngang 15 đặc trưng hàng đầu, chuyển đổi **Random Forest (MDI)** ↔ **XGBoost (Gain)** |
| Sai số theo quý / năm | Bảng MAE từng quý Q1–Q4 và từng năm (đọc từ `gold_model_output/`) |
| Dự báo thủ công | Chỉ cần nhập giá đóng cửa; OHLC bỏ trống tự suy ra, biến vĩ mô bỏ trống lấy phiên gần nhất |

## Demo nhanh 4 mô hình (không cần server)

```bash
cd gold_price_fullstack
python demo/demo.py                     # dự báo phiên kế tiếp + 5 phiên Test gần nhất
python demo/demo.py --close 2650        # tự nhập giá đóng cửa phiên t
python demo/demo.py --date 2024-06-14 --last 10
```

Chi tiết: [`demo/README.md`](demo/README.md).

## API

| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/api/health` | Trạng thái server, mô hình đã nạp, số đặc trưng |
| GET | `/api/metadata` | Siêu thông tin mô hình, chia tập, K đã chọn, khoảng dữ liệu |
| GET | `/api/history?start=&end=` | Chuỗi giá lịch sử |
| GET | `/api/metrics` | Bảng MAE/RMSE/MAPE/R² trên Test |
| GET | `/api/predictions` | Dự báo chi tiết từng phiên Test 2024 |
| GET | `/api/importance?model=rf\|xgb` | Feature importance đủ 25 đặc trưng: `rf` = MDI (%), `xgb` = Gain (%) kèm gain/weight/SHAP |
| GET | `/api/quarterly` | MAE/RMSE theo quý trên Test (đọc `gold_model_output/`) |
| GET | `/api/annual` | Chỉ số theo năm (đọc `gold_model_output/`) |
| GET | `/api/diagnostics` | OLS meta, OOB score, trần ngoại suy RF/XGB |
| GET | `/api/last-session` | Phiên gần nhất + dự báo phiên kế tiếp |
| POST | `/api/predict` | Dự báo thủ công (`gold_close` là bắt buộc) |

`POST /api/predict` chấp nhận payload tối giản — chỉ cần giá đóng cửa:

```bash
curl -s -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' -d '{"gold_close": 2650}'
```

Hoặc payload đầy đủ (OHLC + 5 biến vĩ mô):

```bash
curl -s -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"date":"2024-12-31","gold_open":2620,"gold_high":2640,"gold_low":2610,"gold_close":2635,
       "dxy_close":108.5,"us10y_close":4.55,"sp500_close":5881.6,"wti_close":71.7,"silver_close":29.0}'
```

Quy tắc: `gold_open`/`gold_high`/`gold_low` bỏ trống → suy ra từ `gold_close`; biến vĩ mô bỏ trống
→ lấy giá trị phiên gần nhất trong dataset. Dữ liệu không hợp lệ (giá ≤ 0, `high < close`…)
trả về **422** kèm thông báo lỗi tiếng Việt.

## Kết quả Test 2024 (khớp báo cáo)

| Mô hình | K | MAE | RMSE | MAPE | R² |
|---|---:|---:|---:|---:|---:|
| Naive | — | **18.04** | **23.74** | **0.74%** | 0.9891 |
| Linear Regression | — | 22.05 | 27.07 | 0.91% | 0.9858 |
| Random Forest | 11 | 364.53 | 426.32 | 14.40% | −2.5165 |
| XGBoost | 165 | 369.73 | 432.98 | 14.60% | −2.6271 |

Naive Baseline tốt nhất trên Test 2024 (giá vàng tăng ra ngoài miền đã học — mô hình cây
không ngoại suy được). Xem chi tiết: `outputs/test_metrics.csv` và báo cáo gốc trong repo.

> ⚠️ Công cụ nghiên cứu học thuật — **không phải khuyến nghị đầu tư**.
