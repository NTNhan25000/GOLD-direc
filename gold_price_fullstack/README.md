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
├── outputs/                                # test_metrics, test_predictions, biểu đồ
├── requirements.txt
└── README.md
```

## Cài đặt & chạy

```bash
# 1. Cài dependencies (ở repo root đã có .venv, hoặc tự tạo venv)
python -m pip install -r gold_price_fullstack/requirements.txt

# 2. Huấn luyện (quét K=1..500 trên validation — ~2 phút)
cd gold_price_fullstack
python ml/train.py            # hoặc: python ml/train.py --fast (K preset RF=11, XGB=165)

# 3. Chạy server (backend + frontend cùng cổng)
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Mở `http://localhost:8000` để xem dashboard.

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
| GET | `/api/health` | Trạng thái server & mô hình |
| GET | `/api/metadata` | Siêu thông tin mô hình, chia tập, K đã chọn |
| GET | `/api/history?start=&end=` | Chuỗi giá lịch sử |
| GET | `/api/metrics` | Bảng MAE/RMSE/MAPE/R² trên Test |
| GET | `/api/predictions` | Dự báo chi tiết từng phiên Test 2024 |
| GET | `/api/importance` | Feature importance (Random Forest) |
| GET | `/api/last-session` | Phiên gần nhất + dự báo phiên kế tiếp |
| POST | `/api/predict` | Dự báo thủ công theo dữ liệu phiên do người dùng nhập |

Ví dụ:

```bash
curl -s http://localhost:8000/api/last-session | python -m json.tool
curl -s -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"date":"2024-12-31","gold_open":2620,"gold_high":2640,"gold_low":2610,"gold_close":2635}'
```

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
