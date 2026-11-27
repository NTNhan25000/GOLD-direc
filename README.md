# GOLD-direc — Dự báo giá vàng phiên kế tiếp (Nhóm 8)

Đồ án môn **Trí tuệ Nhân tạo**: dự báo giá đóng cửa **một bước trước** (`t+1`) của hợp đồng
tương lai vàng COMEX `GC=F` (USD/troy ounce), so sánh **Naive Baseline**, **Linear Regression**,
**Random Forest** và **XGBoost** trên cùng một quy trình chống rò rỉ dữ liệu.

- **Dữ liệu:** 2.514 phiên 2015-01-02 → 2024-12-31 (Yahoo Finance: GC=F, DXY, ^TNX, ^GSPC, CL=F, SI=F).
- **Chia tập theo thời gian** (căn theo ngày của giá mục tiêu): Train ≤ 2021 (1.739 mẫu) ·
  Validation 2022–2023 (502) · Test 2024 (252). Không shuffle.
- **25 đặc trưng:** giá trễ 0/1/2/3/5/10 phiên, SMA & độ biến động 5/10/20 phiên, RSI-14,
  lợi suất vàng 1 phiên, biên độ High–Low, giá & lợi suất 1 phiên của DXY/US10Y/S&P 500/WTI/Bạc.
- **K** (số cây / số vòng boosting) chọn bằng RMSE Validation; Test chỉ dùng để đánh giá cuối.

## Kết quả Test 2024 (khớp báo cáo)

| Mô hình | K | MAE (USD/oz) | RMSE | MAPE | R² |
|---|---:|---:|---:|---:|---:|
| **Naive Baseline** | — | **18.038** | **23.743** | **0.744%** | **0.9891** |
| Linear Regression | — | 22.054 | 27.070 | 0.906% | 0.9858 |
| Random Forest | 11 | 364.529 | 426.324 | 14.398% | −2.5165 |
| XGBoost | 165 | 369.730 | 432.978 | 14.597% | −2.6271 |

Naive thắng trên Test 2024 vì giá vàng tăng vượt xa miền quan sát của Train — mô hình cây
không ngoại suy được. Đây là **kết quả hợp lệ về mặt học thuật** và được phân tích trung thực
trong báo cáo; chi tiết xem [`KET_QUA_DU_BAO_GIA_VANG.md`](KET_QUA_DU_BAO_GIA_VANG.md).

## Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate     # (Windows: .venv\Scripts\activate)
python -m pip install -r requirements.txt
```

Repo đã kèm sẵn dataset, **không cần gọi API** khi chạy lại.

## Chạy pipeline

```bash
# 1) Biểu đồ khảo sát dữ liệu & 4 tập chia
python plot_splits.py
#    → gold_train_plot.png, gold_val_plot.png, gold_test_plot.png, gold_comprehensive_analysis.png

# 2) Pipeline chính: 4 mô hình + quét K = 1..500 trên Validation
python gold_predict.py                       # ~6 giây (quét K tăng tốc bằng tiền tố cây tích lũy)
python gold_predict.py --fast                # dùng K preset RF=11, XGB=165
#    → gold_model_output/{test_metrics,k_validation_test,K_test_plan,test_predictions,
#      all_model_test_predictions,annual_test_metrics}.csv

# 3) Phân tích mở rộng (OLS, importance, ablation, ΔClose, biểu đồ)
python gold_extra_outputs.py                 # ~36 giây
#    → gold_model_output/{ols_diagnostics,quarterly_test_metrics,rf_feature_importance,
#      xgb_feature_importance,xgb_learning_curve,model_detailed_summary,
#      ablation_macro_study,delta_target_experiment}.csv + extra_diagnostics.json + 5 PNG
```

## Demo trước giám thị

```bash
python demo_preflight.py      # kiểm tra 12 mục: môi trường, dữ liệu, model, output, API, frontend
```

Script in **PASS/FAIL cho từng mục** kèm lệnh sửa nếu thiếu, rồi in 3 lệnh demo.
Kịch bản chi tiết từng phút (gồm bộ câu hỏi phản biện và phương án dự phòng khi sự cố):
[`HUONG_DAN_DEMO.md`](HUONG_DAN_DEMO.md).

## Code huấn luyện độc lập

[`model_training/`](model_training/) là bản copy tự chạy của `gold_price_fullstack/ml/`
(features + train), kèm dữ liệu — không import gì từ `gold_price_fullstack/`:

```bash
.venv/bin/python model_training/train.py            # quét K 1..500 trên Validation (~6 giây)
.venv/bin/python model_training/train.py --fast     # dùng K preset RF=11, XGB=165 (~5 giây)
.venv/bin/python -m pytest model_training/test_train.py -q   # 6 test
```

Tự tìm lại đúng **K = 11 / 165** và tái lập bảng Test lệch **0.00e+00** so với báo cáo.
Artifact sinh ra dùng lẫn được với `model_joblib/joblib_loader.py`.
Chi tiết: [`model_training/README.md`](model_training/README.md).

## Folder `.joblib` độc lập

[`model_joblib/`](model_joblib/) là folder **tự chạy được một mình**, chứa 4 file mô hình đã
huấn luyện + code nạp + dữ liệu cần thiết. Không import gì từ `gold_price_fullstack/`:

```bash
.venv/bin/python model_joblib/joblib_loader.py        # mô tả model + dự báo + tái lập bảng Test
.venv/bin/python -m pytest model_joblib/test_joblib_loader.py -q   # 6 test
```

Chi tiết (thứ tự imputer → model, lỗi chuẩn hoá 2 lần, lưu ý phiên bản sklearn/xgboost):
[`model_joblib/README.md`](model_joblib/README.md).

## Slide thuyết trình

Bộ slide chính thức: **`slides_thuyet_trinh_nhom8.pptx` — 10 slide, nền trắng, 16:9**
(dựng lại bằng `python make_slides.py`, kiểm tra layout bằng `python check_slides.py`).
Lời thoại từng slide, phân bổ thời gian 13 phút, bộ câu hỏi phản biện và bản rút gọn 7 phút:
[`KICH_BAN_THUYET_TRINH.md`](KICH_BAN_THUYET_TRINH.md).

## Ứng dụng fullstack

```bash
cd gold_price_fullstack
python ml/train.py                     # huấn luyện lại + xuất models/*.joblib (~6 giây)
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Mở `http://localhost:8000` — dashboard gồm: giá & dự báo phiên kế tiếp, biểu đồ lịch sử theo
tập, thực tế vs. dự báo Test 2024, bảng metric, **biểu đồ độ quan trọng đặc trưng (RF/XGB)**,
sai số theo quý/năm, và form dự báo thủ công. Chi tiết: [`gold_price_fullstack/README.md`](gold_price_fullstack/README.md).

```bash
# Demo CLI không cần server
python gold_price_fullstack/demo/demo.py --close 2650

# Kiểm thử tự động (16 test: chống rò rỉ, đúng metric báo cáo, toàn bộ API)
python -m pytest gold_price_fullstack/tests -q
```

## Cấu trúc repository

```
.
├── Báo_cáo_Giá_vàng_nhóm_8.docx          # báo cáo gốc (không chỉnh sửa)
├── README.md                              # file này
├── HUONG_DAN_DEMO.md                      # kịch bản demo + câu hỏi phản biện
├── KICH_BAN_THUYET_TRINH.md               # kịch bản thuyết trình 10 slide + Q&A
├── slides_thuyet_trinh_nhom8.pptx         # bộ slide 10 slide (nền trắng)
├── make_slides.py                         # dựng lại .pptx từ CSV (không gõ tay số)
├── model_training/                        # code huấn luyện độc lập (features + train + data)
├── model_joblib/                          # folder .joblib độc lập (loader + test + data)
├── make_importance_chart.py               # vẽ biểu đồ importance cỡ slide
├── check_slides.py                        # dò tràn chữ / đè lớp trong .pptx
├── preview_slides.py                      # render .pptx ra PNG để tự kiểm tra layout
├── Y_TUONG_SLIDE.md                       # dàn ý & câu chuyện xuyên suốt của slide
├── demo_preflight.py                      # kiểm tra sẵn sàng demo (12 mục)
├── KET_QUA_DU_BAO_GIA_VANG.md             # phân tích kết quả & tiêu chí đề tài
├── CHI_SO_DU_BAO_THEO_NAM.md              # bảng chỉ số theo năm
├── requirements.txt
├── Scripts.py                             # thu thập dữ liệu (yfinance/stooq/local) + chia tập
├── plot_splits.py                         # vẽ biểu đồ khảo sát dữ liệu
├── gold_predict.py                        # pipeline ML chính (4 mô hình, quét K)
├── gold_extra_outputs.py                  # phân tích thống kê & thực nghiệm mở rộng
├── gold_forecast_dataset_yfinance.csv     # dataset gốc
├── gold_{train,val,test}_yfinance.csv     # các tập đã chia
├── gold_model_output/                     # toàn bộ bảng CSV, JSON, biểu đồ của pipeline
└── gold_price_fullstack/                  # ứng dụng web (FastAPI + dashboard + CLI demo + test)
```

## Ghi chú kỹ thuật

- **Chống rò rỉ:** imputer/scaler chỉ `fit` trên Train; mọi đặc trưng tại phiên `t` chỉ dùng dữ liệu ≤ `t`.
- **Quét K tăng tốc:** cây thứ `1..k` của `RandomForest(n_estimators=500, random_state=42)`
  trùng khớp tuyệt đối với mô hình huấn luyện riêng `k` cây (sai khác < 1e-13), và XGBoost
  hỗ trợ `booster.predict(..., iteration_range=(0, k))` — nên quét 500 giá trị K chỉ mất vài giây
  thay vì vài phút. `pytest` có test khẳng định tính tương đương này.
- **Giới hạn ngoại suy:** dự báo cao nhất của RF/XGB trên Test chỉ ~2.080/2.046 USD trong khi giá
  thực tế lên tới 2.801 USD — nguyên nhân gốc của sai số lớn (xem `extra_diagnostics.json`).
  Thực nghiệm đổi mục tiêu sang `ΔClose(t+1)` khắc phục được vấn đề này (MAE XGB 18.00 USD).

> ⚠️ Công cụ nghiên cứu học thuật — **không phải khuyến nghị đầu tư**.
