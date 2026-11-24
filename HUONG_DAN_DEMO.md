# Hướng dẫn demo trước giám thị

> Mục tiêu: trong **5–7 phút** chứng minh được 3 điều —
> (1) kết quả trong báo cáo **tái lập được trên máy**,
> (2) có **sản phẩm chạy thật** (dashboard + API),
> (3) hiểu **vì sao Naive thắng** và đã kiểm chứng bằng thực nghiệm.
>
> Kịch bản dưới đây giả định máy đã cài xong (xem bước 0). Không cần mạng khi demo.

---

## 0. Chuẩn bị trước 10 phút (chỉ làm 1 lần)

```bash
cd /home/user/GOLD-direc

# Cài môi trường (nếu máy mới)
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# Sinh lại mọi mô hình + bảng + biểu đồ (tổng cộng ~50 giây)
cd gold_price_fullstack && ../.venv/bin/python ml/train.py && cd ..
.venv/bin/python plot_splits.py
.venv/bin/python gold_predict.py
.venv/bin/python gold_extra_outputs.py
```

**Kiểm tra lần cuối — đây là thứ đầu tiên bạn nên chạy trước mặt giám thị:**

```bash
.venv/bin/python demo_preflight.py
```

Kết quả phải là **12 dòng PASS** và dòng cuối `🎉 TẤT CẢ ĐÃ SẴN SÀNG`. Nếu có FAIL, script
in luôn lệnh cần chạy để sửa.

> Nếu giám thị muốn xem trên máy của họ: chỉ cần copy cả thư mục, chạy bước 0 rồi bước 1.
> Không có bước nào cần Internet (Chart.js đã nằm sẵn trong `frontend/chart.umd.js`).

---

## 1. Mở đầu — 30 giây

Mở **`Báo_cáo_Giá_vàng_nhóm_8.docx`** ở trang kết quả, chiếu bảng Test 2024:

| Mô hình | K | MAE | RMSE | MAPE | R² |
|---|---:|---:|---:|---:|---:|
| **Naive** | — | **18.038** | **23.743** | **0.744%** | **0.9891** |
| Linear Regression | — | 22.054 | 27.070 | 0.906% | 0.9858 |
| Random Forest | 11 | 364.529 | 426.324 | 14.398% | −2.5165 |
| XGBoost | 165 | 369.730 | 432.978 | 14.597% | −2.6271 |

Nói: *“Điểm bất thường của đề tài là mô hình cây thua Naive rất xa. Em sẽ chứng minh
(i) con số này tái lập được, (ii) tìm ra nguyên nhân định lượng, và (iii) có cách khắc phục.”*

---

## 2. Demo 1 — Tái lập kết quả bằng CLI (60 giây) ⭐ quan trọng nhất

```bash
.venv/bin/python gold_predict.py
```

In ra đúng bảng trên **trong ~5 giây** (quét đủ 500 giá trị K cho cả 2 mô hình).
Chỉ vào terminal: *“MAE, RMSE, MAPE, R² trùng báo cáo tới 6 chữ số. Toàn bộ chỉ mất 5 giây
nhờ quét K bằng tiền tố cây tích lũy — đã có test khẳng định tương đương tuyệt đối với
cách fit từng K riêng lẻ.”*

**Câu hỏi giám thị hay hỏi:** *“Sao quét 500 K mà nhanh vậy?”*
→ Mở `gold_predict.py` dòng ~110, chỉ đoạn `cumsum(tree_preds)/arange`:
cây thứ 1..k của một rừng 500 cây (seed 42) **chính là** rừng k cây. Kiểm chứng bằng
`pytest -k cumulative` — sai khác < 1e-13.

---

## 3. Demo 2 — Giải thích nguyên nhân (90 giây) ⭐ điểm học thuật

```bash
cat gold_model_output/extra_diagnostics.json
```

Chỉ 2 con số:
- `rf_extrapolation.max_pred_USD = 2079.82` — dự báo cao nhất mà RF có thể đưa ra
- `max_actual_USD = 2800.80`, `n_sessions_above_max_pred = 211`

Nói: *“Giá vàng 2024 tăng 27.4%, vượt xa đỉnh Train (2.069 USD). Cây quyết định dự báo bằng
trung bình lá nên **không thể** vượt 2.080 USD — 211/252 phiên nằm ngoài tầm với. Đây là
giới hạn cấu trúc của thuật toán, không phải lỗi code.”*

Rồi mở **`gold_model_output/delta_vs_level_comparison.png`**:
*“Khi đổi mục tiêu sang dự báo **mức thay đổi** ΔClose(t+1) thay vì mức giá, XGBoost đạt
MAE **18.0039** — **vượt Naive** (18.0381) và đúng chiều **55.95%**. Tức vấn đề nằm ở cách
đặt mục tiêu, không phải ở thuật toán.”*

---

## 4. Demo 3 — Dashboard web (2 phút) ⭐ phần “sản phẩm”

Terminal 1:
```bash
cd gold_price_fullstack
../.venv/bin/python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Mở trình duyệt: **http://localhost:8000**

Đi theo thứ tự từ trên xuống, mỗi khối **một câu**:

| Khối | Câu nói |
|---|---|
| 4 thẻ trên cùng | “Giá mới nhất 2.641 USD, Naive sai 18.04 USD/phiên trên Test” |
| **Dự báo phiên kế tiếp** | “Bốn mô hình dự báo phiên sau, kèm chênh lệch so với Naive” |
| Biểu đồ lịch sử | Bấm tab **Test (2024)** — “đây là vùng giá vượt đỉnh Train” |
| **Thực tế vs. Dự báo** | “Hai đường RF/XGB nằm ngang ~2.040 USD, không bám được giá thực — đúng như chẩn đoán” |
| Bảng metric | “Số y hệt báo cáo” |
| **Feature Importance** | Bấm đổi **RF ↔ XGBoost** — “cả hai đều cho thấy close_lag và SMA quyết định; biến vĩ mô đóng góp <1%” |
| Sai số theo quý | “Q1 sai 47 USD, Q4 sai 631 USD — sai số tăng đúng theo mức giá vượt trần” |
| **Dự báo thủ công** | Nhập `gold_close = 2650`, bấm **Dự báo** — “chỉ cần giá đóng cửa, hệ thống tự suy OHLC và lấy biến vĩ mô phiên gần nhất” |

---

## 5. Demo 4 — API thô (30 giây, chỉ khi giám thị hỏi “có API không?”)

Mở `http://localhost:8000/docs` — Swagger UI tự sinh, bấm **Try it out** trên `POST /api/predict`:

```json
{ "gold_close": 2650 }
```

Kết quả:
```json
{"origin_date":"2024-12-31","forecast_next_session":
 {"Naive":2650.0,"LinearRegression":2633.88,"RandomForest":2042.74,"XGBoost":2034.07}}
```

Hoặc trong terminal:
```bash
curl -s -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' -d '{"gold_close":2650}'
curl -s "http://localhost:8000/api/importance?model=xgb" | head -c 300
```

---

## 6. Demo 5 — Chứng minh chất lượng code (30 giây, nếu còn giờ)

```bash
.venv/bin/python -m pytest gold_price_fullstack/tests -q
# → 16 passed
```

Nói: *“16 test tự động: chống rò rỉ dữ liệu (feature tại t chỉ dùng dữ liệu ≤ t), cổng hồi quy
metric khớp báo cáo tới 1e-9, tính tương đương của quét K tăng tốc, và toàn bộ REST API
kể cả validate dữ liệu sai.”*

---

## 7. Bộ câu hỏi phản biện — trả lời trong 1 câu

| Giám thị hỏi | Trả lời |
|---|---|
| **“Mô hình cây thua Naive, vậy đề tài có ý nghĩa gì?”** | “Đây là kết quả trung thực và có giá trị: em chỉ ra **giới hạn ngoại suy** của mô hình cây bằng số liệu (211/252 phiên vượt trần 2.080 USD), và chứng minh khi đổi sang dự báo ΔClose thì XGBoost **vượt** Naive. Bài học: cách đặt mục tiêu quan trọng hơn chọn thuật toán.” |
| **“Sao không dùng LSTM/Transformer?”** | “Với horizon 1 phiên, giá hai phiên liên tiếp gần như nhau nên Naive là baseline rất mạnh; mô hình phức tạp cần thêm dữ liệu tần suất cao (intraday) mới có lợi thế. Bốn mô hình hiện tại đủ để trả lời câu hỏi nghiên cứu và so sánh công bằng.” |
| **“Có rò rỉ dữ liệu không?”** | “Không. Feature tại phiên t chỉ dùng dữ liệu ≤ t (`shift` dương, rolling trailing); imputer/scaler chỉ fit trên Train; chia tập căn theo **ngày giá mục tiêu** không shuffle. Có test `test_no_leakage_last_row_features` kiểm chứng từng giá trị.” |
| **“K chọn thế nào? Có dùng Test để chọn không?”** | “K chọn bằng **RMSE Validation**, quét đủ 1..500; Test chỉ dùng một lần để đánh giá cuối. Bảng đầy đủ ở `k_validation_test.csv`.” |
| **“Vì sao chọn RF=11 cây, ít vậy?”** | “Đó là K tốt nhất trên Validation (RMSE 28.413). Nhưng điểm quan trọng là **mọi K đều thất bại trên Test** — xem `k_validation_curve.png`: đường RF phẳng ở ~32 USD dù K = 1 hay 500, nên không phải chọn sai K mà là giới hạn của mô hình cây.” |
| **“Biến vĩ mô có giúp gì không?”** | “Không, và em đã làm ablation: bỏ 10 biến vĩ mô thì Linear Regression **tốt hơn** (Val RMSE 17.49 vs 19.03). Permutation importance trên Test của phần lớn biến vĩ mô còn âm.” |
| **“Số liệu này có tái lập được không?”** | “Có — chạy `python gold_predict.py` cho đúng bảng trong 5 giây; `demo_preflight.py` kiểm tra khớp tới 6 chữ số; pytest có cổng hồi quy khớp tới 1e-9.” |
| **“Sản phẩm này dùng được thật không?”** | “Đây là công cụ nghiên cứu học thuật, không phải khuyến nghị đầu tư — em ghi rõ trong README và trên dashboard. Về kỹ thuật thì pipeline chạy end-to-end: dữ liệu → đặc trưng → huấn luyện → API → dashboard.” |

---

## 8. Nếu có sự cố

| Sự cố | Xử lý ngay |
|---|---|
| Cổng 8000 bận | `pkill -f uvicorn` hoặc đổi `--port 8001` |
| `ModuleNotFoundError` | `.venv/bin/pip install -r requirements.txt` (mất ~40s) |
| Thiếu `models/*.joblib` | `cd gold_price_fullstack && ../.venv/bin/python ml/train.py` (~5s) |
| Dashboard trắng | Mở **http://127.0.0.1:8000** thay vì `localhost`; kiểm tra terminal có `Application startup complete` |
| Máy giám thị không có Python | Vẫn demo được bằng ảnh: mở thư mục `gold_model_output/` và `gold_price_fullstack/outputs/` — 12 biểu đồ PNG đã lưu sẵn |
| Hết giờ | Bỏ Demo 4 và 5, giữ **Demo 1 + 2 + 3** (đã đủ chứng minh cả 3 mục tiêu) |

---

## 9. Checklist 3 dòng dán lên màn hình

```
[ ] .venv/bin/python demo_preflight.py            → 12 PASS
[ ] .venv/bin/python gold_predict.py              → bảng khớp báo cáo
[ ] cd gold_price_fullstack && ../.venv/bin/python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Các file để mở sẵn trước khi vào phòng:

1. `Báo_cáo_Giá_vàng_nhóm_8.docx` — trang bảng kết quả Test
2. `gold_model_output/delta_vs_level_comparison.png` — biểu đồ khắc phục
3. `gold_model_output/k_validation_curve.png` — biểu đồ quét K
4. `gold_price_fullstack/outputs/test_predictions_plot.png` — biểu đồ dự báo Test
5. Trình duyệt: `http://localhost:8000` (chạy server trước)
