# Ý tưởng slide thuyết trình — Nhóm 8

> File này chỉ chứa **ý tưởng dàn slide**, không phải file trình chiếu.
> Mọi con số dưới đây đã được kiểm chứng bằng script trong repo — dùng trực tiếp được.
> Kịch bản nói + demo chi tiết: [`HUONG_DAN_DEMO.md`](HUONG_DAN_DEMO.md).

---

## 0. Câu chuyện xuyên suốt (quan trọng nhất)

Đừng kể "em làm 4 mô hình rồi so sánh". Hãy kể theo **3 hồi**:

> **① Kết quả bất thường** — mô hình cây thua Naive 20 lần
> → **② Chẩn đoán định lượng** — giới hạn ngoại suy của cây
> → **③ Khắc phục** — đổi mục tiêu sang ΔClose thì XGBoost vượt Naive

**Quy tắc:** slide nào không phục vụ 1 trong 3 hồi này thì cắt.

**Thông điệp 1 câu để mở đầu và kết thúc:** *"Kết quả âm tính vẫn là kết quả khoa học nếu
nhóm giải thích được nguyên nhân và tìm được cách khắc phục."*

---

## 1. Dàn ý 15–19 slide cho 12–15 phút

| # | Slide | Nội dung chính | Hình / file dùng | Câu chốt khi nói |
|---|---|---|---|---|
| 1 | **Bìa** | Tên đề tài, Nhóm 8, môn Trí tuệ Nhân tạo, GVHD | — | — |
| 2 | **Vấn đề** | Vàng là tài sản trú ẩn; dự báo 1 phiên trước là bài toán **hồi quy chuỗi thời gian có giám sát** | `gold_comprehensive_analysis.png` (panel 1 — xu hướng) | "Giá vàng 2015–2024 tăng 2,7 lần, nhưng câu hỏi khó hơn: dự báo **đúng 1 phiên tới**" |
| 3 | **Phát biểu bài toán** | Công thức `ŷ(t+1) = f(X(t))`; timeline rõ **thời điểm phát tín hiệu** (cuối phiên t) | Vẽ timeline 1 dòng: `t` → `t+1` | "Cuối phiên t chỉ biết dữ liệu ≤ t, dự báo close phiên t+1" |
| 4 | **3 câu hỏi nghiên cứu** | ① Phi tuyến có vượt Naive? ② Biến vĩ mô có giúp? ③ Đặc trưng nào quan trọng? | Chữ to, 3 dòng đánh số | "Ba câu này sẽ được trả lời ở slide 11, 13, 14 — đúng thứ tự" |
| 5 | **Dữ liệu** | **2.514 phiên** · 2015-01-02 → 2024-12-31 · 10 cột · **0 NaN** · 6 nguồn Yahoo Finance | `gold_comprehensive_analysis.png` (panel 2 — heatmap) | "S&P 500 tương quan 0,94 và Bạc 0,92 — nhưng tương quan **không** phải nhân quả" |
| 6 | **Chia tập & chống rò rỉ** ⭐ | Train **1.739** / Val **502** / Test **252**; imputer + scaler **chỉ fit trên Train** | `gold_train_plot.png`, `gold_val_plot.png`, `gold_test_plot.png` (3 hình nhỏ cạnh nhau) | "Không shuffle; chia theo **ngày của giá mục tiêu** — đây là chỗ giám thị hay hỏi nhất" |
| 7 | **25 đặc trưng** | 4 nhóm: giá trễ (6) · SMA & biến động (6) · RSI-14 & biên độ (3) · vĩ mô (10) | Bảng 4 dòng, **không** liệt kê 25 tên | "Chỉ 25 biến — không tham lam, vì dữ liệu chỉ 2.514 phiên" |
| 8 | **4 mô hình & lý do chọn** | Naive (baseline) · Linear (diễn giải được) · RF (bagging) · XGB (boosting) | Bảng so sánh bản chất 4 dòng | "4 mô hình là mức tối thiểu để so sánh công bằng" |
| 9 | **Thiết lập thực nghiệm** | Hyperparams: RF `max_features=0.8, min_samples_leaf=2`; XGB `depth=3, lr=0.03, subsample=0.8, colsample=0.8, λ=1` | Bảng 2 cột | "Siêu tham số giữ **cố định** trong lúc quét K" |
| 10 | **Chọn K trên Validation** ⭐ | Quét K = 1..500; RF **K=11** (Val RMSE **28,413**), XGB **K=165** (**34,238**) | `gold_model_output/k_validation_curve.png` | "Chọn K đúng quy trình — và **mọi K đều thất bại trên Test**, nên không phải lỗi chọn K" |
| 11 | **KẾT QUẢ TEST** ⭐⭐ | Bảng 4 mô hình; **in đậm** dòng Naive thắng | `gold_model_output/test_forecast_comparison.png` | "Naive MAE **18,04** vs RF **364,53** — kém **20 lần**. Đây là kết quả thật, không phải lỗi code" |
| 12 | **Vì sao cây thua?** ⭐⭐ | `max_pred_RF = 2.079,82` vs `max_actual = 2.800,80`; **211/252 phiên** vượt trần; OOB MAE **12,11** | `test_forecast_comparison.png` + 2 số phóng to | "Cây dự báo bằng trung bình lá → **không thể** vượt 2.080 USD. Giới hạn **cấu trúc**, không phải bug" |
| 13 | **Ablation biến vĩ mô** (CH2) | 15 biến kỹ thuật: Val RMSE **17,486** · 25 biến: **19,033** → thêm vĩ mô **làm tệ đi** | Bảng 2 dòng | "Câu trả lời CH2 là **KHÔNG** — có bằng chứng định lượng, không phải cảm tính" |
| 14 | **Feature Importance** (CH3) | RF: `close_lag_0` **72,2%** · `close_lag_1` **17,6%** · `sma_10` **8,8%**; vĩ mô **< 1%** | `gold_model_output/xgb_shap_importance.png` | "Nhóm giá trễ + SMA chiếm **> 98%**; biến vĩ mô chỉ đóng góp biên" |
| 15 | **ĐỘT PHÁ: mục tiêu ΔClose** ⭐⭐ | XGBoost MAE **18,0039 < 18,0381** của Naive; **đúng chiều 55,95%** | `gold_model_output/delta_vs_level_comparison.png` | "Đổi **cách đặt mục tiêu** → vượt Naive. Cách đặt mục tiêu quan trọng hơn chọn thuật toán" |
| 16 | **Kết luận** | Trả lời lại đúng 3 câu hỏi ở slide 4, ghi rõ ĐÚNG/SAI | Bảng 3 dòng | "Cả 3 câu hỏi đều có câu trả lời dứt khoát" |
| 17 | **Hạn chế & hướng phát triển** | 1 bước · dữ liệu 2015–2024 · chưa rolling-origin · chưa intraday | 4 gạch đầu dòng | "Ưu tiên số 1: rolling-origin backtesting" |
| 18 | **Sản phẩm** | Dashboard FastAPI + 4 mô hình + **16 test tự động** | Screenshot dashboard (chạy `localhost:8000`) | "Chạy được thật, tái lập được, có kiểm thử tự động" |
| 19 | **Q&A** | Lời cảm ơn + 3 slide backup phía sau | — | — |

---

## 2. Nếu chỉ có 10 phút → cắt còn 10 slide

**Giữ bắt buộc:** 1, 3, 4, 6, 10, 11, 12, 13+14 (gộp), 15, 16

**Cắt hẳn** (chuyển vào nói miệng hoặc backup): 2, 5, 7, 8, 9, 17, 18

**Cách gộp 13 + 14 thành 1 slide:** đặt bảng ablation bên trái, biểu đồ SHAP bên phải,
tiêu đề "Thêm vĩ mô không giúp — và cây chỉ dùng 3 đặc trưng giá".

---

## 3. Ba slide KHÔNG được bỏ

| Slide | Vì sao |
|---|---|
| **11 — Bảng kết quả Test** | Không có nó thì không có đề tài |
| **12 — Chẩn đoán ngoại suy** | Ăn điểm học thuật cao nhất: biến "thất bại" thành "phân tích" |
| **15 — Thực nghiệm ΔClose** | Chứng minh nhóm hiểu vấn đề đến mức tìm được cách khắc phục |

---

## 4. Nguyên tắc thiết kế

1. **1 ý / slide.** Tiêu đề là **một câu khẳng định**, không phải nhãn:
   - ✅ "Cây không thể ngoại suy"
   - ❌ "Kết quả Random Forest"
2. **Mỗi slide 1 con số được phóng to ≥ 24pt.** Ví dụ slide 12: đặt `2.080` và `2.801`
   cạnh nhau, to hơn hẳn phần chữ.
3. **Chữ tối thiểu 20pt**, tối đa **6 dòng/slide**. Bảng quá 5 dòng → tách 2 slide.
4. **Màu nhất quán với biểu đồ** (để giám thị không phải đổi "từ điển" giữa các slide):

   | Đối tượng | Màu |
   |---|---|
   | Thực tế | đen |
   | Naive | xanh dương `#4dabf7` |
   | Linear Regression | cam `#ff7f0e` |
   | Random Forest | xanh lá `#2ca02c` |
   | XGBoost | đỏ `#d62728` |
   | Điểm nhấn / cảnh báo | vàng `#f0b429` |

5. **Không dán code** vào slide. Duy nhất 1 ngoại lệ: nếu bị hỏi "sao quét 500 K mà nhanh
   vậy", bấm sang backup có 3 dòng:
   ```python
   rf = RandomForestRegressor(n_estimators=500, random_state=42).fit(Xt, yt)
   cum = np.cumsum([e.predict(Xv) for e in rf.estimators_], axis=0) / np.arange(1, 501)[:, None]
   ```
6. **Mỗi bảng chỉ 1 dòng được tô đậm.** Tô nhiều = không tô dòng nào.

---

## 5. Ba thứ nên CẮT khỏi slide (dù báo cáo có)

| Nội dung trong báo cáo | Xử lý |
|---|---|
| Mục 1.1–1.3: bối cảnh thị trường vàng, lịch sử AI trong tài chính | Nói miệng **20 giây** ở slide 2, đừng làm slide riêng |
| Mục 2.1–2.4: lý thuyết chi tiết từng mô hình | Giám thị đã biết RF/XGB là gì → chỉ cần **1 bảng so sánh bản chất** |
| Phụ lục 2: bảng 25 hệ số OLS | Đưa vào backup. Slide chỉ cần "**11/25 biến có VIF > 10**" |

---

## 6. Phân bổ thời gian (tổng 13 phút)

| Phần | Slide | Thời gian |
|---|---|---:|
| Mở đầu & bài toán | 1–4 | 2:00 |
| Dữ liệu & thiết lập | 5–9 | 2:30 |
| **Kết quả** | 10–12 | **4:00** ← dành nhiều nhất |
| Trả lời câu hỏi nghiên cứu | 13–14 | 2:00 |
| **Khắc phục** | 15 | **1:30** |
| Kết luận & sản phẩm | 16–18 | 1:00 |
| Đệm | — | ~0:30 |

> Mẹo: **đừng** dành quá 3 phút cho phần dữ liệu. Điểm nằm ở slide 11–15.

---

## 7. Bảng số liệu chính xác để dựng slide (không cần tra lại)

### Kết quả Test 2024
| Mô hình | K | MAE | RMSE | MAPE | R² |
|---|---:|---:|---:|---:|---:|
| **Naive** | — | **18,0381** | **23,7430** | **0,7445%** | **0,9891** |
| Linear Regression | — | 22,0543 | 27,0696 | 0,9063% | 0,9858 |
| Random Forest | 11 | 364,5287 | 426,3243 | 14,3982% | −2,5165 |
| XGBoost | 165 | 369,7296 | 432,9781 | 14,5971% | −2,6271 |

### Chọn K trên Validation
| Mô hình | K tốt nhất | Val MAE | Val RMSE | Val MAPE | Val R² |
|---|---:|---:|---:|---:|---:|
| Random Forest | **11** | 21,399 | **28,413** | 1,123% | 0,9294 |
| XGBoost | **165** | 24,973 | **34,238** | 1,293% | 0,8975 |

Đường quét K tham khảo: RF RMSE 33,879 (K=1) → 29,303 (K=10) → **28,413 (K=11)** → 32,128 (K=50)
→ 31,986 (K=100) → 32,048 (K=165) → 32,195 (K=500).
XGB: 461,844 (K=1) → 360,718 (K=10) → 126,190 (K=50) → 49,784 (K=100) → **34,238 (K=165)** → 37,642 (K=500).

### Chẩn đoán ngoại suy (`extra_diagnostics.json`)
| Chỉ số | RF | XGB |
|---|---:|---:|
| Dự báo cao nhất trên Test | **2.079,82** USD | **2.045,60** USD |
| Giá thực cao nhất trên Test | 2.800,80 USD | 2.800,80 USD |
| Số phiên giá thực vượt trần dự báo | **211 / 252** | **223 / 252** |
| OOB MAE (Train+Val) | 12,11 (R² 0,9968) | — |
| Đỉnh giá Train | 2.069,40 USD (2020-08-06) | |
| Tăng trưởng cả năm Test | **+27,4%** | |

### Ablation biến vĩ mô
| Bộ đặc trưng | Mô hình | Val RMSE | Test MAE |
|---|---|---:|---:|
| Baseline (chỉ Close_t) | Naive | **17,281** | **18,038** |
| **15** biến kỹ thuật | Linear Regression | **17,486** | **18,531** |
| 15 biến kỹ thuật | Random Forest (K=2) | 25,994 | 342,668 |
| 15 biến kỹ thuật | XGBoost (K=217) | 26,751 | 354,763 |
| **25** biến (kỹ thuật + vĩ mô) | Linear Regression | 19,033 | 22,054 |
| 25 biến | Random Forest (K=11) | 28,413 | 364,529 |
| 25 biến | XGBoost (K=165) | 34,238 | 369,730 |

### Thực nghiệm mục tiêu ΔClose(t+1)
| Cách tiếp cận | MAE | RMSE | MAPE | R² | Đúng chiều |
|---|---:|---:|---:|---:|---:|
| Naive (Δ = 0) | 18,0381 | 23,7430 | 0,744% | 0,9891 | — |
| Linear Regression (ΔClose) | 18,9193 | 24,6824 | 0,779% | 0,9882 | 51,59% |
| Random Forest (ΔClose, K=71) | 18,2533 | 23,8148 | 0,753% | 0,9890 | 49,60% |
| **XGBoost (ΔClose, K=21)** | **18,0039** | **23,7063** | **0,743%** | **0,9891** | **55,95%** |

### Sai số theo quý (Test 2024) — dùng cho slide 12 nếu cần
| Quý | Số phiên | Giá TB | MAE Naive | MAE Linear | MAE RF | MAE XGB |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 61 | 2.078 | 11,95 | 13,70 | 47,39 | 47,23 |
| Q2 | 63 | 2.350 | 20,23 | 24,87 | 308,44 | 312,67 |
| Q3 | 64 | 2.499 | 18,09 | 23,71 | 455,24 | 462,70 |
| Q4 | 64 | 2.676 | 21,63 | 25,60 | 631,30 | 640,32 |

### Độ quan trọng đặc trưng

`rf_feature_importance.csv` xếp theo **MDI %**; `xgb_feature_importance.csv` xếp theo **Gain %**
(cả hai đều đã sort giảm dần — đọc trực tiếp từ trên xuống là đúng thứ hạng).

| Hạng | Random Forest (MDI %) | XGBoost (Gain %) |
|---:|---|---|
| 1 | `close_lag_0` — 72,22% | `sma_20` — 29,05% |
| 2 | `close_lag_1` — 17,58% | `close_lag_0` — 26,06% |
| 3 | `sma_10` — 8,77% | `close_lag_1` — 20,70% |
| 4 | `close_lag_2` — 0,93% | `sma_10` — 10,92% |
| 5 | `sma_5` — 0,30% | `close_lag_10` — 4,95% |

> **Lưu ý khi bị hỏi:** MDI và Gain xếp hạng **khác nhau** (`sma_20` đứng 1 theo Gain nhưng
> đứng 8 theo MDI) vì MDI đo mức giảm phương sai mỗi lần chia còn Gain đo mức cải thiện hàm
> mất mát. **Cả hai đều cho cùng kết luận**: nhóm giá trễ + SMA chiếm > 98% (RF) và ~87% (XGB),
> biến vĩ mô < 1%. Nếu muốn dùng một thang đo duy nhất, dùng **TreeSHAP** — đó là lý do
> `xgb_shap_importance.png` lấy SHAP làm trục, không lấy Gain.

### Chẩn đoán OLS (`ols_diagnostics.csv`)
- Train R² = **0,9972** · Durbin–Watson Train = **2,010**, Test = **1,513**
- **11/25** biến có VIF > 10; cao nhất `sma_5` **17.593**, `close_lag_0` **11.266**
- **8** hệ số có p < 0,05; đáng chú ý `close_lag_0` (t = 7,08), `US10Y_Close_return_1d` (t = −5,64),
  `Silver_Close` (t = −3,51), `rsi_14` (t = 2,61)

---

## 8. Ba slide backup cho phần Q&A

Đặt **sau** slide 19, chỉ bấm ra khi bị hỏi:

| Backup | Nội dung | Dùng khi giám thị hỏi |
|---|---|---|
| **A — Đa cộng tuyến** | Ảnh bảng `ols_diagnostics.csv` (VIF, p-value, Durbin–Watson) | "Sao không bỏ biến trùng lặp?" · "Hệ số có ý nghĩa không?" |
| **B — Quét K chi tiết** | Bảng 8 dòng K = 1, 10, 11, 50, 100, 165, 300, 500 | "Sao K nhỏ vậy?" · "Có thử K lớn chưa?" |
| **C — Kiểm thử** | Ảnh chụp `pytest -q` → **16 passed** | "Code có test chưa?" · "Sao biết kết quả đúng?" |

---

## 9. Một câu cho mỗi câu hỏi nghiên cứu (dùng ở slide 16)

| Câu hỏi | Trả lời | Bằng chứng |
|---|---|---|
| **CH1.** Mô hình phi tuyến có vượt Naive? | **Không** khi dự báo mức giá. **Có** khi dự báo ΔClose (XGBoost 18,0039 < 18,0381) | Slide 11 + 15 |
| **CH2.** Biến vĩ mô có cải thiện? | **Không.** Bỏ 10 biến vĩ mô thì Linear tốt hơn (17,486 vs 19,033) | Slide 13 |
| **CH3.** Đặc trưng nào quan trọng nhất? | **Giá trễ + SMA** (> 98% RF, ~87% XGB). Vĩ mô < 1% | Slide 14 |

---

## 10. Danh sách file ảnh để chèn slide

| Slide | File (đường dẫn từ gốc repo) |
|---|---|
| 2, 5 | `gold_comprehensive_analysis.png` (cắt 2 panel) |
| 6 | `gold_train_plot.png` · `gold_val_plot.png` · `gold_test_plot.png` |
| 10 | `gold_model_output/k_validation_curve.png` |
| 11, 12 | `gold_model_output/test_forecast_comparison.png` |
| 14 | `gold_model_output/xgb_shap_importance.png` |
| 15 | `gold_model_output/delta_vs_level_comparison.png` |
| 18 | Screenshot `http://localhost:8000` (chạy server trước) |
| Backup A/B/C | `gold_model_output/ols_diagnostics.csv` · `k_validation_test.csv` · ảnh chụp pytest |
| Dự phòng | `gold_model_output/{xgb_learning_curve,xgb_shap_importance}.png`, `gold_price_fullstack/outputs/*.png` |
