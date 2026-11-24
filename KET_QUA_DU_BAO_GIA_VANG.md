# Kết quả và tiêu chí hoàn thành đề tài dự báo giá vàng

## 1. Mục tiêu theo báo cáo môn Trí tuệ Nhân tạo

Đây là bài toán **hồi quy chuỗi thời gian có giám sát**. Dựa trên thông tin đã có đến cuối phiên giao dịch `t`, dự báo giá đóng cửa hợp đồng tương lai vàng COMEX (`GC=F`) ở phiên giao dịch kế tiếp `t+1`, đơn vị USD/troy ounce.

Mục tiêu học thuật không phải chứng minh một thuật toán luôn thắng, mà là xây dựng quy trình thực nghiệm đúng, so sánh công bằng với mốc cơ sở, đánh giá ngoài mẫu và giải thích kết quả. Một kết quả cho thấy mô hình AI không vượt Naive vẫn hợp lệ nếu được báo cáo trung thực và phân tích đúng.

## 2. Dữ liệu và chia tập

- Dataset đầu vào: `gold_forecast_dataset_yfinance.csv` trong repository.
- Chuỗi mục tiêu: `Gold_Close`.
- Các biến liên thị trường nếu có trong dữ liệu: DXY, US10Y, S&P 500, WTI và Silver.
- Tập Train: nhãn có ngày đến hết 2021-12-31.
- Tập Validation: nhãn từ 2022-01-01 đến hết 2023-12-31.
- Tập Test: nhãn sau 2023-12-31, tương ứng năm 2024 trong dữ liệu hiện có (252 quan sát).
- Chia theo thứ tự thời gian, không shuffle. K được chọn bằng Validation; Test chỉ dùng cho đánh giá cuối.

**Đặc trưng được tạo trong `gold_predict.py`:** giá đóng cửa trễ 0, 1, 2, 3, 5, 10 phiên; SMA 5/10/20; độ biến động lợi suất 5/10/20 phiên; RSI 14; lợi suất vàng 1 phiên; biên độ High–Low; giá và lợi suất một phiên của các biến vĩ mô.

Mục tiêu được tạo bằng `Gold_Close.shift(-1)`: hàng đầu vào tại phiên `t` ghép với đóng cửa phiên giao dịch kế tiếp. Chia tập được căn theo ngày nhãn/target thực tế, không giả định rằng phiên sau luôn là ngày làm việc thông thường.

## 3. Mô hình được so sánh

1. **Naive Baseline:** dự báo đóng cửa phiên sau bằng đóng cửa hiện tại.
2. **Linear Regression:** hồi quy tuyến tính trên các đặc trưng đã chuẩn hóa; scaler chỉ fit trên Train.
3. **Random Forest Regressor:** mô hình cây tổ hợp, thử số cây `K` từ 1 đến 500.
4. **XGBoost Regressor:** mô hình boosting, thử số vòng/cây `K` từ 1 đến 500.

Trong cấu hình hiện tại, Random Forest dùng `max_features=0.8`, `min_samples_leaf=2`, `random_state=42`; XGBoost dùng `max_depth=3`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1`, `random_state=42`. Các tham số này được giữ cố định trong lúc quét K.

## 4. Chọn K trên Validation

Tiêu chí lựa chọn: RMSE Validation thấp nhất. Kết quả quét toàn bộ 500 giá trị cho mỗi mô hình:

| Mô hình | K tốt nhất | Validation MAE | Validation RMSE | Validation MAPE | Validation R² |
|---|---:|---:|---:|---:|---:|
| Random Forest | **11** | **21.399** | **28.413** | **1.123%** | **0.9294** |
| XGBoost | **165** | **24.973** | **34.238** | **1.293%** | **0.8975** |

Trong lần chạy này, K tốt nhất theo Validation là 11 cho Random Forest và 165 cho XGBoost. K chỉ là một siêu tham số; lựa chọn này không đảm bảo mô hình tốt nhất trên Test.

Một số điểm tham khảo trên đường quét:

| K | RF RMSE Validation | XGBoost RMSE Validation |
|---:|---:|---:|
| 1 | 33.879 | 461.844 |
| 10 | 29.303 | 360.718 |
| 11 | **28.413** | 351.129 |
| 50 | 32.128 | 126.190 |
| 100 | 31.986 | 49.784 |
| 165 | 32.048 | **34.238** |
| 300 | 31.828 | 35.177 |
| 500 | 32.195 | 37.642 |

XGBoost có sai số cao khi K quá nhỏ ở cấu hình learning rate 0.03; sai số giảm đến khoảng K=165 rồi tăng lại. RF đạt RMSE tốt nhất ở K=11, nhưng các giá trị lớn hơn cho kết quả Validation tương đối gần nhau. Những con số trên phụ thuộc vào dữ liệu, đặc trưng, cách chia tập và cấu hình tham số.

## 5. Đánh giá cuối trên Test năm 2024

| Mô hình | K | MAE (USD/oz) | RMSE (USD/oz) | MAPE | R² |
|---|---:|---:|---:|---:|---:|
| **Naive Baseline** | — | **18.038** | **23.743** | **0.744%** | **0.9891** |
| Linear Regression | — | 22.054 | 27.070 | 0.906% | 0.9858 |
| Random Forest | 11 | 364.529 | 426.324 | 14.398% | -2.5165 |
| XGBoost | 165 | 369.730 | 432.978 | 14.597% | -2.6271 |

Sai số MAE/RMSE/MAPE thấp hơn là tốt hơn. Trong Test hiện có, **Naive Baseline tốt nhất**, Linear Regression đứng sau; cả hai mô hình cây đều kém rõ rệt. R² âm của mô hình cây nghĩa là dự báo của chúng còn kém hơn cách dự báo hằng bằng trung bình giá thực tế của tập Test theo định nghĩa R².

Một lý do có thể là giá vàng trong năm 2024 tăng mạnh lên ngoài vùng mà cây quyết định học từ dữ liệu Train/Validation; mô hình cây thường khó ngoại suy xu hướng giá vượt miền đã quan sát. Đây là diễn giải phù hợp với kết quả, không phải bằng chứng nhân quả. Naive có lợi thế đặc biệt trong dự báo mức giá tài sản một phiên trước vì giá hai phiên liên tiếp thường gần nhau.

R² cao của Naive không đồng nghĩa mô hình tạo lợi nhuận hay dự đoán đúng hướng biến động; cần so với baseline, xem sai số và giới hạn dữ liệu. Kết quả cũng không phải khuyến nghị đầu tư.

## 6. Phân tích mở rộng (đã tự động hoá trong `gold_extra_outputs.py`)

Toàn bộ các phân tích dưới đây được tạo tự động, không cần thao tác tay:

### 6.1. Chẩn đoán hồi quy tuyến tính (`ols_diagnostics.csv`)

- Hồi quy OLS trên 25 đặc trưng chuẩn hoá: `train R² = 0.9972`; `Durbin–Watson = 2.010` (Train),
  `1.513` (Test) — không có tự tương quan mạnh trong mẫu Train.
- Đa cộng tuyến nặng: **11/25 biến có VIF > 10**, lớn nhất `sma_5` VIF ≈ 17.593 và `close_lag_0`
  VIF ≈ 11.266 — do giá trễ, SMA và giá mức đều gần như cùng một chuỗi. Hệ số riêng lẻ vì vậy
  không nên diễn giải như tác động nhân quả.
- 8 hệ số có ý nghĩa thống kê ở mức 5%, đáng chú ý: `close_lag_0` (t = 7.08), `US10Y_Close_return_1d`
  (t = −5.64), `Silver_Close` (t = −3.51), `rsi_14` (t = 2.61).

### 6.2. Độ quan trọng đặc trưng

`rf_feature_importance.csv` (MDI + Permutation Importance trên Val & Test) và
`xgb_feature_importance.csv` (Gain/Weight/Cover + TreeSHAP):

| Hạng | Random Forest (MDI %) | XGBoost (Gain %) |
|---:|---|---|
| 1 | `close_lag_0` — 72.22% | `sma_20` — 29.05% |
| 2 | `close_lag_1` — 17.58% | `close_lag_0` — 26.06% |
| 3 | `sma_10` — 8.77% | `close_lag_1` — 20.70% |
| 4 | `close_lag_2` — 0.93% | `sma_10` — 10.92% |
| 5 | `sma_5` — 0.30% | `close_lag_10` — 4.95% |

Kết luận: nhóm **giá trễ + SMA chiếm gần như toàn bộ** độ quan trọng; 20 biến còn lại (đặc biệt
là các biến vĩ mô) chỉ đóng góp biên (< 1%). Permutation importance trên Test còn cho thấy nhiều
biến vĩ mô có mức tăng RMSE **âm** — tức thêm chúng không giúp mô hình. Điều này **không** chứng
minh quan hệ nhân quả, chỉ phản ánh cấu trúc dữ liệu.

### 6.3. Ablation biến vĩ mô (`ablation_macro_study.csv`)

| Bộ đặc trưng | Số biến | Mô hình | Val RMSE | Test MAE | Test RMSE |
|---|---:|---|---:|---:|---:|
| Baseline (chỉ Close_t) | 1 | Naive | **17.281** | **18.038** | **23.743** |
| Chỉ kỹ thuật | 15 | Linear Regression | 17.486 | 18.531 | 24.000 |
| Chỉ kỹ thuật | 15 | Random Forest (K=2) | 25.994 | 342.668 | 402.198 |
| Chỉ kỹ thuật | 15 | XGBoost (K=217) | 26.751 | 354.763 | 416.914 |
| Kỹ thuật + vĩ mô | 25 | Linear Regression | 19.033 | 22.054 | 27.070 |
| Kỹ thuật + vĩ mô | 25 | Random Forest (K=11) | 28.413 | 364.529 | 426.324 |
| Kỹ thuật + vĩ mô | 25 | XGBoost (K=165) | 34.238 | 369.730 | 432.978 |

**Trả lời câu hỏi nghiên cứu:** thêm 10 biến vĩ mô **không cải thiện** sai số — với Linear
Regression, Val RMSE tăng từ 17.486 → 19.033 và Test MAE từ 18.531 → 22.054. Với mô hình cây,
bộ 15 biến kỹ thuật thuần cũng tốt hơn bộ 25 biến. Giá trị của dữ liệu liên thị trường trong
thiết lập một bước là rất hạn chế.

### 6.4. Giới hạn ngoại suy — nguyên nhân gốc của sai số cây

`extra_diagnostics.json`:

- RF: dự báo cao nhất trên Test = **2.079.82** USD, trong khi giá thực tế cao nhất = **2.800.80** USD;
  **211/252 phiên** có giá thực vượt mức dự báo tối đa của mô hình.
- XGB: dự báo cao nhất = **2.045.60** USD; **223/252 phiên** vượt trần.
- OOB score của RF (K=11) trên Train+Val: MAE = 12.11, RMSE = 17.05, R² = 0.9968 trên 2.226/2.241
  mẫu có dự báo OOB — mô hình **tốt trong miền đã học**, chỉ thất bại khi ngoại suy.

Đây là bằng chứng định lượng trực tiếp cho giải thích trong báo cáo: cây quyết định dự báo
bằng trung bình lá nên không thể vượt quá giá trị lớn nhất đã thấy khi huấn luyện.

### 6.5. Thực nghiệm đổi mục tiêu sang `ΔClose(t+1)` (`delta_target_experiment.csv`)

Dự báo **mức thay đổi** rồi cộng lại `Close_t` khắc phục hoàn toàn giới hạn ngoại suy:

| Cách tiếp cận | MAE | RMSE | MAPE | R² | Đúng chiều |
|---|---:|---:|---:|---:|---:|
| Naive (Δ = 0) | 18.0381 | 23.7430 | 0.744% | 0.9891 | — |
| Linear Regression (ΔClose) | 18.9193 | 24.6824 | 0.779% | 0.9882 | 51.59% |
| Random Forest (ΔClose, K=71) | 18.2533 | 23.8148 | 0.753% | 0.9890 | 49.60% |
| **XGBoost (ΔClose, K=21)** | **18.0039** | **23.7063** | **0.743%** | **0.9891** | **55.95%** |

XGBoost dự báo ΔClose **vượt Naive** (MAE 18.004 < 18.038) và đúng chiều 55.95% — mức cao hơn
hẳn 50% ngẫu nhiên. Đây là kết quả có giá trị học thuật: **cách đặt mục tiêu quan trọng hơn
việc chọn thuật toán**. Biểu đồ: `delta_vs_level_comparison.png`.

## 7. Kết luận gắn với câu hỏi nghiên cứu

- **Mô hình phi tuyến có vượt Naive không?** Không khi dự báo trực tiếp mức giá; RF/XGBoost kém
  đáng kể trên Test. Nhưng khi dự báo **ΔClose**, XGBoost vượt Naive (MAE 18.004 vs 18.038,
  đúng chiều 55.95%). Vậy câu trả lời phụ thuộc cách đặt mục tiêu, không phụ thuộc thuật toán.
- **Các biến vĩ mô có cải thiện độ chính xác không?** Không. Ablation cho thấy bộ 15 biến kỹ thuật
  thuần tốt hơn hoặc tương đương bộ 25 biến ở mọi mô hình; permutation importance trên Test còn
  âm với phần lớn biến vĩ mô.
- **Đặc trưng nào quan trọng nhất?** `close_lag_0`, `close_lag_1` và các SMA (5/10/20) chiếm
  > 98% độ quan trọng của RF và ~87% Gain của XGBoost. Mức quan trọng không chứng minh nhân quả.
- **Mô hình nên chọn theo kết quả hiện tại?** Naive là mốc chính xác nhất khi dự báo mức giá;
  Linear Regression là mô hình học tốt nhất trong 4 mô hình ở thiết lập đó; XGBoost dự báo ΔClose
  là mô hình học tốt nhất về MAE tổng thể trên Test.

## 8. Các chỉ số và đầu ra khi trình bày môn học

Đã có sẵn, tạo tự động:

- Bảng MAE, RMSE, MAPE, R² của cả bốn mô hình trên Test (`test_metrics.csv`).
- Bảng quét K đầy đủ 1..500 và quy tắc chọn bằng Validation (`k_validation_test.csv`, `K_test_plan.csv`).
- Biểu đồ giá thực tế vs. dự báo trên Test (`test_forecast_comparison.png`), đường cong K
  (`k_validation_curve.png`), learning curve XGBoost (`xgb_learning_curve.png`).
- Hệ số Linear Regression kèm SE/t/p/CI/VIF/Durbin–Watson (`ols_diagnostics.csv`).
- Feature importance (MDI + permutation), TreeSHAP (`rf_feature_importance.csv`,
  `xgb_feature_importance.csv`, `xgb_shap_importance.png`).
- Ablation biến vĩ mô (`ablation_macro_study.csv`) và thực nghiệm ΔClose (`delta_target_experiment.csv`).
- Chẩn đoán ngoại suy định lượng (`extra_diagnostics.json`).
- Sai số theo quý (`quarterly_test_metrics.csv`) và theo năm (`annual_test_metrics.csv`).
- Nêu giới hạn: chỉ dự báo một bước; dữ liệu lịch sử 2015–2024; thị trường không dừng và có cú sốc;
  metric quá khứ không bảo đảm hiệu quả tương lai; kết quả không phải khuyến nghị đầu tư.

## 9. File đầu ra hiện có

| File | Nội dung |
|---|---|
| `gold_predict.py` | Pipeline chính: tạo đặc trưng, huấn luyện, quét K 1..500, đánh giá |
| `gold_extra_outputs.py` | Chẩn đoán OLS, importance/SHAP, ablation, ΔClose, biểu đồ mở rộng |
| `plot_splits.py` | Biểu đồ khảo sát dữ liệu và 4 tập chia |
| `k_validation_test.csv` | RMSE/MAE/MAPE/R² của từng K (RF và XGB) trên Validation — 1.000 dòng |
| `test_metrics.csv` | Kết quả Test của bốn mô hình |
| `test_predictions.csv` | Dự báo **cả bốn mô hình** theo từng quan sát Test |
| `all_model_test_predictions.csv` | Bảng dài: ngày phát, ngày mục tiêu, thực tế, 4 dự báo |
| `annual_test_metrics.csv` | Chỉ số theo năm (dạng dài) |
| `model_detailed_summary.csv` | Val/Test, sai số có dấu, max error, tỷ lệ thắng Naive, đúng chiều |
| `ols_diagnostics.csv` | β, SE, t, p, CI 95%, VIF |
| `rf_feature_importance.csv` | MDI + permutation importance |
| `xgb_feature_importance.csv` | Gain/Weight/Cover + TreeSHAP |
| `xgb_learning_curve.csv` | RMSE/MAE Train vs Val theo K = 1..500 |
| `quarterly_test_metrics.csv` | MAE/RMSE theo quý Q1–Q4 |
| `ablation_macro_study.csv` | So sánh 15 biến kỹ thuật vs. 25 biến |
| `delta_target_experiment.csv` | Thực nghiệm mục tiêu ΔClose |
| `extra_diagnostics.json` | OLS meta, OOB, trần ngoại suy RF/XGB |
| `K_test_plan.csv` | Ghi chú K theo phương pháp |

Để chạy lại toàn bộ: `python -m pip install -r requirements.txt`, sau đó
`python plot_splits.py && python gold_predict.py && python gold_extra_outputs.py`.
