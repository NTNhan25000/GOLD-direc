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

## 6. Kết luận gắn với câu hỏi nghiên cứu

- **Mô hình phi tuyến có vượt Naive không?** Không trong lần thử này; RF/XGBoost kém đáng kể trên Test. Linear Regression cũng chưa vượt Naive.
- **Các biến vĩ mô có cải thiện độ chính xác không?** Kết quả hiện tại chưa tách riêng thí nghiệm chỉ dùng đặc trưng kỹ thuật và thí nghiệm có thêm vĩ mô, do đó chưa thể kết luận. Cần chạy ablation study trên cùng split và cùng quy trình.
- **Đặc trưng nào quan trọng nhất?** Chưa thể kết luận từ bảng metric. Cần xuất feature importance/permutation importance (và có thể SHAP cho XGBoost), kèm cảnh báo rằng mức quan trọng không chứng minh quan hệ nhân quả.
- **Mô hình nên chọn theo kết quả hiện tại?** Naive là benchmark chính xác nhất trên Test; Linear Regression là mô hình học được có kết quả tốt nhất trong bốn mô hình đã đánh giá. Không nên chọn K của cây chỉ vì nó tốt nhất trên Validation khi kết quả Test thất bại.

## 7. Các chỉ số và đầu ra cần có khi trình bày môn học

- Bảng MAE, RMSE, MAPE, R² của cả bốn mô hình trên Test.
- Bảng so sánh K/tham số đã thử và quy tắc chọn bằng Validation.
- Biểu đồ giá thực tế và dự báo trên Test; có thể thêm biểu đồ sai số và đường cong Train/Validation.
- Phân tích hệ số Linear Regression; feature importance/permutation importance của mô hình cây.
- Nếu bám đầy đủ phụ lục báo cáo: có thể bổ sung VIF, p-value/khoảng tin cậy, Durbin–Watson, SHAP, kiểm tra các phiên giá vượt đỉnh Train. Các phân tích này là đầu ra bổ sung và hiện chưa được tạo tự động bởi script.
- So sánh riêng mô hình với/không có biến vĩ mô để trả lời câu hỏi về giá trị của dữ liệu liên thị trường.
- Nêu giới hạn: chỉ dự báo một bước; dữ liệu lịch sử 2015–2024; thị trường không dừng và có cú sốc; metric quá khứ không bảo đảm hiệu quả tương lai.

## 8. File đầu ra hiện có

- `gold_predict.py`: pipeline tạo đặc trưng, huấn luyện, quét K, đánh giá.
- `gold_model_output/k_validation_test.csv`: kết quả của từng K từ 1 đến 500 trên Validation (1.000 dòng).
- `gold_model_output/test_metrics.csv`: kết quả Test của bốn mô hình.
- `gold_model_output/test_predictions.csv`: dự báo Naive và Random Forest theo từng quan sát Test.
- `gold_model_output/K_test_plan.csv`: ghi chú K theo phương pháp.

Để chạy lại: `python -m pip install -r requirements.txt`, sau đó `python gold_predict.py --data gold_forecast_dataset_yfinance.csv`.
