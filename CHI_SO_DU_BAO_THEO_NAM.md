# Bảng chỉ số dự đoán giá vàng theo năm

## Phạm vi

Bảng dưới đây báo cáo bốn chỉ số (**MAE, RMSE, MAPE, R²**) của từng mô hình theo năm của **ngày giá mục tiêu/dự đoán**. Các chỉ số được tính qua toàn bộ phiên giao dịch trong năm đó, đối chiếu mỗi dự đoán với giá đóng cửa thực tế cùng phiên.

Trong thiết lập hiện tại, Test là dữ liệu năm 2024, nên hiện chỉ có thể báo cáo năm **2024** như một năm Test độc lập. Không dùng Train/Validation để điền thêm các năm, vì các dự báo đó không phải kết quả đánh giá ngoài mẫu tương đương. Nếu cần bảng theo nhiều năm độc lập, cần quy trình rolling-origin/backtesting (huấn luyện lại đến từng thời điểm rồi dự báo năm tiếp theo).

## Kết quả Test năm 2024

| Năm dự đoán | Mô hình | Số phiên (N) | MAE (USD/oz) ↓ | RMSE (USD/oz) ↓ | MAPE (%) ↓ | R² ↑ |
|---:|---|---:|---:|---:|---:|---:|
| 2024 | Naive Baseline | 252 | **18.0381** | **23.7430** | **0.7445** | **0.9891** |
| 2024 | Linear Regression | 252 | 22.0543 | 27.0696 | 0.9063 | 0.9858 |
| 2024 | Random Forest (K=11) | 252 | 364.5287 | 426.3243 | 14.3982 | -2.5165 |
| 2024 | XGBoost (K=165) | 252 | 369.7296 | 432.9781 | 14.5971 | -2.6271 |

Mũi tên chỉ chiều mong muốn: MAE/RMSE/MAPE càng thấp càng tốt; R² thường càng cao càng tốt. R² âm của Random Forest/XGBoost cho thấy trên năm Test, các mô hình này kém hơn dự báo hằng bằng trung bình của giá thực tế Test theo định nghĩa R².

## Cách tính

Với từng mô hình và từng năm, gọi giá thực tế là `y_i`, dự báo là `yhat_i`, và số phiên là `N`:

- `MAE = mean(abs(y_i - yhat_i))`
- `RMSE = sqrt(mean((y_i - yhat_i)^2))`
- `MAPE = mean(abs((y_i - yhat_i) / y_i)) * 100`
- `R² = 1 - sum((y_i - yhat_i)^2) / sum((y_i - mean(y))^2)`

Mỗi mô hình được so sánh trên đúng cùng 252 phiên Test. Naive Baseline không có K; K=11 và K=165 của các mô hình cây được chọn trước bằng RMSE Validation, không được chọn dựa trên Test.

## Nhận xét

Naive Baseline cho sai số thấp nhất trong năm Test 2024; Linear Regression đứng thứ hai. Random Forest và XGBoost có sai số lớn và R² âm. Đây là kết quả thực nghiệm của giai đoạn này, không phải khẳng định mô hình cây luôn kém trên các giai đoạn khác. Một nguyên nhân có thể là giá vàng năm 2024 đã tăng ra ngoài vùng giá mà mô hình cây học được, khiến khả năng ngoại suy của cây hạn chế.

## File dữ liệu

- `gold_model_output/annual_test_metrics.csv`: bảng chỉ số dạng CSV theo năm và mô hình.
- `gold_model_output/all_model_test_predictions.csv`: dự báo và giá thực tế của cả bốn mô hình theo từng phiên Test; có `ForecastOrigin` (phiên phát dự báo) và `Date` (ngày của giá mục tiêu).
- `gold_model_output/k_validation_test.csv`: kết quả quét K trên Validation.

Để báo cáo các năm ngoài 2024 một cách công bằng, cần bổ sung đánh giá rolling-origin theo năm; không nên dùng dự đoán in-sample trên Train làm kết quả Test.
