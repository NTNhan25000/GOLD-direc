# GOLD-direc — dự báo giá vàng phiên kế tiếp

Triển khai theo báo cáo: mục tiêu là giá đóng cửa phiên kế tiếp của hợp đồng tương lai vàng COMEX `GC=F` (USD/troy ounce), dự báo một bước trước; so sánh Naive, Linear Regression, Random Forest và XGBoost. Chia dữ liệu theo thời gian Train (đến 2021), Validation (2022–2023), Test (2024); không xáo trộn. Đặc trưng kỹ thuật gồm giá trễ, SMA, biến động, RSI và biến liên thị trường có trong dataset.

## Cài đặt và chạy

```bash
python -m pip install -r requirements.txt
python gold_predict.py --data gold_forecast_dataset_yfinance.csv
```

Dùng file dataset hiện có trong repo, không cần gọi API. Có thể thay đường dẫn bằng `--data`. Kết quả trong `gold_model_output/`: `test_metrics.csv` (MAE, RMSE, MAPE, R² trên test), `test_predictions.csv` và `k_validation_test.csv` (thử mọi K nguyên từ 1 đến 500 cây trên validation; K tốt nhất theo RMSE được dùng để fit lại train+validation và đo test). `K_test_plan.csv` giải thích K theo từng phương pháp; Naive và hồi quy tuyến tính không có tham số số cây K.

Lưu ý: báo cáo mô tả XGBoost, vì vậy cần cài `xgboost`; nếu thiếu, script vẫn chạy Naive/Linear/RF và thông báo bỏ qua XGBoost. Test chỉ được dùng đánh giá sau khi chọn K trên validation. Đây là công cụ nghiên cứu, không phải khuyến nghị đầu tư hay cam kết lợi nhuận.
