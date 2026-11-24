# Demo dự đoán giá vàng — 4 mô hình

Folder này chứa script demo chạy dự đoán giá đóng cửa phiên kế tiếp của hợp đồng
tương lai vàng COMEX (`GC=F`) bằng cả **4 mô hình** đã huấn luyện:

| Mô hình | Tham số | Nguồn |
|---|---|---|
| Naive Baseline | không có | `ŷ(t+1) = C(t)` |
| Linear Regression | w (25 hệ số) + b | `models/linear_regression.joblib` |
| Random Forest | K = 11 cây | `models/randomforest.joblib` |
| XGBoost | K = 165 vòng | `models/xgboost.joblib` |

## Chạy demo

```bash
cd gold_price_fullstack

# Mặc định: dự báo phiên kế tiếp + 5 phiên Test gần nhất đối chiếu thực tế
python demo/demo.py

# Tự nhập giá đóng cửa cho phiên t
python demo/demo.py --close 2650

# Chọn phiên phát dự báo cụ thể + 10 phiên Test
python demo/demo.py --date 2024-06-14 --last 10

# Chỉ dự báo phiên kế tiếp
python demo/demo.py --skip-test
```

> Cần chạy `python ml/train.py` trước một lần để có `models/*.joblib`.
> Script tự tải model từ thư mục `models/` — **không cần chạy server API**.

## Đầu ra

1. **Bảng dự báo phiên kế tiếp**: 4 mô hình, dự báo USD/oz, chênh lệch so với Naive.
2. **Bảng đối chiếu trên Test**: mỗi phiên có giá thực tế `y(t+1)`, dự báo của 4 mô hình
   và mô hình có sai số tuyệt đối nhỏ nhất phiên đó; kết thúc bằng MAE trung bình
   của từng mô hình trên các phiên demo.

Đơn vị: USD/troy ounce. Công cụ nghiên cứu học thuật — không phải khuyến nghị đầu tư.
