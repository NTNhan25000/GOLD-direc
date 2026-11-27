# model_joblib — folder độc lập chứa file `.joblib`

Folder này **tự chạy được một mình**: chỉ cần Python + 4 file `.joblib` + metadata + một
file CSV dữ liệu. Không import gì từ `gold_price_fullstack/`, không cần FastAPI, không cần
chạy lại pipeline.

## Nội dung

```
model_joblib/
├── artifacts/                        ← copy từ gold_price_fullstack/models/
│   ├── imputer.joblib                SimpleImputer (chiến lược median, 25 đặc trưng)
│   ├── linear_regression.joblib      Pipeline: StandardScaler → LinearRegression
│   ├── randomforest.joblib           RandomForestRegressor — K = 11 cây
│   ├── xgboost.joblib                XGBRegressor — K = 165 vòng, depth 3, lr 0,03
│   └── metadata.json                 25 đặc trưng, mốc chia tập, metric Test, K tốt nhất
├── data/
│   └── gold_forecast_dataset.csv     2.514 phiên · 2015-01-02 → 2024-12-31
├── joblib_loader.py                  code nạp + dự báo (chạy trực tiếp được)
└── README.md                         file này
```

| File | Loại đối tượng bên trong | Ghi chú |
|---|---|---|
| `imputer.joblib` | `SimpleImputer` | **Bắt buộc** dùng trước mọi model |
| `linear_regression.joblib` | `Pipeline` | **Đã có scaler bên trong** — không chuẩn hoá thêm |
| `randomforest.joblib` | `RandomForestRegressor` | `n_estimators=11`, `max_features=0.8` |
| `xgboost.joblib` | `XGBRegressor` | `n_estimators=165`, `max_depth=3`, `learning_rate=0.03` |

## Chạy

```bash
# Từ thư mục gốc repo
.venv/bin/python model_joblib/joblib_loader.py
```

Kết quả in ra 3 phần:

1. **Mô tả model** — ngày huấn luyện, phiên bản sklearn/xgboost, số đặc trưng, K tốt nhất.
2. **Dự báo phiên kế tiếp** — 4 mô hình (Naive, LR, RF, XGB) với dữ liệu đến 2024-12-31.
3. **Tái lập bảng Test 2024** — tự tính lại MAE/RMSE/MAPE/R² và so với `metadata.json`.

Lần chạy đã kiểm chứng:

```
   Mô hình                      MAE      RMSE    MAPE %        R²
   Naive Baseline            18.038    23.743     0.744    0.9891
   Linear Regression         22.054    27.070     0.906    0.9858
   Random Forest (K=11)     364.529   426.324    14.398   -2.5165
   XGBoost (K=165)          369.730   432.978    14.597   -2.6271

   Lệch MAE lớn nhất so với metadata.json: 0.00e+00 → KHỚP
```

## Dùng như thư viện

```python
import sys; sys.path.insert(0, "model_joblib")
from joblib_loader import load_models, load_raw, predict, predict_next_session

models = load_models()                    # nạp 1 lần, dùng lại nhiều lần
raw    = load_raw()

# (a) Dự báo phiên kế tiếp sau phiên cuối trong dữ liệu
print(predict_next_session(models, raw))

# (b) Thay giá hiện tại bằng số nhập tay (High/Low tự nới để bao lấy giá mới)
print(predict_next_session(models, raw, close_override=2650))

# (c) Dự báo cho một ngày bất kỳ có trong dữ liệu
print(predict_next_session(models, raw, origin="2024-06-28"))

# (d) Dự báo hàng loạt trên một ma trận đặc trưng bất kỳ
from joblib_loader import make_features
X, y, target_dates = make_features(raw)
print(predict(models, X.tail(5)))
```

## Thứ tự xử lý — chỗ dễ sai nhất

```
dữ liệu thô
   → make_features()      sinh 25 đặc trưng, mỗi dòng chỉ dùng dữ liệu ≤ t
   → imputer.transform()  điền NaN bằng median học từ Train
   → model.predict()      LR tự chuẩn hoá bên trong Pipeline; RF/XGB nhận thẳng
```

Ba lỗi thường gặp:

- **Chuẩn hoá 2 lần** cho Linear Regression (gọi thêm `StandardScaler` ngoài) → số lệch
  hoàn toàn. Pipeline đã có scaler.
- **Đảo thứ tự cột** khi đưa vào model → `predict()` trong loader đã tự sắp lại theo
  `metadata["features"]`, nhưng nếu bạn gọi thẳng `model.predict()` thì phải tự giữ đúng thứ tự.
- **Bỏ qua imputer** → model nhận NaN và trả về NaN hoặc lỗi.

## Lưu ý về phiên bản

File `.joblib` được pickle theo phiên bản thư viện lúc huấn luyện:

```
sklearn 1.9.1 · xgboost 3.2.0 · python 3.11
```

Mở bằng phiên bản cũ hơn có thể chỉ cảnh báo (InconsistentVersionWarning), nhưng nếu
`score_test_split` không còn khớp `metadata.json` thì phải cài đúng phiên bản trong
`requirements.txt` của repo gốc.

## Cập nhật lại artifacts

Khi pipeline gốc huấn luyện lại (`gold_price_fullstack/ml/train.py`), chép lại:

```bash
cp gold_price_fullstack/models/*.joblib gold_price_fullstack/models/metadata.json model_joblib/artifacts/
```

Rồi chạy lại `joblib_loader.py` để xác nhận dòng "KHỚP" ở cuối.
