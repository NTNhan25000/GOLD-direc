# model_training — folder độc lập chứa code huấn luyện

Bản copy của `gold_price_fullstack/ml/train.py` (+ `features.py`), **tự chạy được một mình**:
chỉ cần `data/gold_forecast_dataset.csv`. Không import gì từ `gold_price_fullstack/` — nên có
thể copy nguyên folder sang máy khác rồi chạy.

## Nội dung

```
model_training/
├── data/
│   └── gold_forecast_dataset.csv   2.514 phiên · 2015-01-02 → 2024-12-31
├── features.py                     kỹ nghệ 25 đặc trưng + chia tập + metric
├── train.py                        huấn luyện 4 mô hình, xuất .joblib + báo cáo
├── test_train.py                   6 test đối chiếu với báo cáo
└── outputs/                        sinh ra khi chạy (gitignore)
```

## Chạy

```bash
# Nhanh (~5 giây): dùng K đã chọn sẵn từ báo cáo (RF=11, XGB=165)
.venv/bin/python model_training/train.py --fast

# Đầy đủ: tự quét K = 1..500 trên Validation rồi mới chọn (~6 giây)
.venv/bin/python model_training/train.py

# Ghi ra thư mục khác
.venv/bin/python model_training/train.py --out /tmp/run1

# Test
.venv/bin/python -m pytest model_training/test_train.py -q
```

## Đầu ra

Mặc định ghi vào chính folder này — `artifacts/` và `outputs/` (cả hai đều gitignore,
sinh lại được bất cứ lúc nào):

| File | Nội dung |
|---|---|
| `artifacts/imputer.joblib` | `SimpleImputer(median)` — **chỉ fit trên Train** |
| `artifacts/linear_regression.joblib` | `Pipeline(StandardScaler → LinearRegression)` |
| `artifacts/randomforest.joblib` | `RandomForestRegressor(n_estimators=11)` fit trên Train+Val |
| `artifacts/xgboost.joblib` | `XGBRegressor(n_estimators=165)` fit trên Train+Val |
| `artifacts/metadata.json` | 25 đặc trưng, mốc chia tập, metric Test, K tốt nhất, thời gian train |
| `outputs/test_metrics.csv` | MAE / RMSE / MAPE / R² của 4 mô hình trên Test 2024 |
| `outputs/splits.csv` | số mẫu + khoảng ngày của từng tập |
| `outputs/predictions_test.csv` | dự báo từng phiên Test của cả 4 mô hình |
| `outputs/k_validation_scan.csv` | 1000 dòng quét K (chỉ khi chạy không `--fast`) |
| `outputs/feature_importance_rf.csv` | MDI % của 25 đặc trưng |
| `outputs/feature_importance_xgb.csv` | Gain % + mean\|SHAP\| của 25 đặc trưng |

Nhánh `--fast` **không** sinh `k_validation_scan.csv` (vì không quét).

## Quy trình bên trong

```
CSV thô
  → make_features()       25 đặc trưng; dòng t chỉ dùng dữ liệu ≤ t
  → split_masks()         chia theo NGÀY GIÁ MỤC TIÊU: Train ≤2021, Val 2022–23, Test 2024
  → imputer.fit(Train)    median học từ Train, rồi transform Val/Test
  → Naive                 ŷ = close_lag_0
  → LinearRegression      fit trên Train (scaler chỉ thấy Train)
  → quét K 1..500 trên Val, chọn RMSE thấp nhất
  → fit RF/XGB trên TRAIN+VAL với K đã chọn
  → đánh giá trên Test (chỉ một lần)
```

**Hai điểm dễ làm sai — đừng "sửa":**

1. **RF/XGB fit lại trên Train+Val** sau khi chọn K, còn LR/Naive chỉ dùng Train. Đây là chủ
   đích: số cây K được chọn trên Val nên sau đó phải tận dụng Val để học. Đổi thành Train-only
   sẽ ra số khác báo cáo.
2. **Imputer không bao giờ được fit lại trên Val/Test.** Nó học median từ Train rồi giữ nguyên.

## Tăng tốc quét K

Quét đủ 500 giá trị K chỉ mất vài giây thay vì vài phút nhờ hai thủ thuật:

- **Random Forest:** cây thứ `1..k` của `RandomForest(n_estimators=500, random_state=42)`
  **chính là** rừng `k` cây fit riêng biệt. Vì vậy chỉ fit một lần 500 cây rồi lấy trung bình
  tích lũy trên `estimators_`. Sai khác < 1e-13 (có test riêng trong repo).
- **XGBoost:** `booster.predict(..., iteration_range=(0, k))` cho đúng kết quả của mô hình
  fit `k` vòng.

## Kiểm chứng tái lập

Chạy `train.py` (không `--fast`) rồi so với báo cáo:

```
  RandomForest: K = 11  (val RMSE 28.413)
  XGBoost     : K = 165 (val RMSE 34.238)

           model     K        MAE       RMSE  MAPE_pct        R2
           Naive   NaN  18.038101  23.743037  0.744467  0.989093
LinearRegression   NaN  22.054327  27.069616  0.906270  0.985823
    RandomForest  11.0 364.528736 426.324280 14.398213 -2.516504
         XGBoost 165.0 369.729587 432.978096 14.597089 -2.627128
```

Đã đối chiếu trên máy này: **lệch 0.00e+00** so với `gold_model_output/test_metrics.csv`,
và **lệch 0.00e+00** trên cả 1000 dòng `k_validation_scan.csv` so với
`gold_model_output/k_validation_test.csv`.

## Nối với folder `model_joblib/`

`artifacts/` sinh ra ở đây **dùng lẫn được** với `model_joblib/joblib_loader.py`:

```python
import sys; sys.path.insert(0, "model_joblib")
from joblib_loader import load_models, load_raw, score_test_split

models = load_models("model_training/artifacts")   # nạp model vừa train lại
print(score_test_split(models, load_raw()))        # ra đúng bảng Test trên
```

Hoặc chép đè để dùng lâu dài:

```bash
cp model_training/artifacts/*.joblib model_training/artifacts/metadata.json model_joblib/artifacts/
```

## Lưu ý

- **Không chạy `train.py` với `--out` trỏ vào `gold_price_fullstack/`** nếu chưa muốn ghi đè
  model đang dùng cho dashboard — mặc định của folder này là ghi vào chính nó.
- File `.joblib` gắn với phiên bản thư viện: **sklearn 1.9.1 · xgboost 3.2.0 · Python 3.11**.
  Mở bằng bản cũ hơn có thể chỉ cảnh báo, nhưng nếu metric lệch thì cài đúng phiên bản trong
  `requirements.txt`.
- `features.py` ở đây là **bản copy**; nếu sửa công thức ở `gold_price_fullstack/ml/features.py`
  thì phải sửa cả hai (test `test_metrics_match_report` sẽ báo nếu lệch).
