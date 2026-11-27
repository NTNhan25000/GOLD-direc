# model_training — folder độc lập: code của **từng mô hình**

Bản copy tách rời của `gold_price_fullstack/ml/`, **tự chạy được một mình**: chỉ cần
`data/gold_forecast_dataset.csv`. Không import gì từ `gold_price_fullstack/`.

Mỗi mô hình nằm trong **một file riêng**, tự mang siêu tham số của nó — không có file
"pipeline tổng" bắt buộc phải chạy.

## Nội dung

```
model_training/
├── data/gold_forecast_dataset.csv   2.514 phiên · 2015-01-02 → 2024-12-31
├── features.py                      25 đặc trưng + chia tập + metric + load_raw
│
├── baseline_naive.py       ← MÔ HÌNH 1: Naive Baseline   ŷ(t+1) = Close(t)
├── linear_regression.py    ← MÔ HÌNH 2: Linear Regression (Pipeline: scaler → OLS)
├── random_forest.py        ← MÔ HÌNH 3: Random Forest     (bagging, K = 11 cây)
├── xgboost_model.py        ← MÔ HÌNH 4: XGBoost           (boosting, K = 165 vòng)
│
├── train.py                file gộp: chạy cả 4 như bản gốc, xuất artifacts/ + metadata
├── run_all.py              chạy 4 file rời rồi in bảng so sánh
├── test_models.py          10 test canh 4 file rời khớp nhau & khớp báo cáo
├── test_train.py           6 test canh train.py tái lập báo cáo
└── README.md
```

## Chạy từng mô hình riêng

```bash
.venv/bin/python model_training/baseline_naive.py       # MAE 18,038
.venv/bin/python model_training/linear_regression.py    # MAE 22,054 + 10 hệ số lớn nhất
.venv/bin/python model_training/random_forest.py        # quét K 1..500 (~5 giây) → K = 11
.venv/bin/python model_training/xgboost_model.py        # quét K 1..500 (~2 giây) → K = 165
```

Cờ chung của cả 4 file:

| Cờ | Ý nghĩa |
|---|---|
| `--data PATH` | CSV thô khác (mặc định `data/gold_forecast_dataset.csv`) |
| `--csv` | Xuất dự báo từng phiên Test ra `outputs/predictions_<model>.csv` |
| `--out DIR` | Nơi ghi output (mặc định `model_training/outputs/`) |

Riêng 2 mô hình cây:

| Cờ | Ý nghĩa |
|---|---|
| `--fast` | Bỏ quét K, dùng K preset (RF=11 / XGB=165) |
| `--k N` | Ép một K cụ thể |
| `--k-max N` | Quét tới N (mặc định 500) |
| `--save` | Lưu `artifacts/<model>.joblib` |

Chạy tất cả và so sánh:

```bash
.venv/bin/python model_training/run_all.py --fast
```

```
           model   K    fit_on        MAE       RMSE   MAPE_pct         R2
           Naive   — không học    18.0381    23.7430     0.7445     0.9891
LinearRegression   —     Train    22.0543    27.0696     0.9063     0.9858
    RandomForest  11 Train+Val   364.5287   426.3243    14.3982    -2.5165
         XGBoost 165 Train+Val   369.7296   432.9781    14.5971    -2.6271

=== Dự báo phiên sau 2024-12-31 ===
           Naive    2641.00
LinearRegression    2626.45
    RandomForest    2041.85
         XGBoost    2034.07
```

## Siêu tham số của từng mô hình

| Mô hình | File | Siêu tham số | Có quét K? |
|---|---|---|---|
| **Naive** | `baseline_naive.py` | không có (0 tham số) | không |
| **Linear Regression** | `linear_regression.py` | `fit_intercept=True`, scaler fit trên Train | không |
| **Random Forest** | `random_forest.py` | `max_features=0.8`, `min_samples_leaf=2`, `random_state=42`, `n_jobs=-1` | **K = 1..500** → 11 |
| **XGBoost** | `xgboost_model.py` | `max_depth=3`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1`, `objective=reg:squarederror`, `random_state=42` | **K = 1..500** → 165 |

Chỉ **số cây / số vòng K** được tinh chỉnh, và chỉ trên **Validation**. Test dùng đúng một lần.

## Dùng như thư viện

```python
import sys; sys.path.insert(0, "model_training")

from features import load_raw, make_features, split_masks
from baseline_naive import NaiveBaseline
from linear_regression import LinearRegressionModel
from random_forest import RandomForestModel
from xgboost_model import XGBoostModel

raw = load_raw()
X, y, target_dates = make_features(raw)
m = split_masks(target_dates)
Xtr, ytr = X.loc[m["train"]], y.loc[m["train"]]
Xtv = pd.concat([Xtr, X.loc[m["val"]]])       # cây fit trên Train+Val
ytv = pd.concat([ytr, y.loc[m["val"]]])

models = {
    "Naive": NaiveBaseline().fit(Xtr, ytr),
    "LR":    LinearRegressionModel().fit(Xtr, ytr),
    "RF":    RandomForestModel(k=11).fit(Xtv, ytv),
    "XGB":   XGBoostModel(k=165).fit(Xtv, ytv),
}
preds = {k: v.predict(X.loc[m["test"]]) for k, v in models.items()}
```

Mỗi wrapper đều có `.fit()` / `.predict()` giống sklearn, nên so sánh vòng lặp được.
`LinearRegressionModel` có thêm `.coef` (Series, sắp theo |β| giảm dần) và `.intercept`.
Hai mô hình cây có `.importances` (RF: MDI %, XGB: Gain %).

## Ba điểm dễ làm sai — đừng "sửa"

1. **RF/XGB fit lại trên Train+Val** sau khi chọn K; **LR/Naive chỉ dùng Train**.
   K được chọn trên Val nên sau đó phải tận dụng Val để học. Đổi thành Train-only → số khác báo cáo.
2. **Scaler nằm trong Pipeline của LR.** Không chuẩn hoá thêm lần nữa khi dự báo — sẽ ra số sai.
   Imputer cũng chỉ `fit` trên Train, không bao giờ fit lại trên Val/Test.
3. **Dự báo phiên kế tiếp phải dùng `latest_features(raw)`** (tức `drop_unlabeled=False`).
   Nếu dùng ma trận đã lọc nhãn (`drop_unlabeled=True`), phiên cuối bị loại và bạn dự báo
   **lùi mất một phiên** — lệch khoảng 15–20 USD mà không báo lỗi. Test `test_next_session_uses_last_row` canh chỗ này.

## Tăng tốc quét K

- **Random Forest:** cây thứ `1..k` của `RandomForest(n_estimators=500, random_state=42)`
  **chính là** rừng `k` cây fit riêng biệt → fit một lần rồi lấy trung bình tích lũy trên
  `estimators_` (sai khác < 1e-13).
- **XGBoost:** `booster.predict(..., iteration_range=(0, k))` cho đúng kết quả của mô hình
  fit `k` vòng.

Nhờ vậy quét đủ 500 K chỉ mất ~5 giây (RF) và ~2 giây (XGB).

## Kiểm chứng

```bash
.venv/bin/python -m pytest model_training -q     # 16 test (10 models + 6 train)
```

`test_models.py` canh những điều sau:

- 4 file mô hình rời cho **cùng bộ số** với `model_joblib/` (nguồn dashboard đang dùng) — lệch < 1e-9
- Metric khớp `gold_model_output/test_metrics.csv` — lệch < 1e-9
- Mỗi file tự mang đúng siêu tham số của mình (không đọc ké)
- Naive = `close_lag_0` (đúng định nghĩa)
- Dự báo cây bị chặn dưới 2.200 USD trong khi giá thực Test vượt 2.700 → tái hiện đúng kết luận
- `latest_features` lấy đúng phiên cuối (bẫy ở mục 3 trên)

## Lưu ý

- **Không trỏ `--out` vào `gold_price_fullstack/`** nếu chưa muốn ghi đè model dashboard.
- File `.joblib` gắn với **sklearn 1.9.1 · xgboost 3.2.0 · Python 3.11**.
- `features.py` là **bản copy** — sửa công thức ở `gold_price_fullstack/ml/features.py` thì
  phải sửa cả hai; `test_metrics_match_report` sẽ báo nếu lệch.
