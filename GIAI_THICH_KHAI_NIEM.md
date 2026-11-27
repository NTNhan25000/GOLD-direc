# Giải thích khái niệm, công thức và thuật ngữ

Tài liệu tra cứu cho đề tài **Dự báo giá vàng GC=F phiên kế tiếp**. Mỗi mục gồm: định nghĩa
→ công thức → **ví dụ số thật lấy từ chính repo này** → cạm bẫy đã gặp khi làm.

> Số liệu trong đây đều đọc từ `gold_model_output/*.csv` hoặc tính lại được bằng lệnh đi kèm.
> Nếu chạy lại pipeline mà số đổi, tài liệu này phải cập nhật theo.

## Mục lục

1. [Bài toán & dữ liệu](#1-bài-toán--dữ-liệu)
2. [Đặc trưng (feature engineering)](#2-đặc-trưng-feature-engineering)
3. [Bốn mô hình](#3-bốn-mô-hình)
4. [Đo lường sai số](#4-đo-lường-sai-số)
5. [Chẩn đoán & thống kê](#5-chẩn-đoán--thống-kê)
6. [Độ quan trọng đặc trưng](#6-độ-quan-trọng-đặc-trưng)
7. [Bảng tra thuật ngữ](#7-bảng-tra-thuật-ngữ)

---

## 1. Bài toán & dữ liệu

### 1.1 Hồi quy chuỗi thời gian, dự báo một bước

**Hồi quy** = tìm hàm $f$ sao cho $f(X) \approx y$, với $y$ là số thực (khác **phân loại** —
$y$ là nhãn rời rạc).

**Dự báo một bước** (one-step-ahead): chỉ dự báo phiên ngay sau phiên hiện tại.

$$\hat{y}(t+1) = f\big(X(t)\big), \qquad y(t) = \text{Close}(t+1)$$

Trong đó $X(t)$ là vector 25 đặc trưng **chỉ tính từ dữ liệu đến hết phiên $t$**.

Vì sao không dự báo nhiều bước: sai số tích lũy theo bậc — dự báo $t+5$ dùng $\hat{y}(t+4)$
làm đầu vào, nên sai số một bước bị khuếch đại qua 5 lần.

### 1.2 Rò rỉ dữ liệu (data leakage) và look-ahead bias

**Rò rỉ dữ liệu**: thông tin không được phép biết tại thời điểm dự báo lại lọt vào đặc trưng
hoặc vào bước huấn luyện. **Look-ahead bias** là dạng rò rỉ theo thời gian.

Ba lỗi cụ thể trong bài này, và cách chặn:

| Lỗi | Hậu quả | Cách chặn trong repo |
|---|---|---|
| Dùng `Close(t+1)` khi tính đặc trưng | Model "thấy đáp án", metric đẹp giả tạo | `make_features` chỉ dùng `shift(lag≥0)`; 16 test kiểm chứng |
| `shuffle=True` khi chia tập | Dòng năm 2020 nằm cạnh dòng 2024 → học tương lai | `split_masks` chia theo **ngày**, không shuffle |
| Fit scaler/imputer trên toàn bộ dữ liệu | Median, độ lệch chuẩn bị "nhìn" Val/Test | `imputer.fit(X_train)` rồi `transform(X_val)`, `transform(X_test)` |

**Điểm tinh tế đã gặp:** chia tập phải căn theo **ngày của giá mục tiêu** ($t+1$), không phải
ngày phát dự báo ($t$). Nếu căn theo $t$, dòng có $t = 2023\text{-}12\text{-}29$ rơi vào Val
nhưng nhãn của nó là giá $2024\text{-}01\text{-}02$ — tức phiên đầu Test bị dùng để chọn
siêu tham số.

### 1.3 Chuỗi không dừng (non-stationary)

**Dừng** (stationary) = phân phối không đổi theo thời gian: trung bình, phương sai, tự tương
quan đều hằng. Giá vàng **không dừng** — trung bình của nó tăng từ 1.050 lên 2.800 USD.

Đo bằng **kiểm định ADF** (Augmented Dickey–Fuller): giả thuyết $H_0$ = "chuỗi có nghiệm đơn
vị" (tức không dừng). $p < 0{,}05$ → bác bỏ $H_0$ → chuỗi dừng.

Hệ quả thực tế: mô hình học trên mức giá **không dừng** sẽ học "giá quanh 1.500"
rồi bị sốc khi giá lên 2.800. Đây là gốc rễ của toàn bộ kết quả âm tính trong bài.
Xem [mục 3.3](#33-vì-sao-cây-bị-chặn-ngoại-suy-công-thức-chính-xác).

---

## 2. Đặc trưng (feature engineering)

25 đặc trưng chia 4 nhóm. [xem `model_training/features.py`](model_training/features.py)

### 2.1 Giá trễ (lag)

$$\text{close\_lag}_k(t) = \text{Close}(t-k)$$

Dùng $k \in \{0, 1, 2, 3, 5, 10\}$ → 6 đặc trưng. `close_lag_0` chính là giá hiện tại, và là
đặc trưng **quan trọng nhất** (72,2% MDI ở Random Forest).

### 2.2 Trung bình động (SMA) và biến động

$$\text{SMA}_w(t) = \frac{1}{w}\sum_{i=0}^{w-1}\text{Close}(t-i)$$

$$\text{volatility}_w(t) = \text{std}\Big(\frac{\text{Close}(t-i)}{\text{Close}(t-i-1)} - 1\Big)_{i=0..w-1}$$

Với $w \in \{5, 10, 20\}$ → 6 đặc trưng. SMA làm mượt nhiễu; volatility đo mức bất ổn.

### 2.3 RSI-14 (Relative Strength Index)

$$\text{RSI} = 100 - \frac{100}{1 + \frac{\overline{\text{gain}}_{14}}{\overline{\text{loss}}_{14}}}$$

với $\text{gain} = \max(\Delta\text{Close}, 0)$ và $\text{loss} = \max(-\Delta\text{Close}, 0)$,
lấy trung bình trượt 14 phiên. RSI ∈ [0, 100]; > 70 thường gọi "quá mua", < 30 "quá bán".

**Lưu ý:** khi $\overline{\text{loss}} = 0$ công thức chia cho 0 → code dùng
`loss.replace(0, np.nan)` để tránh `inf`.

### 2.4 Lợi suất và biên độ

$$\text{gold\_return}_{1d}(t) = \frac{\text{Close}(t)}{\text{Close}(t-1)} - 1, \qquad
\text{gold\_range}(t) = \frac{\text{High}(t) - \text{Low}(t)}{\text{Close}(t)}$$

`gold_range` đo biên độ trong phiên so với giá đóng cửa — proxy cho mức bất định nội phiên.

### 2.5 Liên thị trường

5 chuỗi vĩ mô × 2 (mức giá + lợi suất 1 phiên) = 10 đặc trưng:
**DXY** (chỉ số đô la), **US10Y** (lợi suất trái phiếu 10 năm Mỹ), **S&P 500**, **WTI** (dầu),
**Bạc**.

Tương quan với giá vàng trên toàn bộ 2.514 phiên:

| Chuỗi | Tương quan Pearson |
|---|---:|
| S&P 500 | 0,943 |
| Bạc | 0,917 |
| WTI | 0,511 |
| DXY | 0,451 |
| US10Y | 0,413 |

**Cạm bẫy:** tương quan cao **không** có nghĩa là dự báo được. S&P 500 tương quan 0,943 với
vàng nhưng cả hai cùng đi theo xu hướng thời gian — đây là **tương quan giả** do chuỗi không
dừng. Thực nghiệm ablation (mục 5.2) cho thấy thêm 10 biến này **làm mô hình tệ đi**.

### 2.6 Forward-fill và vì sao không nội suy

Sàn vàng và thị trường chứng khoán nghỉ khác ngày. Cột vĩ mô được **forward-fill** (lấy giá
trị gần nhất trước đó):

```
ngày          DXY     forward-fill
...
2024-12-24   108.2    108.2
2024-12-25   NaN  →   108.2      (sàn vàng mở, chứng khoán nghỉ)
2024-12-26   108.4    108.4
```

**Không nội suy tuyến tính** giữa hai quan sát: nội suy tạo ra giá trị **chưa từng tồn tại**
tại thời điểm đó — về bản chất là bịa dữ liệu, và nghiêm trọng hơn là dùng cả giá trị *tương lai*
để nội suy giá trị quá khứ (rò rỉ hai chiều).

---

## 3. Bốn mô hình

### 3.1 Naive Baseline

$$\hat{y}(t+1) = \text{Close}(t)$$

Không huấn luyện, 0 tham số. **Đây không phải "mô hình cho có".** Với dự báo một phiên, giá
hai phiên liên tiếp rất gần nhau ($\text{corr} = 0{,}9946$) nên Naive là **baseline rất mạnh**.

Kết quả Test 2024: **MAE 18,038 USD/oz — thắng cả 3 mô hình học máy.**

Quy tắc: nếu mô hình học máy không vượt Naive, nó **chưa chứng minh được giá trị**.

### 3.2 Linear Regression (OLS)

$$\hat{y} = \beta_0 + \beta_1 x_1 + \cdots + \beta_{25} x_{25}$$

**OLS** (Ordinary Least Squares) chọn $\beta$ để cực tiểu tổng bình phương phần dư:

$$\hat{\beta} = \arg\min_\beta \sum_{i=1}^{n}\big(y_i - \beta_0 - \mathbf{x}_i^\top\beta\big)^2$$

Nghiệm dạng đóng: $\hat{\beta} = (X^\top X)^{-1}X^\top y$ — đây chính là điểm yếu: nếu $X^\top X$
gần suy biến (đa cộng tuyến) thì nghịch đảo cực nhạy, hệ số bùng nổ.

**Chuẩn hoá (StandardScaler):**

$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

Bắt buộc trước khi so sánh độ lớn hệ số (25 biến có đơn vị khác nhau: USD, %, chỉ số).
$\mu$ và $\sigma$ **chỉ tính trên Train**.

Kết quả: Test MAE 22,054 — **kém Naive**. Intercept = 1.417,209; hệ số lớn nhất
`close_lag_0` = 247,223.

**Cạm bẫy đã gặp:** `linear_regression.joblib` là một `Pipeline(scaler → OLS)`. Nếu chuẩn hoá
thêm lần nữa ở ngoài thì đầu vào bị chuẩn hoá 2 lần và dự báo sai hoàn toàn.

### 3.3 Vì sao cây bị chặn ngoại suy (công thức chính xác)

Cây quyết định chia không gian đặc trưng thành các vùng lá $R_1, \dots, R_m$, và dự báo:

$$\hat{f}(x) = \sum_{j=1}^{m} c_j \cdot \mathbb{1}[x \in R_j],
\qquad c_j = \text{mean}\{y_i : x_i \in R_j\}$$

Hai điểm mấu chốt:

1. $c_j$ là **trung bình của các nhãn đã thấy** trong vùng đó → luôn nằm trong
   $[\min y_{\text{train}}, \max y_{\text{train}}]$.
2. Vì vậy với **mọi** $x$ (kể cả $x$ chưa từng thấy):

$$\min y_{\text{train}} \le \hat{f}(x) \le \max y_{\text{train}}$$

Đây là **bất đẳng thức**, không phải hiện tượng thống kê — cây **không thể** dự báo ra ngoài
miền nhãn đã học, dù có bao nhiêu cây.

Số thật của bài này:

| Đại lượng | Giá trị |
|---|---:|
| Giá cao nhất của tập Train (08/2020) | **2.069,40** USD |
| Dự báo cao nhất của Random Forest trên Test | **2.079,82** USD |
| Dự báo cao nhất của XGBoost trên Test | **2.045,60** USD |
| Giá thực cao nhất trên Test 2024 | **2.800,80** USD |
| Số phiên Test có giá thực vượt trần RF | **211 / 252** (83,7%) |

Kiểm chứng bằng code: mỗi cây trong rừng 11 cây có dự báo lớn nhất nằm trong khoảng
1.995,85 – 2.047,12; trung bình 11 cây cho **2.026,13** — vẫn dưới trần Train.

> Đây là lý do **toàn bộ** sai số 364,529 USD đến từ một nguyên nhân duy nhất, không phải
> "model hỏng". Chẩn đoán đúng thì cách khắc phục hiện ra ngay: đổi mục tiêu sang ΔClose.

### 3.4 Bagging và Random Forest

**Bagging** (Bootstrap AGGregating):

1. Lấy $B$ mẫu bootstrap — mỗi mẫu $n$ dòng lấy **có hoàn lại** từ Train.
2. Huấn luyện một cây trên mỗi mẫu.
3. Dự báo = trung bình $B$ cây: $\hat{f}_{\text{bag}}(x) = \frac{1}{B}\sum_{b=1}^{B}\hat{f}_b(x)$.

Mục đích: **giảm phương sai**. Nếu các cây có phương sai $\sigma^2$ và tương quan đôi một
$\rho$, phương sai của trung bình là:

$$\text{Var}\big(\hat{f}_{\text{bag}}\big) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

Khi $B \to \infty$ vẫn còn lại $\rho\sigma^2$ → cần **giảm $\rho$** bằng cách làm các cây
khác nhau. Đó chính là **Random Forest**: ngoài bootstrap mẫu, mỗi lần chia cây chỉ xét một
tập con ngẫu nhiên gồm `max_features` đặc trưng.

Siêu tham số của bài:

| Tham số | Giá trị | Tác dụng |
|---|---|---|
| `n_estimators` (K) | **11** | số cây — quét 1..500 trên Validation |
| `max_features` | 0,8 | mỗi lần chia xét 80% × 25 = 20 đặc trưng → giảm $\rho$ |
| `min_samples_leaf` | 2 | lá phải có ≥ 2 mẫu → chống overfit |
| `random_state` | 42 | tái lập được |

Kết quả: Test MAE **364,529** (kém Naive 20,2 lần) — do giới hạn ở mục 3.3.

**OOB (Out-Of-Bag):** với mỗi dòng $i$, khoảng 37% cây không dùng dòng đó
($\lim_{n\to\infty}(1-1/n)^n = 1/e \approx 0{,}368$). Dự báo OOB chỉ dùng các cây **không
thấy** dòng $i$ → là ước lượng "kiểu validation" miễn phí.

Kết quả OOB của bài (Train+Val): **MAE 12,11 · R² 0,9968** — mô hình **rất giỏi trong miền
học**. Đây là bằng chứng quan trọng: nó không "dở", nó chỉ **không ngoại suy được**.

**Cạm bẫy đã gặp:** sklearn điền `oob_prediction_ = 0.0` cho các dòng **không có** cây nào
bỏ sót nó. Phải lọc `oob_pred > 0` trước khi tính metric, nếu không MAE sẽ bị kéo xuống
một cách vô nghĩa. Trong bài: **2.226 / 2.241** dòng hợp lệ.

### 3.5 Gradient Boosting và XGBoost

**Boosting** — ngược với bagging: cây mới sửa lỗi của cây trước, tuần tự.

$$\hat{f}^{(0)} = 0, \qquad
\hat{f}^{(k)} = \hat{f}^{(k-1)} + \eta \cdot h_k(x)$$

với $h_k$ là cây thứ $k$ fit lên **phần dư**, $\eta$ là learning rate.

Hàm mục tiêu có chính quy của XGBoost:

$$\mathcal{L} = \sum_i \ell\big(y_i, \hat{y}_i\big) + \sum_k \Omega(h_k),
\qquad \Omega(h) = \gamma T + \tfrac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

$T$ = số lá, $w_j$ = giá trị lá, $\lambda$ = `reg_lambda` (L2), $\gamma$ = mức phạt số lá.

Khác biệt bản chất với bagging: boosting **giảm độ chệch** (bias), bagging **giảm phương sai**
(variance). Nhưng **cả hai đều là cây** → **cả hai đều bị chặn ngoại suy** như mục 3.3.
Đó là lý do XGBoost cũng thất bại (MAE 369,730), không phải vì thuật toán kém.

Siêu tham số của bài:

| Tham số | Giá trị | Tác dụng |
|---|---|---|
| `n_estimators` (K) | **165** | số vòng — quét 1..500 trên Validation |
| `max_depth` | 3 | cây nông → ít overfit trên 1.739 mẫu |
| `learning_rate` (η) | 0,03 | bước nhỏ → cần nhiều vòng mới hội tụ |
| `subsample` | 0,8 | mỗi vòng dùng 80% dòng (stochastic boosting) |
| `colsample_bytree` | 0,8 | mỗi cây dùng 80% cột |
| `reg_lambda` (λ) | 1 | L2 trên giá trị lá |

**Vì sao cần tới 165 vòng:** $\eta = 0{,}03$ nhỏ nên phải cộng nhiều cây mới đủ. Bằng chứng:
tại $K = 100$ Val RMSE còn 49,784 USD; xuống 34,238 tại $K = 165$; rồi **tăng lại** 37,641
tại $K = 500$ (bắt đầu overfit).

### 3.6 K là gì, và mẹo quét 500 giá trị trong 5 giây

$K$ = số cây (Random Forest) hoặc số vòng boosting (XGBoost). Đây là **siêu tham số duy nhất
được tinh chỉnh**, và chỉ trên Validation.

Quét ngây thơ = fit 500 mô hình → quá chậm. Hai mẹo:

**Random Forest** — cây thứ $1..k$ của `RandomForest(n_estimators=500, random_state=42)`
**chính là** mô hình $k$ cây fit riêng biệt (cùng seed, cùng thứ tự sinh cây). Nên chỉ fit
một lần rồi lấy **trung bình tích lũy**:

$$\hat{y}_k = \frac{1}{k}\sum_{b=1}^{k}\hat{f}_b(x)$$

Kiểm chứng trên Validation:

| k | RMSE của trung bình tích lũy | RMSE fit riêng k cây | Lệch dự báo lớn nhất |
|---:|---:|---:|---:|
| 1 | 33,878711 | 33,878711 | 0.000e+00 |
| 3 | 31,165811 | 31,165811 | 0.000e+00 |
| 11 | 28,413333 | 28,413333 | 0.000e+00 |
| 50 | 32,128133 | 32,128133 | 4.547e-13 |

**XGBoost** — `booster.predict(DMatrix, iteration_range=(0, k))` cho đúng kết quả mô hình
$k$ vòng, không cần fit lại.

Nhờ vậy: quét đủ **1..500** chỉ mất ~5 giây (RF) và ~2 giây (XGB). Repo có test khẳng định
tính tương đương này (`pytest -k cumulative`).

---

## 4. Đo lường sai số

Cho $n$ phiên Test, $y_i$ = giá thực, $\hat{y}_i$ = dự báo, $e_i = y_i - \hat{y}_i$.

### 4.1 MAE — Mean Absolute Error

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|e_i|$$

Cùng đơn vị với giá (USD/oz) → **dễ diễn giải nhất**. "Trung bình lệch 18 USD/oz".

Nhược điểm: không phân biệt lỗi nhỏ và lỗi rất lớn (mọi lỗi đều tuyến tính).

### 4.2 RMSE — Root Mean Squared Error

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} e_i^2}$$

**Phạt nặng lỗi lớn** vì bình phương trước khi lấy trung bình. Vì vậy RMSE ≥ MAE luôn luôn;
tỉ số RMSE/MAE càng lớn thì càng có ít lỗi cực đoan.

Trong bài: Naive RMSE/MAE = 23,743/18,038 = **1,32** (lỗi lớn nhất 93,70 USD);
Random Forest = 426,324/364,529 = **1,17** (lỗi lớn nhất 759,82 USD).

Vì sao dùng **cả hai**: MAE cho biết "sai bao nhiêu tiền", RMSE cho biết "có lần sai rất
nặng không". Một mô hình có MAE thấp nhưng RMSE cao = thường đúng, thỉnh thoảng sai thảm.

### 4.3 MAPE — Mean Absolute Percentage Error

$$\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{e_i}{y_i}\right|$$

Đơn vị %, so sánh được giữa các tài sản có mức giá khác nhau.

**Cạm bẫy:** MAPE **không đối xứng** — chia cho $y_i$ nên lỗi cùng độ lớn bị phạt nặng hơn
khi giá thấp. Và MAPE **không được** dùng khi $y_i$ có thể bằng 0 hoặc âm.

Trong bài: Naive MAPE = 0,744% nghe rất tốt, nhưng đó là 0,744% của ~2.400 USD = 18 USD/oz.
MAPE thấp **không** có nghĩa sai số nhỏ về tuyệt đối.

### 4.4 R² — hệ số xác định

$$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}
= 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

$SS_{\text{tot}}$ đo tổng biến thiên của giá thực quanh trung bình $\bar{y}$. $R^2$ = tỉ lệ
phương sai được mô hình giải thích.

- $R^2 = 1$: hoàn hảo.
- $R^2 = 0$: mô hình **không hơn** việc đoán bằng $\bar{y}$.
- $R^2 < 0$: mô hình **tệ hơn** đoán bằng $\bar{y}$ ← **hoàn toàn hợp lệ về mặt toán học**,
  không phải lỗi.

Số thật của bài (Test 2024, $n = 252$):

| | Giá trị |
|---|---:|
| $\bar{y}$ | 2.404,71 |
| $\text{std}(y)$ | 227,34 |
| $SS_{\text{tot}}$ | 13.024.753,6 |
| $SS_{\text{res}}$ của Naive | 142.060,4 |
| $R^2$ Naive $= 1 - 142060{,}4/13024753{,}6$ | **0,9891** ✓ |

Vì sao Random Forest có $R^2 = -2{,}5165$: dự báo phẳng quanh 2.040 trong khi giá thực đi từ
2.073 lên 2.800. Mô hình sai **có hệ thống** → $SS_{\text{res}}$ lớn hơn $SS_{\text{tot}}$
rất nhiều.

**Cạm bẫy diễn giải:** nói "$R^2$ âm nên mô hình dở toàn tập" là **sai**. OOB MAE 12,11 và
$R^2$ 0,9968 trên Train+Val cho thấy nó học rất tốt **trong miền học**. Vấn đề là ngoại suy.

### 4.5 Vì sao Naive thắng: nghịch lý của dự báo một bước

Nếu $\text{Close}(t+1) \approx \text{Close}(t) + \varepsilon$ với $\varepsilon$ là nhiễu yếu,
thì mọi mô hình cố "học" $\varepsilon$ sẽ **thêm phương sai** vào dự báo mà không giảm được
độ chệch. Về lý thuyết sai số:

$$\mathbb{E}\big[(\hat{f} - y)^2\big] = \text{Bias}^2 + \text{Variance} + \sigma^2_{\text{irreducible}}$$

Naive có Bias gần 0 và Variance = 0. Model học máy có Bias có thể thấp hơn chút nhưng
Variance lớn hơn nhiều → **thua**. Đây không phải chuyện riêng của vàng; đây là lý do các
cuộc thi dự báo giá tài sản luôn có baseline "last value" và nó thường nằm trong top.

---

## 5. Chẩn đoán & thống kê

### 5.1 Ngoại suy (extrapolation)

**Nội suy** = dự báo trong khoảng đã quan sát. **Ngoại suy** = dự báo ngoài khoảng đó.

Mô hình tuyến tính ngoại suy được (đường thẳng $\beta_0 + \beta_1 x$ kéo dài mãi).
Mô hình cây **không** — xem chứng minh ở mục 3.3.

Trong bài, Test 2024 chính là một **bài kiểm tra ngoại suy**: giá đóng cửa từ 2.073 (02/01)
lên 2.800,80 (cuối năm), **+27,4%**, vượt xa đỉnh Train 2.069,40.

### 5.2 Ablation study

**Ablation** = bỏ bớt thành phần rồi đo tác động, để trả lời "thành phần đó có ích không".

Thiết kế của bài: hai bộ đặc trưng, hai mô hình, 4 tổ hợp.

| Bộ đặc trưng | Mô hình | Val RMSE | Test MAE | Test R² |
|---|---|---:|---:|---:|
| 15 biến kỹ thuật | Linear Regression | **17,486** | **18,531** | 0,9889 |
| 25 biến (+ vĩ mô) | Linear Regression | 19,033 | 22,054 | 0,9858 |
| 15 biến kỹ thuật | Random Forest (K=2) | **25,994** | **342,668** | −2,1298 |
| 25 biến (+ vĩ mô) | Random Forest (K=11) | 28,413 | 364,529 | −2,5165 |

Kết luận: bỏ 10 biến vĩ mô thì **cả hai** mô hình đều tốt hơn. LR: Val RMSE −8,1%,
Test MAE −16%.

Vì sao: 11/25 biến có VIF > 10 (xem 5.3). Biến vĩ mô lặp lại thông tin đã có trong giá vàng.

### 5.3 Đa cộng tuyến và VIF

**Đa cộng tuyến** = hai hay nhiều đặc trưng có tương quan mạnh với nhau, nên chúng "giành
nhau" giải thích cùng một phần phương sai. Hệ số OLS trở nên bất ổn, dấu có thể lật.

**VIF** (Variance Inflation Factor) đo mức độ:

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

với $R_j^2$ là $R^2$ khi hồi quy $x_j$ lên **tất cả các biến còn lại**.

| VIF | Diễn giải |
|---|---|
| 1 | không cộng tuyến |
| 1–5 | chấp nhận được |
| 5–10 | đáng lo |
| > 10 | cộng tuyến nghiêm trọng |

Công thức cho thấy VIF bùng nổ khi $R_j^2 \to 1$. Ví dụ $R_j^2 = 0{,}9999$ → VIF = 10.000.

**Số thật trong bài:** 11/25 biến có VIF > 10; cao nhất là `sma_5` với **VIF = 17.593,1**.
Tính lại bằng tay cho 3 biến (`close_lag_0`, `sma_5`, `close_lag_1`) trên Train:

| Biến | $R^2$ khi hồi quy lên 2 biến còn lại | VIF |
|---|---:|---:|
| close_lag_0 | 0,997243 | 362,7 |
| sma_5 | 0,998354 | 607,4 |
| close_lag_1 | 0,998613 | 721,1 |

Đây là lý do **không được diễn giải hệ số OLS như tác động nhân quả**.

### 5.4 p-value và kiểm định hệ số

Với mỗi hệ số, kiểm định $H_0: \beta_j = 0$ (biến không có tác dụng).

$$t_j = \frac{\hat{\beta}_j}{\text{SE}(\hat{\beta}_j)}$$

$p$-value = xác suất quan sát được $|t|$ lớn như vậy **nếu $H_0$ đúng**. Quy ước: $p < 0{,}05$
→ bác bỏ $H_0$. Trong bài: 8/25 hệ số có $p < 0{,}05$.

**Cạm bẫy:** khi đa cộng tuyến nặng, SE bị thổi phồng → $p$ lớn → **kết luận sai** là "biến
không quan trọng". "Không có ý nghĩa thống kê" ≠ "không có tác dụng".

### 5.5 Durbin–Watson

Kiểm định **tự tương quan bậc 1** của phần dư:

$$\text{DW} = \frac{\sum_{t=2}^{n}(e_t - e_{t-1})^2}{\sum_{t=1}^{n}e_t^2} \approx 2(1 - \rho_1)$$

| DW | Ý nghĩa |
|---|---|
| ≈ 2 | không tự tương quan |
| < 2 | tự tương quan dương (hay gặp trong chuỗi thời gian) |
| > 2 | tự tương quan âm |

Trong bài: **Train 2,010** (tốt) nhưng **Test 1,513** → phần dư trên Test có tự tương quan
dương, dấu hiệu mô hình bỏ sót cấu trúc.

### 5.6 Đổi mục tiêu sang ΔClose

$$\Delta\text{Close}(t+1) = \text{Close}(t+1) - \text{Close}(t), \qquad
\hat{y}(t+1) = \text{Close}(t) + \hat{\Delta}(t+1)$$

**Vì sao khắc phục được:** $\Delta$ là biến **dừng** (trung bình ≈ 0, không có xu hướng) —
khác hoàn toàn với mức giá. Cây học "phiên sau tăng/giảm bao nhiêu USD", giá trị này nằm trong
vài chục USD nên **không bị chặn** bởi đỉnh giá lịch sử. Cộng lại vào giá hiện tại thì mô
hình vẫn theo được xu hướng vượt đỉnh.

Kết quả Test:

| Cách tiếp cận | MAE | Đúng chiều |
|---|---:|---:|
| Naive (Δ = 0) | 18,0381 | — |
| **XGBoost (ΔClose, K=21)** | **18,0039** | **55,95%** |
| Random Forest (ΔClose, K=71) | 18,2533 | 49,60% |
| Linear Regression (ΔClose) | 18,9193 | 51,59% |

**Đọc cho đúng:** mức cải thiện chỉ **0,0342 USD** — đủ để chứng minh *chẩn đoán đúng*,
**không** đủ để dùng giao dịch.

**Đúng chiều** (directional accuracy) = % phiên dự báo đúng dấu tăng/giảm. Đoán ngẫu nhiên
= 50%. XGB được 55,95%.

---

## 6. Độ quan trọng đặc trưng

Bốn cách đo, **không tương đương** — đây là nguồn hiểu lầm phổ biến nhất.

### 6.1 MDI (Mean Decrease in Impurity)

Với mỗi lần chia nút dùng biến $j$, mức giảm "độ vẩn" (variance) được cộng dồn:

$$\text{MDI}_j = \frac{1}{B}\sum_{b=1}^{B}\sum_{\substack{t \in T_b \\ v(t)=j}}
\frac{n_t}{n}\Big(\text{Var}(t) - \frac{n_L}{n_t}\text{Var}(L) - \frac{n_R}{n_t}\text{Var}(R)\Big)$$

Dùng trong `rf_feature_importance.csv`.

**Cạm bẫy:** MDI **thiên vị biến liên tục nhiều giá trị khác nhau** (như mức giá) vì chúng có
nhiều điểm chia hơn. Nó cũng đo trên **dữ liệu huấn luyện**, không phản ánh hiệu quả ngoài mẫu.

Kết quả bài: `close_lag_0` 72,2% · `close_lag_1` 17,6% · `sma_10` 8,8% → **top 3 = 98,6%**.
Tổng 10 biến vĩ mô = **0,11%**.

### 6.2 Gain (XGBoost)

Tổng mức giảm hàm mất mát do các lần chia dùng biến đó, chuẩn hoá thành %:

$$\text{Gain}_j = \sum_{\text{lần chia dùng } j}\Delta\mathcal{L}$$

Tổng Gain = 100% theo định nghĩa. Dùng trong `xgb_feature_importance.csv`.

Kết quả bài: `sma_20` 29,1% · `close_lag_0` 26,1% · `close_lag_1` 20,7%.
→ **Thứ hạng khác MDI nhưng cùng kết luận**: nhóm giá trễ + SMA chiếm gần hết.

### 6.3 Permutation importance

Xáo ngẫu nhiên một cột rồi đo **mức tăng RMSE**:

$$\text{PI}_j = \text{RMSE}\big(X \text{ với cột } j \text{ bị xáo}\big) - \text{RMSE}(X)$$

Đo trên **dữ liệu chưa thấy** nên đáng tin hơn MDI. Nhược điểm: nếu hai biến tương quan cao,
xáo một biến thì biến kia "gánh" được → **đánh giá thấp cả hai**. Với 25 biến tương quan
mạnh trong bài, permutation là **không đáng tin** và repo không dùng làm kết luận chính.

### 6.4 SHAP

Giá trị Shapley từ lý thuyết trò chơi, phân bổ đóng góp của từng biến cho **từng dự báo riêng**:

$$\hat{f}(x) = \phi_0 + \sum_{j=1}^{p}\phi_j(x)$$

Tính đúng đắn về lý thuyết (thoả tính hiệu quả, đối xứng, null player), nhưng tốn
$O(2^p)$ nếu tính chính xác — thực tế dùng TreeSHAP $O(TLD)$.

Repo dùng `mean_abs_SHAP_test_USD` trong `xgb_feature_importance.csv` làm kiểm chứng chéo.

### 6.5 Tổng hợp cho bài này

| Cách đo | Top 1 | Kết luận |
|---|---|---|
| MDI (RF) | close_lag_0 72,2% | giá trễ chi phối |
| Gain (XGB) | sma_20 29,1% | nhóm giá trễ + SMA chi phối |
| SHAP (XGB) | close_lag_0 cao nhất | khớp |

**Câu chốt đúng:** "nhóm giá trễ và SMA chiếm >98% độ quan trọng ở RF; 10 biến vĩ mô
chiếm 0,11% (RF) và 0,37% (XGB)."

**Câu chốt sai:** "close_lag_0 quan trọng 72,2% nên giá hôm nay quyết định 72,2% giá mai."
Độ quan trọng **không phải** quan hệ nhân quả và **không phải** tỉ lệ phương sai.

---

## 7. Bảng tra thuật ngữ

| Thuật ngữ | Viết tắt | Nghĩa | Trong bài |
|---|---|---|---|
| Baseline | — | mốc so sánh tối thiểu | Naive: MAE 18,038 |
| Bagging | — | Bootstrap AGGregatING: nhiều mô hình song song trên mẫu bootstrap, lấy trung bình | Random Forest |
| Boosting | — | mô hình tuần tự, mỗi cái sửa lỗi cái trước | XGBoost |
| Bootstrap | — | lấy mẫu **có hoàn lại** để tạo tập con | mỗi cây RF dùng một mẫu bootstrap |
| Data leakage | — | thông tin tương lai lọt vào huấn luyện | chặn bằng chia theo thời gian + test |
| Directional accuracy | — | % phiên đúng dấu tăng/giảm | XGB ΔClose: 55,95% |
| Durbin–Watson | DW | kiểm định tự tương quan phần dư, lý tưởng ≈ 2 | Train 2,010 · Test 1,513 |
| Extrapolation | — | dự báo ngoài miền đã quan sát | cây không làm được |
| Feature | đặc trưng | biến đầu vào của mô hình | 25 biến |
| Forward-fill | ffill | điền giá trị gần nhất trước đó | cột vĩ mô ngày nghỉ |
| Gain | — | mức giảm hàm mất mát do một biến | XGB, tổng = 100% |
| Gradient boosting | GBDT | boosting dùng gradient của hàm mất mát | XGBoost |
| Imputer | — | điền giá trị thiếu | median, chỉ fit trên Train |
| K | — | số cây / số vòng boosting | RF 11 · XGB 165 |
| Lag | trễ | giá trị các phiên trước | close_lag_0..10 |
| Learning rate | η | hệ số thu nhỏ mỗi bước boosting | 0,03 |
| MAE | — | trung bình trị tuyệt đối sai số | Naive 18,038 |
| MAPE | — | trung bình % sai số | Naive 0,744% |
| MDI | — | mức giảm độ vẩn bình quân | RF importance |
| Naive | — | dự báo = giá trị hiện tại | ŷ(t+1) = Close(t) |
| Non-stationary | — | chuỗi có phân phối đổi theo thời gian | giá vàng |
| OOB | Out-Of-Bag | mẫu không được dùng để huấn luyện một cây | MAE 12,11 |
| OLS | Ordinary Least Squares | hồi quy bình phương tối thiểu | Linear Regression |
| Overfitting | quá khớp | học cả nhiễu, kém khi gặp dữ liệu mới | XGB tại K=500 |
| p-value | — | xác suất thấy kết quả nếu H₀ đúng | 8/25 hệ số < 0,05 |
| Permutation importance | PI | mức tăng RMSE khi xáo một cột | không dùng làm kết luận chính |
| Random Forest | RF | nhiều cây + bootstrap mẫu + mẫu con đặc trưng | K = 11 |
| RMSE | — | căn trung bình bình phương sai số | Naive 23,743 |
| Rolling window | cửa sổ trượt | tính thống kê trên w phiên gần nhất | SMA, volatility |
| RSI | Relative Strength Index | chỉ báo động lượng 0–100 | rsi_14 |
| R² | hệ số xác định | tỉ lệ phương sai giải thích được; **âm là hợp lệ** | Naive 0,9891 |
| Shapley / SHAP | — | phân bổ đóng góp công bằng cho từng biến | kiểm chứng chéo XGB |
| SimpleImputer | — | lớp sklearn điền giá trị thiếu | median |
| SMA | Simple Moving Average | trung bình động đơn giản | sma_5/10/20 |
| StandardScaler | — | chuẩn hoá về trung bình 0, độ lệch chuẩn 1 | trong Pipeline của LR |
| Test set | tập Test | chỉ dùng **một lần** để đánh giá cuối | 252 phiên năm 2024 |
| Validation set | tập Val | dùng để chọn siêu tham số | 502 phiên 2022–2023 |
| Variance Inflation Factor | VIF | mức phóng đại phương sai do đa cộng tuyến | max 17.593,1 (sma_5) |
| XGBoost | XGB | gradient boosting có chính quy | K = 165 |

---

## 8. Những chỗ dễ hiểu sai — đọc trước khi bảo vệ

| Phát biểu sai | Phát biểu đúng |
|---|---|
| "R² âm nên mô hình hỏng" | R² âm = tệ hơn đoán bằng trung bình; mô hình vẫn học tốt **trong miền học** (OOB MAE 12,11) |
| "Random Forest kém nên không dùng RF" | RF đúng cho bài **nội suy**; sai cho bài **ngoại suy** — đây là kết luận chuyển giao được |
| "MAPE 0,744% là sai số rất nhỏ" | 0,744% của 2.400 USD = 18 USD/oz; MAPE thấp không có nghĩa sai số tuyệt đối nhỏ |
| "close_lag_0 quan trọng 72,2% nên nó quyết định 72,2% giá mai" | Độ quan trọng ≠ nhân quả ≠ tỉ lệ phương sai |
| "Biến vĩ mô vô dụng" | Chính xác là: **không cải thiện dự báo một phiên**. Với horizon dài hơn có thể khác |
| "Naive thắng nên học máy vô dụng" | Với dự báo một phiên thì đúng; giá trị của mô hình nằm ở **giải thích nguyên nhân** và ΔClose |
| "ΔClose hơn Naive 0,034 USD là thành công" | Đủ để chứng minh chẩn đoán đúng, **không** đủ để giao dịch |
| "Quét 500 K mà nhanh thế là có gì đó sai" | Trung bình tích lũy trên một rừng 500 cây tương đương **tuyệt đối** với fit riêng từng K (lệch < 1e-13) |
| "VIF 17.593 là lỗi tính toán" | Không: `sma_5` gần như trùng `close_lag` ngắn hạn; đây chính là lý do không diễn giải hệ số OLS |
| "p > 0,05 nghĩa là biến không quan trọng" | Đa cộng tuyến thổi phồng SE → p lớn; "không có ý nghĩa thống kê" ≠ "không có tác dụng" |

---

## 9. Lệnh kiểm chứng lại mọi con số trong tài liệu này

```bash
# Bảng Test + metric (4 mô hình)
.venv/bin/python model_training/run_all.py --fast

# Từng mô hình riêng, có in hệ số / độ quan trọng
.venv/bin/python model_training/linear_regression.py
.venv/bin/python model_training/random_forest.py
.venv/bin/python model_training/xgboost_model.py

# So sánh metric gốc trong gold_model_output
cat gold_model_output/test_metrics.csv
cat gold_model_output/delta_target_experiment.csv
cat gold_model_output/ablation_macro_study.csv

# Chẩn đoán ngoại suy, OOB, OLS
cat gold_model_output/extra_diagnostics.json

# Kiểm tra toàn bộ (38 test)
.venv/bin/python -m pytest gold_price_fullstack/tests model_joblib model_training -q
```
