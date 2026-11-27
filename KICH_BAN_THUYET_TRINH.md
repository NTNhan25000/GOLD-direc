# Kịch bản thuyết trình — Nhóm 8

**Đề tài:** Dự báo giá vàng phiên kế tiếp bằng học máy
**Bộ slide:** `slides_thuyet_trinh_nhom8.pptx` — **10 slide**, nền trắng, 16:9
**Thời lượng chuẩn:** 13 phút nói + 5–7 phút hỏi đáp (xem bản 7 phút ở mục 6 nếu bị cắt)

> Mọi con số trong kịch bản này lấy từ `gold_model_output/*.csv` và
> `extra_diagnostics.json` — trùng khớp với số in trên slide. Nếu chạy lại pipeline
> thì chạy lại `make_slides.py` rồi đối chiếu lại file này.

---

## 1. Phân bổ thời gian

| Slide | Nội dung | Phút | Cộng dồn |
|---|---|---:|---:|
| 1 | Bìa | 0:30 | 0:30 |
| 2 | Đặt vấn đề & 3 câu hỏi nghiên cứu | 1:05 | 1:35 |
| 3 | Phát biểu bài toán, dữ liệu & **tiền xử lý** | 1:50 | 3:35 |
| 4 | Thiết lập thực nghiệm (chia tập, 4 mô hình, K) | 1:20 | 4:55 |
| 5 | Chọn K trên Validation | 0:55 | 5:50 |
| 6 | **Kết quả Test 2024 — kết quả gây bất ngờ** | 1:30 | 7:15 |
| 7 | Chẩn đoán: vì sao cây thua | 1:45 | 9:00 |
| 8 | Ablation (CH2) + độ quan trọng đặc trưng (CH3) | 1:45 | 10:45 |
| 9 | Khắc phục bằng ΔClose | 1:15 | 12:00 |
| 10 | Kết luận, hạn chế, hướng phát triển | 1:00 | 13:00 |

**Nguyên tắc nhịp:** slide 6 và slide 7 là hai slide nặng nhất về học thuật — nói chậm,
dừng 2–3 giây sau mỗi con số chính. Slide 3, 4 là phần thủ tục, nói gọn để dành thời gian.

---

## 2. Chuẩn bị trước khi lên (5 phút)

```bash
# 1) Mở đúng file slide
#    slides_thuyet_trinh_nhom8.pptx  (10 slide, KHÔNG phải bản 22 slide cũ)

# 2) Nếu cần chạy demo số liệu trực tiếp trước giám thị
.venv/bin/python gold_predict.py        # ~5 giây, in bảng Test

# 3) Nếu cần demo dashboard
cd gold_price_fullstack && ../.venv/bin/python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

- Mở sẵn slide 6 ở chế độ Presenter View, ghi nhớ 3 số: **18,038 · 364,529 · ×20,2**.
- Chuẩn bị sẵn một terminal đã `cd` vào repo (nếu muốn demo `gold_predict.py` ở slide 6).
- Tắt thông báo, để chế độ trình chiếu không tự sleep.
- Không mở file 22 slide cũ trong cùng buổi — tránh nhầm bản.

---

## 3. Kịch bản từng slide

### Slide 1 — Bìa (0:30)

**Mục tiêu:** chào, nêu đề tài, nêu quy mô dữ liệu trong một câu.

> "Kính chào thầy cô, em là đại diện Nhóm 8. Đề tài của nhóm là **dự báo giá vàng phiên kế
> tiếp bằng học máy**. Nhóm dùng hợp đồng tương lai vàng COMEX — mã GC=F, đơn vị USD trên
> troy ounce — với **2.514 phiên giao dịch từ 2015 đến hết 2024**, thu từ 6 nguồn trên
> Yahoo Finance. Nhóm so sánh 4 mô hình: Naive Baseline, Linear Regression, Random Forest
> và XGBoost, và đánh giá trên đúng 252 phiên của năm 2024."

**Chuyển:** "Để bắt đầu, em xin nói vì sao đây lại là một bài toán khó."

---

### Slide 2 — Đặt vấn đề & mục tiêu (1:15)

**Mục tiêu:** dựng mức độ khó, rồi đặt ra 3 câu hỏi nghiên cứu để cả bài là đi trả lời.

> "Trong 10 năm qua giá vàng tăng khoảng **2,7 lần**: từ 1.050,8 lên 2.800,8 USD/oz.
> Nhìn xu hướng thì rất rõ — nhưng **dự báo đúng phiên kế tiếp lại là chuyện khác**."

> "Vàng là tài sản trú ẩn, hút vốn khi lạm phát và bất định địa chính trị tăng. Về mặt học
> máy, đây là bài toán **hồi quy chuỗi thời gian, dự báo trước một bước**. Khó ở ba điểm:
> nhiễu cao, chuỗi không dừng, và tín hiệu của một phiên yếu hơn xu hướng rất nhiều.
> Vì vậy yêu cầu là phải **so sánh công bằng với mốc cơ sở** và **đánh giá ngoài mẫu**."

> (chỉ vào 3 thẻ dưới) "Nhóm đặt ra ba câu hỏi — cả phần còn lại của bài là để trả lời dứt
> khoát ba câu này:
> **CH1** — mô hình cây phi tuyến có vượt được Naive Baseline và Linear Regression không?
> **CH2** — thêm 10 biến kinh tế vĩ mô liên thị trường có cải thiện độ chính xác không?
> **CH3** — đặc trưng nào đóng góp nhiều nhất vào quyết định của mô hình?"

**Chuyển:** "Để trả lời được, trước hết phải phát biểu bài toán thật chặt."

---

### Slide 3 — Phát biểu bài toán & dữ liệu (1:30)

**Mục tiêu:** chứng minh đề tài được thiết kế chống rò rỉ dữ liệu ngay từ đầu.

> "Điểm quan trọng nhất của slide này là **mốc thời gian**. Tại cuối phiên t, nhóm chỉ được
> biết dữ liệu đến hết phiên t — và cần dự báo giá đóng cửa phiên t + 1. Công thức là
> **ŷ(t+1) = f(X(t))**, với biến mục tiêu **y(t) = Close(t+1)**."

> (chỉ timeline) "…phiên t−2, t−1 là quá khứ, phiên t là thời điểm phát tín hiệu, phiên
> t+1 mới là cái cần dự báo. **Tuyệt đối không được dùng giá trị Close(t+1) khi tạo đặc
> trưng** — đây là lỗi phổ biến nhất của bài dự báo chuỗi thời gian."

> "Bộ dữ liệu: **2.514 phiên, 10 cột, 0 giá trị khuyết**, từ 2015-01-02 đến 2024-12-31,
> biên giá 1.050,8 – 2.800,8 USD/oz. Sáu nguồn gồm OHLCV vàng và 5 chuỗi vĩ mô: chỉ số
> đô la DXY, lợi suất trái phiếu 10 năm Mỹ, S&P 500, dầu WTI và bạc. Các cột vĩ mô được
> **forward-fill** theo lịch nghỉ của sàn vàng, không nội suy tuyến tính để tránh tạo
> thông tin giả."

> (chỉ 4 nhóm biến) "Từ đó nhóm tạo **25 đặc trưng**: 6 giá trễ, 6 trung bình động và biến
> động, 3 chỉ báo động lượng, và 10 biến liên thị trường."

> (chỉ dải TIỀN XỬ LÝ dưới) "Phần **tiền xử lý** gồm bốn bước, và mỗi bước đều có lý do:
> **Một**, sàn vàng và chứng khoán nghỉ khác ngày nên 5 chuỗi vĩ mô được **forward-fill** —
> tuyệt đối **không nội suy tuyến tính**, vì nội suy sẽ tạo ra giá trị chưa từng tồn tại ở
> thời điểm đó. **Hai**, số NaN còn lại được điền bằng **trung vị của Train**.
> **Ba**, Linear Regression cần **chuẩn hoá** vì 25 biến có đơn vị khác nhau — và cả imputer
> lẫn scaler đều **chỉ fit trên Train**. **Bốn**, nhóm **kiểm tra ngoại lệ**: kết quả là
> 0 NaN, 0 ngày trùng, OHLC nhất quán 100%; có COVID tháng 3/2020 và WTI âm tháng 4/2020
> nhưng đó là **sự kiện thật**, nên nhóm **chủ động giữ lại** thay vì xoá."

> "Và quan trọng: **16 test tự động** kiểm tra rằng mọi đặc trưng tại phiên t chỉ sinh từ
> dữ liệu ≤ t."

**Chuyển:** "Dữ liệu đã sạch và tiền xử lý xong — tiếp theo là thiết lập thực nghiệm."

---

### Slide 4 — Thiết lập thực nghiệm (1:30)

**Mục tiêu:** cho thấy cách chia tập và cách chọn siêu tham số là nghiêm ngặt, không tinh chỉnh trên Test.

> "Nhóm chia **theo thời gian**, không shuffle — để mô hình không bao giờ nhìn thấy tương lai.
> **Train** từ 2015 đến hết 2021 với 1.739 mẫu; **Validation** 2022–2023 với 502 mẫu,
> chỉ dùng để chọn siêu tham số; **Test** là cả năm 2024, 252 mẫu, **chỉ dùng một lần duy
> nhất để đánh giá cuối**."

> "Có ba chi tiết chống rò rỉ: thứ tự thời gian giữ nguyên; tập được căn theo **ngày của giá
> mục tiêu** chứ không theo ngày phát dự báo — nhờ vậy phiên đầu Test không lọt vào Train;
> và **imputer với scaler chỉ fit trên Train**."

> (chỉ 4 thẻ mô hình) "Bốn mô hình là mức tối thiểu để so sánh công bằng: Naive làm mốc cơ
> sở, Linear Regression làm mốc diễn giải được, còn Random Forest và XGBoost đại diện cho
> hai họ phi tuyến — bagging và boosting."

> "Siêu tham số giữ cố định: Random Forest dùng max_features = 0,8 và min_samples_leaf = 2;
> XGBoost dùng max_depth = 3, learning rate 0,03, subsample và colsample 0,8. **Chỉ có một
> thứ được quét là số cây K, từ 1 đến 500, trên Validation** — chọn theo RMSE thấp nhất.
> Quét đủ 500 giá trị chỉ mất khoảng 5 giây nhờ kỹ thuật trung bình tích lũy trên một rừng
> 500 cây, và có test khẳng định tương đương tuyệt đối với cách fit riêng từng K."

**Chuyển:** "Vậy K nào được chọn?"

---

### Slide 5 — Chọn siêu tham số (1:00)

**Mục tiêu:** nêu K đã chọn, và quan trọng hơn — nêu tín hiệu cảnh báo từ đường cong.

> "Kết quả: **Random Forest chọn K = 11 cây**, Val RMSE 28,413, R² 0,9294.
> **XGBoost chọn K = 165 vòng**, Val RMSE 34,238, R² 0,8975."

> (chỉ bảng nhỏ) "Nhưng điều đáng chú ý hơn K là **hình dạng đường cong**. Random Forest
> gần như phẳng từ K = 50 trở đi — dao động quanh 32 USD. Nghĩa là **chỉnh K không cứu được
> mô hình**; vấn đề nằm ở bản chất thuật toán, không ở siêu tham số."

> "XGBoost thì cần tới 165 vòng vì learning rate thấp: tại K = 100 RMSE còn 49,78 USD, tức
> là chưa hội tụ. Và điểm cuối cùng của slide này: **trên Test, mọi giá trị K của cả hai mô
> hình đều kém Naive** — nên đây không phải chuyện chọn K."

**Chuyển:** "Vậy trên Test, cụ thể là kém bao nhiêu?"

---

### Slide 6 — Kết quả Test 2024 (1:30) ⭐

**Mục tiêu:** đưa ra con số gây bất ngờ, và ngay lập tức trấn an rằng đây là kết quả thật.

> "Đây là bảng kết quả chính. **Naive Baseline thắng**: MAE 18,038 USD/oz, RMSE 23,743,
> MAPE 0,744%, R² 0,9891. Linear Regression đứng sau với MAE 22,054. Còn hai mô hình cây
> **thua rất xa**: Random Forest MAE 364,529, XGBoost 369,730 — **kém Naive 20,2 lần**,
> và R² âm."

> (dừng 2 giây) "Con số này nhìn qua giống lỗi code. Nhưng **không phải lỗi**."

> "Thứ nhất, **16 test tự động** khẳng định metric khớp báo cáo tới sai số 1e-9 — ai sửa
> code làm lệch kết quả là test fail ngay. Thứ hai, sai số **bùng nổ đúng lúc giá vượt đỉnh
> lịch sử**: MAE của Random Forest theo quý là 47,39 USD ở quý I, nhưng 308,44 USD ở quý II —
> chính là lúc giá vàng vượt 2.100 USD, cao hơn mọi mức đã thấy khi huấn luyện."

> "Và trên đồ thị, hai đường RF và XGBoost nằm ngang quanh 2.040 USD, không bám được giá
> thực. Đó chính là manh mối cho slide tiếp theo."

**Chuyển:** "Vậy tại sao mô hình cây lại bị chặn?"

---

### Slide 7 — Chẩn đoán nguyên nhân (1:45) ⭐

**Mục tiêu:** đây là đóng góp học thuật của nhóm — giải thích được nguyên nhân bằng con số.

> "Nguyên nhân là **giới hạn ngoại suy của cây quyết định**. Cây dự báo bằng **giá trị trung
> bình của lá**, mà mọi lá đều nằm trong miền giá đã thấy khi huấn luyện. Đỉnh của tập Train
> là **2.069,40 USD**, vào tháng 8 năm 2020. Vì vậy dự báo của cây **bị chặn trên** — đây là
> giới hạn toán học của phép chia không gian, không phải bug."

> "Ba con số chứng minh: dự báo cao nhất mà Random Forest đưa ra là **2.079,82 USD**, trong
> khi giá thực cao nhất trên Test là **2.800,80 USD**. Và có tới **211 trên 252 phiên** —
> tức 84% số phiên — có giá thực vượt trần dự báo của mô hình. XGBoost cũng tương tự:
> 2.045,60 USD và 223/252 phiên."

> "Điểm cuối cùng, rất quan trọng để không hiểu sai: mô hình **không hề yếu trong miền học**.
> OOB MAE trên Train+Val chỉ **12,11 USD**, R² 0,9968. Nó học rất tốt — chỉ là **không thể
> dự báo ra ngoài miền đã học**. Và năm 2024, giá vàng tăng +27,4% — đúng loại biến động mà
> mô hình cây không thể theo."

**Chuyển:** "Vậy còn hai câu hỏi còn lại — biến vĩ mô và độ quan trọng đặc trưng?"

---

### Slide 8 — Ablation (CH2) & độ quan trọng (CH3) (1:45)

**Mục tiêu:** trả lời CH2 và CH3 bằng thực nghiệm, không bằng cảm tính.

> "**CH2 — câu trả lời là KHÔNG, thêm biến vĩ mô làm mô hình tệ đi.** Nhóm làm ablation:
> bỏ 10 biến vĩ mô, chỉ giữ 15 biến kỹ thuật. Với Linear Regression, Val RMSE **giảm từ
> 19,033 xuống 17,486** — tức −8,1% — và Test MAE **giảm từ 22,054 xuống 18,531**, tức −16%.
> Random Forest cũng vậy: Val RMSE từ 28,413 xuống 25,994."

> "Vì sao? Vì biến vĩ mô **tương quan mạnh với nhau và với chính giá vàng** — 11 trong 25
> biến có VIF > 10, cao nhất là sma_5 với VIF 17.593,1. Chúng chủ yếu **lặp lại thông tin đã
> có trong giá vàng**, nên chỉ thêm nhiễu và đa cộng tuyến cho bài dự báo một phiên."

> (chỉ biểu đồ) "**CH3 — đặc trưng quan trọng nhất là nhóm giá trễ và SMA.** Random Forest
> xếp close_lag_0 ở 72,2%, close_lag_1 ở 17,6%, sma_10 ở 8,8% — **top 3 chiếm 98,6%**. Còn
> toàn bộ 10 biến vĩ mô chỉ chiếm **0,11%** ở Random Forest và **0,37%** ở XGBoost. XGBoost
> dùng thang Gain nên xếp hạng hơi khác: sma_20 29,1%, close_lag_0 26,1%, close_lag_1
> 20,7% — nhưng cùng một kết luận."

> "Xin lưu ý: mức quan trọng **không chứng minh quan hệ nhân quả**; nó chỉ nói mô hình dùng
> biến nào để chia không gian."

**Chuyển:** "Nếu nguyên nhân đúng là giới hạn ngoại suy, thì đổi cách đặt mục tiêu phải
khắc phục được. Nhóm đã kiểm chứng điều đó."

---

### Slide 9 — Khắc phục bằng ΔClose (1:15)

**Mục tiêu:** chứng minh chẩn đoán đúng bằng một thực nghiệm sửa được vấn đề.

> "Thay vì dự báo **mức giá**, nhóm dự báo **mức thay đổi**: ΔClose(t+1) = Close(t+1) −
> Close(t), rồi cộng lại vào giá hiện tại. Cây bây giờ chỉ cần học 'phiên sau tăng hay giảm
> bao nhiêu USD' — **không còn bị chặn bởi miền giá đã học**."

> "Kết quả: **XGBoost với K = 21 đạt MAE 18,0039** — **thấp hơn Naive 18,0381** — và đúng
> chiều **55,95%**, so với 50% của đoán ngẫu nhiên. Random Forest ΔClose cũng về sát Naive
> với 18,2533."

> (dừng) "Nhưng nhóm muốn nói thẳng thắn: **mức cải thiện chỉ khoảng 0,034 USD MAE**. Đủ để
> kết luận rằng **chẩn đoán về giới hạn ngoại suy là đúng** và cách đặt mục tiêu quan trọng
> hơn việc đổi thuật toán — nhưng **chưa đủ để dùng giao dịch thật**."

**Chuyển:** "Vậy tổng kết lại, nhóm trả lời ba câu hỏi thế nào?"

---

### Slide 10 — Kết luận, hạn chế, hướng phát triển (1:00)

**Mục tiêu:** chốt 3 câu trả lời trong 20 giây, rồi hạ giọng sang hạn chế để thể hiện sự trung thực.

> (chỉ 3 thẻ) "**CH1**: dự báo trực tiếp mức giá — **không vượt Naive**, kém 20,2 lần;
> nhưng dự báo ΔClose thì **vượt**, 18,0039 so với 18,0381.
> **CH2**: thêm biến vĩ mô **không giúp** — 25 biến cho Test MAE 22,05, còn 15 biến cho
> 18,53, tốt hơn 16%. CH3: **top 3 đặc trưng chiếm 98,6%** độ quan trọng, còn 10 biến vĩ mô
> chỉ 0,11%."

> "Nhóm cũng ý thức rõ **bốn hạn chế**: chỉ dự báo **một bước**; **chỉ chia Test một lần**,
> chưa rolling-origin theo từng năm; chỉ dùng **dữ liệu ngày**, bỏ lỡ biến động trong phiên
> và tin tức; và mới kiểm chứng trên **một thị trường** là vàng COMEX."

> "Bốn hướng phát triển theo thứ tự ưu tiên là: **rolling-origin backtest** để có 10 điểm
> dữ liệu thay vì 1; **mô hình hoá Δ kèm sai số** (thêm quantile); **đặc trưng tần suất
> cao**; và **triển khai có giám sát trôi dữ liệu**."

> (chốt) "Thông điệp cuối cùng của nhóm: **một kết quả âm tính vẫn là khoa học, nếu chỉ ra
> được nguyên nhân và cách khắc phục.** Sản phẩm kèm theo gồm pipeline tái lập bảng Test
> trong khoảng 5 giây, dashboard FastAPI và 16 test tự động. Em xin cảm ơn thầy cô và sẵn
> sàng nhận câu hỏi."

---

## 4. Bộ câu hỏi phản biện & trả lời mẫu

| Câu hỏi | Trả lời (dùng đúng số này) |
|---|---|
| "Sao quét 500 K mà nhanh vậy?" | Cây thứ 1..k của `RandomForest(n_estimators=500, random_state=42)` **chính là** rừng k cây; XGBoost dùng `iteration_range=(0, k)`. Sai khác < 1e-13, có test khẳng định. |
| "Kết quả cây tệ vậy có phải do chọn K sai?" | Không. RF phẳng từ K = 50 (~32 USD); trên Test **mọi K** của cả hai mô hình đều kém Naive. Nguyên nhân là giới hạn ngoại suy. |
| "Có rò rỉ dữ liệu không?" | Không. Chia theo thời gian, không shuffle; căn theo ngày **giá mục tiêu**; imputer/scaler chỉ fit trên Train; 16 pytest kiểm chứng. |
| "Naive thắng thì mô hình học máy vô dụng à?" | Với dự báo 1 phiên thì đúng là Naive rất mạnh. Giá trị của mô hình nằm ở **giải thích nguyên nhân**: ΔClose giúp vượt Naive, và giới hạn ngoại suy là kết luận có ích cho mọi bài dùng cây. |
| "Sao ΔClose chỉ hơn có 0,034 USD?" | Vì tín hiệu 1 phiên rất yếu. Điều nhóm chứng minh là **quan hệ nhân quả giữa cách đặt mục tiêu và sai số**, không phải một mô hình giao dịch. |
| "VIF 17.593,1 là lỗi?" | Không. sma_5 gần như trùng close_lag ngắn hạn nên đa cộng tuyến cực cao; đây là lý do nhóm **không diễn giải hệ số OLS như tác động nhân quả**. |
| "R² âm nghĩa là gì?" | Mô hình còn tệ hơn cả việc lấy **trung bình** của chính nó để dự báo. Với RF: R² = −2,5165. Đây là hệ quả trực tiếp của việc dự báo phẳng ~2.040 USD. |
| "Sản phẩm demo được không?" | Có: `gold_predict.py` in đúng bảng Test trong ~5 giây; dashboard FastAPI có 7 API; `pytest` 16 test pass. |

Chi tiết thêm (bộ câu hỏi dài, phương án khi sự cố máy chiếu): [`HUONG_DAN_DEMO.md`](HUONG_DAN_DEMO.md).

---

## 5. Câu chốt cho từng chuyển slide (nếu bị quên)

1. → 2: "Vì sao đây là bài toán khó."
2. → 3: "Muốn trả lời phải phát biểu bài toán thật chặt."
3. → 4: "Dữ liệu đã sạch, giờ là thiết lập thực nghiệm."
4. → 5: "Vậy K nào được chọn?"
5. → 6: "Trên Test kém bao nhiêu?"
6. → 7: "Tại sao cây bị chặn?"
7. → 8: "Còn biến vĩ mô và độ quan trọng thì sao?"
8. → 9: "Nếu chẩn đoán đúng thì đổi mục tiêu phải sửa được."
9. → 10: "Tổng kết lại ba câu hỏi."

---

## 6. Bản rút gọn 7 phút (nếu hội đồng giới hạn thời gian)

| Slide | Giữ lại | Bỏ / nói một câu |
|---|---|---|
| 1 | 15 giây tự giới thiệu + quy mô dữ liệu | chi tiết nguồn |
| 2 | nhảy thẳng vào 3 câu hỏi | bỏ phần "tài sản trú ẩn" |
| 3 | công thức + "chỉ biết dữ liệu ≤ t" + 16 test | bỏ danh sách 25 đặc trưng |
| 4 | 3 tập + "Test chỉ dùng một lần" | bỏ bảng siêu tham số |
| 5 | K = 11 / 165 + "RF phẳng từ K = 50" | bỏ bảng K chi tiết |
| 6 | **giữ nguyên** (60 giây) | — |
| 7 | **giữ nguyên** (75 giây): 2.079,82 vs 2.800,80 và 211/252 | bỏ phần OOB nếu gấp |
| 8 | CH2: 22,05 → 18,53; CH3: top 3 = 98,6% | bỏ bảng ablation, chỉ nói kết quả |
| 9 | XGB 18,0039 < 18,0381 | bỏ bảng ΔClose, chỉ nêu 1 số |
| 10 | 3 câu trả lời + 4 hướng phát triển | bỏ phần sản phẩm |

**Câu chốt khi bị cắt:** "Nếu chỉ nhớ một điều: mô hình cây dự báo mức giá bị chặn bởi đỉnh
đã học — 2.079,82 so với 2.800,80 USD — và đổi sang dự báo mức thay đổi thì khắc phục được."

---

## 7. Những lỗi cần tránh khi nói

- **Không nói "mô hình tốt"** khi MAE còn 364 USD; luôn kèm bối cảnh "kém Naive 20,2 lần".
- **Không đọc R² âm thành "mô hình dở toàn tập"** — phải nói rõ nó vẫn học tốt trong miền
  học (OOB MAE 12,11 USD); vấn đề là **ngoại suy**.
- **Không nói "biến vĩ mô vô dụng"** — nói chính xác: *không cải thiện dự báo 1 phiên*.
- **Không lẫn ΔClose với mức giá**: 18,0039 USD là MAE của **mức thay đổi**, không phải giá vàng.
- **Không hứa hẹn đầu tư**: luôn kết bằng "công cụ nghiên cứu học thuật, không phải khuyến
  nghị đầu tư".
- Số dùng định dạng Việt Nam: **18,038** (phẩy thập phân), **2.514** (chấm ngăn nghìn).
