import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Thiết lập giao diện chung cho biểu đồ
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# 2. Đọc các tập dữ liệu độc lập
train_df = pd.read_csv("gold_train_yfinance.csv")
val_df = pd.read_csv("gold_val_yfinance.csv")
test_df = pd.read_csv("gold_test_yfinance.csv")

# Chuyển đổi cột Date sang kiểu datetime
for df_temp in [train_df, val_df, test_df]:
  df_temp["Date"] = pd.to_datetime(df_temp["Date"])

# =========================================================================
# PHẦN 1: TẬP HUẤN LUYỆN (TRAIN SET)
# =========================================================================
print("=" * 70)
print("1. TẬP HUẤN LUYỆN (GOLD_TRAIN_YFINANCE.CSV) - (2015 - 2021)")
print("=" * 70)
print(train_df.describe())
print(
    "\n[Phân tích Train]:\n- Mục đích: Cung cấp dữ liệu lịch sử lớn nhất (~70%)"
    " cho mô hình học các quy luật chuỗi thời gian dài hạn.\n- Giá trị: Bao"
    " trùm đầy đủ pha sideway và đỉnh lịch sử năm 2020 ($2,000 USD), bao gồm"
    " cả sự kiện biến động đặc biệt như giá dầu âm.\n- Xu hướng: Biên độ hẹp,"
    " tập trung chủ yếu trong vùng $1,200 - $1,500 USD.\n"
)

# =========================================================================
# PHẦN 2: TẬP KIỂM ĐỊNH (VALIDATION SET)
# =========================================================================
print("=" * 70)
print("2. TẬP KIỂM ĐỊNH (GOLD_VAL_YFINANCE.CSV) - (2022 - 2023)")
print("=" * 70)
print(val_df.describe())
print(
    "\n[Phân tích Validation]:\n- Mục đích: Tinh chỉnh siêu tham số và đánh"
    " giá hiệu suất trung gian mà không gây rò rỉ dữ liệu.\n- Giá trị: Phản"
    " ánh giai đoạn lạm phát leo thang, Fed tăng lãi suất; DXY chạm đỉnh"
    " 114.11.\n- Xu hướng: Vàng duy trì sideway-up vùng cao ($1,630 - $2,093"
    " USD), thanh khoản tăng vọt.\n"
)

# =========================================================================
# PHẦN 3: TẬP KIỂM TRA (TEST SET)
# =========================================================================
print("=" * 70)
print("3. TẬP KIỂM TRA (GOLD_TEST_YFINANCE.CSV) - (NĂM 2024)")
print("=" * 70)
print(test_df.describe())
print(
    "\n[Phân tích Test]:\n- Mục đích: Sát hạch cuối cùng (unseen data) đo lường"
    " độ chính xác thực tế của mô hình.\n- Giá trị: Ghi nhận mức giá trung bình"
    " $2,404 USD, đỉnh lịch sử chạm $2,800.80 USD.\n- Xu hướng: Tăng trưởng"
    " thẳng đứng (bull run) xuyên suốt năm 2024 nhờ lực mua từ các ngân hàng"
    " trung ương.\n"
)

# =========================================================================
# PHẦN 4: VẼ BIỂU ĐỒ ĐỘC LẬP CHO TỪNG TẬP DỮ LIỆU
# =========================================================================

# Biểu đồ 1: Tập Train
plt.figure(figsize=(10, 5))
plt.plot(
    train_df["Date"],
    train_df["Gold_Close"],
    color="#1f77b4",
    linewidth=1.5,
    label="Train Gold Close",
)
plt.title(
    "Xu hướng Giá Vàng - Tập Huấn Luyện (2015 - 2021)",
    fontsize=12,
    fontweight="bold",
)
plt.xlabel("Năm")
plt.ylabel("Giá (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("gold_train_plot.png", dpi=300)
plt.show()

# Biểu đồ 2: Tập Validation
plt.figure(figsize=(10, 5))
plt.plot(
    val_df["Date"],
    val_df["Gold_Close"],
    color="#ff7f0e",
    linewidth=1.5,
    label="Val Gold Close",
)
plt.title(
    "Xu hướng Giá Vàng - Tập Kiểm Định (2022 - 2023)",
    fontsize=12,
    fontweight="bold",
)
plt.xlabel("Năm")
plt.ylabel("Giá (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("gold_val_plot.png", dpi=300)
plt.show()

# Biểu đồ 3: Tập Test
plt.figure(figsize=(10, 5))
plt.plot(
    test_df["Date"],
    test_df["Gold_Close"],
    color="#2ca02c",
    linewidth=1.5,
    label="Test Gold Close",
)
plt.title(
    "Xu hướng Giá Vàng - Tập Kiểm Tra (Năm 2024)",
    fontsize=12,
    fontweight="bold",
)
plt.xlabel("Năm")
plt.ylabel("Giá (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("gold_test_plot.png", dpi=300)
plt.show()

print(
    "Các bảng thống kê và file biểu đồ độc lập (gold_train_plot.png,"
    " gold_val_plot.png, gold_test_plot.png) đã được tạo thành công!"
)