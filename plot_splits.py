"""Thống kê mô tả và trực quan hóa dữ liệu giá vàng GC=F (2015–2024).

Tái tạo toàn bộ biểu đồ khảo sát dữ liệu ở thư mục gốc:
  - gold_train_plot.png              (xu hướng tập Train 2015–2021)
  - gold_val_plot.png                (xu hướng tập Validation 2022–2023)
  - gold_test_plot.png               (xu hướng tập Test 2024)
  - gold_comprehensive_analysis.png  (4 biểu đồ tổng quan & tương quan 2015–2024)

Chạy:
    python plot_splits.py                      # đọc/ghi tại thư mục hiện tại
    python plot_splits.py --dir . --dpi 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args():
    ap = argparse.ArgumentParser(description="Vẽ biểu đồ khảo sát dữ liệu & 4 tập chia giá vàng GC=F")
    ap.add_argument("--dir", default=".", help="Thư mục chứa CSV đầu vào và PNG đầu ra")
    ap.add_argument("--source", default="yfinance", help="Hậu tố nguồn dữ liệu (yfinance/stooq/local)")
    ap.add_argument("--dpi", type=int, default=300)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.dir)
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    def load(name: str) -> pd.DataFrame:
        path = data_dir / name.format(src=args.source)
        if not path.exists():
            raise FileNotFoundError(
                f"Thiếu {path}. Chạy `python Scripts.py` để tạo dataset trước."
            )
        return pd.read_csv(path, parse_dates=["Date"])

    full_df = load("gold_forecast_dataset_{src}.csv")
    train_df = load("gold_train_{src}.csv")
    val_df = load("gold_val_{src}.csv")
    test_df = load("gold_test_{src}.csv")

    splits = [
        (
            "1. TẬP HUẤN LUYỆN (GOLD_TRAIN_YFINANCE.CSV) - (2015 - 2021)",
            train_df,
            "Train Gold Close",
            "#1f77b4",
            "Xu hướng Giá Vàng - Tập Huấn Luyện (2015 - 2021)",
            "gold_train_plot.png",
        ),
        (
            "2. TẬP KIỂM ĐỊNH (GOLD_VAL_YFINANCE.CSV) - (2022 - 2023)",
            val_df,
            "Val Gold Close",
            "#ff7f0e",
            "Xu hướng Giá Vàng - Tập Kiểm Định (2022 - 2023)",
            "gold_val_plot.png",
        ),
        (
            "3. TẬP KIỂM TRA (GOLD_TEST_YFINANCE.CSV) - (NĂM 2024)",
            test_df,
            "Test Gold Close",
            "#2ca02c",
            "Xu hướng Giá Vàng - Tập Kiểm Tra (Năm 2024)",
            "gold_test_plot.png",
        ),
    ]

    for heading, df_part, label, color, title, filename in splits:
        print("=" * 72)
        print(heading)
        print("=" * 72)
        print(df_part.drop(columns=["Date"]).describe().round(2))
        print()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_part["Date"], df_part["Gold_Close"], color=color, linewidth=1.5, label=label)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Năm")
        ax.set_ylabel("Giá (USD/oz)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(data_dir / filename, dpi=args.dpi)
        plt.close(fig)

    # Biểu đồ tổng hợp 4 khung (EDA 2015–2024)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    axes[0, 0].plot(full_df["Date"], full_df["Gold_Close"], color="#d49b27", linewidth=1.4, label="Gold Close")
    axes[0, 0].set_title("1. Xu hướng Biến động Giá Vàng (2015 - 2024)", fontweight="bold")
    axes[0, 0].set_xlabel("Năm")
    axes[0, 0].set_ylabel("Giá (USD/oz)")
    axes[0, 0].legend()

    num_cols = [c for c in full_df.columns if c != "Date"]
    corr = full_df[num_cols].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=axes[0, 1],
        annot_kws={"size": 8},
    )
    axes[0, 1].set_title("2. Ma trận Tương quan Các Chỉ số Tài chính", fontweight="bold")

    axes[1, 0].scatter(
        full_df["Silver_Close"], full_df["Gold_Close"],
        alpha=0.35, s=12, color="#6c757d"
    )
    axes[1, 0].set_title("3. Tương quan: Giá Vàng vs. Giá Bạc", fontweight="bold")
    axes[1, 0].set_xlabel("Giá Bạc (USD)")
    axes[1, 0].set_ylabel("Giá Vàng (USD)")

    axes[1, 1].scatter(
        full_df["SP500_Close"], full_df["Gold_Close"],
        alpha=0.35, s=12, color="#4dabf7"
    )
    axes[1, 1].set_title("4. Tương quan: Giá Vàng vs. S&P 500", fontweight="bold")
    axes[1, 1].set_xlabel("S&P 500 Close")
    axes[1, 1].set_ylabel("Giá Vàng (USD)")

    fig.tight_layout()
    fig.savefig(data_dir / "gold_comprehensive_analysis.png", dpi=200)
    plt.close(fig)

    print(f"Đã xuất thành công vào {data_dir.resolve()}:")
    for name in ("gold_train_plot.png", "gold_val_plot.png", "gold_test_plot.png",
                 "gold_comprehensive_analysis.png"):
        print(f"  - {data_dir / name}")


if __name__ == "__main__":
    main()
