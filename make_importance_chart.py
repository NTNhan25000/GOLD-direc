"""Vẽ lại biểu đồ độ quan trọng đặc trưng cỡ slide (dễ đọc khi chiếu).

Ảnh gốc ở gold_model_output/ vẽ 15 đặc trưng trong khung rất nhỏ nên khi chèn
vào slide 16:9 thì chữ không đọc được. Script này đọc đúng hai file CSV đã có
(không tính lại mô hình) và vẽ bản gọn: top 7 + phần còn lại.

Chạy:
    python make_importance_chart.py     → slide_assets/importance.png

Toàn bộ số liệu vẫn lấy từ gold_model_output/*.csv nên chạy lại pipeline thì
chỉ cần chạy lại script này (make_slides.py không tự gọi).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "slide_assets" / "importance.png"
N = 6  # số đặc trưng hiển thị riêng; phần còn lại gộp thành "Còn lại"

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 9.0


def compose(df: pd.DataFrame, col: str) -> pd.DataFrame:
    top = df.head(N)[["feature", col]].copy()
    rest = df.iloc[N:][col].sum()
    total = df[col].sum()
    top.loc[len(top)] = [f"Còn lại ({len(df) - N} biến)", rest]
    top["pct"] = top[col] / total * 100.0
    return top.iloc[::-1]  # vẽ từ dưới lên


rf = pd.read_csv(ROOT / "gold_model_output" / "rf_feature_importance.csv")
xg = pd.read_csv(ROOT / "gold_model_output" / "xgb_feature_importance.csv").sort_values(
    "gain_pct", ascending=False)

# figsize nhỏ nhưng dpi cao: khi thu vào khung ~6,1 × 2,85 in trên slide thì
# chữ vẫn còn cỡ ~9 pt (nếu vẽ ở khung lớn rồi thu nhỏ, chữ sẽ tí xíu khi chiếu).
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.95), dpi=300,
                         gridspec_kw={"wspace": 0.62})
for ax, df, col, color, ttl in [
    (axes[0], rf, "MDI_pct", "#0B7A52", "Random Forest — MDI (%)"),
    (axes[1], xg, "gain_pct", "#BE3030", "XGBoost — Gain (%)"),
]:
    d = compose(df, col)
    bars = ax.barh(d["feature"], d["pct"], color=color, height=0.72)
    bars[0].set_color("#9AA6B8")  # thanh dưới cùng = phần còn lại: xám
    for b, v in zip(bars, d["pct"]):
        ax.text(b.get_width() + 1.2, b.get_y() + b.get_height() / 2,
                f"{v:.1f}", va="center", ha="left", fontsize=8, color="#0F1726")
    ax.set_title(ttl, fontsize=10, fontweight="bold", color="#0F1726", pad=6)
    ax.set_xlim(0, max(d["pct"]) * 1.30)
    ax.set_xlabel("% đóng góp", fontsize=8, color="#515E74")
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#D3DDEA")
    ax.tick_params(axis="both", labelsize=8, colors="#0F1726", length=0)
    ax.tick_params(axis="y", pad=2)

fig.patch.set_facecolor("white")
for ax in axes:
    ax.set_facecolor("white")
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.06, facecolor="white")
print(f"Đã lưu {OUT.relative_to(ROOT)}")
