"""Dựng bộ slide thuyết trình (.pptx) cho đề tài Nhóm 8 — bản rút gọn 10 slide.

Chạy:
    python make_slides.py
    → slides_thuyet_trinh_nhom8.pptx   (10 slide, nền trắng)

Toàn bộ số liệu lấy trực tiếp từ gold_model_output/*.csv và extra_diagnostics.json,
nên nếu chạy lại pipeline thì chỉ cần chạy lại script này là slide cập nhật theo.
Kịch bản lời thoại từng slide: KICH_BAN_THUYET_TRINH.md
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "slide_assets"
OUT = ROOT / "slides_thuyet_trinh_nhom8.pptx"

# --------------------------------------------------------------------- Thiết kế
# Nền TRẮNG: chữ đậm, panel xám rất nhạt, màu nhấn phải đủ tối để đọc trên giấy trắng.
BG = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x0F, 0x17, 0x26)          # chữ chính
PANEL = RGBColor(0xF5, 0xF7, 0xFB)        # nền panel
PANEL2 = RGBColor(0xE7, 0xED, 0xF6)       # nền panel đậm hơn (header bảng, hộp số)
LINE = RGBColor(0xD3, 0xDD, 0xEA)         # viền mảnh
MUTED = RGBColor(0x51, 0x5E, 0x74)        # chữ phụ
GREY = RGBColor(0x7C, 0x88, 0x9C)

GOLD = RGBColor(0xAE, 0x78, 0x00)         # vàng đậm — chữ/đường nhấn trên nền trắng
GOLD_SOFT = RGBColor(0xFD, 0xF2, 0xD2)    # nền highlight
GOLD_BRIGHT = RGBColor(0xF0, 0xB4, 0x29)  # chỉ dùng cho khối trang trí
GREEN = RGBColor(0x0B, 0x7A, 0x52)
RED = RGBColor(0xBE, 0x30, 0x30)
BLUE = RGBColor(0x18, 0x5C, 0xBC)
ORANGE = RGBColor(0xB8, 0x57, 0x04)
PURPLE = RGBColor(0x5B, 0x39, 0xBC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "Calibri"
W, H = Inches(13.333), Inches(7.5)          # 16:9

prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]


# ------------------------------------------------------------------- Tiện ích
def slide() -> object:
    s = prs.slides.add_slide(BLANK)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG
    bg.line.fill.background()
    bg.shadow.inherit = False
    return s


def txt(s, text, l, t, w, h, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT,
        anchor=MSO_ANCHOR.TOP, spacing=1.0, italic=False):
    box = s.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    lines = text.split("\n") if isinstance(text, str) else list(text)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.bold = bold
        run.font.italic = italic
        run.font.name = FONT
    return box


TITLE_W_IN = 11.9


def fit_font_size(text: str, max_w_in: float, start: float = 30.0, floor: float = 18.0) -> float:
    """Giảm cỡ chữ tới khi tiêu đề vừa MỘT dòng.

    Đo bằng DejaVu (rộng hơn Calibri ~8–12%) rồi nhân biên an toàn 0,88 để
    chắc chắn vừa cả khi mở bằng PowerPoint với Calibri.
    """
    from PIL import ImageFont

    f_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    budget = max_w_in * 0.88
    size = start
    while size > floor:
        f = ImageFont.truetype(f_path, int(size * 4))
        if f.getlength(text) / 4 / 72 <= budget:
            break
        size -= 0.5
    return size


def title(s, text, kicker=None):
    """Tiêu đề slide là MỘT CÂU KHẲNG ĐỊNH, luôn vừa 1 dòng."""
    if kicker:
        txt(s, kicker.upper(), Inches(0.7), Inches(0.34), Inches(11.9), Inches(0.3),
            size=11, color=GOLD, bold=True)
        ty = Inches(0.68)
    else:
        ty = Inches(0.5)
    size = fit_font_size(text, TITLE_W_IN - 0.1)
    txt(s, text, Inches(0.7), ty, Inches(11.9), Inches(0.7), size=size, bold=True, color=INK)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.7), ty + Inches(0.66),
                             Inches(1.5), Pt(4))
    bar.fill.solid()
    bar.fill.fore_color.rgb = GOLD
    bar.line.fill.background()
    bar.shadow.inherit = False


def panel(s, l, t, w, h, fill=PANEL, line=None):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line:
        sh.line.color.rgb = line
        sh.line.width = Pt(1.25)
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    sh.adjustments[0] = 0.06
    return sh


def picture(s, name, l, t, max_w, max_h):
    """Chèn ảnh, giữ tỉ lệ, căn giữa trong khung."""
    path = ASSETS / name
    with Image.open(path) as im:
        iw, ih = im.size
    ratio = iw / ih
    w = max_w
    h = int(w / ratio)
    if h > max_h:
        h = max_h
        w = int(h * ratio)
    left = l + int((max_w - w) / 2)
    top = t + int((max_h - h) / 2)
    s.shapes.add_picture(str(path), Emu(left), Emu(top), Emu(w), Emu(h))


def kpi(s, l, t, w, value, label, color=GOLD, vsize=40):
    panel(s, l, t, w, Inches(1.55), PANEL2)
    txt(s, value, l, t + Inches(0.22), w, Inches(0.7), size=vsize, color=color,
        bold=True, align=PP_ALIGN.CENTER)
    txt(s, label, l + Inches(0.15), t + Inches(1.0), w - Inches(0.3), Inches(0.5),
        size=11, color=MUTED, align=PP_ALIGN.CENTER, spacing=0.95)


def chip(s, l, t, w, value, label, color=GOLD, vsize=24, h=Inches(1.0)):
    """Ô số gọn: số lớn + nhãn nhỏ, dùng cho dải KPI."""
    panel(s, l, t, w, h, PANEL2)
    txt(s, value, l, t + Inches(0.13), w, Inches(0.5), size=vsize, color=color,
        bold=True, align=PP_ALIGN.CENTER)
    txt(s, label, l + Inches(0.16), t + Inches(0.63), w - Inches(0.32), Inches(0.3),
        size=10, color=MUTED, align=PP_ALIGN.CENTER, spacing=0.95)


def table(s, headers, rows, l, t, w, col_w=None, size=12, head_size=10.5,
          highlight_rows=(), highlight_color=GOLD, row_h=Inches(0.42),
          left_cols=()):
    nr, nc = len(rows) + 1, len(headers)
    shape = s.shapes.add_table(nr, nc, l, t, w, row_h * nr)
    tbl = shape.table
    tbl.first_row = True
    if col_w:
        total = sum(col_w)
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = Emu(int(w * cw / total))
    for c, hd in enumerate(headers):
        cell = tbl.cell(0, c)
        cell.text = ""
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if (c == 0 or c in left_cols) else PP_ALIGN.RIGHT
        run = p.add_run()
        run.text = str(hd)
        run.font.size = Pt(head_size)
        run.font.bold = True
        run.font.color.rgb = MUTED
        run.font.name = FONT
        cell.fill.solid()
        cell.fill.fore_color.rgb = PANEL2
        cell.margin_left = cell.margin_right = Inches(0.08)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    for r, row in enumerate(rows, start=1):
        hl = (r - 1) in highlight_rows
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.RIGHT if c else PP_ALIGN.LEFT
            lines = str(val).split("\n")
            for li, line in enumerate(lines):
                pp = p if li == 0 else cell.text_frame.add_paragraph()
                pp.alignment = PP_ALIGN.LEFT if (c == 0 or c in left_cols) else PP_ALIGN.RIGHT
                run = pp.add_run()
                run.text = line
                run.font.size = Pt(size)
                run.font.name = FONT
                run.font.bold = hl
                run.font.color.rgb = highlight_color if hl else INK
            cell.fill.solid()
            cell.fill.fore_color.rgb = GOLD_SOFT if hl else PANEL
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    return tbl


def bullets(s, items, l, t, w, size=17, gap=Inches(0.58), color=MUTED, dot=GOLD):
    for i, (head, body) in enumerate(items):
        y = t + i * gap
        d = s.shapes.add_shape(MSO_SHAPE.OVAL, l, y + Inches(0.09), Inches(0.1), Inches(0.1))
        d.fill.solid()
        d.fill.fore_color.rgb = dot
        d.line.fill.background()
        d.shadow.inherit = False
        box = txt(s, "", l + Inches(0.24), y, w - Inches(0.24), gap, size=size)
        tf = box.text_frame
        p = tf.paragraphs[0]
        r1 = p.add_run()
        r1.text = head + "  "
        r1.font.size = Pt(size)
        r1.font.bold = True
        r1.font.color.rgb = INK
        r1.font.name = FONT
        r2 = p.add_run()
        r2.text = body
        r2.font.size = Pt(size)
        r2.font.color.rgb = color
        r2.font.name = FONT


def footer(s, n, total):
    txt(s, "Nhóm 8 · Dự báo giá vàng GC=F", Inches(0.7), H - Inches(0.52),
        Inches(6), Inches(0.3), size=10, color=GREY)
    txt(s, f"{n} / {total}", W - Inches(1.6), H - Inches(0.52), Inches(0.9),
        Inches(0.3), size=10, color=GREY, align=PP_ALIGN.RIGHT)


# ---------------------------------------------------------------- Dữ liệu thật
m = pd.read_csv(ROOT / "gold_model_output" / "test_metrics.csv").set_index("model")
kv = pd.read_csv(ROOT / "gold_model_output" / "k_validation_test.csv")
ab = pd.read_csv(ROOT / "gold_model_output" / "ablation_macro_study.csv")
dt = pd.read_csv(ROOT / "gold_model_output" / "delta_target_experiment.csv").set_index("approach")
qt = pd.read_csv(ROOT / "gold_model_output" / "quarterly_test_metrics.csv")
rfimp = pd.read_csv(ROOT / "gold_model_output" / "rf_feature_importance.csv")
xgimp = pd.read_csv(ROOT / "gold_model_output" / "xgb_feature_importance.csv").sort_values(
    "gain_pct", ascending=False)
diag = json.loads((ROOT / "gold_model_output" / "extra_diagnostics.json").read_text(encoding="utf-8"))

TOTAL = 10

MACRO = ["DXY_Close", "DXY_Close_return_1d", "US10Y_Close", "US10Y_Close_return_1d",
         "SP500_Close", "SP500_Close_return_1d", "WTI_Close", "WTI_Close_return_1d",
         "Silver_Close", "Silver_Close_return_1d"]


def vn(x, d=2):
    """Định dạng số kiểu Việt Nam: chấm ngăn nghìn, phẩy thập phân (2.079,82)."""
    return f"{x:,.{d}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def f2(x, d=2):
    return vn(x, d)


# ============================================================== SLIDE 1 — Bìa
s = slide()
band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.32), H)
band.fill.solid()
band.fill.fore_color.rgb = GOLD_BRIGHT
band.line.fill.background()
band.shadow.inherit = False

txt(s, "ĐỒ ÁN MÔN TRÍ TUỆ NHÂN TẠO", Inches(1.2), Inches(1.45), Inches(11), Inches(0.4),
    size=13, color=GOLD, bold=True)
txt(s, "Dự báo giá vàng\nphiên kế tiếp bằng học máy", Inches(1.2), Inches(1.95),
    Inches(11), Inches(1.9), size=46, bold=True, color=INK, spacing=1.05)
txt(s, "So sánh Naive · Linear Regression · Random Forest · XGBoost\ntrên hợp đồng tương lai vàng COMEX GC=F (USD/troy ounce), 2015–2024",
    Inches(1.2), Inches(4.05), Inches(11), Inches(1.0), size=16, color=MUTED, spacing=1.35)
line = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.2), Inches(5.3), Inches(2.2), Pt(3))
line.fill.solid()
line.fill.fore_color.rgb = GOLD
line.line.fill.background()
line.shadow.inherit = False
txt(s, "Nhóm 8", Inches(1.2), Inches(5.55), Inches(5), Inches(0.5), size=20, bold=True,
    color=INK)
txt(s, "Dữ liệu: 2.514 phiên giao dịch · 6 nguồn Yahoo Finance · Test 2024 (252 phiên)",
    Inches(1.2), Inches(6.05), Inches(11), Inches(0.4), size=12.5, color=MUTED)

# ================================= SLIDE 2 — Đặt vấn đề & câu hỏi nghiên cứu
s = slide()
title(s, "Vàng tăng 2,7 lần trong 10 năm, nhưng dự báo đúng phiên kế tiếp vẫn rất khó",
      kicker="Đặt vấn đề & mục tiêu")
picture(s, "trend.png", Inches(0.55), Inches(1.72), Inches(7.2), Inches(3.7))

panel(s, Inches(7.95), Inches(1.78), Inches(4.68), Inches(3.55), PANEL, line=LINE)
bullets(s, [
    ("Tài sản trú ẩn", "vàng hút vốn khi lạm phát và bất định địa chính trị tăng"),
    ("Bài toán", "hồi quy chuỗi thời gian, dự báo trước 1 phiên"),
    ("Khó ở đâu", "nhiễu cao, chuỗi không dừng, tín hiệu yếu hơn xu hướng"),
    ("Yêu cầu", "so sánh công bằng với baseline, đánh giá ngoài mẫu"),
], Inches(8.2), Inches(2.0), Inches(4.2), size=13, gap=Inches(0.78))

qs = [
    ("CH1", "Mô hình cây có vượt Naive và Linear Regression?", BLUE),
    ("CH2", "Thêm 10 biến vĩ mô liên thị trường có giúp ích?", ORANGE),
    ("CH3", "Đặc trưng nào đóng góp nhiều nhất?", PURPLE),
]
for i, (tag, body, col) in enumerate(qs):
    x = Inches(0.7) + i * Inches(4.05)
    panel(s, x, Inches(5.55), Inches(3.85), Inches(1.18), PANEL, line=col)
    txt(s, tag, x + Inches(0.26), Inches(5.7), Inches(1.0), Inches(0.3), size=11,
        color=col, bold=True)
    txt(s, body, x + Inches(0.26), Inches(6.0), Inches(3.35), Inches(0.55), size=12.5,
        bold=True, spacing=1.2)
footer(s, 2, TOTAL)

# ================================== SLIDE 3 — Phát biểu bài toán & dữ liệu
s = slide()
title(s, "Cuối phiên t chỉ biết dữ liệu ≤ t, mục tiêu là giá đóng cửa phiên t + 1",
      kicker="Phát biểu bài toán & dữ liệu")

panel(s, Inches(0.7), Inches(1.72), Inches(11.93), Inches(1.12), PANEL2)
txt(s, "ŷ(t + 1) = f( X(t) )        với        y(t) = Close(t + 1)",
    Inches(1.0), Inches(1.9), Inches(11.3), Inches(0.5), size=24, bold=True, color=GOLD,
    align=PP_ALIGN.CENTER)
txt(s, "X(t): 25 đặc trưng chỉ tính từ dữ liệu ≤ t · 16 test tự động kiểm chứng chống rò rỉ",
    Inches(1.0), Inches(2.42), Inches(11.3), Inches(0.3), size=11.5, color=MUTED,
    align=PP_ALIGN.CENTER)

y_axis = Inches(3.55)
ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.2), y_axis, Inches(10.9), Pt(2))
ln.fill.solid()
ln.fill.fore_color.rgb = LINE
ln.line.fill.background()
ln.shadow.inherit = False

for cx, label, sub, col in [
    (Inches(2.0), "… phiên t − 2", "quá khứ", GREY),
    (Inches(4.6), "phiên t − 1", "quá khứ", GREY),
    (Inches(7.2), "phiên t", "thời điểm phát tín hiệu", GOLD),
    (Inches(10.2), "phiên t + 1", "cần dự báo", GREEN),
]:
    d = s.shapes.add_shape(MSO_SHAPE.OVAL, cx - Inches(0.13), y_axis - Inches(0.13),
                           Inches(0.26), Inches(0.26))
    d.fill.solid()
    d.fill.fore_color.rgb = col
    d.line.fill.background()
    d.shadow.inherit = False
    txt(s, label, cx - Inches(1.2), y_axis - Inches(0.62), Inches(2.4), Inches(0.35),
        size=13, bold=True, align=PP_ALIGN.CENTER, color=INK)
    txt(s, sub, cx - Inches(1.2), y_axis + Inches(0.26), Inches(2.4), Inches(0.35),
        size=10.5, color=MUTED, align=PP_ALIGN.CENTER)

chip(s, Inches(0.7), Inches(4.4), Inches(2.9), "2.514", "phiên giao dịch, 2015 → 2024",
     GOLD, 25, Inches(1.02))
chip(s, Inches(3.72), Inches(4.4), Inches(2.9), "10 cột · 0 NaN", "OHLCV vàng + 5 chuỗi vĩ mô",
     BLUE, 18, Inches(1.02))
chip(s, Inches(6.74), Inches(4.4), Inches(2.9), "6", "nguồn dữ liệu Yahoo Finance",
     GREEN, 25, Inches(1.02))
chip(s, Inches(9.76), Inches(4.4), Inches(2.87), "1.050,8 – 2.800,8", "USD/oz, biên giá 10 năm",
     PURPLE, 16, Inches(1.02))

groups = [
    ("Giá trễ  (6)", "close_lag 0 · 1 · 2 · 3 · 5 · 10", BLUE),
    ("SMA & biến động  (6)", "SMA 5/10/20 · vol lợi suất 5/10/20", GREEN),
    ("Động lượng  (3)", "RSI-14 · lợi suất 1 phiên · (H−L)/C", ORANGE),
    ("Liên thị trường  (10)", "DXY · US10Y · S&P 500 · WTI · Bạc", PURPLE),
]
for i, (name, detail, col) in enumerate(groups):
    x = Inches(0.7) + i * Inches(3.02)
    panel(s, x, Inches(5.57), Inches(2.87), Inches(1.02), PANEL, line=LINE)
    txt(s, name, x + Inches(0.22), Inches(5.7), Inches(2.5), Inches(0.3), size=11.5,
        bold=True, color=col)
    txt(s, detail, x + Inches(0.22), Inches(6.03), Inches(2.5), Inches(0.4), size=9.5,
        color=MUTED, spacing=1.1)
footer(s, 3, TOTAL)

# ============================================ SLIDE 4 — Thiết lập thực nghiệm
s = slide()
title(s, "Chia theo thời gian, 4 mô hình, chỉ quét K trên Validation",
      kicker="Thiết lập thực nghiệm")
rows = [
    ["Train", "2015-02-03 → 2021-12-31", "1.739", "fit imputer + scaler"],
    ["Validation", "2022-01-03 → 2023-12-29", "502", "chọn K, không dùng để fit"],
    ["Test", "2024-01-02 → 2024-12-31", "252", "chỉ đánh giá cuối cùng"],
]
table(s, ["Tập", "Khoảng ngày của giá mục tiêu", "Số mẫu", "Vai trò"],
      rows, Inches(0.7), Inches(1.75), Inches(6.95),
      col_w=[1.0, 2.5, 0.85, 2.5], size=12, row_h=Inches(0.38), highlight_rows=(2,))

panel(s, Inches(7.95), Inches(1.75), Inches(4.68), Inches(1.52), PANEL, line=LINE)
txt(s, "Chống rò rỉ dữ liệu", Inches(8.2), Inches(1.9), Inches(4.2), Inches(0.3),
    size=11, color=GOLD, bold=True)
txt(s, "· Không shuffle — giữ nguyên thứ tự thời gian\n"
       "· Căn theo ngày giá mục tiêu, không phải ngày dự báo\n"
       "· Imputer & scaler chỉ fit trên Train",
    Inches(8.2), Inches(2.2), Inches(4.2), Inches(1.0), size=10.5, color=MUTED, spacing=1.35)

models = [
    ("Naive Baseline", "không học · ŷ = Close(t)", "mốc cơ sở bắt buộc", GREY),
    ("Linear Regression", "tuyến tính · 25 hệ số", "mốc diễn giải được", BLUE),
    ("Random Forest", "bagging · 11 cây", "phi tuyến, giảm phương sai", GREEN),
    ("XGBoost", "boosting · 165 vòng", "phi tuyến, giảm độ chệch", ORANGE),
]
for i, (name, mech, role, col) in enumerate(models):
    x = Inches(0.7) + i * Inches(3.02)
    panel(s, x, Inches(3.45), Inches(2.87), Inches(1.02), PANEL, line=col)
    txt(s, name, x + Inches(0.2), Inches(3.58), Inches(2.5), Inches(0.3), size=12.5,
        bold=True, color=col)
    txt(s, mech, x + Inches(0.2), Inches(3.92), Inches(2.5), Inches(0.28), size=10,
        color=INK)
    txt(s, role, x + Inches(0.2), Inches(4.17), Inches(2.5), Inches(0.25), size=9.5,
        color=MUTED)

panel(s, Inches(0.7), Inches(4.65), Inches(6.95), Inches(1.35), PANEL)
txt(s, "Siêu tham số giữ cố định", Inches(0.95), Inches(4.78), Inches(6.4), Inches(0.3),
    size=11, color=GOLD, bold=True)
txt(s, "Random Forest:  max_features = 0,8 · min_samples_leaf = 2 · random_state = 42\n"
       "XGBoost:  max_depth = 3 · lr = 0,03 · subsample = 0,8 · colsample = 0,8 · λ = 1",
    Inches(0.95), Inches(5.08), Inches(6.5), Inches(0.8), size=10.5, color=MUTED, spacing=1.4)

panel(s, Inches(7.95), Inches(4.65), Inches(4.68), Inches(1.35), PANEL)
txt(s, "Quét K ∈ 1..500 trên Validation", Inches(8.2), Inches(4.78), Inches(4.2),
    Inches(0.3), size=11, color=GOLD, bold=True)
txt(s, "Chọn K theo Val RMSE thấp nhất — 500 giá trị chỉ mất ~5 giây nhờ trung bình tích lũy trên một rừng 500 cây (test khẳng định tương đương < 1e-13).",
    Inches(8.2), Inches(5.06), Inches(4.2), Inches(0.85), size=10, color=MUTED, spacing=1.3)

panel(s, Inches(0.7), Inches(6.15), Inches(11.93), Inches(0.5), PANEL2)
txt(s, "Test 2024 chỉ được dùng một lần để đánh giá cuối — không tham gia bất kỳ quyết định nào.",
    Inches(0.95), Inches(6.28), Inches(11.5), Inches(0.3), size=12, color=GOLD, bold=True)
footer(s, 4, TOTAL)

# ================================================ SLIDE 5 — Chọn K
s = slide()
title(s, "Quét K trên Validation: Random Forest = 11 cây, XGBoost = 165 vòng",
      kicker="Chọn siêu tham số")
picture(s, "kcurve.png", Inches(0.5), Inches(1.72), Inches(7.3), Inches(3.5))

rows = [
    ["Random Forest", "11", vn(kv[kv.model == "RandomForest"].RMSE.min(), 3), "0,9294"],
    ["XGBoost", "165", vn(kv[kv.model == "XGBoost"].RMSE.min(), 3), "0,8975"],
]
table(s, ["Mô hình", "K", "Val RMSE", "Val R²"], rows,
      Inches(7.95), Inches(1.8), Inches(4.68), col_w=[1.9, 0.7, 1.15, 1.0],
      size=11, head_size=9.5, row_h=Inches(0.32), highlight_rows=(0, 1))

ks = [1, 11, 50, 100, 165, 500]
rows = []
for k in ks:
    rf_row = kv[(kv.model == "RandomForest") & (kv.K_trees == k)]
    xg_row = kv[(kv.model == "XGBoost") & (kv.K_trees == k)]
    rows.append([str(k), vn(rf_row.RMSE.iloc[0], 3), vn(xg_row.RMSE.iloc[0], 3)])
table(s, ["K", "RF — Val RMSE", "XGB — Val RMSE"], rows,
      Inches(7.95), Inches(3.0), Inches(4.68), col_w=[0.8, 2.0, 2.0],
      size=10, head_size=9, row_h=Inches(0.28), highlight_rows=(1, 4))

panel(s, Inches(0.7), Inches(5.55), Inches(5.85), Inches(1.1), PANEL, line=LINE)
txt(s, "Random Forest", Inches(0.95), Inches(5.68), Inches(5.3), Inches(0.28), size=10.5,
    color=GREEN, bold=True)
txt(s, "Đường RMSE gần như phẳng từ K = 50 (~32 USD) → chỉnh K không cứu được mô hình; vấn đề nằm ở bản chất thuật toán.",
    Inches(0.95), Inches(5.95), Inches(5.35), Inches(0.65), size=11, color=MUTED, spacing=1.25)

panel(s, Inches(6.78), Inches(5.55), Inches(5.85), Inches(1.1), PANEL, line=LINE)
txt(s, "XGBoost", Inches(7.03), Inches(5.68), Inches(5.3), Inches(0.28), size=10.5,
    color=ORANGE, bold=True)
txt(s, "Cần 165 vòng vì learning rate thấp (0,03): tại K = 100 RMSE còn 49,78 USD. Trên Test, mọi K đều kém Naive.",
    Inches(7.03), Inches(5.95), Inches(5.35), Inches(0.65), size=11, color=MUTED, spacing=1.25)
footer(s, 5, TOTAL)

# ============================================ SLIDE 6 — Kết quả Test
s = slide()
title(s, "Trên Test 2024: Naive thắng, mô hình cây kém hơn khoảng 20 lần",
      kicker="Kết quả chính")
rows = []
for model in ["Naive", "LinearRegression", "RandomForest", "XGBoost"]:
    r = m.loc[model]
    k = "—" if pd.isna(r.get("K_trees")) else f"{int(r['K_trees'])}"
    rows.append([
        {"Naive": "Naive Baseline", "LinearRegression": "Linear Regression",
         "RandomForest": "Random Forest", "XGBoost": "XGBoost"}[model],
        k, vn(r["MAE"], 3), vn(r["RMSE"], 3),
        vn(r["MAPE_pct"], 3) + "%", vn(r["R2"], 4),
    ])
table(s, ["Mô hình", "K", "MAE (USD/oz)", "RMSE", "MAPE", "R²"], rows,
      Inches(0.7), Inches(1.78), Inches(11.93),
      col_w=[2.6, 0.8, 1.6, 1.5, 1.3, 1.2], size=13, head_size=11,
      row_h=Inches(0.44), highlight_rows=(0,))

chip(s, Inches(0.7), Inches(4.15), Inches(2.87), "18,04", "MAE Naive (USD/oz)", GREEN, 24,
     Inches(1.12))
chip(s, Inches(3.72), Inches(4.15), Inches(2.87), "22,05", "MAE Linear Regression", BLUE, 24,
     Inches(1.12))
chip(s, Inches(6.74), Inches(4.15), Inches(2.87), "364,53", "MAE Random Forest", RED, 24,
     Inches(1.12))
ratio_mae = m.loc["RandomForest", "MAE"] / m.loc["Naive", "MAE"]
chip(s, Inches(9.76), Inches(4.15), Inches(2.87), "×" + vn(ratio_mae, 1), "lần MAE Naive (Random Forest)", RED, 24,
     Inches(1.12))

panel(s, Inches(0.7), Inches(5.45), Inches(11.93), Inches(1.2), PANEL, line=LINE)
txt(s, "Không phải lỗi code — sai số bùng nổ đúng lúc giá vượt đỉnh lịch sử",
    Inches(0.95), Inches(5.58), Inches(11.4), Inches(0.3), size=11, color=GOLD, bold=True)
txt(s, "16 test tự động xác nhận metric khớp báo cáo tới 1e-9.  MAE của Random Forest theo quý: Q1 = 47,39 → Q2 = 308,44 USD, "
       "đúng lúc giá vàng vượt 2.100 USD — cao hơn mọi mức đã thấy khi huấn luyện. Hai đường RF/XGB trên đồ thị nằm ngang quanh "
       "2.040 USD, không bám được giá thực.",
    Inches(0.95), Inches(5.88), Inches(11.45), Inches(0.7), size=11, color=MUTED, spacing=1.3)
footer(s, 6, TOTAL)

# ==================================== SLIDE 7 — Vì sao cây thua
s = slide()
title(s, "Cây thua vì dự báo bị chặn trên bởi đỉnh giá đã học khi huấn luyện",
      kicker="Chẩn đoán nguyên nhân")
picture(s, "forecast.png", Inches(0.5), Inches(1.72), Inches(7.55), Inches(3.4))
txt(s, "Hai đường RF/XGB nằm ngang quanh 2.040 USD trong khi giá thực 2024 đi từ 2.073 lên 2.800 USD.",
    Inches(0.55), Inches(5.22), Inches(7.45), Inches(0.35), size=10.5, color=MUTED, italic=True)

chip(s, Inches(8.15), Inches(1.72), Inches(4.48),
     vn(diag["rf_extrapolation"]["max_pred_USD"], 2),
     "USD/oz — dự báo cao nhất của Random Forest", ORANGE, 22, Inches(1.0))
chip(s, Inches(8.15), Inches(2.84), Inches(4.48),
     vn(diag["rf_extrapolation"]["max_actual_USD"], 2),
     "USD/oz — giá thực cao nhất trên Test", INK, 22, Inches(1.0))
chip(s, Inches(8.15), Inches(3.96), Inches(4.48),
     f"{diag['rf_extrapolation']['n_sessions_above_max_pred']} / 252",
     "phiên có giá thực vượt trần dự báo", RED, 22, Inches(1.0))

panel(s, Inches(0.7), Inches(5.62), Inches(6.3), Inches(1.05), PANEL, line=LINE)
txt(s, "Nguyên nhân cấu trúc", Inches(0.95), Inches(5.73), Inches(5.8), Inches(0.28),
    size=10.5, color=GOLD, bold=True)
txt(s, "Cây dự báo bằng trung bình lá; mọi lá đều nằm trong miền giá đã học (≤ 2.069,40 USD) nên dự báo bị chặn trên — giới hạn toán học, không phải bug.",
    Inches(0.95), Inches(6.0), Inches(5.85), Inches(0.6), size=10.5, color=MUTED, spacing=1.25)

panel(s, Inches(7.15), Inches(5.62), Inches(5.48), Inches(1.05), PANEL, line=LINE)
txt(s, "Bằng chứng bổ sung", Inches(7.4), Inches(5.73), Inches(5.0), Inches(0.28),
    size=10.5, color=GOLD, bold=True)
txt(s, f"Giá 2024 tăng +27,4% · OOB MAE {vn(diag['rf_oob']['MAE'], 2)} (R² {vn(diag['rf_oob']['R2'], 4)}) "
       f"· XGB dự báo cao nhất {vn(diag['xgb_extrapolation']['max_pred_USD'], 2)}, "
       f"{diag['xgb_extrapolation']['n_sessions_above_max_pred']}/252 phiên vượt trần.",
    Inches(7.4), Inches(6.0), Inches(5.05), Inches(0.6), size=10.5, color=MUTED, spacing=1.25)
footer(s, 7, TOTAL)

# ============================ SLIDE 8 — Ablation + Importance (CH2 & CH3)
s = slide()
title(s, "CH2: bỏ biến vĩ mô thì tốt hơn — CH3: cây chỉ dùng giá trễ và SMA",
      kicker="Ablation & độ quan trọng đặc trưng")

lr15 = ab[(ab.model == "LinearRegression") & (ab.n_features == 15)].iloc[0]
lr25 = ab[(ab.model == "LinearRegression") & (ab.n_features == 25)].iloc[0]
rf15 = ab[(ab.model == "RandomForest") & (ab.n_features == 15)].iloc[0]
rf25 = ab[(ab.model == "RandomForest") & (ab.n_features == 25)].iloc[0]
rows = [
    ["15 biến kỹ thuật", "Linear Regression", vn(lr15.val_RMSE, 3),
     vn(lr15.test_MAE, 3), vn(lr15.test_R2, 4)],
    ["25 biến (+ vĩ mô)", "Linear Regression", vn(lr25.val_RMSE, 3),
     vn(lr25.test_MAE, 3), vn(lr25.test_R2, 4)],
    ["15 biến kỹ thuật", f"Random Forest (K={int(rf15.K)})", vn(rf15.val_RMSE, 3),
     vn(rf15.test_MAE, 3), vn(rf15.test_R2, 4)],
    ["25 biến (+ vĩ mô)", f"Random Forest (K={int(rf25.K)})", vn(rf25.val_RMSE, 3),
     vn(rf25.test_MAE, 3), vn(rf25.test_R2, 4)],
]
table(s, ["Bộ đặc trưng", "Mô hình", "Val RMSE", "Test MAE", "Test R²"], rows,
      Inches(0.7), Inches(1.85), Inches(5.95),
      col_w=[1.5, 1.85, 0.95, 0.95, 0.85], size=10, head_size=9,
      row_h=Inches(0.36), highlight_rows=(0, 2))

txt(s, "Linear Regression: bỏ 10 biến vĩ mô → Val RMSE 19,03 → 17,49 (−8,1%), Test MAE 22,05 → 18,53 (−16%).",
    Inches(0.7), Inches(3.78), Inches(5.7), Inches(0.62), size=10.5, color=INK, spacing=1.25)
txt(s, "Random Forest: 15 biến cũng tốt hơn (Val RMSE 28,41 → 25,99).",
    Inches(0.7), Inches(4.4), Inches(5.7), Inches(0.3), size=10.5, color=MUTED)

picture(s, "importance.png", Inches(6.62), Inches(1.78), Inches(5.95), Inches(2.66))
txt(s, "Hai thang đo khác nhau — MDI đo mức giảm phương sai, Gain đo mức cải thiện hàm mất mát — "
       "nhưng cùng đưa nhóm giá trễ + SMA lên đầu và cùng cho thấy biến vĩ mô gần như bằng 0.",
    Inches(6.62), Inches(4.66), Inches(5.95), Inches(0.8), size=10.5, color=MUTED, spacing=1.3)

macro_mdi = rfimp[rfimp.feature.isin(MACRO)].MDI_pct.sum()
macro_gain = xgimp[xgimp.feature.isin(MACRO)].gain_pct.sum()
panel(s, Inches(0.7), Inches(5.62), Inches(11.93), Inches(1.05), PANEL, line=GOLD)
txt(s, "Thêm biến ≠ thêm thông tin",
    Inches(0.95), Inches(5.73), Inches(5.0), Inches(0.28), size=10.5, color=GOLD, bold=True)
txt(s, f"11/25 biến có VIF > 10 (đa cộng tuyến): biến vĩ mô chủ yếu lặp lại thông tin đã có trong giá vàng, "
       f"nên chỉ thêm nhiễu cho dự báo 1 phiên. Tổng đóng góp 10 biến vĩ mô: {vn(macro_mdi, 2)}% ở RF, "
       f"{vn(macro_gain, 2)}% ở XGB.",
    Inches(0.95), Inches(6.0), Inches(11.45), Inches(0.6), size=10.5, color=MUTED, spacing=1.25)
footer(s, 8, TOTAL)

# ========================================= SLIDE 9 — ΔClose (khắc phục)
s = slide()
title(s, "Đổi mục tiêu sang ΔClose: XGBoost vượt Naive Baseline",
      kicker="Thực nghiệm khắc phục")
picture(s, "delta.png", Inches(0.5), Inches(1.72), Inches(7.5), Inches(3.35))
txt(s, "Cây chỉ cần học “phiên sau tăng/giảm bao nhiêu USD” rồi cộng lại vào giá hiện tại — không còn bị chặn bởi miền giá đã học.",
    Inches(0.55), Inches(5.2), Inches(7.4), Inches(0.45), size=10.5, color=MUTED, italic=True,
    spacing=1.2)

rows = []
for key, label in [("Naive (Δ=0)", "Naive (Δ = 0)"),
                   ("XGBoost (ΔClose, K=21)", "XGBoost (ΔClose, K=21)"),
                   ("RandomForest (ΔClose, K=71)", "Random Forest (K=71)"),
                   ("LinearRegression (ΔClose)", "Linear Regression")]:
    r = dt.loc[key]
    da = "—" if pd.isna(r["directional_accuracy_pct"]) else vn(r["directional_accuracy_pct"], 2) + "%"
    rows.append([label, vn(r["MAE"], 4), da])
table(s, ["Cách tiếp cận", "MAE", "Đúng chiều"], rows,
      Inches(7.95), Inches(1.85), Inches(4.68), col_w=[2.4, 1.2, 1.05],
      size=10, head_size=9, row_h=Inches(0.38), highlight_rows=(1,))

panel(s, Inches(7.95), Inches(3.95), Inches(4.68), Inches(1.2), PANEL, line=GREEN)
txt(s, "XGBoost ΔClose vượt Naive", Inches(8.2), Inches(4.07), Inches(4.2), Inches(0.28),
    size=10.5, color=GREEN, bold=True)
txt(s, f"MAE {vn(dt.loc['XGBoost (ΔClose, K=21)', 'MAE'], 4)} < {vn(dt.loc['Naive (Δ=0)', 'MAE'], 4)} "
       f"· đúng chiều {vn(dt.loc['XGBoost (ΔClose, K=21)', 'directional_accuracy_pct'], 2)}% (ngẫu nhiên 50%)",
    Inches(8.2), Inches(4.36), Inches(4.2), Inches(0.7), size=10.5, color=MUTED, spacing=1.3)

panel(s, Inches(0.7), Inches(5.75), Inches(11.93), Inches(0.95), PANEL, line=GOLD)
txt(s, "Vì sao hiệu quả — và vì sao vẫn chỉ hơn Naive rất ít",
    Inches(0.95), Inches(5.86), Inches(11.4), Inches(0.28), size=10.5, color=GOLD, bold=True)
txt(s, "Dự báo mức thay đổi không bị giới hạn miền giá, nên mô hình vẫn theo được xu hướng vượt đỉnh. Nhưng tín hiệu 1 phiên rất yếu: "
       "mức cải thiện chỉ 0,034 USD MAE — đủ để kết luận về nguyên nhân, chưa đủ để dùng giao dịch.",
    Inches(0.95), Inches(6.12), Inches(11.45), Inches(0.55), size=10.5, color=MUTED, spacing=1.25)
footer(s, 9, TOTAL)

# ==================================== SLIDE 10 — Kết luận & hạn chế
s = slide()
title(s, "Kết luận: ba câu trả lời, ba hạn chế, bốn hướng tiếp theo", kicker="Kết luận")

cards = [
    ("CH1", "Mô hình cây có vượt Naive?", [
        ("✗", "Dự báo trực tiếp mức giá: kém 20,2 lần", RED),
        ("✓", "Dự báo ΔClose: 18,0039 < 18,0381", GREEN),
    ], BLUE),
    ("CH2", "10 biến vĩ mô có giúp ích?", [
        ("✗", "25 biến: Test MAE 22,05", RED),
        ("✓", "15 biến: 18,53 (giảm 16%)", GREEN),
    ], ORANGE),
    ("CH3", "Đặc trưng nào quan trọng nhất?", [
        ("✓", "Top-3 biến = 98,6% độ quan trọng (RF)", GREEN),
        ("✗", "10 biến vĩ mô = 0,11% (RF)", RED),
    ], PURPLE),
]
for i, (tag, question, answers, col) in enumerate(cards):
    x = Inches(0.7) + i * Inches(4.05)
    w = Inches(3.85)
    panel(s, x, Inches(1.72), w, Inches(1.78), PANEL, line=col)
    txt(s, tag, x + Inches(0.26), Inches(1.85), Inches(1.0), Inches(0.28), size=10.5,
        color=col, bold=True)
    txt(s, question, x + Inches(0.26), Inches(2.1), w - Inches(0.52), Inches(0.45),
        size=12.5, bold=True, color=INK, spacing=1.15)
    for j, (mark, body, c2) in enumerate(answers):
        yy = Inches(2.62) + j * Inches(0.4)
        txt(s, mark, x + Inches(0.26), yy, Inches(0.25), Inches(0.3), size=11, bold=True,
            color=c2)
        txt(s, body, x + Inches(0.5), yy, w - Inches(0.74), Inches(0.35), size=10.5,
            color=MUTED, spacing=1.1)

panel(s, Inches(0.7), Inches(3.63), Inches(5.85), Inches(2.25), PANEL, line=LINE)
txt(s, "Hạn chế của đề tài", Inches(0.95), Inches(3.75), Inches(5.3), Inches(0.28),
    size=10.5, color=RED, bold=True)
bullets(s, [
    ("Chỉ 1 bước", "chưa dự báo nhiều phiên hay theo ngày"),
    ("Một lần chia Test", "chưa rolling-origin theo từng năm"),
    ("Chỉ dữ liệu ngày", "bỏ lỡ biến động và tin tức trong phiên"),
    ("Một thị trường", "chưa kiểm chứng ngoài vàng COMEX"),
], Inches(0.95), Inches(4.08), Inches(5.35), size=11, gap=Inches(0.44))

panel(s, Inches(6.78), Inches(3.63), Inches(5.85), Inches(2.25), PANEL, line=LINE)
txt(s, "Hướng phát triển (theo ưu tiên)", Inches(7.03), Inches(3.75), Inches(5.3), Inches(0.28),
    size=10.5, color=GREEN, bold=True)
bullets(s, [
    ("1. Rolling-origin backtest", "10 điểm dữ liệu/năm thay vì 1"),
    ("2. Mô hình hoá Δ và sai số", "mở rộng ΔClose, thêm quantile"),
    ("3. Đặc trưng tần suất cao", "intraday, spread, tin tức"),
    ("4. Triển khai có giám sát", "cảnh báo trực tuyến, phát hiện trôi dữ liệu"),
], Inches(7.03), Inches(4.08), Inches(5.35), size=11, gap=Inches(0.44), dot=GREEN)

panel(s, Inches(0.7), Inches(6.02), Inches(11.93), Inches(0.85), PANEL2, line=GOLD)
txt(s, "Cảm ơn thầy cô đã lắng nghe — Nhóm 8 sẵn sàng nhận câu hỏi.",
    Inches(0.95), Inches(6.12), Inches(11.4), Inches(0.3), size=12.5, bold=True, color=GOLD)
txt(s, "Sản phẩm: gold_predict.py tái lập bảng Test trong ~5 giây · dashboard FastAPI · 16 pytest · công cụ nghiên cứu, không phải khuyến nghị đầu tư.",
    Inches(0.95), Inches(6.45), Inches(11.45), Inches(0.3), size=10, color=MUTED)
footer(s, 10, TOTAL)

prs.save(OUT)
print(f"Đã lưu {OUT.name} — {len(prs.slides._sldIdLst)} slide")
