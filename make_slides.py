"""Dựng bộ slide thuyết trình hoàn chỉnh (.pptx) cho đề tài Nhóm 8.

Chạy:
    python make_slides.py
    → slides_thuyet_trinh_nhom8.pptx   (19 slide chính + 3 backup)

Toàn bộ số liệu lấy trực tiếp từ gold_model_output/*.csv và extra_diagnostics.json,
nên nếu chạy lại pipeline thì chỉ cần chạy lại script này là slide cập nhật theo.
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
GOLD = RGBColor(0xF0, 0xB4, 0x29)
INK = RGBColor(0x10, 0x14, 0x1C)          # nền tối
PANEL = RGBColor(0x1A, 0x20, 0x2E)
PANEL2 = RGBColor(0x23, 0x2B, 0x3D)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
MUTED = RGBColor(0xA8, 0xB4, 0xCB)
GREEN = RGBColor(0x3D, 0xDC, 0x97)
RED = RGBColor(0xFF, 0x6B, 0x6B)
BLUE = RGBColor(0x4D, 0xAB, 0xF7)
ORANGE = RGBColor(0xFF, 0xA9, 0x4D)
PURPLE = RGBColor(0xB1, 0x97, 0xFC)
GREY = RGBColor(0x8A, 0x96, 0xAC)

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
    bg.fill.fore_color.rgb = INK
    bg.line.fill.background()
    bg.shadow.inherit = False
    return s


def txt(s, text, l, t, w, h, size=18, color=WHITE, bold=False, align=PP_ALIGN.LEFT,
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
    txt(s, text, Inches(0.7), ty, Inches(11.9), Inches(0.7), size=size, bold=True)
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


def table(s, headers, rows, l, t, w, col_w=None, size=14, head_size=12,
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
                run.font.color.rgb = highlight_color if hl else WHITE
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0x2C, 0x36, 0x4A) if hl else PANEL
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    return tbl


def bullets(s, items, l, t, w, size=17, gap=Inches(0.58), color=WHITE):
    for i, (head, body) in enumerate(items):
        y = t + i * gap
        dot = s.shapes.add_shape(MSO_SHAPE.OVAL, l, y + Inches(0.09), Inches(0.11), Inches(0.11))
        dot.fill.solid()
        dot.fill.fore_color.rgb = GOLD
        dot.line.fill.background()
        dot.shadow.inherit = False
        box = txt(s, "", l + Inches(0.28), y, w - Inches(0.28), gap, size=size)
        tf = box.text_frame
        p = tf.paragraphs[0]
        r1 = p.add_run()
        r1.text = head + "  "
        r1.font.size = Pt(size)
        r1.font.bold = True
        r1.font.color.rgb = WHITE
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
ols = pd.read_csv(ROOT / "gold_model_output" / "ols_diagnostics.csv")

TOTAL = 19


def vn(x, d=2):
    """Định dạng số kiểu Việt Nam: chấm ngăn nghìn, phẩy thập phân (2.079,82)."""
    return f"{x:,.{d}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def f2(x, d=2):
    return vn(x, d)


# ============================================================== SLIDE 1 — Bìa
s = slide()
band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.32), H)
band.fill.solid()
band.fill.fore_color.rgb = GOLD
band.line.fill.background()
band.shadow.inherit = False

txt(s, "ĐỒ ÁN MÔN TRÍ TUỆ NHÂN TẠO", Inches(1.2), Inches(1.5), Inches(11), Inches(0.4),
    size=14, color=GOLD, bold=True)
txt(s, "Dự báo giá vàng\nphiên kế tiếp bằng học máy", Inches(1.2), Inches(2.05),
    Inches(11), Inches(1.9), size=48, bold=True, spacing=1.05)
txt(s, "So sánh Naive · Linear Regression · Random Forest · XGBoost\ntrên hợp đồng tương lai vàng COMEX GC=F (USD/troy ounce), 2015–2024",
    Inches(1.2), Inches(4.15), Inches(11), Inches(1.0), size=17, color=MUTED, spacing=1.35)
line = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.2), Inches(5.35), Inches(2.2), Pt(3))
line.fill.solid()
line.fill.fore_color.rgb = GOLD
line.line.fill.background()
line.shadow.inherit = False
txt(s, "Nhóm 8", Inches(1.2), Inches(5.65), Inches(5), Inches(0.5), size=20, bold=True)
txt(s, "Dữ liệu: 2.514 phiên giao dịch · 6 nguồn Yahoo Finance · Test 2024",
    Inches(1.2), Inches(6.15), Inches(11), Inches(0.4), size=13, color=MUTED)

# ================================================ SLIDE 2 — Vấn đề
s = slide()
title(s, "Vàng tăng 2,7 lần, nhưng dự báo đúng 1 phiên mới là bài toán khó",
      kicker="Đặt vấn đề")
picture(s, "trend.png", Inches(0.6), Inches(1.85), Inches(7.0), Inches(4.4))
panel(s, Inches(7.9), Inches(1.95), Inches(4.7), Inches(4.2))
bullets(s, [
    ("Tài sản trú ẩn", "vàng hút vốn khi lạm phát và bất định địa chính trị tăng"),
    ("Bài toán học máy", "hồi quy chuỗi thời gian có giám sát, dự báo 1 bước trước"),
    ("Khó ở đâu", "nhiễu cao, dữ liệu không dừng, tín hiệu nhiễu hơn xu hướng"),
    ("Yêu cầu", "so sánh công bằng với mốc cơ sở, đánh giá ngoài mẫu"),
], Inches(8.15), Inches(2.2), Inches(4.2), size=15, gap=Inches(0.95))
footer(s, 2, TOTAL)

# ================================================ SLIDE 3 — Phát biểu bài toán
s = slide()
title(s, "Cuối phiên t chỉ biết dữ liệu ≤ t — dự báo giá đóng cửa phiên t + 1",
      kicker="Phát biểu bài toán")

panel(s, Inches(0.7), Inches(1.85), Inches(11.95), Inches(1.4), PANEL2)
txt(s, "ŷ(t + 1) = f( X(t) )        với        y(t) = Close(t + 1)",
    Inches(1.0), Inches(2.12), Inches(11.4), Inches(0.5), size=26, bold=True, color=GOLD,
    align=PP_ALIGN.CENTER)
txt(s, "X(t) là vector 25 đặc trưng, chỉ tính từ dữ liệu đã biết đến hết phiên t",
    Inches(1.0), Inches(2.7), Inches(11.4), Inches(0.4), size=14, color=MUTED,
    align=PP_ALIGN.CENTER)

# timeline
y_axis = Inches(4.55)
ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.2), y_axis, Inches(10.9), Pt(2))
ln.fill.solid()
ln.fill.fore_color.rgb = RGBColor(0x3A, 0x45, 0x5A)
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
    txt(s, label, cx - Inches(1.2), y_axis - Inches(0.75), Inches(2.4), Inches(0.35),
        size=14, bold=True, align=PP_ALIGN.CENTER)
    txt(s, sub, cx - Inches(1.2), y_axis + Inches(0.28), Inches(2.4), Inches(0.55),
        size=11, color=MUTED, align=PP_ALIGN.CENTER, spacing=0.95)

panel(s, Inches(0.7), Inches(5.55), Inches(5.85), Inches(1.3))
txt(s, "ĐÃ BIẾT tại phiên t", Inches(0.95), Inches(5.72), Inches(5), Inches(0.3),
    size=11, color=GREEN, bold=True)
txt(s, "OHLC, khối lượng vàng + giá đóng cửa DXY, US10Y, S&P 500, WTI, Bạc\n→ dùng để tạo lag, SMA, biến động, RSI, biên độ, lợi suất",
    Inches(0.95), Inches(6.0), Inches(5.4), Inches(0.8), size=11, color=MUTED, spacing=1.2)

panel(s, Inches(6.8), Inches(5.55), Inches(5.85), Inches(1.3))
txt(s, "CHƯA BIẾT", Inches(7.05), Inches(5.72), Inches(5), Inches(0.3),
    size=11, color=ORANGE, bold=True)
txt(s, "Close(t + 1) — chính là biến mục tiêu cần dự báo\n→ tuyệt đối không được dùng giá trị này khi tạo đặc trưng",
    Inches(7.05), Inches(6.0), Inches(5.4), Inches(0.8), size=11, color=MUTED, spacing=1.2)
footer(s, 3, TOTAL)

# ================================================ SLIDE 4 — Câu hỏi nghiên cứu
s = slide()
title(s, "Ba câu hỏi nghiên cứu — sẽ trả lời dứt khoát ở phần kết quả",
      kicker="Mục tiêu")

qs = [
    ("CH1", "Mô hình học máy phi tuyến (Random Forest, XGBoost)\ncó vượt Naive Baseline và Linear Regression không?", BLUE),
    ("CH2", "Thêm 10 biến kinh tế vĩ mô liên thị trường\n(DXY, US10Y, S&P 500, WTI, Bạc) có cải thiện độ chính xác?", ORANGE),
    ("CH3", "Đặc trưng nào đóng góp nhiều nhất\nvào quyết định dự báo của mô hình?", PURPLE),
]
for i, (tag, body, col) in enumerate(qs):
    x = Inches(0.7) + i * Inches(4.05)
    panel(s, x, Inches(2.0), Inches(3.85), Inches(3.1), PANEL, line=col)
    txt(s, tag, x + Inches(0.3), Inches(2.25), Inches(1.5), Inches(0.4), size=13,
        color=col, bold=True)
    txt(s, body, x + Inches(0.3), Inches(2.75), Inches(3.3), Inches(2.0), size=16,
        spacing=1.3)

txt(s, "Nguyên tắc: một kết quả cho thấy mô hình AI không vượt baseline vẫn hợp lệ\nnếu được báo cáo trung thực và giải thích được nguyên nhân.",
    Inches(0.7), Inches(5.55), Inches(11.9), Inches(1.0), size=16, color=MUTED,
    italic=True, spacing=1.35)
footer(s, 4, TOTAL)

# ================================================ SLIDE 5 — Dữ liệu
s = slide()
title(s, "2.514 phiên giao dịch, 10 cột, 0 giá trị khuyết", kicker="Dữ liệu")
kpi(s, Inches(0.7), Inches(1.9), Inches(2.8), "2.514", "phiên giao dịch", GOLD)
kpi(s, Inches(3.65), Inches(1.9), Inches(2.8), "10", "cột dữ liệu", BLUE)
kpi(s, Inches(6.6), Inches(1.9), Inches(2.8), "0", "giá trị NaN", GREEN)
kpi(s, Inches(9.55), Inches(1.9), Inches(3.1), "6", "nguồn Yahoo Finance", PURPLE)

panel(s, Inches(0.7), Inches(3.75), Inches(6.4), Inches(3.05))
txt(s, "Khoảng thời gian", Inches(0.95), Inches(3.92), Inches(5.9), Inches(0.3),
    size=11, color=GOLD, bold=True)
txt(s, "2015-01-02  →  2024-12-31", Inches(0.95), Inches(4.22), Inches(5.9), Inches(0.4),
    size=20, bold=True)
txt(s, "GC=F  ·  DX-Y.NYB  ·  ^TNX  ·  ^GSPC  ·  CL=F  ·  SI=F",
    Inches(0.95), Inches(4.7), Inches(5.9), Inches(0.35), size=13, color=MUTED)
txt(s, "Giá vàng dao động 1.050,8 – 2.800,8 USD/oz.\nCác cột vĩ mô được forward-fill theo lịch nghỉ của sàn vàng,\nkhông nội suy tuyến tính để tránh tạo thông tin giả.",
    Inches(0.95), Inches(5.15), Inches(5.9), Inches(1.05), size=13, color=MUTED, spacing=1.2)
txt(s, "Tương quan ≠ nhân quả: S&P 500 (0,94) và Bạc (0,92) mạnh nhưng không dự báo nổi từng phiên.",
    Inches(0.95), Inches(6.37), Inches(5.9), Inches(0.34), size=10, color=GREY, italic=True)

picture(s, "corr.png", Inches(7.4), Inches(3.6), Inches(5.2), Inches(3.2))
footer(s, 5, TOTAL)

# ================================================ SLIDE 6 — Chia tập & chống rò rỉ
s = slide()
title(s, "Chia theo thời gian và chống rò rỉ dữ liệu ngay từ thiết kế",
      kicker="Thiết lập thực nghiệm")

rows = [
    ["Train", "2015-02-03 → 2021-12-31", "1.739", "fit imputer + scaler"],
    ["Validation", "2022-01-03 → 2023-12-29", "502", "chọn K, không dùng để fit"],
    ["Test", "2024-01-02 → 2024-12-31", "252", "chỉ đánh giá cuối cùng"],
]
table(s, ["Tập", "Khoảng ngày của giá mục tiêu", "Số mẫu", "Vai trò"],
      rows, Inches(0.7), Inches(1.95), Inches(11.9),
      col_w=[1.2, 3.2, 1.0, 3.0], size=15, highlight_rows=(2,))

bullets(s, [
    ("Không shuffle", "thứ tự thời gian được giữ nguyên, mô hình không thấy tương lai"),
    ("Căn theo ngày giá mục tiêu", "không phải ngày phát dự báo — nên phiên đầu Test không lọt vào Train"),
    ("Imputer & scaler chỉ fit trên Train", "median và độ lệch chuẩn không bị “nhìn” dữ liệu Val/Test"),
    ("Kiểm chứng bằng test tự động", "16 pytest kiểm tra feature tại t chỉ sinh từ dữ liệu ≤ t"),
], Inches(0.7), Inches(3.86), Inches(11.9), size=15, gap=Inches(0.58))

# Dải thời gian trực quan: 3 tập nối tiếp nhau
band_y = Inches(6.28)
segs = [(0.7, 7.3, "Train 1.739  ·  2015 → 2021", BLUE),
        (8.12, 2.1, "Val 502  ·  2022–23", ORANGE),
        (10.34, 2.26, "Test 252  ·  2024", GREEN)]
for x, wid, label, col in segs:
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x),
                            band_y, Inches(wid), Inches(0.62))
    sh.fill.solid()
    sh.fill.fore_color.rgb = col
    sh.line.fill.background()
    sh.shadow.inherit = False
    txt(s, label, Inches(x), band_y + Inches(0.16), Inches(wid), Inches(0.32),
        size=12, bold=True, color=INK, align=PP_ALIGN.CENTER)
footer(s, 6, TOTAL)

# ================================================ SLIDE 7 — 25 đặc trưng
s = slide()
title(s, "Chỉ 25 đặc trưng — vừa đủ với 2.514 phiên dữ liệu", kicker="Đặc trưng đầu vào")

groups = [
    ("Giá trễ  (6)", "close_lag_0, 1, 2, 3, 5, 10", BLUE),
    ("Trung bình & biến động  (6)", "SMA 5 / 10 / 20  ·  độ biến động lợi suất 5 / 10 / 20", GREEN),
    ("Động lượng & biên độ  (3)", "RSI-14  ·  lợi suất vàng 1 phiên  ·  (High − Low) / Close", ORANGE),
    ("Liên thị trường  (10)", "DXY  ·  US10Y  ·  S&P 500  ·  WTI  ·  Bạc\n(mỗi biến: mức giá + lợi suất 1 phiên)", PURPLE),
]
y = Inches(1.9)
for name, detail, col in groups:
    panel(s, Inches(0.7), y, Inches(7.6), Inches(0.98), PANEL, line=col)
    txt(s, name, Inches(0.95), y + Inches(0.18), Inches(7.1), Inches(0.35), size=15,
        bold=True, color=col)
    txt(s, detail, Inches(0.95), y + Inches(0.54), Inches(7.1), Inches(0.45), size=12,
        color=MUTED)
    y += Inches(1.1)

panel(s, Inches(8.65), Inches(1.95), Inches(3.95), Inches(4.2), PANEL2)
txt(s, "Vì sao không dùng nhiều hơn?", Inches(8.9), Inches(2.13), Inches(3.5), Inches(0.35),
    size=14, bold=True, color=GOLD)
txt(s, "2.514 phiên là tập nhỏ với mô hình cây. Thêm biến làm tăng nguy cơ overfitting và đa cộng tuyến — OLS cho thấy 11/25 biến có VIF > 10.",
    Inches(8.9), Inches(2.55), Inches(3.45), Inches(1.2), size=12, color=MUTED, spacing=1.3)
txt(s, "25", Inches(8.9), Inches(3.75), Inches(3.45), Inches(0.55), size=30, bold=True, color=GOLD)
txt(s, "biến là mức cân bằng: đủ để kiểm chứng giả thuyết, không quá nhiều để mất kiểm soát.",
    Inches(8.9), Inches(4.3), Inches(3.45), Inches(0.75), size=11, color=MUTED, spacing=1.25)
txt(s, "Mọi đặc trưng tại phiên t chỉ dùng dữ liệu ≤ t.", Inches(8.9), Inches(5.55),
    Inches(3.45), Inches(0.4), size=11, color=GREEN, bold=True)
txt(s, "Toàn bộ đặc trưng tại phiên t chỉ dùng dữ liệu ≤ t — có test tự động kiểm chứng từng giá trị.",
    Inches(0.7), Inches(6.5), Inches(11.9), Inches(0.35), size=12, color=GOLD, bold=True)
footer(s, 7, TOTAL)

# ================================================ SLIDE 8 — 4 mô hình
s = slide()
title(s, "Bốn mô hình: một baseline, một mô hình diễn giải, hai họ phi tuyến",
      kicker="Mô hình thực nghiệm")
rows = [
    ["Naive Baseline", "không học", "ŷ(t+1) = Close(t)", "mốc cơ sở bắt buộc"],
    ["Linear Regression", "tuyến tính, tham số", "25 hệ số + scaler", "diễn giải được hệ số"],
    ["Random Forest", "bagging, 11 cây", "giảm phương sai", "phi tuyến, mạnh trong miền học"],
    ["XGBoost", "boosting, 165 vòng", "giảm độ chệch", "phi tuyến, hiệu quả dữ liệu nhỏ"],
]
table(s, ["Mô hình", "Bản chất", "Cơ chế", "Vai trò trong so sánh"],
      rows, Inches(0.7), Inches(1.95), Inches(11.9),
      col_w=[2.0, 2.0, 2.0, 3.2], size=15, highlight_rows=())
txt(s, "4 mô hình là mức tối thiểu để trả lời câu hỏi nghiên cứu công bằng:\nkhông có baseline thì không biết mô hình có ích hay không; không có mô hình tuyến tính thì không có mốc diễn giải được.",
    Inches(0.7), Inches(4.5), Inches(11.9), Inches(1.1), size=14, color=MUTED, spacing=1.3)
panel(s, Inches(0.7), Inches(5.65), Inches(11.9), Inches(1.25), PANEL2)
txt(s, "Điểm quan trọng:", Inches(0.95), Inches(5.85), Inches(3), Inches(0.3), size=14,
    bold=True, color=GOLD)
txt(s, "Naive không phải “mô hình cho có”. Với dự báo 1 phiên, giá hai phiên liên tiếp rất gần nhau —\nđây là baseline mạnh mà mọi mô hình học máy phải vượt qua để chứng minh giá trị.",
    Inches(0.95), Inches(6.2), Inches(11.4), Inches(0.6), size=13, color=MUTED, spacing=1.25)
footer(s, 8, TOTAL)

# ================================================ SLIDE 9 — Thiết lập
s = slide()
title(s, "Siêu tham số giữ cố định, chỉ quét số cây K trên Validation",
      kicker="Cấu hình thực nghiệm")
rows = [
    ["Random Forest", "max_features = 0.8", "min_samples_leaf = 2", "random_state = 42", "K ∈ 1..500"],
    ["XGBoost", "max_depth = 3", "lr = 0.03 · subsample = 0.8", "colsample = 0.8 · λ = 1", "K ∈ 1..500"],
    ["Linear Regression", "StandardScaler", "chỉ fit scaler trên Train", "—", "không có K"],
    ["Naive", "—", "không huấn luyện", "—", "không có K"],
]
table(s, ["Mô hình", "Tham số 1", "Tham số 2 / 3", "Cố định khác", "Quét K"],
      rows, Inches(0.7), Inches(1.95), Inches(11.9),
      col_w=[2.2, 2.0, 2.4, 2.0, 1.4], size=13, head_size=11)

panel(s, Inches(0.7), Inches(4.1), Inches(5.85), Inches(2.3))
txt(s, "Tiêu chí chọn K", Inches(0.95), Inches(4.28), Inches(5), Inches(0.3), size=12,
    color=GOLD, bold=True)
txt(s, "RMSE trên Validation thấp nhất.", Inches(0.95), Inches(4.6), Inches(5.4), Inches(0.35),
    size=16, bold=True)
txt(s, "Test chỉ được dùng một lần duy nhất để đánh giá cuối,\nkhông tham gia vào bất kỳ quyết định nào.",
    Inches(0.95), Inches(5.15), Inches(5.4), Inches(0.7), size=13, color=MUTED, spacing=1.3)

panel(s, Inches(6.8), Inches(4.1), Inches(5.85), Inches(2.3))
txt(s, "Vì sao quét đủ 500 giá trị K?", Inches(7.05), Inches(4.28), Inches(5.2), Inches(0.3),
    size=12, color=GOLD, bold=True)
txt(s, "Để không bỏ sót K tốt và để chứng minh kết luận\nkhông phụ thuộc vào một lựa chọn K duy nhất.",
    Inches(7.05), Inches(4.6), Inches(5.4), Inches(0.9), size=13, color=MUTED, spacing=1.3)
txt(s, "Quét 500 K chỉ mất ~5 giây", Inches(7.05), Inches(5.55), Inches(5.4), Inches(0.35),
    size=15, bold=True, color=GREEN)
txt(s, "nhờ kỹ thuật trung bình tích lũy trên estimators_ của\nmột rừng 500 cây — có test khẳng định tương đương tuyệt đối.",
    Inches(7.05), Inches(5.9), Inches(5.4), Inches(0.6), size=11, color=MUTED, spacing=1.2)
footer(s, 9, TOTAL)

# ================================================ SLIDE 10 — Chọn K
s = slide()
title(s, "Chọn K đúng quy trình: RF = 11 cây, XGB = 165 vòng boosting",
      kicker="Kết quả trên Validation")
picture(s, "kcurve.png", Inches(0.6), Inches(1.9), Inches(7.6), Inches(3.7))
rows = [
    ["Random Forest", "11", vn(kv[kv.model=='RandomForest'].RMSE.min(), 3), "0,9294"],
    ["XGBoost", "165", vn(kv[kv.model=='XGBoost'].RMSE.min(), 3), "0,8975"],
]
table(s, ["Mô hình", "K", "Val RMSE", "Val R²"], rows,
      Inches(8.35), Inches(2.0), Inches(4.3), col_w=[1.9, 0.7, 1.15, 1.0], size=12,
      head_size=11)

panel(s, Inches(8.5), Inches(3.85), Inches(4.1), Inches(2.55), PANEL2)
txt(s, "Điều quan trọng hơn K", Inches(8.7), Inches(4.03), Inches(3.7), Inches(0.3),
    size=11, color=GOLD, bold=True)
txt(s, "Đường RMSE của Random Forest gần như phẳng từ K = 50 trở đi (~32 USD).\n\nNghĩa là chỉnh K không cứu được mô hình — vấn đề nằm ở bản chất thuật toán, không ở siêu tham số.",
    Inches(8.7), Inches(4.35), Inches(3.7), Inches(2.0), size=12, color=MUTED, spacing=1.3)
txt(s, "XGBoost cần tới 165 vòng vì learning rate thấp (0,03): dưới 100 vòng mô hình chưa hội tụ (RMSE còn 49,8 USD tại K = 100).",
    Inches(0.7), Inches(5.9), Inches(7.6), Inches(0.8), size=12, color=MUTED, spacing=1.3)
footer(s, 10, TOTAL)

# ================================================ SLIDE 11 — KẾT QUẢ TEST
s = slide()
title(s, "Trên Test 2024: Naive thắng — mô hình cây kém 20 lần", kicker="Kết quả chính")
rows = []
for i, model in enumerate(["Naive", "LinearRegression", "RandomForest", "XGBoost"]):
    r = m.loc[model]
    k = "—" if pd.isna(r.get("K_trees")) else f"{int(r['K_trees'])}"
    rows.append([
        {"Naive": "Naive Baseline", "LinearRegression": "Linear Regression",
         "RandomForest": "Random Forest", "XGBoost": "XGBoost"}[model],
        k, vn(r["MAE"], 3), vn(r["RMSE"], 3),
        vn(r["MAPE_pct"], 3) + "%", vn(r["R2"], 4),
    ])
table(s, ["Mô hình", "K", "MAE (USD/oz)", "RMSE", "MAPE", "R²"], rows,
      Inches(0.7), Inches(1.95), Inches(7.4),
      col_w=[2.0, 0.7, 1.4, 1.3, 1.1, 1.0], size=13, highlight_rows=(0,))
picture(s, "forecast.png", Inches(8.35), Inches(1.85), Inches(4.3), Inches(2.6))
txt(s, "Hai đường RF/XGB nằm ngang quanh 2.040 USD,\nkhông bám được giá thực — dấu hiệu bị chặn ngoại suy.",
    Inches(8.35), Inches(4.5), Inches(4.3), Inches(0.8), size=11, color=MUTED, spacing=1.25)

kpi(s, Inches(0.7), Inches(4.55), Inches(2.35), "18,04", "MAE Naive (USD/oz)", GREEN)
kpi(s, Inches(3.2), Inches(4.55), Inches(2.35), "364,53", "MAE Random Forest", RED)
kpi(s, Inches(5.7), Inches(4.55), Inches(2.35), "×20", "kém hơn Naive", RED)
txt(s, "Đây là kết quả thật, không phải lỗi code: 16 test tự động xác nhận metric khớp báo cáo tới 1e-9.",
    Inches(0.7), Inches(6.35), Inches(11.9), Inches(0.4), size=12, color=GOLD, bold=True)
footer(s, 11, TOTAL)

# ================================================ SLIDE 12 — Vì sao
s = slide()
title(s, "Vì sao cây thua: nó không thể dự báo giá cao hơn đỉnh đã học", kicker="Chẩn đoán nguyên nhân")
kpi(s, Inches(0.7), Inches(1.85), Inches(3.6),
    vn(diag["rf_extrapolation"]["max_pred_USD"], 2),
    "USD/oz — dự báo cao nhất mà RF đưa ra", ORANGE, vsize=34)
kpi(s, Inches(4.6), Inches(1.85), Inches(3.6),
    vn(diag["rf_extrapolation"]["max_actual_USD"], 2),
    "USD/oz — giá thực cao nhất trên Test", WHITE, vsize=34)
kpi(s, Inches(8.5), Inches(1.85), Inches(4.1),
    f"{diag['rf_extrapolation']['n_sessions_above_max_pred']} / 252",
    "phiên có giá thực vượt trần dự báo", RED, vsize=34)

panel(s, Inches(0.7), Inches(3.7), Inches(5.85), Inches(2.7))
txt(s, "Nguyên nhân cấu trúc", Inches(0.95), Inches(3.88), Inches(5.4), Inches(0.3),
    size=12, color=GOLD, bold=True)
txt(s, "Cây quyết định dự báo bằng giá trị trung bình của lá.\nMọi lá đều nằm trong miền giá đã thấy khi huấn luyện (≤ 2.069,40 USD).\n\nVì vậy dự báo của cây bị chặn trên — không phải bug, mà là giới hạn toán học của phép chia không gian.",
    Inches(0.95), Inches(4.2), Inches(5.4), Inches(2.1), size=13, color=MUTED, spacing=1.3)

panel(s, Inches(6.8), Inches(3.7), Inches(5.85), Inches(2.7))
txt(s, "Bằng chứng bổ sung", Inches(7.05), Inches(3.88), Inches(5.4), Inches(0.3),
    size=12, color=GOLD, bold=True)
rows = [
    ["Giá vàng 2024 tăng", "+27,4%"],
    ["Đỉnh giá của Train", "2.069,40 (08/2020)"],
    ["OOB MAE (Train+Val)", vn(diag["rf_oob"]["MAE"], 2) + " — giỏi trong miền học"],
    ["XGB dự báo cao nhất", vn(diag["xgb_extrapolation"]["max_pred_USD"], 2)],
    ["Số phiên XGB vượt trần", f"{diag['xgb_extrapolation']['n_sessions_above_max_pred']} / 252"],
]
table(s, ["Chỉ số", "Giá trị"], rows, Inches(7.05), Inches(4.2), Inches(5.4),
      col_w=[2.6, 2.4], size=11, head_size=10, row_h=Inches(0.34))
footer(s, 12, TOTAL)

# ================================================ SLIDE 13 — Ablation
s = slide()
title(s, "Thêm biến vĩ mô làm mô hình tệ đi — CH2 trả lời KHÔNG", kicker="Ablation study")

lr15 = ab[(ab.model == "LinearRegression") & (ab.n_features == 15)].iloc[0]
lr25 = ab[(ab.model == "LinearRegression") & (ab.n_features == 25)].iloc[0]
rf15 = ab[(ab.model == "RandomForest") & (ab.n_features == 15)].iloc[0]
rf25 = ab[(ab.model == "RandomForest") & (ab.n_features == 25)].iloc[0]

rows = [
    ["15 biến kỹ thuật", "Linear Regression", f"{vn(lr15.val_RMSE, 3)}",
     f"{vn(lr15.test_MAE, 3)}", f"{vn(lr15.test_R2, 4)}"],
    ["25 biến (+ vĩ mô)", "Linear Regression", f"{vn(lr25.val_RMSE, 3)}",
     f"{vn(lr25.test_MAE, 3)}", f"{vn(lr25.test_R2, 4)}"],
    ["15 biến kỹ thuật", f"Random Forest (K={int(rf15.K)})", f"{vn(rf15.val_RMSE, 3)}",
     f"{vn(rf15.test_MAE, 3)}", f"{vn(rf15.test_R2, 4)}"],
    ["25 biến (+ vĩ mô)", f"Random Forest (K={int(rf25.K)})", f"{vn(rf25.val_RMSE, 3)}",
     f"{vn(rf25.test_MAE, 3)}", f"{vn(rf25.test_R2, 4)}"],
]
table(s, ["Bộ đặc trưng", "Mô hình", "Val RMSE", "Test MAE", "Test R²"], rows,
      Inches(0.7), Inches(1.95), Inches(11.9),
      col_w=[2.4, 2.6, 1.6, 1.6, 1.4], size=14, highlight_rows=(0, 2))

panel(s, Inches(0.7), Inches(4.35), Inches(5.85), Inches(1.9))
txt(s, "Linear Regression", Inches(0.95), Inches(4.52), Inches(5), Inches(0.3), size=12,
    color=GOLD, bold=True)
txt(s, "Bỏ 10 biến vĩ mô → Val RMSE giảm từ 19,03 xuống 17,49 (−8,1%)\nvà Test MAE giảm từ 22,05 xuống 18,53 (−16%).",
    Inches(0.95), Inches(4.85), Inches(5.4), Inches(1.2), size=13, color=MUTED, spacing=1.3)

panel(s, Inches(6.8), Inches(4.35), Inches(5.85), Inches(1.9))
txt(s, "Vì sao vĩ mô không giúp?", Inches(7.05), Inches(4.52), Inches(5.4), Inches(0.3),
    size=12, color=GOLD, bold=True)
txt(s, "Biến vĩ mô tương quan mạnh với nhau và với chính giá vàng (VIF > 10 ở 11/25 biến),\nnên chỉ thêm nhiễu và đa cộng tuyến thay vì thông tin mới cho dự báo 1 phiên.",
    Inches(7.05), Inches(4.85), Inches(5.4), Inches(1.2), size=13, color=MUTED, spacing=1.3)
footer(s, 13, TOTAL)

# ================================================ SLIDE 14 — Importance
s = slide()
title(s, "Cây chỉ thực sự dùng nhóm giá trễ và SMA — vĩ mô dưới 1%",
      kicker="Đặc trưng quan trọng nhất — CH3")
picture(s, "importance.png", Inches(0.5), Inches(1.8), Inches(7.5), Inches(3.35))
top3_rf = rfimp.head(3)
top3_xg = xgimp.head(3)
txt(s, "Random Forest — MDI %", Inches(0.7), Inches(5.35), Inches(3.5), Inches(0.3),
    size=12, color=GREEN, bold=True)
txt(s, "XGBoost — Gain %", Inches(4.2), Inches(5.35), Inches(3.5), Inches(0.3),
    size=12, color=RED, bold=True)
for i in range(3):
    r, x = top3_rf.iloc[i], top3_xg.iloc[i]
    txt(s, f"{r['feature']} — {vn(r['MDI_pct'], 1)}%", Inches(0.7),
        Inches(5.72) + i * Inches(0.34), Inches(3.4), Inches(0.3), size=13)
    txt(s, f"{x['feature']} — {vn(x['gain_pct'], 1)}%", Inches(4.2),
        Inches(5.72) + i * Inches(0.34), Inches(3.4), Inches(0.3), size=13)

panel(s, Inches(8.3), Inches(1.8), Inches(4.35), Inches(4.75), PANEL2)
txt(s, "Đọc kết quả thế nào", Inches(8.55), Inches(2.0), Inches(3.9), Inches(0.3),
    size=12, color=GOLD, bold=True)
txt(s, "MDI đo mức giảm phương sai mỗi lần chia.\nGain đo mức cải thiện hàm mất mát.\nHai thang đo xếp hạng hơi khác nhau nhưng cùng kết luận.",
    Inches(8.55), Inches(2.35), Inches(3.85), Inches(1.25), size=11, color=MUTED, spacing=1.3)
txt(s, "> 98%", Inches(8.55), Inches(3.55), Inches(3.85), Inches(0.6), size=34,
    bold=True, color=GREEN)
txt(s, "độ quan trọng của RF nằm ở nhóm giá trễ + SMA", Inches(8.55), Inches(4.15),
    Inches(3.85), Inches(0.4), size=11, color=MUTED)
txt(s, "< 1%", Inches(8.55), Inches(4.6), Inches(3.85), Inches(0.6), size=34,
    bold=True, color=RED)
txt(s, "tổng đóng góp của cả 10 biến vĩ mô ở RF", Inches(8.55), Inches(5.2),
    Inches(3.85), Inches(0.4), size=11, color=MUTED)
txt(s, "Mức quan trọng không chứng minh nhân quả.", Inches(8.55), Inches(5.95),
    Inches(3.85), Inches(0.4), size=11, color=GOLD, italic=True)
footer(s, 14, TOTAL)

# ================================================ SLIDE 15 — Delta
s = slide()
title(s, "Đổi cách đặt mục tiêu sang ΔClose: XGBoost vượt Naive",
      kicker="Thực nghiệm khắc phục")
picture(s, "delta.png", Inches(0.55), Inches(1.85), Inches(7.5), Inches(3.4))
rows = []
for key, label in [("Naive (Δ=0)", "Naive (Δ = 0)"),
                   ("XGBoost (ΔClose, K=21)", "XGBoost (ΔClose, K=21)"),
                   ("RandomForest (ΔClose, K=71)", "Random Forest (ΔClose)"),
                   ("LinearRegression (ΔClose)", "Linear Regression (ΔClose)")]:
    r = dt.loc[key]
    da = "—" if pd.isna(r["directional_accuracy_pct"]) else vn(r["directional_accuracy_pct"], 2) + "%"
    rows.append([label, vn(r["MAE"], 4), da])
table(s, ["Cách tiếp cận", "MAE", "Đúng chiều"], rows,
      Inches(8.15), Inches(1.95), Inches(4.5),
      col_w=[2.6, 1.2, 1.1], size=12, head_size=10, highlight_rows=(1,))

panel(s, Inches(8.15), Inches(3.95), Inches(4.5), Inches(1.5), PANEL2)
txt(s, "XGBoost dự báo ΔClose vượt Naive", Inches(8.35), Inches(4.12), Inches(4.1),
    Inches(0.3), size=11, color=GREEN, bold=True)
txt(s, "MAE 18,0039 < 18,0381 · đúng chiều 55,95% (so với 50% ngẫu nhiên)",
    Inches(8.35), Inches(4.42), Inches(4.1), Inches(0.8), size=12, color=WHITE, spacing=1.3)

txt(s, "Vì sao hiệu quả?", Inches(0.7), Inches(5.42), Inches(7.2), Inches(0.3), size=13,
    color=GOLD, bold=True)
txt(s, "Dự báo mức thay đổi (Δ) không bị chặn bởi miền giá đã học: cây chỉ cần học “phiên sau tăng/giảm bao nhiêu USD”,\nrồi cộng lại vào giá hiện tại — nên vẫn theo được xu hướng tăng vượt đỉnh. Đây là minh chứng: cách đặt mục tiêu\nquan trọng hơn việc đổi thuật toán.",
    Inches(0.7), Inches(5.75), Inches(11.9), Inches(1.1), size=13, color=MUTED, spacing=1.35)
footer(s, 15, TOTAL)

# ================================================ SLIDE 16 — Kết luận
s = slide()
title(s, "Ba câu hỏi nghiên cứu — ba câu trả lời dứt khoát", kicker="Kết luận")

cards = [
    ("CH1", "Phi tuyến có vượt Naive?",
     "KHÔNG", "khi dự báo trực tiếp mức giá",
     "CÓ", "khi dự báo ΔClose: 18,0039 < 18,0381", "Slide 11 · 15", BLUE),
    ("CH2", "Biến vĩ mô có giúp?",
     "KHÔNG", "bỏ 10 biến vĩ mô thì Linear tốt hơn",
     "17,49", "Val RMSE thay vì 19,03", "Slide 13", ORANGE),
    ("CH3", "Đặc trưng nào quan trọng?",
     "> 98%", "độ quan trọng của RF nằm ở giá trễ + SMA",
     "< 1%", "tổng đóng góp của 10 biến vĩ mô", "Slide 14", PURPLE),
]
for i, (tag, question, big1, small1, big2, small2, ref, col) in enumerate(cards):
    x = Inches(0.7) + i * Inches(4.05)
    w = Inches(3.85)
    panel(s, x, Inches(1.9), w, Inches(4.2), PANEL, line=col)
    txt(s, tag, x + Inches(0.28), Inches(2.08), Inches(1.2), Inches(0.3), size=12,
        color=col, bold=True)
    txt(s, ref, x + w - Inches(1.4), Inches(2.08), Inches(1.15), Inches(0.3), size=10,
        color=GREY, align=PP_ALIGN.RIGHT)
    txt(s, question, x + Inches(0.28), Inches(2.42), w - Inches(0.56), Inches(0.6),
        size=15, bold=True, spacing=1.15)
    line = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x + Inches(0.28), Inches(3.05),
                              w - Inches(0.56), Pt(1))
    line.fill.solid()
    line.fill.fore_color.rgb = RGBColor(0x36, 0x41, 0x56)
    line.line.fill.background()
    line.shadow.inherit = False
    txt(s, big1, x + Inches(0.28), Inches(3.18), w - Inches(0.56), Inches(0.6),
        size=32, bold=True, color=col)
    txt(s, small1, x + Inches(0.28), Inches(3.78), w - Inches(0.56), Inches(0.5),
        size=11, color=MUTED, spacing=1.2)
    txt(s, big2, x + Inches(0.28), Inches(4.32), w - Inches(0.56), Inches(0.6),
        size=32, bold=True, color=WHITE)
    txt(s, small2, x + Inches(0.28), Inches(4.92), w - Inches(0.56), Inches(0.5),
        size=11, color=MUTED, spacing=1.2)

panel(s, Inches(0.7), Inches(6.4), Inches(11.9), Inches(0.55), PANEL2, line=GOLD)
txt(s, "Thông điệp:  kết quả âm tính vẫn là khoa học — nếu chỉ ra được nguyên nhân và cách khắc phục.",
    Inches(0.95), Inches(6.53), Inches(11.4), Inches(0.32), size=12, color=GOLD, bold=True)
footer(s, 16, TOTAL)

# ================================================ SLIDE 17 — Hạn chế
s = slide()
title(s, "Hạn chế và hướng phát triển", kicker="Trung thực về giới hạn")
panel(s, Inches(0.7), Inches(1.95), Inches(5.85), Inches(4.3))
txt(s, "Hạn chế của đề tài", Inches(0.95), Inches(2.15), Inches(5.4), Inches(0.3), size=12,
    color=RED, bold=True)
bullets(s, [
    ("Chỉ 1 bước", "chưa dự báo nhiều phiên hay theo ngày"),
    ("Dữ liệu 2015–2024", "chưa kiểm chứng trên thị trường khác"),
    ("Chưa rolling-origin", "chỉ một lần chia Test, chưa backtest theo năm"),
    ("Chưa dùng intraday", "chỉ dữ liệu ngày, bỏ lỡ biến động trong phiên"),
], Inches(0.95), Inches(2.55), Inches(5.4), size=14, gap=Inches(0.85))

panel(s, Inches(6.8), Inches(1.95), Inches(5.85), Inches(4.3))
txt(s, "Hướng phát triển (theo ưu tiên)", Inches(7.05), Inches(2.15), Inches(5.4),
    Inches(0.3), size=12, color=GREEN, bold=True)
bullets(s, [
    ("1. Rolling-origin backtest", "đánh giá theo từng năm → 10 điểm dữ liệu thay vì 1"),
    ("2. Mô hình hoá Δ và sai số", "mở rộng thực nghiệm ΔClose, thêm quantile"),
    ("3. Đặc trưng tần suất cao", "intraday, khối lượng, spread, tin tức"),
    ("4. Triển khai thực tế", "cảnh báo trực tuyến, giám sát trôi dữ liệu"),
], Inches(7.05), Inches(2.55), Inches(5.4), size=14, gap=Inches(0.85))
footer(s, 17, TOTAL)

# ================================================ SLIDE 18 — Sản phẩm
s = slide()
title(s, "Sản phẩm bảo vệ: pipeline tái lập được + dashboard + 16 test",
      kicker="Sản phẩm")
cards = [
    ("Pipeline", "gold_predict.py\ngold_extra_outputs.py", "4 mô hình, quét 500 K", BLUE),
    ("Dashboard", "FastAPI + Chart.js", "7 API, biểu đồ tương tác", GREEN),
    ("Kiểm thử", "pytest", "16 test: rò rỉ, metric, API", ORANGE),
    ("Tài liệu", "README + 3 file .md", "cách chạy + kịch bản demo", PURPLE),
]
for i, (name, detail, note, col) in enumerate(cards):
    x = Inches(0.7) + i * Inches(3.05)
    panel(s, x, Inches(1.95), Inches(2.85), Inches(2.4), PANEL, line=col)
    txt(s, name, x + Inches(0.25), Inches(2.15), Inches(2.4), Inches(0.35), size=16,
        bold=True, color=col)
    txt(s, detail, x + Inches(0.25), Inches(2.6), Inches(2.4), Inches(0.8), size=12,
        color=WHITE, spacing=1.3)
    txt(s, note, x + Inches(0.25), Inches(3.5), Inches(2.4), Inches(0.7), size=11,
        color=MUTED, spacing=1.25)

panel(s, Inches(0.7), Inches(4.6), Inches(11.9), Inches(1.75))
txt(s, "Tái lập trong 5 giây", Inches(0.95), Inches(4.78), Inches(5), Inches(0.3), size=12,
    color=GOLD, bold=True)
txt(s, "python gold_predict.py   →   in đúng bảng Test trong báo cáo (khớp tới 6 chữ số thập phân)\npython demo_preflight.py  →   kiểm tra 12 mục sẵn sàng demo trước giám thị",
    Inches(0.95), Inches(5.12), Inches(11.4), Inches(0.9), size=14, spacing=1.4)
txt(s, "Công cụ nghiên cứu học thuật — không phải khuyến nghị đầu tư.",
    Inches(0.7), Inches(6.55), Inches(11.9), Inches(0.35), size=11, color=GREY, italic=True)
footer(s, 18, TOTAL)

# ================================================ SLIDE 19 — Q&A
s = slide()
txt(s, "Cảm ơn thầy cô đã lắng nghe", Inches(1.0), Inches(2.4), Inches(11.3), Inches(1.2),
    size=44, bold=True)
line = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(3.72), Inches(2.2), Pt(4))
line.fill.solid()
line.fill.fore_color.rgb = GOLD
line.line.fill.background()
line.shadow.inherit = False
txt(s, "Nhóm 8 sẵn sàng nhận câu hỏi", Inches(1.0), Inches(4.0), Inches(11), Inches(0.5),
    size=20, color=MUTED)

box = [
    ("Backup A", "Đa cộng tuyến & OLS", "VIF, p-value, Durbin–Watson"),
    ("Backup B", "Quét K chi tiết", "bảng K = 1, 10, 11, 50, 100, 165, 300, 500"),
    ("Backup C", "Kiểm thử tự động", "pytest → 16 passed"),
]
for i, (tag, name, detail) in enumerate(box):
    x = Inches(1.0) + i * Inches(3.85)
    panel(s, x, Inches(5.05), Inches(3.65), Inches(1.3), PANEL2)
    txt(s, tag, x + Inches(0.25), Inches(5.2), Inches(3.2), Inches(0.3), size=10,
        color=GOLD, bold=True)
    txt(s, name, x + Inches(0.25), Inches(5.48), Inches(3.2), Inches(0.35), size=14,
        bold=True)
    txt(s, detail, x + Inches(0.25), Inches(5.88), Inches(3.2), Inches(0.35), size=10,
        color=MUTED)
txt(s, "3 slide dự phòng ngay sau đây — bấm tiếp khi cần trả lời sâu hơn.",
    Inches(1.0), Inches(6.6), Inches(11), Inches(0.35), size=11, color=GREY, italic=True)

# ============================================= SLIDE 20 — Backup A (OLS)
s = slide()
title(s, "Backup A — Đa cộng tuyến và ý nghĩa thống kê của hệ số", kicker="Dự phòng câu hỏi")
top = ols[ols.feature != "intercept"].reindex(
    ols[ols.feature != "intercept"].VIF.fillna(0).sort_values(ascending=False).index).head(6)
rows = [[r.feature, f"{vn(r.coef, 3)}", f"{vn(r.std_err, 3)}",
         f"{vn(r.t_stat, 2)}", f"{vn(r.p_value, 4)}", f"{vn(r.VIF, 1)}"]
        for _, r in top.iterrows()]
table(s, ["Đặc trưng", "β (chuẩn hoá)", "SE", "t", "p-value", "VIF"], rows,
      Inches(0.7), Inches(1.95), Inches(7.6),
      col_w=[2.4, 1.4, 1.1, 1.0, 1.1, 1.0], size=12, head_size=11)

panel(s, Inches(8.6), Inches(1.95), Inches(4.05), Inches(4.3), PANEL2)
txt(s, "Các con số chính", Inches(8.85), Inches(2.15), Inches(3.6), Inches(0.3), size=12,
    color=GOLD, bold=True)
txt(s, f"Train R² = {vn(diag['ols']['train_r2'], 4)}\nDurbin–Watson: Train {vn(diag['ols']['durbin_watson_train'], 3)} · Test {vn(diag['ols']['durbin_watson_test'], 3)}\n\n{diag['ols']['n_vif_gt_10']}/25 biến có VIF > 10\nVIF cao nhất = {vn(diag['ols']['max_vif'], 0)} (sma_5)\n\n{diag['ols']['n_significant_p005']} hệ số có p < 0,05",
    Inches(8.85), Inches(2.5), Inches(3.6), Inches(3.6), size=12, color=MUTED, spacing=1.4)
txt(s, "Hệ số riêng lẻ không nên diễn giải như tác động nhân quả do đa cộng tuyến nặng giữa giá trễ, SMA và giá mức.",
    Inches(0.7), Inches(6.3), Inches(11.9), Inches(0.5), size=12, color=MUTED, italic=True)

# ============================================= SLIDE 21 — Backup B (K scan)
s = slide()
title(s, "Backup B — Quét K đầy đủ: kết luận không phụ thuộc một K duy nhất",
      kicker="Dự phòng câu hỏi")
ks = [1, 10, 11, 50, 100, 165, 300, 500]
rows = []
for k in ks:
    rf_row = kv[(kv.model == "RandomForest") & (kv.K_trees == k)]
    xg_row = kv[(kv.model == "XGBoost") & (kv.K_trees == k)]
    rf_v = vn(rf_row.RMSE.iloc[0], 3) if len(rf_row) else "—"
    xg_v = vn(xg_row.RMSE.iloc[0], 3) if len(xg_row) else "—"
    rows.append([str(k), rf_v, xg_v])
table(s, ["K", "Random Forest — Val RMSE", "XGBoost — Val RMSE"], rows,
      Inches(1.9), Inches(1.95), Inches(8.4),
      col_w=[1.0, 3.2, 3.2], size=14, highlight_rows=(2, 5))
txt(s, "RF tốt nhất tại K = 11 (28,413). XGBoost giảm mạnh tới K = 165 (34,238) rồi tăng nhẹ.\nNhưng trên Test, mọi K của cả hai mô hình đều kém Naive — nên nguyên nhân là giới hạn mô hình, không phải lựa chọn K.",
    Inches(1.9), Inches(5.85), Inches(9.4), Inches(0.9), size=13, color=MUTED, spacing=1.3)

# ============================================= SLIDE 22 — Backup C (tests)
s = slide()
title(s, "Backup C — 16 kiểm thử tự động bảo vệ tính đúng đắn", kicker="Dự phòng câu hỏi")
groups_b = [
    ("Chống rò rỉ dữ liệu", "feature tại t chỉ sinh từ dữ liệu ≤ t; y(t) = Close(t+1)\nchia tập không chồng lấn, đúng mốc thời gian", GREEN),
    ("Cổng hồi quy metric", "MAE 4 mô hình khớp báo cáo tới 1e-9\nnếu ai sửa code làm lệch kết quả → test fail ngay", BLUE),
    ("Tương đương quét K", "trung bình tích lũy = fit riêng từng K (sai khác < 1e-13)\nXGB iteration_range = fit riêng từng K", ORANGE),
    ("Toàn bộ REST API", "health, metrics, importance rf/xgb, predict\nkể cả validate dữ liệu sai trả về 422", PURPLE),
]
for i, (name, detail, col) in enumerate(groups_b):
    y = Inches(1.95) + i * Inches(1.15)
    panel(s, Inches(0.7), y, Inches(11.9), Inches(1.0), PANEL, line=col)
    txt(s, name, Inches(0.95), y + Inches(0.16), Inches(3.4), Inches(0.35), size=14,
        bold=True, color=col)
    txt(s, detail, Inches(4.5), y + Inches(0.14), Inches(7.9), Inches(0.75), size=12,
        color=MUTED, spacing=1.25)
txt(s, "Chạy:  .venv/bin/python -m pytest gold_price_fullstack/tests -q     →     16 passed",
    Inches(0.7), Inches(6.75), Inches(11.9), Inches(0.35), size=13, color=GOLD, bold=True)

OUT_DIR = ROOT
prs.save(OUT)
print(f"Đã lưu {OUT.name} — {len(prs.slides.__iter__.__self__._sldIdLst)} slide")
