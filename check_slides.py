"""Dò lỗi layout trong .pptx: chữ tràn khung, đè lên nhau, tràn mép slide.

    python check_slides.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import ImageFont
from pptx import Presentation
from pptx.util import Emu

ROOT = Path(__file__).resolve().parent
prs = Presentation(str(ROOT / "slides_thuyet_trinh_nhom8.pptx"))

EMU_IN = 914400
SLIDE_W_IN = prs.slide_width / EMU_IN
SLIDE_H_IN = prs.slide_height / EMU_IN
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def text_width_pt(text: str, size_pt: float, bold: bool) -> float:
    """Ước lượng bề rộng chữ (pt) bằng font metric thật."""
    f = ImageFont.truetype(FONT_B if bold else FONT, int(size_pt * 4))  # 4x để chính xác
    return f.getlength(text) / 4


def wrapped_height_in(shape) -> float:
    """Ước lượng chiều cao cần thiết (inch) khi wrap chữ trong shape."""
    if not shape.has_text_frame or not shape.text_frame.text.strip():
        return 0.0
    tf = shape.text_frame
    max_w_pt = (shape.width / EMU_IN) * 72 - 6          # trừ margin
    total_pt = 0.0
    for p in tf.paragraphs:
        if not p.runs:
            total_pt += 10
            continue
        size = p.runs[0].font.size.pt if p.runs[0].font.size else 18
        bold = bool(p.runs[0].font.bold)
        spacing = p.line_spacing or 1.0
        text = "".join(r.text for r in p.runs)
        # đếm số dòng khi wrap theo từ
        words, lines, cur = text.split(" "), 1, ""
        for w in words:
            trial = (cur + " " + w).strip()
            if text_width_pt(trial, size, bold) <= max_w_pt or not cur:
                cur = trial
            else:
                lines += 1
                cur = w
        total_pt += lines * size * 1.22 * spacing + 3
    return total_pt / 72


def rect(shape):
    return (shape.left / EMU_IN, shape.top / EMU_IN,
            (shape.left + shape.width) / EMU_IN, (shape.top + shape.height) / EMU_IN)


issues: list[str] = []

for i, slide in enumerate(prs.slides, start=1):
    # --- tràn mép slide
    for sh in slide.shapes:
        l, t, r, b = rect(sh)
        if l < -0.02 or t < -0.02 or r > SLIDE_W_IN + 0.02 or b > SLIDE_H_IN + 0.02:
            if sh.width > EMU_IN * 0.5:      # bỏ qua nền & thanh trang trí
                issues.append(f"S{i:02d} TRÀN MÉP: {sh.shape_type} "
                              f"({l:.2f},{t:.2f})-({r:.2f},{b:.2f})")

    # --- chữ tràn khung
    for sh in slide.shapes:
        if not sh.has_text_frame or not sh.text_frame.text.strip():
            continue
        need = wrapped_height_in(sh)
        have = sh.height / EMU_IN
        if need > have + 0.06:
            first = sh.text_frame.text.strip().split("\n")[0][:52]
            issues.append(f"S{i:02d} CHỮ TRÀN KHUNG: cần {need:.2f}in / có {have:.2f}in :: “{first}…”")

    # --- text đè lên picture hoặc panel khác
    shapes = list(slide.shapes)
    for a in shapes:
        if not a.has_text_frame or not a.text_frame.text.strip():
            continue
        la, ta, ra, ba = rect(a)
        for b_ in shapes:
            if b_ is a:
                continue
            is_pic = hasattr(b_, "image")
            is_panel = (b_.shape_type == 5 or b_.has_table)      # rounded rect / table
            if not (is_pic or is_panel):
                continue
            lb, tb, rb, bb = rect(b_)
            ox = min(ra, rb) - max(la, lb)
            oy = min(ba, bb) - max(ta, tb)
            if ox > 0.12 and oy > 0.12:
                first = a.text_frame.text.strip().split("\n")[0][:46]
                kind = "ẢNH" if is_pic else "BẢNG/PANEL"
                issues.append(f"S{i:02d} ĐÈ {kind}: “{first}…” "
                              f"overlap {ox:.2f}×{oy:.2f}in")

print(f"Tổng {len(prs.slides._sldIdLst)} slide · {len(issues)} cảnh báo\n")
for s in issues:
    print(" ", s)
