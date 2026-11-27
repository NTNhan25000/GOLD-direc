"""Render .pptx ra PNG để tự kiểm tra layout khi không có LibreOffice.

Đây là renderer đơn giản (chỉ hỗ trợ các shape/text/picture mà make_slides.py dùng),
KHÔNG phải trình render PowerPoint đầy đủ. Dùng để phát hiện tràn chữ / chồng lớp.

    python preview_slides.py            # render tất cả
    python preview_slides.py 11 12      # chỉ slide 11, 12
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Emu

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "slide_preview"
OUTDIR.mkdir(exist_ok=True)

SCALE = 1500 / Emu(0).emu if False else 1.0
PPTX = ROOT / "slides_thuyet_trinh_nhom8.pptx"
prs = Presentation(str(PPTX))

# pixel-per-EMU: 13.333in → 1800px
PX_PER_IN = 135
EMU_PER_IN = 914400


def px(emu: int) -> int:
    return int(emu / EMU_PER_IN * PX_PER_IN)


FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def font(size_pt: float, bold=False):
    size_px = int(size_pt * PX_PER_IN / 72)
    path = FONT_CANDIDATES[1] if bold else FONT_CANDIDATES[0]
    try:
        return ImageFont.truetype(path, size_px)
    except OSError:
        return ImageFont.load_default()


def wrap(draw, text, fnt, max_w):
    """Xuống dòng theo từ; trả về list dòng."""
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=fnt) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render(idx: int):
    slide = prs.slides[idx]
    img = Image.new("RGB", (px(prs.slide_width), px(prs.slide_height)), "#10141C")
    d = ImageDraw.Draw(img)

    for shape in slide.shapes:
        st = shape.shape_type
        L, T = px(shape.left), px(shape.top)
        Wd, Ht = px(shape.width), px(shape.height)

        # ---- picture
        if shape.shape_type == 13 or hasattr(shape, "image"):
            try:
                from io import BytesIO
                im = Image.open(BytesIO(shape.image.blob)).convert("RGB")
                im = im.resize((max(Wd, 1), max(Ht, 1)))
                img.paste(im, (L, T))
                continue
            except Exception:
                pass

        # ---- shape fill
        if shape.shape_type is not None and hasattr(shape, "fill"):
            try:
                if shape.fill.type is not None and shape.fill.type == 1:
                    rgb = shape.fill.fore_color.rgb
                    col = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
                    if Ht <= 6:  # thanh kẻ mảnh
                        d.rectangle([L, T, L + Wd, T + max(Ht, 2)], fill=col)
                    elif shape.shape_type == 1 or Wd < 40 or Ht < 40:  # rect/oval nhỏ
                        if shape.shape_type == 9:  # oval
                            d.ellipse([L, T, L + Wd, T + Ht], fill=col)
                        else:
                            d.rectangle([L, T, L + Wd, T + Ht], fill=col)
                    else:
                        d.rounded_rectangle([L, T, L + Wd, T + Ht], radius=12, fill=col)
            except Exception:
                pass

        # ---- table
        if shape.has_table:
            tbl = shape.table
            x, y = L, T
            for r_i, row in enumerate(tbl.rows):
                rh = px(row.height)
                x = L
                for c_i, cell in enumerate(row.cells):
                    cw = px(tbl.columns[c_i].width)
                    try:
                        rgb = cell.fill.fore_color.rgb
                        fill = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
                    except Exception:
                        fill = "#1A202E"
                    d.rectangle([x, y, x + cw, y + rh], fill=fill, outline="#2A3550")
                    txt = cell.text.replace("\n", " ")
                    if txt:
                        run = cell.text_frame.paragraphs[0].runs
                        size = run[0].font.size.pt if run and run[0].font.size else 12
                        bold = bool(run[0].font.bold) if run else False
                        try:
                            col = run[0].font.color.rgb
                            col = f"#{col[0]:02X}{col[1]:02X}{col[2]:02X}"
                        except Exception:
                            col = "#FFFFFF"
                        fnt = font(size, bold)
                        align_right = cell.text_frame.paragraphs[0].alignment == 3
                        tw = d.textlength(txt, font=fnt)
                        tx = x + cw - tw - 10 if align_right else x + 10
                        d.text((tx, y + (rh - int(size * 1.35)) / 2), txt, font=fnt, fill=col)
                    x += cw
                y += rh
            continue

        # ---- text
        if shape.has_text_frame and shape.text_frame.text.strip():
            tf = shape.text_frame
            y = T
            for p in tf.paragraphs:
                if not p.runs:
                    y += int(14 * PX_PER_IN / 72)
                    continue
                size = p.runs[0].font.size.pt if p.runs[0].font.size else 18
                bold = bool(p.runs[0].font.bold)
                spacing = p.line_spacing or 1.0
                full = "".join(r.text for r in p.runs)
                fnt = font(size, bold)
                # màu: dùng run đầu
                try:
                    rgb = p.runs[0].font.color.rgb
                    col = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
                except Exception:
                    col = "#FFFFFF"
                align = p.alignment
                for line in (full.split("\n") or [""]):
                    for sub in wrap(d, line, fnt, Wd) if Wd > 20 else [line]:
                        tw = d.textlength(sub, font=fnt)
                        tx = L
                        if align == 2:  # center
                            tx = L + (Wd - tw) / 2
                        elif align == 3:  # right
                            tx = L + Wd - tw
                        d.text((tx, y), sub, font=fnt, fill=col)
                        y += int(size * 1.28 * spacing * PX_PER_IN / 72)
                y += int(2 * PX_PER_IN / 72)
    out = OUTDIR / f"slide_{idx + 1:02d}.png"
    img.save(out)
    return out


targets = [int(a) - 1 for a in sys.argv[1:]] or range(len(prs.slides._sldIdLst))
for i in targets:
    print(render(i))
