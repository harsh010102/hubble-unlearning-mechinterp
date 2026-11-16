"""TUM Corporate-Design deck builder.

Builds on the REAL TUM template masters/layouts extracted from `RESPAI presentation.pptx`
(the seminar deck graded 1.3), so the TUM logo, footer, fonts (Arial), theme colors, and
chapter-divider styling are pixel-identical. We start from the template, strip its slides,
and add fresh ones through the captured layouts.

Slide size: 10.0 x 5.625 in (16:9). House style: assertion-evidence (a sentence headline that
states the insight + one big visual), one idea per slide, no em dashes, American spelling.
"""
import copy
from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image

TEMPLATE = "RESPAI presentation.pptx"
UHRENTURM = "deck_assets/title_image2.jpeg"

# --- official TUM theme colors (extracted from the template theme) ---
TUM_BLUE   = RGBColor(0x00, 0x65, 0xBD)   # lt2  primary
TUM_DARK   = RGBColor(0x00, 0x33, 0x59)   # dk2
TUM_D2     = RGBColor(0x00, 0x52, 0x93)   # accent1
TUM_MED    = RGBColor(0x64, 0xA0, 0xC8)   # accent2
TUM_LT     = RGBColor(0x98, 0xC6, 0xEA)   # accent3
TUM_GREEN  = RGBColor(0xA2, 0xAD, 0x00)   # accent4
TUM_ORANGE = RGBColor(0xE3, 0x72, 0x22)   # accent5
TUM_BEIGE  = RGBColor(0xDA, 0xD7, 0xCB)   # accent6
BLACK = RGBColor(0, 0, 0)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GRAY  = RGBColor(0x6A, 0x75, 0x7E)
FONT  = "Arial"

SW, SH = 10.0, 5.625          # slide size in inches
LEFT   = 0.34                 # TUM left margin
CW     = 9.31                 # TUM content width


def _nodash(t):
    t = str(t).replace(" — ", ", ").replace("—", ", ").replace(" – ", ", ")
    return t.replace("’", "'").replace("“", '"').replace("”", '"')


def _style(run, size, color=BLACK, bold=False, italic=False):
    f = run.font
    f.size = Pt(size); f.bold = bold; f.italic = italic
    f.color.rgb = color; f.name = FONT


class Deck:
    def __init__(self, footer_mid):
        self.prs = Presentation(TEMPLATE)
        self._capture_layouts()
        self._strip_slides()
        self._set_footer(f"Harsh Parikh  |  {footer_mid}  |  Chair of Responsible Data Science")

    # ---------- template plumbing ----------
    def _capture_layouts(self):
        by = {}
        for m in self.prs.slide_masters:
            for l in m.slide_layouts:
                by.setdefault(l.name, l)
        self.L_title   = by["Start"]
        self.L_content = by["1_Inhalt + Text"]
        self.L_image   = by["1_Bilder formatfüllend"]
        self.L_divider = by["Kapiteltrenner"]

    def _strip_slides(self):
        # remove both the sldId entries AND their relationships so the old slide
        # parts are truly dropped from the package (else new slides collide on name).
        prs_part = self.prs.part
        lst = self.prs.slides._sldIdLst
        for sid in list(lst):
            rId = sid.get(qn("r:id"))
            lst.remove(sid)
            if rId and rId in prs_part.rels:
                prs_part.drop_rel(rId)

    def _set_footer(self, text):
        for m in self.prs.slide_masters:
            holders = list(m.placeholders)
            for l in m.slide_layouts:
                holders += list(l.placeholders)
            for ph in holders:
                if ph.placeholder_format.type == 15 and ph.has_text_frame:   # FOOTER
                    ph.text_frame.paragraphs[0].text = text
                    for r in ph.text_frame.paragraphs[0].runs:
                        r.font.name = FONT

    def _add(self, layout):
        return self.prs.slides.add_slide(layout)

    def _ph(self, slide, idx):
        for ph in slide.placeholders:
            if ph.placeholder_format.idx == idx:
                return ph
        return None

    def _drop(self, slide, idx):
        ph = self._ph(slide, idx)
        if ph is not None:
            ph._element.getparent().remove(ph._element)

    def _put(self, tf, text, size, color, bold=False, italic=False, first=False,
             space_before=0, align=None):
        """Add a paragraph and style the SAME run that carries the text (robust)."""
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        if space_before:
            p.space_before = Pt(space_before)
        if align is not None:
            p.alignment = align
        r = p.add_run(); r.text = _nodash(text); _style(r, size, color, bold, italic)
        return p

    def _flat(self, shape):
        """Remove theme style (shadow/effect refs) so a drawn shape is flat."""
        el = shape._element
        st = el.find(qn("p:style"))
        if st is not None:
            el.remove(st)
        shape.shadow.inherit = False

    # ---------- primitive drawing ----------
    def tbox(self, slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
        return tf

    def line(self, tf, text, size, color=BLACK, bold=False, italic=False,
             first=False, bullet=False, space=6, align=PP_ALIGN.LEFT):
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(space)
        r = p.add_run(); r.text = ("•  " + _nodash(text)) if bullet else _nodash(text)
        _style(r, size, color, bold, italic)
        return p

    # ---------- slide builders ----------
    def title(self, title, subtitle, presenter, group, date):
        s = self._add(self.L_title)
        for idx in (0, 10, 12):
            self._drop(s, idx)                      # draw our own textboxes (deterministic)
        tf = self.tbox(s, LEFT, 0.92, 5.0, 1.35)
        self._put(tf, title, 30, BLACK, first=True)
        self._put(tf, subtitle, 16, TUM_BLUE, space_before=6)
        cf = self.tbox(s, LEFT, 2.55, 4.9, 1.9)
        self._put(cf, presenter, 12, BLACK, first=True)
        self._put(cf, group, 12, BLACK, bold=True)
        self._put(cf, date, 12, BLACK, bold=True, space_before=10)
        s.shapes.add_picture(UHRENTURM, Inches(5.44), Inches(1.61), Inches(4.18), Inches(3.65))
        return s

    def divider(self, title, subtitle=None):
        s = self._add(self.L_divider)
        tf = self._ph(s, 0).text_frame; tf.clear(); tf.word_wrap = True
        self._put(tf, title, 30, WHITE, first=True)
        if subtitle:
            self._put(tf, subtitle, 16, TUM_LT)
        return s

    def closing(self, big="Thank you!", sub="Q&A"):
        return self.divider(big, sub)

    def content(self, headline, eyebrow=None, hsize=None):
        """Content slide: black assertion headline on top, optional blue eyebrow below it
        (TUM order). Placeholders removed; headline drawn as a textbox so one big visual fits."""
        s = self._add(self.L_content)
        for idx in (0, 1, 14):
            self._drop(s, idx)
        L = len(_nodash(headline))
        HW = 8.45                                  # keep headline clear of the top-right TUM logo
        if eyebrow:
            hs = hsize or (22 if L <= 52 else 18)
            tf = self.tbox(s, LEFT, 0.32, HW, 0.55)
            self._put(tf, headline, hs, BLACK, bold=True, first=True)
            ef = self.tbox(s, LEFT, 0.92, CW, 0.42)
            self._put(ef, eyebrow, 14, TUM_BLUE, bold=True, first=True)
        else:
            hs = hsize or (22 if L <= 60 else (18 if L <= 92 else 15))
            tf = self.tbox(s, LEFT, 0.30, HW, 0.95)
            self._put(tf, headline, hs, BLACK, bold=True, first=True)
        return s

    # ---------- content helpers ----------
    def image(self, slide, path, x=None, y=1.55, maxw=9.31, maxh=3.55, center=True):
        iw, ih = Image.open(path).size; ar = iw / ih
        w = maxw; h = w / ar
        if h > maxh:
            h = maxh; w = h * ar
        if x is None:
            x = LEFT + (maxw - w) / 2 if center else LEFT
        slide.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))
        return x, y, w, h

    def caption(self, slide, text, y=5.02, x=LEFT, w=CW, align=PP_ALIGN.CENTER):
        tf = self.tbox(slide, x, y, w, 0.32)
        self.line(tf, text, 11, GRAY, italic=True, first=True, align=align)

    def callout(self, slide, x, y, w, h, lines, border=TUM_BLUE, fill=WHITE, title=None):
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                     Inches(x), Inches(y), Inches(w), Inches(h))
        box.fill.solid(); box.fill.fore_color.rgb = fill
        box.line.color.rgb = border; box.line.width = Pt(2.0); self._flat(box)
        tf = box.text_frame; tf.word_wrap = True
        tf.margin_left = Inches(0.14); tf.margin_right = Inches(0.14)
        tf.margin_top = Inches(0.10); tf.margin_bottom = Inches(0.10)
        tf.vertical_anchor = MSO_ANCHOR.TOP
        first = True
        if title:
            self.line(tf, title, 13, border, bold=True, first=True, space=4); first = False
        for ln in lines:
            self.line(tf, ln, 12, BLACK, first=first, bullet=True, space=4); first = False
        return box

    def bullets(self, slide, items, x=LEFT, y=1.6, w=CW, size=14, space=9):
        tf = self.tbox(slide, x, y, w, 5.1 - y)
        for i, it in enumerate(items):
            txt, kind = it if isinstance(it, tuple) else (it, "b")
            if kind == "h":
                self.line(tf, txt, size + 2, BLACK, bold=True, first=(i == 0), space=5)
            elif kind == "p":
                self.line(tf, txt, size, BLACK, first=(i == 0), space=space)
            elif kind == "i":
                self.line(tf, txt, size - 1, GRAY, first=(i == 0), bullet=True, space=5)
            else:
                self.line(tf, txt, size, BLACK, first=(i == 0), bullet=True, space=space)
        return tf

    def chevron_bar(self, slide, labels, current, y=2.4, h=0.62, pad=0.08):
        """Native chevron progress bar (current filled TUM blue, rest light-blue outline)."""
        n = len(labels); total = CW; cw = (total - pad * (n - 1)) / n
        fs = 10 if n <= 5 else (9.5 if n == 6 else 8)
        x = LEFT
        for i, lab in enumerate(labels):
            ch = slide.shapes.add_shape(MSO_SHAPE.CHEVRON, Inches(x), Inches(y), Inches(cw), Inches(h))
            try:
                ch.adjustments[0] = 0.28          # shallower arrowhead -> more room for the label
            except Exception:
                pass
            on = (i == current)
            ch.fill.solid(); ch.fill.fore_color.rgb = TUM_BLUE if on else WHITE
            ch.line.color.rgb = TUM_BLUE; ch.line.width = Pt(1.25); self._flat(ch)
            tf = ch.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf.margin_left = Pt(2); tf.margin_right = Pt(2)
            p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
            _style(p.add_run(), fs, WHITE if on else TUM_D2, bold=on); p.runs[0].text = _nodash(lab)
            x += cw + pad
        return y + h

    def agenda(self, labels, current=-1, title="Agenda"):
        s = self.content(title)
        self.chevron_bar(s, labels, current, y=2.35, h=0.7)
        return s

    def save(self, path):
        self.prs.save(path)
        return path
