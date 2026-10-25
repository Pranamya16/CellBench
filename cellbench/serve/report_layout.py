"""The lab-report view as one layout description, drawn by both the web page (SVG) and the PDF export.

Formatting follows the reference result sheet exactly where it can be measured: A4 landscape (842.4 x 597.6 pt), Courier New 11 pt
everywhere and 9.1 pt for the small note, teal status box (#3B98B0, white text), header cells #E7E7E7, body cells #F4F4F4, borders #A8A8A8,
black text, 17.52 pt rows. Coordinates are in points from the TOP-LEFT of the page. Courier / Courier New are monospaced at 0.6 em,
so text widths are exact.

What is deliberately NOT here: the CBC table (red cells, platelets) and the flag boxes, because this prototype does not simulate them.
"""
import textwrap

PAGE = (842.4, 597.6)
FONT, SIZE, SMALL = 11.0, 11.0, 9.1
ROW = 17.52
TEAL, HEAD, BODY, EDGE, INK, WHITE = "#3B98B0", "#E7E7E7", "#F4F4F4", "#A8A8A8", "#000000", "#FFFFFF"
CW = 0.6                              # Courier advance width as a fraction of the font size
COL_L, COL_M, COL_R = 32.88, 278.7, 523.9
W_L, W_M, W_R = 233.0, 232.0, 284.0
LABEL = {"neutrophil": "NEUT", "lymphocyte": "LYMPH", "monocyte": "MONO", "eosinophil": "EO", "ig": "IG"}


class Sheet:
    def __init__(self):
        self.items = []

    def rect(self, x, y, w, h, fill, stroke=None):
        self.items.append(dict(t="rect", x=round(x, 2), y=round(y, 2), w=round(w, 2), h=round(h, 2), fill=fill, stroke=stroke))

    def text(self, x, y, s, size=SIZE, color=INK, anchor="start"):
        """y is the baseline, measured from the top."""
        self.items.append(dict(t="text", x=round(x, 2), y=round(y, 2), s=s, size=size, color=color, anchor=anchor))

    def cell(self, x, y, w, s, fill=BODY, align="left", color=INK, size=SIZE, h=ROW):
        self.rect(x, y, w, h, fill, EDGE)
        pad = 4.0
        base = y + h / 2 + size * 0.34
        if align == "center":
            self.text(x + w / 2, base, s, size, color, "middle")
        elif align == "right":
            self.text(x + w - pad, base, s, size, color, "end")
        else:
            self.text(x + pad, base, s, size, color)

    def table(self, x, y, widths, header, rows, aligns):
        for i, (w, h) in enumerate(zip(widths, header)):
            self.cell(x + sum(widths[:i]), y, w, h, HEAD, "center")
        for r, row in enumerate(rows):
            for i, (w, v) in enumerate(zip(widths, row)):
                self.cell(x + sum(widths[:i]), y + ROW * (r + 1), w, v, BODY, aligns[i])
        return y + ROW * (len(rows) + 1)

    def box(self, x, y, w, h, lines, size=SIZE):
        self.rect(x, y, w, h, BODY, EDGE)
        for k, ln in enumerate(lines):
            self.text(x + 4, y + 4 + size + k * (size + 2.6), ln, size)


def wrap(s, width_pt, size=SIZE):
    chars = max(8, int((width_pt - 8) / (CW * size)))
    return textwrap.wrap(s, chars) or [""]


def build(r):
    """r: dict(patient=..., text=..., source=...). Returns {"page": [w, h], "items": [...]}."""
    p, lab = r["patient"], r["patient"]["lab"]
    s = Sheet()
    # status bar: teal status box, grey cells (same three-cell layout as the reference sheet)
    up = bool(p.get("upload"))
    s.rect(32.88, 33.1, 118.37, 17.52, TEAL)
    s.text(32.88 + 118.37 / 2, 33.1 + 12.8, "Uploaded" if up else "Synthetic", color=WHITE, anchor="middle")
    s.cell(151.73, 33.1, 262.39, "Proof of concept", HEAD, "center")
    third = f"File: {p['name']}  {r.get('stamp', '')}" if up else f"Sample #{p['seed']}  noise: {p['noise']}  {r.get('stamp', '')}"
    s.cell(415.0, 33.1, 393.0, third.rstrip(), HEAD, "center")
    # tab strip, as on the analyser screen (decorative: only the layout is borrowed)
    x = 32.88
    tabs = [("Main", 75.3), ("Graph", 86.8), ("Cumulative", 151.0), ("Q-Flag", 111.6), ("Service", 118.9), ("User", 97.4), ("Lab. Only", 136.8)]
    k = 775.1 / sum(w for _, w in tabs)
    for name, w in tabs:
        s.cell(x, 68.0, w * k, name, HEAD, "center")
        x += w * k
    y0 = 105.0
    # left: differential
    s.text(COL_L, y0 + 4, "DIFF")
    rows = [["WBC", f"{lab['wbc']:.2f}", "10^3/uL"]]
    rows += [[LABEL[n] + "#", f"{a:.2f}", "10^3/uL"] for n, a in zip(lab["groups"], lab["absolute"])]
    rows += [[LABEL[n] + "%", f"{v:.1f}", "%"] for n, v in zip(lab["groups"], lab["pct"])]
    wd = [76.0, 84.0, 73.0]
    y1 = s.table(COL_L, y0 + 12, wd, ["Item", "Data", "Unit"], rows, ["left", "right", "left"])
    # middle: neutrophil research parameters, then what was planted
    s.text(COL_M, y0 + 4, "Research parameters")
    prm = [["NE-SSC", f"{lab['ne_ssc']:.1f}", "ch"], ["NE-SFL", f"{lab['ne_sfl']:.1f}", "ch"], ["NE-FSC", f"{lab['ne_fsc']:.1f}", "ch"],
           ["NE events", f"{lab['ne_events']}", ""], ["Events", f"{lab['events']}", ""]]
    ym = s.table(COL_M, y0 + 12, [76.0, 84.0, 72.0], ["Item", "Data", "Unit"], prm, ["left", "right", "left"])
    s.text(COL_M, ym + 22, "File" if up else "Known truth (demo only)")
    if up:
        truth = f"{p['name']}: {lab['events']} cells. The truth is not known for an uploaded file."
        if p.get("warnings"):
            truth += " Note: " + " ".join(p["warnings"])
    elif p["signal"]:
        truth = ("Every neutrophil was made %d%% less granular." % round(p["drop"] * 100) if p["kind"] == "shift" else
                 "%d%% of neutrophils were made %d%% less granular (%s)." % (round(p["frac"] * 100), round(p["drop"] * 100), p["kind"]))
        truth += f" Planted cells: {p['planted']}."
    else:
        truth = "Normal patient: nothing was planted."
    s.box(COL_M, ym + 30, W_M, 78.0, wrap(truth, W_M))
    # right: plain-language note in a flag-style box
    s.text(COL_R, y0 + 4, "In plain words")
    note = wrap(r["text"], W_R)
    s.box(COL_R, y0 + 12, W_R, max(120.0, 8 + len(note) * (SIZE + 2.6)), note)
    # bottom-left: AI insights, laid out like the reference sheet's Disease / Percentage table
    ya = y1 + 24
    s.text(COL_L, ya, "AI insights")
    t = [["Report averages", f"{round(p['score_report'] * 100)}"], ["Full cell data", f"{round(p['score_cells'] * 100)}"]]
    yb = s.table(COL_L, ya + 8, [176.0, 100.0], ["Hypogranular neutrophils", "Score (0-100)"], t, ["left", "center"])
    s.text(COL_L, yb + 13, "Detector score on an uploaded file." if up else "Detector score on a synthetic patient.", SMALL)
    s.text(COL_L, yb + 25, "Illustrative only. Not a diagnosis.", SMALL)
    # footer: the proof-of-concept statement
    foot = ("Simulated, synthetic data. Proof of concept only: no real patient, no clinical validation. The score shows whether a planted pattern "
            "is visible in made-up cells and is not a chance of disease. WBC is a made-up total. The differential comes from a simple automatic "
            "gate, so the eosinophil and immature-cell lines are approximate.")
    if up:
        foot = ("Uploaded file. Proof of concept only: not for real patient data, no clinical validation. The models were trained only on simulated "
                "patients, so scores for other kinds of data are not meaningful, and a score is never a chance of disease. WBC is a made-up total. "
                "The differential comes from a simple automatic gate, so the eosinophil and immature-cell lines are approximate.")
    for k, ln in enumerate(wrap(foot, 776.0, SMALL)):
        s.text(COL_L, 560.0 + k * 12.0, ln, SMALL)
    return dict(page=list(PAGE), items=s.items, font="Courier New")
