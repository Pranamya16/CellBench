"""Write the lab-report layout as a one-page PDF with no extra dependencies.

Text uses the PDF built-in Courier font, which has exactly the same character width (0.6 em) as Courier New, so the page matches
the reference sheet's font, size and spacing. Only ASCII text is written (other characters are mapped or replaced).
"""
H_FALLBACK = 597.6
MAP = {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "≥": ">=", "≤": "<=", " ": " "}


def _rgb(hexcol):
    h = hexcol.lstrip("#")
    return " ".join(f"{int(h[i:i + 2], 16) / 255:.3f}" for i in (0, 2, 4))


def _esc(s):
    for k, v in MAP.items():
        s = s.replace(k, v)
    s = s.encode("ascii", "replace").decode("ascii")
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def render(layout, title="CellBench synthetic lab result (proof of concept)"):
    W, H = layout["page"]
    ops = []
    for it in layout["items"]:
        if it["t"] == "rect":
            y = H - it["y"] - it["h"]
            ops.append(f"{_rgb(it['fill'])} rg {it['x']} {y:.2f} {it['w']} {it['h']} re f")
            if it.get("stroke"):
                ops.append(f"{_rgb(it['stroke'])} RG 0.75 w {it['x']} {y:.2f} {it['w']} {it['h']} re S")
        else:
            s = it["s"]
            width = len(s) * 0.6 * it["size"]
            x = it["x"] - (width if it["anchor"] == "end" else width / 2 if it["anchor"] == "middle" else 0)
            ops.append(f"BT /F1 {it['size']} Tf {_rgb(it['color'])} rg {x:.2f} {H - it['y']:.2f} Td ({_esc(s)}) Tj ET")
    stream = "\n".join(ops).encode("latin-1")
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>".encode(),
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>",
            f"<< /Title ({_esc(title)}) /Producer (CellBench) >>".encode()]
    out = bytearray(b"%PDF-1.4\n")
    offs = []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for o in offs:
        out += f"{o:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R /Info 6 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)
