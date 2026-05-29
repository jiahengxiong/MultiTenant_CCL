from pathlib import Path
from xml.sax.saxutils import escape


OUT = Path(__file__).with_name("collective_task_dag_corrected.svg")
W, H = 1920, 1080


def svg_text(x, y, text, size=24, weight="400", fill="#111827", anchor="start", italic=False):
    style = f"font-size:{size}px;font-weight:{weight};fill:{fill};"
    if italic:
        style += "font-style:italic;"
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" style="{style}">{escape(text)}</text>'


def rect(x, y, w, h, fill, stroke="#1f4e8c", sw=2, rx=12, opacity=1.0):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def line(x1, y1, x2, y2, stroke="#111827", sw=2, dash=None, marker=True):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    marker_attr = ' marker-end="url(#arrow)"' if marker else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{stroke}" stroke-width="{sw}"{dash_attr}{marker_attr}/>'
    )


def circle(cx, cy, r, fill, stroke="#17406f", sw=2):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def node(cx, cy, label, fill):
    return circle(cx, cy, 20, fill) + svg_text(cx, cy + 8, label, 18, "700", "#111827", "middle")


def arrow_between(x1, y1, x2, y2, color="#111827", dash=None, sw=2):
    return line(x1, y1, x2, y2, color, sw, dash, True)


rank_fill = {
    "0": "#ffbf8a",
    "1": "#b8e7a8",
    "2": "#a9d4ff",
    "3": "#ffe28a",
}


def transmission_chain(x, y, seq, label):
    parts = [svg_text(x - 34, y + 7, label, 18, "700", "#0b3a78", "end")]
    step = 82
    for i, item in enumerate(seq):
        cx = x + i * step
        parts.append(node(cx, y, item.split("->")[0], rank_fill[item.split("->")[0]]))
        parts.append(svg_text(cx, y + 37, item, 13, "700", "#1f2937", "middle"))
        if i < len(seq) - 1:
            parts.append(arrow_between(cx + 22, y, cx + step - 22, y, "#111827", None, 2))
    return "\n".join(parts)


def panel_header(x, y, w, title):
    return rect(x, y, w, 46, "#063b79", "#063b79", 0, 8) + svg_text(
        x + w / 2, y + 31, title, 23, "800", "#ffffff", "middle"
    )


def pill(x, y, w, h, text, fill, stroke):
    return rect(x, y, w, h, fill, stroke, 2, 12) + svg_text(
        x + w / 2, y + h / 2 + 9, text, 22, "800", "#111827", "middle"
    )


def program_box(x, y, w, h, title, subtitle, fill, stroke):
    return (
        rect(x, y, w, h, fill, stroke, 2, 12)
        + svg_text(x + w / 2, y + 33, title, 22, "800", "#111827", "middle")
        + svg_text(x + w / 2, y + 61, subtitle, 22, "800", "#111827", "middle")
    )


def build():
    p = []
    p.append(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
  <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
    <stop offset="0%" stop-color="#ffffff"/>
    <stop offset="100%" stop-color="#eef5ff"/>
  </linearGradient>
  <linearGradient id="callout" x1="0" x2="1">
    <stop offset="0%" stop-color="#fff7df"/>
    <stop offset="100%" stop-color="#fff1bd"/>
  </linearGradient>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/>
  </marker>
  <style>
    text {{ font-family: Avenir Next, Aptos, Helvetica, Arial, sans-serif; }}
  </style>
</defs>
<rect width="{W}" height="{H}" fill="url(#bg)"/>
<rect x="0" y="0" width="132" height="118" fill="#ffffff"/>
<path d="M35 36 L85 82 L101 64 L49 20 Z" fill="#f59e0b"/>
<rect x="132" y="0" width="4" height="118" fill="#063b79"/>
<rect x="0" y="118" width="{W}" height="14" fill="#063b79"/>
<rect x="0" y="132" width="420" height="12" fill="#d8e4f2"/>
<rect x="1660" y="0" width="260" height="18" fill="#063b79"/>
'''
    )
    p.append(svg_text(150, 48, "Collective-Aware Rank Mapping for Multi-Tenant Training", 38, "900", "#073b78"))
    p.append(svg_text(150, 94, "From Collective Semantics to Task-Level Communication Constraints", 32, "800", "#073b78"))

    p.append(rect(235, 156, 1450, 72, "url(#callout)", "#f0bd53", 2, 12))
    p.append(svg_text(960, 187, "We compile collective semantics into transmission tasks,", 31, "900", "#111827", "middle"))
    p.append(svg_text(960, 221, "then add algorithmic dependencies, port-order constraints, and inter-collective release gates.", 30, "900", "#111827", "middle"))

    # Panel 1
    x1, y1, w1, h1 = 38, 252, 332, 620
    p.append(panel_header(x1, y1, w1, "1) Collective Program"))
    p.append(svg_text(x1, y1 + 82, "Per-tenant communication program", 20, "800", "#111827", italic=True))
    p.append(rect(x1 + 30, y1 + 104, w1 - 60, 420, "#ffffff", "#2f6fb3", 2, 28))
    p.append('<rect x="68" y="364" width="272" height="420" rx="28" fill="none" stroke="#2f6fb3" stroke-width="3" stroke-dasharray="10 10"/>')
    p.append(program_box(100, 350, 208, 76, "AllReduce", "size = S1", "#ffd6b6", "#d97706"))
    p.append(svg_text(204, 458, "gap_after = g1", 20, "800", "#ef1717", "middle"))
    p.append(svg_text(204, 503, "...", 33, "900", "#111827", "middle"))
    p.append(program_box(100, 538, 208, 76, "AllGather", "size = S2", "#cdf4c2", "#4c9a42"))
    p.append(svg_text(204, 647, "gap_after = g2", 20, "800", "#ef1717", "middle"))
    p.append(svg_text(204, 693, "...", 33, "900", "#111827", "middle"))
    p.append(program_box(100, 728, 208, 76, "AllToAll", "size = S3", "#cfe5ff", "#2f6fb3"))
    p.append(svg_text(56, 898, "Specifies what and pacing;", 20, "600", "#111827"))
    p.append(svg_text(56, 930, "not an execution schedule.", 20, "800", "#0b4db3"))

    # Panel 2
    x2, y2, w2 = 418, 252, 475
    p.append(panel_header(x2, y2, w2, "2) Per-Collective Expansion"))
    p.append(svg_text(x2 + w2 / 2, y2 + 82, "Example: Ring AllReduce (N = 4)", 20, "800", "#111827", "middle", True))
    p.append(rect(x2, y2 + 104, w2, 278, "#ffffff", "#5b8ec9", 2, 14))
    p.append(svg_text(x2 + w2 / 2, y2 + 132, "Reduce-Scatter: 4 chunks, 3 steps", 19, "800", "#0b4db3", "middle"))
    rs = [
        ("C0", ["0->1", "1->2", "2->3"]),
        ("C1", ["1->2", "2->3", "3->0"]),
        ("C2", ["2->3", "3->0", "0->1"]),
        ("C3", ["3->0", "0->1", "1->2"]),
    ]
    for i, (lab, seq) in enumerate(rs):
        p.append(transmission_chain(x2 + 125, y2 + 172 + i * 48, seq, lab))
    p.append(rect(x2, y2 + 406, w2, 250, "#ffffff", "#5b8ec9", 2, 14))
    p.append(svg_text(x2 + w2 / 2, y2 + 434, "AllGather: 4 chunks, 3 steps", 19, "800", "#0b4db3", "middle"))
    ag = [
        ("C0", ["3->0", "0->1", "1->2"]),
        ("C1", ["0->1", "1->2", "2->3"]),
        ("C2", ["1->2", "2->3", "3->0"]),
        ("C3", ["2->3", "3->0", "0->1"]),
    ]
    for i, (lab, seq) in enumerate(ag):
        p.append(transmission_chain(x2 + 125, y2 + 474 + i * 46, seq, lab))

    # Arrows between panels
    p.append(arrow_between(382, 575, 410, 575, "#063b79", None, 8))
    p.append(arrow_between(907, 575, 935, 575, "#063b79", None, 8))
    p.append(arrow_between(1462, 575, 1490, 575, "#063b79", None, 8))

    # Panel 3
    x3, y3, w3 = 950, 252, 500
    p.append(panel_header(x3, y3, w3, "3) Task DAG + Constraints"))
    p.append(svg_text(x3 + w3 / 2, y3 + 82, "Add resource order and release timing", 20, "800", "#111827", "middle", True))
    p.append(rect(x3, y3 + 104, w3, 524, "#ffffff", "#111827", 2, 18))
    p.append(arrow_between(x3 + 32, y3 + 138, x3 + 100, y3 + 138, "#111827", None, 2))
    p.append(svg_text(x3 + 130, y3 + 145, "Algorithmic dependency", 18, "500"))
    p.append(arrow_between(x3 + 32, y3 + 168, x3 + 100, y3 + 168, "#22a447", "7 7", 2))
    p.append(svg_text(x3 + 130, y3 + 175, "Port-order dependency (same first-hop port)", 18, "500"))
    p.append(line(x3 + 32, y3 + 198, x3 + 100, y3 + 198, "#ef1717", 2, "7 7", False))
    p.append(circle(x3 + 32, y3 + 198, 4, "#ffffff", "#ef1717", 2))
    p.append(circle(x3 + 100, y3 + 198, 4, "#ffffff", "#ef1717", 2))
    p.append(svg_text(x3 + 130, y3 + 205, "Release gate: previous op finish + gap_after", 18, "500"))

    gx, gy, dx, dy = x3 + 88, y3 + 258, 100, 72
    labels = [
        ["0->1", "1->2", "2->3", "3->0"],
        ["1->2", "2->3", "3->0", "0->1"],
        ["2->3", "3->0", "0->1", "1->2"],
        ["3->0", "0->1", "1->2", "2->3"],
    ]
    colors = ["#ffbf8a", "#b8e7a8", "#a9d4ff", "#ffe28a"]
    for r in range(4):
        for c in range(4):
            cx, cy = gx + c * dx, gy + r * dy
            p.append(circle(cx, cy, 24, colors[(r + c) % 4], "#17406f", 2))
            p.append(svg_text(cx, cy + 7, labels[r][c], 15, "800", "#111827", "middle"))
            if c < 3:
                p.append(arrow_between(cx + 26, cy, cx + dx - 28, cy, "#111827", None, 2))
            if r < 3:
                p.append(arrow_between(cx, cy + 26, cx, cy + dy - 28, "#111827", None, 2))
    p.append(arrow_between(gx + 24, gy + 18, gx + dx - 22, gy + dy - 18, "#22a447", "7 7", 2))
    p.append(arrow_between(gx + dx + 24, gy + 18, gx + 2 * dx - 22, gy + dy - 18, "#22a447", "7 7", 2))
    p.append(arrow_between(gx + 2 * dx + 24, gy + 2 * dy + 18, gx + 3 * dx - 22, gy + 3 * dy - 18, "#22a447", "7 7", 2))
    # Release bracket to next collective.
    ybr = gy + 4 * dy + 12
    p.append(line(gx, ybr, gx + 3 * dx, ybr, "#ef1717", 2, "8 8", False))
    for c in range(4):
        p.append(line(gx + c * dx, gy + 3 * dy + 30, gx + c * dx, ybr, "#ef1717", 2, "8 8", False))
        p.append(circle(gx + c * dx, ybr, 5, "#ffffff", "#ef1717", 2))
    p.append(svg_text(gx + 3 * dx + 25, ybr + 6, "gap_after", 18, "800", "#ef1717"))
    for c in range(3):
        cx = gx + c * dx + 42
        p.append(arrow_between(cx, ybr + 12, cx, ybr + 52, "#111827", None, 2))
        p.append(circle(cx, ybr + 76, 20, "#d9d9d9", "#6b7280", 2))
    p.append(svg_text(gx, ybr + 96, "next op initial tasks", 17, "600", "#111827"))
    # Panel 4
    x4, y4, w4 = 1500, 252, 376
    p.append(panel_header(x4, y4, w4, "4) Used by Solvers"))
    cards = [
        (["ILP Oracle"], "mapping + scheduling", "tree"),
        (["Rank Mapping", "Search"], "remap / swap", "magnify"),
        (["Contention", "Estimation"], "link load + overlap", "bars"),
        (["Simulator"], "runtime execution", "server"),
    ]
    for i, (title_lines, sub, kind) in enumerate(cards):
        cy = y4 + 104 + i * 138
        p.append(rect(x4, cy, w4, 110, "#f5f0ff", "#b9a7e0", 2, 16))
        ix, iy = x4 + 58, cy + 55
        if kind == "tree":
            p.append(circle(ix, iy - 28, 8, "#ffffff", "#111827", 4))
            p.append(circle(ix - 28, iy + 8, 8, "#ffffff", "#111827", 4))
            p.append(circle(ix + 28, iy + 8, 8, "#ffffff", "#111827", 4))
            p.append(circle(ix - 48, iy + 40, 8, "#ffffff", "#111827", 4))
            p.append(circle(ix + 48, iy + 40, 8, "#ffffff", "#111827", 4))
            p.append(line(ix, iy - 20, ix - 28, iy, "#111827", 4, None, False))
            p.append(line(ix, iy - 20, ix + 28, iy, "#111827", 4, None, False))
            p.append(line(ix - 28, iy + 16, ix - 48, iy + 32, "#111827", 4, None, False))
            p.append(line(ix + 28, iy + 16, ix + 48, iy + 32, "#111827", 4, None, False))
        elif kind == "magnify":
            p.append(circle(ix - 8, iy - 8, 30, "#ffffff", "#111827", 5))
            p.append(line(ix + 16, iy + 16, ix + 55, iy + 55, "#111827", 7, None, False))
            p.append(line(ix - 25, iy + 2, ix - 10, iy - 12, "#22a447", 4, None, False))
            p.append(line(ix - 10, iy - 12, ix + 8, iy + 4, "#22a447", 4, None, False))
        elif kind == "bars":
            for b, bh in enumerate([28, 48, 70]):
                p.append(rect(ix - 32 + b * 28, iy + 36 - bh, 18, bh, "#111827", "#111827", 0, 0))
        else:
            for b in range(3):
                p.append(rect(ix - 38, iy - 32 + b * 28, 76, 20, "#ffffff", "#111827", 3, 3))
                p.append(circle(ix - 22, iy - 22 + b * 28, 3, "#111827", "#111827", 0))
        if len(title_lines) == 1:
            p.append(svg_text(x4 + 252, cy + 45, title_lines[0], 22, "900", "#111827", "middle"))
            p.append(svg_text(x4 + 252, cy + 76, sub, 19, "600", "#111827", "middle"))
        else:
            p.append(svg_text(x4 + 252, cy + 35, title_lines[0], 22, "900", "#111827", "middle"))
            p.append(svg_text(x4 + 252, cy + 61, title_lines[1], 22, "900", "#111827", "middle"))
            p.append(svg_text(x4 + 252, cy + 91, sub, 19, "600", "#111827", "middle"))

    p.append(rect(200, 960, 1520, 66, "#eaf3ff", "#0b4db3", 2, 12))
    p.append(svg_text(960, 988, "The representation captures collective semantics plus resource-level ordering,", 24, "900", "#111827", "middle"))
    p.append(svg_text(960, 1018, "enabling contention-aware rank mapping, ILP validation, and simulator evaluation.", 24, "900", "#111827", "middle"))
    p.append("</svg>")
    return "\n".join(p)


if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf-8")
    print(OUT)
