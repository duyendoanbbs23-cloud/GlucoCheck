"""Dependency-free SVG charts. Each function returns an HTML string for st.markdown(unsafe_allow_html=True).

Colours are deliberately calm (teal / indigo / amber): no alarming reds, because the app talks to
non-experts about health risk.
"""
from html import escape

INK, MUTED, GRID, TRACK = "#1F2A44", "#6B7690", "#E3E8F2", "#EDF1F8"
TEAL, INDIGO, CORAL, AMBER = "#1F9D8B", "#4F5BD5", "#E8743B", "#F2B84B"
FONT = "font-family:Inter,system-ui,-apple-system,'Segoe UI',Roboto,sans-serif"


def _svg(w, h, body, label=""):
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{escape(label)}" '
            f'style="{FONT};max-width:{w}px;display:block;margin:auto">{body}</svg>')


def _t(x, y, s, size=13, fill=INK, anchor="start", weight=400, halo=False):
    # halo = thin page-coloured outline so text stays readable where a guide line crosses it
    h = ' paint-order="stroke" stroke="#F6F8FC" stroke-width="5" stroke-linejoin="round"' if halo else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{h}>{escape(str(s))}</text>')


def meter(p, avg, thr, labels, vmax=0.6):
    """Horizontal risk meter: calm 'below threshold' zone, 'above threshold' zone, and three markers."""
    W, H, x0, x1, y = 640, 150, 24, 616, 62
    sx = lambda v: x0 + min(max(v, 0), vmax) / vmax * (x1 - x0)
    clipped = p > vmax
    body = (f'<rect x="{x0}" y="{y}" width="{sx(thr) - x0:.1f}" height="22" rx="11" fill="#CDEEE8"/>'
            f'<rect x="{sx(thr):.1f}" y="{y}" width="{x1 - sx(thr):.1f}" height="22" rx="11" fill="#D9DDFB"/>'
            f'<rect x="{sx(thr) - 11:.1f}" y="{y}" width="22" height="22" fill="#D9DDFB"/>')
    # threshold (dashed) below the track
    body += (f'<line x1="{sx(thr):.1f}" x2="{sx(thr):.1f}" y1="{y - 6}" y2="{y + 56}" stroke="{INDIGO}" '
             f'stroke-width="2" stroke-dasharray="4 3"/>')
    body += _t(min(sx(thr) + 6, W - 150), y + 72, f"{labels['thr']} {thr * 100:.0f}%", 12, INDIGO, "start", 600, halo=True)
    # average tick; label sits to the left of its tick and the threshold label to the right of its line,
    # so the two never collide even when the values are close (e.g. 14% vs 15%)
    body += (f'<line x1="{sx(avg):.1f}" x2="{sx(avg):.1f}" y1="{y - 2}" y2="{y + 26}" stroke="{MUTED}" stroke-width="2"/>')
    body += _t(sx(avg) - 6, y + 44, f"{labels['avg']} {avg * 100:.0f}%", 12, MUTED, "end", 400, halo=True)
    # you
    cx = sx(p)
    body += (f'<circle cx="{cx:.1f}" cy="{y + 11}" r="14" fill="#fff" stroke="{INK}" stroke-width="3"/>'
             f'<circle cx="{cx:.1f}" cy="{y + 11}" r="5" fill="{INK}"/>')
    anchor = "end" if cx > W - 90 else "start" if cx < 90 else "middle"
    body += _t(cx, y - 14, f"{labels['you']} {p * 100:.0f}%" + ("+" if clipped else ""), 15, INK, anchor, 700)
    return _svg(W, H, body, f"{labels['you']} {p * 100:.0f}%")


def contrib_bars(items, up_label, down_label, axis_label):
    """items: [(label, value)] already sorted by |value|. Diverging bars around zero."""
    if not items:
        return ""
    W, x0, x1, row = 640, 24, 616, 52
    mid = (x0 + x1) / 2
    m = max(abs(v) for _, v in items) or 1.0
    half = (x1 - x0) / 2 - 8
    H = 30 + row * len(items) + 34
    body = f'<line x1="{mid}" x2="{mid}" y1="14" y2="{H - 38}" stroke="{GRID}" stroke-width="2"/>'
    for i, (lab, v) in enumerate(items):
        y = 30 + i * row
        w = max(abs(v) / m * half, 3)
        col = CORAL if v > 0 else TEAL
        x = mid if v > 0 else mid - w
        body += _t(mid + (-8 if v > 0 else 8), y, lab, 13.5, INK, "end" if v > 0 else "start", 600)
        body += f'<rect x="{x:.1f}" y="{y + 8}" width="{w:.1f}" height="16" rx="8" fill="{col}"/>'
    body += _t(x0, H - 8, "← " + down_label, 12, TEAL, "start", 600)
    body += _t(x1, H - 8, up_label + " →", 12, CORAL, "end", 600)
    return _svg(W, H, body, axis_label)


def line_chart(xs, ys, marks, xlab, ylab, xfmt=lambda v: f"{v:.0f}%", w=640, h=300):
    """marks: [(x, label, colour)] vertical guide lines. xs in 0-1 (probability), ys cost."""
    L, R, T, B = 58, 16, 26, 52
    pw, ph = w - L - R, h - T - B
    x_lo, x_hi = min(xs), max(xs)
    y_hi = max(ys) * 1.08 or 1
    sx = lambda v: L + (v - x_lo) / (x_hi - x_lo) * pw
    sy = lambda v: T + ph - v / y_hi * ph
    body = ""
    for k in range(5):
        v = y_hi * k / 4
        body += (f'<line x1="{L}" x2="{w - R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{GRID}"/>'
                 + _t(L - 8, sy(v) + 4, f"{v:.0f}", 11, MUTED, "end"))
    for v in [x_lo + (x_hi - x_lo) * k / 4 for k in range(5)]:
        body += _t(sx(v), h - B + 18, xfmt(v * 100), 11, MUTED, "middle")
    pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
    body += (f'<polyline points="{pts}" fill="none" stroke="{INDIGO}" stroke-width="3" '
             f'stroke-linejoin="round" stroke-linecap="round"/>')
    for j, (x, lab, col) in enumerate(marks):
        x = min(max(x, x_lo), x_hi)
        body += (f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{T}" y2="{T + ph}" stroke="{col}" '
                 f'stroke-width="2" stroke-dasharray="5 4"/>'
                 f'<circle cx="{sx(x):.1f}" cy="{sy(ys[min(range(len(xs)), key=lambda i: abs(xs[i] - x))]):.1f}" r="5.5" fill="{col}"/>')
        anchor = "end" if sx(x) > w - 130 else "start"
        body += _t(sx(x) + (-6 if anchor == "end" else 6), T + 12 + j * 16, lab, 12, col, anchor, 600)
    body += _t(L + pw / 2, h - 8, xlab, 12, MUTED, "middle")
    body += (f'<text transform="translate(14 {T + ph / 2}) rotate(-90)" font-size="12" fill="{MUTED}" '
             f'text-anchor="middle">{escape(ylab)}</text>')
    return _svg(w, h, body, ylab)


def hbars(rows, vmax, fmt, ref=None, colour=INDIGO, ref_label=""):
    """rows: [(label, value, n_text)]. Horizontal bars with value labels; optional reference line."""
    W, x0, x1, row = 640, 24, 560, 54
    H = 16 + row * len(rows)
    sx = lambda v: x0 + min(v, vmax) / vmax * (x1 - x0)
    body, vals = "", ""
    for i, (lab, v, n_text) in enumerate(rows):
        y = 16 + i * row
        body += _t(x0, y + 4, lab, 13.5, INK, "start", 600)
        if n_text:
            body += _t(x1 + 56, y + 4, n_text, 11, MUTED, "end")
        body += f'<rect x="{x0}" y="{y + 12}" width="{x1 - x0}" height="16" rx="8" fill="{TRACK}"/>'
        body += f'<rect x="{x0}" y="{y + 12}" width="{max(sx(v) - x0, 6):.1f}" height="16" rx="8" fill="{colour}"/>'
        vals += _t(min(sx(v) + 8, x1 + 4), y + 25, fmt(v), 12.5, INK, "start", 700, halo=True)
    if ref is not None:
        body += (f'<line x1="{sx(ref):.1f}" x2="{sx(ref):.1f}" y1="6" y2="{H - 10}" stroke="{MUTED}" '
                 f'stroke-width="1.5" stroke-dasharray="4 3"/>')
        if ref_label:
            body += _t(sx(ref) + 4, H - 1, ref_label, 11, MUTED)
    body += vals   # value labels drawn last, on top of the reference line
    return _svg(W, H + 8, body)


def scatter_cal(pred, obs, xlab, ylab, w=420, h=340):
    """Calibration plot: predicted vs observed rate by decile, with the ideal diagonal."""
    L, R, T, B = 52, 14, 14, 48
    pw, ph = w - L - R, h - T - B
    hi = max(max(pred), max(obs)) * 1.1
    sx = lambda v: L + v / hi * pw
    sy = lambda v: T + ph - v / hi * ph
    body = ""
    for k in range(5):
        v = hi * k / 4
        body += (f'<line x1="{L}" x2="{w - R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{GRID}"/>'
                 + _t(L - 8, sy(v) + 4, f"{v * 100:.0f}%", 11, MUTED, "end")
                 + _t(sx(v), h - B + 18, f"{v * 100:.0f}%", 11, MUTED, "middle"))
    body += f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(hi)}" y2="{sy(hi)}" stroke="{MUTED}" stroke-dasharray="5 4"/>'
    body += "".join(f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="6" fill="{INDIGO}" fill-opacity=".85"/>'
                    for a, b in zip(pred, obs))
    body += _t(L + pw / 2, h - 8, xlab, 12, MUTED, "middle")
    body += (f'<text transform="translate(13 {T + ph / 2}) rotate(-90)" font-size="12" fill="{MUTED}" '
             f'text-anchor="middle">{escape(ylab)}</text>')
    return _svg(w, h, body, ylab)


def step_line(ns, ys, xlab, ylab, mark_n=None, w=420, h=340):
    """AUC vs number of questions (forward-selection curve) with the chosen size highlighted."""
    L, R, T, B = 56, 14, 22, 48
    pw, ph = w - L - R, h - T - B
    lo, hi = min(ys) - 0.02, max(ys) + 0.01
    sx = lambda v: L + (v - min(ns)) / (max(ns) - min(ns)) * pw
    sy = lambda v: T + ph - (v - lo) / (hi - lo) * ph
    body = ""
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        body += (f'<line x1="{L}" x2="{w - R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{GRID}"/>'
                 + _t(L - 8, sy(v) + 4, f"{v:.2f}", 11, MUTED, "end"))
    for n in ns:
        body += _t(sx(n), h - B + 18, n, 11, MUTED, "middle")
    body += (f'<polyline points="{" ".join(f"{sx(n):.1f},{sy(y):.1f}" for n, y in zip(ns, ys))}" fill="none" '
             f'stroke="{INDIGO}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
    for n, y in zip(ns, ys):
        on = n == mark_n
        body += (f'<circle cx="{sx(n):.1f}" cy="{sy(y):.1f}" r="{7 if on else 4.5}" '
                 f'fill="{TEAL if on else INDIGO}" stroke="#fff" stroke-width="2"/>')
    body += _t(L + pw / 2, h - 8, xlab, 12, MUTED, "middle")
    body += (f'<text transform="translate(13 {T + ph / 2}) rotate(-90)" font-size="12" fill="{MUTED}" '
             f'text-anchor="middle">{escape(ylab)}</text>')
    return _svg(w, h, body, ylab)


def icon_array(n, label="", colour=INDIGO):
    """10x10 grid of people-dots: n filled. 'N in 100' is easier to grasp (and calmer) than a bare percentage."""
    n = max(0, min(100, int(n)))
    d, r, pad = 17, 6, 9
    body = ""
    for i in range(100):
        row, col = divmod(i, 10)
        # fill from the bottom-left so the "affected" dots read as a block, not scattered
        k = (9 - row) * 10 + col
        on = k < n
        body += (f'<circle cx="{pad + col * d + r}" cy="{pad + row * d + r}" r="{r}" '
                 f'fill="{colour if on else "#DDE3EF"}"/>')
    size = pad * 2 + 10 * d - (d - 2 * r)
    return _svg(size, size, body, label)
