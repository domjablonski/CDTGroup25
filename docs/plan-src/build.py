"""Build the two plan PDFs from plan_en.html / plan_fr.html.

Usage (from anywhere):  python docs/plan-src/build.py   [en|fr]
Needs Google Chrome or Microsoft Edge (headless printing); PyMuPDF (pip install pymupdf) only
for the page-count message. Figures are generated here as inline SVG, one version per language.
Same pipeline as the Agile Hardware (Minesweeper) plan.
"""
import datetime as dt
import pathlib
import shutil
import subprocess
import sys
import tempfile
from html import escape

HERE = pathlib.Path(__file__).resolve().parent
DOCS = HERE.parent
OUT = {"en": DOCS / "FoodCom_Project_Plan_EN.pdf", "fr": DOCS / "FoodCom_Plan_Projet_FR.pdf"}
FOOTER = {"en": "Food.com project \u00b7 project plan v2.0", "fr": "Projet Food.com \u00b7 plan de projet v2.0"}

ROLE = {1: ("#2b6cb0", "#e6eef8"), 2: ("#b7531c", "#fbece3"), 3: ("#2f855a", "#e5f3ea"),
        4: ("#6b46c1", "#efe9f7"), 5: ("#b83280", "#f9e6f0")}
GREY, GREYL = "#7b8794", "#f1f3f6"
FONT = "font-family='Segoe UI, Arial, sans-serif'"


def t(lang, en, fr):
    return en if lang == "en" else fr


def text(x, y, s, size=10, anchor="middle", weight="normal", fill="#1f2933", style=""):
    return (f"<text x='{x}' y='{y}' font-size='{size}' text-anchor='{anchor}' font-weight='{weight}' "
            f"fill='{fill}' {FONT} {style}>{escape(s)}</text>")


def box(x, y, w, h, fill, stroke, lines, rx=4, dash=None, title_size=11, size=8.6):
    d = f" stroke-dasharray='{dash}'" if dash else ""
    out = [f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{rx}' fill='{fill}' stroke='{stroke}' stroke-width='1.4'{d}/>"]
    if lines:
        total = title_size + (len(lines) - 1) * (size + 2)
        cy = y + h / 2 - total / 2 + title_size * 0.8
        out.append(text(x + w / 2, cy, lines[0], title_size, weight="600"))
        for i, ln in enumerate(lines[1:]):
            out.append(text(x + w / 2, cy + title_size * 0.35 + (i + 1) * (size + 2), ln, size, fill="#3e4c59"))
    return "".join(out)


def arrow(x1, y1, x2, y2, label=None, lx=None, ly=None, color="#52606d", dash=None, anchor="middle"):
    d = f" stroke-dasharray='{dash}'" if dash else ""
    s = f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' stroke-width='1.4' marker-end='url(#ah)'{d}/>"
    if label:
        s += text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2 - 4, label, 7.8,
                  anchor=anchor, fill="#3e4c59")
    return s


DEFS = ("<defs><marker id='ah' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='6' markerHeight='6' "
        "orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='#52606d'/></marker></defs>")


def svg(w, h, body):
    return f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='{w}' height='{h}'>{DEFS}{body}</svg>"


def role_box(k, x, y, w, h, lines, **kw):
    stroke, fill = ROLE[k]
    return box(x, y, w, h, fill, stroke, lines, **kw)


# ---------------------------------------------------------------- timeline
def fig_timeline(lang):
    start, end = dt.date(2026, 11, 9), dt.date(2026, 12, 5)
    x0, x1 = 22, 738
    k = (x1 - x0) / (end - start).days

    def X(d):
        return x0 + (d - start).days * k

    b = []
    sprints = [
        ("S1", dt.date(2026, 11, 10), dt.date(2026, 11, 17), t(lang, "correct on the sample", "juste sur l'\u00e9chantillon")),
        ("S2", dt.date(2026, 11, 17), dt.date(2026, 11, 24), t(lang, "full data + evaluation", "donn\u00e9es compl\u00e8tes + \u00e9val.")),
        ("S3", dt.date(2026, 11, 24), dt.date(2026, 11, 30), t(lang, "writing", "r\u00e9daction")),
        ("S4", dt.date(2026, 11, 30), dt.date(2026, 12, 4), t(lang, "polish", "finitions")),
    ]
    shades = ["#2b6cb0", "#3f7cc0", "#2b6cb0", "#3f7cc0"]
    for i, (n, a, z, sub) in enumerate(sprints):
        xa, xz = X(a), X(z)
        b.append(f"<rect x='{xa+1}' y='62' width='{xz-xa-2}' height='40' rx='3' fill='{shades[i]}'/>")
        b.append(text((xa + xz) / 2, 80, n, 12, weight="700", fill="#fff"))
        b.append(text((xa + xz) / 2, 95, sub, 8.6, fill="#fff"))
    b.append(f"<line x1='{x0}' y1='112' x2='{x1}' y2='112' stroke='#9aa5b1'/>")
    for i in range(4):
        d = dt.date(2026, 11, 10) + dt.timedelta(days=7 * i)
        x = X(d)
        b.append(f"<line x1='{x}' y1='108' x2='{x}' y2='116' stroke='#9aa5b1'/>")
        mon = {11: ("Nov", "nov."), 12: ("Dec", "d\u00e9c.")}[d.month]
        b.append(text(x, 128, f"{t(lang, 'Tue', 'mar.')} {d.day} {mon[0] if lang == 'en' else mon[1]}", 8.8, fill="#52606d"))
        b.append(text(x, 139, t(lang, f"W{10 + i}", f"sem. {10 + i}"), 8.2, fill="#9aa5b1"))
    ms = [
        (dt.date(2026, 11, 12), t(lang, "H1 ingredients v2", "H1 ingr\u00e9dients v2"), 14, "middle", 0.5),
        (dt.date(2026, 11, 19), t(lang, "H2 neighbours, pairs", "H2 voisins, paires"), 14, "middle", 0.5),
        (dt.date(2026, 11, 24), t(lang, "freeze", "gel"), 28, "middle", 0.0),
        (dt.date(2026, 11, 29), t(lang, "full draft", "brouillon"), 14, "middle", 0.5),
        (dt.date(2026, 12, 3), t(lang, "submit", "rendu"), 28, "end", 0.5),
        (dt.date(2026, 12, 4), t(lang, "deadline", "\u00e9ch\u00e9ance"), 14, "start", 0.5),
    ]
    for d, lab, ly, anc, off in ms:
        x = X(d) + off * k
        b.append(f"<line x1='{x}' y1='{ly+12}' x2='{x}' y2='112' stroke='#c77700' stroke-dasharray='3,2'/>")
        b.append(f"<path d='M{x},{ly+6} l6,6 l-6,6 l-6,-6 z' fill='#c77700'/>")
        dx = {"end": -9, "start": 9}.get(anc, 0)
        b.append(text(x + dx, ly + 13 if dx else ly + 2, lab, 9.2, anchor=anc, weight="600", fill="#8a5300"))
    return svg(800, 148, "".join(b))


# ---------------------------------------------------------------- pipeline
def fig_pipeline(lang):
    b = []
    for i, k in enumerate(range(1, 6)):
        stroke, _ = ROLE[k]
        b.append(f"<rect x='{12 + i * 74}' y='8' width='10' height='10' fill='{stroke}'/>")
        b.append(text(27 + i * 74, 17, f"R{k}", 9.2, anchor="start"))
    b.append(box(10, 150, 118, 60, GREYL, "#9aa5b1", ["Kaggle CSV", "RAW_recipes", "231,637 recipes" if lang == "en" else "231 637 recettes"], size=8.2))
    b.append(role_box(1, 150, 136, 114, 88, ["cleaning.py", t(lang, "ingredients v2,", "ingr\u00e9dients v2,"), "calorie_level,", t(lang, "sample", "\u00e9chantillon")], size=8.3))
    b.append(arrow(128, 180, 148, 180))
    b.append(box(282, 165, 128, 30, "#fff", ROLE[1][0], ["recipes.parquet"], title_size=9.4))
    b.append(arrow(264, 180, 280, 180))
    b.append(role_box(2, 440, 40, 150, 60, ["MinHash + LSH", t(lang, "RQ2 similar recipes", "RQ2 recettes similaires")], size=8.6))
    b.append(role_box(3, 440, 250, 150, 60, ["PCY + A-Priori", t(lang, "RQ1 ingredient pairing", "RQ1 associations")], size=8.6))
    b.append(role_box(5, 640, 40, 140, 60, [t(lang, "kNN classifier", "Classifieur kNN"), t(lang, "RQ4 calorie level", "RQ4 niveau de calories")], size=8.6))
    b.append(role_box(4, 640, 250, 140, 60, [t(lang, "PMI graph + Louvain", "Graphe PMI + Louvain"), t(lang, "RQ3 cuisines", "RQ3 cuisines")], size=8.6))
    b.append("<path d='M346,165 V70 H438' fill='none' stroke='#52606d' stroke-width='1.4' marker-end='url(#ah)'/>")
    b.append("<path d='M346,195 V280 H438' fill='none' stroke='#52606d' stroke-width='1.4' marker-end='url(#ah)'/>")
    b.append(arrow(590, 70, 638, 70))
    b.append(text(614, 62, t(lang, "neighbours", "voisins"), 7.8, fill="#3e4c59"))
    b.append(text(614, 84, t(lang, "duplicates", "doublons"), 7.8, fill="#3e4c59"))
    b.append(arrow(515, 100, 515, 248))
    b.append(text(521, 176, t(lang, "duplicates", "doublons"), 7.8, anchor="start", fill="#3e4c59"))
    b.append(arrow(590, 280, 638, 280))
    b.append(text(614, 272, t(lang, "pairs", "paires"), 7.8, fill="#3e4c59"))
    b.append(arrow(410, 172, 658, 102, None, dash="3,2"))
    b.append(text(592, 140, "calorie_level", 7.8, fill="#3e4c59"))
    b.append(arrow(410, 188, 658, 248, None, dash="3,2"))
    b.append(text(592, 218, t(lang, "cuisine (NMI)", "cuisine (NMI)"), 7.8, fill="#3e4c59"))
    b.append(box(440, 340, 340, 40, GREYL, "#9aa5b1", [t(lang, "every role \u2192 results/*.csv \u2192 report (R5) + final_notebook.ipynb (R1)",
                                                         "chaque r\u00f4le \u2192 results/*.csv \u2192 rapport (R5) + final_notebook.ipynb (R1)")], title_size=9.2))
    return svg(790, 388, "".join(b))


# ---------------------------------------------------------------- LSH S-curve
def fig_scurve(lang):
    W, H = 760, 230
    px0, px1, py0, py1 = 60, 470, 200, 20
    b = [f"<rect x='{px0}' y='{py1}' width='{px1-px0}' height='{py0-py1}' fill='#fbfcfd' stroke='#c9d3df'/>"]
    for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        x = px0 + v * (px1 - px0)
        y = py0 - v * (py0 - py1)
        b.append(f"<line x1='{x}' y1='{py0}' x2='{x}' y2='{py0+4}' stroke='#7b8794'/>")
        b.append(text(x, py0 + 15, f"{v:.1f}", 8.6, fill="#52606d"))
        b.append(f"<line x1='{px0-4}' y1='{y}' x2='{px0}' y2='{y}' stroke='#7b8794'/>")
        b.append(text(px0 - 7, y + 3, f"{v:.1f}", 8.6, anchor="end", fill="#52606d"))
    b.append(text((px0 + px1) / 2, py0 + 28, t(lang, "Jaccard similarity s of two recipes", "similarit\u00e9 de Jaccard s de deux recettes"), 9.2, fill="#3e4c59"))
    b.append(text(18, (py0 + py1) / 2, t(lang, "P(candidate)", "P(candidate)"), 9.2, fill="#3e4c59", style=f"transform='rotate(-90 18 {(py0 + py1) / 2})'"))
    xt = px0 + 0.8 * (px1 - px0)
    b.append(f"<line x1='{xt}' y1='{py1}' x2='{xt}' y2='{py0}' stroke='#c77700' stroke-dasharray='4,3'/>")
    b.append(text(xt + 4, py1 + 12, t(lang, "threshold 0.8", "seuil 0,8"), 8.6, anchor="start", fill="#8a5300", weight="600"))
    curves = [(32, 4, ROLE[1][0]), (16, 8, ROLE[2][0]), (8, 16, ROLE[4][0])]
    for bands, rows, col in curves:
        pts = []
        for i in range(101):
            s = i / 100
            p = 1 - (1 - s ** rows) ** bands
            pts.append(f"{px0 + s * (px1 - px0):.1f},{py0 - p * (py0 - py1):.1f}")
        b.append(f"<polyline points='{' '.join(pts)}' fill='none' stroke='{col}' stroke-width='2.2'/>")
    lx = 500
    for i, (bands, rows, col) in enumerate(curves):
        y = 40 + i * 44
        thr = (1 / bands) ** (1 / rows)
        p8 = 1 - (1 - 0.8 ** rows) ** bands
        p5 = 1 - (1 - 0.5 ** rows) ** bands
        b.append(f"<line x1='{lx}' y1='{y}' x2='{lx+24}' y2='{y}' stroke='{col}' stroke-width='2.6'/>")
        b.append(text(lx + 32, y + 4, f"b = {bands}, r = {rows}", 10, anchor="start", weight="600"))
        b.append(text(lx + 32, y + 18, t(lang, f"knee \u2248 {thr:.2f} \u00b7 P(0.8) = {p8:.2f} \u00b7 P(0.5) = {p5:.2f}",
                                         f"coude \u2248 {thr:.2f} \u00b7 P(0,8) = {p8:.2f} \u00b7 P(0,5) = {p5:.2f}"),
                       8.6, anchor="start", fill="#52606d"))
    b.append(text(lx, 186, t(lang, "128 hashes = b bands \u00d7 r rows", "128 hachages = b bandes \u00d7 r lignes"), 9, anchor="start", fill="#3e4c59"))
    b.append(text(lx, 200, t(lang, "R2-S2 picks (b, r) with measured precision/recall", "R2-S2 choisit (b, r) avec pr\u00e9cision/rappel mesur\u00e9s"), 9, anchor="start", fill="#3e4c59"))
    return svg(W, H + 10, "".join(b))


# ---------------------------------------------------------------- verification pyramid
def fig_pyramid(lang):
    layers = [
        (t(lang, "1  Unit tests on toy examples with a known answer", "1  Tests unitaires sur exemples \u00e0 r\u00e9ponse connue"), "#2b6cb0"),
        (t(lang, "2  Contract tests \u2014 signatures, paths, schemas", "2  Tests du contrat \u2014 signatures, chemins, sch\u00e9mas"), "#3576b8"),
        (t(lang, "3  Reference libraries on the 1 % sample", "3  Librairies de r\u00e9f\u00e9rence sur l'\u00e9chantillon"), "#4282c2"),
        (t(lang, "4  Exact ground truth on samples", "4  V\u00e9rit\u00e9 exacte sur \u00e9chantillons"), "#b7531c"),
        (t(lang, "5  Null models and baselines", "5  Mod\u00e8les nuls et baselines"), "#2f855a"),
        (t(lang, "6  Labels + human checks", "6  Labels + v\u00e9rifs humaines"), "#5f6b7a"),
    ]
    b = []
    cx, base, h = 300, 236, 36
    wb, wt = 590, 250
    n = len(layers)
    for i, (lab, col) in enumerate(layers):
        y1 = base - i * h
        y0 = y1 - h + 3
        w1 = wb - (wb - wt) * i / (n - 1)
        w0 = wb - (wb - wt) * (i + 0.85) / (n - 1)
        pts = f"{cx-w1/2},{y1} {cx+w1/2},{y1} {cx+w0/2},{y0} {cx-w0/2},{y0}"
        b.append(f"<polygon points='{pts}' fill='{col}'/>")
        b.append(text(cx, (y0 + y1) / 2 + 4, lab, 10.2, weight="600", fill="#fff"))

    def bracket(i0, i1, lab1, lab2):
        ya = base - i1 * h - h + 4
        yb = base - i0 * h - 1
        x = 618
        out = f"<path d='M{x},{ya} h6 V{yb} h-6' fill='none' stroke='#52606d'/>"
        out += text(x + 12, (ya + yb) / 2 - 2, lab1, 9.6, anchor="start", weight="600")
        out += text(x + 12, (ya + yb) / 2 + 11, lab2, 8.8, anchor="start", fill="#52606d")
        return out
    b.append(bracket(0, 1, t(lang, "GitHub Actions CI", "CI GitHub Actions"), t(lang, "every push and PR", "chaque push et PR")))
    b.append(bracket(2, 3, t(lang, "@pytest.mark.data", "@pytest.mark.data"), t(lang, "locally, real data", "en local, vraies donn\u00e9es")))
    b.append(bracket(4, 5, t(lang, "notebooks", "notebooks"), "results/*.csv"))
    return svg(760, 244, "".join(b))


# ---------------------------------------------------------------- story workflow
def fig_flow(lang):
    steps = [
        ("h", t(lang, "1 Pick a story", "1 Choisir une story"), [t(lang, "card \u2192 In progress", "carte \u2192 In progress"), t(lang, "new Claude session", "nouvelle session Claude")]),
        ("c", t(lang, "2 Read + plan", "2 Lire + planifier"), [t(lang, "CONTRACT + story", "CONTRAT + story"), t(lang, "branch r2/s1-\u2026", "branche r2/s1-\u2026")]),
        ("c", t(lang, "3 Tests first", "3 Tests d'abord"), [t(lang, "toy examples", "exemples jouets"), t(lang, "see them fail", "les voir \u00e9chouer")]),
        ("c", t(lang, "4 Implement", "4 Impl\u00e9menter"), [t(lang, "pytest green", "pytest vert"), t(lang, "results/*.csv + AI log", "results/*.csv + log IA")]),
        ("h", t(lang, "5 Understand", "5 Comprendre"), [t(lang, "read the whole diff", "lire tout le diff"), t(lang, "run pytest yourself", "lancer pytest soi-m\u00eame")]),
        ("t", t(lang, "6 Push + PR", "6 Push + PR"), [t(lang, "CI + /code-review", "CI + /code-review"), t(lang, "teammate review", "revue d'un co\u00e9quipier")]),
        ("t", t(lang, "7 Merge", "7 Fusion"), [t(lang, "squash, card \u2192 Done", "squash, carte \u2192 Done"), t(lang, "status line in chat", "statut dans le chat")]),
    ]
    col = {"h": (GREYL, "#52606d"), "c": (ROLE[3][1], ROLE[3][0]), "t": ("#efe9f7", "#6b46c1")}
    b = []
    w, h, gap = 172, 62, 22
    pos = [(4 + i * (w + gap), 6) for i in range(4)] + [(101 + i * (w + gap), 100) for i in range(3)]
    for i, ((kind, title, lines), (x, y)) in enumerate(zip(steps, pos)):
        f, st = col[kind]
        b.append(box(x, y, w, h, f, st, [title] + lines, title_size=12, size=10))
        if i in (0, 1, 2, 4, 5):
            b.append(arrow(x + w + 1, y + h / 2, x + w + gap - 1, y + h / 2))
    x4, y4 = pos[3]
    x5, y5 = pos[4]
    b.append(f"<path d='M{x4 + w / 2},{y4 + h} V{y4 + h + 16} H{x5 + w / 2} V{y5 - 2}' fill='none' stroke='#52606d' stroke-width='1.4' marker-end='url(#ah)'/>")
    leg = [("h", t(lang, "human (story owner)", "humain (porteur de la story)")), ("c", "Claude Code"), ("t", t(lang, "team + CI", "\u00e9quipe + CI"))]
    for i, (kind, lab) in enumerate(leg):
        f, st = col[kind]
        x = 150 + i * 190
        b.append(f"<rect x='{x}' y='178' width='13' height='13' fill='{f}' stroke='{st}'/>")
        b.append(text(x + 20, 189, lab, 10.5, anchor="start"))
    return svg(770, 198, "".join(b))


def figures(lang):
    return {"timeline": fig_timeline(lang), "pipeline": fig_pipeline(lang), "scurve": fig_scurve(lang),
            "pyramid": fig_pyramid(lang), "flow": fig_flow(lang)}


def find_browser():
    for p in [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]:
        if pathlib.Path(p).exists():
            return p
    for name in ["google-chrome", "chromium", "chromium-browser", "chrome", "msedge"]:
        if shutil.which(name):
            return shutil.which(name)
    sys.exit("No Chrome/Edge found for headless printing.")


def build(lang, browser, tmp):
    html = (HERE / f"plan_{lang}.html").read_text(encoding="utf-8")
    css = (HERE / "style.css").read_text(encoding="utf-8")
    css += ("\n@page { size: A4; margin: 15mm 15mm 16mm 15mm;"
            f" @bottom-left {{ content: '{FOOTER[lang]}'; font-size: 7.5pt; color: #8a94a3; font-family: 'Segoe UI', Arial; }}"
            " @bottom-right { content: counter(page) ' / ' counter(pages); font-size: 7.5pt; color: #8a94a3; font-family: 'Segoe UI', Arial; } }\n")
    html = html.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{css}\n</style>")
    for name, s in figures(lang).items():
        html = html.replace(f"<!--FIG:{name}-->", f"<div class='fig'>{s}</div>")
    assert "<!--FIG:" not in html, "unknown figure placeholder"
    page = tmp / f"plan_{lang}.html"
    page.write_text(html, encoding="utf-8")
    out = OUT[lang]
    printed = tmp / out.name
    cmd = [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--no-first-run",
           f"--user-data-dir={tmp / 'profile'}", f"--print-to-pdf={printed}", page.as_uri()]
    subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    if not printed.exists() or printed.stat().st_size == 0:
        sys.exit(f"{out.name}: the browser produced no PDF")
    try:
        shutil.copyfile(printed, out)
    except PermissionError:
        # Chrome fails silently on a locked file, so print to temp and copy: a PDF viewer may hold the old file.
        out = out.with_suffix(".new.pdf")
        shutil.copyfile(printed, out)
        print(f"WARNING: {OUT[lang].name} is open in another program (locked). Wrote {out.name} instead; "
              "close the viewer and run build.py again.")
    try:
        import fitz  # PyMuPDF, only for the report
        print(f"{out.name}: {fitz.open(out).page_count} pages")
    except ImportError:
        print(f"{out.name}: written")


if __name__ == "__main__":
    browser = find_browser()
    langs = sys.argv[1:] or ["en", "fr"]
    with tempfile.TemporaryDirectory() as d:
        for lang in langs:
            build(lang, browser, pathlib.Path(d))
