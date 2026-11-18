"""PDF-Export (fpdf2, Kernschriften: Umlaute gehen, andere Sonderzeichen werden ersetzt)."""

from fpdf import FPDF

_REPLACE = {"–": "-", "—": "-", "≥": ">=", "≤": "<=", "→": "->", "×": "x", "·": "*", "„": '"', "“": '"', "’": "'", "≈": "~", "−": "-", "σ": "sigma", "φ": "phi", "³": "3"}


def _latin(text):
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


def _para(pdf, text, size=9, bold=False):
    pdf.set_font("Helvetica", "B" if bold else "", size)
    pdf.multi_cell(0, 5, _latin(text), new_x="LMARGIN", new_y="NEXT")


def build_pdf(settings, det_lines, rows, verdicts):
    """settings: dict der Reglerwerte; det_lines: Sätze zum deterministischen Teil; rows: Liste {name, bev, vs_opt}; verdicts: Liste von Texten der Kernfragen."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _latin("Forst: Wann ernten?"), new_x="LMARGIN", new_y="NEXT")
    _para(pdf, "Lehrmodell mit offenen Annahmen (keine Ertragstafel, keine Holzpreisreihe); die Größe der Effekte hängt an σ und φ. Keine Empfehlung für einen Betrieb.")
    pdf.ln(2)
    _para(pdf, "Einstellungen", 10, True)
    _para(pdf, "; ".join(f"{k} = {v}" for k, v in settings.items()))
    pdf.ln(2)
    _para(pdf, "Umtriebszeit (konstanter Preis)", 10, True)
    for line in det_lines:
        _para(pdf, line.replace("**", ""))
    pdf.ln(2)
    _para(pdf, "Bodenertragswert je Regel (Preis als Zufall, exakt)", 10, True)
    pdf.set_font("Helvetica", "B", 8)
    for head, wd in (("Regel", 80), ("BEV in EUR/ha", 40), ("zum Optimum in EUR/ha", 50)):
        pdf.cell(wd, 5, _latin(head), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for r in rows:
        pdf.cell(80, 5, _latin(r["name"])[:50], border=1)
        pdf.cell(40, 5, f"{r['bev']:.1f}", border=1)
        pdf.cell(50, 5, f"{r['vs_opt']:+.1f}", border=1)
        pdf.ln()
    pdf.ln(2)
    _para(pdf, "Kernfragen", 10, True)
    for v in verdicts:
        _para(pdf, v.replace("**", ""))
        pdf.ln(1)
    return bytes(pdf.output())
