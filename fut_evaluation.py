"""Einstellungen, Kennzahlen und Verdikte (reine Funktionen; tests/test_verdicts.py prüft jede Schwelle und jede Zahl im Text).

Geldbeträge in EUR je Hektar, Zinsen im Text in Prozent. Alle Zahlen gelten im Lehrmodell mit seinen Annahmen (siehe „Annahmen, Grenzen und
bekannte Probleme“): es gibt keine echten Holzpreisreihen, σ und φ sind Regler."""

import math
from dataclasses import dataclass

import numpy as np

import fut_constants as C
from fut_model import Modell, Prm, mod


@dataclass(frozen=True)
class Settings:
    zins: float = C.DEFAULT_ZINS          # Prozent je Jahr
    pflanz: float = C.DEFAULT_PFLANZ
    preis: float = C.DEFAULT_PREIS
    sort: float = C.DEFAULT_SORT
    risiko: float = C.DEFAULT_RISIKO
    sigma: float = C.DEFAULT_SIGMA
    phi: float = C.DEFAULT_PHI
    seed: int = C.DEFAULT_SEED


def to_prm(s: Settings, **kw) -> Prm:
    return Prm(preis=float(s.preis), c_pflanz=float(s.pflanz), r=float(s.zins) / 100.0, sort_praemie=float(s.sort), risiko=float(s.risiko),
               sigma=float(s.sigma), phi=float(s.phi), **kw)


def eur(x: float) -> str:
    """Geldbetrag mit Tausenderpunkt und Vorzeichen-Minus, ohne Nachkommastellen."""
    return f"{round(x):,}".replace(",", ".").replace("-", "−")


def zahl(x: float, stellen: int = 1) -> str:
    """Dezimalzahl mit Komma."""
    return f"{x:.{stellen}f}".replace(".", ",").replace("-", "−")


def pct(x: float | None, stellen: int = 2) -> str:
    return "–" if x is None else zahl(100.0 * x, stellen) + " %"


# ---- Kennzahlen -----------------------------------------------------------------------------------------------------------------------
def rangfolge(werte: dict[str, float]) -> str:
    """Namen nach BEV absteigend; Werte, die sich um weniger als 0,005 EUR/ha unterscheiden, gelten als gleich (Zeichen ' = ')."""
    items = sorted(werte.items(), key=lambda kv: -kv[1])
    out = items[0][0]
    for (n0, v0), (n1, v1) in zip(items, items[1:]):
        out += (" = " if abs(v0 - v1) < 0.005 else " > ") + n1
    return out


def abstand(bev: float, referenz: float) -> float:
    """Abstand in EUR/ha: mein Wert minus Referenz."""
    return bev - referenz


def gleichstand_text(anzahl: int, zustaende: int) -> str:
    return f"{anzahl} von {zustaende}"


# ---- Verdikte -------------------------------------------------------------------------------------------------------------------------
def verdict_msy(tf: int, tm: int, bev_f: float, bev_m: float, be_f: float | None, be_m: float | None) -> tuple[str, str]:
    """Kernfrage 1: Was kostet die MSY-Regel (Umtriebszeit der größten durchschnittlichen Zuwachsleistung) gegenüber Faustmann?
    be_f, be_m: Break-even-Zinsen als Dezimalzahl (None: im Suchbereich kein Vorzeichenwechsel)."""
    verlust = bev_f - bev_m
    if verlust <= 1e-9:
        return "success", (f"**Die MSY-Regel verliert hier nichts:** die Faustmann-Umtriebszeit ({tf} Jahre) ist so lang wie die der MSY-Regel ({tm} Jahre). "
                           "Das gilt in der Grundeinstellung nur in einem schmalen Zinsbereich (0,45 % bis 0,52 %, Preset „Kreuzungszins“); sonst verliert sie auf beiden Seiten. "
                           "Bei anderen Reglern verschiebt sich der Bereich (Pflanzung 8000: 1,05 % bis 1,15 %), bei Kalamitätsrisiko 1 gibt es ihn nicht.")
    be = ""
    if be_f is not None and be_m is not None:
        be = f" Break-even-Zins (Bodenwert 0): Faustmann {pct(be_f)}, MSY-Regel {pct(be_m)}."
    if bev_f <= 0:
        return "warning", (f"**Der Boden lohnt sich bei diesem Zins auch mit der besten festen Umtriebszeit nicht** (Bodenwert {eur(bev_f)} EUR/ha bei {tf} Jahren); "
                           f"die MSY-Regel ({tm} Jahre) liegt {eur(verlust)} EUR/ha darunter.{be}")
    if bev_m < 0:
        return "warning", (f"**Zwischen den beiden Break-even-Zinsen:** mit der Faustmann-Umtriebszeit ({tf} Jahre) ist der Boden ein Gewinn ({eur(bev_f)} EUR/ha), "
                           f"mit der MSY-Regel ({tm} Jahre) ein Verlust ({eur(bev_m)} EUR/ha).{be}")
    return "info", (f"**Die MSY-Regel ({tm} Jahre) verliert {eur(verlust)} EUR/ha** gegenüber der Faustmann-Umtriebszeit ({tf} Jahre), "
                    f"das sind {zahl(100.0 * verlust / bev_f, 1)} % des Faustmann-Bodenwerts von {eur(bev_f)} EUR/ha.{be}")


def verdict_hook(gewinn: float, verlust: float, sigma: float, phi: float, zins: float) -> tuple[str, str]:
    """Kernfrage 2: Was ist im Modell mehr wert, das Warten auf den Preis (Optimum minus feste Faustmann-Zeit) oder die richtige Umtriebszeit
    gegenüber der MSY-Regel? gewinn und verlust in EUR/ha; zins in Prozent."""
    bed = f"im Lehrmodell mit σ = {zahl(sigma, 2)}, φ = {zahl(phi, 2)} und Zins {zahl(zins, 2)} %"
    if sigma <= 0:
        return "info", (f"**Ohne Preisstreuung (σ = 0) gibt es nichts abzuwarten:** der Gewinn der Preisflexibilität ist 0; die richtige Umtriebszeit "
                        f"gewinnt gegen die MSY-Regel {eur(verlust)} EUR/ha.")
    if verlust <= 1e-9:
        return "info", (f"**Faustmann und MSY-Regel sind hier gleich gut** (Verlust 0); das Warten auf den Preis bringt zusätzlich {eur(gewinn)} EUR/ha ({bed}).")
    faktor = zahl(gewinn / verlust, 1)
    if gewinn >= verlust:
        return "warning", (f"**Das Warten auf den Preis ist mehr wert als der Streit um die Umtriebszeit:** {eur(gewinn)} EUR/ha Gewinn der Preisflexibilität gegen "
                           f"{eur(verlust)} EUR/ha Verlust der MSY-Regel (Faktor {faktor}), {bed}. Die Größe hängt an σ und φ, für die es hier keine Daten gibt: keine Empfehlung für einen Betrieb.")
    return "success", (f"**Die richtige Umtriebszeit ist mehr wert als das Warten auf den Preis:** {eur(verlust)} EUR/ha Verlust der MSY-Regel gegen nur "
                       f"{eur(gewinn)} EUR/ha Gewinn der Preisflexibilität (Faktor {faktor}), {bed}. Die Größe hängt an σ und φ, für die es hier keine Daten gibt: keine Empfehlung für einen Betrieb.")


def verdict_sigma_stern(sigma_star: float | None, verlust: float, sigma: float, phi: float, zins: float, sigma_max: float = C.SIGMA_MAX) -> tuple[str, str]:
    """Kernfrage 3: Bei welcher Preisstreuung σ* erreicht der Gewinn der Preisflexibilität den Verlust der MSY-Regel? (φ und Zins bleiben fest.)"""
    bed = f"für φ = {zahl(phi, 2)} und Zins {zahl(zins, 2)} %"
    if verlust <= 1e-9:
        return "info", f"**Keine Schwelle:** die MSY-Regel verliert bei diesem Zins nichts ({bed}); jede Preisstreuung macht die Flexibilität zum Mehrwert."
    if sigma_star is None:
        return "info", (f"**Bis σ = {zahl(sigma_max, 2)} reicht der Gewinn der Preisflexibilität nicht an den Verlust der MSY-Regel von {eur(verlust)} EUR/ha heran** ({bed}): "
                        "bei dieser Einstellung gewinnt die richtige Umtriebszeit (im Raster der README bei φ 0,97 und Zins 3 %).")
    lage = "darüber" if sigma > sigma_star else "darunter"
    return "info", (f"**Schwelle σ* = {zahl(sigma_star, 3)}** ({bed}): ab dieser Streuung des Log-Preises ist der Gewinn der Preisflexibilität größer als der Verlust der "
                    f"MSY-Regel von {eur(verlust)} EUR/ha. Ihr σ = {zahl(sigma, 2)} liegt {lage}.")


def verdict_bonitaet(gap_gemeinsam: float, gap_ziel: float, gap_msy: float) -> tuple[str, str]:
    """Tab Bonität: Abstände (EUR/ha, nie positiv) zum Mittel der je Bonität richtigen Faustmann-Umtriebszeiten für eine feste Zeit für alle, den Zieldurchmesser
    (ein A*) und die MSY-Regel je Bonität. Die Zieldurchmesser-Regel ist im Modell wegen der gleichen Zeitskalierung von Volumen und Durchmesser günstig gestellt."""
    text = (f"Gegenüber der je Bonität richtigen Faustmann-Umtriebszeit liegt eine feste Zeit für alle {eur(-gap_gemeinsam)} EUR/ha darunter, der Zieldurchmesser "
            f"{eur(-gap_ziel)} EUR/ha, die MSY-Regel je Bonität {eur(-gap_msy)} EUR/ha (Mittel der drei Bonitäten). "
            "Im Modell wachsen Durchmesser und Volumen in derselben Zeitskalierung: der Zieldurchmesser ist dort fast eine Regel in effektivem Alter, das Ergebnis ist teilweise eingebaut.")
    return "info", text


# ---- Bonität (Tab 3) -------------------------------------------------------------------------------------------------------------------
def bonitaet(p: Prm) -> dict:
    """Drei Bonitäten g = 0,8 / 1,0 / 1,2 (Zeitskalierung des Wachstums), je ein Drittel der Fläche, konstanter Preis. Rückgabe: Mittel der BEV je Regel (EUR/ha).
    Zieldurchmesser: ernten, sobald g·Alter ≥ A* (A* in „effektiven Jahren“ abgestimmt); eine feste Zeit für alle; MSY je Bonität; MSY der mittleren Bonität für alle."""
    gs = C.BONITAET_G
    ms = {g: Modell(mod(p, g=g, sigma=0.0)).skalar() for g in gs}
    ts = list(range(max(C.TF_MIN, p.amin), p.amax + 1))
    bev = {g: {T: ms[g].bev_maske(ms[g].maske_fix(T)) for T in ts} for g in gs}
    tf = {g: max(ts, key=lambda T: bev[g][T]) for g in gs}
    mean = lambda f: float(np.mean([f(g) for g in gs]))
    ref = mean(lambda g: bev[g][tf[g]])
    t_all = max(ts, key=lambda T: mean(lambda g: bev[g][T]))
    best = None
    for a_star in np.arange(20.0, 120.0 + 1e-9, 0.25):
        tg = {g: math.ceil(a_star / g - 1e-9) for g in gs}
        if any(t < ts[0] or t > ts[-1] for t in tg.values()):
            continue
        v = mean(lambda g: bev[g][tg[g]])
        if best is None or v > best[0] + 1e-9:
            best = (v, float(a_star), tg)
    msy_g = {g: Modell(mod(p, g=g, sigma=0.0)).t_msy() for g in gs}
    msy_1 = msy_g[1.0]
    return {"ref": ref, "tf": tf, "t_all": t_all, "fest": mean(lambda g: bev[g][t_all]),
            "ziel": best[0], "a_star": best[1], "t_ziel": best[2],
            "msy_g": mean(lambda g: bev[g][msy_g[g]]), "t_msy_g": msy_g, "msy_1": mean(lambda g: bev[g][msy_1])}
