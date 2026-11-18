"""Preisregel mit drei Parametern (frühestes Hiebalter, Preisindex der Schwelle, spätestes Hiebalter), je Szenario abgestimmt.

Abstimmung: Koordinatensuche (je Runde alle Werte jeder Koordinate durchprobieren, den besten übernehmen) von mehreren Startpunkten, der letzte ist
die feste Umtriebszeit selbst (damit ist die abgestimmte Regel nie schlechter als die beste feste Zeit). Jeder Startpunkt hat Zufall nur in Form
fester Zahlen; das Ergebnis ist deterministisch. Eine Koordinatensuche kann in einem lokalen Optimum enden: die Zahl heißt „abgestimmte Preisregel“,
nicht „beste Preisregel“. Vorgerechnete Zeilen stehen in data/regeln.csv (python fut_grid.py)."""

import csv
import os
from dataclasses import dataclass

import fut_constants as C
from fut_model import Modell, Prm

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RULES_PATH = os.path.join(DATA, "regeln.csv")
FIELDS = ("sigma", "phi", "zins", "pflanz", "preis", "sort", "risiko", "tmin", "k", "tmax", "bev")


@dataclass(frozen=True)
class Regel:
    tmin: int
    k: int
    tmax: int
    bev: float
    quelle: str = "live"   # "live" (hier abgestimmt) oder "Raster" (vorgerechnet)

    def schwelle(self, m: Modell) -> float:
        """Preis (EUR/m³), ab dem die Regel zwischen tmin und tmax erntet."""
        return float(m.preise[self.k])


def abstimmen(m: Modell, tf: int, starts: int | None = None, runden: int | None = None) -> Regel:
    """Koordinatensuche über (tmin, k, tmax); `starts` und `runden` aus fut_constants, wenn nicht angegeben."""
    starts = C.RULE_STARTS if starts is None else starts
    runden = C.RULE_ROUNDS if runden is None else runden
    A, n, amin = m.amax, m.n, max(C.TF_MIN, m.amin)

    def f(x):
        tmin, k, tmax = x
        if tmin > tmax or tmin < amin or tmax > A or k < 0 or k > n - 1:
            return -1e18
        return m.bev(m.auswerten(m.maske_regel(tmin, k, tmax))[0])

    heuristisch = [(tf - 10, n // 2, tf + 15), (tf - 20, n // 2 + 3, tf + 30), (tf, n // 2, tf + 5), (tf - 5, n - 4, tf + 25)]
    fest = (tf, 0, tf)   # Preisindex 0: bei jedem Preis ab tmin ernten, tmin = tmax = T: genau die feste Umtriebszeit
    best = None
    for x0 in heuristisch[:max(1, min(starts, len(heuristisch)))] + [fest]:
        x = [max(x0[0], amin), min(x0[1], n - 1), min(x0[2], A)]
        x[2] = max(x[2], x[0])
        cur = f(x)
        for _ in range(runden):
            besser = False
            for dim, rng in ((0, range(amin, A + 1)), (1, range(0, n)), (2, range(5, A + 1))):
                bb, bv = cur, x[dim]
                for v in rng:
                    y = list(x)
                    y[dim] = v
                    fv = f(y)
                    if fv > bb + 1e-9:
                        bb, bv = fv, v
                if bv != x[dim]:
                    x[dim] = bv
                    cur = bb
                    besser = True
            if not besser:
                break
        if best is None or cur > best[0]:
            best = (cur, *x)
    return Regel(int(best[1]), int(best[2]), int(best[3]), float(best[0]))


def _schluessel(p: Prm):
    return tuple(round(v, 6) for v in (p.sigma, p.phi, p.r * 100.0, p.c_pflanz, p.preis, p.sort_praemie, p.risiko))


def laden(path=RULES_PATH) -> dict:
    """Vorgerechnete Regeln: {Schlüssel: Regel}; leeres Wörterbuch, wenn die Datei fehlt."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            key = tuple(round(float(r[c]), 6) for c in ("sigma", "phi", "zins", "pflanz", "preis", "sort", "risiko"))
            out[key] = Regel(int(r["tmin"]), int(r["k"]), int(r["tmax"]), float(r["bev"]), "Raster")
    return out


def vorgerechnet(p: Prm, table=None):
    """Die vorgerechnete Regel zu genau diesen Parametern, sonst None (Standardwerte von amin, n_preis, g, c_hieb_fix gelten)."""
    table = laden() if table is None else table
    if p.amin != 0 or p.g != 1.0 or p.c_hieb_fix != 0.0 or p.n_preis != C.N_PREIS or p.amax != C.ALTER_MAX:
        return None
    return table.get(_schluessel(p))
