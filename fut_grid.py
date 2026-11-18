"""Vorgerechnete Tabellen (python fut_grid.py): abgestimmte Preisregeln (data/regeln.csv) und die Schwelle σ* (data/sigma_stern.csv).

Die Koordinatensuche der Preisregel braucht je Szenario etwa 4 bis 11 Sekunden (gemessen auf einem Entwicklungsrechner), die App nimmt deshalb für die Szenarien dieser Tabelle die
vorgerechnete Zeile und stimmt nur andere Einstellungen live ab (mit Spinner). Alle Zellen: Pflanzkosten 4000 EUR/ha, Nettoerlös 40 EUR/m³,
keine Sortimentsprämie, Alter bis 150, 21 Preiszustände; sonst nur die hier genannten Parameter verändert."""

import csv
import os
import time

import fut_constants as C
import fut_rules as RU
import fut_stoch as ST
from fut_model import Modell, Prm, mod

BASIS = Prm()
SIGMA_STERN_PATH = os.path.join(RU.DATA, "sigma_stern.csv")
SIGMA_STERN_FIELDS = ("phi", "zins", "verlust", "sigma_stern", "gewinn_010", "gewinn_020", "gewinn_030")


def regel_zellen():
    """Szenarien der vorgerechneten Preisregeln (sigma, phi, zins in %, risiko)."""
    z = [(s, 0.9, 2.0, 0.0) for s in (0.1, 0.2, 0.3, 0.4, 0.5)]            # σ-Sweep
    z += [(0.3, ph, 2.0, 0.0) for ph in (0.0, 0.5, 0.8, 0.97)]              # Persistenz
    z += [(0.3, 0.9, zi, 0.0) for zi in (0.5, 1.0, 3.0, 3.5, 4.0)]           # Zins
    z += [(0.3, 0.9, 2.0, ri) for ri in (0.5, 1.0, 2.0)]                    # Kalamität
    z += [(0.13, 0.9, 2.0, 0.0)]                                            # Preset σ*-Schwelle
    return z


def regeln_bauen(path=RU.RULES_PATH, verbose=True):
    rows = []
    for sigma, phi, zins, risiko in regel_zellen():
        t = time.time()
        p = mod(BASIS, sigma=sigma, phi=phi, r=zins / 100.0, risiko=risiko)
        m = Modell(p)
        tf = m.bestes_fix()[0]
        rg = RU.abstimmen(m, tf)
        rows.append({"sigma": sigma, "phi": phi, "zins": zins, "pflanz": p.c_pflanz, "preis": p.preis, "sort": p.sort_praemie, "risiko": risiko,
                     "tmin": rg.tmin, "k": rg.k, "tmax": rg.tmax, "bev": round(rg.bev, 4)})
        if verbose:
            print(rows[-1], f"{time.time() - t:.1f} s", flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=RU.FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    return rows


def sigma_stern_bauen(path=SIGMA_STERN_PATH, verbose=True):
    rows = []
    for phi in (0.0, 0.9, 0.97):
        for zins in (1.0, 2.0, 3.0):
            p = mod(BASIS, phi=phi, r=zins / 100.0)
            s, verlust = ST.sigma_stern(p)
            g, _, _ = ST.sigma_kurve(p, (0.1, 0.2, 0.3))
            rows.append({"phi": phi, "zins": zins, "verlust": round(verlust, 4), "sigma_stern": "" if s is None else round(s, 4),
                         "gewinn_010": round(float(g[0]), 4), "gewinn_020": round(float(g[1]), 4), "gewinn_030": round(float(g[2]), 4)})
            if verbose:
                print(rows[-1], flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=SIGMA_STERN_FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    return rows


def sigma_stern_laden(path=SIGMA_STERN_PATH):
    """Zeilen der σ*-Tabelle (Zahlen als float, σ* als None, wenn bis σ 0,5 nicht erreicht); leere Liste, wenn die Datei fehlt."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return [{**{k: float(v) for k, v in r.items() if k != "sigma_stern"}, "sigma_stern": float(r["sigma_stern"]) if r["sigma_stern"] != "" else None}
                for r in csv.DictReader(fh)]


if __name__ == "__main__":
    sigma_stern_bauen()
    regeln_bauen()
