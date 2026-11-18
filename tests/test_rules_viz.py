"""Preisregel (Abstimmung, vorgerechnete Zeilen), Abbildungen (Achsen gesperrt, Annotationen am Positionsindex) und PDF-Export."""

import numpy as np
import pytest

import fut_constants as C
import fut_det as D
import fut_evaluation as ev
import fut_grid as G
import fut_pdf_export as PDF
import fut_rules as RU
import fut_stoch as ST
import fut_visualization as V
from fut_model import Modell, Prm, mod

B = Prm()


# ---------------------------------------------------------------------------------------------------------------- Preisregel
def test_abstimmung_ist_nie_schlechter_als_die_feste_zeit_und_nie_besser_als_das_optimum(fast_limits):
    # fast_limits: ein Startpunkt, zwei Runden; die feste Zeit ist trotzdem immer ein Startpunkt
    for sigma, phi in ((0.1, 0.9), (0.3, 0.5)):
        m = Modell(mod(B, sigma=sigma, phi=phi))
        tf, bf = m.bestes_fix()
        regel = RU.abstimmen(m, tf)
        assert regel.bev >= bf - 1e-9
        assert regel.bev <= ST.optimum(m, tf)[0].bev + 1e-9
        assert 0 <= regel.k <= m.n - 1 and 5 <= regel.tmin <= regel.tmax <= m.amax


def test_regel_schwelle_ist_ein_preis_des_gitters():
    m = Modell(mod(B, sigma=0.3, phi=0.9))
    r = RU.Regel(43, 12, 150, 6822.6)
    assert r.schwelle(m) == pytest.approx(50.0, abs=0.05)


def test_vorgerechnet_greift_nur_bei_genau_den_parametern_des_rasters():
    tab = RU.laden()
    p = mod(B, sigma=0.3, phi=0.9)
    assert RU.vorgerechnet(p, tab).quelle == "Raster"
    assert RU.vorgerechnet(mod(p, sigma=0.31), tab) is None and RU.vorgerechnet(mod(p, c_pflanz=4500.0), tab) is None
    assert RU.vorgerechnet(mod(p, amin=40), tab) is None and RU.vorgerechnet(mod(p, g=0.8), tab) is None and RU.vorgerechnet(mod(p, c_hieb_fix=100.0), tab) is None
    assert RU.laden("gibt-es-nicht.csv") == {} and G.sigma_stern_laden("gibt-es-nicht.csv") == []


# ---------------------------------------------------------------------------------------------------------------- Abbildungen
def _alle_figuren():
    p = mod(B, sigma=0.3, phi=0.9)
    m = Modell(p)
    tf, bf = m.bestes_fix()
    opt, mask, _, _ = ST.optimum(m, tf)
    regel = RU.laden()[(0.3, 0.9, 2.0, 4000.0, 40.0, 0.0, 0.0)]
    bahn = ST.bahn_ziehen(m, 5)
    vl = {"Faustmann (fest)": ST.verlauf(m, m.maske_fix(tf), bahn), "MSY (fest)": ST.verlauf(m, m.maske_fix(68), bahn), "Optimum (DP)": ST.verlauf(m, mask, bahn),
          "Preisregel (abgestimmt)": ST.verlauf(m, m.maske_regel(regel.tmin, regel.k, regel.tmax), bahn)}
    ts, bev = D.bev_kurve(B)
    a, v, lauf, schnitt = D.zuwachs(B)
    sig = np.arange(0, 0.51, 0.02)
    figs = {
        "zuwachs": V.fig_zuwachs(a, v, lauf, schnitt, 68, tf),
        "bev": V.fig_bev(ts, bev, tf, bf, 68, m.bev_msy()),
        "bahn": V.fig_bahn(m.preise, bahn.preis_index, vl, regel.schwelle(m)),
        "schwelle": V.fig_schwelle(m.preise, opt.schwelle, regel, tf, 40.0),
        "gewinn": V.fig_gewinn_sigma(sig, np.linspace(0, 5000, len(sig)), 676.0, 0.129, 0.3),
        "regeln": V.fig_regeln([{"name": n, "bev": w} for n, w in (("Faustmann (fest)", 4805.0), ("MSY (fest)", 4129.0), ("Optimum (DP)", 7255.0))]),
        "bonitaet": V.fig_bonitaet(["a", "b", "c", "d"], [-169.0, -18.0, -684.0, -880.0]),
    }
    return figs


def test_alle_achsen_aller_abbildungen_sind_gesperrt():
    for name, fig in _alle_figuren().items():
        achsen = [fig.layout[k] for k in fig.layout if k.startswith(("xaxis", "yaxis"))]
        assert achsen and all(ax.fixedrange is True for ax in achsen), name


def test_abbildung_ohne_preisregel_haelt_stand():
    m = Modell(mod(B, sigma=0.3, phi=0.9))
    bahn = ST.bahn_ziehen(m, 5)
    vl = {"Faustmann (fest)": ST.verlauf(m, m.maske_fix(53), bahn)}
    assert V.fig_bahn(m.preise, bahn.preis_index, vl, None) is not None
    assert V.fig_schwelle(m.preise, ST.optimum(m)[0].schwelle, None, 53, 40.0) is not None


def test_annotationen_auf_kategorieachsen_stehen_am_positionsindex():
    figs = _alle_figuren()
    for name in ("regeln", "bonitaet"):
        xs = [a.x for a in figs[name].layout.annotations]
        assert xs == list(range(len(xs))), name          # 0, 1, 2, ...: kein Name, sondern der Index der Kategorie


# ---------------------------------------------------------------------------------------------------------------- PDF
def test_pdf_ist_ein_pdf_und_vertraegt_sonderzeichen():
    zeilen = [{"name": "Preisregel (abgestimmt)", "bev": 6822.6, "vs_opt": -432.0}, {"name": "Optimum (DP)", "bev": 7254.6, "vs_opt": 0.0}]
    daten = PDF.build_pdf({"Zins %": 2.0, "sigma": 0.3}, ["**MSY** verliert 676 EUR/ha – σ* ≥ 0,129"], zeilen, ["Wärme für ≥ 3 Jahre „Zitat“"])
    assert daten[:5] == b"%PDF-" and len(daten) > 1000
