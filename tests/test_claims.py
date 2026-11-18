"""Die Zahlen der README („Befunde“) werden hier nachgerechnet, als Band (Plattformen lösen Gleichstände und BLAS-Summen verschieden auf, DEMO-PLAYBOOK §5);
die vorgerechneten Tabellen (data/regeln.csv, data/sigma_stern.csv) werden exakt gelesen, zwei Zeilen davon im Orakeltest neu abgestimmt. Alle Zahlen sind Zahlen DES LEHRMODELLS
(keine Ertragstafel, keine Holzpreisreihe). Nicht hier stehen: Laufzeiten, Rechenzeit der Koordinatensuche. Die Zufallsbahn der App ist nicht Gegenstand der Zahlen."""

from pathlib import Path

import numpy as np
import pytest

import fut_constants as C
import fut_det as D
import fut_evaluation as ev
import fut_grid as G
import fut_model as MD
import fut_rules as RU
import fut_stoch as ST
import oracle_helpers as O
from fut_model import Modell, Prm, mod

B = Prm()
C_ROOT = Path(__file__).resolve().parent.parent


def _drei(p):
    m = Modell(p)
    tf, bf = m.bestes_fix()
    return tf, bf, m.bev_msy(), ST.optimum(m, tf)[0].bev


# ---------------------------------------------------------------------------------------------------------------- Modell
def test_wachstumskurve_und_msy():
    m = Modell(B)
    assert [float(MD.volumen(B, a)) for a in (40, 60, 100)] == pytest.approx([336.4, 592.5, 911.3], abs=0.06)
    assert m.t_msy() == 68 and float(m.V[68] / 68) == pytest.approx(9.97, abs=0.005)
    assert float(MD.hazard(mod(B, risiko=1.0), 100)) == pytest.approx(0.022)


def test_preisgitter_diskretisierung_11_21_41_61_zustaende():
    erwartet = {11: 7255.2, 21: 7254.6, 41: 7254.0, 61: 7253.6}
    for n, e in erwartet.items():
        assert _drei(mod(B, sigma=0.3, phi=0.9, n_preis=n))[3] == pytest.approx(e, abs=0.06), n


# ---------------------------------------------------------------------------------------------------------------- deterministisch (Tab 2)
@pytest.mark.parametrize("zins, tf, bf, bm", [(0.25, 71, 121195.7, 120935.3), (0.5, 68, 53261.3, 53261.3), (1.0, 62, 20140.4, 19903.2), (1.5, 57, 9692.7, 9194.2),
                                              (2.0, 53, 4804.7, 4128.6), (2.5, 50, 2079.5, 1301.8), (3.0, 47, 397.0, -422.9), (3.5, 45, -711.8, -1533.6), (5.0, 40, -2434.6, -3130.8)])
def test_tabelle_faustmann_gegen_msy_je_zins(zins, tf, bf, bm):
    t, b, tm, bmsy = D.feste_regeln(mod(B, r=zins / 100.0))
    assert (t, tm) == (tf, 68) and b == pytest.approx(bf, abs=0.06) and bmsy == pytest.approx(bm, abs=0.06)


def test_verlust_der_msy_regel_liegt_im_grundfall_zwischen_0_und_822():
    verluste = []
    for zins in (0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 5.0):
        _, b, _, bm = D.feste_regeln(mod(B, r=zins / 100.0))
        verluste.append(b - bm)
    assert min(verluste) == pytest.approx(0.0, abs=1e-6) and max(verluste) == pytest.approx(821.8, abs=0.1) and verluste.index(max(verluste)) == 7   # bei 3,5 %


def test_prozentverlust_explodiert_wenn_der_bodenwert_gegen_null_geht():
    _, b, _, bm = D.feste_regeln(mod(B, r=0.03))
    assert 100 * (b - bm) / b == pytest.approx(206.5, abs=0.1)


@pytest.mark.parametrize("name, kw, be_f, be_m", [("Grundfall", {}, 3.16, 2.85), ("Nettoerlös 50", {"preis": 50.0}, 3.67, 3.19), ("Pflanzung 8000", {"c_pflanz": 8000.0}, 1.83, 1.81),
                                                  ("Pflanzung 2000", {"c_pflanz": 2000.0}, 4.92, 3.91), ("Sortiment 0,4", {"sort_praemie": 0.4}, 2.81, 2.77),
                                                  ("Risiko 0,5", {"risiko": 0.5}, 3.07, 2.75), ("Risiko 1", {"risiko": 1.0}, 2.98, 2.66), ("Risiko 2", {"risiko": 2.0}, 2.80, 2.46)])
def test_break_even_zinsen(name, kw, be_f, be_m):
    p = mod(B, **kw)
    assert 100 * D.break_even(p, "F") == pytest.approx(be_f, abs=0.006) and 100 * D.break_even(p, "M") == pytest.approx(be_m, abs=0.006)


@pytest.mark.parametrize("kw, lo, hi", [({}, 0.448, 0.520), ({"c_pflanz": 8000.0}, 1.053, 1.152), ({"sort_praemie": 0.4}, 1.455, 1.597)])
def test_kreuzungsintervall_faustmann_gleich_msy(kw, lo, hi):
    a, b = D.kreuzungsintervall(mod(B, **kw))
    assert 100 * a == pytest.approx(lo, abs=0.002) and 100 * b == pytest.approx(hi, abs=0.002)
    # darunter ist Faustmann länger als MSY, darüber kürzer
    tm = Modell(mod(B, **kw)).t_msy()
    assert Modell(mod(B, r=a - 0.0005, **kw)).bestes_fix()[0] > tm and Modell(mod(B, r=b + 0.0005, **kw)).bestes_fix()[0] < tm


def test_kalamitaet_allein_verschiebt_nur_die_umtriebszeit():
    erwartet = {0.0: (53, 4804.7), 0.5: (52, 4376.2), 1.0: (51, 3962.2), 2.0: (50, 3172.6)}
    for ri, (tf, bf) in erwartet.items():
        t, b, _, _ = D.feste_regeln(mod(B, risiko=ri))
        assert t == tf and b == pytest.approx(bf, abs=0.06)
    assert D.feste_regeln(mod(B, risiko=1.0, r=0.0001))[0] == 67          # mit Risiko 1 liegt T_F selbst bei 0,01 % Zins unter 68
    assert D.kreuzungsintervall(mod(B, risiko=1.0)) == (None, None)


# ---------------------------------------------------------------------------------------------------------------- σ, φ, Zins, Risiko
@pytest.mark.parametrize("sigma, opt, gain", [(0.1, 5252.7, 447.9), (0.2, 6137.5, 1332.8), (0.3, 7254.6, 2449.9), (0.4, 8552.7, 3748.0), (0.5, 10007.5, 5202.7)])
def test_sigma_sweep_optimum_und_gewinn(sigma, opt, gain):
    tf, bf, bm, o = _drei(mod(B, sigma=sigma, phi=0.9))
    assert (tf, bf, bm) == (53, pytest.approx(4804.7, abs=0.06), pytest.approx(4128.6, abs=0.06))     # feste Regeln: gleiche Zahlen bei jedem σ (Konstruktion)
    assert o == pytest.approx(opt, abs=0.06) and o - bf == pytest.approx(gain, abs=0.06)


def test_sigma_sweep_preisregel_aus_dem_raster():
    tab = RU.laden()
    erwartet = {0.1: (48, 11, 69, 5124.7), 0.2: (44, 12, 84, 5851.5), 0.3: (43, 12, 150, 6822.6), 0.4: (37, 13, 150, 7990.4), 0.5: (36, 13, 150, 9445.4)}
    for s, (tmin, k, tmax, bev) in erwartet.items():
        r = tab[(s, 0.9, 2.0, 4000.0, 40.0, 0.0, 0.0)]
        assert (r.tmin, r.k, r.tmax) == (tmin, k, tmax) and r.bev == pytest.approx(bev, abs=0.06)
    assert Modell(mod(B, sigma=0.3, phi=0.9)).preise[12] == pytest.approx(50.0, abs=0.05)     # Index 12 = 50 EUR/m³, 25 % über dem Mittel 40
    # Preis-Schwelle der README-Tabelle: Preis der Stufe k im Gitter der jeweiligen Zeile (nicht im σ-0,3-Gitter)
    schwellen = {0.1: 41.6, 0.2: 46.9, 0.3: 50.0, 0.4: 63.2, 0.5: 69.1}
    for s, eur in schwellen.items():
        assert float(Modell(mod(B, sigma=s, phi=0.9)).preise[tab[(s, 0.9, 2.0, 4000.0, 40.0, 0.0, 0.0)].k]) == pytest.approx(eur, abs=0.06), s
    assert "| 0,1 | 48 Jahre, 41,6 EUR/m³, 69 Jahre |" in (C_ROOT / "README.md").read_text(encoding="utf-8")
    assert Modell(mod(B, sigma=0.3, phi=0.9)).preise[0] == pytest.approx(10.0, abs=0.05) and Modell(mod(B, sigma=0.3, phi=0.9)).preise[-1] == pytest.approx(146.3, abs=0.05)


def test_flexibilitaet_gegen_msy_verlust_in_prozent_des_optimums():
    zeilen = {0.1: -21.4, 0.2: -32.7, 0.3: -43.1, 0.4: -51.7, 0.5: -58.8}
    for s, pct in zeilen.items():
        _, _, bm, o = _drei(mod(B, sigma=s, phi=0.9))
        assert 100 * (bm - o) / o == pytest.approx(pct, abs=0.06)


@pytest.mark.parametrize("phi, opt, gain", [(0.0, 12056.4, 7251.6), (0.5, 11129.6, 6324.9), (0.8, 8984.7, 4179.9), (0.9, 7254.6, 2449.9), (0.97, 5316.1, 511.3)])
def test_persistenz_sweep(phi, opt, gain):
    _, bf, _, o = _drei(mod(B, sigma=0.3, phi=phi))
    assert o == pytest.approx(opt, abs=0.06) and o - bf == pytest.approx(gain, abs=0.06)


@pytest.mark.parametrize("zins, bf, bm, opt", [(0.5, 53261.3, 53261.3, 72543.3), (1.0, 20140.4, 19903.2, 27715.8), (3.0, 397.0, -422.9, 1518.2), (3.5, -711.8, -1533.6, 98.7),
                                               (4.0, -1479.3, -2274.2, -874.4)])
def test_zins_sweep(zins, bf, bm, opt):
    _, f, m_, o = _drei(mod(B, sigma=0.3, phi=0.9, r=zins / 100.0))
    assert (f, m_, o) == (pytest.approx(bf, abs=0.06), pytest.approx(bm, abs=0.06), pytest.approx(opt, abs=0.06))


def test_hoher_zins_nur_mit_preisflexibilitaet_im_plus():
    tf, bf, bm, o = _drei(mod(B, sigma=0.3, phi=0.9, r=0.035))
    assert bf < 0 and bm < 0 and o > 0
    regel = RU.laden()[(0.3, 0.9, 3.5, 4000.0, 40.0, 0.0, 0.0)]
    assert regel.bev < 0                     # die abgestimmte Preisregel erreicht den Gewinn nicht: das Optimum braucht die vollen Schwellen je Alter


def test_preisregel_bodenwerte_der_sweeps_aus_dem_raster():
    tab = RU.laden()
    schluessel = lambda s, ph, z, ri: (s, ph, z, 4000.0, 40.0, 0.0, ri)
    erwartet = {schluessel(0.3, 0.0, 2.0, 0.0): 11726.96, schluessel(0.3, 0.5, 2.0, 0.0): 10711.87, schluessel(0.3, 0.8, 2.0, 0.0): 8572.70, schluessel(0.3, 0.97, 2.0, 0.0): 5150.76,
                schluessel(0.3, 0.9, 0.5, 0.0): 70242.87, schluessel(0.3, 0.9, 1.0, 0.0): 26517.06, schluessel(0.3, 0.9, 3.0, 0.0): 1297.21, schluessel(0.3, 0.9, 3.5, 0.0): -70.72,
                schluessel(0.3, 0.9, 4.0, 0.0): -1006.39, schluessel(0.3, 0.9, 2.0, 0.5): 6159.82, schluessel(0.3, 0.9, 2.0, 1.0): 5579.42, schluessel(0.3, 0.9, 2.0, 2.0): 4547.88,
                schluessel(0.13, 0.9, 2.0, 0.0): 5304.16}
    for key, bev in erwartet.items():
        assert tab[key].bev == pytest.approx(bev, abs=0.02), key
    # Preisregel zum Optimum in Prozent: Persistenz (README) und Kalamität (6,0 bis 7,6 %)
    for ph, pct in ((0.0, -2.7), (0.5, -3.8), (0.8, -4.6), (0.9, -6.0), (0.97, -3.1)):
        o = _drei(mod(B, sigma=0.3, phi=ph))[3]
        assert 100 * (tab[schluessel(0.3, ph, 2.0, 0.0)].bev - o) / o == pytest.approx(pct, abs=0.06)
    for ri, pct in ((0.0, -5.96), (0.5, -6.44), (1.0, -6.80), (2.0, -7.55)):
        o = _drei(mod(B, sigma=0.3, phi=0.9, risiko=ri))[3]
        assert 100 * (tab[schluessel(0.3, 0.9, 2.0, ri)].bev - o) / o == pytest.approx(pct, abs=0.006)


def test_kalamitaet_aendert_die_rangfolge_nicht():
    for ri, opt, gain in ((0.0, 7254.6, 2449.9), (0.5, 6584.0, 2207.9), (1.0, 5986.3, 2024.1), (2.0, 4919.1, 1746.4)):
        _, bf, bm, o = _drei(mod(B, sigma=0.3, phi=0.9, risiko=ri))
        assert o == pytest.approx(opt, abs=0.06) and o - bf == pytest.approx(gain, abs=0.06) and o > bf > bm
    gewinne = [_drei(mod(B, sigma=0.3, phi=0.9, risiko=ri)) for ri in (0.0, 0.5, 1.0, 2.0)]
    anteile = [100 * (o - bf) / bf for _, bf, _, o in gewinne]
    assert min(anteile) == pytest.approx(50.5, abs=0.06) and max(anteile) == pytest.approx(55.0, abs=0.06)


def test_rangfolge_und_preisregel_abstand_ueber_alle_18_zellen_des_rasters():
    tab = RU.laden()
    strikt, im_band, mehr = 0, 0, []
    for key, rg in tab.items():
        sigma, phi, zins, _, _, _, risiko = key
        _, bf, bm, o = _drei(mod(B, sigma=sigma, phi=phi, r=zins / 100.0, risiko=risiko))
        assert o >= rg.bev >= bf >= bm
        strikt += int(o > rg.bev > bf > bm)
        if o > 0 and 2.4 <= 100 * (o - rg.bev) / o <= 7.6:
            im_band += 1
        else:
            mehr.append(key)
    assert (len(tab), strikt, im_band) == (18, 17, 15)
    assert sorted((k[2], k[0]) for k in mehr) == [(3.0, 0.3), (3.5, 0.3), (4.0, 0.3)]        # Zins 3, 3,5 und 4 %: dort ist der Abstand größer (14,6 %) oder der Prozentvergleich sinnlos
    assert sorted(tab) == sorted(
        [(s, 0.9, 2.0, 4000.0, 40.0, 0.0, 0.0) for s in (0.1, 0.2, 0.3, 0.4, 0.5, 0.13)] + [(0.3, f, 2.0, 4000.0, 40.0, 0.0, 0.0) for f in (0.0, 0.5, 0.8, 0.97)]
        + [(0.3, 0.9, z, 4000.0, 40.0, 0.0, 0.0) for z in (0.5, 1.0, 3.0, 3.5, 4.0)] + [(0.3, 0.9, 2.0, 4000.0, 40.0, 0.0, r) for r in (0.5, 1.0, 2.0)])   # die 18 Zellen der README
    assert all(rg.k > 0 for rg in tab.values())                       # k = 0 (bei jedem Preis ernten) wird nie gewählt
    assert sorted(k[:3] for k, rg in tab.items() if rg.tmax == C.ALTER_MAX) == sorted([(0.3, 0.0, 2.0), (0.3, 0.5, 2.0), (0.3, 0.8, 2.0), (0.3, 0.9, 0.5), (0.3, 0.9, 1.0),
                                                                                       (0.3, 0.9, 2.0), (0.4, 0.9, 2.0), (0.5, 0.9, 2.0)])   # t_max am Rand: 8 von 18 Zellen


# ---------------------------------------------------------------------------------------------------------------- Struktur des Optimums
def test_schwelle_des_optimums_faellt_mit_dem_alter():
    m = Modell(mod(B, sigma=0.3, phi=0.9))
    o = ST.optimum(m)[0]
    # Stichproben, dazu die Bereichsgrenzen der Stufe 43,7 (74 bis 129), der Stufe 38,2 (130 bis 148) und des Alters 149 (33,4)
    erwartet = {30: 85.5, 35: 74.8, 40: 65.4, 45: 57.2, 50: 57.2, 55: 50.0, 60: 50.0, 70: 50.0, 73: 50.0, 74: 43.7, 75: 43.7, 100: 43.7, 120: 43.7, 129: 43.7,
                130: 38.2, 148: 38.2, 149: 33.4}
    for a, preis in erwartet.items():
        assert float(m.preise[o.schwelle[a]]) == pytest.approx(preis, abs=0.06), a
    stufen = {float(m.preise[o.schwelle[a]]) for a in range(74, 130)}
    assert len(stufen) == 1 and float(m.preise[o.schwelle[130]]) < min(stufen)          # 43,7 gilt nur bis 129, ab 130 ist die Schwelle niedriger
    s = [o.schwelle[a] for a in range(30, 149)]
    assert all(x >= y for x, y in zip(s, s[1:]))                       # nie steigend zwischen Alter 30 und 148
    assert o.gleichstaende == 0 and o.zustaende == 3129 and o.zusammenhaengend


def test_erntemenge_ist_in_jedem_alter_ein_bereich_nach_oben_und_die_schwelle_faellt_ueber_alle_zellen_des_rasters():
    tab = RU.laden()
    for key in tab:
        sigma, phi, zins, _, _, _, risiko = key
        o = ST.optimum(Modell(mod(B, sigma=sigma, phi=phi, r=zins / 100.0, risiko=risiko)))[0]
        assert o.zusammenhaengend and o.fallend, key
        assert o.gleichstaende == 0, key


def test_erntealter_unter_dem_optimum_per_monte_carlo():
    m = Modell(mod(B, sigma=0.3, phi=0.9))
    mask = ST.optimum(m)[1]
    _, _, ages = O.monte_carlo(m, mask, C.MC_PFADE, C.MC_JAHRE, C.MC_SEED, altersstatistik=True)
    assert ages.mean() == pytest.approx(59.8, abs=0.6) and ages.mean() > 53          # später als die feste Faustmann-Zeit 53
    q = np.percentile(ages, [10, 50, 90])
    assert q == pytest.approx([38, 56, 84], abs=2.0) and 20 <= ages.min() <= 21 and ages.max() == 150      # Minimum 20 gilt für diesen Zufalls-Seed (andere Seeds: 21)


# ---------------------------------------------------------------------------------------------------------------- σ*
def test_sigma_stern_tabelle_aus_der_datei_und_live():
    erwartet = {(0.0, 1.0): 0.0075, (0.0, 2.0): 0.0433, (0.0, 3.0): 0.0919, (0.9, 1.0): 0.0328, (0.9, 2.0): 0.1292, (0.9, 3.0): 0.2446, (0.97, 1.0): 0.1096, (0.97, 2.0): 0.3479, (0.97, 3.0): None}
    zeilen = {(r["phi"], r["zins"]): r for r in G.sigma_stern_laden()}
    assert set(zeilen) == set(erwartet)
    for key, e in erwartet.items():
        if e is None:
            assert zeilen[key]["sigma_stern"] is None
        else:
            assert zeilen[key]["sigma_stern"] == pytest.approx(e, abs=0.0006), key
    for key in ((0.9, 2.0), (0.97, 3.0), (0.0, 1.0)):            # live neu gerechnet
        star, verlust = ST.sigma_stern(mod(B, phi=key[0], r=key[1] / 100.0))
        assert verlust == pytest.approx(zeilen[key]["verlust"], abs=0.01)
        assert (star is None and zeilen[key]["sigma_stern"] is None) or star == pytest.approx(zeilen[key]["sigma_stern"], abs=0.0006)


def test_verlust_der_msy_regel_je_zins_in_der_sigma_tabelle():
    zeilen = G.sigma_stern_laden()
    assert {round(r["zins"]): round(r["verlust"], 1) for r in zeilen} == {1: 237.2, 2: 676.2, 3: 819.9}


def test_gewinn_der_flexibilitaet_bei_sigma_0_3_je_phi_und_zins():
    zeilen = {(r["phi"], r["zins"]): r["gewinn_030"] for r in G.sigma_stern_laden()}
    assert zeilen[(0.0, 1.0)] == pytest.approx(20224.7, abs=0.1) and zeilen[(0.9, 2.0)] == pytest.approx(2449.9, abs=0.1) and zeilen[(0.97, 3.0)] == pytest.approx(236.3, abs=0.1)
    assert zeilen[(0.97, 2.0)] < 676.2 < zeilen[(0.9, 2.0)]


def test_gewinn_waechst_mit_sigma_und_schwelle_ist_der_erste_schnittpunkt():
    g, tf, bf = ST.sigma_kurve(mod(B, phi=0.9), np.arange(0.0, 0.51, 0.02))
    assert g[0] == 0.0 and all(b > a for a, b in zip(g, g[1:]))
    star, verlust = ST.sigma_stern(mod(B, phi=0.9))
    assert ST.gewinn(mod(B, phi=0.9, sigma=star - 0.001), (tf, bf)) < verlust <= ST.gewinn(mod(B, phi=0.9, sigma=star + 0.001), (tf, bf))


# ---------------------------------------------------------------------------------------------------------------- Robustheit (README-Tabelle)
@pytest.mark.parametrize("name, kw, bf, opt", [
    ("Grundfall", {}, 4804.7, 7254.6), ("Sortiment 0,4", {"sort_praemie": 0.4}, 3626.8, 5640.7), ("Sortiment 0,6", {"sort_praemie": 0.6}, 3317.1, 5138.5),
    ("frühestes Hiebalter 40", {"amin": 40}, 4804.7, 7146.0), ("frühestes Hiebalter 50", {"amin": 50}, 4804.7, 6716.6), ("frühestes Hiebalter 60", {"amin": 60}, 4635.9, 5921.9),
    ("feste Hiebkosten 2000", {"c_hieb_fix": 2000.0}, 3792.5, 6346.4), ("Sortiment 0,4 + Alter 50 + Risiko 1", {"sort_praemie": 0.4, "amin": 50, "risiko": 1.0}, 2569.8, 4015.6),
    ("σ 0,2 + Sortiment 0,4 + Alter 50", {"sigma": 0.2, "sort_praemie": 0.4, "amin": 50}, 3626.8, 4677.4)])
def test_robustheit_des_gewinns_der_preisflexibilitaet(name, kw, bf, opt):
    p = mod(B, **{"sigma": 0.3, "phi": 0.9, **kw})
    _, f, _, o = _drei(p)
    assert f == pytest.approx(bf, abs=0.06) and o == pytest.approx(opt, abs=0.06)


def test_robustheit_gewinn_liegt_zwischen_27_und_68_prozent():
    anteile = []
    for kw in ({}, {"sort_praemie": 0.4}, {"sort_praemie": 0.6}, {"amin": 40}, {"amin": 50}, {"amin": 60}, {"c_hieb_fix": 2000.0},
               {"sort_praemie": 0.4, "amin": 50, "risiko": 1.0}):
        _, f, _, o = _drei(mod(B, **{"sigma": 0.3, "phi": 0.9, **kw}))
        anteile.append(100 * (o - f) / f)
    assert min(anteile) == pytest.approx(27.7, abs=0.06) and max(anteile) == pytest.approx(67.3, abs=0.06) and len(anteile) == 8
    _, f, _, o = _drei(mod(B, sigma=0.2, phi=0.9, sort_praemie=0.4, amin=50))
    assert 100 * (o - f) / f == pytest.approx(29.0, abs=0.06)


def test_pflanzoption_wirkt_kaum_bei_positivem_und_deutlich_bei_negativem_bodenwert():
    for r, ohne, mit in ((0.02, 7254.6, 7254.6), (0.035, 98.7, 98.8), (0.04, -874.4, 0.0)):
        m = Modell(mod(B, sigma=0.3, phi=0.9, r=r))
        assert m.bev(O.wertiteration(m)[0]) == pytest.approx(ohne, abs=0.06) and m.bev(O.wertiteration(m, True)[0]) == pytest.approx(mit, abs=0.06)
    m = Modell(mod(B, sigma=0.3, phi=0.9, r=0.035))
    diff = m.bev(O.wertiteration(m, True)[0]) - m.bev(O.wertiteration(m)[0])
    assert 0.0 < diff < 0.2                     # bei positivem Bodenwert (3,5 %) wirkt die Einschränkung, aber winzig (+0,12 EUR/ha)


def test_sigma_stern_bei_hohem_zins_und_zellen_bei_sigma_0_3():
    # bei höherem Zins reicht σ 0,3 auch bei φ 0,9 nicht (σ* 0,358 bei 4 %, 0,456 bei 5 %)
    for zins, erwartet in ((4.0, 0.358), (5.0, 0.456)):
        star, _ = ST.sigma_stern(mod(B, phi=0.9, r=zins / 100.0))
        assert star == pytest.approx(erwartet, abs=0.0006) and star > 0.3
    # 3 × 3 Tabelle bei σ 0,3: richtige Umtriebszeit nur bei φ 0,97 und Zins 2 oder 3 %, in den übrigen sieben Zellen die Flexibilität
    zeilen = G.sigma_stern_laden()
    richtige_zeit = sorted((r["phi"], r["zins"]) for r in zeilen if r["gewinn_030"] < r["verlust"])
    assert len(zeilen) == 9 and richtige_zeit == [(0.97, 2.0), (0.97, 3.0)]


# ---------------------------------------------------------------------------------------------------------------- Bonität
@pytest.mark.parametrize("zins, ref, tf, t_all, fest, a_star, t_ziel, ziel, msy_g, msy_1", [
    (1.0, 20189.5, (74, 62, 53), 61, -2.27, 61.25, (77, 62, 52), -0.07, -1.29, -3.66),
    (2.0, 4861.4, (63, 53, 46), 53, -3.48, 53.0, (67, 53, 45), -0.37, -14.07, -18.24),
    (3.0, 450.7, (55, 47, 42), 47, -20.07, 47.0, (59, 47, 40), -2.86, -180.07, -208.87)])
def test_bonitaeten_zieldurchmesser_gegen_feste_zeit(zins, ref, tf, t_all, fest, a_star, t_ziel, ziel, msy_g, msy_1):
    b = ev.bonitaet(mod(B, r=zins / 100.0))
    pc = lambda v: 100 * (v - b["ref"]) / b["ref"]
    assert b["ref"] == pytest.approx(ref, abs=0.06) and tuple(b["tf"][g] for g in C.BONITAET_G) == tf and b["t_all"] == t_all
    assert pc(b["fest"]) == pytest.approx(fest, abs=0.006) and b["a_star"] == a_star and tuple(b["t_ziel"][g] for g in C.BONITAET_G) == t_ziel
    assert pc(b["ziel"]) == pytest.approx(ziel, abs=0.006) and pc(b["msy_g"]) == pytest.approx(msy_g, abs=0.006) and pc(b["msy_1"]) == pytest.approx(msy_1, abs=0.006)
    assert tuple(b["t_msy_g"][g] for g in C.BONITAET_G) == (85, 68, 57)


def test_zieldurchmesser_liegt_in_allen_drei_faellen_naeher_an_faustmann_als_feste_zeit_und_msy():
    for zins in (1.0, 2.0, 3.0):
        b = ev.bonitaet(mod(B, r=zins / 100.0))
        assert b["ref"] >= b["ziel"] > b["fest"] and b["ziel"] > b["msy_g"] > b["msy_1"]


# ---------------------------------------------------------------------------------------------------------------- Handbeispiel
def test_handbeispiel_zahlen():
    import fut_hand as H
    assert H.bruch(H.bev_fest(2)) == "1780/11" and H.bruch(H.bev_fest(4)) == "105330/671"
    pol, bev = H.optimum_aufzaehlung()
    assert H.bruch(bev) == "90963860/432357" and float(bev) == pytest.approx(210.39, abs=0.005)
    assert float(H.faustmann()[1] - H.bev_fest(4)) == pytest.approx(4.84, abs=0.005) and float(bev - H.faustmann()[1]) == pytest.approx(48.57, abs=0.005)


# ---------------------------------------------------------------------------------------------------------------- Modelldetails der README
def test_modelldetails_der_readme_stehen_im_code():
    """Preisgitter, Sortimentsfaktor, Reglergrenzen und Suchparameter des README-Absatzes „Preisgitter, Sortiment, Regler und Preisregel im Einzelnen“ stimmen mit dem Code überein."""
    import inspect
    from math import comb, exp, sqrt
    from pathlib import Path

    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    assert "Preisgitter, Sortiment, Regler und Preisregel im Einzelnen" in readme

    # Preisgitter: 21 gleichabständige Stufen von −ψ bis +ψ, ψ = σ·√20, Mittelwert der Preise exakt der Nettoerlös, stationär binomial, Std. des Log-Preises σ, Autokorrelation φ
    sigma, phi = 0.3, 0.9
    pr, P, pi = MD.preisgitter(mod(B, sigma=sigma, phi=phi))
    x = np.log(pr)
    assert len(pr) == 21 == C.N_PREIS
    assert np.allclose(np.diff(x), 2 * sigma * sqrt(20) / 20) and x[-1] - x[0] == pytest.approx(2 * sigma * sqrt(20))
    assert float(pi @ pr) == pytest.approx(40.0)
    assert np.allclose(pi, [comb(20, j) / 2 ** 20 for j in range(21)]) and np.allclose(pi @ P, pi)
    mean = float(pi @ x)
    var = float(pi @ (x - mean) ** 2)
    assert sqrt(var) == pytest.approx(sigma)
    cov = float(sum(pi[i] * P[i, j] * (x[i] - mean) * (x[j] - mean) for i in range(21) for j in range(21)))
    assert cov / var == pytest.approx(phi)
    assert np.allclose(P.sum(axis=1), 1.0)
    q2 = MD.rouwenhorst(2, phi)                       # Zwei-Zustands-Matrix: Haltewahrscheinlichkeit (1 + φ)/2
    assert q2[0, 0] == pytest.approx((1 + phi) / 2) and q2[0, 1] == pytest.approx((1 - phi) / 2)

    # Sortimentsfaktor: 1 ohne Prämie, sonst logistisch mit Mitte 50 und Breite 10
    assert np.allclose(MD.qualitaet(B, np.arange(0, 151)), 1.0)
    pq = mod(B, sort_praemie=0.4)
    assert float(MD.qualitaet(pq, 50)) == pytest.approx(0.8)
    assert float(MD.qualitaet(pq, 70)) == pytest.approx(0.6 + 0.4 / (1 + exp(-2.0)))

    # Regler
    assert (C.ZINS_MIN, C.ZINS_MAX, C.PFLANZ_MIN, C.PFLANZ_MAX, C.PREIS_MIN, C.PREIS_MAX) == (0.25, 5.0, 2000.0, 8000.0, 20.0, 60.0)
    assert (C.SORT_MIN, C.SORT_MAX, C.RISIKO_MIN, C.RISIKO_MAX, C.SIGMA_MIN, C.SIGMA_MAX, C.PHI_MIN, C.PHI_MAX) == (0.0, 0.6, 0.0, 2.0, 0.0, 0.5, 0.0, 0.97)

    # Preisregel: Index 0 erntet bei jedem Preis ab t_min, t_max zwischen 5 und 150, höchstens 30 Runden je Startpunkt
    m = Modell(mod(B, sigma=0.3))
    mask = m.maske_regel(40, 0, 90)
    assert mask[40:, :].all() and not mask[:40, :].any() and mask[90:, :].all()
    assert m.maske_regel(40, 20, 90)[40:90, :20].sum() == 0 and m.maske_regel(40, 20, 90)[40:90, 20].all()
    assert C.RULE_ROUNDS == 30 and C.TF_MIN == 5 and C.ALTER_MAX == 150
    assert "range(5, A + 1)" in inspect.getsource(RU.abstimmen)

    # σ*: Gitter 0,02, danach 30 Bisektionsschritte
    assert C.SIGMA_GRID_STEP == 0.02 and "range(30)" in inspect.getsource(ST.sigma_stern)

    # Politikiteration: Hieb nur bei Vorsprung über 1e-7, sonst Warten
    assert "> C_ + 1e-7" in inspect.getsource(MD.Kern.optimum)
