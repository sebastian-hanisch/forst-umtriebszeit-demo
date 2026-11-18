"""Orakel-Tests (DEMO-PLAYBOOK §12): jede Kernroutine gegen einen ANDEREN Rechenweg (Rechenwege in tests/oracle_helpers.py und fut_hand.py).

1. geschlossene Faustmann-Formel gegen die Auswertung (13 Fälle), 2. Zyklus-Erwartung mit Kalamität und Notverkauf (9 Fälle),
3. Aufzählung aller festen Umtriebszeiten auf einer Mini-Instanz, 4. Howard-Politikiteration gegen einfache Wertiteration (4 Szenarien),
5. Rückwärtsrekursion gegen das ganze Gleichungssystem (zufällige Politiken), 6. Monte-Carlo gegen die exakte Auswertung (|z| < 3, 3 Politiken),
7. Invarianz fester Regeln gegen σ und φ, 8. σ = 0: Optimum gleich Faustmann, 9. Eigenschaften (Rangfolge der Regeln, Monotonie im Zins),
10. Pflanzoption (Wartezustand) gegen die erzwungene Wiederaufforstung, 11. Preisregel-Abstimmung gegen Aufzählung auf einer Mini-Instanz,
12. Rouwenhorst-Gitter (Mittel, Streuung, Autokorrelation) und Handbeispiel in Brüchen (tests/test_model.py)."""

import numpy as np
import pytest

import fut_constants as C
import fut_rules as RU
import fut_stoch as ST
import oracle_helpers as O
from fut_model import Modell, Prm, mod

B = Prm()   # Nettoerlös 40, Pflanzung 4000, Zins 2 %


def test_01_geschlossene_faustmann_formel():
    n = 0
    for r in (0.01, 0.02, 0.03):
        for T in (40, 55, 70, 100):
            p = mod(B, r=r)
            m = Modell(p)
            a = m.bev_maske(m.maske_fix(T))
            assert a == pytest.approx(O.faust_t(p, T), rel=1e-9), (r, T)
            n += 1
    p = mod(B, sort_praemie=0.4, c_hieb_fix=500.0)
    m = Modell(p)
    assert m.bev_maske(m.maske_fix(60)) == pytest.approx(O.faust_t(p, 60), rel=1e-9)
    assert n == 12


def test_02_zyklus_erwartung_mit_kalamitaet_und_notverkauf():
    for ri in (0.5, 1.0, 2.0):
        for T in (45, 60, 80):
            p = mod(B, risiko=ri, sort_praemie=0.3)
            m = Modell(p)
            assert m.bev_maske(m.maske_fix(T)) == pytest.approx(O.zyklus_geschlossen(p, T), rel=1e-9), (ri, T)


def test_03_aufzaehlung_aller_festen_umtriebszeiten_auf_der_mini_instanz():
    p = mod(B, amax=40, risiko=1.0)
    m = Modell(p)
    werte = {T: O.zyklus_geschlossen(p, T) for T in range(5, 41)}
    T_best = max(werte, key=werte.get)
    assert m.bestes_fix() == (T_best, pytest.approx(werte[T_best], rel=1e-9))
    # Kurve BEV(T) über alle T gleich der Aufzählung
    ts, bev = __import__("fut_det").bev_kurve(p)
    assert list(ts) == list(range(5, 41)) and bev == pytest.approx([werte[T] for T in ts], rel=1e-9)


@pytest.mark.parametrize("s, ph, r, ri", [(0.0, 0.9, 0.02, 0.0), (0.3, 0.9, 0.02, 0.0), (0.3, 0.5, 0.03, 1.0), (0.2, 0.97, 0.02, 1.0)])
def test_04_howard_gegen_wertiteration(s, ph, r, ri):
    m = Modell(mod(B, sigma=s, phi=ph, r=r, risiko=ri))
    R1, _, _, _ = m.optimum(m.bestes_fix()[0])
    R2, _ = O.wertiteration(m)
    assert np.max(np.abs(R1 - R2)) < 1e-5


def test_05_rueckwaertsrekursion_gegen_gesamtsystem():
    p = mod(B, sigma=0.3, phi=0.9, amax=12, n_preis=7, risiko=1.0, r=0.04)
    m = Modell(p)
    rng = np.random.default_rng(1)
    for _ in range(12):
        mask = rng.random((p.amax + 1, m.n)) < rng.choice([0.05, 0.2, 0.5])
        assert m.bev(m.auswerten(mask)[0]) == pytest.approx(O.gesamtsystem(m, mask)[0], rel=1e-10)
    # und das Optimum (Politikiteration) als Gleichungssystem gelöst
    R, S, mask, _ = m.optimum(6)
    assert O.gesamtsystem(m, mask)[0] == pytest.approx(m.bev(R), rel=1e-10)


@pytest.mark.parametrize("name", ["fest", "regel", "optimum"])
def test_06_monte_carlo_gegen_exakt(name):
    p = mod(B, sigma=0.3, phi=0.9, risiko=1.0)
    m = Modell(p)
    mask = {"fest": lambda: m.maske_fix(53), "regel": lambda: m.maske_regel(44, 12, 95), "optimum": lambda: m.optimum(m.bestes_fix()[0])[2]}[name]()
    exakt = m.bev(m.auswerten(mask)[0])
    est, se = O.monte_carlo(m, mask, C.MC_PFADE, C.MC_JAHRE, C.MC_SEED)
    assert abs(est - exakt) / se < 3.0, (name, exakt, est, se)


def test_07_invarianz_fester_regeln_gegen_sigma_und_phi():
    ref = None
    for s, ph in ((0.0, 0.9), (0.1, 0.0), (0.3, 0.9), (0.5, 0.97)):
        m = Modell(mod(B, sigma=s, phi=ph))
        # ohne den schnellen Pfad: volle 21-Zustände-Auswertung
        voll = m.bev(Modell.auswerten(m, m.maske_fix(53))[0])
        ref = voll if ref is None else ref
        assert voll == pytest.approx(ref, abs=1e-6)
        assert m.bev_msy() == pytest.approx(Modell(B).bev_msy(), abs=1e-6)
    assert ref == pytest.approx(4804.74, abs=0.01)


def test_08_ohne_preisstreuung_ist_das_optimum_die_faustmann_zeit():
    for r in (0.01, 0.02, 0.03):
        for ri in (0.0, 1.0):
            m = Modell(mod(B, r=r, risiko=ri))
            tf, bf = m.bestes_fix()
            opt = ST.optimum(m, tf)[0]
            assert opt.bev == pytest.approx(bf, abs=1e-6) and opt.iterationen == 1


def test_09_eigenschaften_rangfolge_und_monotonie_im_zins():
    tab = RU.laden()
    gefunden = 0
    for key, regel in tab.items():
        sigma, phi, zins, pflanz, preis, sort, risiko = key
        p = mod(B, sigma=sigma, phi=phi, r=zins / 100.0, risiko=risiko)
        m = Modell(p)
        tf, bf = m.bestes_fix()
        opt = ST.optimum(m, tf)[0].bev
        assert opt >= regel.bev - 1e-6 and regel.bev >= bf - 1e-6 and bf >= m.bev_msy() - 1e-6, key   # Optimum >= Preisregel >= fest >= MSY
        gefunden += 1
    assert gefunden == len(tab) >= 18
    # monoton fallend im Zins: Bodenwert (fest und Optimum) und Faustmann-Umtriebszeit
    zinsen = (0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04)
    ms = [Modell(mod(B, sigma=0.3, phi=0.9, r=r)) for r in zinsen]
    fest = [m.bestes_fix() for m in ms]
    opt = [ST.optimum(m, f[0])[0].bev for m, f in zip(ms, fest)]
    assert all(a > b for a, b in zip([f[1] for f in fest], [f[1] for f in fest][1:]))
    assert all(a > b for a, b in zip(opt, opt[1:]))
    assert all(a >= b for a, b in zip([f[0] for f in fest], [f[0] for f in fest][1:]))


def test_10_pflanzoption_gegen_erzwungene_wiederaufforstung():
    for r, erwartet in ((0.02, 7254.616), (0.04, -874.388)):
        m = Modell(mod(B, sigma=0.3, phi=0.9, r=r))
        R0, _ = O.wertiteration(m)
        R1, _ = O.wertiteration(m, pflanzoption=True)
        assert m.bev(R0) == pytest.approx(erwartet, abs=0.01)
        assert m.bev(R1) == pytest.approx(max(erwartet, 0.0), abs=0.01)


def test_11_preisregel_abstimmung_gegen_aufzaehlung_auf_der_mini_instanz():
    p = mod(B, sigma=0.3, phi=0.9, amax=30, n_preis=5, r=0.04)
    m = Modell(p)
    tf, bf = m.bestes_fix()
    best = max(m.bev_maske(m.maske_regel(tmin, k, tmax)) for tmin in range(5, 31) for tmax in range(tmin, 31) for k in range(m.n))
    regel = RU.abstimmen(m, tf)
    assert regel.bev <= best + 1e-9 and regel.bev >= bf - 1e-9
    assert regel.bev == pytest.approx(best, abs=1e-6)       # auf dieser Mini-Instanz findet die Koordinatensuche das Gesamtoptimum
    assert m.bev_maske(m.maske_regel(regel.tmin, regel.k, regel.tmax)) == pytest.approx(regel.bev, rel=1e-12)


def test_12_vorgerechnete_regeln_stimmen_mit_der_abstimmung_ueberein():
    tab = RU.laden()
    for key in ((0.2, 0.9, 2.0, 4000.0, 40.0, 0.0, 0.0), (0.3, 0.97, 2.0, 4000.0, 40.0, 0.0, 0.0)):
        p = mod(B, sigma=key[0], phi=key[1], r=key[2] / 100.0)
        m = Modell(p)
        live = RU.abstimmen(m, m.bestes_fix()[0])
        assert (live.tmin, live.k, live.tmax) == (tab[key].tmin, tab[key].k, tab[key].tmax) and live.bev == pytest.approx(tab[key].bev, abs=1e-3)
