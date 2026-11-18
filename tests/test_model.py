"""Einheiten des Kerns an der Mini-Instanz (Handbeispiel, exakte Brüche) und Randfälle jeder Definition:
Notverkauf im ersten Jahr, Zwangshieb, frühestes Hiebalter, Gitterrand des Preises, Kreuzungszins ohne Kreuzung, Gleichstand."""

from fractions import Fraction as Fr
from math import comb

import numpy as np
import pytest

import fut_constants as C
import fut_det as D
import fut_hand as H
import fut_model as MD
import fut_stoch as ST
from conftest import hand_kern
from fut_model import Kern, Modell, Prm, mod


# ---------------------------------------------------------------------------------------------------------------- Handbeispiel
def test_handbeispiel_geschlossene_formel_von_hand():
    # BEV(2) = (5 * 20 - 20 * 1,44) / (1,44 - 1) = 71,2 / 0,44 = 1780/11
    assert H.bev_fest(2) == Fr(1780, 11)
    assert H.bev_fest(1) == Fr(5)                      # (25 - 24) / 0,2
    assert H.faustmann() == (2, Fr(1780, 11)) and H.msy() == 4
    assert H.bev_fest(4) == Fr(105330, 671)


def test_kern_gleich_exakter_bruchrechnung_fuer_alle_256_politiken(hand):
    for pol in H.politiken():
        mask = np.zeros((H.AMAX + 1, 2), dtype=bool)
        for a, i in pol:
            mask[a, i] = True
        R, bev = H.auswerten(pol)
        Rk, _ = hand.auswerten(mask)
        assert hand.bev(Rk) == pytest.approx(float(bev), rel=1e-12, abs=1e-9)
        assert Rk == pytest.approx([float(x) for x in R], rel=1e-12, abs=1e-9)


def test_howard_findet_das_optimum_der_aufzaehlung(hand):
    pol, bev = H.optimum_aufzaehlung()
    R, S, mask, it = hand.optimum(2)
    assert hand.bev(R) == pytest.approx(float(bev), rel=1e-12)
    assert {(a, i) for a in range(1, H.AMAX) for i in (0, 1) if mask[a, i]} == set(pol)
    assert H.beschreibung(pol) == [(1, "nie"), (2, "bei hohem Preis"), (3, "bei hohem Preis"), (4, "bei hohem Preis")]


def test_kern_mit_kalamitaet_gleich_exakter_bruchrechnung():
    haz = [Fr(1, 10), Fr(1, 20), Fr(1, 20), Fr(1, 10), Fr(0)]
    k = hand_kern(haz=[float(x) for x in haz])
    rng = np.random.default_rng(3)
    for _ in range(25):
        mask = rng.random((H.AMAX + 1, 2)) < 0.4
        pol = frozenset((a, i) for a in range(1, H.AMAX) for i in (0, 1) if mask[a, i])
        assert k.bev(k.auswerten(mask)[0]) == pytest.approx(float(H.auswerten(pol, dict(enumerate(haz)))[1]), rel=1e-12)
    pol, bev = H.optimum_aufzaehlung(dict(enumerate(haz)))
    assert k.bev(k.optimum(2)[0]) == pytest.approx(float(bev), rel=1e-12)


def test_feste_politik_bei_zwei_preiszustaenden_hat_den_wert_des_konstanten_preises(hand):
    # Invarianz im Mini-Fall: Mittelwert 5, BEV der festen Umtriebszeit gleich dem bei konstantem Preis 5
    for T in (1, 2, 3, 4, 5):
        assert hand.bev(hand.auswerten(hand.maske_fix(T))[0]) == pytest.approx(float(H.bev_fest(T)), rel=1e-12)


# ---------------------------------------------------------------------------------------------------------------- Randfälle
@pytest.mark.parametrize("zwei_zustaende", [False, True])
def test_notverkauf_im_ersten_jahr_von_hand(zwei_zustaende):
    """amax = 1, Preis 5, V(1) = 5, Rate im Alter 0: 1/10, Notverkauf 1/2, Zins 1/5, Pflanzung 20. R (1 - d) = -20 + d (0,9 * 25 + 0,1 * 12,5) -> R = -5/4."""
    n = 2 if zwei_zustaende else 1
    P = np.full((n, n), 1.0 / n)
    hv = np.full((3, n), 25.0)
    k = Kern(hv, 0.5 * hv, [0.1, 0.0, 0.0], P, np.full(n, 1.0 / n), 1 / 1.2, 20.0, 1)
    R, _ = k.auswerten(np.zeros((2, n), dtype=bool))
    assert R == pytest.approx([-1.25] * n, rel=1e-12)
    # ohne diese Kalamität (Rate 0) wäre es (-20 + 25/1,2) * 6 = 5/6 * ... : die Kalamität im ersten Jahr senkt den Wert
    k0 = Kern(hv, 0.5 * hv, [0.0, 0.0, 0.0], P, np.full(n, 1.0 / n), 1 / 1.2, 20.0, 1)
    assert k0.auswerten(np.zeros((2, n), dtype=bool))[0][0] == pytest.approx((-20 + 25 / 1.2) * 6, rel=1e-12)


def test_zwangshieb_im_hoechsten_alter_wird_erzwungen():
    k = hand_kern()
    leer = np.zeros((H.AMAX + 1, 2), dtype=bool)
    assert k.bev(k.auswerten(leer)[0]) == pytest.approx(k.bev(k.auswerten(k.maske_fix(H.AMAX))[0]), rel=1e-12)
    assert k.bev(k.auswerten(leer)[0]) == pytest.approx(float(H.bev_fest(5)), rel=1e-12)


def test_fruehestes_hiebalter_wird_nie_unterschritten():
    p = mod(Prm(), sigma=0.3, phi=0.9, amin=60)
    m = Modell(p)
    tf, bf = m.bestes_fix()
    assert tf >= 60                                         # feste Zeit darf nicht kürzer sein
    R, S, mask, _ = m.optimum(tf)
    assert not mask[:60].any()                              # keine Hiebe vor dem frühesten Alter, auch nicht bei Höchstpreisen
    assert mask[60:, :].any()
    assert m.bev(R) >= bf - 1e-9
    assert Modell(mod(p, sigma=0.0)).t_msy() == 68          # die MSY-Zeit liegt über dem frühesten Alter
    assert Modell(mod(p, amin=100)).t_msy() == 100          # darunter wäre sie unzulässig: Gleichung gilt ab amin


def test_gitterrand_des_preises_regel_mit_hoechstem_index_erntet_nur_im_obersten_zustand():
    m = Modell(mod(Prm(), sigma=0.3, phi=0.9))
    mask = m.maske_regel(40, m.n - 1, 120)
    assert mask[40:120, m.n - 1].all() and not mask[40:120, : m.n - 1].any()
    assert mask[120:, :].all() and not mask[:40].any()
    assert not m.maske_regel(40, m.n, 120)[40:120].any()    # Index jenseits des Gitters: kein Preis erreicht die Schwelle, nur der Zwangshieb zählt


def test_preisregel_mit_index_null_und_gleichen_altern_ist_die_feste_zeit():
    m = Modell(mod(Prm(), sigma=0.3, phi=0.9))
    assert np.array_equal(m.maske_regel(53, 0, 53), m.maske_fix(53))


def test_gleichstand_hieb_gegen_warten_wird_als_warten_entschieden():
    """Ein Zustand, amax 2, Zins 1 (Abzinsung 1/2), Hiebwerte 10 (Alter 1) und 16 (Alter 2), Pflanzung 7. Hieb im Alter 1 gibt R = -7 + (10 + R)/2 = -4,
    Warten gibt R = -7 + (16 + R)/4 = -4: Gleichstand (Hieb 6 gegen Warten 6). Die Politikiteration lässt dann das Warten stehen."""
    hv = np.array([[0.0], [10.0], [16.0], [16.0]])
    k = Kern(hv, 0.5 * hv, np.zeros(4), [[1.0]], [1.0], 0.5, 7.0, 2)
    assert k.auswerten(k.maske_fix(1))[0][0] == pytest.approx(-4.0) and k.auswerten(k.maske_fix(2))[0][0] == pytest.approx(-4.0)
    R, S, mask, _ = k.optimum(1)
    assert not mask[1, 0] and mask[2, 0]
    assert R[0] == pytest.approx(-4.0) and k.abstand_hieb_warten(R, S)[1, 0] == pytest.approx(0.0, abs=1e-12)


def test_kreuzungsintervall_ohne_kreuzung_und_mit():
    p = Prm()
    lo, hi = D.kreuzungsintervall(p)
    assert lo < hi
    assert Modell(mod(p, r=(lo + hi) / 2)).bestes_fix()[0] == Modell(p).t_msy()
    assert D.kreuzungsintervall(mod(p, risiko=1.0)) == (None, None)


def test_prm_pruefung():
    with pytest.raises(ValueError):
        Prm(phi=1.0)
    with pytest.raises(ValueError):
        Prm(risiko=50.0)
    with pytest.raises(ValueError):
        Prm(sigma=-0.1)


def test_wachstum_und_hazard_werte_von_hand():
    p = Prm()
    # V(a) = 1100 (1 - e^(-0,028 a))^3: a = 0 ist 0; a -> unendlich ist 1100
    assert MD.volumen(p, 0) == 0.0 and MD.volumen(p, 1e4) == pytest.approx(1100.0)
    assert MD.volumen(p, 68) == pytest.approx(1100 * (1 - np.exp(-0.028 * 68)) ** 3)
    assert MD.hazard(mod(p, risiko=1.0), 100) == pytest.approx(0.002 + 0.02)
    assert MD.hazard(mod(p, risiko=2.0), 0) == pytest.approx(0.004)
    assert MD.qualitaet(p, 10) == 1.0
    q = mod(p, sort_praemie=0.4)
    assert MD.qualitaet(q, 50) == pytest.approx(0.8)         # Mitte: 1 - s + s / 2
    assert MD.qualitaet(q, 200) == pytest.approx(1.0, abs=1e-6) and MD.qualitaet(q, 0) == pytest.approx(0.6, abs=0.01)


def test_rouwenhorst_eigenschaften_exakt():
    for n, phi in ((2, 0.3), (5, 0.9), (21, 0.0), (21, 0.97)):
        P = MD.rouwenhorst(n, phi)
        assert P.sum(axis=1) == pytest.approx(1.0)
        pi = np.array([comb(n - 1, j) for j in range(n)], dtype=float)
        pi /= pi.sum()
        assert pi @ P == pytest.approx(pi, abs=1e-12)        # stationär
        x = np.arange(n, dtype=float)
        mu = pi @ x
        var = pi @ (x - mu) ** 2
        cov = sum(pi[i] * P[i, j] * (x[i] - mu) * (x[j] - mu) for i in range(n) for j in range(n))
        assert cov / var == pytest.approx(phi, abs=1e-12)    # Lag-1-Korrelation der Indizes = phi


def test_preisgitter_mittel_streuung_und_autokorrelation():
    for sigma, phi in ((0.1, 0.0), (0.3, 0.9), (0.5, 0.97)):
        m = Modell(mod(Prm(preis=40.0), sigma=sigma, phi=phi))
        x = np.log(m.preise)
        mu = m.pi @ x
        sd = np.sqrt(m.pi @ (x - mu) ** 2)
        cov = sum(m.pi[i] * m.P[i, j] * (x[i] - mu) * (x[j] - mu) for i in range(m.n) for j in range(m.n))
        assert m.pi @ m.preise == pytest.approx(40.0, rel=1e-12)
        assert sd == pytest.approx(sigma, rel=1e-12) and cov / sd**2 == pytest.approx(phi, abs=1e-12)
    assert Modell(Prm()).n == 1                                  # sigma = 0: ein Zustand


# ---------------------------------------------------------------------------------------------------------------- Bahn und Verlauf
def test_bahn_ist_deterministisch_und_gueltig():
    m = Modell(mod(Prm(), sigma=0.3, phi=0.9))
    a, b = ST.bahn_ziehen(m, 7), ST.bahn_ziehen(m, 7)
    assert a == b and a != ST.bahn_ziehen(m, 8)
    assert len(a.preis_index) == C.BAHN_JAHRE + 1 and all(0 <= i < m.n for i in a.preis_index)
    assert max(abs(x - y) for x, y in zip(a.preis_index, a.preis_index[1:])) <= m.n - 1
    assert all(0.0 <= u < 1.0 for u in a.u_kalamitaet)


def test_verlauf_von_hand():
    k = hand_kern()
    bahn = ST.Bahn((0, 1, 1, 0, 1, 1), (0.9,) * 6)
    v = ST.verlauf(k, k.maske_fix(2), bahn)
    assert v.alter == (0, 1, 0, 1, 0, 1) and v.hiebe == (2, 4) and v.notverkaeufe == () and v.mittleres_erntealter == 2.0
    # Optimum-Politik des Handbeispiels: Alter 1 nie, Alter 2 bis 4 nur bei hohem Preis (Index 1)
    opt = k.optimum(2)[2]
    v = ST.verlauf(k, opt, ST.Bahn((0, 0, 1, 0, 0, 0, 1, 0), (0.9,) * 8))
    assert v.hiebe == (2, 6) and v.alter == (0, 1, 0, 1, 2, 3, 0, 1)   # Jahr 2: Alter 2 bei hohem Preis; dann Alter 1 .. 4 ohne hohen Preis: Jahr 6 hoch
    # Kalamität im ersten Jahr (Rate 1/10 im Alter 0): Notverkauf, Alter bleibt 0
    kh = hand_kern(haz=[0.1])
    v = ST.verlauf(kh, kh.maske_fix(5), ST.Bahn((0, 1, 1), (0.0, 0.05, 0.9)))
    assert v.hiebe == (1,) and v.notverkaeufe == (1,) and v.alter == (0, 0, 1)
    # u genau gleich der Rate: keine Kalamität (strikt kleiner)
    v = ST.verlauf(kh, kh.maske_fix(5), ST.Bahn((0, 1), (0.0, 0.1)))
    assert v.notverkaeufe == ()
    assert ST.Verlauf((0,), (), ()).mittleres_erntealter is None
