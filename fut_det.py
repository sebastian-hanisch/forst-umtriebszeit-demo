"""Deterministischer Teil: Faustmann gegen MSY bei konstantem Preis (Zins, Pflanzkosten, Preis, Sortimentsprämie, Kalamität).

Alle Rechnungen laufen über die exakte Auswertung aus fut_model.py. Zinsen sind hier Dezimalzahlen je Jahr (0,02 = 2 %)."""

from dataclasses import dataclass

import numpy as np

import fut_constants as C
import fut_model as MD
from fut_model import Modell, Prm, mod


@dataclass(frozen=True)
class Det:
    tf: int
    bev_f: float
    tm: int
    bev_m: float
    verlust: float       # BEV Faustmann minus BEV MSY-Regel in EUR/ha (nie negativ: Faustmann ist die beste feste Zeit)
    be_f: float | None   # Break-even-Zins der Faustmann-Regel (BEV = 0), None wenn im Suchbereich kein Vorzeichenwechsel
    be_m: float | None   # dasselbe für die MSY-Regel


def feste_regeln(p: Prm):
    """(T_F, BEV_F, T_MSY, BEV_MSY) bei konstantem Preis p.preis (die Preisstreuung wirkt auf feste Regeln nicht)."""
    m = Modell(p)
    tf, bf = m.bestes_fix()
    return tf, bf, m.t_msy(), m.bev_msy()


def bev_kurve(p: Prm, ts=None):
    """BEV je fester Umtriebszeit T (Vektor)."""
    m = Modell(p).skalar()
    ts = list(range(max(C.TF_MIN, p.amin), p.amax + 1)) if ts is None else list(ts)
    return np.array(ts), np.array([m.bev_maske(m.maske_fix(T)) for T in ts])


def zuwachs(p: Prm, amax=None):
    """(Alter 1..amax, V(a), laufender Zuwachs V(a) - V(a-1), durchschnittlicher Zuwachs V(a)/a)."""
    amax = p.amax if amax is None else amax
    a = np.arange(1, amax + 1)
    V = MD.volumen(p, np.arange(0, amax + 1))
    return a, V[1:], np.diff(V), V[1:] / a


def _wurzel(f, lo, hi, schritte=40):
    """Nullstelle einer fallenden Funktion f in [lo, hi] durch Bisektion; None ohne Vorzeichenwechsel (f(lo) < 0 oder f(hi) > 0)."""
    flo, fhi = f(lo), f(hi)
    if flo < 0 or fhi > 0:
        return None
    for _ in range(schritte):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


R_LO, R_HI = 0.0005, 0.12   # Suchbereich der Break-even-Zinsen


def break_even(p: Prm, regel: str):
    """Zins, bei dem der BEV der Regel null wird: 'F' (Faustmann, T mit dem Zins angepasst) oder 'M' (MSY-Umtriebszeit)."""
    def f(r):
        q = mod(p, r=r)
        return feste_regeln(q)[1] if regel == "F" else feste_regeln(q)[3]
    return _wurzel(f, R_LO, R_HI)


def kreuzungsintervall(p: Prm):
    """Zinsintervall (r_lo, r_hi), in dem die Faustmann-Umtriebszeit gleich der MSY-Umtriebszeit ist; (None, None), wenn es im Bereich
    0,01 bis 8 % keines gibt (T_F ist im Zins monoton fallend, die Grenzen kommen per Bisektion)."""
    tm = Modell(p).t_msy()
    def tf(r):
        return Modell(mod(p, r=r)).bestes_fix()[0]
    lo, hi = 0.0001, 0.08
    if tf(lo) < tm or tf(hi) > tm:
        return None, None
    a, b = lo, hi            # kleinster Zins mit T_F <= tm
    for _ in range(40):
        mid = (a + b) / 2
        if tf(mid) <= tm:
            b = mid
        else:
            a = mid
    r_lo = b
    a, b = lo, hi            # größter Zins mit T_F >= tm
    for _ in range(40):
        mid = (a + b) / 2
        if tf(mid) >= tm:
            a = mid
        else:
            b = mid
    return r_lo, a


def analyse(p: Prm) -> Det:
    tf, bf, tm, bm = feste_regeln(p)
    return Det(tf, bf, tm, bm, bf - bm, break_even(p, "F"), break_even(p, "M"))
