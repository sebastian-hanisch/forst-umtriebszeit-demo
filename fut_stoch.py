"""Preis als Zufall: Optimum (Howard-DP über Alter x Preis), Schwelle σ*, Zufallsbahn und Hiebe der vier Regeln auf derselben Bahn.

Exakte Werte (BEV je Regel) kommen aus der exakten Auswertung; die Simulation dient nur der Bahn in der Abbildung."""

from dataclasses import dataclass

import numpy as np

import fut_constants as C
from fut_model import Modell, Prm, mod


@dataclass(frozen=True)
class Opt:
    bev: float
    iterationen: int
    gleichstaende: int          # Zustände (Alter x Preis) mit |Hieb - Warten| < 0,001 EUR (nur erlaubte Alter, ohne Zwangshieb)
    zustaende: int              # Zahl dieser Zustände
    schwelle: tuple             # je Alter 0..amax: niedrigster Preisindex, bei dem geerntet wird (-1: nie)
    zusammenhaengend: bool      # Erntemenge in jedem Alter ein Bereich nach oben (alle Preise ab einer Schwelle)
    fallend: bool               # Schwelle (Preisindex) von Alter zu Alter nie steigend, ab dem ersten Alter mit Hieb bis zum Zwangshieb


def optimum(m: Modell, tf=None):
    """Optimum der Politik: (Opt, mask, R, S). Startpunkt der Howard-Iteration ist die feste Faustmann-Umtriebszeit."""
    if tf is None:
        tf = m.bestes_fix()[0]
    R, S, mask, it = m.optimum(tf)
    d = m.abstand_hieb_warten(R, S)
    lo = max(1, m.amin)
    eng = int((np.abs(d[lo:m.amax]) < 1e-3).sum())
    schwelle, zus = [], True
    for a in range(m.amax + 1):
        hit = np.nonzero(mask[a])[0]
        schwelle.append(int(hit.min()) if len(hit) else -1)
        if len(hit) and hit.max() - hit.min() + 1 != len(hit):
            zus = False
    ab = [s for a, s in enumerate(schwelle) if a >= 1 and s >= 0]
    fall = all(x >= y for x, y in zip(ab, ab[1:]))
    return Opt(m.bev(R), it, eng, (m.amax - lo) * m.n, tuple(schwelle), zus, fall), mask, R, S


def bev_optimum(p: Prm, tf=None) -> float:
    m = Modell(p)
    return optimum(m, tf)[0].bev


def gewinn(p: Prm, tf_und_bev=None):
    """Gewinn der Preisflexibilität: BEV Optimum minus BEV der besten festen Umtriebszeit."""
    m = Modell(p)
    tf, bf = m.bestes_fix() if tf_und_bev is None else tf_und_bev
    return optimum(m, tf)[0].bev - bf


def sigma_kurve(p: Prm, sigmas):
    """Gewinn der Preisflexibilität für jedes sigma (alle übrigen Parameter von p); (Array Gewinn, T_F, BEV_F)."""
    tf, bf = Modell(mod(p, sigma=0.0)).bestes_fix()
    out = []
    for s in sigmas:
        out.append(0.0 if s <= 0 else gewinn(mod(p, sigma=float(s)), (tf, bf)))
    return np.array(out), tf, bf


def sigma_stern(p: Prm, sigma_max=C.SIGMA_MAX, schritt=C.SIGMA_GRID_STEP):
    """Kleinstes σ, bei dem der Gewinn der Preisflexibilität den Verlust der MSY-Regel erreicht.
    Rückgabe (σ*, Verlust der MSY-Regel in EUR/ha). σ* = 0.0 bei Verlust 0 (nichts zu übertreffen); None, wenn der Gewinn bis sigma_max
    nicht reicht. Grobes Gitter, dann Bisektion im Gitterintervall (der Gewinn wächst mit σ)."""
    m0 = Modell(mod(p, sigma=0.0))
    tf, bf = m0.bestes_fix()
    verlust = bf - m0.bev_msy()
    if verlust <= 1e-9:
        return 0.0, verlust
    sig = np.arange(schritt, sigma_max + 1e-9, schritt)
    g = [gewinn(mod(p, sigma=float(s)), (tf, bf)) for s in sig]
    ab = [i for i, v in enumerate(g) if v >= verlust]
    if not ab:
        return None, verlust
    i = ab[0]
    lo, hi = (0.0 if i == 0 else float(sig[i - 1])), float(sig[i])
    for _ in range(30):
        mid = (lo + hi) / 2
        if gewinn(mod(p, sigma=mid), (tf, bf)) >= verlust:
            hi = mid
        else:
            lo = mid
    return hi, verlust


# ---- Zufallsbahn und Hiebe der Regeln ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Bahn:
    preis_index: tuple      # Preiszustand je Jahr 0..J
    u_kalamitaet: tuple     # Gleichverteilte Zahl je Jahr 1..J (Kalamität tritt ein, wenn sie kleiner als die Rate ist; Index 0 unbenutzt)


def bahn_ziehen(m: Modell, seed: int, jahre: int = C.BAHN_JAHRE) -> Bahn:
    """Preiszustände (Start aus der stationären Verteilung) und Kalamitäts-Zufallszahlen. Es werden nur gleichverteilte Zahlen des Generators
    benutzt (Zustand per Suche in der kumulierten Verteilung): die Bahn ist auf jeder Plattform bitgleich."""
    rng = np.random.default_rng(seed)
    u_pre = rng.random(jahre + 1)
    u_kal = rng.random(jahre + 1)
    cum_pi = np.cumsum(m.pi)
    cum = np.cumsum(m.P, axis=1)
    i = min(int(np.searchsorted(cum_pi, u_pre[0], side="right")), m.n - 1)
    zust = [i]
    for t in range(1, jahre + 1):
        i = min(int(np.searchsorted(cum[i], u_pre[t], side="right")), m.n - 1)
        zust.append(i)
    return Bahn(tuple(zust), tuple(float(x) for x in u_kal))


@dataclass(frozen=True)
class Verlauf:
    alter: tuple      # Alter des Bestands je Jahr 0..J (nach einem Hieb 0)
    hiebe: tuple      # Jahre, in denen geerntet wurde (planmäßig oder per Notverkauf)
    notverkaeufe: tuple   # Jahre, in denen eine Kalamität den Notverkauf erzwang

    @property
    def mittleres_erntealter(self):
        if not self.hiebe:
            return None
        return float(np.mean([self.alter[t - 1] + 1 for t in self.hiebe]))


def verlauf(m: Modell, mask, bahn: Bahn) -> Verlauf:
    """Bestand unter der Politik `mask` auf der gegebenen Bahn. Reihenfolge je Jahr: neuer Preiszustand, dann Kalamität (Rate nach Alter zu
    Jahresbeginn), sonst Hieb, wenn die Politik im neuen Alter und Preiszustand erntet. Gleiche Zufallszahl der Kalamität für alle Regeln."""
    mask = np.array(mask, dtype=bool)
    mask[m.amax, :] = True
    age, alter, hiebe, notv = 0, [0], [], []
    for t in range(1, len(bahn.preis_index)):
        j = bahn.preis_index[t]
        a1 = age + 1
        if bahn.u_kalamitaet[t] < m.h[age]:
            hiebe.append(t)
            notv.append(t)
            age = 0
        elif mask[a1, j]:
            hiebe.append(t)
            age = 0
        else:
            age = a1
        alter.append(age)
    return Verlauf(tuple(alter), tuple(hiebe), tuple(notv))
