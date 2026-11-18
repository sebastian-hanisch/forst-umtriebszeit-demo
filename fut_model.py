"""Waldmodell und exakte Auswertung (nur numpy).

Ein Hektar, unendliche Folge von Umtrieben, Jahresschritte, Wiederaufforstung sofort (Faustmann-Annahme). Zustand: Alter a des Bestands
und Preisindex i. Zeitpunkte sind Jahresanfänge: ein Hieb im Alter a bringt hv[a, i] und löst die Wiederaufforstung im selben Zeitpunkt aus
(Kosten c_pflanz, enthalten in R). Eine Kalamität im Jahr von Alter a auf a + 1 (Wahrscheinlichkeit h[a]) erzwingt den Notverkauf
sal[a + 1, j] im neuen Preiszustand j, danach Wiederaufforstung.

R[i] = Bodenertragswert: Wert des unbestockten Bodens im Preiszustand i vor der Pflanzung. BEV = Mittel über die stationäre Preisverteilung.
S[a, i] = Wert eines Bestands im Alter a und Preiszustand i (Hieb jetzt oder Warten gemäß Politik), einschließlich R danach.
Jede Politik wird EXAKT ausgewertet (lineares Gleichungssystem für R), keine Simulation und keine Einschwingzeit; das Optimum findet die
Howard-Politikiteration."""

from dataclasses import dataclass, replace
from math import comb

import numpy as np

import fut_constants as C


@dataclass(frozen=True)
class Prm:
    vmax: float = C.VMAX
    k: float = C.WACHSTUM_K
    pexp: float = C.WACHSTUM_P
    g: float = 1.0                      # Bonität als Zeitskalierung des Wachstums
    preis: float = C.DEFAULT_PREIS      # Nettoerlös je m³ (Mittelwert bei zufälligem Preis)
    c_pflanz: float = C.DEFAULT_PFLANZ
    c_hieb_fix: float = 0.0             # feste Hiebkosten je Hektar (nur für die Robustheitsprüfung)
    r: float = C.DEFAULT_ZINS / 100.0
    sort_praemie: float = C.DEFAULT_SORT
    risiko: float = C.DEFAULT_RISIKO
    h0: float = C.HAZARD_H0
    h1: float = C.HAZARD_H1
    salvage: float = C.NOTVERKAUF
    sigma: float = 0.0                  # 0 = konstanter Preis
    phi: float = C.DEFAULT_PHI
    n_preis: int = C.N_PREIS
    amax: int = C.ALTER_MAX
    amin: int = 0                       # frühestes erlaubtes Hiebalter (nur für die Robustheitsprüfung)

    def __post_init__(self):
        if not (0.0 <= self.phi < 1.0):
            raise ValueError("phi muss in [0, 1) liegen")
        if self.sigma < 0 or self.r <= -1.0 or self.risiko < 0:
            raise ValueError("ungültige Eingabe (sigma, r oder risiko)")
        if self.risiko * (self.h0 + self.h1 * ((self.amax + 1) / 100.0) ** 3) >= 1.0:
            raise ValueError("Kalamitäts-Rate von 1 oder mehr je Jahr")


def mod(p: Prm, **kw) -> Prm:
    return replace(p, **kw)


def volumen(p: Prm, a):
    """Chapman-Richards: V(a) = vmax (1 - exp(-k g a))^p."""
    a = np.asarray(a, dtype=float)
    return p.vmax * (1.0 - np.exp(-p.k * p.g * a)) ** p.pexp


def qualitaet(p: Prm, a):
    """Sortimentsfaktor: 1, oder logistisch von 1 - s (junges Holz) auf 1 (starkes Holz)."""
    a = np.asarray(a, dtype=float)
    if p.sort_praemie <= 0:
        return np.ones_like(a)
    qmin = 1.0 - p.sort_praemie
    return qmin + (1.0 - qmin) / (1.0 + np.exp(-(a - C.SORT_MITTE) / C.SORT_BREITE))


def hazard(p: Prm, a):
    """Kalamitätsrate im Jahr von Alter a auf a + 1."""
    a = np.asarray(a, dtype=float)
    return p.risiko * (p.h0 + p.h1 * (a / 100.0) ** 3)


def rouwenhorst(n: int, phi: float) -> np.ndarray:
    """Übergangsmatrix des symmetrischen Rouwenhorst-Verfahrens (stationär binomial, Lag-1-Korrelation der Indizes = phi)."""
    q = (1.0 + phi) / 2.0
    P = np.array([[q, 1 - q], [1 - q, q]])
    for m in range(3, n + 1):
        Q = np.zeros((m, m))
        Q[:-1, :-1] += q * P
        Q[:-1, 1:] += (1 - q) * P
        Q[1:, :-1] += (1 - q) * P
        Q[1:, 1:] += q * P
        Q[1:-1] /= 2.0
        P = Q
    return P


def preisgitter(p: Prm):
    """(Preise aufsteigend, Übergangsmatrix, stationäre Verteilung). Mittelwert exakt p.preis, Std. des Logs = sigma, Autokorrelation phi."""
    if p.sigma <= 0:
        return np.array([p.preis]), np.array([[1.0]]), np.array([1.0])
    n = p.n_preis
    P = rouwenhorst(n, p.phi)
    pi = np.array([comb(n - 1, j) for j in range(n)], dtype=float)
    pi /= pi.sum()
    psi = p.sigma * np.sqrt(n - 1)
    x = -psi + 2.0 * psi * np.arange(n) / (n - 1)
    pr = np.exp(x)
    pr *= p.preis / float(pi @ pr)
    return pr, P, pi


class Kern:
    """Auswertung und Optimierung auf gegebenen Tabellen (hv, sal, h: je Alter 0..amax+1; P, pi: Preiskette). Die Tabellen kann auch ein
    Handbeispiel liefern (fut_hand.py, Tests): der Kern kennt weder Chapman-Richards noch Rouwenhorst."""

    def __init__(self, hv, sal, h, P, pi, delta, c_pflanz, amax, amin=0):
        self.hv, self.sal, self.h = np.asarray(hv, float), np.asarray(sal, float), np.asarray(h, float)
        self.P, self.pi = np.asarray(P, float), np.asarray(pi, float)
        self.delta, self.c_pflanz, self.amax, self.amin = float(delta), float(c_pflanz), int(amax), int(amin)
        self.n = len(self.pi)

    # ---- exakte Politikauswertung -------------------------------------------------------------------------------------------------
    def auswerten(self, mask, volle_tiefe=False):
        """mask[a, i] (a = 0..amax): True = ernten. mask[amax] wird erzwungen. Rückgabe (R, S)."""
        n, d, P, A = self.n, self.delta, self.P, self.amax
        mask = np.array(mask, dtype=bool)
        mask[A, :] = True
        voll = np.nonzero(mask.all(axis=1))[0]
        Aeff = A if volle_tiefe else int(voll[voll >= 1].min())   # ab dort ist nichts anderes mehr erreichbar
        if n == 1:
            return self._skalar(mask[:, 0], Aeff)
        I = np.eye(n)
        u = np.zeros((A + 1, n))
        M = np.zeros((A + 1, n, n))
        for a in range(Aeff, 0, -1):
            if a == Aeff:
                u[a], M[a] = self.hv[a], I
                continue
            ha = self.h[a]
            Cu = d * (P @ ((1 - ha) * u[a + 1] + ha * self.sal[a + 1]))
            CM = d * (P @ ((1 - ha) * M[a + 1] + ha * I))
            m = mask[a]
            u[a] = np.where(m, self.hv[a], Cu)
            M[a] = np.where(m[:, None], I, CM)
        h0 = self.h[0]
        lhs = I - d * (P @ ((1 - h0) * M[1] + h0 * I))
        rhs = -self.c_pflanz + d * (P @ ((1 - h0) * u[1] + h0 * self.sal[1]))
        R = np.linalg.solve(lhs, rhs)
        S = np.zeros((A + 1, n))
        for a in range(1, Aeff + 1):
            S[a] = u[a] + M[a] @ R
        return R, S

    def _skalar(self, m, Aeff):
        """Schneller Pfad für einen Preiszustand (konstanter Preis): reine Gleitkommaschleife."""
        d = self.delta
        hv, sal, h = self.hv[:, 0], self.sal[:, 0], self.h
        u = [0.0] * (self.amax + 2)
        M = [0.0] * (self.amax + 2)
        for a in range(Aeff, 0, -1):
            if a == Aeff or m[a]:
                u[a], M[a] = float(hv[a]), 1.0
            else:
                ha = float(h[a])
                u[a] = d * ((1 - ha) * u[a + 1] + ha * float(sal[a + 1]))
                M[a] = d * ((1 - ha) * M[a + 1] + ha)
        h0 = float(h[0])
        R = (-self.c_pflanz + d * ((1 - h0) * u[1] + h0 * float(sal[1]))) / (1.0 - d * ((1 - h0) * M[1] + h0))
        S = np.zeros((self.amax + 1, 1))
        for a in range(1, Aeff + 1):
            S[a, 0] = u[a] + M[a] * R
        return np.array([R]), S

    def bev(self, R):
        """Bodenertragswert: Mittel über die stationäre Verteilung der Preise zum Pflanzzeitpunkt."""
        return float(self.pi @ R)

    # ---- Politiken ----------------------------------------------------------------------------------------------------------------
    def maske_fix(self, T):
        m = np.zeros((self.amax + 1, self.n), dtype=bool)
        m[int(T):, :] = True
        return m

    def maske_regel(self, tmin, k, tmax):
        """Preisregel: ab Alter tmin ernten, wenn der Preisindex mindestens k ist; ab Alter tmax immer."""
        m = np.zeros((self.amax + 1, self.n), dtype=bool)
        m[int(tmin):, int(k):] = True
        m[int(tmax):, :] = True
        return m

    def bev_maske(self, mask):
        return self.bev(self.auswerten(mask)[0])

    def optimum(self, start_T, maxit=100):
        """Howard-Politikiteration ab der festen Umtriebszeit start_T. Rückgabe (R, S, mask, Iterationen).
        Gleichstand (|Hieb - Warten| <= 1e-7): Warten. Hieb erst ab amin."""
        mask = self.maske_fix(start_T)
        A = self.amax
        for it in range(maxit):
            R, S = self.auswerten(mask, volle_tiefe=True)
            new = mask.copy()
            H = self.hv + R[None, :]
            for a in range(1, A):
                ha = self.h[a]
                w = (1 - ha) * S[a + 1] + ha * (self.sal[a + 1] + R)
                C_ = self.delta * (self.P @ w)
                new[a] = (H[a] > C_ + 1e-7) & (a >= self.amin)
            new[A] = True
            if np.array_equal(new, mask):
                return R, S, mask, it + 1
            mask = new
        raise RuntimeError("Howard-Iteration nicht konvergiert")

    def abstand_hieb_warten(self, R, S):
        """Für jedes Alter 1..amax-1 und jeden Preiszustand: Hiebwert minus Wartewert (Wert beim Hieb jetzt gegen einen Schritt Warten und dann optimal)."""
        H = self.hv + R[None, :]
        out = np.zeros((self.amax + 1, self.n))
        for a in range(1, self.amax):
            ha = self.h[a]
            w = (1 - ha) * S[a + 1] + ha * (self.sal[a + 1] + R)
            out[a] = H[a] - self.delta * (self.P @ w)
        return out


class Modell(Kern):
    """Kern mit den Tabellen des Waldmodells aus den Parametern Prm."""

    def __init__(self, p: Prm):
        self.p = p
        preise, P, pi = preisgitter(p)
        a = np.arange(p.amax + 2)
        self.V, self.Q, self.hz = volumen(p, a), qualitaet(p, a), hazard(p, a)
        self.preise = preise
        hv = preise[None, :] * (self.Q * self.V)[:, None] - p.c_hieb_fix
        sal = p.salvage * preise[None, :] * (self.Q * self.V)[:, None] - p.c_hieb_fix
        super().__init__(hv, sal, self.hz, P, pi, 1.0 / (1.0 + p.r), p.c_pflanz, p.amax, p.amin)
        self._skalar_modell = None

    def skalar(self):
        """Dasselbe Modell mit konstantem Preis: der BEV fester Regeln hängt weder von sigma noch von phi ab (Preis linear, Mittelwert fest),
        also reicht für sie der schnelle Pfad (tests/test_oracle_*.py prüft die Invarianz)."""
        if self.p.sigma <= 0:
            return self
        if self._skalar_modell is None:
            self._skalar_modell = Modell(mod(self.p, sigma=0.0))
        return self._skalar_modell

    def trange(self):
        return range(max(C.TF_MIN, self.p.amin), self.p.amax + 1)

    def bestes_fix(self, trange=None):
        """(T_F, BEV) der besten festen Umtriebszeit; bei Gleichstand (1e-9) die kürzere."""
        km = self.skalar()
        best = None
        for T in (self.trange() if trange is None else trange):
            b = km.bev_maske(km.maske_fix(T))
            if best is None or b > best[1] + 1e-9:
                best = (T, b)
        return best

    def t_msy(self):
        """Umtriebszeit der größten durchschnittlichen Massenzuwachsleistung (Kulmination von V(a)/a), im erlaubten Bereich ab amin (frühestens Alter 1)."""
        a = np.arange(max(1, self.p.amin), self.p.amax + 1)
        return int(a[np.argmax(self.V[a] / a)])

    def bev_msy(self):
        km = self.skalar()
        return km.bev_maske(km.maske_fix(self.t_msy()))

