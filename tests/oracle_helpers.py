"""Unabhängige Rechenwege für die Orakeltests (kein Code aus fut_model.py außer den Tabellen, die geprüft werden sollen):

- faust_t: geschlossene Faustmann-Formel (konstanter Preis, keine Kalamität),
- zyklus_geschlossen: Zyklus-Erwartung mit Kalamität und Notverkauf als Summenformel,
- gesamtsystem: Politikauswertung als EIN lineares Gleichungssystem über alle Zustände (Alter x Preis) und R, gelöst mit numpy,
- wertiteration: Optimum per einfacher Wertiteration (ohne Politikiteration), optional mit Pflanzoption,
- monte_carlo: Simulation vieler Pfade auf der Preiskette (nur Gleichverteilte aus dem Generator, Hiebe, Kalamität, Notverkauf)."""

import numpy as np


def faust_t(p, T):
    """Dieselbe Formel mit den Waldparametern aus Prm p (V per Chapman-Richards, Q per logistischer Prämie), unabhängig vom Modell-Code geschrieben."""
    V = p.vmax * (1.0 - np.exp(-p.k * p.g * T)) ** p.pexp
    q = 1.0
    if p.sort_praemie > 0:
        qmin = 1.0 - p.sort_praemie
        q = qmin + (1.0 - qmin) / (1.0 + np.exp(-(T - 50.0) / 10.0))
    f = (1.0 + p.r) ** T
    return (p.preis * q * V - p.c_hieb_fix - p.c_pflanz * f) / (f - 1.0)


def zyklus_geschlossen(p, T):
    """BEV mit Kalamität bei konstantem Preis: R = (-c + sum_j P_j d^j N_j + S_T d^T E_T) / (1 - sum_j P_j d^j - S_T d^T) mit P_j = Wahrscheinlichkeit, dass die Kalamität im
    j-ten Jahr eintritt (Rate nach dem Alter j-1), S_T = Überlebenswahrscheinlichkeit bis T, N_j = Notverkaufserlös nach j Jahren, E_T = Hiebserlös."""
    d = 1.0 / (1.0 + p.r)
    a = np.arange(T + 2, dtype=float)
    V = p.vmax * (1.0 - np.exp(-p.k * p.g * a)) ** p.pexp
    if p.sort_praemie > 0:
        qmin = 1.0 - p.sort_praemie
        Q = qmin + (1.0 - qmin) / (1.0 + np.exp(-(a - 50.0) / 10.0))
    else:
        Q = np.ones_like(a)
    h = p.risiko * (p.h0 + p.h1 * (a / 100.0) ** 3)
    surv, num, den = 1.0, -p.c_pflanz, 1.0
    for j in range(1, T + 1):
        pj = surv * h[j - 1]
        num += pj * d**j * (p.salvage * p.preis * Q[j] * V[j] - p.c_hieb_fix)
        den -= pj * d**j
        surv *= 1.0 - h[j - 1]
    num += surv * d**T * (p.preis * Q[T] * V[T] - p.c_hieb_fix)
    den -= surv * d**T
    return num / den


def gesamtsystem(m, mask):
    """BEV einer Politik als ein Gleichungssystem: Unbekannte S[a, i] (a = 1..amax) und R[i]. Gleichungen:
    S[a,i] = hv[a,i] + R[i]            (Hieb, auch im Alter amax),
    S[a,i] = d sum_j P[i,j] ((1-h[a]) S[a+1,j] + h[a] (sal[a+1,j] + R[j]))   (Warten),
    R[i]   = -c + d sum_j P[i,j] ((1-h[0]) S[1,j] + h[0] (sal[1,j] + R[j]))."""
    A_, n, d = m.amax, m.n, m.delta
    mask = np.array(mask, dtype=bool)
    mask[A_, :] = True
    N = A_ * n + n
    M = np.zeros((N, N))
    b = np.zeros(N)
    s = lambda a, i: (a - 1) * n + i
    r = lambda i: A_ * n + i
    for a in range(1, A_ + 1):
        for i in range(n):
            row = s(a, i)
            M[row, row] += 1.0
            if mask[a, i]:
                b[row] = m.hv[a, i]
                M[row, r(i)] -= 1.0
            else:
                for j in range(n):
                    M[row, s(a + 1, j)] -= d * m.P[i, j] * (1 - m.h[a])
                    M[row, r(j)] -= d * m.P[i, j] * m.h[a]
                    b[row] += d * m.P[i, j] * m.h[a] * m.sal[a + 1, j]
    for i in range(n):
        row = r(i)
        M[row, row] += 1.0
        b[row] = -m.c_pflanz
        for j in range(n):
            M[row, s(1, j)] -= d * m.P[i, j] * (1 - m.h[0])
            M[row, r(j)] -= d * m.P[i, j] * m.h[0]
            b[row] += d * m.P[i, j] * m.h[0] * m.sal[1, j]
    x = np.linalg.solve(M, b)
    R = x[A_ * n:]
    return float(m.pi @ R), R


def wertiteration(m, pflanzoption=False, tol=1e-9, maxit=20000):
    """Optimum per Wertiteration über R (Wert des unbestockten Bodens); mit pflanzoption darf der Boden unbepflanzt bleiben (Warten einen Schritt)."""
    A_, d, P = m.amax, m.delta, m.P
    R = np.zeros(m.n)
    for it in range(maxit):
        S = np.zeros((A_ + 1, m.n))
        S[A_] = m.hv[A_] + R
        for a in range(A_ - 1, 0, -1):
            w = (1 - m.h[a]) * S[a + 1] + m.h[a] * (m.sal[a + 1] + R)
            S[a] = np.maximum(m.hv[a] + R, d * (P @ w))
            if a < m.amin:
                S[a] = d * (P @ w)
        plant = -m.c_pflanz + d * (P @ ((1 - m.h[0]) * S[1] + m.h[0] * (m.sal[1] + R)))
        Rn = np.maximum(plant, d * (P @ R)) if pflanzoption else plant
        if np.max(np.abs(Rn - R)) < tol:
            return Rn, it + 1
        R = Rn
    raise RuntimeError("Wertiteration nicht konvergiert")


def monte_carlo(m, mask, n_pfade, jahre, seed, altersstatistik=False):
    """Mittel und Standardfehler des abgezinsten Werts über n_pfade Pfade (Start: unbestockter Boden, Preiszustand aus der stationären Verteilung,
    Pflanzung im Jahr 0). Je Jahr: neuer Preiszustand, Kalamität (Rate nach Alter zu Jahresbeginn) oder Hieb. Optional zusätzlich die Alter der Hiebe der Jahre 101 bis jahre-1."""
    rng = np.random.default_rng(seed)
    d, P = m.delta, m.P
    cum, cum_pi = np.cumsum(P, axis=1), np.cumsum(m.pi)
    i = np.minimum(np.searchsorted(cum_pi, rng.random(n_pfade)), m.n - 1)
    age = np.zeros(n_pfade, dtype=int)
    pv = np.full(n_pfade, -m.c_pflanz)
    mask = np.array(mask, dtype=bool)
    mask[m.amax, :] = True
    alter_hiebe = []
    for t in range(1, jahre + 1):
        u = rng.random(n_pfade)
        j = np.minimum((cum[i] < rng.random(n_pfade)[:, None]).sum(axis=1), m.n - 1)
        kal = u < m.h[age]
        a1 = age + 1
        ern = (~kal) & mask[a1, j]
        pay = np.where(kal, m.sal[a1, j], np.where(ern, m.hv[a1, j], 0.0))
        ersatz = kal | ern
        pv += d**t * (pay - np.where(ersatz, m.c_pflanz, 0.0))
        if altersstatistik and t > 100:
            alter_hiebe.extend(a1[ern].tolist())
        age = np.where(ersatz, 0, a1)
        i = j
    out = (float(pv.mean()), float(pv.std(ddof=1) / np.sqrt(n_pfade)))
    return out + (np.array(alter_hiebe),) if altersstatistik else out
