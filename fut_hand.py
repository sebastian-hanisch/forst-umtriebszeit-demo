"""Handbeispiel in exakten Brüchen (fractions.Fraction): fünf Altersstufen, zwei Preiszustände, ohne Kalamität im App-Beispiel.

Ein eigener Rechenweg, unabhängig von fut_model.py: Eine Politik wird als GESAMTES lineares Gleichungssystem für alle Werte S(Alter, Preis) und
R(Preis) aufgestellt und mit Bruchrechnung gelöst (Gauß-Verfahren), statt wie im Kern rückwärts über die Alter zu rekursieren. Das Optimum ist
die Aufzählung aller Politiken (2^8 = 256), nicht die Politikiteration. Das Beispiel steht in der App im Tab „Von Hand“; tests/ vergleichen es
mit dem Kern (Gleitkomma).

Zahlen: Volumen V = 5, 20, 30, 42, 44 m³ je Hektar im Alter 1 bis 5 (erfunden, damit die MSY-Umtriebszeit 4 und die Faustmann-Umtriebszeit 2
verschieden sind), Pflanzkosten 20, Zins 1/5. Ein konstanter Preis 5 (Abschnitt 1) oder zwei Preiszustände 2 und 8 mit Übergangsmatrix
[[3/4, 1/4], [1/4, 3/4]] und stationärer Verteilung (1/2, 1/2), also Mittelwert 5 (Abschnitt 2). Im Alter 5 wird immer geerntet."""

import itertools
from fractions import Fraction as Fr

V = {0: Fr(0), 1: Fr(5), 2: Fr(20), 3: Fr(30), 4: Fr(42), 5: Fr(44)}
PREIS_KONST = Fr(5)
PREISE = (Fr(2), Fr(8))
P = ((Fr(3, 4), Fr(1, 4)), (Fr(1, 4), Fr(3, 4)))
PI = (Fr(1, 2), Fr(1, 2))
PFLANZ = Fr(20)
ZINS = Fr(1, 5)
AMAX = 5
NAMEN = ("niedrig (2)", "hoch (8)")


def bev_fest(T, preis=PREIS_KONST, r=ZINS, c=PFLANZ):
    """BEV einer festen Umtriebszeit bei konstantem Preis, geschlossene Formel (preis V(T) - c (1+r)^T) / ((1+r)^T - 1) als Bruch."""
    f = (1 + r) ** T
    return (preis * V[T] - c * f) / (f - 1)


def msy():
    """(T_MSY, V/a): Alter mit dem größten durchschnittlichen Zuwachs V(a)/a (im Beispiel eindeutig)."""
    return max(range(1, AMAX + 1), key=lambda a: V[a] / a)


def faustmann():
    """(T_F, BEV): beste feste Umtriebszeit 1..5 (im Beispiel eindeutig)."""
    T = max(range(1, AMAX + 1), key=lambda t: bev_fest(t))
    return T, bev_fest(T)


# ---- exakte Auswertung einer Politik (Gleichungssystem) -----------------------------------------------------------------------------
def loesen(A, b):
    """Gauß-Verfahren mit Brüchen: löst A x = b (A quadratisch, regulär)."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = next(i for i in range(col, n) if M[i][col] != 0)
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [x / pv for x in M[col]]
        for i in range(n):
            if i != col and M[i][col] != 0:
                fac = M[i][col]
                M[i] = [x - fac * y for x, y in zip(M[i], M[col])]
    return [M[i][n] for i in range(n)]


def auswerten(politik, haz=None, salvage=Fr(1, 2)):
    """Politik: Menge von (Alter, Preisindex) mit Hieb für Alter 1..4 (Alter 5 immer). haz: Kalamitätsrate je Alter 0..4 (None: keine).
    Unbekannte: S(a, i) für a = 1..5, i = 0, 1 (Index 2 (a - 1) + i) und R(i) (Index 10 + i). Rückgabe (R-Liste, BEV)."""
    d = 1 / (1 + ZINS)
    haz = haz or {a: Fr(0) for a in range(AMAX)}
    nS, n = 2 * AMAX, 2 * AMAX + 2

    def s(a, i):
        return 2 * (a - 1) + i

    def r(i):
        return nS + i

    A = [[Fr(0)] * n for _ in range(n)]
    b = [Fr(0)] * n
    for a in range(1, AMAX + 1):
        for i in (0, 1):
            row = s(a, i)
            A[row][row] += 1
            if a == AMAX or (a, i) in politik:       # Hieb: Erlös plus Boden danach
                b[row] = PREISE[i] * V[a]
                A[row][r(i)] -= 1
            else:                                     # Warten: ein Jahr, dann neuer Preis; Kalamität mit Rate haz[a]
                for j in (0, 1):
                    A[row][s(a + 1, j)] -= d * P[i][j] * (1 - haz[a])
                    b[row] += d * P[i][j] * haz[a] * salvage * PREISE[j] * V[a + 1]
                    A[row][r(j)] -= d * P[i][j] * haz[a]
    for i in (0, 1):                                  # Pflanzung: Kosten jetzt, ein Jahr bis zum Alter 1 (oder Notverkauf des einjährigen Bestands)
        row = r(i)
        A[row][row] += 1
        b[row] = -PFLANZ
        for j in (0, 1):
            A[row][s(1, j)] -= d * P[i][j] * (1 - haz[0])
            b[row] += d * P[i][j] * haz[0] * salvage * PREISE[j] * V[1]
            A[row][r(j)] -= d * P[i][j] * haz[0]
    x = loesen(A, b)
    R = x[nS:]
    return R, sum(PI[i] * R[i] for i in (0, 1))


def politiken():
    """Alle 256 Politiken als Mengen von (Alter, Preisindex)."""
    zust = [(a, i) for a in range(1, AMAX) for i in (0, 1)]
    for bits in itertools.product((False, True), repeat=len(zust)):
        yield frozenset(z for z, b in zip(zust, bits) if b)


def optimum_aufzaehlung(haz=None):
    """(beste Politik, BEV) über alle 256 Politiken; bei Gleichstand die mit weniger Hieben (Warten)."""
    best = None
    for pol in politiken():
        _, bev = auswerten(pol, haz)
        if best is None or bev > best[1] or (bev == best[1] and len(pol) < len(best[0])):
            best = (pol, bev)
    return best


def feste_politik(T):
    return frozenset((a, i) for a in range(T, AMAX) for i in (0, 1))


def beschreibung(politik):
    """Lesbare Fassung: je Alter 1..4 'nie', 'bei hohem Preis' oder 'immer' (der Hieb im Alter 5 ist erzwungen)."""
    out = []
    for a in range(1, AMAX):
        tief, hoch = (a, 0) in politik, (a, 1) in politik
        out.append((a, "immer" if tief and hoch else "bei hohem Preis" if hoch else "bei niedrigem Preis" if tief else "nie"))
    return out


def bruch(x: Fr) -> str:
    return f"{x.numerator}/{x.denominator}" if x.denominator != 1 else str(x.numerator)
