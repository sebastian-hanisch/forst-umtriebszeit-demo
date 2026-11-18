"""Gemeinsame Fixtures: das Handbeispiel (exakte Brüche) und die Tabellen der Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fut_constants as C  # noqa: E402
import fut_hand as H  # noqa: E402
from fut_model import Kern  # noqa: E402


def hand_kern(haz=None, salvage=0.5):
    """Der Kern auf den Tabellen des Handbeispiels (fut_hand.py): fünf Alter, zwei Preiszustände 2 und 8, Zins 1/5, Pflanzung 20, Alter 5 erzwungen.
    haz: Kalamitätsrate je Alter 0..4 (Liste) oder None (keine)."""
    V = [float(H.V[min(a, 5)]) for a in range(0, 7)]
    V[6] = 44.0
    pr = np.array([float(x) for x in H.PREISE])
    hv = pr[None, :] * np.array(V)[:, None]
    h = np.zeros(7) if haz is None else np.array(list(haz) + [0.0] * (7 - len(haz)))
    return Kern(hv, salvage * hv, h, [[0.75, 0.25], [0.25, 0.75]], [0.5, 0.5], 1 / 1.2, 20.0, H.AMAX)


@pytest.fixture
def hand():
    return hand_kern()


@pytest.fixture(scope="module")
def fast_limits():
    """Für App-Tests: weniger Startpunkte und Runden der Koordinatensuche der Preisregel, falls ein Test sie live abstimmen lässt. Modul-Umfang: test_claims.py rechnet mit den echten Werten."""
    mp = pytest.MonkeyPatch()
    mp.setattr(C, "RULE_STARTS", 1)
    mp.setattr(C, "RULE_ROUNDS", 2)
    yield
    mp.undo()
