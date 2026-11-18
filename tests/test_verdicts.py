"""Verdikte (reine Funktionen in fut_evaluation.py) an handgesetzten Werten: jede Schwelle und jede Zahl im Text; je Preset die erwartete Verdikt-Zeile."""

import pytest

import fut_constants as C
import fut_det as D
import fut_evaluation as ev
import fut_stoch as ST
from fut_model import Modell, mod


def test_formatierung():
    assert ev.eur(4804.74) == "4.805" and ev.eur(12345.6) == "12.346" and ev.eur(-874.39) == "−874" and ev.eur(0.2) == "0"
    assert ev.zahl(3.1572, 2) == "3,16" and ev.zahl(-0.5, 1) == "−0,5"
    assert ev.pct(0.0316, 2) == "3,16 %" and ev.pct(None) == "–"


def test_rangfolge_und_gleichheit():
    assert ev.rangfolge({"A": 3.0, "B": 5.0, "C": 4.0}) == "B > C > A"
    assert ev.rangfolge({"A": 3.0, "B": 3.004}) == "B = A"       # Unterschied unter 0,005 EUR/ha: gleich
    assert ev.rangfolge({"A": 3.0, "B": 3.006}) == "B > A"
    assert ev.abstand(90.0, 100.0) == -10.0                      # mein Wert minus Referenz


# ---- Kernfrage 1 ----------------------------------------------------------------------------------------------------------------------
def test_verdict_msy_gleiche_zeit():
    k, t = ev.verdict_msy(68, 68, 53261.3, 53261.3, 0.0316, 0.0285)
    assert k == "success" and "verliert hier nichts" in t and "68 Jahre" in t


def test_verdict_msy_schwelle_bodenwert_null():
    # BEV Faustmann genau 0: kein Gewinn mehr, also die Zeile „lohnt sich nicht“ (Grenze gehört zu „lohnt nicht“)
    k, t = ev.verdict_msy(45, 68, 0.0, -800.0, None, None)
    assert k == "warning" and "lohnt sich bei diesem Zins auch mit der besten festen Umtriebszeit nicht" in t and "800 EUR/ha darunter" in t
    k, t = ev.verdict_msy(45, 68, -711.8, -1533.6, 0.0316, 0.0285)
    assert k == "warning" and "−712 EUR/ha" in t and "822 EUR/ha darunter" in t and "Faustmann 3,16 %, MSY-Regel 2,85 %" in t


def test_verdict_msy_zwischen_den_break_even_zinsen():
    k, t = ev.verdict_msy(47, 68, 397.0, -422.9, 0.0316, 0.0285)
    assert k == "warning" and "Zwischen den beiden Break-even-Zinsen" in t and "397 EUR/ha" in t and "−423 EUR/ha" in t
    k, t = ev.verdict_msy(47, 68, 397.0, 0.0, 0.0316, 0.0285)     # MSY-Bodenwert genau 0: noch kein Verlust
    assert k == "info" and "verliert 397 EUR/ha" in t


def test_verdict_msy_normalfall_nennt_prozent_und_basis():
    k, t = ev.verdict_msy(53, 68, 4804.7, 4128.6, 0.0316, 0.0285)
    assert k == "info" and "verliert 676 EUR/ha" in t and "14,1 % des Faustmann-Bodenwerts von 4.805 EUR/ha" in t
    assert "Break-even-Zins" not in ev.verdict_msy(53, 68, 4804.7, 4128.6, None, 0.0285)[1]


# ---- Kernfrage 2 ----------------------------------------------------------------------------------------------------------------------
def test_verdict_hook_schwelle_gewinn_gleich_verlust():
    k, t = ev.verdict_hook(676.0, 676.0, 0.13, 0.9, 2.0)         # Gleichheit gehört zur Flexibilität
    assert k == "warning" and "Warten auf den Preis ist mehr wert" in t
    k, t = ev.verdict_hook(675.9, 676.0, 0.13, 0.9, 2.0)
    assert k == "success" and "richtige Umtriebszeit ist mehr wert" in t


def test_verdict_hook_zahlen_faktor_und_bedingung():
    k, t = ev.verdict_hook(2450.0, 676.0, 0.3, 0.9, 2.0)
    assert "2.450 EUR/ha" in t and "676 EUR/ha" in t and "Faktor 3,6" in t and "σ = 0,30, φ = 0,90 und Zins 2,00 %" in t and "keine Empfehlung für einen Betrieb" in t
    k, t = ev.verdict_hook(511.0, 676.0, 0.3, 0.97, 2.0)
    assert k == "success" and "Faktor 0,8" in t and "φ = 0,97" in t


def test_verdict_hook_randfaelle_sigma_null_und_verlust_null():
    k, t = ev.verdict_hook(0.0, 676.0, 0.0, 0.9, 2.0)
    assert k == "info" and "Ohne Preisstreuung" in t and "676 EUR/ha" in t and "Faktor" not in t
    k, t = ev.verdict_hook(2450.0, 0.0, 0.3, 0.9, 0.5)           # kein Verlust: keine Division durch null
    assert k == "info" and "gleich gut" in t and "2.450 EUR/ha" in t


# ---- Kernfrage 3 ----------------------------------------------------------------------------------------------------------------------
def test_verdict_sigma_stern():
    k, t = ev.verdict_sigma_stern(0.129, 676.0, 0.3, 0.9, 2.0)
    assert k == "info" and "σ* = 0,129" in t and "676 EUR/ha" in t and "liegt darüber" in t
    assert "liegt darunter" in ev.verdict_sigma_stern(0.129, 676.0, 0.1, 0.9, 2.0)[1]
    assert "liegt darunter" in ev.verdict_sigma_stern(0.129, 676.0, 0.129, 0.9, 2.0)[1]    # genau auf der Schwelle: noch nicht darüber
    k, t = ev.verdict_sigma_stern(None, 820.0, 0.3, 0.97, 3.0)
    assert "Bis σ = 0,50" in t and "820 EUR/ha" in t and "bei dieser Einstellung gewinnt die richtige Umtriebszeit" in t and "φ 0,97 und Zins 3 %" in t
    k, t = ev.verdict_msy(68, 68, 53261.3, 53261.3, None, None)
    assert k == "success" and "in der Grundeinstellung" in t and "0,45 % bis 0,52 %" in t and "1,05 % bis 1,15 %" in t
    k, t = ev.verdict_sigma_stern(0.0, 0.0, 0.3, 0.9, 0.5)
    assert "Keine Schwelle" in t


def test_verdict_bonitaet_nennt_die_betraege():
    k, t = ev.verdict_bonitaet(-169.4, -18.2, -684.0)
    assert "169 EUR/ha" in t and "18 EUR/ha" in t and "684 EUR/ha" in t and "teilweise eingebaut" in t
    kein_minus = ev.verdict_bonitaet(-1e-12, 0.0, 0.0)[1]
    assert "0 EUR/ha" in ev.verdict_bonitaet(0.0, 0.0, 0.0)[1] and "-0" not in kein_minus and "−0" not in kein_minus


# ---- je Preset die erwartete Verdikt-Zeile -----------------------------------------------------------------------------------------------
def _preset_prm(name):
    pr = C.PRESETS[name]
    return ev.to_prm(ev.Settings(pr["zins"], pr["pflanz"], pr["preis"], pr["sort"], pr["risiko"], pr["sigma"], pr["phi"], pr["seed"]))


def _verdikte(name):
    p = _preset_prm(name)
    det = D.analyse(mod(p, sigma=0.0))
    m = Modell(p)
    tf, bf = m.bestes_fix()
    opt = ST.optimum(m, tf)[0]
    star, verlust = ST.sigma_stern(mod(p, sigma=0.0))
    return (det, ev.verdict_msy(det.tf, det.tm, det.bev_f, det.bev_m, det.be_f, det.be_m), ev.verdict_hook(opt.bev - bf, bf - m.bev_msy(), p.sigma, p.phi, p.r * 100),
            ev.verdict_sigma_stern(star, verlust, p.sigma, p.phi, p.r * 100), opt.bev, bf, star)


def test_preset_kreuzungszins_meldet_keinen_verlust_der_msy_regel():
    det, v1, v2, v3, *_ = _verdikte("Kreuzungszins")
    assert det.tf == det.tm == 68 and v1[0] == "success" and "verliert hier nichts" in v1[1]
    assert "gleich gut" in v2[1] and "Keine Schwelle" in v3[1]


def test_preset_zwischen_zwei_break_even_zinsen():
    det, v1, *_ = _verdikte("Zwischen zwei Break-even-Zinsen")
    assert det.bev_f > 0 > det.bev_m and "Zwischen den beiden Break-even-Zinsen" in v1[1]


def test_preset_sigma_stern_liegt_knapp_darueber():
    det, v1, v2, v3, opt, bf, star = _verdikte("σ*-Schwelle")
    assert star == pytest.approx(0.1292, abs=0.0005) and 0.13 > star
    assert v2[0] == "warning" and "Faktor 1,0" in v2[1] and "liegt darüber" in v3[1]


def test_preset_hoher_zins_nur_mit_flexibilitaet_im_plus():
    det, v1, v2, v3, opt, bf, star = _verdikte("Hoher Zins")
    assert bf < 0 and opt > 0 and "nicht" in v1[1] and "lohnt" in v1[1]


def test_preset_kalamitaetsjahr_aendert_die_rangfolge_nicht():
    det, v1, v2, *_ , = _verdikte("Kalamitätsjahr")
    assert v2[0] == "warning" and det.tf == 50 and det.tm == 68


def test_preset_persistent_und_gedaechtnislos():
    _, _, v2, _, _, _, _ = _verdikte("Preise sehr persistent")
    assert v2[0] == "success" and "richtige Umtriebszeit ist mehr wert" in v2[1]
    _, _, w2, _, _, _, star0 = _verdikte("Preise ohne Gedächtnis")
    assert w2[0] == "warning" and star0 < 0.05
