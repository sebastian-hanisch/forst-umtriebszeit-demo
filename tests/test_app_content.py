"""Inhalt der App: Tabs, Kernfragen, Expander (per Titel, nie per Index), Handbeispiel, Zahlen der Voreinstellung, Formelblöcke, PDF-Knopf, Gleichheit von App- und README-Text."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import fut_constants as C
import fut_texts as TX

pytestmark = pytest.mark.usefixtures("fast_limits")
ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")


@pytest.fixture(scope="module")
def at():
    a = AppTest.from_file(APP, default_timeout=300)
    a.run()
    assert not a.exception, [str(e) for e in a.exception]
    return a


def test_vier_tabs_in_fester_reihenfolge(at):
    assert [t.label for t in at.tabs] == ["✋ Von Hand", "🌲 Umtriebszeit", "💶 Preis als Zufall", "🌳 Bonität"]


def test_expander_stehen_unter_ihren_titeln(at):
    labels = {e.label for e in at.expander}
    assert {"💬 In einfachen Worten", "🔧 Wie wir das erreichen – vollständiger Methodenvergleich", "📐 Mathematische Formulierung", "⚠️ Annahmen, Grenzen und bekannte Probleme"} <= labels


def test_drei_kernfragen_haben_ein_verdikt(at):
    texts = [e.value for e in list(at.success) + list(at.info) + list(at.warning)]
    assert any("Die MSY-Regel (68 Jahre) verliert 676 EUR/ha" in t for t in texts)
    assert any("Das Warten auf den Preis ist mehr wert als der Streit um die Umtriebszeit" in t for t in texts)
    assert any("Schwelle σ* = 0,129" in t for t in texts)


def test_zahlen_der_voreinstellung_in_der_app(at):
    values = {m.label: m.value for m in at.metric}
    assert values["Faustmann"] == "53 Jahre" and values["MSY"] == "68 Jahre"
    assert values["Gewinn der Preisflexibilität (EUR/ha)"] == "2.450"
    assert values["Verlust der MSY-Regel (EUR/ha)"] == "676" and values["Verlust MSY (EUR/ha)"] == "676"
    caps = " ".join(c.value for c in at.caption)
    assert "Rangfolge nach Bodenwert: Optimum (DP) > Preisregel (abgestimmt) > Faustmann (fest) > MSY (fest)" in caps


def test_handbeispiel_zeigt_die_exakten_brueche(at):
    metrics = {m.label: (m.value, m.delta) for m in at.metric}
    assert metrics["Faustmann (feste Zeit 2)"][0] == "1780/11"
    assert metrics["MSY (feste Zeit 4)"][0] == "105330/671"
    assert metrics["Optimum aus allen 256 Politiken"][0] == "90963860/432357"


def test_pdf_knopf_ist_da(at):
    assert any("download" in str(type(el)).lower() for el in at.main)


def test_preisregel_knopf_erscheint_nur_ausserhalb_des_vorgerechneten_rasters():
    a = AppTest.from_file(APP, default_timeout=300)
    a.run()
    assert not [b for b in a.button if b.key == "regel_start"]
    a.slider(key="zins_slider").set_value(4.5).run()          # nicht vorgerechnet
    assert [b for b in a.button if b.key == "regel_start"]
    assert not [r for r in a.caption if "Preisregel (abgestimmt) >" in r.value]
    a.button(key="regel_start").click().run()
    assert not a.exception, [str(e) for e in a.exception]
    assert any("Preisregel (abgestimmt)" in c.value for c in a.caption)


def test_sigma_null_hat_keine_preisschwelle_und_keine_ausnahme():
    a = AppTest.from_file(APP, default_timeout=300)
    a.run()
    a.slider(key="sigma_slider").set_value(0.0).run()
    assert not a.exception, [str(e) for e in a.exception]
    assert any("Bei σ = 0 gibt es nur einen Preis" in i.value for i in a.info)


def test_annahmentext_der_app_steht_in_der_readme():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert TX.ASSUMPTIONS_MARKDOWN.strip() in readme


# ---- Formelblöcke ---------------------------------------------------------------------------------------------------------------------
def test_jede_abgesetzte_formel_steht_in_einem_eigenen_markdown_aufruf_im_quelltext():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    bloecke = re.findall(r'st\.markdown\(r"""\$\$(.*?)\$\$"""\)', src, flags=re.S)
    assert len(bloecke) == 10
    assert all("$$" not in b for b in bloecke)
    assert src.count("$$") == 2 * len(bloecke)             # keine weitere abgesetzte Formel außerhalb dieser Aufrufe
    mathe = src[src.index("📐 Mathematische Formulierung"):]
    assert "%" not in "".join(bloecke)                      # KaTeX liest % als Kommentar
    assert "(1-\\eta(a))" in mathe and "\\delta\\sum_{j}P_{ij}" in mathe


def test_modellbeschreibung_nennt_anfangswerte_reihenfolge_zufall_und_randfaelle():
    t = TX.MATH_NOTE
    for stichwort in ("Anfangswerte", "Reihenfolge", "Zufallsprozess", "Randfälle", "Gleichstand", "Zwangshieb", "stationären Verteilung", "exakt"):
        assert stichwort in t, stichwort
