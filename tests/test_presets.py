"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Raster der Preisregeln deckt jedes Preset ab."""

import pytest

import fut_constants as C
import fut_evaluation as ev
import fut_presets as P
import fut_rules as RU

STEPS = {"zins": (C.ZINS_MIN, C.ZINS_STEP), "pflanz": (C.PFLANZ_MIN, C.PFLANZ_STEP), "preis": (C.PREIS_MIN, C.PREIS_STEP), "sort": (C.SORT_MIN, C.SORT_STEP),
         "risiko": (C.RISIKO_MIN, C.RISIKO_STEP), "sigma": (C.SIGMA_MIN, C.SIGMA_STEP), "phi": (C.PHI_MIN, C.PHI_STEP)}


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_on_the_step_grid():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            assert spec.lo <= preset[key] <= spec.hi, (name, key)
            if key in STEPS:
                lo, step = STEPS[key]
                k = (preset[key] - lo) / step
                assert k == pytest.approx(round(k), abs=1e-9), (name, key)   # der Wert ist mit dem Regler einstellbar


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Grundfall (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_jedes_preset_unterscheidet_sich_vom_grundfall_in_genau_einem_regler():
    grund = C.PRESETS["Grundfall (Voreinstellung)"]
    for name, preset in C.PRESETS.items():
        diff = [k for k in grund if preset[k] != grund[k]]
        assert len(diff) == (0 if name.startswith("Grundfall") else 1), (name, diff)


def test_die_preisregel_ist_fuer_jedes_preset_vorgerechnet():
    tab = RU.laden()
    for name, preset in C.PRESETS.items():
        p = ev.to_prm(ev.Settings(**preset))
        assert RU.vorgerechnet(p, tab) is not None, name


def test_vorgerechnetes_raster_hat_die_erwarteten_zellen():
    tab = RU.laden()
    assert len(tab) == 18 and all(r.quelle == "Raster" for r in tab.values())
    assert all(r.tmin <= r.tmax <= C.ALTER_MAX and r.k >= 0 for r in tab.values())
