"""Konstanten: Modell-Annahmen, Regler-Grenzen, Farben und Presets.

ALLE Zahlen des Waldmodells sind Annahmen (keine Ertragstafel, keine Preisreihe gelesen); die Formen der Kurven (Chapman-Richards-Volumen,
Kalamitäts-Hazard) sind gewählt, nicht angepasst."""

# --- Waldmodell (Annahmen) --------------------------------------------------------------------------------------------------------------
VMAX = 1100.0              # m³ je Hektar, Sättigungsvolumen
WACHSTUM_K = 0.028         # Wachstumsrate des Chapman-Richards-Volumens je Jahr
WACHSTUM_P = 3.0           # Formexponent (S-Form)
HAZARD_H0 = 0.002          # Grundrate der Kalamität je Jahr (vor Skalierung mit dem Risiko)
HAZARD_H1 = 0.02           # altersabhängiger Anteil, wächst mit (Alter/100)³
NOTVERKAUF = 0.5           # Notverkauf: Anteil des Holzwerts bei Kalamität
SORT_MITTE, SORT_BREITE = 50.0, 10.0   # logistischer Anstieg der Sortimentsprämie (Mitte Alter 50, Breite 10)
N_PREIS = 21               # Zustände der diskretisierten Preiskette (Rouwenhorst)
ALTER_MAX = 150            # spätestens in diesem Alter wird geerntet (Zwangshieb, Rand des Modells)
TF_MIN = 5                 # kleinste betrachtete Umtriebszeit

# --- Regler: (Minimum, Maximum, Standard, Schritt) --------------------------------------------------------------------------------------
ZINS_MIN, ZINS_MAX, DEFAULT_ZINS, ZINS_STEP = 0.25, 5.0, 2.0, 0.05            # Prozent je Jahr
PFLANZ_MIN, PFLANZ_MAX, DEFAULT_PFLANZ, PFLANZ_STEP = 2000.0, 8000.0, 4000.0, 500.0    # EUR je Hektar
PREIS_MIN, PREIS_MAX, DEFAULT_PREIS, PREIS_STEP = 20.0, 60.0, 40.0, 5.0       # EUR je m³ (Nettoerlös: Holzpreis minus Ernte)
SORT_MIN, SORT_MAX, DEFAULT_SORT, SORT_STEP = 0.0, 0.6, 0.0, 0.1              # Sortimentsprämie
RISIKO_MIN, RISIKO_MAX, DEFAULT_RISIKO, RISIKO_STEP = 0.0, 2.0, 0.0, 0.5      # Skalierung der Kalamität
SIGMA_MIN, SIGMA_MAX, DEFAULT_SIGMA, SIGMA_STEP = 0.0, 0.5, 0.3, 0.01         # stationäre Std. des Log-Preises
PHI_MIN, PHI_MAX, DEFAULT_PHI, PHI_STEP = 0.0, 0.97, 0.9, 0.01                # Autokorrelation des Log-Preises
SEED_MAX, DEFAULT_SEED = 99999999, 20261118

# --- Suche und Raster -------------------------------------------------------------------------------------------------------------------
RULE_STARTS = 4            # Zahl der heuristischen Startpunkte der Koordinatensuche (Preisregel); dazu kommt immer die feste Umtriebszeit selbst als Start
RULE_ROUNDS = 30           # höchstens so viele Runden der Koordinatensuche je Start
SIGMA_GRID_STEP = 0.02     # Gitter für die Kurve „Gewinn der Preisflexibilität über σ“
BAHN_JAHRE = 150           # Länge der gezeigten Zufallsbahn
BAHN_SEEDS = tuple(range(300, 305))  # feste Zufallsbahnen für Zählungen über mehrere Bahnen (nie eine einzelne als Befund)
MC_SEED, MC_PFADE, MC_JAHRE = 20261118, 20000, 400   # Gegenprobe Monte-Carlo gegen die exakte Auswertung (nur Tests und README)

BONITAET_G = (0.8, 1.0, 1.2)   # Zeitskalierung des Wachstums je Bonität (je ein Drittel der Fläche)

COLOR_FEST = "#7f7f7f"
COLOR_MSY = "#d6604d"
COLOR_REGEL = "#4c78a8"
COLOR_OPT = "#2F6B65"
COLOR_PREIS = "#8c6d1f"

# --- Presets: jeder Eintrag setzt ALLE Regler -------------------------------------------------------------------------------------------
_BASIS = {"zins": DEFAULT_ZINS, "pflanz": DEFAULT_PFLANZ, "preis": DEFAULT_PREIS, "sort": DEFAULT_SORT, "risiko": DEFAULT_RISIKO,
          "sigma": DEFAULT_SIGMA, "phi": DEFAULT_PHI, "seed": DEFAULT_SEED}
PRESETS = {
    "Grundfall (Voreinstellung)": dict(_BASIS),
    "Kreuzungszins": {**_BASIS, "zins": 0.5},
    "Zwischen zwei Break-even-Zinsen": {**_BASIS, "zins": 3.0},
    "σ*-Schwelle": {**_BASIS, "sigma": 0.13},
    "Hoher Zins": {**_BASIS, "zins": 3.5},
    "Kalamitätsjahr": {**_BASIS, "risiko": 2.0},
    "Preise sehr persistent": {**_BASIS, "phi": 0.97},
    "Preise ohne Gedächtnis": {**_BASIS, "phi": 0.0},
}
PRESET_HELP = {
    "Grundfall (Voreinstellung)": "Zins 2 %, Pflanzung 4000 EUR/ha, Nettoerlös 40 EUR/m³, kein Kalamitätsrisiko; Preis log-AR(1) mit σ 0,3 und φ 0,9. Alles Annahmen des Lehrmodells.",
    "Kreuzungszins": "Zins 0,5 %: die feste Faustmann-Umtriebszeit ist so lang wie die der MSY-Regel (68 Jahre), die MSY-Regel verliert hier nichts.",
    "Zwischen zwei Break-even-Zinsen": "Zins 3 %: mit der Faustmann-Umtriebszeit ist der Boden noch ein Gewinn, mit der MSY-Regel schon ein Verlust.",
    "σ*-Schwelle": "σ 0,13: knapp über der Schwelle, ab der der Gewinn der Preisflexibilität den Verlust der MSY-Regel übersteigt (Zins 2 %, φ 0,9).",
    "Hoher Zins": "Zins 3,5 %: mit fester Umtriebszeit ist der Boden ein Verlust; mit der Preisflexibilität (σ 0,3, φ 0,9) bleibt er gerade noch im Plus.",
    "Kalamitätsjahr": "Kalamitätsrisiko 2 (doppelte Rate, Notverkauf mit 50 % Abschlag): verschiebt die Umtriebszeit um wenige Jahre, ändert die Rangfolge der Regeln hier (σ 0,3, φ 0,9, Zins 2 %) nicht.",
    "Preise sehr persistent": "φ 0,97: Preisausschläge halten sehr lange an, Warten bringt weniger; bei σ 0,3 liegt die Flexibilität unter dem Verlust der MSY-Regel (erst ab σ 0,348 darüber).",
    "Preise ohne Gedächtnis": "φ 0: jedes Jahr ein neuer Preis; Warten auf einen guten Preis lohnt am meisten.",
}
