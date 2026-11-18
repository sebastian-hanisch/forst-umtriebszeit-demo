# Forst: Wann ernten? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-forst-umtriebszeit-demo.streamlit.app/)** -->

Fall-Demo für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (`demos.html`): **Wann soll ein Bestand geerntet werden?** Vier Regeln werden auf demselben Waldmodell exakt ausgewertet und verglichen: die Umtriebszeit nach **Faustmann** (größter Bodenertragswert), die **MSY-Regel** (größte durchschnittliche Zuwachsleistung), eine **Preisregel** mit drei je Szenario abgestimmten Parametern und das **Optimum** aus einer Dynamischen Optimierung über Alter × Preis. Die Kernfrage: Ist der Streit zwischen Faustmann und MSY klein gegen den Wert, auf einen guten Preis zu warten, und ab welcher Preisstreuung σ* kippt die Rangfolge?

**Das ist ein Lehrmodell mit offenen Annahmen, keine Forstaussage und keine Empfehlung für einen Betrieb.** Es gibt keine echten Holzpreisreihen darin: die Preisstreuung σ und die Persistenz φ sind Regler, und die Größe des Effekts hängt an ihnen (siehe „Ehrliche Grenzen“). Alle Zahlen dieser README sind Zahlen dieses Modells mit seinen Annahmen.

## Warum dieses Problem

Die klassische Frage der Forstökonomie lautet, wann ein Bestand genutzt werden soll. Der Bodenertragswert (BEV) einer Umtriebszeit gibt den Wert des Bodens bei unendlich vielen gleichen Umtrieben an; seine größte Stelle ist die Faustmann-Umtriebszeit, die im Grundfall oberhalb des Kreuzungszinses (etwa 0,5 %) kürzer ist als die „forstliche“ MSY-Zeit von 68 Jahren. Dieselbe Rechnung wird in der Demo mit einem **zufälligen Preis** erweitert: Wer einen Bestand nicht zum Termin, sondern bei gutem Preis erntet, gewinnt im Modell mehr als der Unterschied zwischen der Faustmann- und der MSY-Umtriebszeit, wenn die Preisstreuung groß genug ist (bei φ 0,97 und Zins 3 % reicht auch σ 0,5 nicht). Der Hook hängt an der **Rangfolge der Regeln und an der Kreuzungs-Streuung σ***, nicht an „die Schwelle aus der DP ist optimal“ (das Muster „Schwelle aus der DP“ steht im Portfolio schon mehrfach, zum Beispiel in `routenresilienz-demo`, `sekretaerproblem-demo` und `revenue-management-demo`); neu sind hier der Zins als Hauptregler, die Frage MSY gegen Faustmann und das Abstimmen der Gegenregeln. Eng verwandt als Rechnung ist die Erneuerung bei festem Zyklus (Altersersatz); die Demo behandelt sie als Anwendungsstück.

## Modell

Ein Hektar, unendliche Folge von Umtrieben, Jahresschritte, Wiederaufforstung sofort (Faustmann-Annahme). Volumen nach Chapman-Richards, Nettoerlös je m³, Pflanzkosten, Zins, Kalamität (Rate nach Alter, Notverkauf mit 50 % des Holzwerts) und ein Preis als log-AR(1) mit 21 Zuständen (Rouwenhorst) mit Streuung σ und Persistenz φ. Alle Formeln stehen im 📐-Expander der App, einschließlich der vollständigen Beschreibung des Modells (Anfangswerte, Reihenfolge, Zufallsprozess, Randfälle). Jede Politik wird **exakt** ausgewertet (lineares Gleichungssystem für den Wert des unbestockten Bodens; keine Simulation, keine Einschwingzeit); die Simulation dient nur der gezeigten Zufallsbahn und der Gegenprobe. Voreinstellung: Zins 2 %, Pflanzung 4000 EUR/ha, Nettoerlös 40 EUR/m³, kein Kalamitätsrisiko, σ 0,3, φ 0,9, Zufalls-Seed 20261118.

**Preisgitter, Sortiment, Regler und Preisregel im Einzelnen (Annahmen):** Der Log-Preis hat 21 gleichabständige Stufen von −ψ bis +ψ mit ψ = σ·√20; der Preis einer Stufe ist e hoch Stufenwert mal einer Konstante, die den Mittelwert der Preise unter der stationären Verteilung genau auf den Nettoerlös setzt. Die stationäre Verteilung ist binomial (20 Versuche, Erfolgswahrscheinlichkeit 1/2); die Übergangsmatrix baut das symmetrische Rouwenhorst-Verfahren aus der Zwei-Zustands-Matrix mit der Haltewahrscheinlichkeit (1 + φ)/2 rekursiv bis 21 Zustände auf. Damit ist die Standardabweichung des Log-Preises genau σ und die Autokorrelation der Stufenindizes genau φ. Der Sortimentsfaktor ist 1, wenn die Sortimentsprämie s gleich 0 ist, sonst 1 − s + s/(1 + e^(−(Alter − 50)/10)) (Mitte Alter 50, Breite 10). Regler (Minimum bis Maximum): Zins 0,25 bis 5 %, Pflanzung 2000 bis 8000 EUR/ha, Nettoerlös 20 bis 60 EUR/m³, Sortimentsprämie 0 bis 0,6, Kalamitätsrisiko 0 bis 2, σ 0 bis 0,5, φ 0 bis 0,97. Die Preisregel hat den Preisindex κ von 0 bis 20 (κ = 0: ernten bei jedem Preis; κ ist nicht die Wachstumsrate k); t_min und t_max liegen zwischen 5 und 150 Jahren, die Koordinatensuche macht höchstens 30 Runden je Startpunkt. σ* wird auf dem Gitter 0,02 gesucht und danach in 30 Bisektionsschritten verfeinert. Bei der Politikiteration (Start: feste Faustmann-Zeit) gilt Hieb erst, wenn der Hiebwert den Wartewert um mehr als 1e-7 EUR/ha übertrifft, sonst wird gewartet.

## Methodik

| Regel | Vorschrift | Parameter |
|---|---|---|
| Faustmann (fest) | feste Umtriebszeit $T$, die den BEV am größten macht, aus allen $T$ von 5 bis 150 Jahren aufgezählt | $T_F$ (abgeleitet) |
| MSY (fest) | $T$ = Alter der größten durchschnittlichen Zuwachsleistung $V(a)/a$ (68 Jahre im Grundfall) | keiner |
| Preisregel (abgestimmt) | ab Alter $t_{\min}$ ernten, wenn der Preis mindestens $q_\kappa$ ist, spätestens bei $t_{\max}$; Koordinatensuche mit fünf Startpunkten, einer davon die feste Umtriebszeit selbst | $t_{\min}$, $\kappa$, $t_{\max}$ |
| Optimum (DP) | Howard-Politikiteration über Alter × Preis, bei Gleichstand wird gewartet; geprüft gegen eine einfache Wertiteration | keiner |

Alle vier Regeln bewerten denselben Wald mit derselben exakten Auswertung. Die abgestimmte Preisregel ist nie schlechter als die beste feste Zeit (die feste Zeit ist einer ihrer Startpunkte) und nie besser als das Optimum. Die Preisregel ist kein Strohmann, hat aber nur eine starre Schwelle; die DP-Schwelle sinkt mit dem Alter. Für die Szenarien dieser README liegen die Parameter der Preisregel in `data/regeln.csv` (18 Zeilen, vorgerechnet mit `python fut_grid.py`); jede andere Einstellung stimmt die App auf Knopfdruck live ab (gemessen auf einem Entwicklungsrechner je nach Einstellung etwa 4 bis 11 Sekunden, auf langsameren Servern länger). Exakte Werte stehen in den Tabellen, die Simulation zeigt nur eine Bahn.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Lehrmodell mit den Annahmen aus „Ehrliche Grenzen“, EUR je Hektar, Bodenertragswert (BEV). Die Spalte „Test“ nennt den Test in `tests/test_claims.py` (Band, kein exakter Vergleich), der die Zahl nachrechnet; was dort nicht steht, ist eine Messung dieser README und nicht im Test.

### Umtriebszeit bei konstantem Preis (Nettoerlös 40 EUR/m³, Pflanzung 4000 EUR/ha, kein Risiko)

| Zins | T_F (Jahre) | BEV Faustmann | BEV MSY (68 Jahre) | Verlust der MSY-Regel |
|---|---|---|---|---|
| 0,25 % | 71 | 121.195,7 | 120.935,3 | 260,4 |
| 0,5 % | 68 | 53.261,3 | 53.261,3 | 0,0 |
| 1 % | 62 | 20.140,4 | 19.903,2 | 237,2 |
| 1,5 % | 57 | 9.692,7 | 9.194,2 | 498,5 |
| 2 % | 53 | 4.804,7 | 4.128,6 | 676,2 |
| 2,5 % | 50 | 2.079,5 | 1.301,8 | 777,7 |
| 3 % | 47 | 397,0 | −422,9 | 819,9 |
| 3,5 % | 45 | −711,8 | −1.533,6 | 821,8 |
| 5 % | 40 | −2.434,6 | −3.130,8 | 696,2 |

(Test `test_tabelle_faustmann_gegen_msy_je_zins`, `test_verlust_der_msy_regel_liegt_im_grundfall_zwischen_0_und_822`.) In diesen neun Zinsstufen liegt der Verlust der MSY-Regel zwischen 0 EUR/ha (bei 0,5 %) und 822 EUR/ha (bei 3,5 %). Der Verlust in Prozent des BEV explodiert, wenn der BEV gegen 0 geht (bei 3 % sind es 206,5 % des Faustmann-BEV); die App zeigt deshalb EUR/ha und Break-even-Zinsen (`test_prozentverlust_explodiert_wenn_der_bodenwert_gegen_null_geht`).

| Fall | Break-even-Zins Faustmann | Break-even-Zins MSY-Regel | Faustmann = MSY für Zinsen von … bis … |
|---|---|---|---|
| Grundfall | 3,16 % | 2,85 % | 0,45 % bis 0,52 % |
| Nettoerlös 50 EUR/m³ | 3,67 % | 3,19 % | nicht ausgewertet |
| Pflanzung 8000 EUR/ha | 1,83 % | 1,81 % | 1,05 % bis 1,15 % |
| Pflanzung 2000 EUR/ha | 4,92 % | 3,91 % | nicht ausgewertet |
| Sortimentsprämie 0,4 | 2,81 % | 2,77 % | 1,45 % bis 1,60 % |
| Kalamitätsrisiko 0,5 | 3,07 % | 2,75 % | nicht ausgewertet |
| Kalamitätsrisiko 1 | 2,98 % | 2,66 % | keiner (T_F bleibt unter 68, selbst bei 0,01 % Zins 67 Jahre) |
| Kalamitätsrisiko 2 | 2,80 % | 2,46 % | nicht ausgewertet |

(Tests `test_break_even_zinsen`, `test_kreuzungsintervall_faustmann_gleich_msy`, `test_kalamitaet_allein_verschiebt_nur_die_umtriebszeit`.) Zwischen den beiden Break-even-Zinsen (Grundfall 2,85 % bis 3,16 %) ist der Boden mit der Faustmann-Umtriebszeit noch ein Gewinn, mit der MSY-Regel ein Verlust. Unterhalb des Kreuzungsintervalls ist die Faustmann-Umtriebszeit länger als die MSY-Zeit, oberhalb kürzer (`test_kreuzungsintervall_faustmann_gleich_msy`). **Kalamität allein ist kein Hook:** Risiko 2 verkürzt T_F bei 2 % Zins von 53 auf 50 Jahre und senkt den BEV von 4.804,7 auf 3.172,6 EUR/ha (`test_kalamitaet_allein_verschiebt_nur_die_umtriebszeit`).

### Preis als Zufall (Zins 2 %, φ 0,9, kein Risiko, Preis log-AR(1) mit 21 Zuständen)

Der BEV der festen Regeln (Faustmann 4.804,7, MSY 4.128,6) ist in jeder Zeile gleich: der Preis geht linear ein, der Mittelwert ist fest, also hängen sie weder von σ noch von φ ab. Das ist Konstruktion, kein Fehler (Orakeltest 7).

| σ | Preisregel (t_min, Preis-Schwelle, t_max) | BEV Preisregel | BEV Optimum | Optimum minus Faustmann | Preisregel zum Optimum | MSY-Regel zum Optimum |
|---|---|---|---|---|---|---|
| 0,1 | 48 Jahre, 41,6 EUR/m³, 69 Jahre | 5.124,7 | 5.252,7 | +447,9 (+9,3 %) | −2,4 % | −21,4 % |
| 0,2 | 44, 46,9, 84 | 5.851,5 | 6.137,5 | +1.332,8 (+27,7 %) | −4,7 % | −32,7 % |
| 0,3 | 43, 50,0, 150 | 6.822,6 | 7.254,6 | +2.449,9 (+51,0 %) | −6,0 % | −43,1 % |
| 0,4 | 37, 63,2, 150 | 7.990,4 | 8.552,7 | +3.748,0 (+78,0 %) | −6,6 % | −51,7 % |
| 0,5 | 36, 69,1, 150 | 9.445,4 | 10.007,5 | +5.202,7 (+108,3 %) | −5,6 % | −58,8 % |

(Tests `test_sigma_sweep_optimum_und_gewinn`, `test_sigma_sweep_preisregel_aus_dem_raster`, `test_flexibilitaet_gegen_msy_verlust_in_prozent_des_optimums`.) Prozentangaben „Optimum minus Faustmann“ sind Prozent des Faustmann-BEV 4.804,7, „zum Optimum“ sind Prozent des Optimums. Die Preis-Schwelle ist der Preis der Stufe κ im Preisgitter der jeweiligen Zeile (die Gitter hängen an σ): Stufe 11 bei σ 0,1, Stufe 12 bei σ 0,2 und 0,3, Stufe 13 bei σ 0,4 und 0,5. Die Preisregel liegt in 15 von 18 vorgerechneten Zellen (σ-, φ-, Zins- und Risiko-Sweeps; die 18 Zeilen von `data/regeln.csv` sind σ 0,1 / 0,2 / 0,3 / 0,4 / 0,5 bei φ 0,9 und Zins 2 %, φ 0 / 0,5 / 0,8 / 0,97 bei σ 0,3 und Zins 2 %, Zins 0,5 / 1 / 3 / 3,5 / 4 % bei σ 0,3 und φ 0,9, Risiko 0,5 / 1 / 2 bei σ 0,3, φ 0,9 und Zins 2 % und dazu σ 0,13 bei φ 0,9 und Zins 2 % für das Preset „σ*-Schwelle“; die Grundzelle σ 0,3, φ 0,9, Zins 2 %, Risiko 0 gehört zu mehreren Sweeps und zählt einmal) zwischen 2,4 und 7,6 % unter dem Optimum; in den drei Zinszellen 3, 3,5 und 4 % ist der Abstand größer (14,6 %) oder der Prozentvergleich sinnlos (Optimum nahe 0 oder negativ). In allen 18 Zellen gilt Optimum ≥ Preisregel ≥ Faustmann ≥ MSY, in 17 strikt (bei Zins 0,5 % ist Faustmann gleich MSY; `test_rangfolge_und_preisregel_abstand_ueber_alle_18_zellen_des_rasters`). In 8 von 18 Zellen (alle mit σ ≥ 0,3, Zins höchstens 2 % und ohne Risiko) liegt t_max am Rand von 150 Jahren: der Zwangshieb ist dort untätig, die Regel hat faktisch zwei Parameter; κ = 0 (bei jedem Preis ernten) wählt die Abstimmung in keiner Zelle.

**Persistenz** (σ 0,3, Zins 2 %): Optimum minus Faustmann +7.251,6 (+150,9 %) bei φ 0; +6.324,9 (+131,6 %) bei 0,5; +4.179,9 (+87,0 %) bei 0,8; +2.449,9 (+51,0 %) bei 0,9; +511,3 (+10,6 %) bei 0,97; Preisregel zum Optimum −2,7 / −3,8 / −4,6 / −6,0 / −3,1 % (`test_persistenz_sweep`, Preisregel aus `data/regeln.csv`).

**Zins** (σ 0,3, φ 0,9): Optimum minus Faustmann +19.282 bei 0,5 %, +7.575 bei 1 %, +2.450 bei 2 %, +1.121 bei 3 %; bei 3 % liegen Faustmann bei 397,0, das Optimum bei 1.518,2 und die MSY-Regel bei −422,9. Bei 3,5 % sind Faustmann (−711,8) und MSY (−1.533,6) im Minus, das Optimum bei +98,7 gerade noch im Plus, die abgestimmte Preisregel mit −70,7 nicht mehr. Bei 4 % sind alle negativ (Faustmann −1.479,3, Optimum −874,4, Preisregel −1.006,4; `test_zins_sweep`, `test_hoher_zins_nur_mit_preisflexibilitaet_im_plus`).

**Kalamität** (σ 0,3, φ 0,9, Zins 2 %): Optimum minus Faustmann bleibt bei +50,5 bis +55,0 % (Risiko 0 bis 2), die Preisregel liegt 6,0 bis 7,6 % unter dem Optimum; die Rangfolge ändert sich nicht (`test_kalamitaet_aendert_die_rangfolge_nicht`).

**Struktur des Optimums** (σ 0,3, φ 0,9, Zins 2 %; `test_schwelle_des_optimums_faellt_mit_dem_alter`): der niedrigste Preis, bei dem das Optimum erntet, ist im Alter 30 gleich 85,5 EUR/m³, 35: 74,8, 40: 65,4, 45 und 50: 57,2, 55 bis 70: 50,0, 74 bis 129: 43,7, 130 bis 148: 38,2, Alter 149: 33,4 (Mittelpreis 40; die Werte gelten nur für die genannten Alter, geprüft an den Bereichsgrenzen 74, 129, 130, 148 und 149). Die Schwelle fällt also mit dem Alter (stufig im 21-Punkte-Gitter); in allen 18 Zellen ist sie von Alter zu Alter nie steigend (ab dem ersten Alter mit Hieb bis zum Zwangshieb), und in jedem Alter ist die Erntemenge ein Bereich nach oben (`test_erntemenge_ist_in_jedem_alter_ein_bereich_nach_oben_und_die_schwelle_faellt_ueber_alle_zellen_des_rasters`). Gleichstände: 0 von 3129 Zuständen mit |Hieb − Warten| < 0,001 EUR. Erntealter unter dem Optimum (Monte-Carlo, Zufalls-Seed 20261118, 20.000 Pfade, Hiebe der Jahre 101 bis 399): Mittel 59,8, 10 / 50 / 90 %-Quantil 38 / 56 / 84, Minimum 20, Maximum 150; das Optimum erntet im Mittel später als die feste Faustmann-Zeit 53 (`test_erntealter_unter_dem_optimum_per_monte_carlo`, Band ± 0,6 Jahre beim Mittel).

### Hook: ab welcher Streuung gewinnt das Warten? (σ*)

Gewinn der Preisflexibilität = Optimum minus feste Faustmann-Zeit; Verlust der MSY-Regel = Faustmann minus MSY (676,2 EUR/ha bei 2 %, 237,2 bei 1 %, 819,9 bei 3 %). σ* ist das kleinste σ, bei dem der Gewinn den Verlust erreicht (Pflanzung 4000, Nettoerlös 40, kein Risiko):

| φ | Zins 1 % | Zins 2 % | Zins 3 % |
|---|---|---|---|
| 0 | 0,0075 | 0,043 | 0,092 |
| 0,9 | 0,033 | 0,129 | 0,245 |
| 0,97 | 0,110 | 0,348 | nicht erreicht bis σ 0,5 |

(`data/sigma_stern.csv`; Tests `test_sigma_stern_tabelle_aus_der_datei_und_live`, `test_verlust_der_msy_regel_je_zins_in_der_sigma_tabelle`, `test_gewinn_der_flexibilitaet_bei_sigma_0_3_je_phi_und_zins`.) Im Grundfall (φ 0,9, Zins 2 %) übertrifft der Gewinn der Preisflexibilität den Verlust der MSY-Regel ab σ* = 0,129; bei σ 0,3 ist der Gewinn 2.449,9 gegen einen Verlust von 676,2 EUR/ha, also das 3,6-Fache. Die Aussage ist **zins- und persistenzabhängig**: bei σ 0,3 gewinnt in der Tabelle die richtige Umtriebszeit bei φ 0,97 und Zins 2 oder 3 % (2 %: 511,3 gegen 676,2), in den übrigen sieben Zellen die Flexibilität; bei höherem Zins reicht σ 0,3 auch bei φ 0,9 nicht (4 %: σ* 0,358, 5 %: σ* 0,456; `test_sigma_stern_bei_hohem_zins_und_zellen_bei_sigma_0_3`). Alles davon hängt an σ und φ, für die es hier keine Daten gibt.

**Robustheit** (σ 0,3, φ 0,9, Zins 2 %; Mindesthiebalter und feste Hiebkosten sind nur in dieser Prüfung einstellbar, nicht in der App):

| Variante | BEV fest | BEV Optimum | Optimum minus fest |
|---|---|---|---|
| Grundfall | 4.804,7 | 7.254,6 | +2.449,9 (+51,0 %) |
| Sortimentsprämie 0,4 | 3.626,8 | 5.640,7 | +2.013,9 (+55,5 %) |
| Sortimentsprämie 0,6 | 3.317,1 | 5.138,5 | +1.821,5 (+54,9 %) |
| frühestes Hiebalter 40 | 4.804,7 | 7.146,0 | +2.341,2 (+48,7 %) |
| frühestes Hiebalter 50 | 4.804,7 | 6.716,6 | +1.911,8 (+39,8 %) |
| frühestes Hiebalter 60 | 4.635,9 | 5.921,9 | +1.286,0 (+27,7 %) |
| feste Hiebkosten 2000 EUR/ha | 3.792,5 | 6.346,4 | +2.553,9 (+67,3 %) |
| Sortiment 0,4 + Alter ≥ 50 + Risiko 1 | 2.569,8 | 4.015,6 | +1.445,8 (+56,3 %) |
| σ 0,2 + Sortiment 0,4 + Alter ≥ 50 | 3.626,8 | 4.677,4 | +1.050,5 (+29,0 %) |

(`test_robustheit_des_gewinns_der_preisflexibilitaet`, `test_robustheit_gewinn_liegt_zwischen_27_und_68_prozent`.) Der Gewinn liegt in den acht Varianten mit σ 0,3 zwischen +27,7 und +67,3 %, mit σ 0,2 und den zwei Einschränkungen bei +29,0 %. Ein Mindesthiebalter ist nur ein grober Ersatz für die Grenzen eines Betriebs mit vielen Beständen und einer Nachhaltigkeitsgrenze (Hiebsatz, Altersklassen), die das Modell nicht hat.

### Drei Bonitäten (Tab „Bonität“, konstanter Preis, kein Risiko)

Bonität als Zeitskalierung des Wachstums (g = 0,8 / 1,0 / 1,2), je ein Drittel der Fläche. Abstand des Mittels der drei Bonitäten zum Mittel der je Bonität richtigen Faustmann-Umtriebszeiten (BEV-Mittel 20.189,5 / 4.861,4 / 450,7 EUR/ha bei 1 / 2 / 3 % Zins), in Prozent dieses Mittels:

| Zins | T_F je Bonität | eine feste Zeit für alle | Zieldurchmesser (ein A*) | MSY je Bonität (85 / 68 / 57 Jahre) | MSY der mittleren Bonität für alle |
|---|---|---|---|---|---|
| 1 % | 74 / 62 / 53 | T 61: −2,27 % | A* 61,25 (T 77 / 62 / 52): −0,07 % | −1,29 % | −3,66 % |
| 2 % | 63 / 53 / 46 | T 53: −3,48 % | A* 53,0 (T 67 / 53 / 45): −0,37 % | −14,07 % | −18,24 % |
| 3 % | 55 / 47 / 42 | T 47: −20,07 % | A* 47,0 (T 59 / 47 / 40): −2,86 % | −180,07 % | −208,87 % |

(`test_bonitaeten_zieldurchmesser_gegen_feste_zeit`, `test_zieldurchmesser_liegt_in_allen_drei_faellen_naeher_an_faustmann_als_feste_zeit_und_msy`.) Der Zieldurchmesser liegt in diesen drei Zinsstufen (1, 2 und 3 %) näher an Faustmann als die feste Zeit und die MSY-Regel. **Vorbehalt:** weil Durchmesser und Volumen hier in derselben Zeitskalierung wachsen, ist die Zieldurchmesser-Regel konstruktionsbedingt eine Regel in effektivem Alter; dass sie nahe an Faustmann liegt, ist teilweise eingebaut. Ohne Daten zum Zusammenhang von Durchmesser, Alter und Bonität nicht belastbar. Prozente bei 3 % beziehen sich auf einen kleinen BEV (450,7 EUR/ha); die App zeigt deshalb EUR/ha.

### Handbeispiel in exakten Brüchen (Tab „Von Hand“)

Volumen 5, 20, 30, 42, 44 m³/ha im Alter 1 bis 5, Pflanzung 20, Zins 1/5, Preis 5: BEV(2) = 1780/11 = 161,82 (Faustmann), BEV(4) = 105330/671 = 156,97 (MSY, 4 Jahre), Verlust 4,84. Mit zwei Preiszuständen 2 und 8 (je 1/2, bleibt mit 3/4), ohne Kalamität und mit Hieb im Alter 5 immer, ist das Optimum aus allen 256 Politiken (je Alter 1 bis 4 und Preiszustand Hieb oder Warten: 2^8), jede als Gleichungssystem in Brüchen gelöst, 90963860/432357 = 210,39; die beste Politik wartet im Alter 1 immer und erntet in den Altern 2 bis 4 nur bei hohem Preis; Gewinn der Preisflexibilität 48,57 gegen einen Verlust der MSY-Regel von 4,84 (`test_handbeispiel_zahlen`, `tests/test_model.py`). Ein Zahlenbeispiel, kein Befund über Wälder.

### Befunde und Korrekturen gegenüber der Vorab-Messreihe

- **Kreuzungszins:** die Vorab-Messreihe nannte 0,45 % (erstes Gitterfeld, bei dem T_F = T_MSY wird). Gemessen ist ein Intervall von 0,45 % bis 0,52 % (Grundfall), in dem beide Zeiten gleich sind; das Preset „Kreuzungszins“ nimmt 0,5 %.
- **σ\*:** Die Schwelle wird jetzt per Bisektion im Gitterintervall bestimmt und bis σ 0,5 gesucht; die Werte der Vorab-Messreihe (bis σ 0,3) bleiben bis auf die Rundung gleich (0,129 bei φ 0,9, Zins 2 %). Neu: bei φ 0,97 und Zins 2 % liegt σ* bei 0,348; bei φ 0,97 und Zins 3 % wird sie bis σ 0,5 nicht erreicht.
- **Volumen im Alter 60:** 592,5 m³/ha (die Vorab-Messreihe rundete auf 593).
- **Erntealter unter dem Optimum:** Mittel 59,8 (Vorab 59,9), Minimum 20 (Vorab 21): die Simulation der Demo zieht die Zufallszahl der Kalamität mit, also ein anderer Zufallsstrom bei gleichem Zufalls-Seed. Quantile 38 / 56 / 84 unverändert.
- **Hoher Zins 3,5 %:** neu gemessen, siehe oben: nur das Optimum bleibt im Plus, die abgestimmte Preisregel nicht.
- **Preisregel:** die feste Umtriebszeit ist jetzt ein fünfter Startpunkt der Koordinatensuche; in keiner der 18 Zellen änderte sich dadurch ein Wert gegenüber den vier Startpunkten der Vorab-Messreihe.

## Ehrliche Grenzen

- **Lehrmodell mit offenen Annahmen, keine Daten:** Weder eine Ertragstafel noch eine Holzpreisreihe wurde gelesen; alle Zahlen sind gewählt. Volumen nach Chapman-Richards mit $V_{\max} = 1100$ m³/ha, $k = 0{,}028$, $p = 3$ (V(40) = 336,4, V(60) = 592,5, V(100) = 911,3 m³/ha; der durchschnittliche Zuwachs kulminiert bei 68 Jahren mit 9,97 m³/(ha·Jahr)), Nettoerlös 40 EUR/m³ (Holzpreis minus Ernte), Pflanzung 4000 EUR/ha (Regler 2000 bis 8000), Zins 0,25 bis 5 %. Die Formen der Kurven sind gewählt, nicht an Daten angepasst; Zahlen wie 68 Jahre oder 3,16 % sind **keine Forstaussagen**, und nichts davon ist eine Empfehlung für einen Betrieb.
- **Der Preisprozess ist die Annahme, an der der Hook hängt:** Preis als log-AR(1) mit der Streuung σ des Log-Preises und der Persistenz φ als **Reglern**; es wurde keine Preisreihe gelesen, σ und φ sind nicht aus Daten geschätzt. Die Größe des Flexibilitätsgewinns hängt an beiden (bei φ = 0,97 und Zins 2 % reicht er bis σ = 0,3 nicht an den Verlust der MSY-Regel heran, bei φ = 0,9 schon ab σ* = 0,129). Eine Kopplung von Preis und Kalamität (etwa niedrigere Preise nach Käferbefall) ist eine Vermutung ohne Beleg und nicht modelliert; das Modell hat unabhängige Preise. Der Preis ist eine Markov-Kette mit 21 Zuständen (Rouwenhorst; Mittelwert des Preises und Streuung des Log-Preises exakt); Gegenprobe mit 11 / 21 / 41 / 61 Zuständen (σ 0,3, φ 0,9, Zins 2 %): Optimum 7255,2 / 7254,6 / 7254,0 / 7253,6 EUR/ha.
- **Zu viel Flexibilität:** ein einzelner Bestand, Verkauf in jedem Jahr zum Marktpreis, keine Mengenwirkung auf den Preis, kein Hiebsatz. Ein Betrieb mit vielen Beständen und einer Nachhaltigkeitsgrenze kann nicht auf jeden Preis reagieren; das ist **nicht gemessen**. Nur ein frühestes Hiebalter dient in der README als grober Ersatz (Gewinn der Flexibilität bei σ 0,3: +27,7 % mit Mindestalter 60 gegen +51,0 % ohne).
- **Kalamität allein ist kein Hook:** Hazard $\eta(a) = \rho\,(0{,}002 + 0{,}02\,(a/100)^3)$ je Jahr (bei Risiko ρ = 1 sind es 2,2 % je Jahr im Alter 100), Notverkauf mit 50 % des Holzwerts, danach Wiederaufforstung. Risiko 2 verkürzt die Faustmann-Umtriebszeit bei 2 % Zins von 53 auf 50 Jahre und senkt den Bodenwert von 4805 auf 3173 EUR/ha; die Rangfolge der Regeln ändert es in den geprüften Fällen (σ 0,3, φ 0,9, Zins 2 %, Risiko 0 bis 2) nicht.
- **Erzwungene Wiederaufforstung** (Faustmann-Annahme). Mit der Option, nicht zu pflanzen oder zu warten, bleibt der Bodenwert bei Zins 2 % gleich (7254,6 EUR/ha bei σ 0,3, φ 0,9), bei Zins 3,5 % (Bodenwert 98,7) steigt er mit der Option nur auf 98,8, bei Zins 4 % wird er 0 statt −874,4 EUR/ha. Die erzwungene Wiederaufforstung wirkt in diesen Messungen (σ 0,3, φ 0,9, Zins 2, 3,5 und 4 %) bei positivem Bodenwert kaum und deutlich erst dort, wo der Bodenwert nahe 0 oder negativ ist.
- **Die Preisregel ist abgestimmt, nicht bewiesen beste:** drei Parameter (frühestes Hiebalter, Preis-Schwelle, spätestes Hiebalter) je Szenario per Koordinatensuche mit fünf Startpunkten; ein lokales Optimum ist möglich. Bei σ ≥ 0,3 (φ 0,9, Zins 2 %) liegt das späteste Hiebalter am Rand von 150 Jahren: der Zwangshieb ist dort untätig, die Regel hat faktisch zwei Parameter.
- **Feste Regeln hängen weder von σ noch von φ ab** (der Preis geht linear ein, der Mittelwert ist fest): in den Zeilen „Faustmann“ und „MSY“ der Tabelle bleibt der Bodenwert bei jedem σ und φ gleich. Das ist Konstruktion, kein Fehler; nur die Preisregel und das Optimum ändern sich.
- **Prozentzahlen am Bodenwert sind fragil** (er ist ein kleiner Rest großer Zahlungen und geht gegen 0): die App zeigt EUR/ha, Break-even-Zinsen und absolute Lücken.
- **Nicht modelliert:** Jahreskosten (Verwaltung, Versicherung), Durchforstung, Mischbestände, Steuern, Subventionen, Risikoaversion, mehrere Altersklassen (Normalwald), Kopplung von Preis und Kalamität, Marktzins gegen Zielrendite des Besitzers.
- **Bonität (Tab):** Drei Bonitäten als Zeitskalierung des Wachstums (g = 0,8 / 1,0 / 1,2), je ein Drittel der Fläche. Weil Durchmesser und Volumen hier in derselben effektiven Zeit g·Alter wachsen, ist die Zieldurchmesser-Regel konstruktionsbedingt eine Regel in effektivem Alter; dass sie nahe an Faustmann liegt, ist teilweise eingebaut. Ohne Daten zum Zusammenhang von Durchmesser, Alter und Bonität nicht belastbar.
- **Quellen (nicht aus dem Volltext):** Faustmann (1849, Jahr aus einer Sekundärquelle der Suche, Original nicht gelesen) und Samuelson (1976), „Economics of Forestry in an Evolving Society“, Economic Inquiry 14(4), 466–492 (Angaben über einen Katalogeintrag der Suche bestätigt, Volltext nicht gelesen). Brazee und Mendelsohn (1988), „Timber Harvesting with Fluctuating Prices“, Forest Science 34(2), 359–372, Reservationspreis-Regel: nur Abstract und Titel über die Suche gelesen, Volltext nicht gelesen. Die Rechnung dieser Demo ist eigenständig, nicht aus diesen Arbeiten übernommen. Das Rouwenhorst-Verfahren und die Chapman-Richards-Kurve sind nur als Verfahrens- und Kurvennamen genannt, ohne nachgeschlagene Quelle. Ob die Schwellen des Optimums der Reservationspreis-Regel von Brazee und Mendelsohn entsprechen, wurde nicht geprüft.

## Verifikation

- **Kern gegen Handrechnung:** das Handbeispiel (fünf Alter, zwei Preiszustände) in exakten Brüchen gegen den Gleitkommakern für alle 256 Politiken, mit und ohne Kalamität; Notverkauf im ersten Jahr von Hand (R = −5/4), Zwangshieb im höchsten Alter, frühestes Hiebalter, Gitterrand des Preises, Gleichstand von Hieb und Warten, Kreuzungszins ohne Kreuzung (`tests/test_model.py`).
- **Orakeltests** (`tests/test_oracle_forst_umtriebszeit.py`, anderer Rechenweg je Kernroutine): geschlossene Faustmann-Formel (12 Fälle), Zyklus-Erwartung mit Kalamität und Notverkauf (9 Fälle), Aufzählung aller festen Umtriebszeiten auf einer Mini-Instanz, Howard-Politikiteration gegen einfache Wertiteration (4 Szenarien), Rückwärtsrekursion gegen das gesamte Gleichungssystem (zufällige Politiken), Monte-Carlo gegen die exakte Auswertung (20.000 Pfade, 400 Jahre, Zufalls-Seed 20261118; |z| < 3 für feste Zeit, Preisregel und Optimum), Invarianz fester Regeln gegen σ und φ, σ = 0 gleich Faustmann, Rangfolge der Regeln und Monotonie im Zins, Pflanzoption, Abstimmung der Preisregel gegen vollständige Aufzählung auf einer Mini-Instanz, vorgerechnete Zeilen gegen Neuabstimmung.
- **Zahlen der README:** `tests/test_claims.py` rechnet die Zahlen der Befunde als Band nach (Tabellen mit Testnamen); die Schwelle des Optimums je Alter ist an den Bereichsgrenzen 74, 129, 130, 148 und 149 und an Stichproben dazwischen geprüft, nicht in jedem Alter einzeln. Nicht im Test stehen Laufzeiten und die Vorab-Werte der früheren Messreihe (Volumen 593, Mittel 59,9, Minimum 21).
- **Verdikte:** reine Funktionen (`fut_evaluation.py`), jede Schwelle und jede Zahl im Text an handgesetzten Werten, je Preset die erwartete Verdikt-Zeile (`tests/test_verdicts.py`).
- **Presets:** vollständig, Werte innerhalb der Reglergrenzen und auf dem Schrittraster, jedes Preset unterscheidet sich vom Grundfall in genau einem Regler, die Preisregel ist für jedes Preset vorgerechnet (`tests/test_presets.py`).
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum), Inhalt (Tabs, Expander per Titel, Kernfragen, Handbeispiel, jede abgesetzte Formel in einem eigenen Markdown-Aufruf), Footer-Test, Laufzeit-Importe.
- **Layout:** Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben; jede Änderung eines Reglers hebt die Markierung auf (`tests/test_layout.py`).
- **Zufall deterministisch:** die Zufallsbahn der App und alle Simulationen nehmen nur gleichverteilte Zahlen des NumPy-Generators mit festem Zufalls-Seed (keine C-Bibliotheksfunktionen); eine Zufallsbahn ist nie ein Befund.
- Die Test-Zahl dieser Demo: 175 Tests.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Kopf, Seitenleiste, vier Tabs, Kernfragen, Expander |
| `data/regeln.csv` | vorgerechnete abgestimmte Preisregeln (18 Szenarien) |
| `data/sigma_stern.csv` | vorgerechnete Schwelle σ* nach φ und Zins |
| `fut_constants.py` | Modell-Annahmen, Regler-Grenzen, Farben, Presets |
| `fut_det.py` | Faustmann gegen MSY bei konstantem Preis, Break-even-Zins, Kreuzungsintervall |
| `fut_evaluation.py` | Einstellungen, Kennzahlen, Verdikte, Bonität |
| `fut_grid.py` | vorgerechnete Tabellen (`python fut_grid.py`) |
| `fut_hand.py` | Handbeispiel in exakten Brüchen |
| `fut_layout.py` | Fester Kopf (Stylesheet) |
| `fut_model.py` | Waldmodell, Preiskette, exakte Auswertung, Howard-Iteration |
| `fut_pdf_export.py` | PDF-Export |
| `fut_presets.py` | Permalink, Presets (Schnellstart in der Seitenleiste), Zufalls-Seed-Button |
| `fut_rules.py` | Preisregel: Abstimmung und vorgerechnete Zeilen |
| `fut_stoch.py` | Optimum, σ*, Zufallsbahn und Verlauf der Regeln |
| `fut_texts.py` | Texte, die App und README gemeinsam führen |
| `fut_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Kern, Orakel, Claims, Verdikte, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Mehrere Altersklassen** (Normalwald, Hiebsatz, Übergang aus einem unregelmäßigen Bestand), Kopplung von Preis und Kalamität, Durchforstung, Jahreskosten, Risikoaversion: nicht modelliert, nicht gemessen.
- **Der Break-even-Zins mit Preisflexibilität** (bei wie hohem Zins lohnt sich der Boden nur mit Warten auf den Preis) ist nicht als eigene Kennzahl in der App; der Preset „Hoher Zins“ zeigt den Fall bei 3,5 %.
- **Eine echte Preisreihe** (Fichten-Stammholz, jährlich) zur Schätzung von σ und φ wurde nicht gelesen. Erst damit wäre der Zahlenkern des Hooks belegt.
- **Das Mindesthiebalter und feste Hiebkosten** sind im Modell, aber keine Regler der App (nur die Robustheitsprüfung nutzt sie).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die vorgerechneten Tabellen entstehen mit `python fut_grid.py` (einige Minuten).

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
