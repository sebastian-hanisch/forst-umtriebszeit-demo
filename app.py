"""Forst: Wann ernten? Faustmann, MSY und das Warten auf den Preis - interaktive Fall-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Lehrmodell mit offenen Annahmen (keine Ertragstafel, keine Holzpreisreihe): Umtriebszeit nach Faustmann gegen MSY bei konstantem Preis, dann Preis als Zufall
(log-AR(1), 21 Zustände) mit fester Zeit, MSY, abgestimmter Preisregel und dem DP-Optimum, alle exakt ausgewertet.
Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import fut_constants as C
import fut_det as D
import fut_evaluation as EV
import fut_grid as G
import fut_hand as H
import fut_rules as RU
import fut_stoch as ST
import fut_texts as TX
import fut_visualization as V
from fut_layout import dunkles_design, kopf_css
from fut_model import Modell, mod
from fut_pdf_export import build_pdf
from fut_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)

st.set_page_config(page_title="Forst: Wann ernten? – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Umtriebszeit und Break-even-Zinsen werden gerechnet ...")
def _det(p):
    return D.analyse(p), D.kreuzungsintervall(p)


@st.cache_data(show_spinner=False)
def _kurven(p):
    ts, bev = D.bev_kurve(p)
    a, v, lauf, schnitt = D.zuwachs(p)
    return ts, bev, (a, v, lauf, schnitt)


@st.cache_data(show_spinner="Das Optimum (Politikiteration) wird gerechnet ...")
def _stoch(p):
    m = Modell(p)
    tf, bf = m.bestes_fix()
    opt, mask, _, _ = ST.optimum(m, tf)
    return {"tf": tf, "bf": bf, "tm": m.t_msy(), "bm": m.bev_msy(), "opt": opt, "mask": mask}


@st.cache_data(show_spinner="Die Preisregel wird abgestimmt (Koordinatensuche, etwa 4 bis 11 Sekunden, auf langsameren Servern länger) ...")
def _regel_live(p, tf):
    return RU.abstimmen(Modell(p), tf)


@st.cache_data(show_spinner=False)
def _regeln_raster():
    return RU.laden()


@st.cache_data(show_spinner="Der Gewinn der Preisflexibilität über σ wird gerechnet ...")
def _sigma(p0):
    """p0: Parameter mit sigma = 0 (σ ist die Achse)."""
    sig = np.round(np.arange(0.0, C.SIGMA_MAX + 1e-9, C.SIGMA_GRID_STEP), 4)
    g, _, _ = ST.sigma_kurve(p0, sig)
    star, verlust = ST.sigma_stern(p0)
    return sig, g, star, verlust


@st.cache_data(show_spinner=False)
def _verlaeufe(p, seed, regel_t):
    m = Modell(p)
    s = _stoch(p)
    bahn = ST.bahn_ziehen(m, seed)
    masken = {"Faustmann (fest)": m.maske_fix(s["tf"]), "MSY (fest)": m.maske_fix(s["tm"]), "Optimum (DP)": s["mask"]}
    if regel_t is not None:
        masken["Preisregel (abgestimmt)"] = m.maske_regel(*regel_t)
    return m.preise, bahn.preis_index, {k: ST.verlauf(m, v, bahn) for k, v in masken.items()}


@st.cache_data(show_spinner="Die drei Bonitäten werden gerechnet ...")
def _bonitaet(p):
    return EV.bonitaet(p)


@st.cache_data(show_spinner="Das Handbeispiel wird in Brüchen gerechnet (256 Politiken) ...")
def _hand():
    pol, bev = H.optimum_aufzaehlung()
    return pol, bev, [(T, H.bev_fest(T)) for T in range(1, H.AMAX + 1)], {T: H.auswerten(H.feste_politik(T))[1] for T in (2, 4)}


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in fut_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🌲 Forst: Wann ernten?")
st.markdown(
    """
Wann soll ein Bestand geerntet werden? Die Demo vergleicht **Faustmann** (die Umtriebszeit, die den Bodenertragswert am größten macht), die **MSY-Regel** (die Zeit der größten durchschnittlichen Zuwachsleistung), eine **Preisregel** und das **DP-Optimum**, das auf einen guten Preis wartet. Die Frage: Ist der Streit um die Umtriebszeit klein gegen den Wert, auf den Preis zu warten, und ab welcher Preisstreuung kippt das?

Das ist ein **Lehrmodell mit offenen Annahmen**: es gibt keine echten Holzpreisreihen darin, die Preisstreuung σ und die Persistenz φ sind Regler, und die Größe des Effekts hängt an ihnen (keine Empfehlung für einen Betrieb). Direkt darunter erklärt der Expander **💬 In einfachen Worten** das Ganze ohne Formeln. Weiter unten: die **Kernfragen**, dann die Expander **🔧 Wie wir das erreichen – vollständiger Methodenvergleich**, **📐 Mathematische Formulierung** und **⚠️ Annahmen, Grenzen und bekannte Probleme**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/forst-umtriebszeit-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Wann soll ein Bestand geerntet werden? Erntet man früh, gibt es wenig Holz, aber das Geld kommt früh; erntet man spät, gibt es viel Holz, aber man wartet lange. Die Anwendung stellt vier Regeln nebeneinander und rechnet für jede, wie viel der Waldboden am Ende wert ist: die **Faustmann-Regel** (das Alter, bei dem der Wert des Bodens am größten ist), die **MSY-Regel** (das Alter mit dem größten durchschnittlichen Holzzuwachs), eine **Preisregel** (erntet, wenn der Preis hoch genug ist) und das **Optimum** (die beste Regel, die für jedes Alter und jeden Preis das Warten gegen das Ernten abwägt).

Der Holzpreis springt hier zufällig von Jahr zu Jahr. Zwei Regler bestimmen ihn: die Preisstreuung (wie weit die Preise auseinanderliegen) und die Persistenz (wie lange ein hoher oder niedriger Preis anhält). Ob sich das Warten auf einen guten Preis mehr lohnt als der Unterschied zwischen Faustmann- und MSY-Regel, hängt an beiden und am Zins. Alle Zahlen sind Annahmen: Es gibt keine echten Holzpreisreihen darin, und das Ergebnis ist keine Empfehlung für einen Betrieb.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/forst-umtriebszeit-erklaert.html)
        """
    )
# erklaerseite:einfach:end
st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Wald und Markt (alle Tabs)**")
    zins = st.slider("Zins (% je Jahr)", *bounds("zins_slider"), key="zins_slider", step=C.ZINS_STEP, help="Kalkulationszins für die Abzinsung (0,25 bis 5 %).")
    pflanz = st.slider("Pflanzkosten (EUR/ha)", *bounds("pflanz_slider"), key="pflanz_slider", step=C.PFLANZ_STEP, help="Kosten der Wiederaufforstung im Jahr des Hiebs.")
    preis = st.slider("Nettoerlös (EUR/m³)", *bounds("preis_slider"), key="preis_slider", step=C.PREIS_STEP,
                      help="Holzpreis minus Erntekosten; bei zufälligem Preis der Mittelwert der Preise.")
    sort = st.slider("Sortimentsprämie", *bounds("sort_slider"), key="sort_slider", step=C.SORT_STEP,
                     help="0 = jedes Holz gleich viel wert; sonst wächst der Preisfaktor logistisch von 1 − Prämie (junges Holz) auf 1 (Mitte Alter 50, Breite 10).")
    risiko = st.slider("Kalamitätsrisiko", *bounds("risiko_slider"), key="risiko_slider", step=C.RISIKO_STEP,
                       help="Skaliert die jährliche Kalamitätsrate 0,002 + 0,02·(Alter/100)³ (bei 1: 2,2 % im Alter 100); Notverkauf mit 50 % des Holzwerts.")
    st.markdown("**Preis als Zufall (nur Tab „Preis als Zufall“ und Kernfragen 2 und 3)**")
    sigma = st.slider("σ: Streuung des Log-Preises", *bounds("sigma_slider"), key="sigma_slider", step=C.SIGMA_STEP,
                      help="Stationäre Standardabweichung des Logarithmus des Preises. 0 = konstanter Preis. Keine Daten: Regler.")
    phi = st.slider("φ: Persistenz des Preises", *bounds("phi_slider"), key="phi_slider", step=C.PHI_STEP,
                    help="Autokorrelation des Log-Preises von Jahr zu Jahr. 0 = jedes Jahr neu gewürfelt, nahe 1 = Preisausschläge halten lange an. Keine Daten: Regler.")
    seed = st.number_input("Zufalls-Seed der Zufallsbahn", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Zufallsbahn", width="stretch", on_click=randomize_seed)

sync_query_params({"zins_slider": float(zins), "pflanz_slider": float(pflanz), "preis_slider": float(preis), "sort_slider": float(sort),
                   "risiko_slider": float(risiko), "sigma_slider": float(sigma), "phi_slider": float(phi), "seed_input": int(seed)})

s = EV.Settings(float(zins), float(pflanz), float(preis), float(sort), float(risiko), float(sigma), float(phi), int(seed))
p = EV.to_prm(s)
p_det = mod(p, sigma=0.0, phi=C.DEFAULT_PHI)   # feste Regeln und Bonität hängen weder von σ noch von φ ab: ein gemeinsamer Cache-Schlüssel

tab_hand, tab_det, tab_stoch, tab_bon = st.tabs(["✋ Von Hand", "🌲 Umtriebszeit", "💶 Preis als Zufall", "🌳 Bonität"])

# --- Von Hand ------------------------------------------------------------------------------------------------------------------------------
with tab_hand:
    pol, hand_bev, fest_tab, fest_bev = _hand()
    st.markdown(
        "Ein Handbeispiel in **exakten Brüchen**, unabhängig vom Kern gerechnet. Volumen im Alter 1 bis 5 Jahre: 5, 20, 30, 42, 44 m³ je Hektar (erfunden), Pflanzkosten 20, Zins 1/5. "
        "Im ersten Abschnitt ist der Preis konstant 5. Der Bodenertragswert einer festen Umtriebszeit $T$ ist $\\mathrm{BEV}(T) = (5\\,V(T) - 20\\cdot 1{,}2^T)/(1{,}2^T - 1)$."
    )
    st.dataframe(
        [{"T (Jahre)": T, "V(T)": str(H.V[T]), "V(T)/T": H.bruch(H.V[T] / T), "BEV (Bruch)": H.bruch(b), "BEV": round(float(b), 3)} for T, b in fest_tab],
        hide_index=True, width="stretch",
    )
    tfh, bfh = H.faustmann()
    st.markdown(
        f"Faustmann: **{tfh} Jahre** mit {H.bruch(bfh)} = {EV.zahl(float(bfh), 2)}. MSY (größtes V(T)/T): **{H.msy()} Jahre** mit {H.bruch(H.bev_fest(H.msy()))} = {EV.zahl(float(H.bev_fest(H.msy())), 2)}. "
        f"Die MSY-Regel verliert {H.bruch(bfh - H.bev_fest(H.msy()))} = {EV.zahl(float(bfh - H.bev_fest(H.msy())), 2)}."
    )
    st.markdown(
        "**Zweiter Abschnitt: zwei Preiszustände.** Preis 2 oder 8 (Mittelwert 5), bleibt mit Wahrscheinlichkeit 3/4, wechselt mit 1/4; stationär je 1/2; im Alter 5 wird immer geerntet; keine Kalamität. "
        "Eine Politik sagt für jedes Alter 1 bis 4 und jeden Preis, ob geerntet wird: $2^8 = 256$ Politiken, jede wird als **ganzes Gleichungssystem** in Brüchen gelöst, die beste gewinnt (kein Verfahren, das rät)."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Faustmann (feste Zeit 2)", H.bruch(fest_bev[2]), delta=f"= {EV.zahl(float(fest_bev[2]), 2)}, gleich dem Wert bei konstantem Preis", delta_color="off")
    c2.metric("MSY (feste Zeit 4)", H.bruch(fest_bev[4]), delta=f"= {EV.zahl(float(fest_bev[4]), 2)}", delta_color="off")
    c3.metric("Optimum aus allen 256 Politiken", H.bruch(hand_bev), delta=f"= {EV.zahl(float(hand_bev), 2)}", delta_color="off")
    st.markdown("Die beste Politik: " + "; ".join(f"Alter {a}: **{t}**" for a, t in H.beschreibung(pol)) + ". "
                f"Der Gewinn der Preisflexibilität ist {EV.zahl(float(hand_bev - fest_bev[2]), 2)} gegen einen Verlust der MSY-Regel von {EV.zahl(float(fest_bev[2] - fest_bev[4]), 2)}: im Handbeispiel ist das Warten auf den Preis mehr wert als der Streit um die Umtriebszeit "
                "(nur ein Zahlenbeispiel, kein Befund über Wälder).")

# --- Umtriebszeit --------------------------------------------------------------------------------------------------------------------------
det, kreuz = _det(p_det)
ts_c, bev_c, (a_c, v_c, lauf_c, schnitt_c) = _kurven(p_det)
with tab_det:
    st.markdown("## 🌲 Umtriebszeit bei konstantem Preis")
    st.caption("Alle Regler außer σ, φ und Zufalls-Seed wirken hier; der Preis ist der Nettoerlös je m³.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Faustmann", f"{det.tf} Jahre", delta=f"BEV {EV.eur(det.bev_f)} EUR/ha", delta_color="off")
    m2.metric("MSY", f"{det.tm} Jahre", delta=f"BEV {EV.eur(det.bev_m)} EUR/ha", delta_color="off")
    m3.metric("Verlust MSY (EUR/ha)", f"{EV.eur(det.verlust)}", delta="Faustmann minus MSY", delta_color="off")
    m4.metric("Break-even-Zins", f"F {EV.pct(det.be_f)}" if det.be_f is not None else "–", delta=f"MSY {EV.pct(det.be_m)}" if det.be_m is not None else "MSY –", delta_color="off")
    if kreuz[0] is None:
        st.caption("Kreuzungszins: bei diesen Einstellungen ist die Faustmann-Umtriebszeit im Zinsbereich 0,01 bis 8 % nirgends gleich der MSY-Umtriebszeit.")
    elif kreuz[1] >= kreuz[0]:
        st.caption(f"Kreuzungszins: die Faustmann-Umtriebszeit ist gleich der MSY-Umtriebszeit für Zinsen von {EV.pct(kreuz[0])} bis {EV.pct(kreuz[1])}; darunter ist sie länger, darüber kürzer.")
    else:
        st.caption(f"Kreuzungszins: die Faustmann-Umtriebszeit springt bei {EV.pct(kreuz[0])} über die MSY-Umtriebszeit, ohne ihr gleich zu sein.")
    ca, cb = st.columns(2)
    ca.plotly_chart(V.fig_zuwachs(a_c, v_c, lauf_c, schnitt_c, det.tm, det.tf), width="stretch", key="zuwachs_chart")
    cb.plotly_chart(V.fig_bev(ts_c, bev_c, det.tf, det.bev_f, det.tm, det.bev_m), width="stretch", key="bev_chart")
    st.caption("Links: die MSY-Zeit liegt am Schnittpunkt von laufendem und durchschnittlichem Zuwachs; sie kennt weder Zins noch Preis. Rechts: der Bodenertragswert je fester Umtriebszeit, das Maximum ist die Faustmann-Zeit.")

# --- Preis als Zufall ----------------------------------------------------------------------------------------------------------------------
stoch = _stoch(p)
with tab_stoch:
    st.markdown("## 💶 Preis als Zufall: vier Regeln, exakt ausgewertet")
    st.caption(f"σ = {EV.zahl(sigma, 2)}, φ = {EV.zahl(phi, 2)}, Zins {EV.zahl(zins, 2)} %. Feste Regeln hängen weder von σ noch von φ ab (der Preis geht linear ein): gleiche Zahlen in diesen Zeilen sind Konstruktion.")
    regel = None
    if p.sigma <= 0:
        regel = RU.Regel(stoch["tf"], 0, stoch["tf"], stoch["bf"], "fest")
    else:
        regel = RU.vorgerechnet(p, _regeln_raster())
        if regel is None:
            if st.button("Preisregel für diese Einstellung abstimmen (etwa 4 bis 11 Sekunden)", key="regel_start"):
                st.session_state["regel_done"] = p
            if st.session_state.get("regel_done") == p:
                regel = _regel_live(p, stoch["tf"])
            else:
                st.info("Für diese Einstellung ist die Preisregel nicht vorgerechnet; der Knopf stimmt sie live ab (Koordinatensuche; gemessen auf einem Entwicklungsrechner je nach Einstellung etwa 4 bis 11 Sekunden, auf langsameren Servern länger).")
    m = Modell(p)
    zeilen = [{"name": "Faustmann (fest)", "bev": stoch["bf"], "par": f"T = {stoch['tf']} Jahre"},
              {"name": "MSY (fest)", "bev": stoch["bm"], "par": f"T = {stoch['tm']} Jahre"}]
    if regel is not None:
        zeilen.append({"name": "Preisregel (abgestimmt)", "bev": regel.bev,
                       "par": f"ab {regel.tmin} Jahren bei Preis ≥ {EV.zahl(regel.schwelle(m), 1)} EUR/m³ (Index {regel.k}), spätestens mit {regel.tmax} Jahren"})
    zeilen.append({"name": "Optimum (DP)", "bev": stoch["opt"].bev, "par": "Schwelle je Alter (Kurve unten)"})
    for z in zeilen:
        z["vs_opt"] = EV.abstand(z["bev"], stoch["opt"].bev)
        z["vs_fest"] = EV.abstand(z["bev"], stoch["bf"])
    st.dataframe(
        [{"Regel": z["name"], "Vorschrift": z["par"], "Bodenwert (EUR/ha)": round(z["bev"], 1), "zum Optimum (EUR/ha)": round(z["vs_opt"], 1), "zu Faustmann (EUR/ha)": round(z["vs_fest"], 1)} for z in zeilen],
        hide_index=True, width="stretch",
    )
    st.caption("Rangfolge nach Bodenwert: " + EV.rangfolge({z["name"]: z["bev"] for z in zeilen}) + ". Abstände in EUR/ha statt Prozent, weil der Bodenwert gegen 0 gehen kann.")
    st.plotly_chart(V.fig_regeln(zeilen), width="stretch", key="regeln_chart")

    gewinn = stoch["opt"].bev - stoch["bf"]
    verlust = stoch["bf"] - stoch["bm"]
    g1, g2 = st.columns(2)
    g1.metric("Gewinn der Preisflexibilität (EUR/ha)", f"{EV.eur(gewinn)}", delta="Optimum minus Faustmann", delta_color="off")
    g2.metric("Verlust der MSY-Regel (EUR/ha)", f"{EV.eur(verlust)}", delta="Faustmann minus MSY", delta_color="off")

    st.markdown("### Wann erntet das Optimum? Niedrigster Preis je Alter")
    if p.sigma > 0:
        st.plotly_chart(V.fig_schwelle(m.preise, stoch["opt"].schwelle, regel, stoch["tf"], p.preis), width="stretch", key="schwelle_chart")
        o = stoch["opt"]
        st.caption(
            f"Die Schwelle des Optimums ist von Alter zu Alter nie steigend, ab dem ersten Alter mit Hieb bis zum Zwangshieb (stufig im 21-Punkte-Preisgitter): {'ja' if o.fallend else 'NEIN'}; "
            f"die gestrichelte Preisregel hat eine starre Schwelle. In jedem Alter erntet das Optimum bei allen Preisen ab einer Schwelle: "
            f"{'ja' if o.zusammenhaengend else 'NEIN'}. Zustände mit einem Abstand von Hieb und Warten unter 0,001 EUR: {EV.gleichstand_text(o.gleichstaende, o.zustaende)} (bei Gleichstand würde gewartet)."
        )
    else:
        st.info("Bei σ = 0 gibt es nur einen Preis: das Optimum ist die Faustmann-Umtriebszeit, es gibt keine Preisschwelle.")

    st.markdown("### Eine Zufallsbahn, vier Regeln")
    preise_b, idx_b, vl = _verlaeufe(p, int(seed), None if regel is None else (regel.tmin, regel.k, regel.tmax))
    st.plotly_chart(V.fig_bahn(preise_b, idx_b, vl, None if regel is None else regel.schwelle(m)), width="stretch", key="bahn_chart")
    st.dataframe(
        [{"Regel": n, "Hiebe in 150 Jahren": len(v.hiebe), "davon Notverkäufe": len(v.notverkaeufe),
          "mittleres Erntealter (Jahre)": None if v.mittleres_erntealter is None else round(v.mittleres_erntealter, 1)} for n, v in vl.items()],
        hide_index=True, width="stretch",
    )
    st.caption(f"Zufalls-Seed {int(seed)}: dieselbe Preisbahn und dieselben Zufallszahlen der Kalamität für alle Regeln; eine einzelne Bahn sagt über den Wert einer Regel wenig, deshalb steht der Wert in der exakten Tabelle oben.")

    st.markdown("### Gewinn der Preisflexibilität über σ: wo kippt die Rangfolge?")
    sig_g, gew_g, star, verl_g = _sigma(mod(p, sigma=0.0))
    st.plotly_chart(V.fig_gewinn_sigma(sig_g, gew_g, verl_g, star, float(sigma)), width="stretch", key="sigma_chart")
    st.caption("Alle Zahlen dieser Kurve: φ und Zins wie eingestellt, exakt ausgewertet, Gitter 0,02 und Bisektion im Gitterintervall für σ*.")
    tab_sigma = G.sigma_stern_laden()
    if tab_sigma:
        st.markdown("**Schwelle σ* nach Persistenz φ und Zins** (Pflanzung 4000, Nettoerlös 40, kein Kalamitätsrisiko; vorgerechnet in `data/sigma_stern.csv`)")
        st.dataframe(
            [{"φ": r["phi"], "Zins (%)": r["zins"], "Verlust der MSY-Regel (EUR/ha)": round(r["verlust"], 1),
              "σ*": None if r["sigma_stern"] is None else round(r["sigma_stern"], 3), "Gewinn bei σ 0,3 (EUR/ha)": round(r["gewinn_030"], 1)} for r in tab_sigma],
            hide_index=True, width="stretch",
            column_config={"σ*": st.column_config.NumberColumn(format="%.3f", help="leer: bis σ = 0,5 nicht erreicht")},
        )

# --- Bonität -------------------------------------------------------------------------------------------------------------------------------
with tab_bon:
    st.markdown("## 🌳 Drei Bonitäten: Zieldurchmesser gegen eine feste Zeit")
    st.warning("Annahme dieses Tabs: Durchmesser und Volumen wachsen in derselben effektiven Zeit g·Alter. Die Zieldurchmesser-Regel ist dann fast eine Regel in effektivem Alter; dass sie nahe an Faustmann liegt, ist teilweise eingebaut und ohne Daten nicht belastbar.")
    bon = _bonitaet(p_det)
    ref = bon["ref"]
    names = ["Eine feste Zeit für alle", "Zieldurchmesser (ein A*)", "MSY je Bonität", "MSY der mittleren Bonität für alle"]
    werte = [bon["fest"], bon["ziel"], bon["msy_g"], bon["msy_1"]]
    st.dataframe(
        [{"Regel": "Faustmann je Bonität (Referenz)", "Umtriebszeiten (g 0,8 / 1,0 / 1,2)": " / ".join(str(bon["tf"][g]) for g in C.BONITAET_G), "Bodenwert, Mittel (EUR/ha)": round(ref, 1),
          "zur Referenz (EUR/ha)": 0.0},
         {"Regel": names[0], "Umtriebszeiten (g 0,8 / 1,0 / 1,2)": " / ".join([str(bon["t_all"])] * 3), "Bodenwert, Mittel (EUR/ha)": round(werte[0], 1), "zur Referenz (EUR/ha)": round(werte[0] - ref, 1)},
         {"Regel": f"{names[1]}: A* = {EV.zahl(bon['a_star'], 2)} Jahre", "Umtriebszeiten (g 0,8 / 1,0 / 1,2)": " / ".join(str(bon["t_ziel"][g]) for g in C.BONITAET_G),
          "Bodenwert, Mittel (EUR/ha)": round(werte[1], 1), "zur Referenz (EUR/ha)": round(werte[1] - ref, 1)},
         {"Regel": names[2], "Umtriebszeiten (g 0,8 / 1,0 / 1,2)": " / ".join(str(bon["t_msy_g"][g]) for g in C.BONITAET_G), "Bodenwert, Mittel (EUR/ha)": round(werte[2], 1),
          "zur Referenz (EUR/ha)": round(werte[2] - ref, 1)},
         {"Regel": names[3], "Umtriebszeiten (g 0,8 / 1,0 / 1,2)": " / ".join([str(bon["t_msy_g"][1.0])] * 3), "Bodenwert, Mittel (EUR/ha)": round(werte[3], 1),
          "zur Referenz (EUR/ha)": round(werte[3] - ref, 1)}],
        hide_index=True, width="stretch",
    )
    st.plotly_chart(V.fig_bonitaet(names, [w - ref for w in werte]), width="stretch", key="bonitaet_chart")
    kind_b, text_b = EV.verdict_bonitaet(werte[0] - ref, werte[1] - ref, werte[2] - ref)
    getattr(st, kind_b)(text_b)

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------
st.markdown("## ❓ Kernfragen")
st.markdown("**1. Was kostet die MSY-Regel gegenüber Faustmann?** (konstanter Preis)")
k1, t1 = EV.verdict_msy(det.tf, det.tm, det.bev_f, det.bev_m, det.be_f, det.be_m)
getattr(st, k1)(t1)
st.markdown("**2. Was ist mehr wert: das Warten auf den Preis oder die richtige Umtriebszeit?**")
k2, t2 = EV.verdict_hook(gewinn, verlust, float(sigma), float(phi), float(zins))
getattr(st, k2)(t2)
st.markdown("**3. Bei welcher Preisstreuung kippt die Rangfolge?**")
sig_g, gew_g, star, verl_g = _sigma(mod(p, sigma=0.0))
k3, t3 = EV.verdict_sigma_stern(star, verl_g, float(sigma), float(phi), float(zins))
getattr(st, k3)(t3)

pdf_bytes = build_pdf(
    {"Zins %": zins, "Pflanzkosten": pflanz, "Nettoerlös": preis, "Sortimentsprämie": sort, "Kalamitätsrisiko": risiko, "sigma": sigma, "phi": phi, "Zufalls-Seed": int(seed)},
    [t1],
    [{"name": z["name"], "bev": z["bev"], "vs_opt": z["vs_opt"]} for z in zeilen],
    [t1, t2, t3],
)
st.download_button("📄 Ergebnis als PDF", data=pdf_bytes, file_name="forst-umtriebszeit.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")

# --- Expander ------------------------------------------------------------------------------------------------------------------------------
with st.expander("🔧 Wie wir das erreichen – vollständiger Methodenvergleich"):
    st.markdown(TX.METHODS_MARKDOWN)

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/forst-umtriebszeit-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind das Wachstum eines Bestands, der Preis des Holzes (konstant oder zufällig), Pflanzkosten und Zins; gesucht ist, wann der Bestand geerntet wird, damit der Wert des Bodens (BEV) so groß wie möglich ist, wenn danach unendlich oft derselbe Umtrieb folgt.

| Symbol | Bedeutung |
|---|---|
| $a$, $T$ | Alter des Bestands in Jahren; feste Umtriebszeit |
| $V(a)$, $V_{\max}$, $k$, $p$, $g$ | Volumen in m³/ha, Sättigungsvolumen, Wachstumsrate, Formexponent, Bonitätsfaktor (Zeitskalierung) |
| $q_i$, $q$ | Nettoerlös je m³ im Preiszustand $i$ (Holzpreis minus Ernte); $q$ ohne Index: der konstante Nettoerlös (nicht der Sortimentsfaktor $Q$) |
| $Q(a)$, $s$ | Sortimentsfaktor und Sortimentsprämie |
| $c$, $c_h$ | Pflanzkosten je Hektar, feste Hiebkosten (in der App 0) |
| $r$, $\delta$ | Zins und Abzinsungsfaktor $1/(1+r)$ |
| $\eta(a)$, $\rho$, $\nu$ | Kalamitätsrate im Jahr von Alter $a$ auf $a+1$, Risikoregler, Anteil des Holzwerts beim Notverkauf ($\nu = 0{,}5$) |
| $P_{ij}$, $\pi_i$ | Übergangswahrscheinlichkeit des Preiszustands und stationäre Verteilung |
| $\sigma$, $\varphi$ | Standardabweichung des Log-Preises, Autokorrelation des Log-Preises |
| $H(a,i)$, $N(a,i)$ | Erlös beim Hieb, Erlös beim Notverkauf (nicht auf 0 begrenzt; mit $c_h>0$ kann er negativ werden) |
| $S(a,i)$, $R(i)$ | Wert eines Bestands vom Alter $a$ im Preiszustand $i$; Wert des unbestockten Bodens vor der Pflanzung |
| $a_{\min}$, $a_{\max}$ | frühestes erlaubtes Hiebalter (in der App 0) und Zwangshiebalter (150) |
| $t_{\min}$, $\kappa$, $t_{\max}$ | Parameter der Preisregel: frühestes Hiebalter, Preisindex der Schwelle ($q_\kappa$ ist die Preis-Schwelle), spätestes Hiebalter |
| $\Delta_{\mathrm{flex}}$, $L_{\mathrm{MSY}}$, $\sigma^{*}$ | Gewinn der Preisflexibilität, Verlust der MSY-Regel, Kreuzungs-Streuung |
| $A^{*}$ | Zieldurchmesser, gemessen als effektives Alter $g\,a$ (Tab Bonität) |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/forst-umtriebszeit-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(r"""$$
V(a)=V_{\max}\,\bigl(1-e^{-k\,g\,a}\bigr)^{p}
$$""")
    st.markdown(r"""$$
Q(a)=1-s+\frac{s}{1+e^{-(a-50)/10}},\qquad H(a,i)=q_i\,Q(a)\,V(a)-c_h,\qquad N(a,i)=\nu\,q_i\,Q(a)\,V(a)-c_h
$$""")
    st.markdown(r"""$$
\eta(a)=\rho\,\bigl(0{,}002+0{,}02\,(a/100)^{3}\bigr)
$$""")
    st.markdown(r"""$$
\mathrm{BEV}(T)=\frac{q\,Q(T)\,V(T)-c_h-c\,(1+r)^{T}}{(1+r)^{T}-1},\qquad T_F=\arg\max_T \mathrm{BEV}(T),\qquad T_{\mathrm{MSY}}=\arg\max_a \frac{V(a)}{a}
$$""")
    st.markdown(r"""$$
S(a,i)=\max\Bigl\{\,H(a,i)+R(i),\ \ \delta\sum_{j}P_{ij}\bigl[(1-\eta(a))\,S(a+1,j)+\eta(a)\,\bigl(N(a+1,j)+R(j)\bigr)\bigr]\Bigr\}\quad (a_{\min}\le a<a_{\max})
$$""")
    st.markdown(r"""$$
S(a_{\max},i)=H(a_{\max},i)+R(i),\qquad S(a,i)=\delta\sum_{j}P_{ij}\bigl[(1-\eta(a))\,S(a+1,j)+\eta(a)\,\bigl(N(a+1,j)+R(j)\bigr)\bigr]\quad (a<a_{\min})
$$""")
    st.markdown(r"""$$
R(i)=-c+\delta\sum_{j}P_{ij}\bigl[(1-\eta(0))\,S(1,j)+\eta(0)\,\bigl(N(1,j)+R(j)\bigr)\bigr],\qquad \mathrm{BEV}=\sum_i \pi_i\,R(i)
$$""")
    st.markdown(r"""$$
\text{Preisregel: ernte im Alter } a \text{ im Zustand } i \iff (a\ge t_{\min}\ \wedge\ i\ge \kappa)\ \vee\ a\ge t_{\max}
$$""")
    st.markdown(r"""$$
\Delta_{\mathrm{flex}}(\sigma)=\mathrm{BEV}_{\mathrm{Opt}}(\sigma)-\mathrm{BEV}_{F},\qquad L_{\mathrm{MSY}}=\mathrm{BEV}_{F}-\mathrm{BEV}_{\mathrm{MSY}},\qquad \sigma^{*}=\min\{\sigma:\ \Delta_{\mathrm{flex}}(\sigma)\ge L_{\mathrm{MSY}}\}
$$""")
    st.markdown(r"""$$
\text{Zieldurchmesser (Tab Bonität): ernte, sobald } g\,a\ge A^{*},\ \text{also im Alter } \lceil A^{*}/g\rceil
$$""")
    st.markdown(TX.MATH_NOTE)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(TX.ASSUMPTIONS_MARKDOWN)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
