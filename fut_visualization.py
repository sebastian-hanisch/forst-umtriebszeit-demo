"""Plotly-Figuren (alle Achsen fest, kein Zoomen: Touch-Scrollen bleibt nicht am Diagramm hängen).

Auf Kategorieachsen (Balken) stehen Annotationen am Positionsindex, nicht am Namen."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import fut_constants as C

REGEL_FARBEN = {"Faustmann (fest)": C.COLOR_FEST, "MSY (fest)": C.COLOR_MSY, "Preisregel (abgestimmt)": C.COLOR_REGEL, "Optimum (DP)": C.COLOR_OPT}


def _fixed(fig, height=340, legend=True):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=60), legend=dict(orientation="h", y=-0.3) if legend else None, showlegend=legend)
    return fig


def fig_zuwachs(a, V, lauf, schnitt, tm, tf, amax_plot=130):
    """Volumen (oben) sowie laufender und durchschnittlicher Zuwachs (unten); senkrecht: Faustmann- und MSY-Umtriebszeit."""
    sel = a <= amax_plot
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.16, row_heights=[0.5, 0.5],
                        subplot_titles=("Volumen V(a) in m³ je Hektar", "Zuwachs in m³ je Hektar und Jahr"))
    fig.add_trace(go.Scatter(x=a[sel], y=V[sel], mode="lines", name="Volumen", line=dict(color=C.COLOR_OPT)), row=1, col=1)
    fig.add_trace(go.Scatter(x=a[sel], y=lauf[sel], mode="lines", name="laufender Zuwachs", line=dict(color=C.COLOR_REGEL)), row=2, col=1)
    fig.add_trace(go.Scatter(x=a[sel], y=schnitt[sel], mode="lines", name="durchschnittlicher Zuwachs V(a)/a", line=dict(color=C.COLOR_MSY)), row=2, col=1)
    for t, name, col in ((tm, f"MSY {tm}", C.COLOR_MSY), (tf, f"Faustmann {tf}", C.COLOR_OPT)):
        if t <= amax_plot:
            for row in (1, 2):
                fig.add_vline(x=t, line=dict(color=col, dash="dot"), row=row, col=1)
    ymax = float(max(lauf[sel].max(), schnitt[sel].max())) * 1.15
    fig.update_yaxes(range=[0, float(V[sel].max()) * 1.1], row=1, col=1)
    fig.update_yaxes(range=[0, ymax], row=2, col=1)
    fig.update_xaxes(range=[0, amax_plot], title_text="Alter in Jahren", row=2, col=1)
    fig.add_annotation(x=min(tm, amax_plot), y=ymax * 0.95, xref="x2", yref="y2", text=f"MSY {tm}", showarrow=False, xanchor="left", font=dict(color=C.COLOR_MSY))
    if tf <= amax_plot:
        fig.add_annotation(x=tf, y=ymax * 0.8, xref="x2", yref="y2", text=f"Faustmann {tf}", showarrow=False, xanchor="right", font=dict(color=C.COLOR_OPT))
    return _fixed(fig, 520)


def fig_bev(ts, bev, tf, bev_f, tm, bev_m, t_von=20):
    """Bodenertragswert je fester Umtriebszeit T mit der Faustmann-Marke (Maximum) und der MSY-Marke."""
    sel = ts >= t_von
    fig = go.Figure(go.Scatter(x=ts[sel], y=bev[sel], mode="lines", name="BEV je Umtriebszeit", line=dict(color=C.COLOR_OPT)))
    fig.add_trace(go.Scatter(x=[tf], y=[bev_f], mode="markers", name=f"Faustmann: {tf} Jahre", marker=dict(size=12, color=C.COLOR_OPT, symbol="star")))
    fig.add_trace(go.Scatter(x=[tm], y=[bev_m], mode="markers", name=f"MSY: {tm} Jahre", marker=dict(size=11, color=C.COLOR_MSY, symbol="diamond")))
    fig.add_hline(y=0, line=dict(color="rgba(128,128,128,0.6)", width=1))
    lo, hi = float(bev[sel].min()), float(bev[sel].max())
    pad = 0.08 * (hi - lo if hi > lo else 1.0)
    fig.update_yaxes(title="Bodenertragswert in EUR/ha", range=[lo - pad, hi + pad])
    fig.update_xaxes(title="feste Umtriebszeit T in Jahren", range=[t_von, float(ts.max())])
    return _fixed(fig, 340)


def fig_bahn(preise, bahn_idx, verlaeufe, schwelle_regel=None):
    """Oben: Preisbahn (EUR/m³) mit Hieben der Preisregel und des Optimums; unten: Alter des Bestands je Regel auf derselben Bahn."""
    jahre = list(range(len(bahn_idx)))
    p_pfad = [float(preise[i]) for i in bahn_idx]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14, row_heights=[0.4, 0.6],
                        subplot_titles=("Preis (Nettoerlös in EUR/m³)", "Alter des Bestands in Jahren (Sprung auf 0 = Hieb)"))
    fig.add_trace(go.Scatter(x=jahre, y=p_pfad, mode="lines", name="Preis", line=dict(color=C.COLOR_PREIS, width=1.5)), row=1, col=1)
    for name, symbol, size in (("Preisregel (abgestimmt)", "circle-open", 11), ("Optimum (DP)", "diamond", 8)):
        if name not in verlaeufe:       # die Preisregel fehlt, solange sie für diese Einstellung nicht abgestimmt ist
            continue
        v = verlaeufe[name]
        ys = [p_pfad[t] for t in v.hiebe]
        fig.add_trace(go.Scatter(x=list(v.hiebe), y=ys, mode="markers", name=f"Hiebe: {name}", marker=dict(symbol=symbol, size=size, color=REGEL_FARBEN[name], line=dict(width=2))),
                      row=1, col=1)
    if schwelle_regel is not None:
        fig.add_hline(y=schwelle_regel, line=dict(color=C.COLOR_REGEL, dash="dot"), row=1, col=1)
    for name, v in verlaeufe.items():
        fig.add_trace(go.Scatter(x=jahre, y=list(v.alter), mode="lines", name=name, line=dict(color=REGEL_FARBEN[name], width=2.2 if "Optimum" in name else 1.4,
                                                                                         dash="solid" if name != "MSY (fest)" else "dash")), row=2, col=1)
    ymax = max(p_pfad) * 1.1
    fig.update_yaxes(range=[0, ymax], row=1, col=1)
    fig.update_yaxes(range=[0, 152], row=2, col=1)
    fig.update_xaxes(range=[0, len(bahn_idx) - 1], title_text="Jahr", row=2, col=1)
    return _fixed(fig, 520)


def fig_schwelle(preise, opt_schwelle, regel, tf, mittel):
    """Niedrigster Preis, bei dem geerntet wird, über dem Alter: Optimum (DP) gegen die starre Preisregel."""
    ages = [a for a, k in enumerate(opt_schwelle) if a >= 1 and k >= 0]
    y_opt = [float(preise[opt_schwelle[a]]) for a in ages]
    fig = go.Figure(go.Scatter(x=ages, y=y_opt, mode="lines", line=dict(shape="hv", color=C.COLOR_OPT, width=2.5), name="Optimum: niedrigster Erntepreis"))
    if regel is not None:
        xs = [regel.tmin, regel.tmax, regel.tmax, max(ages)]
        ys = [float(preise[regel.k])] * 2 + [float(preise[0])] * 2
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(shape="hv", color=C.COLOR_REGEL, dash="dash"), name="Preisregel (Schwelle, ab t_max jeder Preis)"))
    fig.add_hline(y=mittel, line=dict(color=C.COLOR_PREIS, dash="dot"))
    fig.add_vline(x=tf, line=dict(color=C.COLOR_FEST, dash="dot"))
    fig.add_annotation(x=1, y=mittel, text="mittlerer Preis", showarrow=False, xanchor="left", yanchor="bottom", font=dict(color=C.COLOR_PREIS))
    fig.add_annotation(x=tf, y=float(preise[-1]) * 0.9, text=f"Faustmann {tf}", showarrow=False, xanchor="left", font=dict(color=C.COLOR_FEST))
    hi = float(max(max(y_opt), preise[regel.k] if regel is not None else 0.0, mittel)) * 1.15
    fig.update_yaxes(title="Preis in EUR/m³", range=[0, hi])
    fig.update_xaxes(title="Alter in Jahren", range=[0, max(ages)])
    return _fixed(fig, 340)


def fig_gewinn_sigma(sigmas, gewinn, verlust, sigma_stern, sigma_jetzt):
    """Gewinn der Preisflexibilität (Optimum minus feste Faustmann-Zeit) über σ; waagerecht: Verlust der MSY-Regel."""
    fig = go.Figure(go.Scatter(x=list(sigmas), y=list(gewinn), mode="lines+markers", name="Gewinn der Preisflexibilität", line=dict(color=C.COLOR_OPT)))
    fig.add_hline(y=verlust, line=dict(color=C.COLOR_MSY, dash="dash"))
    fig.add_annotation(x=float(sigmas[0]), y=verlust, text="Verlust der MSY-Regel", showarrow=False, xanchor="left", yanchor="bottom", font=dict(color=C.COLOR_MSY))
    fig.add_vline(x=sigma_jetzt, line=dict(color=C.COLOR_PREIS, dash="dot"))
    fig.add_annotation(x=sigma_jetzt, y=0, text="Ihr σ", showarrow=False, xanchor="left", yanchor="bottom", font=dict(color=C.COLOR_PREIS))
    if sigma_stern is not None and sigma_stern > 0:
        fig.add_trace(go.Scatter(x=[sigma_stern], y=[verlust], mode="markers", name=f"σ* = {sigma_stern:.3f}".replace(".", ","),
                                 marker=dict(size=13, color=C.COLOR_MSY, symbol="x")))
    top = float(max(max(gewinn), verlust)) * 1.1 if len(gewinn) else 1.0
    fig.update_yaxes(title="EUR/ha", range=[0, top if top > 0 else 1.0])
    fig.update_xaxes(title="σ: Streuung des Log-Preises", range=[float(sigmas[0]), float(sigmas[-1])])
    return _fixed(fig, 340)


def fig_regeln(zeilen):
    """Bodenertragswert je Regel (Balken), Annotation am Positionsindex der Kategorie."""
    namen = [z["name"] for z in zeilen]
    werte = [z["bev"] for z in zeilen]
    fig = go.Figure(go.Bar(x=namen, y=werte, marker_color=[REGEL_FARBEN.get(n, "#999") for n in namen], showlegend=False))
    lo, hi = min(0.0, min(werte)), max(0.0, max(werte))
    pad = 0.12 * (hi - lo if hi > lo else 1.0)
    for i, w in enumerate(werte):
        fig.add_annotation(x=i, y=w, text=f"{round(w):,}".replace(",", "."), showarrow=False, yshift=12 if w >= 0 else -12)
    fig.update_yaxes(title="Bodenertragswert in EUR/ha", range=[lo - pad, hi + pad])
    return _fixed(fig, 300, legend=False)


def fig_bonitaet(namen, luecken):
    """Abstand zur je Bonität richtigen Faustmann-Umtriebszeit in EUR/ha (Mittel der drei Bonitäten), Balken, Annotation am Positionsindex."""
    fig = go.Figure(go.Bar(x=namen, y=luecken, marker_color=[C.COLOR_FEST, C.COLOR_REGEL, C.COLOR_MSY, "#e08f87"], showlegend=False))
    lo = min(luecken)
    pad = 0.15 * (abs(lo) if lo < 0 else 1.0)
    for i, w in enumerate(luecken):
        fig.add_annotation(x=i, y=w, text=f"{round(w):,}".replace(",", ".").replace("-", "−"), showarrow=False, yshift=-12)
    fig.update_yaxes(title="Abstand zu Faustmann je Bonität in EUR/ha", range=[lo - pad - 20, pad])
    return _fixed(fig, 300, legend=False)
