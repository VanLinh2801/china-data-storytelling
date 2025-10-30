from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "China.cleaned.csv"
FALLBACK_DATASET = ROOT / "China.csv"

st.set_page_config(page_title="China 2000–2020: Overview", layout="wide", page_icon="🧭")

# Styles (Google Fonts + custom CSS)
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
      :root { --red:#d33f49; --gray:#6c757d; --bg:#f8f9fa; --ink:#222; --card:#ffffff; }
      html, body, .main { background: var(--bg); font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Liberation Sans', sans-serif; }
      h1,h2,h3 { color: var(--ink); letter-spacing: -0.3px; }
      .hero { position:relative; padding: 7rem 6rem; border-radius: 24px; background: radial-gradient(80% 120% at 10% 10%, #fff, #f6f6f7); overflow:hidden; }
      .hero::after { content:""; position:absolute; inset:0; background: url('https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/People%27s_Republic_of_China_%28orthographic_projection%29.svg/512px-People%27s_Republic_of_China_%28orthographic_projection%29.svg.png') center/contain no-repeat; opacity:0.06; filter: grayscale(100%); animation: fadeIn 2s ease forwards; }
      .subtitle { color: var(--gray); font-size: 1.15rem; max-width: 900px; }
      .scroll-cue { margin-top: 1.5rem; color: var(--gray); animation: float 2s ease-in-out infinite; }
      @keyframes fadeIn { from { opacity:0; } to { opacity:0.06; } }
      @keyframes float { 0% { transform: translateY(0); } 50% { transform: translateY(6px); } 100% { transform: translateY(0); } }
      .timeline { display:flex; gap: 1.25rem; overflow-x:auto; scroll-snap-type: x mandatory; padding: 1rem 0; }
      .milestone { min-width: 280px; scroll-snap-align: start; background: var(--card); border: 1px solid #eee; border-radius: 16px; padding: 1rem 1.25rem; box-shadow: 0 4px 16px rgba(0,0,0,0.03); }
      .cards { display:grid; grid-template-columns: repeat(5, minmax(160px, 1fr)); gap: 1rem; }
      .card { background: var(--card); border-radius: 16px; padding: 1.25rem; border:1px solid #eee; box-shadow: 0 10px 24px rgba(0,0,0,0.04); transition: transform .2s ease, box-shadow .2s ease; cursor: pointer; }
      .card:hover { transform: translateY(-4px) scale(1.01); box-shadow: 0 16px 36px rgba(0,0,0,0.07); }
      .indicators { display:grid; grid-template-columns: repeat(4, minmax(200px, 1fr)); gap: 1rem; }
      .foot { color: var(--gray); font-size: 0.95rem; }
      .section { margin: 3rem 0 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_dataset() -> pd.DataFrame:
    path = DEFAULT_DATASET if DEFAULT_DATASET.exists() else FALLBACK_DATASET
    df = pd.read_csv(path)
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    return df


def indicator_value(df: pd.DataFrame, code: str, year: int) -> float | None:
    col = "Indicator code" if "Indicator code" in df.columns else "Indicator Code"
    sub = df[(df[col] == code)]
    if sub.empty or str(year) not in df.columns:
        return None
    return float(sub[str(year)].values[0]) if pd.notna(sub[str(year)].values[0]) else None


df = read_dataset()

# Hero
st.markdown(
    """
    <div class="hero">
      <h1>China 2000–2020: Two Decades of Transformation</h1>
      <p class="subtitle">From rapid industrialization to digital innovation — a journey of growth, challenge, and renewal.</p>
      <div class="scroll-cue">↓ Explore</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Timeline
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("Timeline of Change")
st.markdown(
    """
    <div class="timeline">
      <div class="milestone"><h4>2000–2005</h4><b>WTO entry & industrial expansion</b><br/><span class="foot">China joined the WTO in 2001, catalyzing exports and manufacturing.</span></div>
      <div class="milestone"><h4>2006–2010</h4><b>Urbanization surge</b><br/><span class="foot">Megacities expanded and infrastructure boomed across regions.</span></div>
      <div class="milestone"><h4>2011–2015</h4><b>Pollution crisis & green policy</b><br/><span class="foot">Tighter standards and early renewables push began curbing pollution.</span></div>
      <div class="milestone"><h4>2016–2020</h4><b>Digital & innovation era</b><br/><span class="foot">Internet and tech platforms scaled, shifting value toward services.</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Summary indicator cards
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("At a Glance (2000 → 2020)")

gdp2000 = indicator_value(df, "NY.GDP.MKTP.CD", 2000)
gdp2020 = indicator_value(df, "NY.GDP.MKTP.CD", 2020)
urb2000 = indicator_value(df, "SP.URB.TOTL.IN.ZS", 2000)
urb2020 = indicator_value(df, "SP.URB.TOTL.IN.ZS", 2020)
ren2000 = indicator_value(df, "EG.ELC.RNEW.ZS", 2000)
ren2020 = indicator_value(df, "EG.ELC.RNEW.ZS", 2020)
net2000 = indicator_value(df, "IT.NET.USER.ZS", 2000)
net2020 = indicator_value(df, "IT.NET.USER.ZS", 2020)

# Fallbacks if data missing
def _fallback(val, alt):
    return alt if val is None else val

gdp2000 = _fallback(gdp2000, 1.2e12)
gdp2020 = _fallback(gdp2020, 14e12)
urb2000 = _fallback(urb2000, 36)
urb2020 = _fallback(urb2020, 61)
ren2000 = _fallback(ren2000, 8)
ren2020 = _fallback(ren2020, 28)
net2000 = _fallback(net2000, 2)
net2020 = _fallback(net2020, 65)

col_ic = st.container()
with col_ic:
    st.markdown("<div class='indicators'>", unsafe_allow_html=True)
    figA = go.Figure(go.Indicator(mode="number+delta", value=gdp2020/1e12, number={"suffix":" T"}, delta={"reference": gdp2000/1e12, "relative": False, "valueformat":".1f"}, title={"text":"GDP (USD Trillions)"}))
    figB = go.Figure(go.Indicator(mode="number+delta", value=urb2020, number={"suffix":" %"}, delta={"reference": urb2000}, title={"text":"Urban population"}))
    figC = go.Figure(go.Indicator(mode="number+delta", value=ren2020, number={"suffix":" %"}, delta={"reference": ren2000}, title={"text":"Renewable electricity"}))
    figD = go.Figure(go.Indicator(mode="number+delta", value=net2020, number={"suffix":" %"}, delta={"reference": net2000}, title={"text":"Internet users"}))
    st.plotly_chart(figA, use_container_width=True)
    st.plotly_chart(figB, use_container_width=True)
    st.plotly_chart(figC, use_container_width=True)
    st.plotly_chart(figD, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# Optional mini area chart: GDP & Urbanization
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("The Big Picture")
try:
    gdp_series = df[df["Indicator code"] == "NY.GDP.MKTP.CD"][['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020']].T
    gdp_series.index = gdp_series.index.astype(int)
    gdp_series.columns = ["GDP"]
    urb_series = df[df["Indicator code"] == "SP.URB.TOTL.IN.ZS"][gdp_series.index.astype(str)].T
    urb_series.index = urb_series.index.astype(int)
    urb_series.columns = ["Urban_%"]
    mini = pd.concat([gdp_series/1e12, urb_series], axis=1).reset_index().rename(columns={"index":"Year"})
    figMini = go.Figure()
    figMini.add_trace(go.Scatter(x=mini["Year"], y=mini["GDP"], fill="tozeroy", name="GDP (T USD)", line=dict(color="#d33f49")))
    figMini.add_trace(go.Scatter(x=mini["Year"], y=mini["Urban_%"], name="Urban (%)", yaxis="y2", line=dict(color="#6c757d")))
    figMini.update_layout(xaxis_title="Year", yaxis=dict(title="GDP (T)"), yaxis2=dict(title="Urban (%)", overlaying="y", side="right"), height=420, margin=dict(t=10, b=10, l=10, r=10), legend=dict(orientation="h"))
    st.plotly_chart(figMini, use_container_width=True)
except Exception:
    pass


# Navigation cards
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("Chapters")
col1, col2, col3, col4 = st.columns(4)
if col1.button("🏭 Economy\nFrom fields to factories", use_container_width=True):
    st.switch_page("pages/1_📈_Economy.py")
if col2.button("🌆 Urbanization\nThe rise of mega cities", use_container_width=True):
    st.switch_page("pages/2_🏙️_Urbanization_&_Population.py")
if col3.button("🌱 Environment\nThe cost of growth", use_container_width=True):
    st.switch_page("pages/3_🌿_Environment_&_Energy.py")
if col4.button("💡 Innovation & Society\nFrom manufacturing to mind", use_container_width=True):
    st.switch_page("pages/5_🎓_Innovation_&_Society.py")


# Footer
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="foot">
      <em>“This project visualizes how China evolved over two decades — balancing growth, society, and sustainability.”</em>
      <br/>
      Created by VanLinh, 2025 | Data Sources: World Bank, UN Data
    </div>
    """,
    unsafe_allow_html=True,
)


