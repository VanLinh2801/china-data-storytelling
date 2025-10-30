from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ---------------------------
# Config & Constants
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "China.cleaned.csv"
FALLBACK_DATASET = PROJECT_ROOT / "China.csv"

# Comprehensive indicators for innovation and society transformation
INDICATORS = {
    # 1. Innovation & Knowledge Economy
    "RND_PCT_GDP": "GB.XPD.RSDV.GD.ZS",
    "RESEARCHERS_PER_MILL": "SP.POP.SCIE.RD.P6", 
    "PATENTS_RESIDENTS": "IP.PAT.RESD",
    "SCIENTIFIC_ARTICLES": "IP.JRN.ARTC.SC",
    "INTERNET_USERS_PCT": "IT.NET.USER.ZS",
    "BROADBAND_SUBSCRIBERS": "IT.NET.BBND.P2",
    "MOBILE_SUB_PER100": "IT.CEL.SETS.P2",
    "ELEC_USE_PER_CAP": "EG.USE.ELEC.KH.PC",
    
    # 2. Education & Human Capital
    "TERTIARY_ENROLLMENT": "SE.TER.ENRR",
    "GDP_PER_EMPLOYEE": "SL.GDP.PCAP.EM.KD",
    
    # 3. Governance & Inclusion
    "URBAN_POP_PCT": "SP.URB.TOTL.IN.ZS",
    "WOMEN_BUSINESS_LAW": "SG.LAW.INDX",
    "GOV_EFFECTIVENESS": "GE.EST",
    "RULE_OF_LAW": "RL.EST",
    "CONTROL_CORRUPTION": "CC.EST",
    "REGULATORY_QUALITY": "RQ.EST",
    "GENDER_PARLIAMENT": "SG.GEN.PARL.ZS",
    "POPULATION_DENSITY": "EN.POP.DNST",
    "NET_MIGRATION": "SM.POP.NETM",
    
    # Economic indicators for correlation
    "GDP_GROWTH": "NY.GDP.MKTP.KD.ZG",
    "GDP_PER_CAPITA": "NY.GDP.PCAP.CD",
}

YEAR_COLUMNS = [str(y) for y in range(2000, 2021)]


# ---------------------------
# Data Utilities
# ---------------------------
def _read_dataset() -> pd.DataFrame:
    if DEFAULT_DATASET.exists():
        df = pd.read_csv(DEFAULT_DATASET)
    else:
        df = pd.read_csv(FALLBACK_DATASET)
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    return df


def _filter_indicator(df: pd.DataFrame, indicator_code: str) -> pd.DataFrame:
    col_name = "Indicator code" if "Indicator code" in df.columns else "Indicator Code"
    return df[df[col_name] == indicator_code].copy()


def _melt_years(df: pd.DataFrame) -> pd.DataFrame:
    tidy = df.melt(
        id_vars=["Country Name", "Country Code", "Indicator", "Indicator code"],
        value_vars=[c for c in YEAR_COLUMNS if c in df.columns],
        var_name="Year",
        value_name="Value",
    )
    tidy["Year"] = tidy["Year"].astype(int)
    tidy = tidy.dropna(subset=["Value"]).reset_index(drop=True)
    return tidy


def load_indicators(codes: Iterable[str]) -> Dict[str, pd.DataFrame]:
    df = _read_dataset()
    out: Dict[str, pd.DataFrame] = {}
    for code in codes:
        part = _filter_indicator(df, code)
        out[code] = _melt_years(part)[["Year", "Value"]] if not part.empty else pd.DataFrame(columns=["Year", "Value"])
    return out


def decade_filter(df: pd.DataFrame, period: str | None) -> pd.DataFrame:
    if period in (None, "All"):
        return df
    if period == "2000s":
        return df[(df["Year"] >= 2000) & (df["Year"] <= 2009)]
    if period == "2010s":
        return df[(df["Year"] >= 2010) & (df["Year"] <= 2020)]
    return df


def minmax_normalize(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.zeros(len(s)), index=series.index)
    return (s - mn) / (mx - mn) * 100.0


def merge_on_year_named(frames_with_names: List[tuple[pd.DataFrame, str]]) -> pd.DataFrame:
    """Merge multiple indicator frames on 'Year' after renaming 'Value' to a unique provided name.

    frames_with_names: list of (df, name)
    Returns a wide DataFrame with columns: Year, <name1>, <name2>, ...
    """
    result: pd.DataFrame | None = None
    for df, name in frames_with_names:
        if df is None or df.empty:
            continue
        tmp = df[['Year', 'Value']].rename(columns={'Value': name})
        if result is None:
            result = tmp.copy()
        else:
            result = result.merge(tmp, on='Year', how='outer')
    return pd.DataFrame(columns=['Year']) if result is None else result


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Innovation & Society (2000–2020)", layout="wide", page_icon="🎓")

# Minimalistic styling
st.markdown(
    """
    <style>
      :root { --primary-blue: #2b8cbe; --soft-gray: #e9ecef; --ink: #333333; }
      .main > div { padding-top: 0.5rem; }
      h1, h2, h3 { color: var(--ink); font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar sticky filter
st.sidebar.header("⚙️ Filters")
decade = st.sidebar.selectbox("🗓️ Decade", options=["All", "2000s", "2010s"], index=0, key="innovation_decade")

# Glossary (page-specific)
with st.sidebar.expander("Glossary List", expanded=False):
    st.markdown(
        """
        - **R&D**: Research and development expenditure as a share of GDP.
        - **Tertiary enrollment**: Gross enrollment ratio in higher education.
        """
    )

# Header
st.title("Innovation & Society Transformation (2000–2020)")
st.markdown(
    """
    China’s transition from manufacturing to knowledge and digital economy. Explore R&D, talent, connectivity,
    and social inclusion over 2000–2020.
    """
)

# ---------------------------
# Data Loading
# ---------------------------
codes = list(INDICATORS.values())
data = load_indicators(codes)

# Convenience lookups by semantic keys
get = lambda key: data[INDICATORS[key]] if INDICATORS.get(key) in data else pd.DataFrame(columns=["Year","Value"])

# Groupings
internet_df = get("INTERNET_USERS_PCT")
mobile_df = get("MOBILE_SUB_PER100")
broadband_df = get("BROADBAND_SUBSCRIBERS")
rnd_df = get("RND_PCT_GDP")
res_df = get("RESEARCHERS_PER_MILL")
pat_df = get("PATENTS_RESIDENTS")
ter_df = get("TERTIARY_ENROLLMENT")
women_parl_df = get("GENDER_PARLIAMENT")

#############################################
# SECTION A: Digital Adoption & Connectivity #
#############################################
st.markdown("---")
st.subheader("Digital Transformation of Society")

# 1) Internet users (% population)
if not internet_df.empty:
    s_data = decade_filter(internet_df.copy(), decade)
    fig_da1 = go.Figure()
    fig_da1.add_trace(go.Scatter(x=s_data['Year'], y=s_data['Value'], mode='lines+markers',
                                 name='Internet Users (% population)', line=dict(color='#2b8cbe', width=3)))
    fig_da1.update_layout(title="Internet Adoption (S-curve)", xaxis_title="Year", yaxis_title="% Population",
                          height=380, margin=dict(t=50, b=40, l=50, r=40))
    st.plotly_chart(fig_da1, use_container_width=True)

# 2) Mobile vs Broadband
if not mobile_df.empty and not broadband_df.empty:
    dual = merge_on_year_named([
        (mobile_df, 'Mobile_per100'),
        (broadband_df, 'Broadband_per100'),
    ])
    dual = decade_filter(dual, decade)
    fig_da2 = go.Figure()
    fig_da2.add_trace(go.Scatter(x=dual['Year'], y=dual['Mobile_per100'], mode='lines+markers', name='Mobile (per 100)', line=dict(color='#66c2a5', width=3)))
    fig_da2.add_trace(go.Scatter(x=dual['Year'], y=dual['Broadband_per100'], mode='lines+markers', name='Broadband (per 100)', line=dict(color='#a6bddb', width=3)))
    fig_da2.update_layout(title="Mobile vs Broadband Adoption", xaxis_title="Year", yaxis_title="per 100 people",
                          height=360, margin=dict(t=40, b=30, l=50, r=40))
    st.plotly_chart(fig_da2, use_container_width=True)

#########################################################
# SECTION B: Knowledge Production & Innovation Power     #
#########################################################
st.markdown("---")
st.subheader("Innovation Engine & Knowledge Power")

# R&D % GDP + Researchers (dual axis)
if not rnd_df.empty and not res_df.empty:
    rr = merge_on_year_named([
        (rnd_df, 'R&D (% GDP)'),
        (res_df, 'Researchers per million'),
    ])
    rr = decade_filter(rr, decade)
    fig_i1 = go.Figure()
    fig_i1.add_trace(go.Scatter(x=rr['Year'], y=rr['R&D (% GDP)'], mode='lines+markers', name='R&D (% GDP)', yaxis='y1', line=dict(color='#2b8cbe', width=3)))
    fig_i1.add_trace(go.Scatter(x=rr['Year'], y=rr['Researchers per million'], mode='lines+markers', name='Researchers per million', yaxis='y2', line=dict(color='#66c2a5', width=3)))
    fig_i1.update_layout(
        title="R&D and Research Capacity",
        xaxis_title="Year",
        yaxis=dict(title='R&D (% GDP)', side='left'),
        yaxis2=dict(title='Researchers per million', overlaying='y', side='right'),
        height=380, margin=dict(t=40, b=30, l=50, r=50)
    )
    st.plotly_chart(fig_i1, use_container_width=True)

# Patent counts
if not pat_df.empty:
    p = decade_filter(pat_df.copy(), decade)
    fig_i2 = go.Figure()
    fig_i2.add_trace(go.Bar(x=p['Year'], y=p['Value'], name='Patents (residents)', marker_color='#a6bddb'))
    fig_i2.update_layout(title="Patent Counts", xaxis_title="Year", yaxis_title="Count", height=360, margin=dict(t=40, b=30, l=50, r=40), showlegend=False)
    st.plotly_chart(fig_i2, use_container_width=True)

#########################################################
# SECTION C: Education & Inclusion                         #
#########################################################
st.markdown("---")
st.subheader("Education & Inclusion")

# Tertiary enrollment
if not ter_df.empty:
    te = decade_filter(ter_df.copy(), decade)
    fig_s1 = go.Figure()
    fig_s1.add_trace(go.Scatter(x=te['Year'], y=te['Value'], mode='lines+markers', name='Tertiary enrollment (%)', line=dict(color='#2b8cbe', width=3)))
    fig_s1.update_layout(title="Tertiary Education Enrollment", xaxis_title="Year", yaxis_title="Enrollment (%)", height=360, margin=dict(t=40,b=30,l=50,r=40), showlegend=False)
    st.plotly_chart(fig_s1, use_container_width=True)

# Women in parliament (if available)
if not get("GENDER_PARLIAMENT").empty:
    wp_df = get("GENDER_PARLIAMENT")
    wp = decade_filter(wp_df.copy(), decade)
    fig_s2 = go.Figure()
    fig_s2.add_trace(go.Scatter(x=wp['Year'], y=wp['Value'], mode='lines+markers', name='Women in Parliament (%)', line=dict(color='#a6bddb', width=3)))
    if not wp.empty:
        ly = int(wp['Year'].max()); lv = float(wp[wp['Year'] == ly]['Value'].iloc[0])
        fig_s2.add_annotation(x=ly, y=lv, text=f"{lv:.1f}% ({ly})", showarrow=True, arrowhead=2)
    fig_s2.update_layout(title="Women in Parliament", xaxis_title="Year", yaxis_title="Percent of seats", height=320, margin=dict(t=40,b=20,l=50,r=40), showlegend=False)
    st.plotly_chart(fig_s2, use_container_width=True)

# Conclusion
st.markdown("---")
st.subheader("Conclusion")
st.markdown(
    """
    From 2000 to 2020, China intensified R&D and scaled connectivity while tertiary education expanded.
    These shifts underpin the move toward a knowledge-based, innovation-driven society.
    """
)

# Navigation
st.markdown("---")
col_left, col_right = st.columns(2)
with col_left:
    if st.button("← Back: Environment & Energy", use_container_width=True):
        st.switch_page("pages/3_Environment_and_Energy.py")
with col_right:
    if st.button("Next: Home →", use_container_width=True):
        st.switch_page("Home.py")
