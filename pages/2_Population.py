from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------
# Config & Constants
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "China.cleaned.csv"
FALLBACK_DATASET = PROJECT_ROOT / "China.csv"

INDICATORS = {
    "POP_TOTAL": "SP.POP.TOTL",
    "POP_GROWTH": "SP.POP.GROW",
    "BIRTH_RATE": "SP.DYN.CBRT.IN",
    "DEATH_RATE": "SP.DYN.CDRT.IN",
    "FERTILITY_RATE": "SP.DYN.TFRT.IN",
    "POP_0_14": "SP.POP.0014.TO.ZS",
    "POP_15_64": "SP.POP.1564.TO.ZS",
    "POP_65_UP": "SP.POP.65UP.TO.ZS",
    "RURAL_PCT": "SP.RUR.TOTL.ZS",
    "URBAN_PCT": "SP.URB.TOTL.IN.ZS",
    "MEGACITIES_PCT": "EN.URB.MCTY.TL.ZS",
    "SEX_RATIO_BIRTH": "SP.POP.BRTH.MF",
    "UNEMPLOYMENT_TOTAL": "SL.UEM.TOTL.ZS",
    "UNEMPLOYMENT_FEMALE": "SL.UEM.TOTL.FE.ZS",
    "UNEMPLOYMENT_MALE": "SL.UEM.TOTL.MA.ZS",
    "UNEMPLOYMENT_YOUTH": "SL.UEM.1524.ZS",
    "ELECTRICITY_ACCESS": "EG.ELC.ACCS.ZS",
    "URBAN_SANITATION": "SH.STA.WASH.P5.ZS",
    "VEHICLES_PER_1000": "IS.VEH.PCAR.P3",
    "EMP_AGR": "SL.AGR.EMPL.ZS",
    "EMP_IND": "SL.IND.EMPL.ZS",
    "EMP_SRV": "SL.SRV.EMPL.ZS",
    "POP_MALE": "SP.POP.TOTL.MA.IN",
    "POP_FEMALE": "SP.POP.TOTL.FE.IN",
    "LIFE_EXPECTANCY": "SP.DYN.LE00.IN",
    "WORKING_AGE_PCT": "SP.POP.1564.TO.ZS",
    "ELDERLY_PCT": "SP.POP.65UP.TO.ZS",
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
    if period == "2000–2010":
        return df[(df["Year"] >= 2000) & (df["Year"] <= 2010)]
    if period == "2011–2020":
        return df[(df["Year"] >= 2011) & (df["Year"] <= 2020)]
    return df


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Population (2000–2020)", layout="wide", page_icon="🏙️")

st.markdown(
    """
    <style>
      :root { --accent: #4e79a7; --accent-2: #6c757d; --light: #f8f9fa; --ink: #222; }
      .main > div { padding-top: 0.5rem; }
      body { background: var(--light); }
      h1, h2, h3 { color: var(--ink); font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar sticky filter
st.sidebar.header("⚙️ Filters")
period = st.sidebar.selectbox("🗓️ Time period", ["All", "2000–2010", "2011–2020"], index=0, key="urban_period")

# Header
st.title("Population (2000–2020)")
st.markdown(
    """
    Migration reshaped China’s demographics. Explore population levels and growth, gender ratios, age structure, employment, and urban–rural composition.
    """
)

# ---------------------------
# Data load
# ---------------------------
codes = list(INDICATORS.values())
data = load_indicators(codes)

pop_total = data[INDICATORS["POP_TOTAL"]].rename(columns={"Value": "Population"})
pop_growth = data[INDICATORS["POP_GROWTH"]].rename(columns={"Value": "Pop_growth_pct"})
birth_rate = data[INDICATORS["BIRTH_RATE"]].rename(columns={"Value": "Birth_rate"})
death_rate = data[INDICATORS["DEATH_RATE"]].rename(columns={"Value": "Death_rate"})
fertility_rate = data[INDICATORS["FERTILITY_RATE"]].rename(columns={"Value": "Fertility_rate"})
pop_0_14 = data[INDICATORS["POP_0_14"]].rename(columns={"Value": "Age_0_14_pct"})
pop_15_64 = data[INDICATORS["POP_15_64"]].rename(columns={"Value": "Age_15_64_pct"})
pop_65_up = data[INDICATORS["POP_65_UP"]].rename(columns={"Value": "Age_65_up_pct"})

# Load urban/rural data
rural_pct = data[INDICATORS["RURAL_PCT"]].rename(columns={"Value": "Rural_pct"})

# Load sex ratio at birth data
sex_ratio_birth = data[INDICATORS["SEX_RATIO_BIRTH"]].rename(columns={"Value": "Sex_ratio_birth"})

# Load unemployment data
unemployment_total = data[INDICATORS["UNEMPLOYMENT_TOTAL"]].rename(columns={"Value": "Unemployment_total"})
unemployment_female = data[INDICATORS["UNEMPLOYMENT_FEMALE"]].rename(columns={"Value": "Unemployment_female"})
unemployment_male = data[INDICATORS["UNEMPLOYMENT_MALE"]].rename(columns={"Value": "Unemployment_male"})
unemployment_youth = data[INDICATORS["UNEMPLOYMENT_YOUTH"]].rename(columns={"Value": "Unemployment_youth"})
urban_pct = data[INDICATORS["URBAN_PCT"]].rename(columns={"Value": "Urban_pct"})
megacities_pct = data[INDICATORS["MEGACITIES_PCT"]].rename(columns={"Value": "Megacities_pct"})
electricity = data[INDICATORS["ELECTRICITY_ACCESS"]].rename(columns={"Value": "Electricity_access_pct"})
sanitation = data[INDICATORS["URBAN_SANITATION"]].rename(columns={"Value": "Urban_sanitation_pct"})
vehicles = data[INDICATORS["VEHICLES_PER_1000"]].rename(columns={"Value": "Vehicles_per_1000"})
emp_agr = data[INDICATORS["EMP_AGR"]].rename(columns={"Value": "Agriculture"})
emp_ind = data[INDICATORS["EMP_IND"]].rename(columns={"Value": "Industry"})
emp_srv = data[INDICATORS["EMP_SRV"]].rename(columns={"Value": "Services"})
pop_male = data[INDICATORS["POP_MALE"]].rename(columns={"Value": "Male"})
pop_female = data[INDICATORS["POP_FEMALE"]].rename(columns={"Value": "Female"})
life = data[INDICATORS["LIFE_EXPECTANCY"]].rename(columns={"Value": "Life_expectancy"})
working_age = data[INDICATORS["WORKING_AGE_PCT"]].rename(columns={"Value": "Working_age_pct"})
elderly = data[INDICATORS["ELDERLY_PCT"]].rename(columns={"Value": "Elderly_pct"})

# ---------------------------
# Chart 1 – Birth Rate & Death Rate
# ---------------------------
st.subheader("Birth Rate & Death Rate")
col1, col2 = st.columns((2, 2))
with col1:
    birth_death = birth_rate.merge(death_rate, on="Year", how="inner")
    if not birth_death.empty:
        birth_death = decade_filter(birth_death, period)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=birth_death["Year"], y=birth_death["Birth_rate"], name="Birth Rate (per 1,000)", mode="lines+markers", line=dict(color="#4e79a7"), hovertemplate="Year %{x}<br>Birth Rate %{y:.1f}<extra></extra>"))
        fig1.add_trace(go.Scatter(x=birth_death["Year"], y=birth_death["Death_rate"], name="Death Rate (per 1,000)", mode="lines+markers", line=dict(color="#6c757d"), hovertemplate="Year %{x}<br>Death Rate %{y:.1f}<extra></extra>"))
        fig1.update_layout(title="Birth Rate and Death Rate Trends", xaxis_title="Year", yaxis_title="Rate per 1,000 people", height=460, margin=dict(t=90, b=50, l=50, r=30), legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="left", x=0.0))
        fig1.update_xaxes(automargin=True)
        fig1.update_yaxes(automargin=True)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Birth and death rate data not available.")

with col2:
    pop = pop_total.merge(pop_growth, on="Year", how="inner")
    if not pop.empty:
        pop = pop.assign(Population_billions=pop["Population"] / 1e9)
        pop = decade_filter(pop, period)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=pop["Year"], y=pop["Population_billions"], name="Population (billions)", mode="lines+markers", line=dict(color="#4e79a7"), yaxis="y1", hovertemplate="Year %{x}<br>Pop %{y:.2f}B<extra></extra>"))
        fig2.add_trace(go.Scatter(x=pop["Year"], y=pop["Pop_growth_pct"], name="Population growth (%)", mode="lines+markers", line=dict(color="#6c757d"), yaxis="y2", hovertemplate="Year %{x}<br>Growth %{y:.2f}%<extra></extra>"))
        fig2.update_layout(title="Population and Population Growth Percent", xaxis_title="Year", yaxis=dict(title="Population (Billions)", side="left"), yaxis2=dict(title="Population Growth Percent (%)", overlaying="y", side="right"), height=460, margin=dict(t=90, b=50, l=50, r=30), legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="left", x=0.0))
        fig2.update_xaxes(automargin=True)
        fig2.update_yaxes(automargin=True)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Population data not available.")

# ---------------------------
# Section 2 – Gender
# ---------------------------
st.subheader("Gender Analysis")
col3, col4 = st.columns((2, 2))
with col3:
    gender = pop_male.merge(pop_female, on="Year", how="inner")
    if not gender.empty:
        gender["Male_Female_Ratio"] = gender["Male"] / gender["Female"]
        gender = decade_filter(gender, period)
        fig3 = px.line(gender, x="Year", y="Male_Female_Ratio", markers=True, color_discrete_sequence=["#4e79a7"])
        fig3.update_layout(title="Sex Ratio (Male/Female)", yaxis_title="Male / Female", height=460, margin=dict(t=90, b=50, l=50, r=30))
        fig3.update_xaxes(automargin=True)
        fig3.update_yaxes(automargin=True)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Gender series not available.")

with col4:
    if not sex_ratio_birth.empty:
        birth_ratio = decade_filter(sex_ratio_birth, period)
        fig4 = px.line(birth_ratio, x="Year", y="Sex_ratio_birth", markers=True, color_discrete_sequence=["#6c757d"])
        fig4.update_layout(title="Sex Ratio at Birth (Male/Female)", yaxis_title="Male Births / Female Births", height=460, margin=dict(t=90, b=50, l=50, r=30))
        fig4.update_xaxes(automargin=True)
        fig4.update_yaxes(automargin=True)
        # Add reference line for normal ratio (1.05)
        fig4.add_hline(y=1.05, line_dash="dot", line_color="#ff6b6b", annotation_text="Normal Ratio (1.05)")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Sex ratio at birth data not available.")

# ---------------------------
# Section 3 – Age Structure
# ---------------------------
st.subheader("Age Structure Analysis")
col5, col6 = st.columns((2, 2))
with col5:
    # Merge all age group data - simple 3 groups
    age_groups = pop_0_14.merge(pop_15_64, on="Year", how="outer").merge(pop_65_up, on="Year", how="outer")
    if not age_groups.empty:
        age_groups = decade_filter(age_groups, period)
        # Melt data for stacked area chart
        age_long = age_groups.melt(id_vars=["Year"], var_name="Age_Group", value_name="Percentage")
        # Create age group labels
        age_long["Age_Label"] = age_long["Age_Group"].map({
            "Age_0_14_pct": "0-14 years (Children)",
            "Age_15_64_pct": "15-64 years (Working Age)",
            "Age_65_up_pct": "65+ years (Elderly)"
        })
        
        # Filter out NaN values
        age_long = age_long.dropna(subset=["Percentage"])
        
        fig5 = px.area(age_long, x="Year", y="Percentage", color="Age_Label", 
                      color_discrete_map={
                          "0-14 years (Children)": "#4e79a7",
                          "15-64 years (Working Age)": "#6c757d",
                          "65+ years (Elderly)": "#ffb347"
                      },
                      hover_data={"Percentage": ":.1f"})
        fig5.update_layout(title="Population Age Structure by Age Groups", 
                          xaxis_title="Year", yaxis_title="Percentage of Population (%)", 
                          height=460, margin=dict(t=90, b=50, l=50, r=30), 
                          legend_title="Age Groups",
                          legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="left", x=0.0))
        fig5.update_xaxes(automargin=True)
        fig5.update_yaxes(automargin=True)
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Age structure data not available.")

with col6:
    # Calculate median age from age structure data
    if not age_groups.empty:
        # Calculate weighted median age
        age_groups["Median_Age"] = (
            age_groups["Age_0_14_pct"] * 7 +  # Midpoint of 0-14
            age_groups["Age_15_64_pct"] * 39.5 +  # Midpoint of 15-64
            age_groups["Age_65_up_pct"] * 75  # Midpoint of 65+
        ) / 100
        
        median_age = age_groups[["Year", "Median_Age"]].dropna()
        fig6 = px.line(median_age, x="Year", y="Median_Age", markers=True, color_discrete_sequence=["#a0c4ff"])
        fig6.update_layout(title="Median Age of Population", 
                          xaxis_title="Year", yaxis_title="Median Age (Years)", 
                          height=460, margin=dict(t=90, b=50, l=50, r=30))
        fig6.update_xaxes(automargin=True)
        fig6.update_yaxes(automargin=True)
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info("Age structure data not available.")

# ---------------------------
# Section 4 – Employment & Unemployment
# ---------------------------
st.subheader("Employment & Unemployment Analysis")
col7, col8 = st.columns((2, 2))
with col7:
    emp = emp_agr.merge(emp_ind, on="Year", how="outer").merge(emp_srv, on="Year", how="outer")
    if not emp.empty:
        emp = decade_filter(emp, period)
        emp_long = emp.melt(id_vars=["Year"], var_name="Sector", value_name="Percentage")
        fig7 = px.area(emp_long, x="Year", y="Percentage", color="Sector", 
                      color_discrete_map={
                          "Agriculture": "#4e79a7",
                          "Industry": "#6c757d", 
                          "Services": "#a0c4ff"
                      },
                      hover_data={"Percentage": ":.1f"})
        fig7.update_layout(title="Employment Structure by Sector", 
                          xaxis_title="Year", yaxis_title="Percentage of Employment (%)", 
                          height=460, margin=dict(t=90, b=50, l=50, r=30),
                          legend_title="Sector",
                          legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="left", x=0.0))
        fig7.update_xaxes(automargin=True)
        fig7.update_yaxes(automargin=True)
        st.plotly_chart(fig7, use_container_width=True)
    else:
        st.info("Employment structure data not available.")

with col8:
    unemp = unemployment_total.merge(unemployment_female, on="Year", how="outer").merge(unemployment_male, on="Year", how="outer")
    if not unemp.empty:
        unemp = decade_filter(unemp, period)
        fig8 = go.Figure()
        fig8.add_trace(go.Scatter(x=unemp["Year"], y=unemp["Unemployment_total"], 
                                 name="Total", mode="lines+markers", 
                                 line=dict(color="#4e79a7")))
        fig8.add_trace(go.Scatter(x=unemp["Year"], y=unemp["Unemployment_female"], 
                                 name="Female", mode="lines+markers", 
                                 line=dict(color="#a0c4ff")))
        fig8.add_trace(go.Scatter(x=unemp["Year"], y=unemp["Unemployment_male"], 
                                 name="Male", mode="lines+markers", 
                                 line=dict(color="#6c757d")))
        fig8.update_layout(title="Unemployment Rate by Gender", 
                          xaxis_title="Year", yaxis_title="Unemployment Rate (%)", 
                          height=460, margin=dict(t=90, b=50, l=50, r=30),
                          legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="left", x=0.0))
        fig8.update_xaxes(automargin=True)
        fig8.update_yaxes(automargin=True)
        st.plotly_chart(fig8, use_container_width=True)
    else:
        st.info("Unemployment data not available.")

# ---------------------------
# Additional Analysis – Urban vs Rural Population
# ---------------------------
st.subheader("Urban vs Rural Population Structure")
col9, col10 = st.columns((2, 2))
with col9:
    urban_rural = urban_pct.merge(rural_pct, on="Year", how="inner")
    if not urban_rural.empty:
        urban_rural = decade_filter(urban_rural, period)
        fig9 = go.Figure()
        fig9.add_trace(go.Scatter(x=urban_rural["Year"], y=urban_rural["Urban_pct"], 
                                 name="Urban (%)", mode="lines+markers", 
                                 line=dict(color="#4e79a7"), fill="tonexty"))
        fig9.add_trace(go.Scatter(x=urban_rural["Year"], y=urban_rural["Rural_pct"], 
                                 name="Rural (%)", mode="lines+markers", 
                                 line=dict(color="#6c757d"), fill="tozeroy"))
        fig9.update_layout(title="Urban vs Rural Population Distribution", 
                          xaxis_title="Year", yaxis_title="Percentage of Population (%)", 
                          height=460, margin=dict(t=90, b=50, l=50, r=30),
                          legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="left", x=0.0))
        fig9.update_xaxes(automargin=True)
        fig9.update_yaxes(automargin=True)
        st.plotly_chart(fig9, use_container_width=True)
    else:
        st.info("Urban/Rural data not available.")

# ---------------------------
# Chart 10 – Megacities Population Rate
# ---------------------------
with col10:
    if not megacities_pct.empty:
        megacities = decade_filter(megacities_pct, period)
        fig10 = px.line(megacities, x="Year", y="Megacities_pct", 
                       markers=True, color_discrete_sequence=["#ff6b6b"])
        fig10.update_layout(title="Megacities Population Rate", 
                           xaxis_title="Year", yaxis_title="Megacities Population (%)", 
                           height=460, margin=dict(t=90, b=50, l=50, r=30))
        fig10.update_xaxes(automargin=True)
        fig10.update_yaxes(automargin=True)
        st.plotly_chart(fig10, use_container_width=True)
    else:
        st.info("Megacities population data not available.")

# ---------------------------
# Conclusion & Navigation
# ---------------------------
st.markdown("---")
st.subheader("Conclusion")
st.markdown(
    """
    China's demographic transformation shows declining birth rates, stable death rates,
    and rapid urbanization. The population structure shifted toward older demographics with improved
    life expectancy and infrastructure development.
    """
)

col_left, col_right = st.columns(2)
with col_left:
    if st.button("← Back: Economy", use_container_width=True):
        st.switch_page("pages/1_Economy.py")
with col_right:
    if st.button("Next: Environment & Energy →", use_container_width=True):
        st.switch_page("pages/3_Environment_and_Energy.py")
