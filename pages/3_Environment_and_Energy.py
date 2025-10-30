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

# Comprehensive indicators for environmental storytelling
INDICATORS = {
    # 1. Greenhouse Gas Emissions
    "CO2_TOTAL": "EN.GHG.CO2.MT.CE.AR5",
    "CH4_TOTAL": "EN.GHG.CH4.MT.CE.AR5", 
    "N2O_TOTAL": "EN.GHG.N2O.MT.CE.AR5",
    "CO2_INTENSITY": "EN.GHG.CO2.RT.GDP.PP.KD",
    "CO2_PER_CAPITA": "EN.GHG.CO2.PC.CE.AR5",
    "GHG_TOTAL": "EN.GHG.ALL.MT.CE.AR5",
    "GHG_PER_CAPITA": "EN.GHG.ALL.PC.CE.AR5",
    
    # 2. Energy Structure
    "ELEC_COAL": "EG.ELC.COAL.ZS",
    "ELEC_GAS": "EG.ELC.NGAS.ZS", 
    "ELEC_NUCLEAR": "EG.ELC.NUCL.ZS",
    "ELEC_HYDRO": "EG.ELC.HYRO.ZS",
    "ELEC_RENEW": "EG.ELC.RNEW.ZS",
    "FOSSIL_CONSUMPTION": "EG.USE.COMM.FO.ZS",
    "RENEW_FINAL": "EG.FEC.RNEW.ZS",
    
    # 3. Land and Forest Resources
    "FOREST_PCT": "AG.LND.FRST.ZS",
    "FOREST_DEPLETION": "NY.ADJ.DFOR.GN.ZS",
    "AGRI_LAND": "AG.LND.AGRI.ZS",
    "ARABLE_LAND": "AG.LND.ARBL.ZS",
    
    # 4. Pollution and Water Resources
    "PM25": "EN.ATM.PM25.MC.M3",
    "WATER_TOTAL": "ER.H2O.FWTL.K3",
    "WATER_AGRI": "ER.H2O.FWAG.ZS",
    "WATER_INDUSTRY": "ER.H2O.FWIN.ZS", 
    "WATER_DOMESTIC": "ER.H2O.FWDM.ZS",
    "WATER_STRESS": "ER.H2O.FWST.ZS",
    
    # 5. Green Economy and Environmental Costs
    "CO2_DAMAGE": "NY.ADJ.DCO2.GN.ZS",
    "PM_DAMAGE": "NY.ADJ.DPEM.GN.ZS",
    "ENERGY_DAMAGE": "NY.ADJ.DNGY.GN.ZS",
    "NATURAL_RESOURCES": "NY.GDP.TOTL.RT.ZS",
    "GREEN_SAVINGS": "NY.ADJ.NNAT.GN.ZS",
    
    # Economic indicators for correlation
    "GDP_PER_CAPITA": "NY.GDP.PCAP.CD",
    "GDP_GROWTH": "NY.GDP.MKTP.KD.ZG",
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


def normalize_energy_mix(mix_parts: List[pd.DataFrame], labels: List[str]) -> pd.DataFrame:
    df = None
    for part, label in zip(mix_parts, labels):
        if part.empty:
            continue
        part = part.rename(columns={"Value": label})
        df = part if df is None else df.merge(part, on="Year", how="outer")
    if df is None:
        return pd.DataFrame(columns=["Year"] + labels)
    return df


def minmax_normalize(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.zeros(len(s)), index=series.index)
    return (s - mn) / (mx - mn) * 100.0


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Environment & Energy (2000–2020)", layout="wide", page_icon="🌿")

st.markdown(
    """
    <style>
      :root { 
        --green: #2ca25f; --blue: #2b8cbe; --gray: #6c757d; --red: #d62728; 
        --orange: #ff7f0e; --purple: #9467bd; --brown: #8c564b; --pink: #e377c2;
        --light: #f8f9fa; --ink: #222; --dark-green: #1a5f1a;
      }
      .main > div { padding-top: 0.5rem; }
      body { background: var(--light); }
      h1, h2, h3 { color: var(--ink); font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.header("⚙️ Filters")
decade = st.sidebar.selectbox("🗓️ Time Period", ["All", "2000s", "2010s"], index=0, key="env_decade")

# Header
st.title("Environment & Energy Transformation (2000–2020)")
st.markdown(
    """
    China’s growth reshaped its energy system and environment. Explore emissions, energy mix, air quality, forests, and water resources across 2000–2020.
    """
)

# ---------------------------
# Data Loading
# ---------------------------
codes = list(INDICATORS.values())
data = load_indicators(codes)

# Load all data
ghg_data = {k: data[v] for k, v in INDICATORS.items() if k.startswith(('CO2_', 'CH4_', 'N2O_', 'GHG_'))}
energy_data = {k: data[v] for k, v in INDICATORS.items() if k.startswith(('ELEC_', 'FOSSIL_', 'RENEW_'))}
land_data = {k: data[v] for k, v in INDICATORS.items() if k.startswith(('FOREST_', 'AGRI_', 'ARABLE_'))}
pollution_data = {k: data[v] for k, v in INDICATORS.items() if k.startswith(('PM25', 'WATER_'))}
economy_data = {k: data[v] for k, v in INDICATORS.items() if k.startswith(('CO2_DAMAGE', 'PM_DAMAGE', 'ENERGY_DAMAGE', 'NATURAL_', 'GREEN_', 'GDP_'))}

# ---------------------------
# STORY CHAPTER 1: The Rise of Emissions (2000-2010)
# ---------------------------
st.markdown("---")
st.subheader("Greenhouse Gas Emissions")

# Chart 1.1: Total Greenhouse Gas Emissions Trend
col1, col2 = st.columns((2, 1))

with col1:
    if not ghg_data['GHG_TOTAL'].empty:
        ghg_total = ghg_data['GHG_TOTAL'].copy()
        ghg_total = decade_filter(ghg_total, decade)
        ghg_total['Value'] = ghg_total['Value'] / 1000  # Convert to Gt CO2e
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=ghg_total['Year'], y=ghg_total['Value'],
            mode='lines+markers', name='Total GHG Emissions',
            line=dict(color='#d62728', width=3),
            marker=dict(size=8)
        ))
        
        # Add policy milestones
        fig1.add_vline(x=2005, line_dash="dash", line_color="orange", 
                      annotation_text="Kyoto Protocol<br>Implementation")
        fig1.add_vline(x=2010, line_dash="dash", line_color="red", 
                      annotation_text="Peak Emissions<br>Policy Shift")
        
        fig1.update_layout(
            title="Total Greenhouse Gas Emissions (Gt CO2e)",
            xaxis_title="Year", yaxis_title="Emissions (Gt CO2e)",
            height=400, margin=dict(t=60, b=50, l=50, r=30),
            showlegend=True
        )
        st.plotly_chart(fig1, use_container_width=True)

with col2:
    if not ghg_data['CO2_PER_CAPITA'].empty and not economy_data['GDP_PER_CAPITA'].empty:
        co2_pc = ghg_data['CO2_PER_CAPITA'].copy()
        gdp_pc = economy_data['GDP_PER_CAPITA'].copy()
        
        # Merge data
        merged = co2_pc.merge(gdp_pc, on='Year', how='inner')
        merged = decade_filter(merged, decade)
        
        fig2 = px.scatter(
            merged, x='Value_y', y='Value_x',
            title="CO2 per Capita vs GDP per Capita",
            labels={'Value_x': 'CO2 per Capita (t CO2e)', 'Value_y': 'GDP per Capita (US$)'},
            color='Year', size='Year',
            color_continuous_scale='Viridis',
            height=400
        )
        fig2.update_layout(margin=dict(t=60, b=50, l=50, r=30))
        st.plotly_chart(fig2, use_container_width=True)

# Chart 1.2: Greenhouse Gas Composition
if not ghg_data['CO2_TOTAL'].empty and not ghg_data['CH4_TOTAL'].empty and not ghg_data['N2O_TOTAL'].empty:
    st.subheader("Greenhouse Gas Composition Over Time")
    
    # Prepare data for stacked area chart
    co2 = ghg_data['CO2_TOTAL'].copy().rename(columns={'Value': 'CO2'})
    ch4 = ghg_data['CH4_TOTAL'].copy().rename(columns={'Value': 'CH4'})
    n2o = ghg_data['N2O_TOTAL'].copy().rename(columns={'Value': 'N2O'})
    
    # Merge and convert to Mt CO2e
    merged = co2.merge(ch4, on='Year', how='outer').merge(n2o, on='Year', how='outer')
    merged = decade_filter(merged, decade)
    merged[['CO2', 'CH4', 'N2O']] = merged[['CO2', 'CH4', 'N2O']].fillna(0)
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=merged['Year'], y=merged['CO2'], name='CO2', 
                             fill='tonexty', line=dict(color='#d62728')))
    fig3.add_trace(go.Scatter(x=merged['Year'], y=merged['CH4'], name='CH4', 
                             fill='tonexty', line=dict(color='#ff7f0e')))
    fig3.add_trace(go.Scatter(x=merged['Year'], y=merged['N2O'], name='N2O', 
                             fill='tozeroy', line=dict(color='#2ca25f')))
    
    fig3.update_layout(
        title="Greenhouse Gas Emissions by Type (Mt CO2e)",
        xaxis_title="Year", yaxis_title="Emissions (Mt CO2e)",
        height=400, margin=dict(t=60, b=50, l=50, r=30)
    )
    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------
# STORY CHAPTER 2: The Energy Transition (2010-2020)
# ---------------------------
st.markdown("---")
st.subheader("Energy Structure")

# Chart 2.1: Electricity Generation Mix
col1, col2 = st.columns((2, 1))

with col1:
    if not energy_data['ELEC_COAL'].empty:
        # Prepare energy mix data
        coal = energy_data['ELEC_COAL'].copy().rename(columns={'Value': 'Coal'})
        gas = energy_data['ELEC_GAS'].copy().rename(columns={'Value': 'Gas'})
        nuclear = energy_data['ELEC_NUCLEAR'].copy().rename(columns={'Value': 'Nuclear'})
        hydro = energy_data['ELEC_HYDRO'].copy().rename(columns={'Value': 'Hydro'})
        renew = energy_data['ELEC_RENEW'].copy().rename(columns={'Value': 'Renewables'})
        
        # Merge data
        energy_mix = coal.merge(gas, on='Year', how='outer').merge(nuclear, on='Year', how='outer').merge(hydro, on='Year', how='outer').merge(renew, on='Year', how='outer')
        energy_mix = decade_filter(energy_mix, decade)
        energy_mix = energy_mix.fillna(0)
        
        fig4 = go.Figure()
        colors = {'Coal': '#636363', 'Gas': '#bdbdbd', 'Nuclear': '#6c757d', 'Hydro': '#2b8cbe', 'Renewables': '#2ca25f'}
        
        for col in ['Coal', 'Gas', 'Nuclear', 'Hydro', 'Renewables']:
            if col in energy_mix.columns:
                fig4.add_trace(go.Scatter(
                    x=energy_mix['Year'], y=energy_mix[col],
                    name=col, fill='tonexty', line=dict(color=colors[col])
                ))
        
        fig4.update_layout(
            title="Electricity Generation Mix (% of Total)",
            xaxis_title="Year", yaxis_title="Percentage (%)",
            height=400, margin=dict(t=60, b=50, l=50, r=30)
        )
        st.plotly_chart(fig4, use_container_width=True)

with col2:
    # 2020 Energy Composition Pie Chart
    if not energy_mix.empty and 2020 in energy_mix['Year'].values:
        y2020 = energy_mix[energy_mix['Year'] == 2020].iloc[0]
        
        pie_data = {
            'Source': ['Coal', 'Gas', 'Nuclear', 'Hydro', 'Other Renewables'],
            'Percentage': [
                y2020.get('Coal', 0),
                y2020.get('Gas', 0), 
                y2020.get('Nuclear', 0),
                y2020.get('Hydro', 0),
                max(0, y2020.get('Renewables', 0) - y2020.get('Hydro', 0))
            ]
        }
        
        fig5 = px.pie(
            pie_data, values='Percentage', names='Source',
            title="2020 Electricity Generation Mix",
            color_discrete_map={
                'Coal': '#636363', 'Gas': '#bdbdbd', 'Nuclear': '#6c757d',
                'Hydro': '#2b8cbe', 'Other Renewables': '#2ca25f'
            },
            height=400
        )
        fig5.update_layout(margin=dict(t=60, b=50, l=50, r=30))
        st.plotly_chart(fig5, use_container_width=True)

# Chart 2.2: Renewable Energy Growth
if not energy_data['ELEC_RENEW'].empty:
    st.subheader("Renewable Energy Revolution")
    
    renew_data = energy_data['ELEC_RENEW'].copy()
    renew_data = decade_filter(renew_data, decade)
    
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(
        x=renew_data['Year'], y=renew_data['Value'],
        mode='lines+markers', name='Renewable Electricity %',
        line=dict(color='#2ca25f', width=3),
        marker=dict(size=8)
    ))
    
    # Add policy milestones
    fig6.add_vline(x=2013, line_dash="dash", line_color="green", 
                  annotation_text="Blue Sky Action Plan")
    fig6.add_vline(x=2015, line_dash="dash", line_color="blue", 
                  annotation_text="Paris Agreement")
    
    fig6.update_layout(
        title="Renewable Electricity as % of Total Generation",
        xaxis_title="Year", yaxis_title="Percentage (%)",
        height=400, margin=dict(t=60, b=50, l=50, r=30)
    )
    st.plotly_chart(fig6, use_container_width=True)

# ---------------------------
# STORY CHAPTER 3: Air Quality Crisis and Recovery
# ---------------------------
st.markdown("---")
st.subheader("Air Quality")

# Chart 3.1: PM2.5 Air Pollution Trend
if not pollution_data['PM25'].empty:
    col1, col2 = st.columns((2, 1))
    
    with col1:
        pm25_data = pollution_data['PM25'].copy()
        pm25_data = decade_filter(pm25_data, decade)
        
        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(
            x=pm25_data['Year'], y=pm25_data['Value'],
            mode='lines+markers', name='PM2.5 (μg/m³)',
            line=dict(color='#d62728', width=3),
            marker=dict(size=8)
        ))
        
        # Add WHO guideline line
        fig7.add_hline(y=15, line_dash="dash", line_color="green", 
                      annotation_text="WHO Guideline (15 μg/m³)")
        fig7.add_hline(y=35, line_dash="dash", line_color="orange", 
                      annotation_text="China Standard (35 μg/m³)")
        
        # Add policy milestones
        fig7.add_vline(x=2013, line_dash="dash", line_color="blue", 
                      annotation_text="Blue Sky Action Plan")
        fig7.add_vline(x=2018, line_dash="dash", line_color="green", 
                      annotation_text="Plan Completion")
        
        fig7.update_layout(
            title="PM2.5 Air Pollution Levels (μg/m³)",
            xaxis_title="Year", yaxis_title="PM2.5 (μg/m³)",
            height=400, margin=dict(t=60, b=50, l=50, r=30)
        )
        st.plotly_chart(fig7, use_container_width=True)
    
    with col2:
        # PM2.5 Distribution Before/After 2015
        pm25_before = pm25_data[pm25_data['Year'] < 2015]['Value']
        pm25_after = pm25_data[pm25_data['Year'] >= 2015]['Value']
        
        fig8 = go.Figure()
        fig8.add_trace(go.Box(y=pm25_before, name='Before 2015', marker_color='#d62728', boxpoints='all', jitter=0.35, pointpos=0, marker=dict(size=5, opacity=0.6)))
        fig8.add_trace(go.Box(y=pm25_after, name='2015+', marker_color='#2ca25f', boxpoints='all', jitter=0.35, pointpos=0, marker=dict(size=5, opacity=0.6)))
        
        fig8.update_layout(
            title="PM2.5 Distribution Comparison",
            yaxis_title="PM2.5 (μg/m³)",
            height=400, margin=dict(t=60, b=50, l=50, r=30)
        )
        st.plotly_chart(fig8, use_container_width=True)

# ---------------------------
# STORY CHAPTER 4: Forest and Land Restoration
# ---------------------------
st.markdown("---")
st.subheader("Forest & Land Resources")

# Chart 4.1: Forest Cover and Land Use
if not land_data['FOREST_PCT'].empty:
    col1, col2 = st.columns((2, 1))
    
    with col1:
        forest_data = land_data['FOREST_PCT'].copy()
        forest_data = decade_filter(forest_data, decade)
        
        fig9 = go.Figure()
        fig9.add_trace(go.Scatter(
            x=forest_data['Year'], y=forest_data['Value'],
            mode='lines+markers', name='Forest Cover %',
            line=dict(color='#2ca25f', width=3),
            marker=dict(size=8)
        ))
        
        fig9.update_layout(
            title="Forest Area as % of Total Land Area",
            xaxis_title="Year", yaxis_title="Forest Cover (%)",
            height=400, margin=dict(t=60, b=50, l=50, r=30)
        )
        st.plotly_chart(fig9, use_container_width=True)
    
    with col2:
        # Land Use Composition 2020
        if not land_data['AGRI_LAND'].empty and not land_data['ARABLE_LAND'].empty:
            agri_data = land_data['AGRI_LAND'].copy()
            arable_data = land_data['ARABLE_LAND'].copy()
            
            # Get 2020 data
            agri_2020 = agri_data[agri_data['Year'] == 2020]['Value'].iloc[0] if 2020 in agri_data['Year'].values else 0
            arable_2020 = arable_data[arable_data['Year'] == 2020]['Value'].iloc[0] if 2020 in arable_data['Year'].values else 0
            forest_2020 = forest_data[forest_data['Year'] == 2020]['Value'].iloc[0] if 2020 in forest_data['Year'].values else 0
            
            land_use_data = {
                'Land Type': ['Agricultural', 'Arable', 'Forest', 'Other'],
                'Percentage': [agri_2020, arable_2020, forest_2020, 100 - agri_2020 - forest_2020]
            }
            
            fig10 = px.pie(
                land_use_data, values='Percentage', names='Land Type',
                title="2020 Land Use Distribution",
                color_discrete_map={
                    'Agricultural': '#8c564b', 'Arable': '#e377c2', 
                    'Forest': '#2ca25f', 'Other': '#6c757d'
                },
                height=400
            )
            fig10.update_layout(margin=dict(t=60, b=50, l=50, r=30))
            st.plotly_chart(fig10, use_container_width=True)

# ---------------------------
# STORY CHAPTER 5: Water Resources and Stress
# ---------------------------
st.markdown("---")
st.subheader("Water Resources")

# Chart 5.1: Water Stress and Usage
if not pollution_data['WATER_STRESS'].empty:
    col1, col2 = st.columns((2, 1))
    
    with col1:
        water_stress = pollution_data['WATER_STRESS'].copy()
        water_stress = decade_filter(water_stress, decade)
        
        fig11 = go.Figure()
        fig11.add_trace(go.Scatter(
            x=water_stress['Year'], y=water_stress['Value'],
            mode='lines+markers', name='Water Stress Index',
            line=dict(color='#2b8cbe', width=3),
            marker=dict(size=8)
        ))
        
        # Add stress level indicators
        fig11.add_hline(y=25, line_dash="dash", line_color="green", 
                       annotation_text="Low Stress")
        fig11.add_hline(y=50, line_dash="dash", line_color="orange", 
                       annotation_text="Medium Stress")
        fig11.add_hline(y=75, line_dash="dash", line_color="red", 
                       annotation_text="High Stress")
        
        fig11.update_layout(
            title="Water Stress Level (Freshwater Withdrawal as % of Available Resources)",
            xaxis_title="Year", yaxis_title="Water Stress Index",
            height=400, margin=dict(t=60, b=50, l=50, r=30)
        )
        st.plotly_chart(fig11, use_container_width=True)
    
    with col2:
        # Water Usage by Sector (2020)
        if not pollution_data['WATER_AGRI'].empty and not pollution_data['WATER_INDUSTRY'].empty and not pollution_data['WATER_DOMESTIC'].empty:
            agri_water = pollution_data['WATER_AGRI'].copy()
            industry_water = pollution_data['WATER_INDUSTRY'].copy()
            domestic_water = pollution_data['WATER_DOMESTIC'].copy()
            
            # Get 2020 data
            agri_2020 = agri_water[agri_water['Year'] == 2020]['Value'].iloc[0] if 2020 in agri_water['Year'].values else 0
            industry_2020 = industry_water[industry_water['Year'] == 2020]['Value'].iloc[0] if 2020 in industry_water['Year'].values else 0
            domestic_2020 = domestic_water[domestic_water['Year'] == 2020]['Value'].iloc[0] if 2020 in domestic_water['Year'].values else 0
            
            water_usage_data = {
                'Sector': ['Agriculture', 'Industry', 'Domestic'],
                'Percentage': [agri_2020, industry_2020, domestic_2020]
            }
            
            fig12 = px.pie(
                water_usage_data, values='Percentage', names='Sector',
                title="2020 Water Usage by Sector",
                color_discrete_map={
                    'Agriculture': '#8c564b', 'Industry': '#2b8cbe', 'Domestic': '#2ca25f'
                },
                height=400
            )
            fig12.update_layout(margin=dict(t=60, b=50, l=50, r=30))
            st.plotly_chart(fig12, use_container_width=True)

# ---------------------------
# STORY CHAPTER 6: Green Economy and Environmental Costs
# ---------------------------
st.markdown("---")
st.subheader("Environmental Economics")

# Chart 6.1: Environmental Damage Costs
if not economy_data['CO2_DAMAGE'].empty and not economy_data['PM_DAMAGE'].empty and not economy_data['ENERGY_DAMAGE'].empty:
    col1, col2 = st.columns((2, 1))
    
    with col1:
        co2_damage = economy_data['CO2_DAMAGE'].copy().rename(columns={'Value': 'CO2_Damage'})
        pm_damage = economy_data['PM_DAMAGE'].copy().rename(columns={'Value': 'PM_Damage'})
        energy_damage = economy_data['ENERGY_DAMAGE'].copy().rename(columns={'Value': 'Energy_Damage'})
        
        # Merge data
        damage_data = co2_damage.merge(pm_damage, on='Year', how='outer').merge(energy_damage, on='Year', how='outer')
        damage_data = decade_filter(damage_data, decade)
        damage_data = damage_data.fillna(0)
        
        fig13 = go.Figure()
        fig13.add_trace(go.Scatter(x=damage_data['Year'], y=damage_data['CO2_Damage'], 
                                 name='CO2 Damage', fill='tonexty', line=dict(color='#d62728')))
        fig13.add_trace(go.Scatter(x=damage_data['Year'], y=damage_data['PM_Damage'], 
                                 name='PM Damage', fill='tonexty', line=dict(color='#ff7f0e')))
        fig13.add_trace(go.Scatter(x=damage_data['Year'], y=damage_data['Energy_Damage'], 
                                 name='Energy Damage', fill='tozeroy', line=dict(color='#6c757d')))
        
        fig13.update_layout(
            title="Environmental Damage as % of GNI",
            xaxis_title="Year", yaxis_title="Damage (% of GNI)",
            height=400, margin=dict(t=60, b=50, l=50, r=30)
        )
        st.plotly_chart(fig13, use_container_width=True)
    
    with col2:
        # Green Savings Trend
        if not economy_data['GREEN_SAVINGS'].empty:
            green_savings = economy_data['GREEN_SAVINGS'].copy()
            green_savings = decade_filter(green_savings, decade)
            
            fig14 = go.Figure()
            fig14.add_trace(go.Scatter(
                x=green_savings['Year'], y=green_savings['Value'],
                mode='lines+markers', name='Green Savings (% of GNI)',
                line=dict(color='#2ca25f', width=3),
                marker=dict(size=8)
            ))
            
            fig14.update_layout(
                title="Adjusted Net National Savings (% of GNI)",
                xaxis_title="Year", yaxis_title="Green Savings (% of GNI)",
                height=400, margin=dict(t=60, b=50, l=50, r=30)
            )
            st.plotly_chart(fig14, use_container_width=True)

# ---------------------------
# Conclusion
# ---------------------------
st.markdown("---")
st.subheader("Conclusion")
st.markdown(
    """
    From 2000 to 2020, coal dependence eased while renewables scaled, PM2.5 declined after 2013 policies,
    forests expanded, and water stress remained a key constraint. China shows measurable progress toward a cleaner, more efficient energy system.
    """
)

# Navigation
st.markdown("---")
col_left, col_right = st.columns(2)
with col_left:
    if st.button("← Back: Population", use_container_width=True):
        st.switch_page("pages/2_Population.py")
with col_right:
    if st.button("Next: Innovation & Society →", use_container_width=True):
        st.switch_page("pages/4_Innovation_and_Society.py")
