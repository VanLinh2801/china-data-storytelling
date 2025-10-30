from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "China.cleaned.csv"
FALLBACK_DATASET = PROJECT_ROOT / "China.csv"

INDICATORS = {
    "GDP_CURRENT_USD": "NY.GDP.MKTP.CD",
    "GDP_GROWTH": "NY.GDP.MKTP.KD.ZG",
    "GDP_PC_CONST": "NY.GDP.PCAP.KD",
    "SECTOR_AGR": "NV.AGR.TOTL.ZS",
    "SECTOR_IND": "NV.IND.TOTL.ZS",
    "SECTOR_SRV": "NV.SRV.TOTL.ZS",
    "GCF_PCT_GDP": "NE.GDI.FTOT.ZS",
    "CPI_INFL": "FP.CPI.TOTL.ZG",
    "GDP_DEFL": "NY.GDP.DEFL.KD.ZG",
    "CONS_PRIV": "NE.CON.PRVT.ZS",
    "CONS_GOV": "NE.CON.GOVT.ZS",
    "EXPORTS_PCT_GDP": "NE.EXP.GNFS.ZS",
    "IMPORTS_PCT_GDP": "NE.IMP.GNFS.ZS",
    "FDI_IN_PCT_GDP": "BX.KLT.DINV.WD.GD.ZS",
    "RESERVES_USD": "FI.RES.TOTL.CD",
    "EXTERNAL_DEBT_PCT_GNI": "DT.DOD.DECT.GN.ZS",
    "CURRENT_ACCOUNT_PCT_GDP": "BN.CAB.XOKA.GD.ZS",
}

YEAR_COLUMNS = [str(y) for y in range(2000, 2021)]


def _read_dataset() -> pd.DataFrame:
    if DEFAULT_DATASET.exists():
        df = pd.read_csv(DEFAULT_DATASET)
    else:
        df = pd.read_csv(FALLBACK_DATASET)
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    return df


def _filter_indicator(df: pd.DataFrame, indicator_code: str) -> pd.DataFrame:
    subset = df[df["Indicator code"] == indicator_code].copy()
    if subset.empty and "Indicator Code" in df.columns:
        subset = df[df["Indicator Code"] == indicator_code].copy()
    return subset


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


def compute_gdp_trillion_usd(gdp_current_usd: pd.DataFrame) -> pd.DataFrame:
    df = gdp_current_usd.copy()
    df["GDP_trillion_USD"] = df["Value"] / 1e12
    return df[["Year", "GDP_trillion_USD"]]


def compute_net_exports_pct(cons_priv: pd.DataFrame, cons_gov: pd.DataFrame, invest: pd.DataFrame) -> pd.DataFrame:
    merged = (
        cons_priv.rename(columns={"Value": "cons_priv"})
        .merge(cons_gov.rename(columns={"Value": "cons_gov"}), on="Year", how="outer")
        .merge(invest.rename(columns={"Value": "invest"}), on="Year", how="outer")
    )
    merged = merged.dropna(subset=["cons_priv", "cons_gov", "invest"], how="any")
    merged["net_exports"] = 100.0 - (merged["cons_priv"] + merged["cons_gov"] + merged["invest"])
    return merged[["Year", "net_exports"]]


def add_decade_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Decade"] = np.where(df["Year"] <= 2009, "2000s", "2010s")
    return df


def decade_filter(df: pd.DataFrame, decade: str | None) -> pd.DataFrame:
    if decade in (None, "All"):
        return df
    if decade == "2000s":
        return df[(df["Year"] >= 2000) & (df["Year"] <= 2009)]
    if decade == "2010s":
        return df[(df["Year"] >= 2010) & (df["Year"] <= 2020)]
    return df


st.set_page_config(page_title="Economy (2000–2020)", layout="wide", page_icon="📈")

st.markdown(
    """
    <style>
      :root { --primary-red: #d33f49; --soft-gray: #e9ecef; --ink: #333333; }
      .main > div { padding-top: 0.5rem; }
      h1, h2, h3 { color: var(--ink); font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("⚙️ Filters")
decade_choice = st.sidebar.selectbox("🗓️ Decade", options=["All", "2000s", "2010s"], index=0, key="economy_decade")

st.title("China’s Economic Transformation (2000–2020)")
st.markdown(
    """
    China underwent rapid economic growth and a structural shift. Explore GDP, sectors, investment, inflation, and demand composition.
    """
)

needed_codes = [
    INDICATORS["GDP_CURRENT_USD"],
    INDICATORS["GDP_GROWTH"],
    INDICATORS["GDP_PC_CONST"],
    INDICATORS["SECTOR_AGR"],
    INDICATORS["SECTOR_IND"],
    INDICATORS["SECTOR_SRV"],
    INDICATORS["GCF_PCT_GDP"],
    INDICATORS["CPI_INFL"],
    INDICATORS["GDP_DEFL"],
    INDICATORS["CONS_PRIV"],
    INDICATORS["CONS_GOV"],
    INDICATORS["EXPORTS_PCT_GDP"],
    INDICATORS["IMPORTS_PCT_GDP"],
    INDICATORS["FDI_IN_PCT_GDP"],
    INDICATORS["RESERVES_USD"],
    INDICATORS["EXTERNAL_DEBT_PCT_GNI"],
    INDICATORS["CURRENT_ACCOUNT_PCT_GDP"],
]

data = load_indicators(needed_codes)

gdp_current = compute_gdp_trillion_usd(data[INDICATORS["GDP_CURRENT_USD"]])
gdp_growth = data[INDICATORS["GDP_GROWTH"]].rename(columns={"Value": "GDP_growth_pct"})
gdp_pc = data[INDICATORS["GDP_PC_CONST"]].rename(columns={"Value": "GDP_per_capita_const_USD"})
gdp_pc_with_growth = (
    gdp_pc.sort_values("Year").assign(
        GDP_pc_growth_pct=lambda d: d["GDP_per_capita_const_USD"].pct_change() * 100.0
    )
)

sector_agr = data[INDICATORS["SECTOR_AGR"]].rename(columns={"Value": "Agriculture"})
sector_ind = data[INDICATORS["SECTOR_IND"]].rename(columns={"Value": "Industry"})
sector_srv = data[INDICATORS["SECTOR_SRV"]].rename(columns={"Value": "Services"})

invest = data[INDICATORS["GCF_PCT_GDP"]].rename(columns={"Value": "Invest_pct_GDP"})
infl_cpi = data[INDICATORS["CPI_INFL"]].rename(columns={"Value": "CPI_infl_pct"})
infl_defl = data[INDICATORS["GDP_DEFL"]].rename(columns={"Value": "GDP_deflator_pct"})

cons_priv = data[INDICATORS["CONS_PRIV"]]
cons_gov = data[INDICATORS["CONS_GOV"]]
net_exports = compute_net_exports_pct(cons_priv, cons_gov, invest.rename(columns={"Invest_pct_GDP": "Value"}))

exports = data[INDICATORS["EXPORTS_PCT_GDP"]].rename(columns={"Value": "Exports_%GDP"})
imports = data[INDICATORS["IMPORTS_PCT_GDP"]].rename(columns={"Value": "Imports_%GDP"})
fdi_in = data[INDICATORS["FDI_IN_PCT_GDP"]].rename(columns={"Value": "FDI_in_%GDP"})
reserves = data[INDICATORS["RESERVES_USD"]].rename(columns={"Value": "Reserves_USD"})
debt = data[INDICATORS["EXTERNAL_DEBT_PCT_GNI"]].rename(columns={"Value": "External_debt_%GNI"})
cab = data[INDICATORS["CURRENT_ACCOUNT_PCT_GDP"]].rename(columns={"Value": "Current_account_%GDP"})

st.subheader("Topline Dynamics")
left, right = st.columns((2, 2))
with left:
    merged_pc = gdp_pc_with_growth.copy()
    merged_pc = decade_filter(merged_pc, decade_choice)

    fig1_left = go.Figure()
    fig1_left.add_trace(
        go.Scatter(
            x=merged_pc["Year"],
            y=merged_pc["GDP_per_capita_const_USD"],
            name="GDP per capita (const USD)",
            mode="lines",
            line=dict(color="#d33f49"),
            hovertemplate="Year %{x}<br>GDP pc %{y:.0f}<extra></extra>",
            yaxis="y1",
        )
    )
    fig1_left.add_trace(
        go.Scatter(
            x=merged_pc["Year"],
            y=merged_pc["GDP_pc_growth_pct"],
            name="GDP per capita growth (%)",
            mode="lines+markers",
            line=dict(color="#d4a017"),
            hovertemplate="Year %{x}<br>Growth %{y:.1f}%<extra></extra>",
            yaxis="y2",
        )
    )
    fig1_left.update_layout(
        title=dict(text="GDP per Capita and GDP per Capita Growth (%)", y=0.95),
        xaxis=dict(title="Year"),
        yaxis=dict(title="GDP per capita (const USD)", side="left", showgrid=True, zeroline=False),
        yaxis2=dict(title="Growth (%)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="left", x=0.0),
        margin=dict(t=100, b=50, l=50, r=30),
        height=460,
    )
    fig1_left.update_xaxes(automargin=True)
    fig1_left.update_yaxes(automargin=True)
    st.plotly_chart(fig1_left, use_container_width=True)

with right:
    merged_gdp = gdp_current.merge(gdp_growth, on="Year", how="inner")
    merged_gdp = decade_filter(merged_gdp, decade_choice)

    fig1_right = go.Figure()
    fig1_right.add_trace(
        go.Scatter(
            x=merged_gdp["Year"],
            y=merged_gdp["GDP_trillion_USD"],
            name="GDP (trillion USD)",
            fill="tozeroy",
            line=dict(color="#d33f49"),
            hovertemplate="Year %{x}<br>GDP %{y:.2f}T<extra></extra>",
            yaxis="y1",
        )
    )
    fig1_right.add_trace(
        go.Scatter(
            x=merged_gdp["Year"],
            y=merged_gdp["GDP_growth_pct"],
            name="GDP growth (%)",
            mode="lines+markers",
            line=dict(color="#6c757d"),
            hovertemplate="Year %{x}<br>Growth %{y:.1f}%<extra></extra>",
            yaxis="y2",
        )
    )
    fig1_right.update_layout(
        title=dict(text="GDP (Trillion USD) and GDP Growth (%)", y=0.95),
        xaxis=dict(title="Year"),
        yaxis=dict(title="GDP (Trillion USD)", side="left", showgrid=True, zeroline=False),
        yaxis2=dict(title="Growth (%)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="left", x=0.0),
        margin=dict(t=100, b=50, l=50, r=30),
        height=460,
    )
    fig1_right.update_xaxes(automargin=True)
    fig1_right.update_yaxes(automargin=True)
    st.plotly_chart(fig1_right, use_container_width=True)

st.subheader("Investment vs Growth")
col1, col2 = st.columns((2, 2))
with col1:
    inv_growth = invest.rename(columns={"Invest_pct_GDP": "Invest"}).merge(gdp_growth, on="Year", how="inner")
    inv_growth = add_decade_label(inv_growth)
    inv_growth = decade_filter(inv_growth, decade_choice)

    fig3 = px.scatter(
        inv_growth,
        x="Invest",
        y="GDP_growth_pct",
        color="Decade",
        color_discrete_map={"2000s": "#4e79a7", "2010s": "#d33f49"},
        hover_data={"Year": True, "Invest": ":.1f", "GDP_growth_pct": ":.1f"},
        trendline="ols",
    )
    fig3.update_layout(title=dict(text="Investment (% of GDP) vs GDP Growth (%)", y=0.95), margin=dict(t=100, b=50, l=50, r=30), height=460, legend_title="", legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="left", x=0.0))
    fig3.update_xaxes(automargin=True)
    fig3.update_yaxes(automargin=True)
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    infl = infl_cpi.merge(infl_defl, on="Year", how="inner")
    infl = decade_filter(infl, decade_choice)
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=infl["Year"], y=infl["CPI_infl_pct"], name="CPI (%)", mode="lines+markers", line=dict(color="#d33f49")))
    fig4.add_trace(go.Scatter(x=infl["Year"], y=infl["GDP_deflator_pct"], name="GDP Deflator (%)", mode="lines+markers", line=dict(color="#6c757d")))
    fig4.update_layout(title=dict(text="Inflation: CPI vs GDP Deflator", y=0.95), xaxis_title="Year", yaxis_title="Percent (%)", margin=dict(t=100, b=50, l=50, r=30), height=460, legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="left", x=0.0))
    fig4.update_xaxes(automargin=True)
    fig4.update_yaxes(automargin=True)
    st.plotly_chart(fig4, use_container_width=True)

st.subheader("Economic Composition Snapshot")
comp = (
    cons_priv.rename(columns={"Value": "Household consumption %GDP"})
    .merge(cons_gov.rename(columns={"Value": "Government expenditure %GDP"}), on="Year", how="inner")
    .merge(invest.rename(columns={"Invest_pct_GDP": "Investment %GDP"}), on="Year", how="inner")
    .merge(net_exports, on="Year", how="inner")
)
comp = decade_filter(comp, decade_choice)
min_year, max_year = int(comp["Year"].min()), int(comp["Year"].max())
sel_year = st.slider("Year", min_value=min_year, max_value=max_year, value=max_year, step=1, key="treemap_fullwidth_year")
year_row = comp[comp["Year"] == sel_year].iloc[0]
treemap_df = pd.DataFrame(
    {
        "Component": ["Household consumption", "Government expenditure", "Investment", "Net exports"],
        "Percent_GDP": [
            year_row["Household consumption %GDP"],
            year_row["Government expenditure %GDP"],
            year_row["Investment %GDP"],
            year_row["net_exports"],
        ],
    }
)
fig5 = px.treemap(
    treemap_df,
    path=["Component"],
    values="Percent_GDP",
    color="Component",
    color_discrete_sequence=["#d33f49", "#adb5bd", "#6c757d", "#f7a399"],
    hover_data={"Percent_GDP": ":.1f"},
)
fig5.update_traces(texttemplate="%{label}<br>%{value:.1f}%", textposition="middle center")
fig5.update_layout(title=dict(text="GDP Expenditure Composition", y=0.98), margin=dict(t=60, b=40, l=10, r=10), height=520)
st.plotly_chart(fig5, use_container_width=True)

st.subheader("GDP Sectors & Trade Openness")
sec_col, trade_col = st.columns((2, 2))
with sec_col:
    sectors = sector_agr.merge(sector_ind, on="Year").merge(sector_srv, on="Year")
    sectors = decade_filter(sectors, decade_choice)
    sectors_long = sectors.melt(id_vars=["Year"], var_name="Sector", value_name="Percent_GDP")
    fig2 = px.area(
        sectors_long,
        x="Year",
        y="Percent_GDP",
        color="Sector",
        color_discrete_sequence=["#adb5bd", "#6c757d", "#d33f49"],
        hover_data={"Percent_GDP": ":.1f"},
    )
    fig2.update_layout(title=dict(text="Sector Composition (% of GDP)", y=0.98), margin=dict(t=60, b=40, l=50, r=30), height=420, legend_title="", legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="left", x=0.0))
    fig2.update_xaxes(automargin=True)
    fig2.update_yaxes(automargin=True)
    st.plotly_chart(fig2, use_container_width=True)

with trade_col:
    trade = exports.rename(columns={"Exports_%GDP": "Exports (% of GDP)"}).merge(
        imports.rename(columns={"Imports_%GDP": "Imports (% of GDP)"}), on="Year", how="inner"
    )
    if not trade.empty:
        trade = decade_filter(trade, decade_choice)
        long = trade.melt(id_vars=["Year"], var_name="Series", value_name="Percent of GDP")
        fig_trade = px.area(
            long,
            x="Year",
            y="Percent of GDP",
            color="Series",
            color_discrete_map={"Exports (% of GDP)": "#d33f49", "Imports (% of GDP)": "#6c757d"},
            hover_data={"Percent of GDP": ":.1f"},
        )
        fig_trade.add_vline(x=2001, line_dash="dot", line_color="#d4a017")
        fig_trade.add_annotation(x=2001, y=float(long["Percent of GDP"].max()) if not long.empty else 0, text="WTO 2001", showarrow=True, arrowhead=1)
        fig_trade.update_layout(title=dict(text="Exports & Imports (% of GDP)", y=0.98), margin=dict(t=60, b=40, l=50, r=30), height=420, legend_title="", legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="left", x=0.0))
        fig_trade.update_xaxes(automargin=True)
        fig_trade.update_yaxes(automargin=True)
        st.plotly_chart(fig_trade, use_container_width=True)
    else:
        st.info("Trade series not available.")

st.subheader("Finance & External Position")
row_a1, row_a2 = st.columns((2, 2))
with row_a1:
    fdi_g = fdi_in.rename(columns={"FDI_in_%GDP": "FDI inflows (% of GDP)"}).merge(
        gdp_growth.rename(columns={"GDP_growth_pct": "GDP growth (%)"}), on="Year", how="inner"
    )
    if not fdi_g.empty:
        fdi_g = decade_filter(fdi_g, decade_choice)
        fig_fdi = go.Figure()
        fig_fdi.add_trace(go.Scatter(x=fdi_g["Year"], y=fdi_g["FDI inflows (% of GDP)"], name="FDI inflows (% of GDP)", mode="lines+markers", line=dict(color="#d4a017"), yaxis="y1"))
        fig_fdi.add_trace(go.Scatter(x=fdi_g["Year"], y=fdi_g["GDP growth (%)"], name="GDP growth (%)", mode="lines+markers", line=dict(color="#6c757d"), yaxis="y2"))
        fig_fdi.update_layout(title=dict(text="FDI Inflows (% of GDP) vs GDP Growth (%)", y=0.95), xaxis_title="Year", yaxis=dict(title="FDI (% of GDP)", side="left"), yaxis2=dict(title="Growth (%)", overlaying="y", side="right"), height=420, margin=dict(t=80, b=50, l=50, r=30), legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="left", x=0.0))
        st.plotly_chart(fig_fdi, use_container_width=True)
    else:
        st.info("FDI/Growth series not available.")
with row_a2:
    dr = debt.rename(columns={"External_debt_%GNI": "External debt (% of GNI)"}).merge(
        reserves.rename(columns={"Reserves_USD": "Reserves (USD)"}), on="Year", how="inner"
    )
    if not dr.empty:
        dr = decade_filter(dr, decade_choice)
        fig_dr = go.Figure()
        fig_dr.add_trace(go.Scatter(x=dr["Year"], y=dr["External debt (% of GNI)"], name="External debt (% of GNI)", mode="lines+markers", line=dict(color="#6c757d"), yaxis="y1"))
        fig_dr.add_trace(go.Scatter(x=dr["Year"], y=dr["Reserves (USD)"] / 1e9, name="Reserves (USD bn)", mode="lines+markers", line=dict(color="#d33f49"), yaxis="y2"))
        fig_dr.update_layout(title=dict(text="External Debt (% of GNI) vs Foreign Reserves (USD bn)", y=0.95), xaxis_title="Year", yaxis=dict(title="Debt (% of GNI)", side="left"), yaxis2=dict(title="Reserves (USD bn)", overlaying="y", side="right"), height=420, margin=dict(t=80, b=50, l=50, r=30), legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="left", x=0.0))
        st.plotly_chart(fig_dr, use_container_width=True)
    else:
        st.info("Debt/Reserves series not available.")
row_b1, row_b2 = st.columns((2, 2))
with row_b1:
    if not cab.empty:
        ca = decade_filter(cab.rename(columns={"Current_account_%GDP": "Current account (% of GDP)"}), decade_choice)
        fig_ca = px.box(ca.assign(Decade=np.where(ca["Year"] <= 2009, "2000s", "2010s")), x="Decade", y="Current account (% of GDP)", color="Decade", color_discrete_map={"2000s": "#4e79a7", "2010s": "#d33f49"}, points="all")
        fig_ca.update_layout(title=dict(text="Current Account (% of GDP) by Decade", y=0.95), yaxis_title="Current Account (% of GDP)", height=420, margin=dict(t=80, b=50, l=50, r=30), showlegend=False)
        st.plotly_chart(fig_ca, use_container_width=True)
    else:
        st.info("Current account series not available.")
with row_b2:
    vars_df: List[pd.DataFrame] = []
    for df, col in [
        (fdi_in.rename(columns={"FDI_in_%GDP": "FDI inflows (% of GDP)"}), "FDI inflows (% of GDP)"),
        (exports.rename(columns={"Exports_%GDP": "Exports (% of GDP)"}), "Exports (% of GDP)"),
        (imports.rename(columns={"Imports_%GDP": "Imports (% of GDP)"}), "Imports (% of GDP)"),
        (gdp_growth.rename(columns={"GDP_growth_pct": "GDP growth (%)"}), "GDP growth (%)"),
        (debt.rename(columns={"External_debt_%GNI": "External debt (% of GNI)"}), "External debt (% of GNI)"),
    ]:
        if not df.empty:
            vars_df.append(df[["Year", col]])
    if vars_df:
        merged = vars_df[0]
        for x in vars_df[1:]:
            merged = merged.merge(x, on="Year", how="inner")
        merged = decade_filter(merged, decade_choice)
        if merged.shape[1] > 2:
            corr = merged.drop(columns=["Year"]).corr().round(2)
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1)
            fig_corr.update_layout(title=dict(text="Correlation: FDI, Exports, Imports, GDP Growth, External Debt", y=0.95), height=420, margin=dict(t=80, b=50, l=50, r=30))
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough variables for correlation.")
    else:
        st.info("Insufficient data to compute correlations.")

st.subheader("China Global Trade Network")
try:
    trade_regions_path = PROJECT_ROOT / "china_trade_regions.csv"
    if trade_regions_path.exists():
        tri = pd.read_csv(trade_regions_path)
        expected_cols = {"Year", "Region", "Exports_USD", "Imports_USD"}
        if expected_cols.issubset(set(tri.columns)):
            tri = tri.dropna(subset=["Year", "Region"]).copy()
            tri["Year"] = tri["Year"].astype(int)
            trade_type = st.selectbox("Select Trade Mode", ["Exports", "Imports"], key="network_trade_type")
            if trade_type == "Exports":
                value_col = "Exports_USD"
                title_suffix = "Exports"
            else:
                value_col = "Imports_USD"
                title_suffix = "Imports"
            region_colors = {
                "East Asia & Pacific": "#e07a5f",
                "Europe & Central Asia": "#4e79a7",
                "North America": "#d4a017",
                "Latin America & Caribbean": "#2ca02c",
                "Middle East & North Africa": "#ff7f0e",
                "South Asia": "#9467bd",
                "Other Asia": "#8c564b",
                "Sub-Saharan Africa": "#e377c2",
            }
            for y_sel in [2000, 2020]:
                tri_y = tri[tri["Year"] == y_sel]
                if tri_y.empty:
                    st.info(f"No data for {y_sel}.")
                else:
                    tri_sorted = tri_y.sort_values(by=value_col, ascending=False).reset_index(drop=True)
                    regions = tri_sorted["Region"].tolist()
                    china_pos = (0.0, 0.0)
                    q1 = np.quantile(values := tri_sorted[value_col].values, 0.33) if len(tri_sorted) else 0
                    q2 = np.quantile(values, 0.66) if len(tri_sorted) else 0
                    inner_r, mid_r, outer_r = 2.0, 3.2, 4.0
                    inner, mid, outer = [], [], []
                    for rname, v in zip(regions, tri_sorted[value_col].values):
                        if v <= q1:
                            inner.append(rname)
                        elif v <= q2:
                            mid.append(rname)
                        else:
                            outer.append(rname)
                    region_positions = {}
                    for i, region in enumerate(inner):
                        angle = 2 * np.pi * i / max(len(inner), 1) + (0.08 * ((i % 2) - 0.5))
                        x = inner_r * np.cos(angle)
                        y = inner_r * np.sin(angle)
                        region_positions[region] = (x, y)
                    for j, region in enumerate(mid):
                        angle = 2 * np.pi * (j + 0.33) / max(len(mid), 1) + (0.09 * ((j % 2) - 0.5))
                        x = mid_r * np.cos(angle)
                        y = mid_r * np.sin(angle)
                        region_positions[region] = (x, y)
                    for k, region in enumerate(outer):
                        angle = 2 * np.pi * (k + 0.66) / max(len(outer), 1) + (0.1 * ((k % 2) - 0.5))
                        x = outer_r * np.cos(angle)
                        y = outer_r * np.sin(angle)
                        region_positions[region] = (x, y)
                    values = tri_sorted[value_col].values
                    max_value = values.max() if len(values) > 0 else 1
                    min_size, max_size = 24, 70
                    def calculate_size(v):
                        if max_value <= 0:
                            return min_size
                        return min_size + (max_size - min_size) * np.sqrt(float(v) / max_value)
                    fig_network = go.Figure()
                    for region in regions:
                        x0, y0 = china_pos
                        x1, y1 = region_positions[region]
                        fig_network.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(color="rgba(120,120,120,0.25)", width=1), hoverinfo="skip", showlegend=False))
                    for _, row in tri_sorted.iterrows():
                        region = row["Region"]
                        v = row[value_col]
                        x, y = region_positions[region]
                        size = float(calculate_size(v))
                        vb = v / 1e9
                        color = region_colors.get(region, "#6c757d")
                        fig_network.add_trace(go.Scatter(x=[x], y=[y], mode="markers", marker=dict(size=size + 8, color="rgba(0,0,0,0.08)", line=dict(width=0)), hoverinfo="skip", showlegend=False))
                        display_text = (f"${vb/1000:.1f}T" if vb >= 1000 else f"${vb:.1f}B")
                        font_size = int(max(7, min(12, size * 0.15)))
                        fig_network.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", marker=dict(size=size, color=color, line=dict(width=4, color="white"), opacity=0.96), text=[display_text], textposition="middle center", textfont=dict(color="white", size=font_size, family="Inter, Arial"), hovertemplate=f"<b>{region}</b><br>{title_suffix}: ${vb:.1f}B USD<extra></extra>", showlegend=False))
                    for rname in regions:
                        fig_network.add_trace(
                            go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color=region_colors.get(rname, "#6c757d")), name=rname, showlegend=True)
                        )
                    total_b = tri_sorted[value_col].sum() / 1e9
                    fig_network.add_trace(go.Scatter(x=[china_pos[0]], y=[china_pos[1]], mode="markers+text", marker=dict(size=56, color="#d33f49", line=dict(width=6, color="white"), opacity=0.98), text=[(f"${total_b/1000:.1f}T" if total_b >= 1000 else f"${total_b:.1f}B")], textposition="middle center", textfont=dict(color="white", size=10, family="Inter, Arial Black"), hovertemplate=f"<b>China</b><br>Total {title_suffix.lower()}: ${total_b:.1f}B USD<extra></extra>", showlegend=False))
                    fig_network.add_annotation(x=china_pos[0], y=china_pos[1]-0.6, text="China", showarrow=False, font=dict(size=12, color="#2f2f2f"), xanchor="center", yanchor="top")
                    fig_network.update_layout(
                        title=dict(text=f"{title_suffix} — {y_sel}", y=0.94),
                        xaxis=dict(visible=False, range=[-4.4, 4.4]),
                        yaxis=dict(visible=False, range=[-4.4, 4.4]),
                        height=740,
                        margin=dict(t=60, b=30, l=10, r=220),
                        plot_bgcolor="white", paper_bgcolor="white",
                        showlegend=True,
                        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, font=dict(size=11)),
                        shapes=[
                            dict(type="circle", xref="x", yref="y", x0=-1.15, y0=-1.15, x1=1.15, y1=1.15, line=dict(color="rgba(0,0,0,0.06)", width=1), layer="below"),
                            dict(type="circle", xref="x", yref="y", x0=-1.85, y0=-1.85, x1=1.85, y1=1.85, line=dict(color="rgba(0,0,0,0.04)", width=1), layer="below"),
                        ]
                    )
                    st.plotly_chart(fig_network, use_container_width=True)
        else:
            st.info("'china_trade_regions.csv' missing required columns: Year, Region, Exports_USD, Imports_USD.")
    else:
        st.info("Provide 'china_trade_regions.csv' with columns: Year, Region, Exports_USD, Imports_USD to render the network.")
except Exception as e:
    st.info(f"Unable to render trade network: {str(e)}")

st.markdown("---")
st.subheader("Conclusion")
st.markdown(
    """
    From 2000 to 2020, China’s GDP grew more than tenfold, inflation remained generally controlled,
    and the economy shifted from manufacturing towards services.
    """
)

col_left, col_right = st.columns(2)
with col_left:
    if st.button("← Back: Home", use_container_width=True):
        st.switch_page("Home.py")
with col_right:
    if st.button("Next: Population →", use_container_width=True):
        st.switch_page("pages/2_Population.py")
