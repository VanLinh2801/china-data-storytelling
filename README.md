China Data Storytelling (2000–2020)

Live app
 - Deployed on Streamlit: https://china-data-storytelling-hypff9h9aqremixs7hvg8x.streamlit.app/

Overview
An interactive, multi-page Streamlit app exploring China’s transformation from 2000 to 2020 across economy, population, environment & energy, and innovation & society. It uses World Bank indicators (cleaned into China.cleaned.csv) and visualizes trends with Plotly.

Pages & highlights
1) Economy (pages/1_Economy.py)
   - GDP level/growth, GDP per capita, inflation (CPI/deflator)
   - Sector composition (% of GDP), expenditure composition (treemap)
   - Exports/Imports (% of GDP), FDI, reserves, external debt, current account

2) Population (pages/2_Population.py)
   - Birth/Death rates, Population & growth
   - Gender: Sex ratio and Sex ratio at birth (with normal 1.05 reference)
   - Age structure (0–14, 15–64, 65+) and estimated median age
   - Employment by sector and Unemployment rates
   - Urban vs Rural composition and Megacities population rate

3) Environment & Energy (pages/3_Environment_and_Energy.py)
   - GHG emissions (total, per capita, intensity) and composition (CO2/CH4/N2O)
   - Electricity generation mix (coal, gas, nuclear, hydro, renewables)
   - PM2.5 air quality, water stress & usage, forest & land resources
   - Environmental damage costs and green savings

4) Innovation & Society (pages/4_Innovation_and_Society.py)
   - Digital adoption (internet, mobile, broadband)
   - Innovation capacity (R&D, researchers, patents, publications)
   - Education (tertiary enrollment), inclusion (women in parliament)

Glossary (per-page)
Each page’s sidebar includes a “Glossary List” expander explaining the key indicators used on that page (e.g., GDP, CPI, FDI for Economy; PM2.5, GHG for Environment; Sex ratio for Population; R&D for Innovation).

Data
 - Primary dataset: China.cleaned.csv (World Bank indicators; cleaned/standardized columns)
 - Fallback dataset: China.csv (raw export)
 - Optional: china_trade_regions.csv for the Economy trade network demo (columns: Year, Region, Exports_USD, Imports_USD)

Local setup
1) Python
   - Python 3.11+ recommended
2) Create/activate venv (Windows PowerShell)
   - python -m venv venv
   - .\\venv\\Scripts\\Activate.ps1
3) Install dependencies
   - pip install -r requirements.txt
4) Run app (from project root)
   - streamlit run Home.py
   - Or: streamlit run pages/1_Economy.py (open a specific page)

Project structure (key files)
 - Home.py
 - pages/
   - 1_Economy.py
   - 2_Population.py
   - 3_Environment_and_Energy.py
   - 4_Innovation_and_Society.py
 - China.cleaned.csv, China.csv
 - requirements.txt

Configuration & notes
 - The app auto-loads China.cleaned.csv if present; otherwise falls back to China.csv.
 - YEAR range is 2000–2020; some charts include decade filters.
 - All charts are Plotly and responsive (use_container_width=True).

Troubleshooting
 - Module errors: pip install -r requirements.txt; ensure venv is activated.
 - File not found: verify China.cleaned.csv / China.csv are in project root.
 - Plotly rendering issues: update Streamlit/Plotly to latest (see requirements.txt).

License
For personal/educational use. Replace with your organization’s license as needed.

# China 2000–2020 Storytelling Dashboard

This documentation describes the multi-page storytelling dashboard built with Streamlit and Plotly for China’s transformation between 2000 and 2020.

Contents
- 0. Home (Overview)
- 1. Economy (2000–2020)
- 2. Urbanization & Population (2000–2020)
- 3. Environment & Energy (2000–2020)
- 4. Innovation & Society (2000–2020)

Notes
- Data sources: World Bank (primary), UN data (context).
- Years covered: 2000–2020.
- Visual library: Plotly (Express + Graph Objects).
