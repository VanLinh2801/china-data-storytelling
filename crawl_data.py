"""
crawl_china_databank.py
Crawl dữ liệu chỉ số phát triển kinh tế - xã hội Trung Quốc từ trang DataBank World Bank
Giai đoạn: 1960 - 2024
Sau đó filter lấy năm 2000 - 2020 và lưu vào china.csv
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
import time

BASE_URL = "https://api.worldbank.org/v2/country/CHN/indicator/{}?format=json&per_page=20000"
INDICATORS_URL = "https://databank.worldbank.org/source/world-development-indicators"

OUTPUT_FILE = "china.csv"
START_YEAR = 2000
END_YEAR = 2020

def get_indicator_list():
    print("[INFO] Fetching indicator list page...")
    r = requests.get(INDICATORS_URL)
    soup = BeautifulSoup(r.text, "html.parser")
    
    indicators = set()
    for script in soup.find_all("script"):
        if script.string and "seriesCode" in script.string:
            lines = script.string.split("\n")
            for line in lines:
                if "seriesCode" in line and ":" in line:
                    code = line.split(":")[1].replace('"', '').replace(",", "").strip()
                    if len(code) > 2:
                        indicators.add(code)

    print(f"[INFO] Found ~{len(indicators)} indicator codes (raw).")
    return list(indicators)

def crawl_indicator(indicator_code):
    try:
        url = BASE_URL.format(indicator_code)
        r = requests.get(url, timeout=20)
        data = r.json()
        if not isinstance(data, list) or len(data) < 2:
            return None

        rows = []
        for entry in data[1]:
            if entry.get("countryiso3code") == "CHN":
                year = entry.get("date")
                value = entry.get("value")
                rows.append({
                    "Country Name": entry.get("country", {}).get("value"),
                    "Country Code": "CHN",
                    "Indicator Name": entry.get("indicator", {}).get("value"),
                    "Indicator Code": indicator_code,
                    "Year": int(year) if year.isdigit() else None,
                    "Value": value
                })
        return rows
    except Exception:
        return None

def main():
    indicators = get_indicator_list()

    all_rows = []
    print("[INFO] Crawling indicator data...")
    for i, ind in enumerate(indicators):
        print(f"[{i+1}/{len(indicators)}] Crawling {ind} ...")
        rows = crawl_indicator(ind) 
        if rows:
            all_rows.extend(rows)
        time.sleep(0.1)  # respect server

    df = pd.DataFrame(all_rows)

    df = df.dropna(subset=["Year"])
    df = df[(df["Year"] >= START_YEAR) & (df["Year"] <= END_YEAR)]
    df = df[df["Country Code"] == "CHN"]

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[DONE] Saved China data {START_YEAR}-{END_YEAR} → {OUTPUT_FILE}")
    
if __name__ == "__main__":
    main()