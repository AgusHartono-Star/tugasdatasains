import os
import requests
import pandas as pd
import mysql.connector

# =========================
# 1. EXTRACT
# =========================

api_key = os.getenv("CG_API_KEY")

url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

params = {
    "vs_currency": "usd",
    "days": "90",
    "interval": "daily",
    "x_cg_demo_api_key": api_key
}

response = requests.get(url, params=params)
response.raise_for_status()

data = response.json()

# =========================
# 2. TRANSFORM
# =========================

prices = data["prices"]
market_caps = data["market_caps"]
volumes = data["total_volumes"]

df = pd.DataFrame({
    "timestamp": [x[0] for x in prices],
    "price_usd": [x[1] for x in prices],
    "market_cap_usd": [x[1] for x in market_caps],
    "volume_usd": [x[1] for x in volumes]
})

df["date"] = pd.to_datetime(df["timestamp"], unit="ms")

df = df[
    ["date", "price_usd", "market_cap_usd", "volume_usd"]
]

# Cleaning
df = df.drop_duplicates()

# Validasi missing value
df = df.dropna()

# Validasi nilai
df = df[
    (df["price_usd"] >= 0) &
    (df["market_cap_usd"] >= 0) &
    (df["volume_usd"] >= 0)
]

# =========================
# 3. LOAD
# =========================

connection = mysql.connector.connect(
    host="localhost",
    port=3306,
    user="root",
    password="root",
    database="bitcoin_db"
)

cursor = connection.cursor()

# Bersihkan data lama agar dataset mencerminkan
# hasil pengambilan terbaru
cursor.execute("TRUNCATE TABLE bitcoin_market_data")

insert_query = """
INSERT INTO bitcoin_market_data
(date, price_usd, market_cap_usd, volume_usd)
VALUES (%s, %s, %s, %s)
"""

for _, row in df.iterrows():
    cursor.execute(
        insert_query,
        (
            row["date"].to_pydatetime(),
            float(row["price_usd"]),
            float(row["market_cap_usd"]),
            float(row["volume_usd"])
        )
    )

connection.commit()

print("Jumlah data berhasil dimasukkan:", cursor.rowcount)

cursor.close()
connection.close()