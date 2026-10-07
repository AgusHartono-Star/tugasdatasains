import os
import requests
import pandas as pd

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

print("\nData hasil transformasi:")
print(df.head())

print("\nInformasi DataFrame:")
print(df.info())

print("\nJumlah baris dan kolom:")
print(df.shape)

print("\n=== DATA CLEANING ===")

print("\n1. Missing value:")
print(df.isnull().sum())

print("\n2. Duplicate:")
print("Jumlah duplicate:", df.duplicated().sum())

print("\n3. Tipe data:")
print(df.dtypes)

print("\n4. Validasi nilai negatif:")
print("price_usd negatif:", (df["price_usd"] < 0).sum())
print("market_cap_usd negatif:", (df["market_cap_usd"] < 0).sum())
print("volume_usd negatif:", (df["volume_usd"] < 0).sum())

print("\n5. Rentang data:")
print("Tanggal:", df["date"].min(), "sampai", df["date"].max())
print("Harga minimum:", df["price_usd"].min())
print("Harga maksimum:", df["price_usd"].max())