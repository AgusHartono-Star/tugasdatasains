import os
import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "forex_db"
}

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "random_forest_model.joblib"
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "output"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Menghubungkan ke MySQL...")

conn = mysql.connector.connect(**DB_CONFIG)

query = """
SELECT *
FROM training_dataset
ORDER BY datetime
"""

df = pd.read_sql(query, conn)

conn.close()

print(f"Data berhasil dimuat: {len(df)} baris")

df["datetime"] = pd.to_datetime(df["datetime"])

df = df.sort_values("datetime").reset_index(drop=True)


# ============================================================
# 1. HARGA EUR/USD
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["datetime"],
    df["close"],
    label="Close"
)

plt.title("EUR/USD Close Price")
plt.xlabel("Datetime")
plt.ylabel("Price")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "01_close_price.png"),
    dpi=150
)

plt.close()


# ============================================================
# 2. SMA DAN EMA
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["datetime"],
    df["close"],
    label="Close",
    linewidth=1
)

plt.plot(
    df["datetime"],
    df["sma_7"],
    label="SMA 7"
)

plt.plot(
    df["datetime"],
    df["sma_14"],
    label="SMA 14"
)

plt.plot(
    df["datetime"],
    df["sma_30"],
    label="SMA 30"
)

plt.plot(
    df["datetime"],
    df["ema_7"],
    label="EMA 7"
)

plt.plot(
    df["datetime"],
    df["ema_14"],
    label="EMA 14"
)

plt.plot(
    df["datetime"],
    df["ema_30"],
    label="EMA 30"
)

plt.title("EUR/USD Price with SMA and EMA")
plt.xlabel("Datetime")
plt.ylabel("Price")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "02_sma_ema.png"),
    dpi=150
)

plt.close()


# ============================================================
# 3. RSI
# ============================================================

plt.figure(figsize=(14, 5))

plt.plot(
    df["datetime"],
    df["rsi_14"],
    label="RSI 14"
)

plt.axhline(
    70,
    linestyle="--",
    label="Overbought 70"
)

plt.axhline(
    30,
    linestyle="--",
    label="Oversold 30"
)

plt.title("EUR/USD RSI 14")
plt.xlabel("Datetime")
plt.ylabel("RSI")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "03_rsi.png"),
    dpi=150
)

plt.close()


# ============================================================
# 4. MACD
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["datetime"],
    df["macd"],
    label="MACD"
)

plt.plot(
    df["datetime"],
    df["macd_signal"],
    label="Signal"
)

plt.bar(
    df["datetime"],
    df["macd_hist"],
    label="Histogram",
    alpha=0.3
)

plt.title("EUR/USD MACD")
plt.xlabel("Datetime")
plt.ylabel("MACD")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "04_macd.png"),
    dpi=150
)

plt.close()


# ============================================================
# 5. BOLLINGER BANDS
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["datetime"],
    df["close"],
    label="Close"
)

plt.plot(
    df["datetime"],
    df["bollinger_upper"],
    label="Upper Band"
)

plt.plot(
    df["datetime"],
    df["bollinger_middle"],
    label="Middle Band"
)

plt.plot(
    df["datetime"],
    df["bollinger_lower"],
    label="Lower Band"
)

plt.fill_between(
    df["datetime"],
    df["bollinger_lower"],
    df["bollinger_upper"],
    alpha=0.1
)

plt.title("EUR/USD Bollinger Bands")
plt.xlabel("Datetime")
plt.ylabel("Price")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "05_bollinger_bands.png"),
    dpi=150
)

plt.close()


# ============================================================
# 6. RETURN
# ============================================================

plt.figure(figsize=(14, 5))

plt.plot(
    df["datetime"],
    df["return_1d"]
)

plt.axhline(
    0,
    linestyle="--"
)

plt.title("EUR/USD Return")
plt.xlabel("Datetime")
plt.ylabel("Return")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "06_return.png"),
    dpi=150
)

plt.close()


# ============================================================
# 7. VOLATILITY
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["datetime"],
    df["volatility_7d"],
    label="Volatility 7"
)

plt.plot(
    df["datetime"],
    df["volatility_14d"],
    label="Volatility 14"
)

plt.plot(
    df["datetime"],
    df["volatility_30d"],
    label="Volatility 30"
)

plt.title("EUR/USD Rolling Volatility")
plt.xlabel("Datetime")
plt.ylabel("Volatility")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "07_volatility.png"),
    dpi=150
)

plt.close()


# ============================================================
# 8. CORRELATION HEATMAP
# ============================================================

numeric_columns = [
    "open",
    "high",
    "low",
    "close",
    "price_change",
    "return_1d",
    "daily_range",
    "daily_range_pct",
    "volatility_7d",
    "volatility_14d",
    "volatility_30d",
    "lag_close_1",
    "lag_close_2",
    "lag_close_3",
    "lag_close_7",
    "sma_7",
    "sma_14",
    "sma_30",
    "ema_7",
    "ema_14",
    "ema_30",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bollinger_upper",
    "bollinger_middle",
    "bollinger_lower",
    "next_close"
]

corr = df[numeric_columns].corr()

plt.figure(figsize=(16, 13))

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    square=False
)

plt.title("Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "08_correlation_heatmap.png"),
    dpi=150
)

plt.close()


# ============================================================
# 9. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print("Memuat Random Forest model...")

model = joblib.load(MODEL_PATH)

features = [
    "open",
    "high",
    "low",
    "close",
    "price_change",
    "return_1d",
    "daily_range",
    "daily_range_pct",
    "volatility_7d",
    "volatility_14d",
    "volatility_30d",
    "lag_close_1",
    "lag_close_2",
    "lag_close_3",
    "lag_close_7",
    "sma_7",
    "sma_14",
    "sma_30",
    "ema_7",
    "ema_14",
    "ema_30",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bollinger_upper",
    "bollinger_middle",
    "bollinger_lower"
]

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=True
)

plt.figure(figsize=(10, 10))

plt.barh(
    importance["feature"],
    importance["importance"]
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "09_feature_importance.png"),
    dpi=150
)

plt.close()


# ============================================================
# FINISH
# ============================================================

print()
print("========================================")
print("VISUALIZATION SELESAI")
print("========================================")
print(f"Output tersimpan di:")
print(OUTPUT_DIR)
print()
print("File yang dibuat:")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("-", filename)