import pandas as pd
import mysql.connector


# ============================================================
# KONFIGURASI MYSQL
# ============================================================

DB_CONFIG = {
    "host": "host.docker.internal",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "forex_db",
}


# ============================================================
# 1. AMBIL DATA DARI forex_prices
# ============================================================

print("Menghubungkan ke MySQL...")

conn = mysql.connector.connect(**DB_CONFIG)

query = """
SELECT
    id,
    datetime,
    open,
    high,
    low,
    close
FROM forex_prices
ORDER BY datetime ASC
"""

df = pd.read_sql(query, conn)

print(f"Data forex_prices: {len(df)} baris")


# ============================================================
# 2. PERSIAPAN DATA
# ============================================================

df["datetime"] = pd.to_datetime(df["datetime"])

numeric_columns = [
    "open",
    "high",
    "low",
    "close"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_values("datetime").reset_index(drop=True)


# ============================================================
# 3. PRICE FEATURES
# ============================================================

# Perubahan harga dari candle sebelumnya
df["price_change"] = df["close"].diff()

# Return satu candle sebelumnya
df["return_1d"] = df["close"].pct_change()

# Range candle
df["daily_range"] = df["high"] - df["low"]

# Range dalam persentase terhadap close
df["daily_range_pct"] = (
    df["daily_range"] / df["close"]
)

# Volatilitas rolling hourly
df["volatility_7d"] = (
    df["return_1d"].rolling(window=7).std()
)

df["volatility_14d"] = (
    df["return_1d"].rolling(window=14).std()
)

df["volatility_30d"] = (
    df["return_1d"].rolling(window=30).std()
)

# Lag close
df["lag_close_1"] = df["close"].shift(1)
df["lag_close_2"] = df["close"].shift(2)
df["lag_close_3"] = df["close"].shift(3)
df["lag_close_7"] = df["close"].shift(7)


# ============================================================
# 4. PILIH KOLOM UNTUK price_features
# ============================================================

features = df[
    [
        "id",
        "datetime",
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
    ]
].copy()

features.rename(
    columns={"id": "price_id"},
    inplace=True
)

  
# ============================================================
# 5. HAPUS BARIS YANG BELUM MEMILIKI FITUR LENGKAP
# ============================================================

features = features.dropna().reset_index(drop=True)

print(f"Feature rows setelah cleaning: {len(features)}")

# ============================================================
# 5. TECHNICAL INDICATORS
# ============================================================

# SMA
df["sma_7"] = df["close"].rolling(window=7).mean()
df["sma_14"] = df["close"].rolling(window=14).mean()
df["sma_30"] = df["close"].rolling(window=30).mean()

# EMA
df["ema_7"] = df["close"].ewm(span=7, adjust=False).mean()
df["ema_14"] = df["close"].ewm(span=14, adjust=False).mean()
df["ema_30"] = df["close"].ewm(span=30, adjust=False).mean()

# RSI 14
delta = df["close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

rs = avg_gain / avg_loss.replace(0, pd.NA)

df["rsi_14"] = 100 - (100 / (1 + rs))

# MACD
ema_12 = df["close"].ewm(
    span=12,
    adjust=False
).mean()

ema_26 = df["close"].ewm(
    span=26,
    adjust=False
).mean()

df["macd"] = ema_12 - ema_26

df["macd_signal"] = df["macd"].ewm(
    span=9,
    adjust=False
).mean()

df["macd_hist"] = (
    df["macd"] - df["macd_signal"]
)

# Bollinger Bands
bb_middle = df["close"].rolling(
    window=20
).mean()

bb_std = df["close"].rolling(
    window=20
).std()

df["bollinger_middle"] = bb_middle
df["bollinger_upper"] = bb_middle + (2 * bb_std)
df["bollinger_lower"] = bb_middle - (2 * bb_std)


# ============================================================
# PREPARE TECHNICAL INDICATORS
# ============================================================

indicators = df[
    [
        "id",
        "datetime",
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
    ]
].copy()

indicators.rename(
    columns={"id": "price_id"},
    inplace=True
)

indicators = indicators.dropna().reset_index(drop=True)

print(
    f"Technical indicator rows: {len(indicators)}"
)


# ============================================================
# CLEAR OLD TECHNICAL INDICATORS
# ============================================================
cursor = conn.cursor()

cursor.execute(
    "DELETE FROM technical_indicators"
)

conn.commit()

print("technical_indicators lama dibersihkan.")


# ============================================================
# INSERT TECHNICAL INDICATORS
# ============================================================

indicator_query = """
INSERT INTO technical_indicators (
    price_id,
    datetime,
    sma_7,
    sma_14,
    sma_30,
    ema_7,
    ema_14,
    ema_30,
    rsi_14,
    macd,
    macd_signal,
    macd_hist,
    bollinger_upper,
    bollinger_middle,
    bollinger_lower
)
VALUES (
    %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s
)
ON DUPLICATE KEY UPDATE
    datetime = VALUES(datetime),
    sma_7 = VALUES(sma_7),
    sma_14 = VALUES(sma_14),
    sma_30 = VALUES(sma_30),
    ema_7 = VALUES(ema_7),
    ema_14 = VALUES(ema_14),
    ema_30 = VALUES(ema_30),
    rsi_14 = VALUES(rsi_14),
    macd = VALUES(macd),
    macd_signal = VALUES(macd_signal),
    macd_hist = VALUES(macd_hist),
    bollinger_upper = VALUES(bollinger_upper),
    bollinger_middle = VALUES(bollinger_middle),
    bollinger_lower = VALUES(bollinger_lower)
"""

indicator_data = []

for _, row in indicators.iterrows():
    indicator_data.append(
        (
            int(row["price_id"]),
            row["datetime"].to_pydatetime(),
            float(row["sma_7"]),
            float(row["sma_14"]),
            float(row["sma_30"]),
            float(row["ema_7"]),
            float(row["ema_14"]),
            float(row["ema_30"]),
            float(row["rsi_14"]),
            float(row["macd"]),
            float(row["macd_signal"]),
            float(row["macd_hist"]),
            float(row["bollinger_upper"]),
            float(row["bollinger_middle"]),
            float(row["bollinger_lower"]),
        )
    )

cursor.executemany(
    indicator_query,
    indicator_data
)

conn.commit()

print(
    f"Technical indicators inserted: {cursor.rowcount}"
)


# ============================================================
# VERIFY TECHNICAL INDICATORS
# ============================================================

cursor.execute(
    "SELECT COUNT(*) FROM technical_indicators"
)

indicator_count = cursor.fetchone()[0]

print(
    f"Total technical_indicators sekarang: "
    f"{indicator_count}"
)

cursor.execute("""
    SELECT
        price_id,
        datetime,
        sma_7,
        ema_7,
        rsi_14,
        macd,
        bollinger_middle
    FROM technical_indicators
    ORDER BY datetime DESC
    LIMIT 5
""")

print("\n5 technical indicators terbaru:")

for row in cursor.fetchall():
    print(row)
# ============================================================
# 6. BERSIHKAN price_features
# ============================================================

cursor = conn.cursor()

cursor.execute("DELETE FROM price_features")

conn.commit()

print("price_features lama dibersihkan.")


# ============================================================
# 7. INSERT KE price_features
# ============================================================

insert_query = """
INSERT INTO price_features (
    price_id,
    datetime,
    price_change,
    return_1d,
    daily_range,
    daily_range_pct,
    volatility_7d,
    volatility_14d,
    volatility_30d,
    lag_close_1,
    lag_close_2,
    lag_close_3,
    lag_close_7
)
VALUES (
    %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s
)
ON DUPLICATE KEY UPDATE
    datetime = VALUES(datetime),
    price_change = VALUES(price_change),
    return_1d = VALUES(return_1d),
    daily_range = VALUES(daily_range),
    daily_range_pct = VALUES(daily_range_pct),
    volatility_7d = VALUES(volatility_7d),
    volatility_14d = VALUES(volatility_14d),
    volatility_30d = VALUES(volatility_30d),
    lag_close_1 = VALUES(lag_close_1),
    lag_close_2 = VALUES(lag_close_2),
    lag_close_3 = VALUES(lag_close_3),
    lag_close_7 = VALUES(lag_close_7)
"""

data = []

for _, row in features.iterrows():
    data.append(
        (
            int(row["price_id"]),
            row["datetime"].to_pydatetime(),
            float(row["price_change"]),
            float(row["return_1d"]),
            float(row["daily_range"]),
            float(row["daily_range_pct"]),
            float(row["volatility_7d"]),
            float(row["volatility_14d"]),
            float(row["volatility_30d"]),
            float(row["lag_close_1"]),
            float(row["lag_close_2"]),
            float(row["lag_close_3"]),
            float(row["lag_close_7"]),
        )
    )

cursor.executemany(insert_query, data)

conn.commit()

print(f"Berhasil memasukkan {cursor.rowcount} operasi ke price_features.")


# ============================================================
# 8. VERIFIKASI
# ============================================================

cursor.execute(
    "SELECT COUNT(*) FROM price_features"
)

count = cursor.fetchone()[0]

print(f"Total price_features sekarang: {count}")

cursor.execute("""
    SELECT
        price_id,
        datetime,
        price_change,
        return_1d,
        volatility_7d,
        lag_close_1,
        lag_close_7
    FROM price_features
    ORDER BY datetime DESC
    LIMIT 5
""")

print("\n5 data fitur terbaru:")

for row in cursor.fetchall():
    print(row)


# ============================================================
# 9. TUTUP KONEKSI
# ============================================================

cursor.close()
conn.close()

print("\nFeature engineering selesai.")