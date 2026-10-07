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
# 1. CONNECT MYSQL
# ============================================================

print("Menghubungkan ke MySQL...")

conn = mysql.connector.connect(**DB_CONFIG)
cursor = conn.cursor()

print("Berhasil terhubung ke MySQL.")


# ============================================================
# 2. AMBIL DATA GABUNGAN
# ============================================================

query = """
SELECT
    fp.id AS price_id,
    fp.datetime,
    fp.open,
    fp.high,
    fp.low,
    fp.close,

    pf.price_change,
    pf.return_1d,
    pf.daily_range,
    pf.daily_range_pct,
    pf.volatility_7d,
    pf.volatility_14d,
    pf.volatility_30d,
    pf.lag_close_1,
    pf.lag_close_2,
    pf.lag_close_3,
    pf.lag_close_7,

    ti.sma_7,
    ti.sma_14,
    ti.sma_30,
    ti.ema_7,
    ti.ema_14,
    ti.ema_30,
    ti.rsi_14,
    ti.macd,
    ti.macd_signal,
    ti.macd_hist,
    ti.bollinger_upper,
    ti.bollinger_middle,
    ti.bollinger_lower

FROM forex_prices fp

INNER JOIN price_features pf
    ON fp.id = pf.price_id

INNER JOIN technical_indicators ti
    ON fp.id = ti.price_id

ORDER BY fp.datetime ASC
"""

df = pd.read_sql(query, conn)

print(f"Data hasil penggabungan: {len(df)} baris")


# ============================================================
# 3. PERSIAPAN DATA
# ============================================================

df["datetime"] = pd.to_datetime(df["datetime"])

df = df.sort_values("datetime").reset_index(drop=True)


# ============================================================
# 4. BUAT TARGET next_close
# ============================================================

# Target = harga close satu jam berikutnya
df["next_close"] = df["close"].shift(-1)


# ============================================================
# 5. HAPUS DATA YANG TIDAK LENGKAP
# ============================================================

df = df.dropna().reset_index(drop=True)

print(f"Training rows setelah cleaning: {len(df)}")


# ============================================================
# 6. PILIH KOLOM TRAINING DATASET
# ============================================================

training_columns = [
    "price_id",
    "datetime",
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

training_df = df[training_columns].copy()


# ============================================================
# 7. BERSIHKAN training_dataset LAMA
# ============================================================

cursor.execute("DELETE FROM training_dataset")

conn.commit()

print("training_dataset lama dibersihkan.")


# ============================================================
# 8. INSERT TRAINING DATASET
# ============================================================

insert_query = """
INSERT INTO training_dataset (
    price_id,
    datetime,
    open,
    high,
    low,
    close,

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
    lag_close_7,

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
    bollinger_lower,

    next_close
)

VALUES (
    %s, %s, %s, %s, %s, %s,

    %s, %s, %s, %s, %s, %s, %s,

    %s, %s, %s, %s,

    %s, %s, %s,

    %s, %s, %s,

    %s,

    %s, %s, %s,

    %s, %s, %s,

    %s
)

ON DUPLICATE KEY UPDATE
    datetime = VALUES(datetime),
    open = VALUES(open),
    high = VALUES(high),
    low = VALUES(low),
    close = VALUES(close),

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
    lag_close_7 = VALUES(lag_close_7),

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
    bollinger_lower = VALUES(bollinger_lower),

    next_close = VALUES(next_close)
"""


# ============================================================
# 9. SIAPKAN DATA UNTUK MYSQL
# ============================================================

data = []

for _, row in training_df.iterrows():

    data.append(
        (
            int(row["price_id"]),
            row["datetime"].to_pydatetime(),

            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),

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

            float(row["next_close"])
        )
    )


# ============================================================
# 10. INSERT
# ============================================================

cursor.executemany(
    insert_query,
    data
)

conn.commit()

print(
    f"Berhasil memasukkan {cursor.rowcount} operasi "
    "ke training_dataset."
)


# ============================================================
# 11. VERIFIKASI
# ============================================================

cursor.execute(
    "SELECT COUNT(*) FROM training_dataset"
)

count = cursor.fetchone()[0]

print(
    f"Total training_dataset sekarang: {count}"
)


# ============================================================
# 12. LIHAT 5 DATA TERBARU
# ============================================================

cursor.execute("""
    SELECT
        price_id,
        datetime,
        close,
        rsi_14,
        macd,
        next_close
    FROM training_dataset
    ORDER BY datetime DESC
    LIMIT 5
""")

print("\n5 training data terbaru:")

for row in cursor.fetchall():
    print(row)


# ============================================================
# 13. TUTUP KONEKSI
# ============================================================

cursor.close()
conn.close()

print("\nTraining dataset selesai.")