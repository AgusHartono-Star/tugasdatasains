import pandas as pd
import mysql.connector
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# KONFIGURASI MYSQL
# ============================================================

DB_CONFIG = {
    "host": "127.0.0.1",
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

print("Berhasil terhubung ke MySQL.")


# ============================================================
# 2. AMBIL TRAINING DATASET
# ============================================================

query = """
SELECT
    price_id,
    datetime,
    close,
    next_close
FROM training_dataset
ORDER BY datetime ASC
"""

df = pd.read_sql(query, conn)

print(f"Training dataset: {len(df)} baris")


# ============================================================
# 3. PERSIAPAN DATA
# ============================================================

df["datetime"] = pd.to_datetime(df["datetime"])

df = df.sort_values("datetime").reset_index(drop=True)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(df) * 0.80)

test_df = df.iloc[split_index:].copy()

print(f"Data testing: {len(test_df)} baris")

print(
    f"Testing mulai: "
    f"{test_df['datetime'].iloc[0]}"
)

print(
    f"Testing sampai: "
    f"{test_df['datetime'].iloc[-1]}"
)


# ============================================================
# 5. BASELINE PREDICTION
# ============================================================

# Prediksi sederhana:
# harga satu jam berikutnya = harga saat ini

y_actual = test_df["next_close"]

y_baseline = test_df["close"]


# ============================================================
# 6. EVALUASI
# ============================================================

mae = mean_absolute_error(
    y_actual,
    y_baseline
)

rmse = np.sqrt(
    mean_squared_error(
        y_actual,
        y_baseline
    )
)

r2 = r2_score(
    y_actual,
    y_baseline
)


# ============================================================
# 7. HASIL
# ============================================================

print("\n========================================")
print("HASIL BASELINE")
print("========================================")

print(
    "Metode: next_close = close saat ini"
)

print(f"MAE  : {mae:.8f}")
print(f"RMSE : {rmse:.8f}")
print(f"R²   : {r2:.6f}")

print("========================================")


# ============================================================
# 8. CONTOH PREDIKSI
# ============================================================

print("\n5 contoh baseline:")

example = test_df[
    [
        "datetime",
        "close",
        "next_close"
    ]
].copy()

example["baseline_prediction"] = example["close"]

print(
    example.head(5).to_string(index=False)
)


# ============================================================
# 9. TUTUP
# ============================================================

conn.close()

print("\nBaseline evaluation selesai.")