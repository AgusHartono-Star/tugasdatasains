import pandas as pd
import mysql.connector
import numpy as np

from sklearn.ensemble import RandomForestRegressor
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
cursor = conn.cursor()

print("Berhasil terhubung ke MySQL.")


# ============================================================
# 2. AMBIL TRAINING DATASET
# ============================================================

query = """
SELECT
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

print(
    f"Periode data: "
    f"{df['datetime'].min()} sampai {df['datetime'].max()}"
)


# ============================================================
# 4. TENTUKAN FEATURES DAN TARGET
# ============================================================

feature_columns = [
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

target_column = "next_close"

X = df[feature_columns]
y = df[target_column]


# ============================================================
# 5. TRAIN / TEST SPLIT BERDASARKAN WAKTU
# ============================================================

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(f"\nData training : {len(X_train)}")
print(f"Data testing  : {len(X_test)}")

print(
    f"Training sampai: "
    f"{df.iloc[split_index - 1]['datetime']}"
)

print(
    f"Testing mulai : "
    f"{df.iloc[split_index]['datetime']}"
)


# ============================================================
# 6. BUAT MODEL
# ============================================================

print("\nMembuat Random Forest Regressor...")

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 7. TRAIN MODEL
# ============================================================

print("Training model...")

model.fit(X_train, y_train)

print("Training selesai.")


# ============================================================
# 8. PREDIKSI DATA TEST
# ============================================================

print("Melakukan prediksi...")

y_pred = model.predict(X_test)

print("Prediksi selesai.")


# ============================================================
# 9. EVALUASI MODEL
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\n========================================")
print("HASIL EVALUASI MODEL")
print("========================================")

print(f"MAE  : {mae:.8f}")
print(f"RMSE : {rmse:.8f}")
print(f"R²   : {r2:.6f}")

print("========================================")


# ============================================================
# 10. CONTOH HASIL PREDIKSI
# ============================================================

results = pd.DataFrame({
    "datetime": df.iloc[split_index:]["datetime"].values,
    "actual": y_test.values,
    "predicted": y_pred
})

print("\n5 hasil prediksi pertama:")

print(
    results.head(5).to_string(index=False)
)


# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\n10 feature paling berpengaruh:")

print(
    importance.head(10).to_string(index=False)
)


# ============================================================
# 12. SIMPAN HASIL PREDIKSI KE predictions
# ============================================================

print("\nMembersihkan predictions lama...")

cursor.execute("DELETE FROM predictions")

conn.commit()


insert_query = """
INSERT INTO predictions (
    prediction_date,
    actual_close,
    predicted_close,
    error_value,
    absolute_error,
    model_name
)

VALUES (
    %s, %s, %s, %s, %s, %s
)
"""


prediction_data = []

for i in range(len(results)):

    actual = float(results.iloc[i]["actual"])
    predicted = float(results.iloc[i]["predicted"])

    error = actual - predicted
    absolute_error = abs(error)

    prediction_data.append(
        (
            results.iloc[i]["datetime"].to_pydatetime(),
            actual,
            predicted,
            error,
            absolute_error,
            "RandomForestRegressor"
        )
    )


cursor.executemany(
    insert_query,
    prediction_data
)

conn.commit()

print(
    f"Berhasil menyimpan "
    f"{cursor.rowcount} prediksi."
)


# ============================================================
# 13. VERIFIKASI predictions
# ============================================================

cursor.execute(
    "SELECT COUNT(*) FROM predictions"
)

prediction_count = cursor.fetchone()[0]

print(
    f"Total predictions sekarang: "
    f"{prediction_count}"
)


cursor.execute("""
    SELECT
        prediction_date,
        actual_close,
        predicted_close,
        error_value,
        absolute_error,
        model_name
    FROM predictions
    ORDER BY prediction_date DESC
    LIMIT 5
""")


print("\n5 prediksi terbaru:")

for row in cursor.fetchall():
    print(row)


# ============================================================
# 14. TUTUP KONEKSI
# ============================================================

cursor.close()
conn.close()

print("\nTraining model selesai.")