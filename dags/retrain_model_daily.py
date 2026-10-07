import os
import pandas as pd
import mysql.connector
import numpy as np

from joblib import dump

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
# KONFIGURASI MODEL
# ============================================================

MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models"
)

MODEL_DIR = os.path.abspath(MODEL_DIR)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "random_forest_model.joblib"
)


# ============================================================
# FEATURE
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

df = (
    df
    .sort_values("datetime")
    .reset_index(drop=True)
)

print(
    f"Periode data: "
    f"{df['datetime'].min()} "
    f"sampai "
    f"{df['datetime'].max()}"
)


# ============================================================
# 4. FEATURES DAN TARGET
# ============================================================

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

print("\n========================================")
print("TRAIN / TEST SPLIT")
print("========================================")

print(f"Data training : {len(X_train)}")
print(f"Data testing  : {len(X_test)}")

print(
    f"Training sampai : "
    f"{df.iloc[split_index - 1]['datetime']}"
)

print(
    f"Testing mulai  : "
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
# 8. PREDIKSI
# ============================================================

print("Melakukan prediksi pada data testing...")

y_pred = model.predict(X_test)

print("Prediksi selesai.")



# ============================================================
# 9. EVALUASI RANDOM FOREST DAN BASELINE
# ============================================================

# Evaluasi Random Forest
mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

# Baseline: prediksi harga berikutnya sama dengan close saat ini
baseline_pred = X_test["close"].to_numpy()

baseline_mae = mean_absolute_error(
    y_test,
    baseline_pred
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_pred
    )
)

baseline_r2 = r2_score(
    y_test,
    baseline_pred
)

# Menghitung persentase peningkatan dibandingkan baseline
mae_improvement = (
    (baseline_mae - mae) / baseline_mae * 100
    if baseline_mae > 0 else 0
)

rmse_improvement = (
    (baseline_rmse - rmse) / baseline_rmse * 100
    if baseline_rmse > 0 else 0
)

print("\n========================================")
print("PERBANDINGAN RANDOM FOREST VS BASELINE")
print("========================================")

print(f"{'Metrik':<12}{'Random Forest':>16}{'Baseline':>16}")
print("-" * 44)
print(f"{'MAE':<12}{mae:>16.8f}{baseline_mae:>16.8f}")
print(f"{'RMSE':<12}{rmse:>16.8f}{baseline_rmse:>16.8f}")
print(f"{'R²':<12}{r2:>16.6f}{baseline_r2:>16.6f}")

print("-" * 44)
print(f"Perbaikan MAE  : {mae_improvement:.2f}%")
print(f"Perbaikan RMSE : {rmse_improvement:.2f}%")

if mae < baseline_mae and rmse < baseline_rmse:
    print("Kesimpulan: Random Forest lebih baik pada kedua metrik.")
elif mae > baseline_mae and rmse > baseline_rmse:
    print("Kesimpulan: Baseline lebih baik pada kedua metrik.")
else:
    print("Kesimpulan: Hasil bercampur; evaluasi MAE dan RMSE secara terpisah.")

print("========================================")

# ============================================================
# 10. FEATURE IMPORTANCE
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
    importance
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 11. SIMPAN MODEL
# ============================================================

print("\nMenyimpan model...")

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

dump(
    model,
    MODEL_PATH
)

print(
    f"Model berhasil disimpan:"
)

print(
    MODEL_PATH
)


# ============================================================
# 12. TUTUP MYSQL
# ============================================================

conn.close()

print("\nRetraining selesai.")