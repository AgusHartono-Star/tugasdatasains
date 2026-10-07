"""
============================================================
JUDUL PROYEK
============================================================

Prediksi Harga EUR/USD Satu Jam Berikutnya Menggunakan
Machine Learning Random Forest Berbasis Data Historis
dan Indikator Teknikal
"""

"""
============================================================
BUSINESS QUESTIONS
============================================================

Proyek ini dikembangkan untuk menjawab beberapa pertanyaan
bisnis terkait analisis dan prediksi pergerakan harga EUR/USD
berdasarkan data historis hourly dan indikator teknikal.

1. BASIC QUESTION
   Bagaimana perkembangan harga EUR/USD berdasarkan data
   historis dengan interval satu jam?

   Pertanyaan ini digunakan untuk memahami karakteristik dasar
   dan pola pergerakan harga EUR/USD dari waktu ke waktu.

2. ANALYTICAL QUESTION
   Fitur atau indikator teknikal apa yang paling berpengaruh
   terhadap prediksi harga EUR/USD satu jam berikutnya?

   Pertanyaan ini dianalisis menggunakan feature importance
   dari model Random Forest untuk mengetahui fitur yang paling
   banyak digunakan model dalam menghasilkan prediksi.

3. ANALYTICAL QUESTION
   Seberapa akurat model Random Forest dalam memprediksi
   harga EUR/USD satu jam berikutnya dibandingkan dengan
   baseline sederhana?

   Pertanyaan ini dijawab menggunakan metrik evaluasi seperti
   MAE, RMSE, dan R² serta dibandingkan dengan persistence
   baseline.

4. AI / PREDICTION QUESTION
   Seberapa baik model Machine Learning dapat memprediksi
   harga close EUR/USD pada candle satu jam berikutnya
   berdasarkan data historis dan indikator teknikal?

   Pertanyaan ini menjadi fokus utama penggunaan AI/Machine
   Learning dalam proyek untuk menghasilkan prediksi
   next_close EUR/USD.
"""


import os
import json
import requests
import pandas as pd
import mysql.connector

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess
import sys

# =========================================================
# 1. KONFIGURASI
# =========================================================

API_URL = "https://api.twelvedata.com/time_series"

API_PARAMS = {
    "symbol": "EUR/USD",
    "interval": "1h",
    "outputsize": 5000,
}

RAW_FILE = "/tmp/forex_raw.json"
TRANSFORMED_FILE = "/tmp/forex_transformed.csv"


# =========================================================
# 2. EXTRACT
# =========================================================

def extract_data():

    print("=== EXTRACT ===")
    print("Mengambil data EUR/USD dari Twelve Data...")

    api_key = os.getenv("TWELVE_API_KEY")

    if not api_key:
        raise ValueError(
            "TWELVE_API_KEY tidak tersedia di environment Airflow."
        )

    params = API_PARAMS.copy()
    params["apikey"] = api_key

    response = requests.get(
        API_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if data.get("status") == "error":
        raise ValueError(
            f"Twelve Data API Error: {data.get('message')}"
        )

    values = data.get("values", [])

    if not values:
        raise ValueError(
            "Tidak ada data yang diterima dari API."
        )

    print("Data berhasil diambil.")
    print("Jumlah data:", len(values))

    # Simpan hasil extract sementara
    with open(RAW_FILE, "w", encoding="utf-8") as file:
        json.dump(values, file)

    print("Data raw disimpan:", RAW_FILE)


# =========================================================
# 3. TRANSFORM
# =========================================================

def transform_data():

    print("\n=== TRANSFORM ===")

    # Baca hasil Extract
    with open(RAW_FILE, "r", encoding="utf-8") as file:
        values = json.load(file)

    df = pd.DataFrame(values)

    print("Kolom awal:")
    print(df.columns.tolist())

    # Ambil kolom yang diperlukan
    df = df[
        [
            "datetime",
            "open",
            "high",
            "low",
            "close"
        ]
    ]

    # Konversi tipe data
    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    df["open"] = pd.to_numeric(
        df["open"],
        errors="coerce"
    )

    df["high"] = pd.to_numeric(
        df["high"],
        errors="coerce"
    )

    df["low"] = pd.to_numeric(
        df["low"],
        errors="coerce"
    )

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce"
    )

    print("\n=== SEBELUM CLEANING ===")

    print("Jumlah baris:", len(df))
    print("Jumlah kolom:", len(df.columns))

    print("\nMissing value:")
    print(df.isnull().sum())

    print("\nDuplicate:")
    print("Jumlah duplicate:", df.duplicated().sum())

    print("\nStatistik deskriptif:")
    print(df.describe())
    # =====================================================
    # CLEANING
    # =====================================================

    print("\n=== CLEANING ===")

    # Hapus duplicate
    df = df.drop_duplicates()

    # Hapus missing value
    df = df.dropna()

    # Hapus harga tidak valid
    df = df[
        (df["open"] > 0) &
        (df["high"] > 0) &
        (df["low"] > 0) &
        (df["close"] > 0)
    ]

    # Validasi OHLC
    df = df[
        (df["high"] >= df["low"]) &
        (df["high"] >= df["open"]) &
        (df["high"] >= df["close"]) &
        (df["low"] <= df["open"]) &
        (df["low"] <= df["close"])
    ]

    # Urutkan tanggal
    df = df.sort_values("datetime")

    # Reset index
    df = df.reset_index(drop=True)

    print("\n=== SETELAH CLEANING ===")

    print("Jumlah baris:", len(df))
    print("Jumlah kolom:", len(df.columns))

    print("\nMissing value:")
    print(df.isnull().sum())

    print("\nDuplicate:")
    print("Jumlah duplicate:", df.duplicated().sum())

    print("\nDuplicate datetime:")
    print("Jumlah duplicate datetime:", df["datetime"].duplicated().sum())

    print("\nRentang tanggal:")
    print("Awal :", df["datetime"].min())
    print("Akhir:", df["datetime"].max())

    print("\nPreview:")
    print(df.head())

    # Simpan hasil transform
    df.to_csv(
        TRANSFORMED_FILE,
        index=False
    )

    print("\nData hasil transform disimpan:")
    print(TRANSFORMED_FILE)


# =========================================================
# 4. LOAD
# =========================================================

def load_to_mysql():

    print("\n=== LOAD ===")

    # Baca hasil transform
    df = pd.read_csv(
        TRANSFORMED_FILE
    )

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    # Koneksi MySQL Laragon melalui host Windows
    connection = mysql.connector.connect(
        host="host.docker.internal",
        port=3306,
        user="root",
        password="",
        database="forex_db"
    )

    cursor = connection.cursor()

    print(
        "Berhasil terhubung ke MySQL Laragon."
    )

    insert_query = """
    INSERT INTO forex_prices
    (
        datetime,
        open,
        high,
        low,
        close
    )
    VALUES (%s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        open = VALUES(open),
        high = VALUES(high),
        low = VALUES(low),
        close = VALUES(close),
        updated_at = CURRENT_TIMESTAMP
    """

    rows = []

    for _, row in df.iterrows():

        rows.append(
            (
                row["datetime"].to_pydatetime(),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"])
            )
        )

    cursor.executemany(
        insert_query,
        rows
    )

    connection.commit()

    print(
        "Jumlah data diproses:",
        len(rows)
    )

    # Verifikasi
    cursor.execute(
        "SELECT COUNT(*) FROM forex_prices"
    )

    total_data = cursor.fetchone()[0]

    print(
        "Jumlah data dalam forex_prices:",
        total_data
    )

    cursor.close()
    connection.close()

    print("MySQL connection ditutup.")
    print("ETL EUR/USD selesai.")


# =========================================================
# 5. AIRFLOW DAG
# =========================================================
# =========================================================
# 4.1 FEATURE ENGINEERING
# =========================================================

def run_feature_engineering():
    print("=== FEATURE ENGINEERING ===")

    subprocess.run(
        [
            sys.executable,
            "/usr/local/airflow/dags/feature_engineering.py"
        ],
        check=True
    )

    print("Feature engineering selesai.")


# =========================================================
# 4.2 BUILD TRAINING DATASET
# =========================================================

def run_build_training_dataset():
    print("=== BUILD TRAINING DATASET ===")

    subprocess.run(
        [
            sys.executable,
            "/usr/local/airflow/dags/build_training_dataset.py"
        ],
        check=True
    )

    print("Training dataset berhasil diperbarui.")
    

with DAG(
    dag_id="forex_etl",
    start_date=datetime(2026, 9, 29),
    schedule="@hourly",
    catchup=False,
    tags=["forex", "etl", "twelvedata"],
) as dag:

    extract = PythonOperator(
        task_id="extract_data",
        python_callable=extract_data
    )

    transform = PythonOperator(
        task_id="transform_data",
        python_callable=transform_data
    )

    load = PythonOperator(
        task_id="load_to_mysql",
        python_callable=load_to_mysql
    )

    feature_engineering = PythonOperator(
        task_id="feature_engineering",
        python_callable=run_feature_engineering
    )

    build_dataset = PythonOperator(
        task_id="build_training_dataset",
        python_callable=run_build_training_dataset
    )

    extract >> transform >> load
    load >> feature_engineering >> build_dataset
