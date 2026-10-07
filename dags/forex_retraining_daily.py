
import pendulum

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

import os
import pandas as pd
import mysql.connector
import numpy as np

from joblib import dump

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DB_CONFIG = {
    "host": "host.docker.internal",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "forex_db",
}

MODEL_DIR = "/usr/local/airflow/models"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.joblib")

feature_columns = [
    "open", "high", "low", "close",
    "price_change", "return_1d",
    "daily_range", "daily_range_pct",
    "volatility_7d", "volatility_14d", "volatility_30d",
    "lag_close_1", "lag_close_2", "lag_close_3", "lag_close_7",
    "sma_7", "sma_14", "sma_30",
    "ema_7", "ema_14", "ema_30",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bollinger_upper", "bollinger_middle", "bollinger_lower",
]

target_column = "next_close"


def retrain_model():
    conn = mysql.connector.connect(**DB_CONFIG)

    try:
        query = """
        SELECT
            price_id, datetime,
            open, high, low, close,
            price_change, return_1d,
            daily_range, daily_range_pct,
            volatility_7d, volatility_14d, volatility_30d,
            lag_close_1, lag_close_2, lag_close_3, lag_close_7,
            sma_7, sma_14, sma_30,
            ema_7, ema_14, ema_30,
            rsi_14, macd, macd_signal, macd_hist,
            bollinger_upper, bollinger_middle, bollinger_lower,
            next_close
        FROM training_dataset
        ORDER BY datetime ASC
        """

        df = pd.read_sql(query, conn)

    finally:
        conn.close()

    if len(df) < 100:
        raise ValueError(
            f"Dataset terlalu sedikit untuk retraining: {len(df)} baris."
        )

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    df = df.dropna(
        subset=feature_columns + [target_column]
    ).reset_index(drop=True)

    if len(df) < 100:
        raise ValueError(
            f"Data valid setelah cleaning hanya {len(df)} baris."
        )

    X = df[feature_columns]
    y = df[target_column]

    split_index = int(len(df) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Jumlah dataset valid: {len(df)}")
    print(f"Data training: {len(X_train)}")
    print(f"Data testing: {len(X_test)}")
    print(f"Periode terbaru: {df['datetime'].max()}")
    print(f"MAE: {mae:.8f}")
    print(f"RMSE: {rmse:.8f}")
    print(f"R2: {r2:.6f}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    dump(model, MODEL_PATH)

    print(f"Model tersimpan: {MODEL_PATH}")


with DAG(
    dag_id="forex_retraining_daily",
    start_date=pendulum.datetime(2026, 9, 30, tz="Asia/Makassar"),
    schedule="@daily",
    catchup=False,
    tags=["forex", "machine-learning", "retraining"],
) as dag:

    retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model
    )
