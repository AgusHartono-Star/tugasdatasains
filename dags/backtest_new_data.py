import pandas as pd
import numpy as np
import mysql.connector

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)


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
# CONNECT
# ============================================================

print("Menghubungkan ke MySQL...")

conn = mysql.connector.connect(**DB_CONFIG)

print("Berhasil terhubung.")


# ============================================================
# AMBIL DATA RAW
# ============================================================

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

df["datetime"] = pd.to_datetime(df["datetime"])

df = df.sort_values("datetime").reset_index(drop=True)

print(f"Total data raw: {len(df)}")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("Membuat features...")

# Price features
df["price_change"] = df["close"].diff()

df["return_1d"] = df["close"].pct_change()

df["daily_range"] = (
    df["high"] - df["low"]
)

df["daily_range_pct"] = (
    df["daily_range"] / df["close"]
)

df["volatility_7d"] = (
    df["return_1d"].rolling(7).std()
)

df["volatility_14d"] = (
    df["return_1d"].rolling(14).std()
)

df["volatility_30d"] = (
    df["return_1d"].rolling(30).std()
)

df["lag_close_1"] = df["close"].shift(1)
df["lag_close_2"] = df["close"].shift(2)
df["lag_close_3"] = df["close"].shift(3)
df["lag_close_7"] = df["close"].shift(7)


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

# SMA
df["sma_7"] = (
    df["close"].rolling(7).mean()
)

df["sma_14"] = (
    df["close"].rolling(14).mean()
)

df["sma_30"] = (
    df["close"].rolling(30).mean()
)


# EMA
df["ema_7"] = (
    df["close"].ewm(
        span=7,
        adjust=False
    ).mean()
)

df["ema_14"] = (
    df["close"].ewm(
        span=14,
        adjust=False
    ).mean()
)

df["ema_30"] = (
    df["close"].ewm(
        span=30,
        adjust=False
    ).mean()
)


# RSI
delta = df["close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss.replace(0, np.nan)

df["rsi_14"] = (
    100 - (100 / (1 + rs))
)


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

df["macd_signal"] = (
    df["macd"]
    .ewm(span=9, adjust=False)
    .mean()
)

df["macd_hist"] = (
    df["macd"] -
    df["macd_signal"]
)


# Bollinger
bb_middle = (
    df["close"]
    .rolling(20)
    .mean()
)

bb_std = (
    df["close"]
    .rolling(20)
    .std()
)

df["bollinger_middle"] = bb_middle

df["bollinger_upper"] = (
    bb_middle + (2 * bb_std)
)

df["bollinger_lower"] = (
    bb_middle - (2 * bb_std)
)


# ============================================================
# FEATURES YANG DIPAKAI MODEL
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


# ============================================================
# FUNGSI BACKTEST SATU TITIK
# ============================================================

def run_backtest(cutoff_time, target_time):

    print("\n" + "=" * 60)
    print(f"CUTOFF  : {cutoff_time}")
    print(f"TARGET  : {target_time}")
    print("=" * 60)

    cutoff = pd.Timestamp(cutoff_time)
    target = pd.Timestamp(target_time)

    # --------------------------------------------------------
    # DATA TRAIN
    # --------------------------------------------------------

    train = df[
        df["datetime"] <= cutoff
    ].copy()

    # --------------------------------------------------------
    # DATA TARGET
    # --------------------------------------------------------

    target_row = df[
        df["datetime"] == target
    ].copy()

    if target_row.empty:
        print("Target tidak ditemukan.")
        return None

    # Pastikan feature target tersedia
    target_row = target_row.dropna(
        subset=feature_columns
    )

    train = train.dropna(
        subset=feature_columns + ["close"]
    )

    if target_row.empty:
        print("Feature target belum lengkap.")
        return None

    # --------------------------------------------------------
    # TRAIN FEATURES
    # --------------------------------------------------------

    X_train = train[feature_columns]

    y_train = train["close"].shift(-1)

    # Target training:
    # close berikutnya
    X_train = X_train.iloc[:-1]
    y_train = y_train.iloc[:-1]

    # Buang NaN
    valid = y_train.notna()

    X_train = X_train.loc[valid]
    y_train = y_train.loc[valid]

    # --------------------------------------------------------
    # TARGET FEATURES
    # --------------------------------------------------------

    X_target = target_row[
        feature_columns
    ]

    actual = float(
        target_row["close"].iloc[0]
    )

    previous_close = float(
        df[
            df["datetime"] < target
        ]["close"].iloc[-1]
    )

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline_prediction = previous_close

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(
        X_train,
        y_train
    )

    rf_prediction = float(
        rf.predict(X_target)[0]
    )

    # --------------------------------------------------------
    # GRADIENT BOOSTING
    # --------------------------------------------------------

    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=5,
        random_state=42
    )

    gb.fit(
        X_train,
        y_train
    )

    gb_prediction = float(
        gb.predict(X_target)[0]
    )

    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    baseline_error = actual - baseline_prediction
    rf_error = actual - rf_prediction
    gb_error = actual - gb_prediction

    print(f"\nActual close : {actual:.8f}")
    print(
        f"Previous close: {previous_close:.8f}"
    )

    print("\nBASELINE")
    print(
        f"Prediction : {baseline_prediction:.8f}"
    )
    print(
        f"Error      : {baseline_error:.8f}"
    )
    print(
        f"Abs Error  : {abs(baseline_error):.8f}"
    )

    print("\nRANDOM FOREST")
    print(
        f"Prediction : {rf_prediction:.8f}"
    )
    print(
        f"Error      : {rf_error:.8f}"
    )
    print(
        f"Abs Error  : {abs(rf_error):.8f}"
    )

    print("\nGRADIENT BOOSTING")
    print(
        f"Prediction : {gb_prediction:.8f}"
    )
    print(
        f"Error      : {gb_error:.8f}"
    )
    print(
        f"Abs Error  : {abs(gb_error):.8f}"
    )

    return {
        "target_time": target,
        "actual": actual,

        "baseline": baseline_prediction,
        "rf": rf_prediction,
        "gb": gb_prediction,

        "baseline_abs_error":
            abs(baseline_error),

        "rf_abs_error":
            abs(rf_error),

        "gb_abs_error":
            abs(gb_error)
    }


# ============================================================
# BACKTEST 1
# 22:00 → 23:00
# ============================================================

def run_backtest(cutoff_time, target_time):

    target = pd.Timestamp(target_time)

    data = df.sort_values("datetime").reset_index(drop=True)

    target_indices = data.index[
        data["datetime"] == target
    ].tolist()

    if not target_indices:
        print(f"Target tidak ditemukan: {target}")
        return None

    target_idx = target_indices[0]
    feature_idx = target_idx - 1

    if feature_idx < 100:
        print("Data training belum cukup.")
        return None

    # Fitur berasal dari candle yang sudah selesai
    X_target = data.loc[
        [feature_idx], feature_columns
    ]

    # Label training adalah close candle berikutnya.
    # Label harus sudah diketahui sebelum target diprediksi.
    training = data.iloc[:target_idx].copy()
    training["target_close"] = training["close"].shift(-1)

    training = training.dropna(
        subset=feature_columns + ["target_close"]
    )

    if training.empty or X_target.isna().any(axis=None):
        print("Data training atau fitur target tidak lengkap.")
        return None

    X_train = training[feature_columns]
    y_train = training["target_close"]

    actual = float(data.loc[target_idx, "close"])
    previous_close = float(data.loc[feature_idx, "close"])

    # Baseline
    baseline_prediction = previous_close

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_prediction = float(rf.predict(X_target)[0])

    # Gradient Boosting
    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=5,
        random_state=42
    )
    gb.fit(X_train, y_train)
    gb_prediction = float(gb.predict(X_target)[0])

    baseline_error = actual - baseline_prediction
    rf_error = actual - rf_prediction
    gb_error = actual - gb_prediction

    print(f"\nTARGET: {target}")
    print(f"Actual close       : {actual:.8f}")
    print(f"Previous close     : {previous_close:.8f}")
    print(f"Baseline prediction: {baseline_prediction:.8f}")
    print(f"Random Forest      : {rf_prediction:.8f}")
    print(f"Gradient Boosting  : {gb_prediction:.8f}")

    return {
        "target_time": target,
        "actual": actual,
        "baseline": baseline_prediction,
        "rf": rf_prediction,
        "gb": gb_prediction,
        "baseline_abs_error": abs(baseline_error),
        "rf_abs_error": abs(rf_error),
        "gb_abs_error": abs(gb_error)
    }