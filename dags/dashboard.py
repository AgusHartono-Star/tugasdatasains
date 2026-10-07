import os
import textwrap
from datetime import timedelta
import joblib
import mysql.connector
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
# ==================================================
# CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="EUR/USD Market Dashboard",
    page_icon="📈",
    layout="wide",
)
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "random_forest_model.joblib"
)
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "forex_db",
    "connection_timeout": 5,
}
FEATURE_COLUMNS = [
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
st_autorefresh(
    interval=60_000,
    key="forex_auto_refresh",
)
# ==================================================
# CUSTOM CSS
# ==================================================
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(
            circle at 15% 0%,
            rgba(37,99,235,.13),
            transparent 28%
        ),
        linear-gradient(180deg,#0b1120,#0b1220);
    color: #e5edf8;
}
[data-testid="stSidebar"] {
    background: #0e1729;
    border-right: 1px solid #263449;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1500px;
}
.hero {
    background: linear-gradient(
        120deg,
        rgba(30,58,138,.40),
        rgba(17,26,46,.96) 58%,
        rgba(6,95,70,.20)
    );
    border: 1px solid rgba(148,163,184,.20);
    border-radius: 22px;
    padding: 28px;
    margin-bottom: 24px;
}
.eyebrow {
    color: #93c5fd;
    font-size: .75rem;
    font-weight: 800;
    letter-spacing: .16em;
    margin-bottom: 8px;
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -.04em;
    color: #f8fafc;
}
.hero-subtitle {
    color: #a8b7cc;
    margin-top: 8px;
    font-size: .95rem;
}
.status-pill {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(52,211,153,.12);
    color: #6ee7b7;
    border: 1px solid rgba(52,211,153,.24);
    font-size: .78rem;
    font-weight: 700;
    margin-top: 15px;
}
.metric-card {
    background: linear-gradient(145deg,#162238,#0f182a);
    border: 1px solid rgba(148,163,184,.18);
    border-radius: 17px;
    padding: 19px;
    min-height: 125px;
}
.metric-label {
    color: #9aabc2;
    font-size: .75rem;
    font-weight: 700;
    margin-bottom: 12px;
}
.metric-value {
    color: #f1f5f9;
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -.035em;
    overflow-wrap: anywhere;
}
.metric-note {
    color: #94a3b8;
    font-size: .76rem;
    margin-top: 7px;
}
.section-heading {
    font-size: 1.08rem;
    font-weight: 800;
    margin: 12px 0;
    color: #e5edf8;
}
.small-muted {
    color: #94a3b8;
    font-size: .82rem;
}
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(148,163,184,.18);
    border-radius: 12px;
    overflow: hidden;
}
.stButton > button {
    border-radius: 10px;
    border: 1px solid rgba(96,165,250,.35);
    background: rgba(37,99,235,.18);
    color: #dbeafe;
    font-weight: 700;
}
.stButton > button:hover {
    border-color: #60a5fa;
    color: white;
}
hr {
    border-color: rgba(148,163,184,.18);
}
</style>
""", unsafe_allow_html=True)
# ==================================================
# DATABASE AND MODEL
# ==================================================
@st.cache_resource
def load_model(model_path, modified_time):
    return joblib.load(model_path)
def get_connection():
    return mysql.connector.connect(**DB_CONFIG)
def fetch_dataset():
    columns = ["datetime"] + FEATURE_COLUMNS + ["next_close"]
    selected_columns = ", ".join(
        f"`{column}`" for column in columns
    )
    query = f"""
        SELECT {selected_columns}
        FROM training_dataset
        ORDER BY datetime DESC
        LIMIT 9000
    """
    connection = get_connection()
    try:
        df = pd.read_sql(query, connection)
    finally:
        connection.close()
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(
        df["datetime"], errors="coerce"
    )
    df = df.dropna(subset=["datetime"])
    return (
        df.sort_values("datetime")
        .reset_index(drop=True)
    )
def fetch_database_status():
    connection = get_connection()
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM forex_prices"
        )
        raw_rows = cursor.fetchone()[0]
        cursor.execute(
            "SELECT COUNT(*) FROM training_dataset"
        )
        training_rows = cursor.fetchone()[0]
        cursor.execute(
            "SELECT MAX(datetime) FROM forex_prices"
        )
        latest_datetime = cursor.fetchone()[0]
        cursor.close()
        return raw_rows, training_rows, latest_datetime
    finally:
        connection.close()
def format_price(value):
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.5f}"
def metric_card(label, value, note="", accent="#e5edf8"):
    st.markdown(
        textwrap.dedent(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{accent}">
                {value}
            </div>
            <div class="metric-note">{note}</div>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )
# ==================================================
# HEADER
# ==================================================
st.markdown("""
<div class="hero">
    <div class="eyebrow">FOREX · MACHINE LEARNING ANALYTICS</div>
    <div class="hero-title">EUR/USD Market Dashboard</div>
    <div class="hero-subtitle">
        Analisis harga, indikator teknikal, dan estimasi
        candle berikutnya menggunakan Random Forest.
    </div>
    <span class="status-pill">
        Dashboard aktif - Auto-refresh 60 detik
    </span>
</div>
""", unsafe_allow_html=True)
# ==================================================
# SIDEBAR
# ==================================================
# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown("## Pengaturan Dashboard")
    st.divider()
    st.markdown("**Instrumen**")
    st.markdown("EUR / USD")
    st.markdown("**Interval**")
    st.markdown("1 Hour")
    st.markdown("**Model**")
    st.markdown("Random Forest Regressor")
    st.divider()
    st.markdown("**Database**")
    st.code(
        "localhost:3306\nforex_db",
        language=None,
    )
    if st.button(
        "Refresh sekarang",
        use_container_width=True,
    ):
        st.rerun()
    st.caption(
        "Data diproses melalui pipeline Apache Airflow."
    )
# ==================================================
# LOAD DATA
# ==================================================
try:
    if not os.path.exists(MODEL_PATH):
        st.error("File model tidak ditemukan.")
        st.code(MODEL_PATH)
        st.stop()
    model = load_model(
        MODEL_PATH,
        os.path.getmtime(MODEL_PATH),
    )
    data = fetch_dataset()
    raw_rows, training_rows, latest_raw_datetime = (
        fetch_database_status()
    )
except Exception as error:
    st.error("Dashboard gagal mengambil data.")
    st.code(str(error))
    st.info(
        "Pastikan MySQL Laragon aktif dan tabel "
        "forex_db tersedia."
    )
    st.stop()
# ==================================================
# SYSTEM OVERVIEW
# ==================================================
st.markdown(
    '<div class="section-heading">Ringkasan Sistem</div>',
    unsafe_allow_html=True,
)
cols = st.columns(3)
with cols[0]:
    metric_card(
        "DATA HARGA MENTAH",
        f"{raw_rows:,}",
        "Baris forex_prices",
        "#93c5fd",
    )
with cols[1]:
    metric_card(
        "DATA TRAINING",
        f"{training_rows:,}",
        "Baris training_dataset",
        "#6ee7b7",
    )
with cols[2]:
    metric_card(
        "STATUS MODEL",
        "Tersedia",
        "Random Forest",
        "#c4b5fd",
    )
if data.empty:
    st.warning(
        "Tabel training_dataset kosong pada database "
        "yang dibaca dashboard. Periksa task Airflow."
    )
    st.stop()
# ==================================================
# PREDICTION
# ==================================================
latest = data.iloc[-1]
feature_row = (
    latest[FEATURE_COLUMNS]
    .to_frame()
    .T
    .apply(pd.to_numeric, errors="coerce")
)
if feature_row.isna().any().any():
    st.warning(
        "Fitur pada baris terbaru memiliki nilai kosong. "
        "Prediksi belum dapat ditampilkan."
    )
    prediction = None
else:
    try:
        prediction = float(
            model.predict(feature_row)[0]
        )
    except Exception as error:
        st.error("Model gagal membuat prediksi.")
        st.code(str(error))
        st.stop()
last_close = float(latest["close"])
if prediction is not None:
    difference = prediction - last_close
    percentage = (
        difference / last_close * 100
        if last_close != 0 else 0
    )
    if difference > 0:
        direction = "Naik"
        direction_color = "#34d399"
    elif difference < 0:
        direction = "Turun"
        direction_color = "#fb7185"
    else:
        direction = "Tidak berubah"
        direction_color = "#cbd5e1"
else:
    difference = None
    percentage = None
    direction = "Tidak tersedia"
    direction_color = "#cbd5e1"
# ==================================================
# PRICE SUMMARY
# ==================================================
st.markdown(
    '<div class="section-heading">Harga dan Prediksi</div>',
    unsafe_allow_html=True,
)
cols = st.columns(4)
with cols[0]:
    metric_card(
        "HARGA TERAKHIR",
        format_price(last_close),
        str(latest["datetime"]),
    )
with cols[1]:
    metric_card(
        "PREDIKSI BERIKUTNYA",
        format_price(prediction),
        "Estimasi Random Forest",
        "#93c5fd",
    )
with cols[2]:
    metric_card(
        "ARAH PREDIKSI",
        direction,
        "Dibanding harga terakhir",
        direction_color,
    )
with cols[3]:
    percentage_text = (
        f"{percentage:+.3f}%"
        if percentage is not None
        else "—"
    )
    metric_card(
        "PERUBAHAN PREDIKSI",
        percentage_text,
        "Perubahan relatif estimasi",
        direction_color,
    )
st.markdown(
    f'<div class="small-muted">'
    f'Data harga terbaru: '
    f'{latest_raw_datetime if latest_raw_datetime else "Tidak tersedia"}'
    f'</div>',
    unsafe_allow_html=True,
)
st.divider()
# ==================================================
# PRICE CHART
# ==================================================
left, right = st.columns([1.8, 1], gap="large")
with left:
    st.markdown(
        '<div class="section-heading">'
        'Perbandingan Harga Aktual, Random Forest, dan Baseline'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Grafik menggunakan 20% data terakhir sebagai data pengujian. "
        "Baseline memprediksi harga candle berikutnya dengan memakai "
        "harga close candle saat ini."
    )
    # Pastikan data berurutan secara kronologis.
    chart_source = data.sort_values("datetime").reset_index(drop=True).copy()
    # Pembagian kronologis 80% data awal dan 20% data terakhir.
    split_index = int(len(chart_source) * 0.80)
    test_data = chart_source.iloc[split_index:].copy()
    X_test_chart = test_data[FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    valid = (
        X_test_chart.notna().all(axis=1)
        & test_data["next_close"].notna()
        & test_data["close"].notna()
    )
    test_data = test_data.loc[valid].copy()
    X_test_chart = X_test_chart.loc[valid]
    if test_data.empty:
        st.warning(
            "Data pengujian tidak tersedia untuk grafik. "
            "Pastikan training_dataset memiliki data dan fitur lengkap."
        )
    else:
        # Prediksi Random Forest pada periode testing.
        rf_predictions = model.predict(X_test_chart)
        # next_close adalah harga candle berikutnya, jadi waktu target
        # digeser satu candle agar sejajar dengan harga aktual.
        target_times = chart_source["datetime"].shift(-1)
        target_times = target_times.loc[test_data.index].copy()
        target_times = target_times.fillna(
            test_data["datetime"] + pd.Timedelta(hours=1)
        )
        chart_data = pd.DataFrame(
            {
                "Waktu": target_times.to_numpy(),
                "Harga Aktual": test_data["next_close"].to_numpy(),
                "Random Forest": rf_predictions,
                "Baseline": test_data["close"].to_numpy(),
            }
        )
        chart_data = (
            chart_data.dropna()
            .sort_values("Waktu")
            .set_index("Waktu")
        )
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            chart_data.index,
            chart_data["Harga Aktual"],
            label="Harga Aktual",
            linewidth=1.2,
        )
        ax.plot(
            chart_data.index,
            chart_data["Random Forest"],
            label="Random Forest",
            linewidth=1.0,
        )
        ax.plot(
            chart_data.index,
            chart_data["Baseline"],
            label="Baseline",
            linewidth=1.0,
        )
        # Titik prediksi untuk candle satu jam berikutnya.
        # Variabel prediction dan latest sudah dihitung pada bagian PREDICTION.
        if prediction is not None:
            last_time = pd.to_datetime(latest["datetime"])
            next_time = last_time + pd.Timedelta(hours=1)
            ax.scatter(
                [next_time],
                [prediction],
                color="red",
                s=100,
                marker="o",
                edgecolors="white",
                linewidths=1.2,
                label="Prediksi Candle Berikutnya",
                zorder=10,
                clip_on=False,
            )
            ax.annotate(
                f"Prediksi: {prediction:.5f}",
                xy=(next_time, prediction),
                xytext=(8, 10),
                textcoords="offset points",
                color="red",
                fontsize=9,
                fontweight="bold",
                annotation_clip=False,
            )
            ax.set_xlim(
                right=next_time + pd.Timedelta(hours=2)
            )
        ax.set_xlabel("Waktu")
        ax.set_ylabel("Harga EUR/USD")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(
            f"Jumlah candle pengujian: {len(chart_data):,}. "
            "Ketiga garis memakai waktu target candle yang sama."
        )
with right:
    st.markdown(
        '<div class="section-heading">'
        'Indikator Teknikal'
        '</div>',
        unsafe_allow_html=True,
    )
    indicator_names = [
        ("rsi_14", "RSI (14)"),
        ("macd", "MACD"),
        ("macd_signal", "MACD Signal"),
        ("macd_hist", "MACD Histogram"),
        ("sma_7", "SMA 7"),
        ("sma_14", "SMA 14"),
        ("sma_30", "SMA 30"),
        ("ema_7", "EMA 7"),
        ("ema_14", "EMA 14"),
        ("ema_30", "EMA 30"),
        ("bollinger_upper", "Bollinger Upper"),
        ("bollinger_middle", "Bollinger Middle"),
        ("bollinger_lower", "Bollinger Lower"),
    ]
    indicator_rows = []
    for column, label in indicator_names:
        value = pd.to_numeric(
            pd.Series([latest[column]]),
            errors="coerce",
        ).iloc[0]
        indicator_rows.append({
            "Indikator": label,
            "Nilai": (
                "—"
                if pd.isna(value)
                else f"{float(value):.6f}"
            ),
        })
    st.dataframe(
        pd.DataFrame(indicator_rows),
        use_container_width=True,
        hide_index=True,
        height=380,
    )
# ==================================================
# CANDLE HISTORY
# ==================================================
st.divider()
st.markdown(
    '<div class="section-heading">'
    'Riwayat Candle Terbaru'
    '</div>',
    unsafe_allow_html=True,
)
history = data[
    ["datetime", "open", "high", "low", "close"]
].copy()
history = history.sort_values(
    "datetime",
    ascending=False,
).head(20)
history = history.rename(
    columns={
        "datetime": "Waktu",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
    }
)
st.dataframe(
    history,
    use_container_width=True,
    hide_index=True,
)
# ==================================================
# FOOTER
# ==================================================
st.markdown("""
<div class="small-muted" style="margin-top:20px">
    Prediksi merupakan estimasi model, bukan jaminan
    pergerakan pasar atau saran investasi. Data terbaru
    bergantung pada proses ETL dan pembentukan dataset
    yang dijalankan melalui Airflow.
</div>
""", unsafe_allow_html=True)
