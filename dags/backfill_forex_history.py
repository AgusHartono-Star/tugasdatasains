
import os
import time
import calendar
from datetime import date, timedelta

import requests
import mysql.connector


API_KEY = os.getenv("TWELVE_API_KEY")
API_URL = "https://api.twelvedata.com/time_series"

DB_CONFIG = {
    "host": "host.docker.internal",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "forex_db",
}

START_DATE = date(2025, 10, 7)
END_DATE = date(2026, 3, 5)


def fetch_month(session, start_date, end_date):
    params = {
        "symbol": "EUR/USD",
        "interval": "1h",
        "start_date": f"{start_date} 00:00:00",
        "end_date": f"{end_date} 23:59:59",
        "timezone": "UTC",
        "apikey": API_KEY,
    }

    response = session.get(API_URL, params=params, timeout=60)
    response.raise_for_status()
    result = response.json()

    if result.get("status") == "error" or "values" not in result:
        raise RuntimeError(
            f"Twelve Data error untuk {start_date} - {end_date}: "
            f"{result.get('message', result)}"
        )

    return result["values"]


def main():
    if not API_KEY:
        raise RuntimeError("TWELVE_API_KEY tidak tersedia.")

    conn = mysql.connector.connect(**DB_CONFIG)
    session = requests.Session()
    inserted = 0

    try:
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SHOW COLUMNS FROM forex_prices")
        schema = cursor.fetchall()
        columns = {row["Field"]: row for row in schema}

        required = {"datetime", "open", "high", "low", "close"}
        missing = required - set(columns)

        if missing:
            raise RuntimeError(
                f"Kolom tabel tidak sesuai: {sorted(missing)}"
            )

        # Pastikan tidak ada kolom wajib lain yang tidak dapat diisi.
        for row in schema:
            name = row["Field"]
            if (
                name not in required
                and name != "id"
                and row["Null"] == "NO"
                and row["Default"] is None
                and row["Extra"] == ""
            ):
                raise RuntimeError(
                    f"Kolom {name} wajib diisi dan tidak memiliki default. "
                    "Periksa struktur tabel sebelum melanjutkan."
                )

        update_columns = ["open", "high", "low", "close"]
        insert_columns = ["datetime", "open", "high", "low", "close"]

        sql = f"""
            INSERT INTO forex_prices
            ({", ".join(f"`{c}`" for c in insert_columns)})
            VALUES ({", ".join(["%s"] * len(insert_columns))})
            ON DUPLICATE KEY UPDATE
            {", ".join(f"`{c}` = VALUES(`{c}`)" for c in update_columns)}
        """

        # Ambil data per bulan agar setiap permintaan tetap kecil.
        month_start = START_DATE

        while month_start <= END_DATE:
            last_day = calendar.monthrange(
                month_start.year, month_start.month
            )[1]

            month_end = min(
                date(month_start.year, month_start.month, last_day),
                END_DATE,
            )

            print(f"Mengambil {month_start} sampai {month_end}...")

            values = fetch_month(session, month_start, month_end)
            rows = []

            for item in values:
                rows.append((
                    item["datetime"],
                    float(item["open"]),
                    float(item["high"]),
                    float(item["low"]),
                    float(item["close"]),
                ))

            if rows:
                cursor.executemany(sql, rows)
                conn.commit()
                inserted += len(rows)

            print(f"Data diterima: {len(rows)} baris")

            month_start = month_end + timedelta(days=1)
            time.sleep(1)

        cursor.execute("""
            SELECT COUNT(*) AS total,
                   MIN(datetime) AS tanggal_awal,
                   MAX(datetime) AS tanggal_akhir,
                   COUNT(DISTINCT datetime) AS waktu_unik
            FROM forex_prices
        """)

        print("\nVerifikasi forex_prices:")
        print(cursor.fetchone())
        print(f"Total baris yang diproses: {inserted}")

    finally:
        session.close()
        conn.close()


if __name__ == "__main__":
    main()
