"""
Export PostgreSQL tables from indian_real_estate DB to real_estate_data folder.
Credentials are loaded from the .env file in the project root.
"""

import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

# Load .env from the same directory as this script
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

DB_HOST     = os.getenv("PG_HOST")
DB_PORT     = os.getenv("PG_PORT", "5432")
DB_NAME     = os.getenv("PG_DB")
DB_USER     = os.getenv("PG_USER")
DB_PASSWORD = os.getenv("PG_PASSWORD")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "real_estate_data", "real_estate_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TABLES = ["ho_raw_data", "mb_raw_data","mb_rent","ho_rent"]


def export_table(conn, table_name: str) -> None:
    print(f"Exporting {table_name} ...", end=" ", flush=True)
    df = pd.read_sql(f'SELECT * FROM public."{table_name}"', conn)
    out_path = os.path.join(OUTPUT_DIR, f"{table_name}.csv")
    df.to_csv(out_path, index=False)
    print(f"done  →  {out_path}  ({len(df):,} rows)")


def main():
    missing = [k for k, v in {"PG_HOST": DB_HOST, "PG_DB": DB_NAME,
                               "PG_USER": DB_USER, "PG_PASSWORD": DB_PASSWORD}.items() if not v]
    if missing:
        raise ValueError(f"Missing .env keys: {', '.join(missing)}")

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    try:
        for table in TABLES:
            export_table(conn, table)
    finally:
        conn.close()
    print("\nAll tables exported successfully.")


if __name__ == "__main__":
    main()
