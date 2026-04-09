"""
Upload cleaned_data CSV files into PostgreSQL tables.

Source folder:
    artifact/data_transformation/cleaned_data

Target tables:
    apartment.csv     -> cleaned_apt
    builder_floor.csv -> cleaned_builder
    plot.csv          -> cleaned_plot
    res_house.csv     -> cleaned_res_house
    villa.csv         -> cleaned_villa

DB credentials are loaded from .env in project root:
    PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD
Optional:
    PG_SCHEMA (default: public)
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable

import pandas as pd
import psycopg2
from dotenv import load_dotenv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

DB_HOST = os.getenv("PG_HOST")
DB_PORT = os.getenv("PG_PORT", "5432")
DB_NAME = os.getenv("PG_DB")
DB_USER = os.getenv("PG_USER")
DB_PASSWORD = os.getenv("PG_PASSWORD")
DB_SCHEMA = os.getenv("PG_SCHEMA", "public")

CLEANED_DATA_DIR = os.path.join(SCRIPT_DIR, "artifact", "data_transformation", "cleaned_data")

FILE_TABLE_MAP = {
    "apartment": "cleaned_apt",
    "builder_floor": "cleaned_builder",
    "plot": "cleaned_plot",
    "res_house": "cleaned_res_house",
    "villa": "cleaned_villa",
}


def quote_ident(name: str) -> str:
    """Safely quote PostgreSQL identifiers."""
    return '"' + str(name).replace('"', '""') + '"'


def infer_pg_type(series: pd.Series) -> str:
    """Infer a PostgreSQL column type from a pandas Series.

    Notes:
    - Integer-like columns may still contain CSV values such as "1.0".
      Using BIGINT for those causes COPY failures, so numeric columns are
      intentionally mapped to DOUBLE PRECISION for robust loading.
    """
    dtype = series.dtype
    non_null = series.dropna()

    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(dtype):
        return "DOUBLE PRECISION"
    if pd.api.types.is_float_dtype(dtype):
        return "DOUBLE PRECISION"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"

    # Object/string columns: if values are fully numeric-like, load as numeric.
    if not non_null.empty:
        as_num = pd.to_numeric(non_null.astype(str).str.strip(), errors="coerce")
        if as_num.notna().all():
            return "DOUBLE PRECISION"

    return "TEXT"


def infer_schema_from_csv(csv_path: str) -> tuple[Dict[str, str], int]:
    """Read CSV once and infer column SQL types + row count."""
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.convert_dtypes()

    column_types = {col: infer_pg_type(df[col]) for col in df.columns}
    return column_types, len(df)


def create_or_replace_table(cursor, schema: str, table: str, column_types: Dict[str, str]) -> None:
    """Drop existing table and recreate using inferred schema."""
    schema_q = quote_ident(schema)
    table_q = quote_ident(table)

    cursor.execute(f"DROP TABLE IF EXISTS {schema_q}.{table_q}")

    col_defs = ",\n    ".join(
        f"{quote_ident(col)} {sql_type}" for col, sql_type in column_types.items()
    )
    create_sql = f"""
    CREATE TABLE {schema_q}.{table_q} (
        {col_defs}
    )
    """
    cursor.execute(create_sql)


def copy_csv_into_table(cursor, schema: str, table: str, csv_path: str, columns: Iterable[str]) -> None:
    """Bulk-load CSV into a PostgreSQL table using COPY."""
    schema_q = quote_ident(schema)
    table_q = quote_ident(table)
    cols_q = ", ".join(quote_ident(c) for c in columns)

    copy_sql = (
        f"COPY {schema_q}.{table_q} ({cols_q}) "
        "FROM STDIN WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',', QUOTE '\"', ESCAPE '\"')"
    )

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        cursor.copy_expert(copy_sql, f)


def validate_env() -> None:
    required = {
        "PG_HOST": DB_HOST,
        "PG_DB": DB_NAME,
        "PG_USER": DB_USER,
        "PG_PASSWORD": DB_PASSWORD,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(f"Missing .env keys: {', '.join(missing)}")


def upload_one_table(conn, schema: str, source_key: str, dry_run: bool) -> None:
    table_name = FILE_TABLE_MAP[source_key]
    csv_path = os.path.join(CLEANED_DATA_DIR, f"{source_key}.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"\nPreparing {source_key}.csv -> {schema}.{table_name}")
    column_types, row_count = infer_schema_from_csv(csv_path)
    print(f"  rows: {row_count:,}, columns: {len(column_types)}")

    if dry_run:
        print("  dry-run: skipped DB write")
        return

    with conn.cursor() as cursor:
        create_or_replace_table(cursor, schema, table_name, column_types)
        copy_csv_into_table(cursor, schema, table_name, csv_path, column_types.keys())
        cursor.execute(f"SELECT COUNT(*) FROM {quote_ident(schema)}.{quote_ident(table_name)}")
        inserted = cursor.fetchone()[0]

    conn.commit()
    print(f"  done -> inserted {inserted:,} rows")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload cleaned_data CSVs into PostgreSQL cleaned_* tables",
    )
    parser.add_argument(
        "--schema",
        default=DB_SCHEMA,
        help=f"PostgreSQL schema name (default: {DB_SCHEMA})",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(FILE_TABLE_MAP.keys()),
        help="Upload only selected datasets (e.g. apartment plot)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files + inferred schema without writing to DB",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    validate_env()

    if not os.path.isdir(CLEANED_DATA_DIR):
        raise FileNotFoundError(f"cleaned_data folder not found: {CLEANED_DATA_DIR}")

    selected = args.only or list(FILE_TABLE_MAP.keys())
    print("Source folder:", CLEANED_DATA_DIR)
    print("Selected datasets:", ", ".join(selected))

    if args.dry_run:
        for source_key in selected:
            upload_one_table(conn=None, schema=args.schema, source_key=source_key, dry_run=True)
        print("\nDry-run completed.")
        return

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )

    try:
        for source_key in selected:
            try:
                upload_one_table(conn=conn, schema=args.schema, source_key=source_key, dry_run=False)
            except Exception:
                conn.rollback()
                raise
    finally:
        conn.close()

    print("\nAll selected tables uploaded successfully.")


if __name__ == "__main__":
    main()
