import os
import sys

import pandas as pd
import psycopg2
from dotenv import load_dotenv

from real_estate.entity import PlotDataIngestionArtifact, PlotDataIngestionConfig
from real_estate.exception.exception import RealEstateException
from real_estate.logging.logger import logging


class PlotDataIngestion:
    """
    Plot/Land ingestion pipeline:
    1) Fetch ho_raw_data + mb_raw_data from PostgreSQL
    2) Align source schemas and filter plot/land property types
    3) Merge + de-duplicate and save artifact output
    """

    _HO_DROP_COLUMNS = [
        "id",
        "property_id",
        "source",
        "event_type",
        "bhk",
        "sqft_price",
        "bathrooms",
        "balconies",
        "floors",
        "age_of_property",
        "furnishing_type",
        "possession_status",
        "agent_id",
        "agent_name",
        "agent_type",
        "developer_name",
        "developer_uuid",
        "possession_breakdown",
        "listing_url",
        "posting_date",
        "scrape_date",
    ]

    _MB_DROP_COLUMNS = [
        "id",
        "property_id",
        "source",
        "event_type",
        "sqft_price",
        "bhk",
        "bathrooms",
        "balconies",
        "floors",
        "age_of_property",
        "furnishing_type",
        "possession_status",
        "agent_name",
        "company_name",
        "user_type",
        "agent_id",
        "developer_name",
        "developer_id",
        "project_society_name",
        "posting_date",
        "scrape_date",
    ]

    _HO_TYPES = {"Plot", "Agricultural Land"}
    _MB_TYPES = {
        "Residential Plot",
        "Commercial Land",
        "Agricultural Land",
        "Industrial Land",
    }

    _TYPE_MAP = {
        "Residential Plot": "Plot",
        "Plot": "Plot",
        "Agricultural Land": "Agricultural Land",
        "Commercial Land": "Commercial Land",
        "Industrial Land": "Industrial Land",
    }

    _DEDUP_KEYS = [
        "covered_area_value",
        "price_numeric",
        "locality",
        "city",
        "latitude",
        "longitude",
    ]

    def __init__(self, config: PlotDataIngestionConfig = PlotDataIngestionConfig()):
        self.config = config

    @staticmethod
    def _load_env() -> None:
        package_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        candidates = [
            os.path.join(package_root, ".env"),
            os.path.join(os.getcwd(), ".env"),
            os.path.join(os.path.abspath(os.path.join(package_root, "..")), ".env"),
        ]

        loaded = False
        for env_path in candidates:
            if os.path.exists(env_path):
                load_dotenv(env_path)
                loaded = True
                logging.info(f"Loaded environment from {env_path}")
                break

        if not loaded:
            logging.warning(
                ".env file not found. Checked: " + ", ".join(candidates)
            )

    @staticmethod
    def _get_db_params() -> dict:
        db_name = (
            os.getenv("PG_DB")
            or os.getenv("PG_DBNAME")
            or os.getenv("PG_DATABASE")
        )
        return {
            "host": os.getenv("PG_HOST"),
            "port": os.getenv("PG_PORT", "5432"),
            "dbname": db_name,
            "user": os.getenv("PG_USER"),
            "password": os.getenv("PG_PASSWORD"),
        }

    @staticmethod
    def _validate_db_params(params: dict) -> None:
        required_map = {
            "host": "PG_HOST",
            "dbname": "PG_DB (or PG_DBNAME / PG_DATABASE)",
            "user": "PG_USER",
            "password": "PG_PASSWORD",
        }
        missing = [required_map[k] for k in required_map if not params.get(k)]
        if missing:
            raise ValueError(
                "Missing PostgreSQL .env keys: " + ", ".join(missing)
            )

    @staticmethod
    def _drop_if_exists(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")

    @staticmethod
    def _align_union_columns(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
        ordered = []
        seen = set()
        for frame in frames:
            for col in frame.columns:
                if col not in seen:
                    ordered.append(col)
                    seen.add(col)
        return [frame.reindex(columns=ordered) for frame in frames]

    def initiate_data_ingestion(self) -> PlotDataIngestionArtifact:
        try:
            logging.info("PlotDataIngestion started")
            self._load_env()
            params = self._get_db_params()
            self._validate_db_params(params)

            os.makedirs(self.config.raw_data_dir, exist_ok=True)
            os.makedirs(self.config.merged_data_dir, exist_ok=True)

            conn = psycopg2.connect(**params)
            try:
                ho_df = pd.read_sql('SELECT * FROM public."ho_raw_data"', conn)
                mb_df = pd.read_sql('SELECT * FROM public."mb_raw_data"', conn)
            finally:
                conn.close()

            logging.info(f"Fetched ho_raw_data rows={len(ho_df)} cols={len(ho_df.columns)}")
            logging.info(f"Fetched mb_raw_data rows={len(mb_df)} cols={len(mb_df.columns)}")

            ho_df = self._drop_if_exists(ho_df, self._HO_DROP_COLUMNS)
            mb_df = self._drop_if_exists(mb_df, self._MB_DROP_COLUMNS)

            plot_ho = ho_df[ho_df["property_type"].isin(self._HO_TYPES)].copy()
            plot_mb = mb_df[mb_df["property_type"].isin(self._MB_TYPES)].copy()

            plot_ho["source"] = "HousingOnline"
            plot_mb["source"] = "MagicBricks"

            plot_ho["property_type"] = plot_ho["property_type"].map(self._TYPE_MAP)
            plot_mb["property_type"] = plot_mb["property_type"].map(self._TYPE_MAP)

            plot_ho_path = os.path.join(self.config.raw_data_dir, "plot_ho_raw.csv")
            plot_mb_path = os.path.join(self.config.raw_data_dir, "plot_mb_raw.csv")
            plot_ho.to_csv(plot_ho_path, index=False)
            plot_mb.to_csv(plot_mb_path, index=False)

            aligned_frames = self._align_union_columns([plot_ho, plot_mb])
            combined_plot = pd.concat(aligned_frames, ignore_index=True)

            dedup_keys = [k for k in self._DEDUP_KEYS if k in combined_plot.columns]
            before = len(combined_plot)
            if dedup_keys:
                combined_plot = combined_plot.drop_duplicates(subset=dedup_keys, keep="first")
            else:
                combined_plot = combined_plot.drop_duplicates(keep="first")
            combined_plot = combined_plot.reset_index(drop=True)

            combined_plot.to_csv(self.config.merged_file_path, index=False)

            logging.info(f"plot_ho rows={len(plot_ho)}")
            logging.info(f"plot_mb rows={len(plot_mb)}")
            logging.info(f"combined before dedup={before}, after dedup={len(combined_plot)}")
            logging.info(f"Combined plot artifact saved -> {self.config.merged_file_path}")

            return PlotDataIngestionArtifact(merged_file_path=self.config.merged_file_path)

        except Exception as e:
            raise RealEstateException(e, sys)
