import os
import sys
import shutil
import pandas as pd

from real_estate.constant import DATA_DIR, RAW_DATA_FILES, RENT_DATA_FILES
from real_estate.entity import DataIngestionConfig, DataIngestionArtifact
from real_estate.exception.exception import RealEstateException
from real_estate.logging.logger import logging


class DataIngestion:
    """
    Reads all raw CSV files, normalizes source-specific schema differences,
    aligns dtypes on shared columns, concatenates them, and saves the merged file.
    """

    _COLUMN_RENAME_MAP = {
        "ho_raw_data.csv": {
            "developer_uuid": "developer_id",
        },
        "mb_raw_data.csv": {
            "user_type": "agent_type",
        },
        "ho_rent.csv": {
            "developer_uuid": "developer_id",
        },
        "mb_rent.csv": {
            "user_type": "agent_type",
        },
    }

    _DROP_COLUMNS = {
        "possession_breakdown",
        "listing_url",
    }

    _NUMERIC_COLUMNS = {
        "id",
        "covered_area_value",
        "price_numeric",
        "latitude",
        "longitude",
        "corner_property",
        "rectangular_plot",
        "gated_plot",
        "backlane",
    }

    def __init__(
        self,
        config: DataIngestionConfig = DataIngestionConfig(),
        raw_data_files: list[str] | None = None,
    ):
        self.config = config
        self.raw_data_files = raw_data_files if raw_data_files is not None else RAW_DATA_FILES

    # ── helpers ──────────────────────────────────────────────
    @classmethod
    def _normalize_dataframe(cls, df: pd.DataFrame, file_name: str) -> pd.DataFrame:
        """
        Apply source-specific column renames and drop columns that should not
        appear in the merged dataset.
        """
        rename_map = cls._COLUMN_RENAME_MAP.get(file_name.lower(), {})
        if rename_map:
            applicable_renames = {old: new for old, new in rename_map.items() if old in df.columns}
            if applicable_renames:
                df = df.rename(columns=applicable_renames)
                logging.info(f"Normalized columns for {file_name}: {applicable_renames}")

        drop_columns = [column for column in cls._DROP_COLUMNS if column in df.columns]
        if drop_columns:
            df = df.drop(columns=drop_columns)
            logging.info(f"Dropped columns for {file_name}: {drop_columns}")

        return df

    @staticmethod
    def _ordered_union_columns(dataframes: list[pd.DataFrame]) -> list[str]:
        """
        Preserve the first-seen column order while keeping the full normalized union.
        """
        ordered_columns: list[str] = []
        seen_columns: set[str] = set()

        for df in dataframes:
            for column in df.columns:
                if column not in seen_columns:
                    seen_columns.add(column)
                    ordered_columns.append(column)

        return ordered_columns

    @staticmethod
    def _align_dtypes(dataframes: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """
        Normalize only known numeric columns across all DataFrames.

        Non-numeric columns are left as-is so source-specific text and ID fields
        such as developer_id remain intact after schema normalization.
        """
        if len(dataframes) < 2:
            return dataframes

        # find columns common to ALL dataframes
        common_cols = set(dataframes[0].columns)
        for df in dataframes[1:]:
            common_cols &= set(df.columns)
        common_cols = sorted(common_cols)

        # align known numeric types only
        for col in common_cols:
            if col in DataIngestion._NUMERIC_COLUMNS:
                for df in dataframes:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        return dataframes

    # ── main ─────────────────────────────────────────────────
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Data ingestion started")

            # 1. Create artifact directories
            os.makedirs(self.config.raw_data_dir, exist_ok=True)
            os.makedirs(self.config.merged_data_dir, exist_ok=True)

            # 2. Copy raw files into artifact/data_ingestion/raw/
            dataframes: list[pd.DataFrame] = []
            for file_name in self.raw_data_files:
                src = os.path.join(DATA_DIR, file_name)
                dst = os.path.join(self.config.raw_data_dir, file_name)
                try:
                    shutil.copy2(src, dst)
                    logging.info(f"Copied {src} -> {dst}")
                except PermissionError:
                    logging.warning(f"Could not copy {src} -> {dst} (file locked), reading from existing copy")
                # Read from dst if copy succeeded, or from src as fallback
                read_path = dst if os.path.exists(dst) else src
                df = pd.read_csv(read_path)
                df = self._normalize_dataframe(df, file_name)
                dataframes.append(df)

            # 3. Align DataFrames on the normalized union of columns
            common_cols = set(dataframes[0].columns)
            for df in dataframes[1:]:
                common_cols &= set(df.columns)
            common_cols = sorted(common_cols)
            merged_columns = self._ordered_union_columns(dataframes)
            logging.info(f"Common columns after normalization ({len(common_cols)}): {common_cols}")
            logging.info(f"Merged columns after normalization ({len(merged_columns)}): {merged_columns}")

            dataframes = [df.reindex(columns=merged_columns) for df in dataframes]

            # 4. Align dtypes dynamically
            dataframes = self._align_dtypes(dataframes)

            # 5. Concatenate & save
            merged_df = pd.concat(dataframes, ignore_index=True)

            # Remove existing file if locked, then write
            if os.path.exists(self.config.merged_file_path):
                try:
                    os.remove(self.config.merged_file_path)
                except PermissionError:
                    # File is locked — write to a temp file and replace
                    import tempfile
                    tmp = self.config.merged_file_path + ".tmp"
                    merged_df.to_csv(tmp, index=False)
                    logging.warning(f"Original file locked, saved to {tmp}")
                    return DataIngestionArtifact(merged_file_path=tmp)

            merged_df.to_csv(self.config.merged_file_path, index=False)
            logging.info(f"Merged data saved → {self.config.merged_file_path}  shape={merged_df.shape}")

            return DataIngestionArtifact(
                merged_file_path=self.config.merged_file_path,
            )

        except Exception as e:
            raise RealEstateException(e, sys)
