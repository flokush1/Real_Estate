import os
import sys
import shutil
import pandas as pd

from real_estate.constant import DATA_DIR, RAW_DATA_FILES
from real_estate.entity import DataIngestionConfig, DataIngestionArtifact
from real_estate.exception.exception import RealEstateException
from real_estate.logging.logger import logging


class DataIngestion:
    """
    Reads all raw CSV files, aligns dtypes on common columns,
    concatenates them, and saves the merged file.
    """

    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.config = config

    # ── helpers ──────────────────────────────────────────────
    @staticmethod
    def _align_dtypes(dataframes: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """
        For every pair of common columns across all DataFrames,
        convert mismatched dtypes to numeric (coerce errors → NaN).
        """
        if len(dataframes) < 2:
            return dataframes

        # find columns common to ALL dataframes
        common_cols = set(dataframes[0].columns)
        for df in dataframes[1:]:
            common_cols &= set(df.columns)
        common_cols = sorted(common_cols)

        # align types
        for col in common_cols:
            dtypes = {id(df): df[col].dtype for df in dataframes}
            unique_dtypes = set(dtypes.values())
            if len(unique_dtypes) > 1:
                logging.info(f"Dtype mismatch on '{col}': {unique_dtypes} → converting to numeric")
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
            for file_name in RAW_DATA_FILES:
                src = os.path.join(DATA_DIR, file_name)
                dst = os.path.join(self.config.raw_data_dir, file_name)
                try:
                    shutil.copy2(src, dst)
                    logging.info(f"Copied {src} -> {dst}")
                except PermissionError:
                    logging.warning(f"Could not copy {src} -> {dst} (file locked), reading from existing copy")
                # Read from dst if copy succeeded, or from src as fallback
                read_path = dst if os.path.exists(dst) else src
                dataframes.append(pd.read_csv(read_path))

            # 3. Keep only common columns
            common_cols = set(dataframes[0].columns)
            for df in dataframes[1:]:
                common_cols &= set(df.columns)
            common_cols = sorted(common_cols)
            logging.info(f"Common columns ({len(common_cols)}): {common_cols}")

            dataframes = [df[common_cols].copy() for df in dataframes]

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
