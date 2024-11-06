import typer
import polars as pl
import os
import numpy as np
from tqdm import tqdm
import math
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.interpolate import UnivariateSpline
from collections import deque
from src.constants import ROOT_DATA_DIR


class VehicleDataLoader:
    def __init__(self, ROOT_DATA_DIR: str, BASE_DATA_DIR: str, n_workers: int = 8):
        assert BASE_DATA_DIR in ["YaCupTrain", "YaCupTest"]
        self.dataset_dir = os.path.join(ROOT_DATA_DIR, BASE_DATA_DIR)
        self.testcases_ids = np.unique(list(map(int, os.listdir(self.dataset_dir))))
        self.n_workers = n_workers

        self.localization_schema = {
            "stamp_ns": pl.Int64,
            "x": pl.Float32,
            "y": pl.Float32,
            "z": pl.Float32,
            "roll": pl.Float32,
            "pitch": pl.Float32,
            "yaw": pl.Float32,
        }

        self.control_schema = {
            "stamp_ns": pl.Int64,
            "acceleration_level": pl.Float32,
            "steering": pl.Float32,
        }

        self.requested_stamps_schema = {
            "stamp_ns": pl.Int64,
        }

    @staticmethod
    def create_extrapolation(
        x: np.ndarray, y: np.ndarray, x_new: np.ndarray, smoothing: float = 0
    ) -> np.ndarray:
        spl = UnivariateSpline(x, y, k=3, s=smoothing)
        return spl(x_new)

    @staticmethod
    def pad_dataframe(
        df: pl.DataFrame, target_length: int, insert_pos: str = "start"
    ) -> pl.DataFrame:
        cnt_new_val = target_length - len(df)
        if insert_pos == "start":
            new_df = pl.concat([df.head(1)] * cnt_new_val).drop("stamp_ns")
            old_stamp = df["stamp_ns"][:2].to_numpy()
            delta_stamp = old_stamp[1] - old_stamp[0]
            new_stamp = np.arange(
                old_stamp[0] - cnt_new_val * delta_stamp, old_stamp[0], delta_stamp
            )
            new_df = new_df.with_columns(
                pl.Series("stamp_ns", new_stamp).cast(pl.Int64),
                pl.lit(True).alias("deleted_stamp_ns"),
            )
            df = pl.concat([new_df[df.columns], df])
        elif insert_pos == "end":
            new_df = pl.concat([df.tail(1)] * cnt_new_val).drop("stamp_ns")
            old_stamp = df["stamp_ns"][-2:].to_numpy()
            delta_stamp = old_stamp[1] - old_stamp[0]
            new_stamp = (
                np.arange(
                    old_stamp[1], old_stamp[1] + cnt_new_val * delta_stamp, delta_stamp
                )
                + delta_stamp
            )
            new_df = new_df.with_columns(
                pl.Series("stamp_ns", new_stamp).cast(pl.Int64),
                pl.lit(True).alias("deleted_stamp_ns"),
            )
            df = pl.concat([df, new_df[df.columns]])
        return df

    @staticmethod
    def degrees_to_radians(degrees: float) -> float:
        return degrees * (math.pi / 180)

    @staticmethod
    def weighted_rolling_mean(data: np.ndarray, window: int = 100) -> np.ndarray:
        data = np.array(data, dtype=np.float64)
        n = len(data)
        result = np.zeros(n)

        weights = np.linspace(window, 1, window)
        weights = weights / weights.sum()
        window_queue = deque(maxlen=window)

        for i in range(n):
            window_queue.append(data[i])
            if i < window - 1:
                curr_weights = weights[: len(window_queue)]
                curr_weights = curr_weights / curr_weights.sum()
                result[i] = np.sum(np.array(window_queue) * curr_weights)
            else:
                result[i] = np.sum(np.array(window_queue) * weights)
        return result

    def get_testcase_data(self, testcase_id: int) -> pl.DataFrame:
        base_path = os.path.join(self.dataset_dir, str(testcase_id))
        paths = {
            "localization": os.path.join(base_path, "localization.csv"),
            "control": os.path.join(base_path, "control.csv"),
            "metadata": os.path.join(base_path, "metadata.json"),
            "requested_stamps": os.path.join(base_path, "requested_stamps.csv"),
        }

        control = pl.read_csv(
            paths["control"],
            schema=self.control_schema,
        )

        localization = pl.read_csv(
            paths["localization"],
            schema=self.localization_schema,
        )[-1500:]
        localization = localization.with_columns(
            pl.lit(False).alias("deleted_stamp_ns")
        )

        with open(paths["metadata"], "r") as f:
            metadata = json.load(f)

        len_split = 500
        if os.path.exists(paths["requested_stamps"]):
            requested_stamps = pl.read_csv(
                paths["requested_stamps"],
                schema=self.requested_stamps_schema,
            )
            if len(localization) > 125:
                localization = localization[-125:]
            elif len(localization) < 125:
                localization = self.pad_dataframe(localization, 125, "start")

            localization = requested_stamps.join(
                localization, how="full", on="stamp_ns", coalesce=True
            ).sort("stamp_ns")
            if len(localization) < 500:
                localization = self.pad_dataframe(localization, 500, "end")
            len_split = len(localization)

        acceleration_level = self.create_extrapolation(
            control["stamp_ns"].to_numpy(),
            control["acceleration_level"].to_numpy(),
            localization["stamp_ns"].to_numpy(),
        )

        weights_acceleration_diff = acceleration_level - self.weighted_rolling_mean(
            acceleration_level, 50
        )

        steering = self.create_extrapolation(
            control["stamp_ns"].to_numpy(),
            control["steering"].to_numpy(),
            localization["stamp_ns"].to_numpy(),
        )
        acceleration_step = np.clip(acceleration_level, a_min=0, a_max=None)
        braking_step = np.clip(acceleration_level, a_min=None, a_max=0)

        localization = localization.with_columns(
            pl.Series("yaw", np.unwrap(localization.get_column("yaw").to_numpy())),
            pl.Series("roll", np.unwrap(localization.get_column("roll").to_numpy())),
            pl.Series("pitch", np.unwrap(localization.get_column("pitch").to_numpy())),
        )

        localization = localization.with_columns(
            [
                pl.lit(testcase_id).cast(pl.UInt32).alias("testcase_id"),
                pl.Series("weights_acceleration_diff", weights_acceleration_diff).cast(
                    pl.Float32
                ),
                pl.Series("acceleration_level", acceleration_level).cast(pl.Float32),
                pl.Series("acceleration_step", acceleration_step).cast(pl.Float32),
                pl.Series("braking_step", braking_step).cast(pl.Float32),
                pl.Series("steering", self.degrees_to_radians(steering)).cast(
                    pl.Float32
                ),
                (
                    (
                        (
                            pl.col("x").diff() ** 2
                            + pl.col("y").diff() ** 2
                            + pl.col("z").diff() ** 2
                        ).sqrt()
                        / (pl.col("stamp_ns").diff() / 1e9)
                    )
                    .fill_null(strategy="backward")
                    .cast(pl.Float32)
                    .alias("v_total")
                ),
                (
                    (pl.col("yaw").diff() / (pl.col("stamp_ns").diff() / 1e9))
                    .fill_null(strategy="backward")
                    .cast(pl.Float32)
                    .alias("w_yaw")
                ),
            ]
        )
        localization = localization.with_columns(
            (pl.col("v_total").diff() / (pl.col("stamp_ns").diff() / 1e9))
            .fill_null(strategy="backward")
            .cast(pl.Float32)
            .alias("a_total"),
            pl.lit(metadata["tires"]["front"]).cast(pl.Int16).alias("tires_front"),
            pl.lit(metadata["tires"]["rear"]).cast(pl.Int16).alias("tires_rear"),
            pl.lit(metadata["vehicle_id"]).cast(pl.Int16).alias("vehicle_id"),
            pl.lit(metadata["vehicle_model"]).cast(pl.Int16).alias("vehicle_model"),
            pl.lit(metadata["vehicle_model_modification"])
            .cast(pl.Int16)
            .alias("vehicle_model_modification"),
            pl.lit(metadata["location_reference_point_id"])
            .cast(pl.Int16)
            .alias("location_reference_point_id"),
            pl.lit(metadata["ride_date"]).alias("ride_date"),
        )
        # catboost features
        merge_batch = []
        for i in range(0, len(localization), len_split):
            batch_localization = localization[i : i + len_split]
            batch_localization = batch_localization.with_row_index()
            known_values = batch_localization[:125]
            batch_localization = batch_localization.with_columns(
                pl.lit(known_values["x"].mean())
                .cast(pl.Float32)
                .alias("x_known_values_mean"),
                pl.lit(known_values["y"].mean())
                .cast(pl.Float32)
                .alias("y_known_values_mean"),
                pl.lit(known_values["z"].mean())
                .cast(pl.Float32)
                .alias("z_known_values_mean"),
                pl.lit(known_values["yaw"].mean())
                .cast(pl.Float32)
                .alias("yaw_known_values_mean"),
                pl.lit(known_values["roll"].mean())
                .cast(pl.Float32)
                .alias("roll_known_values_mean"),
                pl.lit(known_values["pitch"].mean())
                .cast(pl.Float32)
                .alias("pitch_known_values_mean"),
                pl.lit(known_values["x"].std())
                .cast(pl.Float32)
                .alias("x_known_values_std"),
                pl.lit(known_values["y"].std())
                .cast(pl.Float32)
                .alias("y_known_values_std"),
                pl.lit(known_values["z"].std())
                .cast(pl.Float32)
                .alias("z_known_values_std"),
                pl.lit(known_values["yaw"].std())
                .cast(pl.Float32)
                .alias("yaw_known_values_std"),
                pl.lit(known_values["roll"].std())
                .cast(pl.Float32)
                .alias("roll_known_values_std"),
                pl.lit(known_values["pitch"].std())
                .cast(pl.Float32)
                .alias("pitch_known_values_std"),
                pl.lit(known_values["v_total"].mean())
                .cast(pl.Float32)
                .alias("v_total_mean"),
                pl.lit(known_values["v_total"].std())
                .cast(pl.Float32)
                .alias("v_total_std"),
                pl.lit(known_values["a_total"].mean())
                .cast(pl.Float32)
                .alias("a_total_mean"),
                pl.lit(known_values["a_total"].std())
                .cast(pl.Float32)
                .alias("a_total_std"),
                pl.lit(known_values["v_total"][-1])
                .cast(pl.Float32)
                .alias("v_total_last_known_value"),
                pl.lit(known_values["a_total"][-1])
                .cast(pl.Float32)
                .alias("a_total_last_known_value"),
                pl.lit(known_values["x"][-1])
                .cast(pl.Float32)
                .alias("x_last_known_value"),
                pl.lit(known_values["y"][-1])
                .cast(pl.Float32)
                .alias("y_last_known_value"),
                pl.lit(known_values["z"][-1])
                .cast(pl.Float32)
                .alias("z_last_known_value"),
                pl.lit(known_values["yaw"][-1])
                .cast(pl.Float32)
                .alias("yaw_last_known_value"),
                pl.lit(known_values["roll"][-1])
                .cast(pl.Float32)
                .alias("roll_last_known_value"),
                pl.lit(known_values["pitch"][-1])
                .cast(pl.Float32)
                .alias("pitch_last_known_value"),
            )
            merge_batch.append(batch_localization)
        return pl.concat(merge_batch)

    def get_data(self) -> pl.DataFrame:
        """Parallel processing of testcases"""
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(self.get_testcase_data, testcase_id)
                for testcase_id in self.testcases_ids
            ]

            all_full_data = []
            for future in tqdm(as_completed(futures), total=len(self.testcases_ids)):
                all_full_data.append(future.result())

        all_full_data = pl.concat(all_full_data)
        return all_full_data


def main(n_workers: int = typer.Option(10, help="Number of workers")):
    """
    Prepare the vehicle data for training and testing.
    """
    train_vechicle_dataloader = VehicleDataLoader(
        ROOT_DATA_DIR, "YaCupTrain", n_workers
    )
    train_data_without_norm = train_vechicle_dataloader.get_data()

    test_vechicle_dataloader = VehicleDataLoader(ROOT_DATA_DIR, "YaCupTest", n_workers)
    test_data_without_norm = test_vechicle_dataloader.get_data()

    total_data_without_norm = pl.concat(
        [
            test_data_without_norm.with_columns(pl.lit(True).alias("is_test")),
            train_data_without_norm[test_data_without_norm.columns].with_columns(
                pl.lit(False).alias("is_test")
            ),
        ]
    )

    ride_data_count = total_data_without_norm["ride_date"].value_counts(
        name="ride_date_count"
    )
    ride_data_count = ride_data_count.with_columns(
        (pl.col("ride_date_count") / pl.col("ride_date_count").sum()).cast(pl.Float32)
    )
    total_data_without_norm = total_data_without_norm.join(
        ride_data_count, on="ride_date"
    )

    all_norm_columns = [
        ["vehicle_model"],
        ["vehicle_id"],
        ["vehicle_model_modification"],
    ]
    for col_names in all_norm_columns:
        suffix = "_".join(col_names)
        mean_name = f"acceleration_level_{suffix}_mean"
        std_name = f"acceleration_level_{suffix}_std"
        new_features = (
            total_data_without_norm.drop_nulls()
            .group_by(col_names)
            .agg(
                pl.col("acceleration_level").mean().cast(pl.Float32).alias(mean_name),
                pl.col("acceleration_level").std().cast(pl.Float32).alias(std_name),
            )
        )
        total_data_without_norm = total_data_without_norm.join(
            new_features, on=col_names
        )
        total_data_without_norm = total_data_without_norm.with_columns(
            ((pl.col("acceleration_level") - pl.col(mean_name)) / pl.col(std_name))
            .cast(pl.Float32)
            .alias(f"acceleration_level_{suffix}")
        )
    total_data_without_norm.write_parquet(ROOT_DATA_DIR / "total_data.pa")


if __name__ == "__main__":
    typer.run(main)
