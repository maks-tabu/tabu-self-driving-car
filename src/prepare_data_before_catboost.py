import typer
import polars as pl
from src.constants import ROOT_DATA_DIR, FEATURES_V_TOTAL, FEATURES_W_YAW
from typing import List, Iterator


def create_features(
    df: pl.DataFrame, pred_col_name, n_lags: List[int] = [1, 2, 3, 4, 5]
) -> pl.DataFrame:
    """
    Create features for a given DataFrame, including lags and rolling means.
    """
    sorted_df = df.sort("stamp_ns")

    lag_expressions = [
        pl.col(pred_col_name).shift(lag).alias(f"{pred_col_name}_lag_{lag}")
        for lag in n_lags
    ]

    window_expressions = [
        pl.col(pred_col_name)
        .rolling_mean(window_size=window)
        .alias(f"{pred_col_name}_rolling_mean_{window}")
        for window in [3, 5, 7]
    ]

    result = sorted_df.select(
        [
            pl.col("stamp_ns"),
            pl.col("testcase_id"),
            *lag_expressions,
            *window_expressions,
            pl.lit(sorted_df[pred_col_name].mean()).alias(f"{pred_col_name}_mean"),
            pl.lit(sorted_df[pred_col_name].std()).alias(f"{pred_col_name}_std"),
            pl.lit(sorted_df[pred_col_name].min()).alias(f"{pred_col_name}_min"),
            pl.lit(sorted_df[pred_col_name].max()).alias(f"{pred_col_name}_max"),
        ]
    )

    return result.sort(["testcase_id", "stamp_ns"])


def main():
    data = pl.read_parquet(ROOT_DATA_DIR / "total_data.pa")
    for target in ["w_yaw", "v_total"]:
        if target == "w_yaw":
            all_saved_columns = FEATURES_W_YAW + [target, "testcase_id", "stamp_ns"]
        else:
            all_saved_columns = FEATURES_V_TOTAL + [target, "testcase_id", "stamp_ns"]
        pred_col_name = f"{target}_pred"
        for data_type in ["val", "test"]:
            pred = pl.read_parquet(ROOT_DATA_DIR / f"{data_type}_tsmixer.pa")
            if data_type == "val":
                cur_data = data.filter(~pl.col("is_test")).join(
                    pred, on=["stamp_ns", "testcase_id"]
                )
            elif data_type == "test":
                cur_data = data.filter(pl.col("is_test")).join(
                    pred, on=["stamp_ns", "testcase_id"]
                )
            cur_data = cur_data.join(
                create_features(cur_data, pred_col_name), on=["stamp_ns", "testcase_id"]
            )
            cur_data = cur_data.with_columns(
                pl.col(pred_col_name).log().alias(f"{pred_col_name}_log")
            )
            cur_data = cur_data.with_columns(
                (pl.col(pred_col_name) ** 2).alias(f"{pred_col_name}_kv")
            )

            if data_type == "val":
                f1 = cur_data.group_by("vehicle_id").agg(
                    pl.col(pred_col_name)
                    .mean()
                    .alias(f"{pred_col_name}_vehicle_id_mean"),
                    pl.col(pred_col_name)
                    .std()
                    .alias(f"{pred_col_name}_vehicle_id_std"),
                )
                f2 = cur_data.group_by("vehicle_model").agg(
                    pl.col(pred_col_name)
                    .mean()
                    .alias(f"{pred_col_name}_vehicle_model_mean"),
                    pl.col(pred_col_name)
                    .std()
                    .alias(f"{pred_col_name}_vehicle_model_std"),
                )
                f3 = cur_data.group_by("vehicle_model_modification").agg(
                    pl.col(pred_col_name)
                    .mean()
                    .alias(f"{pred_col_name}_vehicle_model_modification_mean"),
                    pl.col(pred_col_name)
                    .std()
                    .alias(f"{pred_col_name}_vehicle_model_modification_std"),
                )
                f4 = cur_data.group_by(["tires_rear", "tires_front"]).agg(
                    pl.col(pred_col_name).mean().alias(f"{pred_col_name}_tires_mean"),
                    pl.col(pred_col_name).std().alias(f"{pred_col_name}_tires_std"),
                )
                f5 = cur_data.group_by(["ride_date"]).agg(
                    pl.col(pred_col_name)
                    .mean()
                    .alias(f"{pred_col_name}_ride_date_mean"),
                    pl.col(pred_col_name).std().alias(f"{pred_col_name}_ride_date_std"),
                )

            cur_data = cur_data.join(f1, on="vehicle_id").with_columns(
                (
                    (pl.col(pred_col_name) - pl.col(f"{pred_col_name}_vehicle_id_mean"))
                    / pl.col(f"{pred_col_name}_vehicle_id_std")
                ).alias(f"{pred_col_name}_vehicle_id")
            )
            cur_data = cur_data.join(f2, on="vehicle_model").with_columns(
                (
                    (
                        pl.col(pred_col_name)
                        - pl.col(f"{pred_col_name}_vehicle_model_mean")
                    )
                    / pl.col(f"{pred_col_name}_vehicle_model_std")
                ).alias(f"{pred_col_name}_vehicle_model")
            )
            cur_data = cur_data.join(f3, on="vehicle_model_modification").with_columns(
                (
                    (
                        pl.col(pred_col_name)
                        - pl.col(f"{pred_col_name}_vehicle_model_modification_mean")
                    )
                    / pl.col(f"{pred_col_name}_vehicle_model_modification_std")
                ).alias(f"{pred_col_name}_vehicle_model_modification")
            )
            cur_data = cur_data.join(f4, on=["tires_rear", "tires_front"]).with_columns(
                (
                    (pl.col(pred_col_name) - pl.col(f"{pred_col_name}_tires_mean"))
                    / pl.col(f"{pred_col_name}_tires_std")
                ).alias(f"{pred_col_name}_tires")
            )
            cur_data = cur_data.join(f5, on=["ride_date"]).with_columns(
                (
                    (pl.col(pred_col_name) - pl.col(f"{pred_col_name}_ride_date_mean"))
                    / pl.col(f"{pred_col_name}_ride_date_std")
                ).alias(f"{pred_col_name}_ride_date")
            )

            cur_data = cur_data.with_columns(
                (1 / pl.col(pred_col_name)).alias(f"{pred_col_name}_inv")
            )
            cur_data = cur_data.with_columns(
                (1 / pl.col(f"{pred_col_name}_log")).alias(f"{pred_col_name}_log_inv")
            )
            cur_data = cur_data.with_columns(
                (pl.col(pred_col_name) / pl.col("y_known_values_std")).alias(
                    f"{pred_col_name}_y_known_values_std"
                )
            )
            cur_data = cur_data.with_columns(
                (pl.col(pred_col_name) / pl.col("y_known_values_mean")).alias(
                    f"{pred_col_name}_y_known_values_mean"
                )
            )
            cur_data = cur_data.with_columns(
                (pl.col(pred_col_name) / pl.col("x_known_values_std")).alias(
                    f"{pred_col_name}_x_known_values_std"
                )
            )
            cur_data = cur_data.with_columns(
                (pl.col(pred_col_name) / pl.col("x_known_values_mean")).alias(
                    f"{pred_col_name}_x_known_values_mean"
                )
            )
            cur_data = cur_data.with_columns(
                (pl.col(pred_col_name) / pl.col("v_total_mean")).alias(
                    f"{pred_col_name}_v_total_mean"
                )
            )
            cur_data = cur_data.with_columns(
                (pl.col(pred_col_name) / pl.col("v_total_std")).alias(
                    f"{pred_col_name}_v_total_std"
                )
            )
            cur_data[all_saved_columns].write_parquet(
                ROOT_DATA_DIR / f"data_{pred_col_name}_{data_type}.pa"
            )
            print("completed: ", pred_col_name, data_type)


if __name__ == "__main__":
    typer.run(main)
