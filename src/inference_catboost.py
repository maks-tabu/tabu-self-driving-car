import polars as pl
import numpy as np
from src.app import TrainVehicleDataset, get_prediction, get_metrics
from src.constants import (
    DEVICE,
    N_SPLITS,
    TSMIXER_PARS,
    ROOT_DATA_DIR,
    FEATURES_W_YAW,
    FEATURES_V_TOTAL,
    RANDOM_STATE,
)
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from tqdm import tqdm
import typer


def main(
    data_type: str = typer.Option("test", "--type", "-t", help="Type of data"),
    n_workers: int = typer.Option(10, "--nw", "-w", help="Number of workers"),
) -> None:

    data = pl.read_parquet(ROOT_DATA_DIR / "total_data.pa")
    if data_type == "test":
        data = data.filter(pl.col("is_test"))
        folds = []
    elif data_type == "val":
        data = data.filter(~pl.col("is_test"))
        kf = KFold(
            n_splits=TSMIXER_PARS["n_splits"],
            shuffle=True,
            random_state=2 * RANDOM_STATE,
        )
        testcases_ids = np.unique(data["testcase_id"].to_list())
        folds = []
        for train_idx, test_idx in kf.split(testcases_ids):
            folds.append([set(testcases_ids[train_idx]), set(testcases_ids[test_idx])])

    total_pred = []
    current_metrics = {}

    for n_fold in tqdm(range(N_SPLITS)):
        if data_type == "val":
            val_start_points = np.arange(0, 1001, 500)
            val_split = []
            for i in range(len(folds[n_fold][1])):
                val_split.extend((val_start_points + i * 1500).tolist())
            val_data = data.clone().filter(
                pl.col("testcase_id").is_in(folds[n_fold][1])
            )
        elif data_type == "test":
            val_data = data.clone()
            val_split = np.concatenate(
                [
                    [0],
                    val_data.group_by("testcase_id", maintain_order=True)
                    .count()["count"]
                    .cum_sum()
                    .to_numpy()[:-1],
                ]
            )

        for target in ["w_yaw", "v_total"]:
            if target == "w_yaw":
                features_columns = FEATURES_W_YAW
            else:
                features_columns = FEATURES_V_TOTAL
            pred_col_name = f"{target}_pred"
            if DEVICE == "cuda":
                task_type = "GPU"
            else:
                task_type = "CPU"
            model = CatBoostRegressor(task_type=task_type)
            model.load_model(ROOT_DATA_DIR / f"catboost_{target}_{n_fold}.pt")
            catboost_data = pl.read_parquet(
                ROOT_DATA_DIR / f"data_{pred_col_name}_{data_type}.pa"
            )
            catboost_data = catboost_data.filter(
                pl.col("testcase_id").is_in(val_data["testcase_id"].unique())
            )
            catboost_data = catboost_data.with_columns(
                pl.Series(
                    f"{target}_pred_2",
                    model.predict(catboost_data[features_columns].to_pandas()),
                )
            )
            val_data = val_data.join(
                catboost_data[["stamp_ns", "testcase_id", f"{target}_pred_2"]],
                on=["stamp_ns", "testcase_id"],
                how="left",
            )

        val_dataset = TrainVehicleDataset(
            val_data,
            val_split,
            TSMIXER_PARS["total_length"],
            TSMIXER_PARS["prediction_length"],
            TSMIXER_PARS["sequence_length"],
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=TSMIXER_PARS["batch_size"],
            shuffle=False,
            num_workers=n_workers,
        )

        current_metrics[n_fold] = []
        for batch_data in tqdm(val_loader):
            predictions = get_prediction(
                batch_data["prediction"][:, :, 0].to(DEVICE),
                batch_data["prediction"][:, :, 1].to(DEVICE),
                batch_data["target_start_val"].to(DEVICE),
                batch_data["target_time"][:, :, 0].to(DEVICE),
            )
            if data_type == "val":
                current_metrics[n_fold].extend(
                    get_metrics(
                        batch_data["target"].permute(2, 0, 1).to(DEVICE), predictions
                    )
                )

            df_pred = pl.DataFrame(
                {
                    "testcase_id": batch_data["testcase_ids"][:, :, 0]
                    .flatten()
                    .tolist(),
                    "stamp_ns": batch_data["target_time"].flatten().tolist(),
                    "x_pred_2": predictions[0, :, :].flatten().tolist(),
                    "y_pred_2": predictions[1, :, :].flatten().tolist(),
                    "yaw_pred_2": predictions[2, :, :].flatten().tolist(),
                },
                schema=[
                    ("testcase_id", pl.UInt32),
                    ("stamp_ns", pl.Int64),
                    ("x_pred_2", pl.Float32),
                    ("y_pred_2", pl.Float32),
                    ("yaw_pred_2", pl.Float32),
                ],
            )

            total_pred.append(df_pred)

    total_pred = pl.concat(total_pred)
    if data_type == "val":
        for n_fold in range(N_SPLITS):
            print(n_fold, np.mean(current_metrics[n_fold]))
    elif data_type == "test":
        total_pred = total_pred.group_by(["testcase_id", "stamp_ns"]).agg(
            pl.col("x_pred_2").mean().alias("x_pred_2"),
            pl.col("y_pred_2").mean().alias("y_pred_2"),
            pl.col("yaw_pred_2").mean().alias("yaw_pred_2"),
        )
        data = data.join(total_pred, on=["stamp_ns", "testcase_id"], how="left")
        data = data.filter(
            (pl.col("x").is_null())
            & (pl.col("deleted_stamp_ns").is_null() | ~pl.col("deleted_stamp_ns"))
        )
        data = data.sort(["testcase_id", "stamp_ns"])
        data = data.select(
            pl.col("testcase_id"),
            pl.col("stamp_ns"),
            pl.col("x_pred_2").fill_null(strategy="forward").alias("x"),
            pl.col("y_pred_2").fill_null(strategy="forward").alias("y"),
            pl.col("yaw_pred_2").fill_null(strategy="forward").alias("yaw"),
        )
        data.write_csv(ROOT_DATA_DIR / "new_submission.csv")


if __name__ == "__main__":
    typer.run(main)
