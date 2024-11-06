import polars as pl
import torch
import numpy as np
from src.app import get_prediction, get_metrics, TrainVehicleDataset
from torchtsmixer import TSMixerExt
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from src.constants import DEVICE, ROOT_DATA_DIR, RANDOM_STATE, TSMIXER_PARS
import torch.nn.functional as F
import typer
from tqdm import tqdm


def main(
    data_type: str = typer.Option("test", "--type", "-t", help="Type of data"),
    n_workers: int = typer.Option(10, "--nw", "-w", help="Number of workers"),
) -> None:
    current_metrics = {}
    total_pred = []

    data = pl.read_parquet(ROOT_DATA_DIR / "total_data.pa")
    if data_type == "test":
        data = data.filter(pl.col("is_test"))
        folds = []
    elif data_type == "val":
        data = data.filter(~pl.col("is_test"))
        kf = KFold(
            n_splits=TSMIXER_PARS["n_splits"], shuffle=True, random_state=RANDOM_STATE
        )
        testcases_ids = np.unique(data["testcase_id"].to_list())
        folds = []
        for train_idx, test_idx in kf.split(testcases_ids):
            folds.append([set(testcases_ids[train_idx]), set(testcases_ids[test_idx])])

    for n_fold in tqdm(range(TSMIXER_PARS["n_splits"])):
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

        model = TSMixerExt(
            sequence_length=TSMIXER_PARS["sequence_length"],
            prediction_length=TSMIXER_PARS["prediction_length"],
            input_channels=len(val_dataset.x_hist_col),
            extra_channels=len(val_dataset.x_extra_hist_col),
            hidden_channels=TSMIXER_PARS["hidden_channels"],
            output_channels=len(val_dataset.y_true_col),
            static_channels=len(val_dataset.x_static_col),
        ).to(DEVICE)

        model.load_state_dict(
            torch.load(ROOT_DATA_DIR / f"tsmixer_{n_fold}.pth")["model_state_dict"]
        )
        model.eval()

        current_metrics[n_fold] = []
        with torch.no_grad():
            for batch_data in val_loader:
                y_pred = model(
                    x_hist=batch_data["x_hist"].to(DEVICE),
                    x_extra_hist=batch_data["x_extra_hist"].to(DEVICE),
                    x_extra_future=batch_data["x_extra_future"].to(DEVICE),
                    x_static=batch_data["x_static"].to(DEVICE),
                )
                v_total_pred = 55.72991943359375 * y_pred[:, :, 0]
                w_yaw_pred = 0.7887094020843506 * y_pred[:, :, 1]
                predictions = get_prediction(
                    v_total_pred,
                    w_yaw_pred,
                    batch_data["target_start_val"].to(DEVICE),
                    batch_data["target_time"][:, :, 0].to(DEVICE),
                )
                if data_type == "val":
                    current_metrics[n_fold].extend(
                        get_metrics(
                            batch_data["target"].permute(2, 0, 1).to(DEVICE),
                            predictions,
                        )
                    )
                a_total_pred = F.pad(
                    input=torch.diff(v_total_pred, dim=1)
                    / (
                        torch.diff(batch_data["target_time"][:, :, 0].to(DEVICE), dim=1)
                        / 1e9
                    ),
                    pad=(1, 0),
                    mode="constant",
                    value=0,
                )

                df_pred = pl.DataFrame(
                    {
                        "testcase_id": batch_data["testcase_ids"].flatten().tolist(),
                        "stamp_ns": batch_data["target_time"].flatten().tolist(),
                        "x_pred": predictions[0, :, :].flatten().tolist(),
                        "y_pred": predictions[1, :, :].flatten().tolist(),
                        "yaw_pred": predictions[2, :, :].flatten().tolist(),
                        "v_total_pred": v_total_pred.flatten().tolist(),
                        "w_yaw_pred": w_yaw_pred.flatten().tolist(),
                        "a_total_pred": a_total_pred.flatten().tolist(),
                    },
                    schema=[
                        ("testcase_id", pl.UInt32),
                        ("stamp_ns", pl.Int64),
                        ("x_pred", pl.Float32),
                        ("y_pred", pl.Float32),
                        ("yaw_pred", pl.Float32),
                        ("v_total_pred", pl.Float32),
                        ("w_yaw_pred", pl.Float32),
                        ("a_total_pred", pl.Float32),
                    ],
                )
                total_pred.append(df_pred)
    total_pred = pl.concat(total_pred)
    if data_type == "val":
        for n_fold in range(len(folds)):
            print(n_fold, np.mean(current_metrics[n_fold]))
    elif data_type == "test":
        total_pred = total_pred.group_by(["testcase_id", "stamp_ns"]).agg(
            pl.col("x_pred").mean().alias("x_pred"),
            pl.col("y_pred").mean().alias("y_pred"),
            pl.col("yaw_pred").mean().alias("yaw_pred"),
            pl.col("v_total_pred").mean().alias("v_total_pred"),
            pl.col("w_yaw_pred").mean().alias("w_yaw_pred"),
            pl.col("a_total_pred").mean().alias("a_total_pred"),
        )
    total_pred.write_parquet(ROOT_DATA_DIR / f"{data_type}_tsmixer.pa")


if __name__ == "__main__":
    typer.run(main)
