import polars as pl
import numpy as np
from src.constants import RANDOM_STATE, DEVICE, ROOT_DATA_DIR, TSMIXER_PARS
from src.app import get_prediction, get_metrics, TrainVehicleDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torchtsmixer import TSMixerExt
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import typer


def main(
    n_workers: int = typer.Option(10, "--nw", "-w", help="Number of workers")
) -> None:
    train_start_points = np.arange(0, 1001, 250)
    val_start_points = np.arange(0, 1001, 500)
    data = pl.read_parquet(ROOT_DATA_DIR / "total_data.pa")
    data = data.filter(~pl.col("is_test"))
    kf = KFold(
        n_splits=TSMIXER_PARS["n_splits"], shuffle=True, random_state=RANDOM_STATE
    )
    testcases_ids = np.unique(data["testcase_id"].to_list())
    num_epochs = TSMIXER_PARS["num_epochs"]
    folds = []
    for train_idx, test_idx in kf.split(testcases_ids):
        folds.append([set(testcases_ids[train_idx]), set(testcases_ids[test_idx])])

    for n_fold in range(TSMIXER_PARS["n_splits"]):
        train_split = []
        for i in range(len(folds[n_fold][0])):
            train_split.extend((train_start_points + i * 1500).tolist())

        val_split = []
        for i in range(len(folds[n_fold][1])):
            val_split.extend((val_start_points + i * 1500).tolist())

        train_data = data.filter(pl.col("testcase_id").is_in(folds[n_fold][0]))
        val_data = data.filter(pl.col("testcase_id").is_in(folds[n_fold][1]))

        train_dataset = TrainVehicleDataset(
            train_data,
            train_split,
            TSMIXER_PARS["total_length"],
            TSMIXER_PARS["prediction_length"],
            TSMIXER_PARS["sequence_length"],
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=TSMIXER_PARS["batch_size"],
            shuffle=True,
            num_workers=n_workers,
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
            input_channels=len(train_dataset.x_hist_col),
            extra_channels=len(train_dataset.x_extra_hist_col),
            hidden_channels=TSMIXER_PARS["hidden_channels"],
            output_channels=len(train_dataset.y_true_col),
            static_channels=len(train_dataset.x_static_col),
        ).to(DEVICE)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        for epoch in range(TSMIXER_PARS["num_epochs"]):
            model.train()
            train_loss = 0.0
            for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                optimizer.zero_grad()
                y_pred = model(
                    x_hist=batch_data["x_hist"].to(DEVICE),
                    x_extra_hist=batch_data["x_extra_hist"].to(DEVICE),
                    x_extra_future=batch_data["x_extra_future"].to(DEVICE),
                    x_static=batch_data["x_static"].to(DEVICE),
                )
                loss = criterion(y_pred, batch_data["y_true"].to(DEVICE))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            current_metrics = []
            with torch.no_grad():
                for batch_data in val_loader:
                    y_pred = model(
                        x_hist=batch_data["x_hist"].to(DEVICE),
                        x_extra_hist=batch_data["x_extra_hist"].to(DEVICE),
                        x_extra_future=batch_data["x_extra_future"].to(DEVICE),
                        x_static=batch_data["x_static"].to(DEVICE),
                    )
                    v_total_pred = 55.72991943359375 * y_pred[:, :, 0]
                    w_yaw = 0.7887094020843506 * y_pred[:, :, 1]
                    predictions = get_prediction(
                        v_total_pred,
                        w_yaw,
                        batch_data["target_start_val"].to(DEVICE),
                        batch_data["target_time"][:, :, 0].to(DEVICE),
                    )
                    current_metrics.extend(
                        get_metrics(
                            batch_data["target"].permute(2, 0, 1).to(DEVICE),
                            predictions,
                        )
                    )

                    loss = criterion(y_pred, batch_data["y_true"].to(DEVICE))
                    val_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            current_metrics = np.mean(current_metrics)

            print(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f} "
                f"Val metrics: {current_metrics:.4f}"
            )

            scheduler.step()

        checkpoint_path = ROOT_DATA_DIR / f"tsmixer_{n_fold}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            },
            checkpoint_path,
        )


if __name__ == "__main__":
    typer.run(main)
