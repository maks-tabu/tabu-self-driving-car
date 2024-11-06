from typing import Dict, List
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import polars as pl
import numpy as np


def get_prediction(
    v_total_pred: torch.Tensor,
    yaw_pred: torch.Tensor,
    target_start_val: torch.Tensor,
    target_time: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the future position and yaw predictions based on velocity and yaw rate.
    """
    dt = torch.diff(target_time / 1e9)
    yaw_pred = (
        F.pad(
            input=torch.cumsum(yaw_pred[:, :-1] * dt, dim=1),
            pad=(1, 0),
            mode="constant",
            value=0,
        )
        + target_start_val[:, [2]]
    )
    vx_pred = v_total_pred * torch.cos(yaw_pred)
    vy_pred = v_total_pred * torch.sin(yaw_pred)

    x_pred = (
        F.pad(
            input=torch.cumsum(vx_pred[:, :-1] * dt, dim=1),
            pad=(1, 0),
            mode="constant",
            value=0,
        )
        + target_start_val[:, [0]]
    )
    y_pred = (
        F.pad(
            input=torch.cumsum(vy_pred[:, :-1] * dt, dim=1),
            pad=(1, 0),
            mode="constant",
            value=0,
        )
        + target_start_val[:, [1]]
    )
    return torch.stack([x_pred, y_pred, yaw_pred])


def yaw_direction(yaw_value: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.cos(yaw_value), torch.sin(yaw_value)])


def get_metrics(targets: torch.Tensor, predictions: torch.Tensor) -> List[float]:
    """
    Calculates a metric based on the mean squared error between target and predicted points.
    """
    targets_points = torch.concatenate(
        [targets[:-1, :, :], targets[:-1, :, :] + yaw_direction(targets[-1, :, :])],
        axis=2,
    )
    predictions_points = torch.concatenate(
        [
            predictions[:-1, :, :],
            predictions[:-1, :, :] + yaw_direction(predictions[-1, :, :]),
        ],
        axis=2,
    )
    squared_diff = (targets_points - predictions_points) ** 2
    mean_squared_diff = torch.mean(squared_diff, dim=0)
    metric = torch.mean(torch.sqrt(2.0 * mean_squared_diff), axis=1)
    return metric.tolist()


class TrainVehicleDataset(Dataset):
    def __init__(
        self,
        data: pl.DataFrame,
        data_split: List[int],
        total_length: int,
        prediction_length: int,
        sequence_length: int,
    ) -> None:
        self.total_length = total_length
        self.prediction_length = prediction_length
        self.sequence_length = sequence_length
        self.n_spl_pred = self.total_length - self.prediction_length
        self.n_spl_seq = self.n_spl_pred - self.sequence_length
        self.y_true_col = ["v_total", "w_yaw"]
        self.x_hist_col = ["v_total", "w_yaw", "pitch"]

        self.x_extra_hist_col = [
            "braking_step",
            "acceleration_step",
            "steering",
            "weights_acceleration_diff",
            "acceleration_level_vehicle_model",
            "acceleration_level_vehicle_id",
            "acceleration_level_vehicle_model_modification",
        ]
        self.x_extra_future_col = self.x_extra_hist_col
        self.x_static_col = [
            "tires_front",
            "tires_rear",
            "vehicle_id",
            "vehicle_model",
            "vehicle_model_modification",
            "location_reference_point_id",
            "ride_date_count",
        ]
        self.target_col = ["x", "y", "yaw"]
        self.prediction_col = ["v_total_pred_2", "w_yaw_pred_2"]

        # Нормализация
        data = data.clone()
        data = data.with_columns(
            (pl.col("braking_step") + 1379.4670295721216) / 1948.6939697265625,
            (pl.col("acceleration_step") - 3808.2253165094126) / 4510.43310546875,
            (pl.col("steering") + 0.07238475714492636) / 0.8700177669525146,
            (pl.col("weights_acceleration_diff") - 0.8423713965889033)
            / 1918.7613525390625,
            (pl.col("pitch") + 0.0011489910072855687) / 0.02145647630095482,
            pl.col("v_total") / 55.72991943359375,
            pl.col("w_yaw") / 0.7887094020843506,
        )

        if (self.prediction_col[0] in data.columns) and (
            self.prediction_col[1] in data.columns
        ):
            prediction = self._preprocess_data(data, self.prediction_col)
        else:
            prediction = np.array([]).reshape(-1, len(self.prediction_col))

        self.data_arrays = {
            "y_true": self._preprocess_data(data, self.y_true_col),
            "x_hist": self._preprocess_data(data, self.x_hist_col),
            "x_extra_hist": self._preprocess_data(data, self.x_extra_hist_col),
            "x_extra_future": self._preprocess_data(data, self.x_extra_future_col),
            "x_static": self._preprocess_data(data, self.x_static_col),
            "target": self._preprocess_data(data, self.target_col),
            "prediction": prediction,
            "target_time": self._preprocess_data(data, ["stamp_ns"]),
            "testcase_ids": self._preprocess_data(data, ["testcase_id"]).astype(
                np.int64
            ),
        }
        self.data_split = data_split

    def _preprocess_data(self, data: pl.DataFrame, columns: List[str]) -> np.ndarray:
        return data.select(columns).to_numpy()

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        start_idx = self.data_split[index]
        end_idx = start_idx + self.total_length
        return {
            "y_true": torch.tensor(
                self.data_arrays["y_true"][start_idx:end_idx][self.n_spl_pred :],
                dtype=torch.float32,
            ),
            "x_hist": torch.tensor(
                self.data_arrays["x_hist"][start_idx:end_idx][
                    self.n_spl_seq : self.n_spl_pred
                ],
                dtype=torch.float32,
            ),
            "x_extra_hist": torch.tensor(
                self.data_arrays["x_extra_hist"][start_idx:end_idx][
                    self.n_spl_seq : self.n_spl_pred
                ],
                dtype=torch.float32,
            ),
            "x_extra_future": torch.tensor(
                self.data_arrays["x_extra_future"][start_idx:end_idx][
                    self.n_spl_pred :
                ],
                dtype=torch.float32,
            ),
            "x_static": torch.tensor(
                self.data_arrays["x_static"][start_idx], dtype=torch.float32
            ),
            "target_start_val": torch.tensor(
                self.data_arrays["target"][start_idx:end_idx][self.n_spl_pred - 1],
                dtype=torch.float32,
            ),
            "target_time": torch.tensor(
                self.data_arrays["target_time"][start_idx:end_idx][self.n_spl_pred :]
            ),
            "target": torch.tensor(
                self.data_arrays["target"][start_idx:end_idx][self.n_spl_pred :],
                dtype=torch.float32,
            ),
            "testcase_ids": torch.tensor(
                self.data_arrays["testcase_ids"][start_idx:end_idx][self.n_spl_pred :]
            ),
            "prediction": torch.tensor(
                self.data_arrays["prediction"][start_idx:end_idx][self.n_spl_pred :],
                dtype=torch.float32,
            ),
        }

    def __len__(self) -> int:
        return len(self.data_split)
