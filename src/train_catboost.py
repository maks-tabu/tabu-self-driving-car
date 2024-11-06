import polars as pl
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from src.constants import (
    ROOT_DATA_DIR,
    RANDOM_STATE,
    N_SPLITS,
    FEATURES_V_TOTAL,
    FEATURES_W_YAW,
    CATBOOST_PARS,
)
import typer


def main() -> None:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=2 * RANDOM_STATE)
    for target in ["w_yaw", "v_total"]:
        pred_col_name = f"{target}_pred"
        if target == "w_yaw":
            features_columns = FEATURES_W_YAW
        else:
            features_columns = FEATURES_V_TOTAL

        data = pl.read_parquet(ROOT_DATA_DIR / f"data_{pred_col_name}_val.pa")
        testcases_ids = np.unique(data["testcase_id"].to_list())
        folds = []
        for train_idx, test_idx in kf.split(testcases_ids):
            folds.append([set(testcases_ids[train_idx]), set(testcases_ids[test_idx])])

        for n_fold in range(N_SPLITS):
            train_data = data.filter(pl.col("testcase_id").is_in(folds[n_fold][0]))
            val_data = data.filter(pl.col("testcase_id").is_in(folds[n_fold][1]))

            X_train = train_data[features_columns].to_pandas()
            y_train = train_data[target].to_pandas()
            X_val = val_data[features_columns].to_pandas()
            y_val = val_data[target].to_pandas()

            model = CatBoostRegressor(**CATBOOST_PARS)

            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], use_best_model=True)
            model.save_model(ROOT_DATA_DIR / f"catboost_{target}_{n_fold}.pt")


if __name__ == "__main__":
    typer.run(main)
