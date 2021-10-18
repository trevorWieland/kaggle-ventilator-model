import numpy as np
import pandas as pd

from typing import List, Optional, Tuple
from tqdm import tqdm

def scale_data(
        data: pd.DataFrame,
        feature_columns: list = ["R", "C", "time_step", "u_in", "u_out"],
        output_col: str = "pressure"
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """

    """

    scale_columns = feature_columns + [output_col]

    means = data[scale_columns].mean()
    stdevs = data[scale_columns].std()

    data[scale_columns] = (data[scale_columns] - means) / stdevs

    return data, means, stdevs

def unscale_data(
        Y_predictions: np.ndarray ,
        means: pd.Series,
        stdevs: pd.Series
    ) -> np.ndarray:
    """

    """

    Y = (Y_predictions * stdevs[-1] + means[-1]).reshape((Y_predictions.shape[0] * Y_predictions.shape[1], 1))

    return Y

def export_predictions(
        Y: np.ndarray,
        ids: List[int],
        output_path: str
    ) -> pd.DataFrame:
    """

    """
    final_df = pd.DataFrame()
    final_df["id"] = ids
    final_df["pressure"] = Y

    final_df.to_csv(output_path)
    return final_df

def format_input_matrix(
        data: pd.DataFrame,
        feature_columns: list = ["R", "C", "time_step", "u_in", "u_out"],
        index_columns: list = ["breath_id", "id"],
        breath_col: str = "breath_id"
    ) -> np.ndarray:
    """

    """

    gp = data.groupby(index_columns).mean()
    breath_values = list(set(data[breath_col]))

    X = gp[feature_columns].values.reshape((-1, 80, len(feature_columns)))

    return X

def format_output_matrix(
        data: pd.DataFrame,
        output_col: str = "pressure",
        index_columns: list = ["breath_id", "id"],
        breath_col: str = "breath_id"
    ) -> np.ndarray:
    """

    """

    gp = data.groupby(index_columns).mean()
    breath_values = list(set(data[breath_col]))

    Y = gp[output_col].values.reshape((-1, 80))

    return Y

def train_val_split(
        X: np.ndarray,
        Y: np.ndarray,
        train_samples: int = 70000
    ) -> np.ndarray:
    """

    """

    assert X.shape[0] == Y.shape[0]
    assert X.shape[1] == Y.shape[1]
    assert Y.shape[1] == 80

    X_train = X[:train_samples, :, :]
    Y_train = Y[:train_samples, :]

    X_val = X[train_samples:, :, :]
    Y_val = Y[train_samples:, :]

    return X_train, Y_train, X_val, Y_val
