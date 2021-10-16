from abc import ABC, abstractmethod
import pandas as pd

import subprocess


class AbstractModel(ABC):
    """

    """

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        val_X: pd.DataFrame,
        val_Y: pd.DataFrame
    ):
        """
        A method to train the model, using some data
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        A method to use the model, predicting on some data
        """

    @abstractmethod
    def save_model(self, model_path: str):
        """
        Save the current model to a file
        """

    @abstractmethod
    def load_model(self, model_path: str):
        """
        Load the model from a file
        """

    def predict_save(self, data: pd.DataFrame, fpath: str) -> pd.DataFrame:
        """
        A method to use the model, predicting on some data, and save the results
        """

        predictions = self.predict(data)
        assert predictions.shape[0] == 4024000 #Number of rows that the submission is expecting
        predictions.to_csv(fpath)

        return predictions

    def submit(self, fpath: str, message: str) -> str:
        """
        A method to submit a csv file to the submissions on Kaggle
        """

        result = subprocess.run([
            "kaggle",
            "competitions",
            "submit",
            "ventilator-pressure-prediction",
            "-f",
            fpath,
            "-m",
            message
        ])

        return result.stdout
