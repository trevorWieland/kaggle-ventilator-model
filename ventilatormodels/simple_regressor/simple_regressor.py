from ..abstract_model import AbstractModel
import pandas as pd

class SimpleRegressor(AbstractModel):
    """

    """

    def __init__(self):
        """
        Initialize the model, with model structure and hyperparameters
        """

    def fit(self, data: pd.DataFrame):
        """
        A method to train the model, using some data
        """

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        A method to use the model, predicting on some data
        """

    def save_model(self, model_path: str):
        """
        Save the current model to a file
        """

    def load_model(self, model_path: str):
        """
        Load the model from a file
        """
