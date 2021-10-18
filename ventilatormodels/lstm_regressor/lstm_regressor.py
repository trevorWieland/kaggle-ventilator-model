from ..abstract_model import AbstractModel
from ..datautils import *
import pandas as pd
import numpy as np

from typing import Optional, List
from tqdm import tqdm

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Input
from tensorflow.keras.callbacks import EarlyStopping

import pickle

class LSTMRegressor(AbstractModel):
    """

    """

    def __init__(
        self,
        n_timesteps: int = 80,
        n_features: int = 5,
        n_LSTM_layers: int = 2,
        LSTM_size: int = 64,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the model, with model structure and hyperparameters

        Input:
            n_timesteps:
                The number of timesteps to process at once when reading the data. In the Google Data, it is 80 per breath.
            n_features:
                The number of input features to build the model for.
            n_LSTM_layers:
                The number of LSTM layers to have in the model.
            LSTM_size:
                The size of each LSTM layer.
            random_seed:
                The integer seed to use for reproducibility
        """

        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.random_seed = random_seed

        self.model = self._define_model(n_LSTM_layers, LSTM_size)

    def _define_model(
        self,
        n_LSTM_layers,
        LSTM_size
    ) -> Model:
        """

        """
        #Create model input layer
        inputs = Input(shape=(self.n_timesteps, self.n_features))

        #Iterate, adding LSTM layers as needed
        current = inputs
        for i in range(n_LSTM_layers):
            if (i == n_LSTM_layers - 1):
                current = LSTM(LSTM_size, return_sequences=True)(current)
            else:
                current = LSTM(LSTM_size)(current)
                current = RepeatVector(self.n_timesteps)(current)

        #Add the final TimeDistributed Dense layer for output
        current = TimeDistributed(Dense(1))(current)

        #Combine all layers into one model object
        model = Model(inputs, current, name="lstm_regressor")

        #Compile the model for use
        model.compile(optimizer="adam", loss="mae")

        return model


    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        val_X: Optional[np.ndarray] = None,
        val_Y: Optional[np.ndarray] = None,
        epochs: int = 300,
        patience: int = 25,
        verbose: int = 1
    ):
        """
        A method to train the model, using some data
        """

        #Data Checking
        assert X.shape[2] == self.n_features
        assert X.shape[0] == Y.shape[0]
        assert X.shape[1] == Y.shape[1]
        assert X.shape[1] == 80


        #Train model
        if val_X is None or val_Y is None:
            es = EarlyStopping(
                monitor="loss",
                patience=patience,
                verbose=verbose,
                restore_best_weights=True
            )

            self.model.fit(
                X,
                Y,
                epochs=epochs,
                verbose=verbose,
                callbacks=[es]
            )
        else:
            assert val_X.shape[2] == self.n_features
            assert val_X.shape[0] == val_Y.shape[0]
            assert val_X.shape[1] == val_Y.shape[1]
            assert val_X.shape[1] == 80

            es = EarlyStopping(
                monitor="val_loss",
                patience=patience,
                verbose=verbose,
                restore_best_weights=True
            )

            self.model.fit(
                X,
                Y,
                epochs=epochs,
                verbose=verbose,
                validation_data=(val_X, val_Y),
                callbacks=[es]
            )


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        A method to use the model, predicting on some data
        """

        assert X.shape[2] == self.n_features
        assert X.shape[1] == 80

        Y = self.model.predict(X)

        return Y

    def save_model(self, model_path: str):
        """
        Save the current model to a file
        """

    def load_model(self, model_path: str):
        """
        Load the model from a file
        """
