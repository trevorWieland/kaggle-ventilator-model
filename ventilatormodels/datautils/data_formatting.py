import numpy as np
import pandas as pd

from typing import List, Optional, Tuple

def timesplit_data(
    data: pd.DataFrame,
    timesteps: int
) -> np.ndarray:
    """
    Transforms a __WIDE__ pandas dataframe into a timesplit dataset ready for LSTM
    model training.

    (# Rows, # Features) -> (# Rows - # Timesteps, # Timesteps, # Features)

    Input:
        data:
            A pandas dataframe in __WIDE__ format with feature columns and step index
        timesteps:
            The integer number of steps to include in the timesplit dataset for each entry

    Output:
        timesplit:
            A numpy array of shape (# Rows - # Timesteps, # Timesteps, # Features),
            which is the proper formatting for training a LSTM model.
    """

    #Parameter checking
    ##TODO

    #Split data into slices:
    time_slices = [data.values]

    for i in range(1, timesteps):
        temp = data.shift(periods=i)

        time_slices.append(temp.values)

    #Stack and trim nans
    timesplit = np.stack(time_slices, axis=1)

    timesplit = timesplit[timesteps:, :, :]

    return timesplit



def untimesplit_data(
    timesplit: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Transforms a timesplit dataset back into a pandas dataframe in __WIDE__
    format. Uses the mean value for smoothing across duplicate entries.

    (# Rows, # Timesteps, # Features) -> (# Rows, # Features)

    Input:
        timesplit:
            A numpy array of shape (# Rows, # Timesteps, # Features)
        feature_names:
            A list of string names to assign to the new pandas dataframe. Must be of length # Features

    Output:
        data:
            A pandas dataframe in __WIDE__ format with shape (# Rows, # Features)

    """

    #Data Parameter checking
    ##TODO


    #Extract time slices from the data and shift back values
    time_slices = [fast_shift(timesplit[:, i, :], -1*i) for i in range(timesplit.shape[1])]

    #Stack the shifted timeslices back into one array, then take the average of the values to smooth them out
    values = np.stack(time_slices, axis=1).mean(axis=1)

    #Construct a dataframe from the values
    data = pd.DataFrame(columns=feature_names, data=values)

    return data


def fast_shift(
    arr: np.ndarray,
    shift: int,
    fill_value: np.float64 = np.nan
) -> np.ndarray:
    """
    A fast shift function implemented in numpy, to shift values when comparing time slices.

    Input:
        arr:
            A numpy array to shift
        shift:
            The integer number of places to shift
        fill_value:
            What to replace the old shifted values with if nothing fills the place.

    """

    result = np.empty_like(arr)
    if shift > 0:
        result[:shift] = fill_value
        result[shift:] = arr[:-shift]
    elif shift < 0:
        result[shift:] = fill_value
        result[:shift] = arr[-shift:]
    else:
        result[:] = arr
    return result
