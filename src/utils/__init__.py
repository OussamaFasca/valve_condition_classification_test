import pandas as pd
from pandas import DataFrame
from src.config import PROCESSED_DATASET_PATH, SCALER_MODEL_PATH, CLASSIFIER_PATH
from sklearn.preprocessing import StandardScaler
import joblib

# Load processed dataset
def load_processed_dataset() -> DataFrame:
    """
    Loads the processed dataset from a feather file.

    Returns:
        DataFrame: The loaded dataset.

    Raises:
        FileNotFoundError: If the processed dataset file does not exist.
        Exception: If any other error occurs while loading the dataset.
    """
    try:
        df = pd.read_feather(PROCESSED_DATASET_PATH)
        return df
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Processed dataset file not found at {PROCESSED_DATASET_PATH}") from e
    except Exception as e:
        raise Exception(f"An error occurred while loading the processed dataset: {str(e)}") from e

# Load scaling model
def load_scaling_model() -> StandardScaler:
    """
    Loads the scaling model (StandardScaler) from a pickle file.

    Returns:
        StandardScaler: The loaded scaling model.

    Raises:
        FileNotFoundError: If the scaler model file does not exist.
        Exception: If any other error occurs while loading the scaler model.
    """
    try:
        scaler = joblib.load(SCALER_MODEL_PATH)
        return scaler
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Scaler model file not found at {SCALER_MODEL_PATH}") from e
    except Exception as e:
        raise Exception(f"An error occurred while loading the scaler model: {str(e)}") from e

# Load classifier model
def load_classifier():
    """
    Loads the classifier model from a pickle file.

    Returns:
        classifier: The loaded classifier model.

    Raises:
        FileNotFoundError: If the classifier model file does not exist.
        Exception: If any other error occurs while loading the classifier model.
    """
    try:
        classifier = joblib.load(CLASSIFIER_PATH)
        return classifier
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Classifier model file not found at {CLASSIFIER_PATH}") from e
    except Exception as e:
        raise Exception(f"An error occurred while loading the classifier model: {str(e)}") from e

# Setup shared objects
processed_dataset = load_processed_dataset()
standard_scaler = load_scaling_model()
classifier = load_classifier()

# Scale input data using loaded scaler
def scale_input_data(input_data: DataFrame) -> DataFrame:
    """
    Scales the input data using the loaded StandardScaler.

    Args:
        input_data (DataFrame): The input data to be scaled.

    Returns:
        DataFrame: The scaled input data.

    Raises:
        Exception: If any error occurs during scaling.
    """
    try:
        input_data_scaled = standard_scaler.transform(input_data)
        return input_data_scaled
    except Exception as e:
        raise Exception(f"An error occurred while scaling the input data: {str(e)}") from e

# Make prediction using loaded classifier
def make_prediction(scaled_input_data: DataFrame):
    """
    Makes a prediction using the classifier on the scaled input data.

    Args:
        scaled_input_data (DataFrame): The scaled input data for prediction.

    Returns:
        array: The prediction results.

    Raises:
        Exception: If any error occurs during prediction.
    """
    try:
        return classifier.predict(scaled_input_data)
    except Exception as e:
        raise Exception(f"An error occurred while making the prediction: {str(e)}") from e
