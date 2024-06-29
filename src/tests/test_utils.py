import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from sklearn.preprocessing import StandardScaler
from src.utils import (
    load_processed_dataset,
    load_scaling_model,
    load_classifier,
    scale_input_data,
    make_prediction,
    processed_dataset,
    standard_scaler,
    classifier
)

@pytest.fixture
def sample_input_data():
    return pd.DataFrame({
        'fs1_mean_measure_at_10th_second': [0.98779, 0.94283],
        'fs1_mean_measure_at_20th_second': [7.83661, 7.84971],
        'fs1_mean_measure_at_30th_second': [7.68885, 7.70148],
        'fs1_mean_measure_at_40th_second': [7.93828, 7.96244],
        'fs1_mean_measure_at_50th_second': [7.937, 7.9485],
        'fs1_mean_measure_at_60th_second': [7.87036, 7.88693],
        'ps2_mean_measure_at_10th_second': [9.512161, 9.566112],
        'ps2_mean_measure_at_20th_second': [121.12585, 121.08698],
        'ps2_mean_measure_at_30th_second': [131.31226, 131.12716],
        'ps2_mean_measure_at_40th_second': [139.6496, 139.48404],
        'ps2_mean_measure_at_50th_second': [129.96391, 129.83252],
        'ps2_mean_measure_at_60th_second': [125.2377, 125.03253],
    })

def test_load_processed_dataset():
    df = load_processed_dataset()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_load_scaling_model():
    scaler = load_scaling_model()
    assert isinstance(scaler, StandardScaler)

def test_load_classifier():
    clf = load_classifier()
    assert callable(getattr(clf, "predict", None))

def test_scale_input_data(sample_input_data):
    scaled_data = scale_input_data(sample_input_data)
    assert scaled_data.shape == sample_input_data.shape

def test_make_prediction(sample_input_data):
    scaled_data = scale_input_data(sample_input_data)
    predictions = make_prediction(scaled_data)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(sample_input_data)

def test_shared_objects():
    assert isinstance(processed_dataset, pd.DataFrame)
    assert isinstance(standard_scaler, StandardScaler)
    assert callable(getattr(classifier, "predict", None))

def test_end_to_end(sample_input_data):
    scaled_data = scale_input_data(sample_input_data)
    predictions = make_prediction(scaled_data)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(sample_input_data)