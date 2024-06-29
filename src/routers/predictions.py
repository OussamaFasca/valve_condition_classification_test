from fastapi import APIRouter,HTTPException
from src.utils import processed_dataset,scale_input_data,make_prediction
from src.schemas.prediction_data import PredictionInput

predictions_router = APIRouter()

@predictions_router.post("/predict", response_model=dict)
async def predict(data: PredictionInput):
    """
    Endpoint to predict the valve condition based on cycle number.

    Inputs:
    - data: JSON body containing 'cycle_number' which is an integer representing the cycle index.

    Outputs:
    - JSON response with 'Condition Valve' indicating if the valve condition is optimal or non-optimal,
      and 'Classification' which is the predicted classification (1 or 0).

    Raises:
    - HTTPException(404): If the cycle number does not exist in the dataset.
    - HTTPException(500): If there is an internal server error during prediction or scaling.

    Example Usage:
    ```
    POST /predict
    {
        "cycle_number": 123
    }
    ```
    """
    try:
        # Find cycle in dataframe
        df = processed_dataset[processed_dataset.index == data.cycle_number].drop(columns=["Valve condition Status"],axis=1)
        print(len(df.index))
        if len(df.index) == 0:
            raise HTTPException(status_code=404, detail="Cycle number not found in dataset")
    except KeyError:
        raise HTTPException(status_code=404, detail="Cycle number not found in dataset")

    try:
        # Scale input data
        df_scaled = scale_input_data(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scaling input data: {str(e)}")

    try:
        # Make prediction
        classification = int(make_prediction(df_scaled))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

    # Determine condition label based on classification
    condition_label = "Optimale" if classification == 1 else "Non Optimale"

    return {
        "Condition Valve": condition_label,
        "Classification": classification
    }