from pydantic import BaseModel

# Define a Pydantic model for request body
class PredictionInput(BaseModel):
    cycle_number: int