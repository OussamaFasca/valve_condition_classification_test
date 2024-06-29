from fastapi import FastAPI
from src.routers.predictions import predictions_router

app = FastAPI()

#Include Routers
app.include_router(predictions_router)
