from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
from pydantic import BaseModel
from typing import List

# Initialize the FastAPI app
app = FastAPI()

# Load the pre-trained models (make sure to have these models serialized and saved)
models = {
    "ecommerce": pickle.load(open("models/ecommerce_model.pkl", "rb")),
    "bank_churn": pickle.load(open("models/bank_churn_model.pkl", "rb")),
    "orange_telecom": pickle.load(open("models/orange_telecom_model.pkl", "rb"))
}

# Default model (we start with the e-commerce model)
current_model = models["ecommerce"]

# Define the input data schema
class InputData(BaseModel):
    data: List[float]  # A list of features for one sample

# Endpoint to switch between datasets/models
@app.post("/switch-dataset/")
def switch_dataset(dataset: str):
    global current_model
    if dataset in models:
        current_model = models[dataset]
        return {"status": "success", "message": f"Switched to {dataset} model."}
    else:
        raise HTTPException(status_code=404, detail="Dataset not found.")

# Prediction endpoint
@app.post("/predict/")
def predict(input_data: InputData):
    # Input validation: ensure input data matches the model's feature size
    try:
        # Convert input data to numpy array
        input_array = np.array(input_data.data).reshape(1, -1)

        # Perform prediction with the current model
        prediction = current_model.predict(input_array)

        # Return the prediction
        return {"prediction": int(prediction[0])}  # Assuming binary output (churn or not)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Sample welcome message
@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API. Use /predict to get predictions."}


    