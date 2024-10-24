import json
from datetime import datetime
import pandas as pd
import dill
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.DEBUG)
app = FastAPI()

# Load the model
file_name = "model/cars_pipe.pkl"
with open(file_name, 'rb') as file:
    model = dill.load(file)


class CarForm(BaseModel):
    description: str
    id: int
    image_url: str
    lat: float
    long: float
    manufacturer: str
    model: str
    year: int
    odometer: float
    posting_date: datetime
    region: str
    region_url: str
    url: str
    title_status: str
    transmission: str
    state: str
    fuel: str
    price: float


class CarPrediction(BaseModel):
    ID: int
    Price: float
    PredictedCategory: str


@app.post('/predict', response_model=CarPrediction)
def predict(car_form: CarForm):
    import pandas as pd
    logging.debug("Starting prediction...")

    # Create a DataFrame from the car_form data
    df = pd.DataFrame([{
        'description': car_form.description,
        'id': car_form.id,
        'image_url': car_form.image_url,
        'lat': car_form.lat,
        'long': car_form.long,
        'manufacturer': car_form.manufacturer,
        'model': car_form.model,
        'year': car_form.year,
        'odometer': car_form.odometer,
        'posting_date': car_form.posting_date,
        'region': car_form.region,
        'region_url': car_form.region_url,
        'url': car_form.url,
        'title_status': car_form.title_status,
        'transmission': car_form.transmission,
        'state': car_form.state,
        'fuel': car_form.fuel,
        'price': car_form.price
    }])

    logging.debug(f"DataFrame created with columns: {df.columns.tolist()}")

    # Check and log the data type of columns which might cause issues
    logging.debug(f"Data types in DataFrame: {df.dtypes}")

    try:
        # Predict using the loaded model
        category_prediction = model['model'].predict(df)
        predicted_category = category_prediction[0] if category_prediction.size > 0 else "Unknown"
        logging.debug(f"Prediction successful: {predicted_category}")
    except Exception as e:
        logging.error(f"Error in making prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return {
        'ID': car_form.id,
        'Price': car_form.price,
        'PredictedCategory': predicted_category
    }


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']
