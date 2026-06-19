from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialiseer de FastAPI app
app = FastAPI(title="Lantaarnpaal Risico API")

# Laad het model
model = joblib.load("lantaarnpaal_model_definitief.pkl")

# Definieer de input structuur (deze velden moet je meesturen in je JSON)
class CrashInput(BaseModel):
    # Pas deze velden aan naar de features die jouw model echt nodig heeft!
    latitude: float
    longitude: float
    mast_hoogte: int
    temperatuur: int
    wegtype: int
    gladheid_risico: int

@app.get("/")
def read_root():
    return {"message": "De Lantaarnpaal Risico API is live!"}

@app.post("/predict")
def predict(input_data: CrashInput):
    # Zet de input om naar een DataFrame
    data_dict = input_data.dict()
    df = pd.DataFrame([data_dict])
    
    # Doe de voorspelling
    # We gebruiken predict_proba om een percentage (0.0 tot 1.0) te krijgen
    prediction_proba = model.predict_proba(df)
    
    # Pak de kans voor de 'crash' klasse (meestal index 1)
    # Als je model anders is opgebouwd, pas dit dan aan
    risk_score = float(prediction_proba[0][1])
    
    return {"risk_score": round(risk_score, 4)}

# Om de API te runnen (lokaal): uvicorn main:app --reload
