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
    # 1. Zet de JSON input om naar een DataFrame
    data_dict = input_data.dict()
    df = pd.DataFrame([data_dict])
    
    # 2. Dwing de kolommen om de namen te gebruiken die het model verwacht
    if hasattr(model, "feature_names_in_"):
        df.columns = model.feature_names_in_
    else:
        # Als het model met een kale matrix (numpy) is getraind, vallen we hierop terug
        df = df.values 
        
    # 3. Doe de voorspelling
    prediction_proba = model.predict_proba(df)
    
    # 4. Pak de kans voor de 'crash' klasse (index 1)
    risk_score = float(prediction_proba[0][1])
    
    return {"risk_score": round(risk_score, 4)}
