from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
app = FastAPI(title="Lantaarnpaal Risico API")
model = joblib.load("lantaarnpaal_model_definitief.pkl")
class CrashInput(BaseModel):
    # Pas deze velden aan naar de features die jouw model echt nodig heeft!
    latitude: float
    longitude: float
    mast_hoogte: int
    temperatuur: int
    wegtype: int
    gladheid_risico: int

def read_root():
    return {"message": "De Lantaarnpaal Risico API is live!"}

@app.post("/predict")
def predict(input_data: CrashInput):
    # 1. Zet de JSON input om naar een DataFrame
    data_dict = input_data.dict()
    df = pd.DataFrame([data_dict])
    
    if hasattr(model, "feature_names_in_"):
        df.columns = model.feature_names_in_
    else:
        df = df.values 
        
    prediction_proba = model.predict_proba(df)
    
    risk_score = float(prediction_proba[0][1])
    
    return {"risk_score": round(risk_score, 4)}
