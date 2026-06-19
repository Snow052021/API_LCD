import os
import joblib
import json

def init():
    """
    Deze functie wordt één keer uitgevoerd wanneer de container opstart.
    Hier laden we het .pkl model in het geheugen.
    """
    global model
    # AZURE_MODEL_DIR is een ingebouwde omgevingsvariabele die naar je model wijst
    model_path = os.path.join(
        os.getenv("AZURE_MODEL_DIR"), "lantaarnpaal_model_definitief.pkl"
    )
    model = joblib.load(model_path)

def run(raw_data):
    """
    Deze functie wordt telkens uitgevoerd als er een API-aanroep binnenkomt.
    """
    try:
        # Verwacht data in JSON-formaat: {"data": [[waarde1, waarde2, ...]]}
        data = json.loads(raw_data)["data"]
        
        # Voorspelling doen
        predictions = model.predict(data)
        
        # (Optioneel) Kans berekenen als je dat wilt:
        # predictions_proba = model.predict_proba(data)
        
        return {"result": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}
