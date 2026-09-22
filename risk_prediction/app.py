from fastapi import FastAPI
from pydantic import BaseModel
import joblib, json
import pandas as pd
from pathlib import Path

app = FastAPI(title="Road Accident Risk Prediction API")

BASE_DIR = Path(__file__).parent
lr = joblib.load(BASE_DIR / "student_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
columns = joblib.load(BASE_DIR / "columns.pkl")
model_info = json.loads((BASE_DIR / "model_info.json").read_text())

class AccidentInput(BaseModel):
    Year: int
    Month: str
    Day_of_Week: str
    Number_of_Vehicles_Involved: int
    Vehicle_Type_Involved: str
    Weather_Conditions: str
    Road_Type: str
    Road_Condition: str
    Lighting_Conditions: str
    Traffic_Control_Presence: str
    Speed_Limit_kmh: int
    Driver_Age: int
    Driver_License_Status: str
    Alcohol_Involvement: str
    Accident_Location_Details: str
    Hour: int

@app.get("/")
def root():
    return {"status": "ok", "model_info": model_info}

@app.post("/predict")
def predict(accident: AccidentInput):
    raw = {
        "Year": accident.Year, "Month": accident.Month,
        "Day of Week": accident.Day_of_Week,
        "Number of Vehicles Involved": accident.Number_of_Vehicles_Involved,
        "Vehicle Type Involved": accident.Vehicle_Type_Involved,
        "Weather Conditions": accident.Weather_Conditions,
        "Road Type": accident.Road_Type,
        "Road Condition": accident.Road_Condition,
        "Lighting Conditions": accident.Lighting_Conditions,
        "Traffic Control Presence": accident.Traffic_Control_Presence,
        "Speed Limit (km/h)": accident.Speed_Limit_kmh,
        "Driver Age": accident.Driver_Age,
        "Driver License Status": accident.Driver_License_Status,
        "Alcohol Involvement": accident.Alcohol_Involvement,
        "Accident Location Details": accident.Accident_Location_Details,
        "Hour": accident.Hour,
    }
    # IMPORTANT: must use drop_first=True to match training
    new_df = pd.DataFrame([raw])
    new_df = pd.get_dummies(new_df, drop_first=True)
    new_df = new_df.reindex(columns=columns, fill_value=0)
    scaled = scaler.transform(new_df)

    pred = lr.predict(scaled)[0]
    prob = lr.predict_proba(scaled)[0][1]
    return {
        "prediction": "HIGH RISK (Serious/Fatal)" if pred == 1 else "LOW RISK (Minor)",
        "probability_high_risk_percent": round(float(prob) * 100, 1)
    }
