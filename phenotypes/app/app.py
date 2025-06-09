from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import json
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import re
import os

app = Flask(__name__)

class TBIModelInference:
    def __init__(self, coma_model_path, coma_features_path,
                 loc_model_path, loc_features_path,
                 age_scaler_path, weights_path, 
                 kmeans_model_path, kmeans_scaler_path, kmeans_features_path):
        
        # Load symptom models
        self.coma_model = joblib.load(coma_model_path)
        self.loc_model = joblib.load(loc_model_path)

        with open(coma_features_path) as f:
            self.coma_features = json.load(f)
        with open(loc_features_path) as f:
            self.loc_features = json.load(f)

        # Load age scaler & weights
        self.age_scaler = joblib.load(age_scaler_path)
        self.weights = pd.read_csv(weights_path)
        
        #Load KMeans and knn scaler
        self.kmeans = joblib.load(kmeans_model_path)
        self.kmeans_scaler = joblib.load(kmeans_scaler_path)
        with open(kmeans_features_path) as f:
            self.kmeans_features = json.load(f)

    def preprocess_input(self, input_data: dict, feature_order: list) -> pd.DataFrame:
        df = pd.DataFrame([input_data])
        df["age_tbi"] = self.age_scaler.transform(df[["age_tbi"]])
        df = df[feature_order]
        return df

    def predict_symptom(self, input_df: pd.DataFrame, model) -> (float, int):
        proba = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]
        return proba, pred

    def calculate_iiss(self, row: pd.Series) -> float:
        w = self.weights
        w["Column"] = w["Column"].str.strip().str.lower()
        # Groupings
        symp_weights = w[w["Column"].str.contains("imm_symp")]
        headhit_weights = w[w["Column"].str.contains("headhit")]
        injury_weights = w[w["Column"].str.contains("injury_from")]
        demo_weight = w[w["Column"].str.contains("demographic")].iloc[0, 1]

        # --- IMMEDIATE SYMPTOM SCORE ---
        symp_score = 0
        for i in range(len(symp_weights) - 1):  # exclude β row
            col = symp_weights.iloc[i]["Column"]
            weight = symp_weights.iloc[i][1]

            if col in ["imm_symp_coma", "imm_symp_loss_of_consciousness"]:
                proba = row.get(f"predict_{col}", 0)
                symp_score += proba * weight
            else:
                symp_score += row.get(col, 0) * weight

        symp_score *= symp_weights.iloc[-1][1]  # β_s

        # --- HEADHIT SCORE ---
        headhit_score = 0
        for i in range(len(headhit_weights) - 1):
            col = headhit_weights.iloc[i]["Column"]
            weight = headhit_weights.iloc[i][1]
            headhit_score += row.get(col, 0) * weight
        headhit_score *= headhit_weights.iloc[-1][1]  # β_h

        # --- INJURY FROM SCORE ---
        injury_score = 0
        for i in range(len(injury_weights) - 1):
            col = injury_weights.iloc[i]["Column"]
            weight = injury_weights.iloc[i][1]
            injury_score += row.get(col, 0) * weight
        injury_score *= injury_weights.iloc[-1][1]  # β_e

        # --- DEMOGRAPHIC SCORE ---
        demo_score = row["age_tbi"] * demo_weight + demo_weight  # β_d * age + β_d

        # --- IISS TOTAL ---
        return symp_score + headhit_score + injury_score + demo_score

    def run_inference(self, input_data: dict) -> pd.DataFrame:

        # input_data.columns = [col.strip().lower() for col in input_data.columns]

        # Get predictions
        coma_df = self.preprocess_input(input_data, self.coma_features)
        coma_proba, coma_pred = self.predict_symptom(coma_df, self.coma_model)

        loc_df = self.preprocess_input(input_data, self.loc_features)
        loc_proba, loc_pred = self.predict_symptom(loc_df, self.loc_model)
        
        # Create result DataFrame with lowercase columns
        result = pd.DataFrame([input_data])
        result["predict_imm_symp_coma"] = coma_proba
        result["predict_imm_symp_loss_of_consciousness"] = loc_proba
        result["age_tbi"] = self.age_scaler.transform(result[["age_tbi"]])
        result.columns = [col.strip().lower() for col in result.columns]
        result["IISS"] = result.apply(self.calculate_iiss, axis=1)
        # Assign cluster
        result = self.assign_cluster(result)
        return result
    
    def assign_cluster(self, df: pd.DataFrame) -> pd.DataFrame: 
        kmeans_input = df[self.kmeans_features]
        kmeans_scaled = self.kmeans_scaler.transform(kmeans_input)
        df["iiss_cluster"] = self.kmeans.predict(kmeans_scaled)
        return df

# Initialize the model (you'll need to update these paths)
try:
    model_runner = TBIModelInference(
        coma_model_path="artifacts/logreg_model_imm_symp_coma.pkl",
        coma_features_path="artifacts/features_imm_symp_coma.json",
        loc_model_path="artifacts/logreg_model_imm_symp_loss_of_consciousness.pkl",
        loc_features_path="artifacts/features_imm_symp_loss_of_consciousness.json",
        age_scaler_path="artifacts/scaler_age_tbi.pkl",
        weights_path="data/weights.csv",
        kmeans_scaler_path="artifacts/scaler_kmeans.pkl",
        kmeans_features_path="artifacts/features_kmeans.json",
        kmeans_model_path="artifacts/kmeans_iiss.pkl"
    )
except Exception as e:
    print(f"Warning: Could not load models: {e}")
    model_runner = None

def process_event_description(description):
    """Process event description to determine event type flags"""
    description = description.lower()
    
    list_car = [ "car", "Car", "cars", "Cars", "driving", "Driving", "drive", "Drive",
    "driver", "Driver", "driven", "drives", "vehicle", "Vehicle",
    "automobile", "Automobile", "sedan", "Sedan", "convertible", "Convertible",
    "hatchback", "Hatchback", "bumper", "Bumper", "windshield", "Windshield",
    "ran over", "run over", "hit by car", "hit-and-run", "road", "roadway","crash", "Crash", "auto", "Auto", "SUV", "suv",
    "pickup", "Pickup", "minivan", "Minivan", "parking lot",]

    list_fall = ['fall', 'fell', 'Fall', 'Fell',  "tumble", "Tumble", "tumbled", "Tumbled", "stumble", "Stumble",
    "stumbled", "Stumbled", "topple", "Topple", "toppled", "Toppled",
    "collapse", "Collapse", "collapsed", "Collapsed", "slipped", "Slipped",]

    event_desc_car = 1 if any(word in description for word in list_car) else 0
    event_desc_fall = 1 if any(word in description for word in list_fall) else 0
    event_desc_severe = 1 if any(word in description for word in ['severe', 'serious', 'major', 'critical', 'emergency', 'hospital', 'ambulance', 'coma']) else 0
    
    return event_desc_car, event_desc_fall, event_desc_severe

def calculate_age_at_tbi(dob, tbi_date):
    """Calculate age at time of TBI"""
    try:
        birth_date = datetime.strptime(dob, '%Y-%m-%d')
        incident_date = datetime.strptime(tbi_date, '%Y-%m-%d')
        age = (incident_date - birth_date).days / 365.25
        return int(age)
    except:
        return 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Process dates to calculate age at TBI
        age_tbi = calculate_age_at_tbi(data['dob'], data['tbi_date'])
        
        # Process event description
        event_car, event_fall, event_severe = process_event_description(data['event_description'])
        
        # Process gender
        gender_male = 1 if data['gender'] == 'male' else 0
        gender_other = 1 if data['gender'] == 'other' else 0
        
        # Count head hit locations
        head_hit_locations = [
            'headhit_Not_Sure', 'headhit_Neck', 'headhit_Top_Of_Head',
            'headhit_Left_Side_Of_Head', 'headhit_Front_Of_Head', 
            'headhit_Back_Of_Head', 'headhit_All', 'headhit_Whiplash',
            'headhit_Right_Side_Of_Head'
        ]
        num_head_hit_location = sum(1 for loc in head_hit_locations if data.get(loc, 0) == 1)
        
        # Build input data dictionary
        input_data = {
            "num_head_hit_location": num_head_hit_location,
            "headhit_Not_Sure": data.get('headhit_Not_Sure', 0),
            "headhit_Neck": data.get('headhit_Neck', 0),
            "headhit_Top_Of_Head": data.get('headhit_Top_Of_Head', 0),
            "headhit_Left_Side_Of_Head": data.get('headhit_Left_Side_Of_Head', 0),
            "headhit_Front_Of_Head": data.get('headhit_Front_Of_Head', 0),
            "headhit_Back_Of_Head": data.get('headhit_Back_Of_Head', 0),
            "headhit_All": data.get('headhit_All', 0),
            "headhit_Whiplash": data.get('headhit_Whiplash', 0),
            "headhit_Right_Side_Of_Head": data.get('headhit_Right_Side_Of_Head', 0),
            "imm_symp_light_sensitivity": data.get('imm_symp_light_sensitivity', 0),
            "imm_symp_headache": data.get('imm_symp_headache', 0),
            "imm_symp_dazed_or_vacant_stare": data.get('imm_symp_dazed_or_vacant_stare', 0),
            "imm_symp_dizziness": data.get('imm_symp_dizziness', 0),
            "imm_symp_disorientation": data.get('imm_symp_disorientation', 0),
            "imm_symp_nausea": data.get('imm_symp_nausea', 0),
            "imm_symp_confusion": data.get('imm_symp_confusion', 0),
            "imm_symp_incoherent_speech": data.get('imm_symp_incoherent_speech', 0),
            "imm_symp_memory_loss": data.get('imm_symp_memory_loss', 0),
            "event_desc_car": event_car,
            "event_desc_fall": event_fall,
            "event_desc_severe": event_severe,
            "injury_from_Accident": data.get('injury_from_Accident', 0),
            "injury_from_Fall": data.get('injury_from_Fall', 0),
            "injury_from_Collision": data.get('injury_from_Collision', 0),
            "injury_from_Sports": data.get('injury_from_Sports', 0),
            "injury_from_Assault": data.get('injury_from_Assault', 0),
            "injury_from_Stroke": data.get('injury_from_Stroke', 0),
            "injury_from_Surgery": data.get('injury_from_Surgery', 0),
            "age_tbi": age_tbi,
            "gender_male": gender_male,
            "gender_other": gender_other
        }
        
        if model_runner is None:
            # Return mock data for testing
            return jsonify({
                'success': True,
                'iiss_score': 75.2,
                'cluster': 2,
                'message': 'Prediction completed successfully (mock data)'
            })
        
        # Run the model
        result_df = model_runner.run_inference(input_data)
        CLUSTER_LABELS = {
                            0: "Mild",
                            2: "Moderate",
                            1: "Severe"
                        }

        cluster_id = int(result_df['iiss_cluster'].iloc[0])
        severity_label = CLUSTER_LABELS.get(cluster_id, "Unknown")

        
        return jsonify({
            'success': True,
            'iiss_score': float(result_df['IISS'].iloc[0]),
            'cluster': severity_label,
            'message': 'Prediction completed successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)