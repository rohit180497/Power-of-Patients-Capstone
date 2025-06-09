import pandas as pd
import joblib
import json
from sklearn.preprocessing import StandardScaler

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

# Main test runner
def main():
    test_input = {
        "num_head_hit_location": 6,
        "headhit_Not_Sure": 0,
        "headhit_Neck": 1,
        "headhit_Top_Of_Head": 0,
        "headhit_Left_Side_Of_Head": 0,
        "headhit_Front_Of_Head": 1,
        "headhit_Back_Of_Head": 1,
        "headhit_All": 0,
        "headhit_Whiplash": 0,
        "headhit_Right_Side_Of_Head": 1,
        "imm_symp_light_sensitivity": 0,
        "imm_symp_headache": 0,
        "imm_symp_dazed_or_vacant_stare": 1,
        "imm_symp_dizziness": 0,
        "imm_symp_disorientation": 1,
        "imm_symp_nausea": 0,
        "imm_symp_confusion": 1,
        "imm_symp_incoherent_speech": 1,
        "imm_symp_memory_loss": 1,
        "event_desc_car": 0,
        "event_desc_fall": 0,
        "event_desc_severe": 0,
        "injury_from_Accident": 0,
        "injury_from_Fall": 0,
        "injury_from_Collision": 1,
        "injury_from_Sports": 0,
        "injury_from_Assault": 0,
        "injury_from_Stroke": 0,
        "injury_from_Surgery": 0,
        "age_tbi": 30,
        "gender_male": 0,
        "gender_other": 0
    }

    model_runner = TBIModelInference(
        coma_model_path="../artifacts/logreg_model_imm_symp_coma.pkl",
        coma_features_path="../artifacts/features_imm_symp_coma.json",
        loc_model_path="../artifacts/logreg_model_imm_symp_loss_of_consciousness.pkl",
        loc_features_path="../artifacts/features_imm_symp_loss_of_consciousness.json",
        age_scaler_path="../artifacts/scaler_age_tbi.pkl",
        weights_path="../data/raw/weights.csv",
        kmeans_scaler_path="../artifacts/scaler_kmeans.pkl",
        kmeans_features_path="../artifacts/features_kmeans.json",
        kmeans_model_path="../artifacts/kmeans_iiss.pkl"
    )

    result_df = model_runner.run_inference(test_input)
    print(result_df.T)  # Transposed for readability

if __name__ == "__main__":
    main()
