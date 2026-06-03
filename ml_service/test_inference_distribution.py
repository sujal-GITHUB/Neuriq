import os
import sys
import pandas as pd
import numpy as np

# Make sure ml_service can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml_service.inference import AnxietyInferenceEngine

def check_distributions():
    engine = AnxietyInferenceEngine()
    
    test_files = [
        "healthy_baseline_control.csv", 
        "acute_stress_state.csv", 
        "generalized_anxiety_locus.csv",
        "panic_attack_simulation.csv"
    ]
    
    for f in test_files:
        path = os.path.join("public/test_files", f)
        if os.path.exists(path):
            df = pd.read_csv(path)
            signals = np.array([df[ch].values for ch in df.columns])
            channels = list(df.columns)
            
            res = engine.predict_from_signals(signals, channels, 128, "ensemble")
            
            # Map probabilities for Comorbid detection like predict/route.ts does
            prob_low = res["probabilities"]["Healthy Controls"]
            prob_mod = res["probabilities"]["Social Anxiety"]
            prob_high = res["probabilities"]["Acute Stress"]
            
            anxiety_level = "Low"
            if prob_mod >= 25.0 and prob_high >= 25.0:
                anxiety_level = "Both"
            elif res["class_prediction"] == "Social Anxiety Disorder":
                anxiety_level = "Moderate"
            elif res["class_prediction"] == "Acute Stress State":
                anxiety_level = "High"
                
            state_label = {
                "Both": "BOTH STRESS & ANXIETY",
                "High": "STRESS ONLY",
                "Moderate": "ANXIETY ONLY",
                "Low": "HEALTHY"
            }[anxiety_level]
            
            print(f"\nFile: {f}")
            print(f"  Diagnosed State: {state_label}")
            print(f"  Confidence: {res['confidence']}%")
            print(f"  Raw Probabilities: {res['probabilities']}")
        else:
            print(f"File {f} not found.")

if __name__ == "__main__":
    check_distributions()
