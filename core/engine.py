import joblib
from pathlib import Path

def load_data(X):
    pass

def load_pipeline():
    
    MODEL_PATH = Path("./models/xgb_loan_pipeline_v1.pkl")
    pipeline = None
    try:
        pipeline = joblib.load(MODEL_PATH)
        print('The Model / Pipeline was Loaded Successfully.')
    except Exception as e:
        print(f'Oops ! Unable To load Model / Pipeline \n Error Information : {e}')

    return pipeline

def main():
    pass

x = load_pipeline()
