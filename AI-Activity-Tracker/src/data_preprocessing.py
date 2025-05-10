import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

FEATURE_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'features', 'pose_features.csv')
PROC_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

def load_raw():
    return pd.read_csv(FEATURE_CSV)

def preprocess(df):
    df = df.dropna()
    le = LabelEncoder()
    df['activity_code'] = le.fit_transform(df['activity'])
    X = df.drop(['activity', 'activity_code'], axis=1)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, df['activity_code'], le, scaler

def save_processed(X, y):
    os.makedirs(PROC_DIR, exist_ok=True)
    pd.DataFrame(X).to_csv(os.path.join(PROC_DIR, 'X.csv'), index=False)
    pd.DataFrame(y, columns=['activity_code'])\
      .to_csv(os.path.join(PROC_DIR, 'y.csv'), index=False)

if __name__ == "__main__":
    df = load_raw()
    X, y, le, scaler = preprocess(df)
    save_processed(X, y)
    print("Preprocesamiento completado.")
