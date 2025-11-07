import numpy as np
import pandas as pd
import yfinance as yf
import pathlib
import joblib
import warnings
from time import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import inspection

warnings.filterwarnings('ignore')
idx = pd.IndexSlice

path = pathlib.Path(__file__).parent
data = path.parent.parent / "data" / "price" / "AAPL.csv"

df = pd.read_csv(data)
#print(df.info())


def data_prep(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Create binary target: 1 if next day's close > today's close, else 0
    df['target'] = (df['4. close'].shift(-1) > df['4. close']).astype(int)
    
    # Drop the last row (no future price available)
    df = df[:-1].reset_index(drop=True)
    
    target = df['target']
    features = df.drop(['4. close', 'date', 'target'], axis=1)
    return target, features

def train_test_split(target, features, period=6):
    trainDF = features[:-period]
    testDF = features[-period:]
    trainTarget = target[:-period]
    testTarget = target[-period:]
    return trainDF, testDF, trainTarget, testTarget

def train_model(trainDF, trainTarget, testDF, testTarget):
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(trainDF, trainTarget)
    return model

def eval_model(model, testDF, testTarget):
    from sklearn.metrics import accuracy_score, classification_report
    
    y_pred = model.predict(testDF)
    y_pred_proba = model.predict_proba(testDF)[:, 1]
    
    accuracy = accuracy_score(testTarget, y_pred)
    roc_auc = roc_auc_score(testTarget, y_pred_proba)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(testTarget, y_pred))
    
    return roc_auc

def plot_features(model, features):
    importance = model.feature_importances_
    feature_names = features.columns
    plt.barh(feature_names, importance)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.show()

def main():
    df = pd.read_csv(data)
    target, features = data_prep(df)
    print(f"Target distribution:\n{target.value_counts()}\n")
    
    trainDF, testDF, trainTarget, testTarget = train_test_split(target, features)
    model = train_model(trainDF, trainTarget, testDF, testTarget)
    score = eval_model(model, testDF, testTarget)
    plot_features(model, features)

#main init
if __name__ == "__main__":
    main()

