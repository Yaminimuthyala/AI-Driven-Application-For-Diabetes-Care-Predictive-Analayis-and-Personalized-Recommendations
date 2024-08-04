
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from sklearn.preprocessing import LabelEncoder
import imblearn.over_sampling as os
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from common_code import Dataset, Metrics
import argparse


class BaseModel():
    def __init__(self, show_plots=False) -> None:
        self.dataset_object = Dataset("DiabetesDataset.csv")
        self.feature_names = self.dataset_object.data.columns.drop('diabetes').tolist()
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.dataset_object.preprocess_data()
        self.metrics_obj = Metrics(show_plots)
        self.smote = os.SMOTE(random_state=42)
        self.adasyn = os.ADASYN(random_state=42)
        self.random_oversampler = os.RandomOverSampler(random_state=42)
        self.select_k_best = SelectKBest(score_func=chi2, k=5)

    def get_data(self):
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test

    def get_k_best_data(self, X_train, y_train, X_val, X_test):
        X_train_select_k_best = self.select_k_best.fit_transform(X_train, y_train)
        X_val_select_k_best = self.select_k_best.transform(X_val)
        X_test_select_k_best = self.select_k_best.transform(X_test)
        return X_train_select_k_best, X_val_select_k_best, X_test_select_k_best
    
    def get_scaled_data(self, scaler_save_path):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        with open(scaler_save_path, 'wb') as file:
            pickle.dump(scaler, file)
        X_val_scaled = scaler.transform(self.X_val)
        X_test_scaled = scaler.transform(self.X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled

    def get_smote_data(self, X_train_scaled, y_train):
        X_train_smote, y_train_smote = self.smote.fit_resample(X_train_scaled, y_train)
        return X_train_smote, y_train_smote
    
    def get_adasyn_data(self, X_train_scaled, y_train):
        X_train_adasyn, y_train_adasyn = self.adasyn.fit_resample(X_train_scaled, y_train)
        return X_train_adasyn, y_train_adasyn
    
    def get_ros_data(self, X_train_scaled, y_train):
        X_train_ros, y_train_ros = self.random_oversampler.fit_resample(X_train_scaled, y_train)
        return X_train_ros, y_train_ros

    def train_evaluate(self, model, display_text, X_train, y_train, X_val, y_val, X_test, y_test, is_nb_model=False):
        print(f"\nTraining and Evaluating {display_text}")
        if is_nb_model:
            X_train, X_val, X_test = self.get_k_best_data(X_train, y_train, X_val, X_test)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        self.metrics_obj.display_all_metrics(model, display_text, X_train, X_val, X_test, y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred)
        self.metrics_obj.display_classification_report(y_test, y_test_pred)
        self.metrics_obj.plot_all_confusion_matrix(display_text, y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument("--show-plots", action="store_true", default=False, help="Whether to show plots or not")
    args = parser.parse_args()
    
    # Base Models
    base_model = BaseModel(args.show_plots)
    X_train, y_train, X_val, y_val, X_test, y_test = base_model.get_data()    
    base_model.train_evaluate(RandomForestClassifier(random_state=42), "Random Forest On Unbalanced Data", X_train, y_train, X_val, y_val, X_test, y_test)
    base_model.train_evaluate(xgb.XGBClassifier(random_state=42), "XGBoost On Unbalanced Data", X_train, y_train, X_val, y_val, X_test, y_test)
    base_model.train_evaluate(GaussianNB(), "Naive Bayes On Unbalanced Data", X_train, y_train, X_val, y_val, X_test, y_test, is_nb_model=True)

    # Scaled Dataset
    X_train_scaled, X_val_scaled, X_test_scaled = base_model.get_scaled_data("scaler.pkl")
    base_model.train_evaluate(RandomForestClassifier(random_state=42), "Random Forest On Scaled Data", X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)
    base_model.train_evaluate(xgb.XGBClassifier(random_state=42), "XGBoost On Scaled Data", X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)
    
    # Balanced + Scaled dataset for RF and XGBoost
    X_train_smote, y_train_smote = base_model.get_smote_data(X_train_scaled, y_train)
    X_train_adasyn, y_train_adasyn = base_model.get_adasyn_data(X_train_scaled, y_train)
    X_train_ros, y_train_ros = base_model.get_ros_data(X_train_scaled, y_train)

    # RF on balanced dataset
    base_model.train_evaluate(RandomForestClassifier(random_state=42), "Random Forest On SMOTE balanced Data", X_train_smote, y_train_smote, X_val_scaled, y_val, X_test_scaled, y_test)
    base_model.train_evaluate(RandomForestClassifier(random_state=42), "Random Forest On ADASYN balanced Data", X_train_adasyn, y_train_adasyn, X_val_scaled, y_val, X_test_scaled, y_test)
    base_model.train_evaluate(RandomForestClassifier(random_state=42), "Random Forest On Random Oversampler balanced Data", X_train_ros, y_train_ros, X_val_scaled, y_val, X_test_scaled, y_test)

    # XGBoost on balanced dataset
    base_model.train_evaluate(xgb.XGBClassifier(random_state=42), "XGBoost On SMOTE balanced Data", X_train_smote, y_train_smote, X_val_scaled, y_val, X_test_scaled, y_test)
    base_model.train_evaluate(xgb.XGBClassifier(random_state=42), "XGBoost On ADASYN balanced Data", X_train_adasyn, y_train_adasyn, X_val_scaled, y_val, X_test_scaled, y_test)
    base_model.train_evaluate(xgb.XGBClassifier(random_state=42), "XGBoost On Random Oversampler balanced Data", X_train_ros, y_train_ros, X_val_scaled, y_val, X_test_scaled, y_test)

    # Balanced dataset for NB
    X_train_smote, y_train_smote = base_model.get_smote_data(X_train, y_train)
    X_train_adasyn, y_train_adasyn = base_model.get_adasyn_data(X_train, y_train)
    X_train_ros, y_train_ros = base_model.get_ros_data(X_train, y_train)

    # NB on balanced dataset
    base_model.train_evaluate(GaussianNB(), "Naive Bayes On SMOTE balanced Data", X_train_smote, y_train_smote, X_val, y_val, X_test, y_test, is_nb_model=True)
    base_model.train_evaluate(GaussianNB(), "Naive Bayes On ADASYN balanced Data", X_train_adasyn, y_train_adasyn, X_val, y_val, X_test, y_test, is_nb_model=True)
    base_model.train_evaluate(GaussianNB(), "Naive Bayes On Random Oversampler balanced Data", X_train_ros, y_train_ros, X_val, y_val, X_test, y_test, is_nb_model=True)
