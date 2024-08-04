import xgboost as xgb
from model_train import BaseModel
import joblib
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


def get_user_inputs(model):
    with open('scaler.pkl', 'rb') as file:
        loaded_scaler = pickle.load(file)

    gender = input("Enter gender (Female/Male): ")
    age = int(input("Enter age: "))
    hypertension = int(input("Enter hypertension (0 for No, 1 for Yes): "))
    heart_disease = int(input("Enter heart disease (0 for No, 1 for Yes): "))
    smoking_history = input("Enter smoking history (No Info, current, ever, former, never, not current): ")
    bmi = float(input("Enter BMI: "))
    HbA1c_level = float(input("Enter HbA1c level: "))
    blood_glucose_level = float(input("Enter blood glucose level: "))

    gender_encoded = 0 if gender.lower() == 'female' else 1
    smoking_history_mapping = {'No Info': 0, 'current': 1, 'ever': 2, 'former': 3, 'never': 4, 'not current': 5}
    smoking_history_encoded = smoking_history_mapping[smoking_history]
    user_data = np.array([[gender_encoded, age, hypertension, heart_disease, smoking_history_encoded, bmi, HbA1c_level, blood_glucose_level]])

    user_data_scaled = loaded_scaler.transform(user_data)

    prediction = model.predict(user_data_scaled)

    print("\n")
    print("Diabetic" if prediction[0] == 1 else "Not Diabetic")


def train_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test):
    model.fit(X_train, y_train)
    
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    print("Validation Results:")
    print(classification_report(y_val, y_val_pred))
    
    print("Test Results:")
    print(classification_report(y_test, y_test_pred))
    
    print("Detailed Metrics for Test Data:")
    print("Accuracy Score:", accuracy_score(y_test, y_test_pred))
    print("Recall Score:", recall_score(y_test, y_test_pred, average='macro'))  # 'macro' average to treat all classes equally
    print("F1 Score:", f1_score(y_test, y_test_pred, average='macro'))


def stratify_age(diabetes_data):
    diabetes_data['age_group'] = pd.cut(diabetes_data['age'], bins=[0, 30, 60, 90, 100], labels=['0-30', '31-60', '61-80', '81+'])

    X_train, X_temp, y_train, y_temp = train_test_split(
        diabetes_data.drop(columns='diabetes'),
        diabetes_data['diabetes'], 
        test_size=0.3,  
        random_state=42, 
        stratify=diabetes_data['age_group']
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp, 
        test_size=1/3,  
        random_state=42, 
        stratify=X_temp['age_group']
    )

    X_train = X_train.drop(columns='age_group')
    X_val = X_val.drop(columns='age_group')
    X_test = X_test.drop(columns='age_group')

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    ros_stratifying = RandomOverSampler(random_state=42)
    X_train_ros, y_train_ros = ros_stratifying.fit_resample(X_train_scaled, y_train)
    return X_train_ros, y_train_ros, X_val_scaled, y_val, X_test_scaled, y_test





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Best Model Training & Running Script")
    parser.add_argument("--show-plots", action="store_true", default=False, help="Whether to show plots or not")
    args = parser.parse_args()
    
    base_model = BaseModel(args.show_plots)
    X_train, y_train, X_val, y_val, X_test, y_test = base_model.get_data()

    # Scaled Dataset
    X_train_scaled, X_val_scaled, X_test_scaled = base_model.get_scaled_data("scaler.pkl")
    
    # Balanced dataset
    X_train_ros, y_train_ros = base_model.get_ros_data(X_train_scaled, y_train)
    
    xgboost_ros = xgb.XGBClassifier(random_state=42)
    base_model.train_evaluate(xgboost_ros, "XGBoost On Random Oversampler balanced Data", X_train_ros, y_train_ros, X_val_scaled, y_val, X_test_scaled, y_test)

    joblib.dump(xgboost_ros, 'xgboost_ros_model.pkl')
    with open('xgboost_ros_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # User Input
    get_user_inputs(model)
    
    feature_importances = xgboost_ros.feature_importances_
    importances_df = pd.DataFrame({
        'Feature': base_model.feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    print(importances_df)

    # Plotting feature importances
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importances_df, palette="viridis")
    plt.title('Feature Importance from XGBoost with RandomOverSampler', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.xticks(rotation=45)  
    plt.show()

    print("\n\n\n\n'age' is our cofounding variable and we want to stratify our data and train it on XGBOOST randomoversampler model to see the performance\n\n\n\n")

    # Stratify Age
    diabetes_data = base_model.dataset_object.data
    diabetes_data = diabetes_data.drop_duplicates()
    diabetes_data = diabetes_data[diabetes_data['gender'] != 'Other']
    diabetes_data = base_model.dataset_object.do_encoding(diabetes_data)
    X_train_ros, y_train_ros, X_val_scaled, y_val, X_test_scaled, y_test = stratify_age(diabetes_data)
    xgboost_ros_stratifying = xgb.XGBClassifier(random_state=42)
    train_evaluate(xgboost_ros_stratifying, X_train_ros, y_train_ros, X_val_scaled, y_val, X_test_scaled, y_test)
    feature_importances = xgboost_ros_stratifying.feature_importances_
    
    importances_df = pd.DataFrame({
    'Feature': base_model.feature_names,
    'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)  

    print(importances_df)

    # Plotting feature importances after Stratify Age
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importances_df, palette="viridis")
    plt.title('Feature Importance from XGBoost RandomOverSampler on Stratified data', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.xticks(rotation=45)  
    plt.show()
