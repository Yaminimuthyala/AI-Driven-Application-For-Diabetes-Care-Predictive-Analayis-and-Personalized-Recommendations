import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder


class Dataset():
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = self.load_data()

    def load_data(self):
        print("Loading data")
        data = pd.read_csv(self.dataset_path)
        return data

    def display_data(self, num_rows=10):
        print(f"\nFirst {num_rows} rows of dataset")
        print(self.data.head(num_rows))
        print("\nData Statistics")
        print(self.data.describe())
        print("\nData Information")
        print(self.data.info())
        print("\nThere are no Null values in our dataset.")
    
    def data_eda(self, enable_print=True):
        # Handling duplicates
        duplicate_diabetes_data = self.data[self.data.duplicated()]
        print("Number of duplicate rows: ", duplicate_diabetes_data.shape)

        diabetes_data = self.data.drop_duplicates()
        duplicate_diabetes_data = diabetes_data[diabetes_data.duplicated()]
        print("checking whether duplicate rows are dropped or not: ", duplicate_diabetes_data.shape)

        #Displaying unique values in the 'gender' column
        unique_genders = diabetes_data['gender'].unique()
        print("Unique values in the 'gender' column:", unique_genders)
        #Displaying counts of each unique value in the 'gender' column
        gender_counts = diabetes_data['gender'].value_counts()
        print("Counts of each unique value in the 'gender' column:\n", gender_counts)

        # Drop rows with other gender
        diabetes_data = diabetes_data[diabetes_data['gender'] != 'Other']
        return diabetes_data

    def do_encoding(self, diabetes_data):
        categorical_features = diabetes_data.select_dtypes(include=['object']).columns
        label_encoder = LabelEncoder()
        label_mappings={}
        for column in categorical_features:
            diabetes_data[column] = label_encoder.fit_transform(diabetes_data[column])
            label_mappings[column] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))   
        return diabetes_data

    def preprocess_data(self):
        print("\n\nPreprocessing Data")
        diabetes_data = self.data_eda()
        diabetes_data = self.do_encoding(diabetes_data)
        X = diabetes_data.drop(columns='diabetes').values
        y = diabetes_data['diabetes'].values
        return self.train_test_val_split(X, y)
            
    def dataset_validation(self, show_plots=False):
        num_rows, num_columns = self.data.shape
        print("\nNumber of rows:", num_rows)
        print("Number of columns:", num_columns)

        if show_plots:
            print("Distribution of Data")
            self.data.hist(figsize=(10, 10))
            plt.tight_layout()
            plt.show()
    
    def train_test_val_split(self, X, y):
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)  

        print(f"Training set size: {X_train.shape[0]} samples")
        print(f"Validation set size: {X_val.shape[0]} samples")
        print(f"Test set size: {X_test.shape[0]} samples")
        return X_train, y_train, X_val, y_val, X_test, y_test


class Metrics():
    def __init__(self, show_plots=False):
        self.show_plots = show_plots

    def plot_confusion_matrix(self, cm, title="Confusion Matrix"):
        plt.figure(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square=True, cmap='Blues')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.title(title, size = 12)

    def plot_all_confusion_matrix(self, display_text, y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred):
        if not self.show_plots:
            return
        cm_train = confusion_matrix(y_train, y_train_pred)
        cm_val = confusion_matrix(y_val, y_val_pred)
        cm_test = confusion_matrix(y_test, y_test_pred)

        self.plot_confusion_matrix(cm_train, f"Confusion Matrix - Training set - {display_text}")
        self.plot_confusion_matrix(cm_val, f"Confusion Matrix - Validation set - {display_text}")
        self.plot_confusion_matrix(cm_test, f"Confusion Matrix - Test set - {display_text}")
        plt.show()
    
    def display_metrics(self, model, display_text, X_set, y_set, y_set_pred, type):
        print(f"\n{type} dataset - {display_text}")
        print("Accuracy:", accuracy_score(y_set, y_set_pred))
        print("Precision:", precision_score(y_set, y_set_pred))
        print("Recall:", recall_score(y_set, y_set_pred))
        print("F1 Score:", f1_score(y_set, y_set_pred))
        print("ROC AUC:", roc_auc_score(y_set, model.predict_proba(X_set)[:, 1]))
    
    def display_all_metrics(self, model, display_text, X_train, X_val, X_test, y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred):
        print(f"Evaluation Metrics {display_text}:\n")
        self.display_metrics(model, display_text, X_train, y_train, y_train_pred, "Training")
        self.display_metrics(model, display_text, X_val, y_val, y_val_pred, "Validation")
        self.display_metrics(model, display_text, X_test, y_test, y_test_pred, "Testing")

    def display_classification_report(self, y_test, y_test_pred):
        print("Classification Report for Test Data:")
        print(classification_report(y_test, y_test_pred, target_names=['Non-Diabetic(0)', 'Diabetic(1)']))
