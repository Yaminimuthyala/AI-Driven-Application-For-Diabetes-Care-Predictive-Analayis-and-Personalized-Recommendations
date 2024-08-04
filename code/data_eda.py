import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import rand_score, jaccard_score
from sklearn.cluster import KMeans
from common_code import Dataset
from model_train import BaseModel


def bivariate_analysis(data):
    sns.countplot(x='gender', data=data)
    plt.title('Gender Distribution')
    plt.show()

    sns.countplot(x='smoking_history', data=data)
    plt.title('Smoking History Distribution')
    plt.show()

    data.hist(figsize=(10, 10))
    plt.tight_layout()
    plt.show()

    sns.countplot(x='gender', hue='diabetes', data=data, palette='coolwarm', hue_order=[0, 1])

    plt.title('Distribution of Diabetes Cases by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Diabetes', labels=['Non-Diabetic', 'Diabetic'])
    plt.tight_layout()
    plt.show()

    cross_tab = pd.crosstab(data['smoking_history'], data['diabetes'])
    plt.figure(figsize=(10, 6))
    cross_tab.plot(kind='bar', stacked=True, colormap='cividis')
    plt.title('Diabetes Distribution by Smoking History')
    plt.xlabel('Smoking History')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Diabetes', labels=['Non-Diabetic', 'Diabetic'])
    plt.tight_layout()
    plt.show()

    diabetic_data = data[data['diabetes'] == 1]['HbA1c_level']
    non_diabetic_data = data[data['diabetes'] == 0]['HbA1c_level']

    diabetic_kde = gaussian_kde(diabetic_data)
    non_diabetic_kde = gaussian_kde(non_diabetic_data)

    x_values = np.linspace(data['HbA1c_level'].min(), data['HbA1c_level'].max(), 100)

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, diabetic_kde(x_values), color='orange', label='Diabetic')
    plt.plot(x_values, non_diabetic_kde(x_values), color='blue', label='Non-Diabetic')
    plt.title('Estimation of HbA1c Level by Diabetes Status')
    plt.xlabel('HbA1c Level')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

    age_bins = [0, 20, 40, 60, 80, 100]
    age_labels = ['0-20', '20-40', '40-60', '60-80', '80-100']

    data['age_group'] = pd.cut(data['age'], bins=age_bins, labels=age_labels, right=False)

    plt.figure(figsize=(10, 6))
    sns.countplot(x='age_group', hue='diabetes', data=data, palette='inferno')
    plt.title('Distribution of Diabetes Cases by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Diabetes', labels=['Non-Diabetic', 'Diabetic'])
    plt.tight_layout()
    plt.show()

    sns.boxplot(x='diabetes', y='bmi', data=data)
    plt.title('BMI vs Diabetes')

    plt.show()

    diabetic_data = data[data['diabetes'] == 1]['blood_glucose_level']
    non_diabetic_data = data[data['diabetes'] == 0]['blood_glucose_level']

    diabetic_kde = gaussian_kde(diabetic_data)
    non_diabetic_kde = gaussian_kde(non_diabetic_data)

    x_values = np.linspace(data['blood_glucose_level'].min(), data['blood_glucose_level'].max(), 100)

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, diabetic_kde(x_values), color='red', label='Diabetic')
    plt.plot(x_values, non_diabetic_kde(x_values), color='green', label='Non-Diabetic')
    plt.title('KDE of Blood Glucose Levels by Diabetes Status')
    plt.xlabel('Blood Glucose Level')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()


def multivariate_analysis(data):
    data['bmi_category'] = pd.cut(data['bmi'], bins=[0, 18.5, 24.9, 29.9, 40, np.inf], labels=['Underweight', 'Normal', 'Overweight', 'Obese', 'Severely Obese'])

    ct = pd.crosstab(index=[data['diabetes'], data['gender']], columns=data['bmi_category'], normalize='index')

    ct.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Stacked Bar Chart of BMI Categories by Diabetes and Gender')
    plt.xlabel('Diabetes Status, Gender')
    plt.ylabel('Proportion')
    plt.legend(title='BMI Category')
    plt.show()

    g = sns.FacetGrid(data, col="gender", height=5, aspect=1)
    g.map_dataframe(sns.boxplot, x='diabetes', y='age', palette='Set2')
    g.set_axis_labels("Diabetes Status", "Age")
    g.add_legend()
    g.fig.suptitle('Age Distribution by Diabetes Status Across Genders', fontsize=16)
    plt.subplots_adjust(top=0.85)
    plt.show()

    diabetes_counts = data['diabetes'].value_counts()
    plt.figure(figsize=(4, 4))
    plt.pie(diabetes_counts, labels=diabetes_counts.index, autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
    plt.title('Distribution of Diabetes')
    plt.axis('equal')
    plt.show()

def label_encoding_heatmap(data):
    diabetes_data = data.copy()
    categorical_features = diabetes_data.select_dtypes(include=['object']).columns
    label_encoder = LabelEncoder()
    label_mappings={}
    for column in categorical_features:
        diabetes_data[column] = label_encoder.fit_transform(diabetes_data[column])
        label_mappings[column] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    diabetes_data = diabetes_data.drop(columns=['age_group','bmi_category', 'diabetes'])

    corr_matrix = diabetes_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

def cluster_analysis(X_train_scaled, y_train, X_test_scaled, y_test):
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=20)
    kmeans_clusters = kmeans.fit_predict(X_train_scaled)

    # calculate the rand score on training dataset
    rand_statistic = rand_score(y_train, kmeans_clusters)
    print("Rand Index (RI) on training dataset:", rand_statistic)

    jaccard_coefficient = jaccard_score(y_train, kmeans_clusters, average='micro')
    print("Jaccard Coefficient on training dataset:", jaccard_coefficient)

    # clustering on the test data
    predicted_data_clusters = kmeans.predict(X_test_scaled)

    rand_statistic = rand_score(y_test, predicted_data_clusters)
    print("Rand Index (RI) on testing dataset:", rand_statistic)
    jaccard_coefficient = jaccard_score(y_test, predicted_data_clusters, average='micro')
    print("Jaccard Coefficient on testing dataset:", jaccard_coefficient)

    pca = PCA()
    pca.fit(X_train_scaled)

    # use the first two pc
    X_projected = pca.transform(X_test_scaled)[:, :2]

    # Create a color map dictionary to map each unique label to a specific color
    unique_labels = np.unique(y_test)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: color for label, color in zip(unique_labels, colors)}

    # Plot the projected data with color-coded labels
    plt.figure(figsize=(8, 6))
    for label in unique_labels:
        indices = y_test == label
        plt.scatter(X_projected[indices, 0], X_projected[indices, 1], label=f'Label {label}', color=color_map[label], alpha=0.5)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Projection of Data onto First Two Principal Components\n with True Labels')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Create a color map dictionary to map each unique label to a specific color
    unique_labels = np.unique(predicted_data_clusters)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: color for label, color in zip(unique_labels, colors)}

    # Plot the projected data with color-coded labels
    plt.figure(figsize=(8, 6))
    for label in unique_labels:
        indices = predicted_data_clusters == label
        plt.scatter(X_projected[indices, 0], X_projected[indices, 1], label=f'Label {label}', color=color_map[label], alpha=0.5)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Projection of Data onto First Two Principal Components\nwith K-Means Clustering Labels')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    dataset_obj = Dataset("DiabetesDataset.csv")
    dataset_obj.display_data()
    dataset_obj.dataset_validation()
    data = dataset_obj.data_eda()

    bivariate_analysis(data)
    multivariate_analysis(data)
    label_encoding_heatmap(data)
    data = data.drop(columns=['age_group','bmi_category'])
    data = dataset_obj.do_encoding(data)
    X = data.drop(columns='diabetes').values
    y = data['diabetes'].values
    X_train, y_train, X_val, y_val, X_test, y_test = dataset_obj.train_test_val_split(X, y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    cluster_analysis(X_train_scaled, y_train, X_test_scaled, y_test)
    