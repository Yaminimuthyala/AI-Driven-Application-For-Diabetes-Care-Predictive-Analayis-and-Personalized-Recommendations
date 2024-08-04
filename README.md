# AI-Driven-Application-For-Diabetes-Care-Predictive-Analayis-and-Personalized-Recommendations
## Description
This project focuses on predicting diabetes using machine learning techniques, covering data collection, preprocessing, model development, evaluation, and the creation of a real-time, user-friendly interface. It stands out in the field of data mining for diabetes prediction by using advanced sampling techniques and optimizing classification algorithms to address the significant issue of class imbalance, which is often overlooked in standard predictive models. Through targeted data preprocessing and the strategic use of SMOTE, ADASYN, and RandomOverSampler with ensemble and classification classifiers, the project enhances model sensitivity and precision. This approach not only boosts the detection rates of diabetic cases, as shown by improved recall and F1 scores, but also reduces the risk of false positives, which is crucial in medical diagnostics. The comprehensive and balanced methodology ensures the model's effectiveness in accurately predicting diabetes and its applicability in real-world clinical settings. Furthermore, the application includes a real-time, user-friendly interface that offers immediate, personalized risk assessments for predicting diabetes, along with tailored dietary and lifestyle recommendations, thus enhancing user engagement and promoting proactive health management.
### Dataset 
DiabetesDataset.csv

### Code Implementation
Code Folder: Contains all the necessary scripts

**common_code.py**: Data loading and preprocessing steps.

**data_eda.py**: Exploratory data analysis steps.

**model_train.py**: Model development and evaluation steps.

**best_model.py**: Best saved model and analysis of confounding variables.

**web_app.py**: Web application script.

**Standardization Pickle File**: scaler.pkl.

**Best Saved Model Pickle File**: xgboost_ros_model.pkl.


### Technical Report
ProjectReport.pdf

### Install Required Python packages
pip install seaborn scikit-learn xgboost imbalanced-learn streamlit langchain
pip install streamlit pickle-mixin python-dotenv langchain openai
### To run on python virtual environment please use below commands
Create a new virtual environment
python -m venv myenv
### Activate the virtual environment
source myenv/bin/activate --> For macOS/Linux
myenv\Scripts\activate --> For Windows
### Install the necessary packages
pip install langchain==0.0.316 openai==0.28.1 (Or)
pip install langchain==0.1.17 openai==1.26.0
### Run the EDA code
python code/data_eda.py

### Run the model code
python code/model_train.py
python code/model_train.py --show-plots
### Run the best model
python code/best_model.py
python code/best_model.py --show-plots
NOTE: Use --show-plots to see plots while training models or while running best model

### Web Application
cd code

streamlit run web_app.py
