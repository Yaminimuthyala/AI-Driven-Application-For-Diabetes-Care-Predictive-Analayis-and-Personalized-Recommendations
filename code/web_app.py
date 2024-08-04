import streamlit as st
import pickle
import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

st.set_page_config(page_title='Diabetes Prediction Center', layout='wide')

def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #add8e6;
        }
        .stButton > button {
            display: block;
            margin: auto;
            width: 80%;  # Adjusted to better align with the images
            margin-top: 10px;
            margin-bottom: 10px;
        }
        h1, h2, h3, h4, h5, h6, .stMarkdown, .stText {
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True
    )

class ChatBot:
    def __init__(self) -> None:
        self.key_count = 0
        env_file_path = ".env"
        load_dotenv(env_file_path)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.0)
        self.memory = ConversationBufferMemory()
    
    def process_chat(self, user_input):
        conversation = ConversationChain(llm=self.llm, memory=self.memory, verbose=True)
        self.memory.load_memory_variables({})
        response = conversation.predict(input=user_input)
        self.memory.save_context({"input": user_input}, {"output": response})
        user_question = f"You: {user_input}"
        bot_response = f"LifeStyleBot: {response}"
        return user_question, bot_response

chat_bot_obj = ChatBot()

def main_page():
    set_background()
    st.write('<h2>Welcome to Diabetes Prediction Center</h2>', unsafe_allow_html=True)
    st.write("<h4>We provide a quick prediction to assess your risk of diabetes based on your health inputs.</h4>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image('../img/image-1.jpg', use_column_width=True)
        
        if st.button('Get Started'):
            st.session_state['navigation'] = 'options'
            st.experimental_rerun()


def options_page():
    set_background()
    st.title('Choose a Service')

    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        st.image('../img/image5.jpeg', use_column_width=True)
        if st.button('Diabetes Prediction'):
            st.session_state['navigation'] = 'predict'
            st.experimental_rerun()

    with col2:
        st.image('../img/image6.jpeg', use_column_width=True)
        if st.button('LifeStyle Bot'):
            st.session_state['navigation'] = 'chat_bot'
            st.experimental_rerun()

    with col3:
        st.image('../img/bmi_image.jpg', use_column_width=True)
        if st.button('Check your Body Mass Index'):
            st.session_state['navigation'] = 'bmi_calculator'
            st.experimental_rerun()

    if st.button('Go Back'):
        st.session_state['navigation'] = 'home'
        st.experimental_rerun()
        
        
#loading the scaler
with open('../scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

def prediction_page():
    set_background()
    st.title('Diabetes Prediction')

    gender = st.selectbox('Gender:', ['Male', 'Female'])
    age = st.number_input('Age:', min_value=0, max_value=120, step=1)
    hypertension = st.selectbox('Do you have hypertension?', ['Yes', 'No'])
    heart_disease = st.selectbox('Do you have heart disease?', ['Yes', 'No'])
    smoking_history = st.selectbox('Smoking History:', ['No Info', 'current', 'ever', 'former', 'never', 'not current'])
    bmi = st.number_input('Body Mass Index (BMI):', min_value=10.0, max_value=50.0, step=0.1)
    hba1c_level = st.number_input('HbA1c Level:', min_value=4.0, max_value=20.0, step=0.1)
    blood_glucose_level = st.number_input('Blood Glucose Level (mg/dL):', min_value=50, max_value=400, step=1)

    gender = 1 if gender == 'Male' else 0
    hypertension = 1 if hypertension == 'Yes' else 0
    heart_disease = 1 if heart_disease == 'Yes' else 0
    smoking_history = {'No Info': 0, 'current': 1, 'ever': 2, 'former': 3, 'never': 4, 'not current': 5}[smoking_history]


    features = [gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose_level]
    
    # Convert to 2D array and scale
    features_array = [features]
    features_scaled = scaler.transform(features_array)
    
    # Prediction button
    if st.button('Predict'):
        with open('../xgboost_ros_model.pkl', 'rb') as file:
            model = pickle.load(file)
        result = model.predict(features_scaled)[0]
        
        # Display result
        if result == 1:
            st.success('The prediction result is: Diabetic')
        else:
            st.success('The prediction result is: Not Diabetic')

    if st.button('Go Back'):
        st.session_state['navigation'] = 'options'
        st.experimental_rerun()


    
def chat_bot_page():
    set_background()
    st.title('LifeStyle Bot')
    st.write("Ask any diet or health related questions and I'll answer.")

    if st.button('Go Back', key='go_back_chat_bot'):
        st.session_state['navigation'] = 'options'
        st.experimental_rerun()

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'input_key' not in st.session_state:
        st.session_state['input_key'] = 0

    user_input = st.text_input("Your question:", key=f"user_input_{st.session_state['input_key']}", on_change=None)

    button_row = st.columns([1, 1])  # Adjust the proportions to align buttons to the left
    with button_row[0]:  # This is the "Send" button column
        if st.button('Send', key='send_button'):
            if user_input.strip() != "":
                # Process and display chat interaction
                user_question, bot_response = chat_bot_obj.process_chat(user_input)
                st.session_state['chat_history'].append(user_question)
                st.session_state['chat_history'].append(bot_response)
                st.session_state.modified = True
                st.session_state['input_key'] += 1  # Increment key to clear the input field
                st.experimental_rerun()  # Rerun to reflect the updated state
    with button_row[1]:  # This is the "Clear Chat" button column
        if st.button('Clear Chat', key='clear_chat'):
            st.session_state['chat_history'] = []
            st.experimental_rerun()

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in reversed(st.session_state['chat_history']):
            chat_bot_obj.key_count += 1
            st.text_area("Chat", value=message, height=50, disabled=True, key=chat_bot_obj.key_count)

def bmi_calculator_page():
    set_background()
    st.title('Check your Body Mass Index')
    st.write("Enter your weight in pounds and height in feet and inches to calculate your BMI.")

    weight_lbs = st.number_input('Weight in pounds (lbs):', min_value=10.0, max_value=500.0, step=0.1)
    height_ft = st.number_input('Height (feet):', min_value=2, max_value=7, step=1)
    height_in = st.number_input('Height (inches):', min_value=0, max_value=11, step=1)

    total_height_in = height_ft * 12 + height_in
    height_m = total_height_in * 0.0254
    weight_kg = weight_lbs * 0.453592

    if st.button('Calculate BMI'):
        if height_m > 0:
            bmi = weight_kg / (height_m ** 2)
            st.write('Your BMI is: {:.2f}'.format(bmi))

            if bmi < 18.5:
                st.error('You are underweight.')
            elif bmi >= 18.5 and bmi < 25:
                st.success('You have a normal weight.')
            elif bmi >= 25 and bmi < 30:
                st.warning('You are overweight.')
            else:
                st.error('You are obese.')
        else:
            st.error('Height must be greater than zero to calculate BMI.')

    if st.button('Go Back'):
        st.session_state['navigation'] = 'options'
        st.experimental_rerun()


if 'navigation' not in st.session_state:
    st.session_state['navigation'] = 'home'

if st.session_state['navigation'] == 'home':
    main_page()
elif st.session_state['navigation'] == 'options':
    options_page()
elif st.session_state['navigation'] == 'predict':
    prediction_page()
elif st.session_state['navigation'] == 'chat_bot':
    chat_bot_page()
elif st.session_state['navigation'] == 'bmi_calculator':
    bmi_calculator_page()
