import streamlit as st
import pandas as pd
import joblib 
 
st.title('French4U 🇫🇷')
st.write('Enter a French text below and click the button to analyze its difficulty. The difficulty is rated on a scale from 0 to 6, where 0 corresponds to a basic A1 level and 6 denotes proficiency at the C2 level.')
user_input = st.text_area("Insert your text here", height=200)

# Load the model (make sure the model file is in the same directory or specify the full path)
model = joblib.load('https://github.com/allobo49/SBB/best_model_LR_features.joblib')  # Update the path as needed

if st.button('Click here to Analyze Difficulty'):
    if user_input:
        # Assuming your model expects a DataFrame
        data = pd.DataFrame([user_input], columns=['sentence'])
        # Predicting the difficulty
        prediction = model.predict(data)[0]
        # Display the prediction
        st.write(f'The predicted difficulty level of the text is: **{prediction}**')
    else:
        st.write('Please enter a text to analyze.')
