import streamlit as st
import pandas as pd
import joblib 
 
st.title('French4U 🇫🇷')
st.write('Enter a French text below and click the button to analyze its difficulty. The difficulty is rated on a scale from 0 to 6, where 0 corresponds to a basic A1 level and 6 denotes proficiency at the C2 level.')
user_input = st.text_area("Insert your text here", height=200)
st.button('Click here to Analyze Difficulty'):
   
