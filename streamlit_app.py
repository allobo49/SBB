import streamlit as st
import pandas as pd
 
st.title('French4U 🇫🇷')
st.write('Enter a French text below and click the button to analyze its difficulty.')
user_input = st.text_area("Insert your text here", height=200)
