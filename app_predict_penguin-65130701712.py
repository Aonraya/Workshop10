
import streamlit as st
import pickle
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression


# Load model
with open('model-penguin-65130701712.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# Streamlit app
st.title("Penguin Species Prediction")

# Get user input for each variable
island_input = st.selectbox('Select island:', ['Torgersen', 'Biscoe', 'Dream'])
culmen_length_input = st.slider('Enter culmen length:', 0, 50, 37)
culmen_depth_input = st.slider('Enter culmen depth in mm (0 to 50):', min_value=0, max_value=50)
flipper_length_input = st.slider('Enter flipper length in mm (0 to 200):', min_value=0, max_value=200)
body_mass_input = st.slider('Enter body mass in g (1000 to 5000):', min_value=1000, max_value=5000)
sex_input = st.selectbox('Select sex:', ['MALE', 'FEMALE'])

# Create a DataFrame with user input
x_new = pd.DataFrame({
    'island': [island_input],
    'culmen_length_mm': [culmen_length_input],
    'culmen_depth_mm': [culmen_depth_input],
    'flipper_length_mm': [flipper_length_input],
    'body_mass_g': [body_mass_input],
    'sex': [sex_input]
})

# Encoding
x_new['island'] = island_encoder.transform(x_new['island'])
x_new['sex'] = sex_encoder.transform(x_new['sex'])

# Prediction
y_pred_new = model.predict(x_new)
result = species_encoder.inverse_transform(y_pred_new)

# Display result
st.subheader('Prediction Result:')
st.write(f'Predicted Species: {result[0]}')
