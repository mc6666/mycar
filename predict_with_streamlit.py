# https://docs.streamlit.io/en/stable/api.html#streamlit.slider
import streamlit as st
import pandas

# load model
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib 

clf = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

def convert_dict(key, dict1):
    return dict1[key]

# 畫面設計
st.markdown('# 車價迴歸預測')
fueltype = st.sidebar.radio('fuel type:', ['gas', 'diesel'])
aspiration = st.sidebar.radio('aspiration:', ['std', 'turbo'])
curbweight = st.sidebar.slider('curb weight:', 1400, 4100, 2500)
carbody = st.sidebar.selectbox('car body:', ['convertible', 'hatchback', 'sedan', 'wagon', 'hardtop'])
carlength = st.sidebar.slider('car length:', 140, 210, 170)
enginetype = st.sidebar.selectbox('engine type:', ['dohc', 'ohcv', 'ohc', 'l', 'rotor', 'ohcf', 'dohcv'])

if st.sidebar.button('預測'):
    fueltype = convert_dict(fueltype, {'gas':0, 'diesel':1})
    aspiration = convert_dict(aspiration, {'std':0, 'turbo':1})
    carbody = convert_dict(carbody, {'convertible':0, 'hatchback':1, 'sedan':2, 'wagon':3, 'hardtop':4})
    enginetype = convert_dict(enginetype, {'dohc':0, 'ohcv':1, 'ohc':2, 'l':3, 'rotor':4, 'ohcf':5, 'dohcv':6})

    # predict
    X=np.array([[fueltype, aspiration, curbweight, carbody, carlength, enginetype]])
    X=scaler.transform(X)
    st.markdown(f'車價={clf.predict(X)[0]:.2f}')
