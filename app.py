
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# import the model
pipe = pickle.load(open('model.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title('Welcome To Car Valuation')
Intakhab = f'<a href="https://github.com/intakhab1/car-val-intakhab">Developed by @Intakhab</a>'
st.markdown(Intakhab, unsafe_allow_html=True)

# company
company = st.selectbox('Company',df['company'].unique())

# model
car_model = st.selectbox('Model',df['name'].unique())

# year
year = st.selectbox('Year',df['year'].unique())

# fuel_type
fuel_type = st.selectbox('Fuel Type',df['fuel_type'].unique())

# driven
driven = st.number_input('kms driven')

if st.button('Predict Price'):
    prediction = pipe.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                            data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)))
    print(prediction)
    st.title( str(np.round(prediction[0], 2)))






