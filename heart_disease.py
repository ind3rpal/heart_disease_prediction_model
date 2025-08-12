import streamlit as st 
import pandas as pd
import random
import pickle 
import numpy as np

st.header('Heart Disease Prediction Using Machine Learning')

data = '''Heart Disease Prediction using Machine Learning Heart disease prevention is critical, and data-driven prediction systems can significantly aid in early diagnosis and treatment. Machine Learning offers accurate prediction capabilities, enhancing healthcare outcomes. In this project, I analyzed a heart disease dataset with appropriate preprocessing. Multiple classification algorithms were implemented in Python using Scikit-learn and Keras to predict the presence of heart disease.

**Algorithms Used:**

**Logistic Regression**

**Naive Bayes**

**Support Vector Machine (Linear)**

**K-Nearest Neighbors**

**Decision Tree**

**Random Forest**

**XGBoost**

**Artificial Neural Network (1 Hidden Layer, Keras)**
'''

st.markdown(data)

st.image('https://repository-images.githubusercontent.com/543430917/d048e410-7994-4a6f-9180-8a3bd8df902b')

with open('heart_disease_pred.pkl','rb') as f:
    chatgpt = pickle.load(f)

# load data 
url = '''https://github.com/ankitmisk/Heart_Disease_Prediction_ML_Model/blob/main/heart.csv?raw=true'''
df = pd.read_csv(url)

st.sidebar.header('Select Features to Predict Heart Disease')
st.sidebar.image('https://art.pixilart.com/bde1aab50bb21a3.gif')

random.seed(15)

all_values=[]

for i in df.iloc[:,:-1]:
    min_value, max_value = df[i].agg(['min','max'])

    var = st.sidebar.slider(f'Select {i} value',int(min_value),int(max_value),
                      random.randint(int(min_value),int(max_value)))

    all_values.append(var)
    
final_values= [all_values]

ans = chatgpt.predict(final_values)[0]

import time

progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Heart Disease!!')

place = st.empty()
place.image('https://i.makeagif.com/media/1-17-2024/dw-jXM.gif',width = 80)
    
for i in range(100):
    time.sleep(0.05)
    progress_bar.progress(i + 1)

if ans == 0:
    body = f'No Heart Disease Detected'
    # st.subheader(body)
    placeholder.empty()
    place.empty()
    st.success(body)

else:
    body = 'Heart Disease Found'
    placeholder.empty()
    place.empty()
    st.warning(body)
    progress_bar = st.progress(0)

