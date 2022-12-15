import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.externals import joblib
# st.image('./ai.jpg',caption='Machine Learning')
st.title('Songs Gern Prediction Model')
st.write("It will predict the gern of songs from people's age \n #### Data Table ")

age = np.array([34,45,22,65,18,30,17,26,30,32,16])
songs = np.array(['Clasic','Rock','Pop','Rock','Pop','Clasic','Pop','Pop','Clasic','Clasic','Pop'])
data_frame = pd.DataFrame({'Age':age,'Song-Gern':songs})

st.dataframe(data_frame)

target = data_frame.drop(columns=['Age'])
data = data_frame.drop(columns=['Song-Gern'])

st.write('### Target Table and Data Table')
col1,col2 = st.columns(2)
col1.dataframe(data)
col2.dataframe(target)

X_train,X_text,y_train,y_test = train_test_split(data,target,test_size=.2)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)

predictions = model.predict(X_text)

score = accuracy_score(y_test,predictions)
st.write(X_text)
st.write('### Predictions')
st.dataframe({ 'Song-Gern': predictions})
# st.dataframe({'Age':X_text})
st.write('### Accuracy Score ')
st.write(score)