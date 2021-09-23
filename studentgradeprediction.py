#!/usr/bin/env python
# coding: utf-8

# # Predict student grade

# In[1]:

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import time
import random
import sys
import datetime
from sklearn.model_selection import train_test_split

import xgboost as xgb
from xgboost import XGBRegressor





# Creating the Titles and Image
st.title("Welcome to Student Final Grade Prediction Web App")
#st.header("A model to predict  student's grade based on interesting social and demographic features such as family life, social settings, #alcohol consumption etc.  The outcome variable is the final grade for the class which ranges between 0 and 20.")
st.subheader("The Page is divided into 2 categories: \n 1.PowerBI report \n 2.Grade Prediction ")
options = st.selectbox("Please Select", ['Select','PowerBI','Prediction'])

if (options =='PowerBI'):
    st.markdown("""
    <iframe width="600" height="373.5" src="https://app.powerbi.com/view?r=eyJrIjoiNDU4NjMxODEtNGQ4OC00ODkxLThlOWYtODFiYjRmNDA0Yjk2IiwidCI6Ijk3YTkyYjA0LTRjODctNDM0MS05YjA4LWQ4MDUxZWY4ZGNlMiIsImMiOjh9&pageName=ReportSection" frameborder="0" allowFullScreen="true"></iframe>
    """, unsafe_allow_html=True)
    
elif(options =='Prediction'):
    key_features=['G2','G1','age','absences','Dalc','freetime','health','goout','traveltime','famrel','Medu',
     'Walc','Fedu','failures','reason_other','schoolsup_no','studytime','Fjob_services','Mjob_other','Fjob_at_home',
     'sex_F','famsup_no','school_GP','guardian_father','romantic_no','Mjob_services','Mjob_teacher','famsize_GT3',
     'higher_no','nursery_no','activities_no','reason_reputation','reason_home','internet_no','reason_course','address_R',
     'schoolsup_yes','paid_no','guardian_mother','school_MS']
      # load the model from disk
    loaded_model = pickle.load(open('streamlit_student_grade_prediction.pkl', 'rb'))
    uploaded_file = st.sidebar.file_uploader("Choose your csv file")
    if uploaded_file is not None:
        student_por_df = pd.read_csv(uploaded_file)
        
        non_numeric_features = [feat for feat in list(student_por_df) if feat not in list(student_por_df._get_numeric_data())]
        
        for feat in non_numeric_features:
            dummies = pd.get_dummies(student_por_df[feat]).rename(columns=lambda x: feat + '_' + str(x))
            student_por_df = pd.concat([student_por_df, dummies], axis=1)
        student_por_df = student_por_df[[feat for feat in list(student_por_df) if feat not in non_numeric_features]]
        st.write("Final Dataset")
        st.write(student_por_df)
        outcome = 'G3'
        features = [feat for feat in list(student_por_df) if feat not in outcome]
        
        test_dataset=student_por_df[features]
        st.write("Final Features")
        st.write(test_dataset)
        if st.sidebar.button('Predict'):  
            prediction =  loaded_model.predict(test_dataset)
            
            preds_final=pd.DataFrame(prediction,columns=['G3'])
            st.write("Final Predictions")
            st.write(preds_final)
            predicted_students_in_trouble = preds_final[preds_final['G3'] < 10]
            st.write("predicted_students_in_trouble")
            st.write(predicted_students_in_trouble)
            # See which feature they landed well below or well above peers
            for index, row in predicted_students_in_trouble.iterrows():
                st.write('Student ID:', index)
                for feat in key_features:
                    row_df=student_por_df[features][index:index+1]
                    if int(row_df[feat]) < student_por_df[feat].quantile(0.25):
                        st.write('\t', 'Below:', feat, int(row_df[feat]), 'Class:', 
                              np.round(np.mean(student_por_df[feat]),2))
                    if int(row_df[feat]) > student_por_df[feat].quantile(0.75):
                        st.write('\t','Above:', feat, int(row_df[feat]), 'Class:', 
                              np.round(np.mean(student_por_df[feat]),2))


    else:
        st.warning("you need to upload a csv or excel file.")
else:
    pass

    