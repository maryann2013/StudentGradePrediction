#!/usr/bin/env python
# coding: utf-8

# # Predict student grade

# In[1]:

import streamlit as st
import pandas as pd
import pickle
import numpy as np

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
else:
    # load the model from disk
    loaded_model = pickle.load(open('streamlit_student_grade_prediction.pkl', 'rb'))
    uploaded_file = st.file_uploader("Choose your csv file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)
    else:
        st.warning("you need to upload a csv or excel file.")

    # store the inputs
    features = [age, Medu, Fedu, traveltime, studytime, failures, famrel,
           freetime, goout, Dalc, Walc, health, absences, G1, G2,
            school_GP, school_MS, sex_F, sex_M, address_R,
           address_U, famsize_GT3, famsize_LE3, Pstatus_A, Pstatus_T,
           Mjob_at_home, Mjob_health, Mjob_other, Mjob_services,
           Mjob_teacher,Fjob_at_home, Fjob_health, Fjob_other,
           Fjob_services, Fjob_teacher, reason_course, reason_home,
           reason_other, reason_reputation, guardian_father,
           guardian_mother, guardian_other, schoolsup_no, schoolsup_yes,
           famsup_no, famsup_yes, paid_no, paid_yes, activities_no,
           activities_yes, nursery_no, nursery_yes, higher_no,
           higher_yes, internet_no, internet_yes, romantic_no,
           romantic_yes]

    feature_names = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
           'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2',
           'school_GP', 'school_MS', 'sex_F', 'sex_M', 'address_R', 'address_U',
           'famsize_GT3', 'famsize_LE3', 'Pstatus_A', 'Pstatus_T', 'Mjob_at_home',
           'Mjob_health', 'Mjob_other', 'Mjob_services', 'Mjob_teacher',
           'Fjob_at_home', 'Fjob_health', 'Fjob_other', 'Fjob_services',
           'Fjob_teacher', 'reason_course', 'reason_home', 'reason_other',
           'reason_reputation', 'guardian_father', 'guardian_mother',
           'guardian_other', 'schoolsup_no', 'schoolsup_yes', 'famsup_no',
           'famsup_yes', 'paid_no', 'paid_yes', 'activities_no', 'activities_yes',
           'nursery_no', 'nursery_yes', 'higher_no', 'higher_yes', 'internet_no',
           'internet_yes', 'romantic_no', 'romantic_yes']

    key_features=['G2','G1','age','absences','Dalc','freetime','health','goout','traveltime','famrel','Medu',
     'Walc','Fedu','failures','reason_other','schoolsup_no','studytime','Fjob_services','Mjob_other','Fjob_at_home',
     'sex_F','famsup_no','school_GP','guardian_father','romantic_no','Mjob_services','Mjob_teacher','famsize_GT3',
     'higher_no','nursery_no','activities_no','reason_reputation','reason_home','internet_no','reason_course','address_R',
     'schoolsup_yes','paid_no','guardian_mother','school_MS']

    dict_mean = {}
    dict_mean["G2_mean"]= 11.57
    dict_mean["G1_mean"]= 11.4
    dict_mean["age_mean"]= 16.74
    dict_mean["absences_mean"]= 3.66
    dict_mean["Dalc_mean"]= 1.5
    dict_mean["freetime_mean"]= 3.18
    dict_mean["health_mean"]= 3.54
    dict_mean["goout_mean"]= 3.18
    dict_mean["traveltime_mean"]= 1.57
    dict_mean["famrel_mean"]= 3.93
    dict_mean["Medu_mean"]= 2.51
    dict_mean["Walc_mean"]= 2.28
    dict_mean["Fedu_mean"]= 2.31
    dict_mean["failures_mean"]= 0.22
    dict_mean["reason_other_mean"]= 0.11
    dict_mean["schoolsup_no_mean"]= 0.9
    dict_mean["studytime_mean"]= 1.93
    dict_mean["Fjob_services_mean"]= 0.28
    dict_mean["Mjob_other_mean"]= 0.4
    dict_mean["Fjob_at_home_mean"]= 0.06
    dict_mean["sex_F_mean"]= 0.59
    dict_mean["famsup_no_mean"]= 0.39
    dict_mean["school_GP_mean"]= 0.65
    dict_mean["guardian_father_mean"]= 0.24
    dict_mean["romantic_no_mean"]= 0.63
    dict_mean["Mjob_services_mean"]= 0.21
    dict_mean["Mjob_teacher_mean"]= 0.11
    dict_mean["famsize_GT3_mean"]= 0.7
    dict_mean["higher_no_mean"]= 0.11
    dict_mean["nursery_no_mean"]= 0.2
    dict_mean["activities_no_mean"]= 0.51
    dict_mean["reason_reputation_mean"]= 0.22
    dict_mean["reason_home_mean"]= 0.23
    dict_mean["internet_no_mean"]= 0.23
    dict_mean["reason_course_mean"]= 0.44
    dict_mean["address_R_mean"]= 0.3
    dict_mean["schoolsup_yes_mean"]= 0.1
    dict_mean["paid_no_mean"]= 0.94
    dict_mean["guardian_mother_mean"]= 0.7
    dict_mean["school_MS_mean"]= 0.35


    dict_25 = {}
    dict_25["G2_25"]=10.0
    dict_25["G1_25"]= 10.0
    dict_25["age_25"]  =16.0
    dict_25["absences_25"] = 0.0
    dict_25["Dalc_25"]  =1.0
    dict_25["freetime_25"]  =3.0
    dict_25["health_25"]  =2.0
    dict_25["goout_25"]  =2.0
    dict_25["traveltime_25"]  =1.0
    dict_25["famrel_25"]  =4.0
    dict_25["Medu_25"]  =2.0
    dict_25["Walc_25"]  =1.0
    dict_25["Fedu_25"]  =1.0
    dict_25["failures_25"] =0.0
    dict_25["reason_other_25"]  =0.0
    dict_25["schoolsup_no_25"]  =1.0
    dict_25["studytime_25"]  =1.0
    dict_25["Fjob_services_25"]  =0.0
    dict_25["Mjob_other_25"]  =0.0
    dict_25["Fjob_at_home_25"]  =0.0
    dict_25["sex_F_25"]  =0.0
    dict_25["famsup_no_25"]  =0.0
    dict_25["school_GP_25"]  =0.0
    dict_25["guardian_father_25"]  =0.0
    dict_25["romantic_no_25"]  =0.0
    dict_25["Mjob_services_25"]  =0.0
    dict_25["Mjob_teacher_25"]  =0.0
    dict_25["famsize_GT3_25"]  =0.0
    dict_25["higher_no_25"]  =0.0
    dict_25["nursery_no_25"]  =0.0
    dict_25["activities_no_25"]  =0.0
    dict_25["reason_reputation_25"]  =0.0
    dict_25["reason_home_25"]  =0.0
    dict_25["internet_no_25"]  =0.0
    dict_25["reason_course_25"]  =0.0
    dict_25["address_R_25"]  =0.0
    dict_25["schoolsup_yes_25"]  =0.0
    dict_25["paid_no_25"]  =1.0
    dict_25["guardian_mother_25"]  =0.0
    dict_25["school_MS_25"]  =0.0

    dict_75 = {}
    dict_75["G2_75"] =13.0
    dict_75["G1_75"] =13.0
    dict_75["age_75"] =18.0
    dict_75["absences_75"] =6.0
    dict_75["Dalc_75"] =2.0
    dict_75["freetime_75"] =4.0
    dict_75["health_75"] =5.0
    dict_75["goout_75"] =4.0
    dict_75["traveltime_75"] =2.0
    dict_75["famrel_75"] =5.0
    dict_75["Medu_75"] =4.0
    dict_75["Walc_75"] =3.0
    dict_75["Fedu_75"] =3.0
    dict_75["failures_75"] =0.0
    dict_75["reason_other_75"] =0.0
    dict_75["schoolsup_no_75"] =1.0
    dict_75["studytime_75"] =2.0
    dict_75["Fjob_services_75"] =1.0
    dict_75["Mjob_other_75"] =1.0
    dict_75["Fjob_at_home_75"] =0.0
    dict_75["sex_F_75"] =1.0
    dict_75["famsup_no_75"] =1.0
    dict_75["school_GP_75"] =1.0
    dict_75["guardian_father_75"] =0.0
    dict_75["romantic_no_75"] =1.0
    dict_75["Mjob_services_75"] =0.0
    dict_75["Mjob_teacher_75"] =0.0
    dict_75["famsize_GT3_75"] =1.0
    dict_75["higher_no_75"] =0.0
    dict_75["nursery_no_75"] =0.0
    dict_75["activities_no_75"] =1.0
    dict_75["reason_reputation_75"] =0.0
    dict_75["reason_home_75"] =0.0
    dict_75["internet_no_75"] =0.0
    dict_75["reason_course_75"] =1.0
    dict_75["address_R_75"] =1.0
    dict_75["schoolsup_yes_75"] =0.0
    dict_75["paid_no_75"] =1.0
    dict_75["guardian_mother_75"] =1.0
    dict_75["school_MS_75"] =1.0

    
    

