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
    def load_data2():
        df2 = pd.DataFrame({'SchoolMgmt': ['Government','Private'],
                           'Gender': ['Male', 'Female'],
                           'AddressType':['Urban','Rural'],
                           'FamilySize':['Less or equal to 3','Greater than 3'],
                           'ParentCohabitationStatus': ['Individual', 'Joint'],
                           'ExtraEducationalSupport':['Yes','No'],
                           'FamilyEducationalSupport':['Yes','No'],
                            'ExtraPaidClassesWithinTheCourseSubject' :['Yes','No'],   
                           'Extra_curricularActivities' :['Yes','No'], 
                           'AttendedNurserySchool' :['Yes','No'],
                           'WillingToTakeHigherEducation' :['Yes','No'],
                           'InternetAccessAtHome' :['Yes','No'],
                           'RomanticRelationship' :['Yes','No']
                          })
        return df2

    def load_data5():
        df5 = pd.DataFrame({'MotherEducation': ['None','Primary Education','5th to 9th Grade','Secondary Education','Higher Education'],
                           'FatherEducation': ['None','Primary Education','5th to 9th Grade','Secondary Education','Higher Education'],
                           'QualityOfFamilyRelationships' :['Very Bad','Bad','Normal','Good','Excellent'],
                           'FreetimeAfterSchool' :['Very low','low','Normal','High','Very High'],
                           'GoingOutWithFriends' :['Very low','low','Normal','High','Very High'],
                           'WorkdayAlcoholConsumption' :['Very low','low','Normal','High','Very High'],
                           'WeekendAlcoholConsumption' :['Very low','low','Normal','High','Very High'],
                           'CurrentHealthStatus' :['Very Bad','Bad','Normal','Good','Very Good'],
                            'MotherJob': ['Teacher','HealthCare Related','Civil Services','At Home','Other'],
                           'FatherJob': ['Teacher','HealthCare Related','Civil Services','At Home','Other']
                          }) 
        return df5    


    def load_data4():                       
        df4 = pd.DataFrame({
                           'ReasonForSchool':['Close to home' ,'School Reputation', 'Course Preference','Other'],
                           'TravelTime':[' <15 min', '15 to 30 min','30 min to 1 hour','>1 hour'],
                           'StudyTime':['<2 hours','2 to 5 hours', '5 to 10 hours', '>10 hours']
                           })
        return df4


    def load_data3():               
        df3 = pd.DataFrame({'StudentGuardian':['Mother','Father','Other']
                           })
        return df3


    df2 = load_data2()
    df3 = load_data3()  
    df4 = load_data4()
    df5 = load_data5()                       


    st.sidebar.subheader("User Input Parameters")
    # Take the users input
    age = st.sidebar.slider("Student Age", 0, 100)
    failures=st.sidebar.slider("No Of Failures", 0, 3) 
    absences=st.sidebar.slider("No of Absences", 0, 100) 
    G1 =st.sidebar.slider("Term 1 Grade", 0, 20) 
    G2 =st.sidebar.slider("Term 2 Grade", 0, 20) 

    SchoolMgmt=st.sidebar.selectbox("School Type", df2['SchoolMgmt'].unique())
    # converting text input to numeric to get back predictions from backend model.
    if SchoolMgmt == 'Government':
        school_GP = 1
        school_MS=0   
    else:
        school_GP=0
        school_MS = 1

    Gender=st.sidebar.selectbox("Gender", df2['Gender'].unique())
    # converting text input to numeric to get back predictions from backend model.
    if Gender == 'Male':
        sex_M = 1
        sex_F=0   
    else:
        sex_M = 0
        sex_F=1   



    MotherJob=st.sidebar.selectbox("Mother's Job", df5['MotherJob'].unique())
    # converting text input to numeric to get back predictions from backend model.
    if MotherJob == 'Teacher':
        Mjob_at_home = 0
        Mjob_health=0 
        Mjob_other=0
        Mjob_services=0
        Mjob_teacher=1
    elif MotherJob == 'HealthCare Related':
        Mjob_at_home = 0
        Mjob_health=1 
        Mjob_other=0
        Mjob_services=0
        Mjob_teacher=0
    elif MotherJob == 'Civil Services':
        Mjob_at_home = 0
        Mjob_health=0 
        Mjob_other=0
        Mjob_services=1
        Mjob_teacher=0   
    elif MotherJob == 'At Home':
        Mjob_at_home = 1
        Mjob_health=0 
        Mjob_other=0
        Mjob_services=0
        Mjob_teacher=0
    else:
        Mjob_at_home = 0
        Mjob_health=0 
        Mjob_other=1
        Mjob_services=0
        Mjob_teacher=0

    FatherJob=st.sidebar.selectbox("Father's Job", df5['FatherJob'].unique())
    # converting text input to numeric to get back predictions from backend model.
    if FatherJob == 'Teacher':
        Fjob_at_home = 0
        Fjob_health=0 
        Fjob_other=0
        Fjob_services=0
        Fjob_teacher=1
    elif FatherJob == 'HealthCare Related':
        Fjob_at_home = 0
        Fjob_health=1 
        Fjob_other=0
        Fjob_services=0
        Fjob_teacher=0
    elif FatherJob == 'Civil Services':
        Fjob_at_home = 0
        Fjob_health=0 
        Fjob_other=0
        Fjob_services=1
        Fjob_teacher=0   
    elif FatherJob == 'At Home':
        Fjob_at_home = 1
        Fjob_health=0 
        Fjob_other=0
        Fjob_services=0
        Fjob_teacher=0
    else:
        Fjob_at_home = 0
        Fjob_health=0 
        Fjob_other=1
        Fjob_services=0
        Fjob_teacher=0

    MotherEducation=st.sidebar.selectbox("Mother's Education", df5['MotherEducation'].unique())
    # converting text input to numeric to get back predictions from backend model.
    if MotherEducation == 'None':
        Medu = 0
    elif MotherEducation == 'Primary Education':
        Medu = 1
    elif MotherEducation == '5th to 9th Grade':
        Medu = 2
    elif MotherEducation == 'Secondary Education':
        Medu = 3
    else:
        Medu = 4

    FatherEducation=st.sidebar.selectbox("Father's Education", df5['FatherEducation'].unique())
    # converting text input to numeric to get back predictions from backend model.
    if FatherEducation == 'None':
        Fedu = 0
    elif FatherEducation == 'Primary Education':
        Fedu = 1
    elif FatherEducation == '5th to 9th Grade':
        Fedu = 2
    elif FatherEducation == 'Secondary Education':
        Fedu = 3
    else:
        Fedu = 4


    ReasonForSchool=st.sidebar.selectbox("Reason to choose this school", df4['ReasonForSchool'].unique())
    # converting text input to numeric to get back predictions from backend model.
    if ReasonForSchool == 'Close to home':
        reason_home = 1
        reason_other=0
        reason_course=0
        reason_reputation=0
    elif ReasonForSchool == 'School Reputation':
        reason_home = 0
        reason_other=0
        reason_course=0
        reason_reputation=1
    elif ReasonForSchool == 'Course Preference':
        reason_home = 0
        reason_other=0
        reason_course=1
        reason_reputation=0
    else:
        reason_home = 0
        reason_other=1
        reason_course=0
        reason_reputation=0

    StudentGuardian=st.sidebar.selectbox("Student Guardian", df3['StudentGuardian'].unique())
    # converting text input to numeric to get back predictions from backend model.
    if StudentGuardian == 'Mother':
        guardian_mother = 1
        guardian_father=0
        guardian_other=0
    elif StudentGuardian == 'Father':
        guardian_mother = 0
        guardian_father=1
        guardian_other=0
    else:
        guardian_mother = 0
        guardian_father=0
        guardian_other=1


    ExtraEducationalSupport=st.sidebar.selectbox("School providing Extra Educational Support", df2['ExtraEducationalSupport'].unique())
    # converting text input to numeric to get back predictions from backend model.
    if ExtraEducationalSupport == 'Yes':
        schoolsup_no = 0
        schoolsup_yes=1
    else:
        schoolsup_no =1
        schoolsup_yes=0

    ExtraPaidClassesWithinTheCourseSubject=st.sidebar.selectbox("Extra paid classes within the course subject", df2['ExtraPaidClassesWithinTheCourseSubject'].unique())
    # converting text input to numeric to get back predictions from backend model.
    if ExtraPaidClassesWithinTheCourseSubject == 'Yes':
        paid_no = 0
        paid_yes=1
    else:
        paid_no = 1
        paid_yes=0


    Extra_curricularActivities=st.sidebar.selectbox("Extra curricular activities", df2['Extra_curricularActivities'].unique())
    # converting text input to numeric to get back predictions from backend model.
    if Extra_curricularActivities == 'Yes':
        activities_no = 0
        activities_yes=1
    else:
        activities_no = 1
        activities_yes=0


    FamilyEducationalSupport=st.sidebar.selectbox("Family providing Educational Support", df2['FamilyEducationalSupport'].unique())
    # converting text input to numeric to get back predictions from backend model.
    if FamilyEducationalSupport == 'Yes':
        famsup_no = 0
        famsup_yes=1
    else:
        famsup_no = 1
        famsup_yes=0


    TravelTime=st.sidebar.selectbox("Time to travel to School", df4['TravelTime'].unique())
    # converting text input to numeric to get back predictions from backend model.
    if TravelTime == '<15 min':
        traveltime = 1
    elif TravelTime == '15 to 30 min':
        traveltime = 2
    elif TravelTime == '30 min to 1 hour':
        traveltime = 3
    else:
        traveltime = 4



    AttendedNurserySchool=st.sidebar.selectbox("Attended Nursery School", df2['AttendedNurserySchool'].unique())
    # converting text input to numeric to get back predictions from backend model.
    if AttendedNurserySchool == 'Yes':
        nursery_no = 0
        nursery_yes=1
    else:
        nursery_no = 1
        nursery_yes=0                          

    WillingToTakeHigherEducation=st.sidebar.selectbox("Willing to take higher education", df2['WillingToTakeHigherEducation'].unique())
    # converting text input to numeric to get back predictions from backend model.
    if WillingToTakeHigherEducation == 'Yes':
        higher_no = 0
        higher_yes=1
    else:
        higher_no = 1
        higher_yes=0 

    InternetAccessAtHome=st.sidebar.selectbox("Internet Access At Home", df2['InternetAccessAtHome'].unique())
    # converting text input to numeric to get back predictions from backend model.
    if InternetAccessAtHome == 'Yes':
        internet_no = 0
        internet_yes=1
    else:
        internet_no = 1
        internet_yes=0 

    RomanticRelationship=st.sidebar.selectbox("Romantic Relationship", df2['RomanticRelationship'].unique())
    # converting text input to numeric to get back predictions from backend model.
    if RomanticRelationship == 'Yes':
        romantic_yes =1
        romantic_no=0
    else:
        romantic_yes =0
        romantic_no=1

    StudyTime=st.sidebar.selectbox("Hours of Study", df4['StudyTime'].unique())
    # converting text input to numeric to get back predictions from backend model.
    if StudyTime == '<2 hours':
        studytime = 1
    elif TravelTime == '2 to 5 hours':
        studytime = 2
    elif TravelTime == '30 min to 1 hour':
        studytime = 3
    else:
        studytime = 4


    QualityOfFamilyRelationships=st.sidebar.selectbox("Quality of family relationship", df5['QualityOfFamilyRelationships'].unique())
    if QualityOfFamilyRelationships == 'Very Bad':
        famrel = 1  
    elif QualityOfFamilyRelationships == 'Bad':
        famrel = 2
    elif QualityOfFamilyRelationships == 'Normal':
        famrel = 3
    elif QualityOfFamilyRelationships == 'Good':
        famrel = 4
    else:
        famrel = 5 


    GoingOutWithFriends=st.sidebar.selectbox("Going Out With Friends", df5['GoingOutWithFriends'].unique())
    if GoingOutWithFriends == 'Very low':
        goout = 1  
    elif GoingOutWithFriends == 'low':
        goout = 2
    elif GoingOutWithFriends == 'Normal':
        goout = 3
    elif GoingOutWithFriends == 'High':
        goout = 4
    else:
        goout = 5

    WorkdayAlcoholConsumption=st.sidebar.selectbox("Workday Alcohol Consumption", df5['WorkdayAlcoholConsumption'].unique())
    if WorkdayAlcoholConsumption == 'Very low':
        Dalc = 1  
    elif WorkdayAlcoholConsumption == 'low':
        Dalc = 2
    elif WorkdayAlcoholConsumption == 'Normal':
        Dalc = 3
    elif WorkdayAlcoholConsumption == 'High':
        Dalc = 4
    else:
        Dalc = 5

    WeekendAlcoholConsumption=st.sidebar.selectbox("Weekend Alcohol Consumption", df5['WeekendAlcoholConsumption'].unique())
    if WeekendAlcoholConsumption == 'Very low':
        Walc = 1  
    elif WeekendAlcoholConsumption == 'low':
        Walc = 2
    elif WeekendAlcoholConsumption == 'Normal':
        Walc = 3
    elif WeekendAlcoholConsumption == 'High':
        Walc = 4
    else:
        Walc = 5


    CurrentHealthStatus=st.sidebar.selectbox("Current Health Status", df5['CurrentHealthStatus'].unique())
    if CurrentHealthStatus == 'Very Bad':
        health = 1  
    elif CurrentHealthStatus == 'Bad':
        health = 2
    elif CurrentHealthStatus == 'Normal':
        health = 3
    elif CurrentHealthStatus == 'Good':
        health = 4
    else:
        health = 5 



    FreetimeAfterSchool=st.sidebar.selectbox("Freetime after Schoolhours", df5['FreetimeAfterSchool'].unique())
    if FreetimeAfterSchool == 'Very low':
        freetime = 1  
    elif FreetimeAfterSchool == 'low':
        freetime = 2
    elif FreetimeAfterSchool == 'Normal':
        freetime = 3
    elif FreetimeAfterSchool == 'High':
        freetime = 4
    else:
        freetime = 5

    AddressType=st.sidebar.selectbox("Address Type", df2['AddressType'].unique())
    # converting text input to numeric to get back predictions from backend model.
    if AddressType == 'Urban':
        address_R = 0
        address_U=1
    else:
        address_R = 1
        address_U=0




    FamilySize=st.sidebar.selectbox("Family Size", df2['FamilySize'].unique())
    # converting text input to numeric to get back predictions from backend model.
    if FamilySize == 'Less or equal to 3':
        famsize_GT3 = 0
        famsize_LE3=1
    else:
        famsize_GT3 = 1
        famsize_LE3=0

    ParentCohabitationStatus=st.sidebar.selectbox("Parent Cohabitation Status", df2['ParentCohabitationStatus'].unique())
    # converting text input to numeric to get back predictions from backend model.
    if ParentCohabitationStatus == 'Individual':
        Pstatus_T = 0
        Pstatus_A=1
    else:
        Pstatus_T = 1
        Pstatus_A=0


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

    # convert user inputs into an array fr the model

    #final_features = [int(x) for x in features]
    #final_features = [np.array(final_features)]
    final_features = np.array(features).reshape(1, -1)
    final_features_df=pd.DataFrame(final_features,columns=feature_names)



    if st.sidebar.button('Predict'):  
        prediction =  loaded_model.predict(final_features)
        st.write('Final Grade would be : ', prediction[0])

        if prediction[0] <10:
            st.write('Please see the risk Factors below : ')
            for feat in key_features:
                f25=feat+"_25"
                f75=feat+"_75"
                fmean=feat+"_mean"
                if int(final_features_df[feat]) < int(dict_25[f25]):
                    st.write('\t','Student ',feat,':',int(final_features_df[feat]), 'is below the Class Avg:', 
                          np.round(dict_mean[fmean],2))
                if int(final_features_df[feat]) > int(dict_75[f75]):
                    st.write('\t','Student ',feat,':', int(final_features_df[feat]), 'is above the Class Avg:', 
                          np.round(dict_mean[fmean],2))

        # when the submit button is pressed
        #prediction =  loaded_model.predict(final_features)
        #st.balloons()
        #st.success('Final Grade would be : ', prediction[0])

    

