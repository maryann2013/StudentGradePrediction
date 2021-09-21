#!/usr/bin/env python
# coding: utf-8

# # Predict student grade

# In[1]:

import streamlit as st
import pandas as pd
import pickle
import numpy as np


# load the model from disk
loaded_model = pickle.load(open('streamlit_student_grade_prediction.pkl', 'rb'))



# Creating the Titles and Image
st.title("Student's Final Grade Prediction")
st.header("A model to predict  student's grade based on interesting social and demographic features such as family life, social settings, alcohol consumption etc.  The outcome variable is the final grade for the class which ranges between 0 and 20.")


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
                        
                        
                        
# Take the users input
age = st.slider("Student Age", 0, 100)
failures=st.slider("No Of Failures", 0, 3) 
absences=st.slider("No of Absences", 0, 100) 
G1 =st.slider("Term 1 Grade", 0, 20) 
G2 =st.slider("Term 2 Grade", 0, 20) 
    
SchoolMgmt=st.selectbox("School Type", df2['SchoolMgmt'].unique())
# converting text input to numeric to get back predictions from backend model.
if SchoolMgmt == 'Government':
    school_GP = 1
    school_MS=0   
else:
    school_GP=0
    school_MS = 1
    
Gender=st.selectbox("Gender", df2['Gender'].unique())
# converting text input to numeric to get back predictions from backend model.
if Gender == 'Male':
    sex_M = 1
    sex_F=0   
else:
    sex_M = 0
    sex_F=1   

    

MotherJob=st.selectbox("Mother's Job", df5['MotherJob'].unique())
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
    
FatherJob=st.selectbox("Father's Job", df5['FatherJob'].unique())
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
    
MotherEducation=st.selectbox("Mother's Education", df5['MotherEducation'].unique())
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

FatherEducation=st.selectbox("Father's Education", df5['FatherEducation'].unique())
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
    

ReasonForSchool=st.selectbox("Reason to choose this school", df4['ReasonForSchool'].unique())
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

StudentGuardian=st.selectbox("Student Guardian", df3['StudentGuardian'].unique())
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


ExtraEducationalSupport=st.selectbox("School providing Extra Educational Support", df2['ExtraEducationalSupport'].unique())
# converting text input to numeric to get back predictions from backend model.
if ExtraEducationalSupport == 'Yes':
    schoolsup_no = 0
    schoolsup_yes=1
else:
    schoolsup_no =1
    schoolsup_yes=0

ExtraPaidClassesWithinTheCourseSubject=st.selectbox("Extra paid classes within the course subject", df2['ExtraPaidClassesWithinTheCourseSubject'].unique())
# converting text input to numeric to get back predictions from backend model.
if ExtraPaidClassesWithinTheCourseSubject == 'Yes':
    paid_no = 0
    paid_yes=1
else:
    paid_no = 1
    paid_yes=0
    
    
Extra_curricularActivities=st.selectbox("Extra curricular activities", df2['Extra_curricularActivities'].unique())
# converting text input to numeric to get back predictions from backend model.
if Extra_curricularActivities == 'Yes':
    activities_no = 0
    activities_yes=1
else:
    activities_no = 1
    activities_yes=0
    

FamilyEducationalSupport=st.selectbox("Family providing Educational Support", df2['FamilyEducationalSupport'].unique())
# converting text input to numeric to get back predictions from backend model.
if FamilyEducationalSupport == 'Yes':
    famsup_no = 0
    famsup_yes=1
else:
    famsup_no = 1
    famsup_yes=0
    

TravelTime=st.selectbox("Time to travel to School", df4['TravelTime'].unique())
# converting text input to numeric to get back predictions from backend model.
if TravelTime == '<15 min':
    traveltime = 1
elif TravelTime == '15 to 30 min':
    traveltime = 2
elif TravelTime == '30 min to 1 hour':
    traveltime = 3
else:
    traveltime = 4


                            
AttendedNurserySchool=st.selectbox("Attended Nursery School", df2['AttendedNurserySchool'].unique())
# converting text input to numeric to get back predictions from backend model.
if AttendedNurserySchool == 'Yes':
    nursery_no = 0
    nursery_yes=1
else:
    nursery_no = 1
    nursery_yes=0                          

WillingToTakeHigherEducation=st.selectbox("Willing to take higher education", df2['WillingToTakeHigherEducation'].unique())
# converting text input to numeric to get back predictions from backend model.
if WillingToTakeHigherEducation == 'Yes':
    higher_no = 0
    higher_yes=1
else:
    higher_no = 1
    higher_yes=0 
    
InternetAccessAtHome=st.selectbox("Internet Access At Home", df2['InternetAccessAtHome'].unique())
# converting text input to numeric to get back predictions from backend model.
if InternetAccessAtHome == 'Yes':
    internet_no = 0
    internet_yes=1
else:
    internet_no = 1
    internet_yes=0 
    
RomanticRelationship=st.selectbox("Romantic Relationship", df2['RomanticRelationship'].unique())
# converting text input to numeric to get back predictions from backend model.
if RomanticRelationship == 'Yes':
    romantic_yes =1
    romantic_no=0
else:
    romantic_yes =0
    romantic_no=1

StudyTime=st.selectbox("Hours of Study", df4['StudyTime'].unique())
# converting text input to numeric to get back predictions from backend model.
if StudyTime == '<2 hours':
    studytime = 1
elif TravelTime == '2 to 5 hours':
    studytime = 2
elif TravelTime == '30 min to 1 hour':
    studytime = 3
else:
    studytime = 4
    

QualityOfFamilyRelationships=st.selectbox("Quality of family relationship", df5['QualityOfFamilyRelationships'].unique())
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
    

GoingOutWithFriends=st.selectbox("Going Out With Friends", df5['GoingOutWithFriends'].unique())
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
    
WorkdayAlcoholConsumption=st.selectbox("Workday Alcohol Consumption", df5['WorkdayAlcoholConsumption'].unique())
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

WeekendAlcoholConsumption=st.selectbox("Weekend Alcohol Consumption", df5['WeekendAlcoholConsumption'].unique())
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


CurrentHealthStatus=st.selectbox("Current Health Status", df5['CurrentHealthStatus'].unique())
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
    
    
    
FreetimeAfterSchool=st.selectbox("Freetime after Schoolhours", df5['FreetimeAfterSchool'].unique())
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

AddressType=st.selectbox("Address Type", df2['AddressType'].unique())
# converting text input to numeric to get back predictions from backend model.
if AddressType == 'Urban':
    address_R = 0
    address_U=1
else:
    address_R = 1
    address_U=0
       
    


FamilySize=st.selectbox("Family Size", df2['FamilySize'].unique())
# converting text input to numeric to get back predictions from backend model.
if FamilySize == 'Less or equal to 3':
    famsize_GT3 = 0
    famsize_LE3=1
else:
    famsize_GT3 = 1
    famsize_LE3=0

ParentCohabitationStatus=st.selectbox("Parent Cohabitation Status", df2['ParentCohabitationStatus'].unique())
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


# convert user inputs into an array fr the model

#final_features = [int(x) for x in features]
#final_features = [np.array(final_features)]
final_features = pd.DataFrame(features)




if st.button('Predict'):  
    st.success('features are :',final_features)
    # when the submit button is pressed
    #prediction =  loaded_model.predict(final_features)
    #st.balloons()
    #st.success('Final Grade would be : ', prediction[0])
    
    

