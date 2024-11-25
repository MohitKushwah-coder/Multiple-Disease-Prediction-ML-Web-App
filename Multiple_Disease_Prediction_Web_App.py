import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Loading the saved models 

diabtetes_model = pickle.load(open('D:/ML Project/Multiple Disease Prediction/trained_model_diabetes.sav', 'rb'))

heart_model = pickle.load(open('D:/ML Project/Multiple Disease Prediction/trained_model_heart.sav', 'rb'))

brain_model = pickle.load(open('D:/ML Project/Multiple Disease Prediction/trained_model_brain_stroke.sav', 'rb'))


# Sidebar for Navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Brain Stroke Prediction'],
                           icons = ['capsule', 'heart-pulse', 'clipboard2-plus'], 
                           default_index = 0)
    


# Diabetes Prediction main Page
if (selected == 'Diabetes Prediction'):

    # Title
    st.title('Diabetes Prediction using ML')

    # column for input... field 3 columns in line 
    col1, col2, col3 = st.columns(3)
    

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('BloodPressure Level')
    with col1:
        SkinThickness = st.text_input('SkinThickness Value')
    with col2:
        Insulin = st.text_input('Insulin Levels')
    with col3:
        BMI = st.text_input('BMI Value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction Value')
    with col2:
        Age = st.text_input('Age')


    # diabetes prediction
    diabetic_pred  = ''

    # creating button for prediction
    if st.button('Diabetes Test Result'):
        diabetic_pred = diabtetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diabetic_pred[0] == 1):
            diabetic_pred = 'This Person is Diabetic.'
        else:
            diabetic_pred = 'This person is not Diabetic.'

    st.success(diabetic_pred)





# Heart Disease Prediction main page
if (selected == 'Heart Disease Prediction'):

    # Title
    st.title('Heart Disease Prediction using ML')


    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex (1 = Male, 0 = Female)')
    with col3:
        cp = st.text_input('CP Level')
    with col4:
        trestbps = st.text_input('trestbps Level')
    with col1:
        chol = st.text_input('chol Value')
    with col2:
        fbs = st.text_input('fbs Levels')
    with col3:
        restecg = st.text_input('restecg Value')
    with col4:
        thalach = st.text_input('thalach Value')
    with col1:
        exang = st.text_input('exang')
    with col2:
        oldpeak = st.text_input('oldpeak')
    with col3:
        slope = st.text_input('slope')
    with col4:
        ca = st.text_input('ca')
    with col1:
        thal = st.text_input('thal')

    heart_pred = ''
    if st.button('Diabetes Test Result'):

        heart_pred = heart_model.predict([[float(age),float(sex),float(cp),float(trestbps),float(chol),float(fbs),float(restecg),float(thalach),float(exang),float(oldpeak),float(slope),float(ca),float(thal)]])
        

        # alternate for loop
        
        # user_input = [float(x) for x in user_input]

        # diab_prediction = diabetes_model.predict([user_input])

        # if diab_prediction[0] == 1:
        #     diab_diagnosis = 'The person is diabetic'
        # else:
        #     diab_diagnosis = 'The person is not diabetic'


        if (heart_pred[0] == 1):
            heart_pred = 'This Person has Heart Disease.'
        else:
            heart_pred = 'This person has no heart Disease.'
    st.success(heart_pred)




# Brain Stroke Prediction mai page
if (selected == 'Brain Stroke Prediction'):

    # Title
    st.title('Brain Stroke Prediction using ML using ML')


    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.text_input('Gender (1 = Male, 0 = Female)')
    with col2:
        age = st.text_input('Age')
    with col3:
        hypertension = st.text_input('Hypertension (1 = True, 0 = False)' )
    with col1:
        heart_disease = st.text_input('Heart Disease (1 = True, 0 = False)')
    with col2:
        ever_married = st.text_input('Ever Married (1 = Yes, 0 = No)')
    with col3:
        work_type = st.text_input('Work Type (0 = Unknown, 1 = Formerly, 2 = Never, 3 = Smokes)')
    with col1:
        Residence_type = st.text_input('Residence Type (1 = Urban, 0 = Rural)')
    with col2:
        avg_glucose_level = st.text_input('Avg Glucose Level')
    with col3:
        bmi = st.text_input('bmi')
    with col1:
        smoking_status = st.text_input('Smoking Status (0 = Gov-Job, 1 = Never-Worked, 2 = Private, 3 = Self-Emp, 4 = Children)')


    brain_pred = ''

    if st.button('Brain Stroke Test Result'):
        brain_pred = brain_model.predict([[float(gender), float(age), float(hypertension), float(heart_disease) ,float(ever_married), float(work_type), float(Residence_type), float(avg_glucose_level), float(bmi),float(smoking_status)]])

        if (brain_pred[0] == 1):
            brain_pred = 'This Person has no Brain Stroke.'
        else:
            brain_pred = 'This person has no Brain Stroke.'

    st.success(brain_pred)



# streamlit run "D:/ML Project/Multiple Disease Prediction/Multiple Disease Prediction Web App.py"
