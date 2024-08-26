import numpy as np
import pickle
import streamlit as st


#model loading.
loaded_model = pickle.load(open('trained_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

#fuction creation.
def diabetes_prediction(input_data):
    # Changing input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data) #converting input data into numpy array for easy processing.

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) #reshaping the input data.

    std_data = scaler.transform(input_data_reshaped) #standardizing the input data.
    print(std_data)

    prediction = loaded_model.predict(std_data) #making prediction on the input data.
    print(prediction)

    if prediction[0] == 0:
        return 'The person is not diabetic.'
    else:
        return 'The person is diabetic.'
  
    
def main():
    
    #giving webpage title.
    st.title('Diabetes Prediction')
    
    
    #getting input.
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThikness = st.text_input('Skin Thikness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFuction = st.text_input('Diabetes Pedigree Fuction Value')
    Age = st.text_input('Age of the Person')
    
    
    #code for Prediction.
    diagnosis = ''
    
    
    #button creation.
    if st.button('Test'):
        diagnosis = diabetes_prediction([Glucose, BloodPressure, SkinThikness, Insulin, BMI, DiabetesPedigreeFuction, Age])
        
        
    st.success(diagnosis)
    
    
    
    
if __name__ == '__main__':
    main()
