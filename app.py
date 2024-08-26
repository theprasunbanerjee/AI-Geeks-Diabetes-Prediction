import numpy as np
import pickle
import streamlit as st

# Load the model and scaler (ensure both files are uploaded)
loaded_model = pickle.load(open('trained_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

def diabetes_prediction(input_data):
    # Convert input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    
    # Reshape the input data for the model
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Standardize the input data
    std_data = scaler.transform(input_data_reshaped)
    
    # Make a prediction using the loaded model
    prediction = loaded_model.predict(std_data)
    
    if prediction[0] == 0:
        return 'The person is not diabetic.'
    else:
        return 'The person is diabetic.'

def main():
    # Set the title of the web page
    st.title('Diabetes Prediction Web App')

    # Getting input data from the user
    try:
        Glucose = st.text_input('Glucose Level')
        BloodPressure = st.text_input('Blood Pressure Value')
        SkinThickness = st.text_input('Skin Thickness Value')
        Insulin = st.text_input('Insulin Level')
        BMI = st.text_input('BMI Value')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
        Age = st.text_input('Age of the Person')

        # Code for prediction
        diagnosis = ''

        # Prediction button
        if st.button('Diabetes Test Result'):
            diagnosis = diabetes_prediction([
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age)
            ])
            st.success(diagnosis)
    except ValueError:
        st.error("Please enter valid numeric values.")

if __name__ == '__main__':
    main()
