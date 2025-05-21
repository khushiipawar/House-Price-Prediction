#Importing the required libraries:
import streamlit as st  
import pickle
import numpy as np

#loading the pickle file or the salary prediction code we have done yesterday:
model = pickle.load(open(r"D:\Home Price Prediction\house price.pkl","rb"))

#setting the title of the streamlit app:
st.title("🏠 HomeWorth Calculator App ")

#Adding a brief description about the app:
st.write("This app predicts the price of the house based on the area per square feet present.")

#Here while sepcify the input values each and every parameters like the min , max value etc should be of same datatype:
area_per_sqft= st.number_input("Enter the area per square feet:-",min_value=0.0,max_value=500000.0,value=1.0,step=0.5)

if st.button("📈 Predict House Price"): #This will create a button the frontend area.
    area_input=np.array([[area_per_sqft]])
    prediction=model.predict(area_input) #This will help to show the predicted value .
    
    st.success(f"🎯 The estimated Price for {area_input} per square feet: **${prediction[0]:,.2f}**") #helps to print out the predicted output.
    
st.write("The model was trained using the dataset of house price and its corresponding area per square feet.")

#This will help to write the markdown for the user understanding.    
st.markdown(
    """
    ---
    ### About This App
    - 🧠 **Model**: Simple Linear Regression
    - 📈 **Dataset**: 'Historical data of house' price and area per square feet.
    - 🚀 **Goal**: Help users to forecast house price effectively.
    
    **Note:** The predictions are based on the training data and may not reflect real-world outcomes.
    """
)
st.info("🔑 Your input will only be used for house price prediction and not stored.")
st.caption("Made with ❤️ by Khushi Pawar")