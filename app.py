import streamlit as st
import numpy as np
import pickle

with open("random_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Titanic Survival Predictor")
st.write("Enter passenger details below to predict survival:")

pclass = st.selectbox("Pclass", options=[1, 2, 3])

gender = st.selectbox("Sex", options=[0, 1], 
                      format_func=lambda x: "Female" if x==0 else "Male")

age = st.number_input("Age", min_value=0.0)

sibsp = st.number_input("SibSp", min_value=0.0)

fare = st.number_input("Fare", min_value=0.0)

embarked = st.selectbox(
    "Embarked",
    options=[0, 1, 2],
    format_func=lambda x: "S" if x == 0 else "C" if x == 1 else "Q"
)


if st.button("Predict Survived Risk"):
    input_data = np.array([[pclass, gender, age, sibsp, fare, embarked]])
    prediction = model.predict(input_data)[0]
    
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"Passenger Survived ✅")
    else:
        st.error(f"Passenger Did Not Survive ❌")