
import streamlit as st
import joblib
from pathlib import Path

model_path = Path(__file__).resolve().parent.parent / "iris_model.pkl"
model = joblib.load(model_path)

st.title("Iris Flower Prediction")

sl = st.number_input("Sepal Length")
sw = st.number_input("Sepal Width")
pl = st.number_input("Petal Length")
pw = st.number_input("Petal Width")

if st.button("Predict"):
    pred = model.predict([[sl, sw, pl, pw]])
    flowers = ["Setosa", "Versicolor", "Virginica"]
    st.success(flowers[pred[0]])
