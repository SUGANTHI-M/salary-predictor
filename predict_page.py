import streamlit as st 
import pickle
import numpy as np 

def load_model():
    with open('salary_linear.pkl', 'rb') as file:
        data = pickle.load(file)
    return data 

data=load_model()

linear_reg = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("Software Developer Salary Predictor")

    st.write("""### We need some information to predict the salary""")

    countries=('United States', 'United Kingdom', 'Spain', 'Netherlands',
       'Germany', 'Canada', 'Italy', 'Brazil', 'France', 'India',
       'Sweden', 'Poland', 'Australia', 'Russian Federation')

    education=('Bachelor’s degree', 'Master’s degree', 'Less than a Bachelors',
       'Post grad')

    country = st.selectbox("Country",countries)

    education =st.selectbox("Education",education)

    experience = st.slider("Years of experience",0,50,3)

    ok=st.button("Calculate salary")

    if ok:
        X=np.array([[country,education,experience]])
        X[:,0] = le_country.transform(X[:,0])
        X[:,1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = linear_reg.predict(X)
        st.subheader(f"Salary is ${salary[0]:.2f}")