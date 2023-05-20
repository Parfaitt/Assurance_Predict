# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 09:56:29 2023

@author: HP
"""


import numpy as np
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# loading the saved model
loaded_model = pickle.load(open('medical_insurance_cost_predictor.sav', 'rb'))

#creating a function for Prediction
def medical_insurance_cost_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    return prediction

def main():
    def load_data():
        data=pd.read_csv("insurance.csv")
        return data
    
    df=load_data()
    df_sample=df.sample(5)
    st.sidebar.header("Les paramtres d'entrées")
    st.sidebar.write('''
    # PRÉDICTION DES COÛTS D'ASSURANCE MÉDICALE 
    Il s'agit d'un projet d'apprentissage automatique de prédiction des coûts d'assurance   médicale.
    
    Auteur: Parfait Tanoh N'goran
    ''')
    st.title("Contruire le nuage de piont")
    num=df.select_dtypes(exclude="object").columns.to_list()
    var_x=st.selectbox("Choisis la varaible en abscice", num)
    var_y=st.selectbox("Choisis la varaible en ordonée", num)
    fig = px.scatter(
        df, x=var_x , y=var_y, title=str(var_y)+ " vs " +str(var_x)
    )
    
    if st.sidebar.checkbox("Afficher les données brutes",False):
        st.subheader("Jeux de données brutes")
        st.write(df_sample)
        
    if st.sidebar.checkbox("Afficher le nuage de point",False):
        st.subheader("Nuage de point")
        st.write(fig, var_x, var_y)
    seed=123
    #Diagrmme
    #giving a title
    st.title("Application Web de prédiction d'assurance médicale")
    
    #getting input from the user
    
    age = st.text_input('Age')
    sex = st.selectbox('Sex: 0 -> Femme, 1 -> Garçon',[0,1])
    bmi = st.text_input('(BMI) indique le rapport entre votre poids et votre taille','25.5')
    children = st.text_input("Nombre d'enfant/Personnes prise en charge")
    smoker = st.selectbox('Fumeur: 0 -> No, 1 -> Yes', [0,1])
    region = st.text_input('Region: 0 -> NorthEast, 1-> NorthWest, 2-> SouthEast, 3-> SouthWest')
    
    #code for prediction
    diagnosis = ''
    
    # getting the input data from the user
    if st.button("Prédire le Coût d'assurance :"):
        diagnosis = medical_insurance_cost_prediction([age,sex,bmi,children,smoker,region])
        st.success(diagnosis)
    else:
        st.error("Veillez remplir tous les champs vides")

if __name__ == '__main__':
    main()
