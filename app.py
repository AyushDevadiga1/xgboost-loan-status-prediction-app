import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data

# The main function
def main():
    
    def left_sidebar():
       
       input_dict = {}

       x = st.sidebar
       x.header('Want some cash ? ')
       check_box_value = x.checkbox('Are you ready')

       if check_box_value:
        
        x.markdown('<i> Okay ! Hold up... 💵💸💰💲 !!!</i>',unsafe_allow_html=True)

        input_dict['gender'] = x.radio(
            label = 'Gender' ,
            options = ['male','female'] 
        )

        input_dict['person_age'] = x.slider(
            label = 'Age' ,
            min_value = 0, 
            max_value = 100 , 
            value = 18
        )

        input_dict['education'] = x.selectbox(
            label = 'Qualification' ,
            options = ['High School','Bachelor','Associate','Master','Doctorate'] , 
            accept_new_options  = False 
        )

        input_dict['person_income'] = x.number_input(
            label = 'Income' ,
            min_value = 0, 
            max_value = 10000000 , 
            value = 5000
        )

        input_dict['person_emp_exp'] = x.slider(
            label = 'Experienced' ,
            min_value = 0, 
            max_value = 100, 
            value = 0 , 
            step = 1
        )

        input_dict['person_home_ownership'] = x.selectbox(
            label = 'House Ownership' ,
            options = ['OWN','RENT','MORTGAGE','OTHER'] , 
            accept_new_options  = False 
        )

        input_dict['loan_amnt'] = x.number_input(
            label = 'Income' ,
            min_value = 0, 
            max_value = 50000 , 
            value = 2500
        )

        return input_dict
        
        
    # This is just the page metadata
    st.set_page_config(
                        page_title="Loan Data Prediction",
                        page_icon="random", 
                        layout="wide", 
                        initial_sidebar_state="expanded"
                    )



    # st.title(" Loan Approval : Yes or No 😪",text_alignment='center')
    # st.write('Hello')
    
    user_data = left_sidebar()

    # Streamlit already comes with a container called st.container() and to write inside it we will use the with prefix

    with st.container():
        st.title('Loan App Predictor')
        st.write('Want some money ? Try checking this out ' \
        'This is a simple application which on given features gives prediction on whether a user deserves loan or not .' \
        ' The model used in behind scenes is XGBoostClassifier and was trained on some previous loan dataset .')

        col1,col2 = st.columns([4,1])

        with col1:
            st.write('<h4>Prediction<h4>',unsafe_allow_html=True)
            

        with col2:
            st.write('<h5>User Input : <h5>',unsafe_allow_html=True)
            st.write(user_data)

    # with st.container():
    #    st.markdown('<h3> The Dataset : </h3>',unsafe_allow_html=True)
    #    st.write('https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data')
# Basically to make sure the file is run only when called directly
if __name__ == '__main__':
    main()