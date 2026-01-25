import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data

# The main function
def main():

    # Keep this always at the top
    # This is just the page metadata
    st.set_page_config(
                        page_title="Loan Data Prediction",
                        page_icon="random", 
                        layout="wide", 
                        initial_sidebar_state="auto"
                    )


    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Path to your assets folder
    base_path = os.path.dirname(__file__)
    css_path = os.path.join(base_path,"assets", "styles.css")
    local_css(css_path)

    

    def left_sidebar():
       
       input_dict = {}

       x = st.sidebar
       x.header('Enter User Details')
       check_box_value = x.toggle('Toggle to Fill Form ')
    #    check_box_value_1 = x.toggle('For Bulk DATA')

    #    if check_box_value_1:
    #        x.markdown('Got Some files ???')

       if check_box_value:
        
        x.markdown('<i> Please Fill the Details Below </i>',unsafe_allow_html=True)

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
            label = 'Employment Experience Years' ,
            min_value = 0, 
            max_value = 100, 
            value = 0 , 
            step = 1
        )

        input_dict['person_home_ownership'] = x.segmented_control(
            label = 'House Ownership' ,
            options = ['OWN','RENT','MORTGAGE','OTHER'] ,
            default = 'OWN'
        )

        input_dict['loan_amnt'] = x.number_input(
            label = 'Income' ,
            min_value = 0, 
            max_value = 50000 , 
            value = 2500
        )

        input_dict['loan_intent'] = x.selectbox(
            label = 'Loan Intention' ,
            options = ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"] , 
            accept_new_options  = False 
        )

        input_dict['loan_int_rate'] = x.number_input(
            label = 'Loan Interest Rate' ,
            min_value = 5.0, 
            max_value = 25.0 , 
            format = '%.02f',
            value = 5.0
        )

        input_dict['loan_person_income'] = x.slider(
            label = 'Loan Percent Income',
            min_value = 0.0, 
            max_value = 1.0 , 
            step = 0.01,
            value = 0.0
        )
        
        input_dict['cb_person_cred_hist_length'] = x.slider(
            label = 'Credit Bureau Person Credit History Length' ,
            min_value = 1, 
            max_value = 50, 
            value = 0 , 
            step = 1
        )

        input_dict['credit_score'] = x.number_input(
            label = 'Credit Score',
            min_value = 250, 
            max_value = 1000,
            value = 300
        )

        input_dict['previous_loan_defaults_on_file'] = x.radio(
            label = 'Have you defaulted on a loan before?' ,
            options = ['No','Yes'] 
        )

        return input_dict
            
    
    # st.markdown('<div class="loan-banner-strip">Youkoso !!!</div>', unsafe_allow_html=True)
    
    user_data = left_sidebar()

    # Streamlit already comes with a container called st.container() and to write inside it we will use the with prefix

    with st.container():
        st.title('Loan App Predictor')
        st.write('Want some money ? Try checking this out ' \
        'This is a simple application which on given features gives prediction on whether a user deserves loan or not .' \
        ' The model used in behind scenes is XGBoostClassifier and was trained on some previous loan dataset .')

        col1,col2 = st.columns([4,1])

        with col1:
            st.write('<h4>Input Data',unsafe_allow_html=True)

            if(user_data):
                df = pd.DataFrame.from_dict(user_data, orient='index', columns=['Value'])
                st.dataframe(df)
            

        with col2:
            st.write('<h5>User Input :',unsafe_allow_html=True)

            if(user_data):
                st.write(user_data)

    # with st.container():
    #    st.markdown('<h3> The Dataset : </h3>',unsafe_allow_html=True)
    #    st.write('https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data')
# Basically to make sure the file is run only when called directly
if __name__ == '__main__':
    main()