import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from core.engine import load_pipeline
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

    # To modify the inbuilt streamlit style we inject a markdown at the base and use it to control other elements
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Path to your assets folder
    base_path = os.path.dirname(__file__)
    css_path = os.path.join(base_path,"assets", "styles.css")
    local_css(css_path)

    def preprocess_data(X):
        # Load the dataset into a single row
        df = pd.DataFrame([X])
        # Load the Pipeline/Model from the load_pipeline() which is imported above 
        model_pipe = load_pipeline()

        prediction = model_pipe.predict(df)
        prediction_proba = model_pipe.predict_proba(df)
        return prediction,prediction_proba
        

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

        input_dict['person_gender'] = x.radio(
            label = 'Gender' ,
            options = ['male','female'] 
        )

        input_dict['person_age'] = x.slider(
            label = 'Age' ,
            min_value = 0, 
            max_value = 100 , 
            value = 18
        )

        input_dict['person_education'] = x.selectbox(
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
            label = 'Loan Amount' ,
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

        input_dict['loan_percent_income'] = x.slider(
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
            value = 1 , 
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

       return input_dict,check_box_value
            
    
    # st.markdown('<div class="loan-banner-strip">Youkoso !!!</div>', unsafe_allow_html=True)
    
    user_data,toggled_button = left_sidebar()

    # Streamlit already comes with a container called st.container() and to write inside it we will use the with prefix

    with st.container():

        st.title('Loan Approval Prediction from User Data')

        if toggled_button:

            st.info("""
                    💡 **Important Information:**
                    * The results shown initially are based on default placeholder values.
                    * Please update the details in the sidebar to see how your specific profile affects the prediction.
                    """)


            col1,col2 = st.columns([4,1])

            with col1:

                    st.write('<h3> Prediction : ',unsafe_allow_html=True)

                    prediction,prediction_proba = preprocess_data(user_data)


                    if toggled_button:
                        prediction, prediction_proba = preprocess_data(user_data)
                        
                        # Probabilities
                        prob_val = prediction_proba[0][1] if prediction == 1 else prediction_proba[0][0]

                        if prediction == 1:
                            st.markdown(f'''
                                <div class="prediction-card status-accepted">
                                    <span style="font-size: 0.8em; text-transform: uppercase; opacity: 0.7;">System Verdict</span><br>
                                    <b style="font-size: 1.5em;">LOAN APPROVED</b><br>
                                </div>
                            ''', unsafe_allow_html=True)
                        else:
                            st.markdown(f'''
                                <div class="prediction-card status-rejected">
                                    <span style="font-size: 0.8em; text-transform: uppercase; opacity: 0.7;">System Verdict</span><br>
                                    <b style="font-size: 1.5em;">LOAN REJECTED</b><br>
 
                                </div>
                            ''', unsafe_allow_html=True)



                    prob_success = float(prediction_proba[0][1]) 
                    prob_failure = float(prediction_proba[0][0])

                    # Professional Display using Columns and Progress Bars
                    col3, col4 = st.columns(2)

                    with col3:
                        st.write("### Success Probability")
                        st.title(f"{prob_success:.1%}")
                        st.progress(prob_success)

                    with col4:
                        st.write("### Failure Probability")
                        st.title(f"{prob_failure:.1%}")
                        # Using a red progress bar logic via color or standard
                        st.progress(prob_failure)

                    with st.container():

                        # Formatting the raw report into a structured dictionary
                        report_data = {
                            "Class": ["0 (Rejected)", "1 (Approved)", "Accuracy", "Macro Avg", "Weighted Avg"],
                            "Precision": [0.95, 0.85, None, 0.90, 0.93],
                            "Recall": [0.96, 0.84, None, 0.90, 0.93],
                            "F1-Score": [0.96, 0.84, 0.93, 0.90, 0.93],
                            "Support": [6990, 2010, 9000, 9000, 9000]
                        }

                        df_report = pd.DataFrame(report_data)

                        # Displaying in Streamlit
                        st.subheader("Model Classification Report")
                        st.table(df_report)

                        st.subheader("Model Confusion Matrix")
                        img_path = os.path.join(os.path.dirname(__file__), "assets", "conf_1.png")
                        if os.path.exists(img_path):
                            st.image(img_path, use_container_width=True,width=500,caption="Visualizing Model Accuracy")
                        else:
                            # Fallback if image isn't found
                            st.warning("Confusion Matrix image not found in assets folder.")

            with col2:
                if(user_data):
                    st.write('<h5>User Input :',unsafe_allow_html=True)
                    st.json(user_data)

        else:

            st.markdown("""
    ### Can you get the loan you're looking for?
    
    This application is designed to give you an immediate prediction on loan eligibility. Instead of a manual review process, we use a machine learning model to evaluate financial profiles and determine whether a loan is likely to be approved or rejected.

    #### The Data and Intelligence
    The logic behind this app is based on the **Loan Approval Classification dataset**. We trained an **XGBoost Classifier** on thousands of historical records to identify the specific patterns that lead to successful applications. 

    The model considers several key factors:
    *   **Financial Background:** Your annual income and employment experience.
    *   **Credit History:** Your past credit score and any history of default.
    *   **Loan Specifics:** The amount you are requesting and the intended purpose of the funds.

    #### Getting Started
    To see the predictor in action, use the sidebar to toggle the input form. You can adjust the values to see how different scenarios—like increasing your income or improving your credit score—affect the final decision. 

    The results you see initially are based on placeholder values. Update them to reflect your specific data for a real-time prediction.
    """)

            st.divider()

    # with st.container():
    #    st.markdown('<h3> The Dataset : </h3>',unsafe_allow_html=True)
    #    st.write('https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data')
# Basically to make sure the file is run only when called directly
if __name__ == '__main__':
    main()