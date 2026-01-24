import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# The main function
def main():

        # Add a title
    st.set_page_config(
                        page_title="Loan Data Prediction",
                        page_icon="random", 
                        layout="wide", 
                        initial_sidebar_state="expanded"
                    )

    st.title(" Loan Approval : Yes or No 😪",text_alignment='center')

# Basically to make sure the file is run only when called directly
if __name__ == '__main__':
    main()