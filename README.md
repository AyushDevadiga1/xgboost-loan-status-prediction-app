# Loan Eligibility Prediction App

This project is a Streamlit-based web application that leverages an **XGBoost** machine learning model to predict loan approval status in real-time. The application provides an interface for users to input financial and personal details, generating an immediate assessment of loan eligibility along with probability scores and visual insights.

**[Live Application Demo](https://xgboost-loan-status-prediction-app-dwipwvam92goqcxlmpjpwc.streamlit.app/)**

---

## Key Features

*   **Interactive Input Form**: User-friendly sidebar interface for entering applicant details such as age, income, and credit score.
*   **Real-time Prediction**: Immediate loan approval or rejection verdict driven by a pre-trained XGBoost prediction pipeline.
*   **Probability Analysis**: Visual representation of success versus failure probabilities using progress bars.
*   **Model Insights**: Detailed display of model performance metrics, including a Classification Report and Confusion Matrix.
*   **Responsive Design**: Professional user interface built with custom CSS styling for optimal user experience.

## Technology Stack

*   **[Streamlit](https://streamlit.io/)**: Frontend framework for data application development.
*   **[XGBoost](https://xgboost.readthedocs.io/)**: Gradient boosting library used for the core prediction model.
*   **[Scikit-Learn](https://scikit-learn.org/)**: Library for data preprocessing and pipeline management.
*   **[Pandas](https://pandas.pydata.org/)** & **[NumPy](https://numpy.org/)**: Libraries for data manipulation and numerical operations.
*   **[Plotly](https://plotly.com/)** & **[Seaborn](https://seaborn.pydata.org/)**: Libraries utilized for data visualization.

## Project Structure

```
.
├── app.py                  # Main application entry point
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── assets/                 # CSS styles and static assets
├── core/                   # Backend logic and utility functions
│   ├── engine.py           # Model loading logic
│   └── utils.py          
├── models/                 # Serialized model pipelines (.joblib)
└── dataset/                # Dataset used for training
```

## Getting Started

To run this application locally, please follow the steps below:

### Prerequisites

Ensure that Python 3.8 or higher is installed on your system.

### Installation Requirements

1.  **Clone the repository**
    ```bash
    git clone https://github.com/AyushDevadiga1/xgboost-loan-status-prediction-app.git
    cd xgboost-loan-status-prediction-app
    ```

2.  **Create a virtual environment (Recommended)**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

Execute the following command in your terminal to start the Streamlit server:

```bash
streamlit run app.py
```

The application will launch automatically in your default web browser at `http://localhost:8501`.

## Model Methodology

The application utilizes an **XGBoost Classifier** trained on the [Loan Approval Classification Dataset](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data). The model evaluates candidate eligibility based on features such as:
*   Applicant Age & Annual Income
*   Employment Experience
*   Loan Amount & Interest Rate
*   Credit History Length & Credit Score
*   History of Previous Loan Defaults

---

<p align="center">
  Developed by <a href="https://github.com/AyushDevadiga1">Ayush Devadiga</a>
</p>