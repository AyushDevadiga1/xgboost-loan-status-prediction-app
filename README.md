# 💸 Loan Eligibility Prediction App

A powerful Streamlit Application leveraging an **XGBoost** machine learning model to predict loan approval status in real-time. This application allows users to input financial and personal details to receive an immediate assessment of their loan eligibility, complete with probability scores and visual insights.

🔗 **[Live Demo](https://xgboost-loan-status-prediction-app-dwipwvam92goqcxlmpjpwc.streamlit.app/)**

---

## 🚀 Key Features

*   **Interactive Input Form**: Easy-to-use sidebar for entering applicant details (Age, Income, Credit Score, etc.).
*   **Real-time Prediction**: Instant "Approved" or "Rejected" verdict driven by a pre-trained XGBoost pipeline.
*   **Probability Analysis**: Visual breakdown of success vs. failure probabilities using progress bars.
*   **Model Insights**: Access to the model's performance metrics, including a Classification Report and Confusion Matrix.
*   **Responsive Design**: Clean and professional UI built with custom CSS styling.

## 🛠️ Built With

*   **[Streamlit](https://streamlit.io/)** - The frontend framework for data apps.
*   **[XGBoost](https://xgboost.readthedocs.io/)** - High-performance gradient boosting library for the prediction model.
*   **[Scikit-Learn](https://scikit-learn.org/)** - For data preprocessing and pipeline management.
*   **[Pandas](https://pandas.pydata.org/)** & **[NumPy](https://numpy.org/)** - Data manipulation and numerical operations.
*   **[Plotly](https://plotly.com/)** & **[Seaborn](https://seaborn.pydata.org/)** - Data visualization.

## 📂 Project Structure

```
.
├── app.py                  # Main application entry point
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── assets/                 # CSS styles and images
├── core/                   # Backend logic and utility functions
│   ├── engine.py           # Model loading logic
│   └── utils.py          
├── models/                 # Serialized model pipelines (.joblib)
└── dataset/                # Dataset used for training (if applicable)
```

## 🏁 Getting Started

To run this application locally on your machine, follow these steps:

### Prerequisites

Ensure you have Python 3.8 or higher installed.

### Installation

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

### ▶️ Running the App

Execute the following command in your terminal:

```bash
streamlit run app.py
```

The app will open automatically in your default browser at `http://localhost:8501`.

## 🧠 Model Information

The application uses an **XGBoost Classifier** trained on the [Loan Approval Classification Dataset](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data). The model evaluates features such as:
*   Person Age & Income
*   Employment Experience
*   Loan Amount & Interest Rate
*   Credit History Length & Credit Score
*   Previous Loan Defaults

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/AyushDevadiga1">Ayush Devadiga</a>
</p>