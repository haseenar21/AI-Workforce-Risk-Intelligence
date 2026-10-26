# AI Workforce Risk Intelligence

An ML-based application that analyzes employee profiles, workplace automation, and AI adoption patterns to predict potential layoff risk.

## Overview

Artificial Intelligence is transforming workplaces by automating repetitive tasks and changing job requirements. This project uses machine learning to classify potential workforce layoff risk based on employee characteristics, job roles, automation exposure, and AI adoption.

The application provides an interactive Streamlit dashboard where users can explore the dataset, enter employee and job information, and receive a predicted layoff risk category.

## Features

- **Interactive Dashboard:** Explore dataset statistics and risk distributions.
- **Employee Profile Analysis:** Enter personal and professional information.
- **Automation Analysis:** Examine routine tasks and the percentage of tasks automated.
- **AI Adoption Analysis:** Evaluate AI tool usage and training information.
- **Layoff Risk Prediction:** Generate predictions using a trained machine learning model.
- **Additional AI Insights:** Display automation exposure, training intensity, AI engagement, and an adaptability indicator.

## Machine Learning Workflow

1. Load and inspect the dataset.
2. Separate input features and the target variable.
3. Encode categorical features and scale input data.
4. Train and evaluate a Logistic Regression model.
5. Save the trained model and preprocessing artifacts.
6. Load the saved artifacts in the Streamlit application.
7. Generate predictions from user-provided inputs.

## Technology Stack

- **Language:** Python
- **Data Processing:** Pandas
- **Machine Learning:** Scikit-learn
- **Model Persistence:** Joblib
- **User Interface:** Streamlit
- **Visualization:** Streamlit charts
- **Development:** Jupyter Notebook

## Repository Structure

```text
AI-Workforce-Risk-Intelligence/
├── app.py
├── train.ipynb
├── ai-impact-jobs-layoff-risk-dataset.csv
├── lr_layoff.pkl
├── layoff_scaler.pkl
├── layoff_columns.pkl
├── layoff_label_encoder.pkl
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/haseenar21/AI-Workforce-Risk-Intelligence.git
cd AI-Workforce-Risk-Intelligence
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## Dataset

The project uses `ai-impact-jobs-layoff-risk-dataset.csv`.

The dataset contains employee, job, workplace automation, and AI adoption attributes.

Example input features include:

- Age
- Education level
- Years of experience
- Industry and job role
- Company size and job level
- Routine task percentage
- Creativity requirement
- Human interaction level
- AI adoption level
- Number of AI tools used
- AI usage hours per week
- Tasks automated percentage
- AI training hours

**Target variable:** `Layoff_Risk`

The model predicts the risk category based on patterns learned from the training dataset.

## Model Artifacts

The repository includes the following saved files:

| File | Purpose |
|---|---|
| `lr_layoff.pkl` | Trained Logistic Regression model |
| `layoff_scaler.pkl` | Feature scaling |
| `layoff_columns.pkl` | Encoded feature-column alignment |
| `layoff_label_encoder.pkl` | Encoding and decoding risk labels |

These artifacts allow the application to load the trained model without retraining it every time it starts.

## Additional AI Insights

The application calculates supplementary indicators from the user's inputs:

- **Automation Exposure:** Average of routine task percentage and automated task percentage.
- **Training Intensity:** AI training hours relative to years of experience.
- **AI Engagement:** Number of AI tools multiplied by AI usage hours per week.
- **Adaptability Indicator:** Average of creativity requirement, human interaction level, and AI training hours.

These indicators are project-defined calculations, not independently validated measures of employee adaptability or employment outcomes.

## Limitations

- Predictions depend on the quality and representativeness of the training dataset.
- The model identifies statistical patterns and cannot reliably determine an individual's actual future employment status.
- Predicted risk categories should not be interpreted as guarantees.
- The additional indicators are illustrative and are not validated employment-risk metrics.
- The project is intended for educational and analytical purposes, not for making real-world hiring, firing, or employment decisions.

## Future Improvements

- Compare multiple machine learning algorithms.
- Add model evaluation metrics and confusion matrices.
- Improve prediction explanations using feature importance or explainability methods.
- Add model monitoring and dataset quality checks.
- Deploy the application for public access.

## Author

**Haseena Rahaman**

B.Tech Computer Science and Engineering student

GitHub: [@haseenar21](https://github.com/haseenar21)

## Disclaimer

This project is developed for educational purposes. Predictions represent model-generated estimates based on dataset patterns and should not be treated as definitive assessments of an individual's employment future.

