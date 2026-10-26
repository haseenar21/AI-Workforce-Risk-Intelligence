
import streamlit as st
import pandas as pd
import joblib

# 1. Page settings
st.set_page_config(
    page_title="AI Impact & Layoff Risk Predictor",
    page_icon="🤖",
    layout="wide"
)

# 2. Load the trained model and supporting files
model = joblib.load("lr_layoff.pkl")
scaler = joblib.load("layoff_scaler.pkl")
columns = joblib.load("layoff_columns.pkl")
encoder = joblib.load("layoff_label_encoder.pkl")
df = pd.read_csv("ai-impact-jobs-layoff-risk-dataset.csv")

TARGET = "Layoff_Risk"
feature_df = df.drop(columns=[TARGET])

with st.sidebar:
    st.markdown("## 🤖 AI Impact")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["🏠 Dashboard", "🔮 Risk Predictor"]
    )

# 3. App heading
st.title("🤖 AI Impact & Layoff Risk Predictor")
st.write("Understand how AI adoption and workplace automation may relate to workforce layoff risk.")

st.divider()

if page == "🏠 Dashboard":

    st.markdown("### 📊 Dataset Overview")

    a, b, c = st.columns(3)
    a.metric("Total Records", f"{len(df):,}")
    b.metric("Input Features", len(feature_df.columns))
    c.metric("Risk Categories", df[TARGET].nunique())

    st.markdown("### 🧭 Explore the Dataset")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Layoff Risk Distribution")
        counts = df[TARGET].value_counts()
        st.bar_chart(counts)

    with right:
        st.markdown("#### Industry Distribution")
        if "Industry" in df.columns:
            st.bar_chart(df["Industry"].value_counts().head(10))


elif page == "🔮 Risk Predictor":

    st.markdown("### 👤 Employee & Job Profile")
    st.caption("Enter the details to predict the layoff risk.")

    sections = {
        "Personal Profile": [
            "Age", "Education_Level", "Years_of_Experience"
        ],
        "Job Information": [
            "Industry", "Job_Role", "Company_Size", "Job_Level"
        ],
        "Work & Automation": [
            "Routine_Task_Percentage",
            "Creativity_Requirement",
            "Human_Interaction_Level",
            "Tasks_Automated_Percentage"
        ],
        "AI Adoption & Training": [
            "AI_Adoption_Level",
            "Number_of_AI_Tools_Used",
            "AI_Usage_Hours_Per_Week",
            "AI_Training_Hours"
        ]
    }

    with st.form("prediction_form"):

        input_data = {}

        for section_name, section_columns in sections.items():

            st.markdown(f"#### 📌 {section_name}")

            col1, col2 = st.columns(2)

            for i, column in enumerate(section_columns):

                series = feature_df[column].dropna()
                current_col = col1 if i % 2 == 0 else col2

                with current_col:

                    if pd.api.types.is_numeric_dtype(series):

                        input_data[column] = st.number_input(
                            column.replace("_", " "),
                            min_value=float(series.min()),
                            max_value=float(series.max()),
                            value=float(series.median()),
                            key=column
                        )

                    else:
                        options = sorted(series.astype(str).unique())

                        input_data[column] = st.selectbox(
                            column.replace("_", " "),
                            options,
                            key=column
                        )

            st.divider()

        submitted = st.form_submit_button(
            "🔮 Predict Layoff Risk",
            use_container_width=True
        )

    if submitted:

        data = pd.DataFrame([input_data])

        data = pd.get_dummies(
            data, drop_first=True, dtype=int
        )

        data = data.reindex(columns=columns, fill_value=0)

        data_scaled = scaler.transform(data)

        prediction = model.predict(data_scaled)

        result = encoder.inverse_transform(
            prediction.astype(int)
        )[0]

        st.divider()
        st.subheader("📊 Prediction Result")
        st.success(f"Predicted Layoff Risk: {result}")

        routine = input_data["Routine_Task_Percentage"]
        automated = input_data["Tasks_Automated_Percentage"]
        training = input_data["AI_Training_Hours"]
        experience = input_data["Years_of_Experience"]
        tools = input_data["Number_of_AI_Tools_Used"]
        hours = input_data["AI_Usage_Hours_Per_Week"]
        creativity = input_data["Creativity_Requirement"]
        interaction = input_data["Human_Interaction_Level"]

        automation = (routine + automated) / 2
        training_intensity = training / experience if experience > 0 else 0
        engagement = tools * hours
        adaptability = (creativity + interaction + training) / 3

        st.subheader("✨ Additional AI Insights")

        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)

        c1.metric("Automation Exposure", f"{automation:.1f}")
        c2.metric("Training Intensity", f"{training_intensity:.2f}")
        c3.metric("AI Engagement", f"{engagement:.1f}")
        c4.metric("Adaptability Indicator", f"{adaptability:.2f}")
