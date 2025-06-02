from datetime import datetime
import json
import os
import pickle

from omegaconf import OmegaConf
import pandas as pd
import plotly.express as px
import streamlit as st

from src.globals import logger

# Configure Streamlit page
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .prediction-result {
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .survived {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .not-survived {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""",
    unsafe_allow_html=True,
)


class TitanicStreamlitApp:
    def __init__(self):
        self.config = self.load_config()
        self.model = None
        self.load_model()

    def load_config(self):
        """Load configuration from config.yaml"""
        try:
            return OmegaConf.load("config.yaml")
        except Exception as e:
            st.error(f"Failed to load config: {e}")
            return None

    def load_model(self):
        """Load the trained model"""
        try:
            model_path = os.path.join(
                self.config["paths"]["models_parent_dir"],
                f"{self.config['names']['model_name']}.pkl",
            )
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                logger.info(f"Model loaded from {model_path}")
            else:
                st.warning(f"Model file not found at {model_path}")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

    def load_data(self):
        """Load datasets for analysis"""
        data = {}
        try:
            # Load raw data
            train_path = os.path.join(self.config["paths"]["data"]["raw_data"], "train.csv")
            test_path = os.path.join(self.config["paths"]["data"]["raw_data"], "test.csv")

            if os.path.exists(train_path):
                data["train"] = pd.read_csv(train_path)
            if os.path.exists(test_path):
                data["test"] = pd.read_csv(test_path)

        except Exception as e:
            st.error(f"Failed to load data: {e}")

        return data

    def predict_single(self, passenger_data):
        """Make prediction for a single passenger"""
        if self.model is None:
            return None, "Model not loaded"

        try:
            # Convert to DataFrame
            df = pd.DataFrame([passenger_data])

            # Make prediction
            prediction = self.model.predict(df)

            return {
                "prediction": int(prediction),
            }, None

        except Exception as e:
            return None, str(e)


def main():
    app = TitanicStreamlitApp()

    # Header
    st.markdown(
        '<h1 class="main-header">🚢 Titanic Survival Prediction</h1>', unsafe_allow_html=True
    )
    st.markdown("---")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        [
            "🏠 Home",
            "🔮 Single Prediction",
            "📊 Batch Prediction",
            "📈 Data Analysis",
            "🔧 Model Info",
        ],
    )

    if page == "🏠 Home":
        show_home_page(app)
    elif page == "🔮 Single Prediction":
        show_single_prediction_page(app)
    elif page == "📊 Batch Prediction":
        show_batch_prediction_page(app)
    elif page == "📈 Data Analysis":
        show_data_analysis_page(app)
    elif page == "🔧 Model Info":
        show_model_info_page(app)


def show_home_page(app):
    """Display home page with project overview"""
    st.header("Welcome to the Titanic Survival Prediction System")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <div class="metric-card">
        <h3>🎯 Project Goal</h3>
        <p>Predict passenger survival on the Titanic using machine learning algorithms based on passenger characteristics.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="metric-card">
        <h3>🤖 Model</h3>
        <p>Random Forest classifier trained on historical Titanic passenger data with feature engineering.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="metric-card">
        <h3>📊 Features</h3>
        <p>Single predictions, batch processing, data analysis, and model performance metrics.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Quick stats
    if app.model:
        st.success("✅ Model is loaded and ready for predictions!")
    else:
        st.error("❌ Model is not loaded. Please check the model file.")

    # Navigation help
    st.markdown("""
    ## 🧭 Navigation Guide
    
    - **🔮 Single Prediction**: Make predictions for individual passengers
    - **📊 Batch Prediction**: Upload CSV files for bulk predictions
    - **📈 Data Analysis**: Explore the Titanic dataset with interactive visualizations
    - **🔧 Model Info**: View model performance and technical details
    """)


def show_single_prediction_page(app):
    """Display single prediction page"""
    st.header("🔮 Single Passenger Prediction")
    st.markdown("Enter passenger details to predict survival")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Passenger Information")

        passenger_id = st.number_input("Passenger ID (optional)", min_value=0, value=0, step=1)
        name = st.text_input("Full Name", value="John Doe")
        pclass = st.selectbox(
            "Passenger Class", [1, 2, 3], help="1 = First, 2 = Second, 3 = Third"
        )
        sex = st.selectbox("Gender", ["male", "female"])
        age = st.number_input("Age", min_value=0.0, max_value=150.0, value=30.0, step=0.1)

    with col2:
        st.subheader("Family & Travel Details")

        sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, value=0, step=1)
        parch = st.number_input("Parents/Children Aboard", min_value=0, value=0, step=1)
        ticket = st.text_input("Ticket Number", value="A/5 21171")
        fare = st.number_input("Fare", min_value=0.0, value=7.25, step=0.01)
        cabin = st.text_input("Cabin (optional)", value="")
        embarked = st.selectbox(
            "Port of Embarkation",
            ["S", "C", "Q"],
            help="S = Southampton, C = Cherbourg, Q = Queenstown",
        )

    if st.button("🔮 Predict Survival", type="primary"):
        # Prepare passenger data
        passenger_data = {
            "PassengerId": passenger_id if passenger_id > 0 else None,
            "Pclass": pclass,
            "Name": name,
            "Sex": sex,
            "Age": age if age > 0 else None,
            "SibSp": sibsp,
            "Parch": parch,
            "Ticket": ticket,
            "Fare": fare if fare > 0 else None,
            "Cabin": cabin if cabin else None,
            "Embarked": embarked,
        }

        # Make prediction
        result, error = app.predict_single(passenger_data)

        if error:
            st.error(f"Prediction failed: {error}")
        else:
            prediction = result["prediction"]

            # Display result
            col1, col2, col3 = st.columns(3)

            with col2:
                if prediction == 1:
                    st.markdown(
                        """
                    <div class="prediction-result survived">
                    🎉 SURVIVED<br>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                    <div class="prediction-result not-survived">
                    💔 DID NOT SURVIVE<br>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )


def show_batch_prediction_page(app):
    """Display batch prediction page"""
    st.header("📊 Batch Prediction")
    st.markdown("Upload a CSV file with passenger data for bulk predictions")

    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with passenger data. Required columns: Pclass, Name, Sex, SibSp, Parch, Ticket",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("📋 Uploaded Data Preview")
            st.dataframe(df.head(10))

            if st.button("🚀 Run Batch Prediction", type="primary"):
                if app.model is None:
                    st.error("Model not loaded. Cannot make predictions.")
                    return

                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                predictions = []
                total_rows = len(df)

                for idx, row in df.iterrows():
                    # Update progress
                    progress = (idx + 1) / total_rows
                    progress_bar.progress(progress)

                    # Prepare data
                    passenger_data = row.to_dict()

                    # Make prediction
                    result, error = app.predict_single(passenger_data)

                    if result:
                        predictions.append(
                            {
                                "PassengerId": row.get("PassengerId", idx),
                                "Prediction": result["prediction"],
                            }
                        )
                    else:
                        predictions.append(
                            {
                                "PassengerId": row.get("PassengerId", idx),
                                "Prediction": -1,  # Error indicator
                            }
                        )

                # Complete progress
                progress_bar.progress(1.0)
                status_text.text("Batch prediction completed!")

                # Display results
                results_df = pd.DataFrame(predictions)

                col1, col2, col3 = st.columns(3)
                with col1:
                    survived_count = (results_df["Prediction"] == 1).sum()
                    st.metric("Predicted Survivors", survived_count)

                with col2:
                    not_survived_count = (results_df["Prediction"] == 0).sum()
                    st.metric("Predicted Non-Survivors", not_survived_count)

                with col3:
                    survival_rate = survived_count / len(results_df) * 100
                    st.metric("Survival Rate", f"{survival_rate:.1f}%")

                # Results table
                st.subheader("📊 Prediction Results")
                st.dataframe(results_df)

                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name=f"titanic_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")

    # Sample data download
    st.markdown("---")
    st.subheader("📝 Sample Data Format")

    sample_data = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3],
            "Pclass": [3, 1, 3],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley",
                "Heikkinen, Miss. Laina",
            ],
            "Sex": ["male", "female", "female"],
            "Age": [22, 38, 26],
            "SibSp": [1, 1, 0],
            "Parch": [0, 0, 0],
            "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282"],
            "Fare": [7.25, 71.2833, 7.925],
            "Cabin": ["", "C85", ""],
            "Embarked": ["S", "C", "S"],
        }
    )

    st.dataframe(sample_data)

    sample_csv = sample_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Sample CSV",
        data=sample_csv,
        file_name="titanic_sample_data.csv",
        mime="text/csv",
    )


def show_data_analysis_page(app):
    """Display data analysis page"""
    st.header("📈 Data Analysis & Insights")

    data = app.load_data()

    if "train" not in data:
        st.error(
            "Training data not found. Please ensure train.csv exists in the data/raw directory."
        )
        return

    df = data["train"]

    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Passengers", len(df))
    with col2:
        survival_rate = df["Survived"].mean() * 100
        st.metric("Survival Rate", f"{survival_rate:.1f}%")
    with col3:
        avg_age = df["Age"].mean()
        st.metric("Average Age", f"{avg_age:.1f}")
    with col4:
        avg_fare = df["Fare"].mean()
        st.metric("Average Fare", f"${avg_fare:.2f}")

    # Visualizations
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Survival Overview", "👥 Demographics", "💰 Fare Analysis", "🚢 Class Analysis"]
    )

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Survival pie chart
            survival_counts = df["Survived"].value_counts()
            fig = px.pie(
                values=survival_counts.values,
                names=["Did Not Survive", "Survived"],
                title="Overall Survival Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Survival by gender
            survival_by_gender = df.groupby(["Sex", "Survived"]).size().unstack()
            fig = px.bar(survival_by_gender, title="Survival by Gender")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            # Age distribution
            fig = px.histogram(
                df, x="Age", color="Survived", title="Age Distribution by Survival", marginal="box"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Family size analysis
            df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
            family_survival = df.groupby("FamilySize")["Survived"].mean()
            fig = px.bar(
                x=family_survival.index,
                y=family_survival.values,
                title="Survival Rate by Family Size",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            # Fare distribution
            fig = px.box(df, x="Survived", y="Fare", title="Fare Distribution by Survival")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Embarked analysis
            embarked_survival = df.groupby(["Embarked", "Survived"]).size().unstack()
            fig = px.bar(embarked_survival, title="Survival by Embarkation Port")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        col1, col2 = st.columns(2)

        with col1:
            # Class survival
            class_survival = df.groupby(["Pclass", "Survived"]).size().unstack()
            fig = px.bar(class_survival, title="Survival by Passenger Class")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Class and gender
            class_gender = df.groupby(["Pclass", "Sex"])["Survived"].mean().unstack()
            fig = px.imshow(
                class_gender,
                title="Survival Rate by Class and Gender",
                aspect="auto",
                color_continuous_scale="RdYlGn",
            )
            st.plotly_chart(fig, use_container_width=True)


def show_model_info_page(app):
    """Display model information page"""
    st.header("🔧 Model Information")

    if app.model is None:
        st.error("Model not loaded.")
        return

    # Model details
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Model Details")
        st.write(f"**Model Type:** {type(app.model).__name__}")

        if hasattr(app.model, "n_estimators"):
            st.write(f"**Number of Estimators:** {app.model.n_estimators}")
        if hasattr(app.model, "max_depth"):
            st.write(f"**Max Depth:** {app.model.max_depth}")
        if hasattr(app.model, "random_state"):
            st.write(f"**Random State:** {app.model.random_state}")

    with col2:
        st.subheader("⚙️ Configuration")
        if app.config:
            st.write(f"**Model Name:** {app.config['names']['model_name']}")
            st.write(f"**Target Column:** {app.config['dataset']['target_col']}")
            st.write(f"**Test Size:** {app.config['dataset']['test_size']}")

    # Feature importance (if available)
    if hasattr(app.model, "feature_importances_"):
        st.subheader("📊 Feature Importance")

        # Try to get feature names
        feature_names = getattr(app.model, "feature_names_in_", None)
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(app.model.feature_importances_))]

        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": app.model.feature_importances_}
        ).sort_values("Importance", ascending=True)

        fig = px.bar(
            importance_df, x="Importance", y="Feature", orientation="h", title="Feature Importance"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Model performance (if available)
    reports_path = os.path.join(app.config["paths"]["reports_parent_dir"], "random_forest")
    eval_report_path = os.path.join(reports_path, "evaluation_report.json")

    if os.path.exists(eval_report_path):
        st.subheader("📈 Model Performance")
        try:
            with open(eval_report_path, "r") as f:
                eval_report = json.load(f)

            # Display metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                accuracy = eval_report.get("accuracy", 0)
                st.metric("Accuracy", f"{accuracy:.3f}")

            with col2:
                precision = eval_report.get("precision", 0)
                st.metric("Precision", f"{precision:.3f}")

            with col3:
                recall = eval_report.get("recall", 0)
                st.metric("Recall", f"{recall:.3f}")

            with col4:
                f1 = eval_report.get("f1_score", 0)
                st.metric("F1 Score", f"{f1:.3f}")

        except Exception as e:
            st.error(f"Failed to load evaluation report: {e}")


if __name__ == "__main__":
    main()
