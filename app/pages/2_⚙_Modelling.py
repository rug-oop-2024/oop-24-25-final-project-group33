import streamlit as st
import pandas as pd
import os
import json
import time  # Added to simulate progress
from typing import List
from sklearn.model_selection import train_test_split

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.model import Model
from autoop.core.ml.metric import Metric

# Import scikit-learn metrics and models
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# Set page configuration
st.set_page_config(page_title="Modelling", page_icon="📈")

# Helper function to write helper text
def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

# Page title and description
st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

# Load datasets from the registry file
ASSETS_DIR = 'assets'
REGISTRY_FILE = os.path.join(ASSETS_DIR, 'registry.json')

def load_registry():
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, 'r') as f:
            return json.load(f)
    return []

# Load datasets from registry
registry = load_registry()
datasets = [entry for entry in registry if entry['type'] == 'dataset']

# Section: Select Dataset
if datasets:
    st.subheader("Select Dataset")
    dataset_options = [dataset['name'] for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a dataset", dataset_options)

    # Load selected dataset
    selected_dataset_info = next((ds for ds in datasets if ds['name'] == selected_dataset_name), None)
    if selected_dataset_info:
        dataset_path = os.path.join(ASSETS_DIR, selected_dataset_info['asset_path'])
        df = pd.read_csv(dataset_path)
        st.write("Preview of selected dataset:")
        st.dataframe(df.head())

        # Detect features in dataset
        def detect_feature_types(df: pd.DataFrame) -> List[Feature]:
            features = []
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    features.append(Feature(name=col, type='numerical'))
                else:
                    features.append(Feature(name=col, type='categorical'))
            return features

        features = detect_feature_types(df)
        feature_names = [feature.name for feature in features]

        # Section: Select Features
        st.subheader("Select Features")
        input_features = st.multiselect("Select input features", feature_names)
        target_feature = st.selectbox("Select target feature", feature_names)

        if input_features and target_feature:
            # Get Feature objects
            target_feature_obj = next((f for f in features if f.name == target_feature), None)
            input_features_objs = [f for f in features if f.name in input_features]

            # Determine task type
            if target_feature_obj.type == 'categorical':
                task_type = 'classification'
            else:
                task_type = 'regression'

            st.write(f"Detected task type: **{task_type.capitalize()}**")

            # Section: Select Model
            st.subheader("Select Model")
            if task_type == 'classification':
                classification_models = {
                    'Logistic Regression': LogisticRegression,
                    'Decision Tree Classifier': DecisionTreeClassifier,
                    'Random Forest Classifier': RandomForestClassifier
                }
                model_options = list(classification_models.keys())
                selected_model_name = st.selectbox("Select a model", model_options)
                selected_model_class = classification_models[selected_model_name]
            else:
                regression_models = {
                    'Linear Regression': LinearRegression,
                    'Ridge Regression': Ridge,
                    'Decision Tree Regressor': DecisionTreeRegressor
                }
                model_options = list(regression_models.keys())
                selected_model_name = st.selectbox("Select a model", model_options)
                selected_model_class = regression_models[selected_model_name]

            # Section: Select Dataset Split
            st.subheader("Select Dataset Split")
            split_ratio = st.slider("Select train/test split ratio", min_value=0.1, max_value=0.9, value=0.8, step=0.05)

            # Section: Select Evaluation Metrics
            st.subheader("Select Evaluation Metrics")

            if task_type == 'classification':
                classification_metrics = {
                    'Accuracy': accuracy_score,
                    'Precision': precision_score,
                    'Recall': recall_score,
                    'F1 Score': f1_score
                }
                metric_options = list(classification_metrics.keys())
                selected_metrics = st.multiselect("Select evaluation metrics", metric_options)
                selected_metric_functions = [classification_metrics[name] for name in selected_metrics]
            else:
                regression_metrics = {
                    'Mean Squared Error': mean_squared_error,
                    'Mean Absolute Error': mean_absolute_error,
                    'R2 Score': r2_score
                }
                metric_options = list(regression_metrics.keys())
                selected_metrics = st.multiselect("Select evaluation metrics", metric_options)
                selected_metric_functions = [regression_metrics[name] for name in selected_metrics]

            # Section: Pipeline Summary
            st.subheader("Pipeline Summary")
            st.markdown(f"""
            **Dataset:** {selected_dataset_name}  
            **Input Features:** {', '.join(input_features)}  
            **Target Feature:** {target_feature}  
            **Task Type:** {task_type.capitalize()}  
            **Model:** {selected_model_name}  
            **Train/Test Split Ratio:** {split_ratio}  
            **Evaluation Metrics:** {', '.join(selected_metrics)}  
            """)

            # Button to run the pipeline
            if st.button("Run Pipeline"):
                st.write("Training the model...")

                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Split the dataset into train and test sets
                X = df[input_features]
                y = df[target_feature]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - split_ratio), random_state=42)

                # Train the model
                model = selected_model_class()

                # Simulating training progress
                for i in range(1, 101):
                    time.sleep(0.05)  # Simulate training time
                    progress_bar.progress(i / 100)
                    status_text.text(f"Training Progress: {i}%")

                # Train the actual model
                model.fit(X_train, y_train)
                status_text.text("Training Complete!")

                # Make predictions and evaluate
                predictions_train = model.predict(X_train)
                predictions_test = model.predict(X_test)

                st.subheader("Pipeline Results")

                # Training metrics
                st.write("**Training Metrics:**")
                for metric_func in selected_metric_functions:
                    try:
                        if task_type == 'classification':
                            # Use 'macro' or 'weighted' average for non-binary classification
                            metric_value = metric_func(y_train, predictions_train, average='macro')
                        else:
                            metric_value = metric_func(y_train, predictions_train)
                        metric_name = metric_func.__name__.replace('_', ' ').title()
                        st.write(f"{metric_name}: {metric_value}")
                    except Exception as e:
                        st.error(f"Error calculating metric {metric_func.__name__}: {e}")

                # Testing metrics
                st.write("**Testing Metrics:**")
                for metric_func in selected_metric_functions:
                    try:
                        if task_type == 'classification':
                            # Use 'macro' or 'weighted' average for non-binary classification
                            metric_value = metric_func(y_test, predictions_test, average='macro')
                        else:
                            metric_value = metric_func(y_test, predictions_test)
                        metric_name = metric_func.__name__.replace('_', ' ').title()
                        st.write(f"{metric_name}: {metric_value}")
                    except Exception as e:
                        st.error(f"Error calculating metric {metric_func.__name__}: {e}")

else:
    st.write("No datasets available. Please upload a dataset to start modelling.")
