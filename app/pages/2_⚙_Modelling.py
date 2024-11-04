import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import io
import pickle
from typing import List, Literal, Dict, Any
from abc import ABC, abstractmethod
from copy import deepcopy

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.artifact import Artifact

# Import scikit-learn metrics and models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Set page configuration
st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

# Load registry data directly without loading dataset files
try:
    # Adjust the path to your registry file as needed
    registry_file_path = os.path.join('assets', 'registry.json')
    with open(registry_file_path, 'r') as f:
        registry_data = json.load(f)
except Exception as e:
    st.error(f"Error reading registry file: {e}")
    registry_data = []

datasets = []
for data in registry_data:
    if data['type'] == 'dataset':
        dataset_name = data['name']
        asset_path = data['asset_path']
        datasets.append({
            'name': dataset_name,
            'asset_path': asset_path
        })

if datasets:
    dataset_options = [dataset['name'] for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a dataset", dataset_options)
    selected_dataset_info = next((ds for ds in datasets if ds['name'] == selected_dataset_name), None)

    if selected_dataset_info:
        try:
            # Attempt to load the dataset
            selected_dataset_data = automl.registry.load(selected_dataset_name)
            # Create a Dataset object
            selected_dataset = Dataset(
                name=selected_dataset_name,
                asset_path=selected_dataset_info['asset_path'],
                data=selected_dataset_data,
            )

            # Proceed with your existing code, using `selected_dataset`

            # Define the detect_feature_types function
            def detect_feature_types(dataset: Dataset) -> List[Feature]:
                df = dataset.read()
                features = []

                # Identify numerical columns
                numerical_columns = df.select_dtypes(include=['number']).columns
                for col in numerical_columns:
                    features.append(Feature(name=col, type='numerical'))

                # Identify categorical columns
                categorical_columns = df.select_dtypes(exclude=['number']).columns
                for col in categorical_columns:
                    features.append(Feature(name=col, type='categorical'))

                return features

            # Define Model abstract base class and model implementations
            class Model(ABC):
                def __init__(self, model_type: Literal["classification", "regression"]):
                    self.model_type = model_type  # "classification" or "regression"
                    self.parameters: Dict[str, Any] = {}  # Dictionary to hold model parameters

                @abstractmethod
                def fit(self, X: np.ndarray, y: np.ndarray):
                    """Train the model with input data X and target labels y."""
                    pass

                @abstractmethod
                def predict(self, X: np.ndarray) -> np.ndarray:
                    """Predict output using the trained model on input data X."""
                    pass

                def get_params(self) -> Dict[str, Any]:
                    """Get model parameters."""
                    return deepcopy(self.parameters)

                def set_params(self, **params: Any):
                    """Set model parameters."""
                    self.parameters.update(params)

                def to_artifact(self, name: str):
                    """Convert model and its parameters into an artifact."""
                    data = {
                        "model_type": self.model_type,
                        "parameters": self.parameters,
                    }
                    return Artifact(name=name, data=data)

                def save(self, path: str):
                    """Save the model parameters to a file."""
                    with open(path, 'wb') as file:
                        pickle.dump(self.parameters, file)

                def load(self, path: str):
                    """Load the model parameters from a file."""
                    with open(path, 'rb') as file:
                        self.parameters = pickle.load(file)

            # Classification Models

            class LogisticRegressionModel(Model):
                def __init__(self):
                    super().__init__(model_type="classification")
                    self.model = LogisticRegression()
                    self.parameters = self.model.get_params()

                def fit(self, X: np.ndarray, y: np.ndarray):
                    self.model.fit(X, y)
                    self.parameters = self.model.get_params()

                def predict(self, X: np.ndarray) -> np.ndarray:
                    return self.model.predict(X)

            class DecisionTreeClassifierModel(Model):
                def __init__(self):
                    super().__init__(model_type="classification")
                    self.model = DecisionTreeClassifier()
                    self.parameters = self.model.get_params()

                def fit(self, X: np.ndarray, y: np.ndarray):
                    self.model.fit(X, y)
                    self.parameters = self.model.get_params()

                def predict(self, X: np.ndarray) -> np.ndarray:
                    return self.model.predict(X)

            class RandomForestClassifierModel(Model):
                def __init__(self):
                    super().__init__(model_type="classification")
                    self.model = RandomForestClassifier()
                    self.parameters = self.model.get_params()

                def fit(self, X: np.ndarray, y: np.ndarray):
                    self.model.fit(X, y)
                    self.parameters = self.model.get_params()

                def predict(self, X: np.ndarray) -> np.ndarray:
                    return self.model.predict(X)

            # Regression Models

            class LinearRegressionModel(Model):
                def __init__(self):
                    super().__init__(model_type="regression")
                    self.model = LinearRegression()
                    self.parameters = self.model.get_params()

                def fit(self, X: np.ndarray, y: np.ndarray):
                    self.model.fit(X, y)
                    self.parameters = self.model.get_params()

                def predict(self, X: np.ndarray) -> np.ndarray:
                    return self.model.predict(X)

            class RidgeRegressionModel(Model):
                def __init__(self):
                    super().__init__(model_type="regression")
                    self.model = Ridge()
                    self.parameters = self.model.get_params()

                def fit(self, X: np.ndarray, y: np.ndarray):
                    self.model.fit(X, y)
                    self.parameters = self.model.get_params()

                def predict(self, X: np.ndarray) -> np.ndarray:
                    return self.model.predict(X)

            class DecisionTreeRegressorModel(Model):
                def __init__(self):
                    super().__init__(model_type="regression")
                    self.model = DecisionTreeRegressor()
                    self.parameters = self.model.get_params()

                def fit(self, X: np.ndarray, y: np.ndarray):
                    self.model.fit(X, y)
                    self.parameters = self.model.get_params()

                def predict(self, X: np.ndarray) -> np.ndarray:
                    return self.model.predict(X)

            # Define Metric abstract class and specific metric classes
            class Metric(ABC):
                @abstractmethod
                def evaluate(self, predictions, targets):
                    pass

                @abstractmethod
                def __str__(self):
                    pass

            # Classification Metrics
            class AccuracyMetric(Metric):
                def evaluate(self, predictions, targets):
                    return accuracy_score(targets, predictions)

                def __str__(self):
                    return "Accuracy"

            class PrecisionMetric(Metric):
                def evaluate(self, predictions, targets):
                    return precision_score(targets, predictions, average='macro', zero_division=0)

                def __str__(self):
                    return "Precision"

            class RecallMetric(Metric):
                def evaluate(self, predictions, targets):
                    return recall_score(targets, predictions, average='macro', zero_division=0)

                def __str__(self):
                    return "Recall"

            class F1ScoreMetric(Metric):
                def evaluate(self, predictions, targets):
                    return f1_score(targets, predictions, average='macro', zero_division=0)

                def __str__(self):
                    return "F1 Score"

            # Regression Metrics
            class MeanSquaredErrorMetric(Metric):
                def evaluate(self, predictions, targets):
                    return mean_squared_error(targets, predictions)

                def __str__(self):
                    return "Mean Squared Error"

            class MeanAbsoluteErrorMetric(Metric):
                def evaluate(self, predictions, targets):
                    return mean_absolute_error(targets, predictions)

                def __str__(self):
                    return "Mean Absolute Error"

            class R2ScoreMetric(Metric):
                def evaluate(self, predictions, targets):
                    return r2_score(targets, predictions)

                def __str__(self):
                    return "R2 Score"

            # Proceed with your existing code, using `selected_dataset`
            # Read the dataset
            df = selected_dataset.read()

            # Detect features
            features = detect_feature_types(selected_dataset)
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
                        'Logistic Regression': LogisticRegressionModel,
                        'Decision Tree Classifier': DecisionTreeClassifierModel,
                        'Random Forest Classifier': RandomForestClassifierModel
                    }
                    model_options = list(classification_models.keys())
                    selected_model_name = st.selectbox("Select a model", model_options)
                    selected_model_class = classification_models[selected_model_name]
                else:
                    regression_models = {
                        'Linear Regression': LinearRegressionModel,
                        'Ridge Regression': RidgeRegressionModel,
                        'Decision Tree Regressor': DecisionTreeRegressorModel
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
                        'Accuracy': AccuracyMetric,
                        'Precision': PrecisionMetric,
                        'Recall': RecallMetric,
                        'F1 Score': F1ScoreMetric
                    }
                    metric_options = list(classification_metrics.keys())
                    selected_metrics = st.multiselect("Select evaluation metrics", metric_options)
                    selected_metric_classes = [classification_metrics[name] for name in selected_metrics]
                else:
                    regression_metrics = {
                        'Mean Squared Error': MeanSquaredErrorMetric,
                        'Mean Absolute Error': MeanAbsoluteErrorMetric,
                        'R2 Score': R2ScoreMetric
                    }
                    metric_options = list(regression_metrics.keys())
                    selected_metrics = st.multiselect("Select evaluation metrics", metric_options)
                    selected_metric_classes = [regression_metrics[name] for name in selected_metrics]

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
                    # Create an instance of the model
                    model_instance = selected_model_class()

                    # Create instances of the selected metrics
                    metric_instances = [metric_class() for metric_class in selected_metric_classes]

                    # Create the Pipeline instance
                    pipeline = Pipeline(
                        metrics=metric_instances,
                        dataset=selected_dataset,
                        model=model_instance,
                        input_features=input_features_objs,
                        target_feature=target_feature_obj,
                        split=split_ratio,
                    )

                    st.write("Executing pipeline...")
                    # Execute the pipeline
                    try:
                        results = pipeline.execute()

                        # Display the results
                        st.subheader("Pipeline Results")

                        # Training metrics
                        st.write("**Training Metrics:**")
                        for metric, value in results['train_metrics']:
                            st.write(f"{metric}: {value}")

                        # Testing metrics
                        st.write("**Testing Metrics:**")
                        for metric, value in results['test_metrics']:
                            st.write(f"{metric}: {value}")
                    except Exception as e:
                        st.error(f"An error occurred during pipeline execution: {e}")

        except Exception as e:
            st.error(f"Error loading dataset '{selected_dataset_name}': {e}")

            # Provide an option to delete the problematic dataset
            if st.button(f"Delete dataset '{selected_dataset_name}'"):
                try:
                    automl.registry.delete(selected_dataset_name)
                    st.success(f"Deleted dataset '{selected_dataset_name}'")
                    st.experimental_rerun()
                except Exception as del_e:
                    st.error(f"Error deleting dataset: {del_e}")
else:
    st.write("No datasets available. Please upload a dataset.")
