import streamlit as st
import pandas as pd
import os
import json
import time
from typing import List
from sklearn.model_selection import train_test_split
from autoop.core.ml.feature import Feature
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Modelling", page_icon="📈")

# Directory for saving pipelines
PIPELINES_DIR = os.path.join('assets', 'pipelines')
os.makedirs(PIPELINES_DIR, exist_ok=True)


# Load datasets from the registry
def load_registry() -> List[dict]:
    """
    Load registry data from 'assets/registry.json' if it exists.
    Returns:
        List[dict]: List of registry entries or empty list if file not found.
    """
    ASSETS_DIR = 'assets'
    REGISTRY_FILE = os.path.join(ASSETS_DIR, 'registry.json')
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, 'r') as f:
            return json.load(f)
    return []


registry = load_registry()
datasets = [entry for entry in registry if entry['type'] == 'dataset']

# Display page
st.write("# ⚙ Modelling")
st.write(
    "In this section, you can design a machine learning "
    "pipeline to train a model on a dataset."
)

if datasets:
    st.subheader("Select Dataset")
    dataset_options = [dataset['name'] for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a dataset", dataset_options)

    selected_dataset_info = next(
        (ds for ds in datasets if ds['name'] == selected_dataset_name), None
    )
    if selected_dataset_info:
        dataset_path = os.path.join(
            'assets', selected_dataset_info['asset_path']
        )
        df = pd.read_csv(dataset_path)

        # Check for missing values
        if df.isnull().values.any():
            st.error(
                "Dataset contains missing values. "
                "Please provide a dataset with no NaN or missing values."
            )
            st.stop()

        # Check for categorical and numerical features only
        allowed_dtypes = ['int64', 'float64', 'object']
        if not all(dtype in allowed_dtypes for dtype in df.dtypes):
            st.error(
                "Dataset contains unsupported data types. "
                "Only categorical and numerical features are allowed."
            )
            st.stop()

        st.write("Preview of selected dataset:")
        st.dataframe(df.head())

        # Detect features
        features = [
            Feature(
                name=col,
                type='numerical' if df[col].dtype in [
                    'int64', 'float64'
                ] else 'categorical'
            ) for col in df.columns
        ]
        feature_names = [feature.name for feature in features]

        st.subheader("Select Features")
        input_features = st.multiselect("Select input features", feature_names)
        target_feature = st.selectbox("Select target feature", feature_names)

        # Manual feature selection
        if not input_features or not target_feature:
            st.error("Please select input and target features manually.")
            st.stop()

        if target_feature in input_features:
            st.error("Target feature cannot be one of the input features. \
Please select different features.")
            st.stop()

        # Determine task type based on target feature type
        # Determine task type based on target feature type
        task_type = 'classification' if df[target_feature].dtype == 'object\
' else 'regression'
        st.write(f"Detected task type: **{task_type.capitalize()}**")

        # Dynamically set model options based on task type
        if task_type == 'classification':
            model_options = {
                'Logistic Regression': LogisticRegression,
                'Decision Tree Classifier': DecisionTreeClassifier,
                'Random Forest Classifier': RandomForestClassifier
            }
        else:  # Regression models
            model_options = {
                'Linear Regression': LinearRegression,
                'Ridge Regression': Ridge,
                'Decision Tree Regressor': DecisionTreeRegressor
            }

        # Allow user to select from the relevant model options
        selected_model_name = st.selectbox(
            "Select a model", list(model_options.keys()))
        selected_model_class = model_options[selected_model_name]

        # Dataset split
        st.subheader("Select Dataset Split")
        split_ratio = st.slider(
            "Select train/test split ratio",
            min_value=0.1,
            max_value=0.9,
            value=0.8
        )

        # Evaluation metrics
        st.subheader("Select Evaluation Metrics")
        metrics = {
            'classification': {
                'Accuracy': accuracy_score,
                'Precision': precision_score,
                'Recall': recall_score,
                'F1 Score': f1_score
            },
            'regression': {
                'Mean Squared Error': mean_squared_error,
                'Mean Absolute Error': mean_absolute_error,
                'R2 Score': r2_score
            }
        }
        selected_metrics = st.multiselect(
            "Select evaluation metrics",
            list(metrics[task_type].keys())
        )
        selected_metric_functions = [
            metrics[task_type][metric] for metric in selected_metrics
        ]

        # Pipeline name and version
        st.subheader("Save Pipeline")
        pipeline_name = st.text_input("Pipeline Name")
        pipeline_version = st.text_input("Pipeline Version")

        # Function to preprocess data and handle categorical features
        def preprocess_data(df, input_features, target_feature, task_type):
            """
            Preprocesses the dataset for modeling.
            Args:
                df (pd.DataFrame): The input dataset.
                input_features (list): List of input feature names.
                target_feature (str): The target feature name.
                task_type (str): Type of task ('classification' or
                'regression').
            Returns:
                pd.DataFrame: The preprocessed dataset.
            """
            # Ensure df is a DataFrame
            if not isinstance(df, pd.DataFrame):
                raise TypeError("The provided dataset is not a DataFrame.")

            # Select only the specified input and target columns,
            # making a copy to avoid modification issues
            selected_columns = input_features + [target_feature]

            # Ensure selected columns are in the DataFrame
            missing_columns = [
                col for col in selected_columns if col not in df.columns
            ]
            if missing_columns:
                raise KeyError(f"The following columns \
are missing in the dataset: {missing_columns}")

            data = df[selected_columns].copy()

            # Confirm target feature is in data columns after selection
            if target_feature not in data.columns:
                raise KeyError(f"The target feature \
'{target_feature}' is missing from the selected data columns.")
            # One-hot encode categorical features in input_features
            for col in input_features:
                if col in data.columns and pd.api.types.is_object_dtype(
                        data[col]):
                    # Apply one-hot encoding with individual
                    # prefixes for each categorical column
                    data = pd.get_dummies(data, columns=[col], prefix=col)

            # Check if target feature is categorical for classification tasks
            if task_type == 'classification' and pd.api.types.is_object_dtype(
                data[target_feature]
            ):
                # Encode target feature as numeric labels
                data[target_feature] = pd.factorize(data[target_feature])[0]

            # Debugging statements to ensure the target
            # column exists after processing
            print("Columns after preprocessing:", data.columns)
            print(f"Data types after preprocessing:\n{data.dtypes}")
            print(f"Target feature preview: {data[target_feature].head()}")

            return data

        # Preprocess the data before splitting and training
        data = preprocess_data(df, input_features, target_feature, task_type)
        X = data.drop(columns=[target_feature])
        y = data[target_feature]

        # Run pipeline button
        if st.button("Run and Save") and pipeline_name and pipeline_version:
            st.write("Training the model...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=(1 - split_ratio), random_state=42
            )
            model = selected_model_class()

            for i in range(1, 101):
                time.sleep(0.02)
                progress_bar.progress(i / 100)
                status_text.text(f"Training Progress: {i}%")

            model.fit(X_train, y_train)
            st.write("Training Complete!")

            # Evaluate and display metrics
            # Evaluate and display metrics
            st.write("**Pipeline Results:**")
            results = {}
            for metric_func in selected_metric_functions:
                try:
                    # Use average='weighted' for multiclass
                    # classification if necessary
                    if task_type == 'classification' and metric_func in [
                        precision_score, recall_score, f1_score
                    ]:
                        # Detect multiclass setting and apply
                        # 'weighted' average for relevant metrics
                        train_metric = metric_func(
                            y_train, model.predict(X_train), average='weighted'
                        )
                        test_metric = metric_func(
                            y_test, model.predict(X_test), average='weighted')
                    else:
                        train_metric = metric_func(
                            y_train, model.predict(X_train))
                        test_metric = metric_func(
                            y_test, model.predict(X_test))

                    metric_name = metric_func.__name__.replace(
                        "_", " ").title()
                    st.write(f"{metric_name} (Train): {train_metric}")
                    st.write(f"{metric_name} (Test): {test_metric}")
                    results[metric_name] = {
                        "train": train_metric, "test": test_metric
                    }
                except ValueError as e:
                    st.warning(f"Could not calculate \
{metric_func.__name__} due to: {e}")

            # Save the pipeline metadata
            pipeline_data = {
                "pipeline_name": pipeline_name,
                "pipeline_version": pipeline_version,
                "model_type": task_type,
                "model_name": selected_model_name,
                "input_features": input_features,
                "target_feature": target_feature,
                "split_ratio": split_ratio,
                "metrics": results
            }
            pipeline_path = os.path.join(
                PIPELINES_DIR,
                f"{pipeline_name}_{pipeline_version}.json"
            )

            with open(pipeline_path, 'w') as f:
                json.dump(pipeline_data, f, indent=4)

            # Save the trained model as a .pkl file
            model_file_path = os.path.join(
                PIPELINES_DIR,
                f"{pipeline_name}_{pipeline_version}_model.pkl"
            )
            with open(model_file_path, 'wb') as model_file:
                pickle.dump(model, model_file)

            st.success(
                f"Pipeline '{pipeline_name}' (version {pipeline_version}) "
                "has been saved successfully."
            )

            # Visualization for regression
            if task_type == "regression":
                st.write("### Prediction vs Actual Plot")
                y_pred = model.predict(X_test)

                # Calculate additional statistics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Plot the predictions vs actual values and distributions
                fig, ax = plt.subplots(1, 3, figsize=(18, 6))

                # Scatter plot of Actual vs Predicted
                ax[0].scatter(
                    y_test, y_pred,
                    alpha=0.6,
                    edgecolor="k",
                    label="Predictions"
                )
                ax[0].plot(
                    [y_test.min(), y_test.max()],
                    [y_test.min(), y_test.max()],
                    'r--', label="Perfect Fit Line"
                )
                ax[0].set_xlabel("Actual Values")
                ax[0].set_ylabel("Predicted Values")
                ax[0].set_title(
                    f"Prediction vs Actual for {selected_model_name}"
                )
                ax[0].legend(loc="upper left")

                # Add a text box with metrics to the main scatter plot
                textstr = (
                    f"Mean Absolute Error: {mae:.2f}\n"
                    f"Mean Squared Error: {mse:.2f}\n"
                    f"R² Score: {r2:.2f}"
                )
                props = dict(boxstyle='round', facecolor='white', alpha=0.7)
                ax[0].text(
                    0.95, 0.05, textstr, transform=ax[0].transAxes,
                    fontsize=10, verticalalignment='bottom',
                    horizontalalignment='right', bbox=props
                )

                # Plot 2: Actual Distribution
                sns.histplot(
                    y_test, kde=True, color="blue", ax=ax[1], label="Actual"
                )
                ax[1].set_title("Actual Values Distribution")
                ax[1].set_xlabel("Actual Values")
                ax[1].legend()

                # Plot 3: Predicted Distribution
                sns.histplot(
                    y_pred, kde=True, color="green",
                    ax=ax[2], label="Predicted"
                )
                ax[2].set_title("Predicted Values Distribution")
                ax[2].set_xlabel("Predicted Values")
                ax[2].legend()

                # Display the figure
                st.pyplot(fig)


else:
    st.write(
        "No datasets available. Please upload a dataset to start modeling."
    )
