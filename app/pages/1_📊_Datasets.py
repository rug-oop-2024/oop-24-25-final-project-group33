import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

# Initialize AutoMLSystem
automl = AutoMLSystem.get_instance()

# List available datasets
st.title("Dataset Management")
st.subheader("Available Datasets")
datasets = automl.registry.list(type="dataset")

# Display list of datasets
if datasets:
    selected_dataset = st.selectbox("Select a dataset to view or delete", datasets)
    if selected_dataset:
        dataset = automl.registry.get(selected_dataset)
        st.write(f"**Dataset Name:** {dataset.name}")
        st.write(f"**Path:** {dataset.asset_path}")
        st.write("**Sample Data:**")
        st.dataframe(dataset.data.head())

        # Delete Dataset
        if st.button("Delete Dataset"):
            automl.registry.delete(selected_dataset)
            st.success(f"Deleted dataset: {selected_dataset}")
            st.experimental_rerun()
else:
    st.write("No datasets available.")

# Dataset Upload Section
st.subheader("Upload New Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    dataset_name = st.text_input("Enter a name for the dataset")
    if st.button("Save Dataset") and dataset_name:
        new_dataset = Dataset.from_dataframe(name=dataset_name, asset_path=f"{dataset_name}.csv", data=df)
        automl.registry.save(new_dataset)
        st.success(f"Dataset '{dataset_name}' has been saved successfully.")
        st.experimental_rerun()
