import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

# Initialize AutoMLSystem Singleton
automl = AutoMLSystem.get_instance()

# Title and description for dataset management
st.title("Dataset Management")

# Section: List Available Datasets
st.subheader("Available Datasets")
datasets = automl.registry.list(type="dataset")  # Fetch list of datasets from registry

# Display existing datasets with view and delete options
if datasets:
    selected_dataset = st.selectbox("Select a dataset to view or delete", datasets)
    if selected_dataset:
        # Retrieve and display dataset details
        dataset = automl.registry.get(selected_dataset)
        st.write(f"**Dataset Name:** {dataset.name}")
        st.write(f"**Path:** {dataset.asset_path}")
        st.write("**Sample Data:**")
        st.dataframe(dataset.data.head())

        # Delete selected dataset
        if st.button("Delete Dataset"):
            automl.registry.delete(selected_dataset)
            st.success(f"Deleted dataset: {selected_dataset}")
            st.experimental_rerun()  # Refresh page to update dataset list after deletion
else:
    st.write("No datasets available.")

# Section: Upload New Dataset
st.subheader("Upload New Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# If a file is uploaded, show preview and save options
if uploaded_file:
    # Load CSV into a DataFrame and display preview
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # Input for dataset name
    dataset_name = st.text_input("Enter a name for the dataset")
    
    # Save the dataset when button is clicked
    if st.button("Save Dataset") and dataset_name:
        # Convert DataFrame to Dataset object and save to registry
        new_dataset = Dataset.from_dataframe(name=dataset_name, asset_path=f"{dataset_name}.csv", data=df)
        automl.registry.save(new_dataset)
        
        # Confirmation message and refresh
        st.success(f"Dataset '{dataset_name}' has been saved successfully.")
        st.experimental_rerun()
