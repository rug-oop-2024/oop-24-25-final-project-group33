import streamlit as st
import pandas as pd
import pickle
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.artifact import Artifact  # Ensure Artifact class is imported

# Initialize the AutoMLSystem Singleton
automl = AutoMLSystem.get_instance()

# Title for the dataset management page
st.title("Dataset Management")

# Section: List Available Datasets
st.subheader("Available Datasets")
datasets = automl.registry.list(type="dataset")  # Fetch list of datasets from the registry

# Display list of datasets with options to view and delete
if datasets:
    # Create a list of dataset names for the selectbox
    dataset_options = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a dataset to view or delete", dataset_options)

    # Retrieve the selected dataset object
    dataset_data = next((ds for ds in datasets if ds.name == selected_dataset_name), None)

    if dataset_data:
        st.write(f"**Dataset Name:** {dataset_data.name}")
        
        # Access the asset_path safely
        asset_path = getattr(dataset_data, "asset_path", None)
        if asset_path:
            st.write(f"**Path:** {asset_path}")
        
        # Deserialize data and show a sample
        dataset_df = pd.DataFrame(pickle.loads(dataset_data.data))  # Ensure to access data correctly
        st.write("**Sample Data:**")
        st.dataframe(dataset_df.head())

        # Option to delete the selected dataset
        if st.button("Delete Dataset"):
            try:
                automl.registry.delete(selected_dataset_name)  # Use the correct name to delete
                st.success(f"Deleted dataset: {selected_dataset_name}")
                st.experimental_rerun()  # Refresh page to update dataset list after deletion
            except ValueError as e:
                st.error(str(e))  # Display error if the artifact does not exist
else:
    st.write("No datasets available.")

# Section: Upload New Dataset
st.subheader("Upload New Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# If a file is uploaded, show preview and save options
if uploaded_file:
    # Load the CSV file into a DataFrame and display a preview
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # Input field to name the new dataset
    dataset_name = st.text_input("Enter a name for the dataset")

    # Save the dataset when button is clicked
    if st.button("Save Dataset") and dataset_name:
        asset_path = f"{dataset_name}.csv"  # Set asset_path explicitly

        # Create a Dataset object and manually define the artifact's structure
        new_artifact = Artifact(
            name=dataset_name,
            asset_path=asset_path,  # Ensure asset_path is set
            data=pickle.dumps(df),  # Serialize the DataFrame
            type="dataset",
            metadata={"description": f"Dataset for {dataset_name}"}
        )

        # Save the artifact in the registry
        try:
            automl.registry.register(new_artifact)
            st.success(f"Dataset '{dataset_name}' has been saved successfully.")
        except Exception as e:
            st.error(f"Error: Unable to save dataset. {str(e)}")

        # Refresh the page to show the updated list of datasets
        st.experimental_rerun()
