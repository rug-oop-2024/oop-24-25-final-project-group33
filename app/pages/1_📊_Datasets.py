import streamlit as st
import pandas as pd
import os
import json

# Title for the dataset management page
st.title("Dataset Management")

# Directories and files
ASSETS_DIR = 'assets'
OBJECTS_DIR = os.path.join(ASSETS_DIR, 'objects')
REGISTRY_FILE = os.path.join(ASSETS_DIR, 'registry.json')

# Ensure directories exist
os.makedirs(OBJECTS_DIR, exist_ok=True)


# Function to load the registry
def load_registry() -> list:
    """
    Load registry from file if exists, else return empty list.
    Returns:
        list: The registry data.
    """
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, 'r') as f:
            registry = json.load(f)
    else:
        registry = []
    return registry


# Function to save the registry
def save_registry(registry: list) -> None:
    """
    Save the registry to a file.
    Args:
        registry (list): The registry data to save.
    Returns:
        None
    """
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=4)


# Function to get dataset files from the registry
def get_dataset_files() -> list:
    """
    Returns a list of dataset file names from the registry.
    """
    registry = load_registry()
    dataset_files = [e['name'] for e in registry if e['type'] == 'dataset']
    return dataset_files


# Initialize session state for datasets
if 'dataset_files' not in st.session_state:
    st.session_state['dataset_files'] = get_dataset_files()


# Function to refresh dataset list
def refresh_datasets() -> None:
    """
    Refresh dataset files in session state.
    """
    st.session_state['dataset_files'] = get_dataset_files()


# Session state variable to trigger rerun
if 'refresh' not in st.session_state:
    st.session_state.refresh = 0

# Section: List Available Datasets
st.subheader("Available Datasets")

if st.session_state['dataset_files']:
    dataset_options = st.session_state['dataset_files']

    # Before the selectbox, check if the selected dataset is still valid
    if 'selected_dataset_name' in st.session_state:
        if st.session_state.selected_dataset_name not in dataset_options:
            if dataset_options:
                st.session_state.selected_dataset_name = dataset_options[0]
            else:
                st.session_state.selected_dataset_name = None

    # Selectbox for dataset selection
    selected_dataset_name = st.selectbox(
        "Select a dataset to view or delete",
        options=dataset_options,
        key='selected_dataset_name'
    )

    # Load and display the selected dataset
    if selected_dataset_name:
        dataset_path = os.path.join(
            OBJECTS_DIR, selected_dataset_name + '.csv')
        if os.path.exists(dataset_path):
            dataset_df = pd.read_csv(dataset_path)
            st.write(f"**Dataset Name:** {selected_dataset_name}")
            st.write("**Sample Data:**")
            st.dataframe(dataset_df.head())
        else:
            st.error(f"Dataset '{selected_dataset_name}' not found.")
            refresh_datasets()

    # Function to delete dataset
    def delete_dataset() -> None:
        """
        Deletes the selected dataset and updates the registry.
        Parameters:
        None
        Returns:
        None
        """
        dataset_path = os.path.join(
            OBJECTS_DIR, f"{selected_dataset_name}.csv")
        if os.path.exists(dataset_path):
            os.remove(dataset_path)
            st.success(f"Deleted dataset: {selected_dataset_name}")
            # Update the registry
            registry = load_registry()
            registry = [
                e for e in registry
                if not (
                    e['type'] == 'dataset' and e
                    ['name'] == selected_dataset_name
                )
            ]
            save_registry(registry)
            # Refresh the dataset list
            refresh_datasets()
            # Increment refresh to trigger rerun
            st.session_state.refresh += 1
        else:
            st.error(f"Dataset '{selected_dataset_name}' does not exist.")

    # Button to delete the selected dataset
    if st.button("Delete Dataset"):
        delete_dataset()
else:
    st.write("No datasets available.")

# Initialize counters for the upload widgets
if 'uploader_counter' not in st.session_state:
    st.session_state.uploader_counter = 0
if 'dataset_name_counter' not in st.session_state:
    st.session_state.dataset_name_counter = 0

# Section: Upload New Dataset
st.subheader("Upload New Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type="csv",
    key=f"upload_new_dataset_{st.session_state.uploader_counter}"
)

# If a file is uploaded, show preview and save options
if uploaded_file:
    # Load CSV into a DataFrame and display a preview
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # Input for dataset name
    dataset_name = st.text_input(
        "Enter a name for the dataset",
        key=f'dataset_name_{st.session_state.dataset_name_counter}'
    )

    # Function to save the dataset
    def save_dataset() -> None:
        """
        Save the dataset and update the registry.
        Saves the dataset to a CSV file, updates the registry with the new
        dataset entry, and refreshes the dataset list. Increments counters
        to reset widgets and triggers a rerun.
        Args:
            None
        Returns:
            None
        """
        if dataset_name:
            dataset_path = os.path.join(OBJECTS_DIR, dataset_name + '.csv')
            if os.path.exists(dataset_path):
                st.error(f"A dataset with the name '{dataset_name}'\
already exists.")
            else:
                df.to_csv(dataset_path, index=False)
                st.success(f"Dataset '{dataset_name}'\
has been saved successfully.")
                # Update the registry
                registry = load_registry()
                new_entry = {
                    "name": dataset_name,
                    "type": "dataset",
                    "asset_path": os.path.relpath(
                        dataset_path, start=ASSETS_DIR)
                }
                registry.append(new_entry)
                save_registry(registry)
                # Refresh the dataset list
                refresh_datasets()
                # Increment the counters to reset the widgets
                st.session_state.uploader_counter += 1
                st.session_state.dataset_name_counter += 1
                # Increment refresh to trigger rerun
                st.session_state.refresh += 1
        else:
            st.error("Please enter a dataset name.")

    # Button to save the dataset
    if st.button("Save Dataset"):
        save_dataset()

# Add a refresh button at the bottom of the page
if st.button("Refresh Page"):
    # Increment refresh to trigger rerun
    st.session_state.refresh += 1

# Add description under the refresh button
st.write("Refresh the page to see your changes.")
