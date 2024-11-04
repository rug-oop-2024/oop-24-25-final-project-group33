import streamlit as st
import pandas as pd
import os

# Title for the dataset management page
st.title("Dataset Management")

# Directory to store datasets
DATASETS_DIR = "datasets"
if not os.path.exists(DATASETS_DIR):
    os.makedirs(DATASETS_DIR)

# Function to get dataset files
def get_dataset_files():
    return [f for f in os.listdir(DATASETS_DIR) if f.endswith('.csv')]

# Initialize session state for datasets
if 'dataset_files' not in st.session_state:
    st.session_state['dataset_files'] = get_dataset_files()

# Function to refresh dataset list
def refresh_datasets():
    st.session_state['dataset_files'] = get_dataset_files()

# **Session state variable to trigger rerun**
if 'refresh' not in st.session_state:
    st.session_state.refresh = 0

# Section: List Available Datasets
st.subheader("Available Datasets")

if st.session_state['dataset_files']:
    # Create a list of dataset names (remove .csv extension)
    dataset_options = [os.path.splitext(f)[0] for f in st.session_state['dataset_files']]

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
        dataset_path = os.path.join(DATASETS_DIR, selected_dataset_name + '.csv')
        if os.path.exists(dataset_path):
            dataset_df = pd.read_csv(dataset_path)
            st.write(f"**Dataset Name:** {selected_dataset_name}")
            st.write("**Sample Data:**")
            st.dataframe(dataset_df.head())
        else:
            st.error(f"Dataset '{selected_dataset_name}' not found.")
            refresh_datasets()

    # Function to delete dataset
    def delete_dataset():
        dataset_path = os.path.join(DATASETS_DIR, selected_dataset_name + '.csv')
        if os.path.exists(dataset_path):
            os.remove(dataset_path)
            st.success(f"Deleted dataset: {selected_dataset_name}")
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
    def save_dataset():
        if dataset_name:
            dataset_path = os.path.join(DATASETS_DIR, dataset_name + '.csv')
            if os.path.exists(dataset_path):
                st.error(f"A dataset with the name '{dataset_name}' already exists.")
            else:
                df.to_csv(dataset_path, index=False)
                st.success(f"Dataset '{dataset_name}' has been saved successfully.")
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

# **Add description under the refresh button**
st.write("Refresh the page to see your changes.")
