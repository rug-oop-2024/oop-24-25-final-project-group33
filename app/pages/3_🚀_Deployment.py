import streamlit as st
import os
import json

# Set the page configuration
st.set_page_config(page_title="Deployment", page_icon="🚀")

# Title for the deployment page
st.write("# 🚀 Deployment")
st.write("View and manage your saved machine learning pipelines here.")

# Directory where saved pipeline artifacts are stored
PIPELINES_DIR = os.path.join('assets', 'pipelines')

# Ensure the directory exists
if not os.path.exists(PIPELINES_DIR):
    os.makedirs(PIPELINES_DIR)

# Function to get list of saved pipelines
def get_saved_pipelines():
    return [f for f in os.listdir(PIPELINES_DIR) if f.endswith('.json')]

# Load existing pipelines
pipelines = get_saved_pipelines()

# Check if there are any saved pipelines
if pipelines:
    # Create a selectbox to select a pipeline to view
    selected_pipeline_name = st.selectbox("Select a saved pipeline", [os.path.splitext(p)[0] for p in pipelines])

    # Load and display the selected pipeline
    if selected_pipeline_name:
        pipeline_path = os.path.join(PIPELINES_DIR, selected_pipeline_name + '.json')
        
        try:
            # Load the pipeline details from the JSON file
            with open(pipeline_path, 'r') as f:
                pipeline_data = json.load(f)
            
            # Display pipeline details with safe key access
            st.subheader("Pipeline Summary")
            st.write(f"**Name:** {selected_pipeline_name}")
            st.write(f"**Model Type:** {pipeline_data.get('model_type', 'N/A')}")
            st.write(f"**Input Features:** {', '.join(pipeline_data.get('input_features', []))}")
            st.write(f"**Target Feature:** {pipeline_data.get('target_feature', 'N/A')}")
            st.write(f"**Split Ratio:** {pipeline_data.get('split', 'N/A')}")
            st.write("**Metrics:**")
            for metric, values in pipeline_data.get('metrics', {}).items():
                st.write(f"- {metric}: Train: {values.get('train', 'N/A')}, Test: {values.get('test', 'N/A')}")
            
            # Add buttons for further actions (e.g., deploy, delete)
            if st.button("Deploy Pipeline"):
                st.success(f"Pipeline '{selected_pipeline_name}' has been deployed.")
            
            if st.button("Delete Pipeline"):
                os.remove(pipeline_path)
                st.success(f"Pipeline '{selected_pipeline_name}' has been deleted.")
                st.experimental_rerun()  # Refresh the page to update the list of pipelines

        except Exception as e:
            st.error(f"Error loading pipeline '{selected_pipeline_name}': {e}")
else:
    st.write("No saved pipelines available. Please create a pipeline to view it here.")
