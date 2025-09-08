import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="CSV Data Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("📊 CSV Data Visualizer")
    st.markdown("Upload your CSV file and create interactive visualizations")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        help="Upload a CSV file to visualize"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            # Clean numeric-like strings (remove commas, spaces, quotes)
            df = df.replace({',': '', '"': '', ' ':'','-': ''}, regex=True)
            # Convert columns that look numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            # Display basic info
            st.sidebar.header("📋 Dataset Information")
            st.sidebar.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.sidebar.write(f"**Columns:** {', '.join(df.columns.tolist())}")
            
            # Show data preview
            with st.expander("🔍 Preview Data"):
                st.dataframe(df.head(), use_container_width=True)
            
            # Show data types and missing values
            with st.expander("📊 Data Summary"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Data Types:**")
                    dtype_info = df.dtypes.reset_index()
                    dtype_info.columns = ['Column', 'Data Type']
                    st.dataframe(dtype_info, use_container_width=True)
                
                with col2:
                    st.write("**Missing Values:**")
                    missing_info = df.isnull().sum().reset_index()
                    missing_info.columns = ['Column', 'Missing Values']
                    st.dataframe(missing_info, use_container_width=True)
            
            # Data processing options
            st.sidebar.header("⚙️ Data Processing")
            
            # Handle missing values
            if df.isnull().sum().sum() > 0:
                missing_option = st.sidebar.selectbox(
                    "Handle Missing Values:",
                    ["Keep as is", "Drop rows with missing values", "Fill with mean", "Fill with median", "Fill with mode"]
                )
                
                if missing_option == "Drop rows with missing values":
                    df = df.dropna()
                elif missing_option == "Fill with mean":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                elif missing_option == "Fill with median":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                elif missing_option == "Fill with mode":
                    for col in df.columns:
                        if df[col].isnull().sum() > 0:
                            df[col] = df[col].fillna(df[col].mode()[0])
            
            # Column selection
            st.sidebar.header("📈 Visualization Settings")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Visualization type selection
            viz_type = st.sidebar.selectbox(
                "Choose Visualization Type:",
                [
                    "Scatter Plot",
                    "Bubble Plot",  # Added Bubble Plot option
                    "Line Chart",
                    "Bar Chart",
                    "Histogram",
                    "Box Plot",
                    "Violin Plot",
                    "Heatmap",
                    "Pie Chart",
                    "Area Chart"
                ]
            )
            
            # Filter options
            st.sidebar.header("🔍 Filters")
            filter_cols = []
            for col in df.columns:
                if df[col].nunique() < 50:  # Only show filter for columns with reasonable unique values
                    filter_cols.append(col)
            
            selected_filters = {}
            for col in filter_cols:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) > 0:
                    selected_vals = st.sidebar.multiselect(
                        f"Filter {col}:",
                        options=unique_vals,
                        default=unique_vals
                    )
                    selected_filters[col] = selected_vals
            
            # Apply filters
            filtered_df = df.copy()
            for col, vals in selected_filters.items():
                if len(vals) > 0:
                    filtered_df = filtered_df[filtered_df[col].isin(vals)]
            
            # Visualization parameters based on type
            if viz_type == "Scatter Plot":
                if len(numeric_cols) >= 2:
                    x_col = st.sidebar.selectbox("X-axis", df.columns)
                    y_col = st.sidebar.selectbox("Y-axis", numeric_cols)
                    color_col = st.sidebar.selectbox("Color by", [None] + categorical_cols)
                    
                    fig = px.scatter(
                        filtered_df, 
                        x=x_col, 
                        y=y_col, 
                        color=color_col,
                        title=f"Scatter Plot: {x_col} vs {y_col}",
                        hover_data=df.columns.tolist()
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 2 numeric columns for scatter plot")
            
            elif viz_type == "Bubble Plot":  # Added Bubble Plot implementation
                if len(numeric_cols) >= 3:
                    x_col = st.sidebar.selectbox("X-axis", numeric_cols)
                    y_col = st.sidebar.selectbox("Y-axis", numeric_cols)
                    size_col = st.sidebar.selectbox("Size by", numeric_cols)
                    color_col = st.sidebar.selectbox("Color by", [None] + categorical_cols)
                    
                    # Add slider for bubble size scaling
                    size_max = st.sidebar.slider(
                        "Max bubble size", 
                        min_value=10, 
                        max_value=100, 
                        value=50,
                        help="Adjust the maximum size of bubbles"
                    )
                    
                    fig = px.scatter(
                        filtered_df, 
                        x=x_col, 
                        y=y_col, 
                        size=size_col,
                        color=color_col,
                        title=f"Bubble Plot: {x_col} vs {y_col} (Size: {size_col})",
                        hover_data=df.columns.tolist(),
                        size_max=size_max
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 3 numeric columns for bubble plot")
            
            elif viz_type == "Line Chart":
                if len(numeric_cols) >= 1:
                    x_col = st.sidebar.selectbox("X-axis", df.columns)
                    y_col = st.sidebar.selectbox("Y-axis", numeric_cols)
                    color_col = st.sidebar.selectbox("Color by", [None] + categorical_cols)
                    
                    fig = px.line(
                        filtered_df, 
                        x=x_col, 
                        y=y_col, 
                        color=color_col,
                        title=f"Line Chart: {y_col} over {x_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 1 numeric column for line chart")
            
            elif viz_type == "Bar Chart":
                if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                    x_col = st.sidebar.selectbox("X-axis", categorical_cols)
                    y_col = st.sidebar.selectbox("Y-axis", numeric_cols)
                    color_col = st.sidebar.selectbox("Color by", [None] + categorical_cols)
                    
                    fig = px.bar(
                        filtered_df, 
                        x=x_col, 
                        y=y_col, 
                        color=color_col,
                        title=f"Bar Chart: {y_col} by {x_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 1 categorical and 1 numeric column for bar chart")
            
            elif viz_type == "Histogram":
                if len(numeric_cols) >= 1:
                    col = st.sidebar.selectbox("Select column", numeric_cols)
                    color_col = st.sidebar.selectbox("Color by", [None] + categorical_cols)
                    
                    fig = px.histogram(
                        filtered_df, 
                        x=col, 
                        color=color_col,
                        title=f"Histogram of {col}",
                        marginal="box"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 1 numeric column for histogram")
            
            elif viz_type == "Box Plot":
                if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
                    y_col = st.sidebar.selectbox("Y-axis", numeric_cols)
                    x_col = st.sidebar.selectbox("X-axis", [None] + categorical_cols)
                    color_col = st.sidebar.selectbox("Color by", [None] + categorical_cols)
                    
                    fig = px.box(
                        filtered_df, 
                        y=y_col, 
                        x=x_col,
                        color=color_col,
                        title=f"Box Plot of {y_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 1 numeric and 1 categorical column for box plot")
            
            elif viz_type == "Violin Plot":
                if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
                    y_col = st.sidebar.selectbox("Y-axis", numeric_cols)
                    x_col = st.sidebar.selectbox("X-axis", [None] + categorical_cols)
                    color_col = st.sidebar.selectbox("Color by", [None] + categorical_cols)
                    
                    fig = px.violin(
                        filtered_df, 
                        y=y_col, 
                        x=x_col,
                        color=color_col,
                        title=f"Violin Plot of {y_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 1 numeric and 1 categorical column for violin plot")
            
            elif viz_type == "Heatmap":
                if len(numeric_cols) >= 2:
                    selected_numeric = st.sidebar.multiselect(
                        "Select numeric columns for heatmap:",
                        numeric_cols,
                        default=numeric_cols[:min(10, len(numeric_cols))]
                    )
                    
                    if len(selected_numeric) >= 2:
                        corr_matrix = filtered_df[selected_numeric].corr()
                        fig = px.imshow(
                            corr_matrix,
                            title="Correlation Heatmap",
                            aspect="auto",
                            color_continuous_scale="RdBu_r"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Select at least 2 numeric columns for heatmap")
                else:
                    st.warning("Need at least 2 numeric columns for heatmap")
            
            elif viz_type == "Pie Chart":
                if len(categorical_cols) >= 1:
                    col = st.sidebar.selectbox("Select column", categorical_cols)
                    
                    value_counts = filtered_df[col].value_counts()
                    fig = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f"Pie Chart of {col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 1 categorical column for pie chart")
            
            elif viz_type == "Area Chart":
                if len(numeric_cols) >= 1:
                    x_col = st.sidebar.selectbox("X-axis", df.columns)
                    y_col = st.sidebar.selectbox("Y-axis", numeric_cols)
                    color_col = st.sidebar.selectbox("Color by", [None] + categorical_cols)
                    
                    fig = px.area(
                        filtered_df, 
                        x=x_col, 
                        y=y_col, 
                        color=color_col,
                        title=f"Area Chart: {y_col} over {x_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 1 numeric column for area chart")
            
            # Display filtered data
            with st.expander("📋 View Filtered Data"):
                st.write(f"**Filtered data shape:** {filtered_df.shape[0]} rows × {filtered_df.shape[1]} columns")
                st.dataframe(filtered_df, use_container_width=True)
            
            # Download filtered data
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv,
                file_name="filtered_data.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload a CSV file to get started")

if __name__ == "__main__":
    main()
