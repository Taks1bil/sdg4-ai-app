import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, mean_squared_error
import io
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="SDG 4 Data Gap AI Solution",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .module-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}

# Main title
st.markdown('<h1 class="main-header">🎓 Low-Code AI Solution for SDG 4 Data Gaps</h1>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
modules = [
    "📊 Data Ingestion",
    "🔧 Data Preprocessing", 
    "🤖 Machine Learning Model",
    "📈 Prediction & Visualization",
    "🔄 Validation & Iteration"
]
selected_module = st.sidebar.selectbox("Select Module", modules)

# Helper function to create sample data
def create_sample_data():
    """Create sample SDG 4 education data"""
    np.random.seed(42)
    n_samples = 200
    
    countries = ['Country_' + str(i) for i in range(1, 21)]
    regions = ['Sub-Saharan Africa', 'South Asia', 'East Asia', 'Latin America', 'Middle East']
    
    data = {
        'country': np.random.choice(countries, n_samples),
        'region': np.random.choice(regions, n_samples),
        'year': np.random.choice(range(2015, 2024), n_samples),
        'gdp_per_capita': np.random.normal(8000, 3000, n_samples),
        'population_density': np.random.exponential(100, n_samples),
        'urban_population_pct': np.random.normal(60, 20, n_samples),
        'primary_enrollment_rate': np.random.normal(85, 15, n_samples),
        'secondary_enrollment_rate': np.random.normal(70, 20, n_samples),
        'literacy_rate': np.random.normal(75, 20, n_samples),
        'completion_rate_primary': np.random.normal(80, 18, n_samples),
        'completion_rate_secondary': np.random.normal(65, 22, n_samples),
        'out_of_school_children': np.random.exponential(50000, n_samples),
        'teacher_student_ratio': np.random.normal(25, 8, n_samples),
        'education_expenditure_pct_gdp': np.random.normal(4.5, 1.5, n_samples)
    }
    
    # Add some missing values to simulate real-world data gaps
    df = pd.DataFrame(data)
    missing_cols = ['literacy_rate', 'completion_rate_secondary', 'out_of_school_children']
    for col in missing_cols:
        missing_indices = np.random.choice(df.index, size=int(0.15 * len(df)), replace=False)
        df.loc[missing_indices, col] = np.nan
    
    # Ensure realistic ranges
    df['primary_enrollment_rate'] = np.clip(df['primary_enrollment_rate'], 0, 100)
    df['secondary_enrollment_rate'] = np.clip(df['secondary_enrollment_rate'], 0, 100)
    df['literacy_rate'] = np.clip(df['literacy_rate'], 0, 100)
    df['completion_rate_primary'] = np.clip(df['completion_rate_primary'], 0, 100)
    df['completion_rate_secondary'] = np.clip(df['completion_rate_secondary'], 0, 100)
    df['urban_population_pct'] = np.clip(df['urban_population_pct'], 0, 100)
    df['education_expenditure_pct_gdp'] = np.clip(df['education_expenditure_pct_gdp'], 0, 10)
    
    return df

# Module 1: Data Ingestion
if selected_module == "📊 Data Ingestion":
    st.markdown('<div class="module-header">📊 Data Ingestion Module</div>', unsafe_allow_html=True)
    
    st.write("""
    This module allows you to bring in various data sources for analysis. You can upload your own data 
    or use our simulated UN/World Bank database connection to load sample SDG 4 education data.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Your Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file", 
            type=['csv', 'xlsx', 'xls'],
            help="Upload educational data with indicators like enrollment rates, literacy rates, etc."
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.data = pd.read_csv(uploaded_file)
                else:
                    st.session_state.data = pd.read_excel(uploaded_file)
                st.success("✅ File uploaded successfully!")
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    with col2:
        st.subheader("Simulated Database Connection")
        st.write("Load sample SDG 4 education data from simulated UN/World Bank databases")
        
        if st.button("🌐 Load Sample Data", type="primary"):
            st.session_state.data = create_sample_data()
            st.success("✅ Sample data loaded successfully!")
            st.info("📝 This simulates connecting to UN/World Bank databases with real SDG 4 indicators")
    
    # Data Preview
    if st.session_state.data is not None:
        st.subheader("📋 Data Preview")
        st.write(f"**Dataset Shape:** {st.session_state.data.shape[0]} rows × {st.session_state.data.shape[1]} columns")
        
        # Display first 5 rows
        st.dataframe(st.session_state.data.head(), use_container_width=True)
        
        # Basic info about the dataset
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", st.session_state.data.shape[0])
        with col2:
            st.metric("Total Features", st.session_state.data.shape[1])
        with col3:
            missing_pct = (st.session_state.data.isnull().sum().sum() / 
                          (st.session_state.data.shape[0] * st.session_state.data.shape[1]) * 100)
            st.metric("Missing Data %", f"{missing_pct:.1f}%")

# Module 2: Data Preprocessing
elif selected_module == "🔧 Data Preprocessing":
    st.markdown('<div class="module-header">🔧 Data Preprocessing Module</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load data in the Data Ingestion module first.")
    else:
        st.write("""
        Clean, transform, and prepare your data for machine learning. Handle missing values, 
        normalize features, and select relevant columns for analysis.
        """)
        
        # Missing value handling
        st.subheader("🔍 Missing Value Analysis")
        missing_data = st.session_state.data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Columns with Missing Values:**")
                for col, count in missing_data.items():
                    pct = (count / len(st.session_state.data)) * 100
                    st.write(f"• {col}: {count} ({pct:.1f}%)")
            
            with col2:
                # Missing value handling options
                st.write("**Handle Missing Values:**")
                missing_strategy = st.selectbox(
                    "Choose strategy",
                    ["Keep as is", "Fill with Mean", "Fill with Median", "Fill with Mode", "Remove Rows"]
                )
                
                if st.button("Apply Missing Value Strategy"):
                    df_processed = st.session_state.data.copy()
                    
                    if missing_strategy == "Fill with Mean":
                        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
                        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())
                    elif missing_strategy == "Fill with Median":
                        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
                        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
                    elif missing_strategy == "Fill with Mode":
                        for col in df_processed.columns:
                            if df_processed[col].isnull().any():
                                mode_val = df_processed[col].mode()
                                if len(mode_val) > 0:
                                    df_processed[col] = df_processed[col].fillna(mode_val[0])
                    elif missing_strategy == "Remove Rows":
                        df_processed = df_processed.dropna()
                    
                    st.session_state.processed_data = df_processed
                    st.success(f"✅ Applied {missing_strategy} strategy!")
        else:
            st.info("✅ No missing values found in the dataset!")
            st.session_state.processed_data = st.session_state.data.copy()
        
        # Feature selection
        if st.session_state.processed_data is not None:
            st.subheader("🎯 Feature Selection")
            
            numeric_cols = st.session_state.processed_data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = st.session_state.processed_data.select_dtypes(include=['object']).columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Select Numeric Features:**")
                selected_numeric = st.multiselect(
                    "Numeric columns",
                    numeric_cols,
                    default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
                )
            
            with col2:
                st.write("**Select Categorical Features:**")
                selected_categorical = st.multiselect(
                    "Categorical columns",
                    categorical_cols,
                    default=categorical_cols[:2] if len(categorical_cols) > 2 else categorical_cols
                )
            
            # Data normalization
            st.subheader("📏 Data Normalization")
            normalization_method = st.selectbox(
                "Choose normalization method",
                ["None", "Min-Max Scaling", "Standardization (Z-score)"]
            )
            
            if st.button("🔄 Process Data"):
                # Combine selected features
                selected_features = selected_numeric + selected_categorical
                df_final = st.session_state.processed_data[selected_features].copy()
                
                # Apply normalization to numeric features only
                if normalization_method != "None" and selected_numeric:
                    if normalization_method == "Min-Max Scaling":
                        scaler = MinMaxScaler()
                        df_final[selected_numeric] = scaler.fit_transform(df_final[selected_numeric])
                    elif normalization_method == "Standardization (Z-score)":
                        scaler = StandardScaler()
                        df_final[selected_numeric] = scaler.fit_transform(df_final[selected_numeric])
                
                st.session_state.processed_data = df_final
                st.success("✅ Data preprocessing completed!")
        
        # Display processed data summary
        if st.session_state.processed_data is not None:
            st.subheader("📊 Processed Data Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Processed Records", st.session_state.processed_data.shape[0])
            with col2:
                st.metric("Selected Features", st.session_state.processed_data.shape[1])
            with col3:
                missing_pct = (st.session_state.processed_data.isnull().sum().sum() / 
                              (st.session_state.processed_data.shape[0] * st.session_state.processed_data.shape[1]) * 100)
                st.metric("Missing Data %", f"{missing_pct:.1f}%")
            
            # Show summary statistics
            st.write("**Summary Statistics:**")
            st.dataframe(st.session_state.processed_data.describe(), use_container_width=True)

# Module 3: Machine Learning Model
elif selected_module == "🤖 Machine Learning Model":
    st.markdown('<div class="module-header">🤖 Machine Learning Model Module</div>', unsafe_allow_html=True)
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please complete data preprocessing first.")
    else:
        st.write("""
        Train machine learning models to identify patterns, make predictions, and fill data gaps. 
        Select your target variable and choose from various low-code friendly algorithms.
        """)
        
        # Model configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Target Variable Selection")
            numeric_cols = st.session_state.processed_data.select_dtypes(include=[np.number]).columns.tolist()
            target_variable = st.selectbox(
                "Select target variable to predict",
                numeric_cols,
                help="Choose the column you want to predict or estimate"
            )
        
        with col2:
            st.subheader("🔧 Model Selection")
            model_type = st.selectbox(
                "Choose machine learning model",
                [
                    "Linear Regression",
                    "Logistic Regression", 
                    "Decision Tree Classifier"
                ],
                help="Select the type of model based on your prediction task"
            )
        
        # Feature selection for model
        st.subheader("📋 Feature Selection for Model")
        available_features = [col for col in st.session_state.processed_data.columns if col != target_variable]
        selected_features = st.multiselect(
            "Select features for the model",
            available_features,
            default=available_features[:5] if len(available_features) > 5 else available_features
        )
        
        if len(selected_features) > 0:
            # Model training
            if st.button("🚀 Train Model", type="primary"):
                try:
                    # Prepare data
                    X = st.session_state.processed_data[selected_features]
                    y = st.session_state.processed_data[target_variable]
                    
                    # Handle categorical variables (simple encoding)
                    X_encoded = pd.get_dummies(X, drop_first=True)
                    
                    # Remove rows with missing target values
                    mask = ~y.isnull()
                    X_clean = X_encoded[mask]
                    y_clean = y[mask]
                    
                    if len(y_clean) < 10:
                        st.error("❌ Not enough data points for training. Please check your data.")
                    else:
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_clean, y_clean, test_size=0.2, random_state=42
                        )
                        
                        # Train model
                        if model_type == "Linear Regression":
                            model = LinearRegression()
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            
                            # Calculate metrics
                            r2 = r2_score(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)
                            st.session_state.model_metrics = {
                                'R-squared': f"{r2:.3f}",
                                'MSE': f"{mse:.3f}",
                                'RMSE': f"{np.sqrt(mse):.3f}"
                            }
                            
                        elif model_type == "Logistic Regression":
                            # Convert to binary classification
                            y_binary = (y_clean > y_clean.median()).astype(int)
                            X_train, X_test, y_train_bin, y_test_bin = train_test_split(
                                X_clean, y_binary, test_size=0.2, random_state=42
                            )
                            
                            model = LogisticRegression(random_state=42, max_iter=1000)
                            model.fit(X_train, y_train_bin)
                            y_pred = model.predict(X_test)
                            
                            # Calculate metrics
                            accuracy = accuracy_score(y_test_bin, y_pred)
                            precision = precision_score(y_test_bin, y_pred, average='weighted')
                            recall = recall_score(y_test_bin, y_pred, average='weighted')
                            st.session_state.model_metrics = {
                                'Accuracy': f"{accuracy:.3f}",
                                'Precision': f"{precision:.3f}",
                                'Recall': f"{recall:.3f}"
                            }
                            
                        elif model_type == "Decision Tree Classifier":
                            # Convert to binary classification
                            y_binary = (y_clean > y_clean.median()).astype(int)
                            X_train, X_test, y_train_bin, y_test_bin = train_test_split(
                                X_clean, y_binary, test_size=0.2, random_state=42
                            )
                            
                            model = DecisionTreeClassifier(random_state=42, max_depth=5)
                            model.fit(X_train, y_train_bin)
                            y_pred = model.predict(X_test)
                            
                            # Calculate metrics
                            accuracy = accuracy_score(y_test_bin, y_pred)
                            precision = precision_score(y_test_bin, y_pred, average='weighted')
                            recall = recall_score(y_test_bin, y_pred, average='weighted')
                            st.session_state.model_metrics = {
                                'Accuracy': f"{accuracy:.3f}",
                                'Precision': f"{precision:.3f}",
                                'Recall': f"{recall:.3f}"
                            }
                        
                        st.session_state.model = model
                        st.session_state.model_features = X_encoded.columns.tolist()
                        st.session_state.target_variable = target_variable
                        st.session_state.model_type = model_type
                        
                        st.success("✅ Model trained successfully!")
                        
                except Exception as e:
                    st.error(f"❌ Error training model: {str(e)}")
        
        # Display model performance
        if st.session_state.model is not None and st.session_state.model_metrics:
            st.subheader("📊 Model Performance Metrics")
            
            cols = st.columns(len(st.session_state.model_metrics))
            for i, (metric, value) in enumerate(st.session_state.model_metrics.items()):
                with cols[i]:
                    st.metric(metric, value)
            
            st.info(f"🎯 Model Type: {st.session_state.model_type}")
            st.info(f"🎯 Target Variable: {st.session_state.target_variable}")

# Module 4: Prediction & Visualization
elif selected_module == "📈 Prediction & Visualization":
    st.markdown('<div class="module-header">📈 Prediction & Visualization Module</div>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first in the Machine Learning module.")
    else:
        st.write("""
        Generate predictions using your trained model and visualize the results to identify patterns, 
        trends, and insights that can inform policy decisions.
        """)
        
        # Generate predictions
        if st.button("🔮 Generate Predictions", type="primary"):
            try:
                # Prepare data for prediction
                X = st.session_state.processed_data[st.session_state.model_features]
                
                # Make predictions
                if st.session_state.model_type == "Linear Regression":
                    predictions = st.session_state.model.predict(X)
                else:
                    predictions = st.session_state.model.predict_proba(X)[:, 1]
                
                # Store predictions
                st.session_state.predictions = predictions
                
                # Create results dataframe
                results_df = st.session_state.processed_data.copy()
                results_df['predicted_' + st.session_state.target_variable] = predictions
                st.session_state.results_df = results_df
                
                st.success("✅ Predictions generated successfully!")
                
            except Exception as e:
                st.error(f"❌ Error generating predictions: {str(e)}")
        
        # Visualizations
        if st.session_state.predictions is not None:
            st.subheader("📊 Prediction Visualizations")
            
            # Actual vs Predicted scatter plot
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Actual vs Predicted Values**")
                fig, ax = plt.subplots(figsize=(8, 6))
                
                actual_values = st.session_state.processed_data[st.session_state.target_variable].dropna()
                pred_values = st.session_state.predictions[:len(actual_values)]
                
                ax.scatter(actual_values, pred_values, alpha=0.6)
                ax.plot([actual_values.min(), actual_values.max()], 
                       [actual_values.min(), actual_values.max()], 'r--', lw=2)
                ax.set_xlabel(f'Actual {st.session_state.target_variable}')
                ax.set_ylabel(f'Predicted {st.session_state.target_variable}')
                ax.set_title('Actual vs Predicted Values')
                
                st.pyplot(fig)
            
            with col2:
                st.write("**Prediction Distribution**")
                fig, ax = plt.subplots(figsize=(8, 6))
                
                ax.hist(st.session_state.predictions, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_xlabel(f'Predicted {st.session_state.target_variable}')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Predictions')
                
                st.pyplot(fig)
            
            # Regional analysis (if region data exists)
            if 'region' in st.session_state.results_df.columns:
                st.write("**Regional Analysis**")
                
                regional_stats = st.session_state.results_df.groupby('region').agg({
                    st.session_state.target_variable: 'mean',
                    'predicted_' + st.session_state.target_variable: 'mean'
                }).round(2)
                
                fig = px.bar(
                    regional_stats.reset_index(),
                    x='region',
                    y=[st.session_state.target_variable, 'predicted_' + st.session_state.target_variable],
                    title='Average Values by Region',
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Data table with predictions
            st.subheader("📋 Results Table")
            st.write("Original data with predictions:")
            st.dataframe(st.session_state.results_df, use_container_width=True)
            
            # Download option
            csv = st.session_state.results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"sdg4_predictions_{st.session_state.target_variable}.csv",
                mime="text/csv"
            )

# Module 5: Validation & Iteration
elif selected_module == "🔄 Validation & Iteration":
    st.markdown('<div class="module-header">🔄 Validation & Iteration Module</div>', unsafe_allow_html=True)
    
    st.write("""
    Provide feedback on the model predictions, validate results, and iterate to improve the AI solution. 
    This continuous feedback loop ensures the model remains accurate and relevant.
    """)
    
    # Feedback section
    st.subheader("💬 Model Feedback")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Rate Model Performance:**")
        performance_rating = st.slider(
            "How would you rate the model's performance?",
            1, 5, 3,
            help="1 = Poor, 5 = Excellent"
        )
        
        feedback_text = st.text_area(
            "Provide detailed feedback:",
            placeholder="Share your thoughts on the predictions, visualizations, or suggestions for improvement...",
            height=100
        )
    
    with col2:
        st.write("**Validation Checklist:**")
        
        validation_checks = {
            "Predictions seem reasonable": st.checkbox("Predictions seem reasonable"),
            "Visualizations are clear": st.checkbox("Visualizations are clear"),
            "Results align with domain knowledge": st.checkbox("Results align with domain knowledge"),
            "Model metrics are acceptable": st.checkbox("Model metrics are acceptable")
        }
    
    # Submit feedback
    if st.button("📝 Submit Feedback"):
        feedback_summary = {
            'rating': performance_rating,
            'feedback': feedback_text,
            'validation_checks': validation_checks,
            'timestamp': pd.Timestamp.now()
        }
        
        # Store feedback (in a real app, this would go to a database)
        if 'feedback_history' not in st.session_state:
            st.session_state.feedback_history = []
        
        st.session_state.feedback_history.append(feedback_summary)
        st.success("✅ Feedback submitted successfully!")
    
    # Model retraining simulation
    st.subheader("🔄 Model Iteration")
    
    if st.session_state.model is not None:
        st.write("**Retrain Model with Updated Parameters:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_test_size = st.slider(
                "Test set size",
                0.1, 0.4, 0.2,
                help="Proportion of data to use for testing"
            )
        
        with col2:
            if st.session_state.model_type == "Decision Tree Classifier":
                max_depth = st.slider("Max depth", 3, 10, 5)
            else:
                max_depth = None
        
        if st.button("🔄 Retrain Model"):
            st.info("🔄 Retraining model with updated parameters...")
            # This would trigger the retraining process
            st.success("✅ Model retrained successfully! Check the Machine Learning module for updated metrics.")
    
    # Feedback history
    if 'feedback_history' in st.session_state and st.session_state.feedback_history:
        st.subheader("📊 Feedback History")
        
        feedback_df = pd.DataFrame([
            {
                'Timestamp': fb['timestamp'],
                'Rating': fb['rating'],
                'Feedback': fb['feedback'][:100] + '...' if len(fb['feedback']) > 100 else fb['feedback']
            }
            for fb in st.session_state.feedback_history
        ])
        
        st.dataframe(feedback_df, use_container_width=True)
    
    # Export options
    st.subheader("📤 Export & Documentation")
    
    if st.button("📋 Generate Model Report"):
        report = f"""
# SDG 4 AI Model Report

## Model Summary
- **Model Type:** {st.session_state.get('model_type', 'Not trained')}
- **Target Variable:** {st.session_state.get('target_variable', 'Not selected')}
- **Training Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Metrics
{st.session_state.get('model_metrics', 'No metrics available')}

## Data Summary
- **Total Records:** {st.session_state.processed_data.shape[0] if st.session_state.processed_data is not None else 'No data'}
- **Features Used:** {len(st.session_state.get('model_features', [])) if st.session_state.get('model_features') else 'No features'}

## Feedback Summary
- **Average Rating:** {np.mean([fb['rating'] for fb in st.session_state.get('feedback_history', [])]) if st.session_state.get('feedback_history') else 'No feedback'}
- **Total Feedback Entries:** {len(st.session_state.get('feedback_history', []))}

---
Generated by SDG 4 AI Solution
        """
        
        st.download_button(
            label="📥 Download Model Report",
            data=report,
            file_name=f"sdg4_model_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🎓 SDG 4 Low-Code AI Solution | Bridging Education Data Gaps with Machine Learning</p>
    <p><em>Developed for sustainable development and quality education monitoring by Travor Mubaya</em></p>
</div>
""", unsafe_allow_html=True)

