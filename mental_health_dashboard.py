import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from mental_health_analysis import MentalHealthAnalyzer
import json

# Set page configuration
st.set_page_config(
    page_title="Mental Health Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .plot-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🧠 Mental Health Analysis Dashboard")
st.markdown("""
    This dashboard provides insights into mental health patterns, mood tracking, and student mental health statistics.
    Use the sidebar to navigate between different sections.
""")

# Initialize the analyzer
@st.cache_data
def load_data():
    analyzer = MentalHealthAnalyzer()
    analyzer.load_data()
    analyzer.clean_data()
    return analyzer

analyzer = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a section",
    ["Overview", "Mood Analysis", "Diagnosis Analysis", "Student Mental Health", "Correlation Analysis"]
)

# Overview page
if page == "Overview":
    st.header("📊 Overview")
    
    # Create three columns for key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Patients",
            value=len(analyzer.diagnosis_df),
            delta="Active Cases"
        )
    
    with col2:
        st.metric(
            label="Student Survey Responses",
            value=len(analyzer.student_df),
            delta="Total Responses"
        )
    
    with col3:
        st.metric(
            label="Mood Entries",
            value=len(analyzer.mood_df),
            delta="Total Entries"
        )
    
    # Create two columns for main charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Diagnosis Distribution")
        diagnosis_counts = analyzer.diagnosis_df['Diagnosis'].value_counts()
        fig = px.pie(
            values=diagnosis_counts.values,
            names=diagnosis_counts.index,
            title="Distribution of Mental Health Diagnoses"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Student Depression Levels")
        depression_counts = analyzer.student_df['Depression_Score'].value_counts().sort_index()
        fig = px.bar(
            x=depression_counts.index,
            y=depression_counts.values,
            title="Distribution of Depression Scores",
            labels={'x': 'Depression Score', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)

# Mood Analysis page
elif page == "Mood Analysis":
    st.header("😊 Mood Analysis")
    
    # Mood prediction model results
    st.subheader("Mood Prediction Model Performance")
    
    # Create a copy of the mood data for analysis
    mood_df = analyzer.mood_df.copy()
    
    # Convert sub_mood to numerical for analysis
    mood_df['sub_mood_encoded'] = analyzer.label_encoder.fit_transform(mood_df['sub_mood'])
    
    # Create mood mapping dictionary
    mood_mapping = {
        'Good': 3,
        'Average': 2,
        'Bad': 1
    }
    
    # Convert mood to numerical values
    mood_df['mood_numeric'] = mood_df['mood'].map(mood_mapping)
    
    # Create two columns for mood analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Mood distribution over time
        fig = px.line(
            mood_df,
            x='full_date',
            y='mood_numeric',
            title="Mood Trends Over Time",
            labels={'mood_numeric': 'Mood Level', 'full_date': 'Date'}
        )
        fig.update_layout(
            yaxis=dict(
                ticktext=['Bad', 'Average', 'Good'],
                tickvals=[1, 2, 3],
                tickmode='array'
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Activity impact on mood
        activities = mood_df['activities'].str.get_dummies('|')
        activity_mood = pd.concat([activities, mood_df['mood_numeric']], axis=1)
        activity_corr = activity_mood.corr()['mood_numeric'].sort_values(ascending=False)
        
        # Remove the mood_numeric correlation with itself
        activity_corr = activity_corr.drop('mood_numeric')
        
        fig = px.bar(
            x=activity_corr.index[:10],  # Top 10 activities
            y=activity_corr.values[:10],
            title="Top 10 Activities Impacting Mood",
            labels={'x': 'Activity', 'y': 'Correlation with Mood'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add mood distribution by activity
        st.subheader("Mood Distribution by Activity")
        selected_activity = st.selectbox(
            "Select Activity",
            options=activities.columns
        )
        
        activity_mood_data = mood_df[mood_df['activities'].str.contains(selected_activity)]
        mood_counts = activity_mood_data['mood'].value_counts()
        
        fig = px.pie(
            values=mood_counts.values,
            names=mood_counts.index,
            title=f"Mood Distribution for {selected_activity}"
        )
        st.plotly_chart(fig, use_container_width=True)

# Diagnosis Analysis page
elif page == "Diagnosis Analysis":
    st.header("🏥 Diagnosis Analysis")
    
    # Create two columns for diagnosis analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Treatment effectiveness
        treatment_outcomes = analyzer.diagnosis_df.groupby(['Therapy Type', 'Outcome']).size().unstack()
        fig = px.bar(
            treatment_outcomes,
            title="Treatment Outcomes by Therapy Type",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Symptom severity distribution
        fig = px.histogram(
            analyzer.diagnosis_df,
            x='Symptom Severity (1-10)',
            color='Diagnosis',
            title="Symptom Severity Distribution by Diagnosis",
            nbins=10
        )
        st.plotly_chart(fig, use_container_width=True)

# Student Mental Health page
elif page == "Student Mental Health":
    st.header("🎓 Student Mental Health Analysis")
    
    # Create filters
    st.sidebar.subheader("Filters")
    selected_course = st.sidebar.multiselect(
        "Select Course",
        options=analyzer.student_df['Course'].unique(),
        default=analyzer.student_df['Course'].unique()
    )
    
    # Filter data based on selection
    filtered_df = analyzer.student_df[analyzer.student_df['Course'].isin(selected_course)]
    
    # Create three columns for student analysis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Depression score distribution
        fig = px.histogram(
            filtered_df,
            x='Depression_Score',
            title="Depression Score Distribution",
            nbins=6
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Anxiety score distribution
        fig = px.histogram(
            filtered_df,
            x='Anxiety_Score',
            title="Anxiety Score Distribution",
            nbins=6
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Stress level distribution
        fig = px.histogram(
            filtered_df,
            x='Stress_Level',
            title="Stress Level Distribution",
            nbins=6
        )
        st.plotly_chart(fig, use_container_width=True)

# Correlation Analysis page
elif page == "Correlation Analysis":
    st.header("📈 Correlation Analysis")
    
    # Prepare data for correlation analysis
    df_corr = analyzer.student_df.copy()
    
    # Define categorical mappings
    categorical_mappings = {
        'Sleep_Quality': {'Poor': 1, 'Average': 2, 'Good': 3},
        'Physical_Activity': {'Low': 1, 'Moderate': 2, 'High': 3},
        'Diet_Quality': {'Poor': 1, 'Average': 2, 'Good': 3},
        'Social_Support': {'Low': 1, 'Moderate': 2, 'High': 3}
    }
    
    # Convert categorical variables
    for col, mapping in categorical_mappings.items():
        df_corr[col] = df_corr[col].map(mapping)
    
    # Select numeric columns
    numeric_cols = ['Age', 'CGPA', 'Stress_Level', 'Anxiety_Score', 'Sleep_Quality', 
                   'Physical_Activity', 'Diet_Quality', 'Social_Support', 'Financial_Stress']
    
    # Create correlation heatmap
    corr_matrix = df_corr[numeric_cols + ['Depression_Score']].corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Correlation Heatmap of Mental Health Factors",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show detailed correlations with depression
    st.subheader("Detailed Correlations with Depression Score")
    correlations = corr_matrix['Depression_Score'].sort_values(ascending=False)
    
    fig = px.bar(
        x=correlations.index,
        y=correlations.values,
        title="Correlation with Depression Score",
        labels={'x': 'Factor', 'y': 'Correlation Coefficient'}
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Mental Health Analysis Dashboard | Created with Streamlit</p>
    </div>
""", unsafe_allow_html=True) 