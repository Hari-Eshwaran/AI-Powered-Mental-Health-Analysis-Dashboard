import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class MentalHealthAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Load all datasets"""
        try:
            print("Current working directory:", os.getcwd())
            print("Checking if files exist:")
            for file in ['mental_health_diagnosis_treatment_.csv', 'students_mental_health_survey.csv', 
                        'Daylio_Abid.csv', 'survey.csv', 'intents.json']:
                print(f"{file} exists:", os.path.exists(file))
            
            # Load mental health diagnosis data
            print("\nLoading mental health diagnosis data...")
            self.diagnosis_df = pd.read_csv('mental_health_diagnosis_treatment_.csv', encoding='utf-8')
            
            # Load student survey data
            print("Loading student survey data...")
            self.student_df = pd.read_csv('students_mental_health_survey.csv', encoding='utf-8')
            
            # Load mood tracking data
            print("Loading mood tracking data...")
            self.mood_df = pd.read_csv('Daylio_Abid.csv', encoding='utf-8')
            
            # Load general survey data
            print("Loading general survey data...")
            self.survey_df = pd.read_csv('survey.csv', encoding='utf-8')
            
            # Load intents for NLP classification with UTF-8 encoding
            print("Loading intents data...")
            with open('intents.json', 'r', encoding='utf-8') as f:
                self.intents = json.load(f)
                
        except UnicodeDecodeError:
            print("UTF-8 encoding failed, trying with latin-1 encoding...")
            self.diagnosis_df = pd.read_csv('mental_health_diagnosis_treatment_.csv', encoding='latin-1')
            self.student_df = pd.read_csv('students_mental_health_survey.csv', encoding='latin-1')
            self.mood_df = pd.read_csv('Daylio_Abid.csv', encoding='latin-1')
            self.survey_df = pd.read_csv('survey.csv', encoding='latin-1')
            
            with open('intents.json', 'r', encoding='latin-1') as f:
                self.intents = json.load(f)
        except FileNotFoundError as e:
            print(f"Error: Could not find file - {str(e)}")
            raise
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in intents.json - {str(e)}")
            raise
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            raise
    
    def clean_data(self):
        """Clean and preprocess all datasets"""
        # Clean diagnosis data
        self.diagnosis_df = self.diagnosis_df.dropna()
        
        # Clean student survey data
        self.student_df = self.student_df.dropna()
        
        # Clean mood tracking data
        self.mood_df = self.mood_df.dropna()
        
        # Clean survey data
        self.survey_df = self.survey_df.dropna()
    
    def perform_sentiment_analysis(self, text):
        """Perform sentiment analysis on text data"""
        return TextBlob(text).sentiment.polarity
    
    def prepare_nlp_data(self):
        """Prepare data for NLP classification"""
        patterns = []
        labels = []
        
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                patterns.append(pattern)
                labels.append(intent['tag'])
        
        return patterns, labels
    
    def train_mood_prediction_model(self):
        """Train model for mood prediction"""
        # Prepare features and target
        X = self.mood_df[['sub_mood', 'activities']]
        y = self.mood_df['mood']
        
        # Convert categorical variables to numerical
        X['sub_mood'] = self.label_encoder.fit_transform(X['sub_mood'])
        
        # Split activities into separate columns
        activities = X['activities'].str.get_dummies('|')
        X = pd.concat([X['sub_mood'], activities], axis=1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        print("\nMood Prediction Model Performance:")
        print(classification_report(y_test, y_pred))
        
        # Print feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features for Mood Prediction:")
        print(feature_importance.head(10))
        
        return model
    
    def train_diagnosis_model(self):
        """Train model for mental health diagnosis prediction"""
        # Prepare features and target
        X = self.diagnosis_df.drop('Diagnosis', axis=1)
        y = self.diagnosis_df['Diagnosis']
        
        # Identify categorical columns
        categorical_cols = ['Gender', 'Medication', 'Therapy Type', 'Outcome', 'AI-Detected Emotional State']
        
        # Create a copy of features for encoding
        X_encoded = X.copy()
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in X_encoded.columns:
                X_encoded[col] = self.label_encoder.fit_transform(X_encoded[col])
        
        # Convert date columns to numerical features
        date_cols = ['Treatment Start Date']
        for col in date_cols:
            if col in X_encoded.columns:
                X_encoded[col] = pd.to_datetime(X_encoded[col]).astype(np.int64)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        print("\nDiagnosis Prediction Model Performance:")
        print(classification_report(y_test, y_pred))
        
        # Print feature importance
        feature_importance = pd.DataFrame({
            'feature': X_encoded.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features for Diagnosis Prediction:")
        print(feature_importance.head(10))
        
        return model
    
    def analyze_student_mental_health(self):
        """Analyze student mental health data"""
        # Create visualizations
        plt.figure(figsize=(12, 6))
        sns.countplot(data=self.student_df, x='Depression_Score')
        plt.title('Distribution of Depression Scores')
        plt.xlabel('Depression Score')
        plt.ylabel('Count')
        plt.savefig('depression_distribution.png')
        plt.close()
        
        # Calculate statistics
        depression_stats = self.student_df['Depression_Score'].value_counts().sort_index()
        print("\nDepression Score Statistics:")
        print(depression_stats)
        
        # Prepare data for correlation analysis
        df_corr = self.student_df.copy()
        
        # Define categorical columns and their mappings
        categorical_mappings = {
            'Sleep_Quality': {'Poor': 1, 'Average': 2, 'Good': 3},
            'Physical_Activity': {'Low': 1, 'Moderate': 2, 'High': 3},
            'Diet_Quality': {'Poor': 1, 'Average': 2, 'Good': 3},
            'Social_Support': {'Low': 1, 'Moderate': 2, 'High': 3}
        }
        
        # Convert categorical variables to numerical
        for col, mapping in categorical_mappings.items():
            df_corr[col] = df_corr[col].map(mapping)
        
        # Select numeric columns for correlation
        numeric_cols = ['Age', 'CGPA', 'Stress_Level', 'Anxiety_Score', 'Sleep_Quality', 
                       'Physical_Activity', 'Diet_Quality', 'Social_Support', 'Financial_Stress']
        
        # Calculate correlations
        correlations = df_corr[numeric_cols + ['Depression_Score']].corr()['Depression_Score'].sort_values(ascending=False)
        print("\nCorrelations with Depression Score:")
        print(correlations)
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_corr[numeric_cols + ['Depression_Score']].corr(), 
                   annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap of Mental Health Factors')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        print("Loading data...")
        self.load_data()
        
        print("\nCleaning data...")
        self.clean_data()
        
        print("\nTraining mood prediction model...")
        mood_model = self.train_mood_prediction_model()
        
        print("\nTraining diagnosis prediction model...")
        diagnosis_model = self.train_diagnosis_model()
        
        print("\nAnalyzing student mental health data...")
        self.analyze_student_mental_health()
        
        print("\nAnalysis complete! Results have been saved.")

if __name__ == "__main__":
    analyzer = MentalHealthAnalyzer()
    analyzer.run_analysis() 