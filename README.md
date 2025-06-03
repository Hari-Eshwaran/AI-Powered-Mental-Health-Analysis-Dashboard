# AI-Powered Mental Health Analysis Dashboard

This project provides a comprehensive mental health analysis dashboard that combines machine learning, data visualization, and interactive features to analyze mental health patterns, mood tracking, and student mental health statistics.

## 🚀 Features

- **Interactive Dashboard**: Built with Streamlit for an intuitive user interface
- **Mood Analysis**: Track and analyze mood patterns over time
- **Diagnosis Analysis**: Analyze mental health diagnoses and treatment outcomes
- **Student Mental Health**: Comprehensive analysis of student mental health data
- **Correlation Analysis**: Identify relationships between various mental health factors
- **Machine Learning Models**: 
  - Mood prediction model
  - Mental health diagnosis prediction model
  - Sentiment analysis capabilities

## 📊 Data Sources

The project utilizes multiple datasets:
- Mental health diagnosis and treatment data
- Student mental health survey data
- Personal mood tracking data (Daylio)
- General survey data
- Intent classification data for NLP

## 🛠️ Technical Requirements

### Python Dependencies
```
pandas
numpy
scikit-learn
seaborn
matplotlib
textblob
nltk
streamlit
plotly
```

### Installation

1. Clone the repository:
```bash
git clone [https://github.com/Hari-Eshwaran/oneyes.git]
cd [oneyes]
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 🚀 Usage

1. Run the dashboard:
```bash
streamlit run mental_health_dashboard.py
```

2. Access the dashboard through your web browser at `http://localhost:8501`

## 📁 Project Structure

- `mental_health_dashboard.py`: Main Streamlit dashboard application
- `mental_health_analysis.py`: Core analysis and machine learning functionality
- `requirements.txt`: Project dependencies
- Data files:
  - `mental_health_diagnosis_treatment_.csv`
  - `students_mental_health_survey.csv`
  - `Daylio_Abid.csv`
  - `survey.csv`
  - `intents.json`

## 📊 Dashboard Sections

1. **Overview**
   - Key metrics and statistics
   - Diagnosis distribution
   - Student depression levels

2. **Mood Analysis**
   - Mood trends over time
   - Activity impact on mood
   - Mood distribution by activity

3. **Diagnosis Analysis**
   - Treatment effectiveness
   - Symptom severity distribution

4. **Student Mental Health**
   - Depression score distribution
   - Course-wise analysis
   - Mental health factors correlation

5. **Correlation Analysis**
   - Feature importance
   - Correlation heatmaps
   - Statistical insights

## 🤖 Machine Learning Models

### Mood Prediction Model
- Uses Random Forest Classifier
- Predicts mood based on activities and sub-mood
- Provides feature importance analysis

### Diagnosis Prediction Model
- Random Forest-based classification
- Handles categorical and numerical features
- Includes feature importance analysis

### Sentiment Analysis
- TextBlob-based sentiment analysis
- NLP capabilities for text processing

## 📈 Visualizations

The project includes various visualizations:
- Interactive Plotly charts
- Seaborn heatmaps
- Matplotlib plots
- Distribution plots
- Correlation matrices

## 🔒 Data Privacy

- All personal data is anonymized
- Sensitive information is handled with appropriate security measures
- Data processing follows privacy guidelines

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- [Harishwaran P - CTO, Hexpertify]

## 🙏 Acknowledgments

- Thanks to all contributors and data providers
- Special thanks to the open-source community for the amazing tools and libraries 
