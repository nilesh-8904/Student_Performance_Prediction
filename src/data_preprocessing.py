import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_columns = [
            'study_hours_per_week', 'attendance_percentage', 'assignment_score',
            'internal_marks', 'previous_gpa', 'extracurricular_activities',
            'gender', 'parental_education'
        ]
        self.processed_data_path = 'processed_data/'
        
        # Create processed data directory
        os.makedirs(self.processed_data_path, exist_ok=True)
        
    def load_data(self):
        """Load and display basic information about the dataset"""
        self.data = pd.read_csv(self.filepath)
        print("=" * 50)
        print("DATASET INFORMATION")
        print("=" * 50)
        print(f"Shape: {self.data.shape}")
        print(f"\nColumns: {list(self.data.columns)}")
        print("\nFirst 5 rows:")
        print(self.data.head())
        print("\nData Types:")
        print(self.data.dtypes)
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        
    def generate_sample_data(self):
        """Generate sample student performance data if file doesn't exist"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'student_id': range(1, n_samples + 1),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'age': np.random.randint(18, 25, n_samples),
            'study_hours_per_week': np.random.uniform(5, 40, n_samples),
            'attendance_percentage': np.random.uniform(60, 100, n_samples),
            'assignment_score': np.random.uniform(50, 100, n_samples),
            'internal_marks': np.random.uniform(40, 95, n_samples),
            'previous_gpa': np.random.uniform(2.0, 4.0, n_samples),
            'extracurricular_activities': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'parental_education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Calculate final score based on weighted combination of factors
        df['final_score'] = (
            0.3 * df['study_hours_per_week'] / 40 * 100 +
            0.25 * df['attendance_percentage'] +
            0.2 * df['assignment_score'] +
            0.15 * df['internal_marks'] +
            0.1 * df['previous_gpa'] / 4 * 100
        ) + np.random.normal(0, 5, n_samples)
        
        # Ensure scores are within 0-100 range
        df['final_score'] = df['final_score'].clip(0, 100)
        
        # Categorize performance
        df['performance_category'] = pd.cut(
            df['final_score'],
            bins=[0, 60, 75, 90, 100],
            labels=['Poor', 'Average', 'Good', 'Excellent']
        )
        
        self.data = df
        df.to_csv(self.filepath, index=False)
        print(f"Sample dataset generated and saved to {self.filepath}")
        
    def clean_data(self):
        """Clean and preprocess the data"""
        print("\n" + "=" * 50)
        print("DATA CLEANING")
        print("=" * 50)
        
        # Handle missing values
        if self.data.isnull().sum().sum() > 0:
            print("\nHandling missing values...")
            # Fill numeric columns with median
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if self.data[col].isnull().sum() > 0:
                    median_val = self.data[col].median()
                    self.data[col].fillna(median_val, inplace=True)
                    print(f"Filled {col} with median: {median_val}")
            
            # Fill categorical columns with mode
            categorical_columns = self.data.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if self.data[col].isnull().sum() > 0:
                    mode_val = self.data[col].mode()[0]
                    self.data[col].fillna(mode_val, inplace=True)
                    print(f"Filled {col} with mode: {mode_val}")
        
        # Remove duplicates
        initial_rows = len(self.data)
        self.data.drop_duplicates(inplace=True)
        if len(self.data) < initial_rows:
            print(f"Removed {initial_rows - len(self.data)} duplicate rows")
        
        # Encode categorical variables and save encoders
        categorical_mappings = {
            'gender': {'Male': 0, 'Female': 1},
            'parental_education': {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
        }
        
        for col, mapping in categorical_mappings.items():
            if col in self.data.columns:
                self.data[col + '_encoded'] = self.data[col].map(mapping)
                self.encoders[col] = mapping
        
        print("\nData cleaning completed!")
        print(f"Categorical encoders created: {list(self.encoders.keys())}")
        
    def perform_eda(self):
        """Perform exploratory data analysis"""
        print("\n" + "=" * 50)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 50)
        
        # Statistical summary
        print("\nStatistical Summary:")
        print(self.data.describe())
        
        # Correlation matrix
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.data[numeric_cols].corr()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('processed_data/correlation_matrix.png')
        plt.show()
        
        # Distribution of final scores
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data['final_score'], bins=30, kde=True)
        plt.title('Distribution of Final Scores')
        plt.xlabel('Final Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig('processed_data/score_distribution.png')
        plt.show()
        
        # Performance category distribution
        if 'performance_category' in self.data.columns:
            plt.figure(figsize=(8, 6))
            self.data['performance_category'].value_counts().plot(kind='pie', autopct='%1.1f%%')
            plt.title('Performance Category Distribution')
            plt.ylabel('')
            plt.savefig('processed_data/performance_distribution.png')
            plt.show()
        
        # Study hours vs Final Score
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.data, x='study_hours_per_week', y='final_score', 
                       hue='gender', alpha=0.6)
        plt.title('Study Hours vs Final Score')
        plt.xlabel('Study Hours per Week')
        plt.ylabel('Final Score')
        plt.grid(True, alpha=0.3)
        plt.savefig('processed_data/study_hours_vs_score.png')
        plt.show()
        
    def prepare_features(self):
        """Prepare features for machine learning"""
        print("\n" + "=" * 50)
        print("FEATURE PREPARATION")
        print("=" * 50)
        
        # Create feature mapping for consistency
        feature_mapping = {
            'study_hours_per_week': 'study_hours_per_week',
            'attendance_percentage': 'attendance_percentage',
            'assignment_score': 'assignment_score',
            'internal_marks': 'internal_marks',
            'previous_gpa': 'previous_gpa',
            'extracurricular_activities': 'extracurricular_activities',
            'gender': 'gender_encoded',
            'parental_education': 'parental_education_encoded'
        }
        
        # Select and rename features
        features_to_use = []
        for feature in self.feature_columns:
            mapped_feature = feature_mapping.get(feature, feature)
            if mapped_feature in self.data.columns:
                features_to_use.append(mapped_feature)
        
        self.X = self.data[features_to_use]
        self.y = self.data['final_score']
        
        print(f"Features selected: {features_to_use}")
        print(f"Feature matrix shape: {self.X.shape}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        
        # Save the scaler and encoders for GUI
        joblib.dump(self.scaler, 'processed_data/scaler.pkl')
        joblib.dump(self.encoders, 'processed_data/encoders.pkl')
        joblib.dump(features_to_use, 'processed_data/feature_columns.pkl')
        
        print("\nPreprocessing artifacts saved to 'processed_data/' directory")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test

# Usage example
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor('data/student_performance.csv')
    
    # Try to load existing data, generate if not exists
    try:
        preprocessor.load_data()
    except FileNotFoundError:
        print("Data file not found. Generating sample data...")
        preprocessor.generate_sample_data()
        preprocessor.load_data()
    
    # Clean and analyze data
    preprocessor.clean_data()
    preprocessor.perform_eda()
    
    # Prepare features for modeling
    X_train, X_test, y_train, y_test = preprocessor.prepare_features()