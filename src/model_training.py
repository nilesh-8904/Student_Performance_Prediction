import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.model_scores = {}
        self.processed_data_path = 'processed_data/'
        
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple regression models and compare their performance"""
        print("=" * 50)
        print("MODEL TRAINING AND EVALUATION")
        print("=" * 50)
        
        # Define models with optimized parameters
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'SVR': SVR(kernel='rbf', C=1.0)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            # Store results
            self.models[name] = model
            self.model_scores[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'CV_R2_Mean': cv_scores.mean(),
                'CV_R2_Std': cv_scores.std()
            }
            
            print(f"R2 Score: {r2:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"CV R2 Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Find the best model
        self.find_best_model()
        
        # Visualize model comparison
        self.visualize_model_comparison()
        
        # Feature importance for the best model (if applicable)
        if hasattr(self.best_model, 'feature_importances_'):
            self.plot_feature_importance()
            
        return self.best_model
            
    def find_best_model(self):
        """Find the best performing model based on R2 score"""
        best_r2 = -np.inf
        for name, scores in self.model_scores.items():
            if scores['R2'] > best_r2:
                best_r2 = scores['R2']
                self.best_model = self.models[name]
                self.best_model_name = name
        
        print(f"\nBest Model: {self.best_model_name}")
        print(f"Best R2 Score: {best_r2:.4f}")
        
    def visualize_model_comparison(self):
        """Create visualization for model performance comparison"""
        # Prepare data for visualization
        metrics = ['R2', 'RMSE', 'MAE']
        model_names = list(self.model_scores.keys())
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, metric in enumerate(metrics):
            values = [self.model_scores[name][metric] for name in model_names]
            
            if metric == 'R2':
                colors = ['green' if name == self.best_model_name else 'lightblue' 
                         for name in model_names]
            else:
                colors = ['red' if name == self.best_model_name else 'lightblue' 
                         for name in model_names]
            
            bars = axes[idx].bar(model_names, values, color=colors)
            axes[idx].set_title(f'{metric} Comparison')
            axes[idx].set_ylabel(metric)
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                              f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.processed_data_path}/model_comparison.png')
        plt.show()
        
        # Create detailed comparison table
        comparison_df = pd.DataFrame(self.model_scores).T
        print("\nDetailed Model Comparison:")
        print(comparison_df)
        
        # Save comparison
        comparison_df.to_csv(f'{self.processed_data_path}/model_comparison.csv')
        
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        if not hasattr(self.best_model, 'feature_importances_'):
            print("Best model doesn't have feature importance attribute")
            return
            
        # Load feature columns
        try:
            feature_columns = joblib.load(f'{self.processed_data_path}/feature_columns.pkl')
        except:
            feature_columns = [
                'Study Hours', 'Attendance', 'Assignment Score',
                'Internal Marks', 'Previous GPA', 'Extracurricular',
                'Gender', 'Parental Education'
            ][:len(self.best_model.feature_importances_)]
        
        # Create DataFrame for feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': self.best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature')
        plt.title(f'Feature Importance - {self.best_model_name}')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig(f'{self.processed_data_path}/feature_importance.png')
        plt.show()
        
        print("\nFeature Importance:")
        print(importance_df)
        
    def save_model(self, filepath='processed_data/best_model.pkl'):
        """Save the best model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.best_model, filepath)
        print(f"\nBest model saved to {filepath}")
        
    def load_model(self, filepath='processed_data/best_model.pkl'):
        """Load a saved model"""
        self.best_model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        
    def predict(self, features):
        """Make predictions using the best model"""
        if self.best_model is None:
            raise ValueError("No model trained or loaded")
        
        return self.best_model.predict(features)

# Usage example
if __name__ == "__main__":
    # Note: This assumes you've already run the preprocessing script
    from data_preprocessing import DataPreprocessor
    
    # Initialize preprocessor and prepare data
    preprocessor = DataPreprocessor('data/student_performance.csv')
    preprocessor.load_data()
    preprocessor.clean_data()
    X_train, X_test, y_train, y_test = preprocessor.prepare_features()
    
    # Initialize and train models
    trainer = ModelTrainer()
    trainer.train_models(X_train, X_test, y_train, y_test)
    
    # Save the best model
    trainer.save_model()