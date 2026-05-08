import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

class StudentPerformanceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Performance Prediction System")
        self.root.geometry("1200x800")
        
        # Load trained model and preprocessing artifacts
        self.load_artifacts()
        
        # Create main frames
        self.create_frames()
        
        # Create widgets
        self.create_input_widgets()
        self.create_prediction_widgets()
        self.create_visualization_widgets()
        
    def load_artifacts(self):
        """Load model, scaler, encoders, and feature columns"""
        try:
            self.model = joblib.load('processed_data/best_model.pkl')
            self.scaler = joblib.load('processed_data/scaler.pkl')
            self.encoders = joblib.load('processed_data/encoders.pkl')
            self.feature_columns = joblib.load('processed_data/feature_columns.pkl')
            print("All artifacts loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model artifacts: {str(e)}\nPlease run the training script first.")
            self.root.destroy()
            return
    
    def create_frames(self):
        """Create main frames for the GUI"""
        # Input Frame
        self.input_frame = ttk.LabelFrame(self.root, text="Student Information", padding=10)
        self.input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        # Prediction Frame
        self.prediction_frame = ttk.LabelFrame(self.root, text="Prediction Result", padding=10)
        self.prediction_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        # Visualization Frame
        self.viz_frame = ttk.LabelFrame(self.root, text="Data Visualization", padding=10)
        self.viz_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
    
    def create_input_widgets(self):
        """Create input fields for student data"""
        # Input fields
        self.inputs = {}
        
        # Study Hours
        ttk.Label(self.input_frame, text="Study Hours per Week:").grid(row=0, column=0, sticky="w", pady=5)
        self.inputs['study_hours'] = ttk.Scale(self.input_frame, from_=0, to=50, orient="horizontal", length=200)
        self.inputs['study_hours'].grid(row=0, column=1, pady=5)
        self.study_hours_label = ttk.Label(self.input_frame, text="25")
        self.study_hours_label.grid(row=0, column=2, pady=5)
        self.inputs['study_hours'].configure(command=lambda v: self.study_hours_label.config(text=f"{float(v):.1f}"))
        
        # Attendance
        ttk.Label(self.input_frame, text="Attendance Percentage:").grid(row=1, column=0, sticky="w", pady=5)
        self.inputs['attendance'] = ttk.Scale(self.input_frame, from_=0, to=100, orient="horizontal", length=200)
        self.inputs['attendance'].grid(row=1, column=1, pady=5)
        self.attendance_label = ttk.Label(self.input_frame, text="75")
        self.attendance_label.grid(row=1, column=2, pady=5)
        self.inputs['attendance'].configure(command=lambda v: self.attendance_label.config(text=f"{float(v):.1f}"))
        
        # Assignment Score
        ttk.Label(self.input_frame, text="Assignment Score:").grid(row=2, column=0, sticky="w", pady=5)
        self.inputs['assignment'] = ttk.Scale(self.input_frame, from_=0, to=100, orient="horizontal", length=200)
        self.inputs['assignment'].grid(row=2, column=1, pady=5)
        self.assignment_label = ttk.Label(self.input_frame, text="75")
        self.assignment_label.grid(row=2, column=2, pady=5)
        self.inputs['assignment'].configure(command=lambda v: self.assignment_label.config(text=f"{float(v):.1f}"))
        
        # Internal Marks
        ttk.Label(self.input_frame, text="Internal Marks:").grid(row=3, column=0, sticky="w", pady=5)
        self.inputs['internal'] = ttk.Scale(self.input_frame, from_=0, to=100, orient="horizontal", length=200)
        self.inputs['internal'].grid(row=3, column=1, pady=5)
        self.internal_label = ttk.Label(self.input_frame, text="75")
        self.internal_label.grid(row=3, column=2, pady=5)
        self.inputs['internal'].configure(command=lambda v: self.internal_label.config(text=f"{float(v):.1f}"))
        
        # Previous GPA
        ttk.Label(self.input_frame, text="Previous GPA:").grid(row=4, column=0, sticky="w", pady=5)
        self.inputs['gpa'] = ttk.Scale(self.input_frame, from_=0, to=4, orient="horizontal", length=200)
        self.inputs['gpa'].grid(row=4, column=1, pady=5)
        self.gpa_label = ttk.Label(self.input_frame, text="3.0")
        self.gpa_label.grid(row=4, column=2, pady=5)
        self.inputs['gpa'].configure(command=lambda v: self.gpa_label.config(text=f"{float(v):.2f}"))
        
        # Extracurricular Activities
        ttk.Label(self.input_frame, text="Extracurricular Activities:").grid(row=5, column=0, sticky="w", pady=5)
        self.inputs['extracurricular'] = ttk.Combobox(self.input_frame, values=["No", "Yes"], state="readonly", width=15)
        self.inputs['extracurricular'].grid(row=5, column=1, pady=5)
        self.inputs['extracurricular'].set("Yes")
        
        # Gender
        ttk.Label(self.input_frame, text="Gender:").grid(row=6, column=0, sticky="w", pady=5)
        self.inputs['gender'] = ttk.Combobox(self.input_frame, values=["Male", "Female"], state="readonly", width=15)
        self.inputs['gender'].grid(row=6, column=1, pady=5)
        self.inputs['gender'].set("Male")
        
        # Parental Education
        ttk.Label(self.input_frame, text="Parental Education:").grid(row=7, column=0, sticky="w", pady=5)
        self.inputs['parental_edu'] = ttk.Combobox(self.input_frame, 
                                                  values=["High School", "Bachelor", "Master", "PhD"], 
                                                  state="readonly", width=15)
        self.inputs['parental_edu'].grid(row=7, column=1, pady=5)
        self.inputs['parental_edu'].set("Bachelor")
    
    def create_prediction_widgets(self):
        """Create prediction and result display widgets"""
        # Predict button
        self.predict_button = ttk.Button(self.prediction_frame, text="Predict Performance", command=self.predict_performance)
        self.predict_button.grid(row=0, column=0, pady=10)
        
        # Result display
        self.result_frame = ttk.Frame(self.prediction_frame)
        self.result_frame.grid(row=1, column=0, pady=10)
        
        # Predicted score
        ttk.Label(self.result_frame, text="Predicted Final Score:", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w")
        self.score_label = ttk.Label(self.result_frame, text="--", font=("Arial", 16, "bold"), foreground="blue")
        self.score_label.grid(row=0, column=1, padx=10)
        
        # Performance category
        ttk.Label(self.result_frame, text="Performance Category:", font=("Arial", 12, "bold")).grid(row=1, column=0, sticky="w")
        self.category_label = ttk.Label(self.result_frame, text="--", font=("Arial", 14))
        self.category_label.grid(row=1, column=1, padx=10)
        
        # Recommendation
        ttk.Label(self.result_frame, text="Recommendation:", font=("Arial", 12, "bold")).grid(row=2, column=0, sticky="w", pady=10)
        self.recommendation_text = tk.Text(self.result_frame, height=4, width=40, wrap="word")
        self.recommendation_text.grid(row=3, column=0, columnspan=2, pady=5)
        
        # Feature importance button
        self.feature_button = ttk.Button(self.prediction_frame, text="Show Feature Impact", command=self.show_feature_impact)
        self.feature_button.grid(row=2, column=0, pady=10)
    
    def create_visualization_widgets(self):
        """Create visualization widgets"""
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 8))
        self.fig.tight_layout(pad=3.0)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Initial visualizations
        self.update_visualizations()
    
    def encode_features(self, features_dict):
        """Encode categorical features using the same encoders from training"""
        # Encode gender
        if 'gender' in features_dict:
            features_dict['gender'] = self.encoders['gender'].get(features_dict['gender'], 0)
        
        # Encode parental education
        if 'parental_education' in features_dict:
            features_dict['parental_education'] = self.encoders['parental_education'].get(
                features_dict['parental_education'], 0
            )
        
        return features_dict
    
    def predict_performance(self):
        """Make prediction based on input values"""
        try:
            # Collect input values
            features_dict = {
                'study_hours_per_week': self.inputs['study_hours'].get(),
                'attendance_percentage': self.inputs['attendance'].get(),
                'assignment_score': self.inputs['assignment'].get(),
                'internal_marks': self.inputs['internal'].get(),
                'previous_gpa': self.inputs['gpa'].get(),
                'extracurricular_activities': 1 if self.inputs['extracurricular'].get() == "Yes" else 0,
                'gender': self.inputs['gender'].get(),
                'parental_education': self.inputs['parental_edu'].get()
            }
            
            # Encode categorical features
            encoded_features = self.encode_features(features_dict)
            
            # Create feature array in the correct order
            features = []
            for col in self.feature_columns:
                # Map column names to the keys in encoded_features
                if col == 'gender_encoded':
                    features.append(encoded_features['gender'])
                elif col == 'parental_education_encoded':
                    features.append(encoded_features['parental_education'])
                else:
                    features.append(encoded_features.get(col, 0))
            
            # Convert to numpy array and scale
            features_array = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            
            # Ensure prediction is within valid range
            prediction = max(0, min(100, prediction))
            
            # Determine performance category and recommendation
            if prediction < 60:
                category = "Poor"
                color = "red"
                recommendation = "Needs significant improvement. Consider increasing study hours and seeking additional help."
            elif prediction < 75:
                category = "Average"
                color = "orange"
                recommendation = "Good progress. Focus on consistency and try to improve in weaker areas."
            elif prediction < 90:
                category = "Good"
                color = "blue"
                recommendation = "Well done! Maintain current performance and aim for excellence."
            else:
                category = "Excellent"
                color = "green"
                recommendation = "Outstanding performance! Keep up the excellent work and help others."
            
            # Update labels
            self.score_label.config(text=f"{prediction:.2f}", foreground=color)
            self.category_label.config(text=category, foreground=color)
            self.recommendation_text.delete(1.0, tk.END)
            self.recommendation_text.insert(1.0, recommendation)
            
            # Update visualizations
            self.update_visualizations(features, prediction)
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def show_feature_impact(self):
        """Show how each feature impacts the prediction"""
        try:
            # Get current feature values
            current_features = {
                'study_hours_per_week': self.inputs['study_hours'].get(),
                'attendance_percentage': self.inputs['attendance'].get(),
                'assignment_score': self.inputs['assignment'].get(),
                'internal_marks': self.inputs['internal'].get(),
                'previous_gpa': self.inputs['gpa'].get(),
                'extracurricular_activities': 1 if self.inputs['extracurricular'].get() == "Yes" else 0,
                'gender': self.inputs['gender'].get(),
                'parental_education': self.inputs['parental_edu'].get()
            }
            
            # Create display names for features
            display_names = [
                'Study Hours', 'Attendance', 'Assignment Score',
                'Internal Marks', 'Previous GPA', 'Extracurricular',
                'Gender', 'Parental Education'
            ]
            
            # Get values in the correct order
            values = []
            for col in self.feature_columns:
                if col == 'gender_encoded':
                    values.append(self.encoders['gender'].get(current_features['gender'], 0) * 100)
                elif col == 'parental_education_encoded':
                    values.append(self.encoders['parental_education'].get(current_features['parental_education'], 0) * 100)
                else:
                    values.append(current_features.get(col, 0))
            
            # Create impact visualization
            plt.figure(figsize=(10, 6))
            bars = plt.barh(display_names[:len(values)], values)
            plt.xlabel('Current Value')
            plt.title('Feature Values Impact on Performance')
            
            # Color bars based on importance
            for bar, value in zip(bars, values):
                if isinstance(value, (int, float)):
                    if value >= 75:
                        bar.set_color('green')
                    elif value >= 50:
                        bar.set_color('orange')
                    else:
                        bar.set_color('red')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show feature impact: {str(e)}")
    
    def update_visualizations(self, features=None, prediction=None):
        """Update the visualization plots"""
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot 1: Performance distribution (sample data)
        np.random.seed(42)
        sample_scores = np.random.normal(75, 15, 1000)
        self.ax1.hist(sample_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        self.ax1.set_title('Distribution of Student Performance')
        self.ax1.set_xlabel('Final Score')
        self.ax1.set_ylabel('Frequency')
        self.ax1.grid(True, alpha=0.3)
        
        # Add prediction line if available
        if prediction is not None:
            self.ax1.axvline(x=prediction, color='red', linestyle='--', linewidth=2, label=f'Your Score: {prediction:.1f}')
            self.ax1.legend()
        
        # Plot 2: Feature correlation heatmap (sample)
        correlation_data = np.random.rand(5, 5)
        feature_labels = ['Study', 'Attend', 'Assign', 'Internal', 'GPA']
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', 
                   xticklabels=feature_labels, yticklabels=feature_labels,
                   ax=self.ax2, fmt='.2f')
        self.ax2.set_title('Feature Correlation Matrix')
        
        # Refresh canvas
        self.canvas.draw()

def main():
    # Check if model exists
    if not os.path.exists('processed_data/best_model.pkl'):
        messagebox.showwarning("Warning", "Model not found. Please run the training script first.")
        return
        
    root = tk.Tk()
    app = StudentPerformanceGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()