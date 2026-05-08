#!/usr/bin/env python3
"""
Student Performance Prediction System - Main Entry Point
"""

import sys
import os
from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
import tkinter as tk
from tkinter import messagebox
from gui_app import StudentPerformanceGUI

def check_requirements():
    """Check if all required files exist"""
    required_dirs = ['data', 'processed_data']
    for dir_name in required_dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    return True

def main():
    print("=" * 60)
    print("Student Performance Prediction System")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        print("Failed to create required directories")
        return
    
    # Step 1: Data Preprocessing
    print("\nStep 1: Data Preprocessing and EDA")
    print("-" * 40)
    
    preprocessor = DataPreprocessor('data/student_performance.csv')
    
    try:
        preprocessor.load_data()
    except FileNotFoundError:
        print("Data file not found. Generating sample data...")
        preprocessor.generate_sample_data()
        preprocessor.load_data()
    
    preprocessor.clean_data()
    preprocessor.perform_eda()
    X_train, X_test, y_train, y_test = preprocessor.prepare_features()
    
    # Step 2: Model Training
    print("\nStep 2: Model Training")
    print("-" * 40)
    
    trainer = ModelTrainer()
    best_model = trainer.train_models(X_train, X_test, y_train, y_test)
    
    # Save the best model
    trainer.save_model()
    
    print("\nTraining completed successfully!")
    print("=" * 60)
    
    # Step 3: Launch GUI
    print("\nStep 3: Launching GUI Application")
    print("-" * 40)
    print("Close the GUI window to exit the program.")
    
    try:
        root = tk.Tk()
        app = StudentPerformanceGUI(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to launch GUI: {str(e)}")
    
    print("\nProgram completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)