
#🩺 Multi-Model Excess Mortality Forecasting

🎯 Project Overview
This project develops a robust Machine Learning pipeline to forecast potentially excess mortality rates across the United States. Utilizing granular public health data from the National Center for Health Statistics (NCHS), the goal is to identify and quantify the impact of key temporal, geographic, and demographic factors (e.g., State, Age Range, Cause of Death) on mortality gaps.The final deliverable is a production-ready system featuring a comparative analysis of multiple ML models and a live interactive dashboard for data-driven public health strategy and visualization.

✨ Features & Methodology
1. Data Engineering & Preprocessing
   Source Data: NCHS Potentially Excess Deaths from the Five Leading Causes of Death ($\sim$200,000 records).
   Feature Engineering: Conversion of high-cardinality categorical variables (State, Cause of Death, Age Range) using One-Hot Encoding.
   Data Quality: Implementation of advanced imputation techniques to handle missing values and maintain data integrity.
   
2. Multi-Model Comparative Analysis
   A diverse suite of both linear and non-linear algorithms are implemented and rigorously compared for performance on both Regression (predicting the count of excess deaths) and               Classification (predicting high/low risk).
   Implemented Models:
   Logistic Regression (LR)
   Support Vector Machine (SVM)
   K-Nearest Neighbors (KNN)
   Random Forest (RF)
   Optimization: Utilization of Principal Component Analysis (PCA) for dimensionality reduction and extensive Hyperparameter Tuning to maximize $R^2$ and AUC scores.
   
3. Deployment & Accessibility
   Interactive Dashboard: Built with Streamlit to provide a user-friendly interface for visualizing model predictions, key EDA findings, and feature importance.
   API Service: Developed a robust Flask REST API to host the final, optimized predictive model, enabling real-time inference and integration into other applications.

   🛠️ Technology Stack
             Category                      Tools & Libraries                                                                  Description
           Data & Core ML            Python, Pandas, NumPy, Scikit-learn                  Core languages and libraries for data manipulation and model training.
           Visualization                 Matplotlib, Seaborn, Streamlit                   Generating static plots and building the interactive web dashboard.
             Deployment                            Flask                                    Creating a production-ready API endpoint for model serving.
