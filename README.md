# Pneumonia

## Overview

This project is a machine learning-based data product designed to predict pneumonia recovery time based on patient clinical data. The system provides an interactive web interface where users can input patient details, receive real-time recovery time predictions, and generate downloadable reports for clinical reference. The goal is to enhance clinical decision-making by providing accurate, data-driven recovery estimates.

# Features

Machine Learning Model: Utilizes a regression model trained on clinical data to estimate recovery time.
Interactive UI: User-friendly interface allowing input of patient data.
Gauge Visualization: Displays predicted recovery time in an intuitive format.
Medicine Effectiveness Comparison: Helps assess different treatment impacts on recovery.
Downloadable Reports: Allows users to export predictions for medical reference.
Error Handling: Validates inputs and provides user-friendly messages for incorrect or missing data.

## Technologies Used

Backend: Python (Flask for API deployment)
Frontend: Dash for interactive visualizations
Machine Learning: Scikit-Learn for model training and evaluation
Deployment: Render for hosting the application


## Installation

##Prerequisites
Ensure you have the following installed:
--Python 3.8+
--pip (Python package manager)

## Clone Repo
https://github.com/eliwolf18/Pneumonia/tree/main

## Install dependecies
pip install -r requirements.txt

## Run App
python app.py

The application should now be accessible at http://0.0.0.0:8080/

## Usage

Input Patient Data: Enter patient-specific parameters such as age, respiratory rate, and treatment details.
Predict Recovery Time: Click the "Predict" button to generate an estimated recovery duration.
View and Analyze Results: The system displays the estimated recovery time using a gauge visualization.
Compare Treatment Options: Analyze how different medications impact recovery outcomes.
Download Report: Export predictions as a report for clinical documentation.

## Deployment
The system is deployed on Render/Heroku and can be accessed at: https://pneumonia-1-prg3.onrender.com/ 

## Future Improvements

External Dataset Validation: Improve generalization by testing on new datasets.
Explainability Enhancements: Implement SHAP values for better model interpretation.
Performance Optimization: Implement caching mechanisms for faster predictions.
Mobile Compatibility: Optimize UI for better mobile accessibility.

Contact

For questions or collaboration opportunities, reach out at:
Email: keksdarl95@yahoo.com
GitHub: eliwolfe18



