#🎬 Rental Length Prediction

This project predicts rental duration (in days) for movies based on features from a rental dataset. It uses machine learning models, hyperparameter tuning, and feature engineering to identify the best predictor of rental length.

📂 Project Overview

The dataset (rental_info.csv) contains information about movie rentals, including rental and return dates, special features, and other metadata.

Main steps in the workflow:

1. Parsed rental and return dates to calculate rental duration.
2. Created a target variable rental_length_days.
3. Extracted flags for deleted_scenes and behind_the_scenes from special features.
4. Standardized predictors with StandardScaler.
5. Applied Lasso regression for feature selection.
6. Trained multiple models: Decision Tree Regressor, Random Forest Regressor, and Gradient Boosting Regressor.
7. Compared models using Mean Squared Error (MSE) on the test set.


⚙️ Installation & Requirements

Make sure you have Python 3.8+ installed. Install dependencies with:

pip install -r requirements.txt


Required libraries:

pandas
numpy
scikit-learn


🚀 Usage

Place rental_info.csv in the project directory.

Run the script:  python rental_prediction.py

The script will preprocess the dataset.

It will train multiple regression models.

It will evaluate performance using MSE.

It will print the best-performing model and its error.
