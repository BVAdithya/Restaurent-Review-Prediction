# Restaurent-Review-Prediction

This project aims to classify restaurant reviews as positive or negative using a machine learning model. The solution includes a user-friendly graphical user interface (GUI) to allow real-time sentiment analysis of user-provided reviews.

## Features
User-friendly GUI for inputting and analyzing reviews.
Real-time sentiment prediction using a trained Random Forest classifier.
Text preprocessing and feature extraction using NLTK and Scikit-learn.

## Installation
Clone the repository:
git clone https://github.com/yourusername/restaurant-review-sentiment-analysis.git
cd restaurant-review-sentiment-analysis

Install the required libraries:
pip install pandas numpy nltk scikit-learn joblib tkinter

Download NLTK stopwords:

import nltk
nltk.download('stopwords')

Ensure your model files (Restaurant_review_model.pkl and count_v_res.pkl) are in the project directory.

## Project Structure
app.py: Main script to run the GUI application.
model_training.ipynb: Jupyter notebook for training the Random Forest classifier.
Restaurant_review_model.pkl: Trained Random Forest model.
count_v_res.pkl: Trained TF-IDF vectorizer.
data/: Directory containing the dataset used for training (if applicable).

## How It Works
Data Preprocessing: Reviews are cleaned, tokenized, and converted to numerical features using TF-IDF.
Model Training: The Random Forest classifier is trained on the preprocessed data.
Sentiment Prediction: The trained model predicts the sentiment of new reviews based on learned patterns.
