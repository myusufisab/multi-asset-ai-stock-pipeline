import yfinance as yf
from gnews import GNews
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import time

# 1. Initialize FinBERT
print("Loading FinBERT for dataset generation...")
MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def get_sentiment_score(text):
    if not text or str(text).strip() == "": return 0.0
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).tolist()[0]
    return probs[0] - probs[1] # Positive - Negative score

# 2. Simulate historical feature extraction pipeline
def build_historical_dataset(ticker, start_date="2025-01-01", end_date="2026-01-01"):
    print(f"Downloading historical price data for {ticker}...")
    prices = yf.download(ticker, start=start_date, end=end_date)
    prices = prices.reset_index()
    prices['Date'] = pd.to_datetime(prices['Date']).dt.date
    
    # Feature Engineering via Pandas
    prices['Price_Return'] = prices['Close'].pct_change()
    prices['Vol_Change'] = prices['Volume'].pct_change()
    
    # Simulate historical daily news sentiment mapping 
    # (In production, you'd pull from a data warehouse; here we generate structured synthetic sentiment aligned to market days for training)
    print("Generating aligned sentiment features...")
    np.random.seed(42)
    prices['Sentiment'] = np.random.uniform(-0.6, 0.6, len(prices)) + (prices['Price_Return'] * 2)
    prices['Sentiment'] = prices['Sentiment'].clip(-1.0, 1.0)
    
    # Define target: 1 if tomorrow's price is HIGHER than today's close, else 0
    prices['Target'] = (prices['Close'].shift(-1) > prices['Close']).astype(int)
    
    # Drop rows with NaN caused by pct_change or shift
    dataset = prices.dropna()[['Price_Return', 'Vol_Change', 'Sentiment', 'Target']]
    return dataset

if __name__ == "__main__":
    # Build dataset using NVIDIA data
    data = build_historical_dataset("NVDA")
    
    X = data[['Price_Return', 'Vol_Change', 'Sentiment']]
    y = data['Target']
    
    # Split into Train and Test groups
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train Production-Grade Random Forest Model
    print("Training Random Forest Classifier Engine...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    accuracy = rf_model.score(X_test, y_test)
    print(f" Production Model Trained Successfully! Test Accuracy: {accuracy * 100:.2f}%")
    
    # 4. Save the trained model file to disk
    joblib.dump(rf_model, "meta_predictor.pkl")
    print("Model saved as 'meta_predictor.pkl'")
