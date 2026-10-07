import yfinance as yf
from gnews import GNews
import pandas as pd
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def run_pure_pipeline(ticker_symbol):
    ticker_symbol = ticker_symbol.upper()
    print(f"\n STARTING AI ENGINE FOR: {ticker_symbol}...")
    
    # 1. Load Pre-trained AI/ML Models from Disk
    print(" Loading neural network layers (FinBERT & Random Forest)...")
    MODEL_NAME = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
    try:
        meta_model = joblib.load("meta_predictor.pkl")
    except FileNotFoundError:
        print(" ERROR: 'meta_predictor.pkl' not found! Please run 'python train_model.py' first.")
        return

    # 2. Fetch Live Market Pricing via yfinance
    print(" Fetching technical market metrics from Yahoo Finance...")
    stock = yf.Ticker(ticker_symbol)
    history = stock.history(period="5d")
    
    if history.empty or len(history) < 2:
        print(f" ERROR: Ticker symbol '{ticker_symbol}' returned no asset data.")
        return
        
    latest_close = float(history['Close'].values[-1])
    prev_close = float(history['Close'].values[-2])
    
    latest_vol = float(history['Volume'].values[-1])
    prev_vol = float(history['Volume'].values[-2])
    
    price_return = float((latest_close - prev_close) / prev_close)
    vol_change = float((latest_vol - prev_vol) / prev_vol)

    # 3. Scrape Current Financial News Media Headlines
    print(" Scraping current Google News tracking streams...")
    google_news = GNews(language='en', max_results=5)
    raw_news = google_news.get_news(f"{ticker_symbol} Stock")
    headlines = [item['title'] for item in raw_news if 'title' in item]
    combined_text = " ".join(headlines)

    # 4. Evaluate Financial Text Sentiment Matrix via PyTorch
    print(" Extracting semantic signals using deep learning transformer layer...")
    if combined_text.strip():
        inputs = tokenizer(combined_text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).tolist()[0]
        compound_sentiment = round(probabilities[0] - probabilities[1], 4)
    else:
        compound_sentiment = 0.0

    # 5. Run Ensembled Predictive ML Prediction Inference
    print(" Processing features through trained Random Forest classifier...")
    features = [[price_return, vol_change, compound_sentiment]]
    
    # [0] added to safely extract scalar values from the NumPy arrays
    prediction_class = int(meta_model.predict(features)[0])
    probabilities = meta_model.predict_proba(features)[0]
    confidence = round(float(probabilities[prediction_class]) * 100, 2)
    
    forecast_direction = "UP / BULLISH " if prediction_class == 1 else "DOWN / BEARISH "

    # --- PRINT INDUSTRIAL SYSTEM SUMMARY OUTPUT ---
    print("\n" + "="*60)
    print(f" HEADLINES EVALUATED FOR {ticker_symbol}:")
    print("="*60)
    if headlines:
        for idx, headline in enumerate(headlines, 1):
            print(f" [{idx}] {headline}")
    else:
        print(" No recent headlines found.")
        
    print("\n" + "="*60)
    print(f" AI PRODUCTION PIPELINE METRICS FOR: {ticker_symbol}")
    print("="*60)
    print(f"• Current Asset Price:    ${latest_close:.2f}")
    print(f"• Daily Price Delta:      {price_return * 100:.2f}%")
    print(f"• Net Sentiment Vector:   {compound_sentiment}  (-1.0 to +1.0)")
    print("-"*60)
    print(f" FINAL AI FORECAST:     {forecast_direction}")
    print(f" MODEL CONFIDENCE:      {confidence}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Define a list of different global companies you want to evaluate
    portfolio = ["AMZN"]
    
    print(f" Initializing AI Analysis Pipeline for {len(portfolio)} assets...")
    
    for company_ticker in portfolio:
        try:
            run_pure_pipeline(company_ticker)
        except Exception as e:
            print(f" Failed to process ticker {company_ticker}. Error: {e}")
