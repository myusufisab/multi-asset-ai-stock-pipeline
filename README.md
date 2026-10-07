Hi, I’m Yusuf. I’m a data engineer who learned on my own. In my university I did not get to the ML part, so I kept going outside class.

I work on strong machine learning setups and fast backend data pipelines. Instead of only making small scripts, I build full systems. My goal is to connect deep learning transformer models, like PyTorch, with structured market indicator data in a clear way.

I made this project from the ground up. I wanted to see what slows real software down. I worked on things like fast network data loading with multiple threads, API rate limits on exchange servers, and explainability that is not just linear.

(PS: I am not an expert in markets in general i just used some ml techniques to noisy market data to predict so i am open to learning more about the market and how it works.)

Core tools I used:
- Python
- Pandas
- Scikit-Learn
- PyTorch
- Asyncio
- WebSockets

Main areas I focus on:
- Quant work and development
- Backend engineering
- Predictive MLOps

Alright Now how does this work
- First install all the required libraries (pip install pandas yfinance gnews torch transformers scikit-learn joblib numpy
)
- Copy the files train_model.py and run_terminal.py and put it in a folder
- Run the train_model.py and wait for it to finish training
- And finally open run_terminal.py in an editor and find this line
- if __name__ == "__main__":
   
    portfolio = ["AMZN"]
    
    print(f" Initializing AI Analysis Pipeline for {len(portfolio)} assets...")
    
    for company_ticker in portfolio:
        try:
            run_pure_pipeline(company_ticker)
        except Exception as e:
            print(f" Failed to process ticker {company_ticker}. Error: {e}")
  - Change the portfolio parameter (In this case Amazon(AMZN)) to any relevant stock market companies
  - Run the run_terminal.py file and look at the results


If you want to reach me:
m.yusuf.isab@gmail.com
