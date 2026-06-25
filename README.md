# Stock Sentiment Analyzer

&#x20;📈 Stock Sentiment Analyzer



A Streamlit dashboard that combines real-time financial news, FinBERT NLP sentiment analysis, and interactive stock price visualization to measure whether news sentiment correlates with price movement.



&#x20;Features

\- Company logo auto-loaded

\- Live news headlines from NewsAPI (filtered to company-relevant articles only)

\- FinBERT sentiment scoring — Positive / Negative / Neutral

\- Interactive Plotly charts — price vs sentiment overlaid

\- Sentiment pie chart

\- Same-day AND next-day return correlation analysis

\- CSV export

\- Cached data loading for fast re-runs

\- Error handling for invalid tickers and missing news



&#x20;Tech Stack

\- Sentiment model: ProsusAI/FinBERT via HuggingFace

\- News: NewsAPI

\- Stock data: yfinance

\- Visualization: Plotly

\- Frontend: Streamlit



&#x20;Usage

1\. Get a free API key from newsapi.org

2\. Enter ticker (e.g. TSLA), company name, and date range

3\. Click Analyze

