import streamlit as st
from newsapi import NewsApiClient
from transformers import pipeline
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests

st.set_page_config(page_title="Stock Sentiment Analyzer", layout="wide")


@st.cache_resource
def load_model():
    return pipeline("text-classification", model="ProsusAI/finbert")

def is_relevant(title, company, ticker):
    title_lower = title.lower()
    keywords = [w.lower() for w in [company, ticker] if w]
    company_words = [w.lower() for w in company.split() if len(w) > 2]
    all_keywords = list(set(keywords + company_words))
    return any(kw in title_lower for kw in all_keywords)

@st.cache_data(show_spinner=False)
def load_news(api_key, company, ticker, days):
    newsapi = NewsApiClient(api_key=api_key)
    from datetime import datetime, timedelta
    to_date = datetime.today().strftime("%Y-%m-%d")
    from_date = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    news = newsapi.get_everything(
        q=company,
        language="en",
        sort_by="publishedAt",
        from_param=from_date,
        to=to_date,
        page_size=100
    )
    headlines = []
    for article in news["articles"]:
        title = article["title"]
        if not title or title == "[Removed]":
            continue
        if not is_relevant(title, company, ticker):
            continue
        headlines.append({
            "date": article["publishedAt"][:10],
            "headline": title
        })
    return pd.DataFrame(headlines)

@st.cache_data(show_spinner=False)
def load_stock(ticker, days):
    stock = yf.download(ticker, period=f"{days}d", interval="1d", progress=False)
    if stock.empty:
        return pd.DataFrame()
    stock = stock[["Close"]].reset_index()
    stock.columns = ["date", "close"]
    stock["date"] = stock["date"].astype(str)
    stock["price_change"] = stock["close"].pct_change() * 100
    return stock

def get_sentiment(text, model):
    result = model(text[:512])[0]
    label = result["label"]
    score = result["score"]
    if label == "positive":
        return label, score
    elif label == "negative":
        return label, -score
    else:
        return label, 0.0

LOGO_DOMAINS = {
    "tesla": "tesla.com",
    "apple": "apple.com",
    "microsoft": "microsoft.com",
    "nvidia": "nvidia.com",
    "amazon": "amazon.com",
    "google": "google.com",
    "alphabet": "google.com",
    "meta": "meta.com",
    "netflix": "netflix.com",
    "intel": "intel.com",
    "amd": "amd.com",
    "paypal": "paypal.com",
    "uber": "uber.com",
    "airbnb": "airbnb.com",
    "spotify": "spotify.com",
    "twitter": "twitter.com",
    "x": "x.com",
    "openai": "openai.com",
    "samsung": "samsung.com",
    "sony": "sony.com",
}

def get_logo_url(company_name):
    key = company_name.lower().strip()
    domain = LOGO_DOMAINS.get(key, f"{key.replace(' ', '')}.com")
    return f"https://logo.clearbit.com/{domain}"

def logo_works(url):
    """Return True only if Clearbit actually serves a valid image."""
    try:
        r = requests.get(url, timeout=3)
        return r.status_code == 200 and "image" in r.headers.get("Content-Type", "")
    except Exception:
        return False

st.sidebar.header("Configuration")
API_KEY = st.sidebar.text_input("NewsAPI Key", type="password", value=st.secrets.get("NEWS_API_KEY", ""))
ticker = st.sidebar.text_input("Stock Ticker", value="").upper()
company = st.sidebar.text_input("Company Name (for news search)", value="")
days = st.sidebar.slider("Days of stock history", 7, 90, 30)
run = st.sidebar.button("🔍 Analyze", use_container_width=True)

col_logo, col_title = st.columns([1, 9])
with col_logo:
    if company:
        logo_url = get_logo_url(company)
        if logo_works(logo_url):
            st.image(logo_url, width=72)
        else:
            st.markdown(
                f"<div style='width:72px;height:72px;background:linear-gradient(135deg,#00c9ff,#92fe9d);"
                f"border-radius:12px;display:flex;align-items:center;justify-content:center;"
                f"font-size:28px;font-weight:700;color:#0a0a0f'>{company[0].upper()}</div>",
                unsafe_allow_html=True
            )
with col_title:
    if company and ticker:
        st.title(f"{company} ({ticker}) — Stock Sentiment Dashboard")
    else:
        st.title("Stock Sentiment Dashboard")
    st.caption("Powered by FinBERT · Next-day correlation analysis")

st.divider()

if run:
    if not API_KEY:
        st.error("Please enter your NewsAPI key in the sidebar.")
        st.stop()

    finbert = load_model()

    with st.spinner("Fetching news headlines..."):
        df_news = load_news(API_KEY, company, ticker, days)

    if df_news.empty:
        st.error("No recent news found. Try a different company name or expand the date range.")
        st.stop()

    with st.spinner("Running FinBERT sentiment analysis..."):
        df_news[["sentiment", "score"]] = df_news["headline"].apply(
            lambda x: pd.Series(get_sentiment(x, finbert))
        )

    with st.spinner("Fetching stock data..."):
        stock = load_stock(ticker, days)

    if stock.empty:
        st.error(f"Invalid ticker symbol '{ticker}'. Please check and try again.")
        st.stop()

    df_sentiment = df_news.groupby("date").agg(
        avg_score=("score", "mean"),
        headline_count=("headline", "count"),
        negative_count=("sentiment", lambda x: (x == "negative").sum()),
        positive_count=("sentiment", lambda x: (x == "positive").sum())
    ).reset_index()

    df_merged = pd.merge(stock, df_sentiment, on="date", how="left")
    df_merged["next_day_return"] = (
        df_merged["close"].shift(-1) - df_merged["close"]
    ) / df_merged["close"]

    st.subheader("📊 Key Metrics")
    latest_price = float(stock["close"].iloc[-1])
    prev_price = float(stock["close"].iloc[-2]) if len(stock) > 1 else latest_price
    daily_return = ((latest_price - prev_price) / prev_price) * 100
    avg_sentiment = df_news["score"].mean()
    total_headlines = len(df_news)
    neg_count = int(df_news["sentiment"].eq("negative").sum())
    pos_count = int(df_news["sentiment"].eq("positive").sum())
    best_headline = df_news.loc[df_news["score"].idxmax(), "headline"]
    worst_headline = df_news.loc[df_news["score"].idxmin(), "headline"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latest Close", f"${latest_price:.2f}", f"{daily_return:+.2f}%")
    m2.metric("Headlines Analyzed", total_headlines)
    m3.metric("Positive Headlines", pos_count)
    m4.metric("Negative Headlines", neg_count)

    st.subheader("🧭 Overall Sentiment")
    if avg_sentiment > 0.2:
        st.success(f"Overall Sentiment: **Positive** (avg score: {avg_sentiment:.3f})")
    elif avg_sentiment < -0.2:
        st.error(f"Overall Sentiment: **Negative** (avg score: {avg_sentiment:.3f})")
    else:
        st.warning(f"Overall Sentiment: **Neutral** (avg score: {avg_sentiment:.3f})")

    st.markdown(f"🏆 **Most Positive Headline:** {best_headline}")
    st.markdown(f"⚠️ **Most Negative Headline:** {worst_headline}")

    st.divider()

    st.subheader("📈 Price vs. Sentiment")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=stock["date"], y=stock["close"],
            name="Stock Price",
            line=dict(color="#1f77b4", width=2.5),
            fill="tozeroy", fillcolor="rgba(31,119,180,0.07)"
        ),
        secondary_y=False
    )
    sentiment_days = df_merged.dropna(subset=["avg_score"])
    colors = ["#d62728" if s < 0 else "#2ca02c" for s in sentiment_days["avg_score"]]
    fig.add_trace(
        go.Bar(
            x=sentiment_days["date"], y=sentiment_days["avg_score"],
            name="Avg Sentiment Score",
            marker_color=colors, opacity=0.75
        ),
        secondary_y=True
    )
    fig.update_layout(
        legend=dict(orientation="h", y=1.12),
        hovermode="x unified", height=440,
    )
    fig.update_yaxes(title_text="Stock Price (USD)", secondary_y=False)
    fig.update_yaxes(title_text="Sentiment Score", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🥧 Sentiment Distribution")
    pie_fig = px.pie(
        df_news, names="sentiment",
        title="Headline Sentiment Breakdown",
        color="sentiment",
        color_discrete_map={"positive": "#2ca02c", "negative": "#d62728", "neutral": "#7f7f7f"}
    )
    st.plotly_chart(pie_fig, use_container_width=True)

    st.divider()

    st.subheader("🔗 Correlation Analysis")
    df_corr_same = df_merged.dropna(subset=["avg_score", "price_change"])
    df_corr_next = df_merged.dropna(subset=["avg_score", "next_day_return"])

    c1, c2 = st.columns(2)
    with c1:
        if len(df_corr_same) >= 2:
            corr_same = df_corr_same["avg_score"].corr(df_corr_same["price_change"])
            st.metric("Same-Day Correlation", f"{corr_same:.4f}")
            if corr_same > 0.3:
                st.success("Positive — good news coincides with same-day price rises")
            elif corr_same < -0.3:
                st.error("Negative — bad news coincides with same-day price drops")
            else:
                st.warning("Weak same-day correlation")
        else:
            st.info("Not enough same-day data points.")

    with c2:
        if len(df_corr_next) >= 2:
            corr_next = df_corr_next["avg_score"].corr(df_corr_next["next_day_return"])
            st.metric("Next-Day Correlation", f"{corr_next:.4f}")
            if corr_next > 0.3:
                st.success("Positive — today's sentiment predicts tomorrow's gains")
            elif corr_next < -0.3:
                st.error("Negative — today's bad news predicts tomorrow's drops")
            else:
                st.warning("Weak next-day predictive correlation")
        else:
            st.info("Not enough next-day data points.")

    st.divider()

    st.subheader("📰 Headlines Breakdown")
    st.dataframe(
        df_news[["date", "headline", "sentiment", "score"]].sort_values("score"),
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    csv = df_news.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Full Analysis as CSV",
        data=csv,
        file_name=f"{ticker}_sentiment_analysis.csv",
        mime="text/csv",
        use_container_width=True
    )
