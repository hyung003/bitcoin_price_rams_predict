import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import yfinance as yf

# Set page config
st.set_page_config(page_title="Bitcoin Price Prediction", layout="wide", page_icon="💰", initial_sidebar_state="collapsed")

# Predictions
predictions = {
    "Matt": 66969,
    "Vai": 59000,
    "Bailey": 60999,
    "Talent": 65500,
    "Jensen": 62888,
    "Bryan L": 68000,
    "Alvin": 68888,
    "Chris": 70500,
    "Henry": 62865,
    "Simon": 61234
}

# Rate limiting constants
MAX_REQUESTS = 5
REQUEST_WINDOW = 300  # 5 minutes in seconds

# Initialize session state
if 'last_request_time' not in st.session_state:
    st.session_state.last_request_time = time.time()
if 'request_count' not in st.session_state:
    st.session_state.request_count = 0
if 'last_successful_fetch' not in st.session_state:
    st.session_state.last_successful_fetch = None
if 'cached_bitcoin_price' not in st.session_state:
    st.session_state.cached_bitcoin_price = None
if 'cached_historical_data' not in st.session_state:
    st.session_state.cached_historical_data = None

def can_make_request():
    current_time = time.time()
    if current_time - st.session_state.last_request_time > REQUEST_WINDOW:
        st.session_state.request_count = 0
        st.session_state.last_request_time = current_time
    
    if st.session_state.request_count >= MAX_REQUESTS:
        return False
    
    return True

def increment_request_count():
    st.session_state.request_count += 1
    st.session_state.last_request_time = time.time()

def fetch_bitcoin_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "bitcoin" in data and "usd" in data["bitcoin"]:
            return data["bitcoin"]["usd"]
        else:
            return None
    except requests.RequestException:
        return None

def fetch_bitcoin_price_backup():
    try:
        ticker = yf.Ticker("BTC-USD")
        data = ticker.history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
        else:
            return None
    except Exception:
        return None

def get_bitcoin_price():
    if not can_make_request():
        return st.session_state.cached_bitcoin_price, False
    increment_request_count()
    price = fetch_bitcoin_price()
    if price is None:
        price = fetch_bitcoin_price_backup()
    if price is not None:
        st.session_state.cached_bitcoin_price = price
        st.session_state.last_successful_fetch = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return price, True
    return st.session_state.cached_bitcoin_price, False

def fetch_historical_bitcoin_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range?vs_currency=usd&from={int(start_date.timestamp())}&to={int(end_date.timestamp())}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "prices" in data:
            df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.floor('d')
            return df
        else:
            return None
    except requests.RequestException:
        return None

def fetch_historical_bitcoin_data_backup():
    try:
        ticker = yf.Ticker("BTC-USD")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        data = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        if not data.empty:
            df = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'date', 'Close': 'price'})
            return df
        else:
            return None
    except Exception:
        return None

def get_historical_bitcoin_data():
    if not can_make_request():
        return st.session_state.cached_historical_data, False
    increment_request_count()
    data = fetch_historical_bitcoin_data()
    if data is None:
        data = fetch_historical_bitcoin_data_backup()
    if data is not None:
        st.session_state.cached_historical_data = data
        st.session_state.last_successful_fetch = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return data, True
    return st.session_state.cached_historical_data, False

def calculate_rmse(actual, predicted):
    return ((actual - predicted) ** 2) ** 0.5

def refresh_data():
    st.session_state.request_count = 0  # Reset the request count on refresh
    st.experimental_rerun()

st.title('Bitcoin Price Prediction Rankings')

# Refresh button
if st.button('Refresh Data'):
    refresh_data()

# Display current price and last updated time
current_price, is_fresh = get_bitcoin_price()
if current_price is not None:
    st.write(f"Current Bitcoin Price: ${current_price:,.2f}")
    if is_fresh:
        st.write(f"Last updated: {st.session_state.last_successful_fetch}")
    else:
        st.write(f"Using cached data. Last successful fetch: {st.session_state.last_successful_fetch}")
else:
    st.warning("Unable to fetch current Bitcoin price. No cached data available.")

# Calculate RMSE and create rankings
if current_price is not None:
    rankings = []
    for name, prediction in predictions.items():
        rmse = calculate_rmse(current_price, prediction)
        rankings.append({"Name": name, "Prediction": prediction, "RMSE": rmse})

    rankings_df = pd.DataFrame(rankings)
    rankings_df = rankings_df.sort_values('RMSE')

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Prediction Rankings")
        
        # Style the DataFrame
        def color_rmse(val):
            color = '#4CAF50' if val < 2000 else '#FFC107' if val < 4000 else '#F44336'
            return f'color: {color}'
        
        styled_rankings_df = rankings_df.style.applymap(color_rmse, subset=['RMSE']).set_properties(**{
            'text-align': 'center'
        })
        
        st.dataframe(styled_rankings_df.format({'Prediction': '${:,.2f}', 'RMSE': '{:.2f}'}))

    with col2:
        st.subheader("RMSE Comparison")
        fig_bar = go.Figure(data=[
            go.Bar(name='RMSE', x=rankings_df['Name'], y=rankings_df['RMSE'], marker=dict(color=rankings_df['RMSE'].apply(lambda x: '#4CAF50' if x < 2000 else '#FFC107' if x < 4000 else '#F44336')))
        ])
        fig_bar.update_layout(
            xaxis_title='Name',
            yaxis_title='RMSE',
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#000000')
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Historical Bitcoin Prices with Predictions")
    historical_data, is_fresh = get_historical_bitcoin_data()
    
    if historical_data is not None:
        fig = go.Figure()

        # Add the historical price line
        fig.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['price'],
            mode='lines',
            name='Bitcoin Price',
            line=dict(color='#1E88E5', width=2)
        ))

        # Add horizontal prediction lines with hover text
        if current_price is not None:
            for name, prediction in predictions.items():
                fig.add_trace(go.Scatter(
                    x=[historical_data['date'].min(), historical_data['date'].max()],
                    y=[prediction, prediction],
                    mode='lines',
                    name=f'{name} Prediction',
                    line=dict(dash='dash', width=2),
                    hovertext=f'{name} Prediction: ${prediction:,.2f}',
                    hoverinfo='text'
                ))

        fig.update_layout(
            title='Bitcoin Price - Last 365 Days with Predictions',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_white',  # Set to white theme
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            plot_bgcolor='white',  # Set plot background to white
            paper_bgcolor='white',  # Set paper background to white
            font=dict(color='#000000'),  # Set font color to black for better contrast
            xaxis=dict(
                gridcolor='rgba(200, 200, 200, 0.5)',
                showline=True,
                linecolor='rgba(200, 200, 200, 0.5)',
                showgrid=True,
                gridwidth=1
            ),
            yaxis=dict(
                gridcolor='rgba(200, 200, 200, 0.5)',
                showline=True,
                linecolor='rgba(200, 200, 200, 0.5)',
                showgrid=True,
                gridwidth=1
            ),
            margin=dict(l=20, r=20, t=50, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)
        if not is_fresh:
            st.write(f"Using cached data. Last successful fetch: {st.session_state.last_successful_fetch}")
    else:
        st.warning("Unable to fetch historical Bitcoin data. No cached data available.")

    # Calculate daily rankings and create bump chart data
    bump_chart_data = []
    for date, group in historical_data.groupby('date'):
        if date >= datetime(2024, 7, 16):
            daily_price = group['price'].values[0]
            daily_rankings = []
            for name, prediction in predictions.items():
                rmse = calculate_rmse(daily_price, prediction)
                daily_rankings.append({"Name": name, "RMSE": rmse})
            daily_rankings_df = pd.DataFrame(daily_rankings).sort_values('RMSE')
            daily_rankings_df['Rank'] = range(1, len(daily_rankings_df) + 1)
            for index, row in daily_rankings_df.iterrows():
                bump_chart_data.append({'Date': date, 'Name': row['Name'], 'Rank': row['Rank']})

    bump_df = pd.DataFrame(bump_chart_data)

    # Create the bump chart
    fig_bump = px.line(bump_df, x='Date', y='Rank', color='Name', markers=True)
    fig_bump.update_layout(
        yaxis=dict(autorange='reversed'),  # Invert y-axis to have rank 1 at the top
        title="Rank Changes Over Time",
        xaxis_title="Date",
        yaxis_title="Rank",
        template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#000000')
    )

    st.plotly_chart(fig_bump, use_container_width=True)
