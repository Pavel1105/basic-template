
import pandas as pd
import streamlit as st
import yfinance as yf
import requests
import os
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io

# Fetch actual stock prices from Yahoo Finance and save as CSV
def fetch_and_save_actual_prices(ticker: str) -> None:
    try:
        data = yf.download(ticker, start="2000-01-01", end=datetime.today().strftime('%Y-%m-%d'))
        filename = f"{ticker}_Historical_prices.csv"
        data.to_csv(filename)
        st.success(f"Actual prices for {ticker} downloaded and saved successfully as {filename}.")
    except Exception as e:
        st.error(f"Error fetching and saving actual prices for {ticker}: {e}")

# Load stock data from CSV or fetch if CSV is older than 1 day
def load_stock_data(ticker: str) -> pd.DataFrame:
    filename = f"{ticker}_Historical_prices.csv"
    if os.path.exists(filename):
        modification_time = os.path.getmtime(filename)
        if (time.time() - modification_time) > 86400:  # 86400 seconds = 1 day
            fetch_and_save_actual_prices(ticker)
    else:
        fetch_and_save_actual_prices(ticker)

    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        st.error(f"Error: File {filename} not found.")
        return None

# Process stock data to calculate changes over specified periods
def process_stock_data(data: pd.DataFrame, years: int, days: int) -> pd.DataFrame:
    try:
        data['Date'] = pd.to_datetime(data['Date'], utc=True)
        data['Date'] = data['Date'].dt.tz_localize(None)
    except KeyError:
        st.error("Error: 'Date' column not found in the CSV file.")
        return None

    start_date = pd.Timestamp.now() - pd.DateOffset(years=years)
    data = data.loc[data['Date'] >= start_date].copy()

    change_col_name = f'{days} Day Change (%)'
    start_col_name = f'Close at Start {days}-day'
    end_col_name = f'Close at End {days}-day'

    data[change_col_name] = ((data['Close'].shift(-days) - data['Close']) / data['Close']) * 100
    data[start_col_name] = data['Close'].copy()
    data[end_col_name] = data['Close'].shift(-days).copy()

    return data

# Filter top periods based on specified column and criteria
def filter_top_periods(data: pd.DataFrame, column: str, days: int = 20, worst: bool = False) -> pd.DataFrame:
    sorted_data = data.sort_values(by=column, ascending=worst)
    filtered_periods = []
    for index, row in sorted_data.iterrows():
        if all(abs((row['Date'] - d).days) > days for d in filtered_periods):
            filtered_periods.append(row['Date'])
            if len(filtered_periods) == 10:
                break
    return data[data['Date'].isin(filtered_periods)]

# Format the percentage change values and sort the DataFrame by the absolute change
def format_change(df: pd.DataFrame, column: str, sort_ascending: bool = False) -> pd.DataFrame:
    df = df.copy()
    df[column + ' Formatted'] = df[column].apply(lambda x: f"{x:.1f}%")
    df.insert(0, 'Row', range(1, 1 + len(df)))
    return df.sort_values(by=column, ascending=sort_ascending)

# Get the name of the stock from its ticker symbol
def get_stock_name(ticker: str) -> str:
    try:
        stock_info = yf.Ticker(ticker).info
        return stock_info.get('longName', ticker)
    except Exception as e:
        st.error(f"Error fetching stock name: {e}")
        return ticker

# Fetch news articles for the stock from New York Times
def fetch_news(stock_name: str, worst_periods_dates: pd.Series) -> None:
    apikey = os.getenv('NYTIMES_APIKEY', 'your_api_key_here')  # Update with your NYT API key

    for i, worst_date in enumerate(worst_periods_dates):
        if i % 2 == 0:
            start_date = pd.to_datetime(worst_date)
            end_date = start_date + timedelta(days=10)

            start_date_str_api = start_date.strftime('%Y%m%d')
            end_date_str_api = end_date.strftime('%Y%m%d')
            
            start_date_str_print = start_date.strftime('%Y-%m-%d')
            end_date_str_print = end_date.strftime('%Y-%m-%d')

            query_url = f"https://api.nytimes.com/svc/search/v2/articlesearch.json?q={stock_name}&begin_date={start_date_str_api}&end_date={end_date_str_api}&api-key={apikey}"
            
            try:
                response = requests.get(query_url)
                if response.status_code == 200:
                    data = response.json()
                    articles = [(datetime.strptime(article.get('pub_date', ''), '%Y-%m-%dT%H:%M:%S%z'),
                                 article.get('headline', {}).get('main', ''),
                                 article.get('web_url', ''))
                                for article in data.get('response', {}).get('docs', [])]

                    articles.sort(key=lambda x: x[0])

                    st.write(f"Fetching NYT news for the Worst period {start_date_str_print} to {end_date_str_print}:")
                    for pub_date, headline, web_url in articles:
                        formatted_date = pub_date.strftime('%d.%m.%Y')
                        st.write(f"{formatted_date}: {headline} - {web_url}")
                else:
                    st.error(f"Error: {response.status_code} - {response.reason}")

                time.sleep(1)

            except requests.RequestException as e:
                st.error(f"Network error: {e}")

            except ValueError as e:
                st.error(f"JSON decoding error: {e}")

# Visualize top and bottom periods with sorted results
def visualize_periods(best_periods, worst_periods, days, sort_option, sort_by_percentage=True, ticker=None, yy=None):
    plt.figure(figsize=(10, 6))

    if sort_by_percentage:
        worst_periods = worst_periods.sort_values(by=f'{days} Day Change (%)', ascending=False).head(10)
        plt.barh(worst_periods['Date'].dt.strftime('%d.%m.%Y'), worst_periods[f'{days} Day Change (%)'], color='red', label='Worst Periods')

        best_periods = best_periods.sort_values(by=f'{days} Day Change (%)', ascending=False).head(10)
        plt.barh(best_periods['Date'].dt.strftime('%d.%m.%Y'), best_periods[f'{days} Day Change (%)'], color='green', label='Best Periods')

        plt.xlabel('Change (%)')
        plt.title(f'TOP 10 Best and Worst Periods with the Highest and Lowest Changes ({days}-day), for: {ticker}, in the last {yy} years, sorted by: {sort_option}')

    else:
        best_periods = best_periods.sort_values(by='Date', ascending=False).head(10)
        plt.barh(best_periods['Date'].dt.strftime('%d.%m.%Y'), best_periods[f'{days} Day Change (%)'], color='green', label='Best Periods')

        worst_periods = worst_periods.sort_values(by='Date', ascending=False).head(10)
        plt.barh(worst_periods['Date'].dt.strftime('%d.%m.%Y'), worst_periods[f'{days} Day Change (%)'], color='red', label='Worst Periods')

        plt.xlabel('Change (%)')
        plt.title(f'TOP 10 Best and Worst Periods with the Highest and Lowest Changes ({days}-day), for: {ticker}, in the last {yy} years, sorted by: {sort_option}')

    plt.ylabel('Date')
    plt.legend()
    plt.grid(axis='x')
    plt.tight_layout()

    # Save plot to a BytesIO object and use Streamlit to display it
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    st.image(buf)
    buf.close()

# Main function
def main() -> None:
    st.title("Stock Analysis")

    ticker = st.text_input("Enter the ticker:")
    years = st.number_input("Enter the number of years for analysis:", min_value=1, max_value=100, value=5)
    days = st.number_input("Enter the number of days for change calculation:", min_value=1, max_value=365, value=20)

    if ticker:
        st.write(f"Analyzing data for ticker: {ticker}")

        # Load stock data
        stock_data = load_stock_data(ticker)
        if stock_data is not None:
            # Process data
            processed_data = process_stock_data(stock_data, years, days)

            if processed_data is not None:
                # Filter and format periods
                best_periods = filter_top_periods(processed_data, f'{days} Day Change (%)', days, worst=False)
                worst_periods = filter_top_periods(processed_data, f'{days} Day Change (%)', days, worst=True)

                best_periods = format_change(best_periods, f'{days} Day Change (%)')
                worst_periods = format_change(worst_periods, f'{days} Day Change (%)', sort_ascending=False)

                # Visualize data
                sort_option = st.selectbox("Select sorting option:", ["Percentage Change", "Date"])
                sort_by_percentage = sort_option == "Percentage Change"
                visualize_periods(best_periods, worst_periods, days, sort_option, sort_by_percentage, ticker, years)

                # Fetch news
                fetch_news(get_stock_name(ticker), worst_periods['Date'])

if __name__ == "__main__":
    main()

