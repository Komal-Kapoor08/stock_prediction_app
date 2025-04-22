import streamlit as st
import yfinance as yf
import pandas as pd

from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression 

st.title('Stock Price Ticker')
st.sidebar.header('User Input Parameters')

stock_ticker = st.sidebar.text_input('Enter Stock Ticker(e.g.,RELIANCE.BO):','RELIANCE.BO')

start_date = st.sidebar.date_input('Start date',pd.to_datetime('2022-01-01'))

end_date = st.sidebar.date_input('End date',pd.to_datetime('2025-01-01'))


stock_data = yf.download(stock_ticker,start=start_date,end=end_date)
stock_data.reset_index(inplace=True)

stock_data['Days'] = (stock_data.index - stock_data.index.min())

stock_data = stock_data[['Days','Date','Close']]

x = stock_data.drop(['Date','Close'], axis = 'columns')
y = stock_data['Close']

model = LinearRegression()
model.fit(x,y)

st.sidebar.subheader('Future prediction input')
select_date = st.sidebar.date_input('select date',pd.to_datetime('2025-01-01'))
days = (pd.to_datetime(select_date) - stock_data['Date'].min()).days
y_pred = model.predict([[days]])

st.sidebar.write(f'predicted price for{select_date} is:{y_pred[0]}')
#st.write('Stock Data')
#st.write(stock_data)




