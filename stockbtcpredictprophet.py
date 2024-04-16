# https://medium.com/@alexzap922/btc-price-prediction-using-fb-prophet-1bc4e8e5b5aa
#https://github.com/manilwagle/medium/blob/main/Microsoft%20Stock%20Price%20Prediction.ipynb

import pandas as pd
import plotly.express as px
import os
os.chdir('YOURPATH')    # Set working directory
os. getcwd()

#IMPORTS

import requests
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from termcolor import colored as cl

plt.rcParams['figure.figsize'] = (20, 10)
plt.style.use('fivethirtyeight')

#READ BTC DATA

def get_crypto_price(symbol, exchange, start_date = None):
    api_key = 'YOUR_API_KEY'
    api_url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={symbol}&market={exchange}&apikey={api_key}'
    raw_df = requests.get(api_url).json()
    df = pd.DataFrame(raw_df['Time Series (Digital Currency Daily)']).T
    df = df.rename(columns = {'1a. open (USD)': 'Open', '2a. high (USD)': 'High', '3a. low (USD)': 'Low', '4a. close (USD)': 'Close', '5. volume': 'Volume'})
    for i in df.columns:
        df[i] = df[i].astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.iloc[::-1].drop(['1b. open (USD)', '2b. high (USD)', '3b. low (USD)', '4b. close (USD)', '6. market cap (USD)'], axis = 1)
    if start_date:
        df = df[df.index >= start_date]
    return df

btc = get_crypto_price(symbol = 'BTC', exchange = 'USD', start_date = '2021-01-01')
btc.tail()

#BASIC STATISTICS

print(df.info())
print(df.describe())

#DATE COLUMN EDITING

df.reset_index(inplace=True)
df.rename(columns={'index': 'Date'}, inplace=True)

#PLOT BTC PRICE

px.area(df, x='Date', y='Close')
px.box(df, y='Close')

### Boxcox transformation
from statsmodels.base.transform import BoxCox

bc= BoxCox()
df["Close"], lmbda =bc.transform_boxcox(df["Close"])

## Making Prophet variables
data= df[["Date", "Close"]]
data.columns=["ds", "y"]

## Creating model parameters
model_param ={
    "daily_seasonality": False,
    "weekly_seasonality":False,
    "yearly_seasonality":True,
    "seasonality_mode": "multiplicative",
    "growth": "logistic"
}

#!pip install prophet

# Import Prophet
from prophet import Prophet

model = Prophet(**model_param)
data['cap']= data["y"].max() + data["y"].std() * 0.05 

model.fit(data)

# Create future dataframe
future= model.make_future_dataframe(periods=365)

future['cap'] = data['cap'].max()

forecast= model.predict(future)

odel.plot_components(forecast);

model.plot(forecast);# block dots are actual values and blue dots are forecast

## Adding parameters and seasonality and events

from IPython.core.display import display, HTML
display(HTML("<style>div.output_scroll { height: 144em; }</style>"))

model = Prophet(**model_param)

model= model.add_seasonality(name="monthly", period=30, fourier_order=10)
model= model.add_seasonality(name="quarterly", period=92.25, fourier_order=10)

model.add_country_holidays("US")

model.fit(data)

# Create future dataframe
future= model.make_future_dataframe(periods=365)
future['cap'] = data['cap'].max()

forecast= model.predict(future)

model.plot_components(forecast);
model.plot(forecast);

## Hyper parameter Tuning
import itertools
import numpy as np
from prophet.diagnostics import cross_validation, performance_metrics


param_grid={
    "daily_seasonality": [False],
    "weekly_seasonality":[False],
    "yearly_seasonality":[True],
    "growth": ["logistic"],
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5], # to give higher value to prior trend
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0] # to control the flexibility of seasonality components
}

# Generate all combination of parameters
all_params= [
            dict(zip(param_grid.keys(), v))
            for v in itertools.product(*param_grid.values())
]

print(all_params)

rmses= list ()

# go through each combinations
for params in all_params:
    m= Prophet(**params)
    

    m= m.add_seasonality(name= 'monthly', period=15, fourier_order=5)
    m= m.add_seasonality(name= "quarterly", period= 30, fourier_order= 10)
    m.add_country_holidays(country_name="US")
    
    m.fit(data)
    

    df_cv= cross_validation(m, initial="365 days", period="30 days", horizon="365 days")
                            
    df_p= performance_metrics(df_cv, rolling_window=1)
                            
    rmses.append(df_p['rmse'].values[0])
                            
# find teh best parameters
best_params = all_params[np.argmin(rmses)]
                            
print("\n The best parameters are:", best_params)   

forecast.head()

#INVERSE BOX COX TRANSFORM

forecast["yhat"]=bc.untransform_boxcox(x=forecast["yhat"], lmbda=lmbda)
forecast["yhat_lower"]=bc.untransform_boxcox(x=forecast["yhat_lower"], lmbda=lmbda)
forecast["yhat_upper"]=bc.untransform_boxcox(x=forecast["yhat_upper"], lmbda=lmbda)
forecast.plot(x="ds", y=["yhat_lower", "yhat", "yhat_upper"])

