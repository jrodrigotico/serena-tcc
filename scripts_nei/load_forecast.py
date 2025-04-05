import requests
import json
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytz

api_key = '45bd4693a487a8ad3a2a2dd40a1bc30d'

df_total = None

for month in range(1):

    date_start = (datetime(2024, 11, 1) - relativedelta(months=month+1)).strftime('%Y-%m-%d')
    date_end   = (datetime(2024, 11, 1) - relativedelta(months=month)).strftime('%Y-%m-%d')
    
    print(date_start)
    
    url = f"https://api.gridstatus.io/v1/datasets/ercot_load_forecast_by_weather_zone/query?api_key={api_key}&start_time={date_start}T00:00Z&end_time={date_end}T00:00Z"
    
    #url = f"https://api.gridstatus.io/v1/datasets/ercot_standardized_hourly/query?api_key=45bd4693a487a8ad3a2a2dd40a1bc30d&start_time=2022-12-31T00:00Z&end_time=2024-11-01T00:00Z"
    
    
    response = requests.request("GET", url)
    
    
    df = pd.DataFrame(json.loads(response.text)['data'])
    
    df_total = pd.concat([df_total,df], ignore_index=True)

df_total.to_excel("fuel_mix_ercot.xlsx")