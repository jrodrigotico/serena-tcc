import requests
import json
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytz

api_key = '45bd4693a487a8ad3a2a2dd40a1bc30d'

df_total = None

for month in range(12):

    date_start = (datetime(2024, 10, 1) - relativedelta(months=month+1)).strftime('%Y-%m-%d')
    date_end   = (datetime(2024, 10, 1) - relativedelta(months=month)).strftime('%Y-%m-%d')
    
    print(date_start)
    
    url = f"https://api.gridstatus.io/v1/datasets/ercot_temperature_forecast_by_weather_zone/query?api_key={api_key}&start_time={date_start}T00:00Z&end_time={date_end}T00:00Z"
    
    
    response = requests.request("GET", url)
    
    
    df = pd.DataFrame(json.loads(response.text)['data'])
    df['publish_time_utc'] = pd.to_datetime(df['publish_time_utc'])
    
    idx = df.groupby('interval_start_utc')['publish_time_utc'].idxmax()
    max_date_values = df.loc[idx].reset_index(drop=True)
    
    df_total = pd.concat([df_total,max_date_values], ignore_index=True)

df_adj = df_total
df_adj['interval_start_utc'] = pd.to_datetime(df_adj['interval_start_utc'], utc=True)

# Convert to Texas timezone
texas_tz = pytz.timezone('America/Chicago')
df_adj['interval_start_utc'] = df_adj['interval_start_utc'].dt.tz_convert(texas_tz)
df_adj['interval_start_utc'] = df_adj['interval_start_utc'].dt.tz_localize(None)
df_adj['publish_time_utc'] = df_adj['publish_time_utc'].dt.tz_localize(None)







