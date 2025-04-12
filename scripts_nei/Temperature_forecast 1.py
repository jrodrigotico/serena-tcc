import requests
import json
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytz

api_key = '45bd4693a487a8ad3a2a2dd40a1bc30d'

df_total = None
month = 0
for month in range(12):

    date_start = (datetime(2023, 2, 1) - relativedelta(months=month+1)).strftime('%Y-%m-%d')
    date_end   = (datetime(2024, 1, 1) - relativedelta(months=month)).strftime('%Y-%m-%d')
    
    print(date_start)
    
    url = f"https://api.gridstatus.io/v1/datasets/ercot_standardized_hourly/query?api_key={api_key}&start_time={date_start}T00:00Z&end_time={date_end}T00:00Z"
    
    
    response = requests.request("GET", url)
    
    
    df = pd.DataFrame(json.loads(response.text)['data'])
    
    df_total = pd.concat([df_total,df], ignore_index=True)

# df_total.to_excel("load_forecast_1.xlsx")


