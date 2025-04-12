import requests
import json
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pytz
import time


class Temperature:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.gridstatus.io/v1/datasets/ercot_temperature_forecast_by_weather_zone/query"

    def fetch_temperature_forecast(self, start_date, end_date):
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        df_total = pd.DataFrame()
        current = start

        while current < end:
            date_start = current.strftime('%Y-%m-%d')
            date_end = (current + relativedelta(months=1)).strftime('%Y-%m-%d')
            print(f"[Temperatura] Baixando: {date_start} → {date_end}")

            url = f"{self.base_url}?api_key={self.api_key}&start_time={date_start}T00:00Z&end_time={date_end}T00:00Z"

            attempts = 0
            while attempts < 5:
                time.sleep(1)  # Aguarda 1 segundo antes de cada requisição

                response = requests.get(url)

                if response.status_code == 200:
                    df = pd.DataFrame(json.loads(response.text)['data'])
                    df['publish_time_utc'] = pd.to_datetime(df['publish_time_utc'])
                    idx = df.groupby('interval_start_utc')['publish_time_utc'].idxmax()
                    max_date_values = df.loc[idx].reset_index(drop=True)
                    df_total = pd.concat([df_total, max_date_values], ignore_index=True)
                    break  # Sai do loop se a requisição for bem-sucedida
                
                elif response.status_code == 403:
                    print(f"🚨 Erro 403: Acesso negado! Tentando novamente ({attempts + 1}/5)...")
                    time.sleep(2 ** attempts)  # Espera progressivamente mais tempo
                    attempts += 1
                else:
                    print(f"❌ Erro {response.status_code} para {date_start}")
                    break  # Para de tentar se for outro erro

            current += relativedelta(months=1)

        if not df_total.empty:
            df_total['interval_start_utc'] = pd.to_datetime(df_total['interval_start_utc'], utc=True)
            texas_tz = pytz.timezone('America/Chicago')
            df_total['interval_start_utc'] = df_total['interval_start_utc'].dt.tz_convert(texas_tz).dt.tz_localize(None)
            df_total['publish_time_utc'] = df_total['publish_time_utc'].dt.tz_localize(None)
            df_total['hour'] = df_total['interval_start_utc'].dt.hour
            df_total['weekday'] = df_total['interval_start_utc'].dt.weekday.apply(lambda x: 1 if x < 5 else 0)
            df_total['avg_temp_fahrenheit'] = df_total[['coast', 'east', 'far_west', 'north', 'north_central', 'south_central', 'southern', 'west']].astype(float).mean(axis=1)

        return df_total

    def fahrenheit_to_celsius(self, df_total):
        colunas_fahrenheit = ['coast', 'east', 'far_west', 'north', 'north_central', 'south_central', 'southern', 'west']
        for col in colunas_fahrenheit:
            df_total[col + '_tc'] = (df_total[col] - 32) * (5/9)
        
        df_total['avg_temp_celsius'] = df_total[[col + '_tc' for col in colunas_fahrenheit]].astype(float).mean(axis=1)
        
        return df_total

    def get_df_temp(self, start_date, end_date):
        df_fahrenheit = self.fetch_temperature_forecast(start_date, end_date)        
        df_celsius = self.fahrenheit_to_celsius(df_fahrenheit.copy())
        
        df_temp = pd.concat([df_fahrenheit.reset_index(drop=True), 
                            df_celsius.filter(regex='_tc$|avg_temp_celsius').reset_index(drop=True)], axis=1)
        
        return df_temp


class Load:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.gridstatus.io/v1/datasets/ercot_load_forecast_by_weather_zone/query"

    def get_df_load(self, start_date, end_date):
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        df_total = pd.DataFrame()
        current = start

        while current < end:
            date_start = current.strftime('%Y-%m-%d')
            date_end = (current + relativedelta(months=1)).strftime('%Y-%m-%d')
            print(f"[Carga] Baixando: {date_start} → {date_end}")

            url = f"{self.base_url}?api_key={self.api_key}&start_time={date_start}T00:00Z&end_time={date_end}T00:00Z"

            attempts = 0
            while attempts < 5:
                time.sleep(1) 

                response = requests.get(url)

                if response.status_code == 200:
                    df = pd.DataFrame(json.loads(response.text)['data'])
                    df_total = pd.concat([df_total, df], ignore_index=True)
                    break  
                
                elif response.status_code == 403:
                    print(f"🚨 Erro 403: Acesso negado! Tentando novamente ({attempts + 1}/5)...")
                    time.sleep(2 ** attempts)
                    attempts += 1
                else:
                    print(f"❌ Erro {response.status_code} para {date_start}")
                    break  # Para de tentar se for outro erro

            current += relativedelta(months=1)

        if not df_total.empty:
            df_total['interval_start_utc'] = pd.to_datetime(df_total['interval_start_utc'], utc=True)
            df_total['publish_time_utc'] = pd.to_datetime(df_total['publish_time_utc'], utc=True)

            texas_tz = pytz.timezone('America/Chicago')
            df_total['interval_start_utc'] = df_total['interval_start_utc'].dt.tz_convert(texas_tz).dt.tz_localize(None)
            df_total['publish_time_utc_load'] = df_total['publish_time_utc'].dt.tz_convert(texas_tz).dt.tz_localize(None)
            df_total['sum_load'] = df_total[['coast', 'east', 'far_west', 'north', 'north_central', 'south_central', 'southern', 'west']].astype(float).sum(axis=1)

        return df_total
