import requests
import os
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

class LoadData:
    def __init__(self):
        self.data = {}

    def get_data(self, pair, granularity, start_date='2015-10-01', end_date='2016-01-01'):
        url = f"{os.getenv('BASE_URL')}/{os.getenv('OANDA_ACCOUNT_ID')}/instruments/{pair}/candles"

        query_params = {
            "granularity": granularity,
            "from": start_date,
            "to": end_date
        }

        headers = {
            "Authorization": f"Bearer {os.getenv('ACCESS_TOKEN')}",
            "Content-Type": "application/json",
        }

        response = requests.get(url, headers=headers, params=query_params)

        data = response.json()
        df = pd.json_normalize(data['candles'])

        df = df.rename(columns={
            'time': 'time',
            'mid.o': 'o',
            'mid.h': 'h',
            'mid.l': 'l',
            'mid.c': 'c',
            'volume': 'volume'
        })[['time', 'o', 'h', 'l', 'c', 'volume']]

        df[['o', 'h', 'l', 'c']] = df[['o', 'h', 'l', 'c']].astype(float)
        df['volume'] = df['volume'].astype(int)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        self.data = df
        return self.data

    def get_pairs(self):
        url = f'{os.getenv('BASE_URL')}/{os.getenv('OANDA_ACCOUNT_ID')}/instruments'
        headers = {
            "Authorization": f"Bearer {os.getenv('ACCESS_TOKEN')}",
            "Content-Type": "application/json",
        }
        response = requests.get(url, headers=headers)
        response = response.json()
        instrument_names = [item["name"] for item in response["instruments"]]
        return instrument_names
