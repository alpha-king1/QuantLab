import requests
import os
import pandas as pd
from datetime import timedelta

from dotenv import load_dotenv

load_dotenv()

class LoadData:
    def __init__(self):
        self.data = {}

    def _fetch_chunk(self, pair, granularity, start_date, end_date):
            url = (
                f"{os.getenv('BASE_URL')}/"
                f"{os.getenv('OANDA_ACCOUNT_ID')}/"
                f"instruments/{pair}/candles"
            )

            query_params = {
                "granularity": granularity,
                "from": start_date,
                "to": end_date,
            }

            headers = {
                "Authorization": f"Bearer {os.getenv('ACCESS_TOKEN')}",
                "Content-Type": "application/json",
            }

            response = requests.get(
                url,
                headers=headers,
                params=query_params
            )

            response.raise_for_status()

            data = response.json()

            df = pd.json_normalize(data["candles"])

            df = df.rename(columns={
                "time": "time",
                "mid.o": "o",
                "mid.h": "h",
                "mid.l": "l",
                "mid.c": "c",
                "volume": "volume"
            })[
                ["time", "o", "h", "l", "c", "volume"]
            ]

            df[["o", "h", "l", "c"]] = df[
                ["o", "h", "l", "c"]
            ].astype(float)

            df["volume"] = df["volume"].astype(int)

            df["time"] = pd.to_datetime(df["time"])

            df.set_index("time", inplace=True)

            return df

    def get_data(self,pair,granularity,start_date="2015-10-01",end_date="2016-01-01"):

        start = pd.Timestamp(start_date, tz="UTC")
        end = pd.Timestamp(end_date, tz="UTC")

        # Approximate candle duration
        granularity_minutes = {
            "S5": 5 / 60,
            "S10": 10 / 60,
            "S15": 15 / 60,
            "S30": 30 / 60,
            "M1": 1,
            "M2": 2,
            "M4": 4,
            "M5": 5,
            "M10": 10,
            "M15": 15,
            "M30": 30,
            "H1": 60,
            "H2": 120,
            "H3": 180,
            "H4": 240,
            "H6": 360,
            "H8": 480,
            "H12": 720,
            "D": 1440,
            "W": 10080,
        }

        minutes = granularity_minutes[granularity]

        total_minutes = (end - start).total_seconds() / 60

        expected_candles = total_minutes / minutes

        # If under 5000, make one request
        if expected_candles <= 5000:
            df = self._fetch_chunk(
                pair,
                granularity,
                start.isoformat(),
                end.isoformat()
            )

            self.data = df

            return self.data

        # Otherwise split into chunks
        dfs = []

        chunk_duration = timedelta(
            minutes=minutes * 4500
        )

        current_start = start

        while current_start < end:
            current_end = min(
                current_start + chunk_duration,
                end
            )

            print(
                f"Fetching: "
                f"{current_start} -> {current_end}"
            )

            df = self._fetch_chunk(
                pair,
                granularity,
                current_start.isoformat(),
                current_end.isoformat()
            )

            dfs.append(df)

            current_start = current_end

        # Combine everything
        final_df = pd.concat(dfs)

        # Remove duplicate candles
        final_df = final_df[
            ~final_df.index.duplicated(keep="first")
        ]

        # Sort chronologically
        final_df = final_df.sort_index()

        self.data = final_df

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
