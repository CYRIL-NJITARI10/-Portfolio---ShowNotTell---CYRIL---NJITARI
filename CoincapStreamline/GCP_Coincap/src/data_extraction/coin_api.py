import requests
import logging
import pandas as pd
from datetime import datetime, timezone


class CoinAPI:
    def __init__(self, api_key):
        self.api_key = api_key

    def fetch_data(self, url):
        response = requests.get(url)

        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"Error fetching data from {url}: {response.text}")
            raise ResponseError(f"API request failed: {response.status_code}\n{response.text}")

    def get_top_assets_data(self, n):
        url = 'https://api.coincap.io/v2/assets'
        data = self.fetch_data(url)

        ids = []

        for asset in data['data']:
            asset_rank = asset.get("rank")

            if asset_rank is not None and int(asset_rank) <= n:
                asset_id = asset.get("id")
                ids.append(asset_id)

                if len(ids) >= n:
                    break
            else:
                continue

        all_asset_data = {}

        for asset_id in ids:
            history_data = self.get_asset_history(asset_id)
            all_asset_data[asset_id] = history_data

        return all_asset_data

    def get_asset_history(self, asset_id):
        url = f'https://api.coincap.io/v2/assets/{asset_id}/history?interval=d1'
        data = self.fetch_data(url)

        time_prices = {}

        for entry in data['data']:
            time = datetime.fromtimestamp(int(entry.get("time")) // 1000, timezone.utc)
            price = round(float(entry.get("priceUsd")), 3)
            time_prices[time.strftime('%Y-%m-%d')] = price

        return time_prices

    def export_to_csv(self, data, filename_prefix):
        for asset_id, history_data in data.items():
            df = pd.DataFrame(history_data, index=[asset_id])
            df.to_csv(f"{filename_prefix}_{asset_id}.csv", index_label='Date')


class ResponseError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)

"""
coin_api = CoinAPI(api_key)
data = coin_api.get_top_assets_data(10)  # Récupérer les données des 10 meilleurs actifs
coin_api.export_to_csv(data, 'all_assets_data')  # Exporter toutes les données au format CSV
"""

