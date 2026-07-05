import  tpqoa

from Backend.Config.settings import CONFIG


class LoadData(tpqoa.tpqoa):
    def __init__(self):
        self.data = {}
        super().__init__(CONFIG)

    def get_data(self, pair, granularity, start_date='2015', end_date='2016'):
        df = self.get_history(pair, start_date, end_date, granularity, 'M')
        return df

    def get_pairs(self):
        pairs = self.get_instruments()
        return pairs
