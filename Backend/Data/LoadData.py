import  tpqoa

class LoadData(tpqoa.tpqoa):
    def __init__(self, conf_file):
        self.data = {}
        super().__init__(conf_file)

    def get_data(self, pair, granularity, start_date='2015', end_date='2016'):
        df = self.get_history(pair, start_date, end_date, granularity, 'M')
        return df

    def get_pairs(self):
        pairs = self.get_instruments()
        return pairs
