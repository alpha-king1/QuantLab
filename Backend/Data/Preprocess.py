import  numpy as np
import pandas as pd

class Preprocess:
    def __init__(self, datas, forward_return):
        self.data = datas.copy()
        self.forward_return = forward_return

    def preprocess_data(self):
        self.data['highest'] = (self.data['h'].shift(-self.forward_return).rolling(self.forward_return).max() -
                                self.data['c']) / self.data['c']
        self.data['lowest'] = (self.data['l'].shift(-self.forward_return).rolling(self.forward_return).min() -
                               self.data['c']) / self.data['c']
        self.data['body'] = (self.data['c'] - self.data['o']).abs()
        self.data['upper_wick'] = self.data['h'] - self.data[['o', 'c']].max(axis=1)
        self.data['lower_wick'] = self.data[['o', 'c']].min(axis=1) - self.data['l']
        self.data['total_range'] = self.data['h'] - self.data['l']
        self.data['hour'] = self.data.index.hour
        self.data['forward_return'] = (self.data['c'].shift(-self.forward_return) - self.data['c']) / self.data['c']
        self.data['year'] = self.data.index.year
        self.data['time'] = self.data.index
        self.data['true_range'] = np.maximum.reduce([self.data['h'] - self.data['l'],
                                                     self.data['h'] - self.data['c'].shift(periods=1).abs(),
                                                     self.data['l'] - self.data['c'].shift(periods=1).abs()
                                                     ])
        self.data['direction'] = np.sign(self.data['c'] - self.data['o'])
        self.data['ATR_14'] = self.data['true_range'].rolling(14).mean()
        self.data['atr_norm'] = self.data['ATR_14'] / self.data['ATR_14'].rolling(252).mean()

        self.data['vol_regime'] = pd.qcut(
            self.data['atr_norm'],
            q=[0, 0.33, 0.66, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        daily = self.data.resample('D')

        self.data['daily_high'] = daily['h'].transform('max')
        self.data['daily_low'] = daily['l'].transform('min')

        self.data['daily_range'] = self.data['daily_high'] - self.data['daily_low']
        self.data[f'rolling_daily_range_{self.forward_return}'] = self.data['daily_range'].rolling(self.forward_return).mean()
        conditions = [
            self.data['hour'].between(0, 8),
            self.data['hour'].between(8, 9),
            self.data['hour'].between(9, 13),
            self.data['hour'].between(13, 17),
            self.data['hour'].between(17, 22),
            self.data['hour'].between(22, 23)
        ]

        choices = [
            'asian',
            'asian/london',
            'london',
            'london/NY',
            'NY',
            'Closing'
        ]

        self.data['sessions'] = np.select(conditions, choices, default='off')
        session_map = {'asian': 1, 'asian/london': 2, 'london': 3, 'london/NY': 4, 'NY': 5, 'closing': 6}
        self.data['sessions_num'] = self.data['sessions'].map(session_map)
        regime_map = {'Low': 1, 'Medium': 2, 'High': 3}  # Adjust based on your actual labels
        self.data['vol_regime_num'] = self.data['vol_regime'].map(regime_map)

        return self.data

