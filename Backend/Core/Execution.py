import  pandas as pd
from matplotlib import pyplot as plt


class Execution:
    def __init__(self):
        self.equity = []

    def execute_trade(self, data, capital = 1000):
        balance = capital
        self.equity=[]
        self.equity.append({
            'time': data.time.iloc[0],
            'balance': round(balance)
        })
        for i in range(1, len(data)):
            balance = balance + (data.Return.iloc[i-1] * balance)
            self.equity.append({
                'time': data.time.iloc[i],
                'balance': round(balance)
            })
        return {
            'capital': capital,
            'balance': balance
        }

    def plot_equity(self):
        one_percentage_equity_curve = pd.DataFrame(self.equity)
        one_percentage_equity_curve.index = one_percentage_equity_curve.time
        one_percentage_equity_curve['balance'].plot(figsize=(15, 6), title='Equity Curve', grid=True)
        plt.show()

    def execute_trades(self, data, capital = 1000):
        trade_info = self.execute_trade(data, capital)
        equity_curve = self.plot_equity()

