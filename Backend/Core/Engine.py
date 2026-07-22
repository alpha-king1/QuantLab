from Backend.Analytics.Performance import Performance
from Backend.Analytics.Stats import Stats
from Backend.Core.Execution import Execution
from Backend.Data.LoadData import LoadData
from Backend.Data.Preprocess import Preprocess
from Backend.Models.Train_model import Model
from Backend.Strategies.registry import STRATEGIES
import math


class Analyse:
    def __init__(self, pair, granularity,user_strategy, forward_return = 10, start_date='2015-11', end_date='2016-01', capital=100000):
        self.pair = pair
        self.granularity = granularity
        self.start_date = start_date
        self.end_date = end_date
        self.capital = capital
        self.user_strategy = user_strategy
        self.forward_return = forward_return
        self.dataModule = LoadData()
        self.data =  self.dataModule.get_data(self.pair, self.granularity, self.start_date, self.end_date)
        self.preprocessed = Preprocess(self.data, forward_return)
        self.performance = Performance()
        self.execution = Execution()
        self.strategy_class  = STRATEGIES[user_strategy]
        self.strategy = self.strategy_class()
        self.stats = Stats(forward_return)
        self.modelClass = Model()
        self.model = None

    def run_engine(self):
        information = []
        processed_data = self.preprocessed.preprocess_data()
        strategy_return = self.strategy.run(processed_data)
        stats = self.stats.stats(strategy_return, processed_data)
        performance = self.performance.get_performance(strategy_return, processed_data)
        trade = self.execution.execute_trade(strategy_return, self.capital)
        models_info = self.modelClass.train_model(strategy_return, self.user_strategy)
        model_importance = self.modelClass.show_importance()
        self.strategy.x = self.modelClass.x
        self.strategy.model = self.modelClass.model
        filtered_returns = self.strategy.run(processed_data, model = True)
        model_filtered_stats = self.stats.stats(filtered_returns, processed_data)
        model_filtered_performance = self.performance.get_performance(filtered_returns, processed_data)
        model_filtered_execution = self.execution.execute_trade(filtered_returns, 10000)

        data = {
                'stats': stats,
                'performance': performance,
                'trade_evaluation': trade,
                'model_importance': model_importance,
                'model_filtered':
                    {
                        'stats': model_filtered_stats,
                        'performance': model_filtered_performance,
                        'trade_evaluation': model_filtered_execution
                    }
                }

        information.append(self.clean_nans(data))
        return information

    def get_pairs(self):
        return self.dataModule.get_pairs()

    def clean_nans(self, obj):
        if isinstance(obj, float) and math.isnan(obj):
            return None  # Serializes to `null` in JSON (or return 0.0)
        elif isinstance(obj, dict):
            return {k: self.clean_nans(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.clean_nans(v) for v in obj]
        return obj


if __name__ == '__main__':
    yoo = Analyse("XAU_USD", "H1", 'bullish_ob', start_date='2020-10-01', end_date='2021-01-01')
    yoo.run_engine()