from Backend.Analytics.Performance import Performance
from Backend.Analytics.Stats import Stats
from Backend.Core.Execution import Execution
from Backend.Data.LoadData import LoadData
from Backend.Data.Preprocess import Preprocess
from Backend.Models.Train_model import Model
from Backend.Strategies.registry import STRATEGIES
from fastapi.encoders import jsonable_encoder



class Analyse:
    def __init__(self, pair, granularity,user_strategy, forward_return = 10, start_date='2015-11', end_date='2016-01'):
        self.pair = pair
        self.granularity = granularity
        self.start_date = start_date
        self.end_date = end_date
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
        # self.execution.execute_trade(strategy_return, 10000)
        # self.execution.plot_equity()
        # self.model = self.modelClass.train_model(strategy_return, self.user_strategy)
        # self.modelClass.show_importance()
        # self.strategy.x = self.modelClass.x
        # self.strategy.model = self.model
        # filtered_returns = self.strategy.run(processed_data, model = True)
        # self.stats.get_stats(filtered_returns)
        # self.stats.statistical_tests(filtered_returns, processed_data)
        # self.stats.bootstrap_resampling(filtered_returns)
        # self.performance.winloss_rate(filtered_returns)
        # self.performance.yearly(processed_data, filtered_returns)
        # self.performance.column(processed_data, filtered_returns)
        # self.execution.execute_trade(filtered_returns, 10000)
        # self.execution.plot_equity()
        information.append({'stats': stats})
        information.append({'performance': performance})

        return information

    def get_pairs(self):
        return self.dataModule.get_pairs()

if __name__ == '__main__':
    yoo = Analyse("XAU_USD", "H1", 'bearish_ob')
    yoo.run_engine()