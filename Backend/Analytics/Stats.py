from scipy import stats
from scipy.stats import mannwhitneyu
from scipy.stats import ks_2samp
import  numpy as np

class Stats:
    def __init__(self, forward_return):
        self.forward_return = forward_return

    def get_stats(self, returns):
        print(f'total trades: {len(returns)}')

        print(f'{self.forward_return} candles forward return mean: {returns['Return'].mean()}')

        print(f'{self.forward_return} candles forward return median: {returns['Return'].median()}')

        print(f'{self.forward_return} candles forward return std: {returns['Return'].std()}')

        print(f'{self.forward_return} candles forward return skew: {returns['Return'].skew()}')

        print(f'average win: {returns['Return'][returns['Return'] >= 0].mean()}')

        print(f'average_loss: {returns['Return'][returns['Return'] <= 0].mean()}')

        print(f'{returns['MAE'].describe()}')

        print(f'{returns['MFE'].describe()}')

    def statistical_tests(self, returns, data):
        t_stat, p_value = stats.ttest_ind(
            returns['Return'],
            data['forward_return'],
            equal_var=False,
            nan_policy='omit'
        )
        stat, p = mannwhitneyu(returns['Return'].dropna(), data['forward_return'].dropna(),
                               alternative='two-sided')
        stat, p_ks2 = ks_2samp(returns['Return'].dropna(), data['forward_return'].dropna())
        print(f'ks2_test results: {p_ks2}')
        print(f'MW_test result: {p}')
        print(f'ttest result: {p_value}')

    def bootstrap_resampling(self, returns):
        boot_means = []

        for _ in range(10000):
            sample = np.random.choice(returns['Return'].dropna(),
                                      size=len(returns['Return'].dropna()),
                                      replace=True)
            boot_means.append(np.mean(sample))

        lower = np.percentile(boot_means, 2.5)
        upper = np.percentile(boot_means, 97.5)
        print(f'bootstrap resampling result: {lower}')