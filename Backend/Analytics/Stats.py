from scipy import stats
from scipy.stats import mannwhitneyu
from scipy.stats import ks_2samp
import  numpy as np

class Stats:
    def __init__(self, forward_return):
        self.forward_return = forward_return

    def get_stats(self, returns):
        stat = {
            'total_trades': len(returns),
            f'{self.forward_return} candles forward return mean': returns['Return'].mean(),
            f'{self.forward_return} candles forward return median': returns['Return'].median(),
            f'{self.forward_return} candles forward return std': returns['Return'].std(),
            f'{self.forward_return} candles forward return skew': returns['Return'].skew(),
            'average win': returns['Return'][returns['Return'] >= 0].mean(),
            'average_loss': returns['Return'][returns['Return'] <= 0].mean(),
            'MAE': returns['MAE'].describe(),
            'MFE': returns['MFE'].describe()
        }

        return stat

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
        boot_means = []

        for _ in range(10000):
            sample = np.random.choice(returns['Return'].dropna(),
                                      size=len(returns['Return'].dropna()),
                                      replace=True)
            boot_means.append(np.mean(sample))

        lower = np.percentile(boot_means, 2.5)
        upper = np.percentile(boot_means, 97.5)
        statistical_test = {
            'ks2_test results': p_ks2,
            'MW_test result': p,
            'ttest result': p_value,
            'bootstrap resampling result': lower
        }
        return statistical_test

    def stats(self, returns, data):
        stat = self.get_stats(returns)
        statistical_test = self.statistical_tests(returns, data)
        info = {
            'stat': stat,
            'statistical_test': statistical_test
        }
        return info