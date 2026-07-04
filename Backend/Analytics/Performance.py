import numpy as np
import pandas as pd

class Performance:
    def winloss_rate(self, returns):
        returns['failed'] = returns['Return'] <= 0

        failure_rate = returns['failed'].mean()
        print(f'failure rate: {failure_rate}')

        returns['passed'] = returns['Return'] >= 0

        win_rate = returns['passed'].mean()
        print(f'win rate: {win_rate}')

    def max_drawdown(self, equity):

        peak = np.maximum.accumulate(equity)
        drawdown = equity - peak
        return drawdown.min()

    def get_drawdown(self, equity_curves):
        drawdowns = []

        for equity in equity_curves:
            drawdowns.append(self.max_drawdown(equity))

        print("Worst drawdown:", np.min(drawdowns))
        print("Average drawdown:", np.mean(drawdowns))

    def yearly(self, data, returns, column = 'year'):
        result = []

        for year in sorted(data[column].unique()):

            yearly = data[data[column] == year]
            structure_year = returns[returns[column] == year]

            structure = structure_year['Return']
            baseline = yearly['forward_return']

            if len(structure) < 10:
                continue  # skip tiny samples

            struct_mean = structure.mean()
            base_mean = baseline.mean()
            diff = struct_mean - base_mean

            result.append({
                'year': year,
                'structure_mean': struct_mean,
                'baseline_mean': base_mean,
                'mean_diff': diff,
                'sample_size': len(structure)
            })
        yearly_result = pd.DataFrame(result)
        print(yearly_result)

    def column(self,data, returns, column_name = 'vol_regime'):
        vol_results = []

        for regime in data[column_name].unique():

            subset = data[data[column_name] == regime]
            subset_structure = returns[returns[column_name] == regime]

            structure = subset_structure['Return']
            baseline = subset['forward_return']

            if len(structure) < 10:
                continue

            struct_mean = structure.mean()
            base_mean = baseline.mean()
            diff = struct_mean - base_mean

            vol_results.append({
                'vol_regime': regime,
                'structure_mean': struct_mean,
                'baseline_mean': base_mean,
                'mean_diff': diff,
                'sample_size': len(structure)
            })

        result = pd.DataFrame(vol_results)
        print(result)