import pandas as pd
import numpy as np
import tpqoa
from scipy import stats
from scipy.stats import mannwhitneyu
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier # Added this just in case
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

class SupremeBacktester(tpqoa.tpqoa):
    """
        This is a very comprehensive script, used to backtest across alot of strategies, it just runs and give results
        all the user does is to input a comprehensive timeseries dataset that with the following columns:o, h, l, and c
    """
    def __init__(self, data, conf_file, forward_return = 10):
        super().__init__(conf_file)
        self.data = data.copy()
        self.model = None
        self.x = None
        self.forward_return = forward_return
        self.data['highest'] = (self.data['h'].shift(-self.forward_return).rolling(self.forward_return).max()-self.data['c'])/self.data['c']
        self.data['lowest'] = (self.data['l'].shift(-self.forward_return).rolling(self.forward_return).min()-self.data['c'])/self.data['c']

        self.data['body'] = (self.data['c'] - self.data['o']).abs()

        self.data['upper_wick'] = self.data['h'] - self.data[['o', 'c']].max(axis=1)

        self.data['lower_wick'] = self.data[['o', 'c']].min(axis=1) - self.data['l']

        self.data['total_range'] = self.data['h'] - self.data['l']

        self.data['hour'] = self.data.index.hour

        self.data['forward_return'] = (self.data['c'].shift(-self.forward_return) - self.data['c'])/self.data['c']

        self.data['year'] = self.data.index.year

        self.data['time'] = self.data.index

        self.data['true_range'] = np.maximum.reduce([self.data['h'] - self.data['l'],
                                                          self.data['h'] - self.data['c'].shift(periods=1).abs(),
                                                          self.data['l'] - self.data['c'].shift(periods=1).abs()
                                                          ])

        self.data['direction'] = np.sign(self.data['c'] - self.data['o'])

    def bullish_ob(self, displacement_mult=2.0, model = False):
        """
        data: data with ['o', 'h', 'l', 'c', 'ATR_14']
        type: either bullish or bearish
        displacement_mult: How much stronger the move must be than the OB candle to count.
        forward_window: How many candles to look ahead for return after a hit.
        """
        obs = []
        active_zones = []

        for i in range(1, len(self.data) - 1):
            curr = self.data.iloc[i]
            prev = self.data.iloc[i - 1]
            # --- 1. IDENTIFY NEW ORDER BLOCKS ---
            # Bullish OB: Last Bearish candle before a strong Bullish move
            if curr['c'] > curr['o'] and (curr['body']) > (prev['body']) * displacement_mult and \
                    prev['body'] < prev['ATR_14']:
                if prev['c'] < prev['o']:
                    active_zones.append({
                        'type': 'Bullish',
                        'top': prev['o'],
                        'bottom': prev['c'],
                        'created_at': i,
                        'created_time': self.data['time'].iloc[i],
                        'status': 'Active'
                    })

            for zone in active_zones:
                if zone['status'] != 'Active': continue

                # Check for INVALIDATION (Body Close through zone)
                if zone['type'] == 'Bullish' and curr['c'] < zone['bottom']:
                    zone['status'] = 'Invalidated'
                    continue
                hit = False
                if zone['type'] == 'Bullish':
                    # Low enters zone, but Close stays above bottom
                    if curr['l'] <= zone['top'] <= curr['c'] and i - zone['created_at'] > self.forward_return:
                        hit = True
                if hit:
                    if model:
                        current_features = \
                        {
                            'impulse_candle_size':self.data['body'].shift(-1).iloc[zone['created_at']],
                            'created_hour': self.data.index[zone['created_at']].hour,
                            'distance': (self.data['time'].iloc[i] - self.data.index[zone['created_at']]).total_seconds()/3600,
                            'hour_hit': self.data.index[i].hour,
                            'Zone_Top':zone['top'],
                            'Zone_Bottom':zone['bottom'],
                            'zone_size': zone['top'] - zone['bottom'],
                            # 'Created_At': self.data.index[zone['created_at']],
                            'atr_14': self.data['ATR_14'].iloc[i],
                            'vol_regime_num': self.data['vol_regime_num'].iloc[i],
                            'sessions_num': self.data['sessions_num'].iloc[i],
                            'Hit_at': self.data.index[i],
                            'l':self.data['l'].iloc[i],
                            'c':self.data['c'].iloc[i],
                            'o':self.data['o'].iloc[i],
                            'h':self.data['h'].iloc[i],
                            'ATR_14': self.data['ATR_14'].iloc[i]
                        }
                        features = pd.DataFrame([current_features])

                        x_live_aligned = features.reindex(columns=self.x.columns)
                        prob_success = self.model.predict_proba(x_live_aligned)[0][1]

                        # 3. THE ML FILTER: Only take the trade if probability is > 60% (or your chosen threshold)
                        if prob_success > 0.55:
                            future_idx = min(i + self.forward_return, len(self.data) - 1)
                            future_price = self.data.iloc[future_idx]['c']
                            ret = ((future_price - curr['c']) / curr['c']) if zone['type'] == 'Bullish' else (
                                        curr['c'] - future_price)
                            trade_taken = True
                        else:
                            trade_taken = False
                    else:
                        # Capture the Return
                        future_idx = min(i + self.forward_return, len(self.data) - 1)
                        future_price = self.data.iloc[future_idx]['c']
                        ret = ((future_price - curr['c']) / curr['c']) if zone['type'] == 'Bullish' else (
                                    curr['c'] - future_price)
                        trade_taken = True
                # if zone['type'] == 'Bullish':
                #     # Low enters zone, but Close stays above bottom
                #     if curr['l'] <= (zone['top'] + zone['bottom'])/2 and i - zone['created_at'] > self.forward_return:
                #         hit = True
                # if hit:
                #     # Capture the Return
                #     future_idx = min(i + self.forward_return, len(self.data) - 1)
                #     future_price = self.data.iloc[future_idx]['c']
                #     ret = ((future_price - ((zone['top'] + zone['bottom'])/2)) / ((zone['top'] + zone['bottom'])/2)) if zone['type'] == 'Bullish' else (
                #                 curr['c'] - future_price)
                    if trade_taken:
                        obs.append({
                            'year': self.data['year'].iloc[i],
                            'MFE':self.data['highest'].iloc[i],
                            'MAE':self.data['lowest'].iloc[i],
                            'Type': zone['type'],
                            'Created_At': self.data.index[zone['created_at']],
                            'impulse_candle_size':self.data['body'].shift(-1).iloc[zone['created_at']],
                            'created_hour':self.data.index[zone['created_at']].hour,
                            'time': self.data['time'].iloc[i],
                            'distance': (self.data['time'].iloc[i] - self.data.index[zone['created_at']]).total_seconds()/3600,
                            # distance_in_hours = time_diff.total_seconds() / 3600.0
                            'vol_regime': self.data['vol_regime'].iloc[i],
                            'sessions': self.data['sessions'].iloc[i],
                            'Hit_At': self.data.index[i],
                            'l':self.data['l'].iloc[i],
                            'c':self.data['c'].iloc[i],
                            'o':self.data['o'].iloc[i],
                            'h':self.data['h'].iloc[i],
                            'hour_hit': self.data.index[i].hour,
                            'Return': ret,
                            'success': 1 if(ret > 0) else 0,
                            'Zone_Top': zone['top'],
                            'Zone_Bottom': zone['bottom'],
                            'zone_size':zone['top'] - zone['bottom'],
                            'atr_14': self.data['ATR_14'].iloc[i],
                            'sessions_num': self.data['sessions_num'].iloc[i],
                            'vol_regime_num': self.data['vol_regime_num'].iloc[i],
                        })
                        zone['status'] = 'Mitigated'  # Mark as done

        return pd.DataFrame(obs)

    def bearish_ob(self, displacement_mult=2.0, model = False):
        """
            data: data with ['o', 'h', 'l', 'c', 'ATR_14']
            type: either bullish or bearish
            displacement_mult: How much stronger the move must be than the OB candle to count.
            forward_window: How many candles to look ahead for return after a hit.
        """
        obs = []
        active_zones = []
        for i in range(1, len(self.data) - 1):
            curr = self.data.iloc[i]
            prev = self.data.iloc[i - 1]
            if curr['c'] < curr['o'] and (curr['body']) > (prev['body']) * displacement_mult:
                if prev['c'] > prev['o']:
                    active_zones.append({
                        'type': 'Bearish',
                        'top': prev['c'],
                        'bottom': prev['o'],
                        'created_time': self.data['time'].iloc[i],
                        'created_at': i,
                        'status': 'Active'
                    })

            # --- 2. INSPECT ACTIVE ZONES ---
            for zone in active_zones:
                if zone['status'] != 'Active': continue
                if zone['type'] == 'Bearish' and curr['c'] > zone['top']:
                    zone['status'] = 'Invalidated'
                    continue

                hit = False
                if zone['type'] == 'Bearish':  # Bearish
                    if curr['h'] >= zone['bottom'] >= curr['c'] and i - zone['created_at'] > self.forward_return:
                        hit = True

                if hit:
                    if model:
                        current_features = \
                        {
                            'impulse_candle_size':self.data['body'].shift(-1).iloc[zone['created_at']],
                            'created_hour': self.data.index[zone['created_at']].hour,
                            'distance': (self.data['time'].iloc[i] - self.data.index[zone['created_at']]).total_seconds()/3600,
                            'hour_hit': self.data.index[i].hour,
                            'Zone_Top':zone['top'],
                            'Zone_Bottom':zone['bottom'],
                            'zone_size': zone['top'] - zone['bottom'],
                            # 'Created_At': self.data.index[zone['created_at']],
                            'atr_14': self.data['ATR_14'].iloc[i],
                            'vol_regime_num': self.data['vol_regime_num'].iloc[i],
                            'sessions_num': self.data['sessions_num'].iloc[i],
                            'Hit_at': self.data.index[i],
                            'l':self.data['l'].iloc[i],
                            'c':self.data['c'].iloc[i],
                            'o':self.data['o'].iloc[i],
                            'h':self.data['h'].iloc[i],
                            'ATR_14': self.data['ATR_14'].iloc[i]
                        }
                        features = pd.DataFrame([current_features])

                        x_live_aligned = features.reindex(columns=self.x.columns)
                        prob_success = self.model.predict_proba(x_live_aligned)[0][1]

                        # 3. THE ML FILTER: Only take the trade if probability is > 60% (or your chosen threshold)
                        if prob_success > 0.55:
                            future_idx = min(i + self.forward_return, len(self.data) - 1)
                            future_price = self.data.iloc[future_idx]['c']
                            ret = ((curr['c'] - future_price) / curr['c'])
                            trade_taken = True
                        else:
                            trade_taken = False
                    else:
                        future_idx = min(i + self.forward_return, len(self.data) - 1)
                        future_price = self.data.iloc[future_idx]['c']
                        ret = ((curr['c'] - future_price) / curr['c'])
                        trade_taken = True

                    if trade_taken:
                        obs.append({
                            'year': self.data['year'].iloc[i],
                            'MFE':self.data['highest'].iloc[i],
                            'MAE':self.data['lowest'].iloc[i],
                            'Type': zone['type'],
                            'Created_At': self.data.index[zone['created_at']],
                            'impulse_candle_size':self.data['body'].shift(-1).iloc[zone['created_at']],
                            'created_hour':self.data.index[zone['created_at']].hour,
                            'time': self.data['time'].iloc[i],
                            'distance': (self.data['time'].iloc[i] - self.data.index[zone['created_at']]).total_seconds()/3600,
                            # distance_in_hours = time_diff.total_seconds() / 3600.0
                            'vol_regime': self.data['vol_regime'].iloc[i],
                            'sessions': self.data['sessions'].iloc[i],
                            'Hit_At': self.data.index[i],
                            'l':self.data['l'].iloc[i],
                            'c':self.data['c'].iloc[i],
                            'o':self.data['o'].iloc[i],
                            'h':self.data['h'].iloc[i],
                            'hour_hit': self.data.index[i].hour,
                            'Return': ret,
                            'success': 1 if(ret > 0) else 0,
                            'Zone_Top': zone['top'],
                            'Zone_Bottom': zone['bottom'],
                            'zone_size':zone['top'] - zone['bottom'],
                            'atr_14': self.data['ATR_14'].iloc[i],
                            'sessions_num': self.data['sessions_num'].iloc[i],
                            'vol_regime_num': self.data['vol_regime_num'].iloc[i],
                        })
                        zone['status'] = 'Mitigated'  # Mark as done

        return pd.DataFrame(obs)

    def baseline_stat(self, number):
        baseline_1bar = (
            self.data
            .dropna()
            .groupby(['sessions', 'vol_regime'])['forward_return_'+str(number)+'bar']
        )
        baseline_stats = baseline_1bar.agg(
            mean='mean',
            median='median',
            std='std',
            skew='skew'
        )
        return baseline_stats

    def outcome_table(self, structure_name, baseline_stat, structure_baseline_stat):
        outcome_table_bullish_impulse = structure_baseline_stat.join(
        baseline_stat,
        lsuffix='_'+structure_name,
        rsuffix='_base'
        )

        # Add deltas
        outcome_table_bullish_impulse['mean_shift'] = (
            outcome_table_bullish_impulse['mean_'+structure_name] - outcome_table_bullish_impulse['mean_base']
        )

        outcome_table_bullish_impulse['skew_shift'] = (
            outcome_table_bullish_impulse['skew_'+structure_name] - outcome_table_bullish_impulse['skew_base']
        )

        outcome_table_bullish_impulse['std_ratio'] = (
            outcome_table_bullish_impulse['std_'+structure_name] / outcome_table_bullish_impulse['std_base']
        )

        return outcome_table_bullish_impulse

    def distribution_plot(self, name, session, vol_regime, full_data, structure_data):
        ctx = (
            (self.data['sessions'] == session) &
            (self.data['vol_regime'] == vol_regime)
        )

        plt.figure(figsize=(8,5))

        plt.hist(
            full_data.loc[ctx, 'forward_return_3bar'],
            bins=100,
            alpha=0.5,
            label='Baseline',
            density=True
        )

        plt.hist(
            structure_data.loc[ctx, 'forward_return_3bar'],
            bins=100,
            alpha=0.5,
            label= name,
            density=True
        )

        plt.axvline(0, color='black', linestyle='--')
        plt.legend()
        plt.title('3-Bar Forward Return Distribution\n'+session+' + '+vol_regime+' Vol')
        plt.show()

    def stats(self, returns):
        print(f'{self.forward_return} candles forward return mean: {returns['Return'].mean()}')
        print(f'{self.forward_return} candles forward return median: {returns['Return'].median()}')
        print(f'{self.forward_return} candles forward return std: {returns['Return'].std()}')
        print(f'{self.forward_return} candles forward return skew: {returns['Return'].skew()}')
        print(f'average win: {returns['Return'][returns['Return'] >= 0].mean()}')
        print(f'average_loss: {returns['Return'][returns['Return'] <= 0].mean()}')
        print(f'{returns['MAE'].describe()}')
        print(f'{returns['MFE'].describe()}')

    def ttest(self, returns):
        t_stat, p_value = stats.ttest_ind(
            returns['Return'],
            self.data['forward_return'],
            equal_var=False,
            nan_policy='omit'
        )
        print(f'ttest result: {p_value}')

    def MW_test(self, returns):
        stat, p = mannwhitneyu(returns['Return'].dropna(), self.data['forward_return'].dropna(),
                               alternative='two-sided')
        print(f'MW_test result: {p}')

    def ks2_test(self, returns):
        stat, p = ks_2samp(returns['Return'].dropna(), self.data['forward_return'].dropna())
        print(f'ks2_test results: {p}')

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

    def winloss_rate(self, returns):
        returns['failed'] = returns['Return'] <= 0

        failure_rate = returns['failed'].mean()
        print(f'failure rate: {failure_rate}')

        returns['passed'] = returns['Return'] >= 0

        win_rate = returns['passed'].mean()
        print(f'win rate: {win_rate}')

    def yearly(self, column, returns):
        result = []

        for year in sorted(self.data[column].unique()):

            yearly = self.data[self.data[column] == year]
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

    def column(self, column_name, returns):
        vol_results = []

        for regime in self.data[column_name].unique():

            subset = self.data[self.data[column_name] == regime]
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

    def time_control(self):
        '''
            this method is used to set the hourly timeframe and which sessions each timeframe are being acted.
        :return:
        '''
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

    def volatility(self, number=20):
        '''
        this method is to add the volatility column and you can edit mean range by the number param
        :param number: the number is used to get the amounts used to  calculate the range or volatility.
        :return: nothing
        '''
        self.data[f'volatility_{number}'] = self.data['total_range'].rolling(number).mean()

    def daily_range(self, number = 20):
        '''
        this method is used to add the daily range column in a dataset and can only be used with timeframe below daily timeframe
        :input: the amount of dailyh roll up required
        :return: creates the daily high, daily low, daily range, and rolling daily range based on the number inputed
        '''
        daily = self.data.resample('D')

        self.data['daily_high'] = daily['h'].transform('max')
        self.data['daily_low'] = daily['l'].transform('min')

        self.data['daily_range'] = self.data['daily_high'] - self.data['daily_low']
        self.data[f'rolling_daily_range_{number}'] = self.data['daily_range'].rolling(number).mean()

    def ATR(self, number = 20):
        '''
        this is to get the atr of the dataframe, and user can input the numbers of data to use to get the atr
        :param number: input number of rolling true range needed, or just one number
        :return: creates a column where with ATR(number)
        '''
        for n in [number]:
            self.data[f'ATR_{n}'] = self.data['true_range'].rolling(n).mean()

    def vol_regime(self):
        self.data['atr_norm'] = self.data['ATR_14'] / self.data['ATR_14'].rolling(252).mean()

        self.data['vol_regime'] = pd.qcut(
            self.data['atr_norm'],
            q=[0, 0.33, 0.66, 1.0],
            labels=['Low', 'Medium', 'High']
        )

    def arrange_data(self):
        self.time_control()
        self.volatility()
        self.ATR(14)
        self.daily_range()
        self.vol_regime()
        session_map = {'asian': 1, 'asian/london': 2, 'london': 3,'london/NY': 4, 'NY': 5, 'closing': 6}
        self.data['sessions_num'] = self.data['sessions'].map(session_map)

        regime_map = {'Low': 1, 'Medium': 2, 'High': 3} # Adjust based on your actual labels
        self.data['vol_regime_num'] = self.data['vol_regime'].map(regime_map)

    def analyse_strategy(self):
        returned = self.bullish_ob()
        self.stats(returned)
        self.ttest(returned)
        self.MW_test(returned)
        self.ks2_test(returned)
        self.bootstrap_resampling(returned)
        self.yearly('year', returned)
        self.column('vol_regime', returned)
        self.winloss_rate(returned)
        self.train_model(returned, 'bearish_ob')
        returned = self.bullish_ob(model=True)
        self.stats(returned)
        self.ttest(returned)
        self.MW_test(returned)
        self.ks2_test(returned)
        self.bootstrap_resampling(returned)
        self.yearly('year', returned)
        self.column('vol_regime', returned)
        self.winloss_rate(returned)

    def clean_for_ml(self, dat):
        dat = dat.replace([np.inf, -np.inf], np.nan)
        dat = dat.fillna(dat.median())

        return dat

    def train_model(self, returns, strategy_name):
        if strategy_name == 'bullish_ob' or 'bearish_ob':
            x = returns.drop(columns=['success','MFE', 'MAE', 'year', 'Type', 'time', 'Hit_At', 'Created_At', 'vol_regime', 'Return', 'failed', 'passed', 'sessions', 'vol_regime', 'vol_regime_num', 'sessions_num'])
            self.x = x

        y = returns['success']

        split_idx = int(len(x) * 0.7)
        X_train, X_test = x.iloc[:split_idx], x.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        X_train_ml = self.clean_for_ml(X_train)
        X_test_ml = self.clean_for_ml(X_test)
        y_train_ml = self.clean_for_ml(y_train)
        y_test_ml = self.clean_for_ml(y_test)

        # FINAL CHECK: This must print 0
        print("Remaining NaNs:", X_train_ml.isnull().sum().sum())

        # 1. Initialize the models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(max_depth=4),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5)
        }

        # 2. Train and Evaluate
        plt.figure(figsize=(10, 7))  # For the AUC-ROC plot
        best = {}
        best_auc = -1
        for name, clf in models.items():
            # Fit on training data
            clf.fit(X_train_ml, y_train_ml)

            # Get hard predictions (0 or 1)
            preds = clf.predict(X_test_ml)

            # Get probability scores (needed for AUC-ROC)
            probs = clf.predict_proba(X_test_ml)[:, 1]

            # Calculate all metrics
            acc = accuracy_score(y_test_ml, preds)
            prec = precision_score(y_test_ml, preds, zero_division=0)
            rec = recall_score(y_test_ml, preds, zero_division=0)
            f1 = f1_score(y_test_ml, preds, zero_division=0)
            auc = roc_auc_score(y_test_ml, probs)
            best[name] = auc

            print(f"--- {name} ---")
            print(f"Accuracy:  {acc:.2f}")
            print(f"Precision: {prec:.2f}  <-- (Success rate when it signals)")
            print(f"Recall:    {rec:.2f}  <-- (Percentage of all winners we caught)")
            print(f"F1-Score:  {f1:.2f}  <-- (Overall filter quality)")
            print(f"AUC-ROC:   {auc:.2f}  <-- (Overall predictive power)")
            print("\n")


            # Add this model's curve to the plot
            fpr, tpr, _ = roc_curve(y_test_ml, probs)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
            if auc > best_auc:
                best_auc = auc
                self.model = clf

        best_model = max(best, key=best.get)
        print(f"Best model: {best_model} with AUC of {best[best_model]:.2f}")
        # Finalize the AUC-ROC plot
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate (Traps)')
        plt.ylabel('True Positive Rate (Winners)')
        plt.title('ROC Curve Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 3. Visualize the Decision Tree (The "Why")
        plt.figure(figsize=(20, 10))
        plot_tree(models["Decision Tree"],
                  feature_names=X_train_ml.columns,
                  class_names=['Fail', 'Success'],
                  filled=True, fontsize=10)
        plt.show()
