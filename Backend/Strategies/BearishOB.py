import pandas as pd

from Backend.Strategies.base import Strategy


class BearishOB(Strategy):
    def __init__(self, ):
        self.model = None
        self.x = None

    def run(self, data, forward_return = 10, displacement_mult=2.0, model = False):
        """
            data: data with ['o', 'h', 'l', 'c', 'ATR_14']
            type: either bullish or bearish
            displacement_mult: How much stronger the move must be than the OB candle to count.
            forward_window: How many candles to look ahead for return after a hit.
        """
        obs = []
        active_zones = []
        for i in range(1, len(data) - 1):
            curr = data.iloc[i]
            prev = data.iloc[i - 1]
            if curr['c'] < curr['o'] and (curr['body']) > (prev['body']) * displacement_mult:
                if prev['c'] > prev['o']:
                    active_zones.append({
                        'type': 'Bearish',
                        'top': prev['c'],
                        'bottom': prev['o'],
                        'created_time': data['time'].iloc[i],
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
                    if curr['h'] >= zone['bottom'] >= curr['c'] and i - zone['created_at'] > forward_return:
                        hit = True

                if hit:
                    if model:
                        current_features = \
                        {
                            'impulse_candle_size':data['body'].shift(-1).iloc[zone['created_at']],
                            'created_hour': data.index[zone['created_at']].hour,
                            'distance': (data['time'].iloc[i] - data.index[zone['created_at']]).total_seconds()/3600,
                            'hour_hit': data.index[i].hour,
                            'Zone_Top':zone['top'],
                            'Zone_Bottom':zone['bottom'],
                            'zone_size': zone['top'] - zone['bottom'],
                            # 'Created_At': data.index[zone['created_at']],
                            'atr_14': data['ATR_14'].iloc[i],
                            'vol_regime_num': data['vol_regime_num'].iloc[i],
                            'sessions_num': data['sessions_num'].iloc[i],
                            'Hit_at': data.index[i],
                            'l':data['l'].iloc[i],
                            'c':data['c'].iloc[i],
                            'o':data['o'].iloc[i],
                            'h':data['h'].iloc[i],
                            'ATR_14': data['ATR_14'].iloc[i]
                        }
                        features = pd.DataFrame([current_features])

                        x_live_aligned = features.reindex(columns=self.x.columns)
                        prob_success = self.model.predict_proba(x_live_aligned)[0][1]
                        ret = None
                        # 3. THE ML FILTER: Only take the trade if probability is > 60% (or your chosen threshold)
                        if prob_success > 0.55:
                            future_idx = min(i + forward_return, len(data) - 1)
                            future_price = data.iloc[future_idx]['c']
                            ret = ((curr['c'] - future_price) / curr['c'])
                            trade_taken = True
                        else:
                            trade_taken = False
                    else:
                        future_idx = min(i + forward_return, len(data) - 1)
                        future_price = data.iloc[future_idx]['c']
                        ret = ((curr['c'] - future_price) / curr['c'])
                        trade_taken = True

                    if trade_taken:
                        obs.append({
                            'year': data['year'].iloc[i],
                            'MFE':data['highest'].iloc[i],
                            'MAE':data['lowest'].iloc[i],
                            'Type': zone['type'],
                            'Created_At': data.index[zone['created_at']],
                            'impulse_candle_size':data['body'].shift(-1).iloc[zone['created_at']],
                            'created_hour':data.index[zone['created_at']].hour,
                            'time': data['time'].iloc[i],
                            'distance': (data['time'].iloc[i] - data.index[zone['created_at']]).total_seconds()/3600,
                            # distance_in_hours = time_diff.total_seconds() / 3600.0
                            'vol_regime': data['vol_regime'].iloc[i],
                            'sessions': data['sessions'].iloc[i],
                            'Hit_At': data.index[i],
                            'l':data['l'].iloc[i],
                            'c':data['c'].iloc[i],
                            'o':data['o'].iloc[i],
                            'h':data['h'].iloc[i],
                            'hour_hit': data.index[i].hour,
                            'Return': ret,
                            'success': 1 if(ret > 0) else 0,
                            'Zone_Top': zone['top'],
                            'Zone_Bottom': zone['bottom'],
                            'zone_size':zone['top'] - zone['bottom'],
                            'atr_14': data['ATR_14'].iloc[i],
                            'sessions_num': data['sessions_num'].iloc[i],
                            'vol_regime_num': data['vol_regime_num'].iloc[i],
                        })
                        zone['status'] = 'Mitigated'  # Mark as done

        return pd.DataFrame(obs)

