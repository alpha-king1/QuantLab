import pandas as pd

from Backend.Strategies.base import Strategy


class BullishOB(Strategy):
    def __init__(self):
        self.model = None
        self.x = None

    def run(self,data, forward_return = 10, displacement_mult=2.0, model = False):
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
                        'created_time': data['time'].iloc[i],
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
                    if curr['l'] <= zone['top'] <= curr['c'] and i - zone['created_at'] > forward_return:
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
                            ret = ((future_price - curr['c']) / curr['c']) if zone['type'] == 'Bullish' else (
                                        curr['c'] - future_price)
                            trade_taken = True
                        else:
                            trade_taken = False
                    else:
                        # Capture the Return
                        future_idx = min(i + forward_return, len(data) - 1)
                        future_price = data.iloc[future_idx]['c']
                        ret = ((future_price - curr['c']) / curr['c']) if zone['type'] == 'Bullish' else (
                                    curr['c'] - future_price)
                        trade_taken = True
                # if zone['type'] == 'Bullish':
                #     # Low enters zone, but Close stays above bottom
                #     if curr['l'] <= (zone['top'] + zone['bottom'])/2 and i - zone['created_at'] > self.forward_return:
                #         hit = True
                # if hit:
                #     # Capture the Return
                #     future_idx = min(i + self.forward_return, len(data) - 1)
                #     future_price = data.iloc[future_idx]['c']
                #     ret = ((future_price - ((zone['top'] + zone['bottom'])/2)) / ((zone['top'] + zone['bottom'])/2)) if zone['type'] == 'Bullish' else (
                #                 curr['c'] - future_price)
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
