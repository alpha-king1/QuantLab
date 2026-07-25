from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier # Added this just in case
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, roc_curve
import numpy as np
import pandas as pd

def clean_for_ml(dat):
    dat = dat.replace([np.inf, -np.inf], np.nan)
    dat = dat.fillna(dat.median())

    return dat


class Model:
    def __init__(self):
        self.x = None
        self.model = None
        self.x_train = None
        self.best_model_name = None

    def train_model(self, returns, strategy_name):
        x=None
        if strategy_name == 'bullish_ob' or 'bearish_ob':
            x = returns.drop(columns=['success','MFE', 'MAE', 'year', 'Type', 'time', 'Hit_At', 'Created_At', 'vol_regime', 'Return', 'failed', 'passed', 'sessions', 'vol_regime', 'vol_regime_num', 'sessions_num'])
            self.x = x

        y = returns['success']

        split_idx = int(len(self.x) * 0.7)
        x_train, x_test = x.iloc[:split_idx], x.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        self.x_train = x_train

        x_train_ml = clean_for_ml(x_train)
        x_test_ml = clean_for_ml(x_test)
        y_train_ml = clean_for_ml(y_train)
        y_test_ml = clean_for_ml(y_test)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=10000),
            "Decision Tree": DecisionTreeClassifier(max_depth=4),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5)
        }

        best = {}
        best_auc = -1
        best_trading_model = None
        model_info = []
        for name, clf in models.items():
            # Fit on training data
            clf.fit(x_train_ml, y_train_ml)

            # Get hard predictions (0 or 1)
            preds = clf.predict(x_test_ml)

            # Get probability scores (needed for AUC-ROC)
            probs = clf.predict_proba(x_test_ml)[:, 1]

            # Calculate all metrics
            acc = accuracy_score(y_test_ml, preds)
            prec = precision_score(y_test_ml, preds, zero_division=0)
            rec = recall_score(y_test_ml, preds, zero_division=0)
            f1 = f1_score(y_test_ml, preds, zero_division=0)
            auc = roc_auc_score(y_test_ml, probs)
            best[name] = auc

            model_info.append({
                'name': name,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'auc': auc,
            })


            # fpr, tpr, _ = roc_curve(y_test_ml, probs)
            # plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
            if auc > best_auc:
                best_auc = auc
                best_trading_model = clf
                self.model = best_trading_model
                self.best_model_name = name

        return model_info

    def show_importance(self):
        importance = None
        if hasattr(self.model, "feature_importances_"):
            importance = pd.Series(
                self.model.feature_importances_,
                index=self.x.columns
            ).sort_values(ascending=False)

        elif hasattr(self.model, "coef_"):
            importance = pd.Series(
                np.abs(self.model.coef_[0]),
                index=self.x.columns
            ).sort_values(ascending=False)

        else:
            print("This model does not support feature importance.")
        return importance.to_dict()
