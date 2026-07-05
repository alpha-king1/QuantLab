from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier # Added this just in case
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
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

        # FINAL CHECK: This must print 0
        print("Remaining NaNs:", x_train_ml.isnull().sum().sum())

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
        best_trading_model = None
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
                best_trading_model = clf
                self.model = best_trading_model


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
                  feature_names=x_train_ml.columns,
                  class_names=['Fail', 'Success'],
                  filled=True, fontsize=10)
        plt.show()
        return best_trading_model

    def show_importance(self):
        importances = pd.Series(self.model.feature_importances_, index=self.x_train.columns)
        importances.sort_values().plot(kind='barh')
        plt.title("What actually matters for Gold OBs?")
        plt.show()
