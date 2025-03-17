# src/visualize.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_actual_vs_predicted(y_true, y_pred, title, color):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, color=color, edgecolor='black', s=70)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="Perfect Prediction")
    plt.xlabel("Actual Life Expectancy")
    plt.ylabel("Predicted Life Expectancy")
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_future_predictions(model, base_features, years, title, color):
    future_df = pd.DataFrame({"year": years})
    future_df["year_squared"] = future_df["year"] ** 2
    base = pd.DataFrame([base_features] * len(years))
    base["year"] = future_df["year"]
    base["year_squared"] = future_df["year_squared"]

    preds = model.predict(base)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=years, y=preds, marker="o", color=color)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Year")
    plt.ylabel("Predicted Life Expectancy")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
