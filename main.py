import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Load data and define features
df = pd.read_csv('nflx_2014_2023.csv', encoding='latin1')
plt.figure(figsize=(12, 8))

# Define features and target column name
x_col_names = ['rsi_7','rsi_14','cci_7','cci_14','sma_50','ema_50','sma_100','ema_100','macd','bollinger','TrueRange','atr_7','atr_14']
TARGET_COL = 'next_day_close'

# --- 1. Chronological Split ---
split_point = int(len(df) * 0.8)

X = df[x_col_names]
Y = df[TARGET_COL]

X_train = X.iloc[:split_point]
Y_train = Y.iloc[:split_point]

X_test = X.iloc[split_point:]
Y_test = Y.iloc[split_point:]

# ----------------------------------------------------
# 2. Refactor: Define a cleaner fit function (optional, but good practice)
# ----------------------------------------------------
def train_linear_model(X_data, Y_data):
    """Trains a Linear Regression model using provided X and Y data."""
    model = LinearRegression()
    model.fit(X_data, Y_data)
    return model

# ----------------------------------------------------
# 3. Train the Model and Test
# ----------------------------------------------------
# TRAIN the model on the split TRAINING data
LM = train_linear_model(X_train, Y_train)

# Calculate R2 on the TRAINING set (to check for near-perfect fit)
r2_train = LM.score(X_train, Y_train)

# Generate predictions on the UNSEEN TEST data
Y_pred_test = LM.predict(X_test)

# Calculate R2 on the TESTING set (to check for generalization)
r2_test = r2_score(Y_test, Y_pred_test)

# Calculate MAE (Mean Absolute Error) for the actual dollar error
mae_test = mean_absolute_error(Y_test, Y_pred_test)

print("--- Model Results ---")
print(f'R2 (Training Set): {r2_train:.4f} (High value is an overfitting concern)')
print(f'R2 (Testing Set):  {r2_test:.4f} (This is the true measure of performance)')
print(f"MAE (Test Set Error): ${mae_test:.2f} (Average dollar error)")

# Print coefficients for insight
print('\nCoefficients (Feature Weights):')
for name, coef in zip(x_col_names, LM.coef_):
    print(f"  {name}: {coef:.2f}")

# Print a snippet of test predictions
print('\nFirst 5 Test Predictions:')
print(Y_pred_test[:5]) 