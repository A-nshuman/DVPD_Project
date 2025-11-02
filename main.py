import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

trainStartDate = "2020-01-01"
trainEndDate = "2024-01-01"
ticker = '^NSEI'
dfTrain = yf.download(ticker, start=trainStartDate, end=trainEndDate)
valsSeries = dfTrain['Close'].values
valsSeries = valsSeries.tolist()

dates = dfTrain.index
actualVals = list(zip(valsSeries, dates))
Yt_list = [
    (round(price_list[0], 2), timestamp)  
    for price_list, timestamp in actualVals
]
print(Yt_list[0])

def plotChart(vals, predictedVals):
  dates_arr1 = [item[1] for item in predictedVals]
  values_arr1 = [item[0] for item in predictedVals]
  dates_arr2 = [item[1] for item in vals]
  values_arr2 = [item[0] for item in vals]

  diff_arr = []
  diff_dates_arr = []
  for i in range(len(values_arr1)):
    diff_arr.append(values_arr2[i] - values_arr1[i])
    diff_dates_arr.append(dates_arr1[i])

  plt.figure(figsize=(12, 7))
  plt.plot(dates_arr1, values_arr1, label='Predicted Values', marker='', linestyle='-', linewidth=2, color='red')
  plt.plot(dates_arr2, values_arr2, label='Actual Values', marker='', linestyle='-', linewidth=2, color='blue')
  plt.plot(diff_dates_arr, diff_arr, label='Error', marker='', linestyle='-', linewidth=2, color='green')
  plt.title(f'Comparison of {ticker} and Model Predictions Over Time')
  plt.xlabel('Date')
  plt.ylabel('Value (Price)')
  plt.legend()
  plt.grid(True, linestyle=':', alpha=0.6)
  plt.xticks(rotation=45, ha='right')
  plt.tight_layout()
  plt.show()
def armaModel(vals, alpha):
  predictions = []
  phi = 0.5
  theta = 0.5
  predError = 0
  prevError = 0
  i = 1
  while i < len(vals):
    predictedVal = (phi * vals[i-1][0]) + (theta * predError)
    predError = vals[i][0] - predictedVal
    phi = phi + alpha * predError * vals[i-1][0]
    theta = theta + alpha * predError * predError
    predictions.append((round(predictedVal, 2), vals[i][1]))
    prevError = predError
    i = i + 1
  return predictions
def mae(arr1, arr2, testStartPoint):
  sum = 0
  testSize = pd.to_datetime(trainEndDate) - pd.to_datetime(testStartPoint)
  for i in range(testSize.days, len(arr2)):
    sum += abs(arr1[i][0] - arr2[i][0])
  return round(sum / len(arr2), 2)

size = 900
vals = Yt_list[:size]
predictedVals = armaModel(vals, 5e-11)
i = 0
vals = Yt_list[1:size]
errorVal = mae(vals, predictedVals, "2023-01-01")
print(f'MAE: {errorVal}')
plotChart(vals, predictedVals)
