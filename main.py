import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('nflx_2014_2023.csv',encoding='latin1')
plt.figure(figsize=(12,8))

def fitLM(xcol, ycol):
    x = df[[xcol]]
    y = df[ycol]
    model = LinearRegression()
    model.fit(x, y)
    r2 = model.score(x,y)
    return model, r2

LM, r2 = fitLM('ema_100', 'close')

print(f'R2 Fit Parameter: {r2}')
print(f'y = {LM.coef_[0]}x + {LM.intercept_}')