import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('nflx_2014_2023.csv',encoding='latin1')

plt.figure(figsize=(15,10))
sns.lineplot(data=df, x='date',y='close')

plt.show()