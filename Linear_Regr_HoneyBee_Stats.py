#Import all required python libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('honeyproduction.csv')

prod_per_year = df.groupby('year').mean().reset_index()

X = prod_per_year['year']
X = X.values.reshape(-1, 1)
y = prod_per_year['totalprod']

plt.scatter(X,y)

plt.show()


