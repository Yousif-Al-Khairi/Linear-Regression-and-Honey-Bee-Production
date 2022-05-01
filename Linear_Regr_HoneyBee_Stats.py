#Import all required python libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_gradient_at_b(x, y, b, m):
  N = len(x)
  diff = 0
  for i in range(N):
    x_val = x[i]
    y_val = y[i]
    diff += (y_val - ((m * x_val) + b))
  b_gradient = -(2/N) * diff  
  return b_gradient

def get_gradient_at_m(x, y, b, m):
  N = len(x)
  diff = 0
  for i in range(N):
      x_val = x[i]
      y_val = y[i]
      diff += x_val * (y_val - ((m * x_val) + b))
  m_gradient = -(2/N) * diff  
  return m_gradient

def step_gradient(b_current, m_current, x, y, learning_rate):
    b_gradient = get_gradient_at_b(x, y, b_current, m_current)
    m_gradient = get_gradient_at_m(x, y, b_current, m_current)
    b = b_current - (learning_rate * b_gradient)
    m = m_current - (learning_rate * m_gradient)
    return [b, m]
  
def gradient_descent(x, y, learning_rate, num_iterations):
  b = 0
  m = 0
  for i in range(num_iterations):
    b,m = step_gradient(b, m, x, y, learning_rate)
  return [b,m]



df = pd.read_csv('honeyproduction.csv')

prod_per_year = df.groupby('year').mean().reset_index()

X = prod_per_year['year']

X = X.values.reshape(-1, 1)

y = prod_per_year['totalprod']

b, m = gradient_descent(X, y, 0.01, 1000)

y_predictions = [m*x + b for x in X]

plt.plot(X,y,'o')

plt.plot(X,y_predictions)

plt.xlabel("Years")

plt.ylabel("HoneyBee Production Per Year (Lbs)")

plt.show()


