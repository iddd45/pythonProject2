import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# 给予整数的索引
X = df.iloc[0:150, [0, 2]].values
plt.scatter(X[0:50, 0], X[:50, 1], c='blue', marker='x', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], c='red', marker='o', label='versicolor')
plt.scatter(X[100:150, 0], X[100:150, 1], c='green', marker='*', label='virginica')
plt.xlabel('petal width')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.show()

