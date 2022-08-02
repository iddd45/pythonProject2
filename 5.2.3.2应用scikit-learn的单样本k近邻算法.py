import numpy as np
import pandas as pd
from sklearn import datasets

# 加载iris数据集合
scikit_iris = datasets.load_iris()
# 转换为pandas的DataFrame
pd_iris = pd.DataFrame(
    data=np.c_[scikit_iris['data'], scikit_iris['target']],
    columns=np.append(scikit_iris.feature_names, ['y'])
)

x = pd_iris[scikit_iris.feature_names]
y = pd_iris['y']

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
# (2)拟合模型
knn.fit(x, y)
# (3)预测新数据
knn.predict([[3, 2, 5, 6]])
print(knn.predict([[3, 2, 5, 6]]))