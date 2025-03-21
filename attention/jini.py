import numpy as np

def gini_impurity(y):
    classes = np.unique(y)  # 获取唯一类别
    n = len(y)
    gini = 1.0
    for c in classes:
        p = np.sum(y == c) / n  # 计算类别比例
        gini -= p ** 2
    return gini