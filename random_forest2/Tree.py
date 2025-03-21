import numpy as np

def gini_impurity(y):
    classes = np.unique(y)  #获取唯一类别
    n = len(y)
    gini = 1.0
    for c in classes:
        p = np.sum(y == c) / n  #计算类别比例
        gini -= p ** 2
    return gini

class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature    #分裂特征（内部节点）
        self.threshold = threshold#分裂阈值（内部节点）
        self.left = left          #左子树（<=阈值）
        self.right = right        #右子树（>阈值）
        self.value = value        #叶子节点的预测值

def build_tree(X, y, max_depth=5, min_samples_split=2):#特征矩阵,标签向量,最大深度和最小样本分割数
    #终止条件1：样本数过少或纯度足够高
    if len(y) < min_samples_split or gini_impurity(y) < 1e-6:
        return DecisionNode(value=np.argmax(np.bincount(y)))

    #终止条件2：达到最大深度
    if max_depth == 0:
        return DecisionNode(value=np.argmax(np.bincount(y)))

    best_gini = float('inf')#初始化最好基尼不纯度为无穷大
    best_feature, best_threshold = None, None

    #遍历所有特征
    for feature in range(X.shape[1]):
        #取当前特征的所有唯一值作为候选阈值
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            #根据阈值划分数据集
            left_idx = X[:, feature] <= threshold
            y_left = y[left_idx]
            y_right = y[~left_idx]

            #跳过无效分裂（子节点为空）
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            #计算分裂后的加权基尼不纯度
            gini_left = gini_impurity(y_left)
            gini_right = gini_impurity(y_right)
            gini = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)

            #更新最佳分裂点
            if gini < best_gini:
                best_gini = gini
                best_feature = feature
                best_threshold = threshold

    #找到有效分裂时返回叶子节点
    if best_gini == float('inf'):
        return DecisionNode(value=np.argmax(np.bincount(y)))

    #递归构建左右子树
    left_idx = X[:, best_feature] <= best_threshold
    left = build_tree(X[left_idx], y[left_idx], max_depth - 1, min_samples_split)
    right = build_tree(X[~left_idx], y[~left_idx], max_depth - 1, min_samples_split)

    return DecisionNode(feature=best_feature, threshold=best_threshold, left=left, right=right)