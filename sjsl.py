import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt


# 决策树实现
class DecisionNode:
    """决策树节点"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # 分裂特征索引（内部节点）
        self.threshold = threshold  # 分裂阈值（内部节点）
        self.left = left  # 左子树（<=阈值）
        self.right = right  # 右子树（>阈值）
        self.value = value  # 叶子节点的预测值


# 基尼不纯度计算函数
def gini_impurity(y):
    classes = np.unique(y)  # 获取唯一类别
    n = len(y)
    gini = 1.0
    for c in classes:
        p = np.sum(y == c) / n
        gini -= p ** 2
    return gini


# 寻找最佳分裂特征和阈值
def find_best_split(X, y, max_features):  # 考虑特征子集随机选择，构建`max_features`参数
    best_gini = float('inf')
    best_feature, best_threshold = None, None
    zong_features = X.shape[1]

    # 随机选择特征子集
    features = np.random.choice(zong_features, max_features, replace=False)

    for feature in features:
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_idx = X[:, feature] <= threshold
            if np.sum(left_idx) == 0 or np.sum(~left_idx) == 0:
                continue  # 跳过无效分裂

            # 计算加权基尼不纯度
            gini_left = gini_impurity(y[left_idx])
            gini_right = gini_impurity(y[~left_idx])
            weighted_gini = (np.sum(left_idx) * gini_left + np.sum(~left_idx) * gini_right) / len(y)

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold


# 决策树构建(递归)
def build_tree(X, y, max_depth=5, min_samples_split=2, max_features='sqrt'):
    # 终止条件(数量过少、纯度够高或深度已经最神）
    if len(y) < min_samples_split or gini_impurity(y) < 1e-6 or max_depth == 0:
        return DecisionNode(value=np.argmax(np.bincount(y)))

    # 确定max_features数值
    if max_features == 'sqrt':
        max_features = int(np.sqrt(X.shape[1]))
    elif max_features == 'log2':
        max_features = int(np.log2(X.shape[1]))

    # 寻找最佳分裂
    feature, threshold = find_best_split(X, y, max_features)
    if feature is None:
        return DecisionNode(value=np.argmax(np.bincount(y)))

    # 递归构建子树
    left_idx = X[:, feature] <= threshold
    left = build_tree(X[left_idx], y[left_idx], max_depth - 1, min_samples_split, max_features)
    right = build_tree(X[~left_idx], y[~left_idx], max_depth - 1, min_samples_split, max_features)

    return DecisionNode(feature=feature, threshold=threshold, left=left, right=right)


#单棵树预测
def predict_tree(tree, x):
    if tree.value is not None:
        return tree.value
    if x[tree.feature] <= tree.threshold:
        return predict_tree(tree.left, x)
    else:
        return predict_tree(tree.right, x)


#随机森林实现
class RandomForest:
    def __init__(self, n_estimators=100, max_depth=5, max_features='sqrt'):
        self.n_estimators = n_estimators#树数
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
        self.oob_indices_list = []

    #Bootstrap采样并记录OOB索引
    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        oob_indices = np.setdiff1d(np.arange(n_samples), indices)#算差
        return X[indices], y[indices], oob_indices

    #训练随机森林
    def fit(self, X, y):
        self.trees = []
        self.oob_indices_list = []

        for _ in range(self.n_estimators):
            X_sample, y_sample, oob_indices = self.bootstrap_sample(X, y)
            tree = build_tree(X_sample, y_sample,
                              max_depth=self.max_depth,
                              max_features=self.max_features)
            self.trees.append(tree)#调用末尾添加元素函数
            self.oob_indices_list.append(oob_indices)


   #预测（多数投票）
    def predict(self, X):
        preds = np.zeros((len(self.trees), len(X)))
        for i, tree in enumerate(self.trees):
            preds[i] = [predict_tree(tree, x) for x in X]

        # 将预测结果转换为整数类型
        preds = preds.astype(int)

        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds)#多数投票


    #计算OOB误差
    def oob_error(self, X, y):
        n_samples = X.shape[0]
        oob_preds = [[] for _ in range(n_samples)]

        for i, (tree, oob_indices) in enumerate(zip(self.trees, self.oob_indices_list)):
            for idx in oob_indices:
                pred = predict_tree(tree, X[idx])
                oob_preds[idx].append(pred)

        #误差计算
        errors = 0
        valid_samples = 0
        for idx in range(n_samples):
            if len(oob_preds[idx]) > 0:
                majority = Counter(oob_preds[idx]).most_common(1)[0][0]
                if majority != y[idx]:
                    errors += 1
                valid_samples += 1

        return errors / valid_samples if valid_samples > 0 else 0.0


    #计算特征重要性
    def feature_importances_(self, X, y):
        importances = np.zeros(X.shape[1])
        for tree in self.trees:
            self._accumulate_feature_importance(tree, X, y, importances)
        return importances / len(self.trees)

    def _accumulate_feature_importance(self, node, X, y, importances):
        if node.feature is not None:  # 只有内部节点才有特征
            left_idx = X[:, node.feature] <= node.threshold
            right_idx = ~left_idx
            gini_before_split = gini_impurity(np.concatenate((y[left_idx], y[right_idx])))
            gini_left = gini_impurity(y[left_idx])
            gini_right = gini_impurity(y[right_idx])
            importance = gini_before_split - (gini_left * np.sum(left_idx) + gini_right * np.sum(right_idx)) / len(y)
            importances[node.feature] += importance
            # 递归处理子树
            if node.left is not None:
                self._accumulate_feature_importance(node.left, X[left_idx], y[left_idx], importances)
            if node.right is not None:
                self._accumulate_feature_importance(node.right, X[right_idx], y[right_idx], importances)


#示例使用
if __name__ == "__main__":
    # 加载数据
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练单棵决策树
    single_tree = build_tree(X_train, y_train, max_depth=3, max_features='sqrt')
    y_pred_tree = [predict_tree(single_tree, x) for x in X_test]
    accuracy_tree = np.mean(y_pred_tree == y_test)

    # 训练随机森林
    rf = RandomForest(n_estimators=100, max_depth=3, max_features='sqrt')
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    accuracy_rf = np.mean(y_pred_rf == y_test)

    # 计算OOB误差
    oob_error_rate = 1-rf.oob_error(X_train, y_train)

    feature_importances = rf.feature_importances_(X_train, y_train)
    indices = np.argsort(feature_importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), feature_importances[indices],
            color="r", align="center")
    plt.xticks(range(X_train.shape[1]), iris.feature_names)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

    # 打印结果
    print(f"【单棵决策树】测试集准确率: {accuracy_tree:.4f}")
    print(f"【随机森林】测试集准确率: {accuracy_rf:.4f}")
    print(f"【随机森林】OOB误差: {oob_error_rate:.4f}")