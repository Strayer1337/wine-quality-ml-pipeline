import numpy as np


class DecisionTree:
    """Custom Decision Tree dùng Gini impurity, implement bằng NumPy."""

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        prob = counts / counts.sum()
        return 1 - np.sum(prob ** 2)

    def split(self, X, y, feature, threshold):
        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold
        return X[left_idx], X[right_idx], y[left_idx], y[right_idx]

    def best_split(self, X, y):
        best_gini = float("inf")
        best_feature = None
        best_threshold = None

        _, n_features = X.shape

        for feature in range(n_features):
            for threshold in np.unique(X[:, feature]):
                X_left, X_right, y_left, y_right = self.split(X, y, feature, threshold)

                if len(X_left) == 0 or len(X_right) == 0:
                    continue

                weighted_gini = (
                    len(y_left) * self.gini(y_left)
                    + len(y_right) * self.gini(y_right)
                ) / len(y)

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def majority_class(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def build_tree(self, X, y, depth):
        if len(np.unique(y)) == 1:
            return y[0]
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return self.majority_class(y)

        feature, threshold = self.best_split(X, y)
        if feature is None:
            return self.majority_class(y)

        X_left, X_right, y_left, y_right = self.split(X, y, feature, threshold)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": self.build_tree(X_left, y_left, depth + 1),
            "right": self.build_tree(X_right, y_right, depth + 1),
        }

    def fit(self, X, y):
        self.tree = self.build_tree(X, y, 0)

    def predict_one(self, x, node):
        if not isinstance(node, dict):
            return node
        if x[node["feature"]] <= node["threshold"]:
            return self.predict_one(x, node["left"])
        return self.predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])
