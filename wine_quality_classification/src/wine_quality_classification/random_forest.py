import numpy as np


class DecisionTreeRF:
    """Decision Tree với random feature selection, dùng làm base learner cho Random Forest."""

    def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
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
        feature_indices = np.random.choice(n_features, size=self.max_features, replace=False)

        for feature in feature_indices:
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


class RandomForest:
    """Random Forest: ensemble của nhiều DecisionTreeRF với bootstrap sampling."""

    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def bootstrap_sample(self, X, y):
        indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        if self.max_features is None:
            self.max_features = int(np.sqrt(X.shape[1]))

        for _ in range(self.n_trees):
            tree = DecisionTreeRF(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
            )
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Shape: (n_trees, n_samples)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        # Majority vote theo từng sample
        final_preds = []
        for col in range(tree_preds.shape[1]):
            votes = tree_preds[:, col]
            values, counts = np.unique(votes, return_counts=True)
            final_preds.append(values[np.argmax(counts)])

        return np.array(final_preds)
