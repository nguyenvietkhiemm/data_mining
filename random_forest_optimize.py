import numpy as np
from collections import Counter
from scipy.stats import mode # Sử dụng để tối ưu hóa việc bỏ phiếu trong RandomForest

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value # Giá trị dự đoán cho nút lá

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth
                or n_labels == 1  
                or n_samples < self.min_samples_split): 
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        best_feature, best_thresh = self._best_split(X, y, feat_idxs)
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column) 

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr
        
        if best_gain == -1:
             return None, None 

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        if len(y) == 0:
            return 0 
            
        hist = np.bincount(y.astype(int)) 
        ps = hist / len(y)
        ps = ps[ps > 0]
        if len(ps) == 0: 
             return 0
        return -np.sum(ps * np.log2(ps))

    def _most_common_label(self, y):
        if len(y) == 0:
            return None 
        counter = Counter(y)
        if not counter: 
            return None
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        if X.ndim == 1:
            X_processed = X.reshape(1, -1)
        else:
            X_processed = X
        return np.array([self._traverse_tree(x, self.root) for x in X_processed])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        
        num_total_features = X.shape[1]
        if self.n_features is None:
            self.n_features_for_tree = int(np.sqrt(num_total_features))
        else:
            self.n_features_for_tree = min(num_total_features, self.n_features)


        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                n_features=self.n_features_for_tree) 
            
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        if X.ndim == 1:
            X_processed = X.reshape(1, -1)
        else:
            X_processed = X

        tree_predictions = np.array([tree.predict(X_processed) for tree in self.trees])

        if self.n_trees == 1:
            return tree_predictions.flatten()

        try:
           
            valid_predictions = []
            for i in range(tree_predictions.shape[1]): 
                sample_preds = tree_predictions[:, i]
                cleaned_sample_preds = [p for p in sample_preds if p is not None]
                if not cleaned_sample_preds:
                    valid_predictions.append(None) 
                else:
                    counter = Counter(cleaned_sample_preds)
                    valid_predictions.append(counter.most_common(1)[0][0])
            final_predictions = np.array(valid_predictions)

        except ImportError:
            predictions_per_sample = np.swapaxes(tree_predictions, 0, 1)
            final_predictions = np.array([Counter(row_preds[row_preds != np.array(None)]).most_common(1)[0][0]
                                          if len(row_preds[row_preds != np.array(None)]) > 0 else None
                                          for row_preds in predictions_per_sample])
        
        return final_predictions.flatten()