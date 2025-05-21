from sklearn.datasets import load_wine, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import time
import numpy as np
from decesiontree import build_tree, global_predict
from decesiontree_original import build_tree as build_tree_original, global_predict as global_predict_original
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

# Import C++ wrapper (decision_tree.py as provided previously)
# from decision_tree import DecisionTree

import ctypes
import numpy as np
import os
import sys

# Thay đổi đường dẫn này cho phù hợp với file DLL/SO của bạn
if sys.platform == "win32":
    LIB_PATH = os.path.join(os.path.dirname(__file__), "decision_tree.dll")
else:
    LIB_PATH = os.path.join(os.path.dirname(__file__), "libdecision_tree.so")


# Load thư viện động
_dtree = ctypes.CDLL(LIB_PATH)

# Định nghĩa các kiểu dữ liệu trả về và tham số cho các hàm
# typedef void* DecisionTreeHandle;
DecisionTreeHandle = ctypes.c_void_p

# DLL_EXPORT DecisionTreeHandle dt_create(int max_depth, int min_size);
_dtree.dt_create.argtypes = [ctypes.c_int, ctypes.c_int]
_dtree.dt_create.restype = DecisionTreeHandle

# DLL_EXPORT void dt_destroy(DecisionTreeHandle handle);
_dtree.dt_destroy.argtypes = [DecisionTreeHandle]
_dtree.dt_destroy.restype = None

# DLL_EXPORT void dt_fit(DecisionTreeHandle handle, double* X, int* y, int n_samples, int n_features);
_dtree.dt_fit.argtypes = [
    DecisionTreeHandle,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.c_int,
]
_dtree.dt_fit.restype = None

# DLL_EXPORT void dt_predict(DecisionTreeHandle handle, double* X, int n_samples, int n_features, int* out);
_dtree.dt_predict.argtypes = [
    DecisionTreeHandle,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
]
_dtree.dt_predict.restype = None

# DLL_EXPORT void dt_feature_importances(DecisionTreeHandle handle, double* out_importances, int n_features);
_dtree.dt_feature_importances.argtypes = [
    DecisionTreeHandle,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
]
_dtree.dt_feature_importances.restype = None


class DecisionTree:
    def __init__(self, max_depth=5, min_size=1):
        self.handle = _dtree.dt_create(max_depth, min_size)
        self._n_features = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        n_samples, n_features = X.shape
        self._n_features = n_features

        X_ctypes = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        y_ctypes = y.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        _dtree.dt_fit(self.handle, X_ctypes, y_ctypes, n_samples, n_features)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        out = np.zeros(n_samples, dtype=np.int32)

        X_ctypes = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        out_ctypes = out.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        _dtree.dt_predict(self.handle, X_ctypes, n_samples,
                          n_features, out_ctypes)
        return out

    def feature_importances_(self):
        if self._n_features is None:
            raise RuntimeError(
                "You must call fit() before feature_importances_()")
        out = np.zeros(self._n_features, dtype=np.float64)
        out_ctypes = out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        _dtree.dt_feature_importances(
            self.handle, out_ctypes, self._n_features)
        return out

    def __del__(self):
        if hasattr(self, "handle") and self.handle:
            _dtree.dt_destroy(self.handle)
            self.handle = None

class DecisionTreeWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=5, min_size=1, is_optimized=True):
        self.max_depth = max_depth
        self.min_size = min_size
        self.is_optimized = is_optimized
        self.tree = None
        
    def fit(self, X, y):
        if self.is_optimized:
            train_data = np.column_stack((X, y))
            self.tree = build_tree(train_data, self.max_depth, self.min_size)
        else:
            train_data = [list(X[i]) + [y[i]] for i in range(len(y))]
            self.tree = build_tree_original(train_data, self.max_depth, self.min_size)
        return self
        
    def predict(self, X):
        if self.is_optimized:
            return global_predict(self.tree, X)
        else:
            return global_predict_original(self.tree, X)

class DecisionTreeCppWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=5, min_size=1):
        self.max_depth = max_depth
        self.min_size = min_size
        self.clf = DecisionTree(max_depth, min_size)
        self._n_features = None

    def fit(self, X, y):
        # print(X.shape, y.shape)
        X = X.astype(np.float64, copy=False)
        y = y.astype(np.int32, copy=False)
        self.clf.fit(X, y)
        
        self._n_features = X.shape[1]
        return self

    def predict(self, X):
        return self.clf.predict(X)

    @property
    def feature_importances_(self):
        return self.clf.feature_importances_()

def main():
    # Load Wine dataset
    wine = load_wine()
    X = wine.data
    y = wine.target
        
    # Generate synthetic data
    X_syn, y_syn = make_classification(n_samples=10000, 
                                     n_features=13, 
                                     n_classes=3,
                                     n_informative=5,  # Tăng số đặc trưng thông tin
                                     n_clusters_per_class=2,
                                     random_state=42)
    
    # Combine datasets
    X = np.vstack((X, X_syn))
    y = np.concatenate((y, y_syn))
    
    print("\nDistribution of classes in combined dataset:")
    for i in range(len(np.unique(y))):
        count = np.sum(y == i)
        percentage = (count / len(y)) * 100
        print(f"Class {i}: {count} samples ({percentage:.1f}%)")
    
    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # C++ ctypes version
    print("\n[C++ ctypes Version]")
    print("Running cross-validation for C++ ctypes version...")
    clf_cpp = DecisionTreeCppWrapper(max_depth=7, min_size=4)
    start_time = time.time()
    scores_cpp = cross_val_score(clf_cpp, X, y, cv=skf, n_jobs=1)
    end_time = time.time()
    
    print(f"- Build time: {end_time - start_time:.4f} seconds")
    print(f"- Cross-validation scores: {scores_cpp}")
    print(f"- Mean accuracy: {scores_cpp.mean()*100:.2f}%")
    print(f"- Std accuracy: {scores_cpp.std()*100:.2f}%")
    
    # Our implementation
    print("\n[Optimized Version]")
    print("Running cross-validation for optimized version...")
    clf_opt = DecisionTreeWrapper(max_depth=7, min_size=4, is_optimized=True)
    
    start_time = time.time()
    scores_opt = cross_val_score(clf_opt, X, y, cv=skf)
    end_time = time.time()
    
    print(f"- Build time: {end_time - start_time:.4f} seconds")
    print(f"- Cross-validation scores: {scores_opt}")
    print(f"- Mean accuracy: {scores_opt.mean()*100:.2f}%")
    print(f"- Std accuracy: {scores_opt.std()*100:.2f}%")
    
    
    # Scikit-learn implementation
    print("\n[Scikit-learn Version]")
    print("Running cross-validation for scikit-learn version...")
    clf_sk = DecisionTreeClassifier(max_depth=7, min_samples_leaf=4, random_state=42)
    
    start_time = time.time()
    scores_sk = cross_val_score(clf_sk, X, y, cv=skf)
    end_time = time.time()
    
    print(f"- Build time: {end_time - start_time:.4f} seconds")
    print(f"- Cross-validation scores: {scores_sk}")
    print(f"- Mean accuracy: {scores_sk.mean()*100:.2f}%")
    print(f"- Std accuracy: {scores_sk.std()*100:.2f}%")
    
    # Compare predictions on a single split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and predict with our implementation
    clf_opt.fit(X_train, y_train)
    y_pred_opt = clf_opt.predict(X_test)
    
    # Train and predict with scikit-learn
    clf_sk.fit(X_train, y_train)
    y_pred_sk = clf_sk.predict(X_test)
    
    # Train and predict with C++ ctypes
    clf_cpp.fit(X_train, y_train)
    y_pred_cpp = clf_cpp.predict(X_test)
    
    print("\n[Detailed Comparison on Test Set]")
    print("\nOptimized Implementation:")
    print(classification_report(y_test, y_pred_opt))
    
    print("\nScikit-learn Implementation:")
    print(classification_report(y_test, y_pred_sk))
    
    print("\nC++ ctypes Implementation:")
    print(classification_report(y_test, y_pred_cpp))
    
    # # Compare feature importances if available
    # print("\n[Feature Importances Comparison]")
    # print("\nC++ ctypes Implementation:")
    # if hasattr(clf_cpp, 'feature_importances_'):
    #     importances = clf_cpp.feature_importances_
    #     for i, imp in enumerate(importances):
    #         print(f"Feature {i}: {imp:.4f}")
    
    # print("\nScikit-learn Implementation:")
    # for i, imp in enumerate(clf_sk.feature_importances_):
    #     print(f"Feature {i}: {imp:.4f}")
    
if __name__ == "__main__":
    main()