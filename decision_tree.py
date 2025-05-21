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


# import numpy as np
# from sklearn.datasets import load_wine
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report


# # Load the wine dataset from sklearn
# wine = load_wine()
# X = wine.data
# y = wine.target

# # Split into train and test set
# X_train, X_test, y_train, y_test = train_test_split(
# X, y, test_size=0.2, random_state=42, stratify=y
# )

# # Initialize and train the C++ decision tree via ctypes
# dt = DecisionTree(max_depth=7, min_size=4)
# dt.fit(X_train, y_train)

# # Predict on the test set
# y_pred = dt.predict(X_test)

# # Print evaluation metrics
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# # Print feature importances
# importances = dt.feature_importances_()
# print("Feature importances:")
# for idx, imp in enumerate(importances):
#         print(f"Feature {idx}: {imp:.4f}")
