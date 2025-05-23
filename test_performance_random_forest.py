import time
import numpy as np
from collections import Counter
from scipy.stats import mode
from sklearn.datasets import load_wine, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from random_forest_optimize import RandomForest  # SỬA CHỖ NÀY THÔI


class ManualRandomForestWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features

        self.model = RandomForest(
            n_trees=self.n_trees,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_features=self.n_features
        )

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.model = RandomForest(
            n_trees=self.n_trees,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_features=self.n_features
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {
            "n_trees": self.n_trees,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "n_features": self.n_features,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        self.model = RandomForest(
            n_trees=self.n_trees,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_features=self.n_features
        )
        return self


def main():
    wine = load_wine()
    X_wine = wine.data
    y_wine = wine.target

    X_syn, y_syn = make_classification(n_samples=1000,
                                       n_features=X_wine.shape[1],
                                       n_classes=len(np.unique(y_wine)),
                                       n_informative=5,
                                       n_clusters_per_class=2,
                                       random_state=42)

    X = np.vstack((X_wine, X_syn))
    y = np.concatenate((y_wine, y_syn))

    print("\nDistribution of classes in combined dataset:")
    unique_classes, class_counts = np.unique(y, return_counts=True)
    for i, cls_val in enumerate(unique_classes):
        count = class_counts[i]
        percentage = (count / len(y)) * 100
        print(f"Class {cls_val}: {count} samples ({percentage:.1f}%)")

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    n_total_features = X.shape[1]
    manual_rf_n_features_param = 'sqrt'

    print("\n[Manual RandomForest Version]")
    print("Running cross-validation for manual RandomForest version...")
    clf_manual_rf = ManualRandomForestWrapper(n_trees=25,
                                              max_depth=8,
                                              min_samples_split=5,
                                              n_features=n_total_features)  # ban đầu đang là n_features=manual_rf_n_features_param
    start_time = time.time()
    try:
        scores_manual_rf = cross_val_score(
            clf_manual_rf, X, y, cv=skf, n_jobs=-1)
        end_time = time.time()
        print(f"- Build time (CV): {end_time - start_time:.4f} seconds")
        print(f"- Cross-validation scores: {scores_manual_rf}")
        print(f"- Mean accuracy: {scores_manual_rf.mean()*100:.2f}%")
        print(f"- Std accuracy: {scores_manual_rf.std()*100:.2f}%")
    except Exception as e:
        end_time = time.time()
        print(
            f"- Error during manual RF cross-validation (Time: {end_time - start_time:.4f}s): {e}")

    print("\n[Scikit-learn RandomForest Version]")
    print("Running cross-validation for scikit-learn RandomForest version...")
    clf_sk_rf = RandomForestClassifier(n_estimators=25,
                                       max_depth=8,
                                       min_samples_split=5,
                                       min_samples_leaf=2,
                                       max_features='sqrt',
                                       random_state=42,
                                       n_jobs=-1)

    start_time = time.time()
    scores_sk_rf = cross_val_score(clf_sk_rf, X, y, cv=skf)
    end_time = time.time()

    print(f"- Build time (CV): {end_time - start_time:.4f} seconds")
    print(f"- Cross-validation scores: {scores_sk_rf}")
    print(f"- Mean accuracy: {scores_sk_rf.mean()*100:.2f}%")
    print(f"- Std accuracy: {scores_sk_rf.std()*100:.2f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print("\n[Detailed Comparison on Test Set]")

    print("\nManual RandomForest Implementation:")
    try:
        start_fit_manual = time.time()
        clf_manual_rf.fit(X_train, y_train)
        end_fit_manual = time.time()
        print(f"  Fit time: {end_fit_manual - start_fit_manual:.4f}s")

        start_pred_manual = time.time()
        y_pred_manual_rf = clf_manual_rf.predict(X_test)
        end_pred_manual = time.time()
        print(f"  Predict time: {end_pred_manual - start_pred_manual:.4f}s")
        print(classification_report(y_test, y_pred_manual_rf, zero_division=0))
    except Exception as e:
        print(f"- Error during manual RF fit/predict: {e}")

    print("\nScikit-learn RandomForest Implementation:")
    start_fit_sk = time.time()
    clf_sk_rf.fit(X_train, y_train)
    end_fit_sk = time.time()
    print(f"  Fit time: {end_fit_sk - start_fit_sk:.4f}s")

    start_pred_sk = time.time()
    y_pred_sk_rf = clf_sk_rf.predict(X_test)
    end_pred_sk = time.time()
    print(f"  Predict time: {end_pred_sk - start_pred_sk:.4f}s")

    print(classification_report(y_test, y_pred_sk_rf, zero_division=0))


if __name__ == "__main__":
    main()