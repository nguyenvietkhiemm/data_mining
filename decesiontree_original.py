from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import numpy as np
from sklearn.datasets import make_classification
import logging

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Gini index
def gini_index(groups, classes):
    n_instances = float(sum(len(group) for group in groups))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        labels = [row[-1] for row in group]
        for class_val in classes:
            p = labels.count(class_val) / size
            score += p ** 2
        gini += (1.0 - score) * (size / n_instances)
    return gini

# Chia nhóm theo chỉ số và giá trị
def test_split(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# Tìm cách chia tốt nhất
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    best_index, best_value, best_score, best_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < best_score:
                best_index, best_value, best_score, best_groups = index, row[index], gini, groups
    return {'index': best_index, 'value': best_value, 'groups': best_groups}

# Tạo nút lá
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# Đệ quy xây dựng cây
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)

# Xây cây
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

# Dự đoán 1 dòng
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

# Dự đoán cho toàn bộ test set
def global_predict(tree, test_data):
    predictions = []
    for row in test_data:
        predictions.append(predict(tree, row))
    return predictions

def main():
    start_time = time.time()
    
    logging.info("Bắt đầu tải dữ liệu...")
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    logging.info("Đang chia tập train/test...")
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    train_data = [list(X_train[i]) + [y_train[i]] for i in range(len(y_train))]
    test_data = [list(X_test[i]) + [y_test[i]] for i in range(len(y_test))]

    logging.info(f"Kích thước tập train: {len(train_data)} mẫu")
    logging.info(f"Kích thước tập test: {len(test_data)} mẫu")

    logging.info("Bắt đầu xây dựng cây quyết định...")
    build_start = time.time()
    tree = build_tree(train_data, max_depth=6, min_size=3)
    build_time = time.time() - build_start
    logging.info(f"Thời gian xây dựng cây: {build_time:.4f} giây")

    logging.info("Bắt đầu dự đoán...")
    predict_start = time.time()
    predictions = global_predict(tree, X_test)
    predict_time = time.time() - predict_start
    logging.info(f"Thời gian dự đoán: {predict_time:.4f} giây")

    acc = accuracy_score(y_test, predictions)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    logging.info(f"Độ chính xác - Thuật toán viết tay: {round(acc * 100, 2)}%")
    logging.info(f"Tổng thời gian thực thi: {execution_time:.4f} giây")

if __name__ == "__main__":
    main() 