from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import numpy as np

# Gini index - tối ưu hóa với NumPy
def gini_index(groups, classes):
    n_instances = sum(len(group) for group in groups)
    gini = 0.0
    for group in groups:
        if len(group) == 0:
            continue
        size = len(group)
        labels = np.array([row[-1] for row in group])
        score = 0.0
        for class_val in classes:
            p = np.sum(labels == class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini

# Chia nhóm theo chỉ số và giá trị - tối ưu hóa với NumPy
def test_split(index, value, dataset):
    dataset = np.array(dataset)
    left = dataset[dataset[:, index] < value]
    right = dataset[dataset[:, index] >= value]
    return left.tolist(), right.tolist()

# Tìm cách chia tốt nhất - tối ưu hóa với NumPy
def get_split(dataset, depth=0, max_features=None):
    dataset = np.array(dataset)
    class_values = np.unique(dataset[:, -1])
    best_index, best_value, best_score, best_groups = 999, 999, 999, None
    
    # Chọn ngẫu nhiên số đặc trưng nếu max_features được chỉ định
    n_features = dataset.shape[1] - 1
    if max_features is not None:
        n_features = min(max_features, n_features)
    features = np.random.choice(range(dataset.shape[1] - 1), n_features, replace=False)
    
    for index in features:
        # Lấy các giá trị duy nhất và sắp xếp
        values = np.unique(dataset[:, index])
        # Tăng số điểm chia từ 10 lên 20
        if len(values) > 20:
            values = np.percentile(values, np.linspace(0, 100, 20))
        
        for value in values:
            groups = test_split(index, value, dataset)
            gini = gini_index(groups, class_values)
            if gini < best_score:
                best_index, best_value, best_score, best_groups = index, value, gini, groups
    
    if depth % 10 == 0:  # Log mỗi 2 mức độ sâu
        print(f"Depth {depth}, Gini score: {best_score:.4f}")
    
    return {'index': best_index, 'value': best_value, 'groups': best_groups}

# Tạo nút lá - tối ưu hóa với NumPy
def to_terminal(group):
    group = np.array(group)
    values, counts = np.unique(group[:, -1], return_counts=True)
    return values[np.argmax(counts)]

# Đệ quy xây dựng cây - tối ưu hóa với stack
def split(node, max_depth, min_size, depth):
    stack = [(node, depth)]
    
    while stack:
        current_node, current_depth = stack.pop()
        left, right = current_node['groups']
        del(current_node['groups'])
        
        if not left or not right:
            current_node['left'] = current_node['right'] = to_terminal(left + right)
            continue
            
        if current_depth >= max_depth:
            current_node['left'], current_node['right'] = to_terminal(left), to_terminal(right)
            continue
            
        if len(left) <= min_size:
            current_node['left'] = to_terminal(left)
        else:
            current_node['left'] = get_split(left, current_depth + 1)
            stack.append((current_node['left'], current_depth + 1))
            
        if len(right) <= min_size:
            current_node['right'] = to_terminal(right)
        else:
            current_node['right'] = get_split(right, current_depth + 1)
            stack.append((current_node['right'], current_depth + 1))

# Tối ưu hóa hàm predict bằng cách sử dụng vòng lặp thay vì đệ quy
def predict(node, row):
    current = node
    while isinstance(current, dict):
        if row[current['index']] < current['value']:
            current = current['left']
        else:
            current = current['right']
    return current

# Tối ưu hóa hàm global_predict bằng cách vectorize
def global_predict(tree, test_data):
    test_data = np.array(test_data)
    predictions = np.zeros(len(test_data), dtype=int)
    
    for i in range(len(test_data)):
        predictions[i] = predict(tree, test_data[i])
    
    return predictions

def prune_tree(node, X_val, y_val, min_impurity_decrease=0.0):
    if not isinstance(node, dict):
        return node
    
    # Tính độ chính xác trước khi cắt tỉa
    y_pred = global_predict(node, X_val)
    acc_before = accuracy_score(y_val, y_pred)
    
    # Thử cắt tỉa
    left = prune_tree(node['left'], X_val, y_val, min_impurity_decrease)
    right = prune_tree(node['right'], X_val, y_val, min_impurity_decrease)
    
    # Nếu cả hai nhánh là lá, thử gộp
    if not isinstance(left, dict) and not isinstance(right, dict):
        # Tạo node mới với giá trị phổ biến nhất
        merged_node = {'index': node['index'], 'value': node['value'],
                      'left': left, 'right': right}
        y_pred = global_predict(merged_node, X_val)
        acc_after = accuracy_score(y_val, y_pred)
        
        # Nếu độ chính xác không giảm, giữ node gộp
        if acc_after >= acc_before - min_impurity_decrease:
            return to_terminal(np.column_stack((X_val, y_val)))
    
    # Nếu không gộp được, giữ nguyên cấu trúc
    node['left'] = left
    node['right'] = right
    return node

# Xây cây
def build_tree(train, max_depth, min_size, max_features=None, X_val=None, y_val=None, min_impurity_decrease=0.0):
    root = get_split(train, 0, max_features)
    split(root, max_depth, min_size, 1)
    
    # Áp dụng post-pruning nếu có validation set
    if X_val is not None and y_val is not None:
        root = prune_tree(root, X_val, y_val, min_impurity_decrease)
    
    return root

def main():
    start_time = time.time()
    
    print("Bắt đầu tải dữ liệu...")
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    print("Đang chia tập train/test/validation...")
    # Split data into train/test/validation
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    train_data = np.column_stack((X_train, y_train))

    print(f"Kích thước tập train: {len(train_data)} mẫu")
    print(f"Kích thước tập validation: {len(X_val)} mẫu")
    print(f"Kích thước tập test: {len(X_test)} mẫu")

    print("Bắt đầu xây dựng cây quyết định...")
    build_start = time.time()
    # Sử dụng các tham số mới
    tree = build_tree(train_data, 
                     max_depth=7,  # Tăng độ sâu tối đa
                     min_size=4,   # Tăng kích thước tối thiểu
                     max_features=3,  # Chỉ xét 3 đặc trưng tại mỗi node
                     X_val=X_val,  # Thêm validation set
                     y_val=y_val,
                     min_impurity_decrease=0.01)  # Ngưỡng giảm độ không thuần nhất
    build_time = time.time() - build_start
    print(f"Thời gian xây dựng cây: {build_time:.4f} giây")

    print("Bắt đầu dự đoán...")
    predict_start = time.time()
    predictions = global_predict(tree, X_test)
    predict_time = time.time() - predict_start
    print(f"Thời gian dự đoán: {predict_time:.4f} giây")

    acc = accuracy_score(y_test, predictions)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Độ chính xác - Thuật toán viết tay: {round(acc * 100, 2)}%")
    print(f"Tổng thời gian thực thi: {execution_time:.4f} giây")

if __name__ == "__main__":
    main()
