// #include <vector>
// #include <memory>
// #include <set>
// #include <algorithm>
// #include <map>
// #include <numeric>
// #include <iostream>
// #include <limits>

// #ifdef _WIN32
// #define DLL_EXPORT __declspec(dllexport)
// #else
// #define DLL_EXPORT
// #endif

// // LỆNH BUILD: g++ -O3 -std=c++17 -shared -o decision_tree.dll decision_tree.cpp -static-libgcc -static-libstdc++

// extern "C" {

// struct Node {
//     int index;
//     double value;
//     std::shared_ptr<Node> left;
//     std::shared_ptr<Node> right;
//     int prediction;
//     double score;
//     Node() : index(-1), value(0.0), left(nullptr), right(nullptr), prediction(-1), score(0.0) {}
//     Node(int pred) : index(-1), value(0.0), left(nullptr), right(nullptr), prediction(pred), score(0.0) {}
//     Node(int idx, double val, double sc) : index(idx), value(val), left(nullptr), right(nullptr), prediction(-1), score(sc) {}
// };

// class DecisionTree {
// public:
//     DecisionTree(int max_depth = 5, int min_size = 1)
//         : max_depth(max_depth), min_size(min_size) {}

//     void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
//         std::vector<std::vector<double>> dataset;
//         for (size_t i = 0; i < X.size(); i++) {
//             std::vector<double> row = X[i];
//             row.push_back(static_cast<double>(y[i]));
//             dataset.push_back(row);
//         }
//         root = split(dataset, 0);
//     }

//     int predict(const std::vector<double>& row) const {
//         auto node = root;
//         while (node->left) {
//             if (row[node->index] < node->value)
//                 node = node->left;
//             else
//                 node = node->right;
//         }
//         return node->prediction;
//     }

//     std::vector<int> predict(const std::vector<std::vector<double>>& X) const {
//         std::vector<int> preds;
//         for (const auto& row : X)
//             preds.push_back(predict(row));
//         return preds;
//     }

//     std::vector<double> get_feature_importances() const {
//         int n_features = 0;
//         if (root) {
//             std::vector<const Node*> stack = {root.get()};
//             while (!stack.empty()) {
//                 auto node = stack.back(); stack.pop_back();
//                 if (node->left) {
//                     n_features = std::max(n_features, node->index + 1);
//                     stack.push_back(node->left.get());
//                     stack.push_back(node->right.get());
//                 }
//             }
//         }
//         std::vector<double> importances(n_features, 0.0);
//         if (!root) return importances;
//         std::vector<const Node*> stack = {root.get()};
//         while (!stack.empty()) {
//             auto node = stack.back(); stack.pop_back();
//             if (node->left) {
//                 importances[node->index] += node->score;
//                 stack.push_back(node->left.get());
//                 stack.push_back(node->right.get());
//             }
//         }
//         double sum = std::accumulate(importances.begin(), importances.end(), 0.0);
//         if (sum > 0)
//             for (auto& v : importances) v /= sum;
//         return importances;
//     }

// private:
//     std::shared_ptr<Node> root;
//     int max_depth;
//     int min_size;

//     double gini_index(const std::vector<std::vector<std::vector<double>>>& groups, const std::vector<int>& class_values) const {
//         double gini = 0.0;
//         size_t n_instances = 0;
//         for (const auto& group : groups) n_instances += group.size();
//         for (const auto& group : groups) {
//             if (group.empty()) continue;
//             size_t size = group.size();
//             double score = 0.0;
//             std::map<int, int> class_counts;
//             for (const auto& row : group) {
//                 int class_val = static_cast<int>(row.back());
//                 class_counts[class_val]++;
//             }
//             for (const auto& [class_val, count] : class_counts) {
//                 double proportion = static_cast<double>(count) / size;
//                 score += proportion * proportion;
//             }
//             gini += (1.0 - score) * (static_cast<double>(size) / n_instances);
//         }
//         return gini;
//     }

//     std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> test_split(int index, double value, const std::vector<std::vector<double>>& dataset) const {
//         std::vector<std::vector<double>> left, right;
//         for (const auto& row : dataset) {
//             if (row[index] < value) left.push_back(row);
//             else right.push_back(row);
//         }
//         return {left, right};
//     }

//     std::vector<double> get_unique_values(const std::vector<std::vector<double>>& dataset, int index) const {
//         std::set<double> unique_values;
//         for (const auto& row : dataset) unique_values.insert(row[index]);
//         return std::vector<double>(unique_values.begin(), unique_values.end());
//     }

//     std::shared_ptr<Node> get_split(const std::vector<std::vector<double>>& dataset) const {
//     std::vector<int> class_values;
//     for (const auto& row : dataset) class_values.push_back(static_cast<int>(row.back()));
//     int n_features = dataset[0].size() - 1;
//     double b_score = std::numeric_limits<double>::infinity();
//     int b_index = 0;
//     double b_value = 0.0;
//     for (int index = 0; index < n_features; index++) {
//         // Tạo vector các cặp (feature_value, label)
//         std::vector<std::pair<double, int>> feat_label(dataset.size());
//         for (size_t i = 0; i < dataset.size(); ++i)
//             feat_label[i] = {dataset[i][index], (int)dataset[i].back()};
//         std::sort(feat_label.begin(), feat_label.end());
//         for (size_t i = 1; i < feat_label.size(); ++i) {
//             if (feat_label[i-1].second != feat_label[i].second) {
//                 double split_value = (feat_label[i-1].first + feat_label[i].first) / 2.0;
//                 auto groups = test_split(index, split_value, dataset);
//                 std::vector<std::vector<std::vector<double>>> groups_vec = {groups.first, groups.second};
//                 double gini = gini_index(groups_vec, class_values);
//                 if (gini < b_score) {
//                     b_index = index;
//                     b_value = split_value;
//                     b_score = gini;
//                 }
//             }
//         }
//     }
//     return std::make_shared<Node>(b_index, b_value, b_score);
// }

//     std::shared_ptr<Node> to_terminal(const std::vector<std::vector<double>>& group) const {
//         std::map<int, int> counts;
//         for (const auto& row : group) counts[static_cast<int>(row.back())]++;
//         int max_count = 0, prediction = 0;
//         for (const auto& [outcome, count] : counts)
//             if (count > max_count) { max_count = count; prediction = outcome; }
//         return std::make_shared<Node>(prediction);
//     }

//     std::shared_ptr<Node> split(const std::vector<std::vector<double>>& dataset, int depth) const {
//         if (dataset.size() < min_size || depth >= max_depth)
//             return to_terminal(dataset);
//         auto node = get_split(dataset);
//         if (node->score == 0.0)
//             return to_terminal(dataset);
//         auto groups = test_split(node->index, node->value, dataset);
//         node->left = split(groups.first, depth + 1);
//         node->right = split(groups.second, depth + 1);
//         return node;
//     }
// };

// typedef void* DecisionTreeHandle;

// DLL_EXPORT DecisionTreeHandle dt_create(int max_depth, int min_size) {
//     return new DecisionTree(max_depth, min_size);
// }

// DLL_EXPORT void dt_destroy(DecisionTreeHandle handle) {
//     delete static_cast<DecisionTree*>(handle);
// }

// DLL_EXPORT void dt_fit(DecisionTreeHandle handle, double* X, int* y, int n_samples, int n_features) {
//     auto* tree = static_cast<DecisionTree*>(handle);
//     std::vector<std::vector<double>> Xvec(n_samples, std::vector<double>(n_features));
//     for (int i = 0; i < n_samples; ++i)
//         for (int j = 0; j < n_features; ++j)
//             Xvec[i][j] = X[i * n_features + j];
//     std::vector<int> yvec(y, y + n_samples);
//     tree->fit(Xvec, yvec);
// }

// DLL_EXPORT void dt_predict(DecisionTreeHandle handle, double* X, int n_samples, int n_features, int* out) {
//     auto* tree = static_cast<DecisionTree*>(handle);
//     std::vector<std::vector<double>> Xvec(n_samples, std::vector<double>(n_features));
//     for (int i = 0; i < n_samples; ++i)
//         for (int j = 0; j < n_features; ++j)
//             Xvec[i][j] = X[i * n_features + j];
//     auto preds = tree->predict(Xvec);
//     for (int i = 0; i < n_samples; ++i)
//         out[i] = preds[i];
// }

// DLL_EXPORT void dt_feature_importances(DecisionTreeHandle handle, double* out_importances, int n_features) {
//     auto* tree = static_cast<DecisionTree*>(handle);
//     auto importances = tree->get_feature_importances();
//     for (int i = 0; i < n_features; ++i)
//         out_importances[i] = (i < (int)importances.size() ? importances[i] : 0.0);
// }

// } // extern "C" 

#include <vector>
#include <memory>
#include <set>
#include <algorithm>
#include <map> // Vẫn dùng map cho các trường hợp ít lớp, hoặc có thể thay bằng unordered_map
#include <unordered_map> // Để tối ưu việc đếm lớp
#include <numeric>
#include <iostream>
#include <limits>

#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

// LỆNH BUILD: g++ -O3 -std=c++17 -shared -o decision_tree.dll decision_tree.cpp -static-libgcc -static-libstdc++

extern "C" {

struct Node {
    int index; // Chỉ số đặc trưng để chia
    double value; // Giá trị ngưỡng để chia
    std::shared_ptr<Node> left; // Con trái (nhỏ hơn ngưỡng)
    std::shared_ptr<Node> right; // Con phải (lớn hơn hoặc bằng ngưỡng)
    int prediction; // Dự đoán cho nút lá (-1 nếu không phải nút lá)
    double score; // Điểm Gini impurity của nút này (trước khi chia)

    // Constructor cho nút lá
    Node(int pred) : index(-1), value(0.0), left(nullptr), right(nullptr), prediction(pred), score(0.0) {}
    // Constructor cho nút chia
    Node(int idx, double val, double sc) : index(idx), value(val), left(nullptr), right(nullptr), prediction(-1), score(sc) {}
};

class DecisionTree {
public:
    // Constructor của cây quyết định
    DecisionTree(int max_depth = 5, int min_size = 1)
        : max_depth(max_depth), min_size(min_size), n_features_(-1) {}

    // Hàm huấn luyện cây
    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
        if (X.empty() || X[0].empty()) {
            std::cerr << "Lỗi: Dữ liệu huấn luyện X rỗng." << std::endl;
            return;
        }
        n_features_ = X[0].size(); // Số lượng đặc trưng

        // Làm phẳng dữ liệu X để cải thiện hiệu suất truy cập bộ nhớ
        flat_X.clear();
        flat_X.reserve(X.size() * n_features_);
        for (const auto& row : X) {
            flat_X.insert(flat_X.end(), row.begin(), row.end());
        }
        
        // Lưu nhãn
        flat_y = y;

        // Tạo danh sách chỉ mục ban đầu cho toàn bộ tập dữ liệu
        std::vector<int> initial_indices(X.size());
        std::iota(initial_indices.begin(), initial_indices.end(), 0); // Điền 0, 1, 2, ... n_samples-1

        // Bắt đầu quá trình chia cây từ nút gốc
        root = split(flat_X, flat_y, initial_indices, 0);
    }

    // Hàm dự đoán cho một mẫu duy nhất
    int predict(const std::vector<double>& row) const {
        if (!root) {
            std::cerr << "Lỗi: Cây chưa được huấn luyện." << std::endl;
            return -1; // Hoặc một giá trị lỗi khác
        }
        auto node = root;
        // Duyệt cây cho đến khi gặp nút lá (nút không có con trái)
        while (node->left) {
            if (node->index >= row.size()) { // Kiểm tra chỉ số hợp lệ
                std::cerr << "Lỗi: Chỉ số đặc trưng ngoài phạm vi trong quá trình dự đoán." << std::endl;
                return node->prediction != -1 ? node->prediction : 0; // Trả về dự đoán hiện tại nếu có, hoặc 0
            }
            if (row[node->index] < node->value)
                node = node->left;
            else
                node = node->right;
        }
        return node->prediction; // Trả về dự đoán của nút lá
    }

    // Hàm dự đoán cho nhiều mẫu
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const {
        std::vector<int> preds;
        preds.reserve(X.size()); // Cấp phát trước bộ nhớ để tránh reallocations
        for (const auto& row : X)
            preds.push_back(predict(row));
        return preds;
    }

    // Hàm tính toán tầm quan trọng của đặc trưng
    std::vector<double> get_feature_importances() const {
        if (n_features_ == -1) { // Cây chưa được huấn luyện
            return {};
        }
        std::vector<double> importances(n_features_, 0.0);
        if (!root) return importances;

        // Sử dụng BFS/DFS để duyệt cây và cộng dồn điểm Gini
        std::vector<const Node*> stack;
        stack.push_back(root.get());

        while (!stack.empty()) {
            auto node = stack.back();
            stack.pop_back();

            if (node->left) { // Nếu là nút chia (không phải nút lá)
                if (node->index >= 0 && node->index < n_features_) {
                    importances[node->index] += node->score; // Cộng dồn điểm Gini impurity giảm được
                }
                stack.push_back(node->left.get());
                stack.push_back(node->right.get());
            }
        }
        
        // Chuẩn hóa tầm quan trọng
        double sum = std::accumulate(importances.begin(), importances.end(), 0.0);
        if (sum > 0)
            for (auto& v : importances) v /= sum;
        return importances;
    }

private:
    std::shared_ptr<Node> root; // Nút gốc của cây
    int max_depth; // Độ sâu tối đa của cây
    int min_size; // Kích thước nhóm tối thiểu để chia
    int n_features_; // Số lượng đặc trưng, được xác định trong fit()

    // Dữ liệu huấn luyện phẳng (được lưu trữ trong lớp để tránh truyền lại)
    std::vector<double> flat_X;
    std::vector<int> flat_y;

    // Hàm tính chỉ số Gini impurity
    // class_values: Các nhãn lớp duy nhất trong tập dữ liệu (không cần thiết ở đây vì đã có group_labels)
    // group_labels: Các nhãn của các mẫu trong nhóm hiện tại
    double gini_index(const std::vector<std::vector<int>>& groups_indices) const {
        double gini = 0.0;
        size_t n_instances_in_node = 0;
        for (const auto& group_indices : groups_indices) {
            n_instances_in_node += group_indices.size();
        }

        if (n_instances_in_node == 0) return 0.0;

        for (const auto& group_indices : groups_indices) {
            if (group_indices.empty()) continue;
            size_t size = group_indices.size();
            double score = 0.0;
            std::unordered_map<int, int> class_counts; // Dùng unordered_map để đếm nhanh hơn
            
            for (int idx : group_indices) {
                class_counts[flat_y[idx]]++; // Đếm số lượng mẫu cho mỗi lớp trong nhóm
            }

            for (const auto& pair : class_counts) {
                double proportion = static_cast<double>(pair.second) / size;
                score += proportion * proportion;
            }
            // Gini impurity của nhóm = (1 - tổng bình phương các tỷ lệ lớp)
            // Gini tổng thể = tổng (Gini impurity của nhóm * tỷ lệ kích thước nhóm)
            gini += (1.0 - score) * (static_cast<double>(size) / n_instances_in_node);
        }
        return gini;
    }

    // Hàm chia tập dữ liệu thành hai nhóm dựa trên đặc trưng và giá trị ngưỡng
    // Trả về các chỉ mục của mẫu trong nhóm trái và phải
    std::pair<std::vector<int>, std::vector<int>> test_split(int feature_index, double value, const std::vector<int>& current_indices) const {
        std::vector<int> left_indices, right_indices;
        left_indices.reserve(current_indices.size()); // Cấp phát trước bộ nhớ
        right_indices.reserve(current_indices.size()); // Cấp phát trước bộ nhớ

        for (int sample_idx : current_indices) {
            if (feature_index >= n_features_) { // Kiểm tra chỉ số đặc trưng hợp lệ
                std::cerr << "Lỗi: Chỉ số đặc trưng ngoài phạm vi trong test_split." << std::endl;
                // Có thể xử lý bằng cách bỏ qua mẫu này hoặc đưa vào một nhóm mặc định
                continue; 
            }
            if (flat_X[sample_idx * n_features_ + feature_index] < value)
                left_indices.push_back(sample_idx);
            else
                right_indices.push_back(sample_idx);
        }
        return {left_indices, right_indices};
    }

    // Hàm tìm điểm chia tốt nhất cho một tập dữ liệu con
    std::shared_ptr<Node> get_split(const std::vector<int>& current_indices) const {
        // Nếu không có mẫu hoặc chỉ có một mẫu, không thể chia
        if (current_indices.empty()) {
            return std::make_shared<Node>(-1); // Nút rỗng hoặc lỗi
        }

        // Lấy các giá trị lớp duy nhất trong nhóm hiện tại (chỉ để truyền cho gini_index, không dùng trực tiếp)
        // std::vector<int> class_values_in_group;
        // for (int idx : current_indices) class_values_in_group.push_back(flat_y[idx]);
        // std::sort(class_values_in_group.begin(), class_values_in_group.end());
        // class_values_in_group.erase(std::unique(class_values_in_group.begin(), class_values_in_group.end()), class_values_in_group.end());

        double b_score = std::numeric_limits<double>::infinity(); // Điểm Gini tốt nhất
        int b_index = -1; // Chỉ số đặc trưng tốt nhất
        double b_value = 0.0; // Giá trị ngưỡng tốt nhất

        // Duyệt qua tất cả các đặc trưng
        for (int index = 0; index < n_features_; index++) {
            // Tạo vector các cặp (feature_value, label) chỉ cho các mẫu trong nhóm hiện tại
            std::vector<std::pair<double, int>> feat_label_pairs(current_indices.size());
            for (size_t i = 0; i < current_indices.size(); ++i) {
                int sample_idx = current_indices[i];
                feat_label_pairs[i] = {flat_X[sample_idx * n_features_ + index], flat_y[sample_idx]};
            }
            
            // Sắp xếp các cặp theo giá trị đặc trưng
            std::sort(feat_label_pairs.begin(), feat_label_pairs.end());

            // Tìm điểm chia tiềm năng
            for (size_t i = 1; i < feat_label_pairs.size(); ++i) {
                // Chỉ xem xét điểm chia khi nhãn lớp thay đổi (hoặc giá trị đặc trưng thay đổi)
                // Điều này giúp giảm số lượng điểm chia tiềm năng cần kiểm tra
                if (feat_label_pairs[i-1].second != feat_label_pairs[i].second || 
                    feat_label_pairs[i-1].first != feat_label_pairs[i].first) {
                    
                    double split_value = (feat_label_pairs[i-1].first + feat_label_pairs[i].first) / 2.0;
                    
                    // Chia nhóm dựa trên điểm chia tiềm năng
                    auto groups_indices = test_split(index, split_value, current_indices);
                    
                    // Nếu một trong hai nhóm rỗng, điểm chia này không hợp lệ
                    if (groups_indices.first.empty() || groups_indices.second.empty()) {
                        continue;
                    }

                    std::vector<std::vector<int>> groups_vec = {groups_indices.first, groups_indices.second};
                    double gini = gini_index(groups_vec); // Tính Gini impurity cho điểm chia này

                    // Cập nhật điểm chia tốt nhất nếu tìm thấy Gini thấp hơn
                    if (gini < b_score) {
                        b_index = index;
                        b_value = split_value;
                        b_score = gini;
                    }
                }
            }
        }
        // Trả về nút chứa thông tin về điểm chia tốt nhất
        return std::make_shared<Node>(b_index, b_value, b_score);
    }

    // Hàm tạo nút lá (terminal node)
    std::shared_ptr<Node> to_terminal(const std::vector<int>& current_indices) const {
        if (current_indices.empty()) {
            return std::make_shared<Node>(-1); // Nút lá rỗng
        }
        std::unordered_map<int, int> counts; // Dùng unordered_map để đếm nhanh hơn
        for (int idx : current_indices) {
            counts[flat_y[idx]]++;
        }
        int max_count = 0;
        int prediction = -1; // Giá trị mặc định cho trường hợp không có mẫu
        if (!counts.empty()) {
            for (const auto& pair : counts) {
                if (pair.second > max_count) {
                    max_count = pair.second;
                    prediction = pair.first;
                }
            }
        }
        return std::make_shared<Node>(prediction);
    }

    // Hàm đệ quy để xây dựng cây quyết định
    std::shared_ptr<Node> split(const std::vector<double>& data, const std::vector<int>& labels, const std::vector<int>& current_indices, int depth) const {
        // Điều kiện dừng đệ quy:
        // 1. Kích thước nhóm quá nhỏ (min_size)
        // 2. Đạt đến độ sâu tối đa (max_depth)
        // 3. Tất cả các mẫu trong nhóm thuộc cùng một lớp (Gini = 0, sẽ được xử lý bởi get_split)
        
        // Nếu nhóm rỗng, trả về nút lá rỗng
        if (current_indices.empty()) {
            return std::make_shared<Node>(-1);
        }

        // Nếu tất cả các mẫu trong nhóm có cùng nhãn, tạo nút lá
        if (current_indices.size() > 0) {
            int first_label = labels[current_indices[0]];
            bool all_same_label = true;
            for (size_t i = 1; i < current_indices.size(); ++i) {
                if (labels[current_indices[i]] != first_label) {
                    all_same_label = false;
                    break;
                }
            }
            if (all_same_label) {
                return std::make_shared<Node>(first_label);
            }
        }

        if (current_indices.size() < min_size || depth >= max_depth) {
            return to_terminal(current_indices);
        }
        
        // Tìm điểm chia tốt nhất cho nhóm hiện tại
        auto node = get_split(current_indices);

        // Nếu không tìm được điểm chia hợp lệ (ví dụ: b_index vẫn là -1) hoặc Gini = 0, tạo nút lá
        if (node->index == -1 || node->score == 0.0) {
            return to_terminal(current_indices);
        }

        // Chia nhóm thành con trái và con phải
        auto groups_indices = test_split(node->index, node->value, current_indices);
        
        // Xây dựng cây con đệ quy
        node->left = split(data, labels, groups_indices.first, depth + 1);
        node->right = split(data, labels, groups_indices.second, depth + 1);
        
        return node;
    }
};

// --- C API cho Python ---

typedef void* DecisionTreeHandle;

DLL_EXPORT DecisionTreeHandle dt_create(int max_depth, int min_size) {
    return new DecisionTree(max_depth, min_size);
}

DLL_EXPORT void dt_destroy(DecisionTreeHandle handle) {
    delete static_cast<DecisionTree*>(handle);
}

DLL_EXPORT void dt_fit(DecisionTreeHandle handle, double* X_flat, int* y, int n_samples, int n_features) {
    auto* tree = static_cast<DecisionTree*>(handle);
    // Chuyển đổi dữ liệu phẳng từ C-style array sang std::vector<std::vector<double>> tạm thời
    // để phù hợp với hàm fit hiện tại.
    // Trong thực tế, nếu bạn kiểm soát dữ liệu đầu vào từ Python, bạn có thể truyền thẳng
    // std::vector<double> và std::vector<int> vào fit nếu muốn tối ưu hơn nữa.
    
    // Tối ưu hóa: Hàm fit đã được sửa đổi để nhận flat_X và flat_y trực tiếp.
    // Do đó, chúng ta cần sao chép dữ liệu từ con trỏ C vào các vector nội bộ của cây.
    std::vector<std::vector<double>> X_vec_temp(n_samples, std::vector<double>(n_features));
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            X_vec_temp[i][j] = X_flat[i * n_features + j];
        }
    }
    std::vector<int> y_vec_temp(y, y + n_samples);
    tree->fit(X_vec_temp, y_vec_temp);
}

DLL_EXPORT void dt_predict(DecisionTreeHandle handle, double* X_flat, int n_samples, int n_features, int* out) {
    auto* tree = static_cast<DecisionTree*>(handle);
    std::vector<std::vector<double>> X_vec_temp(n_samples, std::vector<double>(n_features));
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            X_vec_temp[i][j] = X_flat[i * n_features + j];
        }
    }
    auto preds = tree->predict(X_vec_temp);
    for (int i = 0; i < n_samples; ++i)
        out[i] = preds[i];
}

DLL_EXPORT void dt_feature_importances(DecisionTreeHandle handle, double* out_importances, int n_features) {
    auto* tree = static_cast<DecisionTree*>(handle);
    auto importances = tree->get_feature_importances();
    for (int i = 0; i < n_features; ++i)
        out_importances[i] = (i < (int)importances.size() ? importances[i] : 0.0);
}

} // extern "C"