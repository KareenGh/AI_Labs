//#include <iostream>
//#include <vector>
//#include <algorithm>
//#include <numeric>
//#include <unordered_map>
//#include <thread>
//#include <mutex>
//
//struct VectorHasher {
//    std::size_t operator()(const std::vector<int>& v) const {
//        std::hash<int> hasher;
//        std::size_t seed = 0;
//        for (int i : v) {
//            seed ^= hasher(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
//        }
//        return seed;
//    }
//};
//
//std::unordered_map<std::vector<int>, bool, VectorHasher> cache;
//std::mutex mtx;
//
//// function to check if sequence is sorted
//bool is_sorted(const std::vector<int>& sequence) {
//    for (int i = 1; i < sequence.size(); i++) {
//        if (sequence[i] < sequence[i - 1]) {
//            return false;
//        }
//    }
//    return true;
//}
//
//// function to apply comparator on sequence
//void apply_comparator(std::vector<int>& sequence, const std::pair<int, int>& comparator) {
//    // Indices for C++ vectors start from 0, not 1, so we subtract 1.
//    int i = comparator.first;
//    int j = comparator.second;
//
//    if (sequence[i] > sequence[j]) {
//        std::swap(sequence[i], sequence[j]);
//    }
//}
//
//// Modified function to apply network on sequence
//std::vector<int> apply_network(std::vector<int> sequence, const std::vector<std::pair<int, int>>& network) {
//    // Check cache first
//    {
//        std::lock_guard<std::mutex> lock(mtx);
//        if (cache.find(sequence) != cache.end()) {
//            return sequence;
//        }
//    }
//
//    for (const auto& comparator : network) {
//        apply_comparator(sequence, comparator);
//    }
//
//    // Store result in cache
//    {
//        std::lock_guard<std::mutex> lock(mtx);
//        cache[sequence] = is_sorted(sequence);
//    }
//
//    return sequence;
//}
//
//// function to check if a single permutation sorts correctly
//void check_permutation(const std::vector<int>& permutation, const std::vector<std::pair<int, int>>& network, bool& isValid) {
//    std::vector<int> temp_sequence = permutation;
//    apply_network(temp_sequence, network);
//    if (!is_sorted(temp_sequence)) {
//        isValid = false;
//    }
//}
//
//
//// function to check if network sorts all possible sequences of size N
//bool check_network(const std::vector<std::pair<int, int>>& network, int N) {
//    // Generate initial sequence
//    std::vector<int> sequence(N);
//    std::iota(sequence.begin(), sequence.end(), 1);  // Fill with 1, 2, ..., N
//
//    bool isValid = true;
//
//    do {
//        // Launch threads to check permutations
//        std::thread t(check_permutation, std::ref(sequence), std::ref(network), std::ref(isValid));
//        t.join();
//
//        if (!isValid) {
//            return false;
//        }
//    } while (std::next_permutation(sequence.begin(), sequence.end()));
//
//    return true;
//}
//
//int main() {
//    std::vector<std::pair<int, int>> network = { {1, 5}, {0, 2}, {0, 1}, {1, 3}, {1, 4}, {2, 3}, {4, 5}, {2, 4}, {0, 1}, {3, 5}, {1, 2}, {3, 4} };
//
//    std::cout << (check_network(network, 6) ? "true" : "false") << std::endl;
//
//    return 0;
//}
//#include <algorithm>
//#include <vector>
//#include <iostream>
//#include <bitset>
//using namespace std;
//
//// Function to check if vector is sorted
//bool isSorted(const vector<int>& v) {
//    for (size_t i = 1; i < v.size(); ++i)
//        if (v[i] < v[i - 1]) return false;
//    return true;
//}
//
//// Function to generate all binary sequences of length n, sort them using the
//// sorting network, and check whether they are sorted
//bool isValidSortingNetwork(int n, const vector<pair<int, int>>& network) {
//    for (int i = 0; i < (1 << n); ++i) {
//        vector<int> sequence(n);
//        for (int j = 0; j < n; ++j)
//            sequence[j] = (i >> j) & 1;
//        for (const auto& p : network)
//            if (sequence[p.first] > sequence[p.second]) swap(sequence[p.first], sequence[p.second]);
//        if (!isSorted(sequence)) return false;
//    }
//    return true;
//}
//
//int main() {
//    vector<pair<int, int>> network = { {8, 13}, {5, 11}, {7, 8}, {5, 7}, {9, 12}, {1, 5}, {0, 10}, {10, 14}, {11, 12}, {12, 15}, {3, 10},
//        {1, 9}, {6, 15}, {6, 9}, {11, 12}, {6, 7}, {0, 6}, {5, 9}, {2, 8}, {2, 12}, {3, 13}, {7, 13}, {2, 4}, {1, 3}, {12, 13}, {8, 10}, 
//        {9, 11}, {6, 12}, {6, 12}, {6, 9}, {2, 5}, {7, 10}, {2, 3}, {2, 11}, {4, 13}, {7, 8}, {6, 7}, {10, 14}, {0, 6}, {7, 10}, {3, 9},
//        {1, 9}, {4, 5}, {7, 11}, {6, 7}, {14, 15}, {5, 11}, {11, 14}, {8, 14}, {5, 7}, {11, 12}, {3, 5}, {10, 13}, {7, 8}, {4, 7}, 
//        {4, 6}, {8, 15}, {8, 10}, {7, 11}, {5, 6}, {0, 13}, {10, 12}, {8, 9}, {7, 8}, {0, 2}, {3, 15}, {6, 8}, {9, 11}, {5, 6}, {9, 10},
//        {2, 3}, {8, 9}, {12, 13}, {6, 7}, {10, 11}, {12, 14}, {13, 14}, {9, 10}, {3, 5}, {10, 12}, {11, 12}, {1, 4}, {3, 4}, {7, 8},
//        {2, 3}, {14, 15}, {5, 6}, {0, 1}, {1, 2}, {4, 5}, {13, 14} };
//
//    // Get the maximum index from the network to calculate the input size
//    int max_index = 0;
//    for (const auto& p : network)
//        max_index = max(max_index, max(p.first, p.second));
//
//    int input_size = max_index + 1;
//
//    if (isValidSortingNetwork(input_size, network))
//        cout << "The network is a valid sorting network.\n";
//    else
//        cout << "The network is not a valid sorting network.\n";
//
//    return 0;
//}



#include <algorithm>
#include <vector>
#include <iostream>
#include <bitset>
#include <unordered_set>
using namespace std;

// Function to check if vector is sorted
bool isSorted(const vector<int>& v) {
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i] < v[i - 1]) return false;
    return true;
}

// Function to generate all binary sequences of length n, sort them using the
// sorting network, and check whether they are sorted
bool isValidSortingNetwork(int n, const vector<pair<int, int>>& network) {
    for (int i = 0; i < (1 << n); ++i) {
        vector<int> sequence(n);
        for (int j = 0; j < n; ++j)
            sequence[j] = (i >> j) & 1;
        for (const auto& p : network)
            if (sequence[p.first] > sequence[p.second]) swap(sequence[p.first], sequence[p.second]);
        if (!isSorted(sequence)) return false;
    }
    return true;
}

int main() {
    vector<pair<int, int>> network = { {0, 15}, {12, 13}, {8, 14}, {4, 5}, {1, 11}, {0, 4}, {10, 15}, {0, 9}, {5, 11}, {5, 14}, {1, 7},
        {0, 2}, {5, 15}, {6, 12}, {1, 8}, {2, 9}, {7, 9}, {13, 14}, {10, 12}, {7, 13}, {3, 13}, {3, 5}, {7, 11}, {12, 15}, {4, 7}, 
        {9, 12}, {4, 7}, {13, 15}, {1, 7}, {1, 10}, {8, 10}, {11, 13}, {12, 14}, {3, 9}, {3, 7}, {1, 2}, {6, 11}, {7, 10}, {10, 15},
        {6, 8}, {2, 8}, {14, 15}, {5, 9}, {3, 6}, {12, 13}, {4, 12}, {13, 14}, {7, 11}, {11, 13}, {1, 6}, {8, 12}, {5, 12}, {0, 7}, 
        {4, 6}, {0, 3}, {2, 5}, {2, 5}, {10, 12}, {4, 10}, {4, 10}, {12, 14}, {3, 4}, {5, 6}, {5, 6}, {6, 7}, {5, 8}, {10, 11}, {7, 9}, 
        {5, 6}, {0, 3}, {7, 10}, {8, 11}, {9, 12}, {1, 3}, {9, 11}, {2, 5}, {6, 8}, {4, 5}, {12, 15}, {5, 6}, {5, 6}, {8, 9}, {12, 13}, 
        {7, 8}, {3, 4}, {8, 10}, {5, 7}, {14, 15}, {6, 7}, {2, 4}, {2, 3}, {9, 12}, {13, 14}, {9, 10}, {11, 12}, {0, 2}, {0, 1} };

    // Get the set of all indices used in the network to calculate the input size
    unordered_set<int> indices;
    for (const auto& p : network) {
        indices.insert(p.first);
        indices.insert(p.second);
    }

    int input_size = indices.size();

    if (isValidSortingNetwork(input_size, network))
        cout << "The network is a valid sorting network.\n";
    else
        cout << "The network is not a valid sorting network.\n";

    return 0;
}
