#ifndef _TEST_MAIN_H__
#define _TEST_MAIN_H__

#include "csr.hpp"
#include <cstddef>
#include <limits>
#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "../sili/lib/headers/unique_vector.hpp"
#include <vector>

// thanks: https://github.com/catchorg/Catch2/issues/929#issuecomment-308663820
#define REQUIRE_MESSAGE(cond, msg)                                                                                     \
    do {                                                                                                               \
        INFO(msg);                                                                                                     \
        REQUIRE(cond);                                                                                                 \
    } while ((void)0, 0)
#define CHECK_MESSAGE(cond, msg)                                                                                       \
    do {                                                                                                               \
        INFO(msg);                                                                                                     \
        CHECK(cond);                                                                                                   \
    } while ((void)0, 0)

template <typename T> struct is_supported_vector : std::false_type {};

template <typename T> struct is_supported_vector<std::vector<T>> : std::true_type {};

template <typename T> struct is_supported_vector<sili::unique_vector<T>> : std::true_type {};

// The following allows us to print pairs and vectors
// thanks: https://stackoverflow.com/a/6245777
namespace aux {
template <std::size_t...> struct seq {};

template <std::size_t N, std::size_t... Is> struct gen_seq : gen_seq<N - 1, N - 1, Is...> {};

template <std::size_t... Is> struct gen_seq<0, Is...> : seq<Is...> {};

template <class Ch, class Tr, class Tuple, std::size_t... Is>
void print_tuple(std::basic_ostream<Ch, Tr> &os, Tuple const &t, seq<Is...>) {
    using swallow = int[];
    (void)swallow{0, (void(os << (Is == 0 ? "" : ", ") << std::get<Is>(t)), 0)...};
}
} // namespace aux

// forward declare
template <class Ch, class Tr, class T> std::ostream &operator<<(std::basic_ostream<Ch, Tr> &o, const std::vector<T> &p);
template <class Ch, class Tr, class S, class T>
std::ostream &operator<<(std::basic_ostream<Ch, Tr> &o, const std::pair<S, T> &p);

template <class Ch, class Tr, class... Args>
auto operator<<(std::basic_ostream<Ch, Tr> &os, std::tuple<Args...> const &t) -> std::basic_ostream<Ch, Tr> & {
    os << "(";
    aux::print_tuple(os, t, aux::gen_seq<sizeof...(Args)>());
    return os << ")";
}

template <class Ch, class Tr, class S, class T>
std::ostream &operator<<(std::basic_ostream<Ch, Tr> &o, const std::pair<S, T> &p) {
    return o << "(" << p.first << ", " << p.second << ")";
}

template <class Ch, class Tr, class T>
std::ostream &operator<<(std::basic_ostream<Ch, Tr> &o, const std::vector<T> &p) {
    o << "{ ";
    for (auto v : p) {
        o << v << " ";
    }
    o << "}";
    return o;
}

template <class Ch, class Tr, class T>
std::ostream &operator<<(std::basic_ostream<Ch, Tr> &o, const sili::unique_vector<T> &p) {
    o << "{ ";
    for (auto v : p) {
        o << v << " ";
    }
    o << "}";
    return o;
}

template <typename VecA, typename VecB>
inline typename std::enable_if<is_supported_vector<VecA>::value && is_supported_vector<VecB>::value, VecA>::type&
vector_diff(const VecA &a, const VecB &b, VecA&& diff=VecA()) {
    std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(diff));
    return diff;
}

template <typename T>
void getNonzeroIndicesAndValues(const std::vector<T> &d,
                                std::vector<size_t> &nonzero_indices,
                                std::vector<T> &nonzero_values,
                                double epsilon = std::numeric_limits<T>::epsilon()) {
    for (size_t i = 0; i < d.size(); ++i) {
        if constexpr (std::is_signed<T>::value) {
            if (std::abs(d[i]) > epsilon) {
                nonzero_indices.push_back(i);
                nonzero_values.push_back(d[i]);
            }
        }else{
            if (d[i] > epsilon) {
                nonzero_indices.push_back(i);
                nonzero_values.push_back(d[i]);
            }
        }
    }
}

template <typename T>
void getNonzeroIndicesAndValues(const sili::unique_vector<T> &d,
                                sili::unique_vector<size_t> &nonzero_indices,
                                sili::unique_vector<T> &nonzero_values,
                                double epsilon = std::numeric_limits<T>::epsilon()) {
    for (size_t i = 0; i < d.size(); ++i) {
        if constexpr (std::is_signed<T>::value) {
            if (std::abs(d[i]) > epsilon) {
                nonzero_indices.push_back(i);
                nonzero_values.push_back(d[i]);
            }
        }else{
            if (d[i] > epsilon) {
                nonzero_indices.push_back(i);
                nonzero_values.push_back(d[i]);
            }
        }
    }
}

template <typename VecA, typename VecB, typename T = typename VecA::value_type>
typename std::enable_if<is_supported_vector<VecA>::value && is_supported_vector<VecB>::value, std::string>::type vector_diff_string(
    const VecA &a,
    const VecB &b,
    double epsilon = std::numeric_limits<T>::epsilon()) {
    std::ostringstream oss;
    oss << "Vector A: " << a << "\n";
    oss << "Vector B: " << b << "\n";
    if (a.size() == b.size()) {
        std::vector<T> d(a.size());
        std::transform(a.begin(), a.end(), b.begin(), d.begin(), std::minus<T>());
        std::vector<size_t> nonzero_indices;
        std::vector<T> nonzero_values;
        getNonzeroIndicesAndValues(d, nonzero_indices, nonzero_values, epsilon);
        oss << "Diff indices: " << nonzero_indices;
        oss << "Diff values: " << nonzero_values;
    } else {
        std::vector<T> diff;
        std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(diff));
        oss << "Diff: " << vector_diff(a, b);
    }
    return oss.str();
}
/*
template<typename T>
std::string vector_diff_string(const sili::unique_vector<T>& a, const sili::unique_vector<T>& b, double
epsilon=std::numeric_limits<T>::epsilon()) { std::ostringstream oss; oss << "Vector A: " << a << "\n"; oss << "Vector B:
" << b << "\n"; if(a.size()==b.size()){ std::vector<T> d(a.size()); std::transform(a.begin(), a.end(), b.begin(),
d.begin(), std::minus<T>()); std::vector<size_t> nonzero_indices; std::vector<T> nonzero_values;
        getNonzeroIndicesAndValues(d, nonzero_indices, nonzero_values, epsilon);
        oss << "Diff indices: " << nonzero_indices;
        oss << "Diff values: " << nonzero_values;
    }else{
        std::vector<T> diff;
        std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(diff));
        oss << "Diff: " << vector_diff(a, b);
    }
    return oss.str();
}
*/
template <class T, class U>
bool almost_equal(const std::vector<T> &a,
                  const std::vector<U> &b,
                  double epsilon = std::numeric_limits<T>::epsilon()) {
    // Check if both vectors are of the same size
    if (a.size() != b.size()) {
        return false; // Vectors are not the same size, so they can't be equal
    }

    // Compare each element within the specified epsilon tolerance
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > epsilon) {
            return false; // The difference between elements exceeds the epsilon tolerance
        }
    }

    return true; // All elements are approximately equal within the epsilon tolerance
}

template <class T, class U>
bool almost_equal(const sili::unique_vector<T> &a,
                  const sili::unique_vector<U> &b,
                  double epsilon = std::numeric_limits<T>::epsilon()) {
    // Check if both vectors are of the same size
    if (a.size() != b.size()) {
        return false; // Vectors are not the same size, so they can't be equal
    }

    // Compare each element within the specified epsilon tolerance
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > epsilon) {
            return false; // The difference between elements exceeds the epsilon tolerance
        }
    }

    return true; // All elements are approximately equal within the epsilon tolerance
}

// Helper function to convert arrays to vectors
template <typename T> std::vector<T> vec(T *arr, size_t size) { return std::vector<T>(arr, arr + size); }

#define CHECK_VECTOR_EQUAL(a, b)                                                                                       \
    do {                                                                                                               \
        INFO(vector_diff_string(a, b, 0));                                                                             \
        bool success = a == b;                                                                                         \
        CHECK(success);                                                                                                \
    } while ((void)0, 0)

#define CHECK_VECTOR_ALMOST_EQUAL_3(a, b, e)                                                                           \
    do {                                                                                                               \
        INFO(vector_diff_string(a, b, e));                                                                             \
        bool success = almost_equal(a, b, e);                                                                          \
        CHECK(success);                                                                                                \
    } while ((void)0, 0)

#define CHECK_VECTOR_ALMOST_EQUAL_2(a, b)                                                                              \
    do {                                                                                                               \
        INFO(vector_diff_string(a, b));                                                                                \
        bool success = almost_equal(a, b);                                                                             \
        CHECK(success);                                                                                                \
    } while ((void)0, 0)

#define CHECK_VECTOR_ALMOST_EQUAL_x(x, a, b, e, FUNC, ...) FUNC

#define CHECK_VECTOR_ALMOST_EQUAL(...)                                                                                 \
    CHECK_VECTOR_ALMOST_EQUAL_x(                                                                                       \
        , ##__VA_ARGS__, CHECK_VECTOR_ALMOST_EQUAL_3(__VA_ARGS__), CHECK_VECTOR_ALMOST_EQUAL_2(__VA_ARGS__), )

#define REQUIRE_VECTOR_EQUAL(a, b)                                                                                     \
    do {                                                                                                               \
        INFO(vector_diff_string(a, b));                                                                                \
        bool success = a == b;                                                                                         \
        REQUIRE(success);                                                                                              \
    } while ((void)0, 0)

// For nested vectors
#define CHECK_NESTED_VECTOR_EQUAL(a, b)                                                                                \
    do {                                                                                                               \
        REQUIRE(a.size() == b.size());                                                                                 \
        for (size_t i = 0; i < a.size(); ++i) {                                                                        \
            INFO("Mismatch at index " << i);                                                                           \
            CHECK_VECTOR_EQUAL(a[i], b[i]);                                                                            \
        }                                                                                                              \
    } while ((void)0, 0)

#define REQUIRE_NESTED_VECTOR_EQUAL(a, b)                                                                              \
    do {                                                                                                               \
        REQUIRE(a.size() == b.size());                                                                                 \
        for (size_t i = 0; i < a.size(); ++i) {                                                                        \
            INFO("Mismatch at index " << i);                                                                           \
            REQUIRE_VECTOR_EQUAL(a[i], b[i]);                                                                          \
        }                                                                                                              \
    } while ((void)0, 0)

template <typename T>
std::string csr_diff_string(const sparse_struct<size_t, CSRPtrs<size_t>, CSRIndices<size_t>, UnaryValues<T>>& csr,
                            const std::vector<size_t>& expected_ptrs,
                            const std::vector<size_t>& expected_indices,
                            const std::vector<T>& expected_values,
                            size_t expected_rows,
                            size_t expected_cols,
                            double epsilon = std::numeric_limits<T>::epsilon()) {
    std::ostringstream oss;

    // Dimensions
    oss << "CSR rows: " << csr.rows << ", Expected rows: " << expected_rows << "\n";
    oss << "CSR cols: " << csr.cols << ", Expected cols: " << expected_cols << "\n";

    // Pointers
    std::vector<size_t> ptrs_vec(csr.ptrs[0].get(), csr.ptrs[0].get() + csr.rows + 1);
    oss << "CSR ptrs: " << ptrs_vec << ", Expected ptrs: " << expected_ptrs << "\n";
    //oss << "Ptrs diff: " << vector_diff_string(ptrs_vec, expected_ptrs, 0) << "\n";

    // Number of non-zero elements
    size_t nnz = csr.nnz();
    oss << "CSR nnz: " << nnz << " ,Expected nnz: " << expected_indices.size() << "\n";

    // Indices
    std::vector<size_t> indices_vec(csr.indices[0].get(), csr.indices[0].get() + nnz);
    oss << "CSR indices: " << indices_vec << ", Expected indices: " << expected_indices << "\n";
    //oss << "Indices diff: " << vector_diff_string(indices_vec, expected_indices, 0) << "\n";

    // Values
    std::vector<T> values_vec(csr.values[0].get(), csr.values[0].get() + nnz);
    oss << "CSR values: " << values_vec << ", Expected values: " << expected_values << "\n";
    //oss << "Values diff: " << vector_diff_string(values_vec, expected_values, epsilon) << "\n";

    return oss.str();
}

template <typename T>
bool csr_almost_equal(const sparse_struct<size_t, CSRPtrs<size_t>, CSRIndices<size_t>, UnaryValues<T>>& csr,
                        const std::vector<size_t>& expected_ptrs,
                        const std::vector<size_t>& expected_indices,
                        const std::vector<T>& expected_values,
                        size_t expected_rows,
                        size_t expected_cols,
                        double epsilon = std::numeric_limits<T>::epsilon()) {
    // Check dimensions
    if (csr.rows != expected_rows || csr.cols != expected_cols) {
        return false;
    }

    // Check ptrs
    std::vector<size_t> ptrs_vec(csr.ptrs[0].get(), csr.ptrs[0].get() + csr.rows + 1);
    if (ptrs_vec != expected_ptrs) {
        return false;
    }

    // Check nnz consistency
    size_t nnz = csr.nnz();
    if (nnz != expected_indices.size() || nnz != expected_values.size()) {
        return false;
    }

    // Check indices
    std::vector<size_t> indices_vec(csr.indices[0].get(), csr.indices[0].get() + nnz);
    if (indices_vec != expected_indices) {
        return false;
    }

    // Check values
    std::vector<T> values_vec(csr.values[0].get(), csr.values[0].get() + nnz);
    if constexpr (std::is_floating_point<T>::value) {
        return almost_equal(values_vec, expected_values, epsilon);
    } else {
        return values_vec == expected_values;
    }
}

#define CHECK_CSR_EQUAL(csr, expected_ptrs, expected_indices, expected_values, expected_rows, expected_cols) \
    do { \
        INFO(csr_diff_string(csr, expected_ptrs, expected_indices, expected_values, expected_rows, expected_cols, 0)); \
        bool success = csr_almost_equal(csr, expected_ptrs, expected_indices, expected_values, expected_rows, expected_cols, 0); \
        CHECK(success); \
    } while ((void)0, 0)

#define CHECK_CSR_ALMOST_EQUAL_3(csr, expected_ptrs, expected_indices, expected_values, expected_rows, expected_cols, epsilon) \
    do { \
        INFO(csr_diff_string(csr, expected_ptrs, expected_indices, expected_values, expected_rows, expected_cols, epsilon)); \
        bool success = csr_almost_equal(csr, expected_ptrs, expected_indices, expected_values, expected_rows, expected_cols, epsilon); \
        CHECK(success); \
    } while ((void)0, 0)

#define CHECK_CSR_ALMOST_EQUAL_2(csr, expected_ptrs, expected_indices, expected_values, expected_rows, expected_cols) \
    do { \
        using ValueType = std::remove_reference_t<decltype(csr.values[0][0])>; \
        INFO(csr_diff_string(csr, expected_ptrs, expected_indices, expected_values, expected_rows, expected_cols, std::numeric_limits<ValueType>::epsilon())); \
        bool success = csr_almost_equal(csr, expected_ptrs, expected_indices, expected_values, expected_rows, expected_cols, std::numeric_limits<ValueType>::epsilon()); \
        CHECK(success); \
    } while ((void)0, 0)

 #define CHECK_CSR_ALMOST_EQUAL_x(x, csr, expected_ptrs, expected_indices, expected_values, expected_rows, expected_cols, epsilon, FUNC, ...) FUNC

#define CHECK_CSR_ALMOST_EQUAL(...) \
        CHECK_CSR_ALMOST_EQUAL_x(, ##__VA_ARGS__, CHECK_CSR_ALMOST_EQUAL_3(__VA_ARGS__), CHECK_CSR_ALMOST_EQUAL_2(__VA_ARGS__))

#endif