/**
 * @file sparse_matrix.hpp
 * @brief Sparse matrix library with CSR and COO format support.
 */

#ifndef __SPARSE_STRUCT_HPP_
#define __SPARSE_STRUCT_HPP_

#include <cstddef>
#include <memory>

/**
 * @brief Type trait to check if a type is a std::array.
 * @tparam T The type to check.
 */
template <typename T>
struct is_std_array : std::false_type {};

/**
 * @brief Specialization of is_std_array for std::array types.
 * @tparam T The element type of the array.
 * @tparam N The size of the array.
 */
template <typename T, std::size_t N>
struct is_std_array<std::array<T, N>> : std::true_type {};

/**
 * @brief Helper variable template to check if a type is a std::array.
 * @tparam T The type to check.
 */
template <typename T>
constexpr bool is_std_array_v = is_std_array<T>::value;


/**
 * @brief Alias template for CSR pointers, stored as an array of unique pointers.
 * @tparam SIZE_TYPE The type used for sizes and indices.
 */
template <class SIZE_TYPE>
using CSRPointers = std::array<std::unique_ptr<SIZE_TYPE[]>, 1>;

/**
 * @brief Alias template for CSR indices, stored as an array of unique pointers.
 * @tparam SIZE_TYPE The type used for sizes and indices.
 */
template <class SIZE_TYPE>
using CSRIndices = std::array<std::unique_ptr<SIZE_TYPE[]>, 1>;

/**
 * @brief Alias template for COO pointers, stored as a single size value (nnz).
 * @tparam SIZE_TYPE The type used for sizes and indices.
 */
template <class SIZE_TYPE>
using COOPointers = SIZE_TYPE;  // just store nnz

/**
 * @brief Alias template for COO indices, stored as two arrays of unique pointers.
 * @tparam SIZE_TYPE The type used for sizes and indices.
 */
template <class SIZE_TYPE>
using COOIndices = std::array<std::unique_ptr<SIZE_TYPE[]>, 2>;

/**
 * @brief Alias template for unary values, stored as one array of unique pointers.
 * @tparam VALUE_TYPE The type of the values stored in the sparse matrix.
 */
template <class VALUE_TYPE>
using UnaryValues = std::array<std::unique_ptr<VALUE_TYPE[]>, 1>;

/**
 * @brief Alias template for binary values, stored as two arrays of unique pointers.
 * @tparam VALUE_TYPE The type of the values stored in the sparse matrix.
 */
template <class VALUE_TYPE>
using BiValues = std::array<std::unique_ptr<VALUE_TYPE[]>, 2>;

/**
 * @brief Alias template for ternary values, stored as three arrays of unique pointers.
 * @tparam VALUE_TYPE The type of the values stored in the sparse matrix.
 */
template <class VALUE_TYPE>
using TriValues = std::array<std::unique_ptr<VALUE_TYPE[]>, 3>;

/**
 * @brief Alias template for quaternary values, stored as four arrays of unique pointers.
 * @tparam VALUE_TYPE The type of the values stored in the sparse matrix.
 */
template <class VALUE_TYPE>
using QuadValues = std::array<std::unique_ptr<VALUE_TYPE[]>, 4>;

/**
 * @brief Alias template for quinary values, stored as five arrays of unique pointers.
 * @tparam VALUE_TYPE The type of the values stored in the sparse matrix.
 */
template <class VALUE_TYPE>
using PentaValues = std::array<std::unique_ptr<VALUE_TYPE[]>, 5>;

/**
 * @brief Helper variable template to determine the number of index arrays.
 * @tparam INDEX_ARRAYS The type of the indices (e.g., std::array or tuple).
 */
template <typename INDEX_ARRAYS>
constexpr std::size_t num_indices = std::tuple_size<INDEX_ARRAYS>::value;

/**
 * @brief A template structure representing a sparse matrix in CSR or COO format.
 *
 * This structure holds pointers, indices, and values for a sparse matrix, along with
 * the number of rows, columns, and optional reserved space. It supports various
 * formats via template parameters.
 *
 * @tparam SIZE_TYPE The type used for sizes and indices (e.g., int, size_t).
 * @tparam PTRS The type for pointers (e.g., CSRPointers or COOPointers).
 * @tparam INDICES The type for indices (e.g., CSRIndices or COOIndices).
 * @tparam VALUES The type for values (e.g., UnaryValues, BiValues).
 *
 * @code
 * // Example: CSR sparse matrix
 * using SIZE_TYPE = int;
 * using PTRS = CSRPointers<SIZE_TYPE>;
 * using INDICES = CSRIndices<SIZE_TYPE>;
 * using VALUES = UnaryValues<double>;
 * PTRS ptrs = ...;  // Initialize pointers
 * INDICES indices = ...;  // Initialize indices
 * VALUES values = ...;  // Initialize values
 * sparse_struct<SIZE_TYPE, PTRS, INDICES, VALUES> matrix(ptrs, indices, values, 10, 10);
 * @endcode
 */
template <class SIZE_TYPE, class PTRS, class INDICES, class VALUES>
struct sparse_struct {
    PTRS ptrs;               // Pointers sub-template
    INDICES indices;         // Indices sub-template
    VALUES values;           // Values sub-template
    SIZE_TYPE rows;
    SIZE_TYPE cols;
    SIZE_TYPE _reserved_space = 0;

    static constexpr std::size_t n_index_arrays = num_indices<INDICES>;
    static constexpr std::size_t n_value_arrays = num_indices<VALUES>;
    static constexpr std::size_t n_pointer_arrays = num_indices<PTRS>;

    /**
     * @brief Default constructor, initializes an empty sparse matrix.
     */
    sparse_struct()
        : rows(0), cols(0), _reserved_space(0) {}

    /**
     * @brief Constructor for pre-allocated arrays with reserved space.
     * @param p Pointers sub-template (moved into the structure).
     * @param ind Indices sub-template (moved into the structure).
     * @param val Values sub-template (moved into the structure).
     * @param num_p Number of rows.
     * @param max_idx Number of columns.
     * @param reserved Reserved space for future expansion.
     */
    sparse_struct(PTRS& p, INDICES& ind, VALUES& val, SIZE_TYPE num_p, SIZE_TYPE max_idx, SIZE_TYPE reserved)
        : ptrs(std::move(p)), indices(std::move(ind)), values(std::move(val)),
          rows(num_p), cols(max_idx), _reserved_space(reserved) {}

    /**
     * @brief Constructor for pre-allocated arrays without reserved space.
     * @param p Pointers sub-template (moved into the structure).
     * @param ind Indices sub-template (moved into the structure).
     * @param val Values sub-template (moved into the structure).
     * @param num_p Number of rows.
     * @param max_idx Number of columns.
     */
    sparse_struct(PTRS& p, INDICES& ind, VALUES& val, SIZE_TYPE num_p, SIZE_TYPE max_idx)
        : sparse_struct(std::move(p), std::move(ind), std::move(val), num_p, max_idx, 0) {}

    /**
     * @brief Get the number of non-zero elements in the sparse matrix.
     *
     * If PTRS is an array type (e.g., CSR), returns the last pointer value.
     * If PTRS is a single value (e.g., COO), returns that value directly.
     *
     * @return The number of non-zero elements.
     */
    SIZE_TYPE nnz() const {
        if constexpr (std::is_array_v<decltype(ptrs)> || is_std_array_v<decltype(ptrs)>) { // Check if ptrs is an array type
            return (ptrs[ptrs.size()-1] != nullptr) ? ptrs[ptrs.size()-1][rows] : 0;
        } else { // ptrs is a single nnz value
            return ptrs;
        }
    }

};

// tri = weight multiplier, backprop, importance (for optim). Adagrad would use 2 for optim, using quad.
// Since all these have the same indices, it's much cheaper to store them in the same csr.
template <class SIZE_TYPE, class VALUE_TYPE>
using CSRSynapses = sparse_struct<SIZE_TYPE, CSRPointers<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, TriValues<VALUE_TYPE> >;
// easier to use in some algorithms
template <class SIZE_TYPE, class VALUE_TYPE>
using COOSynapses = sparse_struct<SIZE_TYPE, COOPointers<SIZE_TYPE>, COOIndices<SIZE_TYPE>, TriValues<VALUE_TYPE> >;

// new weights are pre-optim and didn't contribute to forward, so no values and no importance yet, only grad.
template <class SIZE_TYPE, class VALUE_TYPE>
using CSRSynaptogenesis = sparse_struct<SIZE_TYPE, CSRPointers<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, UnaryValues<VALUE_TYPE> >;

template <class SIZE_TYPE, class VALUE_TYPE>
using CSRInput = sparse_struct<SIZE_TYPE, CSRPointers<SIZE_TYPE>, CSRIndices<SIZE_TYPE>, UnaryValues<VALUE_TYPE> >;

//easier to use in some algorithms
template <class SIZE_TYPE, class VALUE_TYPE>
using COOSynaptogenesis = sparse_struct<SIZE_TYPE, COOPointers<SIZE_TYPE>, COOIndices<SIZE_TYPE>, UnaryValues<VALUE_TYPE> >;

template <class SYNAPSES, class SYNAPTOGENESIS>
struct sparse_weights{
    SYNAPSES connections;
    SYNAPTOGENESIS probes;
};

template <class SIZE_TYPE, class VALUE_TYPE>
using SparseLinearWeights = sparse_weights<CSRSynapses<SIZE_TYPE, VALUE_TYPE>, COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE>>;


#endif