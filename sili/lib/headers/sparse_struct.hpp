#ifndef __SPARSE_STRUCT_HPP_
#define __SPARSE_STRUCT_HPP_

#include "csr.hpp"
#include <cstddef>
#include <memory>

//template checks
template <typename T>
struct is_std_array : std::false_type {};

template <typename T, std::size_t N>
struct is_std_array<std::array<T, N>> : std::true_type {};

template <typename T>
constexpr bool is_std_array_v = is_std_array<T>::value;


// Sub-template for pointers
template <class SIZE_TYPE>
using CSRPointers = std::array<std::unique_ptr<SIZE_TYPE[]>, 1>;

template <class SIZE_TYPE>
using CSRIndices = std::array<std::unique_ptr<SIZE_TYPE[]>, 1>;

template <class SIZE_TYPE>
using COOPointers = SIZE_TYPE;  // just store nnz

template <class SIZE_TYPE>
using COOIndices = std::array<std::unique_ptr<SIZE_TYPE[]>, 2>;

//tuples should also work if different types are needed
template <class VALUE_TYPE>
using UnaryValues = std::array<std::unique_ptr<VALUE_TYPE[]>, 1>;

template <class VALUE_TYPE>
using BiValues = std::array<std::unique_ptr<VALUE_TYPE[]>, 2>;

template <class VALUE_TYPE>
using TriValues = std::array<std::unique_ptr<VALUE_TYPE[]>, 3>;

template <class VALUE_TYPE>
using QuadValues = std::array<std::unique_ptr<VALUE_TYPE[]>, 4>;

template <class VALUE_TYPE>
using PentaValues = std::array<std::unique_ptr<VALUE_TYPE[]>, 5>;

// CSR Struct with sub-templates
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

    // Default constructor
    sparse_struct()
        : rows(0), cols(0), _reserved_space(0) {}

    // Constructor for pre-allocated arrays
    sparse_struct(PTRS& p, INDICES& ind, VALUES& val, SIZE_TYPE num_p, SIZE_TYPE max_idx, SIZE_TYPE reserved)
        : ptrs(std::move(p)), indices(std::move(ind)), values(std::move(val)),
          rows(num_p), cols(max_idx), _reserved_space(reserved) {}

    // Constructor without reserved size
    sparse_struct(PTRS& p, INDICES& ind, VALUES& val, SIZE_TYPE num_p, SIZE_TYPE max_idx)
        : sparse_struct(std::move(p), std::move(ind), std::move(val), num_p, max_idx, 0) {}

    // Get the number of non-zeros
    SIZE_TYPE nnz() const {
        if constexpr (std::is_array_v<decltype(ptrs)> || is_std_array_v<decltype(ptrs)>) { // Check if ptrs is an array type
            return (ptrs[ptrs.size()-1] != nullptr) ? ptrs[ptrs.size()-1][rows] : 0;
        } else { // ptrs is a single nnz value
            return ptrs;
        }
    }

    // Reserve space for indices and values
    /*void reserve(SIZE_TYPE new_reserved) {
        if (new_reserved > _reserved_space) {
            _reserved_space = new_reserved;

            // Resize index arrays
            for (std::size_t i = 0; i < indices.indices.size(); ++i) {
                auto new_indices = std::make_unique<SIZE_TYPE[]>(new_reserved);
                if (indices.indices[i]) {
                    std::copy(indices.indices[i].get(), indices.indices[i].get() + nnz(), new_indices.get());
                }
                indices.indices[i] = std::move(new_indices);
            }

            // Resize value arrays
            for (std::size_t i = 0; i < values.values.size(); ++i) {
                auto new_values = std::make_unique<typename std::remove_reference<decltype(*values.values[i])>::type[]>(new_reserved);
                if (values.values[i]) {
                    std::copy(values.values[i].get(), values.values[i].get() + nnz(), new_values.get());
                }
                values.values[i] = std::move(new_values);
            }
        }
    }*/
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