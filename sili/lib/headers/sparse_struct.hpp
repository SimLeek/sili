/**
 * @file sparse_matrix.hpp
 * @brief Sparse matrix library with CSR and COO format support.
 */

#ifndef __SPARSE_STRUCT_HPP_
#define __SPARSE_STRUCT_HPP_

/*todo: make a sparse_database.hpp version of this
 *  use MySQL to do so: https://x.com/i/grok/share/nf5C5MSMBZqsXfdBWYJ2LoUlC
 *  for sparse linear forward: read the specific rows of a csc, compute the output in parallel
 *  other operations will also need to be modified.
 */
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

//TODO: MOVE THESE TO A BINDINGS FILE AND USE pybind11_add_module(namr ${SOURCES} "${SOURCE_DIR}/bindings.cpp")

/*
#include <sstream>
#include <string>
#include <array>
#include <cstdint>
#include <bit>
#include <stdfloat>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// -----------------------------------------------------------------------------
// Endianness Helpers and Serialization Functions (same as before)
// -----------------------------------------------------------------------------

constexpr bool is_host_little_endian() {
    return std::endian::native == std::endian::little;
}

template <typename T>
T swap_endian(T val) {
    static_assert(std::is_arithmetic_v<T>, "swap_endian only supports arithmetic types.");
    union {
        T val;
        unsigned char bytes[sizeof(T)];
    } src, dest;
    src.val = val;
    for (std::size_t i = 0; i < sizeof(T); i++) {
        dest.bytes[i] = src.bytes[sizeof(T) - 1 - i];
    }
    return dest.val;
}

template <typename T>
void write_value(std::ostream& os, const T& val) {
    T tmp = val;
    if (!is_host_little_endian()) {
        tmp = swap_endian(tmp);
    }
    os.write(reinterpret_cast<const char*>(&tmp), sizeof(T));
}

template <typename T>
void read_value(std::istream& is, T& val) {
    is.read(reinterpret_cast<char*>(&val), sizeof(T));
    if (!is_host_little_endian()) {
        val = swap_endian(val);
    }
}

// Serialization for a single sparse_struct into an existing stream.
template <typename SIZE_TYPE, typename PTRS, typename INDICES, typename VALUES>
void serialize_sparse_struct(std::ostream& os, const sparse_struct<SIZE_TYPE, PTRS, INDICES, VALUES>& mat) {
    // Write dimensions and reserved space.
    write_value(os, mat.rows);
    write_value(os, mat.cols);
    write_value(os, mat._reserved_space);
    // Write nnz.
    SIZE_TYPE nnz = mat.nnz();
    write_value(os, nnz);
    // Write pointer arrays.
    if constexpr(is_std_array_v<decltype(mat.ptrs)>) {
        for (const auto& ptr : mat.ptrs) {
            for (std::size_t i = 0; i < static_cast<std::size_t>(mat.rows) + 1; ++i) {
                write_value(os, ptr.get()[i]);
            }
        }
    } else {
        write_value(os, mat.ptrs);
    }
    // Write indices arrays.
    for (const auto& idx_ptr : mat.indices) {
        for (SIZE_TYPE i = 0; i < nnz; ++i) {
            write_value(os, idx_ptr.get()[i]);
        }
    }
    // Write value arrays.
    using value_t = std::remove_pointer_t<decltype(mat.values[0].get())>;
    for (const auto& val_ptr : mat.values) {
        for (SIZE_TYPE i = 0; i < nnz; ++i) {
            write_value(os, val_ptr.get()[i]);
        }
    }
}

// Deserialization from an existing stream.
template <typename SIZE_TYPE, typename PTRS, typename INDICES, typename VALUES>
void deserialize_sparse_struct(std::istream& is, sparse_struct<SIZE_TYPE, PTRS, INDICES, VALUES>& mat) {
    read_value(is, mat.rows);
    read_value(is, mat.cols);
    read_value(is, mat._reserved_space);
    SIZE_TYPE nnz = 0;
    read_value(is, nnz);
    // Read pointer arrays.
    if constexpr(is_std_array_v<decltype(mat.ptrs)>) {
        for (auto& ptr : mat.ptrs) {
            ptr.reset(new SIZE_TYPE[mat.rows + 1]);
            for (std::size_t i = 0; i < static_cast<std::size_t>(mat.rows) + 1; ++i) {
                read_value(is, ptr.get()[i]);
            }
        }
    } else {
        read_value(is, mat.ptrs);
        nnz = mat.ptrs;
    }
    // Read indices arrays.
    for (auto& idx_ptr : mat.indices) {
        idx_ptr.reset(new SIZE_TYPE[nnz]);
        for (SIZE_TYPE i = 0; i < nnz; ++i) {
            read_value(is, idx_ptr.get()[i]);
        }
    }
    // Read value arrays.
    using value_t = std::remove_pointer_t<decltype(mat.values[0].get())>;
    for (auto& val_ptr : mat.values) {
        val_ptr.reset(new value_t[nnz]);
        for (SIZE_TYPE i = 0; i < nnz; ++i) {
            read_value(is, val_ptr.get()[i]);
        }
    }
}

// -----------------------------------------------------------------------------
// Pickle Support Functions for CSRInput
// -----------------------------------------------------------------------------

template<class SIZE_TYPE, class VALUE_TYPE>
inline py::bytes csr_input_getstate(const CSRInput<SIZE_TYPE, VALUE_TYPE> &self) {
    std::ostringstream oss(std::ios::binary);
    serialize_sparse_struct(oss, self);
    return py::bytes(oss.str());
}

template<class SIZE_TYPE, class VALUE_TYPE>
inline CSRInput<SIZE_TYPE, VALUE_TYPE> csr_input_setstate(py::bytes state) {
    std::string buffer = state;  // Implicit conversion to std::string.
    std::istringstream iss(buffer, std::ios::binary);
    CSRInput<SIZE_TYPE, VALUE_TYPE> obj;
    deserialize_sparse_struct(iss, obj);
    return obj;
}

// -----------------------------------------------------------------------------
// Pickle Support Functions for CSRSynapses
// -----------------------------------------------------------------------------

template<class SIZE_TYPE, class VALUE_TYPE>
inline py::bytes csrsynapses_getstate(const CSRSynapses<SIZE_TYPE, VALUE_TYPE> &self) {
    std::ostringstream oss(std::ios::binary);
    serialize_sparse_struct(oss, self);
    return py::bytes(oss.str());
}

template<class SIZE_TYPE, class VALUE_TYPE>
inline CSRSynapses<SIZE_TYPE, VALUE_TYPE> csrsynapses_setstate(py::bytes state) {
    std::string buffer = state;
    std::istringstream iss(buffer, std::ios::binary);
    CSRSynapses<SIZE_TYPE, VALUE_TYPE> obj;
    deserialize_sparse_struct(iss, obj);
    return obj;
}


//COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE>

template<class SIZE_TYPE, class VALUE_TYPE>
inline py::bytes coosynaptogenesis_getstate(const COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE> &self) {
    std::ostringstream oss(std::ios::binary);
    serialize_sparse_struct(oss, self);
    return py::bytes(oss.str());
}

template<class SIZE_TYPE, class VALUE_TYPE>
inline COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE> coosynaptogenesis_setstate(py::bytes state) {
    std::string buffer = state;
    std::istringstream iss(buffer, std::ios::binary);
    COOSynaptogenesis<SIZE_TYPE, VALUE_TYPE> obj;
    deserialize_sparse_struct(iss, obj);
    return obj;
}

//SparseLinearWeights

template<class SIZE_TYPE, class VALUE_TYPE>
inline py::bytes SparseLinearWeights_getstate(const SparseLinearWeights<SIZE_TYPE, VALUE_TYPE> &self) {
    std::ostringstream oss(std::ios::binary);
    serialize_sparse_struct(oss, self.connections);
    serialize_sparse_struct(oss, self.probes);
    return py::bytes(oss.str());
}

template<class SIZE_TYPE, class VALUE_TYPE>
inline SparseLinearWeights<SIZE_TYPE, VALUE_TYPE> SparseLinearWeights_setstate(py::bytes state) {
    std::string buffer = state;
    std::istringstream iss(buffer, std::ios::binary);
    SparseLinearWeights<SIZE_TYPE, VALUE_TYPE> obj;
    deserialize_sparse_struct(iss, obj.connections);
    deserialize_sparse_struct(iss, obj.probes);
    return obj;
}

// -----------------------------------------------------------------------------
// Pybind11 Module Registration
// -----------------------------------------------------------------------------

template <typename SIZE_TYPE, typename VALUE_TYPE>
void declare_CSRInput(py::module &m, const std::string &size_typestr, const std::string &value_typestr) {
    // Create a unique class name, e.g., "CSRInput_uint32_double"
    std::string pyclass_name = "CSRInput_" + size_typestr + "_" + value_typestr;
    
    // Alias for the specific instantiation
    using CSRInput_t = CSRInput<SIZE_TYPE, VALUE_TYPE>;
    
    // Bind the class to Python
    py::class_<CSRInput_t>(m, pyclass_name.c_str())
        .def(py::init<>())
        .def("nnz", &CSRInput_t::nnz)
        .def(py::pickle(
            // Serialization (getstate)
            [](const CSRInput_t &self) {
                std::ostringstream oss(std::ios::binary);
                serialize_sparse_struct(oss, self);
                return py::bytes(oss.str());
            },
            // Deserialization (setstate)
            [](py::bytes state) {
                std::string buffer = state;
                std::istringstream iss(buffer, std::ios::binary);
                CSRInput_t obj;
                deserialize_sparse_struct(iss, obj);
                return obj;
            }
        ));
}

template <typename SIZE_TYPE, typename VALUE_TYPE>
void declare_SparseLinearWeights(py::module &m, const std::string &size_typestr, const std::string &value_typestr) {
    std::string pyclass_name = "SparseLinearWeights_" + size_typestr + "_" + value_typestr;
    using SparseLinearWeights_t = SparseLinearWeights<SIZE_TYPE, VALUE_TYPE>;
    
    py::class_<SparseLinearWeights_t>(m, pyclass_name.c_str())
        .def(py::init<>())
        .def_readonly("connections", &SparseLinearWeights_t::connections)
        .def_readonly("probes", &SparseLinearWeights_t::probes)
        .def(py::pickle(
            [](const SparseLinearWeights_t &self) {
                std::ostringstream oss(std::ios::binary);
                serialize_sparse_struct(oss, self.connections);
                serialize_sparse_struct(oss, self.probes);
                return py::bytes(oss.str());
            },
            [](py::bytes state) {
                std::string buffer = state;
                std::istringstream iss(buffer, std::ios::binary);
                SparseLinearWeights_t obj;
                deserialize_sparse_struct(iss, obj.connections);
                deserialize_sparse_struct(iss, obj.probes);
                return obj;
            }
        ));
}

PYBIND11_MODULE(sparse_bindings, m) {
    m.doc() = "Bindings for sparse matrix structures with pickle support.";

    declare_CSRInput<uint64_t, float>(m, "u64", "f32");
    declare_CSRInput<uint32_t, float>(m, "u32", "f32");
    declare_CSRInput<uint16_t, float>(m, "u16", "f32");
    declare_CSRInput<uint8_t, float>(m, "u8", "f32");

    declare_SparseLinearWeights<uint64_t, float>(m, "u64", "f32");
    declare_SparseLinearWeights<uint32_t, float>(m, "u32", "f32");
    declare_SparseLinearWeights<uint16_t, float>(m, "u16", "f32");
    declare_SparseLinearWeights<uint8_t, float>(m, "u8", "f32");
}*/



#endif