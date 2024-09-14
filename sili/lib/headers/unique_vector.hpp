// modified from: https://www.coin-or.org/CppAD/Doc/doxydoc/html/pod__vector_8hpp_source.html
// under license Eclipse Public License Version 1.0

#ifndef _UNIQUE_VECTOR_HPP
#define _UNIQUE_VECTOR_HPP

#include <algorithm>
#include <cstdlib>
#include <type_traits>
#include <utility>

#ifndef NDEBUG
#include <iostream>
#include <ostream>
#include <string>
#include <stdexcept>
#define ASSERT(condition, message)                                                                                     \
    do {                                                                                                               \
        if (!(condition)) {                                                                                            \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ << " line " << __LINE__ << ": "           \
                      << message << std::endl;                                                                         \
            std::terminate();                                                                                          \
        }                                                                                                              \
    } while (false)
#else
#define ASSERT(condition, message)                                                                                     \
    do {                                                                                                               \
    } while (false)
#endif

namespace sili {
const size_t cache_line_bytes = 64;

// thx: https://stackoverflow.com/a/28796458
template <typename Test, template <typename...> class Ref> struct is_specialization : std::false_type {};

template <template <typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref> : std::true_type {};

// non-copyable vector. Because there's no point in trying to make fast code when arrays are copied a million times
// behind the scenes
template <class Type> class unique_vector {
  private:
    size_t length_ = 0;
    size_t capacity_ = 0;
    Type *data_ = nullptr;
  public:
    explicit unique_vector(const unique_vector &) = delete;

    explicit unique_vector(unique_vector &&src)
        : length_(std::move(src.length_)), capacity_(std::move(src.capacity_)), data_(std::move(src.data_)) {}
    unique_vector(void)
        : length_(0), capacity_(0), data_(nullptr) { // ASSERT( std::is_standard_layout<size_t>() &&
                                                     // std::is_trivial<size_t>(), "not plain old data");
    }
    unique_vector(size_t n) : length_(0), capacity_(0), data_(nullptr) { extend(n); }

    template <typename wrong_type, typename = std::enable_if<!is_specialization<wrong_type, unique_vector>::value>>
    unique_vector(const std::initializer_list<wrong_type> &init_list) : length_(0), capacity_(0), data_(nullptr) {
        extend(init_list.size());
        if constexpr (!(std::is_standard_layout<Type>() && std::is_trivial<Type>())) {
            if constexpr (std::is_copy_constructible<Type>()) {
                size_t i = 0;
                for (auto &&elem : init_list) {
                    // new(data_ + i) Type();
                    new (&data_[i]) Type(elem);
                    i++;
                }
            } else {
                throw std::invalid_argument("Attempted to initialize a non copy constructable type");
            }
        } else {
            std::copy(init_list.begin(), init_list.end(), data_);
        }
    }

    // handle nested initializer lists without copying
    template <class wrong_type, typename = std::enable_if<!is_specialization<wrong_type, unique_vector>::value>>
    unique_vector(const std::initializer_list<std::initializer_list<wrong_type>> &&init_list)
        : length_(0), capacity_(0), data_(nullptr) {
        extend(init_list.size());
        if constexpr (std::is_copy_constructible<wrong_type>()) {
            size_t i = 0;
            for (const auto &sublist : init_list) {
                // Handle nested lists by constructing elements in place
                new (&data_[i]) Type(sublist);
                i++;
            }
        } else {
            throw std::invalid_argument("Attempted to initialize a non copy constructible type");
        }
    }

    ~unique_vector(void) {
        if (length_ > 0) {
            if constexpr (!(std::is_standard_layout<Type>() &&
                            std::is_trivial<Type>())) { // call destructor for each element
                size_t i;
                for (i = 0; i < length_; i++)
                    (data_ + i)->~Type();
            }
            free(data_);
        }
    }
    size_t size(void) const { return length_; }
    size_t capacity(void) const { return capacity_; }

    Type *data(void) { return data_; }

    const Type *data(void) const { return data_; }

    size_t extend(size_t n) {
        size_t old_length = length_;
        length_ += n;

        if (capacity_ >= length_) {
            if constexpr (!(std::is_standard_layout<Type>() && std::is_trivial<Type>())) {
                size_t i;
                for (i = old_length; i < length_; i++) {
                    // data_[i].~Type();
                    new (&data_[i]) Type();
                }
            }
            return old_length;
        }

        size_t old_capacity = capacity_;
        Type *old_data = data_;

        // get new memory and set capacity
        size_t length_bytes = length_ * sizeof(Type);
        size_t capacity_bytes = (length_bytes << 1);
        if (capacity_bytes < cache_line_bytes) {
            size_t aligned_cache_bytes = size_t((cache_line_bytes + sizeof(Type) - 1) / sizeof(Type)) * sizeof(Type);
            capacity_bytes = aligned_cache_bytes;
        }
        void *v_ptr;
        if constexpr (!(std::is_standard_layout<Type>() && std::is_trivial<Type>())) {
            v_ptr = calloc(length_ << 1,
                           sizeof(Type)); // avoid uninitialized access for pod vectors of pod vectors for example
        } else {
            v_ptr = malloc(capacity_bytes);
        }

        capacity_ = capacity_bytes / sizeof(Type);
        data_ = reinterpret_cast<Type *>(v_ptr);
        ASSERT(length_ <= capacity_, "length must be less than or equal to capacity");

        size_t i;
        if (old_capacity > 0) {
            if constexpr (!(std::is_standard_layout<Type>() && std::is_trivial<Type>())) {
                for (i = 0; i < old_length; i++) {
                    new (&data_[i]) Type(std::move(old_data[i]));
                    old_data[i].~Type();
                }
                for (i = old_length; i < length_; i++) {
                    new (&data_[i]) Type();
                }
            } else {
                std::move(old_data, old_data + old_length, data_);
            }
        }

        return old_length;
    }

    void resize(size_t n) {
        length_ = n;

        if (capacity_ < length_) {
            void *v_ptr;

            if (capacity_ > 0) {
                v_ptr = reinterpret_cast<void *>(data_);
                if constexpr (!(std::is_standard_layout<Type>() && std::is_trivial<Type>())) {
                    for (size_t i = 0; i < length_; i++)
                        (data_ + i)->~Type();
                }
                free(v_ptr);
            }

            size_t length_bytes = length_ * sizeof(Type);
            size_t capacity_bytes = length_bytes << 1;
            v_ptr = malloc(capacity_bytes);
            capacity_ = capacity_bytes / sizeof(Type);
            data_ = reinterpret_cast<Type *>(v_ptr);
            ASSERT(length_ <= capacity_, "Length was greater than capacity");
            if constexpr (!(std::is_standard_layout<Type>() && std::is_trivial<Type>())) {
                for (size_t i = 0; i < capacity_; i++)
                    new (data_ + i) Type();
            }
        }
    }

    Type &operator[](size_t i) {
        ASSERT(i < length_,
               "Attempted to access out of bounds element " + std::to_string(i) + ". Max: " + std::to_string(length_));
        return data_[i];
    }

    const Type &operator[](size_t i) const {
        ASSERT(i < length_,
               "Attempted to access out of bounds element " + std::to_string(i) + ". Max: " + std::to_string(length_));
        return data_[i];
    }

    // Iterator functions
    Type *begin() { return data_; }
    const Type *begin() const { return data_; }
    Type *end() { return data_ + length_; }
    const Type *end() const { return data_ + length_; }
    Type &back() {
        ASSERT(length_ > 0, "Attempted to access back() on an empty vector");
        return *(data_ + length_ - 1);
    }

    const Type &back() const {
        ASSERT(length_ > 0, "Attempted to access back() on an empty vector");
        return *(data_ + length_ - 1);
    }

    template <class OTHER_TYPE> bool operator==(const unique_vector<OTHER_TYPE> &b) const {
        return std::distance(begin(), end()) == std::distance(b.begin(), b.end()) &&
               std::equal(begin(), end(), b.begin());
    }

    void clear(void) {
        if (length_ > 0) {
            if constexpr (!(std::is_standard_layout<Type>() && std::is_trivial<Type>())) {
                size_t i;
                for (i = 0; i < length_; i++)
                    (data_ + i)->~Type();
            }
        }
        length_ = 0;
    }

    unique_vector &operator=(const unique_vector &) = delete;

    void swap(unique_vector &other) {
        std::swap(capacity_, other.capacity_);
        std::swap(length_, other.length_);
        std::swap(data_, other.data_);
    }

    void push_back(const Type &e) {
        size_t i = extend(1);
        data_[i] = e;
    }

    template <typename... Args> void emplace_back(const Args &&...args) {
        size_t i = extend(1);
        data_[i] = Type(args...);
    }
};

} // namespace sili
#endif