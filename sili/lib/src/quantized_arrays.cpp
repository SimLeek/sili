#include <memory>
#include <stdexcept>
#include <cmath>
#include <cstdint>

// Template class for quantized array with arbitrary BITS (1 to 7) for integer storage
//note: accessing nearby values is NOT threadsafe. This class will need to be reworked if it's used for random access
template <int BITS>
class QuantizedIntArray {
    static_assert(BITS > 0 && BITS <= 7, "Bits must be between 1 and 7.");

    std::unique_ptr<uint8_t[]> data;
    int size;

    // Helper function to set BITS-bit value at a specific index
    void setValue(int index, int8_t value) {
        int byteIndex = (index * BITS) / 8;
        int bitOffset = (index * BITS) % 8;

        uint8_t mask = (1 << BITS) - 1;  // BITS-bit mask

        // Adjust the value to be in the correct range
        value &= mask;

        // Clear existing bits and set the new value
        data[byteIndex] &= ~(mask << bitOffset);  // Clear existing bits
        data[byteIndex] |= (value << bitOffset);  // Set the new value

        // Only include this logic if the value spans two bytes
        if constexpr (8 % BITS != 0) {
            if (bitOffset + BITS > 8) {
                int remainingBits = (bitOffset + BITS) - 8;
                data[byteIndex + 1] &= ~(mask >> (BITS - remainingBits));  // Clear upper part in next byte
                data[byteIndex + 1] |= (value >> (BITS - remainingBits));  // Set the upper part of value in next byte
            }
        }
    }

    // Helper function to get BITS-bit value at a specific index
    int8_t getValue(int index) const {
        int byteIndex = (index * BITS) / 8;
        int bitOffset = (index * BITS) % 8;

        uint8_t mask = (1 << BITS) - 1;  // BITS-bit mask

        // Retrieve the bits for the value
        uint8_t value = (data[byteIndex] >> bitOffset) & mask;

        // Only include this logic if the value spans two bytes
        if constexpr (8 % BITS != 0) {
            if (bitOffset + BITS > 8) {
                int remainingBits = (bitOffset + BITS) - 8;
                uint8_t upperPart = data[byteIndex + 1] & ((1 << remainingBits) - 1);
                value |= (upperPart << (BITS - remainingBits));
            }
        }

        // Adjust the value for two's complement interpretation (sign extension)
        if (value & (1 << (BITS - 1))) {
            return value | (0xFF << BITS);  // Sign extend the BITS-bit value
        }

        return value;
    }

public:
    QuantizedIntArray(int size) : size(size) {
        // Calculate how many bytes we need to store the quantized data
        int numBytes = (size * BITS + 7) / 8;
        data.reset(new uint8_t[numBytes]);
    }

    // Overload [] operator for getting and setting quantized values
    int8_t operator[](int index) const {
        if (index < 0 || index >= size) throw std::out_of_range("Index out of range");
        return getValue(index);
    }

    // This proxy object allows setting a value via the [] operator
    struct Proxy {
        QuantizedIntArray* arr;
        int index;

        Proxy(QuantizedIntArray* a, int i) : arr(a), index(i) {}

        // Overload assignment operator to set the value
        Proxy& operator=(int8_t value) {
            arr->setValue(index, value);
            return *this;
        }

        // Implicit conversion to int8_t when retrieving value
        operator int8_t() const {
            return arr->getValue(index);
        }
    };

    Proxy operator[](int index) {
        if (index < 0 || index >= size) throw std::out_of_range("Index out of range");
        return Proxy(this, index);
    }

    // Function to return the size of the array
    int getSize() const {
        return size;
    }
};

// Wrapper class for FLOAT_TYPE version of quantized array
template <class FLOAT_TYPE, int BITS>
class QuantizedFloatArray {
    static_assert(BITS > 0 && BITS <= 7, "Bits must be between 1 and 7.");

    QuantizedIntArray<BITS> quantizedArray;
    
public:
    FLOAT_TYPE multiplier_float;
    FLOAT_TYPE offset_float;
    // Constructor that initializes the quantized integer array and stores the multiplier and offset
    QuantizedFloatArray(int size, FLOAT_TYPE multiplier, FLOAT_TYPE offset)
        : quantizedArray(size), multiplier_float(multiplier), offset_float(offset) {
        }

    // Overload [] operator for getting and setting FLOAT_TYPE values
    /*FLOAT_TYPE operator[](int index) const {
        // Retrieve the quantized integer and apply FLOAT_TYPE transformation
        int8_t quantizedValue = quantizedArray[index];
        return quantizedValue * multiplier_float + offset_float;
    }*/

    struct Proxy {
        QuantizedFloatArray* arr;
        int index;

        Proxy(QuantizedFloatArray* a, int i) : arr(a), index(i) {}

        // Overload assignment operator to set the FLOAT_TYPE value
        Proxy& operator=(float value) {
            // Convert FLOAT_TYPE to quantized integer, then store in the internal quantized array
            int8_t quantizedValue = static_cast<int8_t>(std::round((value - arr->offset_float) / arr->multiplier_float));
            arr->quantizedArray[index] = quantizedValue;
            return *this;
        }

        // Implicit conversion to FLOAT_TYPE when retrieving value
        operator FLOAT_TYPE() const {
            int8_t quantizedValue = arr->quantizedArray[index];
            return quantizedValue * arr->multiplier_float + arr->offset_float;
        }
    };

    Proxy operator[](int index) {
        return Proxy(this, index);
    }

    // Function to return the size of the array
    int getSize() const {
        return quantizedArray.getSize();
    }
};

// Example usage
/*int main() {
    // 4-bit quantized array with FLOAT_TYPE conversion (multiplier = 0.1, offset = 0.0)
    QuantizedFloatArray<float, 4> arr4(10, 0.1f, 0.0f);  // 10 elements

    arr4[0] = -0.8f;
    arr4[1] = 0.7f;
    arr4[2] = -0.3f;
    arr4[3] = 0.4f;

    std::cout << "4-bit Quantized Float Array:" << std::endl;
    for (int i = 0; i < arr4.getSize(); ++i) {
        std::cout << "arr4[" << i << "] = " << arr4[i] << std::endl;
    }

    // 2-bit quantized array with FLOAT_TYPE conversion (multiplier = 0.25, offset = 0.0)
    QuantizedFloatArray<double, 2> arr2(10, 0.25f, 0.0f);  // 10 elements

    arr2[0] = -0.5f;
    arr2[1] = 0.25f;
    arr2[2] = 0.0f;
    arr2[3] = -0.25f;

    std::cout << "2-bit Quantized Float Array:" << std::endl;
    for (int i = 0; i < arr2.getSize(); ++i) {
        std::cout << "arr2[" << i << "] = " << arr2[i] << std::endl;
    }

    return 0;
}*/
