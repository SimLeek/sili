#include <stdint.h>

// Function to perform SIMD-like addition on two 64-bit integers containing packed 2-bit values
uint64_t bitwise_add_simd_2bit_64(uint64_t a, uint64_t b) {
    uint64_t result = 0;
    uint64_t carry = 0;
    uint64_t mask = 0x5555555555555555ULL; // Mask for active 2-bit positions

    for (int i = 0; i < 2; ++i) {
        // Sum the current bits with the carry using XOR
        result |= (a ^ b ^ carry) & mask;
        // Prepare the next carry
        carry = (((a & b) | (carry & (a ^ b))) & mask) << 1;
        mask <<= 1;
    }
    return result;
}

// Function to perform SIMD-like subtraction on two 64-bit integers containing packed 2-bit values
uint64_t bitwise_subtract_simd_2bit_64(uint64_t a, uint64_t b) {
    uint64_t result = 0;
    uint64_t borrow = 0;
    uint64_t mask = 0x5555555555555555ULL; // Mask for active 2-bit positions

    for (int i = 0; i < 2; ++i) {
        // Subtraction with borrow
        result |= ((a ^ b ^ borrow) & mask);
        // Calculate borrow for the next bit
        borrow = ((~a & b) | (~(a ^ b) & borrow)) & mask;
        borrow <<= 1;
        mask <<= 1;
    }
    return result;
}

// Function to perform SIMD-like multiplication on two 64-bit integers containing packed 2-bit values
uint64_t bitwise_multiply_simd_2bit_64(uint64_t a, uint64_t b) {
    uint64_t result = 0;
    uint64_t mask = 0x5555555555555555ULL; // Mask for least significant bits of all 2-bit numbers

    for (int i = 0; i < 2; i++) {
        uint64_t partial_product = 0;
        uint64_t carry = 0;
        uint64_t shifted_a = a;

        for (int j = 0; j < 2; j++) {
            // AND operation for multiplication of current bits
            uint64_t mult = (shifted_a & mask) & (b & mask);
            
            // Add partial product to result
            uint64_t sum = partial_product ^ mult ^ carry;
            carry = ((partial_product & mult) | (carry & (partial_product ^ mult))) & mask;
            
            partial_product = sum;
            shifted_a <<= 1;
        }

        result |= partial_product;
        mask <<= 1;
    }

    return result;
}

// Function to perform SIMD-like division on two 64-bit integers containing packed 2-bit values
uint64_t bitwise_divide_simd_2bit_64(uint64_t a, uint64_t b) {
    uint64_t result = 0;
    uint64_t remainder = 0;
    uint64_t mask = 0xAAAAAAAAAAAAAAAAULL; // Mask for most significant bits of all 2-bit numbers

    for (int i = 0; i < 2; i++) {
        // Shift remainder and bring down next bit from a
        remainder = (remainder << 1) | ((a & mask) ? 0x5555555555555555ULL : 0);
        
        // Check if we can subtract b from remainder
        uint64_t can_subtract = 0;
        uint64_t temp_remainder = remainder;
        uint64_t shifted_b = b & 0x3333333333333333ULL; // Consider only lower 2 bits of each packed number in b
        
        for (int j = 0; j < 2; j++) {
            uint64_t borrow = (~temp_remainder) & shifted_b;
            temp_remainder = temp_remainder ^ shifted_b;
            shifted_b = borrow << 1;
        }
        
        can_subtract = (temp_remainder & 0xAAAAAAAAAAAAAAAAULL) ? 0 : 0x5555555555555555ULL;
        
        // Update result and remainder
        result = (result << 1) | can_subtract;
        remainder = can_subtract ? (temp_remainder & 0x3333333333333333ULL) : remainder;

        mask >>= 1;
    }

    return result;
}