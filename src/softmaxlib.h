#pragma once
// useful typedefs
#include <stdint.h>

// user struct
typedef struct {volatile float *src, *dest;} softmax_ctrl_struct;

// registers
#define CTRL_REG    (volatile unsigned *)(0xC0000008)
#define DST_PTR_REG (volatile unsigned *)(0xC0000004)
#define SRC_PTR_REG (volatile unsigned *)(0xC0000000)

// available memory space, first to last byte
#define ACCEL_MEM_BEGIN (volatile float*)(0xB0000000)
#define ACCEL_MEM_END   (volatile float*)(0xBFFFFFFC)
#define ACCEL_MEM_CAPACITY (unsigned)(0xC0000000 - 0xB0000000)/4

// Library functions
int      softmax_allocate (      softmax_ctrl_struct *user_data, unsigned count);
void     softmax_execute  (const softmax_ctrl_struct *user_data, unsigned count);
unsigned softmax_test_done();

// Utility functions for -nostdlib
void print_str(const char*);
void print_hex(uint32_t);

// Helper for comparing floats
unsigned compare_ulp(float, float, unsigned);

// Things we need for compiling with -nostdlib
int main();
