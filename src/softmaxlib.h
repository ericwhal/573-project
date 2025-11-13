#pragma once
// useful typedefs
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

// user struct
typedef struct {volatile float *src, *dest;} softmax_ctrl_struct;

// registers
static volatile unsigned *CTRL_REG  = (volatile unsigned *)(0xC0000008);
static volatile float **DST_PTR_REG = (volatile float   **)(0xC0000004);
static volatile float **SRC_PTR_REG = (volatile float   **)(0xC0000000);

// available memory space, first to last byte
static volatile float *ACCEL_MEM_BEGIN = (volatile float*)(0xB0000000);
static volatile float *ACCEL_MEM_END   = (volatile float*)(0xBFFFFFFC);
static unsigned ACCEL_MEM_CAPACITY = (unsigned)(0xC0000000 - 0xB0000000)/4;

// Library functions
unsigned softmax_allocate (      softmax_ctrl_struct *user_data, unsigned count);
void     softmax_execute  (const softmax_ctrl_struct *user_data, unsigned count);
unsigned softmax_test_done();

// Utility functions for -nostdlib
void print_str(const char*);
void print_hex(uint32_t);

// Things we need for compiling with -nostdlib
int main();
