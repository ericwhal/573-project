#include "softmaxlib.h"

unsigned softmax_allocate(softmax_ctrl_struct *user_data, unsigned count) {
  if(!softmax_test_done()) return 0;
  unsigned allocated = ACCEL_MEM_CAPACITY - count < count ? ACCEL_MEM_CAPACITY : count;
  user_data->src = ACCEL_MEM_BEGIN;
  user_data->dest = ACCEL_MEM_BEGIN;
  return allocated;
}

void softmax_execute(const softmax_ctrl_struct *user_data, unsigned count) {
  *DST_PTR_REG = user_data->dest;
  *SRC_PTR_REG = user_data->src;
  *CTRL_REG    = (count | (1 << 31));
}

unsigned softmax_test_done() {
  // read ctrl reg
  return (*CTRL_REG & (1 << 30)) > 0;
}

// Things we need for compiling with -nostdlib
// to avoid syscalls
void print_str(const char *s)
{
    register uint32_t a0 asm("a0") = 1;
    register const char *a1 asm("a1") = s;
    register uint32_t a2 asm("a2") = 0;
    while (s[a2]) a2++;
    register uint32_t a7 asm("a7") = 64;
    asm volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a7));
}

void print_hex(uint32_t val)
{
    char buf[12];
    buf[0] = '0';
    buf[1] = 'x';
    for (int i = 0; i < 8; i++) {
        int nibble = (val >> ((7 - i) * 4)) & 0xF;
        buf[2 + i] = (nibble < 10) ? ('0' + nibble) : ('a' + nibble - 10);
    }
    buf[10] = '\n';
    buf[11] = 0;
    print_str(buf);
}

// Spoof _start() to just call main() as if it were A-OK
void _start() {
  print_hex((unsigned)main());
  // exit(0)
  asm volatile("li a7, 93\nli a0, 0\necall");
  
}
