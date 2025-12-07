#include "softmaxlib.h"
#include <gem5/m5ops.h>
#include <math.h>

int softmax_allocate(softmax_ctrl_struct *user_data, unsigned count) {
  // if(!softmax_test_done()) return -1;
  int allocated = ACCEL_MEM_CAPACITY < count ? ACCEL_MEM_CAPACITY : count;
  user_data->src  = ACCEL_MEM_BEGIN;
  user_data->dest = ACCEL_MEM_BEGIN;

  // return allocated;
  return count;
}

// Ignore annoying but safe warning bc we built a 32-bit ptr expectation into a 64-bit system
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpointer-to-int-cast"

void softmax_execute(const softmax_ctrl_struct *user_data, unsigned count) {
  *DST_PTR_REG = (unsigned) user_data->dest;
  *SRC_PTR_REG = (unsigned) user_data->src;
  *CTRL_REG    = (count*sizeof(float)) | (1 << 31);
}
// restore diagnostics
#pragma GCC diagnostic pop

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

unsigned compare_ulp(float a, float b, unsigned ulp)
{
  // bit cast
  unsigned *a_bits_ptr = (int*) &a;
  unsigned *b_bits_ptr = (int*) &b;
  unsigned a_bits = *a_bits_ptr;
  unsigned b_bits = *b_bits_ptr;

  unsigned a_biased = a_bits < 0 ? ~a_bits + 1 : a_bits | (1 << 31);
  unsigned b_biased = b_bits < 0 ? ~b_bits + 1 : b_bits | (1 << 31);

  unsigned distance = a_biased >= b_biased ? a_biased - b_biased : b_biased - a_biased;

  return distance <= ulp;
}

// Spoof _start() to just call main() as if it were A-OK
void _start() {
  // Execute main as 0th ROI, print result
  print_str("_start()\n");

  unsigned main_result = (unsigned) main();

  print_str("exit: ");
  print_hex(main_result);

  m5_exit(0);
}
