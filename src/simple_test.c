#include "softmaxlib.h"
#include <gem5/m5ops.h>

int main() {
  print_str("Testing accelerator with a simple example!\n");

  softmax_ctrl_struct my_data;
  int allocated = softmax_allocate(&my_data, 10);

  if(allocated < 10) return allocated;

  for(int i = 0; i < 10; ++i) {
    my_data.src[i] = i*1.0;
  }

  // ROI
  m5_work_begin(1, 0);
  m5_dump_stats(0,0);

  softmax_execute(&my_data, 10);
  while(!softmax_test_done()) {};

  m5_dump_stats(0,0);
  m5_work_end(1, 0);
  // end ROI

  for(int i = 0; i < 10; ++i) {
    print_str("dst[i] "); print_hex(*(uint32_t*)(my_data.dest + i));
  }

  return 0;
}
