#include "softmaxlib.h"

int main() {
  print_str("Testing accelerator with a simple example!\n");

  softmax_ctrl_struct my_data;
  int allocated = softmax_allocate(&my_data, 10);

  if(allocated < 10) return allocated;

  for(int i = 0; i < 10; ++i) {
    my_data.src[i] = i*1.0;
  }

  softmax_execute(&my_data, 10);
  while(!softmax_test_done()) {};

  for(int i = 0; i < 10; ++i) {
    print_str("dst[i] "); print_hex(*(uint32_t*)(my_data.dest + i));
  }

  return 0;
}
