#include "softmaxlib.h"
#include <stdio.h>
#include <assert.h>

int main() {
  printf("Testing accelerator with a simple example!\n");
  softmax_ctrl_struct my_data;
  unsigned allocated = softmax_allocate(&my_data, 10);
  assert(10 == allocated);

  for(int i = 0; i < 10; ++i) {
    my_data.src[i] = i*1.0;
  }

  softmax_execute(&my_data, 10);
  while(!softmax_test_done()) {};

  for(int i = 0; i < 10; ++i) {
    printf("data[%2i] = %3f\n", i, my_data.dest[i]);
  }

  return 0;
}
