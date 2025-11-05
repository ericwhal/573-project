#include "softmaxlib.h"

unsigned softmax_allocate(softmax_ctrl_struct *user_data, unsigned count) {
  if(!softmax_test_done()) return 0;
  unsigned allocated = ACCEL_MEM_CAPACITY - count < count ? ACCEL_MEM_CAPACITY : count;
  user_data->src = ACCEL_MEM_BEGIN;
  user_data->dest = ACCEL_MEM_BEGIN;
  return allocated;
}

void softmax_execute(softmax_ctrl_struct *user_data, unsigned count) {
  *DST_PTR_REG = user_data->dest;
  *SRC_PTR_REG = user_data->src;
  *CTRL_REG    = (count | (1 << 31));
}

unsigned softmax_test_done() {
  // read ctrl reg
  return (*CTRL_REG & (1 << 30)) > 0;
}

