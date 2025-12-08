#include "softmaxlib.h"
#include <math.h>
#include <gem5/m5ops.h>

// The following header contains three values: two arrays of floats labeled logits and probs, and the (shared) length of the arrays.
// IT IS A GENERATED HEADER BY THE EVAL SCRIPT
// We have to use a macro for data_length because otherwise the compiler freaks
// #define data_length ...;
// float probs[] = { ... };
// float logits[] = { ... };
#include "data.h"

int main() {

  softmax_ctrl_struct my_data;
  int allocated = softmax_allocate(&my_data, data_length);

  // if(allocated < data_length) return allocated;

  for(int i = 0; i < allocated; ++i) {
    my_data.src[i] = logits[i];
  }

  // ------------------------- accel ROI ------------------------- 
  m5_reset_stats(0,0);
  softmax_execute(&my_data, allocated);
  while(!softmax_test_done()) {};
  m5_dump_stats(0,0);
  // ------------------------- end accel ROI ------------------------- 

  // ************************* Check results ************************* 
  int accel_failed = 0;
  for(int i = 0; i < allocated; ++i) {
    accel_failed += !compare_ulp(probs[i], my_data.dest[i], 15);
  }
  // *****************************************************************

  // return code: naive_failed in upper half, accel in lower half
  return accel_failed;
}
