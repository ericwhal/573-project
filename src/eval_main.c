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

  // ------------------------- naive ROI ------------------------- 
  m5_reset_stats(0,0);
  // First pass: exponentiate (in-place) and sum
  float accum = 0;
  for(int i = 0; i < allocated; ++i) {
    logits[i] = isfinite(logits[i]) ? exp(logits[i]) : 0.;
    accum += logits[i];
  }
  // Second pass: divide by sum
  for(int i = 0; i < allocated; ++i) {
    logits[i] /= accum;
  }
  m5_dump_stats(0,0);
  // ------------------------- end naive ROI ------------------------- 

  // *************************  Check results ************************* 
  int naive_failed = 0;
  for(int i = 0; i < allocated; ++i) {
    naive_failed += !compare_ulp(probs[i], logits[i], 15);
  }
  // *****************************************************************

  // return code: naive_failed in upper half, accel in lower half
  return (naive_failed << 16) | accel_failed;
}
