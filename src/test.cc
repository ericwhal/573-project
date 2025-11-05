// user struct
typedef struct {volatile float *src, *dest;} softmax_ctrl_struct;
// registers
static volatile unsigned *ctrl_reg = (volatile unsigned*)(0x60000008);

void softmax_allocate(softmax_ctrl_struct *user_data, unsigned count) {
  user_data->src = (volatile float*)(0x40000000);
  user_data->dest = (volatile float*)(0x40000000);
}

void softmax_execute(softmax_ctrl_struct *user_data, unsigned count) {
  // ctrl reg
  *(volatile unsigned *)(0x60000008) = (count & ((1 << 31) - 1)) | (1 << 30);
  // dest reg
  *(volatile unsigned **)(0x60000004) = (unsigned *)user_data->dest;
  // src reg
  *(volatile unsigned **)(0x60000000) = (unsigned *)user_data->src;
}

unsigned softmax_test_done() {
  // read ctrl reg
  return (*(volatile unsigned *)(0x60000000) & (1 << 29)) > 0;
}

int main() {
  softmax_ctrl_struct my_struct;
  softmax_allocate(&my_struct, 100);
  softmax_execute(&my_struct, 100);
  return softmax_test_done();
}
