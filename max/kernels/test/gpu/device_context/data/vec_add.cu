// Simple vector addition kernel for testing external cubin loading
// Compile with: nvcc -cubin -arch=sm_80 vec_add.cu -o vec_add.cubin

extern "C" __global__ void vec_add(const float *__restrict__ in0,
                                   const float *__restrict__ in1,
                                   float *__restrict__ output, int len) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < len) {
    output[tid] = in0[tid] + in1[tid];
  }
}
