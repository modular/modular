#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cub/cub.cuh>
#include <cuda_fp8.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <functional>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

// build:=> nvcc -arch=sm_86 test.cu -std=c++17 -o test

using SizeType32 = int;
using DataType = __nv_bfloat16;

struct Params {
  void const *act;
  void const *weight;
  float alpha;
  void *output;
  SizeType32 m, n, k;

  Params(void const *_act, void const *_weight, float _alpha, void *_output,
         SizeType32 _m, SizeType32 _n, SizeType32 _k)
      : act(_act), weight(_weight), alpha(_alpha), output(_output), m(_m),
        n(_n), k(_k) {}
};

template <typename InputType, typename OutputType, SizeType32 TILE_M,
          SizeType32 TILE_N, SizeType32 BLOCK_SIZE>
__global__ void cudaCoreGemm(InputType const *__restrict__ act,
                             InputType const *__restrict__ weight, float alpha,
                             OutputType *__restrict__ output, SizeType32 m,
                             SizeType32 n, SizeType32 k) {
  using VecType = int4;
  static constexpr SizeType32 kStepK =
      static_cast<SizeType32>(128 / (8 * sizeof(InputType)));
  static constexpr SizeType32 kTileK = kStepK * BLOCK_SIZE;
  auto tileIdM = static_cast<SizeType32>(blockIdx.x * TILE_M);
  auto tileIdN = static_cast<SizeType32>(blockIdx.y * TILE_N);
  auto tid = static_cast<SizeType32>(threadIdx.x);
  float tile_a[kStepK], tile_w[TILE_N * kStepK];
  float acc[TILE_M * TILE_N];

  static_assert(kStepK % 4 == 0);
  using CvtSrcType = InputType;
  using CvtResType = float;
  static constexpr SizeType32 kCvtCount =
      static_cast<SizeType32>(sizeof(VecType) / sizeof(CvtSrcType));

#pragma unroll
  for (SizeType32 i = 0; i < TILE_M * TILE_N; ++i) {
    acc[i] = 0;
  }
  act += tileIdM * k;
  weight += tileIdN * k;
  output += tileIdM * n + tileIdN;
  for (SizeType32 idxK = tid * kStepK; idxK < k; idxK += kTileK) {
    for (SizeType32 i = 0; i < TILE_N; ++i) {
      auto tile_w_quantized =
          reinterpret_cast<VecType const *>(weight + i * k + idxK)[0];
#pragma unroll
      for (SizeType32 cvtIdx = 0; cvtIdx < kCvtCount; ++cvtIdx) {
        reinterpret_cast<CvtResType *>(tile_w)[i * kCvtCount + cvtIdx] =
            __bfloat162float(
                reinterpret_cast<CvtSrcType *>(&tile_w_quantized)[cvtIdx]);
      }
    }
#pragma unroll
    for (SizeType32 i = 0; i < TILE_M; ++i) {
      auto tile_a_quantized =
          reinterpret_cast<VecType const *>(act + i * k + idxK)[0];
#pragma unroll
      for (SizeType32 cvtIdx = 0; cvtIdx < kCvtCount; ++cvtIdx) {
        reinterpret_cast<CvtResType *>(tile_a)[cvtIdx] = __bfloat162float(
            reinterpret_cast<CvtSrcType *>(&tile_a_quantized)[cvtIdx]);
      }
#pragma unroll
      for (SizeType32 j = 0; j < TILE_N; ++j) {
#pragma unroll
        for (SizeType32 l = 0; l < kStepK; ++l) {
          acc[i * TILE_N + j] =
              fma(tile_a[l], tile_w[j * kStepK + l], acc[i * TILE_N + j]);
        }
      }
    }
  }

  typedef cub::WarpReduce<float> WarpReduce;

  static constexpr SizeType32 kWarpSize = 32;
  static constexpr SizeType32 kWarpNum = BLOCK_SIZE / kWarpSize;
  SizeType32 warpId = tid / kWarpSize, laneId = tid % kWarpSize;
  __shared__ float shmem[TILE_M * TILE_N * kWarpNum];
  __shared__ typename WarpReduce::TempStorage tempStorage[kWarpNum];
#pragma unroll
  for (SizeType32 mi = 0; mi < TILE_M; ++mi) {
#pragma unroll
    for (SizeType32 ni = 0; ni < TILE_N; ++ni) {
      float val = WarpReduce(tempStorage[warpId]).Sum(acc[mi * TILE_N + ni]);
      if (laneId == 0) {
        shmem[mi * TILE_N + ni + warpId * TILE_M * TILE_N] = val;
      }
    }
  }
  __syncthreads();
  for (SizeType32 ii = tid; ii < TILE_M * TILE_N; ii += BLOCK_SIZE) {
    SizeType32 mid = ii / TILE_N, nid = ii % TILE_N;
    float val = 0;
#pragma unroll
    for (SizeType32 jj = 0; jj < kWarpNum; ++jj) {
      val += shmem[jj * TILE_M * TILE_N + ii];
    }
    output[mid * n + nid] = static_cast<OutputType>(val * alpha);
  }
}

template <typename InputType, typename OutputType, SizeType32 TILE_M,
          SizeType32 TILE_N, SizeType32 BLOCK_SIZE>
void cudaCoreGemmKernel(Params const &params, cudaStream_t stream) {
  dim3 block(BLOCK_SIZE);
  dim3 grid(params.m / TILE_M, params.n / TILE_N);
  cudaCoreGemm<InputType, OutputType, TILE_M, TILE_N, BLOCK_SIZE>
      <<<grid, block, 0, stream>>>(
          reinterpret_cast<InputType const *>(params.act),
          reinterpret_cast<InputType const *>(params.weight), params.alpha,
          reinterpret_cast<OutputType *>(params.output), params.m, params.n,
          params.k);
}

template <typename InputType, typename OutputType, int TILE_M, int TILE_N,
          int BLOCK_SIZE>
bool cudaCoreGemmTemplateCaller(Params const &params, cudaStream_t stream) {
  constexpr int cudaCoreGemmTemplateMaxM = 16;
  if (params.m == TILE_M) {
    cudaCoreGemmKernel<InputType, OutputType, TILE_M, TILE_N, BLOCK_SIZE>(
        params, stream);
    return true;
  }
  if constexpr (TILE_M < cudaCoreGemmTemplateMaxM) {
    return cudaCoreGemmTemplateCaller<InputType, OutputType, TILE_M + 1, TILE_N,
                                      BLOCK_SIZE>(params, stream);
  }
  return false;
}

template <typename InputType, typename OutputType>
bool cudaCoreGemmLauncher(Params const &params, cudaStream_t stream) {
  return cudaCoreGemmTemplateCaller<InputType, OutputType, 1, 2, 128>(params,
                                                                      stream);
}

bool cudaCoreGemmDispatcher(Params const &params, cudaStream_t stream) {
  bool dispatched = true;
  dispatched =
      cudaCoreGemmLauncher<__nv_bfloat16, __nv_bfloat16>(params, stream);
  return dispatched;
}

void simple_assert(bool flag) {
  if (!flag) {
    throw std::runtime_error("assert failed");
  }
}

struct CudaBuffer {
  void *_data;
  int _size;

  CudaBuffer(int size_in_bytes) : _size(size_in_bytes) {
    cudaMalloc(&_data, _size);
  }

  template <typename T = void>
  T *data() {
    return reinterpret_cast<T *>(_data);
  }

  void copy_to(void *dst) {
    cudaMemcpy(dst, _data, _size, cudaMemcpyDeviceToHost);
  }

  void copy_from(void *src) {
    cudaMemcpy(_data, src, _size, cudaMemcpyHostToDevice);
  }

  ~CudaBuffer() { cudaFree(_data); }
};

template <typename T>
bool compare(void *_pa, void *_pb, int size) {
  auto pa = reinterpret_cast<T *>(_pa);
  auto pb = reinterpret_cast<T *>(_pb);
  float max_diff = 0.f, tot_diff = 0.f;
  float max_val = 0.f;
  int diff_cnt = 0;
  float threshold = 1e-7;
  for (int n = 0; n < size; ++n) {
    float va = static_cast<float>(pa[n]);
    float vb = static_cast<float>(pb[n]);
    max_val = std::max(max_val, vb);
    float diff = std::abs(va - vb);
    if (diff > threshold) {
      max_diff = std::max(max_diff, diff);
      tot_diff += diff;
      ++diff_cnt;
    }
  }
  float diff_thres = max_val * 2e-3;
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    // bfloat16 has fewer mantissa digits than float16(10 bits for fp16 but only
    // 7 bits for bf16), so the cumulative error will be larger.
    diff_thres *= 3.f;
  } else {
    diff_thres *= 1.5f;
  }
  printf("max diff %f (diff threshold %f), avg diff %f, diff cnt %d/%d\n",
         max_diff, diff_thres, tot_diff / std::max(1, diff_cnt), diff_cnt,
         size);
  return max_diff <= diff_thres;
}

template <typename T1, typename T2>
void random_fill(std::vector<T1> &vec, T2 minv, T2 maxv) {
  std::mt19937 gen(rand());
  std::uniform_real_distribution<float> dis(static_cast<float>(minv),
                                            static_cast<float>(maxv));
  for (auto &v : vec) {
    v = static_cast<T1>(dis(gen));
  }
}

template <typename T1, typename T2>
void constant_fill(std::vector<T1> &vec, T2 value) {
  for (auto &v : vec) {
    v = static_cast<T1>(value);
  }
}

template <typename T1>
void linear_fill(std::vector<T1> &vec, int length) {
  for (int i = 0; i < vec.size(); ++i) {
    vec[i] = static_cast<T1>((i % length) / 100.f);
  }
}

template <typename T>
void print_mat(std::vector<T> const &data, int row, int col, char const *name) {
  assert(data.size() == row * col);
  printf("---------------%s\n", name);
  for (int n = 0; n < data.size(); ++n) {
    float value = static_cast<float>(data[n]);
    printf("%f, ", value);
    if (n % col == col - 1)
      printf("\n");
  }
  printf("\n");
}

template <typename InputType, typename OutputType>
void run_cpu(void *weight, void *activation, float scale, Params const &params,
             void *output) {
  for (int idx_m = 0; idx_m < params.m; ++idx_m) {
    for (int idx_n = 0; idx_n < params.n; ++idx_n) {
      float acc = 0.f;
      for (int idx_k = 0; idx_k < params.k; ++idx_k) {
        InputType a =
            reinterpret_cast<InputType *>(activation)[params.k * idx_m + idx_k];
        InputType w =
            reinterpret_cast<InputType *>(weight)[params.k * idx_n + idx_k];
        acc += static_cast<float>(w) * static_cast<float>(a);
      }
      reinterpret_cast<OutputType *>(output)[idx_m * params.n + idx_n] =
          static_cast<OutputType>(acc * scale);
    }
  }
}

float run_cuda_kernel(Params &params, int warmup, int iter) {
  cudaStream_t s;
  cudaStreamCreate(&s);
  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);
  for (int i = 0; i < warmup; ++i) {
    cudaCoreGemmDispatcher(params, s);
  }
  cudaEventRecord(begin, s);
  for (int i = 0; i < iter; ++i) {
    cudaCoreGemmDispatcher(params, s);
  }
  cudaEventRecord(end, s);
  cudaEventSynchronize(end);
  float time;
  cudaEventElapsedTime(&time, begin, end);
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  cudaStreamDestroy(s);
  return time / iter;
}

template <typename InputType, typename OutputType>
bool benchmark_and_verify(int m, int n, int k, int warmup, int iter,
                          bool debug = false) {
  std::srand(20240123);
  simple_assert(m <= 4);
  printf("mnk (%d, %d, %d), output %s\n", m, n, k, typeid(OutputType).name());
  CudaBuffer d_act(m * k * sizeof(InputType));
  CudaBuffer d_weight(k * n * sizeof(InputType));
  CudaBuffer d_out(m * n * sizeof(OutputType));
  std::vector<InputType> h_act(m * k);
  std::vector<InputType> h_weight(k * n);
  std::vector<float> h_alpha(1);
  std::vector<OutputType> h_out_cuda(m * n), h_out_cublas(m * n),
      h_out_gt(m * n);

  random_fill(h_act, -1.f, 1.f);
  random_fill(h_weight, -1.f, 1.f);
  random_fill(h_alpha, -1.f, 1.f);

  if (debug) {
    print_mat(h_act, m, k, "h_act");
    print_mat(h_weight, k, n, "h_weight");
    print_mat(h_alpha, 1, 1, "h_alpha");
  }

  d_act.copy_from(h_act.data());
  d_weight.copy_from(h_weight.data());

  Params params{
      d_act.data(), d_weight.data(), h_alpha[0], d_out.data(), m, n, k};

  run_cpu<InputType, OutputType>(h_weight.data(), h_act.data(), h_alpha[0],
                                 params, h_out_gt.data());

  float time1;
  time1 = run_cuda_kernel(params, warmup, iter);
  d_out.copy_to(h_out_cuda.data());
  bool pass_cuda_kernel =
      compare<OutputType>(h_out_cuda.data(), h_out_gt.data(), m * n);
  printf("pass correctness =: %d\n", pass_cuda_kernel);

  if (debug) {
    print_mat<OutputType>(h_out_gt, m, n, "h_out_cpu");
    print_mat<OutputType>(h_out_cuda, m, n, "h_out_cuda");
  }

  printf("cuda kernel cost time %.6f\n", time1);
  return pass_cuda_kernel;
}

int main() {
  int warmup = 10, iter = 30;
  benchmark_and_verify<__nv_bfloat16, __nv_bfloat16>(1, 5120, 5120, warmup,
                                                     iter);
  benchmark_and_verify<__nv_bfloat16, __nv_bfloat16>(1, 3072, 3072, warmup,
                                                     iter);
  benchmark_and_verify<__nv_bfloat16, __nv_bfloat16>(1, 3072, 12288, warmup,
                                                     iter);
  benchmark_and_verify<__nv_bfloat16, __nv_bfloat16>(1, 12288, 3072, warmup,
                                                     iter);
  benchmark_and_verify<__nv_bfloat16, __nv_bfloat16>(1, 3072, 32768, warmup,
                                                     iter);
  benchmark_and_verify<__nv_bfloat16, __nv_bfloat16>(1, 32768, 3072, warmup,
                                                     iter);
  benchmark_and_verify<__nv_bfloat16, __nv_bfloat16>(1, 5120, 3072, warmup,
                                                     iter);
  benchmark_and_verify<__nv_bfloat16, __nv_bfloat16>(1, 3072, 5120, warmup,
                                                     iter);
  return 0;
}
