#include "helper.h"
#include <cutlass/gemm/device/gemm.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

/*
To run
(1) Clone the official NVIDIA cutlass repo from git
(2) But this script in an example folder eg examples/00_basic_gemm
(3) Fix the naming and Make + run
*/

std::ofstream handle;

void recordResult(double runtime, std::vector<int> config) {
  std::string out = std::to_string(runtime);
  for (auto x : config) {
    out += "," + std::to_string(x);
  }
  out += "\n";
  handle << out;
}

template <typename TShape, typename WShape>
void CutlassSgemmNN(int M, int N, int K, float alpha, float const *A, int lda,
                    float const *B, int ldb, float beta, float *C, int ldc) {

  using ColumnMajor = cutlass::layout::ColumnMajor;
  using RowMajor = cutlass::layout::RowMajor;
  using ThreadBlockShape = TShape;
  using WarpBlockShape = WShape;
  using CutlassGemm = cutlass::gemm::device::Gemm<
      float, // Matrix A data type
      RowMajor,
      float, // Matrix B data type
      RowMajor,
      float, // Matrix C data type
      RowMajor,
      float,                      // Element Accumulator
      cutlass::arch::OpClassSimt, // To indicate whether cuda or tensor cores
      cutlass::arch::Sm80,        // arch to tune for
      ThreadBlockShape,           // threadblock level tile shape
      WarpBlockShape              // warp level tile shape
      >;

  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;
  typename CutlassGemm::Arguments args({M, N, K}, {A, lda}, {B, ldb}, {C, ldc},
                                       {C, ldc}, {alpha, beta});

  int iter = 100;
  cutlass::Status status;
  GpuTimer timer;
  timer.start();
  for (int i = 0; i < iter; i++) {
    status = gemm_operator(args);
  }
  timer.stop();
  float timeX = ((timer.elapsed_millis() / (float)iter) / 1000);

  printf("status => %s [%d %d %d %d]\n", cutlassGetStatusString(status),
         ThreadBlockShape::kM, ThreadBlockShape::kN, WarpBlockShape::kM,
         WarpBlockShape::kN);
  if (status != cutlass::Status::kSuccess) {
    return;
  }
  recordResult(timeX, {K, ThreadBlockShape::kM, ThreadBlockShape::kN,
                       WarpBlockShape::kM, WarpBlockShape::kN});
}

template <typename TShape, typename WShape>
void TestCutlassGemmInter(int M, int N, int K, float alpha, float beta) {

  int lda = K;
  int ldb = N;
  int ldc = N;

  float *h_a, *h_b, *h_c;
  h_a = (float *)malloc(M * K * sizeof(float));
  h_b = (float *)malloc(K * N * sizeof(float));
  h_c = (float *)malloc(M * N * sizeof(float));

  for (int i = 0; i < (M * K); i++) {
    h_a[i] = (float)i;
  }
  for (int i = 0; i < (K * N); i++) {
    h_b[i] = (float)(i + 1);
  }
  for (int i = 0; i < (M * N); i++) {
    h_c[i] = (float)0;
  }

  float *A, *B, *C_cutlass;
  cudaMalloc((void **)&A, M * K * sizeof(float));
  cudaMalloc((void **)&B, K * N * sizeof(float));
  cudaMalloc((void **)&C_cutlass, M * N * sizeof(float));

  cudaMemcpy(A, h_a, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B, h_b, K * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(C_cutlass, h_c, M * N * sizeof(float), cudaMemcpyHostToDevice);
  CutlassSgemmNN<TShape, WShape>(M, N, K, alpha, A, lda, B, ldb, beta,
                                 C_cutlass, ldc);

  cudaFree(C_cutlass);
  cudaFree(B);
  cudaFree(A);
}

template <typename TShape>
void TestCutlassGemm(int M, int N, int K, float alpha, float beta) {
  TestCutlassGemmInter<TShape, cutlass::gemm::GemmShape<32, 32, 8>>(
      M, N, K, alpha, beta);
  TestCutlassGemmInter<TShape, cutlass::gemm::GemmShape<32, 64, 8>>(
      M, N, K, alpha, beta);
  TestCutlassGemmInter<TShape, cutlass::gemm::GemmShape<64, 64, 8>>(
      M, N, K, alpha, beta);
}

void benchmark() {
  float scalars[2] = {1, 0};

  TestCutlassGemm<cutlass::gemm::GemmShape<64, 64, 8>>(1024, 1024, 1024,
                                                       scalars[0], scalars[1]);
  TestCutlassGemm<cutlass::gemm::GemmShape<128, 64, 8>>(1024, 1024, 1024,
                                                        scalars[0], scalars[1]);
  TestCutlassGemm<cutlass::gemm::GemmShape<128, 128, 8>>(
      1024, 1024, 1024, scalars[0], scalars[1]);
  TestCutlassGemm<cutlass::gemm::GemmShape<256, 64, 8>>(1024, 1024, 1024,
                                                        scalars[0], scalars[1]);
  TestCutlassGemm<cutlass::gemm::GemmShape<256, 128, 8>>(
      1024, 1024, 1024, scalars[0], scalars[1]);

  TestCutlassGemm<cutlass::gemm::GemmShape<64, 64, 8>>(2048, 2048, 2048,
                                                       scalars[0], scalars[1]);
  TestCutlassGemm<cutlass::gemm::GemmShape<128, 64, 8>>(2048, 2048, 2048,
                                                        scalars[0], scalars[1]);
  TestCutlassGemm<cutlass::gemm::GemmShape<128, 128, 8>>(
      2048, 2048, 2048, scalars[0], scalars[1]);
  TestCutlassGemm<cutlass::gemm::GemmShape<256, 64, 8>>(2048, 2048, 2048,
                                                        scalars[0], scalars[1]);
  TestCutlassGemm<cutlass::gemm::GemmShape<256, 128, 8>>(
      2048, 2048, 2048, scalars[0], scalars[1]);

  TestCutlassGemm<cutlass::gemm::GemmShape<64, 64, 8>>(4096, 4096, 4096,
                                                       scalars[0], scalars[1]);
  TestCutlassGemm<cutlass::gemm::GemmShape<128, 64, 8>>(4096, 4096, 4096,
                                                        scalars[0], scalars[1]);
  TestCutlassGemm<cutlass::gemm::GemmShape<128, 128, 8>>(
      4096, 4096, 4096, scalars[0], scalars[1]);
  TestCutlassGemm<cutlass::gemm::GemmShape<256, 64, 8>>(4096, 4096, 4096,
                                                        scalars[0], scalars[1]);
  TestCutlassGemm<cutlass::gemm::GemmShape<256, 128, 8>>(
      4096, 4096, 4096, scalars[0], scalars[1]);

  TestCutlassGemm<cutlass::gemm::GemmShape<64, 64, 8>>(8192, 8192, 8192,
                                                       scalars[0], scalars[1]);
  TestCutlassGemm<cutlass::gemm::GemmShape<128, 64, 8>>(8192, 8192, 8192,
                                                        scalars[0], scalars[1]);
  TestCutlassGemm<cutlass::gemm::GemmShape<128, 128, 8>>(
      8192, 8192, 8192, scalars[0], scalars[1]);
  TestCutlassGemm<cutlass::gemm::GemmShape<256, 64, 8>>(8192, 8192, 8192,
                                                        scalars[0], scalars[1]);
  TestCutlassGemm<cutlass::gemm::GemmShape<256, 128, 8>>(
      8192, 8192, 8192, scalars[0], scalars[1]);

  TestCutlassGemm<cutlass::gemm::GemmShape<64, 64, 8>>(16384, 16384, 16384,
                                                       scalars[0], scalars[1]);
  TestCutlassGemm<cutlass::gemm::GemmShape<128, 64, 8>>(16384, 16384, 16384,
                                                        scalars[0], scalars[1]);
  TestCutlassGemm<cutlass::gemm::GemmShape<128, 128, 8>>(
      16384, 16384, 16384, scalars[0], scalars[1]);
  TestCutlassGemm<cutlass::gemm::GemmShape<256, 64, 8>>(16384, 16384, 16384,
                                                        scalars[0], scalars[1]);
  TestCutlassGemm<cutlass::gemm::GemmShape<256, 128, 8>>(
      16384, 16384, 16384, scalars[0], scalars[1]);
}

int main(int argc, const char *arg[]) {
  handle.open("cutlass_results.csv");
  benchmark();
  handle.close();
}
