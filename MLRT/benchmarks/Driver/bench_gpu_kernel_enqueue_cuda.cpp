//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This file benchmarks the time taken to enqueue (not execute) CUDA kernels
// using the CUDA driver API (cuLaunchKernelEx).
//
//===----------------------------------------------------------------------===//

#include "cuda_common.h"

#include "Support/MicroBenchmark.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>

namespace {

// A minimal PTX kernel with a single buffer argument as we're only measuring
// enqueue time, not execution time
static const char *kernelPTX = R"PTX(
.version 8.0
.target sm_90
.address_size 64

.visible .entry small_kernel(
  .param .b64 param_0
)
{
  .reg .b64 %rd<2>;
  ld.param.u64 %rd1, [param_0];
  ret;
}
)PTX";

// Constants
constexpr size_t NUM_KERNELS_PER_ITERATION = 128;
constexpr size_t NUM_WARMUP_ITERATIONS = 3;
constexpr size_t NUM_ITERATIONS = 1000;

// Benchmark the time taken to enqueue a minimal kernel
int minimalKernelEnqueue() {
  llvm::errs() << "CUDA Kernel Enqueue Benchmark\n";
  llvm::errs() << "==============================\n";

  // Load CUDA driver functions
  if (!CudaCommon::loadCUDAFunctions()) {
    llvm::errs() << "Failed to load CUDA driver functions\n";
    return 1;
  }

  // Initialize CUDA driver
  if (CudaCommon::cuInit(0) != CudaCommon::CUDA_SUCCESS) {
    llvm::errs() << "Failed to initialize CUDA driver\n";
    return 1;
  }

  // Get device 0
  CudaCommon::CUdevice device;
  if (CudaCommon::cuDeviceGet(&device, 0) != CudaCommon::CUDA_SUCCESS) {
    llvm::errs() << "Failed to get CUDA device 0\n";
    return 1;
  }

  // Print device information
  char deviceName[256];
  if (CudaCommon::cuDeviceGetName(deviceName, sizeof(deviceName), device) ==
      CudaCommon::CUDA_SUCCESS) {
    llvm::errs() << "Device: " << deviceName << "\n";
  }

  // Create or get primary context
  CudaCommon::CUcontext context;
  if (CudaCommon::cuDevicePrimaryCtxRetain(&context, device) !=
      CudaCommon::CUDA_SUCCESS) {
    llvm::errs() << "Failed to get primary context\n";
    return 1;
  }

  // Set context current
  if (CudaCommon::cuCtxSetCurrent(context) != CudaCommon::CUDA_SUCCESS) {
    CudaCommon::cuDevicePrimaryCtxRelease(device);
    llvm::errs() << "Failed to set current context\n";
    return 1;
  }

  // Load PTX module
  CudaCommon::CUmodule module;
  if (CudaCommon::cuModuleLoadDataEx(&module, kernelPTX, 0, nullptr, nullptr) !=
      CudaCommon::CUDA_SUCCESS) {
    CudaCommon::cuDevicePrimaryCtxRelease(device);
    llvm::errs() << "Failed to load PTX module\n";
    return 1;
  }

  // Get kernel function
  CudaCommon::CUfunction function;
  if (CudaCommon::cuModuleGetFunction(&function, module, "small_kernel") !=
      CudaCommon::CUDA_SUCCESS) {
    CudaCommon::cuModuleUnload(module);
    CudaCommon::cuDevicePrimaryCtxRelease(device);
    llvm::errs() << "Failed to get kernel function\n";
    return 1;
  }

  // Create stream
  CudaCommon::CUstream stream;
  if (CudaCommon::cuStreamCreate(&stream, 0) != CudaCommon::CUDA_SUCCESS) {
    CudaCommon::cuModuleUnload(module);
    CudaCommon::cuDevicePrimaryCtxRelease(device);
    llvm::errs() << "Failed to create stream\n";
    return 1;
  }

  // Allocate device buffer
  constexpr size_t bufferSize = 64;
  CudaCommon::CUdeviceptr buffer;
  CudaCommon::CUresult allocResult =
      CudaCommon::cuMemAlloc(&buffer, bufferSize);
  if (allocResult != CudaCommon::CUDA_SUCCESS) {
    const char *errStr = nullptr;
    CudaCommon::cuGetErrorString(allocResult, &errStr);
    llvm::errs() << "Failed to allocate device buffer: "
                 << (errStr ? errStr : "unknown") << " (code: " << allocResult
                 << ")\n";
    CudaCommon::cuStreamDestroy(stream);
    CudaCommon::cuModuleUnload(module);
    CudaCommon::cuDevicePrimaryCtxRelease(device);
    return 1;
  }

  // Setup launch configuration
  CudaCommon::CUlaunchConfig config;
  config.gridDimX = 1;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = 1;
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = 0;
  config.hStream = stream;
  config.attrs = nullptr;
  config.numAttrs = 0;

  // Setup kernel parameters
  void *kernelParams[] = {&buffer};

  // Create the benchmark
  M::MicroBenchmark bench(
      "GPU Kernel Enqueue", [&](M::MicroBenchmark::State &state) {
        for (auto _ : state) {
          // Launch multiple kernels per iteration to get meaningful timings
          for (size_t i = 0; i < NUM_KERNELS_PER_ITERATION; ++i) {
            CudaCommon::CUresult result = CudaCommon::cuLaunchKernelEx(
                &config, function, kernelParams, nullptr);
            if (result != CudaCommon::CUDA_SUCCESS) {
              state.reportError(M::Error("Kernel launch failed."));
              return;
            }
          }
        }
      });

  // Configure benchmark options
  M::MicroBenchmark::RunOptions runOpts;
  runOpts.minRuntime = std::chrono::seconds(1);
  runOpts.maxRuntime = std::chrono::seconds(600);
  runOpts.maxBenchmarkIterations = NUM_ITERATIONS;
  runOpts.maxBatchSize = 100;
  runOpts.warmupIterations = NUM_WARMUP_ITERATIONS;

  // Add epilogue to synchronize the stream after each batch
  runOpts.epilogueFunction = [&](M::MicroBenchmark::State &state) {
    CudaCommon::CUresult result = CudaCommon::cuStreamSynchronize(stream);
    if (result != CudaCommon::CUDA_SUCCESS) {
      state.reportError(M::Error("Stream synchronization failed."));
    }
  };

  // Run the benchmark
  M::ErrorOrSuccess runResult = bench.run(runOpts);
  if (M::failed(runResult)) {
    llvm::errs() << "Benchmark failed: " << runResult.takeError() << "\n";
    CudaCommon::cuMemFree(buffer);
    CudaCommon::cuStreamDestroy(stream);
    CudaCommon::cuModuleUnload(module);
    CudaCommon::cuDevicePrimaryCtxRelease(device);
    return 1;
  }

  // Report the results
  bench.report(llvm::outs(), M::MicroBenchmark::ReportOptions());

  // Retrieve mean and median for total and per kernel
  double meanTotalMs =
      bench.measurement(M::MicroBenchmark::ReportMetric::kMeanLatency,
                        M::MicroBenchmark::TimeUnit::kMilliseconds);
  double medianTotalMs =
      bench.measurement(M::MicroBenchmark::ReportMetric::kMedianLatency,
                        M::MicroBenchmark::TimeUnit::kMilliseconds);

  double meanPerKernelUs =
      bench.measurement(M::MicroBenchmark::ReportMetric::kMeanLatency,
                        M::MicroBenchmark::TimeUnit::kMicroseconds) /
      NUM_KERNELS_PER_ITERATION;
  double medianPerKernelUs =
      bench.measurement(M::MicroBenchmark::ReportMetric::kMedianLatency,
                        M::MicroBenchmark::TimeUnit::kMicroseconds) /
      NUM_KERNELS_PER_ITERATION;

  llvm::errs() << "\nResults:\n";
  llvm::errs() << "  Mean (total):        " << meanTotalMs << " ms\n";
  llvm::errs() << "  Median (total):      " << medianTotalMs << " ms\n";
  llvm::errs() << "  Mean (per kernel):   " << meanPerKernelUs << " μs\n";
  llvm::errs() << "  Median (per kernel): " << medianPerKernelUs << " μs\n";

  // Cleanup
  CudaCommon::cuMemFree(buffer);
  CudaCommon::cuStreamDestroy(stream);
  CudaCommon::cuModuleUnload(module);
  CudaCommon::cuDevicePrimaryCtxRelease(device);

  return 0;
}

} // namespace

int main(int argc, char **argv) { return minimalKernelEnqueue(); }
