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
// This file benchmarks the time taken to enqueue (not execute) GPU kernels
// using the AsyncRT Device and DeviceContext API.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"

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
  llvm::errs() << "GPU Kernel Enqueue Benchmark (DeviceContext API)\n";
  llvm::errs() << "================================================\n";

  // Check if CUDA devices are available
  int numDevices =
      M::Driver::Device::numberOfDevices(M::Driver::Device::cudaAPI);
  if (numDevices == 0) {
    llvm::errs() << "No CUDA devices available\n";
    return 1;
  }

  // Create DeviceContext for device 0
  M::ErrorOr<M::Driver::DeviceContextRef> ctxOrErr =
      M::Driver::DeviceContext::create(M::Driver::Device::cudaAPI, 0);
  if (M::failed(ctxOrErr)) {
    llvm::errs() << "Failed to create device context: " << ctxOrErr.getError()
                 << "\n";
    return 1;
  }
  M::Driver::DeviceContextRef ctx = ctxOrErr.takeValue();

  // Print device information
  llvm::errs() << "Device: " << ctx->device().name() << "\n";

  // Load the PTX kernel
  M::ErrorOr<M::Driver::DeviceFunctionRef> funcOrErr =
      ctx->loadFunction("small_kernel_module", "small_kernel", kernelPTX);
  if (M::failed(funcOrErr)) {
    llvm::errs() << "Failed to load kernel function: " << funcOrErr.getError()
                 << "\n";
    return 1;
  }
  M::Driver::DeviceFunctionRef function = funcOrErr.takeValue();

  // Allocate device buffer
  constexpr size_t bufferSize = 64;
  M::ErrorOr<M::Driver::DeviceBufferRef> bufferOrErr =
      ctx->enqueueCreateBuffer(bufferSize);
  if (M::failed(bufferOrErr)) {
    llvm::errs() << "Failed to enqueue device buffer allocation: "
                 << bufferOrErr.getError() << "\n";
    return 1;
  }
  M::Driver::DeviceBufferRef buffer = bufferOrErr.takeValue();
  M::ErrorOrSuccess syncOrErr = buffer->context().synchronize();
  if (M::failed(syncOrErr)) {
    llvm::errs() << "Failed to sync after enqueueing device buffer allocation: "
                 << syncOrErr.getError() << "\n";
    return 1;
  }

  // Setup launch configuration
  constexpr uint32_t gridX = 1;
  constexpr uint32_t gridY = 1;
  constexpr uint32_t gridZ = 1;
  constexpr uint32_t blockX = 1;
  constexpr uint32_t blockY = 1;
  constexpr uint32_t blockZ = 1;
  constexpr uint32_t sharedMemBytes = 0;

  // Prepare arguments for direct kernel launch
  void *bufferPtr = buffer->ptr();
  void *args[] = {&bufferPtr};

  // Create the benchmark
  M::MicroBenchmark bench(
      "GPU Kernel Enqueue", [&](M::MicroBenchmark::State &state) {
        for (auto _ : state) {
          // Launch multiple kernels per iteration to get meaningful timings
          for (size_t i = 0; i < NUM_KERNELS_PER_ITERATION; ++i) {
            M::ErrorOrSuccess result = ctx->enqueueFunctionExecDirect(
                *function, gridX, gridY, gridZ, blockX, blockY, blockZ,
                sharedMemBytes, nullptr, 0, args, 0, nullptr);
            if (M::failed(result)) {
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
    M::ErrorOrSuccess result = ctx->synchronize();
    if (M::failed(result)) {
      state.reportError(M::Error("Stream synchronization failed."));
    }
  };

  // Run the benchmark
  M::ErrorOrSuccess runResult = bench.run(runOpts);
  if (M::failed(runResult)) {
    llvm::errs() << "Benchmark failed: " << runResult.takeError() << "\n";
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

  return 0;
}

} // namespace

int main(int argc, char **argv) { return minimalKernelEnqueue(); }
