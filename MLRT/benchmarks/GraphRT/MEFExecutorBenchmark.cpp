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
// This file benchmarks MEF executor performance characteristics including
// primitive dispatch, control flow, invokable calls, and dataflow scheduling.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/HostSystem.h"
#include "MLRT/GraphRT/GRTTestcase.h"
#include "MLRT/GraphRT/MEFExecutor/MEFFile.h"
#include "MLRT/GraphRT/MEFExecutor/MEFInvokable.h"
#include "MLRT/GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "MLRT/GraphRT/Primitives.h"
#include "Support/ML/SizeUtils.h"

#include "benchmark/benchmark.h"
#include "tools/cpp/runfiles/runfiles.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

#include <memory>
#include <string>

using namespace M;
using namespace M::AsyncRT;
using bazel::tools::cpp::runfiles::Runfiles;

namespace {

//===----------------------------------------------------------------------===//
// Benchmark Infrastructure
//===----------------------------------------------------------------------===//

/// Global runfiles instance for loading MEF files.
static std::unique_ptr<Runfiles> gRunfiles;

/// Get or create the global runfiles instance.
static Runfiles *getRunfiles() {
  if (!gRunfiles) {
    std::string error;
    gRunfiles = std::unique_ptr<Runfiles>(Runfiles::CreateForTest(&error));
    if (!gRunfiles) {
      llvm::errs() << "Failed to create runfiles: " << error << "\n";
      std::abort();
    }
  }
  return gRunfiles.get();
}

/// Global primitive registry shared by all benchmarks.
static PrimitiveRegistry *getPrimitiveRegistry() {
  static PrimitiveRegistry *registry = []() {
    auto *reg = new PrimitiveRegistry();
    registerAllGraphRTPrimitives(*reg);
    reg->finalize();
    return reg;
  }();
  return registry;
}

/// Holds a loaded MEF file and its backing buffer.
struct LoadedMEF {
  std::unique_ptr<llvm::MemoryBuffer> buffer;
  RCRef<MEFFile> file;
};

/// Load a MEF file from runfiles with proper alignment.
static LoadedMEF loadMEFFile(llvm::StringRef name) {
  std::string path = getRunfiles()->Rlocation("_main/MLRT/benchmarks/GraphRT/" +
                                              name.str() + ".mef");

  // Read the file into memory
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrErr) {
    llvm::errs() << "Failed to open MEF file: " << path << "\n";
    std::abort();
  }

  // Create an aligned copy of the buffer for MEF parsing
  auto alignedBuffer = createAndCopyToAlignedBuffer(
      (*bufferOrErr)->getBufferStart(), (*bufferOrErr)->getBufferEnd());

  auto mefFileOrErr =
      MEFFile::open(alignedBuffer->getBuffer(), *getPrimitiveRegistry());
  if (failed(mefFileOrErr)) {
    llvm::errs() << "Failed to parse MEF file: " << path << ": "
                 << mefFileOrErr.getError() << "\n";
    std::abort();
  }

  return LoadedMEF{std::move(alignedBuffer), mefFileOrErr.takeValue()};
}

/// Find a testcase by name in a MEF file.
static GRTTestcase findTestcase(MEFFile *mefFile, llvm::StringRef name) {
  llvm::SmallVector<GRTTestcase, 16> testcases;
  auto *moduleInv = mefFile->getTopLevelInvokable();

  auto err = moduleInv->walkPrimitives<GRTTestcase>([&](GRTTestcase test) {
    if (test.getName() == name)
      testcases.push_back(test);
  });

  if (failed(err) || testcases.empty()) {
    llvm::errs() << "Failed to find testcase: " << name << "\n";
    std::abort();
  }

  return testcases[0];
}

/// Create a cpuDevice with 32 threads for async primitive execution.
static CPUDeviceRef createBenchmarkCPUDevice() {
  CPUDeviceOptions opts;
  opts.numThreads = 32;
  opts.withMainWillNotDonate();
  return getOrCreateCPUDevice(CPUDeviceSource::Test, opts);
}

//===----------------------------------------------------------------------===//
// Primitive Dispatch Benchmarks
//===----------------------------------------------------------------------===//

/// Baseline benchmark: schedules one async primitive but does NOT wait for
/// completion. This measures pure scheduling/dispatch overhead without the
/// cost of synchronization. Compare with BM_PrimitiveDispatch_1 to see the
/// additional cost of waiting for the result.
static void BM_PrimitiveDispatch_Baseline(benchmark::State &state) {
  auto loadedMef = loadMEFFile("bench_primitives");
  auto testcase =
      findTestcase(loadedMef.file.getPointer(), "bench_primitive_baseline");
  auto cpuDevice = createBenchmarkCPUDevice();

  for (auto _ : state) {
    // Intentionally not awaiting - testcase has no outputs so this just
    // measures the cost of scheduling the async work.
    testcase.getBody()->invoke({}, *cpuDevice);
  }
}
BENCHMARK(BM_PrimitiveDispatch_Baseline)
    ->UseRealTime()
    ->Iterations(10000)
    ->DisplayAggregatesOnly(true);

static void BM_PrimitiveDispatch_1(benchmark::State &state) {
  auto loadedMef = loadMEFFile("bench_primitives");
  auto testcase =
      findTestcase(loadedMef.file.getPointer(), "bench_primitive_1");
  auto cpuDevice = createBenchmarkCPUDevice();

  for (auto _ : state) {
    auto results = testcase.getBody()->invoke({}, *cpuDevice);
    await(results);
  }
}
BENCHMARK(BM_PrimitiveDispatch_1)
    ->UseRealTime()
    ->Iterations(10000)
    ->DisplayAggregatesOnly(true);

static void BM_PrimitiveDispatch_10(benchmark::State &state) {
  auto loadedMef = loadMEFFile("bench_primitives");
  auto testcase =
      findTestcase(loadedMef.file.getPointer(), "bench_primitive_10");
  auto cpuDevice = createBenchmarkCPUDevice();

  for (auto _ : state) {
    auto results = testcase.getBody()->invoke({}, *cpuDevice);
    await(results);
  }
}
BENCHMARK(BM_PrimitiveDispatch_10)
    ->UseRealTime()
    ->Iterations(10000)
    ->DisplayAggregatesOnly(true);

//===----------------------------------------------------------------------===//
// Control Flow Benchmarks - If/Else
//===----------------------------------------------------------------------===//

static void BM_ControlFlowIf_True(benchmark::State &state) {
  auto loadedMef = loadMEFFile("bench_control_flow");
  auto testcase = findTestcase(loadedMef.file.getPointer(), "bench_if_true");
  auto cpuDevice = createBenchmarkCPUDevice();

  for (auto _ : state) {
    auto results = testcase.getBody()->invoke({}, *cpuDevice);
    await(results);
  }
}
BENCHMARK(BM_ControlFlowIf_True)
    ->UseRealTime()
    ->Iterations(10000)
    ->DisplayAggregatesOnly(true);

static void BM_ControlFlowIf_False(benchmark::State &state) {
  auto loadedMef = loadMEFFile("bench_control_flow");
  auto testcase = findTestcase(loadedMef.file.getPointer(), "bench_if_false");
  auto cpuDevice = createBenchmarkCPUDevice();

  for (auto _ : state) {
    auto results = testcase.getBody()->invoke({}, *cpuDevice);
    await(results);
  }
}
BENCHMARK(BM_ControlFlowIf_False)
    ->UseRealTime()
    ->Iterations(10000)
    ->DisplayAggregatesOnly(true);

//===----------------------------------------------------------------------===//
// Control Flow Benchmarks - While Loop
//===----------------------------------------------------------------------===//

static void BM_ControlFlowWhile_1(benchmark::State &state) {
  auto loadedMef = loadMEFFile("bench_control_flow");
  auto testcase = findTestcase(loadedMef.file.getPointer(), "bench_while_1");
  auto cpuDevice = createBenchmarkCPUDevice();

  for (auto _ : state) {
    auto results = testcase.getBody()->invoke({}, *cpuDevice);
    await(results);
  }
}
BENCHMARK(BM_ControlFlowWhile_1)
    ->UseRealTime()
    ->Iterations(10000)
    ->DisplayAggregatesOnly(true);

static void BM_ControlFlowWhile_10(benchmark::State &state) {
  auto loadedMef = loadMEFFile("bench_control_flow");
  auto testcase = findTestcase(loadedMef.file.getPointer(), "bench_while_10");
  auto cpuDevice = createBenchmarkCPUDevice();

  for (auto _ : state) {
    auto results = testcase.getBody()->invoke({}, *cpuDevice);
    await(results);
  }
}
BENCHMARK(BM_ControlFlowWhile_10)
    ->UseRealTime()
    ->Iterations(10000)
    ->DisplayAggregatesOnly(true);

//===----------------------------------------------------------------------===//
// Control Flow Benchmarks - Repeat Loop
//===----------------------------------------------------------------------===//

static void BM_ControlFlowRepeat_1(benchmark::State &state) {
  auto loadedMef = loadMEFFile("bench_control_flow");
  auto testcase = findTestcase(loadedMef.file.getPointer(), "bench_repeat_1");
  auto cpuDevice = createBenchmarkCPUDevice();

  for (auto _ : state) {
    auto results = testcase.getBody()->invoke({}, *cpuDevice);
    await(results);
  }
}
BENCHMARK(BM_ControlFlowRepeat_1)
    ->UseRealTime()
    ->Iterations(10000)
    ->DisplayAggregatesOnly(true);

static void BM_ControlFlowRepeat_10(benchmark::State &state) {
  auto loadedMef = loadMEFFile("bench_control_flow");
  auto testcase = findTestcase(loadedMef.file.getPointer(), "bench_repeat_10");
  auto cpuDevice = createBenchmarkCPUDevice();

  for (auto _ : state) {
    auto results = testcase.getBody()->invoke({}, *cpuDevice);
    await(results);
  }
}
BENCHMARK(BM_ControlFlowRepeat_10)
    ->UseRealTime()
    ->Iterations(10000)
    ->DisplayAggregatesOnly(true);

//===----------------------------------------------------------------------===//
// Invokable Call Benchmarks
//===----------------------------------------------------------------------===//

static void BM_InvokableCall_Depth1(benchmark::State &state) {
  auto loadedMef = loadMEFFile("bench_invokable");
  auto testcase =
      findTestcase(loadedMef.file.getPointer(), "bench_call_depth_1");
  auto cpuDevice = createBenchmarkCPUDevice();

  for (auto _ : state) {
    auto results = testcase.getBody()->invoke({}, *cpuDevice);
    await(results);
  }
}
BENCHMARK(BM_InvokableCall_Depth1)
    ->UseRealTime()
    ->Iterations(10000)
    ->DisplayAggregatesOnly(true);

static void BM_InvokableCall_Depth10(benchmark::State &state) {
  auto loadedMef = loadMEFFile("bench_invokable");
  auto testcase =
      findTestcase(loadedMef.file.getPointer(), "bench_call_depth_10");
  auto cpuDevice = createBenchmarkCPUDevice();

  for (auto _ : state) {
    auto results = testcase.getBody()->invoke({}, *cpuDevice);
    await(results);
  }
}
BENCHMARK(BM_InvokableCall_Depth10)
    ->UseRealTime()
    ->Iterations(10000)
    ->DisplayAggregatesOnly(true);

static void BM_InvokableCall_ChainCall5(benchmark::State &state) {
  auto loadedMef = loadMEFFile("bench_invokable");
  auto testcase =
      findTestcase(loadedMef.file.getPointer(), "bench_chain_call_5");
  auto cpuDevice = createBenchmarkCPUDevice();

  for (auto _ : state) {
    auto results = testcase.getBody()->invoke({}, *cpuDevice);
    await(results);
  }
}
BENCHMARK(BM_InvokableCall_ChainCall5)
    ->UseRealTime()
    ->Iterations(10000)
    ->DisplayAggregatesOnly(true);

//===----------------------------------------------------------------------===//
// Dataflow Scheduling Benchmarks
//===----------------------------------------------------------------------===//

static void BM_DataflowParallel_2(benchmark::State &state) {
  auto loadedMef = loadMEFFile("bench_dataflow");
  auto testcase = findTestcase(loadedMef.file.getPointer(), "bench_parallel_2");
  auto cpuDevice = createBenchmarkCPUDevice();

  for (auto _ : state) {
    auto results = testcase.getBody()->invoke({}, *cpuDevice);
    await(results);
  }
}
BENCHMARK(BM_DataflowParallel_2)
    ->UseRealTime()
    ->Iterations(10000)
    ->DisplayAggregatesOnly(true);

static void BM_DataflowParallel_8(benchmark::State &state) {
  auto loadedMef = loadMEFFile("bench_dataflow");
  auto testcase = findTestcase(loadedMef.file.getPointer(), "bench_parallel_8");
  auto cpuDevice = createBenchmarkCPUDevice();

  for (auto _ : state) {
    auto results = testcase.getBody()->invoke({}, *cpuDevice);
    await(results);
  }
}
BENCHMARK(BM_DataflowParallel_8)
    ->UseRealTime()
    ->Iterations(10000)
    ->DisplayAggregatesOnly(true);

static void BM_DataflowSequential_8(benchmark::State &state) {
  auto loadedMef = loadMEFFile("bench_dataflow");
  auto testcase =
      findTestcase(loadedMef.file.getPointer(), "bench_sequential_8");
  auto cpuDevice = createBenchmarkCPUDevice();

  for (auto _ : state) {
    auto results = testcase.getBody()->invoke({}, *cpuDevice);
    await(results);
  }
}
BENCHMARK(BM_DataflowSequential_8)
    ->UseRealTime()
    ->Iterations(10000)
    ->DisplayAggregatesOnly(true);

static void BM_DataflowParallelInvoke_8(benchmark::State &state) {
  auto loadedMef = loadMEFFile("bench_dataflow");
  auto testcase =
      findTestcase(loadedMef.file.getPointer(), "bench_parallel_invoke_8");
  auto cpuDevice = createBenchmarkCPUDevice();

  for (auto _ : state) {
    auto results = testcase.getBody()->invoke({}, *cpuDevice);
    await(results);
  }
}
BENCHMARK(BM_DataflowParallelInvoke_8)
    ->UseRealTime()
    ->Iterations(10000)
    ->DisplayAggregatesOnly(true);

//===----------------------------------------------------------------------===//
// GPU Simultaneous Ready Benchmarks
// Measures serialization overhead when multiple primitives become ready at
// the same time due to sharing inputs (simulates multi-device allreduce).
// Uses test.sleep to simulate ~1us GPU kernel launch latency.
//===----------------------------------------------------------------------===//

static void BM_GPUSimultaneousReady_2(benchmark::State &state) {
  auto loadedMef = loadMEFFile("bench_dataflow");
  auto testcase = findTestcase(loadedMef.file.getPointer(),
                               "bench_gpu_simultaneous_ready_2");
  auto cpuDevice = createBenchmarkCPUDevice();

  for (auto _ : state) {
    auto results = testcase.getBody()->invoke({}, *cpuDevice);
    await(results);
  }
}
BENCHMARK(BM_GPUSimultaneousReady_2)
    ->UseRealTime()
    ->Iterations(10000)
    ->DisplayAggregatesOnly(true);

static void BM_GPUSimultaneousReady_8(benchmark::State &state) {
  auto loadedMef = loadMEFFile("bench_dataflow");
  auto testcase = findTestcase(loadedMef.file.getPointer(),
                               "bench_gpu_simultaneous_ready_8");
  auto cpuDevice = createBenchmarkCPUDevice();

  for (auto _ : state) {
    auto results = testcase.getBody()->invoke({}, *cpuDevice);
    await(results);
  }
}
BENCHMARK(BM_GPUSimultaneousReady_8)
    ->UseRealTime()
    ->Iterations(10000)
    ->DisplayAggregatesOnly(true);

} // namespace
