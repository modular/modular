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

#include "../lib/Metal/Common.h"
#include "Fixture.h"
#include "HAL/M_driver_bundle.h"
#include "HAL/M_driver_common.h"
#include <chrono>
#include <map>
#include <string>
#include <thread>

#include "llvm/ADT/ScopeExit.h"

// Extend test-cases per-plugin beyond the common ones in PluginTest.cpp

/// NOTE: assumes `s` is null terminated
M_driver_slice M_driver_slice_from(const char *s) {
  uint64_t size = strlen(s);
  return M_driver_slice{s, size};
}

M_driver_slice M_driver_slice_from(std::string &s) {
  return M_driver_slice{s.data(), s.size()};
}

static std::string vecAddMsl = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct FloatBufPtr      { device float*       ptr; };
struct ConstFloatBufPtr { device const float* ptr; };

kernel void vec_add_msl(
    device FloatBufPtr&       out_ref [[buffer(0)]],
    device ConstFloatBufPtr&  in0_ref [[buffer(1)]],
    device ConstFloatBufPtr&  in1_ref [[buffer(2)]],
    constant uint&            length  [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    device float*       output = out_ref.ptr;
    device const float* input0 = in0_ref.ptr;
    device const float* input1 = in1_ref.ptr;
    if (tid < length)
        output[tid] = input0[tid] + input1[tid];
}
)MSL";

// Duration-controlled marker store. The loop-carried fma chain cannot be
// strength-reduced away, so `iters` really controls wall time; each thread
// then stores `marker`, which the host observes through the UMA pointer.
static std::string controlledStoreMsl = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct UintBufPtr      { device uint*       ptr; };
struct ConstUintBufPtr { device const uint* ptr; };

kernel void controlled_store(
    device UintBufPtr&      out_ref   [[buffer(0)]],
    device ConstUintBufPtr& iters_ref [[buffer(1)]],
    constant uint&          marker    [[buffer(2)]],
    constant uint&          length    [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= length) return;

    float acc = 0.0f;
    uint iters = iters_ref.ptr[tid];
    for (uint i = 0; i < iters; ++i)
        acc = fma(acc, 1.0000001f, 1.0f);

    out_ref.ptr[tid] = marker + (acc > 1e30f ? 1u : 0u);
}
)MSL";

TEST_F(DriverPluginTest, CustomStatusMessage) {
  char errorMessage[1024];
  ASSERT_EQ(M_DRIVER_STATUS_SUCCESS,
            this->status_message(driver, M_DRIVER_STATUS_UNIMPLEMENTED,
                                 errorMessage, sizeof(errorMessage)));
  ASSERT_STREQ(errorMessage, "Unimplemented");
  ASSERT_EQ(M_DRIVER_STATUS_SUCCESS,
            this->status_message(driver, M_DRIVER_STATUS_UNINITIALIZED,
                                 errorMessage, sizeof(errorMessage)));
  ASSERT_STREQ(errorMessage, "Uninitialized");
}

// Our macOS CI is Apple silicon, which is one pool shared with the CPU.
TEST_F(DriverPluginDeviceTest, HostUnifiedPropertyIsTrue) {
  for (M_driver_device_t device : devices) {
    int64_t unified = -1;
    ASSERT_SUCCESS(device_property(
        device, M_DRIVER_DEVICE_PROPERTY_HOST_UNIFIED, &unified));
    EXPECT_EQ(unified, 1);
  }
}

TEST_F(DriverPluginContextTest, FunctionLoad) {
  for (M_driver_context_t context : contexts) {
    M_driver_runtime_bundle_t library;
    std::string fileType("msl");
    M_driver_static_bundle staticBundle{.mapped_data =
                                            M_driver_slice_from(vecAddMsl),
                                        .file_type = fileType.c_str(),
                                        .file_type_len = fileType.size()};
    M_driver_bundle_compilation_options opts{"", 0, -1};
    ASSERT_SUCCESS(bundle_load(context, &staticBundle, &opts, &library));
    std::string funcName("vec_add_msl");
    M_driver_function_t function;
    ASSERT_SUCCESS(function_load(context, library, funcName.c_str(),
                                 funcName.size(), &function));
    ASSERT_SUCCESS(function_unload(context, function));
  }
}

// Allocating a buffer at or above the device's maxBufferLength must be
// rejected with a descriptive error rather than an opaque out-of-memory
// result. Before the maxBufferLength guard was ported from the device context,
// an oversized request fell through to newBuffer(), which returns null and
// yields a generic out-of-memory status with no indication that the request
// simply exceeded the per-buffer size limit.
TEST_F(DriverPluginContextTest, AllocRejectsOversizedBuffer) {
  for (M_driver_context_t context : contexts) {
    // Far larger than maxBufferLength on any Apple GPU (and beyond any
    // physical memory), so the allocation cannot possibly succeed.
    constexpr uint64_t oversized = 1ULL << 50; // 1 PiB

    M_driver_memory_t memory = nullptr;
    M_driver_status_t status = memory_alloc_sync(context, oversized, &memory);
    ASSERT_NE(M_DRIVER_STATUS_SUCCESS, status)
        << "Oversized allocation must not succeed";

    // The error must explain that the request exceeded the per-buffer limit,
    // not merely report a generic out-of-memory condition.
    std::string message = getStatusMessage(status);
    EXPECT_NE(message.find("larger than allowed"), std::string::npos)
        << "Expected a maxBufferLength guard message, got: " << message;
  }
}

// KV-cache sizing caps a buffer at this value, so it must be the guard's limit.
TEST_F(DriverPluginContextTest, MaxAllocationSizeMatchesAllocGuard) {
  ASSERT_EQ(devices.size(), contexts.size());
  for (size_t i = 0; i < contexts.size(); i++) {
    uint64_t maxAlloc = 0;
    ASSERT_SUCCESS(device_property(
        devices[i], M_DRIVER_DEVICE_PROPERTY_MAX_ALLOCATION_SIZE, &maxAlloc));
    ASSERT_GT(maxAlloc, 0u);

    M_driver_memory_t memory = nullptr;
    M_driver_status_t status =
        memory_alloc_sync(contexts[i], maxAlloc + 1, &memory);
    ASSERT_NE(M_DRIVER_STATUS_SUCCESS, status);
    std::string message = getStatusMessage(status);
    EXPECT_NE(message.find("larger than allowed"), std::string::npos)
        << "Expected the maxBufferLength guard to fire, got: " << message;
  }
}

struct DriverMetalPluginFunctionTest : DriverPluginQueueTest {
  void SetUp() override {
    DriverPluginQueueTest::SetUp();
    if (this->HasFatalFailure() || this->IsSkipped())
      return;

    for (size_t i = 0; i < contexts.size(); i++) {
      M_driver_context_t context = contexts[i];
      M_driver_runtime_bundle_t library;
      std::string fileType("msl");
      M_driver_static_bundle staticBundle{.mapped_data =
                                              M_driver_slice_from(vecAddMsl),
                                          .file_type = fileType.c_str(),
                                          .file_type_len = fileType.size()};
      M_driver_bundle_compilation_options opts{"", 0, -1};
      ASSERT_SUCCESS(bundle_load(context, &staticBundle, &opts, &library));
      libraries.push_back(library);
      std::string funcName("vec_add_msl");
      M_driver_function_t function;
      ASSERT_SUCCESS(function_load(context, library, funcName.c_str(),
                                   funcName.size(), &function));
      functions.push_back(function);
    }
  }

  void TearDown() override {
    for (size_t i = 0; i < contexts.size(); i++) {
      M_driver_context_t context = contexts[i];
      M_driver_function_t function = functions[i];
      M_driver_runtime_bundle_t library = libraries[i];
      EXPECT_SUCCESS(function_unload(context, function));
      EXPECT_SUCCESS(bundle_unload(context, library));
    }
    DriverPluginQueueTest::TearDown();
  }

  std::vector<M_driver_runtime_bundle_t> libraries;
  std::vector<M_driver_function_t> functions;
};

TEST_F(DriverMetalPluginFunctionTest, QueueExecute) {
  for (size_t i = 0; i < contexts.size(); i++) {
    [[maybe_unused]] M_driver_context_t context = contexts[i];
    [[maybe_unused]] M_driver_queue_t queue = queues[i];
    [[maybe_unused]] M_driver_function_t function = functions[i];

    // Test the function with some data
    constexpr size_t length = 1024;
    constexpr size_t elemSize = sizeof(float);
    constexpr size_t memorySize = length * elemSize;

    // Cleanup allocated device memory
    std::vector<M_driver_memory_t> memories;
    auto defer = llvm::scope_exit([&]() {
      for (M_driver_memory_t memory : memories)
        memory_free_sync(context, memory);
    });

    // Create device buffers
    M_driver_memory_t in0Memory;
    ASSERT_SUCCESS(memory_alloc_sync(context, memorySize, &in0Memory));
    memories.push_back(in0Memory);

    M_driver_memory_t in1Memory;
    ASSERT_SUCCESS(memory_alloc_sync(context, memorySize, &in1Memory));
    memories.push_back(in1Memory);

    M_driver_memory_t outMemory;
    ASSERT_SUCCESS(memory_alloc_sync(context, memorySize, &outMemory));
    memories.push_back(outMemory);

    // Initialize host buffers
    std::vector<float> in0Host(length);
    std::vector<float> in1Host(length);
    std::vector<float> outHost(length);

    for (size_t i = 0; i < length; ++i) {
      in0Host[i] = static_cast<float>(i);
      in1Host[i] = 2.0f;
      outHost[i] = 0.0f;
    }

    // Copy data to device
    M_driver_memory_view toView0{in0Memory, /*offset=*/0, memorySize};
    ASSERT_SUCCESS(queue_copy_to_device(queue, &toView0, in0Host.data()));
    M_driver_memory_view toView1{in1Memory, /*offset=*/0, memorySize};
    ASSERT_SUCCESS(queue_copy_to_device(queue, &toView1, in1Host.data()));
    ASSERT_SUCCESS(queue_synchronize(queue));

    constexpr uint32_t blockDim = 256;
    M_driver_queue_execute_config_t config = {
        .mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU,
        .config = {.gpu = {
                       .grid = {length / blockDim, 1, 1},
                       .block = {blockDim, 1, 1},
                       .shared_mem_bytes = 0,
                       .attributes = nullptr,
                       .num_attributes = 0,
                   }}};

    void *outPtr = nullptr;
    ASSERT_SUCCESS(memory_property(outMemory, "address", &outPtr));
    ASSERT_NE(nullptr, outPtr);
    void *in0Ptr = nullptr;
    ASSERT_SUCCESS(memory_property(in0Memory, "address", &in0Ptr));
    ASSERT_NE(nullptr, in0Ptr);
    void *in1Ptr = nullptr;
    ASSERT_SUCCESS(memory_property(in1Memory, "address", &in1Ptr));
    ASSERT_NE(nullptr, in1Ptr);
    uint32_t lengthValue = static_cast<uint32_t>(length);
    void *args[] = {&outPtr, &in0Ptr, &in1Ptr, &lengthValue};
    uint64_t argSizes[] = {sizeof(outPtr), sizeof(in0Ptr), sizeof(in1Ptr),
                           sizeof(lengthValue)};

    ASSERT_SUCCESS(queue_execute(queue, function, &config, args, argSizes, 4));

    // Copy results back
    M_driver_memory_view fromView{outMemory, /*offset=*/0, memorySize};
    ASSERT_SUCCESS(queue_copy_from_device(queue, outHost.data(), &fromView));
    ASSERT_SUCCESS(queue_synchronize(queue));

    // Verify results
    for (size_t i = 0; i < length; ++i) {
      float expected = static_cast<float>(i) + 2.0f;
      EXPECT_FLOAT_EQ(outHost[i], expected)
          << "Mismatch at index " << i << ": expected " << expected << ", got "
          << outHost[i];
    }
  }
}

// Regression test: an invalid argument (null pointer or zero size) must be
// rejected with an error *before* a compute encoder is opened. The zero-size
// path previously returned mid-encode, leaving the encoder open; Metal asserts
// when a command buffer is dropped with an open encoder, aborting the process.
TEST_F(DriverMetalPluginFunctionTest, QueueExecuteRejectsInvalidArgs) {
  for (size_t i = 0; i < contexts.size(); i++) {
    M_driver_queue_t queue = queues[i];
    M_driver_function_t function = functions[i];

    M_driver_queue_execute_config_t config = {
        .mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU,
        .config = {.gpu = {
                       .grid = {1, 1, 1},
                       .block = {1, 1, 1},
                       .shared_mem_bytes = 0,
                       .attributes = nullptr,
                       .num_attributes = 0,
                   }}};

    // Four non-null argument pointers matching the kernel's argument count.
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;

    // A zero-sized argument is rejected — this is the path that used to leak an
    // open encoder.
    void *args[] = {&a0, &a1, &a2, &a3};
    uint64_t zeroSizeArgSizes[] = {sizeof(a0), 0, sizeof(a2), sizeof(a3)};
    EXPECT_NE(
        M_DRIVER_STATUS_SUCCESS,
        queue_execute(queue, function, &config, args, zeroSizeArgSizes, 4));

    // A null argument is rejected.
    void *nullArgs[] = {&a0, nullptr, &a2, &a3};
    uint64_t argSizes[] = {sizeof(a0), sizeof(a1), sizeof(a2), sizeof(a3)};
    EXPECT_NE(M_DRIVER_STATUS_SUCCESS,
              queue_execute(queue, function, &config, nullArgs, argSizes, 4));

    // The queue must still be usable afterwards: no encoder was leaked and no
    // command buffer was dropped with work pending.
    EXPECT_SUCCESS(queue_synchronize(queue));
  }
}

// Requesting more dynamic threadgroup (shared) memory than the device supports
// must be rejected. Prior to wiring up `shared_mem_bytes`, the field was
// silently ignored, so an absurd request would "succeed" and the kernel would
// run as if no shared memory had been asked for.
TEST_F(DriverMetalPluginFunctionTest, QueueExecuteOversizedSharedMemFails) {
  for (size_t i = 0; i < contexts.size(); i++) {
    M_driver_context_t context = contexts[i];
    M_driver_queue_t queue = queues[i];
    M_driver_function_t function = functions[i];

    constexpr size_t length = 1024;
    constexpr size_t memorySize = length * sizeof(float);

    std::vector<M_driver_memory_t> memories;
    auto defer = llvm::scope_exit([&]() {
      for (M_driver_memory_t memory : memories)
        memory_free_sync(context, memory);
    });

    M_driver_memory_t in0Memory, in1Memory, outMemory;
    ASSERT_SUCCESS(memory_alloc_sync(context, memorySize, &in0Memory));
    memories.push_back(in0Memory);
    ASSERT_SUCCESS(memory_alloc_sync(context, memorySize, &in1Memory));
    memories.push_back(in1Memory);
    ASSERT_SUCCESS(memory_alloc_sync(context, memorySize, &outMemory));
    memories.push_back(outMemory);

    void *outPtr = nullptr, *in0Ptr = nullptr, *in1Ptr = nullptr;
    ASSERT_SUCCESS(memory_property(outMemory, "address", &outPtr));
    ASSERT_SUCCESS(memory_property(in0Memory, "address", &in0Ptr));
    ASSERT_SUCCESS(memory_property(in1Memory, "address", &in1Ptr));
    uint32_t lengthValue = static_cast<uint32_t>(length);
    void *args[] = {&outPtr, &in0Ptr, &in1Ptr, &lengthValue};
    uint64_t argSizes[] = {sizeof(outPtr), sizeof(in0Ptr), sizeof(in1Ptr),
                           sizeof(lengthValue)};

    constexpr uint32_t blockDim = 256;
    // Far larger than any Apple GPU threadgroup memory limit (32 KiB today).
    constexpr uint32_t oversizedSharedMem = 1u << 30; // 1 GiB
    M_driver_queue_execute_config_t config = {
        .mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU,
        .config = {.gpu = {
                       .grid = {length / blockDim, 1, 1},
                       .block = {blockDim, 1, 1},
                       .shared_mem_bytes = oversizedSharedMem,
                       .attributes = nullptr,
                       .num_attributes = 0,
                   }}};

    M_driver_status_t status =
        queue_execute(queue, function, &config, args, argSizes, 4);
    EXPECT_NE(M_DRIVER_STATUS_SUCCESS, status)
        << "execute() must reject shared_mem_bytes that exceeds the device "
           "threadgroup memory limit";
  }
}

// A valid, in-limit, non-zero `shared_mem_bytes` request must still be accepted
// and produce correct results. This guards against the validation rejecting
// legitimate requests and exercises the setThreadgroupMemoryLength() binding
// path on a regular kernel.
TEST_F(DriverMetalPluginFunctionTest, QueueExecuteValidSharedMem) {
  for (size_t i = 0; i < contexts.size(); i++) {
    M_driver_context_t context = contexts[i];
    M_driver_queue_t queue = queues[i];
    M_driver_function_t function = functions[i];

    constexpr size_t length = 1024;
    constexpr size_t memorySize = length * sizeof(float);

    std::vector<M_driver_memory_t> memories;
    auto defer = llvm::scope_exit([&]() {
      for (M_driver_memory_t memory : memories)
        memory_free_sync(context, memory);
    });

    M_driver_memory_t in0Memory, in1Memory, outMemory;
    ASSERT_SUCCESS(memory_alloc_sync(context, memorySize, &in0Memory));
    memories.push_back(in0Memory);
    ASSERT_SUCCESS(memory_alloc_sync(context, memorySize, &in1Memory));
    memories.push_back(in1Memory);
    ASSERT_SUCCESS(memory_alloc_sync(context, memorySize, &outMemory));
    memories.push_back(outMemory);

    std::vector<float> in0Host(length), in1Host(length), outHost(length, 0.0f);
    for (size_t j = 0; j < length; ++j) {
      in0Host[j] = static_cast<float>(j);
      in1Host[j] = 2.0f;
    }

    M_driver_memory_view toView0{in0Memory, /*offset=*/0, memorySize};
    ASSERT_SUCCESS(queue_copy_to_device(queue, &toView0, in0Host.data()));
    M_driver_memory_view toView1{in1Memory, /*offset=*/0, memorySize};
    ASSERT_SUCCESS(queue_copy_to_device(queue, &toView1, in1Host.data()));
    ASSERT_SUCCESS(queue_synchronize(queue));

    void *outPtr = nullptr, *in0Ptr = nullptr, *in1Ptr = nullptr;
    ASSERT_SUCCESS(memory_property(outMemory, "address", &outPtr));
    ASSERT_SUCCESS(memory_property(in0Memory, "address", &in0Ptr));
    ASSERT_SUCCESS(memory_property(in1Memory, "address", &in1Ptr));
    uint32_t lengthValue = static_cast<uint32_t>(length);
    void *args[] = {&outPtr, &in0Ptr, &in1Ptr, &lengthValue};
    uint64_t argSizes[] = {sizeof(outPtr), sizeof(in0Ptr), sizeof(in1Ptr),
                           sizeof(lengthValue)};

    constexpr uint32_t blockDim = 256;
    // Modest, comfortably within the Apple GPU threadgroup memory limit.
    constexpr uint32_t validSharedMem = blockDim * sizeof(float); // 1 KiB
    M_driver_queue_execute_config_t config = {
        .mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU,
        .config = {.gpu = {
                       .grid = {length / blockDim, 1, 1},
                       .block = {blockDim, 1, 1},
                       .shared_mem_bytes = validSharedMem,
                       .attributes = nullptr,
                       .num_attributes = 0,
                   }}};

    ASSERT_SUCCESS(queue_execute(queue, function, &config, args, argSizes, 4));
    M_driver_memory_view fromView{outMemory, /*offset=*/0, memorySize};
    ASSERT_SUCCESS(queue_copy_from_device(queue, outHost.data(), &fromView));
    ASSERT_SUCCESS(queue_synchronize(queue));

    for (size_t j = 0; j < length; ++j) {
      float expected = static_cast<float>(j) + 2.0f;
      EXPECT_FLOAT_EQ(outHost[j], expected)
          << "Mismatch at index " << j << ": expected " << expected << ", got "
          << outHost[j];
    }
  }
}

// Event/ordering contract of the queue spine at the M_driver ABI boundary.
// Oracles: is_event_ready + host reads of GPU stores via the UMA pointer
// (completion handlers are commit-ordered and hide GPU completion order).
struct DriverMetalPluginEventTest : DriverPluginQueueTest {
  static constexpr uint32_t kLength = 64;
  static constexpr uint32_t kLongIters = 30000000; // ~400ms on M5
  static constexpr uint32_t kShortIters = 1000000; // ~15ms

  void SetUp() override {
    DriverPluginQueueTest::SetUp();
    if (this->HasFatalFailure() || this->IsSkipped())
      return;

    for (M_driver_context_t context : contexts) {
      M_driver_runtime_bundle_t library;
      std::string fileType("msl");
      M_driver_static_bundle staticBundle{
          .mapped_data = M_driver_slice_from(controlledStoreMsl),
          .file_type = fileType.c_str(),
          .file_type_len = fileType.size()};
      M_driver_bundle_compilation_options opts{"", 0, -1};
      ASSERT_SUCCESS(bundle_load(context, &staticBundle, &opts, &library));
      libraries.push_back(library);
      std::string funcName("controlled_store");
      M_driver_function_t function;
      ASSERT_SUCCESS(function_load(context, library, funcName.c_str(),
                                   funcName.size(), &function));
      functions.push_back(function);
    }
  }

  void TearDown() override {
    for (M_driver_memory_t memory : allocations)
      memory_free_sync(owner[memory], memory);
    for (M_driver_memory_t memory : pinnedAllocations)
      memory_free_pinned(owner[memory], memory);
    for (size_t i = 0; i < libraries.size(); i++) {
      EXPECT_SUCCESS(function_unload(contexts[i], functions[i]));
      EXPECT_SUCCESS(bundle_unload(contexts[i], libraries[i]));
    }
    DriverPluginQueueTest::TearDown();
  }

  // Allocates kLength uint32 words, returns the device handle and the
  // UMA-visible host pointer ("address" is the GPU address and is not
  // host-dereferenceable); freed in TearDown.
  void allocWords(M_driver_context_t context, M_driver_memory_t *outMemory,
                  volatile uint32_t **outHost) {
    ASSERT_SUCCESS(
        memory_alloc_sync(context, kLength * sizeof(uint32_t), outMemory));
    allocations.push_back(*outMemory);
    owner[*outMemory] = context;
    void *host = nullptr;
    ASSERT_SUCCESS(memory_property(*outMemory, "host_address", &host));
    ASSERT_NE(nullptr, host);
    *outHost = static_cast<volatile uint32_t *>(host);
  }

  void allocPinnedWords(M_driver_context_t context,
                        M_driver_memory_t *outMemory, uint32_t **outHost) {
    ASSERT_SUCCESS(
        memory_alloc_pinned(context, kLength * sizeof(uint32_t), outMemory));
    pinnedAllocations.push_back(*outMemory);
    owner[*outMemory] = context;
    void *host = nullptr;
    ASSERT_SUCCESS(memory_property(*outMemory, "host_address", &host));
    ASSERT_NE(nullptr, host);
    *outHost = static_cast<uint32_t *>(host);
  }

  // Launches controlled_store: every lane spins `iters` then stores `marker`
  // into `out`. `itersMemory` must already hold the per-lane counts.
  void launchStore(M_driver_queue_t queue, M_driver_function_t function,
                   M_driver_memory_t outMemory, M_driver_memory_t itersMemory,
                   uint32_t marker) {
    void *outPtr = nullptr, *itersPtr = nullptr;
    ASSERT_SUCCESS(memory_property(outMemory, "address", &outPtr));
    ASSERT_SUCCESS(memory_property(itersMemory, "address", &itersPtr));
    uint32_t length = kLength;
    void *args[] = {&outPtr, &itersPtr, &marker, &length};
    uint64_t argSizes[] = {sizeof(outPtr), sizeof(itersPtr), sizeof(marker),
                           sizeof(length)};
    M_driver_queue_execute_config_t config = {
        .mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU,
        .config = {.gpu = {
                       .grid = {1, 1, 1},
                       .block = {kLength, 1, 1},
                       .shared_mem_bytes = 0,
                       .attributes = nullptr,
                       .num_attributes = 0,
                   }}};
    ASSERT_SUCCESS(queue_execute(queue, function, &config, args, argSizes, 4));
  }

  void fillIters(volatile uint32_t *host, uint32_t iters) {
    for (uint32_t i = 0; i < kLength; ++i)
      host[i] = iters;
  }

  // Polls the UMA pointer until every lane shows `marker` or the deadline
  // passes; returns whether the store became fully visible.
  bool pollStore(volatile uint32_t *host, uint32_t marker, int deadlineMs) {
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(deadlineMs);
    while (std::chrono::steady_clock::now() < deadline) {
      bool all = true;
      for (uint32_t i = 0; i < kLength && all; ++i)
        all = (host[i] == marker);
      if (all)
        return true;
      std::this_thread::yield();
    }
    return false;
  }

  bool eventReady(M_driver_context_t context, M_driver_event_t event) {
    bool ready = false;
    EXPECT_SUCCESS(is_event_ready(context, event, &ready));
    return ready;
  }

  std::vector<M_driver_runtime_bundle_t> libraries;
  std::vector<M_driver_function_t> functions;
  std::vector<M_driver_memory_t> allocations;
  std::vector<M_driver_memory_t> pinnedAllocations;
  std::map<M_driver_memory_t, M_driver_context_t> owner;
};

// A record gathers EVERY prior op (union fan-in): recorded behind a long and
// a short launch, the event is not ready while the long one runs, and
// synchronizing it makes both outputs host-visible.
TEST_F(DriverMetalPluginEventTest, RecordEventObservesAllPriorWork) {
  for (size_t i = 0; i < contexts.size(); i++) {
    M_driver_context_t context = contexts[i];
    M_driver_queue_t queue = queues[i];

    M_driver_memory_t outLong, outShort, itersLong, itersShort;
    volatile uint32_t *outLongHost, *outShortHost, *itersLongHost,
        *itersShortHost;
    allocWords(context, &outLong, &outLongHost);
    allocWords(context, &outShort, &outShortHost);
    allocWords(context, &itersLong, &itersLongHost);
    allocWords(context, &itersShort, &itersShortHost);
    fillIters(itersLongHost, kLongIters);
    fillIters(itersShortHost, kShortIters);

    launchStore(queue, functions[i], outLong, itersLong, /*marker=*/1);
    launchStore(queue, functions[i], outShort, itersShort, /*marker=*/1);

    M_driver_event_t event;
    ASSERT_SUCCESS(
        event_create(context, M_DRIVER_EVENT_FLAG_CPU_VISIBLE, &event));
    auto defer = llvm::scope_exit([&] { event_destroy(context, event); });
    ASSERT_SUCCESS(queue_record_event(queue, event));

    // The long launch is still running when we get here (record is async).
    EXPECT_FALSE(eventReady(context, event));

    ASSERT_SUCCESS(event_synchronize(context, event));
    EXPECT_TRUE(eventReady(context, event));
    EXPECT_EQ(outLongHost[0], 1u);
    EXPECT_EQ(outLongHost[kLength - 1], 1u);
    EXPECT_EQ(outShortHost[0], 1u);
    EXPECT_EQ(outShortHost[kLength - 1], 1u);
  }
}

// A record is a fan-in point, not a barrier: an op committed after it becomes
// host-visible while the record is still pending. DISABLED: execute()'s
// useResource-everything FIXME makes hazard tracking serialize the kernels.
TEST_F(DriverMetalPluginEventTest, DISABLED_RecordEventIsNotABarrier) {
  for (size_t i = 0; i < contexts.size(); i++) {
    M_driver_context_t context = contexts[i];
    M_driver_queue_t queue = queues[i];

    M_driver_memory_t outLong, outShort, itersLong, itersShort;
    volatile uint32_t *outLongHost, *outShortHost, *itersLongHost,
        *itersShortHost;
    allocWords(context, &outLong, &outLongHost);
    allocWords(context, &outShort, &outShortHost);
    allocWords(context, &itersLong, &itersLongHost);
    allocWords(context, &itersShort, &itersShortHost);
    fillIters(itersLongHost, kLongIters);
    fillIters(itersShortHost, kShortIters);

    launchStore(queue, functions[i], outLong, itersLong, /*marker=*/1);

    M_driver_event_t event;
    ASSERT_SUCCESS(
        event_create(context, M_DRIVER_EVENT_FLAG_CPU_VISIBLE, &event));
    auto defer = llvm::scope_exit([&] { event_destroy(context, event); });
    ASSERT_SUCCESS(queue_record_event(queue, event));

    launchStore(queue, functions[i], outShort, itersShort, /*marker=*/1);

    // The short op overtakes the pending record.
    EXPECT_TRUE(pollStore(outShortHost, 1, /*deadlineMs=*/2000));
    EXPECT_FALSE(eventReady(context, event));

    ASSERT_SUCCESS(event_synchronize(context, event));
    EXPECT_EQ(outLongHost[0], 1u);
    EXPECT_EQ(outLongHost[kLength - 1], 1u);
  }
}

// Re-recording rebinds the event to the newest work: ready after the first
// record drains, pending again once re-recorded behind a long launch, and the
// second synchronize observes the second marker.
TEST_F(DriverMetalPluginEventTest, EventReRecordTracksNewWork) {
  for (size_t i = 0; i < contexts.size(); i++) {
    M_driver_context_t context = contexts[i];
    M_driver_queue_t queue = queues[i];

    M_driver_memory_t out, iters;
    volatile uint32_t *outHost, *itersHost;
    allocWords(context, &out, &outHost);
    allocWords(context, &iters, &itersHost);
    fillIters(itersHost, kLongIters);

    M_driver_event_t event;
    ASSERT_SUCCESS(
        event_create(context, M_DRIVER_EVENT_FLAG_CPU_VISIBLE, &event));
    auto defer = llvm::scope_exit([&] { event_destroy(context, event); });

    launchStore(queue, functions[i], out, iters, /*marker=*/1);
    ASSERT_SUCCESS(queue_record_event(queue, event));
    ASSERT_SUCCESS(event_synchronize(context, event));
    EXPECT_TRUE(eventReady(context, event));
    EXPECT_EQ(outHost[kLength - 1], 1u);

    launchStore(queue, functions[i], out, iters, /*marker=*/2);
    ASSERT_SUCCESS(queue_record_event(queue, event));
    // A re-record must not stay ready off the OLD signal while the new work
    // is still running.
    EXPECT_FALSE(eventReady(context, event));

    ASSERT_SUCCESS(event_synchronize(context, event));
    EXPECT_TRUE(eventReady(context, event));
    EXPECT_EQ(outHost[0], 2u);
    EXPECT_EQ(outHost[kLength - 1], 2u);
  }
}

// queue_synchronize drains everything previously submitted, including work
// that overlapped or completed out of order, and is idempotent on an empty
// queue.
TEST_F(DriverMetalPluginEventTest, SynchronizeDrainsAllWork) {
  for (size_t i = 0; i < contexts.size(); i++) {
    M_driver_context_t context = contexts[i];
    M_driver_queue_t queue = queues[i];

    ASSERT_SUCCESS(queue_synchronize(queue));
    ASSERT_SUCCESS(queue_synchronize(queue));

    M_driver_memory_t outLong, outShort, itersLong, itersShort;
    volatile uint32_t *outLongHost, *outShortHost, *itersLongHost,
        *itersShortHost;
    allocWords(context, &outLong, &outLongHost);
    allocWords(context, &outShort, &outShortHost);
    allocWords(context, &itersLong, &itersLongHost);
    allocWords(context, &itersShort, &itersShortHost);
    fillIters(itersLongHost, kLongIters);
    fillIters(itersShortHost, kShortIters);

    launchStore(queue, functions[i], outLong, itersLong, /*marker=*/1);
    launchStore(queue, functions[i], outShort, itersShort, /*marker=*/1);
    ASSERT_SUCCESS(queue_synchronize(queue));

    EXPECT_EQ(outLongHost[0], 1u);
    EXPECT_EQ(outLongHost[kLength - 1], 1u);
    EXPECT_EQ(outShortHost[0], 1u);
    EXPECT_EQ(outShortHost[kLength - 1], 1u);
  }
}

// wait_for_events gates a SECOND queue's later work on an event recorded
// behind long work on the first queue, without blocking the host.
TEST_F(DriverMetalPluginEventTest, WaitForEventsGatesSecondQueue) {
  for (size_t i = 0; i < contexts.size(); i++) {
    M_driver_context_t context = contexts[i];
    M_driver_queue_t queueA = queues[i];

    M_driver_queue_t queueB;
    ASSERT_SUCCESS(queue_create(context, &queueB));
    auto deferQ = llvm::scope_exit([&] { queue_destroy(context, queueB); });

    M_driver_memory_t outLong, itersLong;
    volatile uint32_t *outLongHost, *itersLongHost;
    allocWords(context, &outLong, &outLongHost);
    allocWords(context, &itersLong, &itersLongHost);
    fillIters(itersLongHost, kLongIters);

    launchStore(queueA, functions[i], outLong, itersLong, /*marker=*/1);
    M_driver_event_t eventA;
    ASSERT_SUCCESS(
        event_create(context, M_DRIVER_EVENT_FLAG_CPU_VISIBLE, &eventA));
    auto deferA = llvm::scope_exit([&] { event_destroy(context, eventA); });
    ASSERT_SUCCESS(queue_record_event(queueA, eventA));

    ASSERT_SUCCESS(queue_wait_for_events(queueB, &eventA, 1));
    M_driver_event_t probeB;
    ASSERT_SUCCESS(
        event_create(context, M_DRIVER_EVENT_FLAG_CPU_VISIBLE, &probeB));
    auto deferB = llvm::scope_exit([&] { event_destroy(context, probeB); });
    ASSERT_SUCCESS(queue_record_event(queueB, probeB));

    // B's probe is pending while A's long launch runs.
    EXPECT_FALSE(eventReady(context, probeB));

    ASSERT_SUCCESS(event_synchronize(context, probeB));
    EXPECT_EQ(outLongHost[0], 1u);
    EXPECT_EQ(outLongHost[kLength - 1], 1u);
  }
}

// A pinned-host copy is enqueued like cuMemcpyHtoDAsync: the call returns while
// earlier work still runs, and the data lands once the queue drains.
TEST_F(DriverMetalPluginEventTest, PinnedCopyToDeviceDoesNotBlock) {
  for (size_t i = 0; i < contexts.size(); i++) {
    M_driver_context_t context = contexts[i];
    M_driver_queue_t queue = queues[i];

    M_driver_memory_t outLong, itersLong, dst, src;
    volatile uint32_t *outLongHost, *itersLongHost, *dstHost;
    uint32_t *srcHost;
    allocWords(context, &outLong, &outLongHost);
    allocWords(context, &itersLong, &itersLongHost);
    allocWords(context, &dst, &dstHost);
    allocPinnedWords(context, &src, &srcHost);
    fillIters(itersLongHost, kLongIters);
    for (uint32_t j = 0; j < kLength; ++j)
      srcHost[j] = 100 + j;

    launchStore(queue, functions[i], outLong, itersLong, /*marker=*/1);
    M_driver_memory_view dstView{dst, /*offset=*/0, kLength * sizeof(uint32_t)};
    ASSERT_SUCCESS(queue_copy_to_device(queue, &dstView, srcHost));
    EXPECT_NE(outLongHost[kLength - 1], 1u)
        << "copy_to_device from pinned memory waited for prior work";

    ASSERT_SUCCESS(queue_synchronize(queue));
    for (uint32_t j = 0; j < kLength; ++j)
      ASSERT_EQ(dstHost[j], 100 + j) << "index " << j;
  }
}

// A pinned-host readback is enqueued like cuMemcpyDtoHAsync, ordered after the
// kernel that produces its source.
TEST_F(DriverMetalPluginEventTest, PinnedCopyFromDeviceDoesNotBlock) {
  for (size_t i = 0; i < contexts.size(); i++) {
    M_driver_context_t context = contexts[i];
    M_driver_queue_t queue = queues[i];

    M_driver_memory_t outLong, itersLong, dst;
    volatile uint32_t *outLongHost, *itersLongHost;
    uint32_t *dstHost;
    allocWords(context, &outLong, &outLongHost);
    allocWords(context, &itersLong, &itersLongHost);
    allocPinnedWords(context, &dst, &dstHost);
    fillIters(itersLongHost, kLongIters);
    for (uint32_t j = 0; j < kLength; ++j)
      dstHost[j] = 0;

    launchStore(queue, functions[i], outLong, itersLong, /*marker=*/7);
    M_driver_memory_view srcView{outLong, /*offset=*/0,
                                 kLength * sizeof(uint32_t)};
    ASSERT_SUCCESS(queue_copy_from_device(queue, dstHost, &srcView));
    EXPECT_NE(outLongHost[kLength - 1], 7u)
        << "copy_from_device into pinned memory waited for prior work";

    ASSERT_SUCCESS(queue_synchronize(queue));
    for (uint32_t j = 0; j < kLength; ++j)
      ASSERT_EQ(dstHost[j], 7u) << "index " << j;
  }
}

// An interior pointer into a pinned allocation resolves to that allocation, not
// to the pageable fallback.
TEST_F(DriverMetalPluginEventTest, PinnedSubRangeCopyDoesNotBlock) {
  static constexpr uint32_t kSkip = 16;
  for (size_t i = 0; i < contexts.size(); i++) {
    M_driver_context_t context = contexts[i];
    M_driver_queue_t queue = queues[i];

    M_driver_memory_t outLong, itersLong, dst, src;
    volatile uint32_t *outLongHost, *itersLongHost, *dstHost;
    uint32_t *srcHost;
    allocWords(context, &outLong, &outLongHost);
    allocWords(context, &itersLong, &itersLongHost);
    allocWords(context, &dst, &dstHost);
    allocPinnedWords(context, &src, &srcHost);
    fillIters(itersLongHost, kLongIters);
    for (uint32_t j = 0; j < kLength; ++j) {
      srcHost[j] = 100 + j;
      dstHost[j] = 0;
    }

    launchStore(queue, functions[i], outLong, itersLong, /*marker=*/1);
    M_driver_memory_view dstView{dst, /*offset=*/kSkip * sizeof(uint32_t),
                                 (kLength - kSkip) * sizeof(uint32_t)};
    ASSERT_SUCCESS(queue_copy_to_device(queue, &dstView, srcHost + kSkip));
    EXPECT_NE(outLongHost[kLength - 1], 1u)
        << "copy_to_device from a pinned sub-range waited for prior work";

    ASSERT_SUCCESS(queue_synchronize(queue));
    for (uint32_t j = 0; j < kSkip; ++j)
      ASSERT_EQ(dstHost[j], 0u) << "index " << j;
    for (uint32_t j = kSkip; j < kLength; ++j)
      ASSERT_EQ(dstHost[j], 100 + j) << "index " << j;
  }
}

// Pageable copies stay synchronous, as cuMemcpyHtoDAsync / cuMemcpyDtoHAsync
// are for pageable memory: each waits for prior work before returning.
TEST_F(DriverMetalPluginEventTest, PageableCopiesWaitForPriorWork) {
  for (size_t i = 0; i < contexts.size(); i++) {
    M_driver_context_t context = contexts[i];
    M_driver_queue_t queue = queues[i];

    M_driver_memory_t outLong, itersLong, dst;
    volatile uint32_t *outLongHost, *itersLongHost, *dstHost;
    allocWords(context, &outLong, &outLongHost);
    allocWords(context, &itersLong, &itersLongHost);
    allocWords(context, &dst, &dstHost);
    fillIters(itersLongHost, kLongIters);
    std::vector<uint32_t> pageable(kLength, 5);
    M_driver_memory_view dstView{dst, /*offset=*/0, kLength * sizeof(uint32_t)};

    launchStore(queue, functions[i], outLong, itersLong, /*marker=*/1);
    ASSERT_SUCCESS(queue_copy_to_device(queue, &dstView, pageable.data()));
    EXPECT_EQ(outLongHost[kLength - 1], 1u);
    EXPECT_EQ(dstHost[kLength - 1], 5u);

    launchStore(queue, functions[i], outLong, itersLong, /*marker=*/2);
    M_driver_memory_view srcView{outLong, /*offset=*/0,
                                 kLength * sizeof(uint32_t)};
    ASSERT_SUCCESS(queue_copy_from_device(queue, pageable.data(), &srcView));
    EXPECT_EQ(pageable[0], 2u);
    EXPECT_EQ(pageable[kLength - 1], 2u);
  }
}
