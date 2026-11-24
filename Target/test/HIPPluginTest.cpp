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

#include "../lib/HIP/Error.h"
#include "Fixture.h"
#include "HAL/M_driver_bundle.h"
#include "HAL/M_driver_common.h"
#include "HAL/M_driver_memory.h"

#include "hip/hiprtc.h"
#include "llvm/ADT/ScopeExit.h"

#include <cstdio>
#include <dlfcn.h>
#include <vector>

/// NOTE: assumes `s` is null terminated
M_driver_slice M_driver_slice_from(const char *s) {
  uint64_t size = strlen(s);
  return M_driver_slice{s, size};
}

// Extend test-cases per-plugin beyond the common ones in PluginTest.cpp

TEST_F(DriverPluginTest, CustomStatusMessage) {
  char errorMessage[1024];
  ASSERT_EQ(M_DRIVER_STATUS_SUCCESS,
            this->status_message(driver, M_DRIVER_STATUS_UNIMPLEMENTED,
                                 errorMessage, sizeof(errorMessage)));
  ASSERT_STREQ(errorMessage, "Unimplemented");
  ASSERT_EQ(M_DRIVER_STATUS_SUCCESS,
            this->status_message(driver, M_DRIVER_STATUS_BAD_BUNDLE,
                                 errorMessage, sizeof(errorMessage)));
  ASSERT_STREQ(errorMessage, "Bad bundle");
  ASSERT_EQ(M_DRIVER_STATUS_SUCCESS,
            this->status_message(driver, M_DRIVER_STATUS_BAD_FUNCTION,
                                 errorMessage, sizeof(errorMessage)));
  ASSERT_STREQ(errorMessage, "Bad function");
}

// Simple HIP C++ kernel source for runtime compilation via HIPRTC.
static const char *vecAddSrc = R"HIP(
extern "C" __global__ void vec_add(const float *a, const float *b, float *c,
                                   long n) {
  long idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    c[idx] = a[idx] + b[idx];
}
)HIP";

/// Helper: compile vecAddSrc to HSACO binary via HIPRTC (dlopen).
/// Returns the compiled code object bytes, or an empty vector on failure.
static std::vector<char> compileHIPKernel() {
  // dlopen libhiprtc.so at test time rather than linking against it,
  // keeping the plugin itself free of any HIPRTC dependency.
  void *rtcLib = dlopen("libhiprtc.so", RTLD_NOW);
  if (!rtcLib) {
    fprintf(stderr, "HIPPluginTest: dlopen(libhiprtc.so) failed: %s\n",
            dlerror());
    return {};
  }

  auto createProg = reinterpret_cast<decltype(&hiprtcCreateProgram)>(
      dlsym(rtcLib, "hiprtcCreateProgram"));
  auto compileProg = reinterpret_cast<decltype(&hiprtcCompileProgram)>(
      dlsym(rtcLib, "hiprtcCompileProgram"));
  auto getCodeSize = reinterpret_cast<decltype(&hiprtcGetCodeSize)>(
      dlsym(rtcLib, "hiprtcGetCodeSize"));
  auto getCode = reinterpret_cast<decltype(&hiprtcGetCode)>(
      dlsym(rtcLib, "hiprtcGetCode"));
  auto destroyProg = reinterpret_cast<decltype(&hiprtcDestroyProgram)>(
      dlsym(rtcLib, "hiprtcDestroyProgram"));

  if (!createProg || !compileProg || !getCodeSize || !getCode || !destroyProg) {
    fprintf(stderr, "HIPPluginTest: failed to resolve HIPRTC symbols\n");
    dlclose(rtcLib);
    return {};
  }

  hiprtcProgram prog = nullptr;
  if (createProg(&prog, vecAddSrc, "vec_add.hip", 0, nullptr, nullptr) !=
      HIPRTC_SUCCESS) {
    dlclose(rtcLib);
    return {};
  }

  if (compileProg(prog, 0, nullptr) != HIPRTC_SUCCESS) {
    destroyProg(&prog);
    dlclose(rtcLib);
    return {};
  }

  size_t codeSize = 0;
  if (getCodeSize(prog, &codeSize) != HIPRTC_SUCCESS || codeSize == 0) {
    destroyProg(&prog);
    dlclose(rtcLib);
    return {};
  }

  std::vector<char> code(codeSize);
  if (getCode(prog, code.data()) != HIPRTC_SUCCESS) {
    destroyProg(&prog);
    dlclose(rtcLib);
    return {};
  }

  destroyProg(&prog);
  dlclose(rtcLib);
  return code;
}

/// Helper: build a static code bundle from compiled HSACO bytes.
/// Note: fileType must outlive the returned bundle (non-owning pointer).
static M_driver_static_bundle makeStaticBundle(const std::vector<char> &hsaco,
                                               const std::string &fileType) {
  return M_driver_static_bundle{
      .mapped_data = {.data = hsaco.data(), .size = hsaco.size()},
      .file_type = fileType.c_str(),
      .file_type_len = fileType.size(),
  };
}

TEST_F(DriverPluginDeviceTest, HostUnifiedPropertyIsImplemented) {
  for (M_driver_device_t device : devices) {
    int64_t unified = -1;
    ASSERT_SUCCESS(device_property(
        device, M_DRIVER_DEVICE_PROPERTY_HOST_UNIFIED, &unified));
    EXPECT_TRUE(unified == 0 || unified == 1) << "got " << unified;
  }
}

TEST_F(DriverPluginContextTest, BundleLoadNullBundle) {
  M_driver_runtime_bundle_t runtime = nullptr;
  M_driver_bundle_compilation_options opts{"", 0, -1};
  M_driver_status_t status = bundle_load(contexts[0], nullptr, &opts, &runtime);
  EXPECT_EQ(status, M_DRIVER_STATUS_BAD_BUNDLE);
  EXPECT_EQ(runtime, nullptr);
}

TEST_F(DriverPluginContextTest, BundleLoadIncompatibleFileType) {
  std::string fileType("ptx");
  M_driver_static_bundle badBundle{.mapped_data = {.data = "fake", .size = 4},
                                   .file_type = fileType.c_str(),
                                   .file_type_len = fileType.size()};
  M_driver_bundle_compilation_options opts{"", 0, -1};
  M_driver_runtime_bundle_t runtime = nullptr;
  M_driver_status_t status =
      bundle_load(contexts[0], &badBundle, &opts, &runtime);
  EXPECT_EQ(status, M_DRIVER_STATUS_INCOMPATIBLE);
  EXPECT_EQ(runtime, nullptr);
}

TEST_F(DriverPluginContextTest, BundleLoadAndUnload) {
  std::vector<char> hsaco = compileHIPKernel();
  ASSERT_FALSE(hsaco.empty()) << "Failed to compile HIP kernel via HIPRTC";

  std::string fileType("hsaco");
  auto bundle = makeStaticBundle(hsaco, fileType);
  M_driver_bundle_compilation_options opts{"", 0, -1};
  M_driver_runtime_bundle_t runtime = nullptr;
  ASSERT_SUCCESS(bundle_load(contexts[0], &bundle, &opts, &runtime));
  ASSERT_NE(runtime, nullptr);

  EXPECT_SUCCESS(bundle_unload(contexts[0], runtime));
}

TEST_F(DriverPluginContextTest, FunctionLoadFromBundle) {
  std::vector<char> hsaco = compileHIPKernel();
  ASSERT_FALSE(hsaco.empty()) << "Failed to compile HIP kernel via HIPRTC";

  std::string fileType("hsaco");
  auto bundle = makeStaticBundle(hsaco, fileType);
  M_driver_bundle_compilation_options opts{"", 0, -1};
  M_driver_runtime_bundle_t runtime = nullptr;
  ASSERT_SUCCESS(bundle_load(contexts[0], &bundle, &opts, &runtime));
  ASSERT_NE(runtime, nullptr);

  std::string funcName("vec_add");
  M_driver_function_t func = nullptr;
  ASSERT_SUCCESS(function_load(contexts[0], runtime, funcName.c_str(),
                               funcName.size(), &func));
  ASSERT_NE(func, nullptr);

  EXPECT_SUCCESS(function_unload(contexts[0], func));
  EXPECT_SUCCESS(bundle_unload(contexts[0], runtime));
}

TEST_F(DriverPluginContextTest, FunctionOccupancy) {
  std::vector<char> hsaco = compileHIPKernel();
  ASSERT_FALSE(hsaco.empty()) << "Failed to compile HIP kernel via HIPRTC";

  std::string fileType("hsaco");
  auto bundle = makeStaticBundle(hsaco, fileType);
  M_driver_bundle_compilation_options opts{"", 0, -1};
  M_driver_runtime_bundle_t runtime = nullptr;
  ASSERT_SUCCESS(bundle_load(contexts[0], &bundle, &opts, &runtime));

  std::string funcName("vec_add");
  M_driver_function_t func = nullptr;
  ASSERT_SUCCESS(function_load(contexts[0], runtime, funcName.c_str(),
                               funcName.size(), &func));

  int32_t numBlocks = 0;
  ASSERT_SUCCESS(function_occupancy_max_active_blocks(
      contexts[0], func, /*block_size=*/128,
      /*dynamic_shared_memory_bytes=*/0, &numBlocks));
  EXPECT_GT(numBlocks, 0);

  EXPECT_SUCCESS(function_unload(contexts[0], func));
  EXPECT_SUCCESS(bundle_unload(contexts[0], runtime));
}

TEST_F(DriverPluginContextTest, FunctionLoadBadName) {
  std::vector<char> hsaco = compileHIPKernel();
  ASSERT_FALSE(hsaco.empty()) << "Failed to compile HIP kernel via HIPRTC";

  std::string fileType("hsaco");
  auto bundle = makeStaticBundle(hsaco, fileType);
  M_driver_bundle_compilation_options opts{"", 0, -1};
  M_driver_runtime_bundle_t runtime = nullptr;
  ASSERT_SUCCESS(bundle_load(contexts[0], &bundle, &opts, &runtime));
  ASSERT_NE(runtime, nullptr);

  std::string funcName("nonexistent_kernel");
  M_driver_function_t func = nullptr;
  M_driver_status_t status = function_load(
      contexts[0], runtime, funcName.c_str(), funcName.size(), &func);
  EXPECT_NE(status, M_DRIVER_STATUS_SUCCESS);
  EXPECT_EQ(func, nullptr);

  EXPECT_SUCCESS(bundle_unload(contexts[0], runtime));
}

TEST_F(DriverPluginContextTest, FunctionLoadNullBundle) {
  std::string funcName("vec_add");
  M_driver_function_t func = nullptr;
  M_driver_status_t status = function_load(
      contexts[0], nullptr, funcName.c_str(), funcName.size(), &func);
  EXPECT_EQ(status, M_DRIVER_STATUS_BAD_BUNDLE);
  EXPECT_EQ(func, nullptr);
}

TEST_F(DriverPluginContextTest, FunctionLoadNullOut) {
  std::vector<char> hsaco = compileHIPKernel();
  ASSERT_FALSE(hsaco.empty()) << "Failed to compile HIP kernel via HIPRTC";

  std::string fileType("hsaco");
  auto bundle = makeStaticBundle(hsaco, fileType);
  M_driver_bundle_compilation_options opts{"", 0, -1};
  M_driver_runtime_bundle_t runtime = nullptr;
  ASSERT_SUCCESS(bundle_load(contexts[0], &bundle, &opts, &runtime));

  std::string funcName("vec_add");
  EXPECT_EQ(function_load(contexts[0], runtime, funcName.c_str(),
                          funcName.size(), nullptr),
            M_DRIVER_STATUS_INVALID_ARG);

  EXPECT_SUCCESS(bundle_unload(contexts[0], runtime));
}

TEST_F(DriverPluginContextTest, BundleUnloadNullHandle) {
  M_driver_status_t status = bundle_unload(contexts[0], nullptr);
  EXPECT_EQ(status, M_DRIVER_STATUS_BAD_BUNDLE);
}

TEST_F(DriverPluginContextTest, EventCreateDestroyMultiple) {
  for (M_driver_context_t ctx : contexts) {
    constexpr int kNumEvents = 8;
    M_driver_event_t events[kNumEvents];
    for (int i = 0; i < kNumEvents; i++) {
      events[i] = nullptr;
      ASSERT_SUCCESS(event_create(ctx, M_DRIVER_EVENT_FLAG_NONE, &events[i]));
      ASSERT_NE(events[i], nullptr);
    }
    for (int i = 0; i < kNumEvents; i++) {
      EXPECT_SUCCESS(event_destroy(ctx, events[i]));
    }
  }
}

TEST_F(DriverPluginContextTest, EventNullContext) {
  M_driver_event_t event = nullptr;
  ASSERT_EQ(M_DRIVER_STATUS_BAD_CONTEXT,
            event_create(nullptr, M_DRIVER_EVENT_FLAG_NONE, &event));

  ASSERT_EQ(M_DRIVER_STATUS_BAD_CONTEXT, event_destroy(nullptr, event));

  ASSERT_EQ(M_DRIVER_STATUS_BAD_CONTEXT, event_synchronize(nullptr, event));

  bool isReady = false;
  ASSERT_EQ(M_DRIVER_STATUS_BAD_CONTEXT,
            is_event_ready(nullptr, event, &isReady));
}

TEST_F(DriverPluginContextTest, EventDestroyNull) {
  for (M_driver_context_t ctx : contexts) {
    ASSERT_EQ(M_DRIVER_STATUS_BAD_EVENT, event_destroy(ctx, nullptr));
  }
}

TEST_F(DriverPluginContextTest, EventSynchronizeNull) {
  for (M_driver_context_t ctx : contexts) {
    ASSERT_EQ(M_DRIVER_STATUS_BAD_EVENT, event_synchronize(ctx, nullptr));
  }
}

TEST_F(DriverPluginContextTest, EventIsReadyNull) {
  for (M_driver_context_t ctx : contexts) {
    bool isReady = false;
    ASSERT_EQ(M_DRIVER_STATUS_BAD_EVENT,
              is_event_ready(ctx, nullptr, &isReady));
  }
}

// Uses synchronous (context-based) memory allocation.
TEST_F(DriverPluginQueueTest, VecAdd) {
  std::vector<char> hsaco = compileHIPKernel();
  ASSERT_FALSE(hsaco.empty()) << "Failed to compile HIP kernel via HIPRTC";

  std::string fileType("hsaco");
  auto staticBundle = makeStaticBundle(hsaco, fileType);
  M_driver_bundle_compilation_options opts{"", 0, -1};
  M_driver_runtime_bundle_t bundle = nullptr;
  ASSERT_SUCCESS(bundle_load(contexts[0], &staticBundle, &opts, &bundle));
  ASSERT_NE(bundle, nullptr);
  auto bundleCleanup =
      llvm::scope_exit([&]() { bundle_unload(contexts[0], bundle); });

  std::string funcName("vec_add");
  M_driver_function_t func = nullptr;
  ASSERT_SUCCESS(function_load(contexts[0], bundle, funcName.c_str(),
                               funcName.size(), &func));
  auto funcCleanup =
      llvm::scope_exit([&]() { function_unload(contexts[0], func); });

  constexpr size_t length = 1024;
  constexpr size_t bufSize = length * sizeof(float);

  std::vector<float> inHostA(length);
  std::vector<float> inHostB(length);
  std::vector<float> outHost(length, 0.0f);

  for (size_t i = 0; i < length; i++) {
    inHostA[i] = static_cast<float>(i);
    inHostB[i] = 2.0f;
  }

  M_driver_memory_t inDevA = nullptr;
  M_driver_memory_t inDevB = nullptr;
  M_driver_memory_t outDev = nullptr;
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bufSize, &inDevA));
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bufSize, &inDevB));
  ASSERT_SUCCESS(memory_alloc_sync(contexts[0], bufSize, &outDev));
  auto memCleanup = llvm::scope_exit([&]() {
    memory_free_sync(contexts[0], outDev);
    memory_free_sync(contexts[0], inDevB);
    memory_free_sync(contexts[0], inDevA);
  });

  M_driver_memory_view toViewA{inDevA, /*offset=*/0, bufSize};
  ASSERT_SUCCESS(queue_copy_to_device(queues[0], &toViewA, inHostA.data()));
  M_driver_memory_view toViewB{inDevB, /*offset=*/0, bufSize};
  ASSERT_SUCCESS(queue_copy_to_device(queues[0], &toViewB, inHostB.data()));

  void *outPtr = nullptr;
  void *inPtrA = nullptr;
  void *inPtrB = nullptr;
  ASSERT_SUCCESS(
      memory_property(outDev, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &outPtr));
  ASSERT_SUCCESS(
      memory_property(inDevA, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &inPtrA));
  ASSERT_SUCCESS(
      memory_property(inDevB, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &inPtrB));

  void *args[] = {&inPtrA, &inPtrB, &outPtr, (void *)&length};
  uint64_t argSizes[] = {sizeof(inPtrA), sizeof(inPtrB), sizeof(outPtr),
                         sizeof(length)};

  constexpr uint32_t blockDim = 32;
  M_driver_queue_execute_config_t config{};
  config.mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU;
  config.config.gpu = {.grid = {length / blockDim, 1, 1},
                       .block = {blockDim, 1, 1},
                       .shared_mem_bytes = 0,
                       .attributes = nullptr,
                       .num_attributes = 0};
  ASSERT_SUCCESS(queue_execute(queues[0], func, &config, args, argSizes, 4));

  M_driver_memory_view fromView{outDev, /*offset=*/0, bufSize};
  ASSERT_SUCCESS(queue_copy_from_device(queues[0], outHost.data(), &fromView));
  ASSERT_SUCCESS(queue_synchronize(queues[0]));

  for (size_t i = 0; i < length; i++)
    EXPECT_EQ(outHost[i], static_cast<float>(i) + 2.0f);
}
