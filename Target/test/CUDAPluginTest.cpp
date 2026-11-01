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

#include "../lib/CUDA/Error.h"
#include "Fixture.h"
#include "HAL/M_driver_bundle.h"
#include "HAL/M_driver_common.h"
#include "HAL/M_driver_memory.h"

#include "llvm/ADT/ScopeExit.h"

#include <vector>

/// NOTE: assumes `s` is null terminated
M_driver_slice M_driver_slice_from(const char *s) {
  uint64_t size = strlen(s);
  return M_driver_slice{s, size};
}

M_driver_slice M_driver_slice_from(std::string &s) {
  return M_driver_slice{s.data(), s.size()};
}

// Extend test-cases per-plugin beyond the common ones in PluginTest.cpp

static std::string vecAddPtx = R"PTX(
// From max/kernels/test/gpu/vec_add.ptx

.version 6.3
.target sm_75
.address_size 64

	// .globl	vec_add

.visible .entry vec_add(
	.param .b64 vec_add_param_0,
	.param .b64 vec_add_param_1,
	.param .b64 vec_add_param_2,
	.param .b64 vec_add_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .b32 	%r<28>;
	.reg .f32 	%f<4>;
	.reg .b64 	%rd<12>;

	mov.u32 	%r1, %tid.x;
	cvt.s64.s32 	%rd6, %r1;
	ld.param.u64 	%rd8, [vec_add_param_3];
	mov.u32 	%r2, %ntid.x;
	mov.u32 	%r3, %ctaid.x;
	mul.wide.s32 	%rd9, %r2, %r3;
	add.s64 	%rd10, %rd9, %rd6;
	setp.ge.s64 	%p1, %rd10, %rd8;
	@%p1 bra 	$L__BB0_2;
	ld.param.u64 	%rd4, [vec_add_param_0];
	ld.param.u64 	%rd5, [vec_add_param_1];
	ld.param.u64 	%rd7, [vec_add_param_2];
	shl.b64 	%rd11, %rd10, 2;
	add.s64 	%rd1, %rd4, %rd11;
	add.s64 	%rd2, %rd7, %rd11;
	add.s64 	%rd3, %rd5, %rd11;
	ld.u8 	%r4, [%rd3];
	ld.u8 	%r5, [%rd3+1];
	shl.b32 	%r6, %r5, 8;
	or.b32  	%r7, %r6, %r4;
	ld.u8 	%r8, [%rd3+2];
	shl.b32 	%r9, %r8, 16;
	ld.u8 	%r10, [%rd3+3];
	shl.b32 	%r11, %r10, 24;
	or.b32  	%r12, %r11, %r9;
	or.b32  	%r13, %r12, %r7;
	mov.b32 	%f1, %r13;
	ld.u8 	%r14, [%rd2];
	ld.u8 	%r15, [%rd2+1];
	shl.b32 	%r16, %r15, 8;
	or.b32  	%r17, %r16, %r14;
	ld.u8 	%r18, [%rd2+2];
	shl.b32 	%r19, %r18, 16;
	ld.u8 	%r20, [%rd2+3];
	shl.b32 	%r21, %r20, 24;
	or.b32  	%r22, %r21, %r19;
	or.b32  	%r23, %r22, %r17;
	mov.b32 	%f2, %r23;
	add.rn.f32 	%f3, %f1, %f2;
	mov.b32 	%r24, %f3;
	st.u8 	[%rd1], %r24;
	shr.u32 	%r25, %r24, 16;
	st.u8 	[%rd1+2], %r25;
	shr.u32 	%r26, %r24, 24;
	st.u8 	[%rd1+3], %r26;
	shr.u32 	%r27, %r24, 8;
	st.u8 	[%rd1+1], %r27;
$L__BB0_2:
	ret;
}
)PTX";

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

TEST_F(DriverPluginDeviceTest, HostUnifiedPropertyIsImplemented) {
  for (M_driver_device_t device : devices) {
    int64_t unified = -1;
    ASSERT_SUCCESS(device_property(
        device, M_DRIVER_DEVICE_PROPERTY_HOST_UNIFIED, &unified));
    EXPECT_TRUE(unified == 0 || unified == 1) << "got " << unified;
  }
}

TEST_F(DriverPluginContextTest, CurrentContextNonNullAfterSetCurrent) {
  for (M_driver_context_t context : contexts) {
    ASSERT_SUCCESS(context_set_current(context));
    M_driver_native_handle_t active = 0;
    ASSERT_SUCCESS(context_get_current_driver_context(context, &active));
    // After set_current the thread's active CUcontext is this context's
    // primary context, never null.
    EXPECT_NE(active, 0);
  }
}

TEST_F(DriverPluginContextTest, FunctionLoadUnload) {
  std::string fileType("ptx");
  M_driver_static_bundle staticBundle{.mapped_data =
                                          M_driver_slice_from(vecAddPtx),
                                      .file_type = fileType.c_str(),
                                      .file_type_len = fileType.size()};
  M_driver_bundle_compilation_options opts{"", 0, -1};
  M_driver_runtime_bundle_t bundle = nullptr;
  ASSERT_SUCCESS(bundle_load(contexts[0], &staticBundle, &opts, &bundle));
  ASSERT_NE(bundle, nullptr);

  std::string funcName("vec_add");
  M_driver_function_t func = nullptr;
  ASSERT_SUCCESS(function_load(contexts[0], bundle, funcName.c_str(),
                               funcName.size(), &func));
  ASSERT_NE(func, nullptr);
  EXPECT_SUCCESS(function_unload(contexts[0], func));
  EXPECT_SUCCESS(bundle_unload(contexts[0], bundle));
}

TEST_F(DriverPluginContextTest, FunctionLoadNullOut) {
  std::string fileType("ptx");
  M_driver_static_bundle staticBundle{.mapped_data =
                                          M_driver_slice_from(vecAddPtx),
                                      .file_type = fileType.c_str(),
                                      .file_type_len = fileType.size()};
  M_driver_bundle_compilation_options opts{"", 0, -1};
  M_driver_runtime_bundle_t bundle = nullptr;
  ASSERT_SUCCESS(bundle_load(contexts[0], &staticBundle, &opts, &bundle));
  auto bundleCleanup =
      llvm::scope_exit([&]() { bundle_unload(contexts[0], bundle); });

  std::string funcName("vec_add");
  EXPECT_EQ(function_load(contexts[0], bundle, funcName.c_str(),
                          funcName.size(), nullptr),
            M_DRIVER_STATUS_INVALID_ARG);
}

TEST_F(DriverPluginContextTest, FunctionIntrospection) {
  std::string fileType("ptx");
  M_driver_static_bundle staticBundle{.mapped_data =
                                          M_driver_slice_from(vecAddPtx),
                                      .file_type = fileType.c_str(),
                                      .file_type_len = fileType.size()};
  M_driver_bundle_compilation_options opts{"", 0, -1};
  M_driver_runtime_bundle_t bundle = nullptr;
  ASSERT_SUCCESS(bundle_load(contexts[0], &staticBundle, &opts, &bundle));
  auto bundleCleanup =
      llvm::scope_exit([&]() { bundle_unload(contexts[0], bundle); });

  std::string funcName("vec_add");
  M_driver_function_t func = nullptr;
  ASSERT_SUCCESS(function_load(contexts[0], bundle, funcName.c_str(),
                               funcName.size(), &func));
  auto funcCleanup =
      llvm::scope_exit([&]() { function_unload(contexts[0], func); });

  uint64_t nativeHandle = 0;
  ASSERT_SUCCESS(function_property(contexts[0], func,
                                   M_DRIVER_FUNCTION_PROPERTY_NATIVE_HANDLE,
                                   &nativeHandle));
  EXPECT_NE(nativeHandle, 0u);

  ASSERT_SUCCESS(function_set_attribute(
      contexts[0], func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
      64 * 1024));

  int32_t numBlocks = 0;
  ASSERT_SUCCESS(function_occupancy_max_active_blocks(
      contexts[0], func, /*block_size=*/128,
      /*dynamic_shared_memory_bytes=*/0, &numBlocks));
  EXPECT_GT(numBlocks, 0);
}

TEST_F(DriverPluginContextTest, TensorMapEncodeTiled) {
  M_driver_device_t device = nullptr;
  ASSERT_SUCCESS(device_get(driver, 0, &device));
  int32_t ccMajor = 0;
  ASSERT_SUCCESS(device_attribute(
      device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, &ccMajor));
  if (ccMajor < 9)
    GTEST_SKIP() << "TMA requires sm_90+";

  constexpr uint64_t kDim = 64;
  M_driver_memory_t mem = nullptr;
  ASSERT_SUCCESS(
      memory_alloc_sync(contexts[0], kDim * kDim * sizeof(float), &mem));
  void *addr = nullptr;
  ASSERT_SUCCESS(memory_property(mem, M_DRIVER_MEMORY_PROPERTY_ADDRESS, &addr));

  alignas(64) char tensorMap[128] = {};
  uint64_t globalDim[2] = {kDim, kDim};
  uint64_t globalStrides[1] = {kDim * sizeof(float)};
  uint32_t boxDim[2] = {16, 16};
  uint32_t elementStrides[2] = {1, 1};
  ASSERT_SUCCESS(tensor_map_encode_tiled(
      contexts[0], tensorMap, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2,
      reinterpret_cast<uint64_t>(addr), globalDim, globalStrides, boxDim,
      elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  ASSERT_SUCCESS(memory_free_sync(contexts[0], mem));
}
// Uses synchronous (context-based) memory allocation.
TEST_F(DriverPluginQueueTest, VecAdd) {
  std::string fileType("ptx");
  M_driver_static_bundle staticBundle{.mapped_data =
                                          M_driver_slice_from(vecAddPtx),
                                      .file_type = fileType.c_str(),
                                      .file_type_len = fileType.size()};
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

  void *args[] = {&outPtr, &inPtrA, &inPtrB, (void *)&length};
  uint64_t argSizes[] = {sizeof(outPtr), sizeof(inPtrA), sizeof(inPtrB),
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

// Uses asynchronous (queue-based) memory allocation.
TEST_F(DriverPluginQueueTest, VecAddAsync) {
  std::string fileType("ptx");
  M_driver_static_bundle staticBundle{.mapped_data =
                                          M_driver_slice_from(vecAddPtx),
                                      .file_type = fileType.c_str(),
                                      .file_type_len = fileType.size()};
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

  // All operations on a single stream — only sync at the end.
  M_driver_memory_t inDevA = nullptr;
  M_driver_memory_t inDevB = nullptr;
  M_driver_memory_t outDev = nullptr;
  ASSERT_SUCCESS(memory_alloc_async(queues[0], bufSize, &inDevA));
  ASSERT_SUCCESS(memory_alloc_async(queues[0], bufSize, &inDevB));
  ASSERT_SUCCESS(memory_alloc_async(queues[0], bufSize, &outDev));
  auto memCleanup = llvm::scope_exit([&]() {
    memory_free_async(queues[0], outDev);
    memory_free_async(queues[0], inDevB);
    memory_free_async(queues[0], inDevA);
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

  void *args[] = {&outPtr, &inPtrA, &inPtrB, (void *)&length};
  uint64_t argSizes[] = {sizeof(outPtr), sizeof(inPtrA), sizeof(inPtrB),
                         sizeof(length)};

  constexpr uint32_t blockDim = 32;
  M_driver_queue_execute_config_t config{};
  config.mode = M_DRIVER_QUEUE_EXECUTE_MODE_GPU;
  config.config.gpu = {.grid = {static_cast<uint32_t>(length / blockDim), 1, 1},
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
