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

#include "MLRT/lib/Driver/DeviceContext/Metal/MetalDeviceContext.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"

// Suppress warnings from metal-cpp headers
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wdtor-name"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wundef"

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"

#pragma clang diagnostic pop

#include "HostMapping.h"
#include "TestUtils.h"
#include "gtest/gtest.h"

#include <algorithm> // for std::min
#include <atomic>
#include <chrono>
#include <cmath> // for std::ceil, std::sqrt
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

namespace M::Driver {

// Flag to control performance test output
static constexpr bool kPrintPerformanceMetrics =
    false; // Set to false to disable performance output

// Helper struct to hold timing breakdowns
struct TimingBreakdown {
  double dataTransferToDevice;
  double computation;
  double dataTransferFromDevice;
  double total;
};

// Define size ranges for different tests
struct SizeRange {
  const char *name;
  std::vector<size_t> sizes;
  const char *description;
};

static const std::vector<SizeRange> SIZE_RANGES = {
    {"Small",
     {64 * 1024, 256 * 1024, 1024 * 1024}, // 64KB to 1MB
     "Small vectors (64KB - 1MB) - Testing CPU vs GPU overhead"},
    {"Medium",
     {4 * 1024 * 1024, 16 * 1024 * 1024}, // 4MB to 16MB
     "Medium vectors (4MB - 16MB) - Testing typical workloads"},
    {"Large",
     {32 * 1024 * 1024, 64 * 1024 * 1024}, // 32MB to 64MB
     "Large vectors (32MB - 64MB) - Testing memory bandwidth"}};

// Base class for Metal device tests
class MetalDeviceContextTestBase : public ::testing::Test {
protected:
  void SetUp() override {
    ErrorOr<DeviceRef> deviceOr = Device::create(Device::metalAPI);
    if (deviceOr.isError()) {
      GTEST_SKIP() << "Failed to create Metal device: " << deviceOr.getError();
      return;
    }
    device_ = std::move(*deviceOr);

    // Create a context for the device
    ErrorOr<DeviceContextRef> ctx = device_->createContext();
    if (ctx.isError()) {
      GTEST_SKIP() << "Failed to create device context: " << ctx.getError();
      return;
    }
    ctx_ = std::move(*ctx);

    // Metal device context is ready to use
  }

  void TearDown() override {
    ctx_.reset();
    device_.reset();
  }

  // Helper function to get Metal shader source that emits a known marker via
  // os_log (MSL 3.1 / macOS 14+). Compilation is attempted at test time; if
  // os_log is unavailable the load will fail and the test skips.
  std::string getPrintKernelSource() {
    return R"(
      #include <metal_stdlib>
      using namespace metal;

      kernel void print_test_kernel(uint tid [[thread_position_in_grid]])
      {
          if (tid == 0)
              os_log("gpu_print_test_marker\n");
      }
    )";
  }

  // Helper function to get Metal shader source code
  std::string getMetalShaderSource() {
    return R"(
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
    )";
  }

  // Two distinct entry points in a single MSL blob. Used to check that the
  // function cache keys on the entry-point name, not just the blob: loading
  // `fill_one` and `fill_two` from the same source must resolve to distinct
  // pipelines that write their distinct constants (1.0 vs 2.0), not alias.
  std::string getTwoKernelShaderSource() {
    return R"(
      #include <metal_stdlib>
      using namespace metal;

      struct FloatBufPtr { device float* ptr; };

      kernel void fill_one(
          device FloatBufPtr& out_ref [[buffer(0)]],
          constant uint&      length  [[buffer(1)]],
          uint tid [[thread_position_in_grid]])
      {
          device float* out = out_ref.ptr;
          if (tid < length)
              out[tid] = 1.0f;
      }

      kernel void fill_two(
          device FloatBufPtr& out_ref [[buffer(0)]],
          constant uint&      length  [[buffer(1)]],
          uint tid [[thread_position_in_grid]])
      {
          device float* out = out_ref.ptr;
          if (tid < length)
              out[tid] = 2.0f;
      }
    )";
  }

  // CPU-released queue gate: an enqueued wait on a value only release()
  // signals (from the CPU) holds back everything committed after it.
  struct HostGate {
    DeviceEventRef event;
    void release() {
      MTL::SharedEvent *evt = MetalDeviceEvent::cast(*event).sharedEvent();
      if (evt->signaledValue() < 1)
        evt->setSignaledValue(1);
    }
    // A failed ASSERT must not wedge the queue; release() is idempotent.
    ~HostGate() {
      if (event)
        release();
    }
  };
  HostGate makeGate(DeviceContextRef &ctx) {
    NS::SharedPtr<MTL::SharedEvent> evt = NS::TransferPtr(
        MetalDevice::cast(*device_).metalDevice()->newSharedEvent());
    if (!evt) {
      ADD_FAILURE() << "newSharedEvent failed";
      return HostGate{};
    }
    HostGate gate{MetalDeviceEventRef::create(MetalDevice::cast(*device_),
                                              std::move(evt),
                                              /*waitValue=*/1)};
    EXPECT_FALSE(ctx->enqueueWaitFor(*gate.event).isError());
    return gate;
  }

  // Launches fill_one (writes 1.0f to `length` floats). loadFunction only
  // builds pipeline state (no queue work), so this is safe behind a gate.
  void launchFillOne(DeviceContextRef &ctx, DeviceBufferRef &out,
                     uint32_t length) {
    ErrorOr<DeviceFunctionRef> fnOr =
        ctx->loadFunction("", "fill_one", getTwoKernelShaderSource());
    ASSERT_FALSE(fnOr.isError()) << fnOr.getError();
    DeviceFunctionRef fn = fnOr.takeValue();
    void *outPtr = out->ptr();
    void *kernelArgs[] = {&outPtr, &length};
    size_t argSizes[] = {sizeof(outPtr), sizeof(length)};
    bool argIsDevicePtr[] = {true, false};
    DeviceBuffer *kernelBuffers[] = {out.getPointer()};
    MetalEnqueueFunctionArgs metalArgs = {kernelArgs, argSizes, argIsDevicePtr,
                                          kernelBuffers, 1};
    void *args[] = {&metalArgs};
    ASSERT_FALSE(ctx->enqueueFunctionExecDirect(*fn, 1, 1, 1, 64, 1, 1, 0,
                                                nullptr, 0, args, 2, argSizes)
                     .isError());
  }

  DeviceRef device_;
  DeviceContextRef ctx_;
};

// Test class for MSL-specific tests
class MSLDeviceContextTest : public MetalDeviceContextTestBase {};

// Test fixture is needed to enable testing of private methods in DeviceContext.
class MetalDeviceContextTest : public ::testing::Test {
protected:
  void SetUp() override {
    // No common setup needed.
  }
};

TEST_F(MetalDeviceContextTest, Smoke) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::metalAPI));

  std::string deviceName = ctx->device().name();
  EXPECT_NE(deviceName.find("Apple"), std::string::npos) << deviceName;
  std::cerr << "Name of default MTL device: " << deviceName << "\n";

  EXPECT_EQ(ctx->device().api(), Device::metalAPI);
}

TEST_F(MetalDeviceContextTest, NumDevices) {
  int num = Device::numberOfDevices(Device::metalAPI);
  EXPECT_EQ(num, 1);
  std::cerr << "Number of GPU devices: " << num << "\n";
}

TEST_F(MetalDeviceContextTestBase, BinaryMetallibExecution) {
  ASSERT_TRUE(ctx_);

  // Get device info
  const MetalDevice &metalDevice = MetalDevice::cast(*device_);
  MTL::Device *mtlDevice = metalDevice.metalDevice();
  ASSERT_TRUE(mtlDevice) << "Failed to get Metal device";

  // Get Metal shader source code
  std::string metalSource = getMetalShaderSource();
  ASSERT_FALSE(metalSource.empty()) << "Failed to get Metal shader source";

  // Load the function from MSL source code
  ErrorOr<DeviceFunctionRef> libFunctionOr =
      ctx_->loadFunction("vec_add_module", "vec_add_msl", metalSource);
  ASSERT_FALSE(libFunctionOr.isError())
      << "Failed to load Metal function: " << libFunctionOr.getError();
  DeviceFunctionRef libFunction = libFunctionOr.takeValue();

  // Test the function with some data
  const size_t length = 1024;
  const size_t elemSize = sizeof(float);

  // Create device buffers
  ErrorOr<DeviceBufferRef> in0DeviceOr =
      ctx_->enqueueCreateBuffer(length * elemSize);
  ASSERT_FALSE(in0DeviceOr.isError()) << "Failed to create in0Device buffer";
  DeviceBufferRef in0Device = in0DeviceOr.takeValue();

  ErrorOr<DeviceBufferRef> in1DeviceOr =
      ctx_->enqueueCreateBuffer(length * elemSize);
  ASSERT_FALSE(in1DeviceOr.isError()) << "Failed to create in1Device buffer";
  DeviceBufferRef in1Device = in1DeviceOr.takeValue();

  ErrorOr<DeviceBufferRef> outDeviceOr =
      ctx_->enqueueCreateBuffer(length * elemSize);
  ASSERT_FALSE(outDeviceOr.isError()) << "Failed to create outDevice buffer";
  DeviceBufferRef outDevice = outDeviceOr.takeValue();

  // Initialize host buffers
  float *in0Host = new float[length];
  float *in1Host = new float[length];
  float *outHost = new float[length];

  for (size_t i = 0; i < length; ++i) {
    in0Host[i] = static_cast<float>(i);
    in1Host[i] = 2.0f;
    outHost[i] = 0.0f;
  }

  // Copy data to device
  EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*in0Device, in0Host));
  EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*in1Device, in1Host));
  EXPECT_NO_ERR(ctx_->synchronize());

  void *outPtr = outDevice->ptr();
  void *in0Ptr = in0Device->ptr();
  void *in1Ptr = in1Device->ptr();
  uint32_t lengthValue = static_cast<uint32_t>(length);
  void *kernelArgs[] = {&outPtr, &in0Ptr, &in1Ptr, &lengthValue};
  size_t argSizes[] = {sizeof(outPtr), sizeof(in0Ptr), sizeof(in1Ptr),
                       sizeof(lengthValue)};
  bool argIsDevicePtr[] = {true, true, true, false};
  DeviceBuffer *kernelBuffers[] = {
      outDevice.getPointer(), in0Device.getPointer(), in1Device.getPointer()};
  MetalEnqueueFunctionArgs metalArgs = {
      kernelArgs, argSizes, argIsDevicePtr, kernelBuffers,
      sizeof(kernelBuffers) / sizeof(DeviceBuffer *)};
  void *args[] = {&metalArgs};

  // Execute kernel
  constexpr uint32_t blockDim = 256;
  EXPECT_NO_ERR(ctx_->enqueueFunctionExecDirect(
      *libFunction, length / blockDim, 1, 1, blockDim, 1, 1, 0, nullptr, 0,
      args, sizeof(kernelArgs) / sizeof(void *), argSizes));

  // Copy results back
  EXPECT_NO_ERR(ctx_->enqueueCopyFromDevice(outHost, *outDevice));
  EXPECT_NO_ERR(ctx_->synchronize());

  // Verify results
  for (size_t i = 0; i < std::min(size_t(5), length); ++i) {
    float expected = static_cast<float>(i) + 2.0f;
    EXPECT_FLOAT_EQ(outHost[i], expected)
        << "Mismatch at index " << i << ": expected " << expected << ", got "
        << outHost[i];
  }

  // Cleanup
  delete[] in0Host;
  delete[] in1Host;
  delete[] outHost;
}

// Loading the same (module, code, entry point) twice must return the same
// cached compute pipeline state rather than recompiling the shader and creating
// a fresh pipeline on every launch. Metal previously had no function cache (the
// CUDA/HIP backends do), so each load rebuilt the pipeline on the critical
// path. The pipeline-state pointer is the observable cache-hit signal: on a hit
// both `MetalDeviceFunction`s wrap the one cached pipeline, so the pointers are
// equal; without the cache they differ.
TEST_F(MetalDeviceContextTestBase, LoadFunctionCachesPipelineState) {
  ASSERT_TRUE(ctx_);

  std::string src = getMetalShaderSource();

  ErrorOr<DeviceFunctionRef> firstOr =
      ctx_->loadFunction("vec_add_module", "vec_add_msl", src);
  ASSERT_FALSE(firstOr.isError())
      << "First load failed: " << firstOr.getError();
  DeviceFunctionRef first = firstOr.takeValue();

  ErrorOr<DeviceFunctionRef> secondOr =
      ctx_->loadFunction("vec_add_module", "vec_add_msl", src);
  ASSERT_FALSE(secondOr.isError())
      << "Second load failed: " << secondOr.getError();
  DeviceFunctionRef second = secondOr.takeValue();

  MTL::ComputePipelineState *firstPipeline =
      MetalDeviceFunction::cast(*first).getPipelineState();
  MTL::ComputePipelineState *secondPipeline =
      MetalDeviceFunction::cast(*second).getPipelineState();
  ASSERT_NE(firstPipeline, nullptr);
  EXPECT_EQ(firstPipeline, secondPipeline)
      << "Second load did not reuse the cached pipeline state";
}

// Correctness guard for the cache key: loading two distinct entry points from
// the same MSL blob must return two distinct functions, not alias the first.
// The empty module name forces the code-hash keying path (see
// FunctionCacheKey::moduleCacheName), where a key that omitted the entry-point
// name would collide and silently return `fill_one` for both. Each kernel
// writes a distinct constant (1.0 vs 2.0) so a mis-keyed cache is observable as
// a wrong result, not just a wrong handle. Mirrors the CUDA backend's
// LoadFunctionDistinctEntryPointsSameBlob.
TEST_F(MetalDeviceContextTestBase, LoadFunctionDistinctEntryPointsSameBlob) {
  ASSERT_TRUE(ctx_);

  std::string src = getTwoKernelShaderSource();

  ErrorOr<DeviceFunctionRef> funcOneOr =
      ctx_->loadFunction(/*moduleName=*/"", "fill_one", src);
  ASSERT_FALSE(funcOneOr.isError())
      << "Failed to load fill_one: " << funcOneOr.getError();
  DeviceFunctionRef funcOne = funcOneOr.takeValue();

  ErrorOr<DeviceFunctionRef> funcTwoOr =
      ctx_->loadFunction(/*moduleName=*/"", "fill_two", src);
  ASSERT_FALSE(funcTwoOr.isError())
      << "Failed to load fill_two: " << funcTwoOr.getError();
  DeviceFunctionRef funcTwo = funcTwoOr.takeValue();

  // Pre-fix, funcTwo would alias funcOne (same blob key) and report "fill_one".
  EXPECT_EQ(funcOne->name(), "fill_one");
  EXPECT_EQ(funcTwo->name(), "fill_two");
  EXPECT_NE(MetalDeviceFunction::cast(*funcOne).getPipelineState(),
            MetalDeviceFunction::cast(*funcTwo).getPipelineState());

  const size_t length = 256;
  const size_t elemSize = sizeof(float);
  const uint32_t blockDim = 64;
  const uint32_t gridSize =
      static_cast<uint32_t>((length + blockDim - 1) / blockDim);

  auto runAndCheck = [&](DeviceFunctionRef &fn, float expected) {
    ErrorOr<DeviceBufferRef> bufOr =
        ctx_->enqueueCreateBuffer(length * elemSize);
    ASSERT_FALSE(bufOr.isError()) << "Failed to create buffer";
    DeviceBufferRef buf = bufOr.takeValue();

    std::vector<float> host(length, -1.0f);
    EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*buf, host.data()));
    EXPECT_NO_ERR(ctx_->synchronize());

    void *bufPtr = buf->ptr();
    uint32_t lengthValue = static_cast<uint32_t>(length);
    void *kernelArgs[] = {&bufPtr, &lengthValue};
    size_t argSizes[] = {sizeof(bufPtr), sizeof(lengthValue)};
    bool argIsDevicePtr[] = {true, false};
    DeviceBuffer *kernelBuffers[] = {buf.getPointer()};
    MetalEnqueueFunctionArgs metalArgs = {kernelArgs, argSizes, argIsDevicePtr,
                                          kernelBuffers, 1};
    void *args[] = {&metalArgs};

    EXPECT_NO_ERR(ctx_->enqueueFunctionExecDirect(
        *fn, gridSize, 1, 1, blockDim, 1, 1, 0, nullptr, 0, args,
        sizeof(kernelArgs) / sizeof(void *), argSizes));

    EXPECT_NO_ERR(ctx_->enqueueCopyFromDevice(host.data(), *buf));
    EXPECT_NO_ERR(ctx_->synchronize());

    for (size_t i = 0; i < std::min(size_t(5), length); ++i)
      EXPECT_FLOAT_EQ(host[i], expected)
          << "Mismatch at index " << i << ": expected " << expected << ", got "
          << host[i];
  };

  runAndCheck(funcOne, 1.0f);
  runAndCheck(funcTwo, 2.0f);
}

TEST_F(MetalDeviceContextTestBase, DriverVersion) {
  ASSERT_TRUE(device_);
  auto driverVersion = device_->getDriverVersion();
  ASSERT_FALSE(driverVersion.isError());
  std::cout << "Metal Driver Version: " << driverVersion->toString() << "\n";
  ASSERT_GT(driverVersion->major(), 0);
  ASSERT_GE(driverVersion->minor(), 0);
  ASSERT_GE(driverVersion->patch(), 0);
}

TEST_F(MetalDeviceContextTestBase, BinaryMetallibExecutionMultipleSizes) {
  ASSERT_TRUE(ctx_);

  // Get device info
  const MetalDevice &metalDevice = MetalDevice::cast(*device_);
  MTL::Device *mtlDevice = metalDevice.metalDevice();
  ASSERT_TRUE(mtlDevice) << "Failed to get Metal device";

  // Get Metal shader source code
  std::string metalSource = getMetalShaderSource();
  ASSERT_FALSE(metalSource.empty()) << "Failed to get Metal shader source";

  // Load the function from MSL source code
  ErrorOr<DeviceFunctionRef> libFunctionOr =
      ctx_->loadFunction("vec_add_module", "vec_add_msl", metalSource);
  ASSERT_FALSE(libFunctionOr.isError())
      << "Failed to load Metal function: " << libFunctionOr.getError();
  DeviceFunctionRef libFunction = libFunctionOr.takeValue();

  // Test different vector sizes
  std::vector<size_t> testSizes = {256, 1024, 4096, 16384, 65536};
  const uint32_t blockDim = 256;

  for (size_t length : testSizes) {
    if (kPrintPerformanceMetrics) {
      std::cout << "Testing vector size: " << length << std::endl;
    }

    const size_t elemSize = sizeof(float);

    // Create device buffers
    ErrorOr<DeviceBufferRef> in0DeviceOr =
        ctx_->enqueueCreateBuffer(length * elemSize);
    ASSERT_FALSE(in0DeviceOr.isError())
        << "Failed to create in0Device buffer for size " << length;
    DeviceBufferRef in0Device = in0DeviceOr.takeValue();

    ErrorOr<DeviceBufferRef> in1DeviceOr =
        ctx_->enqueueCreateBuffer(length * elemSize);
    ASSERT_FALSE(in1DeviceOr.isError())
        << "Failed to create in1Device buffer for size " << length;
    DeviceBufferRef in1Device = in1DeviceOr.takeValue();

    ErrorOr<DeviceBufferRef> outDeviceOr =
        ctx_->enqueueCreateBuffer(length * elemSize);
    ASSERT_FALSE(outDeviceOr.isError())
        << "Failed to create outDevice buffer for size " << length;
    DeviceBufferRef outDevice = outDeviceOr.takeValue();

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
    EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*in0Device, in0Host.data()));
    EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*in1Device, in1Host.data()));
    EXPECT_NO_ERR(ctx_->synchronize());

    void *outPtr = outDevice->ptr();
    void *in0Ptr = in0Device->ptr();
    void *in1Ptr = in1Device->ptr();
    uint32_t lengthValue = static_cast<uint32_t>(length);
    void *kernelArgs[] = {&outPtr, &in0Ptr, &in1Ptr, &lengthValue};
    size_t argSizes[] = {sizeof(outPtr), sizeof(in0Ptr), sizeof(in1Ptr),
                         sizeof(lengthValue)};
    bool argIsDevicePtr[] = {true, true, true, false};
    DeviceBuffer *kernelBuffers[] = {
        outDevice.getPointer(), in0Device.getPointer(), in1Device.getPointer()};
    MetalEnqueueFunctionArgs metalArgs = {
        kernelArgs, argSizes, argIsDevicePtr, kernelBuffers,
        sizeof(kernelBuffers) / sizeof(DeviceBuffer *)};
    void *args[] = {&metalArgs};

    // Execute kernel
    uint32_t gridSize =
        static_cast<uint32_t>((length + blockDim - 1) / blockDim);
    EXPECT_NO_ERR(ctx_->enqueueFunctionExecDirect(
        *libFunction, gridSize, 1, 1, blockDim, 1, 1, 0, nullptr, 0, args,
        sizeof(kernelArgs) / sizeof(void *), argSizes));
    EXPECT_NO_ERR(ctx_->synchronize());

    // Copy results back
    EXPECT_NO_ERR(ctx_->enqueueCopyFromDevice(outHost.data(), *outDevice));
    EXPECT_NO_ERR(ctx_->synchronize());

    // Verify results for a sample of elements
    // check all elements
    for (size_t i = 0; i < length; ++i) {
      float expected = static_cast<float>(i) + 2.0f;
      EXPECT_FLOAT_EQ(outHost[i], expected)
          << "Mismatch at index " << i << " for size " << length
          << ": expected " << expected << ", got " << outHost[i];
    }

    if (kPrintPerformanceMetrics) {
      std::cout << "✓ Vector size " << length << " passed" << std::endl;
    }
  }
}

TEST_F(MetalDeviceContextTestBase, BinaryMetallibExecutionPerformance) {
  ASSERT_TRUE(ctx_);

  // Get device info
  const MetalDevice &metalDevice = MetalDevice::cast(*device_);
  MTL::Device *mtlDevice = metalDevice.metalDevice();
  ASSERT_TRUE(mtlDevice) << "Failed to get Metal device";

  // Get Metal shader source code
  std::string metalSource = getMetalShaderSource();
  ASSERT_FALSE(metalSource.empty()) << "Failed to get Metal shader source";

  // Load the function from MSL source code
  ErrorOr<DeviceFunctionRef> libFunctionOr =
      ctx_->loadFunction("vec_add_module", "vec_add_msl", metalSource);
  ASSERT_FALSE(libFunctionOr.isError())
      << "Failed to load Metal function: " << libFunctionOr.getError();
  DeviceFunctionRef libFunction = libFunctionOr.takeValue();

  // Performance test with larger vectors
  std::vector<size_t> performanceSizes = {1024, 16384, 65536, 1048576};
  const uint32_t blockDim = 256;
  const int numRuns = 5; // Run multiple times for better average

  if (kPrintPerformanceMetrics) {
    std::cout << "\n=== Metal GPU Vector Addition Performance ===\n";
    std::cout << std::setw(12) << "Size" << " | " << std::setw(15)
              << "Avg Time (ms)"
              << " | " << std::setw(15) << "Throughput (GB/s)" << "\n";
    std::cout << std::string(50, '-') << "\n";
  }

  for (size_t length : performanceSizes) {
    const size_t elemSize = sizeof(float);
    const size_t totalBytes =
        length * elemSize * 3; // 3 buffers (2 input, 1 output)

    double totalExecutionTime = 0.0;

    for (int run = 0; run < numRuns; ++run) {
      // Create device buffers
      ErrorOr<DeviceBufferRef> in0DeviceOr =
          ctx_->enqueueCreateBuffer(length * elemSize);
      ASSERT_FALSE(in0DeviceOr.isError())
          << "Failed to create in0Device buffer";
      DeviceBufferRef in0Device = in0DeviceOr.takeValue();

      ErrorOr<DeviceBufferRef> in1DeviceOr =
          ctx_->enqueueCreateBuffer(length * elemSize);
      ASSERT_FALSE(in1DeviceOr.isError())
          << "Failed to create in1Device buffer";
      DeviceBufferRef in1Device = in1DeviceOr.takeValue();

      ErrorOr<DeviceBufferRef> outDeviceOr =
          ctx_->enqueueCreateBuffer(length * elemSize);
      ASSERT_FALSE(outDeviceOr.isError())
          << "Failed to create outDevice buffer";
      DeviceBufferRef outDevice = outDeviceOr.takeValue();

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
      EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*in0Device, in0Host.data()));
      EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*in1Device, in1Host.data()));
      EXPECT_NO_ERR(ctx_->synchronize());

      void *outPtr = outDevice->ptr();
      void *in0Ptr = in0Device->ptr();
      void *in1Ptr = in1Device->ptr();
      uint32_t lengthValue = static_cast<uint32_t>(length);
      void *kernelArgs[] = {&outPtr, &in0Ptr, &in1Ptr, &lengthValue};
      size_t argSizes[] = {sizeof(outPtr), sizeof(in0Ptr), sizeof(in1Ptr),
                           sizeof(lengthValue)};
      bool argIsDevicePtr[] = {true, true, true, false};
      DeviceBuffer *kernelBuffers[] = {outDevice.getPointer(),
                                       in0Device.getPointer(),
                                       in1Device.getPointer()};
      MetalEnqueueFunctionArgs metalArgs = {
          kernelArgs, argSizes, argIsDevicePtr, kernelBuffers,
          sizeof(kernelBuffers) / sizeof(DeviceBuffer *)};
      void *args[] = {&metalArgs};

      // Time the kernel execution only
      std::chrono::high_resolution_clock::time_point startTime =
          std::chrono::high_resolution_clock::now();

      uint32_t gridSize =
          static_cast<uint32_t>((length + blockDim - 1) / blockDim);
      EXPECT_NO_ERR(ctx_->enqueueFunctionExecDirect(
          *libFunction, gridSize, 1, 1, blockDim, 1, 1, 0, nullptr, 0, args,
          sizeof(kernelArgs) / sizeof(void *), argSizes));
      EXPECT_NO_ERR(ctx_->synchronize());

      std::chrono::high_resolution_clock::time_point endTime =
          std::chrono::high_resolution_clock::now();

      // Copy results back (not timed)
      EXPECT_NO_ERR(ctx_->enqueueCopyFromDevice(outHost.data(), *outDevice));
      EXPECT_NO_ERR(ctx_->synchronize());

      // Verify correctness for first run
      if (run == 0) {
        // check all elements
        for (size_t i = 0; i < length; ++i) {
          float expected = static_cast<float>(i) + 2.0f;
          EXPECT_FLOAT_EQ(outHost[i], expected)
              << "Mismatch at index " << i << " for size " << length;
        }
      }

      std::chrono::microseconds duration =
          std::chrono::duration_cast<std::chrono::microseconds>(endTime -
                                                                startTime);
      totalExecutionTime +=
          duration.count() / 1000.0; // Convert to milliseconds
    }

    double avgTime = totalExecutionTime / numRuns;
    double throughput = (totalBytes / 1e9) / (avgTime / 1000.0); // GB/s

    if (kPrintPerformanceMetrics) {
      std::cout << std::setw(12) << length << " | " << std::setw(15)
                << std::fixed << std::setprecision(3) << avgTime << " | "
                << std::setw(15) << std::setprecision(2) << throughput << "\n";
    }
  }

  if (kPrintPerformanceMetrics) {
    std::cout << std::string(50, '-') << "\n\n";
  }
}

TEST_F(MetalDeviceContextTestBase, BinaryMetallibExecutionCPUComparison) {
  ASSERT_TRUE(ctx_);

  // Get device info
  const MetalDevice &metalDevice = MetalDevice::cast(*device_);
  MTL::Device *mtlDevice = metalDevice.metalDevice();
  ASSERT_TRUE(mtlDevice) << "Failed to get Metal device";

  // Get Metal shader source code
  std::string metalSource = getMetalShaderSource();
  ASSERT_FALSE(metalSource.empty()) << "Failed to get Metal shader source";

  // Load the function from MSL source code
  ErrorOr<DeviceFunctionRef> libFunctionOr =
      ctx_->loadFunction("vec_add_module", "vec_add_msl", metalSource);
  ASSERT_FALSE(libFunctionOr.isError())
      << "Failed to load Metal function: " << libFunctionOr.getError();
  DeviceFunctionRef libFunction = libFunctionOr.takeValue();

  // Comparison test with different vector sizes
  std::vector<size_t> comparisonSizes = {1024, 16384, 65536, 1048576};
  const uint32_t blockDim = 256;
  const int numRuns = 10; // More runs for better averaging

  if (kPrintPerformanceMetrics) {
    std::cout << "\n=== CPU vs Metal GPU Vector Addition Performance "
                 "Comparison ===\n";
    std::cout << std::setw(12) << "Size" << " | " << std::setw(12) << "CPU (ms)"
              << " | " << std::setw(12) << "GPU (ms)" << " | " << std::setw(12)
              << "Speedup"
              << " | " << std::setw(15) << "CPU (GB/s)" << " | "
              << std::setw(15) << "GPU (GB/s)" << "\n";
    std::cout << std::string(90, '-') << "\n";
  }

  for (size_t length : comparisonSizes) {
    const size_t elemSize = sizeof(float);
    const size_t totalBytes =
        length * elemSize * 3; // 3 buffers (2 input, 1 output)

    // Prepare host data (shared between CPU and GPU tests)
    std::vector<float> in0Host(length);
    std::vector<float> in1Host(length);
    std::vector<float> cpuResult(length);
    std::vector<float> gpuResult(length);

    for (size_t i = 0; i < length; ++i) {
      in0Host[i] = static_cast<float>(i);
      in1Host[i] = 2.0f;
      cpuResult[i] = 0.0f;
      gpuResult[i] = 0.0f;
    }

    // === CPU BENCHMARK ===
    double totalCpuTime = 0.0;
    for (int run = 0; run < numRuns; ++run) {
      std::chrono::high_resolution_clock::time_point startTime =
          std::chrono::high_resolution_clock::now();

      // Simple CPU vector addition
      for (size_t i = 0; i < length; ++i) {
        cpuResult[i] = in0Host[i] + in1Host[i];
      }

      std::chrono::high_resolution_clock::time_point endTime =
          std::chrono::high_resolution_clock::now();
      std::chrono::microseconds duration =
          std::chrono::duration_cast<std::chrono::microseconds>(endTime -
                                                                startTime);
      totalCpuTime += duration.count() / 1000.0; // Convert to milliseconds
    }
    double avgCpuTime = totalCpuTime / numRuns;

    // === GPU BENCHMARK ===
    double totalGpuTime = 0.0;
    for (int run = 0; run < numRuns; ++run) {
      // Create device buffers
      ErrorOr<DeviceBufferRef> in0DeviceOr =
          ctx_->enqueueCreateBuffer(length * elemSize);
      ASSERT_FALSE(in0DeviceOr.isError())
          << "Failed to create in0Device buffer";
      DeviceBufferRef in0Device = in0DeviceOr.takeValue();

      ErrorOr<DeviceBufferRef> in1DeviceOr =
          ctx_->enqueueCreateBuffer(length * elemSize);
      ASSERT_FALSE(in1DeviceOr.isError())
          << "Failed to create in1Device buffer";
      DeviceBufferRef in1Device = in1DeviceOr.takeValue();

      ErrorOr<DeviceBufferRef> outDeviceOr =
          ctx_->enqueueCreateBuffer(length * elemSize);
      ASSERT_FALSE(outDeviceOr.isError())
          << "Failed to create outDevice buffer";
      DeviceBufferRef outDevice = outDeviceOr.takeValue();

      // Copy data to device
      EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*in0Device, in0Host.data()));
      EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*in1Device, in1Host.data()));
      EXPECT_NO_ERR(ctx_->synchronize());

      // Mimic how Mojo passes pointers
      void *outPtr = outDevice->ptr();
      void *in0Ptr = in0Device->ptr();
      void *in1Ptr = in1Device->ptr();
      uint32_t lengthValue = static_cast<uint32_t>(length);
      void *kernelArgs[] = {&outPtr, &in0Ptr, &in1Ptr, &lengthValue};
      size_t argSizes[] = {sizeof(outPtr), sizeof(in0Ptr), sizeof(in1Ptr),
                           sizeof(lengthValue)};
      bool argIsDevicePtr[] = {true, true, true, false};
      DeviceBuffer *kernelBuffers[] = {outDevice.getPointer(),
                                       in0Device.getPointer(),
                                       in1Device.getPointer()};
      MetalEnqueueFunctionArgs metalArgs = {
          kernelArgs, argSizes, argIsDevicePtr, kernelBuffers,
          sizeof(kernelBuffers) / sizeof(DeviceBuffer *)};
      void *args[] = {&metalArgs};

      // Time the kernel execution only
      std::chrono::high_resolution_clock::time_point startTime =
          std::chrono::high_resolution_clock::now();

      uint32_t gridSize =
          static_cast<uint32_t>((length + blockDim - 1) / blockDim);
      EXPECT_NO_ERR(ctx_->enqueueFunctionExecDirect(
          *libFunction, gridSize, 1, 1, blockDim, 1, 1, 0, nullptr, 0, args,
          sizeof(kernelArgs) / sizeof(void *), argSizes));
      EXPECT_NO_ERR(ctx_->synchronize());

      std::chrono::high_resolution_clock::time_point endTime =
          std::chrono::high_resolution_clock::now();

      // Copy results back (for the last run only, to verify correctness)
      if (run == numRuns - 1) {
        EXPECT_NO_ERR(
            ctx_->enqueueCopyFromDevice(gpuResult.data(), *outDevice));
        EXPECT_NO_ERR(ctx_->synchronize());
      }

      std::chrono::microseconds duration =
          std::chrono::duration_cast<std::chrono::microseconds>(endTime -
                                                                startTime);
      totalGpuTime += duration.count() / 1000.0; // Convert to milliseconds
    }
    double avgGpuTime = totalGpuTime / numRuns;

    // === CORRECTNESS VERIFICATION ===
    // Verify both CPU and GPU produce the same results
    // check all elements
    for (size_t i = 0; i < length; ++i) {
      float expected = static_cast<float>(i) + 2.0f;
      EXPECT_FLOAT_EQ(cpuResult[i], expected)
          << "CPU mismatch at index " << i << " for size " << length;
      EXPECT_FLOAT_EQ(gpuResult[i], expected)
          << "GPU mismatch at index " << i << " for size " << length;
      EXPECT_FLOAT_EQ(cpuResult[i], gpuResult[i])
          << "CPU/GPU mismatch at index " << i << " for size " << length;
    }

    // === PERFORMANCE CALCULATIONS ===
    double speedup = avgCpuTime / avgGpuTime;
    double cpuThroughput = (totalBytes / 1e9) / (avgCpuTime / 1000.0); // GB/s
    double gpuThroughput = (totalBytes / 1e9) / (avgGpuTime / 1000.0); // GB/s

    if (kPrintPerformanceMetrics) {
      std::cout << std::setw(12) << length << " | " << std::setw(12)
                << std::fixed << std::setprecision(3) << avgCpuTime << " | "
                << std::setw(12) << std::fixed << std::setprecision(3)
                << avgGpuTime << " | " << std::setw(12) << std::fixed
                << std::setprecision(2) << speedup << "x"
                << " | " << std::setw(15) << std::fixed << std::setprecision(2)
                << cpuThroughput << " | " << std::setw(15) << std::fixed
                << std::setprecision(2) << gpuThroughput << "\n";
    }
  }

  if (kPrintPerformanceMetrics) {
    std::cout << std::string(90, '-') << "\n";
    std::cout << "Note: GPU timing includes only kernel execution, not memory "
                 "transfers.\n";
    std::cout << "      CPU uses single-threaded implementation for fair "
                 "comparison.\n\n";
  }
}
// Returns true if MTL::LogState can be created on the current platform
// (requires macOS 14+). Used to skip print tests on older systems.
static bool isLogStateSupported(MTL::Device *mtlDevice) {
  NS::SharedPtr<MTL::LogStateDescriptor> desc =
      NS::TransferPtr(MTL::LogStateDescriptor::alloc()->init());
  NS::Error *error = nullptr;
  NS::SharedPtr<MTL::LogState> ls =
      NS::TransferPtr(mtlDevice->newLogState(desc.get(), &error));
  return !error && ls;
}

// Test: GPU print output is captured on stdout when logging is enabled (the
// default). The kernel calls os_log() with a known marker string; after
// synchronisation the string must appear in stdout because the log handler
// installed by getOrCreateLogState() writes there.
TEST_F(MetalDeviceContextTestBase, MetalPrint_CapturesOutput) {
  const MetalDevice &metalDevice = MetalDevice::cast(*device_);
  if (!isLogStateSupported(metalDevice.metalDevice()))
    GTEST_SKIP() << "MTL::LogState not supported on this platform";

  ErrorOr<DeviceFunctionRef> fnOr = ctx_->loadFunction(
      "print_test_module", "print_test_kernel", getPrintKernelSource());
  if (fnOr.isError())
    GTEST_SKIP() << "os_log not available in MSL on this platform: "
                 << fnOr.getError();
  DeviceFunctionRef fn = fnOr.takeValue();

  MetalEnqueueFunctionArgs metalArgs = {nullptr, nullptr, nullptr, nullptr, 0};
  void *args[] = {&metalArgs};
  testing::internal::CaptureStdout();
  EXPECT_NO_ERR(ctx_->enqueueFunctionExecDirect(
      *fn, 1, 1, 1, 1, 1, 1, /*sharedMemBytes=*/0,
      /*attrs=*/nullptr, /*numAttrs=*/0, args));
  EXPECT_NO_ERR(ctx_->synchronize());
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find("gpu_print_test_marker"), std::string::npos)
      << "Expected GPU print marker in stdout; got: " << output;
}

// Test: METAL_DISABLE_LOGGING attribute prevents the log state from being
// attached to the command buffer, so no GPU print output reaches stdout.
TEST_F(MetalDeviceContextTestBase, MetalPrint_DisabledByAttribute) {
  const MetalDevice &metalDevice = MetalDevice::cast(*device_);
  if (!isLogStateSupported(metalDevice.metalDevice()))
    GTEST_SKIP() << "MTL::LogState not supported on this platform";

  ErrorOr<DeviceFunctionRef> fnOr =
      ctx_->loadFunction("print_test_module_disabled", "print_test_kernel",
                         getPrintKernelSource());
  if (fnOr.isError())
    GTEST_SKIP() << "os_log not available in MSL on this platform: "
                 << fnOr.getError();
  DeviceFunctionRef fn = fnOr.takeValue();

  Device::Attribute attrs[] = {Device::Attribute::METAL_DISABLE_LOGGING};
  MetalEnqueueFunctionArgs metalArgs = {nullptr, nullptr, nullptr, nullptr, 0};
  void *args[] = {&metalArgs};
  testing::internal::CaptureStdout();
  EXPECT_NO_ERR(ctx_->enqueueFunctionExecDirect(
      *fn, 1, 1, 1, 1, 1, 1, /*sharedMemBytes=*/0, static_cast<void *>(attrs),
      /*numAttrs=*/1, args));
  EXPECT_NO_ERR(ctx_->synchronize());
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(output.find("gpu_print_test_marker"), std::string::npos)
      << "Expected no GPU print output with logging disabled; got: " << output;
}

// Regression test: memAlloc must donate newBuffer's +1 directly to the
// tracker. A previous version added a spurious buf->retain(), leaving the
// MTL::Buffer at refcount 2; memFree's release() then leaked it.
TEST_F(MetalDeviceContextTestBase, MemAllocDonatesRefcountToTracker) {
  ASSERT_TRUE(ctx_);
  const MetalDevice &metalDev = MetalDevice::cast(*device_);
  const size_t baseline = metalDev.getTrackedBuffers().size();

  void *ptr = nullptr;
  EXPECT_NO_ERR(
      ctx_->memAlloc(&ptr, /*byteSize=*/4096, Device::AllocMode::HOST));
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(metalDev.getTrackedBuffers().size(), baseline + 1);

  MTL::Buffer *buf = metalDev.getTrackedBuffer(ptr);
  ASSERT_NE(buf, nullptr);
  EXPECT_EQ(buf->retainCount(), 1u);

  EXPECT_NO_ERR(ctx_->memFree(ptr, Device::FreeKind::HOST));
  EXPECT_EQ(metalDev.getTrackedBuffers().size(), baseline);
}

// Rounding out to a page boundary could alias memory that is not mapped.
TEST_F(MetalDeviceContextTestBase, WrapHostMemoryRejectsBadRanges) {
  ASSERT_TRUE(ctx_);
  HostMapping mapping(2 * hostPageSize());
  ASSERT_NE(mapping.data(), nullptr);

  EXPECT_ERR_MSG(ctx_->wrapHostMemory(nullptr, hostPageSize()),
                 "pointer is null");
  EXPECT_ERR_MSG(ctx_->wrapHostMemory(mapping.data(), 0), "byteSize is zero");
  EXPECT_ERR_MSG(ctx_->wrapHostMemory(mapping.bytes() + 64, hostPageSize()),
                 "not page-aligned");
  EXPECT_ERR_MSG(ctx_->wrapHostMemory(mapping.data(), hostPageSize() + 64),
                 "not a multiple of the page size");
}

// Once the handle drops the Metal buffer is gone, but the pages are not.
TEST_F(MetalDeviceContextTestBase, WrapHostMemoryLeavesPagesToTheCaller) {
  ASSERT_TRUE(ctx_);
  const MetalDevice &metalDev = MetalDevice::cast(*device_);
  const size_t baseline = metalDev.getTrackedBuffers().size();

  HostMapping mapping(4 * hostPageSize());
  ASSERT_NE(mapping.data(), nullptr);
  std::memset(mapping.data(), 0xA5, mapping.byteSize());

  void *wrappedDevPtr = nullptr;
  {
    ASSIGN_NO_ERR(DeviceBufferRef, wrapped,
                  ctx_->wrapHostMemory(mapping.data(), mapping.byteSize()));
    EXPECT_EQ(wrapped->byteSize(), mapping.byteSize());
    EXPECT_TRUE(wrapped->isHost());
    // The aliasing buffer's GPU address, which maps back to the caller's pages.
    EXPECT_NE(wrapped->ptr(), mapping.data());
    ASSIGN_NO_ERR(void *, host, ctx_->getHostAccessiblePointer(wrapped->ptr()));
    EXPECT_EQ(host, mapping.data());
    wrappedDevPtr = wrapped->ptr();
    EXPECT_EQ(metalDev.getTrackedBuffers().size(), baseline + 1);
  }

  // `synchronize` alone must drain the release, as it does on CUDA.
  EXPECT_NO_ERR(ctx_->synchronize());
  EXPECT_EQ(metalDev.getTrackedBuffers().size(), baseline);

  const uint8_t *bytes = mapping.bytes();
  EXPECT_TRUE(std::all_of(bytes, bytes + mapping.byteSize(), [](uint8_t b) {
    return b == 0xA5;
  })) << "dropping the handle disturbed the caller's pages";
}

// A wrapped range is a kernel operand: the kernel writes through it and the
// caller reads the result from its own pages, with no copy back.
TEST_F(MetalDeviceContextTestBase, WrapHostMemoryFeedsKernel) {
  ASSERT_TRUE(ctx_);
  std::string metalSource = getMetalShaderSource();
  ASSERT_FALSE(metalSource.empty());
  ASSIGN_NO_ERR(
      DeviceFunctionRef, fn,
      ctx_->loadFunction("vec_add_module", "vec_add_msl", metalSource));

  constexpr uint32_t blockDim = 256;
  constexpr size_t length = 1024;
  const size_t bytes = length * sizeof(float);
  const size_t mapBytes =
      ((bytes + hostPageSize() - 1) / hostPageSize()) * hostPageSize();

  HostMapping outMap(mapBytes);
  ASSERT_NE(outMap.data(), nullptr);
  std::memset(outMap.data(), 0, mapBytes);

  ASSIGN_NO_ERR(DeviceBufferRef, in0, ctx_->enqueueCreateBuffer(bytes));
  ASSIGN_NO_ERR(DeviceBufferRef, in1, ctx_->enqueueCreateBuffer(bytes));
  ASSIGN_NO_ERR(DeviceBufferRef, wrappedOut,
                ctx_->wrapHostMemory(outMap.data(), mapBytes));

  std::vector<float> a(length), b(length);
  for (size_t i = 0; i < length; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = 2.0f;
  }
  EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*in0, a.data()));
  EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*in1, b.data()));
  EXPECT_NO_ERR(ctx_->synchronize());

  void *outPtr = wrappedOut->ptr();
  void *in0Ptr = in0->ptr();
  void *in1Ptr = in1->ptr();
  uint32_t lengthValue = static_cast<uint32_t>(length);
  void *kernelArgs[] = {&outPtr, &in0Ptr, &in1Ptr, &lengthValue};
  size_t argSizes[] = {sizeof(outPtr), sizeof(in0Ptr), sizeof(in1Ptr),
                       sizeof(lengthValue)};
  bool argIsDevicePtr[] = {true, true, true, false};
  DeviceBuffer *kernelBuffers[] = {wrappedOut.getPointer(), in0.getPointer(),
                                   in1.getPointer()};
  MetalEnqueueFunctionArgs metalArgs = {kernelArgs, argSizes, argIsDevicePtr,
                                        kernelBuffers, 3};
  void *args[] = {&metalArgs};

  EXPECT_NO_ERR(ctx_->enqueueFunctionExecDirect(*fn, length / blockDim, 1, 1,
                                                blockDim, 1, 1, 0, nullptr, 0,
                                                args, 4, argSizes));
  EXPECT_NO_ERR(ctx_->synchronize());

  const float *observed = static_cast<const float *>(outMap.data());
  for (size_t i = 0; i < length; ++i)
    ASSERT_FLOAT_EQ(observed[i], a[i] + b[i]) << "index " << i;
}

// The unregister runs from the last participating handler, so draining the
// owner alone is not enough once another context has copied the range.
TEST_F(MetalDeviceContextTestBase, WrapHostMemoryReleaseNeedsEveryContext) {
  ASSERT_TRUE(ctx_);
  ASSIGN_NO_ERR(DeviceContextRef, foreign, device_->createContext());
  const MetalDevice &metalDev = MetalDevice::cast(*device_);
  const size_t baseline = metalDev.getTrackedBuffers().size();

  const size_t byteSize = hostPageSize();
  HostMapping mapping(byteSize);
  ASSERT_NE(mapping.data(), nullptr);

  void *devPtr = nullptr;
  {
    ASSIGN_NO_ERR(DeviceBufferRef, wrapped,
                  ctx_->wrapHostMemory(mapping.data(), byteSize));
    devPtr = wrapped->ptr();
    ASSIGN_NO_ERR(DeviceBufferRef, deviceBuf,
                  foreign->enqueueCreateBuffer(byteSize));
    EXPECT_NO_ERR(wrapped->copyTo(*deviceBuf));
    EXPECT_NO_ERR(foreign->synchronize());
    // The device-buffer allocation is tracked too, so count only the wrap.
    EXPECT_GE(metalDev.getTrackedBuffers().size(), baseline + 1);
  }

  EXPECT_NO_ERR(ctx_->synchronize());
  EXPECT_NO_ERR(ctx_->pollPendingEvents());
  EXPECT_NE(
      metalDev.getTrackedBuffers().count(reinterpret_cast<uintptr_t>(devPtr)),
      0u)
      << "owner-only drain released the wrap; the foreign handler still held a "
         "reference";

  EXPECT_NO_ERR(foreign->synchronize());
  EXPECT_NO_ERR(foreign->pollPendingEvents());
  EXPECT_EQ(
      metalDev.getTrackedBuffers().count(reinterpret_cast<uintptr_t>(devPtr)),
      0u);
}

// Each handle carries its own range, so they drop independently, in any order.
TEST_F(MetalDeviceContextTestBase, WrapHostMemoryChunksOfOneMapping) {
  ASSERT_TRUE(ctx_);
  const MetalDevice &metalDev = MetalDevice::cast(*device_);
  const size_t baseline = metalDev.getTrackedBuffers().size();

  constexpr size_t kChunks = 3;
  const size_t chunkBytes = 2 * hostPageSize();
  HostMapping mapping(kChunks * chunkBytes);
  ASSERT_NE(mapping.data(), nullptr);

  std::vector<DeviceBufferRef> handles;
  for (size_t i = 0; i < kChunks; ++i) {
    ASSIGN_NO_ERR(
        DeviceBufferRef, chunk,
        ctx_->wrapHostMemory(mapping.bytes() + i * chunkBytes, chunkBytes));
    handles.push_back(std::move(chunk));
  }
  EXPECT_EQ(metalDev.getTrackedBuffers().size(), baseline + kChunks);

  handles.erase(handles.begin() + 1); // middle, then last, then first
  handles.pop_back();
  handles.clear();

  EXPECT_NO_ERR(ctx_->synchronize());
  EXPECT_NO_ERR(ctx_->pollPendingEvents());
  EXPECT_EQ(metalDev.getTrackedBuffers().size(), baseline);
}

// A wrapped range is GPU-addressable, not bookkeeping: the copy resolves it
// through the tracker and blits rather than falling back to a memcpy.
TEST_F(MetalDeviceContextTestBase, WrapHostMemoryFeedsDeviceCopy) {
  ASSERT_TRUE(ctx_);
  const size_t byteSize = hostPageSize();
  HostMapping mapping(byteSize);
  ASSERT_NE(mapping.data(), nullptr);

  const size_t count = byteSize / sizeof(uint32_t);
  uint32_t *src = static_cast<uint32_t *>(mapping.data());
  for (size_t i = 0; i < count; ++i)
    src[i] = static_cast<uint32_t>(i * 7 + 3);

  ASSIGN_NO_ERR(DeviceBufferRef, wrapped,
                ctx_->wrapHostMemory(mapping.data(), byteSize));

  // Without this the copy below would fall back to a memcpy and still pass.
  ASSIGN_NO_ERR(void *, resolved,
                ctx_->getHostAccessiblePointer(wrapped->ptr()));
  EXPECT_EQ(resolved, mapping.data());
  EXPECT_NE(wrapped->ptr(), mapping.data());

  ASSIGN_NO_ERR(DeviceBufferRef, deviceBuf,
                ctx_->enqueueCreateBuffer(byteSize));
  EXPECT_NO_ERR(wrapped->copyTo(*deviceBuf));
  EXPECT_NO_ERR(ctx_->synchronize());

  ASSIGN_NO_ERR(void *, readback,
                ctx_->getHostAccessiblePointer(deviceBuf->ptr()));
  const uint32_t *observed = static_cast<const uint32_t *>(readback);
  for (size_t i = 0; i < count; ++i)
    ASSERT_EQ(observed[i], src[i]) << "mismatch at index " << i;
}

// Test: chunkAlloc rejects a request exactly equal to maxBufferLength.
// allocateNewBuffer uses `byteSize >= maxBufferLength`, so the boundary value
// must fail even though it is only one byte over the usable limit.
TEST_F(MetalDeviceContextTestBase, ChunkAlloc_AtMaxBufferLength_ReturnsError) {
  MetalDevice &metalDev = MetalDevice::cast(*device_);
  size_t maxLen = metalDev.metalDevice()->maxBufferLength();

  void *ptr = nullptr;
  EXPECT_ERR_MSG(metalDev.chunkAlloc(&ptr, maxLen, /*isHost=*/false),
                 "larger than allowed");
  EXPECT_EQ(ptr, nullptr);
}

// Test: chunkAlloc rejects SIZE_MAX — the largest possible size_t value.
TEST_F(MetalDeviceContextTestBase, ChunkAlloc_SizeMax_ReturnsError) {
  MetalDevice &metalDev = MetalDevice::cast(*device_);

  void *ptr = nullptr;
  EXPECT_ERR_MSG(metalDev.chunkAlloc(&ptr, SIZE_MAX, /*isHost=*/false),
                 "larger than allowed");
  EXPECT_EQ(ptr, nullptr);
}

// Test: chunkAlloc with a normal size succeeds, buffer is tracked, chunkFree
// removes it.
TEST_F(MetalDeviceContextTestBase, ChunkAlloc_ValidSize_SucceedsAndFrees) {
  MetalDevice &metalDev = MetalDevice::cast(*device_);
  const size_t baseline = metalDev.getTrackedBuffers().size();

  void *ptr = nullptr;
  EXPECT_NO_ERR(metalDev.chunkAlloc(&ptr, /*byteSize=*/4096, /*isHost=*/false));
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(metalDev.getTrackedBuffers().size(), baseline + 1);
  EXPECT_NE(metalDev.getTrackedBuffer(ptr), nullptr);

  EXPECT_NO_ERR(metalDev.chunkFree(ptr, /*isHost=*/false));
  EXPECT_EQ(metalDev.getTrackedBuffers().size(), baseline);
}

// getMemoryInfo reports recommendedMaxWorkingSetSize as total and
// (total - currentAllocatedSize) as free.
TEST_F(MetalDeviceContextTestBase, GetMemInfo_TotalAndFree) {
  const MetalDevice &metalDev = MetalDevice::cast(*device_);
  MTL::Device *mtlDev = metalDev.metalDevice();
  size_t expectedTotal = mtlDev->recommendedMaxWorkingSetSize();
  size_t used = mtlDev->currentAllocatedSize();
  size_t expectedFree = expectedTotal > used ? expectedTotal - used : 0;

  ASSIGN_NO_ERR(Device::MemInfo, info, metalDev.getMemoryInfo());
  EXPECT_EQ(info.total, expectedTotal);
  EXPECT_EQ(info.free, expectedFree);
}

// maxSingleAllocationSize on Metal is maxBufferLength.
TEST_F(MetalDeviceContextTestBase,
       MaxSingleAllocationSize_MatchesBufferLength) {
  const MetalDevice &metalDev = MetalDevice::cast(*device_);
  ASSIGN_NO_ERR(size_t, actual, metalDev.maxSingleAllocationSize());
  EXPECT_EQ(actual, metalDev.metalDevice()->maxBufferLength());
  EXPECT_GT(actual, 0u);
}

// Against the Metal query rather than a hardcoded true, so an Intel Mac driving
// a discrete GPU reports the truth instead of failing.
TEST_F(MetalDeviceContextTestBase, IsHostUnified) {
  const MetalDevice &metalDev = MetalDevice::cast(*device_);
  ASSIGN_NO_ERR(bool, unified, metalDev.isHostUnified());
  EXPECT_EQ(unified, metalDev.metalDevice()->hasUnifiedMemory());
}

// Metal device allocations are `MTLStorageModeShared`, so the host can read the
// bytes a blit wrote once the queue has drained -- no device-to-host copy.
TEST_F(MetalDeviceContextTestBase, GetHostAccessiblePointer_ReadsDeviceWrites) {
  ASSERT_TRUE(ctx_);
  constexpr size_t length = 64;

  ASSIGN_NO_ERR(DeviceBufferRef, buf,
                ctx_->enqueueCreateBuffer(length * sizeof(uint32_t)));

  std::vector<uint32_t> src(length);
  for (size_t i = 0; i < length; ++i)
    src[i] = static_cast<uint32_t>(i * 3 + 1);
  EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*buf, src.data()));
  EXPECT_NO_ERR(ctx_->synchronize());

  ASSIGN_NO_ERR(void *, hostPtr, ctx_->getHostAccessiblePointer(buf->ptr()));
  ASSERT_NE(hostPtr, nullptr);
  const uint32_t *observed = static_cast<const uint32_t *>(hostPtr);
  for (size_t i = 0; i < length; ++i)
    EXPECT_EQ(observed[i], src[i]) << "mismatch at index " << i;
}

// The range map resolves interior pointers, so a sub-buffer and a hand-computed
// interior address must both land at the parent's host address plus the offset.
TEST_F(MetalDeviceContextTestBase, GetHostAccessiblePointer_ResolvesOffsets) {
  ASSERT_TRUE(ctx_);
  constexpr size_t byteSize = 4096;
  constexpr size_t offset = 256;

  ASSIGN_NO_ERR(DeviceBufferRef, parent, ctx_->enqueueCreateBuffer(byteSize));
  ASSIGN_NO_ERR(void *, parentHost,
                ctx_->getHostAccessiblePointer(parent->ptr()));

  ASSIGN_NO_ERR(DeviceBufferRef, sub,
                parent->createSubBuffer(offset, byteSize - offset));
  ASSIGN_NO_ERR(void *, subHost, ctx_->getHostAccessiblePointer(sub->ptr()));
  EXPECT_EQ(static_cast<char *>(subHost),
            static_cast<char *>(parentHost) + offset);

  const void *interior = static_cast<const char *>(parent->ptr()) + offset;
  ASSIGN_NO_ERR(void *, interiorHost, ctx_->getHostAccessiblePointer(interior));
  EXPECT_EQ(interiorHost, subHost);
}

// Host allocations already hand out `contents()`, so feeding one back in must
// return the same address rather than shifting it.
TEST_F(MetalDeviceContextTestBase, GetHostAccessiblePointer_HostBufferIsSame) {
  ASSERT_TRUE(ctx_);
  ASSIGN_NO_ERR(DeviceBufferRef, host, ctx_->createHostBuffer(1024));
  ASSIGN_NO_ERR(void *, hostPtr, ctx_->getHostAccessiblePointer(host->ptr()));
  EXPECT_EQ(hostPtr, host->ptr());
}

TEST_F(MetalDeviceContextTestBase, GetHostAccessiblePointer_UnknownPointer) {
  ASSERT_TRUE(ctx_);
  uint32_t onStack = 0;
  EXPECT_ERR_MSG(ctx_->getHostAccessiblePointer(&onStack),
                 "was not allocated by this device");
  EXPECT_ERR_MSG(ctx_->getHostAccessiblePointer(nullptr), "pointer is null");
}

// A never-recorded event is complete (CUDA semantics), waiting on it is a
// no-op, and Metal events cannot carry timing.
TEST_F(MetalDeviceContextTestBase, EventCreateUnrecordedIsReady) {
  ASSIGN_NO_ERR(DeviceEventRef, event, ctx_->eventCreate(false));
  ASSIGN_NO_ERR(bool, ready, event->isReady());
  EXPECT_TRUE(ready);
  EXPECT_NO_ERR(event->synchronize());
  EXPECT_NO_ERR(ctx_->enqueueWaitFor(*event));
  EXPECT_NO_ERR(ctx_->synchronize());
  EXPECT_ERR_MSG(ctx_->eventCreate(true), "timing");
}

// enqueueEvent() marks the point after all previously committed work; its
// synchronize() alone makes the gated kernel's output host-visible.
TEST_F(MetalDeviceContextTestBase, EventRecordTracksQueueCompletion) {
  ASSIGN_NO_ERR(DeviceBufferRef, out,
                ctx_->enqueueCreateBuffer(64 * sizeof(float)));
  float zeros[64] = {};
  EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*out, zeros));
  EXPECT_NO_ERR(ctx_->synchronize());

  HostGate gate = makeGate(ctx_);
  ASSERT_TRUE(gate.event);
  launchFillOne(ctx_, out, 64);
  ASSIGN_NO_ERR(DeviceEventRef, event, ctx_->enqueueEvent());

  ASSIGN_NO_ERR(bool, readyWhileGated, event->isReady());
  EXPECT_FALSE(readyWhileGated);

  gate.release();
  EXPECT_NO_ERR(event->synchronize());
  ASSIGN_NO_ERR(bool, readyAfter, event->isReady());
  EXPECT_TRUE(readyAfter);

  ASSIGN_NO_ERR(void *, outHost, ctx_->getHostAccessiblePointer(out->ptr()));
  EXPECT_FLOAT_EQ(static_cast<float *>(outHost)[0], 1.0f);
  EXPECT_FLOAT_EQ(static_cast<float *>(outHost)[63], 1.0f);
}

// Re-recording an event rebinds it to the newest work: complete after the
// first record drains, pending again once re-recorded behind a gate.
TEST_F(MetalDeviceContextTestBase, EventReRecordTracksLatestWork) {
  ASSIGN_NO_ERR(DeviceEventRef, event, ctx_->eventCreate(false));
  EXPECT_NO_ERR(ctx_->enqueueEvent(*event));
  EXPECT_NO_ERR(event->synchronize());

  HostGate gate = makeGate(ctx_);
  ASSERT_TRUE(gate.event);
  EXPECT_NO_ERR(ctx_->enqueueEvent(*event));
  ASSIGN_NO_ERR(bool, readyWhileGated, event->isReady());
  EXPECT_FALSE(readyWhileGated);

  gate.release();
  EXPECT_NO_ERR(event->synchronize());
}

// enqueueWaitFor(event) must gate work committed later on another context
// (another Metal queue) without blocking the host.
TEST_F(MetalDeviceContextTestBase, EnqueueWaitForGatesAcrossContexts) {
  ASSIGN_NO_ERR(DeviceContextRef, ctxB, device_->createContext());

  ASSIGN_NO_ERR(DeviceBufferRef, out,
                ctx_->enqueueCreateBuffer(64 * sizeof(float)));
  float zeros[64] = {};
  EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*out, zeros));
  EXPECT_NO_ERR(ctx_->synchronize());

  HostGate gate = makeGate(ctx_);
  ASSERT_TRUE(gate.event);
  launchFillOne(ctx_, out, 64);
  ASSIGN_NO_ERR(DeviceEventRef, eventA, ctx_->enqueueEvent());

  EXPECT_NO_ERR(ctxB->enqueueWaitFor(*eventA));
  ASSIGN_NO_ERR(DeviceEventRef, eventB, ctxB->enqueueEvent());
  ASSIGN_NO_ERR(bool, bReadyWhileGated, eventB->isReady());
  EXPECT_FALSE(bReadyWhileGated);

  gate.release();
  EXPECT_NO_ERR(eventB->synchronize());
  ASSIGN_NO_ERR(void *, outHost, ctx_->getHostAccessiblePointer(out->ptr()));
  EXPECT_FLOAT_EQ(static_cast<float *>(outHost)[0], 1.0f);
}

// Sequential record/synchronize cycles keep the value bookkeeping exact:
// the final signaled value is the record count and no cycle wedges.
TEST_F(MetalDeviceContextTestBase, EventReuseSequentialHammer) {
  constexpr uint64_t kRecords = 200;
  ASSIGN_NO_ERR(DeviceEventRef, event, ctx_->eventCreate(false));
  for (uint64_t i = 0; i < kRecords; ++i) {
    EXPECT_NO_ERR(ctx_->enqueueEvent(*event));
    EXPECT_NO_ERR(event->synchronize());
    ASSIGN_NO_ERR(bool, ready, event->isReady());
    EXPECT_TRUE(ready) << "cycle " << i;
  }
  EXPECT_EQ(MetalDeviceEvent::cast(*event).sharedEvent()->signaledValue(),
            kRecords);
}

// A waiter captures the event's value at wait time (CUDA semantics): a
// re-record does not retarget waits already taken.
TEST_F(MetalDeviceContextTestBase, EventReRecordStaleWaiterKeepsOldTarget) {
  ASSIGN_NO_ERR(DeviceContextRef, ctxB, device_->createContext());
  ASSIGN_NO_ERR(DeviceEventRef, event, ctx_->eventCreate(false));

  HostGate gate1 = makeGate(ctx_);
  ASSERT_TRUE(gate1.event);
  EXPECT_NO_ERR(ctx_->enqueueEvent(*event)); // value 1, behind gate1

  EXPECT_NO_ERR(ctxB->enqueueWaitFor(*event)); // captures value 1
  ASSIGN_NO_ERR(DeviceEventRef, probeB, ctxB->enqueueEvent());

  HostGate gate2 = makeGate(ctx_);
  ASSERT_TRUE(gate2.event);
  EXPECT_NO_ERR(ctx_->enqueueEvent(*event)); // value 2, behind gate2

  gate1.release();
  // The stale waiter (value 1) unblocks even though value 2 is pending.
  EXPECT_NO_ERR(probeB->synchronize());
  ASSIGN_NO_ERR(bool, eventReady, event->isReady());
  EXPECT_FALSE(eventReady);

  gate2.release();
  EXPECT_NO_ERR(event->synchronize());
}

// Racing re-records from multiple threads must serialize through the record
// mutex: no value regression, no lost signal, exact final count.
TEST_F(MetalDeviceContextTestBase, EventConcurrentReRecords) {
  constexpr int kThreads = 4;
  constexpr int kRecordsPerThread = 50;
  ASSIGN_NO_ERR(DeviceEventRef, event, ctx_->eventCreate(false));

  std::vector<std::thread> threads;
  std::atomic<int> failures{0};
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&] {
      for (int i = 0; i < kRecordsPerThread; ++i)
        if (ctx_->enqueueEvent(*event).isError())
          failures.fetch_add(1);
    });
  }
  for (std::thread &t : threads)
    t.join();
  EXPECT_EQ(failures.load(), 0);

  EXPECT_NO_ERR(event->synchronize());
  EXPECT_EQ(MetalDeviceEvent::cast(*event).sharedEvent()->signaledValue(),
            static_cast<uint64_t>(kThreads * kRecordsPerThread));
}

// An event is not bound to the context that created it: re-recording on a
// second context (another queue) tracks that queue's completion.
TEST_F(MetalDeviceContextTestBase, EventReuseAcrossContexts) {
  ASSIGN_NO_ERR(DeviceContextRef, ctxB, device_->createContext());
  ASSIGN_NO_ERR(DeviceEventRef, event, ctx_->eventCreate(false));

  EXPECT_NO_ERR(ctx_->enqueueEvent(*event));
  EXPECT_NO_ERR(event->synchronize());

  HostGate gateB = makeGate(ctxB);
  ASSERT_TRUE(gateB.event);
  EXPECT_NO_ERR(ctxB->enqueueEvent(*event));
  ASSIGN_NO_ERR(bool, readyWhileGated, event->isReady());
  EXPECT_FALSE(readyWhileGated);

  gateB.release();
  EXPECT_NO_ERR(event->synchronize());
}

// A device-to-host-pointer copy must observe writes from work still pending
// when the copy was enqueued.
TEST_F(MetalDeviceContextTestBase, HostCopyDrainsPriorKernel) {
  ASSIGN_NO_ERR(DeviceBufferRef, out,
                ctx_->enqueueCreateBuffer(64 * sizeof(float)));
  float zeros[64] = {};
  EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*out, zeros));
  EXPECT_NO_ERR(ctx_->synchronize());

  HostGate gate = makeGate(ctx_);
  ASSERT_TRUE(gate.event);
  launchFillOne(ctx_, out, 64);

  std::thread releaser([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    gate.release();
  });
  float observed[64] = {};
  EXPECT_NO_ERR(
      ctx_->enqueueCopyFromDevice(observed, out->ptr(), sizeof(observed)));
  releaser.join();
  EXPECT_FLOAT_EQ(observed[0], 1.0f);
  EXPECT_FLOAT_EQ(observed[63], 1.0f);
}

// The timer must measure GPU work enqueued between start and stop, not the
// host's enqueue time: stopTimer returns only after the gated kernel ran, and
// the elapsed time spans the gate.
TEST_F(MetalDeviceContextTestBase, TimerCoversEnqueuedWork) {
  ASSIGN_NO_ERR(DeviceBufferRef, out,
                ctx_->enqueueCreateBuffer(64 * sizeof(float)));
  float zeros[64] = {};
  EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*out, zeros));
  EXPECT_NO_ERR(ctx_->synchronize());

  constexpr auto kGateHold = std::chrono::milliseconds(100);
  ASSIGN_NO_ERR(DeviceTimerRef, timer, ctx_->startTimer());
  HostGate gate = makeGate(ctx_);
  ASSERT_TRUE(gate.event);
  launchFillOne(ctx_, out, 64);
  std::thread releaser([&] {
    std::this_thread::sleep_for(kGateHold);
    gate.release();
  });
  ErrorOr<std::chrono::nanoseconds> elapsed = ctx_->stopTimer(*timer);
  releaser.join();
  ASSERT_FALSE(elapsed.isError()) << elapsed.getError();

  ASSIGN_NO_ERR(void *, outHost, ctx_->getHostAccessiblePointer(out->ptr()));
  EXPECT_FLOAT_EQ(static_cast<float *>(outHost)[63], 1.0f);
  // Half the hold leaves room for the start marker's own scheduling latency.
  EXPECT_GE(*elapsed, kGateHold / 2);
}

// As with CUDA events, the start marker is ordered behind earlier work, so
// work queued before startTimer is not counted.
TEST_F(MetalDeviceContextTestBase, TimerExcludesEarlierWork) {
  constexpr auto kGateHold = std::chrono::milliseconds(200);
  HostGate gate = makeGate(ctx_);
  ASSERT_TRUE(gate.event);
  ASSIGN_NO_ERR(DeviceTimerRef, timer, ctx_->startTimer());
  std::thread releaser([&] {
    std::this_thread::sleep_for(kGateHold);
    gate.release();
  });
  ErrorOr<std::chrono::nanoseconds> elapsed = ctx_->stopTimer(*timer);
  releaser.join();
  ASSERT_FALSE(elapsed.isError()) << elapsed.getError();
  EXPECT_LT(*elapsed, kGateHold / 2);
}

// A cross-context re-record (value 2, its own queue) must not signal ahead
// of an earlier record (value 1) still gated behind real prior work.
TEST_F(MetalDeviceContextTestBase,
       EventCrossContextReRecordKeepsMonotonicValues) {
  ASSIGN_NO_ERR(DeviceBufferRef, out,
                ctx_->enqueueCreateBuffer(64 * sizeof(float)));
  float zeros[64] = {};
  EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*out, zeros));
  EXPECT_NO_ERR(ctx_->synchronize());

  ASSIGN_NO_ERR(DeviceEventRef, event, ctx_->eventCreate(false));

  HostGate gateA = makeGate(ctx_);
  ASSERT_TRUE(gateA.event);
  launchFillOne(ctx_, out, 64);
  EXPECT_NO_ERR(ctx_->enqueueEvent(*event)); // value 1, behind gateA + fillOne

  ASSIGN_NO_ERR(DeviceContextRef, ctxB, device_->createContext());
  EXPECT_NO_ERR(ctxB->enqueueEvent(*event)); // value 2, ungated on ctxB
  // Give ctxB's signal(2) time to land; without the chain it has no
  // dependency on ctx_ and fires almost immediately after commit.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  ASSIGN_NO_ERR(bool, readyBeforeFillOneRan, event->isReady());
  EXPECT_FALSE(readyBeforeFillOneRan);

  gateA.release();
  EXPECT_NO_ERR(event->synchronize());
  ASSIGN_NO_ERR(void *, outHost, ctx_->getHostAccessiblePointer(out->ptr()));
  EXPECT_FLOAT_EQ(static_cast<float *>(outHost)[0], 1.0f);
}

// enqueueWaitFor(DeviceContext&) is a self-wait no-op and gates the waiter's
// later work on the target's pending work without blocking the host.
TEST_F(MetalDeviceContextTestBase, WaitForContextGatesAcrossQueues) {
  EXPECT_NO_ERR(ctx_->enqueueWaitFor(*ctx_));
  EXPECT_NO_ERR(ctx_->synchronize());

  ASSIGN_NO_ERR(DeviceBufferRef, out,
                ctx_->enqueueCreateBuffer(64 * sizeof(float)));
  float zeros[64] = {};
  EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*out, zeros));
  EXPECT_NO_ERR(ctx_->synchronize());

  HostGate gate = makeGate(ctx_);
  ASSERT_TRUE(gate.event);
  launchFillOne(ctx_, out, 64);

  ASSIGN_NO_ERR(DeviceContextRef, ctxB, device_->createContext());
  EXPECT_NO_ERR(ctxB->enqueueWaitFor(*ctx_));
  ASSIGN_NO_ERR(DeviceEventRef, probe, ctxB->enqueueEvent());
  ASSIGN_NO_ERR(bool, probeReadyWhileGated, probe->isReady());
  EXPECT_FALSE(probeReadyWhileGated);

  gate.release();
  EXPECT_NO_ERR(probe->synchronize());
  ASSIGN_NO_ERR(void *, outHost, ctx_->getHostAccessiblePointer(out->ptr()));
  EXPECT_FLOAT_EQ(static_cast<float *>(outHost)[0], 1.0f);
  EXPECT_FLOAT_EQ(static_cast<float *>(outHost)[63], 1.0f);
}

// A D2D copy must wait for the queue that produced its source on the GPU,
// not read whatever the blit happens to see.
TEST_F(MetalDeviceContextTestBase, MemCopyDtoDWaitsForProducingContext) {
  if (ctx_->syncMode())
    GTEST_SKIP() << "asserts async copy ordering; device-sync-mode is on";
  ASSIGN_NO_ERR(DeviceContextRef, ctxB, device_->createContext());

  ASSIGN_NO_ERR(DeviceBufferRef, src,
                ctx_->enqueueCreateBuffer(64 * sizeof(float)));
  ASSIGN_NO_ERR(DeviceBufferRef, dst,
                ctxB->enqueueCreateBuffer(64 * sizeof(float)));
  float zeros[64] = {};
  EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*src, zeros));
  EXPECT_NO_ERR(ctxB->enqueueCopyToDevice(*dst, zeros));
  EXPECT_NO_ERR(ctx_->synchronize());
  EXPECT_NO_ERR(ctxB->synchronize());

  HostGate gate = makeGate(ctx_);
  ASSERT_TRUE(gate.event);
  launchFillOne(ctx_, src, 64);

  EXPECT_NO_ERR(ctxB->enqueueCopyDeviceToDevice(*dst, *src));
  ASSIGN_NO_ERR(DeviceEventRef, probe, ctxB->enqueueEvent());

  // Without the bracket the copy has no dependency on ctx_ and lands at once.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  ASSIGN_NO_ERR(bool, probeReadyWhileGated, probe->isReady());
  EXPECT_FALSE(probeReadyWhileGated);

  gate.release();
  EXPECT_NO_ERR(probe->synchronize());
  ASSIGN_NO_ERR(void *, dstHost, ctxB->getHostAccessiblePointer(dst->ptr()));
  EXPECT_FLOAT_EQ(static_cast<float *>(dstHost)[0], 1.0f);
  EXPECT_FLOAT_EQ(static_cast<float *>(dstHost)[63], 1.0f);
}

// Opting out is the caller's promise to order the copy itself, so the
// destination queue must be left independent.
TEST_F(MetalDeviceContextTestBase, MemCopyDtoDHonorsCrossStreamSyncOptOut) {
  ASSIGN_NO_ERR(DeviceContextRef, ctxB, device_->createContext());

  ASSIGN_NO_ERR(DeviceBufferRef, src,
                ctx_->enqueueCreateBuffer(64 * sizeof(float)));
  ASSIGN_NO_ERR(DeviceBufferRef, dst,
                ctxB->enqueueCreateBuffer(64 * sizeof(float)));
  float zeros[64] = {};
  EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*src, zeros));
  EXPECT_NO_ERR(ctx_->synchronize());

  HostGate gate = makeGate(ctx_);
  ASSERT_TRUE(gate.event);
  launchFillOne(ctx_, src, 64);

  EXPECT_NO_ERR(
      ctxB->enqueueCopyDeviceToDevice(*dst, *src, /*crossStreamSync=*/false));
  ASSIGN_NO_ERR(DeviceEventRef, probe, ctxB->enqueueEvent());

  // Poll rather than synchronize: a regression gating ctxB on ctx_ would block
  // forever, and a bounded wait reports that as a failure instead of a hang.
  bool completedWhileGated = false;
  for (int i = 0; i < 200 && !completedWhileGated; ++i) {
    ASSIGN_NO_ERR(bool, ready, probe->isReady());
    completedWhileGated = ready;
    if (!ready)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_TRUE(completedWhileGated);

  gate.release();
  EXPECT_NO_ERR(ctx_->synchronize());
  EXPECT_NO_ERR(ctxB->synchronize());
}

// A destination owned by another context would silently get the wrong
// queue's ordering.
TEST_F(MetalDeviceContextTestBase, MemCopyDtoDRejectsForeignDestination) {
  ASSIGN_NO_ERR(DeviceContextRef, ctxB, device_->createContext());
  ASSIGN_NO_ERR(DeviceBufferRef, src,
                ctx_->enqueueCreateBuffer(64 * sizeof(float)));
  ASSIGN_NO_ERR(DeviceBufferRef, dst,
                ctxB->enqueueCreateBuffer(64 * sizeof(float)));

  EXPECT_ERR_MSG(ctx_->enqueueCopyDeviceToDevice(*dst, *src),
                 "does not match context of dst");
}

// Runs `enqueue` while the queue is gated and reports whether it returned
// before the gate opened. A call that drains the queue blocks until the
// timeout releases the gate instead.
static bool returnsWhileGated(const std::function<void()> &enqueue,
                              std::function<void()> releaseGate) {
  std::atomic<bool> returned{false};
  std::atomic<bool> timedOut{false};
  std::thread releaser([&] {
    for (int i = 0; i < 200 && !returned.load(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    timedOut = !returned.load();
    releaseGate();
  });
  enqueue();
  returned = true;
  releaser.join();
  return !timedOut.load();
}

template <typename T>
static void expectAllEqual(const void *data, size_t count, T expected) {
  const T *words = static_cast<const T *>(data);
  for (size_t i = 0; i < count; ++i)
    ASSERT_EQ(words[i], expected) << "index " << i;
}

// A memset behind a pending kernel that writes the same buffer must land
// after that kernel on the GPU without the host waiting for it. Byte-uniform
// patterns take a blit and wider ones a helper kernel, so both are covered.
TEST_F(MetalDeviceContextTestBase, SetMemoryOrdersBehindPendingWriter) {
  if (ctx_->syncMode())
    GTEST_SKIP() << "asserts an asynchronous memset; device-sync-mode is on";
  constexpr uint32_t kLen = 64;
  ASSIGN_NO_ERR(DeviceBufferRef, out,
                ctx_->enqueueCreateBuffer(kLen * sizeof(float)));
  ASSIGN_NO_ERR(void *, outHost, ctx_->getHostAccessiblePointer(out->ptr()));

  struct Pattern {
    uint64_t val;
    size_t valSize;
  };
  for (Pattern p :
       {Pattern{0, 4}, Pattern{0xA5, 1}, Pattern{0x3C00, 2},
        Pattern{0xFF800000, 4}, Pattern{0x0123456789ABCDEFull, 8}}) {
    SCOPED_TRACE(testing::Message()
                 << "val=" << p.val << " valSize=" << p.valSize);
    HostGate gate = makeGate(ctx_);
    ASSERT_TRUE(gate.event);
    launchFillOne(ctx_, out, kLen);
    EXPECT_TRUE(returnsWhileGated(
        [&] { EXPECT_NO_ERR(ctx_->enqueueSetMemory(*out, p.val, p.valSize)); },
        [&] { gate.release(); }));
    EXPECT_NO_ERR(ctx_->synchronize());

    const size_t count = out->byteSize() / p.valSize;
    switch (p.valSize) {
    case 1:
      expectAllEqual<uint8_t>(outHost, count, static_cast<uint8_t>(p.val));
      break;
    case 2:
      expectAllEqual<uint16_t>(outHost, count, static_cast<uint16_t>(p.val));
      break;
    case 4:
      expectAllEqual<uint32_t>(outHost, count, static_cast<uint32_t>(p.val));
      break;
    default:
      expectAllEqual<uint64_t>(outHost, count, p.val);
    }
  }
}

// A memset must not overwrite a buffer that a pending kernel still reads,
// on either the blit or the helper-kernel path.
TEST_F(MetalDeviceContextTestBase, SetMemoryWaitsForPendingReader) {
  if (ctx_->syncMode())
    GTEST_SKIP() << "asserts an asynchronous memset; device-sync-mode is on";
  ASSIGN_NO_ERR(DeviceFunctionRef, fn,
                ctx_->loadFunction("vec_add_module", "vec_add_msl",
                                   getMetalShaderSource()));
  constexpr uint32_t kLen = 256;
  const size_t bytes = kLen * sizeof(float);
  ASSIGN_NO_ERR(DeviceBufferRef, out, ctx_->enqueueCreateBuffer(bytes));
  ASSIGN_NO_ERR(DeviceBufferRef, in0, ctx_->enqueueCreateBuffer(bytes));
  ASSIGN_NO_ERR(DeviceBufferRef, in1, ctx_->enqueueCreateBuffer(bytes));
  ASSIGN_NO_ERR(void *, outHost, ctx_->getHostAccessiblePointer(out->ptr()));
  ASSIGN_NO_ERR(void *, in0Host, ctx_->getHostAccessiblePointer(in0->ptr()));
  std::vector<float> a(kLen), b(kLen, 2.0f);
  for (uint32_t i = 0; i < kLen; ++i)
    a[i] = static_cast<float>(i);
  EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*in1, b.data()));

  for (uint32_t fill : {0u, 0xFF800000u}) {
    SCOPED_TRACE(testing::Message() << "fill=" << fill);
    EXPECT_NO_ERR(ctx_->enqueueCopyToDevice(*in0, a.data()));
    EXPECT_NO_ERR(ctx_->synchronize());

    HostGate gate = makeGate(ctx_);
    ASSERT_TRUE(gate.event);
    void *outPtr = out->ptr();
    void *in0Ptr = in0->ptr();
    void *in1Ptr = in1->ptr();
    uint32_t length = kLen;
    void *kernelArgs[] = {&outPtr, &in0Ptr, &in1Ptr, &length};
    size_t argSizes[] = {sizeof(outPtr), sizeof(in0Ptr), sizeof(in1Ptr),
                         sizeof(length)};
    bool argIsDevicePtr[] = {true, true, true, false};
    DeviceBuffer *kernelBuffers[] = {out.getPointer(), in0.getPointer(),
                                     in1.getPointer()};
    MetalEnqueueFunctionArgs metalArgs = {kernelArgs, argSizes, argIsDevicePtr,
                                          kernelBuffers, 3};
    void *args[] = {&metalArgs};
    EXPECT_NO_ERR(ctx_->enqueueFunctionExecDirect(
        *fn, 1, 1, 1, kLen, 1, 1, 0, nullptr, 0, args, 4, argSizes));
    EXPECT_TRUE(returnsWhileGated(
        [&] { EXPECT_NO_ERR(ctx_->enqueueSetMemory(*in0, fill, 4)); },
        [&] { gate.release(); }));
    EXPECT_NO_ERR(ctx_->synchronize());

    for (uint32_t i = 0; i < kLen; ++i)
      ASSERT_FLOAT_EQ(static_cast<float *>(outHost)[i], a[i] + b[i])
          << "index " << i;
    expectAllEqual<uint32_t>(in0Host, kLen, fill);
  }
}

// Interior ranges keep their neighbors, a multi-MiB fill covers its ragged
// tail, host buffers take the same path, and a size that is not a whole
// number of elements is rejected as on CUDA.
TEST_F(MetalDeviceContextTestBase, SetMemoryRangesAndSizes) {
  constexpr size_t kLen = 64;
  ASSIGN_NO_ERR(DeviceBufferRef, buf, ctx_->enqueueCreateBuffer(kLen * 4));
  ASSIGN_NO_ERR(void *, host, ctx_->getHostAccessiblePointer(buf->ptr()));
  char *base = static_cast<char *>(buf->ptr());
  auto *bytes = static_cast<uint8_t *>(host);

  EXPECT_NO_ERR(ctx_->enqueueSetMemory(*buf, 0x11, 1));
  EXPECT_NO_ERR(ctx_->enqueueSetMemory(base + 3, 5, 0xEE, 1));
  EXPECT_NO_ERR(ctx_->enqueueSetMemory(base + 16, 128, 0x3C00, 2));
  EXPECT_NO_ERR(ctx_->synchronize());
  for (size_t i = 0; i < kLen * 4; ++i) {
    uint8_t expected = 0x11;
    if (i >= 3 && i < 8)
      expected = 0xEE;
    else if (i >= 16 && i < 144)
      expected = (i % 2 == 0) ? 0x00 : 0x3C;
    ASSERT_EQ(bytes[i], expected) << "byte " << i;
  }

  const size_t bigBytes = 3 * (size_t(1) << 20) + 40;
  ASSIGN_NO_ERR(DeviceBufferRef, big, ctx_->enqueueCreateBuffer(bigBytes));
  EXPECT_NO_ERR(ctx_->enqueueSetMemory(*big, 0x0123456789ABCDEFull, 8));
  EXPECT_NO_ERR(ctx_->synchronize());
  ASSIGN_NO_ERR(void *, bigHost, ctx_->getHostAccessiblePointer(big->ptr()));
  expectAllEqual<uint64_t>(bigHost, bigBytes / 8, 0x0123456789ABCDEFull);

  ASSIGN_NO_ERR(DeviceBufferRef, hostBuf, ctx_->createHostBuffer(256));
  EXPECT_NO_ERR(ctx_->enqueueSetMemory(*hostBuf, 0xDEADBEEF, 4));
  EXPECT_NO_ERR(ctx_->synchronize());
  expectAllEqual<uint32_t>(hostBuf->ptr(), 64, 0xDEADBEEF);

  EXPECT_ERR_MSG(ctx_->enqueueSetMemory(base, 6, 0, 4), "misaligned");
  EXPECT_ERR_MSG(ctx_->enqueueSetMemory(base + 2, 4, 0, 4), "aligned");
}

// Byte `i` of a fill of `val` (pattern size `valSize`) starting at `dst`.
static uint8_t patternByte(uint64_t val, size_t valSize, size_t i) {
  return static_cast<uint8_t>(val >> (8 * (i % valSize)));
}

// The helper kernel writes whole 16-byte blocks with vector stores and the
// edge blocks byte by byte. Sweep element-aligned offsets that are not
// 16-byte aligned, ranges inside one block, across a block boundary, and
// longer than a threadgroup's worth of blocks with a ragged tail, plus one
// range longer than one pass of the kernel's capped grid. The bytes around
// each range must keep their old value.
TEST_F(MetalDeviceContextTestBase, SetMemoryPatternKernelEdges) {
  constexpr uint8_t kGuard = 0x5A;
  constexpr size_t kBufBytes = 64 * 1024;
  ASSIGN_NO_ERR(DeviceBufferRef, buf, ctx_->enqueueCreateBuffer(kBufBytes));
  ASSIGN_NO_ERR(void *, host, ctx_->getHostAccessiblePointer(buf->ptr()));
  char *base = static_cast<char *>(buf->ptr());
  const auto *bytes = static_cast<const uint8_t *>(host);

  for (size_t valSize : {size_t(2), size_t(4), size_t(8)}) {
    const uint64_t val = 0x0123456789ABCDEFull & (~0ull >> (64 - 8 * valSize));
    for (size_t offset :
         {size_t(0), valSize, 16 - valSize, 16 + valSize, size_t(40)}) {
      for (size_t len : {valSize, 2 * valSize, size_t(16), 48 + valSize,
                         size_t(256 * 16 * 3 + 8)}) {
        SCOPED_TRACE(testing::Message() << "valSize=" << valSize << " offset="
                                        << offset << " len=" << len);
        EXPECT_NO_ERR(ctx_->enqueueSetMemory(*buf, kGuard, 1));
        EXPECT_NO_ERR(ctx_->enqueueSetMemory(base + offset, len, val, valSize));
        EXPECT_NO_ERR(ctx_->synchronize());
        for (size_t i = 0; i < kBufBytes; ++i) {
          const uint8_t expected = (i >= offset && i < offset + len)
                                       ? patternByte(val, valSize, i - offset)
                                       : kGuard;
          ASSERT_EQ(bytes[i], expected) << "byte " << i;
        }
      }
    }
  }

  // 256 MiB is one pass of the kernel's grid (65536 threadgroups of 256
  // threads, 16 bytes each); go past it so threads loop.
  const size_t bigBytes = (size_t(256) << 20) + 4096 + 64;
  ASSIGN_NO_ERR(DeviceBufferRef, big, ctx_->enqueueCreateBuffer(bigBytes));
  ASSIGN_NO_ERR(void *, bigHost, ctx_->getHostAccessiblePointer(big->ptr()));
  const auto *bigBytesHost = static_cast<const uint8_t *>(bigHost);
  const size_t bigOffset = 6;
  const size_t bigLen = bigBytes - 32;
  EXPECT_NO_ERR(ctx_->enqueueSetMemory(*big, kGuard, 1));
  EXPECT_NO_ERR(ctx_->enqueueSetMemory(
      static_cast<char *>(big->ptr()) + bigOffset, bigLen, 0xFF80, 2));
  EXPECT_NO_ERR(ctx_->synchronize());
  for (size_t i = 0; i < bigBytes; ++i) {
    const uint8_t expected = (i >= bigOffset && i < bigOffset + bigLen)
                                 ? patternByte(0xFF80, 2, i - bigOffset)
                                 : kGuard;
    if (bigBytesHost[i] != expected) {
      ADD_FAILURE() << "byte " << i << ": " << int(bigBytesHost[i])
                    << " != " << int(expected);
      break;
    }
  }
}

// Microbenchmark, run on demand with
//   ./bazelw test --config=release
//     //MLRT/unittests/Driver:MetalDeviceContextTest --test_output=streamed
//     --test_arg=--gtest_also_run_disabled_tests
//     --test_arg=--gtest_filter=*SetMemoryFillTiming*
// Times a BF16 -inf fill (a 2-byte pattern whose bytes differ) of 2 MiB and
// 1 GiB on an idle queue: the time for enqueueSetMemory to return, and the
// time until the fill has completed.
TEST_F(MetalDeviceContextTestBase, DISABLED_SetMemoryFillTiming) {
  using Clock = std::chrono::steady_clock;
  constexpr uint64_t kBf16NegInf = 0xFF80;
  struct Case {
    const char *name;
    size_t bytes;
    int iters;
  };
  for (Case c : {Case{"2 MiB", size_t(2) << 20, 50},
                 Case{"1 GiB", size_t(1) << 30, 10}}) {
    ASSIGN_NO_ERR(DeviceBufferRef, buf, ctx_->enqueueCreateBuffer(c.bytes));
    EXPECT_NO_ERR(ctx_->synchronize());

    std::vector<double> enqueueUs, totalUs;
    for (int i = -1; i < c.iters; ++i) {
      Clock::time_point t0 = Clock::now();
      EXPECT_NO_ERR(ctx_->enqueueSetMemory(*buf, kBf16NegInf, 2));
      Clock::time_point t1 = Clock::now();
      EXPECT_NO_ERR(ctx_->synchronize());
      Clock::time_point t2 = Clock::now();
      // The first call pays one-time costs (the helper pipeline's compile,
      // the first touch of the pages), so it is reported apart.
      if (i < 0) {
        std::cout << "[fill-timing] " << c.name << " first call: "
                  << std::chrono::duration<double, std::micro>(t2 - t0).count()
                  << " us\n";
        continue;
      }
      enqueueUs.push_back(
          std::chrono::duration<double, std::micro>(t1 - t0).count());
      totalUs.push_back(
          std::chrono::duration<double, std::micro>(t2 - t0).count());
    }

    ASSIGN_NO_ERR(void *, host, ctx_->getHostAccessiblePointer(buf->ptr()));
    const auto *words = static_cast<const uint16_t *>(host);
    EXPECT_EQ(words[0], kBf16NegInf);
    EXPECT_EQ(words[c.bytes / 2 - 1], kBf16NegInf);

    auto report = [&](const char *what, std::vector<double> &us) {
      std::sort(us.begin(), us.end());
      std::cout << "[fill-timing] " << c.name << " " << what << ": median "
                << us[us.size() / 2] << " us, min " << us.front() << " us, max "
                << us.back() << " us (n=" << us.size() << ")\n";
    };
    report("enqueue", enqueueUs);
    report("enqueue+sync", totalUs);
  }
}

} // namespace M::Driver
