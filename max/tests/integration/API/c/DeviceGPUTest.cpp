//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// GPU Device API tests - requires GPU hardware
//
//===----------------------------------------------------------------------===//

#include "Utils.h"
#include "max/c/common.h"
#include "max/c/device.h"
#include "max/c/tensor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cstring>
#include <vector>

using namespace M;

TEST_F(GPUTest, CreateGPUDevice) {
  EXPECT_EQ(M_getDeviceType(gpuDevice), M_ACCELERATOR);
  EXPECT_EQ(M_getDeviceId(gpuDevice), 0);
  EXPECT_EQ(M_isHostDevice(gpuDevice), 0);
  EXPECT_STREQ(M_getDeviceLabel(gpuDevice), "gpu");
}

TEST_F(GPUTest, SynchronizeGPUDevice) {
  M_synchronizeDevice(gpuDevice, status);
  EXPECT_SUCCESS(status, "Failed to synchronize GPU device");
}

TEST_F(GPUTest, MoveTensorToGPU) {
  // Create a tensor on host
  std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f,
                                  5.0f, 6.0f, 7.0f, 8.0f};
  int64_t shape[] = {2, 4};
  M_TensorSpec *spec = M_newTensorSpec(shape, 2, M_FLOAT32, "testTensor", host);
  M_AsyncTensorMap *tensorMap = M_newAsyncTensorMap(context);

  M_borrowTensorInto(tensorMap, inputData.data(), spec, status);
  EXPECT_SUCCESS(status, "Failed to borrow tensor");

  M_AsyncTensor *hostTensor =
      M_getTensorByNameFrom(tensorMap, "testTensor", status);
  EXPECT_SUCCESS(status, "Failed to get host tensor");
  EXPECT_THAT(hostTensor, ::testing::NotNull());

  // Verify tensor is on host
  const M_Device *tensorDevice = M_getTensorDevice(hostTensor);
  EXPECT_THAT(tensorDevice, ::testing::NotNull());
  EXPECT_EQ(M_isHostDevice(tensorDevice), 1);

  // Move tensor to GPU
  M_AsyncTensor *gpuTensor =
      M_copyTensorToDevice(hostTensor, gpuDevice, status);
  EXPECT_SUCCESS(status, "Failed to copy tensor to GPU");
  EXPECT_THAT(gpuTensor, ::testing::NotNull());

  // Verify tensor is on GPU
  const M_Device *gpuTensorDevice = M_getTensorDevice(gpuTensor);
  EXPECT_THAT(gpuTensorDevice, ::testing::NotNull());
  EXPECT_EQ(M_isHostDevice(gpuTensorDevice), 0);
  EXPECT_EQ(M_getDeviceType(gpuTensorDevice), M_ACCELERATOR);
  EXPECT_EQ(M_getDeviceId(gpuTensorDevice), M_getDeviceId(gpuDevice));

  M_freeTensor(gpuTensor);
  M_freeTensor(hostTensor);
  M_freeAsyncTensorMap(tensorMap);
  M_freeTensorSpec(spec);
}

TEST_F(GPUTest, MoveTensorFromGPU) {
  // Create a tensor and move it to GPU first
  std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f};
  int64_t shape[] = {4};
  M_TensorSpec *spec =
      M_newTensorSpec(shape, 1, M_FLOAT32, "testTensor", gpuDevice);
  M_AsyncTensorMap *tensorMap = M_newAsyncTensorMap(context);

  M_borrowTensorInto(tensorMap, inputData.data(), spec, status);
  EXPECT_SUCCESS(status, "Failed to borrow tensor");

  M_AsyncTensor *gpuTensor =
      M_getTensorByNameFrom(tensorMap, "testTensor", status);
  EXPECT_SUCCESS(status, "Failed to get gpu tensor");
  M_Device *gpuTensorDevice = M_getTensorDevice(gpuTensor);
  EXPECT_EQ(M_isHostDevice(gpuTensorDevice), 0);
  EXPECT_EQ(M_getDeviceType(gpuTensorDevice), M_ACCELERATOR);
  EXPECT_EQ(M_getDeviceId(gpuTensorDevice), M_getDeviceId(gpuDevice));
  M_freeDevice(gpuTensorDevice);

  M_AsyncTensor *hostTensor = M_copyTensorToDevice(gpuTensor, host, status);
  EXPECT_SUCCESS(status, "Failed to copy tensor back to host");
  EXPECT_THAT(hostTensor, ::testing::NotNull());

  M_Device *device = M_getTensorDevice(hostTensor);
  EXPECT_THAT(device, ::testing::NotNull());
  EXPECT_EQ(M_isHostDevice(device), 1);
  M_freeDevice(device);

  M_freeTensor(hostTensor);
  M_freeTensor(gpuTensor);
  M_freeAsyncTensorMap(tensorMap);
  M_freeTensorSpec(spec);
}

TEST_F(GPUTest, RoundTripTensorData) {
  // Create a tensor with known data
  std::vector<float> originalData = {1.0f, 2.0f, 3.0f, 4.0f,
                                     5.0f, 6.0f, 7.0f, 8.0f};
  int64_t shape[] = {2, 4};
  M_TensorSpec *spec = M_newTensorSpec(shape, 2, M_FLOAT32, "testTensor", host);
  M_AsyncTensorMap *tensorMap = M_newAsyncTensorMap(context);

  M_borrowTensorInto(tensorMap, originalData.data(), spec, status);
  EXPECT_SUCCESS(status, "Failed to borrow tensor");

  M_AsyncTensor *hostTensor =
      M_getTensorByNameFrom(tensorMap, "testTensor", status);
  EXPECT_SUCCESS(status, "Failed to get host tensor");

  M_AsyncTensor *gpuTensor =
      M_copyTensorToDevice(hostTensor, gpuDevice, status);
  EXPECT_SUCCESS(status, "Failed to copy tensor to GPU");

  M_AsyncTensor *roundTripped = M_copyTensorToDevice(gpuTensor, host, status);
  EXPECT_SUCCESS(status, "Failed to copy tensor back to host");

  EXPECT_EQ(M_getTensorNumElements(roundTripped), originalData.size());
  const float *resultData =
      static_cast<const float *>(M_getTensorData(roundTripped));
  EXPECT_THAT(resultData, ::testing::NotNull());

  for (size_t i = 0; i < originalData.size(); ++i)
    EXPECT_FLOAT_EQ(resultData[i], originalData[i])
        << "Data mismatch at index " << i;

  M_freeTensor(roundTripped);
  M_freeTensor(gpuTensor);
  M_freeTensor(hostTensor);
  M_freeAsyncTensorMap(tensorMap);
  M_freeTensorSpec(spec);
}

TEST_F(GPUTest, TensorSpecWithGPUDevice) {
  int64_t shape[] = {1, 28, 28};
  M_TensorSpec *spec =
      M_newTensorSpec(shape, 3, M_FLOAT32, "gpuTensor", gpuDevice);
  EXPECT_THAT(spec, ::testing::NotNull());

  EXPECT_EQ(M_getDeviceTypeFromSpec(spec), M_ACCELERATOR);
  EXPECT_EQ(M_getDeviceIdFromSpec(spec), M_getDeviceId(gpuDevice));

  M_freeTensorSpec(spec);
}

TEST_F(GPUTest, BorrowTensorIntoWithGPUSpec) {
  // Create a tensor spec on GPU device
  std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                  6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
  int64_t shape[] = {2, 5};
  M_TensorSpec *spec =
      M_newTensorSpec(shape, 2, M_FLOAT32, "gpuTensor", gpuDevice);
  EXPECT_THAT(spec, ::testing::NotNull());

  // Verify the spec has GPU device info
  EXPECT_EQ(M_getDeviceTypeFromSpec(spec), M_ACCELERATOR);
  EXPECT_EQ(M_getDeviceIdFromSpec(spec), 0);

  // Borrow tensor into the tensor map with GPU spec
  M_AsyncTensorMap *tensorMap = M_newAsyncTensorMap(context);
  M_borrowTensorInto(tensorMap, inputData.data(), spec, status);
  EXPECT_SUCCESS(status, "Failed to borrow tensor with GPU spec");

  // Retrieve the tensor and verify its properties
  M_AsyncTensor *tensor = M_getTensorByNameFrom(tensorMap, "gpuTensor", status);
  EXPECT_SUCCESS(status, "Failed to get GPU tensor by name");
  EXPECT_THAT(tensor, ::testing::NotNull());

  EXPECT_EQ(M_getTensorNumElements(tensor), 10u);
  EXPECT_EQ(M_getTensorType(tensor), M_FLOAT32);

  // Verify the tensor is associated with a GPU device
  M_Device *tensorDevice = M_getTensorDevice(tensor);
  EXPECT_THAT(tensorDevice, ::testing::NotNull());
  EXPECT_EQ(M_getDeviceType(tensorDevice), M_ACCELERATOR);
  EXPECT_EQ(M_getDeviceId(tensorDevice), 0);
  EXPECT_EQ(M_isHostDevice(tensorDevice), 0);

  M_freeDevice(tensorDevice);
  M_freeTensor(tensor);
  M_freeAsyncTensorMap(tensorMap);
  M_freeTensorSpec(spec);
}

TEST_F(GPUTest, ExecuteModelWithGPUTensors) {
  // This test requires the MEF file from the MEFTest
  const char *mefPath = getenv("TEST_MEF_PATH");

  // Load the MEF file
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, mefPath);

  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  EXPECT_SUCCESS(status, "Failed to compile model");
  EXPECT_THAT(compiledModel, ::testing::NotNull());

  M_AsyncModel *model = M_initModel(context, compiledModel, nullptr, status);
  EXPECT_SUCCESS(status, "Failed to initialize model");
  EXPECT_THAT(model, ::testing::NotNull());

  // Create input tensors on host first
  float vector1[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float vector2[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

  int64_t shape[1] = {8};
  M_TensorSpec *inputSpec1 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input0", gpuDevice);
  M_TensorSpec *inputSpec2 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input1", gpuDevice);

  M_AsyncTensorMap *inputs = M_newAsyncTensorMap(context);
  M_borrowTensorInto(inputs, vector1, inputSpec1, status);
  EXPECT_SUCCESS(status, "Failed to add first input");
  M_borrowTensorInto(inputs, vector2, inputSpec2, status);
  EXPECT_SUCCESS(status, "Failed to add second input");

  M_AsyncTensorMap *outputs =
      M_executeModelSync(context, model, inputs, status);
  EXPECT_SUCCESS(status, "Failed to execute model");
  EXPECT_THAT(outputs, ::testing::NotNull());

  M_AsyncTensor *outputTensor =
      M_getTensorByNameFrom(outputs, "output0", status);
  EXPECT_SUCCESS(status, "Failed to retrieve output tensor");

  EXPECT_EQ(M_getTensorNumElements(outputTensor), 8u);

  M_Device *outputDevice = M_getTensorDevice(outputTensor);
  EXPECT_EQ(M_getDeviceType(outputDevice), M_ACCELERATOR);
  M_freeDevice(outputDevice);

  M_AsyncTensor *outputHostTensor =
      M_copyTensorToDevice(outputTensor, host, status);
  EXPECT_SUCCESS(status, "Failed to transfer result to host");
  const float *outputData =
      static_cast<const float *>(M_getTensorData(outputHostTensor));
  EXPECT_THAT(outputData, ::testing::NotNull());

  for (int i = 0; i < 8; i++)
    EXPECT_FLOAT_EQ(outputData[i], 9.0f);

  M_freeTensor(outputTensor);
  M_freeTensor(outputHostTensor);
  M_freeAsyncTensorMap(outputs);
  M_freeAsyncTensorMap(inputs);
  M_freeTensorSpec(inputSpec2);
  M_freeTensorSpec(inputSpec1);
  M_freeModel(model);
  M_freeCompiledModel(compiledModel);
}

TEST_F(GPUTest, ExecuteModelWithIncorrectDeviceTensors) {
  // This test requires the MEF file from the MEFTest
  const char *mefPath = getenv("TEST_MEF_PATH");

  // Load the MEF file
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, mefPath);

  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  EXPECT_SUCCESS(status, "Failed to compile model");
  EXPECT_THAT(compiledModel, ::testing::NotNull());

  M_AsyncModel *model = M_initModel(context, compiledModel, nullptr, status);
  EXPECT_SUCCESS(status, "Failed to initialize model");
  EXPECT_THAT(model, ::testing::NotNull());

  float vector1[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float vector2[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

  int64_t shape[1] = {8};
  M_TensorSpec *inputSpec1 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input0", host);
  M_TensorSpec *inputSpec2 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input1", host);

  M_AsyncTensorMap *inputs = M_newAsyncTensorMap(context);
  M_borrowTensorInto(inputs, vector1, inputSpec1, status);
  EXPECT_SUCCESS(status, "Failed to add first input");
  M_borrowTensorInto(inputs, vector2, inputSpec2, status);
  EXPECT_SUCCESS(status, "Failed to add second input");

  M_AsyncTensorMap *outputs =
      M_executeModelSync(context, model, inputs, status);
  EXPECT_FAILURE(status, "input 'input0' on incorrect device");
  EXPECT_THAT(outputs, ::testing::IsNull());

  M_freeAsyncTensorMap(inputs);
  M_freeTensorSpec(inputSpec2);
  M_freeTensorSpec(inputSpec1);
  M_freeModel(model);
  M_freeCompiledModel(compiledModel);
}
