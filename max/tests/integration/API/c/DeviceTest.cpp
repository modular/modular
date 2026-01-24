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
// Device API tests
//
//===----------------------------------------------------------------------===//

#include "max/c/device.h"
#include "Utils.h"
#include "max/c/common.h"
#include "max/c/tensor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace M;

TEST_F(APITest, CreateCPUDevice) {
  M_Device *device = M_newDevice(M_HOST, 0, status);
  EXPECT_SUCCESS(status, "Failed to create CPU device");
  EXPECT_THAT(device, ::testing::NotNull());
  EXPECT_EQ(M_getDeviceType(device), M_HOST);
  EXPECT_EQ(M_getDeviceId(device), 0);
  EXPECT_EQ(M_isHostDevice(device), 1);
  EXPECT_STREQ(M_getDeviceLabel(device), "cpu");
  M_freeDevice(device);
}

TEST_F(APITest, DeviceSynchronize) {
  M_Device *device = M_newDevice(M_HOST, 0, status);
  EXPECT_SUCCESS(status, "Failed to create CPU device");
  EXPECT_THAT(device, ::testing::NotNull());

  // Synchronize should succeed for CPU device
  M_synchronizeDevice(device, status);
  EXPECT_SUCCESS(status, "Failed to synchronize CPU device");

  M_freeDevice(device);
}

TEST_F(APITest, GetAcceleratorCount) {
  // This test just verifies the function doesn't crash
  // The actual count depends on the hardware
  int count = M_getAcceleratorCount();
  EXPECT_GE(count, 0);
}

TEST_F(APITest, CopyTensorToSameDevice) {
  // Create a tensor on CPU and copy it to CPU
  float inputData[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
  int64_t shape[] = {1, 10};
  M_TensorSpec *spec = M_newTensorSpec(shape, 2, M_FLOAT32, "testTensor", host);
  M_AsyncTensorMap *tensorMap = M_newAsyncTensorMap(context);

  M_borrowTensorInto(tensorMap, inputData, spec, status);
  EXPECT_SUCCESS(status, "Failed to add tensor");

  M_AsyncTensor *tensor =
      M_getTensorByNameFrom(tensorMap, "testTensor", status);
  EXPECT_SUCCESS(status, "Failed to get tensor");

  M_Device *cpuDevice = M_newDevice(M_HOST, 0, status);
  EXPECT_SUCCESS(status, "Failed to create CPU device");

  // Copy tensor to CPU device - should succeed
  M_AsyncTensor *copiedTensor = M_copyTensorToDevice(tensor, cpuDevice, status);
  EXPECT_SUCCESS(status, "Failed to copy tensor to device");
  EXPECT_THAT(copiedTensor, ::testing::NotNull())
      << "Copy to device should return a valid tensor";

  EXPECT_EQ(M_getTensorNumElements(copiedTensor), 10u);
  EXPECT_EQ(M_getTensorType(copiedTensor), M_FLOAT32);
  const float *copiedData = (const float *)M_getTensorData(copiedTensor);
  for (int i = 0; i < 10; ++i)
    EXPECT_EQ(inputData[i], copiedData[i]);

  M_freeTensor(tensor);
  // Copied tensor is still available
  EXPECT_EQ(M_getTensorNumElements(copiedTensor), 10u);
  EXPECT_EQ(M_getTensorType(copiedTensor), M_FLOAT32);

  M_freeTensor(copiedTensor);

  M_freeDevice(cpuDevice);
  M_freeAsyncTensorMap(tensorMap);
  M_freeTensorSpec(spec);
}

TEST_F(APITest, CreateDeviceWithInvalidType) {
  // Test that creating a device with an invalid type fails
  M_DeviceType invalidType = static_cast<M_DeviceType>(999);
  M_Device *device = M_newDevice(invalidType, 0, status);

  EXPECT_THAT(device, ::testing::IsNull())
      << "Creating device with invalid type should return null";
  EXPECT_FAILURE(status, "Invalid device type");
}
