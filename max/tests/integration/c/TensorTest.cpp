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
// Tensor manipulation tests
//
//===----------------------------------------------------------------------===//

#include "max/c/tensor.h"
#include "Utils.h"
#include "max/c/common.h"
#include "max/c/context.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <vector>

using namespace M;

TEST_F(APITest, GetTensorSpecDetails) {
  int64_t shape[] = {1, 28, 28};
  M_TensorSpec *inputSpec =
      M_newTensorSpec(shape, std::size(shape), M_FLOAT32, "testTensor", host);
  EXPECT_EQ(0, M_isDynamicRanked(inputSpec));
  ASSERT_EQ(static_cast<int64_t>(std::size(shape)), M_getRank(inputSpec));
  EXPECT_EQ(M_FLOAT32, M_getDtype(inputSpec));
  for (size_t i = 0, e = std::size(shape); i < e; ++i) {
    EXPECT_EQ(M_getDimAt(inputSpec, i), shape[i]);
  }
  M_freeTensorSpec(inputSpec);
}

TEST_F(APITest, TensorSpecDetailsWithNullptr) {
  ASSERT_EQ(0u, M_getRank(nullptr));
  EXPECT_EQ(M_UNKNOWN, M_getDtype(nullptr));
  EXPECT_EQ(M_getDimAt(nullptr, 1), 0);
}

TEST_F(APITest, TensorSpecDetailsIndexingError) {
  int64_t shape[] = {1, 28, 28};
  M_TensorSpec *inputSpec =
      M_newTensorSpec(shape, std::size(shape), M_FLOAT32, "testTensor", host);
  EXPECT_EQ(M_getDimAt(inputSpec, 5), 0);
  M_freeTensorSpec(inputSpec);
}

TEST_F(APITest, GetTensorByName) {
  int numInputs = 10;
  std::vector<float> inputTensor(numInputs);
  int64_t shape[] = {1, 10};
  M_TensorSpec *inputSpec1 =
      M_newTensorSpec(shape, std::size(shape), M_FLOAT32, "testTensor1", host);
  M_TensorSpec *inputSpec2 =
      M_newTensorSpec(shape, std::size(shape), M_FLOAT32, "testTensor2", host);
  M_AsyncTensorMap *tensorMap = M_newAsyncTensorMap(context);

  M_borrowTensorInto(tensorMap, inputTensor.data(), inputSpec1, status);
  M_borrowTensorInto(tensorMap, inputTensor.data(), inputSpec2, status);

  const char *name1 = "testTensor1";
  const char *name2 = "testTensor2";

  M_AsyncTensor *tensor1 = M_getTensorByNameFrom(tensorMap, name1, status);
  EXPECT_SUCCESS(status, "Failed to get tensor1 by name");
  EXPECT_EQ(M_FLOAT32, M_getTensorType(tensor1));
  EXPECT_EQ(10u, M_getTensorNumElements(tensor1));

  M_TensorSpec *tensorSpec = M_getTensorSpec(tensor1);
  ASSERT_NE(nullptr, tensorSpec);
  EXPECT_EQ(M_FLOAT32, M_getDtype(tensorSpec));
  EXPECT_EQ(std::size(shape), (size_t)M_getRank(tensorSpec));
  for (size_t i = 0, e = std::size(shape); i < e; ++i) {
    EXPECT_EQ(M_getDimAt(tensorSpec, i), shape[i]);
  }

  const float *tensorData = (const float *)(M_getTensorData(tensor1));
  EXPECT_TRUE(std::equal(tensorData, tensorData + inputTensor.size(),
                         inputTensor.cbegin()));

  M_AsyncTensor *tensor2 = M_getTensorByNameFrom(tensorMap, name2, status);
  EXPECT_EQ(M_FLOAT32, M_getTensorType(tensor2));
  EXPECT_EQ(10u, M_getTensorNumElements(tensor2));

  M_freeTensor(tensor2);
  M_freeTensor(tensor1);
  M_freeAsyncTensorMap(tensorMap);
  M_freeTensorSpec(tensorSpec);
  M_freeTensorSpec(inputSpec2);
  M_freeTensorSpec(inputSpec1);
}

TEST_F(APITest, QueryingInvalidTensorError) {
  EXPECT_EQ(nullptr, M_getTensorData(nullptr));
  EXPECT_EQ(nullptr, M_getTensorSpec(nullptr));
}

TEST_F(APITest, GetTensorByNameWithInvalidTensorMap) {
  // oops, user forgot to create inputs and put them into a tensor array
  // before trying to retrieve a tensor out
  M_AsyncTensorMap *tensorMap = nullptr;
  M_AsyncTensor *tensor =
      M_getTensorByNameFrom(tensorMap, "testTensor", status);

  EXPECT_EQ(nullptr, tensor);
  EXPECT_FAILURE(status, "Invalid 'M_AsyncTensorMap'*.");
}

TEST_F(APITest, GetTensorByNameWithInvalidName) {
  int numInputs = 2;
  std::vector<float> inputTensor(numInputs);
  int64_t shape[] = {1, 10};
  M_TensorSpec *inputSpec1 =
      M_newTensorSpec(shape, std::size(shape), M_FLOAT32, "testTensor1", host);
  M_TensorSpec *inputSpec2 =
      M_newTensorSpec(shape, std::size(shape), M_FLOAT32, "testTensor2", host);
  M_AsyncTensorMap *tensorMap = M_newAsyncTensorMap(context);

  M_borrowTensorInto(tensorMap, inputTensor.data(), inputSpec1, status);
  M_borrowTensorInto(tensorMap, inputTensor.data(), inputSpec2, status);

  M_AsyncTensor *tensor =
      M_getTensorByNameFrom(tensorMap, "testTensorBadName", status);
  EXPECT_EQ(nullptr, tensor);
  EXPECT_FAILURE(status,
                 "No tensor with name testTensorBadName in tensor map.");

  M_freeAsyncTensorMap(tensorMap);
  M_freeTensorSpec(inputSpec2);
  M_freeTensorSpec(inputSpec1);
}

TEST_F(APITest, BorrowTensorByNameInvalidTensorMap) {
  int numInputs = 1;
  std::vector<float> inputTensor(numInputs);
  int64_t shape[] = {1, 28, 28};
  M_TensorSpec *inputSpec =
      M_newTensorSpec(shape, std::size(shape), M_FLOAT32, "testTensor", host);
  M_AsyncTensorMap *tensorMap =
      nullptr; // oops, user forgot to create a tensor array
  M_borrowTensorInto(tensorMap, inputTensor.data(), inputSpec, status);
  EXPECT_FAILURE(status, "Invalid tensor array.");
  M_freeTensorSpec(inputSpec);
}

TEST_F(APITest, BorrowTensorByNameInvalidInput) {
  int64_t shape[] = {1, 28, 28};
  M_TensorSpec *inputSpec =
      M_newTensorSpec(shape, std::size(shape), M_FLOAT32, "testTensor", host);
  M_AsyncTensorMap *tensorMap = M_newAsyncTensorMap(context);
  void *inputTensor = nullptr;
  M_borrowTensorInto(tensorMap, inputTensor, inputSpec, status);
  EXPECT_FAILURE(status, "Invalid input tensor.");
  M_freeAsyncTensorMap(tensorMap);
  M_freeTensorSpec(inputSpec);
}

TEST_F(APITest, BorrowTensorByNameInvalidTensorSpec) {
  int numInputs = 1;
  std::vector<float> inputTensor(numInputs);
  M_TensorSpec *inputSpec = nullptr;
  M_AsyncTensorMap *tensorMap = M_newAsyncTensorMap(context);
  M_borrowTensorInto(tensorMap, inputTensor.data(), inputSpec, status);
  EXPECT_FAILURE(status, "Invalid tensor spec.");
  M_freeAsyncTensorMap(tensorMap);
}

TEST_F(APITest, TensorSpecDevice) {
  int64_t shape[] = {1, 28, 28};
  M_TensorSpec *spec = M_newTensorSpec(shape, 3, M_FLOAT32, "testTensor", host);
  EXPECT_THAT(spec, ::testing::NotNull());

  // Check that device properties are stored in the spec
  EXPECT_EQ(M_getDeviceTypeFromSpec(spec), M_getDeviceType(host));
  EXPECT_EQ(M_getDeviceIdFromSpec(spec), M_getDeviceId(host));

  // Check basic tensor spec properties still work
  EXPECT_EQ(M_getRank(spec), 3);
  EXPECT_EQ(M_getDtype(spec), M_FLOAT32);

  M_freeTensorSpec(spec);
}

TEST_F(APITest, TensorSpecWithNullDevice) {
  int64_t shape[] = {1, 28, 28};
  M_TensorSpec *spec =
      M_newTensorSpec(shape, 3, M_FLOAT32, "testTensor", nullptr);
  EXPECT_THAT(spec, ::testing::IsNull());

  M_freeTensorSpec(spec);
}

TEST_F(APITest, GetTensorDeviceNull) {
  M_Device *device = M_getTensorDevice(nullptr);
  EXPECT_EQ(device, nullptr);
}

TEST_F(APITest, GetTensorDeviceReturnsDevice) {
  // Create a tensor and verify we can get its device
  float inputData[10] = {0};
  int64_t shape[] = {1, 10};
  M_TensorSpec *spec = M_newTensorSpec(shape, 2, M_FLOAT32, "testTensor", host);
  M_AsyncTensorMap *tensorMap = M_newAsyncTensorMap(context);

  M_borrowTensorInto(tensorMap, inputData, spec, status);
  EXPECT_SUCCESS(status, "Failed to add tensor");

  M_AsyncTensor *tensor =
      M_getTensorByNameFrom(tensorMap, "testTensor", status);
  EXPECT_SUCCESS(status, "Failed to get tensor");
  EXPECT_THAT(tensor, ::testing::NotNull());

  // A tensor should have a device (at minimum, CPU)
  M_Device *device = M_getTensorDevice(tensor);
  EXPECT_THAT(device, ::testing::NotNull())
      << "Tensor should have an associated device";
  if (device) {
    EXPECT_EQ(M_isHostDevice(device), 1) << "Borrowed tensor should be on CPU";
    M_freeDevice(device);
  }

  M_freeTensor(tensor);
  M_freeAsyncTensorMap(tensorMap);
  M_freeTensorSpec(spec);
}
