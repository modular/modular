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

// Tests for the weights registry C API.
// The MEF file is generated at build time by gen_external_weights_mef.py.
// The test expects EXTERNAL_WEIGHTS_MEF_PATH environment variable.

#include "Utils.h"
#include "max/c/common.h"
#include "max/c/context.h"
#include "max/c/model.h"
#include "max/c/tensor.h"
#include "max/c/types.h"
#include "max/c/weights.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

static const char *mefPath() { return getenv("EXTERNAL_WEIGHTS_MEF_PATH"); }

using M::APITest;

TEST_F(APITest, WeightsRegistryCreateAndFree) {
  float weightData[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  const char *names[1] = {"my_weight"};
  const void *data[1] = {weightData};

  M_WeightsRegistry *registry = M_newWeightsRegistry(names, data, 1, status);
  EXPECT_SUCCESS(status, "Failed to create weights registry");
  EXPECT_THAT(registry, ::testing::NotNull());

  M_freeWeightsRegistry(registry);
}

TEST_F(APITest, WeightsRegistryCreateEmpty) {
  M_WeightsRegistry *registry =
      M_newWeightsRegistry(nullptr, nullptr, 0, status);
  EXPECT_SUCCESS(status, "Failed to create empty weights registry");
  EXPECT_THAT(registry, ::testing::NotNull());

  M_freeWeightsRegistry(registry);
}

TEST_F(APITest, WeightsRegistryFreeNull) {
  // Should be a no-op, not crash.
  M_freeWeightsRegistry(nullptr);
}

TEST_F(APITest, WeightsRegistryNullNamesError) {
  M_WeightsRegistry *registry =
      M_newWeightsRegistry(nullptr, nullptr, 1, status);
  EXPECT_TRUE(M_isError(status));
  EXPECT_THAT(registry, ::testing::IsNull());
}

TEST_F(APITest, WeightsRegistryExecuteWithWeights) {
  // Load the MEF that uses an external weight.
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, mefPath());

  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  EXPECT_SUCCESS(status, "Failed to compile model");
  EXPECT_THAT(compiledModel, ::testing::NotNull());

  // Create weight data: [10, 20, 30, 40]
  float weightData[4] = {10.0f, 20.0f, 30.0f, 40.0f};
  const char *names[1] = {"my_weight"};
  const void *data[1] = {weightData};

  M_WeightsRegistry *registry = M_newWeightsRegistry(names, data, 1, status);
  EXPECT_SUCCESS(status, "Failed to create weights registry");
  EXPECT_THAT(registry, ::testing::NotNull());

  // Initialize model with weights
  M_AsyncModel *model = M_initModel(context, compiledModel, registry, status);
  EXPECT_SUCCESS(status, "Failed to initialize model with weights");
  EXPECT_THAT(model, ::testing::NotNull());

  // Create input data: [1, 2, 3, 4]
  float inputData[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  int64_t shape[1] = {4};
  M_TensorSpec *inputSpec =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input0", host);
  EXPECT_THAT(inputSpec, ::testing::NotNull());

  M_AsyncTensorMap *inputs = M_newAsyncTensorMap(context);
  M_borrowTensorInto(inputs, inputData, inputSpec, status);
  EXPECT_SUCCESS(status, "Failed to add input");

  // Execute: output = input + weight = [11, 22, 33, 44]
  M_AsyncTensorMap *outputs =
      M_executeModelSync(context, model, inputs, status);
  EXPECT_SUCCESS(status, "Failed to execute model");
  EXPECT_THAT(outputs, ::testing::NotNull());

  M_AsyncTensor *outputTensor =
      M_getTensorByNameFrom(outputs, "output0", status);
  EXPECT_SUCCESS(status, "Failed to retrieve output tensor");
  EXPECT_THAT(outputTensor, ::testing::NotNull());

  const float *outputData = (const float *)M_getTensorData(outputTensor);
  EXPECT_THAT(outputData, ::testing::NotNull());
  EXPECT_EQ(M_getTensorNumElements(outputTensor), 4);

  EXPECT_EQ(outputData[0], 11.0f);
  EXPECT_EQ(outputData[1], 22.0f);
  EXPECT_EQ(outputData[2], 33.0f);
  EXPECT_EQ(outputData[3], 44.0f);

  M_freeTensor(outputTensor);
  M_freeAsyncTensorMap(outputs);
  M_freeAsyncTensorMap(inputs);
  M_freeTensorSpec(inputSpec);
  M_freeModel(model);
  M_freeWeightsRegistry(registry);
  M_freeCompiledModel(compiledModel);
}
