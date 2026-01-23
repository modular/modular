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

// Test that loads a MEF file and verifies execution via the C API.
// The MEF file is generated at build time by gen_vector_add_mef.py.
// The test expects TEST_MEF_PATH environment variable to contain the path.

#include "Utils.h"
#include "max/c/common.h"
#include "max/c/context.h"
#include "max/c/model.h"
#include "max/c/tensor.h"
#include "max/c/types.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cstdio>

static const char *mefPath() { return getenv("TEST_MEF_PATH"); }

using M::APITest;

TEST_F(APITest, CompileModel) {
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, mefPath());

  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  EXPECT_SUCCESS(status, "Failed to compile model");
  EXPECT_THAT(compileConfig, ::testing::IsNull());
  EXPECT_THAT(compiledModel, ::testing::NotNull());

  M_freeCompiledModel(compiledModel);
}

// TODO(EMF-XXXX): Hits assert freeing the runtime
// TEST_F(APITest, LoadModel) {
//   M_CompileConfig *compileConfig = M_newCompileConfig();
//   M_setModelPath(compileConfig, mefPath());

//   M_AsyncCompiledModel *compiledModel =
//       M_compileModelSync(context, &compileConfig, status);
//   EXPECT_SUCCESS(status, "Failed to compile model");
//   EXPECT_THAT(compileConfig, ::testing::IsNull());
//   EXPECT_THAT(compiledModel, ::testing::NotNull());

//   M_AsyncModel *model = M_initModel(context, compiledModel, NULL, status);
//   EXPECT_SUCCESS(status, "Failed to initialize model");
//   EXPECT_THAT(model, ::testing::NotNull());

//   // Just load, don't execute - verify cleanup works
//   M_freeModel(model);
//   M_freeCompiledModel(compiledModel);
// }

TEST_F(APITest, ExecuteModel) {
  // Load the MEF file
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, mefPath());

  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  EXPECT_SUCCESS(status, "Failed to compile model");
  EXPECT_THAT(compileConfig, ::testing::IsNull());
  EXPECT_THAT(compiledModel, ::testing::NotNull());

  M_AsyncModel *model = M_initModel(context, compiledModel, NULL, status);
  EXPECT_SUCCESS(status, "Failed to initialize model");
  EXPECT_THAT(model, ::testing::NotNull());

  // Create tensor specifications for inputs
  int64_t shape[1] = {8};
  M_TensorSpec *inputSpec1 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input0", host);
  M_TensorSpec *inputSpec2 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input1", host);
  EXPECT_THAT(inputSpec1, ::testing::NotNull());
  EXPECT_THAT(inputSpec2, ::testing::NotNull());

  // Create input data - two vectors of 8 elements each
  // Expected output: [1+8, 2+7, 3+6, 4+5, 5+4, 6+3, 7+2, 8+1] =
  // [9,9,9,9,9,9,9,9]
  float vector1[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float vector2[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

  M_AsyncTensorMap *inputs = M_newAsyncTensorMap(context);
  EXPECT_THAT(inputs, ::testing::NotNull());

  M_borrowTensorInto(inputs, vector1, inputSpec1, status);
  EXPECT_SUCCESS(status, "Failed to add first input");
  M_borrowTensorInto(inputs, vector2, inputSpec2, status);
  EXPECT_SUCCESS(status, "Failed to add second input");

  M_AsyncTensorMap *outputs =
      M_executeModelSync(context, model, inputs, status);
  EXPECT_SUCCESS(status, "Failed to execute model");
  EXPECT_THAT(outputs, ::testing::NotNull());

  // Get the output tensor
  M_AsyncTensor *outputTensor =
      M_getTensorByNameFrom(outputs, "output0", status);
  EXPECT_SUCCESS(status, "Failed to retrieve output tensor");
  EXPECT_THAT(outputTensor, ::testing::NotNull());

  const float *outputData = (const float *)M_getTensorData(outputTensor);
  EXPECT_THAT(outputData, ::testing::NotNull());
  EXPECT_EQ(M_getTensorNumElements(outputTensor), 8);

  for (auto i = 0; i < 8; i++)
    EXPECT_EQ(outputData[i], 9.0f);

  M_freeTensor(outputTensor);
  M_freeAsyncTensorMap(outputs);
  M_freeAsyncTensorMap(inputs);
  M_freeTensorSpec(inputSpec2);
  M_freeTensorSpec(inputSpec1);
  M_freeModel(model);
  M_freeCompiledModel(compiledModel);
}

TEST_F(APITest, ExecuteModelMultipleTimes) {
  // Verify we can execute the same model multiple times with the same inputs
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, mefPath());

  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  EXPECT_SUCCESS(status, "Failed to compile model");

  M_AsyncModel *model = M_initModel(context, compiledModel, NULL, status);
  EXPECT_SUCCESS(status, "Failed to initialize model");

  int64_t shape[1] = {8};
  M_TensorSpec *inputSpec1 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input0", host);
  M_TensorSpec *inputSpec2 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input1", host);

  float vector1[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float vector2[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

  M_AsyncTensorMap *inputs = M_newAsyncTensorMap(context);
  M_borrowTensorInto(inputs, vector1, inputSpec1, status);
  EXPECT_SUCCESS(status, "Failed to add first input");
  M_borrowTensorInto(inputs, vector2, inputSpec2, status);
  EXPECT_SUCCESS(status, "Failed to add second input");

  // Execute twice with the same inputs
  for (int run = 0; run < 2; ++run) {
    M_AsyncTensorMap *outputs =
        M_executeModelSync(context, model, inputs, status);
    EXPECT_SUCCESS(status, "Failed to execute model on run");
    EXPECT_THAT(outputs, ::testing::NotNull());

    M_AsyncTensor *outputTensor =
        M_getTensorByNameFrom(outputs, "output0", status);
    EXPECT_SUCCESS(status, "Failed to retrieve output tensor");

    const float *outputData = (const float *)M_getTensorData(outputTensor);
    for (auto i = 0; i < 8; i++)
      EXPECT_EQ(outputData[i], 9.0f);

    M_freeTensor(outputTensor);
    M_freeAsyncTensorMap(outputs);
  }

  // Verify input data is still intact after executions
  M_AsyncTensor *input1 = M_getTensorByNameFrom(inputs, "input0", status);
  EXPECT_SUCCESS(status, "Failed to retrieve input tensor after execution");
  const float *input1Data = (const float *)M_getTensorData(input1);
  for (auto i = 0; i < 8; i++)
    EXPECT_EQ(input1Data[i], vector1[i]);
  M_freeTensor(input1);

  M_freeAsyncTensorMap(inputs);
  M_freeTensorSpec(inputSpec2);
  M_freeTensorSpec(inputSpec1);
  M_freeModel(model);
  M_freeCompiledModel(compiledModel);
}

TEST_F(APITest, ExecuteModelInputsInDifferentOrder) {
  // Verify inputs can be provided in a different order than MEF expects
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, mefPath());

  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  EXPECT_SUCCESS(status, "Failed to compile model");

  M_AsyncModel *model = M_initModel(context, compiledModel, NULL, status);
  EXPECT_SUCCESS(status, "Failed to initialize model");

  int64_t shape[1] = {8};
  M_TensorSpec *inputSpec0 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input0", host);
  M_TensorSpec *inputSpec1 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input1", host);

  float vector1[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float vector2[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

  M_AsyncTensorMap *inputs = M_newAsyncTensorMap(context);

  // Add inputs in REVERSE order (input1 first, then input0)
  M_borrowTensorInto(inputs, vector2, inputSpec1, status);
  EXPECT_SUCCESS(status, "Failed to add input1");
  M_borrowTensorInto(inputs, vector1, inputSpec0, status);
  EXPECT_SUCCESS(status, "Failed to add input0");

  M_AsyncTensorMap *outputs =
      M_executeModelSync(context, model, inputs, status);
  EXPECT_SUCCESS(status, "Failed to execute model");
  EXPECT_THAT(outputs, ::testing::NotNull());

  M_AsyncTensor *outputTensor =
      M_getTensorByNameFrom(outputs, "output0", status);
  EXPECT_SUCCESS(status, "Failed to retrieve output tensor");

  const float *outputData = (const float *)M_getTensorData(outputTensor);
  for (auto i = 0; i < 8; i++)
    EXPECT_EQ(outputData[i], 9.0f);

  M_freeTensor(outputTensor);
  M_freeAsyncTensorMap(outputs);
  M_freeAsyncTensorMap(inputs);
  M_freeTensorSpec(inputSpec1);
  M_freeTensorSpec(inputSpec0);
  M_freeModel(model);
  M_freeCompiledModel(compiledModel);
}

TEST_F(APITest, ExecuteModelMissingInput) {
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, mefPath());

  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  EXPECT_SUCCESS(status, "Failed to compile model");

  M_AsyncModel *model = M_initModel(context, compiledModel, NULL, status);
  EXPECT_SUCCESS(status, "Failed to initialize model");

  int64_t shape[1] = {8};
  M_TensorSpec *inputSpec1 =
      M_newTensorSpec(shape, 1, M_FLOAT32, "input0", host);
  // intentionally don't adding "input1"

  float vector1[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  M_AsyncTensorMap *inputs = M_newAsyncTensorMap(context);
  M_borrowTensorInto(inputs, vector1, inputSpec1, status);
  EXPECT_SUCCESS(status, "Failed to add first input");

  M_AsyncTensorMap *outputs =
      M_executeModelSync(context, model, inputs, status);
  EXPECT_FAILURE(status, "input named 'input1' not provided");
  EXPECT_THAT(outputs, ::testing::IsNull());

  M_freeAsyncTensorMap(inputs);
  M_freeTensorSpec(inputSpec1);
  M_freeModel(model);
  M_freeCompiledModel(compiledModel);
}
