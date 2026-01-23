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

// Test symbolic dimension handling in the C API.
// Tests verify that models with static/symbolic dimensions properly validate
// input shapes and execute correctly with various input configurations.

#include "Support/PlatformUtils.h"
#include "Utils.h"
#include "max/c/common.h"
#include "max/c/context.h"
#include "max/c/model.h"
#include "max/c/tensor.h"
#include "max/c/types.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cstdio>

static const char *staticMefPath() { return getenv("STATIC_MEF_PATH"); }
static const char *symbolicMefPath() { return getenv("SYMBOLIC_MEF_PATH"); }
static const char *sameSymbolicMefPath() {
  return getenv("SAME_SYMBOLIC_MEF_PATH");
}
static const char *algebraicMefPath() { return getenv("ALGEBRAIC_MEF_PATH"); }

using M::APITest;

TEST_F(APITest, StaticDimensionMismatch) {
#if MODULAR_MACOS
  GTEST_SKIP() << "Disabled on macOS due to platform-specific issues";
#endif
  // Load static_dimensions.mef expecting [8, 10]
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, staticMefPath());

  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  EXPECT_SUCCESS(status, "Failed to compile model");

  M_AsyncModel *model = M_initModel(context, compiledModel, NULL, status);
  EXPECT_SUCCESS(status, "Failed to initialize model");

  int64_t wrong_shape[] = {8, 12};
  M_TensorSpec *inputSpec = M_newTensorSpec(wrong_shape, std::size(wrong_shape),
                                            M_FLOAT32, "input0", host);

  float data[8 * 12];

  M_AsyncTensorMap *inputs = M_newAsyncTensorMap(context);
  M_borrowTensorInto(inputs, data, inputSpec, status);
  EXPECT_SUCCESS(status, "Failed to borrow tensor");

  M_AsyncTensorMap *outputs =
      M_executeModelSync(context, model, inputs, status);
  EXPECT_FAILURE(status, "input 'input0': Buffer of type [(8, 12), f32] does "
                         "not match expected type [(8, 10), f32]");
  EXPECT_THAT(outputs, ::testing::IsNull());

  // Cleanup
  M_freeAsyncTensorMap(inputs);
  M_freeTensorSpec(inputSpec);
  M_freeModel(model);
  M_freeCompiledModel(compiledModel);
}

TEST_F(APITest, SymbolicDimensionDifferentShapes) {
#if MODULAR_MACOS
  GTEST_SKIP() << "Disabled on macOS due to platform-specific issues";
#endif
  // Load symbolic_dimensions.mef with ["batch", 10]
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, symbolicMefPath());

  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  EXPECT_SUCCESS(status, "Failed to compile model");

  M_AsyncModel *model = M_initModel(context, compiledModel, NULL, status);
  EXPECT_SUCCESS(status, "Failed to initialize model");

  // Create input data
  float data[10 * 10];

  {
    // Test with batch size 8
    int64_t shape[] = {8, 10};
    M_TensorSpec *inputSpec =
        M_newTensorSpec(shape, std::size(shape), M_FLOAT32, "input0", host);

    M_AsyncTensorMap *inputs = M_newAsyncTensorMap(context);
    M_borrowTensorInto(inputs, data, inputSpec, status);
    EXPECT_SUCCESS(status, "Failed to borrow tensor");

    M_AsyncTensorMap *outputs =
        M_executeModelSync(context, model, inputs, status);
    EXPECT_SUCCESS(status, "Failed to execute model");
    EXPECT_THAT(outputs, ::testing::NotNull());

    // Verify output shape
    M_AsyncTensor *outputTensor =
        M_getTensorByNameFrom(outputs, "output0", status);
    EXPECT_SUCCESS(status, "Failed to get output tensor");
    EXPECT_THAT(outputTensor, ::testing::NotNull());

    M_TensorSpec *outputSpec = M_getTensorSpec(outputTensor);
    EXPECT_EQ(M_getDimAt(outputSpec, 0), 8);

    M_freeTensorSpec(outputSpec);
    M_freeTensor(outputTensor);
    M_freeAsyncTensorMap(outputs);
    M_freeAsyncTensorMap(inputs);
    M_freeTensorSpec(inputSpec);
  }

  {
    // Test with batch size 10
    int64_t shape[] = {10, 10};
    M_TensorSpec *inputSpec =
        M_newTensorSpec(shape, std::size(shape), M_FLOAT32, "input0", host);

    M_AsyncTensorMap *inputs = M_newAsyncTensorMap(context);
    M_borrowTensorInto(inputs, data, inputSpec, status);
    EXPECT_SUCCESS(status, "Failed to borrow tensor");

    M_AsyncTensorMap *outputs =
        M_executeModelSync(context, model, inputs, status);
    EXPECT_SUCCESS(status, "Failed to execute model");
    EXPECT_THAT(outputs, ::testing::NotNull());

    // Verify output shape
    M_AsyncTensor *outputTensor =
        M_getTensorByNameFrom(outputs, "output0", status);
    EXPECT_SUCCESS(status, "Failed to get output tensor");
    EXPECT_THAT(outputTensor, ::testing::NotNull());

    M_TensorSpec *outputSpec = M_getTensorSpec(outputTensor);
    EXPECT_EQ(M_getDimAt(outputSpec, 0), 10);

    M_freeTensorSpec(outputSpec);
    M_freeTensor(outputTensor);
    M_freeAsyncTensorMap(outputs);
    M_freeAsyncTensorMap(inputs);
    M_freeTensorSpec(inputSpec);
  }

  // Cleanup
  M_freeModel(model);
  M_freeCompiledModel(compiledModel);
}

TEST_F(APITest, SameSymbolicDimensionMismatch) {
#if MODULAR_MACOS
  GTEST_SKIP() << "Disabled on macOS due to platform-specific issues";
#endif
  // Load same_symbolic_two_inputs.mef with two ["seq_len", 768] inputs
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, sameSymbolicMefPath());

  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  EXPECT_SUCCESS(status, "Failed to compile model");

  M_AsyncModel *model = M_initModel(context, compiledModel, NULL, status);
  EXPECT_SUCCESS(status, "Failed to initialize model");

  int64_t shape1[] = {10, 768};
  int64_t shape2[] = {12, 768}; // Different seq_len

  M_TensorSpec *inputSpec1 =
      M_newTensorSpec(shape1, std::size(shape1), M_FLOAT32, "input0", host);
  M_TensorSpec *inputSpec2 =
      M_newTensorSpec(shape2, std::size(shape2), M_FLOAT32, "input1", host);

  float data1[10 * 768], data2[12 * 768];

  M_AsyncTensorMap *inputs = M_newAsyncTensorMap(context);
  M_borrowTensorInto(inputs, data1, inputSpec1, status);
  EXPECT_SUCCESS(status, "Failed to borrow first tensor");
  M_borrowTensorInto(inputs, data2, inputSpec2, status);
  EXPECT_SUCCESS(status, "Failed to borrow second tensor");

  M_AsyncTensorMap *outputs =
      M_executeModelSync(context, model, inputs, status);
  EXPECT_FAILURE(status, "symbolic dimension");
  EXPECT_THAT(outputs, ::testing::IsNull());

  // Cleanup
  M_freeAsyncTensorMap(inputs);
  M_freeTensorSpec(inputSpec2);
  M_freeTensorSpec(inputSpec1);
  M_freeModel(model);
  M_freeCompiledModel(compiledModel);
}

TEST_F(APITest, AlgebraicDimensionMeetsConstraints) {
#if MODULAR_MACOS
  GTEST_SKIP() << "Disabled on macOS due to platform-specific issues";
#endif
  // Load algebraic_dimensions.mef with input [batch, 64] -> [batch, 4, 16]
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, algebraicMefPath());

  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  EXPECT_SUCCESS(status, "Failed to compile model");

  M_AsyncModel *model = M_initModel(context, compiledModel, NULL, status);
  EXPECT_SUCCESS(status, "Failed to initialize model");

  // reshape to [8, 4, 16] should succeed
  int64_t shape[] = {8, 64};
  M_TensorSpec *inputSpec =
      M_newTensorSpec(shape, std::size(shape), M_FLOAT32, "input0", host);

  float data[8 * 64];

  M_AsyncTensorMap *inputs = M_newAsyncTensorMap(context);
  M_borrowTensorInto(inputs, data, inputSpec, status);
  EXPECT_SUCCESS(status, "Failed to borrow tensor");

  // Execute - should succeed with correct dimensions
  M_AsyncTensorMap *outputs =
      M_executeModelSync(context, model, inputs, status);
  EXPECT_SUCCESS(status, "Failed to execute model");
  EXPECT_THAT(outputs, ::testing::NotNull());

  // Verify output shape is [8, 4, 16]
  M_AsyncTensor *outputTensor =
      M_getTensorByNameFrom(outputs, "output0", status);
  EXPECT_SUCCESS(status, "Failed to get output tensor");
  EXPECT_THAT(outputTensor, ::testing::NotNull());

  M_TensorSpec *outputSpec = M_getTensorSpec(outputTensor);
  EXPECT_EQ(M_getRank(outputSpec), 3);
  EXPECT_EQ(M_getDimAt(outputSpec, 0), 8);
  EXPECT_EQ(M_getDimAt(outputSpec, 1), 4);
  EXPECT_EQ(M_getDimAt(outputSpec, 2), 16);

  M_freeTensorSpec(outputSpec);

  // Cleanup
  M_freeTensor(outputTensor);
  M_freeAsyncTensorMap(outputs);
  M_freeAsyncTensorMap(inputs);
  M_freeTensorSpec(inputSpec);
  M_freeModel(model);
  M_freeCompiledModel(compiledModel);
}

TEST_F(APITest, AlgebraicDimensionFailsConstraints) {
  // Load algebraic_dimensions.mef expecting [batch, dim]
  // where `dim` must be divisible by 4
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, algebraicMefPath());

  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  EXPECT_SUCCESS(status, "Failed to compile model");

  M_AsyncModel *model = M_initModel(context, compiledModel, NULL, status);
  EXPECT_SUCCESS(status, "Failed to initialize model");

  int64_t shape[] = {8, 63}; // 63 not divisible by 4
  M_TensorSpec *inputSpec =
      M_newTensorSpec(shape, std::size(shape), M_FLOAT32, "input0", host);

  float data[8 * 63];

  M_AsyncTensorMap *inputs = M_newAsyncTensorMap(context);
  M_borrowTensorInto(inputs, data, inputSpec, status);
  EXPECT_SUCCESS(status, "Failed to borrow tensor");

  // Execute - should fail due to shape mismatch (63 != 64)
  M_AsyncTensorMap *outputs =
      M_executeModelSync(context, model, inputs, status);
  EXPECT_FAILURE(status, "");
  EXPECT_THAT(outputs, ::testing::IsNull());

  // Cleanup
  M_freeAsyncTensorMap(inputs);
  M_freeTensorSpec(inputSpec);
  M_freeModel(model);
  M_freeCompiledModel(compiledModel);
}
