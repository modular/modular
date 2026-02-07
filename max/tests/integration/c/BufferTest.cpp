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

// Test memory borrowing, slices, and mutable buffers in the C API.
// Tests verify that tensors can borrow memory from interior pointers and
// that mutations to borrowed memory are properly reflected.

#include "Utils.h"
#include "max/c/common.h"
#include "max/c/context.h"
#include "max/c/device.h"
#include "max/c/model.h"
#include "max/c/tensor.h"
#include "max/c/types.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cstdio>

static const char *mutateMefPath() { return getenv("MUTATE_MEF_PATH"); }
static const char *vectorAddMefPath() { return getenv("VECTOR_ADD_MEF_PATH"); }

using M::APITest;

TEST_F(APITest, BorrowInteriorPointer) {
  // Load vector_add model for testing (simple computation)
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, vectorAddMefPath());

  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  EXPECT_SUCCESS(status, "Failed to compile model");

  M_AsyncModel *model = M_initModel(context, compiledModel, NULL, status);
  EXPECT_SUCCESS(status, "Failed to initialize model");

  // Create large buffer with known pattern
  float largeBuffer[100];
  for (int i = 0; i < 100; i++)
    largeBuffer[i] = static_cast<float>(i);

  // Create TensorSpec for slice [8] from the large buffer
  int64_t shape[] = {8};
  M_TensorSpec *inputSpec1 =
      M_newTensorSpec(shape, std::size(shape), M_FLOAT32, "input0", host);
  M_TensorSpec *inputSpec2 =
      M_newTensorSpec(shape, std::size(shape), M_FLOAT32, "input1", host);

  // Borrow from interior pointers (offset 50 and 60)
  M_AsyncTensorMap *inputs = M_newAsyncTensorMap(context);
  M_borrowTensorInto(inputs, &largeBuffer[50], inputSpec1, status);
  EXPECT_SUCCESS(status, "Failed to borrow first tensor");
  M_borrowTensorInto(inputs, &largeBuffer[60], inputSpec2, status);
  EXPECT_SUCCESS(status, "Failed to borrow second tensor");

  // Execute model (vector add)
  M_AsyncTensorMap *outputs =
      M_executeModelSync(context, model, inputs, status);
  EXPECT_SUCCESS(status, "Failed to execute model");
  EXPECT_THAT(outputs, ::testing::NotNull());

  // Verify model used the correct slices
  // Expected: largeBuffer[50:58] + largeBuffer[60:68]
  M_AsyncTensor *outputTensor =
      M_getTensorByNameFrom(outputs, "output0", status);
  EXPECT_SUCCESS(status, "Failed to get output tensor");
  EXPECT_THAT(outputTensor, ::testing::NotNull());

  const float *outputData = (const float *)M_getTensorData(outputTensor);
  for (int i = 0; i < 8; i++) {
    float expected = largeBuffer[50 + i] + largeBuffer[60 + i];
    EXPECT_FLOAT_EQ(outputData[i], expected);
  }

  // Cleanup
  M_freeTensor(outputTensor);
  M_freeAsyncTensorMap(outputs);
  M_freeAsyncTensorMap(inputs);
  M_freeTensorSpec(inputSpec2);
  M_freeTensorSpec(inputSpec1);
  M_freeModel(model);
  M_freeCompiledModel(compiledModel);
}

TEST_F(APITest, BufferMutatesBorrowedMemory) {
  // Load mutate_cpu model (x = x * 2 + 1)
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, mutateMefPath());

  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  EXPECT_SUCCESS(status, "Failed to compile model");

  M_AsyncModel *model = M_initModel(context, compiledModel, NULL, status);
  EXPECT_SUCCESS(status, "Failed to initialize model");

  // Create buffer with known values
  float buffer[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  // Store original values for verification
  float original[10];
  std::memcpy(original, buffer, sizeof(buffer));

  int64_t shape[] = {10};
  M_TensorSpec *inputSpec =
      M_newTensorSpec(shape, std::size(shape), M_FLOAT32, "input0", host);

  M_AsyncTensorMap *inputs = M_newAsyncTensorMap(context);
  M_borrowTensorInto(inputs, buffer, inputSpec, status);
  EXPECT_SUCCESS(status, "Failed to borrow tensor");

  // Execute model (mutates buffer: x = x * 2 + 1)
  M_AsyncTensorMap *outputs =
      M_executeModelSync(context, model, inputs, status);
  EXPECT_SUCCESS(status, "Failed to execute model");
  EXPECT_THAT(outputs, ::testing::NotNull());

  // Verify original buffer was mutated
  for (int i = 0; i < 10; i++) {
    float expected = original[i] * 2.0f + 1.0f;
    EXPECT_FLOAT_EQ(buffer[i], expected) << "Buffer not mutated at index " << i;
  }

  // Cleanup
  M_freeAsyncTensorMap(outputs);
  M_freeAsyncTensorMap(inputs);
  M_freeTensorSpec(inputSpec);
  M_freeModel(model);
  M_freeCompiledModel(compiledModel);
}

TEST_F(APITest, BufferMutatesInteriorPointer) {
  // Combine interior pointer borrowing with mutation
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, mutateMefPath());

  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  EXPECT_SUCCESS(status, "Failed to compile model");

  M_AsyncModel *model = M_initModel(context, compiledModel, NULL, status);
  EXPECT_SUCCESS(status, "Failed to initialize model");

  // Create large buffer
  float largeBuffer[100];
  for (int i = 0; i < 100; i++)
    largeBuffer[i] = static_cast<float>(i);

  // Store original values from section [50-59]
  float original[10];
  std::memcpy(original, &largeBuffer[50], sizeof(original));

  // Borrow from interior pointer
  int64_t shape[] = {10};
  M_TensorSpec *inputSpec =
      M_newTensorSpec(shape, std::size(shape), M_FLOAT32, "input0", host);

  M_AsyncTensorMap *inputs = M_newAsyncTensorMap(context);
  M_borrowTensorInto(inputs, &largeBuffer[50], inputSpec, status);
  EXPECT_SUCCESS(status, "Failed to borrow tensor");

  // Execute model (mutates slice)
  M_AsyncTensorMap *outputs =
      M_executeModelSync(context, model, inputs, status);
  EXPECT_SUCCESS(status, "Failed to execute model");

  // Verify only [50-59] was modified, rest unchanged
  for (int i = 0; i < 100; i++) {
    if (i >= 50 && i < 60) {
      float expected = original[i - 50] * 2.0f + 1.0f;
      EXPECT_FLOAT_EQ(largeBuffer[i], expected)
          << "Slice not mutated at index " << i;
    } else {
      EXPECT_FLOAT_EQ(largeBuffer[i], static_cast<float>(i))
          << "Unexpected modification at index " << i;
    }
  }

  // Cleanup
  M_freeAsyncTensorMap(outputs);
  M_freeAsyncTensorMap(inputs);
  M_freeTensorSpec(inputSpec);
  M_freeModel(model);
  M_freeCompiledModel(compiledModel);
}

TEST_F(APITest, BorrowedTensorsMayBeExternallyMutated) {
  // Load vector_add model for this test
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, vectorAddMefPath());

  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  EXPECT_SUCCESS(status, "Failed to compile model");

  M_AsyncModel *model = M_initModel(context, compiledModel, NULL, status);
  EXPECT_SUCCESS(status, "Failed to initialize model");

  // Create buffers
  float buffer1[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  float buffer2[8] = {8, 7, 6, 5, 4, 3, 2, 1};

  int64_t shape[] = {8};
  M_TensorSpec *inputSpec1 =
      M_newTensorSpec(shape, std::size(shape), M_FLOAT32, "input0", host);
  M_TensorSpec *inputSpec2 =
      M_newTensorSpec(shape, std::size(shape), M_FLOAT32, "input1", host);

  M_AsyncTensorMap *inputs = M_newAsyncTensorMap(context);
  M_borrowTensorInto(inputs, buffer1, inputSpec1, status);
  EXPECT_SUCCESS(status, "Failed to borrow first tensor");
  M_borrowTensorInto(inputs, buffer2, inputSpec2, status);
  EXPECT_SUCCESS(status, "Failed to borrow second tensor");

  // Modify buffers AFTER borrowing but BEFORE execution
  buffer1[0] = 100.0f;
  buffer1[5] = 200.0f;
  buffer2[2] = 300.0f;

  // Execute model
  M_AsyncTensorMap *outputs =
      M_executeModelSync(context, model, inputs, status);
  EXPECT_SUCCESS(status, "Failed to execute model");
  EXPECT_THAT(outputs, ::testing::NotNull());

  // Verify model saw the modified values
  M_AsyncTensor *outputTensor =
      M_getTensorByNameFrom(outputs, "output0", status);
  EXPECT_SUCCESS(status, "Failed to get output tensor");

  const float *outputData = (const float *)M_getTensorData(outputTensor);
  EXPECT_FLOAT_EQ(outputData[0], 100.0f + 8.0f); // 100 + 8 = 108
  EXPECT_FLOAT_EQ(outputData[2], 3.0f + 300.0f); // 3 + 300 = 303
  EXPECT_FLOAT_EQ(outputData[5], 200.0f + 3.0f); // 200 + 3 = 203

  // Cleanup
  M_freeTensor(outputTensor);
  M_freeAsyncTensorMap(outputs);
  M_freeAsyncTensorMap(inputs);
  M_freeTensorSpec(inputSpec2);
  M_freeTensorSpec(inputSpec1);
  M_freeModel(model);
  M_freeCompiledModel(compiledModel);
}
