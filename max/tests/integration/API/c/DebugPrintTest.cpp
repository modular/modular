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

// Test debug printing configuration in the C API.
// Tests verify that debug print options can be configured and that
// output is produced in the expected formats.

#include "Support/PlatformUtils.h"
#include "Utils.h"
#include "max/c/common.h"
#include "max/c/context.h"
#include "max/c/model.h"
#include "max/c/tensor.h"
#include "max/c/types.h"
#include "llvm/ADT/SmallSet.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

static const char *debugPrintMefPath() {
  return getenv("DEBUG_PRINT_MEF_PATH");
}

static std::string captureModelStdout(M_RuntimeContext *context, M_Device *host,
                                      M_Status *status) {
  M_CompileConfig *compileConfig = M_newCompileConfig();
  M_setModelPath(compileConfig, debugPrintMefPath());

  M_AsyncCompiledModel *compiledModel =
      M_compileModelSync(context, &compileConfig, status);
  EXPECT_SUCCESS(status, "Failed to compile model");

  M_AsyncModel *model = M_initModel(context, compiledModel, nullptr, status);
  EXPECT_SUCCESS(status, "Failed to initialize model");

  // Prepare inputs
  float vector1[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float vector2[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

  int64_t shape[] = {8};
  M_TensorSpec *inputSpec1 =
      M_newTensorSpec(shape, std::size(shape), M_FLOAT32, "input0", host);
  M_TensorSpec *inputSpec2 =
      M_newTensorSpec(shape, std::size(shape), M_FLOAT32, "input1", host);

  M_AsyncTensorMap *inputs = M_newAsyncTensorMap(context);
  M_borrowTensorInto(inputs, vector1, inputSpec1, status);
  EXPECT_SUCCESS(status);
  M_borrowTensorInto(inputs, vector2, inputSpec2, status);
  EXPECT_SUCCESS(status);

  testing::internal::CaptureStdout();

  M_AsyncTensorMap *outputs =
      M_executeModelSync(context, model, inputs, status);
  EXPECT_SUCCESS(status, "Failed to execute model");

  M_freeTensorSpec(inputSpec1);
  M_freeTensorSpec(inputSpec2);
  M_freeAsyncTensorMap(inputs);
  M_freeAsyncTensorMap(outputs);
  M_freeModel(model);
  M_freeCompiledModel(compiledModel);

  return testing::internal::GetCapturedStdout();
}

using M::APITest;

TEST_F(APITest, DebugPrintDefault) {
#if MODULAR_MACOS
  GTEST_SKIP() << "Disabled on macOS due to platform-specific issues";
#endif
  std::string output = captureModelStdout(context, host, status);
  // Default: M_COMPACT

  // clang-format off
  EXPECT_EQ(
      output,
      "input_vector_1 = tensor([[1.0000, 2.0000, 3.0000, ..., 6.0000, 7.0000, 8.0000]], dtype=f32, shape=[8])\n"
      "input_vector_2 = tensor([[8.0000, 7.0000, 6.0000, ..., 3.0000, 2.0000, 1.0000]], dtype=f32, shape=[8])\n"
      "sum_result = tensor([[9.0000, 9.0000, 9.0000, ..., 9.0000, 9.0000, 9.0000]], dtype=f32, shape=[8])\n"
      "final_output = tensor([[18.0000, 18.0000, 18.0000, ..., 18.0000, 18.0000, 18.0000]], dtype=f32, shape=[8])\n"
  );
  // clang-format on
}

TEST_F(APITest, DebugPrintCompact) {
#if MODULAR_MACOS
  GTEST_SKIP() << "Disabled on macOS due to platform-specific issues";
#endif
  // Set compact format; precision is ignored
  M_setDebugPrintOptions(context, M_COMPACT, 4, "");
  std::string output = captureModelStdout(context, host, status);

  // clang-format off
  EXPECT_EQ(
      output,
      "input_vector_1 = tensor([[1.0000, 2.0000, 3.0000, ..., 6.0000, 7.0000, 8.0000]], dtype=f32, shape=[8])\n"
      "input_vector_2 = tensor([[8.0000, 7.0000, 6.0000, ..., 3.0000, 2.0000, 1.0000]], dtype=f32, shape=[8])\n"
      "sum_result = tensor([[9.0000, 9.0000, 9.0000, ..., 9.0000, 9.0000, 9.0000]], dtype=f32, shape=[8])\n"
      "final_output = tensor([[18.0000, 18.0000, 18.0000, ..., 18.0000, 18.0000, 18.0000]], dtype=f32, shape=[8])\n"
  );
  // clang-format on
}

TEST_F(APITest, DebugPrintFull) {
#if MODULAR_MACOS
  GTEST_SKIP() << "Disabled on macOS due to platform-specific issues";
#endif
  // Set full format with precision 6
  M_setDebugPrintOptions(context, M_FULL, 6, "");
  std::string output = captureModelStdout(context, host, status);

  // clang-format off
  EXPECT_EQ(
      output,
      "input_vector_1 = tensor<8xf32> [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]\n"
      "input_vector_2 = tensor<8xf32> [8.000000e+00, 7.000000e+00, 6.000000e+00, 5.000000e+00, 4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00]\n"
      "sum_result = tensor<8xf32> [9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00]\n"
      "final_output = tensor<8xf32> [1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01, 1.800000e+01]\n"
  );
  // clang-format on
}

TEST_F(APITest, DebugPrintNone) {
#if MODULAR_MACOS
  GTEST_SKIP() << "Disabled on macOS due to platform-specific issues";
#endif
  M_setDebugPrintOptions(context, M_NONE, 0, "");
  std::string output = captureModelStdout(context, host, status);
  EXPECT_EQ(output, "");
}

TEST_F(APITest, DebugPrintBinary) {
#if MODULAR_MACOS
  GTEST_SKIP() << "Disabled on macOS due to platform-specific issues";
#endif
  // Create temporary directory for binary output
  auto tmp = std::filesystem::temp_directory_path();
  std::string tmpDir = (tmp / "max_debug_print_test_XXXXXX").string();
  ASSERT_NE(mkdtemp(tmpDir.data()), nullptr)
      << "Failed to create temp directory";

  // Set binary format
  M_setDebugPrintOptions(context, M_BINARY, 0, tmpDir.c_str());
  std::string output = captureModelStdout(context, host, status);
  EXPECT_EQ(output, "");

  llvm::SmallSet<std::string, 8> expectedFiles = {
      "input_vector_1", "input_vector_2", "sum_result", "final_output"};
  llvm::SmallSet<std::string, 8> createdFiles;
  for (auto entry : std::filesystem::directory_iterator(tmpDir))
    createdFiles.insert(entry.path().filename().string());

  EXPECT_EQ(expectedFiles, createdFiles);

  std::filesystem::remove_all(tmpDir);
}

TEST_F(APITest, DebugPrintBinaryMaxCheckpoint) {
#if MODULAR_MACOS
  GTEST_SKIP() << "Disabled on macOS due to platform-specific issues";
#endif
  // Create temporary directory for binary output
  auto tmp = std::filesystem::temp_directory_path();
  std::string tmpDir = (tmp / "max_debug_print_test_XXXXXX").string();
  ASSERT_NE(mkdtemp(tmpDir.data()), nullptr)
      << "Failed to create temp directory";

  // Set binary format
  M_setDebugPrintOptions(context, M_BINARY_MAX_CHECKPOINT, 0, tmpDir.c_str());
  std::string output = captureModelStdout(context, host, status);
  EXPECT_EQ(output, "");

  llvm::SmallSet<std::string, 8> expectedFiles = {
      "input_vector_1.max", "input_vector_2.max", "sum_result.max",
      "final_output.max"};
  llvm::SmallSet<std::string, 8> createdFiles;
  for (auto entry : std::filesystem::directory_iterator(tmpDir))
    createdFiles.insert(entry.path().filename().string());

  EXPECT_EQ(expectedFiles, createdFiles);

  std::filesystem::remove_all(tmpDir);
}
