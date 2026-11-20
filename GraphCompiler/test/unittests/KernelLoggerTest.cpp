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

#include "GraphCompiler/MGPDialect/Primitives/KernelLogger.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/HostSystem.h"
#include "AsyncRT/Support/UnknownLocationDecoder.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <sstream>

using namespace M;
using namespace M::MGP;

namespace {

class KernelLoggerTest : public ::testing::Test {
protected:
  AsyncRT::CPUDeviceRef cpuDevice;

  void SetUp() override {
    cpuDevice = AsyncRT::getOrCreateCPUDevice(
        AsyncRT::CPUDeviceSource::Test, AsyncRT::CPUDeviceOptions().forDebug());
  }
};

TEST_F(KernelLoggerTest, SyncLoggerBehavesCorrectly) {
  // Create empty operands and results for basic testing
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> operands;
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> results;
  results.push_back(
      AsyncRT::AnyAsyncValueRef::createReady<int>(*cpuDevice, 42));

  // Capture log output
  std::string logOutput;
  llvm::raw_string_ostream logStream(logOutput);

  // Test that logger correctly logs sync events
  {
    KernelSyncLogger<decltype(operands)> logger("test_kernel", operands,
                                                results, logStream);
    // Logger destructor will trigger completion log
  }

  // Validate log output contains both launch and completion events
  EXPECT_THAT(logOutput, ::testing::HasSubstr(
                             "[KERNEL] SYNC LAUNCH test_kernel on CPU [id="));
  EXPECT_THAT(logOutput, ::testing::HasSubstr(
                             "[KERNEL] SYNC COMPLETE test_kernel on CPU [id="));
  EXPECT_THAT(logOutput, ::testing::HasSubstr("ms"));
}

TEST_F(KernelLoggerTest, AsyncLoggerBehavesCorrectly) {
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> operands;
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> results;
  results.push_back(
      AsyncRT::AnyAsyncValueRef::createReady<int>(*cpuDevice, 42));

  // Capture log output
  std::string logOutput;
  llvm::raw_string_ostream logStream(logOutput);

  // Test that async logger works correctly
  {
    KernelAsyncLogger<decltype(operands)> logger("async_kernel", operands,
                                                 results, logStream);
    // Logger destructor will trigger completion log
  }

  // Validate log output contains both launch and completion events with ASYNC
  // labels
  EXPECT_THAT(logOutput, ::testing::HasSubstr(
                             "[KERNEL] ASYNC LAUNCH async_kernel on CPU [id="));
  EXPECT_THAT(
      logOutput,
      ::testing::HasSubstr("[KERNEL] ASYNC COMPLETE async_kernel on CPU [id="));
}

TEST_F(KernelLoggerTest, LogsNormalCompletionEvent) {
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> operands;
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> results;
  results.push_back(
      AsyncRT::AnyAsyncValueRef::createReady<int>(*cpuDevice, 42));

  // Capture log output
  std::string logOutput;
  llvm::raw_string_ostream logStream(logOutput);

  // Test normal completion logging
  {
    KernelSyncLogger<decltype(operands)> logger("normal_kernel", operands,
                                                results, logStream);
    // Logger destructor will trigger completion log
  }

  // Validate log output contains launch and completion events
  EXPECT_THAT(logOutput, ::testing::HasSubstr(
                             "[KERNEL] SYNC LAUNCH normal_kernel on CPU [id="));
  EXPECT_THAT(
      logOutput,
      ::testing::HasSubstr("[KERNEL] SYNC COMPLETE normal_kernel on CPU [id="));
  EXPECT_THAT(logOutput, ::testing::Not(::testing::HasSubstr("SYNC ERROR")));
}

TEST_F(KernelLoggerTest, GeneratesUniqueKernelIds) {
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> operands;
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> results1;
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> results2;
  results1.push_back(
      AsyncRT::AnyAsyncValueRef::createReady<int>(*cpuDevice, 42));
  results2.push_back(
      AsyncRT::AnyAsyncValueRef::createReady<int>(*cpuDevice, 43));

  // Capture log output
  std::string logOutput;
  llvm::raw_string_ostream logStream(logOutput);

  {
    KernelSyncLogger<decltype(operands)> logger1("kernel1", operands, results1,
                                                 logStream);
  }

  {
    KernelSyncLogger<decltype(operands)> logger2("kernel2", operands, results2,
                                                 logStream);
  }

  // Parse kernel IDs from LAUNCH events to verify they are unique
  std::string output = logOutput;
  size_t pos1 = output.find("SYNC LAUNCH kernel1");
  ASSERT_NE(pos1, std::string::npos);
  size_t idPos1 = output.find("[id=", pos1);
  ASSERT_NE(idPos1, std::string::npos);
  size_t idEnd1 = output.find("]", idPos1);
  ASSERT_NE(idEnd1, std::string::npos);
  std::string id1Str = output.substr(idPos1 + 4, idEnd1 - idPos1 - 4);

  size_t pos2 = output.find("SYNC LAUNCH kernel2");
  ASSERT_NE(pos2, std::string::npos);
  size_t idPos2 = output.find("[id=", pos2);
  ASSERT_NE(idPos2, std::string::npos);
  size_t idEnd2 = output.find("]", idPos2);
  ASSERT_NE(idEnd2, std::string::npos);
  std::string id2Str = output.substr(idPos2 + 4, idEnd2 - idPos2 - 4);

  // Kernel IDs should be different
  EXPECT_NE(id1Str, id2Str);
}

TEST_F(KernelLoggerTest, LogsCorrectKernelAndDeviceInfo) {
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> operands;
  llvm::SmallVector<AsyncRT::AnyAsyncValueRef> results;
  results.push_back(
      AsyncRT::AnyAsyncValueRef::createReady<int>(*cpuDevice, 42));

  // Capture log output
  std::string logOutput;
  llvm::raw_string_ostream logStream(logOutput);

  {
    KernelSyncLogger<decltype(operands)> logger("my_kernel", operands, results,
                                                logStream);
  }

  // Validate kernel name and device info appear in log output
  EXPECT_THAT(logOutput, ::testing::HasSubstr("my_kernel"));
  EXPECT_THAT(logOutput, ::testing::HasSubstr("on CPU"));
  EXPECT_THAT(logOutput, ::testing::HasSubstr("[id="));
}

} // namespace
