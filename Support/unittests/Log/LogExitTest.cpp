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
// Exits the process while the default logger's consumer thread still drains
// records into the stdout sink. The stdout stream must outlive the logger
// through static destruction.

#include "Support/Log.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

#include "gtest/gtest.h"

namespace {

TEST(LogExitDeathTest, ExitWhileDrainingToStdout) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        for (int i = 0; i < 2000; ++i)
          MLOG_WARN("record {}", i);
        std::exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}

// The first log record comes from an atexit handler, so the logger is built
// during exit. llvm::outs() is built after the handler registers, so exit
// destroys it before the handler runs.
TEST(LogExitDeathTest, FirstLogFromExitHandler) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        std::atexit([] {
          MLOG_WARN("record from an exit handler");
          M::Log::getDefaultLog().flush();
        });
        llvm::outs().flush();
        std::exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}

} // namespace
