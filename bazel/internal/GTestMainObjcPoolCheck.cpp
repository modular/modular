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

/// Custom gtest main for macOS that detects ObjC autorelease pool leaks.
///
/// Requires OBJC_DEBUG_MISSING_POOLS=YES in the process environment (set via
/// the Bazel test env in modular_cc_test.bzl — must be present before the ObjC
/// runtime initializes.
///
/// When active, the ObjcPoolCheck environment intercepts stderr via a
/// pipe+reader thread, tees output in real-time to the original stderr, and
/// fails the test suite in TearDown if any "MISSING POOLS" warnings were
/// captured.
///
/// This mirrors LLVM's UnitTestMain/TestMain.cpp but adds the pool check.

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>

// ---------------------------------------------------------------------------
// ObjcPoolCheck environment
// ---------------------------------------------------------------------------

namespace {

class ObjcPoolCheck : public ::testing::Environment {
public:
  void SetUp() override {
    if (!std::getenv("OBJC_DEBUG_MISSING_POOLS"))
      return;

    // Save original stderr.
    savedStderr = dup(STDERR_FILENO);
    ASSERT_NE(savedStderr, -1) << "dup(STDERR_FILENO) failed";

    // Create pipe: pipeFds[0]=read, pipeFds[1]=write.
    int pipeFds[2];
    ASSERT_EQ(pipe(pipeFds), 0) << "pipe() failed";

    readFd = pipeFds[0];

    // Redirect stderr to the write end of the pipe.
    ASSERT_NE(dup2(pipeFds[1], STDERR_FILENO), -1) << "dup2() failed";
    close(pipeFds[1]); // Close the original write fd; stderr IS the write end.

    // Spawn reader thread: reads from pipe, tees to original stderr, captures.
    readerThread = std::thread([this]() {
      char buf[4096];
      ssize_t n;
      while ((n = read(readFd, buf, sizeof(buf))) > 0) {
        // Tee to original stderr for real-time output.
        write(savedStderr, buf, n);
        // Accumulate for final check.
        std::lock_guard<std::mutex> lock(mu);
        captured.append(buf, n);
      }
      close(readFd);
    });

    active = true;
  }

  void TearDown() override {
    if (!active)
      return;

    // Restore stderr — this closes the pipe's write end, causing reader EOF.
    dup2(savedStderr, STDERR_FILENO);
    close(savedStderr);

    readerThread.join();

    std::lock_guard<std::mutex> lock(mu);
    if (captured.find("MISSING POOLS:") != std::string::npos) {
      FAIL() << "OBJC_DEBUG_MISSING_POOLS detected autorelease-without-pool "
                "warnings in stderr:\n"
             << captured;
    }
  }

private:
  bool active = false;
  int savedStderr = -1;
  int readFd = -1;
  std::thread readerThread;
  std::mutex mu;
  std::string captured;
};

} // namespace

// ---------------------------------------------------------------------------
// Custom main (mirrors LLVM's UnitTestMain/TestMain.cpp)
// ---------------------------------------------------------------------------

const char *TestMainArgv0;

int main(int argc, char **argv) {
  // Skip setting up signal handlers for tests that need to test things without
  // them configured.
  if (!getenv("LLVM_PROGRAM_TEST_NO_STACKTRACE_HANDLER")) {
    llvm::sys::PrintStackTraceOnErrorSignal(argv[0],
                                            true /* Disable crash reporting */);
  }

  // Initialize both gmock and gtest.
  testing::InitGoogleMock(&argc, argv);

  ::testing::AddGlobalTestEnvironment(new ObjcPoolCheck());

  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Make it easy for a test to re-execute itself by saving argv[0].
  TestMainArgv0 = argv[0];

  return RUN_ALL_TESTS();
}
