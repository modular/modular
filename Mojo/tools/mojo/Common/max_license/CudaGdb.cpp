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

#include "CudaGdb.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "Mojo/Support/Configuration.h"
#include "Support/Driver/DriverSupport.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Program.h"
#include <filesystem>

#define DRIVER_OPTIONS_PATH "Debug/DebugOptions.inc"
#define DRIVER_OPTIONS_ENUM_ONLY
#include "Support/Driver/OptTable.inc"

using namespace M;

/// Returns the path to the `cuda-gdb` executable, or an error if not found.
static ErrorOr<std::string> getCudaGdb() {
  auto executable = llvm::sys::findProgramByName("cuda-gdb");
  if (!executable)
    return Error("unable to resolve the cuda-gdb path");
  return executable.get();
}

int M::invokeCudaGdb(const State &state, ArrayRef<std::string> cudaGdbArgs,
                     ArrayRef<std::string> runArgs,
                     std::optional<std::string> cudaGdbPathGiven,
                     bool breakOnLaunch, bool dryRun) {
  std::string cudaGdbPath;
  if (cudaGdbPathGiven.has_value()) {
    cudaGdbPath = cudaGdbPathGiven.value();
  } else {
    // Find the path to the cuda-gdb executable.
    ErrorOr<std::string> cgdb = getCudaGdb();
    if (failed(cgdb))
      return state.reportError(cgdb.getError());
    cudaGdbPath = cgdb.get();
  }
  SmallVector<StringRef> subprocessArgs = {cudaGdbPath};

  if (breakOnLaunch) {
    subprocessArgs.push_back("-ex");
    subprocessArgs.push_back("set cuda break_on_launch application");
  }

  llvm::append_range(subprocessArgs, cudaGdbArgs);

  if (!runArgs.empty()) {
    subprocessArgs.push_back("--args");
    llvm::append_range(subprocessArgs, runArgs);
  }

  if (dryRun) {
    llvm::interleave(
        subprocessArgs, llvm::outs(),
        [](StringRef arg) {
          // This is just a simple shell quoting mechanism. We don't use
          // anything fancy because the dry-run command is intended for
          // internal development.
          if (arg.contains(' '))
            llvm::outs() << "'" << arg << "'";
          else
            llvm::outs() << arg;
        },
        " ");
    llvm::outs() << "\n";
    return 0;
  }

  std::optional<std::string> useLegacyEnv =
      llvm::sys::Process::GetEnv("CUDBG_USE_LEGACY_DEBUGGER");
  // The cuda-gdb documentation always says to use the value "1" for the
  // variable.  It is possible that cuda-gdb accepts other values as well, but
  // for our messaging purposes let's just test for "1".
  bool useLegacy = useLegacyEnv.has_value() && (useLegacyEnv.value() == "1");

  if (useLegacy) {
    llvm::outs()
        << "This will run cuda-gdb using the Legacy Debugger backend.\n"
        << "If you have issues stopping at GPU breakpoints, try using the "
           "Unified Debugger\n"
        << "backend by removing the "
           "`CUDBG_USE_LEGACY_DEBUGGER=1` environment variable.\n";
  } else {
    llvm::outs()
        << "This will run cuda-gdb using the Unified Debugger backend.\n"
        << "If the debugger shows errors mentioning "
           "`CUDBG_ERROR_COMMUNICATION_FAILURE`,\n"
        << "retry this command prepending the environment variable "
           "`CUDBG_USE_LEGACY_DEBUGGER=1`\n";
  }

  return llvm::sys::ExecuteAndWait(cudaGdbPath, subprocessArgs);
}
