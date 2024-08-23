//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Support/Debugging.h"
#include "KGEN/Support/Configuration.h"
#include "Support/ErrorOr.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include <csignal>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace M;

void M::waitForDebuggerToAttach() {
  llvm::errs() << "Waiting for debugger to attach...\nCurrent pid is "
               << llvm::sys::Process::getProcessId() << ".\n";

#ifdef _WIN32
  while (!IsDebuggerPresent())
    Sleep(1000);
#else
  std::raise(SIGSTOP);
#endif
}

void M::attachToNewRemoteDebugSession() {
  StringRef initializationError =
      "couldn't initiate the debug session. You might want to attach manually "
      "to this process";

  // Find the path to the mojo executable.
  ErrorOr<KGEN::MojoConfig> configOr = KGEN::MojoConfig::open();
  if (failed(configOr)) {
    llvm::errs() << "error: " << initializationError << ": "
                 << configOr.takeError() << "\n";
  } else {
    std::error_code ec;
    StringRef mojo = configOr->getDriverPath();
    if (!std::filesystem::exists(mojo.str(), ec) || ec) {
      llvm::errs()
          << "error: " << initializationError
          << ": unable to resolve the mojo path from the modular.cfg\n";
    } else {
      std::string pidStr = std::to_string(llvm::sys::Process::getProcessId());
      SmallVector<StringRef> args{mojo, "debug", "--vscode", "--pid", pidStr};

      // `mojo debug --vscode` succeeds if lldb-dap was launched, but it might
      // still be possible that the actual attach failed.
      int exitCode = llvm::sys::ExecuteAndWait(mojo, args, /*Env=*/std::nullopt,
                                               /*Redirects=*/{});
      if (exitCode != 0) {
        llvm::errs()
            << "error: the remote debugger seems to have failed to attach. "
               "You might need attach manually to this process\n";
      }
    }
  }
  waitForDebuggerToAttach();
}
