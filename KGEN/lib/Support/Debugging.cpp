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

void M::attachToRemoteDebugger() {
  // Find the path to the mojo executable.
  ErrorOr<KGEN::MojoConfig> configOr = KGEN::MojoConfig::open();
  if (failed(configOr)) {
    llvm::errs() << configOr.takeError() << "\n";
    return;
  }

  std::error_code ec;
  StringRef mojo = configOr->getDriverPath();
  if (!std::filesystem::exists(mojo.str(), ec) || ec) {
    llvm::errs() << "error: unable to resolve the mojo path\n";
    return;
  }

  std::string pidStr = std::to_string(llvm::sys::Process::getProcessId());
  SmallVector<StringRef> args{mojo, "debug", "--rpc", "--pid", pidStr};

  int exitCode = llvm::sys::ExecuteAndWait(mojo, args, /*Env=*/std::nullopt,
                                           /*Redirects=*/{});
  if (exitCode != 0) {
    llvm::errs()
        << "error: unable to attach to the remote debugger. You might need "
           "attach manually to this process. Its pid is "
        << pidStr << ".\n";
  }

  llvm::errs() << "Waiting for debugger to attach...\n";

#ifdef _WIN32
  while (!IsDebuggerPresent())
    Sleep(1000);
#else
  std::raise(SIGSTOP);
#endif
}
