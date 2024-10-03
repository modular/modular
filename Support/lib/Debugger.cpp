//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Debugger.h"
#include "Support/Process.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Process.h"

#include <chrono>
#include <thread>

using namespace M;

void M::Debugger::attach(int timeoutSeconds) {
  llvm::errs() << "======== Pausing for debugger attach ========\n";
  llvm::errs() << "Process name: " << getProcessExecutablePath() << "\n";
  llvm::errs() << "Process ID: " << llvm::sys::Process::getProcessId() << "\n";
  llvm::errs() << "Waiting for " << timeoutSeconds << " seconds...\n";
  std::this_thread::sleep_for(std::chrono::seconds(timeoutSeconds));
  llvm::errs() << "======= Resuming execution ========\n";
}
