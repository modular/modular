//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Init/Init.h"
#include "Support/CommandLine.h"
#include "Support/CrashReporting.h"

using namespace M;

namespace {

struct CLOptions {
  cl::opt<bool> simulate{
      "simulate", M::cl::desc("Simulate crash rather than actually crashing")};
};

} // namespace

int main(int argc, char **argv) {
  CLOptions clOptions;
  llvm::cl::ParseCommandLineOptions(argc, argv, "Modular Crash Test Dummy");

  auto ctxOr = Init::createContext("crash-test-dummy");
  if (ctxOr.isError()) {
    llvm::errs() << "could not create context: " << ctxOr.getError() << "\n";
    return EXIT_FAILURE;
  }

  if (clOptions.simulate)
    generateNonFatalDump();
  else
    std::abort();

  return EXIT_SUCCESS;
}
