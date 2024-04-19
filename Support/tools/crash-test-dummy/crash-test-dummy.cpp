//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

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
  initCrashpadForProgram("crash-test-dummy");

  if (clOptions.simulate)
    generateNonFatalDump();
  else
    std::abort();

  return EXIT_SUCCESS;
}
