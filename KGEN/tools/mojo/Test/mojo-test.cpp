//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-test.h"
#include "../Common/Telemetry.h"

#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/MojoTesting/Test.h"
#include "KGEN/Package/Package.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/Driver/DriverSupport.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/Telemetry/Telemetry.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace mlir;
using namespace M::KGEN::Mojo;

//===----------------------------------------------------------------------===//
// Command line argument parsing
//===----------------------------------------------------------------------===//

#define DRIVER_OPTIONS_PATH "Test/TestOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct TestOptTable : public llvm::opt::PrecomputedOptTable {
  TestOptTable() : llvm::opt::PrecomputedOptTable(InfoTable, PrefixTable) {}
};

} // namespace

/// Parses the command line arguments from the given `state` object. Its return
/// value is either an integer exit code signaling that program execution should
/// exit immediately with that code, or nullopt, signifying program execution
/// should continue.
static std::optional<int> parseArgs(const State &state,
                                    llvm::opt::InputArgList &args) {
  // First, parse all arguments, in order to find the index of the input
  // argument.
  TestOptTable options;
  unsigned unused = 0;
  args = options.ParseArgs(state.arguments, unused, unused);

  // If those arguments include `--help`, print help before checking any other
  // arguments.
  if (args.hasArg(options::OPT_help)) {
    return state.printHelp(
#include "Test/TestOptionsHelpText.inc"
    );
  }

  // Otherwise, within this subset of arguments that appear before the input,
  // unknown arguments are rejected.
  if (int result = state.rejectUnknownArguments(args, options::OPT_UNKNOWN))
    return result;

  return {};
}

//===----------------------------------------------------------------------===//
// Mojo test input
//===----------------------------------------------------------------------===//

static int test(const State &state) {
  llvm::opt::InputArgList args;
  if (std::optional<int> exitCode = parseArgs(state, args))
    return *exitCode;

  // Initialize the LLCL runtime. We don't allow users to configure runtime
  // options, such as the allocator or the work queue threading model.
  std::unique_ptr<LLCL::Runtime> runtime = LLCL::createUniqueRuntime();
  auto &telemetryCtx =
      runtime->emplaceContext<M::Telemetry::TelemetryContext>();

  // Initialize telemetry, making sure to redact any arguments that may contain
  // user-sensitive data.
  initializeTelemetry(telemetryCtx, state, args, /*privateArgs=*/{});

  // If an input was provided, use that as the test id. Otherwise, fallback to
  // the current working directory.
  TestID testID;
  if (args.hasArg(options::OPT_INPUT)) {
    testID = TestID(args.getLastArgValue(options::OPT_INPUT));
  } else {
    testID = TestID(std::filesystem::current_path().string());
  }
  std::optional<Test> test = Test::discoverFromID(testID);

  // If we're only collecting, exit early.
  if (args.hasArg(options::OPT_collect_only)) {
    if (test)
      llvm::outs() << *test << "\n";
    return 0;
  }

  // TODO: Add support for processing discovered tests.
  llvm::outs() << "Total Discovered Tests: 0\n";
  return 0;
}

void M::registerTestSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("test", test);
}
