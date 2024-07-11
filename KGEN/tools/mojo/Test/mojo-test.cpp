//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-test.h"
#include "../../common/Telemetry.h"

#include "AsyncRT/Init/Init.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/MojoTesting/Test.h"
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
static std::optional<int> parseArgs(State &state,
                                    llvm::opt::InputArgList &args) {
  TestOptTable options;
  unsigned unused = 0;
  args = options.ParseArgs(state.arguments, unused, unused);

  // If `--help` appears anywhere in the argument list, print help before
  // checking any other arguments.
  if (args.hasArg(options::OPT_help)) {
    return state.printHelp(
#include "Test/TestOptionsHelpText.inc"
    );
  }

  if (int result = state.parseDiagnosticFormatArguments(
          args, options::OPT_diagnostic_format))
    return result;
  if (int result = state.rejectUnknownArguments(args, options::OPT_UNKNOWN))
    return result;

  return {};
}

//===----------------------------------------------------------------------===//
// Mojo test input
//===----------------------------------------------------------------------===//

static int test(const State &subcommandState) {
  State state = subcommandState;
  llvm::opt::InputArgList args;
  if (std::optional<int> exitCode = parseArgs(state, args))
    return *exitCode;

  // Create our context.
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "mojo", Init::Options().withRuntimeOptions(LLCL::RuntimeOptions()));
  if (ctxOr.isError())
    return state.reportError(ctxOr.getError());
  ContextRef ctx = std::move(*ctxOr);

  // Initialize telemetry, making sure to redact any arguments that may contain
  // user-sensitive data.
  auto &telemetryCtx = *ctx->get<M::Telemetry::TelemetryContext>();
  auto scopedThread = logToolInvocationEventAsync(
      telemetryCtx, StringRef(state.subcommand), args,
      /*privateArgs=*/{options::OPT_I});

  // Grab the additional import paths if they were provided.
  std::vector<std::string> additionalImportPaths;
  if (args.hasArg(options::OPT_I))
    additionalImportPaths = args.getAllArgValues(options::OPT_I);

  // If an input was provided, use that as the test id. Otherwise, fallback to
  // the current working directory.
  TestID testID;
  if (args.hasArg(options::OPT_INPUT)) {
    testID = TestID(args.getLastArgValue(options::OPT_INPUT));
  } else {
    testID = TestID(std::filesystem::current_path().string());
  }

  // If the input is a directory instead of an individual test
  // then add it to the import search path.
  // Implements MOTO-521
  std::filesystem::path testIDFilePath = testID.getFilePath();
  if (std::filesystem::is_directory(testIDFilePath)) {
    additionalImportPaths.push_back(testIDFilePath);
  }

  LLCL::Runtime &runtime = *ctx->get<LLCL::Runtime>();
  ErrorOr<std::optional<Test>> testOr =
      Test::discoverFromID(runtime, testID, additionalImportPaths);
  if (testOr.isError())
    return state.reportError(testOr.getError());
  std::optional<Test> test = std::move(*testOr);

  // Utility functor used to format the output for a given result.
  auto emitOutput = [&](const auto &result) {
    switch (state.diagnosticFormat) {
    case DiagnosticFormat::Text:
      result.print(llvm::outs());
      break;
    case DiagnosticFormat::JSON:
      llvm::json::OStream jsonOS(llvm::outs(), /*IndentSize=*/2);
      jsonOS.value(toJSON(result));
      break;
    }
    llvm::outs() << "\n";
  };

  // If we're only collecting, exit early.
  if (args.hasArg(options::OPT_collect_only)) {
    if (!test)
      return 0;
    emitOutput(*test);
    return 0;
  }

  if (!test) {
    llvm::outs() << "Total Discovered Tests: 0\n";
    return 0;
  }

  // Execute the test and print the results.
  TestExecutionResult result = test->execute(runtime, additionalImportPaths);
  emitOutput(result);
  return result.getKind() == TestExecutionResult::kSuccess ? 0 : 1;
}

void M::registerTestSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("test", test);
}
