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
#include "KGEN/Support/Configuration.h"
#include "Support/Driver/DriverSupport.h"
#include "Support/FileSystemExtras.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/Telemetry/Telemetry.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include <llvm/ADT/DenseSet.h>
#include <mlir/Support/IndentedOstream.h>

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
// Test entrypoint building
//===----------------------------------------------------------------------===//

/// Returns the path to the mojo tool, or an error if not found.
static ErrorOr<std::string> getMojoDriver() {
  ErrorOr<KGEN::MojoConfig> configOr = KGEN::MojoConfig::open();
  if (failed(configOr))
    return Error(Twine("failed to parse 'modular.cfg': ") +
                 configOr.getError());
  std::error_code ec;
  StringRef driver = configOr->getDriverPath();
  if (!std::filesystem::exists(driver.str(), ec) || ec)
    return Error("unable to resolve the mojo program path");
  return driver.str();
}

static std::vector<TestID> filterUnitTests(const Test &root) {
  std::vector<TestID> unitTests;
  std::function<void(const Test &test)> visit = [&unitTests,
                                                 &visit](const Test &test) {
    TestID id = test.getTestID();
    if (id.getTest()) {
      if (!id.getTestSuite()) {
        unitTests.push_back(std::move(id));
      }
    } else {
      for (const Test &child : test.getChildren()) {
        visit(child);
      }
    }
  };
  visit(root);

  return unitTests;
}

static ErrorOr<TempFile> generateEntrypointSource(ArrayRef<TestID> unitTests) {
  ErrorOr<TempFile> sourceFile = TempFile::create("test-%%%%%%.mojo");
  if (!failed(sourceFile)) {
    llvm::raw_fd_ostream rawOs(sourceFile->getFD(), /*shouldClose=*/false);
    mlir::raw_indented_ostream os(rawOs);
    for (const TestID &id : unitTests)
      os << "import `" << id.getFilePath().stem() << "`\n";

    os << "from sys import argv\n";
    os << "fn main() raises:\n";

    os.indent();
    os << "var testName = argv()[1]\n";

    for (const TestID &id : unitTests) {
      StringRef testName = *id.getTest();
      assert(testName.consume_back("()") &&
             "id does not reference a valid test");

      ErrorOr<SmallVector<std::string>> names =
          TestID::parseScopedName(testName);
      assert(!names.isError());
      assert(names->size() == 1 && "id does not reference a valid test");

      os << formatv("if testName == \"{0}\":\n", id.strref());
      os.indent();
      os << formatv("`{0}`.`{1}`()\n", id.getFilePath().stem(),
                    StringRef(names->back()));
      os.unindent();
    }

    os.unindent();
  }

  return sourceFile;
}

static ErrorOr<TempFile> buildEntrypoint(std::vector<std::string> buildArgs,
                                         ArrayRef<TestID> unitTests,
                                         const TempFile &sourceFile) {
  ErrorOr<std::string> driverPath = getMojoDriver();
  if (driverPath.isError())
    return Error(driverPath.getError());

  // Append each test file's directory as an include path. This allows the
  // entrypoint source to import each test function.
  std::vector<std::string> testIncludeDirs{""};
  for (const TestID &id : unitTests)
    llvm::append_range(
        buildArgs,
        ArrayRef<std::string>{"-I", id.getFilePath().parent_path().string()});

  ErrorOr<TempFile> entrypointOutput =
      TempFile::create("test-entrypoint-%%%%%%");
  if (entrypointOutput.isError())
    return Error(entrypointOutput.getError());

  entrypointOutput->close();

  buildArgs.emplace_back("-o");
  std::string outputPath = entrypointOutput->getPath().string();
  buildArgs.push_back(entrypointOutput->getPath().string());
  buildArgs.push_back(sourceFile.getPath().string());

  SmallVector<StringRef> buildCommand{*driverPath, "build"};
  llvm::append_range(buildCommand, buildArgs);

  std::string errorMessage;
  int result =
      llvm::sys::ExecuteAndWait(*driverPath, buildCommand, std::nullopt,
                                std::nullopt, 0, 0, &errorMessage);
  if (!errorMessage.empty())
    return Error(errorMessage);

  if (result)
    return Error(llvm::formatv(
        "couldn't build the test executable. Exit code {0}", result));

  return entrypointOutput;
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
      "mojo", Init::Options().withRuntimeOptions(AsyncRT::RuntimeOptions()));
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

  // Collect compilation args now. For some reason, if we do this after reading
  // OPT_INPUT, the argument parser ends up in an invalid state.
  std::vector<std::string> buildArgStrings;
  {
    // We need to re-parse the options, otherwise we'll get errors due to
    // options being disposed after we extract them.
    TestOptTable options;
    unsigned unused = 0;
    auto args = options.ParseArgs(state.arguments, unused, unused);
    auto filtered =
        args.filtered(options::OPT_CompilationOptionGroup,
                      options::OPT_ExperimentalCompilationOptionGroup,
                      options::OPT_TargetOptionGroup);
    for (auto &arg : filtered) {
      buildArgStrings.push_back(arg->getSpelling().str());
      assert(arg->getNumValues() <= 1);
      for (const char *value : arg->getValues())
        buildArgStrings.emplace_back(value);
    }
  }

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
    buildArgStrings.emplace_back("-I");
    buildArgStrings.push_back(testIDFilePath.string());
  }

  AsyncRT::Runtime &runtime = *ctx->get<AsyncRT::Runtime>();
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

  std::vector<TestID> unitTests = filterUnitTests(*test);
  ErrorOr<TempFile> entrypointSource = generateEntrypointSource(unitTests);
  if (entrypointSource.isError())
    return state.reportError(entrypointSource.getError());

  ErrorOr<TempFile> entrypoint =
      buildEntrypoint(std::move(buildArgStrings), unitTests, *entrypointSource);
  if (entrypoint.isError())
    return state.reportError(entrypoint.getError());

  if (args.hasArg(options::OPT_keep_entrypoint)) {
    entrypointSource->keep();
    entrypoint->keep();
    llvm::errs() << "Entrypoint source can be found at "
                 << entrypointSource->getPath() << "\n";
    llvm::errs() << "Built entrypoint can be found at " << entrypoint->getPath()
                 << "\n";
  }

  // Execute the test and print the results.
  TestExecutionResult result = test->execute(runtime, additionalImportPaths);
  emitOutput(result);
  return result.getKind() == TestExecutionResult::kSuccess ? 0 : 1;
}

void M::registerTestSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("test", test);
}
