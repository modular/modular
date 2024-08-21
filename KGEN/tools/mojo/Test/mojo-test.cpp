//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mojo-test.h"
#include "../../common/Telemetry.h"

#include "AsyncRT/Runtime/Runtime.h"
#include "Init/Init.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/MojoTesting/Test.h"
#include "KGEN/Support/Configuration.h"
#include "Support/Driver/DriverSupport.h"
#include "Support/FileSystemExtras.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/Telemetry/Telemetry.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Regex.h"
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

/// Extracts a set of options from the subcommand's arguments as strings
/// suitable for passing through to another process, like the compilation flags.
template <typename... OptFilters>
static std::vector<std::string> extractOptionsAndValues(State &state,
                                                        OptFilters... filters) {
  // We need to re-parse the options, otherwise we'll get errors due to
  // options being disposed after we extract them.
  TestOptTable options;
  unsigned unused = 0;
  auto args = options.ParseArgs(state.arguments, unused, unused);

  auto filtered = args.filtered(filters...);

  std::vector<std::string> extractedTokens;
  for (auto &arg : filtered) {
    extractedTokens.push_back(arg->getSpelling().str());
    for (const char *value : arg->getValues())
      extractedTokens.emplace_back(value);
  }

  return extractedTokens;
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

/// Collects a flat list of unit tests from a test hierarchy. In the process,
/// also searches for the presence of doctests.
static std::pair<std::vector<TestID>, bool> filterUnitTests(const Test &root) {
  std::vector<TestID> unitTests;
  bool hasDocTest = false;
  std::function<void(const Test &test)> visit =
      [&unitTests, &visit, &hasDocTest](const Test &test) {
        TestID id = test.getTestID();
        if (id.getTest()) {
          if (!id.getTestSuite()) {
            unitTests.push_back(std::move(id));
          } else {
            hasDocTest = true;
          }
        } else {
          for (const Test &child : test.getChildren()) {
            visit(child);
          }
        }
      };
  visit(root);

  return {unitTests, hasDocTest};
}

static ErrorOr<TempFile> generateEntrypointSource(ArrayRef<TestID> unitTests) {
  ErrorOr<TempFile> sourceFile = TempFile::create("test-%%%%%%.mojo");
  if (!failed(sourceFile)) {
    llvm::raw_fd_ostream rawOs(sourceFile->getFD(), /*shouldClose=*/false);
    mlir::raw_indented_ostream os(rawOs);
    for (const TestID &id : unitTests)
      os << "import `" << id.getFilePath().stem() << "`\n";

    os << "from sys import argv\n";
    os << "from testing import assert_not_equal\n";
    os << "fn main() raises:\n";

    os.indent();
    os << "var executed = 0\n";
    os << "var testName = argv()[1]\n";

    for (const TestID &id : unitTests) {
      std::optional<StringRef> testId = id.getTest();
      if (!testId.has_value())
        return Error("an unexpected empty test id was found");

      StringRef testName = *testId;
      if (!testName.consume_back("()"))
        return Error(
            llvm::formatv("test id with invalid format {0}", testName));

      ErrorOr<SmallVector<std::string>> names =
          TestID::parseScopedName(testName);
      if (names.isError())
        return names.takeError();

      os << llvm::formatv("if testName == 'all' or testName == '{0}::{1}()':\n",
                          id.getFilePath(), llvm::join(*names, "."));
      os.indent();
      os << formatv("`{0}`.`{1}`()\n", id.getFilePath().stem(),
                    StringRef(names->back()));
      os << "executed += 1\n";
      os.unindent();
    }

    os << "print(testName)\n";
    os << "assert_not_equal(executed, 0, \"no tests were executed\")\n";

    os.unindent();
  }

  return sourceFile;
}

static ErrorOrSuccess buildEntrypoint(std::vector<std::string> buildArgs,
                                      ArrayRef<TestID> unitTests,
                                      const TempFile &sourceFile,
                                      std::string outputPath) {
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

  buildArgs.emplace_back("-o");
  buildArgs.emplace_back(outputPath);
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

  return SuccessType();
}

//===----------------------------------------------------------------------===//
// Test debugging
//===----------------------------------------------------------------------===//

static ErrorOrSuccess launchDebug(ArrayRef<std::string> options,
                                  StringRef entrypointPath) {
  ErrorOr<std::string> driverPath = getMojoDriver();
  if (driverPath.isError())
    return Error(driverPath.getError());

  SmallVector<StringRef> debugCommand{
      *driverPath,
      "debug",
  };

  llvm::append_range(debugCommand, options);
  llvm::append_range(debugCommand, ArrayRef<StringRef>{entrypointPath, "all"});

  std::string errorMessage;
  int result =
      llvm::sys::ExecuteAndWait(*driverPath, debugCommand, std::nullopt,
                                std::nullopt, 0, 0, &errorMessage);
  if (!errorMessage.empty())
    return Error(errorMessage);
  else if (result != 0)
    return Error(llvm::formatv(
        "Debug command exited with non-zero exit code {0}", result));

  return SuccessType();
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
  std::vector<std::string> buildArgStrings =
      extractOptionsAndValues(state, options::OPT_CompilationOptionGroup,
                              options::OPT_ExperimentalCompilationOptionGroup,
                              options::OPT_TargetOptionGroup);

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

  std::optional<llvm::Regex> filterRegex = std::nullopt;
  if (args.hasArg(options::OPT_filter)) {
    StringRef filterString = args.getLastArgValue(options::OPT_filter);

    filterRegex = llvm::Regex(filterString.data(), llvm::Regex::IgnoreCase);

    std::string error;
    if (!filterRegex->isValid(error))
      return state.reportError(llvm::formatv("Invalid regex pattern `{0}`: {1}",
                                             filterString, error));
  }

  AsyncRT::Runtime &runtime = *ctx->get<AsyncRT::Runtime>();
  ErrorOr<std::optional<Test>> testOr =
      Test::discoverFromID(runtime, testID, additionalImportPaths);
  if (testOr.isError())
    return state.reportError(testOr.getError());
  std::optional<Test> test = std::move(*testOr);

  // If we're only collecting, exit early.
  if (args.hasArg(options::OPT_collect_only)) {
    if (!test)
      return 0;

    switch (state.diagnosticFormat) {
    case DiagnosticFormat::Text:
      test->print(llvm::outs(), filterRegex);
      break;
    case DiagnosticFormat::JSON:
      llvm::json::OStream jsonOS(llvm::outs(), /*IndentSize=*/2);
      jsonOS.value(toJSON(*test));
      break;
    }
    llvm::outs() << "\n";
    return 0;
  }

  if (!test) {
    llvm::outs() << "Total Discovered Tests: 0\n";

    // If the user specified a test ID but we still didn't find any tests, we
    // should suggest the use of `--filter` instead.
    if (args.hasArg(options::OPT_INPUT))
      llvm::outs() << llvm::formatv(
          "No tests were discovered with the test ID {0}. Did you mean to use "
          "--filter instead?\n",
          testID);

    return 0;
  }

  auto [unitTests, hasDocTest] = filterUnitTests(*test);
  ErrorOr<TempFile> entrypointSource = generateEntrypointSource(unitTests);
  if (entrypointSource.isError())
    return state.reportError(entrypointSource.getError());

  // TempFile will delete the file it points to once it goes out of scope.
  // Keeping it here ensures that the executable lives long enough for us to use
  // it.
  std::optional<TempFile> entrypointTemp = std::nullopt;
  std::string entrypointPath;
  if (args.hasArg(options::OPT_entrypoint_path)) {
    entrypointPath = args.getLastArgValue(options::OPT_entrypoint_path).str();
  } else {
    ErrorOr<TempFile> outputOrErr = TempFile::create("test-entrypoint-%%%%%%");
    if (outputOrErr.isError())
      return state.reportError(outputOrErr.getError());
    entrypointTemp.emplace(std::move(*outputOrErr));
    entrypointTemp->close();
    entrypointPath = entrypointTemp->getPath().string();
  }

  ErrorOrSuccess buildResult = buildEntrypoint(
      buildArgStrings, unitTests, *entrypointSource, entrypointPath);
  if (buildResult.isError())
    return state.reportError(buildResult.getError());

  if (args.hasArg(options::OPT_keep_entrypoint)) {
    entrypointSource->keep();
    if (entrypointTemp)
      entrypointTemp->keep();
    llvm::errs() << "Entrypoint source can be found at "
                 << entrypointSource->getPath() << "\n";
    llvm::errs() << "Built entrypoint can be found at " << entrypointPath
                 << "\n";
  }

  if (args.hasArg(options::OPT_no_execute)) {
    llvm::errs() << "Skipping test execution because --no-execute was passed\n";
    return 0;
  }

  if (args.hasArg(options::OPT_debug)) {
    if (hasDocTest)
      llvm::errs() << "WARNING: Doctests were discovered, but will not be "
                      "executed when `--debug` is passed.\n";

    std::vector<std::string> debugOptions = extractOptionsAndValues(
        state, options::OPT_DebuggerOptionGroup, options::OPT_RPCOptionGroup);
    ErrorOrSuccess debugResult = launchDebug(debugOptions, entrypointPath);

    if (debugResult.isError())
      return state.reportError(debugResult.getError());

    return 0;
  }

  // Execute the test and print the results.
  TestExecutionResult result = test->execute(
      runtime, entrypointPath, additionalImportPaths, filterRegex);

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

  return result.getKind() == TestExecutionResult::kSuccess ? 0 : 1;
}

void M::registerTestSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("test", test);
}
