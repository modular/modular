//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LITDialect/LITInterfaces.h"
#include "KGEN/MojoJupyter/Kernel.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DocString.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/MojoTesting/Test.h"
#include "KGEN/MojoTooling/ParserDriver.h"
#include "KGEN/Support/Configuration.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "LLCL/CompilerSupport/Context.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/Filesystem/Paths.h"
#include "Support/Init/Init.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Transport.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Process.h"

using namespace M;
using namespace M::KGEN::Mojo;
using namespace M::KGEN::LIT;
using namespace M::Mojo::Jupyter;
using namespace mlir::lsp;

namespace {
/// A test that can be executed.
struct ExecutableTest {
  ExecutableTest(TestID id, StringRef contents)
      : id(std::move(id)), contents(contents.str()) {}

  TestID id;
  std::string contents;
};
} // namespace

//===----------------------------------------------------------------------===//
// Test Execution
//===----------------------------------------------------------------------===//

/// Emit an initialization failure for the given tests.
static void
emitInitializationError(ArrayRef<ExecutableTest> tests,
                        function_ref<void(TestExecutionResult)> emitFn,
                        StringRef error) {
  // Emit an intialization error for the first test, and skip the rest.
  emitFn(TestExecutionResult::buildInitError(tests[0].id, error));

  // Emit skipped results for the rest of the tests.
  for (const ExecutableTest &test : tests.drop_front())
    emitFn(TestExecutionResult::buildSkip(test.id));
}

/// Execute the given tests, emitting results to the given function.
static void executeTests(StringRef workingDirectory,
                         ArrayRef<std::string> includeDirs,
                         ArrayRef<ExecutableTest> tests,
                         function_ref<void(TestExecutionResult)> emitFn) {
  auto emitInitError = [&](StringRef error) {
    emitInitializationError(tests, emitFn, error);
  };

  // Determine the path of the repl entry point.
  ErrorOr<KGEN::MojoConfig> config = KGEN::MojoConfig::open();
  if (failed(config))
    return emitInitError("failed to open modular.cfg");
  StringRef exePath = config->getREPLEntryPoint();

  // Build the output function that emits to the current cell result.
  std::string error, stdOut, stdErr;
  auto outputFn = [&](StringRef kind, StringRef msg) {
    if (kind == "error")
      error += msg;
    else if (kind == "stdout")
      stdOut += msg;
    else if (kind == "stderr")
      stdErr += msg;
  };

  // Initialize the kernel.
  MojoKernel kernel(outputFn, /*initializeMatPlotLib=*/false);
  if (failed(kernel.initialize(exePath, workingDirectory, includeDirs,
                               /*lldbInitFile=*/{})))
    return emitInitError("failed to initialize test executor kernel");

  for (auto [index, cell] : llvm::enumerate(tests)) {
    // Clear out the previous output.
    error.clear();
    stdOut.clear();
    stdErr.clear();

    // Execute the cell.
    std::string cellName = ("CodeBlock [" + Twine(index) + "]").str();
    auto now = std::chrono::steady_clock::now();
    bool hadError = kernel.executeAndWait(cellName, cell.contents) ==
                    ExecutionFinishedState::kFinishedError;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - now);

    // Emit the result of the cell.
    TestExecutionResult::Kind kind = TestExecutionResult::Kind::kSuccess;
    if (hadError) {
      kind = TestExecutionResult::kExecutionError;
      if (error.empty())
        error = "execution failed";
    } else if (!error.empty()) {
      kind = TestExecutionResult::kExecutionError;
    }
    emitFn(TestExecutionResult(kind, cell.id, duration, error, stdOut, stdErr));

    // If the cell failed, skip the rest of the cells.
    if (kind != TestExecutionResult::kSuccess) {
      for (auto &cell : tests.drop_front(index + 1))
        emitFn(TestExecutionResult::buildSkip(cell.id));
      break;
    }
  }
}

//===----------------------------------------------------------------------===//
// Test Initialization

/// Return a set of executable tests defined by the doc string of the given
/// operation, or nullopt if the operation does not define a test suite.
static ErrorOr<std::vector<ExecutableTest>>
getDocTestFromDecl(const TestID &testID, Operation *declOp) {
  auto astDeclOp = dyn_cast_if_present<ASTDeclInterface>(declOp);
  if (!astDeclOp)
    return Error("id does not reference a valid test suite");
  DocStringAttr docStringAttr = astDeclOp.getDocStringAttr();
  if (!docStringAttr)
    return Error("id does not reference a valid test suite");
  DocString docString(docStringAttr);
  auto codeBlocks = docString.getCodeBlocks();
  if (codeBlocks.empty())
    return Error("id does not reference a valid test suite");

  // Grab the index of the code block to execute.
  size_t index;
  if (testID.getTest()->getAsInteger(10, index) || index >= codeBlocks.size())
    return Error("id does not reference a valid test within the suite");

  // Create an executable test for the code block, which also tests all of the
  // previous blocks.
  std::vector<ExecutableTest> tests;
  for (size_t i : llvm::seq<size_t>(0, index + 1)) {
    tests.emplace_back(testID.withTest(std::to_string(i)),
                       codeBlocks[i].getRawCode().str());
  }
  return tests;
}

/// Return an executable test for a unit test defined by the given test id.
static ErrorOr<std::vector<ExecutableTest>>
getUnitTest(const std::filesystem::path &path, const TestID &testID) {
  // We currently only expect to find unit tests at the top-level of a file, and
  // have no parameters/results.
  if (testID.getTestSuite())
    return Error("id does not reference a valid test");
  StringRef testName = *testID.getTest();
  if (!testName.consume_back("()"))
    return Error("id does not reference a valid test");

  // Our unit tests are currently quite simple, so keep things equally simple
  // here and define a repl expression that imports the test module and invokes
  // the test function.
  std::string contents =
      llvm::formatv("import `{0}`\n`{0}`.`{1}`()", path.stem(), testName);
  return std::vector<ExecutableTest>{ExecutableTest{testID, contents}};
}

/// Return an executable test for the given Test ID, or Error if the test ID
/// does not correspond to a valid test.
static ErrorOr<std::vector<ExecutableTest>>
getTestFromID(const std::filesystem::path &path, const TestID &id,
              ContextRef ctx, ArrayRef<std::string> includeDirs) {
  // Setup a source manager to handle diagnostics.
  llvm::SourceMgr sourceManager;
  sourceManager.setIncludeDirs(includeDirs);
  std::string diagnosticBuffer;
  sourceManager.setDiagHandler(
      [](const llvm::SMDiagnostic &diag, void *rawBuffer) {
        std::string &diagnosticBuffer = *static_cast<std::string *>(rawBuffer);
        diagnosticBuffer += diag.getMessage().str();
      },
      &diagnosticBuffer);

  // Parse the file.
  MLIRContext mlirCtx{MLIRContext::Threading::DISABLED};
  registerContext(mlirCtx, ctx);
  KGEN::CompilationOptions compilationOptions;
  ParserConfig parserConfig(&mlirCtx, compilationOptions);
  MojoParserContext parserContext(sourceManager, parserConfig);
  MojoASTDeclRef moduleDecl = parserContext.parseFileOrPackage(path);
  if (!moduleDecl || !moduleDecl.getIfOperation())
    return Error(diagnosticBuffer);
  std::optional<StringRef> suiteName = id.getTestSuite();

  mlir::SymbolTableCollection symbolTable;
  Operation *op = moduleDecl->getIfOperation();

  // Process the case where we're looking for a test in a specific test suite.
  if (suiteName) {
    ErrorOr<SmallVector<std::string>> scopes =
        TestID::parseScopedName(*suiteName);
    if (scopes.isError())
      return scopes.takeError();

    // We currently don't have any other tests defined in test suites than
    // doc tests, so bail early if we're not looking for a doc test.
    if (scopes->pop_back_val() != "__doc__")
      return Error("id does not reference a valid test suite");

    // Resolve the operation defining the test suite.
    for (StringRef it : *scopes)
      if (!(op = symbolTable.lookupSymbolIn(op, StringAttr::get(&mlirCtx, it))))
        return Error("id does not reference a valid test suite");

    return getDocTestFromDecl(id, op);
  }

  // If we're not looking for a specific test suite, we're looking for a unit
  // test defined in the module itself.
  if (!symbolTable.lookupSymbolIn(op, StringAttr::get(&mlirCtx, *id.getTest())))
    return Error("id does not reference a valid test");
  return getUnitTest(path, id);
}

static mlir::LogicalResult runTestExecutor(const TestID &id, ContextRef ctx,
                                           bool prettyOutput,
                                           ArrayRef<std::string> includeDirs) {
  JSONTransport transport(stdin, llvm::outs(), JSONStreamStyle::Standard,
                          prettyOutput);
  MessageHandler messageHandler(transport);

  // Grab a notification handler to use for test results.
  llvm::unique_function<void(TestExecutionResult)> onTestResultFn =
      messageHandler.outgoingNotification<TestExecutionResult>(
          "execution/result");
  auto emitError = [&](StringRef message) {
    onTestResultFn(TestExecutionResult::buildInitError(id, message));
    return failure();
  };

  // Check that the id corresponds to a specific test.
  std::optional<StringRef> testName = id.getTest();
  if (!testName)
    return emitError("id does not correspond to a specific test");

  // Process the test ID.
  std::filesystem::path path = id.getFilePath();
  if (!Filesystem::isMojoSourceFile(path))
    return emitError("id does not correspond to a valid mojo source file");

  // Get the tests to execute.
  ErrorOr<std::vector<ExecutableTest>> tests =
      getTestFromID(path, id, ctx, includeDirs);
  if (tests.isError())
    return emitError(tests.getError());

  // Execute the tests, using the top-level directory as the working directory.
  std::filesystem::path workingDirectory = path.parent_path();
  while (Filesystem::isMojoSourcePackagePath(workingDirectory))
    workingDirectory = workingDirectory.parent_path();
  executeTests(workingDirectory.string(), includeDirs, *tests, onTestResultFn);
  return success();
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  llvm::InitLLVM il(argc, argv, /*InstallPipeSignalExitHandler=*/false);
  llvm::PrettyStackTraceProgram prettyStackTrace(argc, argv);

  llvm::cl::opt<Logger::Level> logLevel{
      "log",
      llvm::cl::desc("Verbosity of log messages written to stderr"),
      llvm::cl::values(
          clEnumValN(Logger::Level::Error, "error", "Error messages only"),
          clEnumValN(Logger::Level::Info, "info",
                     "High level execution tracing"),
          clEnumValN(Logger::Level::Debug, "verbose", "Low level details")),
      llvm::cl::init(Logger::Level::Error),
  };
  llvm::cl::opt<bool> prettyPrint{
      "pretty",
      llvm::cl::desc("Pretty-print JSON output"),
      llvm::cl::init(false),
  };
  llvm::cl::opt<std::string> testID{
      llvm::cl::Positional,
      llvm::cl::desc("<Test ID>"),
  };
  llvm::cl::list<std::string> includeDirs{
      "I", llvm::cl::desc("Append directory to the search path list used to "
                          "resolve imported modules in a test")};
  llvm::cl::ParseCommandLineOptions(argc, argv, "Mojo Test Executor");

  // Configure the logger.
  Logger::setLogLevel(logLevel);

  // Create our context.
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "mojo-test-executor", Init::Options().withRuntimeOptions());
  if (ctxOr.isError()) {
    llvm::errs() << "failed to create context: " << ctxOr.getError() << "\n";
    return 1;
  }

  // Run the executor.
  return failed(
      runTestExecutor(TestID(testID), ctxOr->copy(), prettyPrint, includeDirs));
}
