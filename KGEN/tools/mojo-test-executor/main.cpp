//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/Runtime.h"
#include "Init/Init.h"
#include "KGEN/MojoJupyter/Kernel.h"
#include "KGEN/MojoParser/DocString.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/MojoTesting/Test.h"
#include "KGEN/MojoTooling/ParserDriver.h"
#include "KGEN/Support/Configuration.h"
#include "Support/FileSystemExtras.h"
#include "Support/Filesystem/Paths.h"
#include "Support/Process.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Transport.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/ADT/StringExtras.h>

using namespace M;
using namespace M::KGEN::Mojo;
using namespace M::KGEN::LIT;
using namespace M::Mojo::Jupyter;
using namespace mlir::lsp;

namespace {
/// How to execute an executable test.
enum class TestExecutionKind {
  /// Use the passed entrypoint from mojo-test.
  Entrypoint,
  /// Use the REPL kernel to invoke a group of tests at once.
  REPL,
};

/// A test that can be executed.
struct ExecutableTest {
  ExecutableTest(TestID id, StringRef contents, TestExecutionKind kind)
      : id(std::move(id)), contents(contents.str()), kind(kind) {}

  TestID id;
  std::string contents;
  TestExecutionKind kind;
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

static TestExecutionResult executeMojoRunTest(StringRef workingDirectory,
                                              StringRef entrypointFile,
                                              const ExecutableTest &test) {

  auto emitInitError = [&](StringRef error) {
    return TestExecutionResult::buildInitError(test.id, error);
  };
  // Create a temporary output file for the test executor.
  ErrorOr<TempFile> outFileOr = TempFile::create("test-out-%%%%%%.txt");
  if (failed(outFileOr))
    return emitInitError(outFileOr.getError());

  ErrorOr<TempFile> errFileOr = TempFile::create("test-err-%%%%%%.txt");
  if (failed(errFileOr))
    return emitInitError(errFileOr.getError());

  // The test ID contains string escapes. The test entrypoint will not be
  // looking for these, so we need to undo those escape sequences here in order
  // to correctly invoke some tests.
  ErrorOr<SmallVector<std::string>> names =
      TestID::parseScopedName(*test.id.getTest());
  std::string unescapedId = llvm::join(*names, ".");
  std::string fullId =
      llvm::formatv("{0}::{1}", test.id.getFilePath(), unescapedId);

  std::string out = outFileOr->getPath().string();
  std::string errPath = errFileOr->getPath().string();
  const std::optional<StringRef> redirects[] = {
      /*stdin=*/std::nullopt,
      /*stdout=*/out,
      /*stderr=*/errPath,
  };
  SmallVector<StringRef> args = {entrypointFile, fullId};

  auto now = std::chrono::steady_clock::now();
  auto exitCode =
      llvm::sys::ExecuteAndWait(entrypointFile, args, {}, redirects);
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - now);

  auto kind = exitCode == 0 ? TestExecutionResult::kSuccess
                            : TestExecutionResult::kExecutionError;
  StringRef error = kind == TestExecutionResult::kSuccess
                        ? ""
                        : "Unhandled exception caught during execution";

  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> stdOutBufOr =
      toModularErrorOr(llvm::MemoryBuffer::getFile(out));
  if (failed(stdOutBufOr))
    return emitInitError(stdOutBufOr.getError());

  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> stdErrBufOf =
      toModularErrorOr(llvm::MemoryBuffer::getFile(errPath));
  if (failed(stdErrBufOf))
    return emitInitError(stdErrBufOf.getError());
  std::string stdOut = stdOutBufOr->get()->getBuffer().str();
  std::string stdErr = stdErrBufOf->get()->getBuffer().str();

  return TestExecutionResult{kind,        test.id,           duration,
                             error.str(), std::move(stdOut), std::move(stdErr)};
}

static void executeReplTests(ContextRef ctx, StringRef workingDirectory,
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
  if (failed(kernel.initialize(ctx, exePath, workingDirectory, includeDirs,
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

/// Execute the given tests, emitting results to the given function.
static void executeTests(ContextRef ctx, StringRef workingDirectory,
                         ArrayRef<std::string> includeDirs,
                         std::vector<ExecutableTest> tests,
                         StringRef entrypointFile,
                         function_ref<void(TestExecutionResult)> emitFn) {

  std::vector<ExecutableTest> docTests;
  for (auto &test : tests) {
    if (test.kind == TestExecutionKind::REPL)
      docTests.push_back(std::move(test));
    else
      emitFn(executeMojoRunTest(workingDirectory, entrypointFile, test));
  }
  executeReplTests(std::move(ctx), workingDirectory, includeDirs, docTests,
                   emitFn);
}

//===----------------------------------------------------------------------===//
// Test Initialization

/// Return an executable test for a doc test defined by the given test.
static ErrorOr<ExecutableTest> getDocTest(llvm::SourceMgr &sourceMgr,
                                          const std::filesystem::path &path,
                                          const Test &test) {
  // Use the location of the test to pull out the code block for the doc test.
  std::optional<Test::SourceRange> sourceRange = test.getSourceRange();
  if (!sourceRange)
    return Error("id does not reference a valid test suite");

  // Read in the source file to find the code block.
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileContents =
      toModularErrorOr(llvm::MemoryBuffer::getFile(path.string()));
  if (fileContents.isError())
    return fileContents.takeError();

  // Extract out the code block based on the source location of the test.
  unsigned bufferId =
      sourceMgr.AddNewSourceBuffer(std::move(*fileContents), SMLoc());
  SMLoc startLoc = sourceMgr.FindLocForLineAndColumn(
      bufferId, sourceRange->startLine + 1, 0);
  SMLoc endLoc =
      sourceMgr.FindLocForLineAndColumn(bufferId, sourceRange->endLine, 0);
  if (!startLoc.isValid() || !endLoc.isValid())
    return Error("unable to resolve source location of test");

  StringRef codeBlock(startLoc.getPointer(),
                      endLoc.getPointer() - startLoc.getPointer());
  return ExecutableTest{test.getTestID(), codeBlock, TestExecutionKind::REPL};
}

/// Return an executable test for a unit test defined by the given test id.
static ErrorOr<ExecutableTest> getUnitTest(const std::filesystem::path &path,
                                           const TestID &testID) {
  // We currently only expect to find unit tests at the top-level of a file, and
  // have no parameters/results.
  if (testID.getTestSuite())
    return Error("id does not reference a valid test");
  StringRef testName = *testID.getTest();
  if (!testName.consume_back("()"))
    return Error("id does not reference a valid test");

  ErrorOr<SmallVector<std::string>> names = TestID::parseScopedName(testName);
  if (names.isError())
    return names.takeError();
  // We only expect top-level unit tests right now.
  if (names->size() != 1)
    return Error("id does not reference a valid test");

  return ExecutableTest{testID, "", TestExecutionKind::Entrypoint};
}

/// Return an executable test for the given Test ID, or Error if the test ID
/// does not correspond to a valid test.
static ErrorOr<ExecutableTest>
getExecutableTest(llvm::SourceMgr &sourceMgr, const std::filesystem::path &path,
                  const Test &test) {
  // Process the case where we're looking for a test in a specific test suite.
  if (std::optional<StringRef> suiteName = test.getTestID().getTestSuite()) {
    ErrorOr<SmallVector<std::string>> scopes =
        TestID::parseScopedName(*suiteName);
    if (scopes.isError())
      return scopes.takeError();

    // We currently don't have any other tests defined in test suites than
    // doc tests, so bail early if we're not looking for a doc test.
    if (scopes->pop_back_val() != "__doc__")
      return Error("id does not reference a valid test suite");

    return getDocTest(sourceMgr, path, test);
  }

  // If we're not looking for a specific test suite, we're looking for a unit
  // test.
  return getUnitTest(path, test.getTestID());
}

static mlir::LogicalResult runTestExecutor(ContextRef ctx, ArrayRef<Test> tests,
                                           bool prettyOutput,
                                           ArrayRef<std::string> includeDirs,
                                           StringRef entrypointFile) {
  JSONTransport transport(stdin, llvm::outs(), JSONStreamStyle::Standard,
                          prettyOutput);
  MessageHandler messageHandler(transport);

  // Grab a notification handler to use for test results.
  llvm::unique_function<void(TestExecutionResult)> onTestResultFn =
      messageHandler.outgoingNotification<TestExecutionResult>(
          "execution/result");
  auto emitError = [&](StringRef message) {
    onTestResultFn(TestExecutionResult::buildInitError(
        tests.front().getTestID(), message));
    for (const Test &test : tests.drop_front())
      onTestResultFn(TestExecutionResult::buildSkip(test.getTestID()));
    return failure();
  };

  // Check that the file path is a mojo source file.
  std::filesystem::path path = tests.front().getTestID().getFilePath();
  if (!Filesystem::isMojoSourceFile(path))
    return emitError("id does not correspond to a valid mojo source file");

  llvm::SourceMgr sourceMgr;
  std::vector<ExecutableTest> executableTests;
  for (const Test &test : tests) {
    // Check that each test corresponds to the same file, we aren't expecting
    // tests to span multiple files. This should never happen, but we should
    // verify it here just in case (`mojo` should only spawn the executor with
    // known valid sets of tests).
    std::filesystem::path path = test.getTestID().getFilePath();
    if (path != tests.front().getTestID().getFilePath())
      return emitError("unexpected tests spanning multiple files");

    // Check that the id corresponds to a specific test.
    std::optional<StringRef> testName = test.getTestID().getTest();
    if (!testName)
      return emitError("id does not correspond to a specific test");

    // Get the test to execute.
    ErrorOr<ExecutableTest> executableTest =
        getExecutableTest(sourceMgr, path, test);
    if (executableTest.isError())
      return emitError(executableTest.getError());
    executableTests.push_back(std::move(*executableTest));
  }

  // Execute the tests, using the top-level directory as the working directory.
  std::filesystem::path workingDirectory = path.parent_path();
  while (Filesystem::isMojoSourcePackagePath(workingDirectory))
    workingDirectory = workingDirectory.parent_path();
  executeTests(std::move(ctx), workingDirectory.string(), includeDirs,
               std::move(executableTests), entrypointFile, onTestResultFn);
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
  llvm::cl::opt<std::string> testFile{
      llvm::cl::Positional, llvm::cl::desc("<File with json tests to run>"),
      llvm::cl::init("-")};
  llvm::cl::opt<std::string> entrypointFile{
      llvm::cl::Positional,
      llvm::cl::desc("An executable entrypoint produced by mojo-test")};
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

  // Read in the set of tests to execute.
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> inputTestsFile =
      toModularErrorOr(llvm::MemoryBuffer::getFileOrSTDIN(testFile));
  if (inputTestsFile.isError()) {
    llvm::errs() << "failed to open test file: " << inputTestsFile.getError()
                 << "\n";
    return 1;
  }
  ErrorOr<std::vector<Test>> tests = toModularErrorOr(
      llvm::json::parse<std::vector<Test>>((*inputTestsFile)->getBuffer()));
  if (tests.isError()) {
    llvm::errs() << "failed to parse test file: " << tests.getError() << "\n";
    return 1;
  }
  if (tests->empty())
    return 0;

  // Run the executor.
  return failed(runTestExecutor(*ctxOr, *tests, prettyPrint, includeDirs,
                                entrypointFile));
}
