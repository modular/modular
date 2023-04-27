//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines an entry point for a dummy executable used by the Mojo
// REPL. This provides an anchor point for the debugger to run REPL expressions,
// as LLDB requires an in-memory target.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CompilerRT.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>
#include <iostream>

using namespace M;

//===----------------------------------------------------------------------===//
// Kernel C-API
//===----------------------------------------------------------------------===//

/// This is copied from Kernel.cpp so we have symbolic names for the possible
/// output states.
enum ExecutionFinishedState : int {
  kNotFinished = 0,
  kFinishedSuccessfully = 1,
  kFinishedError = 2,
};

using OutputFn = void (*)(const char *, const char *);

using OpaqueMojoKernel = void *;

MODULAR_EXPORT OpaqueMojoKernel initMojoKernel(OutputFn outputFn,
                                               const char *mojoReplExe);
MODULAR_EXPORT void startMojoExecution(OpaqueMojoKernel kernel,
                                       const char *cellId, const char *code,
                                       int storeHistory);
MODULAR_EXPORT int checkMojoExecutionFinished(OpaqueMojoKernel kernel);
MODULAR_EXPORT void destroyMojoKernel(OpaqueMojoKernel kernel);

//===----------------------------------------------------------------------===//
// MojoKernel
//===----------------------------------------------------------------------===//

namespace {
class MojoKernel {
public:
  MojoKernel(OutputFn outputFn, const char *mojoReplExe)
      : kernel(initMojoKernel(outputFn, mojoReplExe)) {}
  ~MojoKernel() {
    if (kernel)
      destroyMojoKernel(kernel);
  }

  /// The kernel is valid if it is non-null.
  operator bool() const { return kernel; }

  //===--------------------------------------------------------------------===//
  // Execution
  //===--------------------------------------------------------------------===//

  /// Start a new cell execution.
  void startExecution(const char *cellId, const char *code) {
    startMojoExecution(kernel, cellId, code, /*storeHistory=*/true);
  }

  /// Check if the current execution has finished.
  int hasExecutionFinished() const {
    return checkMojoExecutionFinished(kernel);
  }

private:
  /// The internal kernel object returned from the jupyter kernel.
  OpaqueMojoKernel kernel;
};
} // namespace

//===----------------------------------------------------------------------===//
// REPL Executor
//===----------------------------------------------------------------------===//

/// This function provides a REPL-like experience that calls into the Jupyter
/// kernel. You can enter code at the prompt and execute it - the prompt itself
/// will tell you the current 'cell'. This does not change automatically because
/// we need to be able to (for example) print the logs generated for the current
/// cell. In order to switch cells, you can use `:next-cell` or
/// `:prev-cell`. In order to exit cleanly, use `:exit`.
static void executeAsREPL(MojoKernel &kernel, StringRef currentCell = "") {
  int idx = 0;
  std::string cellPrefix =
      currentCell.empty() ? "[0] > " : ("[" + currentCell + "] > ").str();
  std::cout << cellPrefix;
  for (std::string line; std::getline(std::cin, line);) {
    // Allow the program to exit cleanly.
    if (line == ":exit")
      break;

    // Print the prompt at the end.
    auto scope = llvm::make_scope_exit([&]() { std::cout << cellPrefix; });

    // Allow the user control over which cell is executing. This is useful for
    // things like executing an expression, and then dumping the logs.
    if (line == ":next-cell") {
      cellPrefix = "[" + std::to_string(++idx) + "] > ";
      continue;
    }
    if (line == ":prev-cell") {
      cellPrefix = "[" + std::to_string(--idx) + "] > ";
      continue;
    }

    kernel.startExecution(cellPrefix.c_str(), line.c_str());
    while (!kernel.hasExecutionFinished())
      continue;
  }
}

//===----------------------------------------------------------------------===//
// Jupyter Executor
//===----------------------------------------------------------------------===//

static LogicalResult executeNotebook(MojoKernel &kernel, StringRef notebookPath,
                                     bool debugOnFailure) {
  std::string errorMsg;
  std::unique_ptr<llvm::MemoryBuffer> notebookFile =
      mlir::openInputFile(notebookPath, &errorMsg);
  if (!notebookFile) {
    llvm::errs() << "error opening notebook file: " << errorMsg << "\n";
    return failure();
  }

  // Parse the notebook file into a json object.
  auto notebookJSON = llvm::json::parse(notebookFile->getBuffer());
  if (!notebookJSON) {
    llvm::errs() << "error parsing notebook file: " << notebookJSON.takeError()
                 << "\n";
    return failure();
  }
  auto notebook = notebookJSON->getAsObject();
  if (!notebook) {
    llvm::errs() << "error parsing notebook file: not a json object\n";
    return failure();
  }

  // Check that we can actually find the cells.
  auto *cells = notebook->getArray("cells");
  if (!cells) {
    llvm::errs() << "error parsing notebook file: no cells found\n";
    return failure();
  }

  for (const auto &[index, cell] : llvm::enumerate(*cells)) {
    // We only care about code cells.
    auto *cellObj = cell.getAsObject();
    auto *source = cellObj->getArray("source");
    if (cellObj->getString("cell_type") != "code" || !source)
      continue;

    // Concatenate all of the lines of the cell into a single string.
    std::string code;
    for (const auto &line : *source)
      if (std::optional<StringRef> lineStr = line.getAsString())
        code += *lineStr;
    if (code.empty())
      continue;
    code += "\n\n";

    // Execute the cell code.
    std::string cellName = ("notebook_cell_" + Twine(index)).str();
    kernel.startExecution(cellName.c_str(), code.c_str());
    // If we finish with an error and we're in debug mode, drop into REPL mode
    // in the current cell.
    int finishState;
    do {
      finishState = kernel.hasExecutionFinished();

      // We hit an error, exit failure.
      if (finishState == kFinishedError) {
        // Drop into REPL mode if requested.
        if (debugOnFailure)
          executeAsREPL(kernel, cellName);

        return failure();
      }
    } while (finishState == kNotFinished);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

int main(int argc, char *argv[]) {
  llvm::cl::opt<std::string> notebookPath(
      llvm::cl::desc("Optional notebook to execute, if not present the "
                     "executor will run in REPL mode"),
      llvm::cl::Positional, llvm::cl::Optional);
  llvm::cl::opt<bool> debugOnFailure(
      "debug-on-failure", llvm::cl::desc("Drop into REPL mode on cell failure"),
      llvm::cl::init(false));
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Determine the path of the repl entry point.
  std::optional<std::string> pathOr =
      llvm::sys::Process::GetEnv("MODULAR_PATH");
  std::filesystem::path exePath = std::filesystem::path(pathOr.value_or(".")) /
                                  ".derived" / "build" / "lib" /
                                  "mojo-repl-entry-point";

  // Initialize the kernel.
  MojoKernel kernel(
      [](const char *kind, const char *msg) {
        llvm::outs() << "[" << kind << "] " << msg << "\n";
      },
      exePath.c_str());

  // If we have a notebook path, execute it, otherwise run in REPL mode.
  if (notebookPath.getNumOccurrences()) {
    if (failed(executeNotebook(kernel, notebookPath, debugOnFailure)))
      return 1;
  } else {
    executeAsREPL(kernel);
  }
  return 0;
}
