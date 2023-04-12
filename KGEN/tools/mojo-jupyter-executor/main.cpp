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
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>
#include <iostream>

namespace M::KGEN::Mojo {
class MojoKernel;
class ExpressionExecutionState;
} // namespace M::KGEN::Mojo

using namespace M::KGEN::Mojo;

using OutputFn = void (*)(const char *, const char *);

extern "C" MojoKernel *initMojoKernel(OutputFn outputFn,
                                      const char *mojoReplExe);
extern "C" ExpressionExecutionState *
startMojoExecution(MojoKernel *kernel, const char *cellId, const char *code);
extern "C" int checkMojoExecutionFinished(MojoKernel *kernel,
                                          ExpressionExecutionState *state);
extern "C" void destroyMojoKernel(MojoKernel *kernel);

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

/// This main function provides a REPL-like experience that calls into the
/// Jupyter kernel. You can enter code at the prompt and execute it - the prompt
/// itself will tell you the current 'cell'. This does not change automatically
/// because we need to be able to (for example) print the logs generated for the
/// current cell. In order to switch cells, you can use `:next-cell` or
/// `:prev-cell`. In order to exit cleanly, use `:exit`.

int main(int argc, char *argv[]) {
  std::optional<std::string> pathOr =
      llvm::sys::Process::GetEnv("MODULAR_PATH");
  std::filesystem::path exePath = std::filesystem::path(pathOr.value_or(".")) /
                                  ".derived" / "build" / "lib" /
                                  "mojo-repl-entry-point";

  auto *kernel = initMojoKernel(
      [](const char *kind, const char *msg) {
        llvm::outs() << "[" << kind << "] " << msg << "\n";
      },
      exePath.c_str());
  int idx = 0;
  std::cout << "[" << idx << "] > ";
  for (std::string line; std::getline(std::cin, line);) {
    // Allow the program to exit cleanly.
    if (line == ":exit")
      break;

    // Print the prompt at the end.
    auto scope =
        llvm::make_scope_exit([&]() { std::cout << "[" << idx << "] > "; });

    // Allow the user control over which cell is executing. This is useful for
    // things like executing an expression, and then dumping the logs.
    if (line == ":next-cell") {
      ++idx;
      continue;
    }
    if (line == ":prev-cell") {
      --idx;
      continue;
    }

    auto *execution =
        startMojoExecution(kernel, std::to_string(idx).c_str(), line.c_str());
    while (!checkMojoExecutionFinished(kernel, execution))
      continue;
  }
  destroyMojoKernel(kernel);
}
