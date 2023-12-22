//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/Runtime.h"
#include "LSPServer.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Transport.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include <csignal>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace M;
using namespace M::KGEN::LIT;
using namespace mlir::lsp;

int main(int argc, char **argv) {
  llvm::InitLLVM IL(argc, argv, /*InstallPipeSignalExitHandler=*/false);
  llvm::PrettyStackTraceProgram X(argc, argv);
  llvm::setBugReportMsg(
      "Please submit a bug report to https://github.com/modularml/mojo/issues "
      "and include the crash backtrace along with all the relevant source "
      "codes with the contents they had at crash time.\n");

  llvm::cl::opt<JSONStreamStyle> inputStyle{
      "input-style",
      llvm::cl::desc("Input JSON stream encoding"),
      llvm::cl::values(clEnumValN(JSONStreamStyle::Standard, "standard",
                                  "usual LSP protocol"),
                       clEnumValN(JSONStreamStyle::Delimited, "delimited",
                                  "messages delimited by `// -----` lines, "
                                  "with // comment support")),
      llvm::cl::init(JSONStreamStyle::Standard),
      llvm::cl::Hidden,
  };
  llvm::cl::opt<bool> mojoTest{
      "mojo-test",
      llvm::cl::desc(
          "Abbreviation for -input-style=delimited -pretty -log=verbose. "
          "Intended to simplify mojo tests"),
      llvm::cl::init(false),
  };
  // FIXME(23386): delete this when proper package decls support is added.
  llvm::cl::opt<bool> parseStdlib{
      "parse-stdlib",
      llvm::cl::desc(
          "Parses the input file(s) as the Mojo standard library, which allows "
          "jumping to definitions in files in the sdlib"),
      llvm::cl::init(false),
      llvm::cl::Hidden,
  };
  llvm::cl::opt<Logger::Level> logLevel{
      "log",
      llvm::cl::desc("Verbosity of log messages written to stderr"),
      llvm::cl::values(
          clEnumValN(Logger::Level::Error, "error", "Error messages only"),
          clEnumValN(Logger::Level::Info, "info",
                     "High level execution tracing"),
          clEnumValN(Logger::Level::Debug, "verbose", "Low level details")),
      // We are still in basic development mode, so we set the logLevel to Debug
      // to get more additional information for troubleshooting. When we become
      // more confident of the LSP, we can switch this back to Info.
      llvm::cl::init(Logger::Level::Debug),
  };
  llvm::cl::opt<bool> prettyPrint{
      "pretty",
      llvm::cl::desc("Pretty-print JSON output"),
      llvm::cl::init(false),
  };
  llvm::cl::opt<bool> suspended{
      "suspended",
      llvm::cl::desc("Launch the server in a suspended state waiting for a "
                     "debugger to attach"),
      llvm::cl::init(false),
  };

  llvm::cl::ParseCommandLineOptions(argc, argv, "Mojo LSP Language Server");

  // When testing, updating flags that make the server a bit easier to interact
  // with.
  if (mojoTest) {
    inputStyle = JSONStreamStyle::Delimited;
    logLevel = Logger::Level::Debug;
    prettyPrint = true;
  }

  // Wait for the server to shutdown when testing.
  bool waitOnShutdown = mojoTest;

  // Configure the logger.
  Logger::setLogLevel(logLevel);

  // Configure the transport used for communication.
  llvm::sys::ChangeStdinToBinary();
  JSONTransport transport(stdin, llvm::outs(), inputStyle, prettyPrint);

  // Register the additionally supported URI schemes for the server.
  URIForFile::registerSupportedScheme("vscode-notebook-cell");

  if (suspended) {
    Logger::info(
        "The server has been launched in a suspended state. It's pid "
        "is {0} and you can use a debugger to attach to it with, for example, "
        "`lldb -p {0}`.",
        llvm::sys::Process::getProcessId());
#ifdef _WIN32
    while (!IsDebuggerPresent())
      Sleep(1000);
#else
    std::raise(SIGSTOP);
#endif
  }

  // Start the server.
  // When testing we use a single thread to provide deterministic output.
  return failed(runMojoLSPServer(transport, /*singleThreaded=*/mojoTest,
                                 waitOnShutdown, parseStdlib));
}
