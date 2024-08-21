//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/Runtime.h"
#include "CompilationServer.h"
#include "KGEN/Support/Debugging.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/Transport.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"

using namespace M;
using namespace M::KGEN;
using namespace mlir::lsp;

int main(int argc, char **argv) {
  llvm::InitLLVM IL(argc, argv, /*InstallPipeSignalExitHandler=*/false);
  llvm::PrettyStackTraceProgram X(argc, argv);
  llvm::setBugReportMsg(
      "Compilation server has encountered an internal error. "
      "Please submit a Modular internal bug report "
      "and include the crash backtrace along with your command line"
      "invocation.\n");

  llvm::cl::opt<JSONStreamStyle> inputStyle{
      "input-style",
      llvm::cl::desc("Input JSON stream encoding"),
      llvm::cl::values(clEnumValN(JSONStreamStyle::Standard, "standard",
                                  "usual Compilation Server Protocol"),
                       clEnumValN(JSONStreamStyle::Delimited, "delimited",
                                  "messages delimited by `// -----` lines, "
                                  "with // comment support "
                                  "to facilitate debugging")),
      llvm::cl::init(JSONStreamStyle::Standard),
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
      // Print maximum info, while compilation server is under development.
      llvm::cl::init(Logger::Level::Debug),
  };
  llvm::cl::opt<bool> prettyPrint{
      "pretty",
      llvm::cl::desc("Pretty-print JSON output"),
      llvm::cl::init(false),
  };
  llvm::cl::opt<bool> testMode{
      "test",
      llvm::cl::desc(
          "This flags sets up the server in test mode. It effectively sets the "
          "options `-input-style=delimited -pretty -log=verbose "),
      llvm::cl::init(false),
  };
  llvm::cl::opt<bool> attach{
      "attach-debugger-on-startup",
      llvm::cl::desc("Launch the server and start a debug session attached to "
                     "it on VS Code"),
      llvm::cl::init(false),
  };

  llvm::cl::ParseCommandLineOptions(argc, argv, "Compilation Server");

  // When testing, set the flags that make it easier to interact with server.
  if (testMode) {
    inputStyle = JSONStreamStyle::Delimited;
    logLevel = Logger::Level::Debug;
    prettyPrint = true;
  }

  // Configure the logger.
  Logger::setLogLevel(logLevel);

  // Configure the transport used for communication.
  llvm::sys::ChangeStdinToBinary();
  JSONTransport transport(stdin, llvm::outs(), inputStyle, prettyPrint);

  if (attach)
    attachToNewRemoteDebugSession();

  // Start the server.
  // Use single-thread mode for now to keep things simple.
  return failed(runCompilationServer(transport));
}
