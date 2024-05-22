//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "../../common/lsp-protocol/Protocol.h"
#include "../LSPBatchClient.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

using namespace M;
namespace lsp = mlir::lsp;

int main(int argc, char **argv) {
  llvm::InitLLVM IL(argc, argv, /*InstallPipeSignalExitHandler=*/false);
  llvm::PrettyStackTraceProgram X(argc, argv);

  llvm::cl::opt<bool> attachDebugger{
      "attach-debugger",
      llvm::cl::desc("Launch the LSP and start a debug session attached to "
                     "it on VS Code."),
      llvm::cl::init(false),
  };

  llvm::cl::opt<std::string> inputFile{
      "inputFile", llvm::cl::desc("The input file to be processed by the LSP."),
      llvm::cl::Positional, llvm::cl::Required};

  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "This simple LSP client receives an input file and spawns an LSP server "
      "to process it. This tool is intended to be used for debugging purposes "
      "and it's supposed to be modified to replicate any desired workflows.");

  auto bufferOr = toModularErrorOr(llvm::MemoryBuffer::getFile(inputFile));
  if (failed(bufferOr))
    llvm::report_fatal_error(Twine("Error reading the file ") + inputFile +
                             ": " + bufferOr.getError());
  llvm::MemoryBuffer &buffer = *bufferOr->get();
  Document doc("file://" + inputFile, buffer.getBuffer());

  // We modify the environment to guarantee that IO files are preserved for
  // inspection.
  setenv("PRESERVE_LSP_IO_FILES", "1", /*overwrite=*/true);

  LSPBatchClient(/*attachDebugger=*/attachDebugger)
      .open(doc)
      .documentSymbol(doc,
                      [](const std::vector<mlir::lsp::DocumentSymbol> &) {
                        // This is left here for demonstrative purposes.
                        // Whenever you need to use this client, just specify
                        // the requests you want to send. You can use this
                        // lambda to print the results, but you can probably
                        // more easily just inspect the stdout/stderr files.
                      })
      .execute();
}
