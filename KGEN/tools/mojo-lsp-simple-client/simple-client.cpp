//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "../mojo-lsp-test-client/LSPBatchClient.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace M;
namespace lsp = llvm::lsp;

int main(int argc, char **argv) {
  llvm::InitLLVM il(argc, argv, /*InstallPipeSignalExitHandler=*/false);
  llvm::PrettyStackTraceProgram x(argc, argv);

  llvm::cl::opt<bool> attachDebugger{
      "attach-debugger",
      llvm::cl::desc("Launch the LSP and start a debug session attached to "
                     "it on VS Code."),
      llvm::cl::init(false),
  };

  llvm::cl::opt<bool> keepIOFiles{
      "keep-io-files",
      llvm::cl::desc(
          "Preserve the server's stdin/stdout/stderr temp files and print "
          "their paths, even on success. Useful for inspecting raw LSP "
          "traffic. On failure the files are always preserved."),
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

  if (keepIOFiles)
    setenv("PRESERVE_LSP_IO_FILES", "1", /*overwrite=*/true);

  // By default, we include the following requests that don't require any
  // special input.
  auto result =
      LSPBatchClient(/*attachDebugger=*/attachDebugger)
          .open(doc)
          .documentSymbol(doc,
                          [](const std::vector<llvm::lsp::DocumentSymbol> &) {
                            // This is left here for demonstrative purposes.
                            // Whenever you need to use this client, just
                            // specify the requests you want to send. You can
                            // use this lambda to print the results, but you can
                            // probably more easily just inspect the
                            // stdout/stderr files.
                          })
          .semanticTokensFull(doc, [&](ArrayRef<Mojo::LSP::SemanticToken>) {})
          .hoverNullable(doc, {0, 0},
                         [&](const std::optional<lsp::Hover2> &) {})
          .execute();

  if (failed(result.err) && result.serverIOFiles) {
    // Stream the server's stderr inline so crash details are immediately
    // visible without manually cat-ing the temp file.
    if (auto stderrBuf =
            llvm::MemoryBuffer::getFile(result.serverIOFiles->serverStderr)) {
      StringRef content = (*stderrBuf)->getBuffer();
      llvm::errs() << content;
    }
  }

  return failed(result.err) ? EXIT_FAILURE : EXIT_SUCCESS;
}
