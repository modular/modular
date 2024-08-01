//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"
#include <fstream>

using namespace M;

static void dumpIOFile(StringRef streamName, StringRef path) {
  auto bufferOr = toModularErrorOr(llvm::MemoryBuffer::getFile(path));
  if (failed(bufferOr))
    llvm::report_fatal_error(Twine("Error reading the file ") + path + ": " +
                             bufferOr.getError());

  llvm::MemoryBuffer &buffer = *bufferOr->get();
  llvm::errs() << "=====================================\n"
               << "LSP Server " << streamName << " (" << path << "):\n"
               << buffer.getBuffer() << "\n\n";
}

/// Create a test client that asserts the execution doesn't fail and also dumps
/// the contents of server IO files upon errors.
LSPBatchClient M::createTestClient(bool attachDebugger) {
  auto dumpStreamsOnError = [](const LSPBatchClient::ExecutionResult &result) {
    if (failed(result.err)) {
      llvm::errs() << "Fatal error: " << result.err.getError() << "\n";
      if (result.serverIOFiles)
        dumpIOFile("stderr", result.serverIOFiles->serverStderr);
    }

    ASSERT_FALSE(result.err.isError());
  };
  return LSPBatchClient(attachDebugger, dumpStreamsOnError);
}

Document M::createDocumentFromInputFile(StringRef fileName) {
  std::string fullPath = std::filesystem::canonical(
                             std::filesystem::path(std::getenv("MODULAR_PATH")))
                             .lexically_normal() /
                         "KGEN" / "unittests" / "mojo-lsp-server" / "inputs" /
                         fileName.str();
  auto bufferOr = toModularErrorOr(llvm::MemoryBuffer::getFile(fullPath));
  if (failed(bufferOr))
    llvm::report_fatal_error(Twine("Error reading the file ") + fullPath +
                             ": " + bufferOr.getError());
  llvm::MemoryBuffer &buffer = *bufferOr->get();
  return Document("file://" + fullPath, buffer.getBuffer());
}
