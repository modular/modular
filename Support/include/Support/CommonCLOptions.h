//===- Support/CommonCLOptions.h ------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMMONCLOPTIONS_H
#define SUPPORT_COMMONCLOPTIONS_H

#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"

namespace M {
using llvm::cl::opt;

/// Contains command-line options that are shared among most of our binaries.
struct CommonCLOptions {
  // Specify the input file for a given binary
  cl::opt<std::string> inputFilename{cl::Positional, cl::desc("<input file>"),
                                     cl::init("-")};

  cl::opt<bool> verifyDiagnostics{
      "verify-diagnostics",
      cl::desc("Check that emitted diagnostics match "
               "expected-* lines on the corresponding line"),
      cl::init(false)};

  /// Open the filename specified on the command line and return a memory
  /// buffer, or an error message on failure.
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> openInputFile() {
    std::string errorMsg;
    auto result = mlir::openInputFile(inputFilename, &errorMsg);
    if (result)
      return result;
    return Error(errorMsg);
  }

  /// The common case for all our driver-like tools is to fail early with an
  /// exit error status.  This takes care of that bit of boilerplate.
  std::unique_ptr<llvm::MemoryBuffer>
  openInputFileOrExit(const char *toolName) {
    auto errorOrInputFile = openInputFile();
    if (failed(errorOrInputFile)) {
      errs() << toolName << ": " << errorOrInputFile.takeError() << '\n';
      exit(1);
    }
    return errorOrInputFile.takeValue();
  }
};

} // namespace M

#endif // SUPPORT_COMMONCLOPTIONS_H
