//===- Support/CommonCLOptions.h ------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMMONCLOPTIONS_H
#define SUPPORT_COMMONCLOPTIONS_H

#include "Support/CommandLine.h"
#include "Support/ErrorOr.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
namespace M {

/// Contains command-line options that are shared among most of our binaries.
struct CommonCLOptions {
  // Specify the input file for a given binary
  cl::opt<std::string> inputFilename{llvm::cl::Positional,
                                     cl::desc("<input file>"), cl::init("-")};

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
      llvm::errs() << toolName << ": " << errorOrInputFile.takeError() << '\n';
      exit(1);
    }
    return errorOrInputFile.takeValue();
  }

  /// This method creates an MLIR context with the specified memory buffer as
  /// the primary file configured in the source mgr.  It configures it for
  /// diagnostic printing based on the setting of the -verify-diagnostics flag.
  /// This invokes the `bodyFn` callable with the MLIRContext that is set up.
  template <typename BodyFn>
  LogicalResult
  configureMLIRContextAndExecute(std::unique_ptr<llvm::MemoryBuffer> &&buffer,
                                 BodyFn &&bodyFn) const {
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
    return configureMLIRContextAndExecute(sourceMgr,
                                          std::forward<BodyFn>(bodyFn));
  }

  /// This method creates an MLIR context with the specified memory buffer as
  /// the primary file configured in the source mgr.  It configures it for
  /// diagnostic printing based on the setting of the -verify-diagnostics flag.
  /// This invokes the `bodyFn` callable with the MLIRContext and SourceMgr that
  /// is set up.
  template <typename BodyFn>
  LogicalResult configureMLIRContextAndSourceMgrAndExecute(
      std::unique_ptr<llvm::MemoryBuffer> &&buffer, BodyFn &&bodyFn) const {
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
    return configureMLIRContextAndExecute(
        sourceMgr, [&sourceMgr, bodyFn = std::forward<BodyFn>(bodyFn)](
                       mlir::MLIRContext *ctx) { bodyFn(ctx, sourceMgr); });
  }

  /// This method creates an MLIR context and configures it for diagnostic
  /// printing based on the setting of the -verify-diagnostics flag.  This
  /// invokes the `bodyFn` callable with the MLIRContext that is set up.
  template <typename BodyFn>
  LogicalResult configureMLIRContextAndExecute(llvm::SourceMgr &sourceMgr,
                                               BodyFn &&bodyFn) const {
    mlir::MLIRContext context;
    if (verifyDiagnostics) {
      mlir::SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr,
                                                                &context);
      bodyFn(&context);
      return sourceMgrHandler.verify();
    }

    mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    bodyFn(&context);
    return success();
  }
};

} // namespace M

#endif // SUPPORT_COMMONCLOPTIONS_H
