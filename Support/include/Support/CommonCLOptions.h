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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

namespace llvm {
class ToolOutputFile;
}

namespace M {

/// Contains functionality that's common to all tools.
class CLOptionsBase {
public:
  CLOptionsBase(StringRef programName) : programName(programName) {}

  StringRef getProgramName() const { return programName; }

  int reportError(Twine errorMessage) const {
    llvm::errs() << programName << ": " << errorMessage << "\n";
    return EXIT_FAILURE;
  }

  /// Open the filename specified as the argument and return a memory buffer, or
  /// an error message on failure.
  static ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  openInputFile(StringRef inputFilename) {
    std::string errorMsg;
    auto result = mlir::openInputFile(inputFilename, &errorMsg);
    if (result)
      return result;
    return Error(errorMsg);
  }

  cl::opt<bool> verifyDiagnostics{
      "verify-diagnostics",
      cl::desc("Check that emitted diagnostics match "
               "expected-* lines on the corresponding line"),
      cl::init(false)};

private:
  /// This is the value of argv[0] when the program launches, used for reporting
  /// error messages.
  StringRef programName;
};

/// Contains command-line options that are shared among most of our binaries.
class CommonCLOptions : public CLOptionsBase {
public:
  CommonCLOptions(StringRef programName) : CLOptionsBase(programName) {}

  // Specify the input file for a given binary
  cl::opt<std::string> inputFilename{llvm::cl::Positional,
                                     cl::desc("<input file>"), cl::init("-")};

  /// Open the filename specified on the command line and return a memory
  /// buffer, or an error message on failure.
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> openInputFile() {
    return CLOptionsBase::openInputFile(inputFilename);
  }

  /// The common case for all our driver-like tools is to fail early with an
  /// exit error status.  This takes care of that bit of boilerplate.
  std::unique_ptr<llvm::MemoryBuffer> openInputFileOrExit() {
    auto errorOrInputFile = openInputFile();
    if (failed(errorOrInputFile))
      exit(reportError(Twine(errorOrInputFile.getError())));
    return errorOrInputFile.takeValue();
  }

  //===--------------------------------------------------------------------===//
  // Emission Options
  //===--------------------------------------------------------------------===//

  cl::opt<std::string> outputFilename{"o", cl::desc("Output filename"),
                                      cl::value_desc("filename"),
                                      cl::init("-")};

  /// Determine an output file name and open it.
  std::unique_ptr<llvm::ToolOutputFile>
  getOutputFile(bool hasBinaryOutput) const;

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
        sourceMgr,
        [&sourceMgr, bodyFn = std::forward<BodyFn>(bodyFn)](
            mlir::MLIRContext *ctx) -> LogicalResult {
          return bodyFn(ctx, sourceMgr);
        });
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
      // If diagnostic verification is enabled, we don't propagate the result.
      (void)bodyFn(&context);
      return sourceMgrHandler.verify();
    }

    mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    return bodyFn(&context);
  }
};

} // namespace M

#endif // SUPPORT_COMMONCLOPTIONS_H
