//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMMONCLOPTIONS_H
#define SUPPORT_COMMONCLOPTIONS_H

#include "Support/CommandLine.h"
#include "Support/ErrorOr.h"
#include "Support/FileSystemExtras.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"

namespace llvm {
class ToolOutputFile;
}

namespace M {

/// Contains functionality that's common to all tools.
class CLOptionsBase {
public:
  /// When the 'skipInitLLVM' flag is true, this initializer does not call
  /// InitLLVM.
  CLOptionsBase(int &argc, char **&argv, bool skipInitLLVM = false) {
    if (!skipInitLLVM)
      llvmInitializer.emplace(argc, argv);
    // On windows, InitLLVM may mutate argv, so make sure to get the fresh
    // value.
    programName = argv[0];

    static constexpr StringLiteral bugReportMsg =
        "PLEASE submit a bug report to "
        "https://github.com/modularml/modular/issues and include the crash "
        "backtrace.\n";

    llvm::setBugReportMsg(bugReportMsg.data());
  }

  StringRef getProgramName() const { return programName; }

  int reportError(Twine errorMessage) const {
    llvm::errs() << programName << ": " << errorMessage << "\n";
    return EXIT_FAILURE;
  }

private:
  /// This tells LLVM to print stack traces on crashes, and also handles
  /// multibyte command line options on windows.
  std::optional<llvm::InitLLVM> llvmInitializer;

  /// This is the value of argv[0] when the program launches, used for reporting
  /// error messages.
  StringRef programName;
};

/// Contains command-line options that are shared among most of our binaries.
class CommonCLOptions : public CLOptionsBase {
private:
  llvm::cl::OptionCategory CommonOptionsCategory{"Common command line options"};

public:
  using CLOptionsBase::CLOptionsBase;

  cl::opt<bool> verifyDiagnostics{
      "verify-diagnostics",
      cl::desc("Check that emitted diagnostics match "
               "expected-* lines on the corresponding line"),
      cl::init(false), llvm::cl::cat(CommonOptionsCategory)};

  // Specify the input file for a given binary
  cl::opt<std::string> inputFilename{llvm::cl::Positional,
                                     cl::desc("<input file>"), cl::init("-"),
                                     llvm::cl::cat(CommonOptionsCategory)};

  // Specify the alignment for a given binary file.
  cl::opt<int> inputFileAlignment{"input-file-alignment",
                                  cl::desc("Alignment for opening input file"),
                                  llvm::cl::cat(CommonOptionsCategory)};

  /// Open the filename specified on the command line and return a memory
  /// buffer, or an error message on failure.
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  openInputFile(StringRef inputFile,
                std::optional<llvm::Align> align = std::nullopt) {
    align = (inputFileAlignment != 0) ? llvm::Align(inputFileAlignment) : align;
    return M::openInputFile(inputFile, align.value_or(defaultAlignment));
  }

  /// The common case for all our driver-like tools is to fail early with an
  /// exit error status.  This takes care of that bit of boilerplate.
  /// Takes an optional alignment with priority:
  /// CLI alignment > align argument > default alignment.
  std::unique_ptr<llvm::MemoryBuffer>
  openInputFileOrExit(std::optional<llvm::Align> align = std::nullopt) {
    return openInputFileOrExit(inputFilename, align);
  }

  std::unique_ptr<llvm::MemoryBuffer>
  openInputFileOrExit(StringRef inputFile,
                      std::optional<llvm::Align> align = std::nullopt) {
    auto errorOrInputFile = openInputFile(inputFile, align);
    if (failed(errorOrInputFile))
      exit(reportError(Twine(errorOrInputFile.getError())));
    return errorOrInputFile.takeValue();
  }

  //===--------------------------------------------------------------------===//
  // Emission Options
  //===--------------------------------------------------------------------===//

  cl::opt<std::string> outputFilename{"o", cl::desc("Output filename"),
                                      cl::value_desc("filename"), cl::init("-"),
                                      llvm::cl::cat(CommonOptionsCategory)};

  /// Determine an output file name and open it.
  std::unique_ptr<llvm::ToolOutputFile>
  getOutputFile(bool hasBinaryOutput, StringRef fileExtension = ".mef") const;

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

  //===--------------------------------------------------------------------===//
  // Intermediate Files Options
  //===--------------------------------------------------------------------===//

  cl::opt<bool> saveTemps{
      "save-temps",
      cl::desc("Store the usual 'temporary' intermediate files permanently in "
               "the directory specified by -temps-dir (defaults to the output "
               "directory); name them as auxiliary output files."),
      llvm::cl::Optional, llvm::cl::cat(CommonOptionsCategory)};

  cl::opt<std::string> tempsDir{
      "temps-dir", cl::init(""),
      cl::desc(
          "The directory in which to store 'temporary' intermediate files. No "
          "files will be saved here unless `-save-temps` is also specified."),
      llvm::cl::Optional, llvm::cl::cat(CommonOptionsCategory)};

  /// Determine an intermediate file with extension `ext` and open it.
  std::unique_ptr<llvm::ToolOutputFile>
  getIntermediateFile(StringRef inputName, StringRef ext) const;

  LogicalResult emitArchive(StringRef object) const;

  //===--------------------------------------------------------------------===//
  // Hacks and workarounds
  //===--------------------------------------------------------------------===//

  // TODO(#20024): Until we have a better solution we need to disable oneDNN
  // optimizations in stock tf since they appear to be incompatible with the
  // tf eager C API.
  cl::opt<bool> disableTFOneDNN{
      "disable-tf-onednn",
      cl::desc("Disable the stock TF oneDNN optimizations."), cl::init(false),
      llvm::cl::cat(CommonOptionsCategory)};

private:
  /// Default alignment for input files.
  /// Used only when both client code and CLI do not specify alignment.
  static constexpr llvm::Align defaultAlignment = llvm::Align::Constant<64>();
};

} // namespace M

#endif // SUPPORT_COMMONCLOPTIONS_H
