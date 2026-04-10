//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// kgen-opt driver.
//
// Dispatches to the MLIR or LLVM IR processing path based on the input file
// extension:
//   .mlir, .mojopkg, stdin ('-')         → MLIR IR passes
//   .ll, .bc                             → LLVM IR passes
//   no extension / unrecognised extension → WARNING + MLIR path
//
//===----------------------------------------------------------------------===//

#include "KGEN/tools/kgen-opt/LLVMDriver.h"
#include "KGEN/tools/kgen-opt/MLIRDriver.h"

#include "mlir/IR/DialectRegistry.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"

using namespace M;

//===----------------------------------------------------------------------===//
// IR kind detection
//===----------------------------------------------------------------------===//

namespace {
enum class IRKind { MLIR, LLVM };

/// Determine the IR kind from the input filename extension.
///
///   .mlir, .mojopkg, stdin ('-'), empty  → MLIR (no warning)
///   .ll, .bc                             → LLVM
///   no extension / unknown extension     → WARNING + MLIR
IRKind detectIRKind(llvm::StringRef filename) {
  if (filename == "-" || filename.empty())
    return IRKind::MLIR;

  llvm::StringRef ext = llvm::sys::path::extension(filename);

  if (ext == ".ll" || ext == ".bc")
    return IRKind::LLVM;

  if (ext == ".mlir" || ext == ".mojopkg")
    return IRKind::MLIR;

  if (ext.empty()) {
    llvm::errs() << "WARNING: No file extension for '" << filename
                 << "'; defaulting to MLIR path.\n";
  } else {
    llvm::errs() << "WARNING: Unrecognised file extension '" << ext << "' for '"
                 << filename << "'; defaulting to MLIR path.\n";
  }

  return IRKind::MLIR;
}

} // namespace

//===----------------------------------------------------------------------===//
// main
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  // Must be registered before any cl::opt construction.
  static llvm::codegen::RegisterCodeGenFlags cfg;

  // Initialize LLVM (signal handlers, pretty stack traces, …).
  llvm::InitLLVM y(argc, argv);

  // -----------------------------------------------------------------------
  // Register ALL command-line options before the single parse call.
  // (registerAndParseCLIOptions cannot be used: it registers conflicting
  // positional and -o options and calls cl::ParseCommandLineOptions itself.)
  // -----------------------------------------------------------------------
  mlir::DialectRegistry registry;
  KGEN::Tool::registerMLIRDialectsAndPasses(registry);

  // Shared: input and output filenames.
  static llvm::cl::opt<std::string> inputFilenameOpt(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::init("-"));
  static llvm::cl::opt<std::string> outputFilenameOpt(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));

  if (!KGEN::Tool::registerMLIRPathCLOptions(registry, argc, argv))
    return 1;
  KGEN::Tool::registerLLVMPathCLOptions();

  // -----------------------------------------------------------------------
  // Single option parse.
  // -----------------------------------------------------------------------
  llvm::cl::ParseCommandLineOptions(argc, argv, "kgen optimizer driver\n");

  // -----------------------------------------------------------------------
  // Definitive IR kind detection and dispatch.
  // -----------------------------------------------------------------------
  const IRKind kind = detectIRKind(inputFilenameOpt.getValue());

  if (kind == IRKind::LLVM)
    return KGEN::Tool::runLLVMPath(inputFilenameOpt.getValue(),
                                   outputFilenameOpt.getValue());

  return failed(KGEN::Tool::runMLIRPath(
      inputFilenameOpt.getValue(), outputFilenameOpt.getValue(), registry));
}
