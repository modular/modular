//===- KGEN/CLOptions.h ---------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_CLOPTIONS_H
#define KGEN_CLOPTIONS_H

#include "Support/CommonCLOptions.h"
#include "Support/ErrorOr.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "llvm/Support/CommandLine.h"

namespace M {
namespace KGEN {
class ExecutionEngine;
}

//===----------------------------------------------------------------------===//
// ExecutableKernel
//===----------------------------------------------------------------------===//

/// This struct provides a way to parse a kernel name and signature from the
/// command line.
struct ExecutableKernel {
  std::string name;
  std::string signature;

  /// Verify that the signature of this kernel passed in on the command line
  /// matches the signature of the kernel as it exists in the IR.
  ErrorOrSuccess
  verifyKernelSignature(mlir::LLVM::LLVMFunctionType kernelType) const;
  /// Execute this kernel and print its result(s).
  ErrorOrSuccess executeAndPrint(KGEN::ExecutionEngine &engine) const;
};

/// Parse ExecutableKernel objects from the command line flags provided.
class ExecutableKernelParser : public llvm::cl::parser<ExecutableKernel> {
public:
  using llvm::cl::parser<ExecutableKernel>::parser;

  bool parse(llvm::cl::Option &o, StringRef argName, StringRef argValue,
             ExecutableKernel &val);
};

//===----------------------------------------------------------------------===//
// EmittableKernel
//===----------------------------------------------------------------------===//

/// This struct provides a way to parse a kernel name and an output object file
/// from the command line.
struct EmittableKernel {
  std::string name;
  std::string outputFilename;
};

/// Parse EmittableKernel objects from the command line flags provided.
class EmittableKernelParser : public llvm::cl::parser<EmittableKernel> {
public:
  using llvm::cl::parser<EmittableKernel>::parser;

  bool parse(llvm::cl::Option &o, StringRef argName, StringRef argValue,
             EmittableKernel &val);
};
} // namespace M

#endif // KGEN_CLOPTIONS_H
