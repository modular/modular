//===- KGEN/CLOptions.h ---------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_CLOPTIONS_H
#define KGEN_CLOPTIONS_H

#include "Support/CommonCLOptions.h"
#include "Support/ErrorOr.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/CommandLine.h"

namespace M {
namespace KGEN {
class ExecutionEngine;
}

/// What to do with a given kernel.
enum class Command {
  kEmit,
  kExecute,
};

//===----------------------------------------------------------------------===//
// CommandLineKernel
//===----------------------------------------------------------------------===//

/// This struct gives us a standard way to specify a kernel, its signature, and
/// its output filename on the command line. It also gives us a way to execute
/// this kernel.
struct CommandLineKernel {
  std::string name;
  std::string signature;
  std::string outputFilename;

  /// Verify that the signature of this kernel passed in on the command line
  /// matches the signature of the kernel as it exists in the IR.
  ErrorOrSuccess
  verifyKernelSignature(mlir::LLVM::LLVMFunctionType kernelType) const;
  /// Execute this kernel and print its result(s).
  ErrorOrSuccess executeAndPrint(KGEN::ExecutionEngine &engine) const;
};

/// Provide a parser for the CommandLineKernel object.
class CommandLineKernelParser : public llvm::cl::parser<CommandLineKernel> {
public:
  using llvm::cl::parser<CommandLineKernel>::parser;

  bool parse(llvm::cl::Option &o, StringRef argName, StringRef argValue,
             CommandLineKernel &val);
};

class KGENCLOptions : public CommonCLOptions {
public:
  using CommonCLOptions::CommonCLOptions;

  cl::opt<Command> cmd{
      cl::desc("The command to execute"),
      cl::values(
          clEnumValN(Command::kEmit, "emit", "Emit kernels as object files."),
          clEnumValN(Command::kExecute, "execute", "Execute kernels.")),
      llvm::cl::Required};

  cl::list<CommandLineKernel, bool, CommandLineKernelParser> kernels{
      "kernel", cl::desc("Specifies the kernels to handle. Defaults to an "
                         "empty list, which will do nothing.")};

  Optional<CommandLineKernel>
  shouldHandleKernel(mlir::LLVM::LLVMFuncOp kernel) const {
    auto found = llvm::find_if(kernels, [&](const CommandLineKernel &ek) {
      return ek.name == kernel.getName();
    });
    if (found == kernels.end())
      return None;
    return *found;
  }
};
} // namespace M

#endif // KGEN_CLOPTIONS_H
