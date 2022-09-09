//===- KGEN/CLOptions.h ---------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_CLOPTIONS_H
#define KGEN_CLOPTIONS_H

#include "KGEN/KGENDialect/KGENOps.h"
#include "Support/CommonCLOptions.h"
#include "Support/ErrorOr.h"
#include "llvm/Support/CommandLine.h"

namespace M {
namespace KGEN {
class CompiledFunc;
class ExecutionEngine;
}

/// What to do with a given KGEN file.
enum class Command {
  kGenLibraryFile,
  kElaborate,
  kEmit,
  kExecute,
};

//===----------------------------------------------------------------------===//
// CommandLineFunc
//===----------------------------------------------------------------------===//

/// This struct gives us a standard way to specify a func, its signature, and
/// its output filename on the command line. It also gives us a way to execute
/// this func. The format of this option is:
///
///  func ::= name `:` (signature)? `:` output-filename
///  signature ::= return-type`(`arg-types...`)`
///
/// where name and output-filename are just strings. Providing the signature is
/// optional.
struct CommandLineFunc {
  std::string name;
  std::string signature;
  std::string outputFilename;

  /// Verify that the signature of this func passed in on the command line
  /// matches the signature of the func as it exists in the IR.
  ErrorOrSuccess verifyFuncSignature(mlir::FunctionType funcType) const;
  /// Execute this func and print its result(s).
  ErrorOrSuccess executeAndPrint(KGEN::CompiledFunc &compiledFunc) const;
};

/// Provide a parser for the CommandLineFunc object.
class CommandLineFuncParser : public llvm::cl::parser<CommandLineFunc> {
public:
  using llvm::cl::parser<CommandLineFunc>::parser;

  bool parse(llvm::cl::Option &o, StringRef argName, StringRef argValue,
             CommandLineFunc &val);
};

class KGENCLOptions : public CommonCLOptions {
public:
  using CommonCLOptions::CommonCLOptions;

  cl::opt<Command> cmd{
      cl::desc("The command to execute"),
      cl::values(
          clEnumValN(Command::kGenLibraryFile, "gen-lib",
                     "Generate a distributable library file."),
          clEnumValN(Command::kElaborate, "elaborate", "Elaborate the input."),
          clEnumValN(Command::kEmit, "emit", "Emit funcs as object files."),
          clEnumValN(Command::kExecute, "execute", "Execute funcs.")),
      llvm::cl::Required};

  cl::list<CommandLineFunc, bool, CommandLineFuncParser> funcs{
      "func", cl::desc("Specifies the funcs to handle. Defaults to an "
                       "empty list, which will do nothing.")};

  Optional<CommandLineFunc> shouldHandleFunc(KGEN::FuncOp func) const {
    auto found = llvm::find_if(funcs, [&](const CommandLineFunc &ek) {
      return ek.name == func.getName();
    });
    if (found == funcs.end())
      return None;
    return *found;
  }
};
} // namespace M

#endif // KGEN_CLOPTIONS_H
