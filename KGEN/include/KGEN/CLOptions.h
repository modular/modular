//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_CLOPTIONS_H
#define KGEN_CLOPTIONS_H

#include "KGEN/CompilationOptions.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "Support/CommonCLOptions.h"
#include "Support/ErrorOr.h"
#include "llvm/Support/CommandLine.h"

namespace M {
namespace KGEN {
class CompiledFunc;
class ExecutionEngine;
} // namespace KGEN

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
///  func ::= name `:` (signature)?
///  signature ::= return-type`(`arg-types...`)`
///
/// where name is just a string. Providing the signature is optional.
struct CommandLineFunc {
  std::string name;
  std::string signature;

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
      "func", cl::desc("Specifies the funcs to execute.")};

  cl::opt<KGEN::CompilationOptions::DebugInfoLevel> debugInfoLevel{
      "debug-level",
      cl::desc("The level of debug info to use during compilation"),
      cl::values(clEnumValN(KGEN::CompilationOptions::kNoDebug, "none",
                            "Disable all debug info."),
                 clEnumValN(KGEN::CompilationOptions::kLineTablesOnly,
                            "line-tables",
                            "Only generate debug info for line number tables."),
                 clEnumValN(KGEN::CompilationOptions::kFullDebugInfo, "full",
                            "Generate full debug info.")),
      cl::init(KGEN::CompilationOptions::kNoDebug)};

  cl::opt<KGEN::CompilationOptions::DebugAtLevel> debugAtLevel{
      "debug-at",
      cl::desc("Generate debug info for the giving abstraction, instead of the "
               "input"),
      cl::values(clEnumValN(KGEN::CompilationOptions::kDebugAtLLVM, "llvm",
                            "Generate debug info for the LLVM level."))};

  /// Return a compilation options object based on the command line options.
  KGEN::CompilationOptions getCompilationOptions() const {
    // Grab the optimization level. For now use an aggressive default.
    unsigned optLevel = 3;
    if (optLevel0)
      optLevel = 0;
    else if (optLevel1)
      optLevel = 1;
    else if (optLevel2)
      optLevel = 2;

    // Grab the debug-at level.
    Optional<KGEN::CompilationOptions::DebugAtLevel> debugAt;
    if (debugAtLevel.getNumOccurrences())
      debugAt = debugAtLevel;
    return KGEN::CompilationOptions{optLevel, debugInfoLevel, debugAt};
  }

  Optional<CommandLineFunc> shouldExecuteFunc(StringRef func) const {
    auto found = llvm::find_if(
        funcs, [&](const CommandLineFunc &ek) { return ek.name == func; });
    if (found == funcs.end())
      return None;
    return *found;
  }

  std::string getOutputPath() const;

  LogicalResult emitObject(StringRef object) const;

private:
  cl::opt<bool> optLevel0{"O0", cl::desc("Disable all optimizations")};
  cl::opt<bool> optLevel1{
      "O1", cl::desc("Enable optimizations, but favor compilation speed")};
  cl::opt<bool> optLevel2{"O2", cl::desc("Enable most optimizations")};
  cl::opt<bool> optLevel3{"O3",
                          cl::desc("Aggresively enable all optimizations")};
};
} // namespace M

#endif // KGEN_CLOPTIONS_H
