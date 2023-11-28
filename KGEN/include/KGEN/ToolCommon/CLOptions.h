//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLCOMMON_CLOPTIONS_H
#define KGEN_TOOLCOMMON_CLOPTIONS_H

#include "KGEN/ToolCommon/CompilationOptions.h"
#include "LLCL/Runtime/RuntimeCLOptions.h"
#include "Support/CommonCLOptions.h"
#include "Support/ErrorOr.h"
#include "Support/Profiling/TimeProfiler.h"
#include "llvm/Support/CommandLine.h"
#include <filesystem>

namespace M {
namespace KGEN {
class CompiledFunc;
class ExecutionEngine;

/// What to do with a given KGEN file.
enum class Command {
  kGenLibraryFile,
  kElaborate,
  kEmitLLVM,
  kEmitAssembly,
  kEmitHeader,
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
  ErrorOrSuccess executeAndPrint(CompiledFunc &compiledFunc) const;
};

/// Provide a parser for the CommandLineFunc object.
class CommandLineFuncParser : public llvm::cl::parser<CommandLineFunc> {
public:
  using llvm::cl::parser<CommandLineFunc>::parser;

  bool parse(llvm::cl::Option &o, StringRef argName, StringRef argValue,
             CommandLineFunc &val);
};

//===----------------------------------------------------------------------===//
// CLOptions
//===----------------------------------------------------------------------===//

class KGENCommonOptions : public LLCL::RuntimeWorkQueueCLOptions {
private:
  llvm::cl::OptionCategory KGENOptionsCategory{"KGEN common options"};

public:
  cl::opt<CompilationOptions::DebugInfoLevel> debugInfoLevel{
      "debug-level",
      cl::desc("The level of debug information to use during compilation"),
      cl::values(
          clEnumValN(CompilationOptions::kNoDebug, "none",
                     "Disable all debug information."),
          clEnumValN(CompilationOptions::kSynthetic, "synthetic",
                     "Generate synthetic debug information."),
          clEnumValN(CompilationOptions::kLineTablesOnly, "line-tables",
                     "Only generate debug information for line number tables."),
          clEnumValN(CompilationOptions::kFullDebugInfo, "full",
                     "Generate full debug information.")),
      cl::init(CompilationOptions::kNoDebug),
      llvm::cl::cat(KGENOptionsCategory)};

  cl::opt<CompilationOptions::DebugAtLevel> debugAtLevel{
      "debug-at",
      cl::desc("Generate debug information for the giving abstraction, instead "
               "of the input"),
      cl::values(clEnumValN(CompilationOptions::kDebugAtLLVM, "llvm",
                            "Generate debug information for the LLVM level.")),
      llvm::cl::cat(KGENOptionsCategory)};

  cl::opt<CompilationOptions::DebugInfoLanguage> debugInfoLanguage{
      "debug-info-language",
      llvm::cl::desc("The DWARF language to specify in the debug info. "
                     "Either `C` or `Mojo`. Defaults to `Mojo`."),
      llvm::cl::values(
          clEnumValN(CompilationOptions::kLangC, "C", "C language."),
          clEnumValN(CompilationOptions::kLangMojo, "Mojo", "Mojo language")),
      llvm::cl::init(CompilationOptions::kLangMojo),
      llvm::cl::cat(KGENOptionsCategory)};

  cl::opt<bool> enableXRayInstrumentation{
      "xray-instrument",
      cl::desc("Enable XRay instrumentation for the generated code."),
      cl::init(false), llvm::cl::cat(KGENOptionsCategory)};

  cl::opt<bool> enableSearch{
      "enable-search", cl::init(false),
      cl::desc("Do search when an evaluator is provided."),
      llvm::cl::cat(KGENOptionsCategory)};

  cl::list<std::string> includePaths{
      "I", cl::desc("Path to use to search for included files."),
      llvm::cl::cat(KGENOptionsCategory)};

  cl::list<std::string> linkPaths{
      "L", cl::desc("Path to use to search for linked libraries/objects."),
      llvm::cl::cat(KGENOptionsCategory)};

  cl::list<std::string> defines{
      "D",
      cl::desc("Defines passed into Mojo through the environment parameter."),
      llvm::cl::cat(KGENOptionsCategory)};

  cl::opt<bool> enableMLIRCrashReproducer{
      "enable-mlir-crash-repro",
      cl::desc("Enable MLIR pass manager crash reproducer generation."),
      cl::init(false), llvm::cl::cat(KGENOptionsCategory)};

  cl::opt<bool> enableLocalMLIRReproducer{
      "enable-mlir-local-repro",
      cl::desc("If set, MLIR will attempt to generate a local reproducer."),
      cl::init(false), llvm::cl::cat(KGENOptionsCategory)};

  cl::opt<bool> timeTrace{
      "time-trace",
      cl::desc("Turn on time profiler. Generates JSON file "
               "called kgen.trace.json in the derived directory."),
      llvm::cl::cat(KGENOptionsCategory)};

  cl::opt<int> timeTraceGranularity{
      "time-trace-granularity",
      cl::desc("Minimum time granularity (in microseconds) "
               "traced by time profiler."),
      cl::init(0), llvm::cl::cat(KGENOptionsCategory)};

  cl::opt<std::string> targetTriple{
      "target-triple",
      cl::desc("Compilation target triple. Defaults to the host target."),
      cl::init(llvm::sys::getDefaultTargetTriple()),
      llvm::cl::cat(KGENOptionsCategory)};

  cl::opt<std::string> targetCpu{
      "target-cpu",
      cl::desc("Compilation target CPU. Defaults to the host CPU."),
      cl::init(llvm::sys::getHostCPUName().str()),
      llvm::cl::cat(KGENOptionsCategory)};

  cl::opt<std::string> targetFeatures{
      "target-features",
      cl::desc(
          "Compilation target CPU features. Defaults to the host features."),
      cl::init(getHostCPUFeatures()), llvm::cl::cat(KGENOptionsCategory)};

  cl::opt<std::string> march{
      "march", cl::desc("Architecture to generate code for (see --version)"),
      llvm::cl::cat(KGENOptionsCategory)};

  cl::opt<std::string> mcpu{"mcpu", cl::desc("CPU to generate code for"),
                            llvm::cl::cat(KGENOptionsCategory)};

  cl::opt<std::string> mtune{"mtune", cl::desc("CPU to tune code for"),
                             llvm::cl::cat(KGENOptionsCategory)};

  KGENCommonOptions() : RuntimeWorkQueueCLOptions(WorkQueueType::kThreadPool) {}

  /// Get the include directories that exist on the file system.
  std::vector<std::string> getIncludePaths() const {
    std::vector<std::string> result;
    result.reserve(includePaths.size());
    for (auto &path : includePaths)
      if (std::filesystem::is_directory(path))
        result.push_back(path);
    return result;
  }

  /// Return a compilation options object based on the command line options.
  CompilationOptions getCompilationOptions() {
    // Grab the optimization level. For now use an aggressive default.
    unsigned optLevel = 3;
    if (optLevel0)
      optLevel = 0;
    else if (optLevel1)
      optLevel = 1;
    else if (optLevel2)
      optLevel = 2;

    // Grab the debug-at level.
    std::optional<CompilationOptions::DebugAtLevel> debugAt;
    if (debugAtLevel.getNumOccurrences())
      debugAt = debugAtLevel;
    return CompilationOptions(enableSearch, optLevel, debugInfoLevel, debugAt,
                              sanitizerOptions.getBits(),
                              enableXRayInstrumentation, targetTriple,
                              targetCpu, targetFeatures, linkPaths);
  }

private:
  cl::opt<bool> optLevel0{"O0", cl::desc("Disable all optimizations"),
                          llvm::cl::cat(KGENOptionsCategory)};
  cl::opt<bool> optLevel1{
      "O1", cl::desc("Enable optimizations, but favor compilation speed"),
      llvm::cl::cat(KGENOptionsCategory)};
  cl::opt<bool> optLevel2{"O2", cl::desc("Enable most optimizations"),
                          llvm::cl::cat(KGENOptionsCategory)};
  cl::opt<bool> optLevel3{"O3",
                          cl::desc("Aggressively enable all optimizations"),
                          llvm::cl::cat(KGENOptionsCategory)};

  using SanitizerKind = Sanitizers::SanitizerKind;
  llvm::cl::bits<SanitizerKind> sanitizerOptions{
      "sanitize", cl::desc("Enable the given sanitizer"),
      cl::values(clEnumValN(SanitizerKind::kAddress, "address",
                            "Enable address sanitizer"),
                 clEnumValN(SanitizerKind::kThread, "thread",
                            "Enable thread sanitizer")),
      llvm::cl::cat(KGENOptionsCategory)};
};

class KGENCLOptions : public KGENCommonOptions, public CommonCLOptions {
  llvm::cl::OptionCategory KGENCLOptionsCategory{"KGEN Command line options"};

public:
  using CommonCLOptions::CommonCLOptions;

  cl::opt<Command> cmd{
      cl::desc("The command to execute"),
      cl::values(
          clEnumValN(Command::kGenLibraryFile, "gen-lib",
                     "Generate a distributable library file."),
          clEnumValN(Command::kElaborate, "elaborate", "Elaborate the input."),
          clEnumValN(Command::kEmitLLVM, "emit-llvm", "Emit funcs as LLVM IR."),
          clEnumValN(Command::kEmitAssembly, "emit-asm",
                     "Emit the funcs as assembly."),
          clEnumValN(Command::kEmit, "emit", "Emit funcs as object files."),
          clEnumValN(
              Command::kEmitHeader, "emit-header",
              "Emit a C header file with declarations of exported functions."),
          clEnumValN(Command::kExecute, "execute", "Execute funcs.")),
      llvm::cl::Required, llvm::cl::cat(KGENCLOptionsCategory)};

  cl::list<CommandLineFunc, bool, CommandLineFuncParser> funcs{
      "func", cl::desc("Specifies the funcs to execute."),
      llvm::cl::cat(KGENCLOptionsCategory)};

  std::optional<CommandLineFunc> shouldExecuteFunc(StringRef func) const {
    auto found = llvm::find_if(
        funcs, [&](const CommandLineFunc &ek) { return ek.name == func; });
    if (found == funcs.end())
      return std::nullopt;
    return *found;
  }
};

//===----------------------------------------------------------------------===//
// TraceProfiler
//===----------------------------------------------------------------------===//

/// Common trace profiler setup.
struct TraceProfiler {
  TraceProfiler(bool enabled, int timeTraceGranularity) {
    if (enabled)
      initialize(timeTraceGranularity);
  }
  ~TraceProfiler();

private:
  void initialize(int timeTraceGranularity);

  std::optional<TimeTraceProfiler> profiler;
  std::filesystem::path outputFilePath;
};
} // namespace KGEN
} // namespace M

#endif // KGEN_TOOLCOMMON_CLOPTIONS_H
