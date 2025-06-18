//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLCOMMON_CLOPTIONS_H
#define KGEN_TOOLCOMMON_CLOPTIONS_H

#include "AsyncRT/Runtime/RuntimeCLOptions.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "Support/CommonCLOptions.h"
#include "Support/ErrorOr.h"
#include "Support/Profiling/TimeProfiler.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include <filesystem>

namespace M {
namespace KGEN {
class CompiledFunc;
class ExecutionEngine;

/// What to do with a given KGEN file.
enum class Command {
  kElaborate,
  kEmit,
  kEmitAssembly,
  kEmitAssemblyVerbose,
  kEmitHeader,
  kEmitLLVM,
  kEmitLLVMOpt,
  kEmitSharedObject,
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

class KGENCommonOptions : public AsyncRT::RuntimeOptions {

public:
  KGENCommonOptions()
      : AsyncRT::RuntimeOptions(
            AsyncRT::RuntimeOptions::WorkQueueType::kThreadPool) {}

  CompilationOptions::DebugInfoLevel debugInfoLevel{
      CompilationOptions::kNoDebug};

  CompilationOptions::DebugAtLevel debugAtLevel{
      CompilationOptions::kDebugUnset};

  CompilationOptions::DebugInfoLanguage debugInfoLanguage{
      CompilationOptions::kLangMojo};

  std::string stripFilePrefix;

  SmallVector<std::string> includePaths;

  SmallVector<std::string> defines;

  bool enableMLIRCrashReproducer{false};

  bool enableLocalMLIRReproducer{true};

  bool timeTrace{false};

  int timeTraceGranularity{0};

  std::string targetTriple{llvm::sys::getDefaultTargetTriple()};

  std::string targetCpu{llvm::sys::getHostCPUName().str()};

  std::string targetAccelerator{""};

  std::string targetFeatures{getHostCPUFeatures()};

  std::string march;

  std::string mcpu;

  std::string mtune;

  std::string mcmodel;

  std::string largeDataThreshold;

  std::string loopUnrollingWarnThreshold;

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
  CompilationOptions getCompilationOptions() const {
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
    if (debugAtLevel != CompilationOptions::kDebugUnset)
      debugAt = debugAtLevel;
    return CompilationOptions(optLevel, debugInfoLevel, debugAt,
                              sanitizerOptions, targetTriple, targetCpu,
                              targetFeatures, targetAccelerator);
  }

  bool optLevel0{false};

  bool optLevel1{false};

  bool optLevel2{false};

  bool optLevel3{false};

  unsigned sanitizerOptions{0};
};
//===----------------------------------------------------------------------===//
// Pass options
//===----------------------------------------------------------------------===//

class KGENPassCLOptions {
#ifndef MODULAR_PRODUCTION
#define DEFINE_CL_OPTION_GETTER(TYPE, VAR_NAME)                                \
  static std::optional<TYPE> VAR_NAME() {                                      \
    if (!passOptions->VAR_NAME.getNumOccurrences() &&                          \
        !passOptions->VAR_NAME.getDefault().hasValue())                        \
      return std::nullopt;                                                     \
    return passOptions->VAR_NAME;                                              \
  }

#define DEFINE_CL_OPTION_WITH_DEFAULT(TYPE, VAR_NAME, STR_NAME, DESC,          \
                                      DEFAULT_VALUE)                           \
  llvm::cl::opt<TYPE> VAR_NAME{STR_NAME, llvm::cl::desc(DESC),                 \
                               llvm::cl::init(DEFAULT_VALUE)};

#define DEFINE_CL_OPTION(TYPE, VAR_NAME, STR_NAME, DESC)                       \
  llvm::cl::opt<TYPE> VAR_NAME{STR_NAME, llvm::cl::desc((DESC))};

#else

#define DEFINE_CL_OPTION_GETTER(TYPE, VAR_NAME)                                \
  static std::optional<TYPE> VAR_NAME() { return passOptions->VAR_NAME; }

#define DEFINE_CL_OPTION(TYPE, VAR_NAME, STR_NAME, DESC)                       \
  std::optional<TYPE> VAR_NAME = std::nullopt;

#define DEFINE_CL_OPTION_WITH_DEFAULT(TYPE, VAR_NAME, STR_NAME, DESC,          \
                                      DEFAULT_VALUE)                           \
  TYPE VAR_NAME = DEFAULT_VALUE;
#endif // !MODULAR_PRODUCTION
public:
  /// Register all options
  static void registerOptions() { *passOptions; }

  DEFINE_CL_OPTION_GETTER(uint64_t, automaticInlineThreshold);
  DEFINE_CL_OPTION_GETTER(uint64_t, parametricInlineThreshold);
  DEFINE_CL_OPTION_GETTER(uint64_t, parametricInlineEstimatedLoopTripCount);
  DEFINE_CL_OPTION_GETTER(size_t, stackReusePromoteToGlobalThreshold);
  DEFINE_CL_OPTION_GETTER(size_t, kgenVerifierMaxErrors);

private:
  struct PassOptions {
    DEFINE_CL_OPTION(uint64_t, automaticInlineThreshold,
                     "kgen-automatic-inline-threshold",
                     "The threshold to automatically inline functions. "
                     "It has higher priority over pass's heuristic.");

    DEFINE_CL_OPTION(uint64_t, parametricInlineThreshold,
                     "kgen-parametric-inline-threshold",
                     "The threshold to inline parametric functions. "
                     "It has higher priority over pass's heuristic.");

    DEFINE_CL_OPTION_WITH_DEFAULT(uint64_t,
                                  parametricInlineEstimatedLoopTripCount,
                                  "kgen-parametric-inline-avg-loop-trip-count",
                                  "Estimated loop trip count that will be used "
                                  "in InlineParametric pass's heuristic.",
                                  4);

    DEFINE_CL_OPTION_WITH_DEFAULT(
        size_t, stackReusePromoteToGlobalThreshold,
        "kgen-stack-reuse-promote-to-global-threshold",
        "Threshold in byte above which a read-only stack "
        "allocations are promoted to global.",
        1024);

    DEFINE_CL_OPTION_WITH_DEFAULT(size_t, kgenVerifierMaxErrors,
                                  "kgen-verifier-max-errors",
                                  "Specify maximum number of errors "
                                  "KGENVerifier pass can emit at once.",
                                  10);
  };

  static llvm::ManagedStatic<PassOptions> passOptions;

#undef DEFINE_CL_OPTION
#undef DEFINE_CL_OPTION_WITH_DEFAULT
#undef DEFINE_CL_OPTION_GETTER
};

//===----------------------------------------------------------------------===//
// CLOptions
//===----------------------------------------------------------------------===//

class KGENCommonCLOptions : public AsyncRT::RuntimeCLOptions {
public:
  KGENCommonCLOptions(KGENCommonOptions &opts)
      : RuntimeCLOptions(opts), options(opts) {}
  KGENCommonOptions &options;

private:
  llvm::cl::OptionCategory KGENOptionsCategory{"KGEN common options"};

  M::cl::MOpt<CompilationOptions::DebugInfoLevel, true> debugInfoLevel{
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
      llvm::cl::location(options.debugInfoLevel),
      llvm::cl::cat(KGENOptionsCategory)};

  M::cl::MOpt<CompilationOptions::DebugAtLevel, true> debugAtLevel{
      "debug-at",
      cl::desc("Generate debug information for the giving abstraction, instead "
               "of the input"),
      cl::values(clEnumValN(CompilationOptions::kDebugAtLLVM, "llvm",
                            "Generate debug information for the LLVM level.")),
      llvm::cl::location(options.debugAtLevel),
      llvm::cl::cat(KGENOptionsCategory)};

  M::cl::MOpt<CompilationOptions::DebugInfoLanguage, true> debugInfoLanguage{
      "debug-info-language",
      llvm::cl::desc("The DWARF language to specify in the debug info. "
                     "Either `C` or `Mojo`. Defaults to `Mojo`."),
      llvm::cl::values(
          clEnumValN(CompilationOptions::kLangC, "C", "C language."),
          clEnumValN(CompilationOptions::kLangMojo, "Mojo", "Mojo language")),
      llvm::cl::location(options.debugInfoLanguage),
      llvm::cl::cat(KGENOptionsCategory)};

  M::cl::MOpt<std::string, true> stripFilePrefix{
      "strip-file-prefix",
      llvm::cl::desc(
          "Strip this prefix from filenames used for diagnostics & debugging."),
      llvm::cl::location(options.stripFilePrefix),
      llvm::cl::cat(KGENOptionsCategory)};

  M::cl::MListOpt<std::string, SmallVector<std::string>> includePaths{
      "I", cl::desc("Path to use to search for included files."),
      llvm::cl::location(options.includePaths),
      llvm::cl::cat(KGENOptionsCategory)};

  M::cl::MListOpt<std::string, SmallVector<std::string>> defines{
      "D",
      cl::desc("Defines passed into Mojo through the environment parameter."),
      llvm::cl::location(options.defines), llvm::cl::cat(KGENOptionsCategory)};

  M::cl::MOpt<bool, true> enableMLIRCrashReproducer{
      "enable-mlir-crash-repro",
      cl::desc("Enable MLIR pass manager crash reproducer generation."),
      llvm::cl::location(options.enableMLIRCrashReproducer),
      llvm::cl::cat(KGENOptionsCategory)};

  M::cl::MOpt<bool, true> enableLocalMLIRReproducer{
      "enable-mlir-local-repro",
      cl::desc("If set, MLIR will attempt to generate a local reproducer."),
      llvm::cl::location(options.enableLocalMLIRReproducer),
      llvm::cl::cat(KGENOptionsCategory)};

  M::cl::MOpt<bool, true> timeTrace{
      "time-trace",
      cl::desc("Turn on time profiler. Generates JSON file "
               "called kgen.trace.json in the derived directory."),
      llvm::cl::location(options.timeTrace),
      llvm::cl::cat(KGENOptionsCategory)};

  M::cl::MOpt<int, true> timeTraceGranularity{
      "time-trace-granularity",
      cl::desc("Minimum time granularity (in microseconds) "
               "traced by time profiler."),
      llvm::cl::location(options.timeTraceGranularity),
      llvm::cl::cat(KGENOptionsCategory)};

  M::cl::MOpt<std::string, true> targetTriple{
      "target-triple",
      cl::desc("Compilation target triple. Defaults to the host target."),
      llvm::cl::location(options.targetTriple),
      llvm::cl::cat(KGENOptionsCategory)};

  M::cl::MOpt<std::string, true> targetCpu{
      "target-cpu",
      cl::desc("Compilation target CPU. Defaults to the host CPU."),
      llvm::cl::location(options.targetCpu),
      llvm::cl::cat(KGENOptionsCategory)};

  M::cl::MOpt<std::string, true> targetFeatures{
      "target-features",
      cl::desc(
          "Compilation target CPU features. Defaults to the host features."),
      llvm::cl::location(options.targetFeatures),
      llvm::cl::cat(KGENOptionsCategory)};

  M::cl::MOpt<std::string, true> march{
      "march", cl::desc("Architecture to generate code for (see --version)"),
      llvm::cl::location(options.march), llvm::cl::cat(KGENOptionsCategory)};

  M::cl::MOpt<std::string, true> mcpu{
      "mcpu", cl::desc("CPU to generate code for"),
      llvm::cl::location(options.mcpu), llvm::cl::cat(KGENOptionsCategory)};

  M::cl::MOpt<std::string, true> mtune{
      "mtune", cl::desc("CPU to tune code for"),
      llvm::cl::location(options.mtune), llvm::cl::cat(KGENOptionsCategory)};

  M::cl::MOpt<std::string, true> acceleratorTarget{
      "target-accelerator",
      cl::desc("The target architecture for the accelerator"),
      llvm::cl::location(options.targetAccelerator),
      llvm::cl::cat(KGENOptionsCategory), llvm::cl::Hidden};

  M::cl::MOpt<std::string, true> mcmodel{
      "mcmodel",
      cl::desc(
          "CodeModel for object file, i.e. small (default), medium or large."),
      llvm::cl::location(options.mcmodel), llvm::cl::cat(KGENOptionsCategory),
      llvm::cl::Hidden};

  M::cl::MOpt<std::string, true> largeDataThreshold{
      "large-data-threshold",
      cl::desc("Choose large data threshold for x86_64 medium code model."),
      llvm::cl::location(options.largeDataThreshold),
      llvm::cl::cat(KGENOptionsCategory), llvm::cl::Hidden};

  M::cl::MOpt<std::string, true> loopUnrollingWarnThreshold{
      "loop-unrolling-warn-threshold",
      cl::desc("Threshold to warn if a loop is unrolled for too many times, "
               "default value is 65536, set to 0 to disable the warning."),
      llvm::cl::location(options.loopUnrollingWarnThreshold),
      llvm::cl::cat(KGENOptionsCategory), llvm::cl::Hidden};

private:
  M::cl::MOpt<bool, true> optLevel0{"O0", cl::desc("Disable all optimizations"),
                                    llvm::cl::location(options.optLevel0),
                                    llvm::cl::cat(KGENOptionsCategory)};
  M::cl::MOpt<bool, true> optLevel1{
      "O1", cl::desc("Enable optimizations, but favor compilation speed"),
      llvm::cl::location(options.optLevel1),
      llvm::cl::cat(KGENOptionsCategory)};
  M::cl::MOpt<bool, true> optLevel2{"O2", cl::desc("Enable most optimizations"),
                                    llvm::cl::location(options.optLevel2),
                                    llvm::cl::cat(KGENOptionsCategory)};
  M::cl::MOpt<bool, true> optLevel3{
      "O3", cl::desc("Aggressively enable all optimizations"),
      llvm::cl::location(options.optLevel3),
      llvm::cl::cat(KGENOptionsCategory)};

  using SanitizerKind = Sanitizers::SanitizerKind;
  M::cl::MBitsOpt<SanitizerKind, unsigned> sanitizerOptions{
      "sanitize", cl::desc("Enable the given sanitizer"),
      cl::values(clEnumValN(SanitizerKind::kAddress, "address",
                            "Enable address sanitizer"),
                 clEnumValN(SanitizerKind::kThread, "thread",
                            "Enable thread sanitizer")),
      llvm::cl::location(options.sanitizerOptions),
      llvm::cl::cat(KGENOptionsCategory)};
};

class KGENOptions : public KGENCommonOptions, public CommonOptions {
public:
  Command cmd{/*required*/};

  llvm::SmallVector<CommandLineFunc> funcs{};
  std::optional<CommandLineFunc> shouldExecuteFunc(StringRef func) const {
    auto found = llvm::find_if(
        funcs, [&](const CommandLineFunc &ek) { return ek.name == func; });
    if (found == funcs.end())
      return std::nullopt;
    return *found;
  }
};

/// Provide a parser for the KGENCLOptions object.
class KGENCLOptionsParser : public llvm::cl::parser<Command> {

public:
  using llvm::cl::parser<Command>::parser;

  bool parse(llvm::cl::Option &o, StringRef argName, StringRef argValue,
             Command &val);
};

class KGENCLOptions : public KGENCommonCLOptions, public CommonCLOptions {

public:
  KGENOptions &options;
  KGENCLOptions(int argc, char **argv, KGENOptions &opts,
                bool skipInitLLVM = false)
      : KGENCommonCLOptions(opts),
        CommonCLOptions(argc, argv, opts, skipInitLLVM), options(opts) {}

private:
  llvm::cl::OptionCategory KGENCLOptionsCategory{"KGEN Command line options"};
  M::cl::MOpt<Command, true, KGENCLOptionsParser> cmd{
      cl::desc("The command to execute"),
      cl::values(
          clEnumValN(Command::kElaborate, "elaborate", "Elaborate the input."),
          clEnumValN(Command::kEmitLLVM, "emit-llvm", "Emit funcs as LLVM IR."),
          clEnumValN(Command::kEmitLLVMOpt, "emit-llvm=opt",
                     "Emit funcs as optimized LLVM IR."),
          clEnumValN(Command::kEmitAssembly, "emit-asm",
                     "Emit the funcs as assembly."),
          clEnumValN(
              Command::kEmitAssemblyVerbose, "emit-asm=verbose",
              "Emit the funcs as vevrbose assembly with preserved comments"),
          clEnumValN(Command::kEmit, "emit", "Emit funcs as an object file."),
          clEnumValN(Command::kEmitHeader, "emit-header",
                     "Emit a C header file with declarations of "
                     "exported functions."),
          clEnumValN(Command::kEmitSharedObject, "emit=shared",
                     "Emit funcs as a shared object (ELF only)."),
          clEnumValN(Command::kExecute, "execute", "Execute funcs.")),
      llvm::cl::location(options.cmd),
      llvm::cl::Required,
      llvm::cl::ValueOptional,
      llvm::cl::cat(KGENCLOptionsCategory)};

  M::cl::MListOpt<CommandLineFunc, llvm::SmallVector<CommandLineFunc>,
                  CommandLineFuncParser>
      funcs{"func", cl::desc("Specifies the funcs to execute."),
            llvm::cl::location(options.funcs),
            llvm::cl::cat(KGENCLOptionsCategory)};
};

} // namespace KGEN
} // namespace M

#endif // KGEN_TOOLCOMMON_CLOPTIONS_H
