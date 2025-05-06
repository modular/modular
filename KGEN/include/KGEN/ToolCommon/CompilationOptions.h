//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLCOMMON_COMPILATIONOPTIONS_H
#define KGEN_TOOLCOMMON_COMPILATIONOPTIONS_H

#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "Support/Compiler/Sanitizers.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/MArchTarget/MArchTarget.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/TargetParser/Host.h"
#include <cstddef>

namespace M::KGEN {
/// This class provides a set of options used to control the compilation of
/// KGEN modules.
class CompilationOptions {
public:
  /// The debug info level to use when compiling.
  enum DebugInfoLevel {
    /// Disable debug info generation.
    kNoDebug,

    /// Generate synthetic debug info.
    kSynthetic,

    /// Emit only debug info necessary for generating line number tables.
    kLineTablesOnly,

    /// Generate complete debug info.
    kFullDebugInfo,
  };

  // The language to specify in the debug info.
  enum DebugInfoLanguage {
    kLangC = llvm::dwarf::DW_LANG_C,
    kLangMojo = llvm::dwarf::DW_LANG_Mojo,
  };

  /// The compilation abstraction level to generate debug info for, used in
  /// tandem with DebugInfoLevel.
  enum DebugAtLevel {
    kDebugUnset,
    /// Generate debug info for the LLVM output.
    kDebugAtLLVM
  };

  CompilationOptions(
      unsigned optimizationLevel = 3, DebugInfoLevel debugLevel = kNoDebug,
      std::optional<DebugAtLevel> debugAtLevel = std::nullopt,
      Sanitizers sanitizers = Sanitizers(),
      std::string targetTriple = llvm::sys::getDefaultTargetTriple(),
      std::string targetCpu = llvm::sys::getHostCPUName().str(),
      std::string targetFeatures = getHostCPUFeatures(),
      std::string targetAccelerator =
          M::AsyncRT::Device::getAcceleratorArchOrEmpty(),
      DebugInfoLanguage debugInfoLanguage = kLangMojo,
      std::string searchPaths = "");

  /// Return the corresponding codegen optimization level for the current option
  /// set.
  llvm::CodeGenOptLevel getCodeGenOptLevel() const;

  /// Return the corresponding debuginfo emission level for the current option
  /// set.
  DebugInfo::EmissionKind getDIEmissionKind() const;

  /// Return the debug info level to use when parsing an input file.
  DebugInfoLevel getDebugInfoLevelForInput() const {
    return debugAtLevel ? kNoDebug : debugLevel;
  }

  /// Print the compilation options to the given stream.
  void print(raw_ostream &os) const;

  /// Parse command line defines, adding defaults based on compilation options.
  ErrorOr<EnvAttr> parseDefinesWithDefaults(MLIRContext *ctx,
                                            ArrayRef<std::string> defines);

  /// Get debugLevel as a string that matches how EnvAttr is being set.
  StringRef getDebugLevelString();

  /// Save temporary files to a file with the given prefix.
  void setSaveTemps(std::string prefix) { saveTempsPrefix = prefix; }

  unsigned optimizationLevel = 3;
  DebugInfoLevel debugLevel = kNoDebug;
  std::optional<DebugAtLevel> debugAtLevel;
  Sanitizers sanitizers = Sanitizers();
  bool sharedLibasan = false;
  std::string externalLibasan = {};
  std::string targetTriple = llvm::sys::getDefaultTargetTriple();
  std::string targetCpu = llvm::sys::getHostCPUName().str();
  std::string targetFeatures = getHostCPUFeatures();
  std::string targetDataLayout = "";
  std::optional<llvm::CodeModel::Model> mcmodel = {};
  std::optional<uint64_t> largeDataThreshold = {};
  int64_t loopUnrollingWarnThreshold = 65536;

  std::string targetAccelerator =
      M::AsyncRT::Device::getAcceleratorArchOrEmpty();
  bool isCrossCompilation = false;

  llvm::Reloc::Model relocModel = llvm::Reloc::Model::PIC_;
  DebugInfoLanguage debugInfoLanguage = kLangMojo;

  std::string saveTempsPrefix = "";
  std::string searchPaths = "";

  bool verboseOutput = false;

  // HACK: to disable llvm splitting for some cases.
  // - mojo REPL (#35345)
  // - graph compiler's compilation path where heuristics is needed for
  // performance.
  // - ...
  bool enableLLVMPerFunctionSplitting = false;
  bool enableParallelLLC = true;

  std::string emissionOptions = "";

  // Maximum number of threads to be used by AsyncRT. 0 means use all available.
  size_t numThreads = 0;

  bool disableWarnings = false;
};

bool isGPUTriple(const llvm::Triple &triple);
bool isGPUBackend(const CompilationOptions &options);
bool isNVPTXBackend(const CompilationOptions &options);
bool isAMDBackend(const CompilationOptions &options);

} // namespace M::KGEN

#endif // KGEN_TOOLCOMMON_COMPILATIONOPTIONS_H
