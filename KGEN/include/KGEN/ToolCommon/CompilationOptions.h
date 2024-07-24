//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLCOMMON_COMPILATIONOPTIONS_H
#define KGEN_TOOLCOMMON_COMPILATIONOPTIONS_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "Support/Compiler/Sanitizers.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/MArchTarget/MArchTarget.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/TargetParser/Host.h"

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
      bool enableSearch = true, unsigned optimizationLevel = 3,
      DebugInfoLevel debugLevel = kNoDebug,
      std::optional<DebugAtLevel> debugAtLevel = std::nullopt,
      Sanitizers sanitizers = Sanitizers(),
      bool enableXRayInstrumentation = false,
      std::string targetTriple = llvm::sys::getDefaultTargetTriple(),
      std::string targetCpu = llvm::sys::getHostCPUName().str(),
      std::string targetFeatures = getHostCPUFeatures(),
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

  /// Save temporary files to a file with the given prefix.
  void setSaveTemps(std::string prefix) { saveTempsPrefix = prefix; }

  bool enableSearch = true;
  unsigned optimizationLevel = 3;
  DebugInfoLevel debugLevel = kNoDebug;
  std::optional<DebugAtLevel> debugAtLevel;
  Sanitizers sanitizers = Sanitizers();
  bool enableXRayInstrumentation = false;
  std::string targetTriple = llvm::sys::getDefaultTargetTriple();
  std::string targetCpu = llvm::sys::getHostCPUName().str();
  std::string targetFeatures = getHostCPUFeatures();
  llvm::Reloc::Model relocModel = llvm::Reloc::Model::PIC_;
  DebugInfoLanguage debugInfoLanguage = kLangMojo;

  std::string saveTempsPrefix = "";
  bool emitAllElaboratorDiags = false;
  std::string searchPaths = "";

  // HACK: to disable llvm splitting for some cases.
  // - mojo REPL (#35345)
  // - graph compiler's compilation path where heuristics is needed for
  // performance.
  // - ...
  bool enableLLVMPerFunctionSplitting = false;
  bool enableParallelLLC = true;
};
} // namespace M::KGEN

#endif // KGEN_TOOLCOMMON_COMPILATIONOPTIONS_H
