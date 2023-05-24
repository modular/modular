//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILATIONOPTIONS_H
#define KGEN_COMPILATIONOPTIONS_H

#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/Target.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/ErrorHandling.h"
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

  /// The compilation abstraction level to generate debug info for, used in
  /// tandem with DebugInfoLevel.
  enum DebugAtLevel {
    /// Generate debug info for the LLVM output.
    kDebugAtLLVM
  };

  /// The sanitizers enabled for the compilation.
  class Sanitizers {
  public:
    /// The various sanitizers that can be enabled.
    enum SanitizerKind { kAddress, kThread };

    Sanitizers(unsigned sanitizerMask = 0) : sanitizerMask(sanitizerMask) {}

    /// Check if the given sanitizer is enabled.
    bool has(SanitizerKind sanitizer) const {
      return sanitizerMask & (1 << sanitizer);
    }

    /// Returns if any sanitizer is enabled.
    operator bool() const { return sanitizerMask != 0; }

  private:
    unsigned sanitizerMask;
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
      std::vector<std::string> linkDirs = {}, bool explicitLinking = false)
      : enableSearch(enableSearch), optimizationLevel(optimizationLevel),
        debugLevel(debugLevel), debugAtLevel(debugAtLevel),
        sanitizers(sanitizers),
        enableXRayInstrumentation(enableXRayInstrumentation),
        targetTriple(std::move(targetTriple)), targetCpu(std::move(targetCpu)),
        targetFeatures(std::move(targetFeatures)),
        linkDirs(std::move(linkDirs)), explicitLinking(explicitLinking) {}

  /// Return the corresponding codegen optimization level for the current option
  /// set.
  llvm::CodeGenOpt::Level getCodeGenOptLevel() const {
    switch (optimizationLevel) {
    case 0:
      return llvm::CodeGenOpt::None;
    case 1:
      return llvm::CodeGenOpt::Less;
    case 2:
      return llvm::CodeGenOpt::Default;
    default:
      return llvm::CodeGenOpt::Aggressive;
    }
  }

  /// Return the corresponding debuginfo emission level for the current option
  /// set.
  DebugInfo::EmissionKind getDIEmissionKind() const {
    switch (debugLevel) {
    case kNoDebug:
      return DebugInfo::EmissionKind::None;
    case kSynthetic:
    case kLineTablesOnly:
      return DebugInfo::EmissionKind::LineTablesOnly;
    case kFullDebugInfo:
      return DebugInfo::EmissionKind::Full;
    }
    llvm_unreachable("unhandled debug level");
  }

  /// Return the debug info level to use when parsing an input file.
  DebugInfoLevel getDebugInfoLevelForInput() const {
    return debugAtLevel ? kNoDebug : debugLevel;
  }

  /// Print the compilation options to the given stream.
  void print(raw_ostream &os) const {
    os << "CompilationOptions { optimizationLevel: " << optimizationLevel;
    if (debugLevel != kNoDebug) {
      os << ", debugLevel: "
         << (debugLevel == kLineTablesOnly ? "line-tables" : "full");
    }
    if (debugAtLevel) {
      os << ", debugAtLevel: ";
      switch (*debugAtLevel) {
      case kDebugAtLLVM:
        os << "llvm";
        break;
      }
    }
    if (sanitizers) {
      os << ", sanitizers: ";
      if (sanitizers.has(Sanitizers::kAddress))
        os << " address";
      if (sanitizers.has(Sanitizers::kThread))
        os << " thread";
    }
    if (enableXRayInstrumentation)
      os << ", enableXRayInstrumentation";

    if (explicitLinking)
      os << ", explicitLinking";

    os << ", linkDirs: [";
    llvm::interleaveComma(linkDirs, os);
    os << "]";
    os << " }";
  }

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
  std::vector<std::string> linkDirs = {};
  bool explicitLinking = false;

  std::string saveTempsPrefix = "";
};
} // namespace M::KGEN

#endif // KGEN_COMPILATIONOPTIONS_H
