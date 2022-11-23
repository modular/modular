//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILATIONOPTIONS_H
#define KGEN_COMPILATIONOPTIONS_H

#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/CodeGen.h"

namespace M::KGEN {
/// This class provides a set of options used to control the compilation of
/// KGEN modules.
struct CompilationOptions {
  /// The debug info level to use when compiling.
  enum DebugInfoLevel {
    /// Disable debug info generation.
    kNoDebug,

    /// Emit only debug info necessary for generating line number tables.
    kLineTablesOnly,

    /// Generate complete debug info.
    kFullDebugInfo,
  };

  /// The compilation abstraction level to generate debug info for, used in
  /// tadem with DebugInfoLevel.
  enum DebugAtLevel {
    /// Generate debug info for the LLVM output.
    kDebugAtLLVM
  };

  CompilationOptions(unsigned optimizationLevel = 3,
                     DebugInfoLevel debugLevel = kNoDebug,
                     Optional<DebugAtLevel> debugAtLevel = llvm::None)
      : optimizationLevel(optimizationLevel), debugLevel(debugLevel),
        debugAtLevel(debugAtLevel) {}

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
    case kLineTablesOnly:
      return DebugInfo::EmissionKind::LineTablesOnly;
    case kFullDebugInfo:
      return DebugInfo::EmissionKind::Full;
    }
  }

  unsigned optimizationLevel : 2;
  DebugInfoLevel debugLevel = kNoDebug;
  Optional<DebugAtLevel> debugAtLevel;
};
} // namespace M::KGEN

#endif // KGEN_COMPILATIONOPTIONS_H
