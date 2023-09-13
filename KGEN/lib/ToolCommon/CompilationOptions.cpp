//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/CompilationOptions.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/ErrorHandling.h"

using namespace M;
using namespace KGEN;

CompilationOptions::CompilationOptions(
    bool enableSearch, unsigned optimizationLevel, DebugInfoLevel debugLevel,
    std::optional<DebugAtLevel> debugAtLevel, Sanitizers sanitizers,
    bool enableXRayInstrumentation, std::string targetTriple,
    std::string targetCpu, std::string targetFeatures,
    std::vector<std::string> linkDirs)
    : enableSearch(enableSearch), optimizationLevel(optimizationLevel),
      debugLevel(debugLevel), debugAtLevel(debugAtLevel),
      sanitizers(sanitizers),
      enableXRayInstrumentation(enableXRayInstrumentation),
      targetTriple(std::move(targetTriple)), targetCpu(std::move(targetCpu)),
      targetFeatures(std::move(targetFeatures)), linkDirs(std::move(linkDirs)) {
}

llvm::CodeGenOpt::Level CompilationOptions::getCodeGenOptLevel() const {
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

DebugInfo::EmissionKind CompilationOptions::getDIEmissionKind() const {
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

void CompilationOptions::print(raw_ostream &os) const {
  os << "CompilationOptions { enableSearch: " << enableSearch
     << ", optimizationLevel: " << optimizationLevel;
  if (debugLevel != kNoDebug) {
    os << ", debugLevel: "
       << (debugLevel == kLineTablesOnly ? "line-tables"
           : debugLevel == kSynthetic    ? "synthetic"
                                         : "full");
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
    os << ", sanitizers:";
    sanitizers.print(os);
  }
  if (enableXRayInstrumentation)
    os << ", enableXRayInstrumentation";

  os << ", linkDirs: [";
  llvm::interleaveComma(linkDirs, os);
  os << "]";
  os << " }";
}
