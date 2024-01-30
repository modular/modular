//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/CompilationOptions.h"
#include "Support/MDialect/MAttrs.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/ErrorHandling.h"

using namespace M;
using namespace KGEN;

CompilationOptions::CompilationOptions(
    bool enableSearch, unsigned optimizationLevel, DebugInfoLevel debugLevel,
    std::optional<DebugAtLevel> debugAtLevel, Sanitizers sanitizers,
    bool enableXRayInstrumentation, std::string targetTriple,
    std::string targetCpu, std::string targetFeatures,
    std::vector<std::string> linkDirs, DebugInfoLanguage debugInfoLanguage,
    std::string searchPaths)
    : enableSearch(enableSearch), optimizationLevel(optimizationLevel),
      debugLevel(debugLevel), debugAtLevel(debugAtLevel),
      sanitizers(sanitizers),
      enableXRayInstrumentation(enableXRayInstrumentation),
      targetTriple(std::move(targetTriple)), targetCpu(std::move(targetCpu)),
      targetFeatures(std::move(targetFeatures)), linkDirs(std::move(linkDirs)),
      debugInfoLanguage(debugInfoLanguage), searchPaths(searchPaths) {}

llvm::CodeGenOptLevel CompilationOptions::getCodeGenOptLevel() const {
  if (auto level = llvm::CodeGenOpt::getLevel(optimizationLevel))
    return *level;
  // Default to "Aggressive" optimizations.
  return llvm::CodeGenOptLevel::Aggressive;
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
    case kDebugUnset:
      // do nothing
      break;
    }
  }
  if (sanitizers) {
    os << ", sanitizers:";
    sanitizers.print(os);
  }
  if (enableXRayInstrumentation)
    os << ", enableXRayInstrumentation";

  os << ", relocModel: " << stringifyRelocationModel(relocModel);

  os << ", linkDirs: [";
  llvm::interleaveComma(linkDirs, os);
  os << "]";
  os << ", debugInfoLang: " << debugInfoLanguage;
  os << " }";
}
