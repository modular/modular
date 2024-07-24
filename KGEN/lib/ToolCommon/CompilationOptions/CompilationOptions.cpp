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
    DebugInfoLanguage debugInfoLanguage, std::string searchPaths)
    : enableSearch(enableSearch), optimizationLevel(optimizationLevel),
      debugLevel(debugLevel), debugAtLevel(debugAtLevel),
      sanitizers(sanitizers),
      enableXRayInstrumentation(enableXRayInstrumentation),
      targetTriple(std::move(targetTriple)), targetCpu(std::move(targetCpu)),
      targetFeatures(std::move(targetFeatures)),
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

ErrorOr<EnvAttr>
CompilationOptions::parseDefinesWithDefaults(MLIRContext *ctx,
                                             ArrayRef<std::string> defines) {
  // Add defaults from compilation options.  Add them as strings before parsing
  // so that if a user defines them as well, they get an error for defining it a
  // second time.
  SmallVector<std::string> definesWithDefaults;
  switch (debugLevel) {
  case kFullDebugInfo:
    definesWithDefaults.push_back("DEBUG_LEVEL=full");
    break;
  case kLineTablesOnly:
    definesWithDefaults.push_back("DEBUG_LEVEL=line-tables");
    break;
  default:
    break;
  }
  definesWithDefaults.push_back("OPTIMIZATION_LEVEL=" +
                                Twine(optimizationLevel).str());
  for (std::string define : defines)
    definesWithDefaults.push_back(define);
  return EnvAttr::parseDefines(ctx, definesWithDefaults);
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

  os << ", debugInfoLang: " << debugInfoLanguage;
  os << " }";
}
