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
using namespace AsyncRT;

CompilationOptions::CompilationOptions(
    unsigned optimizationLevel, DebugInfoLevel debugLevel,
    std::optional<DebugAtLevel> debugAtLevel, Sanitizers sanitizers,
    std::string targetTriple, std::string targetCpu, std::string targetFeatures,
    std::string targetAccelerator, DebugInfoLanguage debugInfoLanguage,
    std::string searchPaths)
    : optimizationLevel(optimizationLevel), debugLevel(debugLevel),
      debugAtLevel(debugAtLevel), sanitizers(sanitizers),
      targetTriple(std::move(targetTriple)), targetCpu(std::move(targetCpu)),
      targetFeatures(std::move(targetFeatures)),
      targetAccelerator(std::move(targetAccelerator)),
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
    definesWithDefaults.push_back("__DEBUG_LEVEL=full");
    break;
  case kLineTablesOnly:
    definesWithDefaults.push_back("__DEBUG_LEVEL=line-tables");
    break;
  default:
    break;
  }
  definesWithDefaults.push_back("__OPTIMIZATION_LEVEL=" +
                                Twine(optimizationLevel).str());
  for (std::string define : defines)
    definesWithDefaults.push_back(define);
  return EnvAttr::parseDefines(ctx, definesWithDefaults);
}

StringRef CompilationOptions::getDebugLevelString() {
  switch (debugLevel) {
  case kFullDebugInfo:
    return "full";
  case kLineTablesOnly:
    return "line-tables";
  default:
    return "";
  }
}

void CompilationOptions::print(raw_ostream &os) const {
  os << "CompilationOptions { optimizationLevel: " << optimizationLevel;
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

  os << ", relocModel: " << stringifyRelocationModel(relocModel);

  if (!targetAccelerator.empty())
    os << ", targetAccelerator: " << targetAccelerator;

  os << ", debugInfoLang: " << debugInfoLanguage;

  if (numThreads != 0)
    os << ", numThreads: " << numThreads;

  os << " }";
}

bool M::KGEN::isGPUTriple(const llvm::Triple &triple) {
  return triple.isNVPTX() || triple.isAMDGPU();
}

bool M::KGEN::isGPUBackend(const CompilationOptions &options) {
  llvm::Triple triple(options.targetTriple);
  return isGPUTriple(triple);
}

bool M::KGEN::isNVPTXBackend(const CompilationOptions &options) {
  return llvm::Triple(options.targetTriple).isNVPTX();
}

bool M::KGEN::isAMDBackend(const CompilationOptions &options) {
  return llvm::Triple(options.targetTriple).isAMDGCN();
}
