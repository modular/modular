//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/CompilationOptions.h"
#include "Support/MDialect/MAttrs.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TargetParser/ARMTargetParser.h"
#include <string>

using namespace M;
using namespace KGEN;
using namespace M::Driver;
using namespace std::string_literals;

CompilationOptions::CompilationOptions(
    unsigned optimizationLevel, DebugInfoLevel debugLevel,
    std::optional<DebugAtLevel> debugAtLevel, Sanitizers sanitizers,
    std::string targetTriple, std::string targetCpu, std::string targetFeatures,
    std::string targetAccelerator, int elaborationErrorLimit,
    bool elaborationErrorIncludePrelude,
    ErrorVerboseLevel elaborationErrorVerbose, unsigned elaborationMaxDepth,
    DebugInfoLanguage debugInfoLanguage, std::string searchPaths,
    SmallVector<std::string> extraSearchPaths)
    : optimizationLevel(optimizationLevel), debugLevel(debugLevel),
      debugAtLevel(debugAtLevel), sanitizers(sanitizers),
      targetTriple(std::move(targetTriple)), targetCpu(std::move(targetCpu)),
      targetFeatures(std::move(targetFeatures)),
      targetAccelerator(std::move(targetAccelerator)),
      debugInfoLanguage(debugInfoLanguage), searchPaths(searchPaths),
      extraSearchPaths(extraSearchPaths),
      elaborationErrorLimit(elaborationErrorLimit),
      elaborationErrorIncludePrelude(elaborationErrorIncludePrelude),
      elaborationErrorVerbose(elaborationErrorVerbose),
      elaborationMaxDepth(elaborationMaxDepth) {

  if (this->targetCpu.empty())
    setDefaultCPU();
}

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
  definesWithDefaults.push_back(
      "__SANITIZE_ADDRESS="s +
      (sanitizers.has(Sanitizers::kAddress) ? "1" : "0"));
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

void CompilationOptions::setDefaultCPU() {
  llvm::Triple triple(targetTriple);
  if (isHexagonBackend(*this)) {
    // Set hexagon default CPU same as
    // https://github.com/llvm/llvm-project/blob/8d59cca1ab9cf4e39e43bf695e415de9ccd41115/clang/lib/Driver/ToolChains/Hexagon.cpp#L804
    targetCpu = "hexagonv68";
  } else if (isARMBackend(*this)) {
    targetCpu = llvm::ARM::getDefaultCPU(triple.getArchName());
  } else if (triple.getArch() !=
             llvm::Triple(llvm::sys::getDefaultTargetTriple()).getArch()) {
    // When cross-compiling, the host CPU is invalid for the target arch.
    // Clear it so LLVM selects the target's baseline CPU instead.
    targetCpu = "";
  } else {
    // Native target with no explicit CPU: use the host CPU.
    targetCpu = llvm::sys::getHostCPUName();
  }
}

bool M::KGEN::isGPUTriple(const llvm::Triple &triple) {
  // llvm::Triple defines isAMDGPU and isAMDGCN functions. The main difference
  // is that isAMDGPU checks for TeraScale muarch, which we don't support.
  return triple.isNVPTX() || triple.isAMDGCN() || isMetalTriple(triple);
}

bool M::KGEN::isHexagonTriple(const llvm::Triple &triple) {
  return triple.getArch() == llvm::Triple::hexagon;
}

bool M::KGEN::isMetalTriple(const llvm::Triple &triple) {
  // Metal GPU targets use ARM64 during compilation, then get converted to AIR
  // iOS/tvOS/watchOS don't have discrete GPUs suitable for compute kernels
  StringRef tripleStr = triple.str();

  return tripleStr.starts_with("air64-");
}

bool M::KGEN::isGPUBackend(const CompilationOptions &options) {
  llvm::Triple triple(options.targetTriple);
  return isGPUTriple(triple);
}

bool M::KGEN::isNVPTXBackend(const CompilationOptions &options) {
  llvm::Triple triple(options.targetTriple);
  return triple.isNVPTX();
}

bool M::KGEN::isAMDGPUBackend(const CompilationOptions &options) {
  llvm::Triple triple(options.targetTriple);
  return triple.isAMDGCN();
}

bool M::KGEN::isMetalBackend(const CompilationOptions &options) {
  // check compilation options for Metal backend
  return options.targetAccelerator == "metal" ||
         llvm::StringRef(options.targetTriple).starts_with("air64-");
}

bool M::KGEN::isHexagonBackend(const CompilationOptions &options) {
  llvm::Triple triple(options.targetTriple);
  return isHexagonTriple(triple);
}

bool M::KGEN::isARMBackend(const CompilationOptions &options) {
  llvm::Triple triple(options.targetTriple);
  return triple.isARM();
}
