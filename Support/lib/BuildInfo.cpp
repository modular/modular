//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/BuildInfo.h"
#include "Config/Version.h"
#include "Support/AlignedAlloc.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

void BuildInfo::print(llvm::raw_ostream &os) const {
  os << "modular-version: " << modularVersion;
  os << "\ngit-revision: " << gitRevision;
  os << "\nbuild-type: " << buildType;
  os << "\nkernels-build-type: " << kernelsBuildType;
  os << "\nllcl-max-profiling-level: "
     << llvm::format("0%04o", llclMaxProfilingLevel);
  os << "\npreferred-mem-alignment: " << preferredMemoryAlignment;
  os << "\nllvm-targets: ";
  llvm::interleaveComma(llvmTargets, os);
  os << "\n";
}

void BuildInfo::print(llvm::json::OStream &json) const {
  json.objectBegin();
  json.attribute("modular-version", modularVersion);
  json.attribute("git-revision", gitRevision);
  json.attribute("build-type", buildType);
  json.attribute("kernels-build-type", kernelsBuildType);
  json.attribute("llcl-max-profiling-level", llclMaxProfilingLevel);
  json.attribute("preferred-mem-alignment", preferredMemoryAlignment);
  json.attribute("llvm-targets", llvm::json::Array(llvmTargets));
  json.objectEnd();
}

void BuildInfo::print(BuildProperty property, llvm::raw_ostream &os) const {
  switch (property) {
  case BuildProperty::ModularVersion:
    os << modularVersion;
    break;
  case BuildProperty::GitRevision:
    os << gitRevision;
    break;
  case BuildProperty::BuildType:
    os << buildType;
    break;
  case BuildProperty::KernelsBuildType:
    os << kernelsBuildType;
    break;
  case BuildProperty::LLCLMaxProfilingLevel:
    os << llclMaxProfilingLevel;
    break;
  case BuildProperty::PreferredMemoryAlignment:
    os << preferredMemoryAlignment;
    break;
  case BuildProperty::LLVMTargets:
    llvm::interleaveComma(llvmTargets, os);
    break;
  }
  os << "\n";
}

BuildInfo M::getBuildInfo() {
  BuildInfo buildInfo;

  ModularVersion modularVersion = getModularVersion();
  buildInfo.modularVersion = getModularVersionString();
  buildInfo.gitRevision = modularVersion.revision;
  buildInfo.buildType = modularVersion.buildType;
  buildInfo.kernelsBuildType = MODULAR_KERNELS_BUILD_TYPE;
  buildInfo.llclMaxProfilingLevel = MODULAR_ASYNCRT_MAX_PROFILING_LEVEL;
  buildInfo.preferredMemoryAlignment = kPreferredMemoryAlignment;

  StringRef(LLVM_TARGETS_BUILT).split(buildInfo.llvmTargets, " ");

  return buildInfo;
}
