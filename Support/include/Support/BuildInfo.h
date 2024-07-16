//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Utilities for interrogating build-time settings.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_BUILDINFO_H
#define SUPPORT_BUILDINFO_H

#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace llvm::json {
class OStream;
}

namespace M {

enum class BuildProperty {
  ModularVersion,
  GitRevision,
  BuildType,
  KernelsBuildType,
  AsyncRTMaxProfilingLevel,
  PreferredMemoryAlignment,
  LLVMTargets,
};

struct BuildInfo {
  std::string modularVersion;
  std::string gitRevision;
  std::string buildType;
  std::string kernelsBuildType;
  int asyncrtMaxProfilingLevel;
  size_t preferredMemoryAlignment;
  SmallVector<StringRef> llvmTargets;

  void print(llvm::raw_ostream &os) const;
  void print(llvm::json::OStream &json) const;
  void print(BuildProperty property, llvm::raw_ostream &os) const;
};

/// Get information about this build of Modular.
BuildInfo getBuildInfo();

} // namespace M

#endif // SUPPORT_BUILDINFO_H
