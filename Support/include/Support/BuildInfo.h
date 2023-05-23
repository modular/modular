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

#include "Config/Version.h"
#include "Support/LLVMForwardDecls.h"

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
  LLCLMaxProfilingLevel,
  SIMDBitWidth,
  PreferredMemoryAlignment,
};

struct BuildInfo {
  std::string modularVersion;
  std::string gitRevision;
  std::string buildType;
  std::string kernelsBuildType;
  int llclMaxProfilingLevel;
  // This is the SIMD bit-width Modular was built for (aka
  // kPreferredSIMDBitWidth, controlled by compiler flags at compile-time), and
  // does not change if you move the binary between machines.  It does not take
  // into account any detected processor capabilities at run-time (it is not
  // host info).
  size_t simdBitWidth;
  size_t preferredMemoryAlignment;

  void print(llvm::raw_ostream &os) const;
  void print(llvm::json::OStream &json) const;
  void print(BuildProperty property, llvm::raw_ostream &os) const;
};

/// Get information about this build of Modular.
BuildInfo getBuildInfo();

} // namespace M

#endif // SUPPORT_BUILDINFO_H
