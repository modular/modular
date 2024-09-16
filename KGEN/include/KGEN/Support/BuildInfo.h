//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_SUPPORT_BUILDINFO_H
#define KGEN_SUPPORT_BUILDINFO_H

#include "Config/Version.h"

#include <string>

namespace M::KGEN {
std::string getBuildID();
} // namespace M::KGEN

#define KGEN_VERSION_STRING                                                    \
  M::getModularVersionString() + getBuildID() + "-" +                          \
      M::getModularVersion().buildType

#endif // KGEN_SUPPORT_BUILDINFO_H
