//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Config/Version.h"
#include "GeneratedVersion.h"

using namespace M;

ModularVersion M::getModularVersion() {
  ModularVersion version{};
  version.major = MODULAR_VERSION_MAJOR;
  version.minor = MODULAR_VERSION_MINOR;
  version.patch = MODULAR_VERSION_PATCH;
  version.revision = MODULAR_VERSION_REVISION;
  version.buildType = MODULAR_BUILD_TYPE_LOWER;
  return version;
}

const char *M::getModularVersionString() { return MODULAR_VERSION_STRING; }
