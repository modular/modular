//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Config/Version.h"
#include "GeneratedVersion.h"

using namespace M;

ProjectVersion M::getMAXVersion() {
  return ProjectVersion{
      .major = MAX_VERSION_MAJOR,
      .minor = MAX_VERSION_MINOR,
      .patch = MAX_VERSION_PATCH,
      .label = MAX_VERSION_LABEL,
      .revision = MODULAR_VERSION_REVISION,
      .buildType = MODULAR_BUILD_TYPE_LOWER,
  };
}

const char *M::getMAXVersionString() { return MAX_VERSION_STRING; }

ProjectVersion M::getMojoVersion() {
  return ProjectVersion{
      .major = MOJO_VERSION_MAJOR,
      .minor = MOJO_VERSION_MINOR,
      .patch = MOJO_VERSION_PATCH,
      .label = MOJO_VERSION_LABEL,
      .revision = MODULAR_VERSION_REVISION,
      .buildType = MODULAR_BUILD_TYPE_LOWER,
  };
}

const char *M::getMojoVersionString() { return MOJO_VERSION_STRING; }

// TODO: Remove

M::ModularVersion M::getModularVersion() { return M::getMAXVersion(); }

const char *M::getModularVersionString() { return MODULAR_VERSION_STRING; }
