//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Information about the Modular build version.
//
//===----------------------------------------------------------------------===//

#ifndef CONFIG_VERSION_H
#define CONFIG_VERSION_H

namespace M {

struct ProjectVersion final {
  int major;
  int minor;
  int patch;
  const char *label;    // version label like "-rc.1"
  const char *revision; // Truncated Git SHA
  const char *buildType;
};

ProjectVersion getMAXVersion();
ProjectVersion getMojoVersion();
const char *getMAXVersionString();
const char *getMojoVersionString();

// TODO: Remove
using ModularVersion = ProjectVersion;
ModularVersion getModularVersion();
const char *getModularVersionString();

} // namespace M

#endif // CONFIG_VERSION_H
