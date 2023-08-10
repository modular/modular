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

struct ModularVersion final {
  int major;
  int minor;
  int patch;
  const char *label;    // version label like "-rc.1"
  const char *revision; // Truncated Git SHA
  const char *buildType;
};

ModularVersion getModularVersion();
const char *getModularVersionString();

} // namespace M

#endif // CONFIG_VERSION_H
