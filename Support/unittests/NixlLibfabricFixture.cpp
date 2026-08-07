//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Stands in for libfabric in NixlPluginDirTest. Built twice against different
// version scripts to mirror the two copies that coexist in a serving image: the
// EFA build we stage next to the plugins (exports FABRIC_1.8) and the older
// distro build that other components can drag in (stops at FABRIC_1.0). Both
// carry the same SONAME, which is what makes them collide.
//
// The toolchain compiles with hidden visibility, which a version script cannot
// override, so the exported entry points say so explicitly.

#define FIXTURE_EXPORT extern "C" __attribute__((visibility("default")))

FIXTURE_EXPORT int fi_fixture_open() { return 0; }

#if FIXTURE_HAS_FABRIC_1_8
FIXTURE_EXPORT int fi_fixture_open_v18() { return 0; }
#endif
