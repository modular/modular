//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Stands in for cuda/libplugin_LIBFABRIC.so in NixlPluginDirTest: it binds a
// symbol that only the EFA libfabric fixture exports at FABRIC_1.8, so it loads
// against that copy and fails against the distro one -- exactly how the real
// plugin behaves.

extern "C" int fi_fixture_open_v18();

extern "C" __attribute__((visibility("default"))) int nixl_plugin_init() {
  return fi_fixture_open_v18();
}
