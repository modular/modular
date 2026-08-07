//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/NixlPluginDir.h"

#include "gtest/gtest.h"

#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace {

// Fixtures staged as test data by the BUILD rule, reachable from the runfiles
// root. Both libfabric builds carry SONAME libfabric.so.1; the plugin binds a
// symbol only the EFA one exports, at FABRIC_1.8.
constexpr const char *kEfaLibfabric =
    "Support/unittests/nixl_libfabric_efa_fixture.so";
constexpr const char *kDistroLibfabric =
    "Support/unittests/nixl_libfabric_distro_fixture.so";
constexpr const char *kPlugin = "Support/unittests/libplugin_FIXTURE.so";

// Reproduces the staged package layout: transport plugins in
// <prefix>/lib/nixl/<flavor>, libfabric flat in <prefix>/lib. The distro copy
// sits outside the prefix, standing in for /usr/lib/x86_64-linux-gnu.
class StagedLayout {
public:
  explicit StagedLayout(const std::string &name, bool stageLibfabric = true)
      : prefix_(std::filesystem::temp_directory_path() /
                ("nixl-plugin-dir-" + name)) {
    std::filesystem::remove_all(prefix_);
    std::filesystem::create_directories(pluginDir());
    std::filesystem::create_directories(prefix_ / "distro");
    std::filesystem::copy_file(kPlugin, plugin());
    std::filesystem::copy_file(kDistroLibfabric, distroLibfabric());
    if (stageLibfabric)
      std::filesystem::copy_file(kEfaLibfabric,
                                 prefix_ / "lib" / "libfabric.so.1");
  }

  ~StagedLayout() {
    std::error_code ec;
    std::filesystem::remove_all(prefix_, ec);
  }

  std::filesystem::path pluginDir() const {
    return prefix_ / "lib" / "nixl" / "cuda";
  }
  std::filesystem::path plugin() const {
    return pluginDir() / "libplugin_FIXTURE.so";
  }
  std::filesystem::path distroLibfabric() const {
    return prefix_ / "distro" / "libfabric.so.1";
  }
  std::filesystem::path errorFile() const { return prefix_ / "dlerror.txt"; }

private:
  std::filesystem::path prefix_;
};

// Whether the plugin loaded, plus the loader's complaint when it did not.
struct LoadResult {
  bool loaded = false;
  std::string error;
};

// Loads the plugin the way NIXL does, with the foreign libfabric pulled into
// the process the way another component would, optionally claiming the staged
// copy first. Runs in a forked child because dlopen state is process-wide, and
// which copy claimed the SONAME first is precisely what is under test.
LoadResult loadPluginAlongsideForeignLibfabric(const StagedLayout &layout,
                                               bool preloadStaged) {
  const pid_t pid = fork();
  if (pid == 0) {
    if (preloadStaged)
      M::preloadStagedLibfabric(layout.pluginDir());
    ::dlopen(layout.distroLibfabric().c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!::dlopen(layout.plugin().c_str(), RTLD_NOW | RTLD_LOCAL)) {
      std::ofstream(layout.errorFile()) << ::dlerror();
      ::_exit(1);
    }
    ::_exit(0);
  }
  int status = 0;
  ::waitpid(pid, &status, 0);

  LoadResult result;
  result.loaded = WIFEXITED(status) && WEXITSTATUS(status) == 0;
  if (std::ifstream err{layout.errorFile()})
    std::getline(err, result.error);
  return result;
}

void requestBackend(const char *backend) {
  setenv("MODULAR_NIXL_TRANSFER_BACKEND", backend, /*overwrite=*/1);
}

// The trap this all exists for: an older libfabric that got into the process
// first owns the SONAME, and the loader never reconsiders the plugin's rpath,
// so the plugin becomes permanently unloadable. Asserts the loader's behavior
// rather than ours -- if this ever stops failing to load, the hazard is gone
// and preloadStagedLibfabric can go with it.
TEST(PreloadStagedLibfabric, AForeignLibfabricLoadedFirstBreaksThePlugin) {
  StagedLayout layout("hijacked");

  const LoadResult result =
      loadPluginAlongsideForeignLibfabric(layout, /*preloadStaged=*/false);

  EXPECT_FALSE(result.loaded);
  EXPECT_NE(result.error.find("FABRIC_1.8"), std::string::npos)
      << "expected a version-mismatch error, got: " << result.error;
}

// ... which preloading the staged copy prevents, because ours gets the SONAME
// and is a strict superset of the older copy's symbol versions.
TEST(PreloadStagedLibfabric, PreloadingTheStagedCopyKeepsThePluginLoadable) {
  StagedLayout layout("preloaded");
  requestBackend("libfabric");

  const LoadResult result =
      loadPluginAlongsideForeignLibfabric(layout, /*preloadStaged=*/true);

  EXPECT_TRUE(result.loaded) << "plugin failed to load: " << result.error;
}

TEST(PreloadStagedLibfabric, AcceptsAnyCasingOfTheBackendRequest) {
  StagedLayout layout("casing");
  requestBackend("LibFabric");
  EXPECT_TRUE(M::preloadStagedLibfabric(layout.pluginDir()));
}

// Only the libfabric plugin binds the versioned symbols that make the SONAME
// race matter, and the EFA build pulls in the CUDA driver stack, so no other
// backend should pay for it.
TEST(PreloadStagedLibfabric, SkipsBackendsThatDoNotNeedIt) {
  StagedLayout layout("ucx");
  requestBackend("ucx");
  EXPECT_FALSE(M::preloadStagedLibfabric(layout.pluginDir()));
}

// Layouts without a staged libfabric (bazel runfiles, ROCm-only packages) are
// ordinary, not errors.
TEST(PreloadStagedLibfabric, ToleratesAnUnstagedLibfabric) {
  StagedLayout layout("missing", /*stageLibfabric=*/false);
  requestBackend("libfabric");
  EXPECT_FALSE(M::preloadStagedLibfabric(layout.pluginDir()));
}

} // namespace
