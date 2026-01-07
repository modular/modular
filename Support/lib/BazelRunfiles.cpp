//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/BazelRunfiles.h"
#include "rules_cc/cc/runfiles/runfiles.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

using namespace M;
using llvm::StringRef;
using rules_cc::cc::runfiles::Runfiles;

struct RunfileMapping {
  llvm::StringLiteral
      configKey; // The config key (e.g., "mojo-max.driver_path")
  llvm::StringLiteral workspace; // Empty for _main, otherwise external
                                 // workspace (e.g., "llvm-project")
  llvm::StringLiteral path;    // Path within workspace (without lib prefix/ext)
  bool isSharedLibrary;        // If true, add lib prefix and platform extension
  llvm::StringLiteral libName; // Shared library name (without lib prefix/ext).
};

#ifdef __APPLE__
static constexpr llvm::StringLiteral kSharedLibExt = ".dylib";
#else
static constexpr llvm::StringLiteral kSharedLibExt = ".so";
#endif

static constexpr RunfileMapping kRunfileMappings[] = {
    {"crash_reporting.handler_path", "crashpad", "modular-crashpad-handler",
     false, ""},
    {"mojo-max.driver_path", "", "KGEN/tools/mojo/mojo", false, ""},
    {"mojo-max.lld_path", "llvm-project", "lld/lld", false, ""},
    {"mojo-max.lldb_path", "llvm-project", "lldb/lldb", false, ""},
    {"mojo-max.lsp_server_path", "",
     "KGEN/tools/mojo-lsp-server/mojo-lsp-server", false, ""},
    {"mojo-max.repl_entry_point", "",
     "KGEN/tools/mojo-repl-entry-point/mojo-repl-entry-point", false, ""},

    // Shared libraries
    {"mojo-max.mgprt_path", "", "GenericML", true, "MGPRT"},
    {"mojo-max.compilerrt_path", "", "KGEN", true, "KGENCompilerRTShared"},
    {"mojo-max.lldb_plugin_path", "", "KGEN", true, "MojoLLDB"},

    // Directory paths
    {"nixl_plugin_dir", "", "MLRT/Driver", false, ""},
};

/// Returns nullptr if runfiles cannot be initialized (not running under Bazel)
static Runfiles *getRunfiles() {
  static std::unique_ptr<Runfiles> runfiles =
      []() -> std::unique_ptr<Runfiles> {
    auto rf = std::unique_ptr<Runfiles>(
        Runfiles::CreateForTest(BAZEL_CURRENT_REPOSITORY, nullptr));
    if (rf)
      return rf;

    std::string execPath =
        llvm::sys::fs::getMainExecutable(nullptr, (void *)&getRunfiles);
    return std::unique_ptr<Runfiles>(
        Runfiles::Create(execPath, BAZEL_CURRENT_REPOSITORY, nullptr));
  }();

  return runfiles.get();
}

static std::string buildRunfilePath(const RunfileMapping &mapping) {
  std::string result;

  if (mapping.workspace.empty()) {
    result = "_main/";
  } else {
    result = mapping.workspace.str() + "/";
  }

  result += mapping.path.str();

  if (mapping.isSharedLibrary) {
    result += "/lib";
    result += mapping.libName.str();
    result += kSharedLibExt.str();
  }

  return result;
}

std::optional<std::string> M::findConfigWithRunfiles(StringRef key) {
  std::string lowerKey = key.lower();
  const RunfileMapping *mapping = nullptr;
  for (const auto &m : kRunfileMappings) {
    if (m.configKey == lowerKey) {
      mapping = &m;
      break;
    }
  }

  if (!mapping)
    return std::nullopt;

  Runfiles *rf = getRunfiles();
  if (!rf)
    return std::nullopt;

  std::string runfilePath = buildRunfilePath(*mapping);
  std::string rlocation = rf->Rlocation(runfilePath);
  if (rlocation.empty())
    llvm::report_fatal_error(
        llvm::Twine("[BazelRunfiles] Runfile lookup failed for key '") + key +
        "': Rlocation returned empty for path '" + runfilePath +
        "'. This indicates a build configuration error - the runfile mapping "
        "exists but the file is not in the runfiles manifest.");

  // If the file isn't part of the runfiles, return nothing so looks
  // fallthrough. It might still fail later.
  std::error_code ec;
  if (!std::filesystem::exists(rlocation, ec))
    return std::nullopt;

  return rlocation;
}
