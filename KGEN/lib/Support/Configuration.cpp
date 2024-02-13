//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Support/Configuration.h"

using namespace M;
using namespace M::KGEN;

#define _STRINGIFY(str) #str
#define _X_STRINGIFY(str) _STRINGIFY(str)
#define STRINGIFY_MOJO_CONFIG(path) _X_STRINGIFY(MOJO_CONFIG_SECTION) path

//===----------------------------------------------------------------------===//
// MojoConfig
//===----------------------------------------------------------------------===//

ErrorOr<MojoConfig> MojoConfig::open() {
  ErrorOr<Config> config = Config::open();
  if (config.isError())
    return config.takeError();
  return MojoConfig(std::move(*config));
}

//===----------------------------------------------------------------------===//
// Parser Configurations

void MojoConfig::getParserImportPaths(SmallVectorImpl<StringRef> &paths) {
  StringRef importPaths =
      config.getValue(STRINGIFY_MOJO_CONFIG(".import_path"));
  importPaths.split(paths, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
}

//===----------------------------------------------------------------------===//
// LLDB Configurations

StringRef MojoConfig::getLLDBPluginPath() {
  return config.getValue(STRINGIFY_MOJO_CONFIG(".lldb_plugin_path"));
}

StringRef MojoConfig::getLLDBPath() {
  return config.getValue(STRINGIFY_MOJO_CONFIG(".lldb_path"));
}

//===----------------------------------------------------------------------===//
// JIT Configurations

StringRef MojoConfig::getCompilerRTPath() {
  return config.getValue(STRINGIFY_MOJO_CONFIG(".compilerrt_path"));
}

StringRef MojoConfig::getStaticCompilerRTPath() {
  return config.getValue(STRINGIFY_MOJO_CONFIG(".compilerrt_static_path"));
}

StringRef MojoConfig::getOrcRTPath() {
  return config.getValue(STRINGIFY_MOJO_CONFIG(".orcrt_path"));
}

//===----------------------------------------------------------------------===//
// Python Configurations

StringRef MojoConfig::getPythonLib() {
  return config.getValue(STRINGIFY_MOJO_CONFIG(".python_lib"));
}

//===----------------------------------------------------------------------===//
// Driver Configurations

StringRef MojoConfig::getDriverPath() {
  return config.getValue(STRINGIFY_MOJO_CONFIG(".driver_path"));
}

StringRef MojoConfig::getMBlackPath() {
  return config.getValue(STRINGIFY_MOJO_CONFIG(".mblack_path"));
}

StringRef MojoConfig::getREPLEntryPoint() {
  return config.getValue(STRINGIFY_MOJO_CONFIG(".repl_entry_point"));
}

void MojoConfig::getSystemLibraryLinkArgs(SmallVectorImpl<StringRef> &libs) {
  StringRef systemLibsArg =
      config.getValue(STRINGIFY_MOJO_CONFIG(".system_libs"));
  systemLibsArg.split(libs, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
}
