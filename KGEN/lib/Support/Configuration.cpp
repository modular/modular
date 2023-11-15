//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Support/Configuration.h"

using namespace M;
using namespace M::KGEN;

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
  StringRef importPaths = config.getValue("mojo.import_path");
  importPaths.split(paths, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
}

//===----------------------------------------------------------------------===//
// LLDB Configurations

StringRef MojoConfig::getLLDBPluginPath() {
  return config.getValue("mojo.lldb_plugin_path");
}

StringRef MojoConfig::getLLDBPath() {
  return config.getValue("mojo.lldb_path");
}

//===----------------------------------------------------------------------===//
// JIT Configurations

StringRef MojoConfig::getCompilerRTPath() {
  return config.getValue("mojo.compilerrt_path");
}

StringRef MojoConfig::getStaticCompilerRTPath() {
  return config.getValue("mojo.compilerrt_static_path");
}

StringRef MojoConfig::getOrcRTPath() {
  return config.getValue("mojo.orcrt_path");
}

//===----------------------------------------------------------------------===//
// Python Configurations

StringRef MojoConfig::getPythonLib() {
  return config.getValue("mojo.python_lib");
}

//===----------------------------------------------------------------------===//
// Driver Configurations

StringRef MojoConfig::getMBlackPath() {
  return config.getValue("mojo.mblack_path");
}

StringRef MojoConfig::getREPLEntryPoint() {
  return config.getValue("mojo.repl_entry_point");
}

void MojoConfig::getSystemLibraryLinkArgs(SmallVectorImpl<StringRef> &libs) {
  StringRef systemLibsArg = config.getValue("mojo.system_libs");
  systemLibsArg.split(libs, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
}
