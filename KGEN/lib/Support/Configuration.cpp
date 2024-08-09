//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Support/Configuration.h"
#include "Support/Settings/Settings.h"

using namespace M;
using namespace M::KGEN;

#define _STRINGIFY(str) #str
#define _X_STRINGIFY(str) _STRINGIFY(str)
#define STRINGIFY_MOJO_CONFIG(path) _X_STRINGIFY(MOJO_CONFIG_SECTION) path

#ifndef MOJO_CONFIG_SECTION
#error "Expected MOJO_CONFIG_SECTION to be set"
#endif

ErrorOr<std::filesystem::path> MojoConfig::getConfigFilePath() const {
  if (!std::holds_alternative<Config>(configSource))
    return Error("Configuration file path unavailable from settings");
  return std::get<Config>(configSource).getConfigFilePath();
}

static StringRef getValueFrom(Config &config, StringLiteral key) {
  return config.getValue(key);
}
static StringRef getValueFrom(Settings *settings, StringLiteral key) {
  return settings->get<StringRef>(key);
}

StringRef MojoConfig::getValue(StringLiteral key) {
  return std::visit([key](auto &source) { return getValueFrom(source, key); },
                    configSource);
}

//===----------------------------------------------------------------------===//
// MojoConfig
//===----------------------------------------------------------------------===//

ErrorOr<MojoConfig> MojoConfig::open() {
  ErrorOr<Config> config = Config::open();
  if (config.isError())
    return config.takeError();
  return MojoConfig(std::move(*config));
}

MojoConfig MojoConfig::fromContext(ContextRef ctx) {
  return MojoConfig(ctx->get<Settings>());
}

//===----------------------------------------------------------------------===//
// Parser Configurations

void MojoConfig::getParserImportPaths(SmallVectorImpl<StringRef> &paths) {
  StringRef importPaths = getValue(STRINGIFY_MOJO_CONFIG(".import_path"));
  importPaths.split(paths, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
}

//===----------------------------------------------------------------------===//
// Compiled Mojo File Configurations

StringRef MojoConfig::getMOGGLibPath() {
  return getValue(STRINGIFY_MOJO_CONFIG(".mogglib_path"));
}

StringRef MojoConfig::getMOGGLibDebuginfoPath() {
  return getValue(STRINGIFY_MOJO_CONFIG(".mogglib_debuginfo_path"));
}

StringRef MojoConfig::getMOGGPrimitivesLibPath() {
  return getValue(STRINGIFY_MOJO_CONFIG(".moggprimitiveslib_path"));
}

StringRef MojoConfig::getMOGGPrimitivesLibDebuginfoPath() {
  return getValue(STRINGIFY_MOJO_CONFIG(".moggprimitiveslib_debuginfo_path"));
}

//===----------------------------------------------------------------------===//
// LLDB Configurations

StringRef MojoConfig::getLLDBPluginPath() {
  return getValue(STRINGIFY_MOJO_CONFIG(".lldb_plugin_path"));
}

StringRef MojoConfig::getLLDBPath() {
  return getValue(STRINGIFY_MOJO_CONFIG(".lldb_path"));
}

//===----------------------------------------------------------------------===//
// JIT Configurations

StringRef MojoConfig::getCompilerRTPath() {
  return getValue(STRINGIFY_MOJO_CONFIG(".compilerrt_path"));
}

StringRef MojoConfig::getStaticCompilerRTPath() {
  return getValue(STRINGIFY_MOJO_CONFIG(".compilerrt_static_path"));
}

StringRef MojoConfig::getOrcRTPath() {
  return getValue(STRINGIFY_MOJO_CONFIG(".orcrt_path"));
}

//===----------------------------------------------------------------------===//
// Driver Configurations

StringRef MojoConfig::getDriverPath() {
  return getValue(STRINGIFY_MOJO_CONFIG(".driver_path"));
}

StringRef MojoConfig::getJupyterPath() {
  return getValue(STRINGIFY_MOJO_CONFIG(".jupyter_path"));
}

StringRef MojoConfig::getLSPServerPath() {
  return getValue(STRINGIFY_MOJO_CONFIG(".lsp_server_path"));
}

StringRef MojoConfig::getBuildServerPath() {
  return getValue(STRINGIFY_MOJO_CONFIG(".build_server_path"));
}

StringRef MojoConfig::getMBlackPath() {
  return getValue(STRINGIFY_MOJO_CONFIG(".mblack_path"));
}

StringRef MojoConfig::getREPLEntryPoint() {
  return getValue(STRINGIFY_MOJO_CONFIG(".repl_entry_point"));
}

StringRef MojoConfig::getTestExecutorPath() {
  return getValue(STRINGIFY_MOJO_CONFIG(".test_executor_path"));
}

StringRef MojoConfig::getLinkerDriver() {
  return getValue(STRINGIFY_MOJO_CONFIG(".linker_driver"));
}

void MojoConfig::getSystemLibraryLinkArgs(SmallVectorImpl<StringRef> &libs) {
  StringRef systemLibsArg = getValue(STRINGIFY_MOJO_CONFIG(".system_libs"));
  systemLibsArg.split(libs, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
}
