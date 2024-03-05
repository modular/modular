//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_SUPPORT_CONFIGURATION_H
#define KGEN_SUPPORT_CONFIGURATION_H

#include "Support/Configuration.h"
#include "Support/ErrorOr.h"

#include <filesystem>

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// MojoConfig
//===----------------------------------------------------------------------===//

/// This class provides easy and type-safe access to values in the Mojo section
/// of the modular configuration file.
class MojoConfig {
public:
  /// Returns the path to the modular.cfg file.
  ErrorOr<std::filesystem::path> getConfigFilePath() const;

  /// Open the default configuration, and parse it.
  static ErrorOr<MojoConfig> open();

  //===--------------------------------------------------------------------===//
  // Parser Configurations
  //===--------------------------------------------------------------------===//

  /// Return the default Mojo parser import paths.
  void getParserImportPaths(SmallVectorImpl<StringRef> &paths);

  //===--------------------------------------------------------------------===//
  // LLDB Configurations
  //===--------------------------------------------------------------------===//

  /// Return the path to the lldb-vscode executable within the mojo install.
  StringRef getLLDBVSCodePath();

  /// Return the path to the Mojo lldb plugin within the mojo install.
  StringRef getLLDBPluginPath();

  /// Return the path to the lldb executable within the mojo install.
  StringRef getLLDBPath();

  /// Return the default lldb visualizers to use when starting an LLDB debug
  /// session.
  void getLLDBVisualizers(SmallVectorImpl<StringRef> &paths);

  //===--------------------------------------------------------------------===//
  // JIT Configurations
  //===--------------------------------------------------------------------===//

  /// Return the path to the kgen-compiler-rt library.
  StringRef getCompilerRTPath();

  /// Return the path to the static kgen-compiler-rt library.
  StringRef getStaticCompilerRTPath();

  /// Return the path to the orc-rt library.
  StringRef getOrcRTPath();

  //===--------------------------------------------------------------------===//
  // Python Configurations
  //===--------------------------------------------------------------------===//

  /// Return the path to the default python shared library to load for Mojo
  /// python interop.
  StringRef getPythonLib();

  //===--------------------------------------------------------------------===//
  // Driver Configurations
  //===--------------------------------------------------------------------===//

  /// Return the path to the `mojo` driver in the mojo install.
  StringRef getDriverPath();

  /// Return the path to the Mojo jupyter library.
  StringRef getJupyterPath();

  /// Return the path to the Mojo LSP server.
  StringRef getLSPServerPath();

  /// Return the path to the mblack executable in the mojo install.
  StringRef getMBlackPath();

  /// Return the path to the REPL entry point executable in the mojo install.
  StringRef getREPLEntryPoint();

  /// Return the system libraries to link with Mojo when building a standalone
  /// binary.
  void getSystemLibraryLinkArgs(SmallVectorImpl<StringRef> &libs);

private:
  MojoConfig(Config config) : config(std::move(config)) {}

  Config config;
};
} // namespace M::KGEN

#endif // KGEN_SUPPORT_CONFIGURATION_H
