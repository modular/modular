//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_CONFIGURATION_H
#define SUPPORT_CONFIGURATION_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringMap.h"
#include <filesystem>

namespace llvm {
class SourceMgr;
}

namespace M {
/// This class provides an overrideable config map. The on-disk representation
/// is essentially an INI file, since it's easy to parse, but every
/// configuration can be overridden with an environment variable.
/// For example, given the file:
///
///  # file.cfg
///  [section] # section
///  key = value # property
///
/// `section.key` could be lazily overridden with an environment variable
/// `MODULAR_SECTION_KEY` (if it exists) when the user gets the value. Note that
/// the identifiers in the file are normalized to lowercase to avoid potential
/// conflicts with an environment variable. Note that if there is an override,
/// and the config is later dumped, the config will contain the overridden value
/// rather than the original value.
///
/// There is a canonical location for the modular config file, it is located at
///
///   Config::getConfigFilePath()
///
/// which is at
///
///   Config::getModularConfigFolderPath()
///
///
/// If there is no file at the canonical location, that does not constitute an
/// error, it simply means that any values in the config must be derived from
/// the environment.
class Config {
public:
  Config() = default;
  // No copying.
  Config(const Config &other) = delete;
  Config(Config &&other) = default;
  Config &operator=(Config &&other) = default;

  /// Open the default configuration, and parse it.
  static ErrorOr<Config> open();

  /// Provides a simple ini-style parser.
  ErrorOrSuccess parseFrom(StringRef buffer, llvm::SourceMgr *mgr = nullptr);

  /// Get a value with a possible override from the environment.
  StringRef getValue(StringRef key);

  /// Get a value, and if that's missing return the default value.
  StringRef getValueOr(StringRef key, StringRef defaultValue);

  /// Get a boolean value with possible override from the environment.  Default
  /// is returned if not set.  Error is returned if set, but to a value that
  /// cannot be interpreted as a boolean.
  bool getValueAsBool(StringRef key, bool defaultValue);

  /// Set a value - this will override anything that was already set for that
  /// key.
  void setValue(StringRef key, StringRef value);

  /// Get all the values contained in the config.
  const llvm::StringMap<std::string> &getAllValues() const { return kv; }

  /// Populate the env overrides for any configuration that has one. Does
  /// nothing if env overrides are disabled.
  void populateEnvOverrides();

  // Enable or disable the functionality that allows environment variables to
  // override the existing variables on read.
  void setEnvOverride(bool newVal);

  /// Get the path to the canonical modular home directory.
  ///
  /// If create is true, then this directory will be created if it does not
  /// exist. If it cannot be created, then an error will be returned. This is
  /// the default, and should be used by most callers. However, in some cases
  /// callers may choose to not create the directory by setting create to false.
  /// In this case, the caller should check for existence of the returned path,
  /// as this may represent where the directory *would* be created.
  ///
  /// On systems that follow the XDG Base Directory Specification, this will be
  /// the $XDG_DATA_HOME/modular folder (typically $HOME/.local/share/modular)
  ///
  /// On other systems except Windows, will typically be $HOME/.modular
  static ErrorOr<std::filesystem::path>
  getModularDataFolderPath(bool create = true);

  /// Get the path to the canonical modular config folder.
  ///
  /// The semantics for create are the same as getModularDataFolderPath.
  ///
  /// NOTE: This will be the same as the modular data folder on systems that
  /// don't follow the XDG Base Directory Specification.
  ///
  /// On systems that do follow the XDG Base Directory Specification, this will
  /// be the $XDG_CONFIG_HOME/modular folder (typically $HOME/.config/modular)
  ///
  /// On other systems except Windows, will typically be $HOME/.modular
  static ErrorOr<std::filesystem::path>
  getModularConfigFolderPath(bool create = true);

  /// Get the path to the canonical modular config file.
  /// Often $XDG_CONFIG_HOME/modular/modular.cfg or $HOME/.modular/modular.cfg
  static ErrorOr<std::filesystem::path> getConfigFilePath(bool create = false);

private:
  /// Nested sections are just delimited with a `.`. Access is done with dot
  /// notation. This is a map of property -> value, with each property prefixed
  /// by its section.
  llvm::StringMap<std::string> kv;
  bool allowEnvOverride = true;
};

/// Given a file name, find that file in one of the modular search paths. If the
/// file does not exist in those paths, returns std::nullopt. If the file does
/// exist, returns the full path to that file.
std::optional<std::filesystem::path> findModularFile(StringRef fileName);
} // namespace M

#endif // SUPPORT_CONFIGURATION_H
