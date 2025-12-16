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
#include <optional>
#include <string>

namespace llvm {
class SourceMgr;
}

namespace M {
/// This class provides an overridable config map. The on-disk representation
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
  /// Returns an empty string if not found.
  StringRef getValue(StringRef key);

  /// Get a value with a possible override from the environment.
  /// Returns nothing if not found.
  std::optional<StringRef> maybeGetValue(StringRef key);

  /// Get a path with a possible override from the environment.
  /// If not found, returns getValue("package_root") / relativePath, using the
  /// same config section as `key`.
  StringRef getPath(StringRef key, StringRef relativePath);

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

  /// Get the path to the canonical modular home directory.
  ///
  /// If create is true, then this directory will be created if it does not
  /// exist. If it cannot be created, then an error will be returned. This is
  /// the default, and should be used by most callers. However, in some cases
  /// callers may choose to not create the directory by setting create to false.
  /// In this case, the caller should check for existence of the returned path,
  /// as this may represent where the directory *would* be created.
  ///
  /// The precedence of how configuration options affect the data folder path:
  ///
  /// 1. When MODULAR_HOME is set: $MODULAR_HOME
  /// 2. When MODULAR_DERIVED_PATH is set: $MODULAR_DERIVED_PATH
  /// 3. When TEST_TMPDIR is set: $TEST_TMPDIR
  /// 4. If $HOME/.modular directory exists: $HOME/.modular
  /// 5. Otherwise, follow the XDG Base Directory Specification on systems that
  ///    support it: $XDG_DATA_HOME or its default $HOME/.local/share/modular
  static ErrorOr<std::filesystem::path>
  getModularDataFolderPath(bool create = true);

  /// Get the path to the canonical modular config folder.
  ///
  /// The semantics for create are the same as getModularDataFolderPath.
  ///
  /// NOTE: This will be the same as the modular data folder on systems that
  /// don't follow the XDG Base Directory Specification.
  ///
  /// The precedence of how configuration options affect the data folder path:
  ///
  /// 1. When MODULAR_HOME is set: $MODULAR_HOME
  /// 2. When MODULAR_DERIVED_PATH is set: $MODULAR_DERIVED_PATH
  /// 3. When TEST_TMPDIR is set: $TEST_TMPDIR
  /// 4. If $HOME/.modular directory exists: $HOME/.modular
  /// 5. Otherwise, follow the XDG Base Directory Specification on systems that
  ///    support it: $XDG_CONFIG_HOME or its default $HOME/.config/modular
  static ErrorOr<std::filesystem::path>
  getModularConfigFolderPath(bool create = true);

  /// Get the path to the canonical modular cache folder.
  ///
  /// The semantics for create are the same as getModularDataFolderPath.
  ///
  /// The precedence of how configuration options affect the data folder path:
  ///
  /// 1. When the global `cache_dir` config file option or MODULAR_CACHE_DIR is
  ///    set, use that
  /// 2. When MODULAR_HOME is set: $MODULAR_HOME/cache
  /// 3. When MODULAR_DERIVED_PATH is set: $MODULAR_DERIVED_PATH/cache
  /// 4. When TEST_TMPDIR is set: $TEST_TMPDIR
  /// 5. If $HOME/.modular directory exists: $HOME/.modular
  /// 6. Otherwise, follow the XDG Base Directory Specification on systems that
  ///    support it: $XDG_CACHE_HOME or its default $HOME/.cache/modular
  ErrorOr<std::filesystem::path> getModularCacheFolderPath(bool create = true);

  /// Get the path to the canonical modular config file.
  /// Often $XDG_CONFIG_HOME/modular/modular.cfg or $HOME/.modular/modular.cfg
  static ErrorOr<std::filesystem::path> getConfigFilePath(bool create = false);

private:
  /// Nested sections are just delimited with a `.`. Access is done with dot
  /// notation. This is a map of property -> value, with each property prefixed
  /// by its section.
  llvm::StringMap<std::string> kv;
};

/// Given a file name, find that file in one of the modular search paths. If the
/// file does not exist in those paths, returns std::nullopt. If the file does
/// exist, returns the full path to that file.
std::optional<std::filesystem::path> findModularFile(StringRef fileName);
} // namespace M

#endif // SUPPORT_CONFIGURATION_H
