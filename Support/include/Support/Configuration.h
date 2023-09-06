//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_CONFIGURATION_H
#define SUPPORT_CONFIGURATION_H

#include "Support/ADT/SmartVariant.h"
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
///   Config::getModularHomeDirPath() / ".modular"
///
/// which is found through the MODULAR_HOME environment variable, having
/// `.modular` in the user's PATH, the MODULAR_DERIVED_PATH environment variable
/// for in-tree builds, or if all else fails, just in the current working
/// directory.
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

  /// Copy all the values from another config object into current object.
  /// Error if any of the keys from incoming config already exist.
  ErrorOrSuccess copyFrom(const Config &other);

  /// Get a value with a possible override from the environment.
  StringRef getValue(StringRef key);

  /// Get a value, and if that's missing return the default value.
  StringRef getValueOr(StringRef key, StringRef defaultValue);

  /// Get a boolean value with possible override from the environment.  Default
  /// is returned if not set.  Error is returned if set, but to a value that
  /// cannot be interpreted as a boolean.
  ErrorOr<bool> getValueAsBool(StringRef key, bool defaultValue);

  /// Set a value - this will override anything that was already set for that
  /// key.
  void setValue(StringRef key, StringRef value);

  /// Given a section name, get a list of all the values in that section. Global
  /// properties (properties without a section) can be listed by simply using an
  /// empty string for the section.
  void
  getValuesInSection(StringRef section,
                     SmallVectorImpl<std::pair<StringRef, StringRef>> &values);

  /// Get all the values contained in the config.
  const llvm::StringMap<std::string> &getAllValues() const { return kv; }

  /// Flush the configs to the provided stream.
  // TODO: Preserve user comments.
  void flush(llvm::raw_ostream &os);
  /// Flush the configuration to the canonical location.
  ErrorOrSuccess flush();

  /// Get the path to the canonical modular home directory.
  static std::filesystem::path getModularHomeDirPath();

  /// Get the path to the canonical modular config file.
  static std::filesystem::path getConfigFilePath();

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
