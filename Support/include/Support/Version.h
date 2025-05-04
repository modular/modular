//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_VERSION_H
#define SUPPORT_VERSION_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace M {
/// This class provides the canonical version printer/parser at Modular. It
/// implements the SemVer 2.0.0 spec.
class Version {
public:
  /// Default-construct an empty version (0.0.0)
  Version() = default;
  /// Construct a version with a major/minor/patch/label. This performs no
  /// parsing of any kind.
  Version(unsigned major, unsigned minor, unsigned patch, std::string l)
      : major(major), minor(minor), patch(patch), label(std::move(l)) {}

  /// Parse the version from a string. This follows standard semver parsing
  /// rules.
  static ErrorOr<Version> parse(llvm::StringRef str);

  /// Compare two Version objects according to SemVer 2.0.0 section 11. In this
  /// context, "less than" means "lower precedence".
  bool operator<(const Version &other) const;

  bool operator<=(const Version &other) const {
    return (*this < other) || (*this == other);
  }
  bool operator>(const Version &other) const { return !(*this <= other); }
  bool operator>=(const Version &other) const { return !(*this < other); }

  /// Check if two version objects are semantically equal.
  bool operator==(const Version &other) const {
    return major == other.major && minor == other.minor &&
           patch == other.patch && label == other.label;
  }

  /// Check if two version objects are not equal.
  bool operator!=(const Version &other) const { return !(*this == other); }

  /// Get the major version.
  unsigned getMajor() const { return major; }

  /// Get the minor version.
  unsigned getMinor() const { return minor; }

  /// Get the patch version.
  unsigned getPatch() const { return patch; }

  /// Get the label (any extra identifiers after the patch).
  StringRef getLabel() const { return label; }

  /// Print this version to a string.
  std::string toString() const;

private:
  unsigned major = 0;
  unsigned minor = 0;
  unsigned patch = 0;
  std::string label = "";
};

/// Print a Version object.
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Version &other);
} // namespace M

#endif // SUPPORT_VERSION_H
