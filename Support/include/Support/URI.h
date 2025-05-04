//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_URI_H
#define SUPPORT_URI_H

#include "Support/ErrorOr.h"
#include <filesystem>

namespace M {

/// This class is used to parse and store URIs. Currently it will parse the
/// scheme, authority and path components from a URI (where everything after
/// authority is considered part of the path). These components can be queried
/// independently after parsing.
///
/// If parsing a non-URI, it will assume that it is a local filesystem path
/// (see `parse` method for details).
class URI {
public:
  /// Creates a URI with form "file://path" (note that it does not modify the
  /// path).
  URI(const std::filesystem::path path) : scheme("file"), path(path.string()) {}

  /// Parse a URI string. If the string is not a valid URI, it will assume that
  /// it is a local filesystem path: it will set the scheme to "file", empty
  /// authority (which is equivalent to localhost), and will set path to be the
  /// input string without modification.
  static ErrorOr<URI> parse(StringRef uri);

  /// Get URI scheme component.
  StringRef getScheme() const { return scheme; }

  /// Get URI authority component.
  StringRef getAuthority() const { return authority; }

  /// Get URI path component.
  StringRef getPath() const { return path; }

private:
  URI() = default;

  std::string scheme;
  std::string authority;
  std::string path;
};

} // namespace M

#endif // SUPPORT_URI_H
