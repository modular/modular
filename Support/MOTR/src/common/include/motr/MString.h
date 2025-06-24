//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOTR_COMMON_MSTRING_H
#define MOTR_COMMON_MSTRING_H

#include "motr/Hash.h"
#include "motr/StringLibrary.h"
#include <string>

namespace M::motr {

struct MString {
  // returns the global StringLibrary singleton
  static StringLibrary &getStringLibrary();

  Hash::Value hash;

  // creates an invalid MString
  MString();

  // asserts if hash is not found in the StringLibrary
  MString(Hash::Value hash, bool failIfNotFound = true);

  // inserts the string into the StringLibrary if it is not found
  MString(std::string_view str);

  bool valid() const;

  constexpr bool operator<(const MString &other) const {
    return hash < other.hash;
  }
  constexpr bool operator==(const MString &other) const {
    return hash == other.hash;
  }
  constexpr bool operator!=(const MString &other) const {
    return !(hash == other.hash);
  }

  bool operator==(const std::string_view &other) const { return sv() == other; }
  bool operator!=(const std::string_view &other) const { return sv() != other; }

  // returns the string_view of the string in the StringLibrary
  // if the string is not found, it will return an empty string_view
  std::string_view sv(bool createPlaceholder = false) const;

  // returns a copy of the string in the StringLibrary
  // if the string is not found, it will return an empty string
  std::string str(bool createPlaceholder = false) const;
};

// MStringSafe is a convenience class
// wrapper of MString that does not assert if the string is not found
// and always returns a placeholder string of the hash if not found
// equivalents:
// MString{hash, false} => MStringSafe{hash}
// MString{hash, false}.sv(true) => MStringSafe{hash}.sv()
struct MStringSafe {
  MString mstr;
  MStringSafe(Hash::Value hash) : mstr(hash, false) {}
  std::string_view sv() const { return mstr.sv(true); }
  std::string str() const { return mstr.str(true); }
  operator std::string_view() const { return sv(); }
  operator std::string() const { return str(); }
};

} // namespace M::motr

namespace std {
template <>
struct hash<M::motr::MString> {
  size_t operator()(const M::motr::MString &h) const { return h.hash.v; }
};
} // namespace std

#endif // MOTR_COMMON_MSTRING_H
