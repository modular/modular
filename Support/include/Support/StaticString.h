//===- Support/StaticString.h -----------------------------------*- C++ -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_STATIC_STRING_H
#define SUPPORT_STATIC_STRING_H

#include "Support/LLVM.h"

namespace M {

/// This is a "const char *" equivalent struct that forces construction from a
/// static C string.  This can be used by APIs that don't want to manage
/// lifetimes of strings.
class StaticString final {
public:
  /// Implicitly construct a StaticString with a static string (char array).
  template <size_t n>
  /*implicit*/ constexpr StaticString(const char (&value)[n])
      : value(const_cast<char *>(value)) {}

  /// May also explicitly construct with a `const char*` when you know the
  /// lifetime is static.
  static constexpr StaticString get(const char *ptr) {
    return StaticString(const_cast<char *>(ptr), /*unused*/ 4);
  }

  /// Return the value of the string. Because we want this class to be
  /// copyable/movable we can't const-qualify the char *, but we also don't want
  /// it to be modifiable so we only provide a getter.
  char *getValue() const { return value; }

private:
  constexpr StaticString(const char *value, int /*unused*/)
      : value(const_cast<char *>(value)) {}

  char *value;
};

} // end namespace M

#endif // SUPPORT_STATIC_STRING_H