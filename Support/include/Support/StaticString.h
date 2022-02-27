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
  /// Implicitly construct an Error with a static error string.
  template <size_t n>
  /*implicit*/ constexpr StaticString(const char (&value)[n]) : value(value) {}

  /// May also explicitly construct with a `const char*` when you know the
  /// lifetime is static.
  static constexpr StaticString get(const char *ptr) {
    return StaticString(ptr, /*unused*/ 4);
  }

  /// This is the value of the string, it is public since it is immutable.
  const char *const value;

private:
  constexpr StaticString(const char *value, int /*unused*/) : value(value) {}
};

} // end namespace M

#endif // SUPPORT_STATIC_STRING_H