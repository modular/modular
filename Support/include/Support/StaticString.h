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
  /// Construct an Error with a static error string.
  template <size_t n>
  constexpr StaticString(const char (&value)[n]) : value(value) {}

  /// This is the value of the string, it is public since it is immutable.
  const char *const value;
};

} // end namespace M

#endif // SUPPORT_STATIC_STRING_H