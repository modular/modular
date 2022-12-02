//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides information for working with 'special functions' in Lit
// like the __new__ function.
//
//===----------------------------------------------------------------------===//

#ifndef SPECIAL_FUNCTIONS_H
#define SPECIAL_FUNCTIONS_H

namespace M::KGEN::LIT {

enum class SpecialFunctionKind : uint8_t {
  // This is not a special function.  This enumerator should always have value
  // zero so it can be used as a false condition in an if.
  kNormal = 0,

#define SF(ENUM, NAME, NUMGPERANDS, FLAGS) ENUM,
#include "SpecialFunctions.def"
};

struct SpecialFunctionInfo {
  const char *name = nullptr;
  SpecialFunctionKind kind = SpecialFunctionKind::kNormal;

  /// This is the number of operands that this special function requires, or -1
  /// if variadic.
  int numOperands = -1;
  unsigned flags = 0;

  /// This is a bitmask of flags that describes requirements of the special
  /// function.
  enum {
    /// This is an implicitly static method like __new__ even if not declared
    /// as such.
    kImplicitlyStaticMethod = 1 << 0,

    /// This must be an instance method of a type.
    kInstMethod = 1 << 1,

    /// On a method of struct, the self must be passed ByRef.  This is true for
    /// in-place operators like += / __iadd__.  This implies an instance method.
    kByRefSelfInstMethod = (1 << 2) | kInstMethod,
  };

  bool isByRefSelfInstMethod() const {
    return (flags & kByRefSelfInstMethod) == kByRefSelfInstMethod;
  }

  /// Return a record that describes special functions like __init__.  The
  /// kind field identifies it.
  static const SpecialFunctionInfo &get(StringRef name) {
    return get(getKind(name));
  }
  static const SpecialFunctionInfo &get(SpecialFunctionKind kind);

  /// Given a function name like "__init__" return the special function kind
  /// that corresponds to it.
  static SpecialFunctionKind getKind(StringRef name);
};

} // namespace M::KGEN::LIT

#endif // SPECIAL_FUNCTIONS_H
