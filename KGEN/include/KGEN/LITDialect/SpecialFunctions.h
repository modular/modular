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

#ifndef KGEN_KGENDIALECT_SPECIAL_FUNCTIONS_H
#define KGEN_KGENDIALECT_SPECIAL_FUNCTIONS_H

namespace M::KGEN::LIT {

enum class SpecialFunctionKind : uint8_t {
  // This is not a special function.  This enumerator should always have value
  // zero so it can be used as a false condition in an if.
  kNormal = 0,

#define SF(ENUM, NAME, MINOPERANDS, MAXOPERANDS, EXPRNODE, FLAGS) ENUM,
#include "KGEN/LITDialect/SpecialFunctions.def"
};

class SpecialFunctionInfo {
public:
  const char *name = nullptr;
  SpecialFunctionKind kind = SpecialFunctionKind::kNormal;

  /// The minimum number of arguments that this special function requires.
  unsigned minNumArguments = 0;

  /// The maximum number of arguments that this special function requires, or -1
  /// if variadic.
  int maxNumArguments = -1;

  unsigned flags = 0;

  /// This is a bitmask of flags that describes requirements of the special
  /// function.
  enum {
    /// This is an implicitly static method like __new__ even if not declared
    /// as such.
    kImplicitlyStaticMethod = 1 << 0,

    /// This must be an instance method of a type.
    kInstMethod = 1 << 1,

    /// On a method of a struct, the self must be passed as Owned argument
    /// convention.
    kRequiresOwnedSelfInstMethod = (1 << 2) | kInstMethod,

    /// This is true when this represents a "reversed" operator like __radd__.
    kReversedOperator = 1 << 3,

    /// This is true when the operation is supposed to return None.
    kNoneResult = 1 << 4,

    /// This method must return Self.
    kSelfResult = 1 << 5,

    /// This method is a struct initializer, it takes 'inout self'
    /// and returns None.
    kInitializer = (1 << 6) | kInstMethod | kNoneResult,

    /// This method cannot be declared to raise an error.
    kCannotRaise = 1 << 7,
  };

  /// Return true if this is any kind of instance method.
  bool isInstMethod() const { return (flags & kInstMethod) != 0; }

  bool requiresOwnedSelfInstMethod() const {
    return (flags & kRequiresOwnedSelfInstMethod) ==
           kRequiresOwnedSelfInstMethod;
  }

  /// Return true if this is a reversed operator.
  bool isReversed() const { return (flags & kReversedOperator) != 0; }

  /// Return true if this special function must return None.
  bool hasNoneResult() const { return (flags & kNoneResult) != 0; }

  /// Return true if this special function is an initializer.
  bool isInitializer() const { return (flags & kInitializer) == kInitializer; }

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

#endif // KGEN_KGENDIALECT_SPECIAL_FUNCTIONS_H
