//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LITDIALECT_LIFETIME_TRACKABLE_H
#define KGEN_LITDIALECT_LIFETIME_TRACKABLE_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"

namespace M::KGEN {
namespace LIT {

/// This class provide an abstraction for analyzing lifetime-trackable values,
/// e.g. variable definitions and owned arguments to functions.  This class can
/// also be used to query whether something is lifetime trackable or not, by
/// building a LifetimeTrackable and then querying it for null.
struct LifetimeTrackable {
  /// This constructor checks to see if the value is trackable, and if so
  /// identifies it.  If not, this returns a null value.
  LifetimeTrackable(Value value);

  /// This constructor checks to see if the value is trackable or a field of a
  /// trackable.  If so it identifies the underlying object being referenced. If
  /// not, this returns a null value.
  static Value findUnderlyingValueFromField(Value value);

  operator bool() const { return name != StringAttr(); }

  /// This is the user's declared name for the value declaration, or null if
  /// this isn't a tracked value.
  StringAttr name;

  /// This is true if the SSA value is a pointer to the logical storage instead
  /// of being the value itself.  This is always true for values of memory-only
  /// type.
  bool isIndirect = false;

  /// This is true if the value is uninitialized at function entry, false if it
  /// starts out initialized.
  bool startsUninit = false;

  /// This is true if the value is uninitialized at function exit, false if it
  /// ends up defined (e.g. as with a byref argument).
  bool endsUninit = false;

  /// True if this is a InitSelf argument: the self parameter in an
  /// __init__/__copyinit__ method.  These have magic behavior so they become
  /// fully initialized when all their fields are initialized.
  bool isFullObjectLiveOnEntry = false;

  /// Return the type of the underlying value, looking through the pointer type
  /// if this is an indirect reference.
  Type getValueType(Value value) const {
    return getTypeOrPointeeType(value.getType(), isIndirect);
  }

  /// When isIndirect is true, this strips off the top level PointerType
  /// from the specified type, otherwise it returns it unmodified.
  static Type getTypeOrPointeeType(Type type, bool isIndirect);
};
} // namespace LIT
} // namespace M::KGEN

#endif // KGEN_LITDIALECT_LIFETIME_TRACKABLE_H
