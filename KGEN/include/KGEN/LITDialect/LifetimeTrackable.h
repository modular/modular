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

class CachedTypeLifetimeFinder {
public:
  /// This method finds all the lifetimes buried in the specified type,
  /// returning them as a list.
  SmallVector<TypedAttr> findLifetimesInType(Type type) {
    return findLifetimesInTypes(type);
  }

  /// This finds all the lifetimes in the specified set of types, possibly
  /// eliding duplicates.
  SmallVector<TypedAttr> findLifetimesInTypes(ArrayRef<Type> types);

private:
  llvm::DenseSet<const void *> typesAndAttrsWithoutLifetimes;
};

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

  /// This value feels true'y when it is initialized by something that can be
  /// lifetime tracked.
  operator bool() const { return !!name; }

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

  /// This enum indicates the expected initialization state of a value upon
  /// return from a function. A function can return normally or return in an
  /// error state.
  enum ExitInitState {
    /// Value is never initialized upon function exit.
    EndsUninit,
    /// Value is always initialized upon function exit (e.g. as with a inout
    /// argument).
    EndsInit,
    /// Value is initialized upon a normal function exit (e.g. as with a
    /// byref_result or init_self argument).
    InitOnNormal,
    /// Value is initialized upon an error function exit (e.g. as with a
    /// byref_error argument).
    InitOnError
  };

  /// The expected initialization state of the value upon exit from a function.
  ExitInitState endInitState = ExitInitState::EndsUninit;

  /// True if this is a InitSelf argument: the self parameter in an
  /// __init__/__copyinit__ method.  These have magic behavior so they become
  /// fully initialized when all their fields are initialized.
  bool isFullObjectLiveOnEntry = false;
};

//===----------------------------------------------------------------------===//
// OperationValueEffects
//===----------------------------------------------------------------------===//

enum class ResultEffect {
  /// This is an ignorable result value, e.g. a value of trivial type.
  ignore,

  /// The result defines a new reg value, e.g. an owned register  result of a
  /// function call.
  regDefine,

  /// The result is a ref that starts uninitialized when defined, but is
  /// initialized by the end of the function.
  memDefineUninitToInit,

  /// The result is a ref that starts uninitialized when defined, and is also
  /// uninitialized by the end of the function.
  memDefineUninitToUninit,

  /// The result is a ref that starts initialized when defined, and is also
  /// initialized by the end of the function.
  memDefineInitToInit,

  /// The result is a ref that starts initialized when defined, but is
  /// uninitialized by the end of the function.
  memDefineInitToUninit,
};

enum class OperandEffect {
  /// This reads a register value and uses it, but does not consume it, e.g.
  /// a borrowed_reg argument.
  regUse,

  /// This takes ownership of an inreg value, e.g. owned_reg argument or
  /// RefStoreOp (which transfers ownership from the operand to the memory).
  regConsume,

  /// This is used by operations that load the value, things like RefLoadOp,
  /// LoadConsumeOp, OwnershipUseOp, and passing a borrowed operand.
  memLoad,

  /// This is store to the pointer that overwrites whatever is in it with a new
  /// owned value.  For example, RefStoreOp, InitSelf and ByRefResult call
  /// operands all do this.
  memStoreOwned,

  /// inout arg to a function call.  Value must be initialized before the
  /// operation, may be mutated, but then is still live afterward.
  memInOut,

  /// This loads a value from the operand and takes ownership of the result, for
  /// example, owned operands (e.g. __del__) and LoadConsume.
  memConsume,

  /// This indicates that the full-object should be considered destroyed, but
  /// any fields within it are still valid.
  memMarkDestroyed,
};

/// This is the result value of `getOperationValueEffects`, indicating
/// out-of-bound effects (aka special cases) and whether the op is unknown.
enum class OverallOpValueEffect {
  /// this indicates that the returned value effects cover everything.
  allHandled = 0,

  /// This is returned when the operation is unknown.
  unknownOp,

  /// This is a terminator op like return or unreachable.
  terminatorOp,

  /// This is HLCF::BreakOp, HLCF::ContinueOp, LIT::TryRaiseOp, which all
  /// perform local control flow.
  localControlFlowOp,

  /// This is HLCF::IfOp, ParamIfOp, which are all if-like.
  ifLikeOp,

  /// This is HLCF::ElifOp specifically.
  elifOp,

  /// This is HLCF::LoopOp.
  loopOp,

  /// This is LIT::TryOp.
  tryOp,
};

/// This computes the effects that an operation has on any operands, result
/// values, and other declared lifetimes. This information is used by both
/// phases of CheckLifetimes.
OverallOpValueEffect
getOperationEffects(Operation &op,
                    SmallVectorImpl<std::pair<Value, OperandEffect>> &operands,
                    SmallVectorImpl<ResultEffect> &results,
                    SmallVectorImpl<TypedAttr> &lifetimes,
                    CachedTypeLifetimeFinder &lifetimeFinder);

} // namespace LIT
} // namespace M::KGEN

#endif // KGEN_LITDIALECT_LIFETIME_TRACKABLE_H
