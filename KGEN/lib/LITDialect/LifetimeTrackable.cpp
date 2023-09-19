//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LITDialect/LifetimeTrackable.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// LifetimeTrackable
//===----------------------------------------------------------------------===//

LifetimeTrackable::LifetimeTrackable(Value v) {
  if (!v) // Null value isn't tracked.
    return;

  // LetReg starts out initialized with its own value.
  if (auto letReg = v.getDefiningOp<LetRegDeclOp>()) {
    name = letReg.getNameAttr();
    isIndirect = false;
    startsUninit = true; // Initialized at its definition point.
    endsUninit = true;
    return;
  }
  // VarLetDeclOp is uninit and ends that way.
  if (auto varLet = v.getDefiningOp<VarLetDeclOp>()) {
    name = varLet.getNameAttr();
    isIndirect = true;
    startsUninit = true;
    endsUninit = true;
    return;
  }
  // VarLetDecl2Op is uninit and ends that way.
  if (auto varLet = v.getDefiningOp<VarLetDecl2Op>()) {
    name = varLet.getNameAttr();
    isIndirect = true;
    startsUninit = true;
    endsUninit = true;
    return;
  }

  if (auto loadConsume = v.getDefiningOp<LoadConsumeOp>()) {
    name = StringAttr::get(v.getContext(), "(anonymous value)");
    isIndirect = false;
    startsUninit = true;
    endsUninit = true;
    return;
  }

  // The lit.ownership.end.lifetime op ends a register/mem lifetime and creates
  // a new one.  This defines the properties of its new lifetime.
  if (auto endLifetime = v.getDefiningOp<OwnershipEndLifetimeOp>()) {
    name = StringAttr::get(v.getContext(), "(consumed value)");
    isIndirect = !endLifetime.getIsReg();
    startsUninit = true;
    endsUninit = true;
    return;
  }

  // The lit.ownership.make.pointer.lvalue op takes an address and projects to a
  // liveness tracked indirect value.
  if (auto makePointer = v.getDefiningOp<OwnershipMakePointerLValue>()) {
    name = StringAttr::get(v.getContext(), "(pointee value)");
    isIndirect = true;
    startsUninit = !makePointer.getLiveOnEntry();
    endsUninit = !makePointer.getLiveOnExit();
    return;
  }

  /// Owned results of function calls are tracked as being initialized when
  /// defined but needing to be destroyed by the end of function.
  if (OpResult res = dyn_cast<OpResult>(v)) {
    if (auto call = dyn_cast<KGENCallOpInterface>(res.getOwner())) {
      if (call.getCalleeType().hasOwnedRegisterResult()) {
        name = StringAttr::get(v.getContext(), "<call result>");
        isIndirect = false;
        startsUninit = true;
        endsUninit = true;
      }
    }
  }

  // If this is a function argument, check to see what ownership it has.
  auto bbArg = dyn_cast<BlockArgument>(v);
  if (!bbArg || !bbArg.getOwner())
    return;
  SignatureType signature;
  Operation *parentOp = bbArg.getOwner()->getParentOp();
  if (auto func = dyn_cast<LIT::FuncOp>(parentOp)) {
    signature = func.getSignature();
  } else if (auto func = dyn_cast<ParamDeclareRegionOp>(parentOp)) {
    signature = func.getSignature();
  } else
    return;

  unsigned argIdx = bbArg.getArgNumber();
  switch (signature.getInputConvention(argIdx)) {
  case ValueInputConvention::OwnedInReg: // This gets an LValue slot.
  case ValueInputConvention::BorrowedInReg:
  case ValueInputConvention::BorrowedInMem:
    // These are immutable so don't need to be tracked.
    return;

  case ValueInputConvention::OwnedInMem:
    isIndirect = true;
    startsUninit = false;
    endsUninit = true;
    break;
  case ValueInputConvention::ByRefResult:
    // FIXME(Issue#12196): __result__ slots in raising functions cannot properly
    // model the behavior when an error is thrown, so we give up tracking them.
    if (signature.isThrows())
      return;
    isIndirect = true;
    startsUninit = true;
    endsUninit = false;
    break;
  case ValueInputConvention::InitSelf:
    // Unlike byref-result, we allow memberwise initialization of 'self' in an
    // init method to construct a full value.
    isIndirect = true;
    startsUninit = true;
    endsUninit = false;
    isFullObjectLiveOnEntry = true;
    break;
  case ValueInputConvention::ByRef:
    isIndirect = true;
    startsUninit = false;
    endsUninit = false;
    break;
  }

  ArrayRef<StringAttr> argNames = signature.getArgNames();
  name = argNames[argIdx];
  if (!name.size()) {
    name = StringAttr::get(v.getContext(), "(positional-only argument # " +
                                               Twine(argIdx) + ")");
  }
}

/// This constructor checks to see if the value is trackable or a field of a
/// trackable.  If so it identifies the underlying object being referenced. If
/// not, this returns a null value.
Value LifetimeTrackable::findUnderlyingValueFromField(Value value) {
  // If there are any GEP operations into the struct, dig through them.
  bool hadGEP = false;

  // TODO(references): Eliminate pointer operations and be exclusively about
  // references.
  while (1) {
    if (auto structGEP = value.getDefiningOp<StructGEPOp>()) {
      hadGEP = true;
      value = structGEP.getContainer();
    } else if (auto refToPointer = value.getDefiningOp<RefToPointerOp>()) {
      value = refToPointer.getRef();
    } else {
      break;
    }
  }

  // Check if there is a base value.
  LifetimeTrackable result(value);
  // If we had a GEP of this value but it doesn't have indirect storage, then
  // we aren't actually tracking the pointers off this value field sensitively,
  // so we can't be confident about what is going on with it.
  if (!result || (hadGEP && !result.isIndirect))
    return Value();
  // Otherwise, use whatever we found.
  return value;
}

/// When isIndirect is true, this strips off the top level pointer from the
/// specified type, otherwise it returns it unmodified.
Type LifetimeTrackable::getTypeOrPointeeType(Type type, bool isIndirect) {
  // If this is a direct value, use the type directly.
  if (!isIndirect)
    return type;

  // TODO(references): Remove support for raw pointers.
  if (auto refType = dyn_cast<RefType>(type))
    return refType.getElementAsType();
  return llvm::cast<PointerType>(type).getElementAsType();
}
