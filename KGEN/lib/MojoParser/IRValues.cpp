//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the IR Value classes.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/IRValues.h"
#include "ExprNodes.h"
#include "IREmitter.h"
#include "OverloadSet.h"

#include "KGEN/Interpreter/InterpreterAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/ExprNode.h"
#include "KGEN/POPDialect/POPTypes.h"

#include "llvm/Support/SMLoc.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// IRValue Implementation Logic.
//===----------------------------------------------------------------------===//

static raw_ostream &printStorage(raw_ostream &os,
                                 const AnyValue::Storage &storage,
                                 bool isDump = false) {
  if (isa<NullRepresentation>(storage)) {
    os << "<NULL IR Value>\n";
  } else if (auto val = dyn_cast<PValue>(storage)) {
    if (isDump)
      os << "P: ";
    os << val.get();
  } else if (auto val = dyn_cast<SRValue>(storage)) {
    if (isDump)
      os << "SR: ";
    os << val;
  } else if (auto val = dyn_cast<MRValue>(storage)) {
    if (isDump)
      os << "MR: ";
    os << val;
  } else if (auto val = dyn_cast<SBValue>(storage)) {
    if (isDump)
      os << "SB: ";
    os << val;
  } else if (auto val = dyn_cast<MBValue>(storage)) {
    if (isDump)
      os << "MB: ";
    os << val;
  } else if (auto val = dyn_cast<MBPValue>(storage)) {
    if (isDump)
      os << "MBP: ";
    os << val;
  } else if (auto val = dyn_cast<OverloadSetUValue>(storage)) {
    if (isDump)
      os << "OverloadSetUValue: ";
    os << '"' << val->baseName << "\" " << val->fnDecls.size()
       << " candidates.\n";
    val->dump();
  } else if (isa<InitializerUValue>(storage)) {
    if (isDump)
      os << "InitializerUValue";
    switch (cast<InitializerUValue>(storage).syntax) {
    case InitializerUValue::kSliceLiteral:
      os << "[SliceLiteral]:";
      break;
    case InitializerUValue::kListLiteral:
      os << "[ListLiteral]:";
      break;
    case InitializerUValue::kDictLiteral:
      os << "[DictLiteral]:";
      break;
    case InitializerUValue::kSetInitLiteral:
      os << "[SetInitLiteral]:";
      break;
    }
    os << cast<InitializerUValue>(storage).get();
  } else if (auto val = dyn_cast<MLValue>(storage)) {
    if (isDump)
      os << "ML: ";
    os << val;
  } else if (auto val = dyn_cast<RLValue>(storage)) {
    if (isDump)
      os << "RL: ";
    os << val;
  } else if (auto val = dyn_cast<PMBValue>(storage)) {
    if (isDump)
      os << "PMB: ";
    os << val.get();
  } else if (auto val = dyn_cast<PMRValue>(storage)) {
    if (isDump)
      os << "PMR: ";
    os << val.get();
  } else if (auto dlv = dyn_cast<DLValue>(storage)) {
    if (isDump)
      os << "DLV ";
    if (!dlv)
      os << "<<NULL>>";
    else
      dlv->print(os);
  } else {
    os << "<UNKNOWN IRVALUE>";
  }
  return os;
}

raw_ostream &LIT::operator<<(raw_ostream &os, PValue value) {
  return printStorage(os, value);
}
raw_ostream &LIT::operator<<(raw_ostream &os, OverloadSetUValue value) {
  return printStorage(os, value);
}
raw_ostream &LIT::operator<<(raw_ostream &os, InitializerUValue value) {
  return printStorage(os, value);
}
raw_ostream &LIT::operator<<(raw_ostream &os, UValue value) {
  return printStorage(os, value.getStorage());
}
raw_ostream &LIT::operator<<(raw_ostream &os, RValue value) {
  return printStorage(os, value.getStorage());
}
raw_ostream &LIT::operator<<(raw_ostream &os, CValue value) {
  return printStorage(os, value.getStorage());
}
raw_ostream &operator<<(raw_ostream &os, LValue value) {
  return printStorage(os, value.getStorage());
}
raw_ostream &operator<<(raw_ostream &os, BValue value) {
  return printStorage(os, value.getStorage());
}
raw_ostream &LIT::operator<<(raw_ostream &os, AnyValue value) {
  return printStorage(os, value.getStorage());
}

void PValue::dump() const { printStorage(llvm::errs(), *this, true) << '\n'; }
void PMBValue::dump() const { printStorage(llvm::errs(), *this, true) << '\n'; }
void PMRValue::dump() const { printStorage(llvm::errs(), *this, true) << '\n'; }

void CValue::dump() const {
  printStorage(llvm::errs(), getStorage(), true) << '\n';
}
void UValue::dump() const {
  printStorage(llvm::errs(), getStorage(), true) << '\n';
}
void RValue::dump() const {
  printStorage(llvm::errs(), getStorage(), true) << '\n';
}
void LValue::dump() const {
  printStorage(llvm::errs(), getStorage(), true) << '\n';
}
void BValue::dump() const {
  printStorage(llvm::errs(), getStorage(), true) << '\n';
}
void AnyValue::dump() const {
  printStorage(llvm::errs(), getStorage(), true) << '\n';
}

ASTType AnyValue::getRValueTypeIfResolvable() const {
  if (auto cValue = getIfCValue())
    return cValue.getRValueType();
  // Otherwise, try to narrow an overload set to a PValue.
  if (auto ovSet = getIfOverloadSet())
    if (auto pValue = ovSet->getIfPValue())
      return pValue.getRValueType();
  // Initializer lists have no implied type.
  return ASTType();
}

static ASTType getTypeFrom(AnyValue::Storage storage) {
  if (isa<NullRepresentation>(storage))
    return {};
  if (auto attr = dyn_cast<PValue>(storage))
    return attr.get().getType();
  if (auto value = dyn_cast<SRValue>(storage))
    return value.getType();
  if (auto value = dyn_cast<MRValue>(storage))
    return value.getType();
  if (auto value = dyn_cast<SBValue>(storage))
    return value.getType();
  if (auto value = dyn_cast<MBValue>(storage))
    return value.getType();
  if (auto value = dyn_cast<MBPValue>(storage))
    return value.getType();
  if (auto value = dyn_cast<MLValue>(storage))
    return value.getType();
  if (auto value = dyn_cast<RLValue>(storage))
    return value.getType();
  if (auto value = dyn_cast<PMBValue>(storage))
    return value.getType();
  if (auto value = dyn_cast<PMRValue>(storage))
    return value.getType();
  if (auto value = dyn_cast<DLValue>(storage))
    return value->elementType;
  assert(!isa<OverloadSetUValue>(storage) && "overloaded rvalue has no type");
  llvm_unreachable("unknown IRValue");
}

ASTType RValue::getType() const { return getTypeFrom(storage); }
ASTType CValue::getType() const { return getTypeFrom(storage); }
ASTType BValue::getType() const { return getTypeFrom(storage); }
ASTType LValue::getType() const { return getTypeFrom(storage); }

PValue::PValue(Type value)
    : storage(value
                  ? TypeParamAttr::get(value, ASTType(value).extractMetaType())
                  : Attribute()) {}

/// If this value /is/ a type return it.
ASTType PValue::getIfTypeValue() const {
  TypedAttr attr = get();
  // If this is a parameter expression of type value, use ParamType to turn
  // it into a type.
  if (LIT::isTypeExpr(attr))
    return ParamType::get(attr);
  return {};
}

/// Given an RValue, return a PMBValue that wraps it.
PMBValue PMBValue::getFromPValue(PValue value) {
  auto origin = ComptimeOriginAttr::get(value.get().getContext(), false);
  return {StoreToMemAttr::get(value, RefType::get(value.getType(), origin))};
}

/// Return the RValue that this wraps.
TypedAttr PMBValue::getUnderlyingRValue() const {
  return cast<StoreToMemAttr>(storage).getValue();
}

/// Given an RValue, return a PMRValue that wraps it.
PMRValue PMRValue::getFromPValue(PValue value) {
  auto origin = ComptimeOriginAttr::get(value.get().getContext(), true);
  return {StoreToMemAttr::get(value, RefType::get(value.getType(), origin))};
}

/// Return the RValue that this wraps.
TypedAttr PMRValue::getUnderlyingRValue() const {
  return cast<StoreToMemAttr>(storage).getValue();
}

/// If this value is a type, then return it.  This can happen when this is a
/// PValue with a type metatype (e.g. a computed type) or if it is some other
/// value that has struct metatype type.
ASTType VariantValueStorageBase::getIfTypeValue() const {
  // We can only evaluate this on CValues.
  auto cv = CValue::getFrom(storage);
  if (!cv)
    return {};

  // If this is a PValue, then we can use it directly.
  if (auto value = dyn_cast<PValue>(storage))
    return value.getIfTypeValue();

  // Otherwise, check to see if this is some other sort of value that is
  // returning a struct metatype.  If so, we know it is a singleton result.
  if (auto structMeta = sugarDynCast<StructMetaType>(cv.getRValueType()))
    return structMeta.getType();

  return {};
}

/// This method looks through references to return the element type.
ASTType RValue::getRValueType() const {
  auto type = getType();
  if (isa<MRValue, PMRValue>(storage))
    return type.getReferenceElementType();
  return type;
}

ASTType CValue::getRValueType() const {
  auto type = getType();
  if (isMValue())
    return type.getReferenceElementType();
  if (auto rl = dyn_cast<RLValue>(storage))
    return rl.getRValueType();
  return type;
}

ASTType LValue::getRValueType() const {
  auto type = getType();
  if (auto ml = dyn_cast<MLValue>(storage))
    return ml.getRValueType();
  if (auto rl = dyn_cast<RLValue>(storage))
    return rl.getRValueType();
  return type;
}

ASTType BValue::getRValueType() const {
  auto type = getType();
  if (isMValue())
    return type.getReferenceElementType();
  return type;
}

/// Given an MValue, return the underlying reference.
Value VariantValueStorageBase::getMValueReference() const {
  if (auto lvalue = dyn_cast<MLValue>(storage))
    return lvalue;
  if (auto rl = dyn_cast<RLValue>(storage))
    return rl;
  if (auto rvalue = dyn_cast<MRValue>(storage))
    return rvalue;
  if (auto bvalue = dyn_cast<MBValue>(storage))
    return bvalue;
  if (auto mbpvalue = dyn_cast<MBPValue>(storage))
    return mbpvalue;
  assert(!isa<PMBValue>(storage) && !isa<PMRValue>(storage) &&
         "getMValueReference() doesn't work on PMBValue or PMRValue");
  llvm_unreachable("invalid use of non-MValue");
}

RefType VariantValueStorageBase::getMValueType() const {
  if (auto pmbvalue = dyn_cast<PMBValue>(storage))
    return pmbvalue.getRefType();
  if (auto pmrvalue = dyn_cast<PMRValue>(storage))
    return pmrvalue.getRefType();
  return sugarCast<RefType>(getMValueReference().getType());
}

/// Given an S*Value, return the underlying register.
Value VariantValueStorageBase::getSValueRegister() const {
  if (auto rvalue = dyn_cast<SRValue>(storage))
    return rvalue;
  if (auto bvalue = dyn_cast<SBValue>(storage))
    return bvalue;
  llvm_unreachable("invalid use of non-SValue");
}

/// Given an S*Value or M*Value, return the underlying register/reference.  If
/// not, return a null Value.
Value VariantValueStorageBase::getMlirValue() const {
  if (auto lvalue = dyn_cast<MLValue>(storage))
    return lvalue;
  if (auto rvalue = dyn_cast<MRValue>(storage))
    return rvalue;
  if (auto bvalue = dyn_cast<MBValue>(storage))
    return bvalue;
  if (auto mbpvalue = dyn_cast<MBPValue>(storage))
    return mbpvalue;
  if (auto rvalue = dyn_cast<SRValue>(storage))
    return rvalue;
  if (auto bvalue = dyn_cast<SBValue>(storage))
    return bvalue;
  if (auto rlvalue = dyn_cast<RLValue>(storage))
    return rlvalue;
  return Value();
}

void PMBValue::check() const {
#ifndef NDEBUG
  auto storeToMem = ::sugarDynCast<StoreToMemAttr>(get());
  assert(storeToMem && "PMBValue can only be used for a comptime memory value");
  assert(::sugarIsa<RefType>(storeToMem.getType()) &&
         "PMBValue should be a ref");
  assert(::sugarIsa<ComptimeOriginAttr>(getRefType().getOrigin()) &&
         "PMBValue should have a comptime origin");
  assert(getRefType().isMutableKnown(false) && "PMBValue should be immutable");
#endif
}

void PMRValue::check() const {
#ifndef NDEBUG
  auto storeToMem = ::sugarDynCast<StoreToMemAttr>(get());
  assert(storeToMem && "PMRValue can only be used for a comptime memory value");
  assert(::sugarIsa<RefType>(storeToMem.getType()) &&
         "PMRValue should be a ref");
  assert(::sugarIsa<ComptimeOriginAttr>(getRefType().getOrigin()) &&
         "PMRValue should have a comptime origin");
  assert(getRefType().isMutableKnown(true) && "PMRValue should be mutable");
#endif
}

void MRValue::check() const {
  assert(::sugarIsa<RefType>(Value::getType()) &&
         ::sugarCast<RefType>(Value::getType()).isMutableKnown(true) &&
         "MRValue can only be used for a mutable reference");
}

void MLValue::check() const {
  assert(::sugarIsa<RefType>(Value::getType()) &&
         ::sugarCast<RefType>(Value::getType()).isMutableKnown(true) &&
         "MLValue can only be used for a mutable reference");
}

void RLValue::check() const {
  assert(::sugarIsa<RefType>(Value::getType()) &&
         ::sugarCast<RefType>(Value::getType()).isMutableKnown(true) &&
         "RLValue can only be used for a mutable reference");
  assert(::sugarIsa<RefType>(
             ::sugarCast<RefType>(Value::getType()).getElementType()) &&
         "RLValue should be ref of ref");
}

void MBValue::check() const {
  // MBValue allow any mutability.
  assert(::sugarIsa<RefType>(Value::getType()));
}

void MBPValue::check() const {
  assert(::sugarIsa<RefType>(Value::getType()) &&
         ::sugarCast<RefType>(Value::getType()).getMutabilityClass() ==
             OriginType::Parametric &&
         "MBPValue can only be used for a parametric mutability reference");
}

/// Given a value of !lit.ref type, return an MLValue/MBValue/MBPValue
/// depending on the mutability of the reference.
CValue CValue::getMValueForRef(Value refValue) {
  switch (sugarCast<RefType>(refValue.getType()).getMutabilityClass()) {
  case OriginType::Mutable:
    return MLValue(refValue);
  case OriginType::Immutable:
    return MBValue(refValue);
  case OriginType::Parametric:
    return MBPValue(refValue);
  }
}

//===----------------------------------------------------------------------===//
// OverloadSetUValue
//===----------------------------------------------------------------------===//

OverloadSetUValue::OverloadSetUValue() = default;
OverloadSetUValue::OverloadSetUValue(const OverloadSetUValue &existing)
    : storage(existing.storage.copy()) {}
OverloadSetUValue::OverloadSetUValue(RCRef<OverloadSetWrapper> storage)
    : storage(std::move(storage)) {}
OverloadSetUValue::~OverloadSetUValue() = default;

OverloadSetUValue &
OverloadSetUValue::operator=(const OverloadSetUValue &existing) {
  storage = existing.storage.copy();
  return *this;
}

OverloadSetUValue OverloadSetUValue::create(OverloadSet &&set) {
  return OverloadSetUValue(takeRCRef(new OverloadSetWrapper{std::move(set)}));
}

//===----------------------------------------------------------------------===//
// InitializerUValue
//===----------------------------------------------------------------------===//

/// This provides a wrapper around CallOperands which is reference counted,
/// allowing InitializerUValue to maintain it while still being copyable.
struct InitializerUValue::ImplWrapper
    : public NonAtomicallyReferenceCounted<ImplWrapper> {
  ImplWrapper(CallOperands &&operands) : operands(std::move(operands)) {}
  CallOperands operands;
};

InitializerUValue::InitializerUValue(const InitializerUValue &existing)
    : syntax(existing.syntax), storage(existing.storage.copy()) {}
InitializerUValue::InitializerUValue(Syntax syntax, RCRef<ImplWrapper> storage)
    : syntax(syntax), storage(std::move(storage)) {}
InitializerUValue::~InitializerUValue() = default;

InitializerUValue &
InitializerUValue::operator=(const InitializerUValue &existing) {
  storage = existing.storage.copy();
  return *this;
}

InitializerUValue InitializerUValue::create(Syntax syntax,
                                            CallOperands &&operands) {
  return InitializerUValue(syntax,
                           takeRCRef(new ImplWrapper{std::move(operands)}));
}

const CallOperands &InitializerUValue::get() const { return storage->operands; }

static void addNoneLiteralMarker(CallOperands &operands, StringRef kwargName,
                                 IREmitter &emitter) {
  // Emit the None in a parameter context so we don't eagerly generated IR into
  // the body of any current function.
  auto paramEmitter = emitter.getParamEmitter(EC_CollectionLiteral);
  SimpleLiteralNode noneLiteral(SimpleLiteralNode::kNoneLiteral,
                                operands.callExpr->getLoc());
  RValue noneValue =
      paramEmitter.emitExprRValue(&noneLiteral, EC_CollectionLiteral);
  assert(noneValue && "failed to emit None literal");
  operands.addKeyword(StringAttr::get(paramEmitter.getContext(), kwargName),
                      {noneValue, operands.callExpr});
}

ASTType InitializerUValue::getDefaultType(SharedState &shared) const {
  auto loc = get().callExpr->getLoc();
  switch (syntax) {
  case Syntax::kListLiteral:
    return shared.getStandardCollectionType(loc, "List");
  case Syntax::kDictLiteral:
    return shared.getStandardCollectionType(loc, "Dict");
  case Syntax::kSetInitLiteral:
    return shared.getStandardCollectionType(loc, "Set");
  case Syntax::kSliceLiteral:
    return {}; // TODO: This could default to Slice.
  }
}

/// Given an inferred type for this initializer list, return the operands that
/// we should use to try to construct it.  This returns failure if invalid.
CallOperands
InitializerUValue::getOperandsForInferredType(ASTType type, ExprDest &dest,
                                              IREmitter &emitter) const {
  // FIXME: need the correct ExprDest here.
  CallOperands operands(get(), dest);
  switch (syntax) {
  case Syntax::kSliceLiteral:
    addNoneLiteralMarker(operands, "__slice_literal__", emitter);
    break;
  case Syntax::kListLiteral:
    addNoneLiteralMarker(operands, "__list_literal__", emitter);
    break;
  case Syntax::kDictLiteral:
    addNoneLiteralMarker(operands, "__dict_literal__", emitter);
    break;
  case Syntax::kSetInitLiteral:
    // Given we have an inferred type, we can interrogate it a bit.  If there
    // are any keyword arguments, then we leave this as an initializer list.
    if (llvm::any_of(operands.values, [](const auto &operand) {
          return operand.keyword != StringAttr();
        }))
      break;

    // Otherwise if this is an empty initializer list, check to see if the type
    // conforms to the dict protocol.  If so, we emit this as a dict literal so
    // {} turns into a dict with PythonObject.

    // Convert MySet[*(0, 0)] to MySet[?] so we can infer the parameter(s).
    type = type.getWithUnknownParametersReplaced(emitter.shared);
    if (operands.values.empty()) {
      auto getEmptyList = [&]() -> AnyValue {
        return InitializerUValue::create(
            InitializerUValue::kListLiteral,
            CallOperands(CallSyntax::kTypeCall, operands.callExpr));
      };

      // Call __init__(keys=[], values=[], __dict_literal__=())
      CallOperands dictOperands(CallSyntax::kTypeCall, operands.callExpr);
      dictOperands.add({getEmptyList(), operands.callExpr});
      dictOperands.add({getEmptyList(), operands.callExpr});
      addNoneLiteralMarker(dictOperands, "__dict_literal__", emitter);
      CallOperands dictCopy(dictOperands, operands.dest);
      FailureOr<PValue> pValue = OverloadSet::canConstructType(
          type, std::move(dictOperands), emitter.declScope);
      if (succeeded(pValue) && pValue.value())
        return dictCopy;
    }

    // Otherwise, check to see if we can emit this as a set literal. It will
    // take precedent over initializer list emission, because (e.g.)
    // PythonObject's set literal ctor takes a required keyword argument.
    CallOperands setOperands(CallSyntax::kTypeCall, operands.callExpr);
    addNoneLiteralMarker(setOperands, "__set_literal__", emitter);
    FailureOr<PValue> pValue = OverloadSet::canConstructType(
        type, std::move(setOperands), emitter.declScope);
    if (succeeded(pValue) && pValue.value()) {
      addNoneLiteralMarker(operands, "__set_literal__", emitter);
      break;
    }
    // Otherwise, leave it alone as an initializer list.
    break;
  }
  return operands;
}

/// Emit this as a CValue if it can be resolved, otherwise emit an ambiguity
/// error and return null.
CValue InitializerUValue::emitAsCValue(IREmitter &emitter, ExprDest &dest) {

  // If we have the inferred contextual type, we can emit the constructor call.
  if (ASTType expectedType = dest.getExpectedTypeIfSpecified()) {
    CallOperands operands =
        getOperandsForInferredType(expectedType, dest, emitter);
    return emitter.emitConstructorCall(expectedType, std::move(operands), dest);
  }

  // For a list or set literal, we need to unify the elements into a common
  // element type.
  auto unifyOperands = [&](ArrayRef<OperandValue> operands) -> CallOperands {
    assert(!operands.empty() && "empty operands cannot be unified");

    // Emit all the values as CValues without a contextual type.
    SmallVector<CValue> elements;
    for (const auto &operand : operands) {
      auto value = emitter.emitCValue(operand, EC_CollectionLiteral);
      if (!value)
        return CallOperands(CallSyntax::kTypeCall, operand.expr);
      elements.push_back(value);
    }

    // Okay, now we can pairwise merge the elements into the first element to
    // get a final unified element type (as the first element's type).
    const ExprNode *lhsExpr = get()[0].expr;
    for (size_t i = 1; i != elements.size(); ++i) {
      if (failed(emitter.coerceTypesToEachOther(lhsExpr->getLoc(), elements[0],
                                                lhsExpr, elements[i],
                                                get()[i].expr, {})))
        return CallOperands(CallSyntax::kTypeCall, lhsExpr);
    }

    // If that succeeded, then the final result type of the first element is
    // the unified element type, which could have changed across each of the
    // elements. Form the constructor's operand list with a consistent element
    // type which will be used for the constructor call, allowing it to infer
    // the element type.
    CallOperands result(CallSyntax::kTypeCall, get().callExpr);
    for (auto [i, elt] : llvm::enumerate(elements)) {
      auto *expr = get()[i].expr;

      // Make sure all of the elements agree with the first element's unified.
      if (failed(emitter.coerceTypesToEachOther(lhsExpr->getLoc(), elements[0],
                                                lhsExpr, elements[i],
                                                get()[i].expr, {})))
        return CallOperands(CallSyntax::kTypeCall, expr);
      result.add({elt, expr});
    }
    return result;
  };

  // Otherwise, handle defaulting.
  switch (syntax) {
  case Syntax::kSliceLiteral:
    // TODO: This could get more aggressive, looking through implicit
    // conversions etc.
    emitter.emitError(get().callExpr->getLoc(),
                      "cannot emit slice expression without a contextual type");
    return {};
  case Syntax::kListLiteral: {
    if (get().empty()) {
      emitter.emitError(get().callExpr->getLoc(),
                        "cannot emit an empty list without a contextual type");
      return {};
    }

    auto operands = unifyOperands(get().values);
    if (operands.values.empty())
      return {};

    // Add the __list_literal__ kwarg.
    addNoneLiteralMarker(operands, "__list_literal__", emitter);
    auto listType = emitter.shared.getStandardCollectionType(
        get().callExpr->getLoc(), "List");
    if (!listType)
      return {};
    return emitter.emitConstructorCall(listType, std::move(operands), dest);
  }
  case Syntax::kDictLiteral: {
    // Let the nested list literals try to infer their own common element
    // types recursively.  We just default to Dict.
    auto dictType = emitter.shared.getStandardCollectionType(
        get().callExpr->getLoc(), "Dict");
    if (!dictType)
      return {};
    CallOperands operands(get(), dest);
    addNoneLiteralMarker(operands, "__dict_literal__", emitter);
    return emitter.emitConstructorCall(dictType, std::move(operands), dest);
  }
  case Syntax::kSetInitLiteral: {
    // If there are values with no keywords, then this can be emitted as a
    // set, inferring the element type from the values.  If there are
    // keywords, or if it is empty, then this is an error.
    bool hasKWArg = llvm::any_of(get().values, [](const auto &operand) {
      return operand.keyword != StringAttr();
    });
    if (hasKWArg || get().values.empty()) {
      auto diag = emitter.emitError(get().callExpr->getLoc());
      if (dest.getContext() == EC_DefaultArgument) {
        diag << "cannot infer initializer list type for default value from "
                "parametric type; use an explicit constructor call";
      } else {
        diag << "cannot emit initializer list without a contextual type";
      }
      return {};
    }

    // Otherwise, all values, just pass them into Set constructor.
    auto operands = unifyOperands(get().values);
    if (operands.values.empty())
      return {};

    // Add the __set_literal__ kwarg.
    addNoneLiteralMarker(operands, "__set_literal__", emitter);

    auto setType = emitter.shared.getStandardCollectionType(
        get().callExpr->getLoc(), "Set");
    if (!setType)
      return {};
    return emitter.emitConstructorCall(setType, std::move(operands), dest);
  }
  }
  return {};
}
