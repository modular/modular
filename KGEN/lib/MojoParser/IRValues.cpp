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
#include "CallEmission.h"
#include "ExprNodes.h"
#include "IREmitter.h"

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
    os << '"' << val->baseName << "\" " << val->fnDecls.size() << " candidates";
  } else if (isa<InitializerUValue>(storage)) {
    if (isDump)
      os << "InitializerUValue";
    switch (cast<InitializerUValue>(storage).syntax) {
    case InitializerUValue::kSlice:
      os << "[Slice]:";
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
  if (auto value = dyn_cast<DLValue>(storage))
    return value->elementType;
  assert(!isa<OverloadSetUValue>(storage) && "overloaded rvalue has no type");
  llvm_unreachable("unknown IRValue");
}

ASTType RValue::getType() const { return getTypeFrom(storage); }
ASTType CValue::getType() const { return getTypeFrom(storage); }
ASTType BValue::getType() const { return getTypeFrom(storage); }
ASTType LValue::getType() const { return getTypeFrom(storage); }

/// Given a type value, attempt to extract a metatype.
static Type extractMetaType(Type type) {
  // The metatype is stored on the type.
  if (auto declRef = dyn_cast<LIT::StructType>(type))
    return StructMetaType::get(LIT::StructType::get(
        declRef.getSymbol(), declRef.getParamValues(), declRef.getSignature()));
  // The metatype is the type of the carried type expression.
  if (auto paramRef = dyn_cast<ParamType>(type))
    return paramRef.getParam().getType();
  if (auto traitRef = dyn_cast<TraitType>(type))
    return traitRef.getMetaType();

  // Otherwise, this is a generic MLIR type.
  return TypeType::get(type.getContext());
}

PValue::PValue(Type value)
    : storage(value ? TypeParamAttr::get(value, extractMetaType(value))
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
  if (auto structMeta = dyn_cast<StructMetaType>(cv.getRValueType()))
    return structMeta.getType();

  return {};
}

/// This method looks through references to return the element type.
ASTType RValue::getRValueType() const {
  auto type = getType();
  if (isa<MRValue>(storage))
    return type.getReferenceElementType();
  return type;
}

ASTType CValue::getRValueType() const {
  auto type = getType();
  if (isMValue())
    return type.getReferenceElementType();
  return type;
}

ASTType LValue::getRValueType() const {
  auto type = getType();
  if (isa<MLValue>(storage))
    return type.getReferenceElementType();
  return type;
}

ASTType BValue::getRValueType() const {
  auto type = getType();
  if (isa<MBValue, MBPValue>(storage))
    return type.getReferenceElementType();
  return type;
}

/// Given an MValue, return the underlying reference.
Value VariantValueStorageBase::getMValueReference() const {
  if (auto lvalue = dyn_cast<MLValue>(storage))
    return lvalue;
  if (auto rvalue = dyn_cast<MRValue>(storage))
    return rvalue;
  if (auto bvalue = dyn_cast<MBValue>(storage))
    return bvalue;
  if (auto mbpvalue = dyn_cast<MBPValue>(storage))
    return mbpvalue;
  llvm_unreachable("invalid use of non-MValue");
}

RefType VariantValueStorageBase::getMValueType() const {
  return cast<RefType>(getMValueReference().getType());
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
  return Value();
}

void MRValue::check() const {
  assert(::isa<RefType>(Value::getType()) &&
         ::cast<RefType>(Value::getType()).isMutableKnown(true) &&
         "MRValue can only be used for a mutable reference");
}
void MLValue::check() const {
  assert(::isa<RefType>(Value::getType()) &&
         ::cast<RefType>(Value::getType()).isMutableKnown(true) &&
         "MLValue can only be used for a mutable reference");
}
void MBValue::check() const {
  // MBValue allow any mutability.
  assert(::isa<RefType>(Value::getType()));
}

void MBPValue::check() const {
  assert(::isa<RefType>(Value::getType()) &&
         ::cast<RefType>(Value::getType()).getMutabilityClass() ==
             OriginType::Parametric &&
         "MBPValue can only be used for a parametric mutability reference");
}

/// Given a value of !lit.ref type, return an MLValue/MBValue/MBPValue
/// depending on the mutability of the reference.
CValue CValue::getMValueForRef(Value refValue) {
  switch (cast<RefType>(refValue.getType()).getMutabilityClass()) {
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
    : syntax(existing.syntax), expr(existing.expr),
      storage(existing.storage.copy()) {}
InitializerUValue::InitializerUValue(Syntax syntax, const ExprNode *expr,
                                     RCRef<ImplWrapper> storage)
    : syntax(syntax), expr(expr), storage(std::move(storage)) {}
InitializerUValue::~InitializerUValue() = default;

InitializerUValue &
InitializerUValue::operator=(const InitializerUValue &existing) {
  storage = existing.storage.copy();
  return *this;
}

InitializerUValue InitializerUValue::create(Syntax syntax, const ExprNode *expr,
                                            CallOperands &&operands) {
  return InitializerUValue(syntax, expr,
                           takeRCRef(new ImplWrapper{std::move(operands)}));
}

const CallOperands &InitializerUValue::get() const { return storage->operands; }

static void addEmptyTuple(CallOperands &operands, StringRef kwargName,
                          const ExprNode *expr, IREmitter &emitter) {
  // Emit the tuple in a parameter context so we don't eagerly generated IR into
  // the body of any current function.
  auto paramEmitter = emitter.getParamEmitter(EC_CollectionLiteral);

  TupleNode emptyTuple(expr->getLoc(), {});
  if (auto tupleValue =
          paramEmitter.emitExprRValue(&emptyTuple, EC_CollectionLiteral))
    operands.add(StringAttr::get(paramEmitter.getContext(), kwargName),
                 {tupleValue, expr});
};

/// Given an inferred type for this initializer list, return the operands that
/// we should use to try to construct it.  This returns failure if invalid.
CallOperands
InitializerUValue::getOperandsForInferredType(ASTType type,
                                              IREmitter &emitter) const {
  CallOperands operands(get());
  switch (syntax) {
  case Syntax::kSlice:
    break;
  case Syntax::kListLiteral:
    addEmptyTuple(operands, "__list_literal__", expr, emitter);
    break;
  case Syntax::kDictLiteral:
    addEmptyTuple(operands, "__dict_literal__", expr, emitter);
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
        return InitializerUValue::create(InitializerUValue::kListLiteral, expr,
                                         CallOperands());
      };

      // Call __init__(keys=[], values=[], __dict_literal__=())
      CallOperands dictOperands;
      dictOperands.add({getEmptyList(), expr});
      dictOperands.add({getEmptyList(), expr});
      addEmptyTuple(dictOperands, "__dict_literal__", expr, emitter);
      CallOperands dictCopy(dictOperands);
      FailureOr<PValue> pValue = OverloadSet::canConstructType(
          type, std::move(dictOperands), expr, emitter.declScope,
          /*isImplicitConversion=*/false);
      if (succeeded(pValue) && pValue.value())
        return dictCopy;
    }

    // Otherwise, check to see if we can emit this as a set literal. It will
    // take precedent over initializer list emission, because (e.g.)
    // PythonObject's set literal ctor takes a required keyword argument.
    CallOperands setOperands;
    addEmptyTuple(setOperands, "__set_literal__", expr, emitter);
    FailureOr<PValue> pValue = OverloadSet::canConstructType(
        type, std::move(setOperands), expr, emitter.declScope,
        /*isImplicitConversion=*/false);
    if (succeeded(pValue) && pValue.value()) {
      addEmptyTuple(operands, "__set_literal__", expr, emitter);
      break;
    }
    // Otherwise, leave it alone as an initializer list.
    break;
  }
  return operands;
};

/// Emit this as a CValue if it can be resolved, otherwise emit an ambiguity
/// error and return null.
CValue InitializerUValue::emitAsCValue(IREmitter &emitter, ValueDest &dest) {

  // If we have the inferred contextual type, we can emit the constructor call.
  if (ASTType expectedType = dest.getExpectedTypeIfSpecified()) {
    CallOperands operands = getOperandsForInferredType(expectedType, emitter);
    return emitter.emitConstructorCall(expectedType, std::move(operands), expr,
                                       CallSyntax::kTypeCall, dest);
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
        return {};
      elements.push_back(value);
    }

    // Okay, now we can pairwise merge the elements into the first element to
    // get a final unified element type (as the first element's type).
    const ExprNode *lhsExpr = get()[0].expr;
    for (size_t i = 1; i != elements.size(); ++i) {
      if (failed(emitter.coerceTypesToEachOther(lhsExpr->getLoc(), elements[0],
                                                lhsExpr, elements[i],
                                                get()[i].expr, {})))
        return {};
    }

    // If that succeeded, then the final result type of the first element is
    // the unified element type, which could have changed across each of the
    // elements. Form the constructor's operand list with a consistent element
    // type which will be used for the constructor call, allowing it to infer
    // the element type.
    CallOperands result;
    for (auto [i, elt] : llvm::enumerate(elements)) {
      auto *expr = get()[i].expr;

      // Make sure all of the elements agree with the first element's unified.
      if (failed(emitter.coerceTypesToEachOther(lhsExpr->getLoc(), elements[0],
                                                lhsExpr, elements[i],
                                                get()[i].expr, {})))
        return {};
      result.add({elt, expr});
    }
    return result;
  };

  // Otherwise, handle defaulting.
  switch (syntax) {
  case Syntax::kSlice:
    emitter.emitError(expr->getLoc(),
                      "cannot emit slice expression without a contextual type");
    return {};
  case Syntax::kListLiteral: {
    if (get().empty()) {
      emitter.emitError(expr->getLoc(),
                        "cannot emit an empty list without a contextual type");
      return {};
    }

    auto operands = unifyOperands(get().values);
    if (operands.values.empty())
      return {};

    // Add the __list_literal__ kwarg.
    addEmptyTuple(operands, "__list_literal__", expr, emitter);
    auto listType =
        emitter.shared.getStandardCollectionType(expr->getLoc(), "List");
    if (!listType)
      return {};
    return emitter.emitConstructorCall(listType, std::move(operands), expr,
                                       CallSyntax::kTypeCall, dest);
  }
  case Syntax::kDictLiteral: {
    // Let the nested list literals try to infer their own common element
    // types recursively.  We just default to Dict.
    auto dictType =
        emitter.shared.getStandardCollectionType(expr->getLoc(), "Dict");
    if (!dictType)
      return {};
    CallOperands operands(get());
    addEmptyTuple(operands, "__dict_literal__", expr, emitter);
    return emitter.emitConstructorCall(dictType, std::move(operands), expr,
                                       CallSyntax::kTypeCall, dest);
  }
  case Syntax::kSetInitLiteral: {
    // If there are values with no keywords, then this can be emitted as a
    // set, inferring the element type from the values.  If there are
    // keywords, or if it is empty, then this is an error.
    bool hasKWArg = llvm::any_of(get().values, [](const auto &operand) {
      return operand.keyword != StringAttr();
    });
    if (hasKWArg || get().values.empty()) {
      emitter.emitError(
          expr->getLoc(),
          "cannot emit initializer list without a contextual type");
      return {};
    }

    // Otherwise, all values, just pass them into Set constructor.
    auto operands = unifyOperands(get().values);
    if (operands.values.empty())
      return {};

    // Add the __set_literal__ kwarg.
    addEmptyTuple(operands, "__set_literal__", expr, emitter);

    auto setType =
        emitter.shared.getStandardCollectionType(expr->getLoc(), "Set");
    if (!setType)
      return {};
    return emitter.emitConstructorCall(setType, std::move(operands), expr,
                                       CallSyntax::kTypeCall, dest);
  }
  }
  return {};
}
