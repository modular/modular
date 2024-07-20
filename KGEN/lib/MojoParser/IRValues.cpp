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
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/CallEmission.h"
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
  } else if (auto val = dyn_cast<OverloadSetUValue>(storage)) {
    if (isDump)
      os << "OverloadSetUValue: ";
    os << '"' << val->baseName << "\" " << val->fnDecls.size() << " candidates";
  } else if (auto val = dyn_cast<InitializerUValue>(storage)) {
    if (isDump)
      os << "InitializerUValue: ";
    os << val.get();
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
    return declRef.getMetaType();
  // The metatype is the type of the carried type expression.
  if (auto paramRef = dyn_cast<ParamRefType>(type))
    return paramRef.getParam().getType();
  if (auto traitRef = dyn_cast<TraitType>(type))
    return traitRef.getMetaType();

  // Otherwise, this is a generic MLIR type.
  return TypeType::get(type.getContext());
}

PValue::PValue(Type value)
    : storage(value ? TypeConstantAttr::get(value, extractMetaType(value))
                    : Attribute()) {}

/// If this value /is/ a type return it.
ASTType PValue::getIfTypeValue() const {
  TypedAttr attr = get();
  // If this is a parameter expression of type value, use ParamRefType to turn
  // it into a type.
  if (LIT::isTypeExpr(attr))
    return ParamRefType::get(attr);
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
  if (isa<MLValue, MRValue, MBValue>(storage))
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
  if (isa<MBValue>(storage))
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

// TODO(lifetimes): remove pedantic checks.
void MRValue::check() const { assert(::isa<RefType>(Value::getType())); }
void MLValue::check() const { assert(::isa<RefType>(Value::getType())); }
void MBValue::check() const { assert(::isa<RefType>(Value::getType())); }

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

/// This provides a wrapper around OperandContainer which is reference counted,
/// allowing InitializerUValue to maintain it while still being copyable.
struct InitializerUValue::CallOperandsWrapper
    : public NonAtomicallyReferenceCounted<CallOperandsWrapper> {
  CallOperandsWrapper(ArrayRef<ASTExprAnd<AnyValue>> operands)
      : operands(operands.begin(), operands.end()) {}
  SmallVector<ASTExprAnd<AnyValue>> operands;
};

InitializerUValue::InitializerUValue() {}
InitializerUValue::InitializerUValue(const InitializerUValue &existing)
    : storage(existing.storage.copy()) {}
InitializerUValue::InitializerUValue(RCRef<CallOperandsWrapper> storage)
    : storage(std::move(storage)) {}
InitializerUValue::~InitializerUValue() = default;

InitializerUValue &
InitializerUValue::operator=(const InitializerUValue &existing) {
  storage = existing.storage.copy();
  return *this;
}

InitializerUValue
InitializerUValue::create(ArrayRef<ASTExprAnd<AnyValue>> operands) {
  return InitializerUValue(
      takeRCRef(new CallOperandsWrapper{std::move(operands)}));
}

OperandContainer InitializerUValue::get() const {
  return OperandContainer(storage->operands);
}
