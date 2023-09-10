//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the IR Value classes.
//
//===----------------------------------------------------------------------===//

#include "IRValues.h"
#include "CallEmission.h"
#include "ExprNode.h"

#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/LITDialect/LITOps.h"
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
  } else if (auto val = dyn_cast<ORValue>(storage)) {
    if (isDump)
      os << "OR: ";
    os << '"' << val->baseName << "\" " << val->fnDecls.size() << " candidates";
  } else if (auto val = dyn_cast<SLValue>(storage)) {
    if (isDump)
      os << "SLV: ";
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
raw_ostream &LIT::operator<<(raw_ostream &os, ORValue value) {
  return printStorage(os, value);
}
raw_ostream &LIT::operator<<(raw_ostream &os, CRValue value) {
  return printStorage(os, value.getStorage());
}
raw_ostream &LIT::operator<<(raw_ostream &os, URValue value) {
  return printStorage(os, value.getStorage());
}
raw_ostream &LIT::operator<<(raw_ostream &os, RValue value) {
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
void CRValue::dump() const {
  printStorage(llvm::errs(), getStorage(), true) << '\n';
}
void URValue::dump() const {
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
  if (auto value = dyn_cast<SLValue>(storage))
    return value.getType();
  if (auto value = dyn_cast<DLValue>(storage))
    return value->elementType;
  assert(!isa<ORValue>(storage) && "overloaded rvalue has no type");
  llvm_unreachable("unknown IRValue");
}

ASTType CRValue::getType() const { return getTypeFrom(storage); }
ASTType CValue::getType() const { return getTypeFrom(storage); }
ASTType BValue::getType() const { return getTypeFrom(storage); }
ASTType LValue::getType() const { return getTypeFrom(storage); }

PValue::PValue(Type value)
    : storage(value ? ParameterizedTypeConstantAttr::get(value) : Attribute()) {
}

/// If this value /is/ a type return it.
ASTType PValue::getIfTypeValue() const {
  auto attr = get();
  if (auto type = dyn_cast<TypeConstantAttr>(attr))
    return type.getValue();

  // If this is a parameter expression of type value, use ParamRefType to turn
  // it into a type.
  if (isa<MLIRTypeType>(attr.getType()))
    return ParamRefType::get(attr);
  return {};
}

/// This method looks through the pointer in a MRValue to return the
/// underlying type.
ASTType CRValue::getRValueType() const {
  if (isa<MRValue>(storage))
    return getType().getPointerElementType();
  return getType();
}

ASTType CValue::getRValueType() const {
  if (isa<SLValue, MRValue, MBValue>(storage))
    return getType().getPointerElementType();
  return getType();
}

ASTType LValue::getRValueType() const {
  if (isa<SLValue>(storage))
    return getType().getPointerElementType();
  return getType();
}

ASTType BValue::getRValueType() const {
  if (isa<MBValue>(storage))
    return getType().getPointerElementType();
  return getType();
}

//===----------------------------------------------------------------------===//
// ORValue
//===----------------------------------------------------------------------===//

ORValue::ORValue() {}
ORValue::ORValue(const ORValue &existing) : storage(existing.storage.copy()) {}
ORValue::ORValue(RCRef<OverloadSetWrapper> storage)
    : storage(std::move(storage)) {}
ORValue::~ORValue() {}

ORValue &ORValue::operator=(const ORValue &existing) {
  storage = existing.storage.copy();
  return *this;
}

ORValue ORValue::create(OverloadSet &&set) {
  return ORValue(takeRCRef(new OverloadSetWrapper{std::move(set)}));
}

//===----------------------------------------------------------------------===//
// DLValue / BaseDLValue
//===----------------------------------------------------------------------===//

DLValue::~DLValue() {}

DLValue &DLValue::operator=(const DLValue &existing) {
  storage = existing.storage.copy();
  return *this;
}

BaseDLValue::~BaseDLValue() {
  // vtable anchor.
}

//===----------------------------------------------------------------------===//
// DiscardDLValue
//===----------------------------------------------------------------------===//

DiscardDLValue::DiscardDLValue(ASTType elementType, const ExprNode *expr)
    : BaseDLValue(elementType), expr(expr) {}

void DiscardDLValue::print(raw_ostream &os) const { os << "discard pattern"; }

//===----------------------------------------------------------------------===//
// StoredAttributeRefDLValue
//===----------------------------------------------------------------------===//

StoredAttributeRefDLValue::StoredAttributeRefDLValue(
    ASTExprAnd<DLValue> baseVal, StructFieldOp fieldOp, ASTType elementType,
    const ExprNode *expr)
    : BaseDLValue(elementType), expr(expr), baseVal(baseVal), fieldOp(fieldOp) {
}

StructFieldOp StoredAttributeRefDLValue::getField() const {
  return cast<StructFieldOp>(fieldOp);
}

void StoredAttributeRefDLValue::print(raw_ostream &os) const {
  os << "stored attr '" << getField().getName() << " : ";
  baseVal.ir->print(os);
}

//===----------------------------------------------------------------------===//
// SubscriptDLValue
//===----------------------------------------------------------------------===//

SubscriptDLValue::SubscriptDLValue(ArrayRef<FuncOperand> selfAndIndicesValue,
                                   ASTType elementType, const ExprNode *expr)
    : BaseDLValue(elementType), expr(expr),
      selfAndIndicesValue(selfAndIndicesValue.begin(),
                          selfAndIndicesValue.end()) {}

/// Return true if this is a subscript, false if this is an attribute access.
bool SubscriptDLValue::isSubscript() const {
  return expr->kind == ExprNode::kSubscript;
}

void SubscriptDLValue::print(raw_ostream &os) const {
  os << (isSubscript() ? "(subscript): " : "(property): ") << elementType
     << " ";
}

//===----------------------------------------------------------------------===//
// TupleDLValue
//===----------------------------------------------------------------------===//

TupleDLValue::TupleDLValue(ArrayRef<FuncOperand> eltLValues, ASTType tupleType,
                           const ExprNode *expr)
    : BaseDLValue(tupleType), expr(expr),
      eltLValues(eltLValues.begin(), eltLValues.end()) {
  for (auto &elt : eltLValues)
    assert(elt.ir.getIfLValue() && "element must be an lvalue");
}

void TupleDLValue::print(raw_ostream &os) const {
  os << "(tuple lvalue): " << elementType << " ";
}

//===----------------------------------------------------------------------===//
// GlobalDLValue
//===----------------------------------------------------------------------===//

GlobalDLValue::GlobalDLValue(GlobalVarDeclOp op, ASTType type, SMLoc loc)
    : BaseDLValue(type), op(op), loc(loc) {}

GlobalVarDeclOp GlobalDLValue::getGlobal() const {
  return cast<GlobalVarDeclOp>(op);
}

void GlobalDLValue::print(raw_ostream &os) const {
  GlobalVarDeclOp opMut = getGlobal();
  os << "(global lvalue): " << opMut.getSymName() << " : " << opMut.getType();
}
