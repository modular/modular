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
  } else if (auto val = dyn_cast<XRValue>(storage)) {
    if (isDump)
      os << "XR: ";
    os << val;
  } else if (auto val = dyn_cast<SBValue>(storage)) {
    if (isDump)
      os << "SB: ";
    os << val;
  } else if (auto val = dyn_cast<MBValue>(storage)) {
    if (isDump)
      os << "MB: ";
    os << val;
  } else if (auto val = dyn_cast<XBValue>(storage)) {
    if (isDump)
      os << "XB: ";
    os << val;
  } else if (auto val = dyn_cast<ORValue>(storage)) {
    if (isDump)
      os << "OR: ";
    os << '"' << val->baseName << "\" " << val->fnDecls.size() << " candidates";
  } else if (auto val = dyn_cast<MLValue>(storage)) {
    if (isDump)
      os << "ML: ";
    os << val;
  } else if (auto val = dyn_cast<XLValue>(storage)) {
    if (isDump)
      os << "XL: ";
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
  if (auto value = dyn_cast<XRValue>(storage))
    return value.getType();
  if (auto value = dyn_cast<SBValue>(storage))
    return value.getType();
  if (auto value = dyn_cast<MBValue>(storage))
    return value.getType();
  if (auto value = dyn_cast<XBValue>(storage))
    return value.getType();
  if (auto value = dyn_cast<MLValue>(storage))
    return value.getType();
  if (auto value = dyn_cast<XLValue>(storage))
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

/// Given a type value, attempt to extract a metatype.
static Type extractMetaType(Type type) {
  // The metatype is stored on the type.
  if (auto declRef = dyn_cast<DeclRefType>(type))
    return declRef.getMetaType();
  // The metatype is the type of the carried type expression.
  if (auto paramRef = dyn_cast<ParamRefType>(type))
    return paramRef.getParam().getType();
  // TODO: build AnyTypeType?
  // Otherwise, this is a generic MLIR type.
  return AnyRegTypeType::get(type.getContext());
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

/// This method looks through the pointer in a MRValue to return the
/// underlying type.
ASTType CRValue::getRValueType() const {
  auto type = getType();
  if (isa<MRValue>(storage))
    return type.getPointerElementType();
  if (isa<XRValue>(storage))
    return type.getReferenceElementType();
  return type;
}

ASTType CValue::getRValueType() const {
  auto type = getType();
  if (isa<MLValue, MRValue, MBValue>(storage))
    return type.getPointerElementType();
  if (isa<XLValue, XRValue, XBValue>(storage))
    return type.getReferenceElementType();
  return type;
}

ASTType LValue::getRValueType() const {
  auto type = getType();
  if (isa<MLValue>(storage))
    return type.getPointerElementType();
  if (isa<XLValue>(storage))
    return type.getReferenceElementType();
  return type;
}

ASTType BValue::getRValueType() const {
  auto type = getType();
  if (isa<MBValue>(storage))
    return type.getPointerElementType();
  if (isa<XBValue>(storage))
    return type.getReferenceElementType();
  return type;
}

// TODO(lifetimes): remove pedantic checks.
void MRValue::check() const { assert(::isa<PointerType>(Value::getType())); }
void MLValue::check() const { assert(::isa<PointerType>(Value::getType())); }
void MBValue::check() const { assert(::isa<PointerType>(Value::getType())); }
void XRValue::check() const { assert(::isa<RefType>(Value::getType())); }
void XLValue::check() const { assert(::isa<RefType>(Value::getType())); }
void XBValue::check() const { assert(::isa<RefType>(Value::getType())); }

//===----------------------------------------------------------------------===//
// ORValue
//===----------------------------------------------------------------------===//

ORValue::ORValue() = default;
ORValue::ORValue(const ORValue &existing) : storage(existing.storage.copy()) {}
ORValue::ORValue(RCRef<OverloadSetWrapper> storage)
    : storage(std::move(storage)) {}
ORValue::~ORValue() = default;

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

DLValue::~DLValue() = default;

DLValue &DLValue::operator=(const DLValue &existing) {
  storage = existing.storage.copy();
  return *this;
}

BaseDLValue::~BaseDLValue() = default; // vtable anchor.

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

SubscriptDLValue::SubscriptDLValue(
    SmallVectorImpl<FuncOperand> &&posOperands,
    SmallDenseMap<StringAttr, FuncOperand> &&kwOperands, ASTType elementType,
    const ExprNode *expr)
    : BaseDLValue(elementType), expr(expr), posOperands(std::move(posOperands)),
      kwOperands(std::move(kwOperands)) {}

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
