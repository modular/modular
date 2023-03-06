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
#include "LitExprCalls.h"

#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "llvm/Support/SMLoc.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// IRValue Implementation Logic.
//===----------------------------------------------------------------------===//

using VariantStorage = SmartVariant<NullRepresentation, PRValue, SRValue,
                                    MRValue, ORValue, LValue>;

static raw_ostream &printStorage(raw_ostream &os, const VariantStorage &storage,
                                 bool isDump = false) {
  if (isa<NullRepresentation>(storage)) {
    os << "<NULL IR Value>\n";
  } else if (auto val = dyn_cast<PRValue>(storage)) {
    if (isDump)
      os << "PR: ";
    os << val.get();
  } else if (auto val = dyn_cast<SRValue>(storage)) {
    if (isDump)
      os << "SR: ";
    os << val;
  } else if (auto val = dyn_cast<MRValue>(storage)) {
    if (isDump)
      os << "MR: ";
    os << val;
  } else if (auto val = dyn_cast<ORValue>(storage)) {
    if (isDump)
      os << "OR: ";
    os << '"' << val->baseName << "\" " << val->fnDecls.size() << " candidates";
  } else if (auto val = dyn_cast<LValue>(storage)) {
    if (isDump)
      os << "LV: ";
    os << val;
  } else {
    os << "<UNKNOWN IRVALUE>";
  }
  return os;
}

raw_ostream &LIT::operator<<(raw_ostream &os, PRValue value) {
  return printStorage(os, value);
}
raw_ostream &LIT::operator<<(raw_ostream &os, ORValue value) {
  return printStorage(os, value);
}
raw_ostream &LIT::operator<<(raw_ostream &os, CRValue value) {
  return printStorage(os, value.getStorage());
}
raw_ostream &LIT::operator<<(raw_ostream &os, RValue value) {
  return printStorage(os, value.getStorage());
}
raw_ostream &LIT::operator<<(raw_ostream &os, AnyValue value) {
  return printStorage(os, value.getStorage());
}

void CRValue::dump() const {
  printStorage(llvm::errs(), getStorage(), true) << '\n';
}
void RValue::dump() const {
  printStorage(llvm::errs(), getStorage(), true) << '\n';
}
void AnyValue::dump() const {
  printStorage(llvm::errs(), getStorage(), true) << '\n';
}

static ASTType getTypeFrom(VariantStorage storage) {
  if (auto attr = dyn_cast<PRValue>(storage))
    return attr.get().getType();
  if (auto value = dyn_cast<SRValue>(storage))
    return value.getType();
  if (auto value = dyn_cast<MRValue>(storage))
    return value.getType();
  if (auto value = dyn_cast<LValue>(storage))
    return value.getType();
  assert(!isa<ORValue>(storage) && "overloaded rvalue has no type");

  // Otherwise null.
  return Type();
}

ASTType CRValue::getType() const { return getTypeFrom(storage); }
ASTType RValue::getType() const { return getTypeFrom(storage); }
ASTType AnyValue::getType() const { return getTypeFrom(storage); }

PRValue::PRValue(Type value)
    : storage(value ? ParameterizedTypeConstantAttr::get(value) : Attribute()) {
}

/// If this value /is/ a type return it.
ASTType PRValue::getIfTypeValue() const {
  auto attr = get();
  if (auto type = dyn_cast<TypeConstantAttr>(attr))
    return type.getValue();

  // If this is a parameter expression of type value, use ParamRefType to turn
  // it into a type.
  if (isa<MLIRTypeType>(attr.getType()))
    return ParamRefType::get(attr);
  return {};
}

/// This method returns the type of this value when projected as an RValue.
/// Since LValue's are always stored by-pointer, this strips it off.
ASTType LValue::getRValueType() const {
  return getType().getPointerElementType();
}

/// MRValue's represent the address of the stored value.  This returns the
/// RValue type, the declared type of the value.
ASTType MRValue::getRValueType() const {
  return getType().getPointerElementType();
}

/// This method looks through the pointer in a MRValue to return the
/// underlying type.
ASTType CRValue::getRValueType() const {
  if (isa_and_nonnull<MRValue>(storage))
    return getType().getPointerElementType();
  return getType();
}

/// This method returns the type of this value when projected as an RValue.
/// If this is an LValue or MRValue, it strips off the pointer type.
ASTType AnyValue::getRValueType() const {
  if (isa_and_nonnull<LValue, MRValue>(storage))
    return getType().getPointerElementType();
  return getType();
}

//===----------------------------------------------------------------------===//
// ORValue
//===----------------------------------------------------------------------===//

ORValue::ORValue() {}
ORValue::ORValue(const ORValue &existing) : storage(existing.storage.copy()) {}
ORValue::ORValue(LLCL::RCRef<OverloadSetWrapper> storage)
    : storage(std::move(storage)) {}
ORValue::~ORValue() {}

ORValue &ORValue::operator=(const ORValue &existing) {
  storage = existing.storage.copy();
  return *this;
}

ORValue ORValue::create(OverloadSet &&set) {
  return ORValue(LLCL::takeRCRef(new OverloadSetWrapper{std::move(set)}));
}
