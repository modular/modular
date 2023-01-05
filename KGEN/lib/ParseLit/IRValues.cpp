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
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/SMLoc.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// IRValues Implementation Logic.
//===----------------------------------------------------------------------===//

using VariantStorage = PointerUnion<MValue, DRValue, LValue>;

static raw_ostream &printStorage(raw_ostream &os, VariantStorage storage,
                                 bool isDump = false) {
  if (storage.isNull()) {
    os << "<NULL IR Value>\n";
  } else if (auto val = dyn_cast<MValue>(storage)) {
    if (isDump)
      os << "M: ";
    os << val.get();
  } else if (auto val = dyn_cast<DRValue>(storage)) {
    if (isDump)
      os << "DR: ";
    os << val;
  } else if (auto val = dyn_cast<LValue>(storage)) {
    if (isDump)
      os << "LV: ";
    os << val;
  } else {
    os << "<UNKNOWN IRVALUE>";
  }
  return os;
}

raw_ostream &LIT::operator<<(raw_ostream &os, MValue value) {
  return printStorage(os, value);
}
raw_ostream &LIT::operator<<(raw_ostream &os, RValue value) {
  return printStorage(os, value.getStorage());
}
raw_ostream &LIT::operator<<(raw_ostream &os, AnyValue value) {
  return printStorage(os, value.getStorage());
}

void RValue::dump() const {
  printStorage(llvm::errs(), getStorage(), true) << '\n';
}
void AnyValue::dump() const {
  printStorage(llvm::errs(), getStorage(), true) << '\n';
}

static std::string getStorageAsString(VariantStorage storage) {
  std::string result;
  llvm::raw_string_ostream os(result);
  printStorage(os, storage);
  return os.str();
}

mlir::Diagnostic &LIT::operator<<(mlir::Diagnostic &diag, MValue value) {
  return diag << '\'' << getStorageAsString(value) << '\'';
}
mlir::Diagnostic &LIT::operator<<(mlir::Diagnostic &diag, RValue value) {
  return diag << '\'' << getStorageAsString(value.getStorage()) << '\'';
}
mlir::Diagnostic &LIT::operator<<(mlir::Diagnostic &diag, AnyValue value) {
  return diag << '\'' << getStorageAsString(value.getStorage()) << '\'';
}

static Type getTypeFrom(VariantStorage storage) {
  if (storage.isNull())
    return Type();
  if (auto attr = dyn_cast<MValue>(storage))
    return attr.get().getType();
  if (auto value = dyn_cast<DRValue>(storage))
    return value.getType();
  return cast<LValue>(storage).getType();
}

Type RValue::getType() const { return getTypeFrom(storage); }
Type AnyValue::getType() const { return getTypeFrom(storage); }

MValue::MValue(Type value)
    : storage(ParameterizedTypeConstantAttr::get(value)) {}

/// If this value /is/ a type return it.
ASTType MValue::getIfTypeValue() const {
  auto attr = get();
  if (auto type = dyn_cast<ConcreteTypeConstantAttr>(attr))
    return type.getValue();
  if (auto type = dyn_cast<ParameterizedTypeConstantAttr>(attr))
    return type.getValue();

  // If this is a parameter expression of type value, use ParamRefType to turn
  // it into a type.
  if (isa<MLIRTypeType>(attr.getType()))
    return ParamRefType::get(attr);
  return {};
}

/// This method returns the type of this value when projected as an RValue.
/// If this is already an RValue, it is the type of the value.  If this is
/// an LValue, it strips off the pointer type.
ASTType AnyValue::getRValueType() const {
  if (auto lv = getIfLValue())
    return lv.getRValueType();
  return getType();
}

/// This method returns the type of this value when projected as an RValue.
/// If this is already an RValue, it is the type of the value.  If this is
/// an LValue, it strips off the pointer type.
ASTType LValue::getRValueType() const {
  TypedAttr attrType = llvm::cast<POP::PointerType>(getType()).getElementType();
  Type type = MValue(attrType).getIfTypeValue();
  assert(type && "LValue element type shouldn't be a parameter");
  return type;
}
