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
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/SMLoc.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// IRValues Implementation Logic.
//===----------------------------------------------------------------------===//

using VariantStorage = PointerUnion<MAValue, ASTType, DRValue, LValue>;

static raw_ostream &printStorage(raw_ostream &os, VariantStorage storage,
                                 bool isDump = false) {
  if (storage.isNull()) {
    os << "<NULL IR Value>\n";
  } else if (auto val = dyn_cast<MAValue>(storage)) {
    if (isDump)
      os << "MA: ";
    os << val.get();
  } else if (auto val = dyn_cast<ASTType>(storage)) {
    if (isDump)
      os << "MT: ";
    os << val;
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
  return printStorage(os, value.getStorage());
}
raw_ostream &LIT::operator<<(raw_ostream &os, RValue value) {
  return printStorage(os, value.getStorage());
}
raw_ostream &LIT::operator<<(raw_ostream &os, AnyValue value) {
  return printStorage(os, value.getStorage());
}

void MValue::dump() const {
  printStorage(llvm::errs(), getStorage(), true) << '\n';
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
  return diag << '\'' << getStorageAsString(value.getStorage()) << '\'';
}
mlir::Diagnostic &LIT::operator<<(mlir::Diagnostic &diag, RValue value) {
  return diag << '\'' << getStorageAsString(value.getStorage()) << '\'';
}
mlir::Diagnostic &LIT::operator<<(mlir::Diagnostic &diag, AnyValue value) {
  return diag << '\'' << getStorageAsString(value.getStorage()) << '\'';
}

static Type getTypeFrom(VariantStorage storage, MLIRContext *context) {
  if (storage.isNull())
    return Type();
  if (auto attr = dyn_cast<MAValue>(storage))
    return attr.get().getType();
  if (auto value = dyn_cast<DRValue>(storage))
    return value.getType();
  if (auto value = dyn_cast<LValue>(storage))
    return value.getType();

  if (isa<ASTType>(storage))
    return MLIRTypeType::get(context);

  // TODO: Handle ASTType.
  llvm_unreachable("unhandled case ASTType");
}

Type MValue::getType(MLIRContext *context) const {
  return getTypeFrom(storage, context);
}
Type RValue::getType(MLIRContext *context) const {
  return getTypeFrom(storage, context);
}
Type AnyValue::getType(MLIRContext *context) const {
  return getTypeFrom(storage, context);
}

/// Lower this MValue to a TypedAttr.  If this contains an ASTType, it is
/// lowered to an MLIRType and wrapped in a ParameteredTypeConstantAttr.
TypedAttr MValue::lowerToAttribute(LitSharedState &shared, Location loc) const {
  assert(!isNull() && "Cannot emit null attribute");

  // If this is already an attribute, return it.
  if (auto attr = getIfMAValue())
    return attr;

  // If this is a type, convert it.
  auto astType = getIfMTValue();
  assert(astType && "Unknown MValue kind");
  return ParameterizedTypeConstantAttr::get(shared.getMLIRType(astType, loc));
}

/// Lower this MValue to a TypedAttr.  If this contains an ASTType, it is
/// lowered to an MLIRType and wrapped in a ParameteredTypeConstantAttr.
TypedAttr MValue::lowerToAttribute(LitSharedState &shared,
                                   llvm::SMLoc loc) const {
  assert(!isNull() && "Cannot emit null attribute");

  // If this is already an attribute, return it.
  if (auto attr = getIfMAValue())
    return attr;

  // If this is a type, convert it.
  auto astType = getIfMTValue();
  assert(astType && "Unknown MValue kind");
  return ParameterizedTypeConstantAttr::get(shared.getMLIRType(astType, loc));
}
