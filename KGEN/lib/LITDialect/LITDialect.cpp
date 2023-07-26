//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LIT dialect.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LITDialect/LITDialect.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// LITDialectFoldInterface
//===----------------------------------------------------------------------===//

namespace {
struct LITDialectFoldInterface : public mlir::DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  /// Never hoist a constant out of a declaration scope. We could scan the
  /// parameters declarations to find the highest scope a constant could be
  /// hoisted into, but that is expensive to do.
  bool shouldMaterializeInto(Region *region) const override {
    return isa<DeclInterface>(region->getParentOp());
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// LITOpAsmDialectInterface
//===----------------------------------------------------------------------===//

namespace {
struct LITOpAsmDialectInterface : public mlir::OpAsmDialectInterface {
  using mlir::OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (!attr)
      return AliasResult::NoAlias;

    return TypeSwitch<Attribute, AliasResult>(attr)
        .Case([&](DocStringAttr attr) {
          // Doc strings are nearly always long, so make sure to print them as
          // aliases.
          os << "doc_string";
          return AliasResult::OverridableAlias;
        })
        .Default([](Attribute) { return AliasResult::NoAlias; });
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#include "KGEN/LITDialect/LITDialect.cpp.inc"

void LITDialect::initialize() {
  // Register attributes.
  registerAttributes();
  addInterfaces<LITDialectFoldInterface, LITOpAsmDialectInterface>();

  // Register types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "KGEN/LITDialect/LITTypes.cpp.inc"
      >();

  // Give the lifetime type a pretty kgen type.
  auto *kgenDialect = getContext()->getOrLoadDialect<KGENDialect>();
  kgenDialect->registerPrettyType(
      "lifetime",
      [](AsmParser &p) -> Type { return LifetimeType::get(p.getContext()); },
      TypeID::get<LifetimeType>(),
      +[](AsmPrinter &p, Type type) { p << "lifetime"; });

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "KGEN/LITDialect/LIT.cpp.inc"
      >();
}

Operation *LITDialect::materializeConstant(OpBuilder &b, Attribute value,
                                           Type type, Location loc) {
  return b.create<ParamConstantOp>(loc, type, cast<TypedAttr>(value));
}

//===----------------------------------------------------------------------===//
// Type implementations.
//===----------------------------------------------------------------------===//

RefType RefType::get(bool isMutable, TypedAttr elementType,
                     TypedAttr lifetime) {
  auto *ctx = elementType.getContext();
  return get(ctx, BoolAttr::get(ctx, isMutable), elementType, lifetime);
}

RefType RefType::get(bool isMutable, Type elementType, TypedAttr lifetime) {
  return get(isMutable, TypeConstantAttr::get(elementType), lifetime);
}

Type RefType::getElementAsType() {
  TypedAttr elemType = getElementType();
  if (auto typeCst = llvm::dyn_cast<TypeConstantAttr>(elemType))
    return typeCst.getValue();
  assert(::isa<MLIRTypeType>(elemType.getType()) &&
         "parameter expr must have metatype type");
  return ParamRefType::get(elemType);
}

REPLResultRefType REPLResultRefType::get(Type elementType) {
  auto *ctx = elementType.getContext();
  return get(ctx, elementType);
}

/// Print/Parse a parameter value that is known to have `lifetime` type.
static void printLifetimeParamValue(AsmPrinter &p, TypedAttr value) {
  printParamValue(p, value);
}
static ParseResult parseLifetimeParamValue(AsmParser &p, TypedAttr &value) {
  return parseParamValue(p, value,
                         LifetimeType::get(p.getBuilder().getContext()));
}

/// Print/Parse the 'mut' keyword as 1, and its absence as 0.
static void printMutFlag(AsmPrinter &p, BoolAttr value) {
  if (value.getValue())
    p << "mut ";
}
static ParseResult parseMutFlag(AsmParser &p, BoolAttr &value) {
  value =
      BoolAttr::get(p.getContext(), succeeded(p.parseOptionalKeyword("mut")));
  return success();
}

#define GET_TYPEDEF_CLASSES
#include "KGEN/LITDialect/LITTypes.cpp.inc"
