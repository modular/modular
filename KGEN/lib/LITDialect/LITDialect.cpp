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
// Dialect specification.
//===----------------------------------------------------------------------===//

// Pull in the dialect definition.
#include "KGEN/LITDialect/LITDialect.cpp.inc"

void LITDialect::initialize() {
  // Register attributes.
  registerAttributes();
  addInterfaces<LITDialectFoldInterface>();

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

#define GET_TYPEDEF_CLASSES
#include "KGEN/LITDialect/LITTypes.cpp.inc"

Operation *LITDialect::materializeConstant(OpBuilder &b, Attribute value,
                                           Type type, Location loc) {
  return b.create<ParamConstantOp>(loc, type, cast<TypedAttr>(value));
}
