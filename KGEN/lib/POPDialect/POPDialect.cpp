//===- POPDialect.cpp -----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the POP dialect.
//
//===----------------------------------------------------------------------===//

#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

namespace {
/// This class defines the interface for handling inlining for pop
/// dialect operations.
struct POPInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All pop dialect ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }
};
} // namespace

// Pull in the dialect definition.
#include "KGEN/POPDialect/POPDialect.cpp.inc"

// Register operations.
void POPDialect::initialize() {
  registerAttributes();

  addOperations<
#define GET_OP_LIST
#include "KGEN/POPDialect/POP.cpp.inc"
      >();
  addInterfaces<POPInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// Constant
//===----------------------------------------------------------------------===//

/// Registered hook to materialize a constant operation from a "pop" dialect
/// op that is folded.
Operation *POPDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  if (ConstantOp::isBuildableWith(value, type))
    return builder.create<ConstantOp>(loc, type, value);
  return nullptr;
}
