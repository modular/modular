//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ZAPDialect/ZAPDialect.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace M;
using namespace M::KGEN;

namespace {
/// This class defines the interface for handling inlining for zap
/// dialect operations.
struct ZAPInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All zap dialect ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ZAPDialect
//===----------------------------------------------------------------------===//

void ZAP::ZAPDialect::initialize() {
  registerOperations();
  registerTypes();
  addInterface<ZAPInlinerInterface>();
}

Operation *ZAP::ZAPDialect::materializeConstant(OpBuilder &b, Attribute value,
                                                Type type, Location loc) {
  if ((value.isa<IntegerAttr>() && type.isa<IndexType>()) ||
      (value.isa<DTypeConstantAttr>() && type.isa<DTypeType>()))
    return b.create<ParamConstantOp>(loc, value.cast<TypedAttr>());
  return nullptr;
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "KGEN/ZAPDialect/ZAPDialect.cpp.inc"
