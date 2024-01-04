//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_FOLDGLOBALCONSTLOADS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct FoldGlobalConstLoads
    : M::KGEN::impl::FoldGlobalConstLoadsBase<FoldGlobalConstLoads> {
  void runOnOperation() override;
};
} // namespace

/// All statically addressed loads from a GlobalConstant can be elided and
/// replaced with the constant directly. This improves code generation as
/// constants can be propagated but GlobalConstants can't.
void FoldGlobalConstLoads::runOnOperation() {
  FuncOp func = getOperation();

  SmallVector<Operation *> toDelete;

  func.walk([&](POP::GlobalConstantOp op) {
    auto arrayAttr = dyn_cast<POP::ArrayAttr>(op.getValueAttr());
    if (!arrayAttr)
      return;

    Location loc = op.getLoc();
    OpBuilder builder{op};

    // Replace the given load with the value from the array at index.
    auto replaceOpWithValueAt = [&](POP::LoadOp load, size_t index) {
      TypedAttr constVal = arrayAttr.getValues()[index];
      auto newVal = builder.create<ParamConstantOp>(loc, constVal);
      load.replaceAllUsesWith({newVal->getResult(0)});
      toDelete.push_back(load);
    };

    // Replace an indirect access op like offset or gep.
    auto replaceIndirectLoadsWithValueAt = [&](Operation *indirect,
                                               Value indexVal) {
      // Ensure the index is a constant >0
      APInt index;
      if (!matchPattern(indexVal, mlir::m_ConstantInt(&index)) ||
          index.isNegative())
        return;

      // Replace any indirectly used loads.
      for (Operation *user : indirect->getUsers()) {
        if (auto load = dyn_cast<POP::LoadOp>(user))
          replaceOpWithValueAt(load, index.getLimitedValue());
      }
    };

    // Walk the users and replace short chains leading to loads with the
    // constant being accessed directly.
    for (Operation *user : op->getUsers()) {
      // Fold geps + loads -> constant.
      if (auto gep = dyn_cast<POP::ArrayGEPOp>(user))
        replaceIndirectLoadsWithValueAt(gep, gep.getIndex());

      // Bitcast of array global -> pointer with optional offset loads
      if (auto bitcast = dyn_cast<POP::PointerBitcastOp>(user)) {
        // Expecting a pointer...
        auto ptr = dyn_cast<PointerType>(bitcast.getOutput().getType());
        if (!ptr)
          return;

        // We are expecting the element type to remain the same.
        if (ptr.getElementType() != arrayAttr.getType().getElementType())
          return;

        for (Operation *bcastUser : bitcast->getUsers()) {
          // Load is easy, implictly first element.
          if (auto load = dyn_cast<POP::LoadOp>(bcastUser))
            replaceOpWithValueAt(load, 0);

          // Otherwise we need to look through the offsets.
          if (auto offset = dyn_cast<POP::OffsetOp>(bcastUser))
            replaceIndirectLoadsWithValueAt(offset, offset.getIndex());
        }
      }
    }
  });

  numLoadsFolded = toDelete.size();
  for (Operation *op : toDelete)
    op->erase();
}
