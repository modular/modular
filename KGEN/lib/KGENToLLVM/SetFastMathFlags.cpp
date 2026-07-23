//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/KGENPasses.h"

#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"

using namespace mlir;

namespace M::KGEN {
#define GEN_PASS_DEF_SETFASTMATHFLAGS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {

struct SetFastMathFlagsPass
    : public M::KGEN::impl::SetFastMathFlagsBase<SetFastMathFlagsPass> {
  using SetFastMathFlagsBase::SetFastMathFlagsBase;

  void runOnOperation() override {
    using M::KGEN::POP::AddOp;
    using M::KGEN::POP::FastmathFlags;
    using M::KGEN::POP::MulOp;
    using M::KGEN::POP::SubOp;

    // Only clearing `contract` changes anything.
    if (contract)
      return;

    auto strip = [](auto op) {
      op.setFastmathFlags(op.getFastmathFlags() & ~FastmathFlags::contract);
    };
    getOperation()->walk([&](Operation *op) {
      if (auto add = dyn_cast<AddOp>(op))
        strip(add);
      else if (auto sub = dyn_cast<SubOp>(op))
        strip(sub);
      else if (auto mul = dyn_cast<MulOp>(op))
        strip(mul);
    });
  }
};

} // namespace
