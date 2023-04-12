//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_LIFTANDFOLDAPPLY
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

static void liftAndFoldApplys(Location loc, Region *body) {
  DenseMap<ParamOperatorAttr, Attribute> lifted;
  mlir::AttrTypeReplacer replacer;

  ImplicitLocOpBuilder b(loc, OpBuilder(loc.getContext()));
  unsigned counter = 0;

  // Lift 'apply' operators into `kgen.param.apply` operations.
  replacer.addReplacement([&](ParamOperatorAttr op) -> Attribute {
    if (op.getOpcode() != POC::Apply)
      return op;
    if (auto it = lifted.find(op); it != lifted.end())
      return it->second;
    // Generate a name for the lifted parameter.
    auto decl = ParamDeclAttr::get(
        b.getStringAttr("(lifted)apply_" + Twine(counter++)), op.getType());

    // Explicitly recurse on all operands to lift nested 'apply' operators.
    TypedAttr callee = replacer.replace(op.getOperands().front());
    SmallVector<TypedAttr> operands;
    for (TypedAttr operand : op.getOperands().drop_front())
      operands.push_back(replacer.replace(operand));
    b.create<ParamApplyOp>(decl, callee, operands);
    return lifted.try_emplace(op, ParamDeclRefAttr::get(decl)).first->second;
  });

  body->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Walk over nested parameter scopes, since lifted apply operators with name
    // shadowing can cause collisions.
    if (isa<DeclInterface>(op)) {
      for (Region &region : op->getRegions())
        liftAndFoldApplys(op->getLoc(), &region);
      return WalkResult::skip();
    }

    // Insert the apply operations as close to the original location of the
    // 'apply' operator as possible.
    b.setLoc(op->getLoc());
    b.setInsertionPoint(op);
    replacer.replaceElementsIn(op, /*replaceAttrs=*/true, /*replaceLocs=*/true,
                               /*replaceTypes=*/true);
    return WalkResult::advance();
  });
}

namespace {
struct LiftAndFoldApplyPass : impl::LiftAndFoldApplyBase<LiftAndFoldApplyPass> {
  using LiftAndFoldApplyBase::LiftAndFoldApplyBase;

  void runOnOperation() override {
    for (auto func : getOperation().getOps<GeneratorOp>())
      liftAndFoldApplys(func.getLoc(), &func.getBodyRegion());
  }
};
} // namespace
