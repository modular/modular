//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// KGEN List Lowering
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "llvm/ADT/TypeSwitch.h"
using namespace M;
using namespace KGEN;

// After all the types and attrs have been bulk replaced, replace this op.
static void expandListOp(POP::ListGetOp getOp) {
  // We know the operand will now have array type, which isn't what the
  // accessors expect, so we use the untyped accessors.
  OpBuilder b(getOp);
  auto arrayGetOp = b.create<POP::StructExtractOp>(
      getOp.getLoc(), getOp.getType(), getOp->getOperand(0),
      cast<IntegerAttr>(getOp.getIndex()));
  getOp.replaceAllUsesWith(arrayGetOp.getResult());
  getOp->erase();
}

// After all the types and attrs have been bulk replaced, replace this op.
static void expandListOp(POP::ListCreateOp createOp) {
  OpBuilder b(createOp);
  // We know the result will now have array type, which isn't what the
  // accessors expect, so we use the untyped accessors.
  auto arrayCreateOp = b.create<POP::StructCreateOp>(
      createOp.getLoc(), createOp->getResult(0).getType(),
      createOp.getOperands());
  createOp.replaceAllUsesWith(arrayCreateOp.getResult());
  createOp->erase();
}

static void expandListOp(POP::SIMDShuffleOp shuffleOp) {
  // Shuffles don't actually want their mask lowered.  They are the only
  // operation like this.
  auto maskAttr = shuffleOp->getAttrOfType<POP::StructAttr>("mask");
  assert(maskAttr && "mask should have been lowered");

  // Get the ListType/ListAttr back.
  auto indexType = IndexType::get(shuffleOp.getContext());
  auto listType = KGEN::ListType::get(indexType, maskAttr.getValues().size());
  auto listAttr = KGEN::ListAttr::get(maskAttr.getValues(), listType);
  shuffleOp->setAttr("mask", listAttr);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERKGENTOPOP
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerKGENToPOPPass
    : public M::KGEN::impl::LowerKGENToPOPBase<LowerKGENToPOPPass> {
  using LowerKGENToPOPBase::LowerKGENToPOPBase;

  void runOnOperation() override;
};
} // namespace

void LowerKGENToPOPPass::runOnOperation() {
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([](KGEN::ListType list) -> Type {
    SmallVector<TypedAttr> elts(*list.getResolvedLength(),
                                list.getElementType());
    return POP::StructType::get(list.getContext(), elts);
  });
  replacer.addReplacement([&](KGEN::ListAttr list) -> Attribute {
    SmallVector<TypedAttr> newElts;
    newElts.reserve(list.getValues().size());
    for (auto elt : list.getValues())
      newElts.push_back(replacer.replace(elt));

    auto newType = cast<POP::StructType>(replacer.replace(list.getType()));
    return POP::StructAttr::get(newElts, newType);
  });

  // Replace all list types + attrs with corresponding array types/attrs.
  replacer.recursivelyReplaceElementsIn(getOperation(),
                                        /*replaceAttrs=*/true,
                                        /*replaceLocs=*/true,
                                        /*replaceTypes=*/true);

  // Rewrite list ops to array ops.
  getOperation()->walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<POP::ListGetOp, POP::ListCreateOp, POP::SIMDShuffleOp>(
            [&](auto op) { expandListOp(op); });
  });
}
