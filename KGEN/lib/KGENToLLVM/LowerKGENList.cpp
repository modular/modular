//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// KGEN List -> Array Lowering.
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
  auto arrayGetOp = b.create<POP::ArrayGetOp>(
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
  auto arrayCreateOp = b.create<POP::ArrayCreateOp>(
      createOp.getLoc(), createOp->getResult(0).getType(),
      createOp.getOperands());
  createOp.replaceAllUsesWith(arrayCreateOp.getResult());
  createOp->erase();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERKGENLIST
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerKGENListPass
    : public M::KGEN::impl::LowerKGENListBase<LowerKGENListPass> {
  using LowerKGENListBase::LowerKGENListBase;

  void runOnOperation() override;
};
} // namespace

void LowerKGENListPass::runOnOperation() {
  mlir::AttrTypeReplacer replacer;
  // Lower list types into array types of the same length.
  replacer.addReplacement([](KGEN::ListType list) -> Type {
    return POP::ArrayType::get(list.getLength(), list.getElementType());
  });
  replacer.addReplacement([&](KGEN::ListAttr list) -> Attribute {
    SmallVector<TypedAttr> newElts;
    newElts.reserve(list.getValues().size());
    for (auto elt : list.getValues())
      newElts.push_back(replacer.replace(elt));

    auto newType = cast<POP::ArrayType>(replacer.replace(list.getType()));
    return POP::ArrayAttr::get(newElts, newType);
  });

  // Replace all list types + attrs with corresponding array types/attrs.
  replacer.recursivelyReplaceElementsIn(getOperation(),
                                        /*replaceAttrs=*/true,
                                        /*replaceLocs=*/true,
                                        /*replaceTypes=*/true);

  // Rewrite list ops to array ops.
  getOperation()->walk([&](Operation *op) {
    TypeSwitch<Operation *>(op).Case<POP::ListGetOp, POP::ListCreateOp>(
        [&](auto op) { expandListOp(op); });
  });
}
