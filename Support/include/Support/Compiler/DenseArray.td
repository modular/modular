//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef DENSEARRAY_TD
#define DENSEARRAY_TD

include "mlir/IR/OpBase.td"

def DenseIntArrayAttr : Attr<CPred<"$_self.isa<DenseIntArrayAttr>()">,
    "Dense integer array"> {
  let storageType = "DenseIntArrayAttr";
  let returnType = "DenseIntArrayAttr";
}

def DenseSI32ArrayAttr : ConfinedAttr<DenseIntArrayAttr, [AttrConstraint<CPred<
    "$_self.cast<DenseIntArrayAttr>().getElementType().isSignedInteger(32)"
>>]> {
  let constBuilderCall = "$_builder.getAttr<DenseIntArrayAttr>($0, IntegerType::Signed)";
  let returnType = "ArrayRef<int32_t>";
  let convertFromStorage = "$_self.asArrayRef<int32_t>()";
}

#endif // DENSEARRAY_TD
