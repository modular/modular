//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Compiler/MLIRDenseAttr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"

using namespace M;

DenseResourceElementsAttr M::createResourceAttr(MLIRContext *ctx,
                                                ArrayRef<char> data,
                                                const Twine &name) {
  auto resourceManager =
      mlir::DenseResourceElementsHandle::getManagerInterface(ctx);

  // Pretend this is a "tensor" of data.
  auto attrType = RankedTensorType::get(
      {(int64_t)data.size()}, IntegerType::get(ctx, 8, IntegerType::Unsigned));
  auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyWithAlign(
      ArrayRef<char>(data.begin(), data.size()),
      /*align=*/8);

  // Some convenience typedefs to simplify this code a little bit.
  using HandleTy = mlir::DialectResourceBlobHandle<mlir::BuiltinDialect>;
  auto *dialect = cast<mlir::BuiltinDialect>(resourceManager.getDialect());
  return DenseResourceElementsAttr::get(
      attrType, resourceManager.getBlobManager().insert<HandleTy>(
                    dialect, name.str(), std::move(blob)));
}
