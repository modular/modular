//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/Compiler/MLIRDenseAttr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"

using namespace M;

/// Utility function to insert a new asm blob into the context.
static DenseResourceElementsAttr insertAsmBlob(MLIRContext *ctx,
                                               RankedTensorType attrType,
                                               const Twine &name,
                                               mlir::AsmResourceBlob blob) {
  using HandleTy = mlir::DialectResourceBlobHandle<mlir::BuiltinDialect>;
  auto resourceManager =
      mlir::DenseResourceElementsHandle::getManagerInterface(ctx);
  auto *dialect = cast<mlir::BuiltinDialect>(resourceManager.getDialect());
  return DenseResourceElementsAttr::get(
      attrType, resourceManager.getBlobManager().insert<HandleTy>(
                    dialect, name.str(), std::move(blob)));
}

DenseResourceElementsAttr M::createResourceAttr(MLIRContext *ctx,
                                                ArrayRef<char> data,
                                                const Twine &name) {
  // Pretend this is a "tensor" of data.
  auto attrType = RankedTensorType::get(
      {(int64_t)data.size()}, IntegerType::get(ctx, 8, IntegerType::Unsigned));
  auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyWithAlign(
      ArrayRef<char>(data.begin(), data.size()),
      /*align=*/8);
  return insertAsmBlob(ctx, attrType, name, std::move(blob));
}

DenseResourceElementsAttr
M::createResourceAttr(MLIRContext *ctx, BufferRef data, const Twine &name) {
  // Pretend this is a "tensor" of data.
  auto attrType =
      RankedTensorType::get({(int64_t)data->getBufferSize()},
                            IntegerType::get(ctx, 8, IntegerType::Unsigned));

  // Build an unmanaged blob to represent the data, using a deleter that holds
  // on to the reference (to avoid copying).
  StringRef buffer = data->getBuffer();
  auto blob = mlir::UnmanagedAsmResourceBlob::allocateWithAlign(
      ArrayRef<char>(buffer.data(), buffer.size()), /*align=*/8,
      [data = std::move(data)](void *, unsigned, unsigned) {});
  return insertAsmBlob(ctx, attrType, name, std::move(blob));
}
