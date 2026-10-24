//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Direct dialect conversion: a single-pass, eager alternative to MLIR's
// dialect conversion framework for lowerings whose patterns never fail after
// mutating IR, so no rollback log, value mapping table or legality solver is
// kept. Type changes across unconverted ops are bridged with
// builtin.unrealized_conversion_cast ops, which a later reconcile removes.
// Patterns derive from DirectOpConversionPattern<OpT> and live in an ordinary
// RewritePatternSet; a pattern must not modify IR before returning failure.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMPILER_DIRECTCONVERSION_H
#define SUPPORT_COMPILER_DIRECTCONVERSION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

namespace M {

/// The rewriter handed to direct conversion patterns. IR is mutated eagerly;
/// replaceOp inserts a bridge cast when a replacement value's type differs
/// from the replaced one.
class DirectRewriter : public mlir::PatternRewriter {
public:
  DirectRewriter(mlir::MLIRContext *ctx,
                 const mlir::TypeConverter *typeConverter)
      : mlir::PatternRewriter(ctx), typeConverter(typeConverter) {}

  /// Return `value` converted to its target type: `value` itself if the type
  /// converter leaves the type unchanged, the bridged original if `value` is
  /// a bridge cast result, or a cached unrealized_conversion_cast otherwise.
  /// Null if the converter rejects the type, as upstream's spelling does.
  mlir::Value remap(mlir::Value value);

  /// ConversionPatternRewriter-compatible spellings.
  mlir::Value getRemappedValue(mlir::Value value) { return remap(value); }
  mlir::LogicalResult
  getRemappedValues(mlir::ValueRange values,
                    llvm::SmallVectorImpl<mlir::Value> &result);

  void replaceOp(mlir::Operation *op, mlir::ValueRange newValues) override;
  void replaceOp(mlir::Operation *op, mlir::Operation *newOp) override;
  void eraseOp(mlir::Operation *op) override;
  void eraseBlock(mlir::Block *block) override;

  /// In-place block signature conversion: rewrites the block argument list
  /// per `conversion` and bridges uses of the old arguments. Returns the
  /// (unchanged) block.
  mlir::Block *applySignatureConversion(
      mlir::Block *block, mlir::TypeConverter::SignatureConversion &conversion);

  /// Convert the types of block arguments within the given region, using
  /// `converter` for non-entry blocks and `entryConversion` (if given) for the
  /// entry block.
  mlir::FailureOr<mlir::Block *> convertRegionTypes(
      mlir::Region *region, const mlir::TypeConverter &converter,
      mlir::TypeConverter::SignatureConversion *entryConversion = nullptr);

  const mlir::TypeConverter *getTypeConverter() const { return typeConverter; }

  /// Driver hook: called for every operation erased through this rewriter
  /// (including ops nested in erased ops/blocks).
  llvm::function_ref<void(mlir::Operation *)> onErase = nullptr;

private:
  const mlir::TypeConverter *typeConverter;
  llvm::DenseMap<mlir::Value, mlir::Value> castCache;
};

/// Base class for direct conversion patterns. Derives RewritePattern so the
/// patterns fit in a RewritePatternSet; only applyDirectConversion may invoke
/// them (it supplies the DirectRewriter the final override casts to).
class DirectConversionPattern : public mlir::RewritePattern {
public:
  DirectConversionPattern(const mlir::TypeConverter &typeConverter,
                          llvm::StringRef opName, mlir::MLIRContext *ctx,
                          mlir::PatternBenefit benefit = 1)
      : mlir::RewritePattern(opName, benefit, ctx),
        typeConverter(&typeConverter) {}

  const mlir::TypeConverter *getTypeConverter() const { return typeConverter; }

protected:
  const mlir::TypeConverter *typeConverter;
};

template <typename SourceOp>
class DirectOpConversionPattern : public DirectConversionPattern {
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  DirectOpConversionPattern(const mlir::TypeConverter &typeConverter,
                            mlir::MLIRContext *ctx,
                            mlir::PatternBenefit benefit = 1)
      : DirectConversionPattern(typeConverter, SourceOp::getOperationName(),
                                ctx, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const final {
    // Only the direct conversion driver invokes these patterns.
    auto &directRewriter = static_cast<DirectRewriter &>(rewriter);
    auto sourceOp = mlir::cast<SourceOp>(op);
    llvm::SmallVector<mlir::Value> remapped;
    remapped.reserve(op->getNumOperands());
    if (mlir::failed(
            directRewriter.getRemappedValues(op->getOperands(), remapped))) {
      return mlir::failure();
    }
    OpAdaptor adaptor(remapped, sourceOp);
    return matchAndRewrite(sourceOp, adaptor, directRewriter);
  }

  virtual mlir::LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  DirectRewriter &rewriter) const = 0;
};

/// Apply the given direct conversion patterns to every op nested under `root`,
/// in pre-order, so blocks must be listed in dominance order. Ops created
/// during conversion are visited too, so a pattern must not match its own
/// output. An op whose patterns all fail stays unconverted; afterwards the
/// first remaining op that `mustConvert` selects fails the conversion with an
/// error on that op. `skipOp` ops are neither converted nor required.
mlir::LogicalResult applyDirectConversion(
    mlir::Operation *root, const mlir::TypeConverter *typeConverter,
    const mlir::FrozenRewritePatternSet &patterns,
    llvm::function_ref<bool(mlir::Operation *)> skipOp = nullptr,
    llvm::function_ref<bool(mlir::Operation *)> mustConvert = nullptr);

} // namespace M

#endif // SUPPORT_COMPILER_DIRECTCONVERSION_H
