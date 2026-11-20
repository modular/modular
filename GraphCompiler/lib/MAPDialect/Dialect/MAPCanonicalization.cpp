//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MOGGDialect/Support/TypeTraits.h"
#include "Support/AssertStream.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace M::MAP;

namespace {

struct DedupeIterInputs : OpInterfaceRewritePattern<IterOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(IterOpInterface op,
                                PatternRewriter &rewriter) const override {
    return success(op.dedupeInputs(rewriter));
  }
};

// `DedupeIterInputs` for a capturing closure's captures.
// `LoopInvariantViewMotion` relies on it to fold duplicates from `addCapture`.
struct DedupeCapturingClosureCaptures
    : OpInterfaceRewritePattern<CapturingFusionClosure> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(CapturingFusionClosure op,
                                PatternRewriter &rewriter) const override {
    Block::BlockArgListType captureArgs = op.getCaptureBlockArgs();

    llvm::DenseMap<Value, BlockArgument> seen;
    llvm::SmallVector<unsigned> duplicates;
    for (auto [idx, capture] : llvm::enumerate(op.getCaptures()))
      if (!seen.insert({capture, captureArgs[idx]}).second)
        duplicates.push_back(idx);
    if (duplicates.empty())
      return failure();

    for (unsigned idx : duplicates)
      rewriter.replaceAllUsesWith(captureArgs[idx],
                                  seen.lookup(op.getCaptures()[idx]));
    rewriter.modifyOpInPlace(op.getOperation(), [&] {
      for (unsigned idx : llvm::reverse(duplicates))
        op.eraseCaptureAt(idx);
    });
    return success();
  }
};

// Drops captures whose shadowed block arg has no use. Unlike an opaque's
// inputs, a capture carries no scheduling edge, so this is always safe.
struct DropDeadCapturingClosureCaptures
    : OpInterfaceRewritePattern<CapturingFusionClosure> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(CapturingFusionClosure op,
                                PatternRewriter &rewriter) const override {
    Block::BlockArgListType captureArgs = op.getCaptureBlockArgs();
    llvm::SmallVector<unsigned> dead;
    for (auto [idx, arg] : llvm::enumerate(captureArgs))
      if (arg.use_empty())
        dead.push_back(idx);
    if (dead.empty())
      return failure();

    rewriter.modifyOpInPlace(op.getOperation(), [&] {
      for (unsigned idx : llvm::reverse(dead))
        op.eraseCaptureAt(idx);
    });
    return success();
  }
};

// The largest constant worth rematerializing inside a kernel body rather than
// passing at runtime.
constexpr int64_t kMaxConstantSize = 2048;

// The `mo.constant` producing `val` if it can be rematerialized in a kernel
// body, else null.
M::MO::ConstantOp getFusableConstant(Value val) {
  // A rebind only reinterprets the type.
  if (auto rebind = val.getDefiningOp<M::MO::RebindOp>())
    val = rebind.getInput();

  auto cst = val.getDefiningOp<M::MO::ConstantOp>();
  if (!cst || !M::MO::isOnHost(cst))
    return {};

  ElementsAttr elems = cst.getValueAttr();
  if (elems.getNumElements() > kMaxConstantSize)
    return {};
  if (!M::MOGG::TypeTraits::isSupportedMOGGType(elems.getElementType()))
    return {};
  if (!isa<M::ArrayElementsAttr, DenseResourceElementsAttr>(elems))
    return {};

  return cst;
}

// Rematerializes small constant inputs inside the (isolated) body and drops
// them from `inputs`, the lowered kernel's parameter list. Duplicate clones are
// uniqued by the folder since `mo.constant` is `ConstantLike`.
struct FuseConstantIterOpaqueInputs : OpRewritePattern<IterOpaqueOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IterOpaqueOp op,
                                PatternRewriter &rewriter) const override {
    // Folding into a device kernel miscompiles it into illegal memory accesses
    // (GEX-3839).
    if (!M::MO::isOnHost(op.getOperation()))
      return failure();

    rewriter.setInsertionPointToStart(&op.getBody().front());
    llvm::SmallVector<unsigned> folded;
    for (auto [idx, input] : llvm::enumerate(op.getInputs())) {
      M::MO::ConstantOp cst = getFusableConstant(input);
      if (!cst)
        continue;
      // The body holds only MAP ops. Typed against the input, since a
      // looked-through rebind retypes it.
      Value inBody = ConstantOp::create(rewriter, cst.getLoc(), input.getType(),
                                        cst.getValueAttr());
      rewriter.replaceAllUsesWith(op.getInputBlockArgs()[idx], inBody);
      folded.push_back(idx);
    }
    if (folded.empty())
      return failure();

    rewriter.modifyOpInPlace(op, [&] {
      for (unsigned idx : llvm::reverse(folded))
        op.eraseInputsAt(idx);
    });
    return success();
  }
};

// Drops inputs whose shadowed block arg has no use, except chains: a chain
// input sequences the kernel against other memory ops at the graph level even
// when the body never consumes it, so dropping it would change scheduling.
struct DropDeadIterOpaqueInputs : OpRewritePattern<IterOpaqueOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IterOpaqueOp op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<unsigned> dead;
    for (auto [idx, arg] : llvm::enumerate(op.getInputBlockArgs()))
      if (arg.use_empty() && !llvm::isa<M::MO::ChainType>(arg.getType()))
        dead.push_back(idx);
    if (dead.empty())
      return failure();

    rewriter.modifyOpInPlace(op, [&] {
      for (unsigned idx : llvm::reverse(dead))
        op.eraseInputsAt(idx);
    });
    return success();
  }
};

// Walks `tensor` out to the MO-level value the kernel reads: through closure
// captures, stride views, and the enclosing `map.iter.opaque`'s inputs. Null
// for an output block argument or a value computed inside the kernel.
static Value resolveMOSource(Value tensor) {
  while (true) {
    if (auto arg = dyn_cast<BlockArgument>(tensor)) {
      Operation *parent = arg.getOwner()->getParentOp();
      if (auto closure = dyn_cast_or_null<CapturingFusionClosure>(parent)) {
        tensor = closure.getCaptureFor(arg);
        if (!tensor)
          return nullptr;
        continue;
      }
      if (auto opaque = dyn_cast_or_null<IterOpaqueOp>(parent)) {
        tensor = opaque.getInputFor(arg);
        if (!tensor)
          return nullptr;
        continue;
      }
      return nullptr;
    }
    Operation *def = tensor.getDefiningOp();
    if (auto view = dyn_cast_or_null<StrideViewOpInterface>(def)) {
      tensor = view.getViewedTensor();
      continue;
    }
    return tensor;
  }
}

// Reads the single element of `elems` as a float or integer scalar attribute,
// the only kinds with a Mojo literal to emit.
static TypedAttr getSingleElementAttr(mlir::ElementsAttr elems, M::DType dtype,
                                      Type elemType) {
  if (isa<mlir::FloatType>(elemType)) {
    auto blob = M::MO::tryGetConstantBlob<double>(elems, dtype);
    if (!blob || blob->size() != 1)
      return nullptr;
    return mlir::FloatAttr::get(elemType, blob->front());
  }
  if (isa<mlir::IntegerType>(elemType)) {
    auto blob = M::MO::tryGetConstantBlob<int64_t>(elems, dtype);
    if (!blob || blob->size() != 1)
      return nullptr;
    return mlir::IntegerAttr::get(elemType, blob->front());
  }
  return nullptr;
}

// The constant elements `source` holds, or null. `m_Constant` covers both a
// host `map.constant` folded by `FuseConstantIterOpaqueInputs` and a device
// `mo.constant` operand it refused (GEX-3839).
static mlir::ElementsAttr getConstantElements(Value source) {
  mlir::ElementsAttr elems;
  if (matchPattern(source, m_Constant(&elems)))
    return elems;

  auto srcTensor = dyn_cast<M::MO::TensorValue>(source);
  if (!srcTensor)
    return nullptr;

  // A host constant moved to the device (`ops.constant(...).to(device)`)
  // arrives through `mo.transfer`, which `tryGetConstantOpsOfRank` does not see
  // through. Placement is irrelevant when inlining a literal.
  size_t rank = srcTensor.getType().getRank();
  while (auto transfer = srcTensor.getDefiningOp<M::MO::TransferOp>())
    srcTensor = cast<M::MO::TensorValue>(transfer.getInput());

  // Handles the constant / rebind / concat / reshape chains; more than one
  // constant means a concatenation, which cannot be a single element.
  SmallVector<M::MO::ConstantOp> constants =
      M::MO::tryGetConstantOpsOfRank(srcTensor, rank);
  if (constants.size() != 1)
    return nullptr;
  return constants.front().getValueAttr();
}

// Replaces a load of a one-element compile-time constant with its element as
// a literal. A one-element source makes the index irrelevant, whatever view
// sits on the index or tensor.
//
// Rooted on the load so it never mutates across the isolated-from-above
// boundary; the dead capture and input are shed by the `DropDead*` patterns.
struct InlineConstantScalarLoad : OpRewritePattern<LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LoadOp op,
                                PatternRewriter &rewriter) const override {
    Value source = resolveMOSource(op.getTensor());
    if (!source)
      return failure();
    auto srcTy = dyn_cast<M::MO::TensorType>(source.getType());
    if (!srcTy)
      return failure();

    // Also excludes dynamic shapes, whose `getNumElements()` is nullopt.
    if (srcTy.getNumElements() != 1)
      return failure();

    mlir::ElementsAttr elems = getConstantElements(source);
    if (!elems)
      return failure();

    Type elemType = op.getResult().getType();
    TypedAttr scalar = getSingleElementAttr(elems, srcTy.getDtype(), elemType);
    if (!scalar)
      return failure();

    rewriter.replaceOpWithNewOp<ConstantScalarOp>(op, elemType, scalar);
    return success();
  }
};

} // namespace

// Fusion can bind the same value several times in `inputs`, which becomes the
// lowered kernel's parameter list. Uniquing also lets CSE collapse redundant
// loads and re-exposes a shared producer as single-use.
bool IterOpInterface::dedupeInputs(RewriterBase &rewriter) {
  Block::BlockArgListType inputArgs = getInputBlockArgs();
  llvm::DenseMap<Value, BlockArgument> seen;
  llvm::SmallVector<unsigned> duplicates;
  for (auto [idx, input] : llvm::enumerate(getInputs()))
    if (!seen.insert({input, inputArgs[idx]}).second)
      duplicates.push_back(idx);
  if (duplicates.empty())
    return false;

  for (unsigned idx : duplicates)
    rewriter.replaceAllUsesWith(inputArgs[idx], seen.lookup(getInputs()[idx]));
  rewriter.modifyOpInPlace(getOperation(), [&] {
    for (unsigned idx : llvm::reverse(duplicates))
      eraseInputsAt(idx, 1);
  });
  return true;
}

// Applies to every op implementing `IterOpInterface`.
void MAPDialect::getCanonicalizationPatterns(RewritePatternSet &results) const {
  results.add<DedupeIterInputs>(results.getContext());
}

void IterOpaqueOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<DropDeadIterOpaqueInputs, FuseConstantIterOpaqueInputs>(context);
}

void LoadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<InlineConstantScalarLoad>(context);
}

void IterElementwiseBodyOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<DedupeCapturingClosureCaptures, DropDeadCapturingClosureCaptures>(
      context);
}

void IterPrologueOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<DedupeCapturingClosureCaptures, DropDeadCapturingClosureCaptures>(
      context);
}

void IterEpilogueOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<DedupeCapturingClosureCaptures, DropDeadCapturingClosureCaptures>(
      context);
}
