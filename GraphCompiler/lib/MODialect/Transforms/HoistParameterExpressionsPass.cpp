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

#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "GraphCompiler/MODialect/Transforms/ShapeInferencePassesCommon.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace M;
using namespace MO;
using namespace M::MO::ShapeInferencePassCommon;

namespace M::MO {

#define GEN_PASS_DEF_HOISTPARAMETEREXPRESSIONS

#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

} // namespace M::MO

namespace {

struct RewriteParamExprTypes : public RewritePattern {
  RewriteParamExprTypes(
      MLIRContext *context,
      llvm::DenseMap<Region *,
                     llvm::DenseMap<KGEN::ParamOperatorAttr, std::string>>
          &&regionToExprToParam)
      : RewritePattern(mlir::Pattern::MatchAnyOpTypeTag(), 1, context),
        regionToExprToParam(regionToExprToParam) {}
  mutable llvm::DenseMap<Region *,
                         llvm::DenseMap<KGEN::ParamOperatorAttr, std::string>>
      regionToExprToParam;

  /// Declares a new parameter for each parameter expression in `ty`'s shape
  /// not yet in `regionToExprToParam`, and records it there. Returns true if
  /// `ty` contains any parameter expression (unknown `?` dims are ignored).
  bool collectComplexParams(DeviceAwareRewriter &rewriter, Operation *op,
                            Type ty) const {
    // Bundle element shapes can carry param-expression dims too.
    if (auto bundleTy = dyn_cast<MO::BundleType>(ty)) {
      bool found = false;
      for (Type elemTy : bundleTy.getElementTypes())
        if (collectComplexParams(rewriter, op, elemTy))
          found = true;
      return found;
    }

    auto graphOp = op->getParentOfType<GraphOp>();

    bool found = false;
    auto tensorTy = dyn_cast<MO::TensorType>(ty);
    if (!tensorTy)
      return false;
    auto shapeAttr = tensorTy.getShape();
    for (TypedAttr dimValue : shapeAttr.getValues()) {
      auto paramOp = dyn_cast<KGEN::ParamOperatorAttr>(dimValue);
      if (!paramOp)
        continue;
      found = true;
      Region *parentRegion = op->getParentRegion();
      auto &map = regionToExprToParam[parentRegion];
      if (auto [it, inserted] = map.insert({paramOp, {}}); inserted) {
        std::string exprName = graphOp.getNewParamName("param_expr_");
        it->second = exprName;
        auto dimType = ShapeAttr::getDimType(op->getContext());
        rewriter.create<KGEN::ParamDeclareOp>(
            op->getLoc(), KGEN::ParamDeclAttr::get(exprName, dimType), paramOp);
      }
    }
    return found;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &origRewriter) const override {
    if (isa<KGEN::ParamDeclareOp>(op))
      return origRewriter.notifyMatchFailure(op, "skip parameter declarations");
    // Skip control-flow ops whose types are pinned by the graph signature or
    // whose regions carry non-tensor block arguments. Exceptions:
    //  - mo.call result types can be freely rewritten (consumers pick them up
    //    via the output rebinds).
    //  - mo.if has no region arguments and its results aren't the signature,
    //    so it is aliased like any op. Otherwise an un-aliased `ParamExpr`
    //    reaches MOGG lowering and asserts in `getTensorDynamicDimensions`.
    // TODO(GEX-3580): Revisit whether mo.call should implement
    // MOControlOpInterface at all; dropping it would remove that exception.
    if (isa<MO::MOControlOpInterface>(op) && !isa<MO::CallOp, MO::IfOp>(op))
      return origRewriter.notifyMatchFailure(op, "skip control flow");

    // mo.parallel bodies must stay kernel-only (no rebinds). The parallel op
    // is rewritten as a whole, and `recursivelyReplaceElementsIn` updates its
    // body types.
    if (op->getParentOfType<MO::ParallelOp>())
      return origRewriter.notifyMatchFailure(op, "skip parallel body ops");

    DeviceAwareRewriter rewriter(op, origRewriter);

    llvm::DenseSet<Value> rewriteInputs;
    bool containsComplexExprs = false;
    for (auto [operand, ty] :
         llvm::zip(op->getOperands(), op->getOperandTypes())) {
      bool foundComplex = collectComplexParams(rewriter, op, ty);
      // Bundle operands get no input rebind (`MO::RebindOp` is tensor-only);
      // `AttrTypeReplacer` rewrites their element types in place.
      if (foundComplex && isa<MO::TensorType>(ty)) {
        containsComplexExprs = true;
        rewriteInputs.insert(operand);
      }
    }
    for (Type resType : op->getResultTypes())
      if (collectComplexParams(rewriter, op, resType))
        containsComplexExprs = true;

    // Also scan region block-arg and nested result types, for e.g. a
    // `mo.parallel` whose outer types are alias-clean but whose body still
    // references the un-aliased form.
    op->walk([&](Block *block) {
      for (BlockArgument arg : block->getArguments())
        if (collectComplexParams(rewriter, op, arg.getType()))
          containsComplexExprs = true;
    });
    op->walk([&](Operation *nested) {
      if (nested == op)
        return;
      for (Type ty : nested->getResultTypes())
        if (collectComplexParams(rewriter, op, ty))
          containsComplexExprs = true;
    });

    if (!containsComplexExprs)
      return rewriter.notifyMatchFailure(op, "no complex expressions");

    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([&](KGEN::ParamOperatorAttr paramOp) -> Attribute {
      Region *parentRegion = op->getParentRegion();
      if (auto it = regionToExprToParam[parentRegion].find(paramOp);
          it != regionToExprToParam[parentRegion].end()) {
        auto dimType = ShapeAttr::getDimType(op->getContext());
        return KGEN::ParamDeclRefAttr::get(it->second, dimType);
      }
      return paramOp;
    });

    llvm::SmallVector<Type> resTypes(op->getResultTypes());

    rewriter.setInsertionPoint(op);

    llvm::DenseMap<Value, Value> inputRemap;
    for (Value inp : rewriteInputs) {
      inputRemap[inp] = rewriter.create<MO::RebindOp>(
          op->getLoc(), replacer.replace(inp.getType()), inp);
    }

    replacer.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/true,
                                          /*replaceTypes=*/true);

    // Rebind outputs to original types.
    rewriter.setInsertionPointAfter(op);
    for (auto [res, ty] : llvm::zip(op->getResults(), resTypes)) {
      if (isa<MO::TensorType>(ty)) {
        auto rebind = rewriter.create<MO::RebindOp>(op->getLoc(), ty, res);
        rewriter.replaceAllUsesExcept(res, rebind, rebind);
      }
    }

    // `mo.yield` operand types must equal the rewritten `mo.if` result types,
    // but `recursivelyReplaceElementsIn` only rewrites values produced inside
    // the regions. Rebind yields of outer values (or differently canonicalized
    // expressions) to the result type.
    if (auto ifOp = dyn_cast<MO::IfOp>(op)) {
      TypeRange resultTys = ifOp.getResultTypes();
      for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
        auto yieldOp = cast<MO::YieldOp>(region->front().getTerminator());
        rewriter.setInsertionPoint(yieldOp);
        llvm::SmallVector<Value> newYields(yieldOp.getOperands().begin(),
                                           yieldOp.getOperands().end());
        bool changed = false;
        for (auto [i, ty] : llvm::enumerate(resultTys)) {
          if (isa<MO::TensorType>(ty) && newYields[i].getType() != ty) {
            newYields[i] = rewriter.create<MO::RebindOp>(yieldOp.getLoc(), ty,
                                                         newYields[i]);
            changed = true;
          }
        }
        if (changed)
          rewriter.modifyOpInPlace(yieldOp,
                                   [&] { yieldOp->setOperands(newYields); });
      }
    }

    llvm::SmallVector<Value> inputs;
    for (Value inp : op->getOperands()) {
      if (auto it = inputRemap.find(inp); it != inputRemap.end())
        inputs.push_back(it->second);
      else
        inputs.push_back(inp);
    }
    rewriter.modifyOpInPlace(op, [&] { op->setOperands(inputs); });

    return LogicalResult::success();
  }
};

struct HoistParameterExpressionsPass
    : public M::MO::impl::HoistParameterExpressionsBase<
          HoistParameterExpressionsPass> {
  using HoistParameterExpressionsBase::HoistParameterExpressionsBase;

  /// Maps each `<expr>` in `kgen.param.declare name = <expr>` to `name`, per
  /// region.
  static llvm::DenseMap<Region *,
                        llvm::DenseMap<KGEN::ParamOperatorAttr, std::string>>
  collectExistingExprToParamMap(GraphOp graphOp) {
    llvm::DenseMap<Region *,
                   llvm::DenseMap<KGEN::ParamOperatorAttr, std::string>>
        regionToExprToParam;
    graphOp.walk([&](KGEN::ParamDeclareOp paramDeclOp) {
      Region *parentRegion = paramDeclOp->getParentRegion();
      auto &map = regionToExprToParam[parentRegion];
      if (auto paramExpr =
              dyn_cast<KGEN::ParamOperatorAttr>(paramDeclOp.getValue())) {
        map.insert({paramExpr, paramDeclOp.getName().str()});
      }
    });
    return regionToExprToParam;
  }

  void runOnOperation() override {
    // Replaces each KGEN::ParamOperatorAttr in tensor types with a reference to
    // a new alias parameter declared to it, rebinding inputs and outputs.
    auto *ctx = &getContext();

    RewritePatternSet patterns{ctx};
    patterns.add<RewriteParamExprTypes>(
        ctx, collectExistingExprToParamMap(getOperation()));
    patterns.add<FoldConsecutiveRebinds>(ctx);
    // Folds the identity rebinds introduced here without a separate
    // canonicalization pass.
    RebindOp::getCanonicalizationPatterns(patterns, ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
