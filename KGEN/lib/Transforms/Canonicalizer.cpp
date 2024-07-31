//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ToolCommon/KGENPasses.h"

#include "KGEN/CustomDialect/CustomDialect.h"
#include "KGEN/CustomDialect/CustomUtils.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/Compiler/BytecodeReaderWriter.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Rewrite.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Return true if the region's body is empty (only contains a terminator).
static bool isEmpty(Region &region) {
  assert(llvm::hasSingleElement(region));
  return llvm::hasSingleElement(region.front());
}

//===----------------------------------------------------------------------===//
// Canonicalization Patterns
//===----------------------------------------------------------------------===//

namespace {

/// Canonicalize ifs with no bodies an N results to N selects. This also removes
/// trivially dead ifs.
struct IfToSelect : public OpRewritePattern<HLCF::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(HLCF::IfOp op,
                                PatternRewriter &b) const override {
    auto thenYield = dyn_cast<HLCF::YieldOp>(op.getThenTerminator());
    auto elseYield = dyn_cast<HLCF::YieldOp>(op.getElseTerminator());
    if (!isEmpty(op.getThenRegion()) || !isEmpty(op.getElseRegion()) ||
        !thenYield || !elseYield)
      return b.notifyMatchFailure(op.getLoc(),
                                  "bodies aren't empty with 'yield'");

    // Replace each result with a 'select' of the yield operands.
    SmallVector<Value> replacements;
    for (auto [i, result] : llvm::enumerate(op.getResults())) {
      replacements.push_back(b.create<POP::SelectOp>(op.getLoc(), op.getCond(),
                                                     thenYield.getOperand(i),
                                                     elseYield.getOperand(i)));
    }

    b.replaceOp(op, replacements);
    return success();
  }
};

struct IfYieldSelect : public OpRewritePattern<HLCF::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(HLCF::IfOp op,
                                PatternRewriter &b) const override {
    auto thenYield = dyn_cast<HLCF::YieldOp>(op.getThenTerminator());
    auto elseYield = dyn_cast<HLCF::YieldOp>(op.getElseTerminator());
    // Constructing dominance info is cheap because we have single-block
    // regions.
    mlir::DominanceInfo domInfo;
    auto dominatesIf = [&](Value value) {
      // Block arguments always dominate the IfOp, because it itself can never
      // have block arguments. Otherwise, check dominance of the defining
      // operation.
      return isa<BlockArgument>(value) ||
             domInfo.properlyDominates(value.getDefiningOp(), op);
    };

    // If only one branch ends in a yield, then we can replace dominating
    // results entirely with that yield.
    if (!thenYield != !elseYield) {
      bool anyChanged = false;
      if (!thenYield)
        thenYield = elseYield;
      for (auto [result, operand] :
           llvm::zip(op.getResults(), thenYield.getOperands())) {
        if (dominatesIf(operand)) {
          b.replaceAllUsesWith(result, operand);
          anyChanged = true;
        }
      }
      return success(anyChanged);
    }

    // The end of the IfOp is unreachable.
    if (!thenYield)
      return failure();

    // Both branches end in a yield. We can hoist each into a select.
    bool anyChanged = false;
    for (auto [result, trueVal, falseVal] :
         llvm::zip(op.getResults(), thenYield.getOperands(),
                   elseYield.getOperands())) {
      if (dominatesIf(trueVal) && dominatesIf(falseVal)) {
        Value select = b.create<POP::SelectOp>(op.getLoc(), op.getCond(),
                                               trueVal, falseVal);
        b.replaceAllUsesWith(result, select);
        anyChanged = true;
      }
    }
    return success(anyChanged);
  }
};

/// Canonicalize `!pop.scalar<index>` computations to `index` operations.
class IndexifyComparison : public OpRewritePattern<POP::CastToBuiltinOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(POP::CastToBuiltinOp op,
                                PatternRewriter &b) const override {
    if (op.getInput().getType().getResolvedDType() != KGENDType::kBool)
      return b.notifyMatchFailure(op.getLoc(), "not bool dtype");

    auto cmp = op.getInput().getDefiningOp<POP::CmpOp>();
    if (!cmp || !cmp->hasOneUse())
      return b.notifyMatchFailure(op.getLoc(),
                                  "input isn't single-use comparison");

    auto cast = cmp.getLhs().getDefiningOp<POP::CastFromBuiltinOp>();
    if (!cast || !cast->hasOneUse() ||
        cast.getResult().getType().getResolvedDType() != KGENDType::index)
      return b.notifyMatchFailure(op.getLoc(),
                                  "LHS isn't single-use cast to index dtype");

    POP::SIMDAttr rhs;
    if (!mlir::matchPattern(cmp.getRhs(), mlir::m_Constant(&rhs)))
      return b.notifyMatchFailure(op.getLoc(), "RHS isn't constant");

    auto cst = b.create<mlir::index::ConstantOp>(
        op.getLoc(), rhs.getValues().front().getIndexVal());
    b.replaceOpWithNewOp<mlir::index::CmpOp>(
        op, getIndexCmpPredicate(cmp.getPred()), cast.getInput(), cst);
    b.eraseOp(cmp);
    b.eraseOp(cast);
    return success();
  }

private:
  /// Get the equivalent index comparison predicate. POP treats the `index`
  /// dtype as signed.
  static mlir::index::IndexCmpPredicate
  getIndexCmpPredicate(POP::CmpPredicate pred) {
    switch (pred) {
    case POP::CmpPredicate::EQ:
      return mlir::index::IndexCmpPredicate::EQ;
    case POP::CmpPredicate::NE:
      return mlir::index::IndexCmpPredicate::NE;
    case POP::CmpPredicate::LT:
      return mlir::index::IndexCmpPredicate::SLT;
    case POP::CmpPredicate::GT:
      return mlir::index::IndexCmpPredicate::SGT;
    case POP::CmpPredicate::LE:
      return mlir::index::IndexCmpPredicate::SLE;
    case POP::CmpPredicate::GE:
      return mlir::index::IndexCmpPredicate::SGE;
    }
    llvm_unreachable("invalid cmp predicate");
  }
};

/// Replace:
///
/// ```mlir
/// %0 = index.cmp <pred>(%a, %b)
/// %1 = cast_from_builtin %0 : i1 to scalar<bool>
/// %2 = xor %1, %simd_bool_0
/// %3 = cast_to_builtin %2 : scalar<bool> to i1
/// ```
///
/// With:
///
/// ```mlir
/// %3 = index.cmp <not pred>(%a, %b)
/// ```
struct InvertComparison : OpRewritePattern<POP::CastToBuiltinOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(POP::CastToBuiltinOp op,
                                PatternRewriter &b) const override {
    if (op.getInput().getType().getResolvedDType() != KGENDType::kBool)
      return b.notifyMatchFailure(op.getLoc(), "not bool dtype");

    auto notOp = op.getInput().getDefiningOp<POP::XOrOp>();
    if (!notOp)
      return b.notifyMatchFailure(op.getLoc(), "parent isn't xor");

    POP::SIMDAttr zeroAttr;
    if (!mlir::matchPattern(notOp.getRhs(), mlir::m_Constant(&zeroAttr)) ||
        zeroAttr.getValues().front().getBoolVal() != true)
      return b.notifyMatchFailure(notOp.getLoc(), "not xor with true");

    auto inCast = notOp.getLhs().getDefiningOp<POP::CastFromBuiltinOp>();
    if (!inCast)
      return b.notifyMatchFailure(notOp.getLoc(), "lhs parent isn't cast");

    auto cmpOp = inCast.getInput().getDefiningOp<mlir::index::CmpOp>();
    if (!cmpOp)
      return b.notifyMatchFailure(inCast.getLoc(), "parent isn't cmp");

    b.replaceOpWithNewOp<mlir::index::CmpOp>(
        op, getInvertedPred(cmpOp.getPred()), cmpOp.getLhs(), cmpOp.getRhs());
    return success();
  }

private:
  static mlir::index::IndexCmpPredicate
  getInvertedPred(mlir::index::IndexCmpPredicate pred) {
    switch (pred) {
    case mlir::index::IndexCmpPredicate::EQ:
      return mlir::index::IndexCmpPredicate::NE;
    case mlir::index::IndexCmpPredicate::NE:
      return mlir::index::IndexCmpPredicate::EQ;

    case mlir::index::IndexCmpPredicate::SLT:
      return mlir::index::IndexCmpPredicate::SGE;
    case mlir::index::IndexCmpPredicate::SLE:
      return mlir::index::IndexCmpPredicate::SGT;
    case mlir::index::IndexCmpPredicate::SGT:
      return mlir::index::IndexCmpPredicate::SLE;
    case mlir::index::IndexCmpPredicate::SGE:
      return mlir::index::IndexCmpPredicate::SLT;

    case mlir::index::IndexCmpPredicate::ULT:
      return mlir::index::IndexCmpPredicate::UGE;
    case mlir::index::IndexCmpPredicate::ULE:
      return mlir::index::IndexCmpPredicate::UGT;
    case mlir::index::IndexCmpPredicate::UGT:
      return mlir::index::IndexCmpPredicate::ULE;
    case mlir::index::IndexCmpPredicate::UGE:
      return mlir::index::IndexCmpPredicate::ULT;
    }
    llvm_unreachable("invalid cmp predicate");
  }
};

/// Canonicalize
/// `(i < x ? x - i : 0) > 0` to `i < x`. or
/// `(x > i ? x - i : 0) > 0` to `x > i`. or
/// `(x > 0 ? x : 0) > 0` to `x > 0`. or
/// `(0 < x ? x : 0) > 0` to `0 < x`.
/// This is a common pattern
/// in for loop constructs.
/// TODO: Generalize this pattern?
struct SimplifyCompareSelect : OpRewritePattern<mlir::index::CmpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::index::CmpOp op,
                                PatternRewriter &b) const override {
    if (op.getPred() != mlir::index::IndexCmpPredicate::SGT)
      return b.notifyMatchFailure(op.getLoc(), "predicate is not `sgt`");

    IntegerAttr cmpRhs;
    if (!mlir::matchPattern(op.getRhs(), mlir::m_Constant(&cmpRhs)) ||
        !cmpRhs.getValue().isZero())
      return b.notifyMatchFailure(op.getLoc(), "RHS is not zero");

    auto select = op.getLhs().getDefiningOp<POP::SelectOp>();
    if (!select)
      return b.notifyMatchFailure(op.getLoc(), "LHS is not a select");

    auto indexCmp = select.getCondition().getDefiningOp<mlir::index::CmpOp>();
    if (!indexCmp)
      return b.notifyMatchFailure(op.getLoc(),
                                  "select condition is not a comparison.");

    auto falseValue = select.getFalseValue();
    auto trueVal = select.getTrueValue();
    auto trueValSub = trueVal.getDefiningOp<mlir::index::SubOp>();

    IntegerAttr falseV;
    if (!mlir::matchPattern(falseValue, mlir::m_Constant(&falseV)) ||
        !falseV.getValue().isZero())
      return b.notifyMatchFailure(op.getLoc(),
                                  "Select's false value is not zero");

    if (indexCmp.getPred() == mlir::index::IndexCmpPredicate::SLT) {
      if (!(indexCmp.getLhs() == falseValue && indexCmp.getRhs() == trueVal)) {
        if (!trueValSub || trueValSub.getLhs() != indexCmp.getRhs() ||
            trueValSub.getRhs() != indexCmp.getLhs())
          return b.notifyMatchFailure(op.getLoc(),
                                      "select true value is not `x - i`");
      }
    } else if (indexCmp.getPred() == mlir::index::IndexCmpPredicate::SGT) {
      if (!(indexCmp.getLhs() == trueVal && indexCmp.getRhs() == falseValue)) {
        if (!trueValSub || trueValSub.getLhs() != indexCmp.getLhs() ||
            trueValSub.getRhs() != indexCmp.getRhs())
          return b.notifyMatchFailure(op.getLoc(),
                                      "select true value is not `x - i`");
      }
    } else {
      return b.notifyMatchFailure(
          op.getLoc(), "select condition is not `slt` or `sgt` comparison");
    }

    IntegerAttr falseVal;
    if (!mlir::matchPattern(select.getFalseValue(),
                            mlir::m_Constant(&falseVal)) ||
        falseVal != cmpRhs)
      return b.notifyMatchFailure(op.getLoc(),
                                  "select false value is not zero");

    // Just replace the whole thing with `i < x` or `x > i`.
    b.replaceOp(op, indexCmp);
    return success();
  }
};

/// Given an if, the condition argument is known to be true within the 'then'
/// region and false in the 'else' region. Propagate this by replacing the
/// condition with a constant in both regions.
struct ConditionPropagation : OpRewritePattern<HLCF::IfOp> {
  ConditionPropagation(MLIRContext *ctx)
      : OpRewritePattern(ctx, /*benefit=*/9) {}

  LogicalResult matchAndRewrite(HLCF::IfOp op,
                                PatternRewriter &b) const override {
    // The pattern matches if the condition has uses in either region. Lazily
    // create the true and false constants.
    Value trueCst, falseCst;
    for (OpOperand &use : op.getCond().getUses()) {
      if (op.getThenRegion().isAncestor(use.getOwner()->getParentRegion())) {
        if (!trueCst)
          trueCst = b.create<mlir::index::BoolConstantOp>(op.getLoc(), true);
        use.set(trueCst);
      } else if (op.getElseRegion().isAncestor(
                     use.getOwner()->getParentRegion())) {
        if (!falseCst)
          falseCst = b.create<mlir::index::BoolConstantOp>(op.getLoc(), false);
        use.set(falseCst);
      }
    }
    return success(trueCst || falseCst);
  }
};

/// A canonicalization pattern for an op in the `custom` dialect.
struct CustomOpPattern : RewritePattern {
  CustomOpPattern(StringAttr opName,
                  std::function<mlir::LogicalResult(mlir::Operation *,
                                                    mlir::PatternRewriter &)>
                      canonicalizationFn)
      : RewritePattern(opName.strref(), /*benefit=*/9, opName.getContext()),
        canonicalizationFn(canonicalizationFn) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &b) const override {
    return canonicalizationFn(op, b);
  }

private:
  std::function<mlir::LogicalResult(mlir::Operation *, mlir::PatternRewriter &)>
      canonicalizationFn;
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_CANONICALIZER
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct Canonicalizer : public impl::CanonicalizerBase<Canonicalizer> {
  Canonicalizer(CompileCanonicalizationFnsFn compileCanonicalizationFnsFn = {})
      : compileCanonicalizationFnsFn(std::move(compileCanonicalizationFnsFn)) {}

  /// Initialize the canonicalizer by building the starting set of patterns.
  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);
    addNonCustomCanonicalizationPatterns(context, owningPatterns);
    patterns = mlir::FrozenRewritePatternSet(std::move(owningPatterns));
    return success();
  }

  /// Get the JIT'ed canonicalization patterns. If there are canonicalization
  /// patterns registered but none are JIT'ed, JIT them all and return them.
  ErrorOr<const DenseMap<StringAttr, std::function<LogicalResult(
                                         Operation *, PatternRewriter &)>> &>
  getOrJITCustomCanonicalizationPatterns(
      DenseResourceElementsAttr opImplModuleAttr);

  /// Update the frozen pattern set after a possible change in the custom ops
  /// patterns.
  ErrorOrSuccess updatePatterns(MLIRContext *context,
                                DenseResourceElementsAttr opImplsModuleAttr);

  /// Add all canonicalization patterns besides the ones from the `custom`
  /// dialect into the pattern set.
  void addNonCustomCanonicalizationPatterns(MLIRContext *context,
                                            RewritePatternSet &patterns);

  void runOnOperation() override;

  /// The patterns that the canonicalizer runs.
  mlir::FrozenRewritePatternSet patterns = {};

  /// Does the pattern has custom op patterns. As we assume that custom op
  /// canonicalization patterns cannot change after being registered, this is
  /// enough to know if `patterns` contains all canonicalization patterns.
  bool hasCustomPatterns = false;

  /// The function to JIT compile canonicalization patterns from the `custom`
  /// dialect.
  CompileCanonicalizationFnsFn compileCanonicalizationFnsFn = {};
};

void Canonicalizer::runOnOperation() {
  // If we do not have custom patterns, check if there are some defined.
  if (!hasCustomPatterns) {
    auto theModule = getOperation()->getParentOfType<ModuleOp>();
    if (!theModule)
      theModule = mlir::cast<ModuleOp>(getOperation());

    auto customOpImplsModuleAttr =
        theModule->getAttrOfType<DenseResourceElementsAttr>(
            Custom::kCustomOpImplModuleAttr);

    // Update the canonicalization patterns if some were defined.
    if (customOpImplsModuleAttr) {
      auto errorOnUpdate = updatePatterns(customOpImplsModuleAttr.getContext(),
                                          customOpImplsModuleAttr);
      if (errorOnUpdate.isError()) {
        mlir::emitError(getOperation()->getLoc())
            << "error while loading custom op canonicalization patterns: "
            << errorOnUpdate.getError() << "\n";
        signalPassFailure();
        return;
      }
    }
  }

  // Run the canonicalization patterns
  mlir::GreedyRewriteConfig config;
  config.enableRegionSimplification = mlir::GreedySimplifyRegionLevel::Disabled;
  (void)applyPatternsAndFoldGreedily(getOperation(), patterns, config);
}

void Canonicalizer::addNonCustomCanonicalizationPatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  // Add the "static" canonicalization patterns.
  for (auto *dialect : context->getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);
  for (mlir::RegisteredOperationName op : context->getRegisteredOperations())
    op.getCanonicalizationPatterns(patterns, context);

  patterns
      .insert<IfToSelect, IfYieldSelect, IndexifyComparison, InvertComparison,
              SimplifyCompareSelect, ConditionPropagation>(context);
}

ErrorOr<const DenseMap<
    StringAttr, std::function<LogicalResult(Operation *, PatternRewriter &)>> &>
Canonicalizer::getOrJITCustomCanonicalizationPatterns(
    DenseResourceElementsAttr opImplModuleAttr) {
  auto *customDialect =
      opImplModuleAttr.getContext()->getLoadedDialect<Custom::CustomDialect>();
  // Lock the mutex as we are going to read and possibly write in the
  // canonicalization functions.
  llvm::sys::SmartScopedWriter<true> lock(customDialect->canonicalizationMutex);

  // First, try to see if the canonicalization patterns are already loaded. If
  // they are, we can safely return a reference to them, as we know the field
  // won't be modified anymore (as it is already loaded).
  if (customDialect->areCanonicalizationFnLoaded)
    return customDialect->canonicalizationFns;

  // Then, this means we need to JIT the canonicalization patterns.
  // Get the op canonicalization patterns symbols from the op
  // implementation module.
  OwningOpRef<ModuleOp> opImplsModule =
      readOpFromBytecodeFile<ModuleOp>(opImplModuleAttr);
  SymbolTable opImplsTable(*opImplsModule);
  auto opImplOp = CustomOpImplsOp::lookupOp(*opImplsModule);
  DenseMap<StringAttr, SymbolConstantAttr> canonicalizationSyms;
  for (auto opImplAttr : opImplOp.getImpls())
    if (auto canonicalizationSym = opImplAttr.getOpCanonicalization()) {
      canonicalizationSyms.try_emplace(opImplAttr.getOpName(),
                                       canonicalizationSym);

      // Set the operation as exported so it doesn't get DCE'd.
      opImplsTable
          .lookup<ExportInterface>(
              canonicalizationSym.getSymbol().getLeafReference())
          .setExported();
    }

  ErrorOr<TargetInfoAttr> targetOr =
      getTargetInfoFor(&getContext(), llvm::sys::getDefaultTargetTriple(),
                       llvm::sys::getHostCPUName(), getHostCPUFeatures());
  if (targetOr.isError())
    return targetOr.takeError();
  TargetInfoAttr target = targetOr.takeValue();

  // JIT them.
  auto errorOrCanonFn = compileCanonicalizationFnsFn(
      *opImplsModule, canonicalizationSyms, target);
  if (errorOrCanonFn.isError())
    return errorOrCanonFn.takeError();

  // Insert jit'ed canonicalization patterns to the custom dialect.
  for (auto &[name, capiCanonFn] : errorOrCanonFn.takeValue()) {
    auto canonFunc = [func = capiCanonFn](Operation *op,
                                          PatternRewriter &rewriter) mutable {
      // Both the operation and the rewriter are passed as pointers, as the
      // mojo canonicalization pattern is marked as inout.
      MlirOperation c_op = wrap(op);
      MlirRewriterBase c_rewriter = wrap(&rewriter);
      return mlir::success(func(&c_op, &c_rewriter));
    };
    customDialect->canonicalizationFns.try_emplace(name, canonFunc);
  }

  // Return them.
  return customDialect->canonicalizationFns;
}

ErrorOrSuccess
Canonicalizer::updatePatterns(MLIRContext *context,
                              DenseResourceElementsAttr opImplsModuleAttr) {
  // Mark the patterns as updated.
  hasCustomPatterns = true;
  RewritePatternSet owningPatterns(context);

  // Add the non custom patterns.
  addNonCustomCanonicalizationPatterns(context, owningPatterns);

  // Add the canonicalization patterns from the custom dialect.
  auto errorOrFns = getOrJITCustomCanonicalizationPatterns(opImplsModuleAttr);
  if (errorOrFns.isError())
    return errorOrFns.takeError();

  for (auto [name, canonFn] : errorOrFns.get())
    owningPatterns.insert<CustomOpPattern>(name, canonFn);

  // Update the pattern set, and the number of custom ops patterns that was
  // used to compute the set.
  patterns = mlir::FrozenRewritePatternSet(std::move(owningPatterns));
  return success();
}

} // namespace

std::unique_ptr<mlir::Pass> KGEN::createCanonicalizer(
    CompileCanonicalizationFnsFn compileCanonicalizationFnsFn) {
  return std::make_unique<Canonicalizer>(
      std::move(compileCanonicalizationFnsFn));
}
