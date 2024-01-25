//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/Threading/ThreadLocalCache.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "llvm/Support/Mutex.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_LIFTANDFOLDAPPLY
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

using LiftedMapStack =
    SmallVector<std::pair<Region *, DenseMap<ParamOperatorAttr, Attribute>>>;

/// Recursively lift and fold apply operators within the provided body.
static void liftAndFoldApply(Region *body, ImplicitLocOpBuilder &b,
                             ParameterCollector &collector,
                             LiftedMapStack &lifted, unsigned &counter,
                             const ParameterUseDefGraph &graph,
                             const ParameterUseDefGraph &topLevel,
                             unsigned &numDedupedApplies) {
  mlir::AttrTypeReplacer replacer;
  lifted.emplace_back().first = body;

  // Skip over parameterized signatures. We cannot generically pull out 'apply'
  // operators from within signature types because the parameter expressions may
  // reference signature parameters. This OK, however, since every parametric
  // signature requires some concretization point in the elaborator to be
  // useful, and we will pull out the 'apply' operator at those points.
  replacer.addReplacement([](SignatureType signature) {
    if (signature.getInputParamTypes().empty() &&
        signature.getResultParamTypes().empty())
      return std::make_pair(signature, WalkResult::advance());
    return std::make_pair(signature, WalkResult::skip());
  });

  replacer.addReplacement([&](ParamOperatorAttr op)
                              -> std::pair<Attribute, WalkResult> {
    // Expressions cannot be hoisted out of a condition because it only
    // conditionally evaluates both branches. Moving expressions out changes the
    // semantics of the program.
    if (op.getOpcode() == POC::Cond)
      return {op, WalkResult::skip()};

    if (op.getOpcode() != POC::Apply)
      return {op, WalkResult::advance()};

    // When we encounter an 'apply' operator, check the lifted map for this
    // scope for an operator of the same value in the current scope that has
    // already been lifted.
    DenseMap<ParamOperatorAttr, Attribute> &curMap = lifted.back().second;
    if (auto it = curMap.find(op); it != curMap.end()) {
      // Deduplicated an 'apply'.
      ++numDedupedApplies;
      return {it->second, WalkResult::advance()};
    }

    // Collect all parameter uses within the operator so we can check the
    // declaring operations to determine the highest scope into which we could
    // lift this operator.
    SmallVector<ParamDeclRefAttr> uses;
    bool hasConstExpr;
    {
      CompilerTimeTraceScope traceScope("collectParameters");
      collector.collectUsesFromAttr(op, uses, hasConstExpr);
    }

    // Baseline is the top-level scope, which would be valid for empty uses.
    Region *upperBound = topLevel.scope;
    for (ParamDeclRefAttr use : uses) {
      const ParamDeclaration &decl = graph.decls.at(use.getName());
      if (upperBound->isProperAncestor(decl.scope))
        upperBound = decl.scope;
    }

    // Walk backwards starting from the next-nearest scope to determine the
    // lowest scope in which this operator has been lifted. This is important
    // because we don't want to hoist invalid operators out of conditionals.
    Attribute existing;
    for (std::pair<Region *, DenseMap<ParamOperatorAttr, Attribute>> &frame :
         llvm::reverse(lifted)) {
      // Skip the current scope.
      if (frame.first != body) {
        // Look for a lifted operator in this scope's cache.
        if (auto it = frame.second.find(op); it != frame.second.end()) {
          existing = it->second;
          break;
        }
      }
      // Stop once we reach the upper bound.
      if (frame.first == upperBound)
        break;
    }

    // If we didn't find an existing lifted operator, then lift the operator at
    // the current scope.
    if (!existing) {
      // Explicit recurse on the operator.
      Type type = replacer.replace(op.getType());
      TypedAttr callee =
          cast<TypedAttr>(replacer.replace(op.getOperands().front()));
      SmallVector<TypedAttr> operands;
      for (TypedAttr operand : op.getOperands().drop_front())
        operands.push_back(cast<TypedAttr>(replacer.replace(operand)));

      // Generate a name for the lifted parameter.
      auto decl = ParamDeclAttr::get(
          b.getStringAttr("(lifted)apply_" + Twine(counter++)), type);

      // Create the operation and set the value of the lifted operator.
      b.create<ParamApplyOp>(decl, callee, operands);
      existing = ParamDeclRefAttr::get(decl);
    } else {
      // Deduplicated an 'apply' with the existing parameter.
      ++numDedupedApplies;
    }

    // Map the created or existing parameter into the current scope. This also
    // has the effect of allowing searches for the same operator to end early in
    // nested scopes.
    curMap.try_emplace(op, existing);
    return {existing, WalkResult::advance()};
  });

  // If the parent is a function, extract 'apply' operators and place them at
  // the start of the body.
  if (auto func = dyn_cast<GeneratorOp>(body->getParentOp())) {
    b.setLoc(func.getLoc());
    b.setInsertionPointToStart(func.getBody());
    replacer.replaceElementsIn(func, /*replaceAttrs=*/false,
                               /*replaceLocs=*/true, /*replaceTypes=*/true);
    // Certain generator attributes need to be evaluatable in isolation.
    // Specially handle them here.
    NamedAttrList attrs = func->getAttrDictionary();
    attrs.set(func.getFunctionTypeAttrName(),
              replacer.replace(func.getFunctionTypeAttr()));
    attrs.set(func.getInputParamsAttrName(),
              replacer.replace(func.getInputParamsAttr()));
    attrs.set(func.getResultParamsAttrName(),
              replacer.replace(func.getResultParamsAttr()));
    attrs.set(func.getDecoratorsAttrName(),
              replacer.replace(func.getDecoratorsAttr()));
    func->setAttrs(attrs.getDictionary(func.getContext()));
  }

  body->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Insert the apply operations as close to the original location of the
    // 'apply' operator as possible.
    b.setLoc(op->getLoc());
    b.setInsertionPoint(op);
    replacer.replaceElementsIn(op, /*replaceAttrs=*/true, /*replaceLocs=*/true,
                               /*replaceTypes=*/true);

    // Walk over nested parameter scopes, since lifted apply operators with name
    // shadowing can cause collisions.
    if (isa<DeclInterface>(op)) {
      for (Region &region : op->getRegions()) {
        liftAndFoldApply(&region, b, collector, lifted, counter,
                         topLevel.nestedScopes.at(&region), topLevel,
                         numDedupedApplies);
      }
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });

  lifted.pop_back();
}

/// Entry point for `liftAndFoldApply`. This function keeps a cache of lifted
/// operators for each scope.
static void liftAndFoldApply(Region *body,
                             ParameterCollector::Analysis &paramCache,
                             const ParameterUseDefGraph &topLevel,
                             unsigned &numDedupedApplies) {
  ImplicitLocOpBuilder b(body->getParentOp()->getLoc(),
                         OpBuilder(body->getContext()));
  LiftedMapStack lifted;
  unsigned counter = 0;
  ParameterCollector collector(paramCache);
  liftAndFoldApply(body, b, collector, lifted, counter, topLevel, topLevel,
                   numDedupedApplies);
}

namespace {
struct LiftAndFoldApplyPass : impl::LiftAndFoldApplyBase<LiftAndFoldApplyPass> {
  using LiftAndFoldApplyBase::LiftAndFoldApplyBase;

  void runOnOperation() override {
    auto &paramCache = getAnalysis<ParameterCollector::Analysis>();
    // Sum the number of deduplicated operators over all work items.
    std::atomic<unsigned> totNumDedupedApplies = 0;

    // Give each thread a copy of the parameter cache, rather than each work
    // item.
    ThreadLocalCache<ParameterCollector::Analysis> threadCaches(
        paramCache, getContext().isMultithreadingEnabled()
                        ? getContext().getThreadPool().getThreadCount()
                        : 1);

    auto workFunc = [&threadCaches, &totNumDedupedApplies](GeneratorOp func) {
      ParameterUseDefGraph graph(func.getBodyRegion());
      ParameterCollector::Analysis &cache = threadCaches.getThreadLocalCache();
      graph.calculate(cache);
      unsigned numDedupedApplies = 0;
      liftAndFoldApply(&func.getBodyRegion(), cache, graph, numDedupedApplies);
      totNumDedupedApplies += numDedupedApplies;
    };
    mlir::parallelForEach(&getContext(), getOperation().getOps<GeneratorOp>(),
                          workFunc);

    markAllAnalysesPreserved();
    numDedupedApplies = totNumDedupedApplies;
  }
};
} // namespace
