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

#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/Compiler/Threading.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Mutex.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_VERIFYPARAMETERS
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

/// Function to walk all op users of parameters and substitute parameters based
/// on the values currently in the evaluator.
static void processOp(Operation *op, ParameterEvaluator &evaluator) {
  SmallVector<NamedAttribute> attrs;
  bool changed = false;
  for (const NamedAttribute &attr : op->getAttrs()) {
    Attribute newAttr = evaluator.getReboundAttribute(attr.getValue());
    attrs.emplace_back(attr.getName(), newAttr);
    changed |= newAttr != attr.getValue();
  }
  if (changed)
    op->setAttrs(DictionaryAttr::getWithSorted(op->getContext(), attrs));

  for (OpResult result : op->getResults())
    result.setType(evaluator.getReboundType(result.getType()));
  for (Region &region : op->getRegions())
    for (BlockArgument arg : region.getArguments())
      arg.setType(evaluator.getReboundType(arg.getType()));
}

/// Propagate trivial parameter declarations in the region, given the use-def
/// graph for that region and the top-level graph to lookup nested regions.
static void propagateTrivialParameters(Region *region,
                                       const ParameterUseDefGraph &graph,
                                       const ParameterUseDefGraph &topLevel,
                                       ParameterEvaluator evaluator) {
  // Collect the defining operations in topological order. The same operation
  // can define multiple parameters, so punt them according to their most
  // dominated definition. Do this by collecting them in reverse.
  llvm::SetVector<Operation *> defOps;
  for (StringAttr param : llvm::reverse(graph.params))
    defOps.insert(graph.defs.at(param).defOp);
  for (Operation *op : llvm::reverse(defOps)) {
    if (auto decl = dyn_cast<DeclInterface>(op);
        decl && op == region->getParentOp()) {
      // For parent decl ops, bind input parameters to themselves.
      for (ParamDeclAttr decl : decl.getInputParams()) {
        decl = cast<ParamDeclAttr>(evaluator.getReboundAttribute(decl));
        evaluator.setDeclBinding(decl, ParamDeclRefAttr::get(decl));
      }
      // All required parameters are bound for the parent op. Process it now.
      // Skip the top-level declaration since it cannot reference parameters
      // declared inside it.
      if (op != topLevel.scope->getParentOp())
        processOp(op, evaluator);
    } else if (auto declare = dyn_cast<ParamDeclareOp>(op)) {
      // If the value of the declared parameter is "trivial", i.e. a simple
      // constant, then propagate it.
      TypedAttr value = declare.getValue();
      if (!isa<DeferredAttr>(value))
        value = evaluator.getReboundAttribute(value);

      // The type of the parameter may change. Try to rebind it.
      auto decl = cast<ParamDeclAttr>(
          evaluator.getReboundAttribute(declare.getParamDecl()));
      evaluator.setDeclBinding(decl, value);
      declare.erase();
    } else {
      // If this is any other operation, just walk its definitions in the
      // current scope.
      cast<ParamOpInterface>(op).walkDefinitions(
          [&](ParamDeclAttr decl, const ParamDefValue &value) {
            decl = cast<ParamDeclAttr>(evaluator.getReboundAttribute(decl));
            evaluator.setDeclBinding(decl, ParamDeclRefAttr::get(decl));
          });
      processOp(op, evaluator);
    }
  }

  for (Operation *op : graph.paramOps) {
    processOp(op, evaluator);

    // Peephole rebinds that have been resolved to the same types.
    if (auto rebind = dyn_cast<RebindOp>(op);
        rebind && rebind.getInput().getType() == rebind.getType()) {
      rebind.replaceAllUsesWith(rebind.getInput());
      rebind.erase();
      continue;
    }

    // GeneratorUser ops need concretize callee to be called when the callee
    // attr is updated.
    if (auto generatorUser = dyn_cast<GeneratorUserOpInterface>(op)) {
      auto callee = FuncSymbolGenerator::dynCast(generatorUser.getCallee());
      if (callee && callee.isFullyBound()) {
        IRRewriter b{OpBuilder(op)};
        generatorUser.concretizeCallee(b, callee);
      }
    }
  }

  // Any op might contain a parametric location, so we go through all of them.
  auto rebindLoc = [&](Location loc) {
    return cast<Location>(evaluator.getReboundAttribute(loc));
  };
  OpRegionBlockWalker walker(
      [&](Operation *op) {
        if (auto inlined = dyn_cast<DebugInfo::InlinedSubprogramScoped>(op))
          if (LocationAttr loc = inlined.getCallLocAttr())
            inlined.setCallLocAttr(rebindLoc(loc));
        // DeclInterface's location might reference parameters declared by it
        // (e.g. in case of a parametric argument making it into a subprogram
        // scope type), so we will handle it when we recurse into it.
        if (isa<DeclInterface>(op))
          return WalkResult::skip();
        op->setLoc(rebindLoc(op->getLoc()));
        return WalkResult::advance();
      },
      nullptr,
      [&](Block *block) {
        for (BlockArgument arg : block->getArguments())
          arg.setLoc(rebindLoc(arg.getLoc()));
        return WalkResult::advance();
      });
  walker.walk(region);
  // Don't process the top-level decl operation. It cannot reference
  // declarations in its body and its location is shared across threads.
  if (region->getParentOp() != topLevel.scope->getParentOp())
    if (auto declScope = dyn_cast<DeclInterface>(region->getParentOp()))
      declScope->setLoc(rebindLoc(declScope->getLoc()));

  // Recurse into nested parameter scopes.
  for (Region *region : graph.nestedDecls) {
    propagateTrivialParameters(region, topLevel.nestedScopes.at(region),
                               topLevel, evaluator);
  }
}

namespace {
struct VerifyParametersPass : impl::VerifyParametersBase<VerifyParametersPass> {
  using VerifyParametersBase::VerifyParametersBase;

  void runOnOperation() override {
    using ParamCache = ParameterCollector::Analysis;

    auto &analysis = getAnalysis<mlir::SymbolTableAnalysis>();
    mlir::LockedSymbolTableCollection sharedSymtabs(analysis.getSymbolTables());
    auto &paramCache = getAnalysis<ParamCache>();

    std::vector<std::pair<Region *, size_t>> declRegions;
    ModuleOp module = getOperation();
    for (auto decl : module.getOps<DeclInterface>())
      for (Region &region : decl->getRegions())
        declRegions.emplace_back(&region, declRegions.size());

    LIT::LITSymTabEvaluationContext evaluationContext(module, sharedSymtabs);

    // Save the use-def graphs from the verification pass for simplification.
    std::vector<ParameterUseDefGraph> graphs;
    if (simplifyParameters) {
      graphs.reserve(declRegions.size());
      for (size_t i = 0, e = declRegions.size(); i != e; ++i)
        graphs.emplace_back(nullptr);
    }

    auto workFunc =
        [&evaluationContext, &graphs, simplify = bool(simplifyParameters)](
            ParamCache &paramCache, std::pair<Region *, size_t> item) {
          auto [declRegion, i] = item;
          ParameterUseDefGraph graph(*declRegion);
          if (failed(graph.verify(&evaluationContext, paramCache)))
            return failure();
          if (simplify)
            graphs[i] = std::move(graph);
          return mlir::success();
        };

    if (failed(paramCache.failableParallelForEach(&getContext(), declRegions,
                                                  std::move(workFunc))))
      return signalPassFailure();

    // This pass does not modify any IR, so mark all analyses as preserved. In
    // addition, this signals the pass manager that the MLIR verifier need not
    // run after this pass.
    if (!simplifyParameters) {
      markAllAnalysesPreserved();
      return;
    }

    VerboseCompilerTimeTraceScope traceScope("propagateTrivialParameters");
    auto propagate = [&](std::pair<Region *, size_t> item) {
      auto [declRegion, i] = item;
      ParameterUseDefGraph &graph = graphs[i];
      ParameterEvaluator evaluator;
      evaluator.setEvaluationContext(&evaluationContext);
      propagateTrivialParameters(declRegion, graph, graph,
                                 std::move(evaluator));
    };

    // Evaluation reads a function only through its top-level op, which
    // simplification never changes, so function bodies run in parallel. Their
    // nested scopes are comptime regions that no other declaration can name.
    // Struct and trait bodies are read in full, so they run first, serially.
    std::vector<std::pair<Region *, size_t>> functionRegions;
    for (auto item : declRegions) {
      if (isa<GeneratorOp>(item.first->getParentOp()))
        functionRegions.push_back(item);
      else
        propagate(item);
    }
    mlir::parallelForEach(&getContext(), functionRegions, propagate);
  }
};
} // namespace
