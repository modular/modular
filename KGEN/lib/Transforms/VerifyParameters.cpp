//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Interpreter/InterpreterInterface.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/Compiler/Threading.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Mutex.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_VERIFYPARAMETERS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
class ParameterSimplifier : public ParameterEvaluator, public InterpreterState {
public:
  ParameterSimplifier(bool enableInterp, ModuleOp module,
                      SymbolTableCollection &symtabs)
      : InterpreterState(module.getContext()), enableInterp(enableInterp),
        module(module), symtabs(symtabs) {}
  ParameterSimplifier(const ParameterSimplifier &other)
      : ParameterEvaluator(other.getParameterValues()),
        InterpreterState(other.getContext()), enableInterp(other.enableInterp),
        module(other.module), symtabs(other.symtabs) {}

  FailureOr<TypedAttr> evaluateExpression(ParamOperatorAttr op) override;
  ErrorOr<Region *> lookupFunctionBody(SymbolRefAttr symbol) override;
  Operation *lookupTypeDefinition(SymbolRefAttr symbol) override;

private:
  bool enableInterp;
  ModuleOp module;
  SymbolTableCollection &symtabs;
};
} // namespace

FailureOr<TypedAttr>
ParameterSimplifier::evaluateExpression(ParamOperatorAttr op) {
  if (!enableInterp)
    return failure();

  if (op.getOpcode() != POC::Apply && op.getOpcode() != POC::ApplyResultSlot)
    return failure();

  // We can only fold direct calls.
  auto ref = dyn_cast<SymbolConstantAttr>(op.getOperands().front());
  if (!ref)
    return failure();

  // All inputs must be simple constants.
  ArrayRef<TypedAttr> inputs = op.getOperands().drop_front();
  if (!llvm::all_of(inputs, ParameterAttr::isSimpleConstant))
    return failure();

  SmallVector<Attribute> arguments;
  for (TypedAttr input : inputs)
    arguments.push_back(input);

  ErrorOr<Region *> bodyOr = lookupFunctionBody(ref.getSymbol());
  if (bodyOr.isError())
    return failure();
  Region &body = **bodyOr;

  TypedAttr value;
  if (op.getOpcode() == POC::Apply) {
    ErrorTreeOr<SmallVector<Attribute>> result = executeRegion(body, arguments);
    if (result.isError()) {
      DEBUG_WITH_TYPE(
          "simple-interpreter",
          result.takeError().emit(
              (InFlightDiagnostic(*)(Location))mlir::emitError, "called from"));
      return failure();
    }
    value = cast<TypedAttr>(result->front());
  } else {
    auto func = cast<FuncInterface>(body.getParentOp());
    bool isInitSelf = func.getSignature().hasInitSelfArg();
    Value resultArg =
        isInitSelf ? body.getArgument(0) : body.getArguments().back();
    Type resultType = cast<PointerType>(resultArg.getType()).getElementType();
    ErrorTreeOr<TypedAttr> result = executeRegionWithResultSlot(
        body, arguments, isInitSelf, createUninitializedValueOf(resultType));
    if (result.isError()) {
      DEBUG_WITH_TYPE(
          "simple-interpreter",
          result.takeError().emit(
              (InFlightDiagnostic(*)(Location))mlir::emitError, "called from"));
      return failure();
    }
    value = result.takeValue();
  }

  if (!ParameterAttr::isSimpleConstant(value))
    return failure();
  return value;
}

ErrorOr<Region *>
ParameterSimplifier::lookupFunctionBody(SymbolRefAttr symbol) {
  auto func = symtabs.lookupSymbolIn<mlir::FunctionOpInterface>(module, symbol);
  assert(func && "invalid function reference");
  if (func.isExternal())
    return Error("external function reference");
  return &func.getFunctionBody();
}

Operation *ParameterSimplifier::lookupTypeDefinition(SymbolRefAttr symbol) {
  return symtabs.lookupSymbolIn(module, symbol);
}

/// Function to walk all op users of parameters and substitute parameters based
/// on the values currently in the evaluator.
static void processOp(Operation *op, ParameterSimplifier &evaluator) {
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
                                       ParameterSimplifier evaluator) {
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
        evaluator.setParameterValue(decl, ParamDeclRefAttr::get(decl));
      }
      // All required parameters are bound for the parent op. Process it now.
      // Skip the top-level declaration since it cannot reference parameters
      // declared inside it.
      if (op != topLevel.scope->getParentOp())
        processOp(op, evaluator);
    } else if (auto declare = dyn_cast<ParamDeclareOp>(op)) {
      // If the value of the declared parameter is "trivial", i.e. a simple
      // constant, then propagate it. We can only safely refine the attribute
      // (interpret calls) if its type is not parametric. If the type is
      // parametric, we risk creating unequal types across function calls if
      // there are dependent parameters.
      Attribute value =
          cast<TypedAttr>(evaluator.getReboundAttribute(declare.getValue()));

      // The type of the parameter may change. Try to rebind it.
      auto decl = cast<ParamDeclAttr>(
          evaluator.getReboundAttribute(declare.getParamDecl()));
      if (isa<ParamDeclRefAttr, TypeConstantAttr>(value) ||
          ParameterAttr::isSimpleConstant(value)) {
        evaluator.setParameterValue(decl, value);
        declare.erase();
      } else {
        evaluator.setParameterValue(decl, ParamDeclRefAttr::get(decl));
        processOp(op, evaluator);
      }
    } else {
      // If this is any other operation, just walk its definitions in the
      // current scope.
      cast<ParamOpInterface>(op).walkDefinitions(
          [&](ParamDeclAttr decl, const ParamDefValue &value) {
            decl = cast<ParamDeclAttr>(evaluator.getReboundAttribute(decl));
            evaluator.setParameterValue(decl, ParamDeclRefAttr::get(decl));
          });
      // Nested regions can declare parameters, so we cannot fully rebind the
      // operation now. It will be handled later when this function recurses.
      if (!isa<ParamDeclareRegionOp>(op))
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
    }
  }

  // Any op might contain a parametric location, so we go through all of them.
  auto rebindLoc = [&](Location loc) {
    return cast<Location>(evaluator.getReboundAttribute(loc));
  };
  region->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (auto inlined = dyn_cast<DebugInfo::InlinedSubprogramScoped>(op))
      if (LocationAttr loc = inlined.getCallLocAttr())
        inlined.setCallLocAttr(rebindLoc(loc));
    // DeclInterface's location might reference parameters declared by it (e.g.
    // in case of a parametric argument making it into a subprogram scope type),
    // so we will handle it when we recurse into it.
    if (isa<DeclInterface>(op))
      return WalkResult::skip();
    op->setLoc(rebindLoc(op->getLoc()));
    return WalkResult::advance();
  });
  // Don't process the top-level decl operation. It cannot reference
  // declarations in its body and its location is shared across threads.
  if (region->getParentOp() != topLevel.scope->getParentOp())
    if (auto declScope = dyn_cast<DeclInterface>(region->getParentOp()))
      declScope->setLoc(rebindLoc(declScope->getLoc()));

  // Recurse into nested parameter scopes.
  for (Region *region : graph.nestedDecls)
    propagateTrivialParameters(region, topLevel.nestedScopes.at(region),
                               topLevel, evaluator);
}

namespace {
struct VerifyParametersPass : impl::VerifyParametersBase<VerifyParametersPass> {
  using VerifyParametersBase::VerifyParametersBase;

  void runOnOperation() override {
    using ParamCache = ParameterCollector::Analysis;

    auto &analysis = getAnalysis<mlir::SymbolTableAnalysis>();
    mlir::LockedSymbolTableCollection sharedSymtabs(analysis.getSymbolTables());
    auto &paramCache = getAnalysis<ParamCache>();
    bool emptyCache = paramCache.parameterLess.empty();

    std::vector<std::pair<Region *, size_t>> declRegions;
    ModuleOp module = getOperation();
    for (auto decl : module.getOps<DeclInterface>())
      for (Region &region : decl->getRegions())
        declRegions.emplace_back(&region, declRegions.size());

    // Because parameter simplification invokes the interpreter, we cannot
    // simplify in parallel: functions may be modified as they are being
    // interpreted. Save the use-def graphs from the verification pass here.
    std::vector<ParameterUseDefGraph> graphs;
    if (simplifyParameters) {
      graphs.reserve(declRegions.size());
      for (size_t i = 0, e = declRegions.size(); i != e; ++i)
        graphs.emplace_back(nullptr);
    }

    auto workFunc =
        [&sharedSymtabs, &graphs, simplify = bool(simplifyParameters)](
            ParamCache &paramCache, std::pair<Region *, size_t> item) {
          auto [declRegion, i] = item;
          ParameterUseDefGraph graph(*declRegion);
          if (failed(graph.verify(sharedSymtabs, paramCache)))
            return failure();
          if (simplify)
            graphs[i] = std::move(graph);
          return mlir::success();
        };

    auto consolidateFn = [emptyCache](ParamCache &original,
                                      ArrayRef<ParamCache> threadCaches) {
      // Consolidate the caches, but only when the original cache is empty.
      // In reality, the cache does not grow much after the first run of
      // this pass on an input IR, so consolidation is only worthwhile on
      // the first run of the pass, when the cache is empty.
      if (emptyCache)
        return;
      for (const ParamCache &threadCache : threadCaches) {
        original.parameterLess.insert(threadCache.parameterLess.begin(),
                                      threadCache.parameterLess.end());
      }
    };

    if (failed(failableParallelForEach(&getContext(), declRegions,
                                       std::move(workFunc), paramCache,
                                       consolidateFn)))
      return signalPassFailure();

    // This pass does not modify any IR, so mark all analyses as preserved. In
    // addition, this signals the pass manager that the MLIR verifier need not
    // run after this pass.
    if (!simplifyParameters) {
      markAllAnalysesPreserved();
      return;
    }

    CompilerTimeTraceScope traceScope("propagateTrivialParameters");
    for (auto [declRegion, i] : declRegions) {
      ParameterUseDefGraph &graph = graphs[i];
      propagateTrivialParameters(
          declRegion, graph, graph,
          ParameterSimplifier(enableInterp, module,
                              analysis.getSymbolTables()));
    }
  }
};
} // namespace
