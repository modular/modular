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
// Flattens `kgen.struct` function arguments into leaf fields. Which arguments
// and which leaves is the policy's decision
// (`FunctionArgumentDecompositionPolicy` in KGENPasses.h); this file only knows
// how to rewrite a signature, its body and its call sites once told.
//===----------------------------------------------------------------------===//

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENDialect.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"

#include <optional>

using namespace mlir;
using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_DECOMPOSEFUNCTIONARGUMENTS
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {

//===----------------------------------------------------------------------===//
// Default policy
//===----------------------------------------------------------------------===//

/// No per-leaf metadata; pointer-typed fields are left as-is rather than
/// dereferenced (see `createAlwaysDecomposeStructPolicy`'s docstring).
void collectStructLeaves(Type t, SmallVector<unsigned> &path,
                         SmallVector<DecomposedArgLeaf> &out) {
  if (auto st = dyn_cast<StructType>(t)) {
    if (auto elts = st.getElementTypes()) {
      for (auto [i, f] : llvm::enumerate(*elts)) {
        path.push_back(static_cast<unsigned>(i));
        collectStructLeaves(f, path, out);
        path.pop_back();
      }
      return;
    }
  }
  DecomposedArgLeaf leaf;
  leaf.type = t;
  leaf.path.assign(path.begin(), path.end());
  out.push_back(std::move(leaf));
}

class AlwaysDecomposeStructPolicy : public FunctionArgumentDecompositionPolicy {
public:
  FailureOr<std::optional<SmallVector<DecomposedArgLeaf>>>
  planArgument(FuncOp func, unsigned argIdx, Type argType) override {
    if (!isa<StructType>(argType))
      return std::optional<SmallVector<DecomposedArgLeaf>>(std::nullopt);
    SmallVector<DecomposedArgLeaf> leaves;
    SmallVector<unsigned> path;
    collectStructLeaves(argType, path, leaves);
    return std::optional<SmallVector<DecomposedArgLeaf>>(std::move(leaves));
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct DecomposeFunctionArgumentsPass
    : public M::KGEN::impl::DecomposeFunctionArgumentsBase<
          DecomposeFunctionArgumentsPass> {
  DecomposeFunctionArgumentsPass()
      : policy(createAlwaysDecomposeStructPolicy()) {}
  explicit DecomposeFunctionArgumentsPass(
      std::shared_ptr<FunctionArgumentDecompositionPolicy> policy)
      : policy(std::move(policy)) {}
  // MLIR may clone a pass before running it; the policy must survive that.
  DecomposeFunctionArgumentsPass(const DecomposeFunctionArgumentsPass &other)
      : DecomposeFunctionArgumentsBase(other), policy(other.policy) {}

  void runOnOperation() override;

private:
  std::shared_ptr<FunctionArgumentDecompositionPolicy> policy;

  struct FuncPlan {
    // One entry per original input; `nullopt` means "leave untouched".
    SmallVector<std::optional<SmallVector<DecomposedArgLeaf>>> inputs;
    bool changed = false;
    /// (start, len) range into the decomposed signature per original input.
    SmallVector<std::pair<unsigned, unsigned>> inputRanges;
  };

  FailureOr<FuncPlan> planFor(FuncOp func,
                              FunctionArgumentDecompositionPolicy &policy);
  LogicalResult rewriteFunc(FuncOp func, const FuncPlan &plan);
  LogicalResult
  rewriteBody(FuncOp func, ArrayRef<BlockArgument> origArgs,
              ArrayRef<Type> origArgTypes,
              ArrayRef<SmallVector<Value>> leafValuesPerOrigArg,
              ArrayRef<SmallVector<SmallVector<unsigned>>> argLeafPaths);
  LogicalResult
  rewriteCalls(ModuleOp module,
               const DenseMap<Operation *, FuncPlan> &plansByFunc);
};

//===----------------------------------------------------------------------===//
// FuncPlan construction
//===----------------------------------------------------------------------===//

FailureOr<DecomposeFunctionArgumentsPass::FuncPlan>
DecomposeFunctionArgumentsPass::planFor(
    FuncOp func, FunctionArgumentDecompositionPolicy &policy) {
  FuncPlan plan;
  FunctionType ft = func.getFunctionType();
  unsigned newIdx = 0;
  for (auto [i, t] : llvm::enumerate(ft.getInputs())) {
    FailureOr<std::optional<SmallVector<DecomposedArgLeaf>>> planned =
        policy.planArgument(func, static_cast<unsigned>(i), t);
    if (failed(planned))
      return mlir::failure();
    unsigned len = planned->has_value() ? (*planned)->size() : 1;
    plan.inputRanges.emplace_back(newIdx, len);
    newIdx += len;
    plan.changed |= planned->has_value();
    plan.inputs.push_back(std::move(*planned));
  }
  return plan;
}

//===----------------------------------------------------------------------===//
// Body rewrite
//===----------------------------------------------------------------------===//

/// Walks a `struct.extract` / `struct.gep` / `pop.load` chain back to its
/// block arg, collecting the field indices; a load is transparent. Returns
/// null if `value` does not reduce to a block arg this way. A non-constant
/// index ends the walk unless `ignoreSymbolicIndices`, in which case only the
/// root is meaningful, not `outPath`.
static BlockArgument traceChainToArg(Value value,
                                     SmallVector<unsigned> &outPath,
                                     bool ignoreSymbolicIndices = false) {
  outPath.clear();
  auto step = [&](Attribute index, Value container) -> bool {
    IntegerAttr idx = dyn_cast<IntegerAttr>(index);
    if (!idx && !ignoreSymbolicIndices)
      return false;
    outPath.push_back(idx ? static_cast<unsigned>(idx.getInt()) : 0);
    value = container;
    return true;
  };
  while (true) {
    if (auto ba = dyn_cast<BlockArgument>(value)) {
      std::reverse(outPath.begin(), outPath.end());
      return ba;
    }
    Operation *def = value.getDefiningOp();
    if (!def)
      return {};
    if (auto extr = dyn_cast<StructExtractOp>(def)) {
      if (!step(extr.getIndex(), extr.getContainer()))
        return {};
      continue;
    }
    if (auto gep = dyn_cast<StructGEPOp>(def)) {
      if (!step(gep.getIndex(), gep.getContainer()))
        return {};
      continue;
    }
    if (auto ld = dyn_cast<POP::LoadOp>(def)) {
      value = ld.getPtr();
      continue;
    }
    return {};
  }
}

/// The inverse of `traceChainToArg`: descends `cur` along `path`, loading
/// through pointers before extracting. The empty-path check comes first so a
/// pointer-typed leaf the policy left uncrossed comes back unloaded.
static Value materializeLeaf(OpBuilder &b, Location loc, Value cur,
                             Type curType, ArrayRef<unsigned> path) {
  if (path.empty())
    return cur;
  if (auto pt = dyn_cast<PointerType>(curType))
    return materializeLeaf(b, loc, POP::LoadOp::create(b, loc, cur),
                           pt.getElementType(), path);
  auto st = cast<StructType>(curType);
  unsigned idx = path.front();
  Type fieldType = (*st.getElementTypes())[idx];
  Value field = StructExtractOp::create(b, loc, cur, idx);
  return materializeLeaf(b, loc, field, fieldType, path.drop_front(1));
}

/// The type reached by descending `curType` along `path`, pointers
/// transparent; null if `path` does not resolve.
static Type typeAtPath(Type curType, ArrayRef<unsigned> path) {
  if (path.empty())
    return curType;
  if (auto pt = dyn_cast<PointerType>(curType))
    return typeAtPath(pt.getElementType(), path);
  auto st = dyn_cast<StructType>(curType);
  if (!st)
    return {};
  auto elts = st.getElementTypes();
  if (!elts || path.front() >= elts->size())
    return {};
  return typeAtPath((*elts)[path.front()], path.drop_front(1));
}

/// Rebuilds the value at `prefix`, which lies above the leaves, via nested
/// `struct.create`. Null if some position under it has no leaf: this only
/// combines leaves the policy produced, it cannot synthesize one.
static Value materializeAggregate(OpBuilder &b, Location loc, Type targetType,
                                  ArrayRef<unsigned> prefix,
                                  ArrayRef<SmallVector<unsigned>> leafPaths,
                                  ArrayRef<Value> leafValues) {
  for (auto [path, value] : llvm::zip(leafPaths, leafValues))
    if (path.size() == prefix.size() && llvm::equal(path, prefix))
      return value;
  auto st = dyn_cast<StructType>(targetType);
  if (!st)
    return {};
  auto elts = st.getElementTypes();
  if (!elts)
    return {};
  SmallVector<Value> children;
  children.reserve(elts->size());
  SmallVector<unsigned> childPrefix(prefix.begin(), prefix.end());
  for (auto [i, fieldType] : llvm::enumerate(*elts)) {
    childPrefix.push_back(static_cast<unsigned>(i));
    Value child = materializeAggregate(b, loc, fieldType, childPrefix,
                                       leafPaths, leafValues);
    childPrefix.pop_back();
    if (!child)
      return {};
    children.push_back(child);
  }
  return StructCreateOp::create(b, loc, targetType, children);
}

LogicalResult DecomposeFunctionArgumentsPass::rewriteBody(
    FuncOp func, ArrayRef<BlockArgument> origArgs, ArrayRef<Type> origArgTypes,
    ArrayRef<SmallVector<Value>> leafValuesPerOrigArg,
    ArrayRef<SmallVector<SmallVector<unsigned>>> argLeafPaths) {
  DenseMap<BlockArgument, unsigned> origArgIndex;
  for (auto [i, ba] : llvm::enumerate(origArgs))
    origArgIndex[ba] = i;

  // Checked before substitution: a path is not tracked as "still the original
  // pointer" versus "rebuilt", so a safe write cannot be told from one that
  // would never reach backing storage.
  // TODO(npanchen): a store landing exactly on an uncrossed pointer leaf is
  // safe; allow it once a leaf path carries that provenance.
  WalkResult result = func.walk([&](POP::StoreOp st) {
    SmallVector<unsigned> path;
    BlockArgument root = traceChainToArg(st.getPtr(), path);
    if (root && origArgIndex.count(root)) {
      st.emitOpError(
          "pop.store against a decomposed pointer arg: the pointer is "
          "stripped during decomposition, so the underlying value cannot be "
          "written through it");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return mlir::failure();

  // A leaf is a constant path, so a symbolic index into a decomposed arg
  // can never match one; name that here rather than as a missing leaf later.
  result = func.walk([&](Operation *op) {
    Attribute index;
    Value container;
    if (auto extr = dyn_cast<StructExtractOp>(op)) {
      index = extr.getIndex();
      container = extr.getContainer();
    } else if (auto gep = dyn_cast<StructGEPOp>(op)) {
      index = gep.getIndex();
      container = gep.getContainer();
    } else {
      return WalkResult::advance();
    }
    if (isa<IntegerAttr>(index))
      return WalkResult::advance();
    SmallVector<unsigned> path;
    BlockArgument root =
        traceChainToArg(container, path, /*ignoreSymbolicIndices=*/true);
    if (!root || !origArgIndex.count(root))
      return WalkResult::advance();
    op->emitOpError("indexes a decomposed arg with a non-constant index");
    return WalkResult::interrupt();
  });
  if (result.wasInterrupted())
    return mlir::failure();

  // A traced path that reaches or passes a leaf is re-rooted on that leaf;
  // one that stops short of the leaves (an intermediate sub-struct handed
  // on whole) is rebuilt from them. Only the final consumer's operand is
  // checked: a chain op's own operand traces to the un-narrowed root and
  // would misfire the rebuild case one step early. Chain ops left without
  // uses are erased below.
  SmallVector<Operation *> allOps;
  func.walk([&](Operation *op) { allOps.push_back(op); });

  for (Operation *op : allOps) {
    if (!op->getBlock())
      continue;
    if (isa<StructExtractOp, StructGEPOp, POP::LoadOp>(op))
      continue;
    for (OpOperand &operand : op->getOpOperands()) {
      Value v = operand.get();
      SmallVector<unsigned> path;
      BlockArgument root = traceChainToArg(v, path);
      if (!root)
        continue;
      auto it = origArgIndex.find(root);
      if (it == origArgIndex.end())
        continue;
      unsigned origIdx = it->second;
      ArrayRef<SmallVector<unsigned>> leafPaths = argLeafPaths[origIdx];
      ArrayRef<Value> leafValues = leafValuesPerOrigArg[origIdx];
      bool matched = false;
      for (unsigned li = 0; li < leafPaths.size(); ++li) {
        ArrayRef<unsigned> pathRef(path);
        if (leafPaths[li].size() <= pathRef.size() &&
            llvm::equal(leafPaths[li],
                        pathRef.take_front(leafPaths[li].size()))) {
          OpBuilder b(op);
          operand.set(materializeLeaf(
              b, op->getLoc(), leafValues[li], leafValues[li].getType(),
              pathRef.drop_front(leafPaths[li].size())));
          matched = true;
          break;
        }
      }
      if (matched)
        continue;
      Type targetType = typeAtPath(origArgTypes[origIdx], path);
      if (!targetType)
        continue;
      OpBuilder b(op);
      if (Value rebuilt = materializeAggregate(b, op->getLoc(), targetType,
                                               path, leafPaths, leafValues))
        operand.set(rebuilt);
    }
  }

  // To a fixpoint: deeper extracts die only once their consumers are gone.
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> dead;
    func.walk([&](Operation *op) {
      if (!isa<StructExtractOp, StructGEPOp, POP::LoadOp>(op))
        return;
      if (mlir::isOpTriviallyDead(op))
        dead.push_back(op);
    });
    for (Operation *op : dead) {
      op->erase();
      changed = true;
    }
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Signature helpers
//===----------------------------------------------------------------------===//

/// The decomposed input types and conventions for `plan` over `sig`. A leaf
/// inherits its original argument's convention unless the policy set one.
static void
decomposeSignature(ArrayRef<std::optional<SmallVector<DecomposedArgLeaf>>> plan,
                   FuncType sig, SmallVectorImpl<Type> &inputs,
                   SmallVectorImpl<ArgConvention> &convs) {
  ArrayRef<ArgConvention> origConvs = sig.getArgConventions();
  for (auto [argIdx, ap] : llvm::enumerate(plan)) {
    ArgConvention origConv = origConvs[argIdx];
    if (!ap) {
      inputs.push_back(sig.getValues().getInput(argIdx));
      convs.push_back(origConv);
      continue;
    }
    for (const DecomposedArgLeaf &leaf : *ap) {
      inputs.push_back(leaf.type);
      convs.push_back(leaf.convention.value_or(origConv));
    }
  }
}

/// `sig` with its values and per-argument conventions replaced, everything
/// else kept. `FuncType::getWithValuesReplaced` would keep the old-arity
/// conventions and let `FuncType::get` resize them, shifting every convention
/// after a decomposed argument onto the wrong leaf.
static FuncType withSignature(FuncType sig, FunctionType values,
                              ArrayRef<ArgConvention> convs) {
  return FuncType::get(values, convs, sig.getFnEffects(), sig.getMetadata(),
                       sig.getArgListAttrs());
}

//===----------------------------------------------------------------------===//
// Function rewrite
//===----------------------------------------------------------------------===//

LogicalResult
DecomposeFunctionArgumentsPass::rewriteFunc(FuncOp func, const FuncPlan &plan) {
  MLIRContext *ctx = func.getContext();
  Block *entry =
      func.getBodyRegion().empty() ? nullptr : &func.getBodyRegion().front();
  // `rewriteBody` needs the pre-decomposition types to rebuild a use that
  // does not narrow to a single leaf.
  SmallVector<Type> origArgTypes(func.getFunctionType().getInputs().begin(),
                                 func.getFunctionType().getInputs().end());

  // Results are unchanged: writable leaves are only annotated through the
  // policy's per-leaf attributes, and a target pass turns those into its
  // calling convention.
  FuncType origFuncType = func.getFuncTypeGenerator().getBody();
  SmallVector<Type> newInputs;
  SmallVector<ArgConvention> newConvs;
  decomposeSignature(plan.inputs, origFuncType, newInputs, newConvs);
  SmallVector<Type> newResults(func.getFunctionType().getResults().begin(),
                               func.getFunctionType().getResults().end());

  SmallVector<SmallVector<Value>> leafValuesPerOrigArg;
  SmallVector<SmallVector<SmallVector<unsigned>>> argLeafPaths;
  SmallVector<BlockArgument> origArgs;
  if (entry) {
    for (BlockArgument ba : entry->getArguments())
      origArgs.push_back(ba);

    // Append all new args first; original args are erased after body rewrite.
    SmallVector<BlockArgument> newArgs;
    for (Type t : newInputs)
      newArgs.push_back(entry->addArgument(t, func.getLoc()));

    unsigned cursor = 0;
    for (auto [argIdx, ap] : llvm::enumerate(plan.inputs)) {
      if (!ap) {
        BlockArgument newArg = newArgs[cursor++];
        origArgs[argIdx].replaceAllUsesWith(newArg);
        leafValuesPerOrigArg.push_back({newArg});
        argLeafPaths.push_back({SmallVector<unsigned>()});
        continue;
      }
      SmallVector<Value> leaves;
      SmallVector<SmallVector<unsigned>> paths;
      for (const DecomposedArgLeaf &leaf : *ap) {
        leaves.push_back(newArgs[cursor++]);
        paths.push_back(leaf.path);
      }
      leafValuesPerOrigArg.push_back(std::move(leaves));
      argLeafPaths.push_back(std::move(paths));
    }
  }

  auto newFuncType = FunctionType::get(ctx, newInputs, newResults);
  func.setFuncTypeGenerator(FuncTypeGeneratorType::get(
      /*inputParamTypes=*/{},
      withSignature(origFuncType, newFuncType, newConvs),
      /*genMetadata=*/{}));

  if (entry) {
    if (failed(rewriteBody(func, origArgs, origArgTypes, leafValuesPerOrigArg,
                           argLeafPaths)))
      return mlir::failure();
  }

  // A surviving use has no leaf coverage under it; diagnose on that op
  // rather than trip `eraseArgument`'s use-empty assert.
  if (entry) {
    for (unsigned i = origArgs.size(); i-- > 0;) {
      if (!origArgs[i].use_empty())
        return (*origArgs[i].user_begin())
            ->emitOpError("references a decomposed arg through a path with "
                          "no matching leaf");
      entry->eraseArgument(i);
    }
  }

  // `kgen.func` verifies a non-empty `fnArgAttrs` has one entry per
  // argument, so rebuild it whenever any leaf has attributes or the function
  // already carried some.
  ArrayAttr oldArgMeta = func.getFnArgAttrs();
  auto emptyDict = DictionaryAttr::get(ctx);
  SmallVector<Attribute> newArgMeta;
  bool anyNonEmptyLeafMeta = false;
  for (auto [origIdx, ap] : llvm::enumerate(plan.inputs)) {
    if (!ap) {
      newArgMeta.push_back((oldArgMeta && !oldArgMeta.empty())
                               ? oldArgMeta[origIdx]
                               : Attribute(emptyDict));
      continue;
    }
    for (const DecomposedArgLeaf &leaf : *ap) {
      DictionaryAttr d = leaf.argAttrs ? leaf.argAttrs : emptyDict;
      if (!d.empty())
        anyNonEmptyLeafMeta = true;
      newArgMeta.push_back(d);
    }
  }
  if (anyNonEmptyLeafMeta || (oldArgMeta && !oldArgMeta.empty())) {
    assert((!oldArgMeta || oldArgMeta.empty() ||
            oldArgMeta.size() == plan.inputs.size()) &&
           "fnArgAttrs size must equal original arg count");
    func.setFnArgAttrsAttr(ArrayAttr::get(ctx, newArgMeta));
  }

  // `[newStart, newLength, wasDecomposed]` per original arg, so a target pass
  // can remap old-arg-indexed metadata without this pass knowing about it.
  SmallVector<Attribute> argMap;
  argMap.reserve(plan.inputs.size());
  for (auto [origIdx, ap] : llvm::enumerate(plan.inputs)) {
    auto [start, len] = plan.inputRanges[origIdx];
    argMap.push_back(DenseI32ArrayAttr::get(ctx, {static_cast<int32_t>(start),
                                                  static_cast<int32_t>(len),
                                                  ap.has_value() ? 1 : 0}));
  }
  func->setAttr(kDecomposedArgMapAttrName, ArrayAttr::get(ctx, argMap));

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Call rewrite: keep call sites in sync with a decomposed callee
//===----------------------------------------------------------------------===//

// `kgen.call` caches the callee's signature in its callee attribute, so a
// call to a decomposed callee needs its operands and that type refreshed
// together.
LogicalResult DecomposeFunctionArgumentsPass::rewriteCalls(
    ModuleOp module, const DenseMap<Operation *, FuncPlan> &plansByFunc) {
  if (plansByFunc.empty())
    return mlir::success();

  mlir::SymbolTable symtab(module);
  SmallVector<KGEN::CallOp> calls;
  module.walk([&](KGEN::CallOp call) { calls.push_back(call); });

  for (KGEN::CallOp call : calls) {
    auto callee = symtab.lookup<FuncOp>(call.getCalleeSymbol().getAttr());
    if (!callee)
      continue;
    auto it = plansByFunc.find(callee.getOperation());
    if (it == plansByFunc.end())
      continue;
    const FuncPlan &calleePlan = it->second;

    FuncType origSig = call.getCalleeType().getBody();
    FunctionType origFuncTy = origSig.getValues();
    // Operands and the cached type are both indexed by original-arg position
    // below, so a call that drifted from its callee cannot be expanded.
    if (call.getNumOperands() != calleePlan.inputs.size() ||
        origFuncTy.getNumInputs() != calleePlan.inputs.size())
      return call.emitOpError(
          "cached callee signature disagrees with the callee's argument list, "
          "so its decomposed operands cannot be expanded");

    OpBuilder b(call);
    SmallVector<Value> newOperands;
    newOperands.reserve(origFuncTy.getNumInputs());
    for (auto [argIdx, operand] : llvm::enumerate(call.getOperands())) {
      const std::optional<SmallVector<DecomposedArgLeaf>> &ap =
          calleePlan.inputs[argIdx];
      if (!ap) {
        newOperands.push_back(operand);
        continue;
      }
      Type origArgType = origFuncTy.getInput(argIdx);
      for (const DecomposedArgLeaf &leaf : *ap)
        newOperands.push_back(
            materializeLeaf(b, call.getLoc(), operand, origArgType, leaf.path));
    }

    // Only inputs are decomposed, so results never change here.
    call->setOperands(newOperands);
    SmallVector<Type> newInputs;
    SmallVector<ArgConvention> newConvs;
    decomposeSignature(calleePlan.inputs, origSig, newInputs, newConvs);
    auto newFuncTy = FunctionType::get(call.getContext(), newInputs,
                                       origFuncTy.getResults());
    FuncType newSig = withSignature(origSig, newFuncTy, newConvs);
    auto newSigGen = FuncTypeGeneratorType::get(/*inputParamTypes=*/{}, newSig,
                                                /*genMetadata=*/{});
    call.setCalleeAttr(
        SymbolConstantAttr::get(call.getCalleeSymbol(), newSigGen));
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// runOnOperation
//===----------------------------------------------------------------------===//

void DecomposeFunctionArgumentsPass::runOnOperation() {
  ModuleOp module = getOperation();

  SmallVector<FuncOp> funcs;
  module.walk([&](FuncOp f) { funcs.push_back(f); });

  DenseMap<Operation *, FuncPlan> plansByFunc;
  for (FuncOp f : funcs) {
    FailureOr<FuncPlan> plan = planFor(f, *policy);
    if (failed(plan)) {
      signalPassFailure();
      return;
    }
    if (!plan->changed)
      continue;
    if (failed(rewriteFunc(f, *plan))) {
      signalPassFailure();
      return;
    }
    plansByFunc.try_emplace(f.getOperation(), std::move(*plan));
  }

  if (failed(rewriteCalls(module, plansByFunc))) {
    signalPassFailure();
    return;
  }
}

} // namespace

std::unique_ptr<FunctionArgumentDecompositionPolicy>
M::KGEN::createAlwaysDecomposeStructPolicy() {
  return std::make_unique<AlwaysDecomposeStructPolicy>();
}

std::unique_ptr<mlir::Pass> M::KGEN::createDecomposeFunctionArguments(
    std::unique_ptr<FunctionArgumentDecompositionPolicy> policy) {
  return std::make_unique<DecomposeFunctionArgumentsPass>(
      std::shared_ptr<FunctionArgumentDecompositionPolicy>(std::move(policy)));
}
