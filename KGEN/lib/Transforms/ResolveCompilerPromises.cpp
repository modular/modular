//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "CallGraphUtils.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "LLCL/Support/ForkJoin.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_RESOLVECOMPILERPROMISES
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct ResolveCompilerPromisesPass
    : impl::ResolveCompilerPromisesBase<ResolveCompilerPromisesPass> {
  explicit ResolveCompilerPromisesPass(LLCL::Runtime *runtime = nullptr)
      : runtime(runtime) {}

  void runOnOperation() override;

  LLCL::Runtime *runtime;
};
} // namespace

std::unique_ptr<Pass>
KGEN::createResolveCompilerPromises(LLCL::Runtime &runtime) {
  return std::make_unique<ResolveCompilerPromisesPass>(&runtime);
}

namespace {
/// Op-like wrapper for call operations and capture list operations that must
/// behave as edges in the callgraph.
class CallLikeOp {
public:
  // Required methods for LLVM-style RTTI.
  CallLikeOp(Operation *op) : op(op) {
    assert((!op || classof(op)) && "not a call-like op");
  }
  static bool classof(Operation *op) {
    return isa_and_nonnull<KGENCallOpInterface, CaptureListCreateOp,
                           CaptureListCopyOp>(op);
  }
  operator bool() const { return op; }
  operator Operation *() const { return op; }
  Operation &operator*() { return *op; }

  // Required CallGraph interface.
  TypedAttr getCallee() {
    // Micro-optimization: `isa` on operations is faster than interfaces.
    if (auto create = dyn_cast<CaptureListCreateOp>(op))
      return create.getCallee();
    if (auto copy = dyn_cast<CaptureListCopyOp>(op))
      return copy.getCallee();
    return cast<KGENCallOpInterface>(op).getCallee();
  }

private:
  /// The underlying operation.
  Operation *op;
};

struct CallGraphNode
    : public CallGraphNodeBase<CallGraphNode, FuncOp, CallLikeOp> {
  using CallGraphNodeBase::CallGraphNodeBase;

  std::vector<std::pair<StringAttr, Type>> requiredPromises;
};

struct CallGraph : public CallGraphBase<CallGraph, CallGraphNode> {
  explicit CallGraph(LLCL::Runtime &runtime, const SymbolTable &symtab,
                     TargetInfoAttr targetInfo)
      : worklist(runtime), symtab(symtab), targetInfo(targetInfo) {}
  using Base = CallGraphBase<CallGraph, CallGraphNode>;

  /// It turns out we have to rely on propagation of the `capturing` bit on
  /// functions in the parser, where a function with a `capturing` signature in
  /// its input or result parameters also needs to be marked as `capturing`,
  /// because otherwise `kgen.create_closure` does not know whether to create a
  /// function pointer or a closure with a capture list.
  ///
  /// In that case, we can also rely on the bit to form the edges in the graph
  /// to be processed, circumventing the issue with cycles.
  bool shouldAddToGraph(CallLikeOp call, CallGraphNode *node) {
    return node->func.getSignature().isCapturing() ||
           isa<CaptureListCreateOp, CaptureListCopyOp>(*call);
  }

  /// Process a single function. Resolve all the promises in the function by
  /// processing `pop.compiler.global_load` and `pop.compiler.global_store` ops,
  /// prepending arguments if necessary.
  void resolvePromises(CallGraphNode *node);

  /// Lookup the call graph node for the given operation.
  CallGraphNode *getCalleeNode(TypedAttr symbol) {
    auto callee = symtab.lookup<FuncOp>(
        cast<SymbolConstantAttr>(symbol).getSymbol().getRootReference());
    assert(callee);
    return &nodes.find(callee)->second;
  }

  /// Initialize the worklist and run to completion.
  void run();

  LLCL::ForkJoin worklist;
  const SymbolTable symtab;
  TargetInfoAttr targetInfo;
};
} // namespace

/// Walk the operations contained within operation in reverse, in post order.
/// That means `op` is visited after all the ops in its regions. Ops are visited
/// in reverse order in each region, starting from the last region of each op.
static void reversePostOrderWalk(Operation *op,
                                 function_ref<void(Operation *)> walkFn) {
  for (Region &region : llvm::reverse(op->getRegions())) {
    // There shouldn't be more than one block here.
    assert(region.getBlocks().size() <= 1 && "unexpected CFG");
    if (!region.hasOneBlock())
      continue;
    Block &block = region.front();
    // Ops can get deleted, so make sure to early inc.
    for (Operation &op : llvm::make_early_inc_range(llvm::reverse(block)))
      reversePostOrderWalk(&op, walkFn);
  }
  walkFn(op);
}

/// Propagate from `kgen.capture_list.create` the set of required promises from
/// the callee into the current function. Query them and pack them into a
/// heap-allocated slot.
static void resolveCaptureListCreate(CaptureListCreateOp op,
                                     TargetInfoAttr target,
                                     ArrayRef<Value> captures) {
  auto clType = StructType::get(
      op.getContext(),
      llvm::map_to_vector(captures, [](Value v) { return v.getType(); }));

  ImplicitLocOpBuilder b(op->getLoc(), OpBuilder(op));
  // Allocate the memory for the actual capture state.
  Value alloc = b.create<POP::AlignedAllocOp>(
      PointerType::get(clType),
      b.create<mlir::index::ConstantOp>(*clType.getTypeAlign(target)),
      b.create<mlir::index::ConstantOp>(*clType.getTypeSize(target)));
  for (auto [i, capture] : llvm::enumerate(captures))
    b.create<POP::StoreOp>(capture, b.create<StructGEPOp>(alloc, i));

  Value opaque = b.create<POP::PointerBitcastOp>(
      PointerType::get(KGEN::NoneType::get(op.getContext())), alloc);
  op->replaceAllUsesWith(ValueRange(opaque));
  op.erase();
}

/// Using knowledge of the required promises, emit IR for a copy of the capture
/// list for a particular closure.
static void resolveCaptureListCopy(CaptureListCopyOp op, TargetInfoAttr target,
                                   ArrayRef<Type> captures) {
  auto clType = StructType::get(op.getContext(), captures);

  ImplicitLocOpBuilder b(op->getLoc(), OpBuilder(op));
  // Allocate the memory for the copy.
  Value alloc = b.create<POP::AlignedAllocOp>(
      PointerType::get(clType),
      b.create<mlir::index::ConstantOp>(*clType.getTypeAlign(target)),
      b.create<mlir::index::ConstantOp>(*clType.getTypeSize(target)));

  // Emit the memcpy.
  Value orig =
      b.create<POP::PointerBitcastOp>(PointerType::get(clType), op.getOrig());
  b.create<POP::StoreOp>(b.create<POP::LoadOp>(orig), alloc);

  Value opaque = b.create<POP::PointerBitcastOp>(
      PointerType::get(KGEN::NoneType::get(op.getContext())), alloc);
  op->replaceAllUsesWith(ValueRange(opaque));
  op.erase();
}

/// Rewrite `kgen.capture_list.expand %cl` given the current set of required
/// promises by propagating the required promises into the node of the enclosing
/// function.
static void resolveCaptureListExpand(
    CaptureListExpandOp op, CallGraphNode *node,
    llvm::MapVector<StringAttr, SmallVector<POP::CompilerGlobalLoadOp>>
        &requiredPromises) {
  ImplicitLocOpBuilder b(op->getLoc(), OpBuilder(op));
  b.setInsertionPoint(op);

  // Compute the capture list type based on the set of required promises.
  SmallVector<Type> types =
      llvm::map_to_vector(requiredPromises, [](auto &nameLoads) {
        return nameLoads.second.front().getType();
      });
  auto clType = StructType::get(op.getContext(), types);
  Value captures = b.create<POP::PointerBitcastOp>(PointerType::get(clType),
                                                   op.getCaptureList());

  for (auto [idx, nameLoads] : llvm::enumerate(requiredPromises)) {
    // Extract each captured value out of the capture state and use it as
    // input to the call of the closure.
    auto &[name, loads] = nameLoads;
    Value promise = b.create<POP::LoadOp>(b.create<StructGEPOp>(captures, idx));
    Type type = loads.front().getType();
    for (POP::CompilerGlobalLoadOp load : loads) {
      load.replaceAllUsesWith(promise);
      load.erase();
    }
    // Forward the required promise onto the node of the enclosing function.
    // The invariant is that the enclosing function is not 'capturing', so the
    // promises must be propagated through `kgen.capture_list.create`.
    node->requiredPromises.emplace_back(name, type);
  }

  // Clear the required promises set. It has been fully satisfied by this op.
  requiredPromises.clear();
  op.erase();
}

void CallGraph::resolvePromises(CallGraphNode *node) {
  FuncOp func = node->func;
  CompilerTimeTraceScope traceScope("resolvePromises", [func]() mutable {
    return func.getSymNameAttr().str();
  });

  llvm::MapVector<StringAttr, SmallVector<POP::CompilerGlobalLoadOp>>
      requiredPromises;

  reversePostOrderWalk(func, [&](Operation *op) {
    // When we encounter a load, mark it as a requested promise within the
    // function.
    if (auto load = dyn_cast<POP::CompilerGlobalLoadOp>(op)) {
      requiredPromises[load.getNameAttr()].push_back(load);
      return;
    }

    // When a store is encountered, resolve every load that requested this
    // value.
    if (auto store = dyn_cast<POP::CompilerGlobalStoreOp>(op)) {
      auto it = requiredPromises.find(store.getNameAttr());
      if (it == requiredPromises.end()) {
        store.erase();
        return;
      }
      SmallVector<POP::CompilerGlobalLoadOp> leftover;
      for (POP::CompilerGlobalLoadOp load : it->second) {
        // Make sure the store dominates the load in terms of regions.
        if (store->getParentRegion()->isAncestor(load->getParentRegion())) {
          load.replaceAllUsesWith(store.getValue());
          load.erase();
        } else {
          leftover.push_back(load);
        }
      }
      // If there no leftover ops, then the promise is not pending anymore.
      if (leftover.empty())
        requiredPromises.erase(it);
      else
        it->second = std::move(leftover);
      store.erase();
      return;
    }

    auto computeRequiredCaptures =
        [&requiredPromises](Operation *op, CallGraphNode *calleeNode) {
          ImplicitLocOpBuilder b(op->getLoc(), OpBuilder(op));
          // Create new loads to keep the state. Because the walk uses an early
          // inc, it will not visit the loads twice.
          SmallVector<Value> captures;
          for (auto [name, type] : calleeNode->requiredPromises) {
            auto load = b.create<POP::CompilerGlobalLoadOp>(type, name);
            requiredPromises[name].push_back(load);
            captures.push_back(load);
          }
          return captures;
        };

    // When a call is encountered, look up the required promises of the
    // function it is calling. Rewrite the call to provide them.
    if (auto call = dyn_cast<KGENCallOpInterface>(op)) {
      auto symbol = cast<SymbolConstantAttr>(call.getCallee());
      // Calls to functions that are not capturing cannot have captures.
      if (!symbol.getType().isCapturing())
        return;
      auto callee =
          symtab.lookup<FuncOp>(symbol.getSymbol().getRootReference());
      assert(callee);
      CallGraphNode *calleeNode = &nodes.find(callee)->second;

      // Exit early if there is nothing to do.
      if (calleeNode->requiredPromises.empty())
        return;

      SmallVector<Value> captures = computeRequiredCaptures(call, calleeNode);
      // Prepend the captures and update the callee signature on the call. The
      // function already has the updated signature. `kgen.create_closure`
      // applies arguments from the front, so we cannot append.
      call->insertOperands(0, captures);
      call.setCalleeAttr(
          SymbolConstantAttr::get(symbol.getSymbol(), callee.getSignature()));
      return;
    }

    if (auto create = dyn_cast<CaptureListCreateOp>(op)) {
      CallGraphNode *closureNode = getCalleeNode(create.getCallee());
      SmallVector<Value> captures = computeRequiredCaptures(op, closureNode);
      resolveCaptureListCreate(create, targetInfo, captures);
      return;
    }

    if (auto copy = dyn_cast<CaptureListCopyOp>(op)) {
      CallGraphNode *closureNode = getCalleeNode(copy.getCallee());
      SmallVector<Type> captures = llvm::to_vector(
          llvm::make_second_range(closureNode->requiredPromises));
      resolveCaptureListCopy(copy, targetInfo, captures);
      return;
    }

    if (auto expand = dyn_cast<CaptureListExpandOp>(op)) {
      resolveCaptureListExpand(expand, node, requiredPromises);
      return;
    }
  });

  // At the end of the walk, assess the leftover required promises. Prepend
  // them to the signature and block arguments.
  Block *body = func.getBody();

  unsigned i = 0;
  for (auto &[name, loads] : requiredPromises) {
    assert(!loads.empty() && "no requested loads?");
    Type type = loads.front().getType();
    node->requiredPromises.emplace_back(name, type);
    Value arg = body->insertArgument(i++, type, func.getLoc());
    for (POP::CompilerGlobalLoadOp load : loads) {
      load.replaceAllUsesWith(arg);
      load.erase();
    }
  }
  SignatureType sig = func.getSignature();
  // TODO: what conventions should we use here?
  SmallVector<ArgConvention> convs(i, ArgConvention::None);
  convs.append(sig.getArgConventions().begin(), sig.getArgConventions().end());
  assert(body->getNumArguments() == convs.size());

  // Update the function signature.
  auto fnType = FunctionType::get(func.getContext(), body->getArgumentTypes(),
                                  func.getResultTypes());
  func.setSignature(SignatureType::get(fnType, convs, sig.getFnEffects()));

  // HACK HACK HACK https://github.com/modularml/modular/issues/22959
  // HACK: If captures went up to an exported function, propagate them through
  // the ABI boundary by encoding the capture names on the function.
  if (func.isExported() && !node->requiredPromises.empty()) {
    SmallVector<StringAttr> captures =
        llvm::to_vector(llvm::make_first_range(node->requiredPromises));
    func->setAttr("kgen.cross_device_captures",
                  StringArrayAttr::get(func.getContext(), captures));
  }

  // Now go schedule all the nodes that have been made available.
  for (CallGraphNode *node : node->callers)
    if (++node->numProcessedCalls == node->callsites.size())
      worklist.fork([this, node] { resolvePromises(node); });
}

void CallGraph::run() {
  for (CallGraphNode &node : llvm::make_second_range(nodes))
    if (node.callsites.empty())
      worklist.fork([node = &node, this] { resolvePromises(node); });
  worklist.join();

  // FIXME: Actually, this pass should be able to handle cycles. I.e., the
  // following should work:
  //
  // ```mlir
  // kgen.func @recursive_closure() capturing {
  //   %0 = pop.compiler.global_load "var" : index
  //   use(%0)
  //   pop.compiler.global_store "var", %0 : index
  //   kgen.call @recursive_closure()
  // }
  // ```
  //
  // Fixed-point iterating on this should yield:
  //
  // ```mlir
  // kgen.func @recursive_closure(%x: index) capturing {
  //   use(%x)
  //   pop.compiler.global_store "var", %x : index
  //   kgen.call @recursive_closure()
  // }
  // ```
  //
  // And the next iteration will converge:
  //
  // ```mlir
  // kgen.func @recursive_closure(%x: index) capturing {
  //   use(%x)
  //   kgen.call @recursive_closure(%x)
  // }
  //
  // In order to do this without killing compile time, the fixed-point iteration
  // should be bound to SCCs. That is, the pass needs to organize CG nodes into
  // SCCs and then resolve the SCCs as a DAG, where each SCC is resolved by
  // fixed-point iterating until convergence.
}

void ResolveCompilerPromisesPass::runOnOperation() {
  CompilerTimeTraceScope traceScope(
      "ResolveCompilerPromisesPass::runOnOperation");
  auto rt =
      ConditionallyOwnedPointer<LLCL::Runtime>::takeIfNeeded(runtime, []() {
        return LLCL::createUniqueRuntime(LLCL::RuntimeOptions().forDebug())
            .release();
      });
  const SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

  CallGraph cg(*rt, symtab, getTargetInfo(getOperation()));
  cg.build(getOperation(), symtab);
  cg.run();
}
