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
struct CallGraphNode
    : public CallGraphNodeBase<CallGraphNode, FuncOp, KGENCallOpInterface> {
  using CallGraphNodeBase::CallGraphNodeBase;

  std::vector<std::pair<StringAttr, Type>> requiredPromises;
};

struct CallGraph : public CallGraphBase<CallGraph, CallGraphNode> {
  explicit CallGraph(LLCL::Runtime &runtime, const SymbolTable &symtab)
      : worklist(runtime), symtab(symtab) {}

  /// It turns out we have to rely on propagation of the `capturing` bit on
  /// functions in the parser, where a function with a `capturing` signature in
  /// its input or result parameters also needs to be marked as `capturing`,
  /// because otherwise `kgen.create_closure` does not know whether to create a
  /// function pointer or a closure with a capture list.
  ///
  /// In that case, we can also rely on the bit to form the edges in the graph
  /// to be processed, circumventing the issue with cycles.
  bool shouldInline(CallGraphNode *node) {
    return node->func.getSignature().isCapturing();
  }

  /// Process a single function. Resolve all the promises in the function by
  /// processing `pop.compiler.global_load` and `pop.compiler.global_store` ops,
  /// prepending arguments if necessary.
  void resolvePromises(CallGraphNode *node);

  /// Initialize the worklist and run to completion.
  void run();

  LLCL::ForkJoin worklist;
  const SymbolTable symtab;
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

      // Create new loads to keep the state. Because the walk uses an early
      // inc, it will not visit the loads twice.
      SmallVector<Value> captures;
      ImplicitLocOpBuilder b(call.getLoc(), OpBuilder(call));
      for (auto [name, type] : calleeNode->requiredPromises) {
        auto load = b.create<POP::CompilerGlobalLoadOp>(type, name);
        requiredPromises[name].push_back(load);
        captures.push_back(load);
      }

      // Prepend the captures and update the callee signature on the call. The
      // function already has the updated signature. `kgen.create_closure`
      // applies arguments from the front, so we cannot append.
      call->insertOperands(0, captures);
      call.setCalleeAttr(
          SymbolConstantAttr::get(symbol.getSymbol(), callee.getSignature()));
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
  SmallVector<ValueInputConvention> convs(i, ValueInputConvention::None);
  convs.append(sig.getInputConventions().begin(),
               sig.getInputConventions().end());
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
  auto rt = ConditionallyOwnedPointer<LLCL::Runtime>::allocateIfNeeded(
      runtime, LLCL::createLeakCheckAllocator(LLCL::createMallocAllocator()),
      LLCL::createSingleThreadWorkQueue());
  const SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

  CallGraph cg(*rt, symtab);
  cg.build(getOperation(), symtab);
  cg.run();
}
