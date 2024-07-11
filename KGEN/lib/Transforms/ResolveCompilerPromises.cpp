//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Support/ForkJoin.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "KGEN/TransformUtils/SCCUtils.h"
#include "KGEN/TransformUtils/Walkers.h"
#include "Support/Context.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/SCCIterator.h"

using namespace M;
using namespace KGEN;

namespace {

//===----------------------------------------------------------------------===//
// CallLikeOp
//===----------------------------------------------------------------------===//

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
  Operation *operator->() { return op; }

  /// Required CallGraph interface.
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

//===----------------------------------------------------------------------===//
// CallGraphNode
//===----------------------------------------------------------------------===//

/// A basic callgraph node, containing state for this dataflow analysis across
/// the callgraph.
struct CallGraphNode : public SCCNode<CallGraphNode, FuncOp, CallLikeOp> {
  using SCCNode::SCCNode;

  /// The current set of required promises for the function.
  llvm::MapVector<StringAttr, std::pair<Type, unsigned>> requiredPromises;
};

//===----------------------------------------------------------------------===//
// CallGraph
//===----------------------------------------------------------------------===//

struct CallGraph : public SCCGraph<CallGraph, CallGraphNode> {
  CallGraph(const SymbolTable &symtab, TargetInfoAttr target)
      : symtab(symtab), target(target) {}

  /// Only add nodes to the graph that are to functions that are capturing,
  /// since those are the only functions we need to handle.
  bool shouldAddToGraph(CallLikeOp call, CallGraphNode *node) {
    return node->func.getSignature().isCapturing() ||
           isa<CaptureListCreateOp, CaptureListCopyOp>(*call);
  }

  /// Lookup the call graph node for the given symbol reference.
  CallGraphNode *getCalleeNode(TypedAttr symbol) {
    auto callee = symtab.lookup<FuncOp>(
        cast<SymbolConstantAttr>(symbol).getSymbol().getRootReference());
    assert(callee);
    return &nodes.find(callee)->second;
  }

  /// Run an iteration of the analysis and transformation on a single node.
  /// Return true if anything changed.
  bool doAnalysis(CallGraphNode *node);
  void doRewrite(const CallGraphNode *node);

  const SymbolTable &symtab;
  TargetInfoAttr target;
};
} // namespace

/// Propagate from `kgen.capture_list.create` the set of required promises from
/// the callee into the current function. Query them and pack them into a
/// heap-allocated slot.
static void resolveCaptureListCreate(CaptureListCreateOp op,
                                     TargetInfoAttr target) {
  ValueRange captures = op->getOperands();

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
static void resolveCaptureListCopy(CaptureListCopyOp op,
                                   CallGraphNode *calleeNode,
                                   TargetInfoAttr target) {
  auto clType = StructType::get(
      op.getContext(),
      llvm::map_to_vector(calleeNode->requiredPromises,
                          [](auto &promise) { return promise.second.first; }));

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
static void resolveCaptureListExpand(CaptureListExpandOp op) {
  ImplicitLocOpBuilder b(op->getLoc(), OpBuilder(op));
  b.setInsertionPoint(op);

  // Compute the capture list type based on the set of required promises.
  auto clType =
      StructType::get(op.getContext(), llvm::to_vector(op->getResultTypes()));
  Value captures = b.create<POP::PointerBitcastOp>(PointerType::get(clType),
                                                   op.getCaptureList());

  for (auto [idx, value] : llvm::enumerate(op->getResults())) {
    // Extract each captured value out of the capture state and use it as
    // input to the call of the closure.
    Value promise = b.create<POP::LoadOp>(b.create<StructGEPOp>(captures, idx));
    value.replaceAllUsesWith(promise);
  }

  // Clear the required promises set. It has been fully satisfied by this op.
  op.erase();
}

void CallGraph::doRewrite(const CallGraphNode *node) {
  FuncOp func = node->func;
  func.walk([this](Operation *op) {
    if (isa<POP::CompilerGlobalStoreOp>(op))
      op->erase();
    else if (auto create = dyn_cast<CaptureListCreateOp>(op))
      resolveCaptureListCreate(create, target);
    else if (auto copy = dyn_cast<CaptureListCopyOp>(op))
      resolveCaptureListCopy(copy, getCalleeNode(copy.getCallee()), target);
    else if (auto expand = dyn_cast<CaptureListExpandOp>(op))
      resolveCaptureListExpand(expand);
  });
}

bool CallGraph::doAnalysis(CallGraphNode *node) {
  FuncOp func = node->func;
  llvm::MapVector<StringAttr, SmallVector<POP::CompilerGlobalLoadOp>>
      requiredPromises;
  unsigned curNumPromises = node->requiredPromises.size();

  VerboseCompilerTimeTraceScope traceScope(
      "resolvePromises", [func]() mutable { return func.getSymName().str(); });

  // This functor will, given an operation that points to another node, create
  // new required promises based on any additional required promises from the
  // callee.
  auto computeRequiredCaptures = [&requiredPromises](Operation *op,
                                                     CallGraphNode *calleeNode,
                                                     unsigned fulfilled) {
    ImplicitLocOpBuilder b(op->getLoc(), OpBuilder(op));
    // Create new loads to keep the state. Because the walk uses an early
    // inc, it will not visit the loads twice.
    SmallVector<Value> captures;
    for (auto [name, type] :
         llvm::drop_begin(calleeNode->requiredPromises, fulfilled)) {
      auto load = b.create<POP::CompilerGlobalLoadOp>(type.first, name);
      requiredPromises[name].push_back(load);
      captures.push_back(load);
    }
    return captures;
  };

  /// This functor will, given the current set of required promises, transfer
  /// any new ones to the enclosing function.
  auto consumeRequiredPromises = [node,
                                  &requiredPromises](ValueRange fulfilled) {
    SmallVector<std::pair<Type, SmallVector<POP::CompilerGlobalLoadOp>>>
        newTypes;
    for (auto &[name, loads] : requiredPromises) {
      // If this required promise is already fulfilled on the node, then replace
      // the request immediately.
      if (auto it = node->requiredPromises.find(name);
          it != node->requiredPromises.end()) {
        auto [type, index] = it->second;
        for (POP::CompilerGlobalLoadOp load : loads) {
          load.replaceAllUsesWith(fulfilled[index]);
          load.erase();
        }
        continue;
      }
      // Otherwise, save the request for the new promise.
      Type type = loads.front().getType();
      node->requiredPromises.insert(
          {name, {type, node->requiredPromises.size()}});
      newTypes.emplace_back(type, std::move(loads));
    }
    // Consume the current state.
    requiredPromises.clear();
    return newTypes;
  };

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
      if (it == requiredPromises.end())
        return;

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
      return;
    }

    // When a call is encountered, look up the required promises of the
    // function it is calling. Rewrite the call to provide them.
    if (auto call = dyn_cast<KGENCallOpInterface>(op)) {
      auto symbol = cast<SymbolConstantAttr>(call.getCallee());
      SignatureType sig = symbol.getType();
      // Calls to functions that are not capturing cannot have captures.
      if (!sig.isCapturing())
        return;
      CallGraphNode *calleeNode = getCalleeNode(symbol);
      FuncOp callee = calleeNode->func;

      // Exit early if there is nothing to do.
      if (calleeNode->requiredPromises.empty())
        return;

      unsigned fulfilled =
          calleeNode->requiredPromises.size() -
          (calleeNode->func.getNumArguments() - sig.getNumArguments());
      SmallVector<Value> captures =
          computeRequiredCaptures(call, calleeNode, fulfilled);
      // Prepend the captures and update the callee signature on the call. The
      // function already has the updated signature. `kgen.create_closure`
      // applies arguments from the front, so we cannot append.
      call->insertOperands(0, captures);
      call.setCalleeAttr(
          SymbolConstantAttr::get(symbol.getSymbol(), callee.getSignature()));
      return;
    }

    if (auto create = dyn_cast<CaptureListCreateOp>(op)) {
      CallGraphNode *calleeNode = getCalleeNode(create.getCallee());
      SmallVector<Value> captures =
          computeRequiredCaptures(create, calleeNode, create->getNumOperands());
      create->insertOperands(0, captures);
      return;
    }

    if (auto expand = dyn_cast<CaptureListExpandOp>(op)) {
      // Consume all promises.
      auto newTypes = consumeRequiredPromises(expand->getResults());
      if (newTypes.empty())
        return;

      // Now fulfill any new promises by resizing the results on the op.
      OperationState state(expand.getLoc(), op->getName(), op->getOperands(),
                           op->getResultTypes());
      for (auto [type, _] : newTypes)
        state.types.push_back(type);

      OpBuilder b(op);
      Operation *newOp = b.create(state);
      op->replaceAllUsesWith(
          llvm::drop_end(newOp->getResults(), newTypes.size()));
      for (auto [newPromise, loads] :
           llvm::zip(llvm::drop_begin(newOp->getResults(), op->getNumResults()),
                     newTypes)) {
        for (POP::CompilerGlobalLoadOp load : loads.second) {
          load.replaceAllUsesWith(newPromise);
          load.erase();
        }
      }
      op->erase();
      return;
    }
  });

  if (!func.getSignature().isCapturing())
    return node->requiredPromises.size() != curNumPromises;

  // At the end of the walk, assess the leftover required promises. Prepend
  // them to the signature and block arguments.
  auto newTypes = consumeRequiredPromises(
      func.getArguments().take_back(node->requiredPromises.size()));
  if (newTypes.empty())
    return false;

  Block *body = func.getBody();
  unsigned i = 0;
  for (auto &[type, loads] : newTypes) {
    Value arg = body->insertArgument(i++, type, func.getLoc());
    for (POP::CompilerGlobalLoadOp load : loads) {
      load.replaceAllUsesWith(arg);
      load.erase();
    }
  }

  SignatureType sig = func.getSignature();
  // TODO: What conventions do we use for captures.
  SmallVector<ArgConvention> convs(i, ArgConvention::BorrowedInReg);
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

  return true;
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_RESOLVECOMPILERPROMISES
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct ResolveCompilerPromisesPass
    : impl::ResolveCompilerPromisesBase<ResolveCompilerPromisesPass> {
  void runOnOperation() override;
};
} // namespace

void ResolveCompilerPromisesPass::runOnOperation() {
  VerboseCompilerTimeTraceScope traceScope(
      "ResolveCompilerPromisesPass::runOnOperation");
  const SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

  LLCL::Runtime &runtime = *loadContext(&getContext())->get<LLCL::Runtime>();
  CallGraph cg(symtab, getTargetInfo(getOperation()));
  cg.build(getOperation(), symtab);
  cg.run(runtime);
}
