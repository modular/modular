//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "InliningUtils.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/Compiler/TimeProfilerTimingManager.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// inlineRegion
//===----------------------------------------------------------------------===//

std::pair<Operation *, bool> KGEN::inlineRegion(IRMapping &map,
                                                KGENCallOpInterface call,
                                                Region &region, bool takeBody) {
  StringAttr label = StringAttr::get(call.getContext(), "inlined_cf_scope");

  mlir::IRRewriter b{OpBuilder(call)};
  Operation *scope;
  if (isa<CallOp>(&*call)) {
    scope = b.create<HLCF::LoopOp>(call.getLoc(), call->getResultTypes(),
                                   ValueRange(), label);
  } else if (auto asyncCall = dyn_cast<LIT::AsyncCallOp>(&*call)) {
    // Nested function-like op should retain scoped location of the callee.
    scope = b.create<LIT::AsyncExecuteOp>(region.getParentOp()->getLoc(),
                                          asyncCall.getType(), call.getLoc());
  } else if (auto createClosure = dyn_cast<CreateClosureOp>(&*call)) {
    // Nested function-like op should retain scoped location of the callee.
    scope = b.create<StageClosureOp>(region.getParentOp()->getLoc(),
                                     createClosure.getType(), call.getLoc());
  } else {
    llvm::report_fatal_error("unknown call operation '" +
                             call->getName().getStringRef() +
                             "' in inlining pass -- please file a bug!");
  }

  Region &scopeBody = scope->getRegion(0);
  bool returnAtEnd = isa<ReturnOp>(region.front().getTerminator());
  if (takeBody) {
    scopeBody.takeBody(region);
    for (auto [value, arg] :
         llvm::zip(call->getOperands(), scopeBody.getArguments()))
      arg.replaceAllUsesWith(value);
    scopeBody.front().eraseArguments(0, call->getNumOperands());
  } else {
    Block *block = b.createBlock(&scopeBody);
    for (auto [value, arg] :
         llvm::zip(call->getOperands(), region.getArguments()))
      map.map(arg, value);
    for (BlockArgument trailing :
         region.getArguments().drop_front(call->getNumOperands()))
      map.map(trailing,
              block->addArgument(trailing.getType(), trailing.getLoc()));
    for (Operation &op : region.getOps())
      b.clone(op, map);
  }

  unsigned numReturns = 0;
  scopeBody.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<ReturnOp>(op)) {
      b.setInsertionPoint(op);
      if (isa<CallOp>(&*call)) {
        b.replaceOpWithNewOp<HLCF::BreakOp>(op, op->getOperands(), label);
      } else if (isa<CreateClosureOp, LIT::AsyncCallOp>(*&call)) {
        // Just `return` is ok.
      } else {
        llvm::report_fatal_error("unknown call operation '" +
                                 call->getName().getStringRef() +
                                 "' in inlining pass -- please file a bug!");
      }

      ++numReturns;
      return WalkResult::skip();
    }
    if (isa<LIT::AsyncExecuteOp, StageClosureOp>(op))
      return WalkResult::skip();
    return WalkResult::advance();
  });
  b.replaceOp(call, scope->getResults());
  return std::make_pair(scope, numReturns == 1 && returnAtEnd);
}

//===----------------------------------------------------------------------===//
// foldTrivialLoop
//===----------------------------------------------------------------------===//

void KGEN::foldTrivialLoop(Operation *op) {
  CompilerTimeTraceScope traceScope("foldTrivialLoop");

  auto loop = dyn_cast<HLCF::LoopOp>(op);
  if (!loop)
    return;

  mlir::IRRewriter b{OpBuilder(op)};

  Block &body = loop.getBody().front();
  Operation *term = body.getTerminator();
  b.inlineBlockBefore(&body, loop);
  b.replaceOp(loop, term->getOperands());
  b.eraseOp(term);
}

//===----------------------------------------------------------------------===//
// updateScopeDebugInfo
//===----------------------------------------------------------------------===//

/// Starting from an inlining scope, update debug information as appropriate and
/// fold the scope if requested. Recurse on nested scopes.
static void updateScopeDebugInfoFrom(Operation *scope, IntegerAttr tag,
                                     StringAttr updateAttrName,
                                     bool stripValues) {
  // Unpack the bits.
  auto value = static_cast<uint8_t>(tag.getInt());
  auto singleExit = static_cast<bool>(value & 1);
  auto noDebug = static_cast<bool>(value >> 1);

  // The scope operations contains the location of the call.
  Region &body = scope->getRegion(0);
  Location callLoc = scope->getLoc();

  bool scopeIsNotSubprogram = !isa<DebugInfo::SubprogramScoped>(scope);
  body.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Erase `debuginfo.value` operations when inlining without debug info.
    if ((noDebug || stripValues) && isa<DebugInfo::ValueOp>(op)) {
      op->erase();
      return WalkResult::skip();
    }

    // Inline the location if needed.
    if (noDebug || scopeIsNotSubprogram)
      DebugInfo::updateInlinedLoc(op, callLoc, noDebug);

    // Recurse into the body if needed and allowed.
    if (isa<HLCF::LoopOp>(op)) {
      if (auto tag = op->getAttrOfType<IntegerAttr>(updateAttrName)) {
        updateScopeDebugInfoFrom(op, tag, updateAttrName, stripValues);
        return WalkResult::skip();
      }
    } else if (isa<LIT::AsyncExecuteOp, StageClosureOp>(op)) {
      if (auto tag = op->getAttrOfType<IntegerAttr>(updateAttrName)) {
        updateScopeDebugInfoFrom(op, tag, updateAttrName,
                                 isa<FileLineColLoc>(op->getLoc()));
        return WalkResult::skip();
      }
    }
    return WalkResult::advance();
  });

  // If this scope is a trivial control-flow scope, fold it away.
  if (singleExit)
    foldTrivialLoop(scope);
}

void KGEN::updateScopeDebugInfo(FuncOp func, StringAttr updateAttrName) {
  CompilerTimeTraceScope updateScopeDebugInfo(
      "updateScopeDebugInfo", [&func] { return func.getSymName().str(); });
  func.getBody()->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (!isa<HLCF::LoopOp, LIT::AsyncExecuteOp, StageClosureOp>(op))
      return WalkResult::advance();
    auto tag = op->getAttrOfType<IntegerAttr>(updateAttrName);
    if (!tag)
      return WalkResult::advance();

    // If the surrounding function lacks debug info, then debug value operations
    // have no anchor. Erase them.
    Location surroundingFuncLoc =
        isa<HLCF::LoopOp>(op) ? func->getLoc() : op->getLoc();
    updateScopeDebugInfoFrom(op, tag, updateAttrName,
                             isa<FileLineColLoc>(surroundingFuncLoc));
    return WalkResult::skip();
  });
}

//===----------------------------------------------------------------------===//
// PerThreadPassManager
//===----------------------------------------------------------------------===//

/// This class manages a pass manager instance for each thread.
PerThreadPassManagers::PerThreadPassManagers(
    MLIRContext *ctx, function_ref<void(mlir::OpPassManager &)> buildFuncPasses)
    : ctx(ctx), buildFuncPasses(buildFuncPasses) {
  // Reserve the thread-local cache map so that it never resizes.
  pms.reserve(ctx->isMultithreadingEnabled()
                  ? ctx->getThreadPool().getThreadCount()
                  : 1);
}

/// Get the pass manager for the current thread, initializing it if one does
/// not exist.
mlir::PassManager &PerThreadPassManagers::getPassManager() {
  int64_t threadId = llvm::get_threadid();
  {
    llvm::sys::SmartScopedReader<true> lock(mutex);
    if (auto it = pms.find(threadId); it != pms.end())
      return *it->second;
  }

  // Emplace a new pass manager for this thread.
  mutex.lock();
  mlir::PassManager &pm =
      *pms.try_emplace(threadId, std::make_unique<mlir::PassManager>(
                                     ctx, FuncOp::getOperationName()))
           .first->second;
  mutex.unlock();

  // Initialize the pass manager.
  buildFuncPasses(pm);
  pm.enableVerifier(false);
  // Enable time tracing on the nested pass manager.
  pm.enableTiming(std::make_unique<TimeProfilerTimingManager>());
  return pm;
}

uint64_t KGEN::getNumOperations(Operation *op) {
  if (!op)
    return 0;

  uint64_t result = 0;
  op->walk([&](Operation *) { ++result; });
  return result;
}
