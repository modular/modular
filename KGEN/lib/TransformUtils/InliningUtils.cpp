//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/TransformUtils/InliningUtils.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/Support/CompilerProfiling.h"
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

std::pair<Operation *, bool> KGEN::inlineRegion(mlir::RewriterBase &b,
                                                IRMapping &map, Operation *call,
                                                Region &region, bool takeBody) {
  // NOTE: All IR mutation must pass through the `RewriterBase`.
  // In-place mutation to `scope` is okay because it's a new operation.
  Operation *scope;
  std::function<void(Operation *)> handleReturn;

  // If the operation defines a call interface, use it to prepare inlining.
  if (auto itf = dyn_cast<KGENCallOpInterface>(call)) {
    FailureOr<InlineResult> result = itf.prepInline(b);
    if (failed(result)) {
      llvm::report_fatal_error("unexpected failure in inlining of call '" +
                               call->getName().getStringRef() +
                               "' -- please file a bug!");
    }
    std::tie(scope, handleReturn) = std::move(*result);
  } else {
    // Otherwise, assume this is inlining a direct call.
    StringAttr label = b.getStringAttr("inlined_cf_scope");
    scope = b.create<HLCF::LoopOp>(call->getLoc(), call->getResultTypes(),
                                   ValueRange(), label);
    handleReturn = [label, &b](Operation *op) {
      b.replaceOpWithNewOp<HLCF::BreakOp>(op, op->getOperands(), label);
    };
  }

  // Update the location if the scope defines a subprogram.
  if (auto inlinedSubScoped =
          dyn_cast<DebugInfo::InlinedSubprogramScoped>(scope)) {
    inlinedSubScoped->setLoc(region.getParentOp()->getLoc());
    inlinedSubScoped.setCallLocAttr(call->getLoc());
  }

  Region &scopeBody = scope->getRegion(0);
  bool returnAtEnd = isa<ReturnOp>(region.front().getTerminator());
  if (takeBody) {
    b.inlineRegionBefore(region, scopeBody, scopeBody.end());
    for (auto [value, arg] :
         llvm::zip(call->getOperands(), scopeBody.getArguments()))
      b.replaceAllUsesWith(arg, value);
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
      handleReturn(op);
      ++numReturns;
      return WalkResult::skip();
    }
    if (isa<FunctionLike>(op))
      return WalkResult::skip();

    if (auto sourceLocOp = dyn_cast<SourceLocOp>(op))
      processSourceLocOp(sourceLocOp, call->getLoc(), b);

    return WalkResult::advance();
  });
  b.replaceOp(call, scope->getResults());
  return std::make_pair(scope, numReturns == 1 && returnAtEnd);
}

//===----------------------------------------------------------------------===//
// processSourceLocOp
//===----------------------------------------------------------------------===//

void KGEN::processSourceLocOp(SourceLocOp sourceLocOp, Location callLoc,
                              mlir::RewriterBase &b) {
  // The inline count is decremented until it reaches 0. When that happens, we
  // capture the caller's location, and replace the op.
  if (auto &props = sourceLocOp.getProperties();
      int64_t inlineCount = props.getInlineCount()) {
    b.modifyOpInPlace(sourceLocOp,
                      [&] { props.setInlineCount(inlineCount - 1); });
    return;
  }

  // Extract the source location, even in the presence of debuginfo.
  FileLineColLoc fileLoc = DebugInfo::extractSourceLoc(callLoc);
  Location opLoc = sourceLocOp.getLoc();

  // Replace the source location op with constants.
  b.setInsertionPoint(sourceLocOp);
  b.replaceAllUsesWith(
      sourceLocOp.getLine(),
      b.create<ParamConstantOp>(opLoc, b.getIndexAttr(fileLoc.getLine())));
  b.replaceAllUsesWith(
      sourceLocOp.getCol(),
      b.create<ParamConstantOp>(opLoc, b.getIndexAttr(fileLoc.getColumn())));
  b.replaceAllUsesWith(
      sourceLocOp.getFileName(),
      b.create<ParamConstantOp>(
          opLoc, StringAttr::get(fileLoc.getFilename().getValue(),
                                 b.getType<StringType>())));
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

void KGEN::updateScopeDebugInfoFrom(Operation *scope, IntegerAttr tag,
                                    StringAttr updateAttrName) {
  // Unpack the bits.
  auto value = static_cast<uint8_t>(tag.getInt());
  auto singleExit = static_cast<bool>(value & 1);

  // The scope operations contains the location of the call.
  Region &body = scope->getRegion(0);
  Location callLoc = scope->getLoc();

  bool insideInlinedSubprogram = isa<DebugInfo::InlinedSubprogramScoped>(scope);
  body.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Inline the location if not inside an inlined subprogram.
    if (!insideInlinedSubprogram)
      DebugInfo::updateInlinedLoc(op, callLoc);

    // Don't recurse into nested functions.
    if (isa<FuncInterface>(op))
      return WalkResult::skip();

    // Recurse into the body if needed and allowed.
    if (isa<DebugInfo::InlinedSubprogramScoped>(op)) {
      // Recurse inside if the inlined subprogram has a tag (deferred update).
      IntegerAttr tag;
      if (updateAttrName &&
          (tag = op->getAttrOfType<IntegerAttr>(updateAttrName)))
        updateScopeDebugInfoFrom(op, tag, updateAttrName);

      // Always skip walking directly into subprogram scopes.
      return WalkResult::skip();
    } else if (updateAttrName && isa<HLCF::LoopOp>(op)) {
      if (auto tag = op->getAttrOfType<IntegerAttr>(updateAttrName)) {
        updateScopeDebugInfoFrom(op, tag, updateAttrName);
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
    if (!isa<HLCF::LoopOp, FunctionLike>(op))
      return WalkResult::advance();
    auto tag = op->getAttrOfType<IntegerAttr>(updateAttrName);
    if (!tag)
      return WalkResult::advance();

    updateScopeDebugInfoFrom(op, tag, updateAttrName);
    return WalkResult::skip();
  });
}

void KGEN::maybeUpdateDebugInfo(Operation *scope,
                                std::optional<StringAttr> updateAttrName,
                                bool singleExit) {
  if (updateAttrName) {
    // We don't know where the op will end up, so tag it with an attribute.
    // Encode information {singleExit} as bits.
    IntegerAttr tag =
        OpBuilder(scope->getContext()).getI8IntegerAttr(singleExit);
    if (*updateAttrName) {
      // Deferred debuginfo update.
      scope->setAttr(*updateAttrName, tag);
    } else {
      // Immediate debuginfo update.
      // This will also foldTrivialLoops if applicable.
      updateScopeDebugInfoFrom(scope, tag, nullptr);
    }
  } else if (singleExit) {
    foldTrivialLoop(scope);
  }
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
                  ? ctx->getThreadPool().getMaxConcurrency()
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
