//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENPasses.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "LLVMLoweringUtils.h"
#include "Support/Compiler/OperationUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace M;
using namespace KGEN;
using namespace mlir::LLVM;

//===----------------------------------------------------------------------===//
// TweakSpilledAllocas
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_TWEAKSPILLEDALLOCAS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct TweakSpilledAllocasPass
    : public KGEN::impl::TweakSpilledAllocasBase<TweakSpilledAllocasPass> {
  using TweakSpilledAllocasBase::TweakSpilledAllocasBase;

  void runOnOperation() override;
};
} // namespace

/// In a CFG representation, we want to strip lifetimes where operations
/// dominate like `alloca > await > lifetime.end`. In a region representation,
/// we can get the right order with a forward traversal, since control-flow can
/// cross between parent and child regions but not jump across operations.
static LogicalResult tweakSpilledAllocas(Operation *func) {
  enum SpillKind {
    /// Alloca is not spilled.
    NOT_SPILLED,
    /// Alloca is spilled while live.
    SPILLED_LIVE,
    /// Alloca is spilled while dead.
    SPILLED_DEAD
  };
  struct AllocaInfo {
    LifetimeStartOp start = nullptr;
    SpillKind spill = SpillKind::NOT_SPILLED;
  };
  DenseMap<AllocaOp, AllocaInfo> allocas;
  auto processOp = [&](Operation *op) -> LogicalResult {
    // Be defensive about invalid IR.
    if (auto alloca = dyn_cast<AllocaOp>(op)) {
      allocas.insert({alloca, {}});

    } else if (auto start = dyn_cast<LifetimeStartOp>(op)) {
      auto alloca =
          dyn_cast_or_null<AllocaOp>(start.getOperand().getDefiningOp());
      if (LLVM_UNLIKELY(!alloca))
        return start.emitOpError("operand not defined by an `llvm.alloca`");
      auto it = allocas.find(alloca);
      if (LLVM_UNLIKELY(it == allocas.end() || it->second.start))
        return start.emitOpError(
            "duplicate `llvm.intr.lifetime.start` marker for alloca");
      it->second.start = start;

    } else if (auto end = dyn_cast<LifetimeEndOp>(op)) {
      auto alloca =
          dyn_cast_or_null<AllocaOp>(end.getOperand().getDefiningOp());
      if (LLVM_UNLIKELY(!alloca))
        return end.emitOpError("operand not defined by an `llvm.alloca`");
      auto it = allocas.find(alloca);
      if (LLVM_UNLIKELY(it == allocas.end() || !it->second.start))
        return it->first.emitOpError(
            "alloca with no `llvm.intr.lifetime.start` marker");
      switch (it->second.spill) {
      case SpillKind::NOT_SPILLED:
        break;
      case SpillKind::SPILLED_LIVE:
        // Alloca is spilled while live. Remove the lifetime markers.
        end.erase();
        it->second.start.erase();
        break;
      case SpillKind::SPILLED_DEAD:
        // Alloca is spilled while dead. Move the alloca to before the lifetime
        // start marker to take it off the coroutine frame.
        alloca->moveBefore(it->second.start);
        break;
      }
      // Deactivate the alloca.
      allocas.erase(it);

    } else if (auto await = dyn_cast<POP::CoroutineAwaitOp>(op)) {
      // All active allocas are spilled.
      for (auto &[alloca, info] : allocas) {
        // If the lifetime start marker of the alloca has been encountered, the
        // alloca is live while spilled.
        info.spill =
            info.start ? SpillKind::SPILLED_LIVE : SpillKind::SPILLED_DEAD;
      }
    }
    return success();
  };

  WalkResult result = func->walk([&](Operation *op) {
    if (failed(processOp(op)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

void TweakSpilledAllocasPass::runOnOperation() {
  if (failed(tweakSpilledAllocas(getOperation())))
    return signalPassFailure();
}

//===----------------------------------------------------------------------===//
// Coroutine Lowering
//===----------------------------------------------------------------------===//

namespace {
/// This struct contains "cached" types and attributes. These are types and
/// attributes that are commonly used throughout the lowerings. They are cached
/// in this struct so that they are not hashed multiple times.
struct TypeAttrCache {
  Type i1Type;
  Type i8Type;
  Type i32Type;
  Type i64Type;
  Type ptrType;
  Type i8PtrType;
  Type tokenType;
  Type asyncFnType;
};

/// This struct contains information about the machinery of a coroutine.
struct Coroutine {
  /// The suspend block. This block contains the call to the end intrinsic
  /// when the coroutine should be suspended but is not complete.
  Block *suspend;
  /// The cleanup block. This block deallocates any memory associated with the
  /// coroutine handle.
  Block *cleanup;
  /// The coroutine handle value.
  Value handle;
};
} // namespace

/// Get the LLVM type of the coroutine promise.
static LLVMStructType getCoroutinePromiseType(LLVMBuilder &b,
                                              const TypeAttrCache &cache,
                                              POP::CoroutineType coroType) {
  // Pack the result types into a struct.
  SmallVector<Type> promiseTypes;

  // Add the async context: the callback function and a context pointer.
  promiseTypes.push_back(LLVMStructType::getLiteral(
      b.getContext(), {cache.ptrType, cache.ptrType}));
  for (Type type : coroType.getResultTypes()) {
    type = b.convertType(type);
    if (!type)
      return {};
    promiseTypes.push_back(type);
  }
  return LLVMStructType::getLiteral(b.getContext(), promiseTypes);
}

/// Convert a function to a coroutine by generating the necessary LLVM coroutine
/// machinery: set up the promise, allocate the coroutine handle, and insert the
/// suspend and exit points.
static FailureOr<Coroutine>
createCoroutineFunction(LLVMBuilder &b, const TypeAttrCache &cache,
                        LLVMFuncOp func, POP::CoroutineType coroType) {
  // Create a new coroutine entry block that sets up the coroutine.
  auto *coroEntry = new Block;
  SmallVector<Value> coroEntryArgs;
  for (BlockArgument arg : func.getArguments())
    coroEntryArgs.push_back(
        coroEntry->addArgument(arg.getType(), arg.getLoc()));
  b.setInsertionPointToStart(coroEntry);

  // Initialize the coroutine promise. The coroutine promise contains a
  // coroutine context and space for the result storage.
  LLVMStructType promiseType = getCoroutinePromiseType(b, cache, coroType);
  if (!promiseType)
    return func.emitError("failed to convert coroutine type");
  // Explicitly specify the alignment.
  Value promiseMem = b.create<AllocaOp>(LLVMPointerType::get(promiseType),
                                        b.create<ConstantOp>(cache.i32Type, 1),
                                        b.getTypeABIAlign(promiseType));
  Value promiseMemI8 = b.create<BitcastOp>(cache.i8PtrType, promiseMem);

  // Initialize the coroutine frame.
  Value cstNullPtr = b.create<IntToPtrOp>(
      cache.i8PtrType, b.create<ConstantOp>(b.getIndexType(), 0));
  Value coroId =
      b.create<CoroIdOp>(cache.tokenType, b.create<CoroAlignOp>(cache.i32Type),
                         promiseMemI8, cstNullPtr, cstNullPtr);

  // Determine whether the coroutine requires dynamic memory allocation.
  Value needDynAlloc =
      b.create<CallIntrinsicOp>(b.getI1Type(), "llvm.coro.alloc", coroId)
          .getResult(0);
  auto *coroBegin = new Block;
  auto *coroAlloc = new Block;
  b.create<CondBrOp>(needDynAlloc, coroAlloc, coroBegin, cstNullPtr);

  // Allocate memory for the coroutine frame. Use `malloc` and `free` as the
  // coroutine frame allocators.
  // TODO: Switch this to an aligned alloc.
  b.setInsertionPointToStart(coroAlloc);
  Value coroSize = b.create<CoroSizeOp>(b.getIndexType());
  auto allocCall =
      b.create<POP::ExternalCallOp>(cache.ptrType, "malloc", coroSize);
  auto allocMemCast =
      b.create<BitcastOp>(cache.i8PtrType, allocCall.getResult(0));
  b.create<BrOp>(allocMemCast.getResult(), coroBegin);

  // Create the coroutine frame and handle.
  b.setInsertionPointToStart(coroBegin);
  Value coroMem = coroBegin->addArgument(cache.i8PtrType, b.getLoc());
  Value coroHdl = b.create<CoroBeginOp>(cache.i8PtrType, coroId, coroMem);

  // Immediately suspend the coroutine. This function returns a coroutine
  // suspended at its entry block so the caller can set up the async context.
  Value cstFalse = b.create<ConstantOp>(cache.i1Type, false);
  Value suspendState = b.create<CoroSuspendOp>(
      cache.i8Type, b.create<CoroSaveOp>(cache.tokenType, coroHdl), cstFalse);
  Region &body = func.getBody();
  Block *funcEntry = &body.front();
  auto *cleanup = new Block;
  auto *suspend = new Block;
  b.create<SwitchOp>(suspendState, suspend, ValueRange(),
                     ArrayRef<int32_t>{0, 1},
                     ArrayRef<Block *>{funcEntry, cleanup},
                     ArrayRef<ValueRange>{coroEntryArgs, {}});

  // In the suspend block, emit the coroutine end marker and return the handle.
  b.setInsertionPointToStart(suspend);
  b.create<CoroEndOp>(cache.i1Type, cstNullPtr, cstFalse);
  b.create<ReturnOp>(coroHdl);

  // In the cleanup block, check if we need to deallocate the frame memory.
  b.setInsertionPointToStart(cleanup);
  Value coroFreeMem = b.create<CoroFreeOp>(cache.i8PtrType, coroId, coroHdl);
  Value needDynFree =
      b.create<ICmpOp>(ICmpPredicate::ne, coroFreeMem, cstNullPtr);
  auto *cleanupFree = new Block;
  b.create<CondBrOp>(needDynFree, cleanupFree, suspend);
  b.setInsertionPointToStart(cleanupFree);
  b.create<POP::ExternalCallOp>(
      "free", b.create<BitcastOp>(cache.ptrType, coroFreeMem).getResult());
  b.create<BrOp>(suspend);

  // For each return in the function before it is transformed into a coroutine,
  // insert a final suspend.
  auto *trap = new Block;
  b.setInsertionPointToStart(trap);
  b.create<CallIntrinsicOp>(TypeRange(), "llvm.trap", ValueRange());
  b.create<UnreachableOp>();
  for (auto returnOp : llvm::make_early_inc_range(body.getOps<ReturnOp>())) {
    b.setInsertionPoint(returnOp);
    Value saveTok = b.create<CoroSaveOp>(cache.tokenType, coroHdl);
    Value finalSuspend = b.create<CoroSuspendOp>(
        cache.i8Type, saveTok, b.create<ConstantOp>(cache.i1Type, true));
    b.create<SwitchOp>(
        finalSuspend, suspend, ValueRange(), ArrayRef<int32_t>{0, 1},
        ArrayRef<Block *>{trap, cleanup}, ArrayRef<ValueRange>{{}, {}});
    returnOp->erase();
  }

  // Add all the blocks to the function body.
  body.push_front(coroBegin);
  body.push_front(coroAlloc);
  body.push_front(coroEntry);
  body.push_back(trap);
  body.push_back(cleanup);
  body.push_back(cleanupFree);
  body.push_back(suspend);

  // Tag the function with the required LLVM attribute.
  func.setPassthroughAttr(b.getArrayAttr(b.getStringAttr("presplitcoroutine")));

  // Return the coroutine machinery.
  return Coroutine{suspend, cleanup, coroHdl};
}

//===----------------------------------------------------------------------===//
// CoroutineAwaitOp

/// Lower a `pop.coroutine.await` to a non-final suspend and inline the region.
static void lowerCoroutineAwait(LLVMBuilder &b, const TypeAttrCache &cache,
                                const Coroutine &coroutine,
                                POP::CoroutineAwaitOp op) {
  b.setLoc(op.getLoc());

  // Split the block before the await operation. Insert the suspend point at the
  // end of the current block and resume at the new block.
  Block *cur = op->getBlock();
  Block *resume = op->getBlock()->splitBlock(op);
  b.setInsertionPointToEnd(cur);

  // Suspend the current coroutine.
  auto savePoint = b.create<CoroSaveOp>(cache.tokenType, coroutine.handle);

  // Inline the awaited region.
  for (Operation &op :
       llvm::make_early_inc_range(llvm::reverse(op.getBody().front())))
    op.moveAfter(savePoint);

  // Insert the suspension point. Resume after the operation.
  Value suspendState = b.create<CoroSuspendOp>(
      cache.i8Type, savePoint, b.create<ConstantOp>(cache.i1Type, false));
  b.create<SwitchOp>(suspendState, coroutine.suspend, ValueRange(),
                     ArrayRef<int32_t>{0, 1},
                     ArrayRef<Block *>{resume, coroutine.cleanup},
                     ArrayRef<ValueRange>{{}, {}});

  // In the resume block, load the coroutine results from the promise.
  b.setInsertionPointToStart(resume);
  op.erase();
}

//===----------------------------------------------------------------------===//
// CoroutinePromiseOp

/// Lower `pop.coroutine.promise` to get the promise of the coroutine with the
/// right type.
static LogicalResult lowerCoroutinePromise(LLVMBuilder &b,
                                           const TypeAttrCache &cache,
                                           POP::CoroutinePromiseOp op) {
  b.setLoc(op.getLoc());
  b.setInsertionPoint(op);
  Value hdl = b.create<mlir::UnrealizedConversionCastOp>(cache.i8PtrType,
                                                         op.getCoroutine())
                  .getResult(0);

  LLVMStructType promiseType =
      getCoroutinePromiseType(b, cache, op.getCoroutine().getType());
  Type ptrType = b.convertType(op.getType());
  if (!promiseType || !ptrType)
    return op.emitError("failed to convert coroutine type");

  // Retrieve the pointer to the beginning of the coroutine promise.
  auto coroPromiseOp = b.create<CallIntrinsicOp>(
      cache.i8PtrType, "llvm.coro.promise",
      ValueRange{
          hdl,
          b.create<ConstantOp>(cache.i32Type, b.getTypeABIAlign(promiseType)),
          b.create<ConstantOp>(cache.i1Type, false)});

  // The next element is a pair of pointers. Skip over it to get to the results.
  Value promise = b.create<GEPOp>(
      ptrType, coroPromiseOp.getResult(0),
      GEPArg(llvm::divideCeil(b.getIndexTypeBitwidth(), CHAR_BIT) * 2),
      /*inbounds=*/true);

  op.replaceAllUsesWith(promise);
  op.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// CoroutineResumeOp

/// Lower `pop.coroutine.resume` to `llvm.coro.resume`.
static void lowerCoroutineResume(LLVMBuilder &b, const TypeAttrCache &cache,
                                 POP::CoroutineResumeOp op) {
  b.setLoc(op.getLoc());
  b.setInsertionPoint(op);
  Value hdl = b.create<mlir::UnrealizedConversionCastOp>(cache.i8PtrType,
                                                         op.getCoroutine())
                  .getResult(0);

  b.create<CoroResumeOp>(hdl);
  op.erase();
}

//===----------------------------------------------------------------------===//
// CoroutineDestroyOp

/// Lower `pop.coroutine.destroy` to the coroutine destroy intrinsic.
static LogicalResult lowerCoroutineDestroy(LLVMBuilder &b,
                                           const TypeAttrCache &cache,
                                           POP::CoroutineDestroyOp op) {
  b.setLoc(op.getLoc());
  b.setInsertionPoint(op);
  Value hdl = b.create<mlir::UnrealizedConversionCastOp>(cache.i8PtrType,
                                                         op.getCoroutine())
                  .getResult(0);

  // Call the coroutine destroy intrinsic.
  b.create<CallIntrinsicOp>(TypeRange(), "llvm.coro.destroy", hdl);

  op.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERKGENCOROUTINES
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerKGENCoroutinesPass
    : public KGEN::impl::LowerKGENCoroutinesBase<LowerKGENCoroutinesPass> {
  using LowerKGENCoroutinesBase::LowerKGENCoroutinesBase;

  void runOnOperation() override;
};
} // namespace

void LowerKGENCoroutinesPass::runOnOperation() {
  LLVMFuncOp func = getOperation();

  // Configure the builder.
  TargetInfoAttr target = lookupTargetInfo(func);
  if (!target) {
    mlir::emitError(func.getLoc(),
                    "could not find an enclosing target specification");
    return signalPassFailure();
  }
  POPToLLVMTypeConverter typeConverter(target);

  ImplicitLocOpBuilder opBuilder(getOperation()->getLoc(), &getContext());
  LLVMBuilder b(opBuilder, typeConverter);

  // Initialize the type cache.
  TypeAttrCache cache{b.getI1Type(),
                      b.getI8Type(),
                      b.getI32Type(),
                      b.getI64Type(),
                      b.getType<LLVMPointerType>(),
                      LLVMPointerType::get(b.getI8Type()),
                      b.getType<LLVMTokenType>(),
                      nullptr};

  // Collect all the relevant ops.
  SmallVector<POP::CoroutineHandleOp> handles;
  SmallVector<POP::CoroutineOpaqueHandleOp> opaques;
  SmallVector<POP::CoroutineAwaitOp> awaits;
  func.walk([&](Operation *op) {
    if (auto handle = dyn_cast<POP::CoroutineHandleOp>(op)) {
      handles.push_back(handle);
    } else if (auto opaque = dyn_cast<POP::CoroutineOpaqueHandleOp>(op)) {
      opaques.push_back(opaque);
    } else if (auto await = dyn_cast<POP::CoroutineAwaitOp>(op)) {
      awaits.push_back(await);
    } else if (auto promise = dyn_cast<POP::CoroutinePromiseOp>(op)) {
      if (failed(lowerCoroutinePromise(b, cache, promise)))
        return signalPassFailure();
    } else if (auto resume = dyn_cast<POP::CoroutineResumeOp>(op)) {
      lowerCoroutineResume(b, cache, resume);
    } else if (auto destroy = dyn_cast<POP::CoroutineDestroyOp>(op)) {
      if (failed(lowerCoroutineDestroy(b, cache, destroy)))
        return signalPassFailure();
    }
  });

  // The presence of `pop.coroutine.handle` inside a function indicates
  // that it is a coroutine.
  if (handles.empty()) {
    // Replace opaque handles with a null pointer.
    for (POP::CoroutineOpaqueHandleOp opaque : opaques) {
      b.setInsertionPoint(opaque);
      Value cstNullPtr = b.create<IntToPtrOp>(
          cache.i8PtrType, b.create<ConstantOp>(b.getIndexType(), 0));
      opaque.replaceAllUsesWith(cstNullPtr);
      opaque.erase();
    }
    return;
  }

  POP::CoroutineType coroType = handles.front().getType();
  b.setLoc(func.getLoc());
  FailureOr<Coroutine> coroutine =
      createCoroutineFunction(b, cache, func, coroType);
  if (failed(coroutine))
    return signalPassFailure();

  for (POP::CoroutineHandleOp op : handles) {
    op.replaceAllUsesWith(coroutine->handle);
    op.erase();
  }
  for (POP::CoroutineOpaqueHandleOp op : opaques) {
    op.replaceAllUsesWith(coroutine->handle);
    op.erase();
  }
  for (POP::CoroutineAwaitOp op : awaits)
    lowerCoroutineAwait(b, cache, *coroutine, op);
}

//===----------------------------------------------------------------------===//
// LowerKGENCoroutinesAsync
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERKGENCOROUTINESASYNC
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerKGENCoroutinesAsyncPass
    : public KGEN::impl::LowerKGENCoroutinesAsyncBase<
          LowerKGENCoroutinesAsyncPass> {
  using LowerKGENCoroutinesAsyncBase::LowerKGENCoroutinesAsyncBase;

  void runOnOperation() override;
};
} // namespace

static LogicalResult lowerCoroutinePromiseAsync(LLVMBuilder &b,
                                                TypeAttrCache &cache,
                                                POP::CoroutinePromiseOp op) {
  b.setLoc(op.getLoc());
  b.setInsertionPoint(op);

  Value hdl = b.create<mlir::UnrealizedConversionCastOp>(cache.i8PtrType,
                                                         op.getCoroutine())
                  .getResult(0);

  Type ptrType = b.convertType(op.getType());
  if (!ptrType)
    return op.emitError("failed to convert coroutine type");

  // The context contains the parent context, the resume function pointer, and
  // then the callback closure. Skip over them (4 pointers) to reach the
  // results.
  Value promise = b.create<GEPOp>(
      ptrType, hdl,
      GEPArg(llvm::divideCeil(b.getIndexTypeBitwidth(), CHAR_BIT) * 4),
      /*inbounds=*/true);

  op.replaceAllUsesWith(promise);
  op.erase();
  return success();
}

static void lowerCoroutineResumeAsync(LLVMBuilder &b, TypeAttrCache &cache,
                                      POP::CoroutineResumeOp op) {
  b.setLoc(op.getLoc());
  b.setInsertionPoint(op);

  // The resume function is the second element of the context.
  auto hdl = b.create<mlir::UnrealizedConversionCastOp>(cache.i8PtrType,
                                                        op.getCoroutine())
                 .getResult(0);
  // Bitcast `i8*` to `void(i8*)**` in the GEP.
  Value resumeFnPtr = b.create<GEPOp>(
      LLVMPointerType::get(cache.asyncFnType), hdl,
      GEPArg(llvm::divideCeil(b.getIndexTypeBitwidth(), CHAR_BIT)),
      /*inbounds=*/true);
  b.create<CallOp>(TypeRange(), ValueRange{b.create<LoadOp>(resumeFnPtr), hdl});
  op.erase();
}

static void lowerCoroutineDestroyAsync(LLVMBuilder &b, TypeAttrCache &cache,
                                       POP::CoroutineDestroyOp op) {
  b.setLoc(op.getLoc());
  b.setInsertionPoint(op);

  // Just free the coroutine context.
  auto hdl = b.create<mlir::UnrealizedConversionCastOp>(cache.i8PtrType,
                                                        op.getCoroutine())
                 .getResult(0);
  b.create<POP::ExternalCallOp>(
      "free", b.create<BitcastOp>(cache.ptrType, hdl).getResult());
  op.erase();
}

/// When a coroutine completes, it invokes it completion callback:
///
/// ```
/// llvm.func @__kgen_coro_end_fn(%opaqueCtxt: !llvm.ptr<i8>) {
///   %ctxt = llvm.bitcast %opaqueCtxt : !llvm.ptr<i8> to !llvm.ptr<struct(...
///   %clsFnPtr = llvm.getelementptr %ctxt[0, 2, 0]
///   %clsArgPtr = llvm.getlementptr %ctxt[0, 2, 1]
///   %ctxtFn = llvm.load %ctxtFnPtr
///   %ctxtArg = llvm.load %ctxtArgPtr
///   llvm.call %ctxtFn(%ctxtArg)
///   llvm.return
/// }
/// ```
static LLVMFuncOp synthesizeCoroEndFunc(SymbolTable &symtab, LLVMBuilder &b,
                                        TypeAttrCache &cache) {
  OpBuilder::InsertionGuard guard(b);
  Location prevLoc = b.getLoc();
  b.setLoc(symtab.getOp()->getLoc());
  b.clearInsertionPoint();

  auto endFn = b.create<LLVMFuncOp>(
      "__kgen_coro_end_fn",
      LLVMFunctionType::get(LLVMVoidType::get(b.getContext()), cache.i8PtrType),
      Linkage::Internal);

  b.setInsertionPointToStart(endFn.addEntryBlock());
  Value closure = b.create<GEPOp>(
      LLVMPointerType::get(LLVMStructType::getLiteral(
          b.getContext(), {cache.asyncFnType, cache.i8PtrType})),
      endFn.getArgument(0),
      GEPArg(llvm::divideCeil(b.getIndexTypeBitwidth(), CHAR_BIT) * 2),
      /*inbounds=*/true);
  Value closureFn = b.create<LoadOp>(
      b.create<GEPOp>(LLVMPointerType::get(cache.asyncFnType), closure,
                      ArrayRef<GEPArg>{0, 0}, /*inbounds=*/true));
  Value closureArg = b.create<LoadOp>(
      b.create<GEPOp>(LLVMPointerType::get(cache.i8PtrType), closure,
                      ArrayRef<GEPArg>{0, 1}, /*inbounds=*/true));
  b.create<CallOp>(TypeRange(), ValueRange{closureFn, closureArg});
  b.create<ReturnOp>(ValueRange());
  symtab.insert(endFn);

  b.setLoc(prevLoc);
  return endFn;
}

/// Parent-callee context relations are managed by the standard library via the
/// completion callback. Create a function that forwards the context.
static LLVMFuncOp synthesizeCoroCtxtProjFn(SymbolTable &symtab, LLVMBuilder &b,
                                           TypeAttrCache &cache) {
  OpBuilder::InsertionGuard guard(b);
  Location prevLoc = b.getLoc();
  b.setLoc(symtab.getOp()->getLoc());
  b.clearInsertionPoint();

  auto projFn = b.create<LLVMFuncOp>(
      "__kgen_coro_ctxt_proj_fn",
      LLVMFunctionType::get(cache.i8PtrType, cache.i8PtrType),
      Linkage::Internal);

  b.setInsertionPointToStart(projFn.addEntryBlock());
  b.create<ReturnOp>(projFn.getArgument(0));
  symtab.insert(projFn);

  b.setLoc(prevLoc);
  return projFn;
}

namespace {
struct CoroutineInfo {
  LLVMFuncOp asyncFn;
  Value hdl;
  Type contextPtrType;
  int64_t contextBaseSize;
};
} // namespace

static FailureOr<CoroutineInfo>
createAsyncCoroutine(SymbolTable &symtab, LLVMFuncOp func,
                     POP::CoroutineType coroType, LLVMBuilder &b,
                     TypeAttrCache &cache, LLVMFuncOp coroEndFn) {
  b.setLoc(func.getLoc());
  b.clearInsertionPoint();

  // The coroutine context contains the parent context (null if no parent),
  // the next resume function pointer, a callback closure in the form of
  // `{ void(i8*)*, i8* }`, the function results, and the function arguments,
  // and then trailing coroutine frame:
  //
  //   { i8*, void(i8*)*, { void(i8*)*, i8* }, ResTs..., ArgTs..., FrameT }
  //
  SmallVector<Type, 16> contextTypes{
      cache.i8PtrType, cache.asyncFnType,
      LLVMStructType::getLiteral(b.getContext(),
                                 {cache.ptrType, cache.ptrType})};
  for (Type resultType : coroType.getResultTypes()) {
    resultType = b.convertType(resultType);
    if (!resultType)
      return mlir::emitError(func.getLoc(),
                             "could not convert coroutine result type");
    contextTypes.push_back(resultType);
  }
  llvm::append_range(contextTypes, func.getArgumentTypes());
  auto contextType = LLVMStructType::getLiteral(b.getContext(), contextTypes);
  auto contextPtrType = LLVMPointerType::get(contextType);
  // Compute the base size of the context to populate into the global async
  // function pointer as required by the LLVM async coroutine intrinsics. LLVM's
  // async lowering will update the field with the total size.
  int64_t contextBaseSize = b.getTypeAllocSize(contextType);

  // The async function implementation is always void(i8*). Create a new
  // function with that signature and make the current function the wrapper
  // function. The wrapper will load the arguments into the context and store
  // the ramp function as the resume function.
  auto asyncFn = b.create<LLVMFuncOp>(
      (func.getSymName() + "_af").str(),
      LLVMFunctionType::get(LLVMVoidType::get(b.getContext()), cache.i8PtrType),
      func.getLinkage());
  symtab.insert(asyncFn, func->getIterator());

  // Construct the async function pointer. We have to synthesize this massive
  // global constant.
  auto afpType = LLVMStructType::getLiteral(b.getContext(),
                                            {cache.i32Type, cache.i32Type});
  // constant struct <{ i32, i32 }>
  auto afp =
      b.create<GlobalOp>(afpType, /*isConstant=*/true,
                         /*linkage=*/func.getLinkage(),
                         (func.getSymName() + "_afp").str(), Attribute());
  symtab.insert(afp, func->getIterator());
  // = <{ i32 trunc (
  //      i64 sub (
  //        i64 ptrtoint void(i8*)* @async_fn to i64),
  //        i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32 }>,
  //                      <{ i32, i32 }>* @async_fn_afp, i32 0, i32 1) to i64)
  //      )
  //      to i32),
  //      <context_base_size>
  b.createBlock(&afp.getBodyRegion());
  Value afpValue = b.create<UndefOp>(afpType);
  Value afpEndPtr = b.create<GEPOp>(LLVMPointerType::get(cache.i32Type),
                                    b.create<AddressOfOp>(afp),
                                    ArrayRef<GEPArg>{0, 1}, /*inbounds=*/true);
  Value afpOffset = b.create<TruncOp>(
      cache.i32Type,
      b.create<SubOp>(
          b.create<PtrToIntOp>(cache.i64Type, b.create<AddressOfOp>(asyncFn)),
          b.create<PtrToIntOp>(cache.i64Type, afpEndPtr)));
  afpValue = b.create<InsertValueOp>(afpValue, afpOffset, 0);
  afpValue = b.create<InsertValueOp>(
      afpValue, b.create<ConstantOp>(cache.i32Type, contextBaseSize), 1);
  b.create<ReturnOp>(afpValue);

  // Move the body of the coroutine into the new async function.
  Region &asyncFnBody = asyncFn.getBody();
  asyncFnBody.takeBody(func.getBody());

  // Generate coroutine machinery.
  b.setLoc(asyncFn.getLoc());
  b.setInsertionPointToStart(&asyncFnBody.front());
  auto coroIdOp = b.create<CallIntrinsicOp>(
      cache.tokenType, "llvm.coro.id.async",
      ValueRange{
          b.create<ConstantOp>(b.getI32IntegerAttr(contextBaseSize)),
          // FIXME: `malloc` provides no alignment guarantees.
          b.create<ConstantOp>(b.getI32IntegerAttr(1)),
          b.create<ConstantOp>(b.getI32IntegerAttr(0)),
          b.create<BitcastOp>(cache.i8PtrType, b.create<AddressOfOp>(afp))});
  Value hdl = b.create<CoroBeginOp>(
      cache.i8PtrType, coroIdOp.getResult(0),
      b.create<IntToPtrOp>(cache.i8PtrType,
                           b.create<ConstantOp>(b.getI32IntegerAttr(0))));

  // The coroutine handle is specially handled by the coroutine splitting pass
  // to be replaced by the frame pointer value in each resume function. It is
  // the tail to the context pointer. Retrieve any context values, including the
  // async context itself, through the handle to prevent the async context value
  // from being considered "spilled" and put into the frame. Storing the async
  // context in itself causes it to be an aliasing pointer. In addition, push
  // all uses the context and arguments to the latest use sites to prevent them
  // from being put on the coroutine frame.

  // Replace argument uses with values from the async context.
  asyncFnBody.addArgument(cache.i8PtrType, func.getLoc());
  b.setInsertionPointToStart(&asyncFnBody.front());
  // Arguments are located at the end.
  constexpr int64_t resOffset = 3;
  int64_t argOffset = resOffset + coroType.getResultTypes().size();
  for (auto [idx, arg] :
       llvm::enumerate(asyncFnBody.getArguments().drop_back())) {
    b.setLoc(arg.getLoc());
    for (OpOperand &use : llvm::make_early_inc_range(arg.getUses())) {
      b.setInsertionPoint(use.getOwner());
      // Obtain start of the context from the frame pointer.
      Value contextPtr = b.create<GEPOp>(
          contextPtrType, hdl, GEPArg(-contextBaseSize), /*inbounds=*/true);
      Value argPtr = b.create<GEPOp>(
          LLVMPointerType::get(arg.getType()), contextPtr,
          ArrayRef<GEPArg>{0, argOffset + idx}, /*inbounds=*/true);
      use.set(b.create<LoadOp>(argPtr));
    }
  }
  asyncFnBody.front().eraseArguments(0, asyncFnBody.getNumArguments() - 1);

  // Returns in the async function are all void. Generate the coroutine end
  // marker that invokes the callback closure.
  asyncFnBody.walk([&](ReturnOp ret) {
    b.setInsertionPoint(ret);
    b.setLoc(ret.getLoc());
    Value contextPtr = b.create<GEPOp>(
        contextPtrType, hdl, GEPArg(-contextBaseSize), /*inbounds=*/true);
    b.create<CallIntrinsicOp>(
        cache.i1Type, "llvm.coro.end.async",
        ValueRange{hdl, b.create<ConstantOp>(b.getBoolAttr(false)),
                   b.create<AddressOfOp>(coroEndFn), contextPtr});
    ret->eraseOperands(0, ret.getNumOperands());
  });

  // Generate the code to marshall the async function arguments and store the
  // ramp function as the initial resume function.
  b.setLoc(func.getLoc());
  b.setInsertionPointToStart(func.addEntryBlock());
  // Read the required context size from the async function pointer.
  // NOTE: Hide the global read behind a "prepare" intrinsic to prevent the size
  // from being inlined into the allocator call until after coroutine splitting,
  // when it gets updated with the frame size.
  auto prepare = b.create<CallIntrinsicOp>(
      cache.i8PtrType, "llvm.coro.prepare.async",
      Value(b.create<BitcastOp>(cache.i8PtrType, b.create<AddressOfOp>(afp))));
  Value contextSize = b.create<LoadOp>(
      b.create<GEPOp>(LLVMPointerType::get(cache.i32Type),
                      b.create<BitcastOp>(LLVMPointerType::get(afp.getType()),
                                          prepare.getResult(0)),
                      ArrayRef<GEPArg>{0, 1}, /*inbounds=*/true));
  auto allocCall = b.create<POP::ExternalCallOp>(
      cache.ptrType, "malloc",
      Value(b.create<ZExtOp>(cache.i64Type, contextSize)));
  Value contextValue =
      b.create<BitcastOp>(contextPtrType, allocCall.getResult(0));

  b.create<StoreOp>(b.create<AddressOfOp>(asyncFn),
                    b.create<GEPOp>(LLVMPointerType::get(cache.asyncFnType),
                                    contextValue, ArrayRef<GEPArg>{0, 1},
                                    /*inbounds=*/true));
  for (auto [idx, arg] : llvm::enumerate(func.getArguments())) {
    Value argPtr = b.create<GEPOp>(
        LLVMPointerType::get(arg.getType()), contextValue,
        ArrayRef<GEPArg>{0, argOffset + idx}, /*inbounds=*/true);
    b.create<StoreOp>(arg, argPtr);
  }
  b.create<ReturnOp>(b.create<BitcastOp>(cache.i8PtrType, contextValue));
  return {{asyncFn, hdl, contextPtrType, contextBaseSize}};
}

static LogicalResult
lowerCoroutineAwaitAsync(SymbolTable &symtab, LLVMBuilder &b,
                         CoroutineInfo &coro, TypeAttrCache &cache,
                         LLVMFuncOp coroProjFn, POP::CoroutineAwaitOp op) {
  b.setLoc(op.getLoc());

  // Outline the body of the await into a function.
  SmallVector<Value> captures;
  (void)operationIsIsolatedFromAbove(op, &captures);

  Block *awaitBody = &op.getBody().front();
  SmallVector<Type> captureTypes;
  for (Value &capture : captures) {
    Type captureType = b.convertType(capture.getType());
    if (!captureType)
      return op.emitError("failed to convert captured value type");
    captureTypes.push_back(captureType);

    Value arg = awaitBody->addArgument(capture.getType(), op.getLoc());
    Value valueInBody = capture;
    if (arg.getType() != captureType) {
      b.setInsertionPointToStart(&op.getBody().front());
      // Materialize source and destination conversions.
      Type srcType = arg.getType();
      arg.setType(captureType);
      arg =
          b.create<mlir::UnrealizedConversionCastOp>(srcType, arg).getResult(0);
      b.setInsertionPoint(op);
      capture = b.create<mlir::UnrealizedConversionCastOp>(captureType, capture)
                    .getResult(0);
    }
    valueInBody.replaceUsesWithIf(arg, [&](OpOperand &use) {
      return op->isProperAncestor(use.getOwner());
    });
  }

  b.clearInsertionPoint();
  auto suspendFn = b.create<LLVMFuncOp>(
      (coro.asyncFn.getSymName() + ".suspend").str(),
      LLVMFunctionType::get(LLVMVoidType::get(b.getContext()), captureTypes),
      Linkage::Internal);
  suspendFn.getBody().takeBody(op.getBody());
  b.setInsertionPointToEnd(&suspendFn.getBody().front());
  b.create<ReturnOp>(ValueRange());
  symtab.insert(suspendFn, coro.asyncFn->getIterator());

  b.setInsertionPoint(op);
  Value resumeFn = b.create<CallIntrinsicOp>(
                        cache.i8PtrType, "llvm.coro.async.resume", ValueRange())
                       .getResult(0);
  Value contextPtr =
      b.create<GEPOp>(coro.contextPtrType, coro.hdl,
                      GEPArg(-coro.contextBaseSize), /*inbounds=*/true);
  Value resumeFnPtr =
      b.create<GEPOp>(LLVMPointerType::get(cache.i8PtrType), contextPtr,
                      ArrayRef<GEPArg>{0, 1}, /*inbounds=*/true);
  b.create<StoreOp>(resumeFn, resumeFnPtr);

  SmallVector<Value> suspendAsyncArgs{
      b.create<ConstantOp>(b.getI32IntegerAttr(0)), resumeFn,
      b.create<BitcastOp>(cache.i8PtrType, b.create<AddressOfOp>(coroProjFn)),
      b.create<AddressOfOp>(suspendFn)};
  llvm::append_range(suspendAsyncArgs, captures);
  auto suspendRetType = LLVMStructType::getLiteral(
      b.getContext(), {cache.i8PtrType, cache.i8PtrType, cache.i8PtrType});
  // FIXME: For some reason, `call_intrinsic` fails to resolve the overload.
  b.create<POP::ExternalCallOp>(
      suspendRetType, "llvm.coro.suspend.async.sl_p0p0p0s", suspendAsyncArgs,
      TypeAttr::get(b.getFunctionType(
          {cache.i32Type, cache.i8PtrType, cache.i8PtrType}, suspendRetType)));
  op.erase();
  return success();
}

static LogicalResult
lowerCoroutineFunction(SymbolTable &symtab, LLVMFuncOp func, LLVMBuilder &b,
                       TypeAttrCache &cache,
                       function_ref<LLVMFuncOp()> getCoroEndFn,
                       function_ref<LLVMFuncOp()> getCoroProjFn) {
  // Collect all the relevant ops.
  SmallVector<POP::CoroutineHandleOp> handles;
  SmallVector<POP::CoroutineOpaqueHandleOp> opaques;
  SmallVector<POP::CoroutineAwaitOp> awaits;
  WalkResult result = func.walk([&](Operation *op) {
    if (auto handle = dyn_cast<POP::CoroutineHandleOp>(op)) {
      handles.push_back(handle);
    } else if (auto opaque = dyn_cast<POP::CoroutineOpaqueHandleOp>(op)) {
      opaques.push_back(opaque);
    } else if (auto await = dyn_cast<POP::CoroutineAwaitOp>(op)) {
      awaits.push_back(await);
    } else if (auto promise = dyn_cast<POP::CoroutinePromiseOp>(op)) {
      if (failed(lowerCoroutinePromiseAsync(b, cache, promise)))
        return WalkResult::interrupt();
    } else if (auto resume = dyn_cast<POP::CoroutineResumeOp>(op)) {
      lowerCoroutineResumeAsync(b, cache, resume);
    } else if (auto destroy = dyn_cast<POP::CoroutineDestroyOp>(op)) {
      lowerCoroutineDestroyAsync(b, cache, destroy);
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  // The presence of `pop.coroutine.handle` inside a function indicates
  // that it is a coroutine.
  if (handles.empty()) {
    // Replace opaque handles with a null pointer.
    for (POP::CoroutineOpaqueHandleOp opaque : opaques) {
      b.setInsertionPoint(opaque);
      Value cstNullPtr = b.create<IntToPtrOp>(
          cache.i8PtrType, b.create<ConstantOp>(b.getIndexType(), 0));
      opaque.replaceAllUsesWith(cstNullPtr);
      opaque.erase();
    }
    return success();
  }

  POP::CoroutineType coroType = handles.front().getType();
  FailureOr<CoroutineInfo> coro =
      createAsyncCoroutine(symtab, func, coroType, b, cache, getCoroEndFn());
  if (failed(coro))
    return failure();

  // The coroutine handle is now the first argument of the coroutine function.
  for (POP::CoroutineHandleOp op : handles) {
    b.setInsertionPoint(op);
    Value contextPtr =
        b.create<GEPOp>(coro->contextPtrType, coro->hdl,
                        GEPArg(-coro->contextBaseSize), /*inbounds=*/true);
    op.replaceAllUsesWith(
        b.create<BitcastOp>(cache.i8PtrType, contextPtr).getResult());
    op.erase();
  }
  for (POP::CoroutineOpaqueHandleOp op : opaques) {
    b.setInsertionPoint(op);
    Value contextPtr =
        b.create<GEPOp>(coro->contextPtrType, coro->hdl,
                        GEPArg(-coro->contextBaseSize), /*inbounds=*/true);
    op.replaceAllUsesWith(
        b.create<BitcastOp>(cache.i8PtrType, contextPtr).getResult());
    op.erase();
  }
  for (POP::CoroutineAwaitOp op : awaits) {
    if (failed(lowerCoroutineAwaitAsync(symtab, b, *coro, cache,
                                        getCoroProjFn(), op)))
      return failure();
  }
  return success();
}

void LowerKGENCoroutinesAsyncPass::runOnOperation() {
  // Configure the builder.
  TargetInfoAttr target = getTargetInfo(getOperation());
  if (!target) {
    mlir::emitError(getOperation()->getLoc(),
                    "could not find an enclosing target specification");
    return signalPassFailure();
  }
  POPToLLVMTypeConverter typeConverter(target);
  ImplicitLocOpBuilder opBuilder(getOperation().getLoc(), &getContext());
  LLVMBuilder b(opBuilder, typeConverter);

  // Initialize the type cache.
  // TODO: Move this into the pass class itself.
  TypeAttrCache cache{b.getI1Type(),
                      b.getI8Type(),
                      b.getI32Type(),
                      b.getI64Type(),
                      b.getType<LLVMPointerType>(),
                      LLVMPointerType::get(b.getI8Type()),
                      b.getType<LLVMTokenType>(),
                      nullptr};
  cache.asyncFnType = LLVMPointerType::get(
      LLVMFunctionType::get(b.getType<LLVMVoidType>(), cache.i8PtrType));

  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

  LLVMFuncOp coroEndFn, coroProjFn;
  auto getCoroEndFn = [&] {
    if (!coroEndFn)
      coroEndFn = synthesizeCoroEndFunc(symtab, b, cache);
    return coroEndFn;
  };
  auto getCoroProjFn = [&] {
    if (!coroProjFn)
      coroProjFn = synthesizeCoroCtxtProjFn(symtab, b, cache);
    return coroProjFn;
  };

  // TODO: Do this in parallel.
  for (auto func : getOperation().getOps<LLVMFuncOp>())
    if (failed(lowerCoroutineFunction(symtab, func, b, cache, getCoroEndFn,
                                      getCoroProjFn)))
      return signalPassFailure();
}
