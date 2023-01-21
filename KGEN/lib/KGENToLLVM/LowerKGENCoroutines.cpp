//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENPasses.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "LLVMLoweringUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace M;
using namespace KGEN;
using namespace mlir::LLVM;

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
  Type ptrType;
  Type i8PtrType;
  Type tokenType;
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
  /// The value of the coroutine promise.
  Value promise;
  /// The type of the promise.
  LLVMStructType promiseType;
};
} // namespace

/// Get the LLVM type of the coroutine promise.
static LLVMStructType getCoroutinePromiseType(LLVMBuilder &b,
                                              const TypeAttrCache &cache,
                                              POP::CoroutineType coroType) {
  // Pack the result types into a struct.
  SmallVector<Type> promiseTypes(coroType.getResultTypes());
  for (Type &type : promiseTypes) {
    type = b.convertType(type);
    if (!type)
      return {};
  }
  // Add the async context: a ref-counted pointer and a compact runtime pointer.
  promiseTypes.push_back(LLVMStructType::getLiteral(
      b.getContext(), {cache.ptrType, cache.i8Type}));
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
  // Explicitly specify the alignemnt.
  Value promiseMem = b.create<AllocaOp>(LLVMPointerType::get(promiseType),
                                        b.create<ConstantOp>(cache.i32Type, 1),
                                        b.getTypeABIAlignment(promiseType));
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
  return Coroutine{suspend, cleanup, coroHdl, promiseMem, promiseType};
}

/// Given the coroutine handle, return the pointer to the coroutine promise.
static Value getCoroutinePromise(LLVMBuilder &b, const TypeAttrCache &cache,
                                 Value hdl, LLVMStructType promiseType) {
  auto coroPromiseOp = b.create<CallIntrinsicOp>(
      cache.i8PtrType, "llvm.coro.promise",
      ValueRange{hdl,
                 b.create<ConstantOp>(cache.i32Type,
                                      b.getTypeABIAlignment(promiseType)),
                 b.create<ConstantOp>(cache.i1Type, false)});
  return {b.create<BitcastOp>(LLVMPointerType::get(promiseType),
                              coroPromiseOp.getResult(0))};
}

//===----------------------------------------------------------------------===//
// CoroutineHandleOp

/// Replace `pop.coroutine.handle` with the current coroutine handle.
static void lowerCoroutineHandle(const Coroutine &coroutine,
                                 POP::CoroutineHandleOp op) {
  op.replaceAllUsesWith(coroutine.handle);
  op.erase();
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

  // Cast it to just the promise results.
  Value promise = b.create<BitcastOp>(
      ptrType, getCoroutinePromise(b, cache, hdl, promiseType));

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

/// Lower `pop.coroutine.destroy` to a runtime call and the coroutine destroy
/// intrinsic.
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

  // Set LLVM lowering options.
  mlir::LowerToLLVMOptions options(&getContext());
  if (indexBitwidth != mlir::kDeriveIndexBitwidthFromDataLayout)
    options.overrideIndexBitwidth(indexBitwidth);

  // Configure the builder.
  POPToLLVMTypeConverter typeConverter(func.getLoc(), options);
  mlir::DataLayout dl;
  ImplicitLocOpBuilder opBuilder(getOperation()->getLoc(), &getContext());
  LLVMBuilder b(opBuilder, typeConverter, dl);

  // Initialize the type cache.
  TypeAttrCache cache{b.getI1Type(),
                      b.getI8Type(),
                      b.getI32Type(),
                      b.getType<LLVMPointerType>(),
                      LLVMPointerType::get(b.getI8Type()),
                      b.getType<LLVMTokenType>()};

  // Collect all the relevant ops.
  SmallVector<POP::CoroutineHandleOp> handles;
  SmallVector<POP::CoroutineAwaitOp> awaits;
  func.walk([&](Operation *op) {
    if (auto handle = dyn_cast<POP::CoroutineHandleOp>(op)) {
      handles.push_back(handle);
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
  if (handles.empty())
    return;

  POP::CoroutineType coroType = handles.front().getType();
  b.setLoc(func.getLoc());
  FailureOr<Coroutine> coroutine =
      createCoroutineFunction(b, cache, func, coroType);
  if (failed(coroutine))
    return signalPassFailure();

  for (POP::CoroutineHandleOp op : handles)
    lowerCoroutineHandle(*coroutine, op);
  for (POP::CoroutineAwaitOp op : awaits)
    lowerCoroutineAwait(b, cache, *coroutine, op);
}
