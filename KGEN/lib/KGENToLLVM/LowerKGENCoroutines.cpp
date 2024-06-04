//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CODialect/COOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "LLVMLoweringUtils.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/DebugInfoDialect/IR/DebugInfoInterfaces.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/BinaryFormat/Dwarf.h"

using namespace M;
using namespace KGEN;
using namespace CO;
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
static LogicalResult tweakSpilledAllocas(Operation *func,
                                         unsigned &numLocalVars) {
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

    } else if (auto await = dyn_cast<SuspendOp>(op)) {
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
  unsigned numLocalVars = 0;
  if (failed(tweakSpilledAllocas(getOperation(), numLocalVars)))
    return signalPassFailure();
  this->numLocalVars = numLocalVars;
}

//===----------------------------------------------------------------------===//
// Coroutine Lowering
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERKGENCOROUTINESASYNC
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
/// This struct contains "cached" types and attributes. These are types and
/// attributes that are commonly used throughout the lowerings. They are cached
/// in this struct so that they are not hashed multiple times.
struct TypeAttrCache {
  Type i1Type;
  Type i8Type;
  Type i32Type;
  Type i64Type;
  Type indexType;
  Type i8PtrType;
  Type tokenType;
  Type opaquePtr;
};

struct LowerKGENCoroutinesAsyncPass
    : public KGEN::impl::LowerKGENCoroutinesAsyncBase<
          LowerKGENCoroutinesAsyncPass> {
  using LowerKGENCoroutinesAsyncBase::LowerKGENCoroutinesAsyncBase;

  void runOnOperation() override;
};
} // namespace

static Value getOffsetToResults(LLVMBuilder &b, TypeAttrCache &cache,
                                Value hdl) {
  // The context contains the resume function pointer, the callback closure, and
  // the 2 result slots. Skip over them (5 pointers) to reach the results.
  return b.create<GEPOp>(cache.opaquePtr, b.getI8Type(), hdl,
                         GEPArg(b.getPointerByteWidth() * 5),
                         /*inbounds=*/true);
}

static LogicalResult lowerCoroutineSetResults(LLVMBuilder &b,
                                              TypeAttrCache &cache,
                                              SetResultsOp op) {
  // Handle the case where there are no result values.
  ValueRange values = op.getValues();
  if (values.empty()) {
    op.erase();
    return success();
  }

  b.setLoc(op.getLoc());
  b.setInsertionPoint(op);

  // Load out the results as a whole struct. There should only be a small number
  // of results.
  SmallVector<Type> resultTypes;
  resultTypes.reserve(values.size());
  for (Type type : values.getType()) {
    resultTypes.push_back(b.convertType(type));
    if (!resultTypes.back())
      return op.emitError("failed to convert result type");
  }
  Value hdl = b.createConversion(cache.opaquePtr, op.getCoroutine());
  Value resultsPtr = getOffsetToResults(b, cache, hdl);

  // If there is a single result, don't bother wrapping it into a struct.
  if (resultTypes.size() == 1) {
    b.create<StoreOp>(b.createConversion(resultTypes.front(), values.front()),
                      resultsPtr);
    op.erase();
    return success();
  }

  // Pack the results into a struct and then store them.
  auto structType = LLVMStructType::getLiteral(b.getContext(), resultTypes);
  Value result = b.create<UndefOp>(structType);
  for (auto [i, value, type] : llvm::enumerate(values, resultTypes)) {
    result =
        b.create<InsertValueOp>(result, b.createConversion(type, value), i);
  }
  b.create<StoreOp>(result, resultsPtr);

  op.erase();
  return success();
}

static LogicalResult lowerCoroutineCallback(LLVMBuilder &b,
                                            TypeAttrCache &cache,
                                            GetCallbackPtrOp op) {
  b.setLoc(op.getLoc());
  b.setInsertionPoint(op);

  Value hdl = b.createConversion(cache.opaquePtr, op.getCoroutine());
  Type elType = b.convertType(op.getType().getElementType());
  if (!elType)
    return op.emitError("failed to convert callback type");

  // The context contains the resume function pointer, and then the callback
  // closure. Skip over the resume function (1 pointer).
  Value callback =
      b.create<GEPOp>(cache.opaquePtr, b.getI8Type(), hdl,
                      GEPArg(b.getPointerByteWidth()), /*inbounds=*/true);

  op.replaceAllUsesWith(callback);
  op.erase();
  return success();
}

static LogicalResult lowerCoroutineGetResults(LLVMBuilder &b,
                                              TypeAttrCache &cache,
                                              GetResultsOp op) {
  // Peephole the case where there are no results.
  if (op.getNumResults() == 0) {
    op.erase();
    return success();
  }

  b.setLoc(op.getLoc());
  b.setInsertionPoint(op);

  // Load out the results as a whole struct. There should only be a small number
  // of results.
  SmallVector<Type> resultTypes;
  resultTypes.reserve(op.getNumResults());
  for (Type type : op.getResultTypes()) {
    resultTypes.push_back(b.convertType(type));
    if (!resultTypes.back())
      return op.emitError("failed to convert result type");
  }
  Value hdl = b.createConversion(cache.opaquePtr, op.getCoroutine());
  Value resultsPtr = getOffsetToResults(b, cache, hdl);

  // Handle the case where there is a single result and no need to pack the
  // results into a struct.
  if (resultTypes.size() == 1) {
    Value result = b.create<LoadOp>(resultTypes.front(), resultsPtr);
    op.replaceAllUsesWith(
        ArrayRef(b.createConversion(op.getResultTypes().front(), result)));
    op.erase();
    return success();
  }

  // Load the packed results as a struct and extract them.
  auto structType = LLVMStructType::getLiteral(b.getContext(), resultTypes);
  Value resultPack = b.create<LoadOp>(structType, resultsPtr);
  SmallVector<Value> results;
  results.reserve(op.getNumResults());
  for (auto [i, type, origType] :
       llvm::enumerate(resultTypes, op.getResultTypes())) {
    results.push_back(b.createConversion(
        origType, b.create<ExtractValueOp>(type, resultPack, i)));
  }

  op.replaceAllUsesWith(results);
  op.erase();
  return success();
}

static void lowerSetByRefResults(LLVMBuilder &b, TypeAttrCache &cache,
                                 SetByRefErrorAndResultOp op) {
  b.setLoc(op.getLoc());
  b.setInsertionPoint(op);

  Value hdl = b.createConversion(cache.opaquePtr, op.getCoroutine());
  Value err = b.createConversion(op.getOperand(1));
  Value res = b.createConversion(op.getOperand(2));

  b.create<StoreOp>(err, b.create<GEPOp>(res.getType(), b.getI8Type(), hdl,
                                         GEPArg(b.getPointerByteWidth() * 3),
                                         /*inbounds=*/true));
  b.create<StoreOp>(res, b.create<GEPOp>(err.getType(), b.getI8Type(), hdl,
                                         GEPArg(b.getPointerByteWidth() * 4),
                                         /*inbounds=*/true));
  op.erase();
}

static void lowerGetByRefResults(LLVMBuilder &b, TypeAttrCache &cache,
                                 GetByRefErrorAndResultOp op) {
  b.setLoc(op.getLoc());
  b.setInsertionPoint(op);

  Value hdl = b.createConversion(cache.opaquePtr, op.getCoroutine());
  Type errType = b.convertType(op.getError().getType());
  Type resType = b.convertType(op.getResult().getType());

  Value err = b.create<LoadOp>(
      errType,
      b.create<GEPOp>(cache.opaquePtr, b.getI8Type(), hdl,
                      GEPArg(b.getPointerByteWidth() * 3), /*inbounds=*/true));
  Value res = b.create<LoadOp>(
      resType,
      b.create<GEPOp>(cache.opaquePtr, b.getI8Type(), hdl,
                      GEPArg(b.getPointerByteWidth() * 4), /*inbounds=*/true));
  op.replaceAllUsesWith(ValueRange{err, res});
  op.erase();
}

static void lowerCoroutineResumeAsync(LLVMBuilder &b, TypeAttrCache &cache,
                                      CO::ResumeOp op) {
  b.setLoc(op.getLoc());
  b.setInsertionPoint(op);

  // The resume function is the first element of the context.
  Value hdl = b.createConversion(cache.opaquePtr, op.getCoroutine());
  Value resumeFn = b.create<LoadOp>(cache.opaquePtr, hdl);
  op.replaceAllUsesWith(resumeFn);
  op.erase();
}

static void lowerCoroutineDestroyAsync(LLVMBuilder &b, TypeAttrCache &cache,
                                       DestroyOp op) {
  b.setLoc(op.getLoc());
  b.setInsertionPoint(op);

  // Just free the coroutine context.
  Value hdl = b.createConversion(cache.opaquePtr, op.getCoroutine());
  b.create<POP::AlignedFreeOp>(
      b.createConversion(PointerType::get(cache.i8Type), hdl));
  op.erase();
}

/// When a coroutine completes, it invokes its completion callback, passing
/// itself and its opaque argument.
///
/// ```
/// llvm.func @__kgen_coro_end_fn(%opaqueCtxt: !llvm.ptr<i8>) {
///   %ctxt = llvm.bitcast %opaqueCtxt : !llvm.ptr<i8> to !llvm.ptr<struct(...
///   %clsFnPtr = llvm.getelementptr %ctxt[0, 2, 0]
///   %clsArgPtr = llvm.getelementptr %ctxt[0, 2, 1]
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

  LLVMFuncOp endFn = b.createFunc(
      "__kgen_coro_end_fn",
      LLVMFunctionType::get(LLVMVoidType::get(b.getContext()), cache.opaquePtr),
      Linkage::Internal);

  b.setInsertionPointToStart(endFn.addEntryBlock(b));
  Value ctxt = endFn.getArgument(0);
  Type closureType = LLVMStructType::getLiteral(
      b.getContext(), {cache.opaquePtr, cache.opaquePtr});
  Value closure =
      b.create<GEPOp>(cache.opaquePtr, b.getI8Type(), ctxt,
                      GEPArg(b.getPointerByteWidth()), /*inbounds=*/true);
  auto ptrToAsyncFn =
      b.create<GEPOp>(cache.opaquePtr, closureType, closure,
                      ArrayRef<GEPArg>{0, 0}, /*inbounds=*/true);
  Value closureFn = b.create<LoadOp>(cache.opaquePtr, ptrToAsyncFn);
  Value ptrToClosureArg =
      b.create<GEPOp>(cache.opaquePtr, closureType, closure,
                      ArrayRef<GEPArg>{0, 1}, /*inbounds=*/true);
  Value closureArg = b.create<LoadOp>(cache.opaquePtr, ptrToClosureArg);
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

  // TODO: Replace i8PtrType with opaquePtr next llvm update.
  LLVMFuncOp projFn =
      b.createFunc("__kgen_coro_ctxt_proj_fn",
                   LLVMFunctionType::get(cache.opaquePtr, cache.opaquePtr),
                   Linkage::Internal);

  b.setInsertionPointToStart(projFn.addEntryBlock(b));
  b.create<ReturnOp>(projFn.getArgument(0));
  symtab.insert(projFn);

  b.setLoc(prevLoc);
  return projFn;
}

namespace {
class CoroutineInfo {
public:
  CoroutineInfo(LLVMFuncOp asyncFn, int64_t contextBaseSize,
                LLVMStructType contextType, Value hdlValue, Type hdlType)
      : asyncFn(asyncFn), contextBaseSize(contextBaseSize),
        contextType(contextType), hdlValue(hdlValue), hdlType(hdlType) {
    assert(llvm::isa<LLVMPointerType>(hdlValue.getType()) &&
           "handle type must be a pointer");
  }
  Type getContextType() const { return contextType; }
  Value getHdlValue() const { return hdlValue; }
  Type getHdlType() const { return hdlType; }

  LLVMFuncOp asyncFn;
  int64_t contextBaseSize;

private:
  LLVMStructType contextType;
  Value hdlValue;
  Type hdlType;
};
} // namespace

/// Create a coroutine function by moving the body of the async function into a
/// new function containing the coroutine machinery. In the original function,
/// generate the code to form the coroutine context and handle. If the coroutine
/// has no suspends, `coro-split` will replace the handle value with `undef`,
/// causing everything to explode because we rely on the handle to marshall
/// arguments and results. Instead, we detect if the coroutine has no suspends,
/// and in that case we don't generate any coroutine machinery.
static FailureOr<CoroutineInfo>
createAsyncCoroutine(SymbolTable &symtab, LLVMFuncOp func,
                     ArrayRef<Type> resultTypes, LLVMBuilder &b,
                     TypeAttrCache &cache, LLVMFuncOp coroEndFn,
                     bool noSuspend) {
  b.setLoc(func.getLoc());
  b.clearInsertionPoint();

  // The coroutine context contains the next resume function pointer, a callback
  // closure in the form of `{ void(i8*)*, i8* }`, 2 result slots for in-memory
  // results, the function results, and the function arguments, and then
  // trailing coroutine frame:
  //
  //   { void*, { void*, void* }, void*, void*, ResTs..., ArgTs..., FrameT }
  //
  MLIRContext *ctx = b.getContext();
  SmallVector<Type, 16> contextTypes{
      cache.opaquePtr,
      LLVMStructType::getLiteral(ctx, {cache.opaquePtr, cache.opaquePtr}),
      cache.opaquePtr, cache.opaquePtr};
  for (Type resultType : resultTypes) {
    resultType = b.convertType(resultType);
    if (!resultType)
      return mlir::emitError(func.getLoc(),
                             "could not convert coroutine result type");
    contextTypes.push_back(resultType);
  }
  llvm::append_range(contextTypes, func.getArgumentTypes());
  auto contextType = LLVMStructType::getLiteral(ctx, contextTypes);
  // Compute the base size of the context to populate into the global async
  // function pointer as required by the LLVM async coroutine intrinsics. LLVM's
  // async lowering will update the field with the total size.
  int64_t contextBaseSize = b.getTypeAllocSize(contextType);
  int64_t contextBaseAlign = b.getTypeABIAlign(contextType);

  // The async function implementation is always void(i8*). Create a new
  // function with that signature and make the current function the wrapper
  // function. The wrapper will load the arguments into the context and store
  // the ramp function as the resume function.
  LLVMFuncOp asyncFn = b.createFunc(
      (func.getSymName() + "_af").str(),
      LLVMFunctionType::get(LLVMVoidType::get(ctx), cache.opaquePtr),
      func.getLinkage());
  symtab.insert(asyncFn, func->getIterator());

  // Construct the async function pointer. We have to synthesize this massive
  // global constant.
  auto afpType =
      LLVMStructType::getLiteral(ctx, {cache.i32Type, cache.i32Type});
  // constant struct <{ i32, i32 }>
  auto afp = b.create<GlobalOp>(
      afpType, /*isConstant=*/true, /*linkage=*/Linkage::Internal,
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
  auto addressOfOp = b.create<AddressOfOp>(
      LLVMPointerType::get(b.getContext(), afp.getAddrSpace()),
      afp.getSymName());
  Value afpEndPtr = b.create<GEPOp>(cache.opaquePtr, afpType, addressOfOp,
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
  if (auto scope = DebugInfo::extractScopeFrom<DebugInfo::DISubprogramAttr>(
          asyncFn.getLoc(), DebugInfo::LocWalkPolicy::CalleePriority)) {
    DebugInfo::updateSubprogram(
        asyncFn, asyncFn.getSymNameAttr(),
        DebugInfo::SourceNameAttr::get("async_function", scope.getName()));
  }
  BlockArgument asyncCtxArg =
      asyncFnBody.addArgument(cache.opaquePtr, asyncFn.getLoc());

  // Generate coroutine machinery.
  b.setLoc(asyncFn.getLoc());
  b.setInsertionPointToStart(&asyncFnBody.front());
  Value hdl;
  Type hdleType = b.getI8Type();
  if (noSuspend) {
    // If there are no suspend points, don't create a coroutine.
    hdl = b.create<GEPOp>(cache.opaquePtr, b.getI8Type(), asyncCtxArg,
                          GEPArg(contextBaseSize),
                          /*inbounds=*/true);
  } else {
    // TODO: Remove bitcast next llvm update.
    auto typedPtr =
        b.create<BitcastOp>(cache.i8PtrType, b.create<AddressOfOp>(afp));
    auto coroIdOp = b.create<CallIntrinsicOp>(
        cache.tokenType, "llvm.coro.id.async",
        ValueRange{b.create<ConstantOp>(b.getI32IntegerAttr(contextBaseSize)),
                   b.create<ConstantOp>(b.getI32IntegerAttr(contextBaseAlign)),
                   b.create<ConstantOp>(b.getI32IntegerAttr(0)), typedPtr});
    // TODO: Replace cache.i8PtrType with cache.opaquePtr next llvm update.
    auto typedPtrFromInt = b.create<IntToPtrOp>(
        cache.i8PtrType, b.create<ConstantOp>(b.getI32IntegerAttr(0)));
    hdl = b.create<CoroBeginOp>(cache.opaquePtr, coroIdOp.getResult(0),
                                typedPtrFromInt);
  }

  // The coroutine handle is specially handled by the coroutine splitting pass
  // to be replaced by the frame pointer value in each resume function. It is
  // the tail to the context pointer. Retrieve any context values, including the
  // async context itself, through the handle to prevent the async context value
  // from being considered "spilled" and put into the frame. Storing the async
  // context in itself causes it to be an aliasing pointer. In addition, push
  // all uses the context and arguments to the latest use sites to prevent them
  // from being put on the coroutine frame.

  // We'll use the debug scope if available when replacing argument uses below.
  DebugInfo::DIScopeAttr asyncFnScope;
  if (auto fusedLoc = dyn_cast<mlir::FusedLocWith<DebugInfo::DIScopeAttr>>(
          asyncFn.getLoc()))
    asyncFnScope = fusedLoc.getMetadata();

  // Replace argument uses with values from the async context. Arguments are
  // located at the end.
  constexpr int64_t resOffset = 4; // 4 elements
  int64_t argOffset = resOffset + resultTypes.size();
  for (auto [idx, arg] :
       llvm::enumerate(asyncFnBody.getArguments().drop_back())) {
    if (auto argLoc = arg.getLoc()->findInstanceOf<FileLineColLoc>();
        asyncFnScope && argLoc) {
      b.setLoc(FusedLoc::get(ctx, {argLoc}, asyncFnScope));
    } else {
      b.setLoc(arg.getLoc());
    }
    for (OpOperand &use : llvm::make_early_inc_range(arg.getUses())) {
      // If the coroutine does not suspend, we want to load as early as possible
      // to avoid generating these loads inside of nested loops.
      if (!noSuspend)
        b.setInsertionPoint(use.getOwner());

      // Obtain start of the context from the frame pointer.
      Value contextPtr =
          b.create<GEPOp>(cache.opaquePtr, hdleType, hdl,
                          GEPArg(-contextBaseSize), /*inbounds=*/true);
      Value argPtr = b.create<GEPOp>(
          cache.opaquePtr, contextType, contextPtr,
          ArrayRef<GEPArg>{0, static_cast<int32_t>(argOffset + idx)},
          /*inbounds=*/true);
      use.set(b.create<LoadOp>(arg.getType(), argPtr));
    }
    assert(arg.use_empty() && "didn't replace all uses?");
  }
  asyncFnBody.front().eraseArguments(0, asyncFnBody.getNumArguments() - 1);

  // Returns in the async function are all void. Generate the coroutine end
  // marker that invokes the callback closure.
  asyncFnBody.walk([&](ReturnOp ret) {
    b.setInsertionPoint(ret);
    b.setLoc(ret.getLoc());
    Value contextPtr =
        b.create<GEPOp>(cache.opaquePtr, hdleType, hdl,
                        GEPArg(-contextBaseSize), /*inbounds=*/true);
    if (noSuspend) {
      b.create<CallOp>(coroEndFn, Value(contextPtr));
    } else {
      b.create<CallIntrinsicOp>(
          cache.i1Type, "llvm.coro.end.async",
          ValueRange{hdl, b.create<ConstantOp>(b.getBoolAttr(false)),
                     b.create<AddressOfOp>(coroEndFn), contextPtr});
    }
    ret->eraseOperands(0, ret.getNumOperands());
  });

  // Generate the code to marshall the async function arguments and store the
  // ramp function as the initial resume function.
  b.setLoc(func.getLoc());
  b.setInsertionPointToStart(func.addEntryBlock(b));
  // Read the required context size from the async function pointer.
  // NOTE: Hide the global read behind a "prepare" intrinsic to prevent the size
  // from being inlined into the allocator call until after coroutine splitting,
  // when it gets updated with the frame size.
  // TODO: remove bitcast next llvm update.
  auto typedArg =
      b.create<BitcastOp>(cache.i8PtrType, b.create<AddressOfOp>(afp));
  auto i8prepare = b.create<CallIntrinsicOp>(
      cache.i8PtrType, "llvm.coro.prepare.async", Value(typedArg));
  auto prepare = b.create<BitcastOp>(cache.opaquePtr, i8prepare.getResult(0));
  Value contextSize = b.create<LoadOp>(
      cache.i32Type,
      b.create<GEPOp>(cache.opaquePtr, afp.getType(), prepare,
                      ArrayRef<GEPArg>{0, 1}, /*inbounds=*/true));
  Value allocCall = b.create<POP::AlignedAllocOp>(
      PointerType::get(cache.i8Type),
      ArrayRef<Value>{
          b.createConversion(
              b.getType<IndexType>(),
              b.create<ConstantOp>(cache.indexType, contextBaseAlign)),
          b.createConversion(b.getType<IndexType>(),
                             b.create<ZExtOp>(cache.indexType, contextSize))});
  Value contextValue = b.createConversion(cache.opaquePtr, allocCall);
  auto gep = b.create<GEPOp>(cache.opaquePtr, contextType, contextValue,
                             ArrayRef<GEPArg>{0, 0}, /*inbounds=*/true);
  b.create<StoreOp>(b.create<AddressOfOp>(asyncFn), gep);
  for (auto [idx, arg] : llvm::enumerate(func.getArguments())) {
    Value argPtr = b.create<GEPOp>(
        cache.opaquePtr, contextType, contextValue,
        ArrayRef<GEPArg>{0, static_cast<int32_t>(argOffset + idx)},
        /*inbounds=*/true);
    b.create<StoreOp>(arg, argPtr);
  }
  b.create<ReturnOp>(contextValue);
  return CoroutineInfo(asyncFn, contextBaseSize, contextType, hdl, hdleType);
}

static LogicalResult lowerCoroutineSuspendAsync(
    SymbolTable &symtab, LLVMBuilder &b, CoroutineInfo &coro,
    TypeAttrCache &cache, LLVMFuncOp coroProjFn, SuspendOp op, unsigned index) {
  b.setLoc(op.getLoc());

  // Outline the body of the await into a function.
  llvm::SetVector<Value> uniqueCaptures;
  mlir::getUsedValuesDefinedAbove(op->getRegions(), uniqueCaptures);
  SmallVector<Value, 0> captures = uniqueCaptures.takeVector();

  // Replace uses of the current coroutine handle argument.
  Region &awaitBody = op.getBody();
  b.setInsertionPointToStart(&awaitBody.front());
  Value hdlArg = awaitBody.getArgument(0);
  Value coroHdl = b.createConversion(hdlArg.getType(), hdlArg);
  hdlArg.setType(cache.opaquePtr);
  hdlArg.replaceAllUsesExcept(coroHdl, coroHdl.getDefiningOp());

  SmallVector<Type> captureTypes{cache.opaquePtr};
  for (Value &capture : captures) {
    Type captureType = b.convertType(capture.getType());
    if (!captureType)
      return op.emitError("failed to convert captured value type");
    captureTypes.push_back(captureType);

    Value arg = awaitBody.addArgument(capture.getType(), op.getLoc());
    Value valueInBody = capture;
    if (arg.getType() != captureType) {
      b.setInsertionPointToStart(&awaitBody.front());
      // Materialize source and destination conversions.
      Type srcType = arg.getType();
      arg.setType(captureType);
      arg = b.createConversion(srcType, arg);
      b.setInsertionPoint(op);
      capture = b.createConversion(captureType, capture);
    }
    mlir::replaceAllUsesInRegionWith(valueInBody, arg, op.getRegion());
  }

  b.clearInsertionPoint();
  MLIRContext *ctx = b.getContext();
  LLVMFuncOp suspendFn = b.createFunc(
      (coro.asyncFn.getSymName() + "_suspend_" + Twine(index)).str(),
      LLVMFunctionType::get(LLVMVoidType::get(ctx), captureTypes),
      Linkage::Internal);
  symtab.insert(suspendFn, coro.asyncFn->getIterator());
  cast<SuspendEndOp>(awaitBody.front().getTerminator()).erase();
  suspendFn.getBody().takeBody(awaitBody);

  // If possible, we need to add a subprogram scope to the new function.
  auto fileLoc = op.getLoc()->findInstanceOf<FileLineColLoc>();
  auto scope = DebugInfo::extractScopeFrom<DebugInfo::DISubprogramAttr>(
      op.getLoc(), DebugInfo::LocWalkPolicy::CalleePriority);
  if (scope) {
    // Use unresolved types now for simplicity, these will get resolved during
    // compilation.
    auto mapUnresolvedType = [&](Type type) -> DebugInfo::DIType {
      return DebugInfo::DIUnresolvedMLIRType::get(type);
    };
    auto spType = DebugInfo::DISubroutineType::get(
        ctx, llvm::map_to_vector(captureTypes, mapUnresolvedType), {});

    // The insertion into the symtab might change the name, so we extract it.
    StringAttr suspName = suspendFn.getSymNameAttr();
    Location newLoc = FusedLoc::get(
        op.getContext(), Location(fileLoc),
        scope.cloneWith(DebugInfo::SourceNameAttr::get(
                            "suspend." + Twine(index), scope.getName()),
                        suspName, spType));

    // Okay, we can now overwrite the location with a scoped one. We also set
    // the builder location so anything else we insert (e.g. return) is correct.
    suspendFn->setLoc(newLoc);
    b.setLoc(newLoc);

    // We also need to ensure the ops in the body have matching scope. We can do
    // this by treating them as if they were inlined.
    suspendFn.getBody().walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
      DebugInfo::updateInlinedLoc(op, newLoc);
      if (isa<DebugInfo::SubprogramScoped>(op))
        return WalkResult::skip();
      return WalkResult::advance();
    });
  } else if (fileLoc) {
    suspendFn->setLoc(fileLoc);
    b.setLoc(fileLoc);
  }

  b.setInsertionPointToEnd(&suspendFn.getBody().front());
  b.create<ReturnOp>(ValueRange());

  b.setLoc(op.getLoc());
  b.setInsertionPoint(op);
  // TODO: replace i8PtrType with opaquePtr after llvm update
  Value resumeFn = b.create<CallIntrinsicOp>(
                        cache.i8PtrType, "llvm.coro.async.resume", ValueRange())
                       .getResult(0);
  Value contextPtr =
      b.create<GEPOp>(cache.opaquePtr, coro.getHdlType(), coro.getHdlValue(),
                      GEPArg(-coro.contextBaseSize), /*inbounds=*/true);
  Value resumeFnPtr =
      b.create<GEPOp>(cache.opaquePtr, coro.getContextType(), contextPtr,
                      ArrayRef<GEPArg>{0, 0}, /*inbounds=*/true);
  b.create<StoreOp>(resumeFn, resumeFnPtr);

  // TODO: replace i8PtrType with opaquePtr after llvm update
  auto typedPtr =
      b.create<BitcastOp>(cache.i8PtrType, b.create<AddressOfOp>(coroProjFn));
  SmallVector<Value> suspendAsyncArgs{
      b.create<ConstantOp>(b.getI32IntegerAttr(0)), resumeFn, typedPtr,
      b.create<AddressOfOp>(suspendFn), contextPtr};
  llvm::append_range(suspendAsyncArgs, captures);

  // TODO: replace i8PtrType with opaquePtr after llvm update
  auto suspendRetType = LLVMStructType::getLiteral(
      ctx, {cache.i8PtrType, cache.i8PtrType, cache.i8PtrType});
  // FIXME: For some reason, `call_intrinsic` fails to resolve the overload.
  // TODO: replace i8PtrType with opaquePtr after llvm update
  b.create<POP::ExternalCallOp>(
      suspendRetType, "llvm.coro.suspend.async.sl_p0p0p0s", suspendAsyncArgs,
      b.getFunctionType({cache.i32Type, cache.i8PtrType, cache.i8PtrType},
                        suspendRetType));
  op.erase();
  return success();
}

static LogicalResult
lowerCoroutineFunction(SymbolTable &symtab, LLVMFuncOp func, LLVMBuilder &b,
                       TypeAttrCache &cache,
                       function_ref<LLVMFuncOp()> getCoroEndFn,
                       function_ref<LLVMFuncOp()> getCoroProjFn) {
  // Collect all the relevant ops.
  SmallVector<HandleOp> handles;
  SmallVector<SuspendOp> awaits;
  WalkResult result = func.walk([&](Operation *op) {
    if (auto handle = dyn_cast<HandleOp>(op)) {
      handles.push_back(handle);
    } else if (auto await = dyn_cast<SuspendOp>(op)) {
      awaits.push_back(await);
    } else if (auto setResults = dyn_cast<SetResultsOp>(op)) {
      if (failed(lowerCoroutineSetResults(b, cache, setResults)))
        return WalkResult::interrupt();
    } else if (auto callback = dyn_cast<GetCallbackPtrOp>(op)) {
      if (failed(lowerCoroutineCallback(b, cache, callback)))
        return WalkResult::interrupt();
    } else if (auto getResults = dyn_cast<GetResultsOp>(op)) {
      if (failed(lowerCoroutineGetResults(b, cache, getResults)))
        return WalkResult::interrupt();
    } else if (auto setByRefResults = dyn_cast<SetByRefErrorAndResultOp>(op)) {
      lowerSetByRefResults(b, cache, setByRefResults);
    } else if (auto setByRefResults = dyn_cast<GetByRefErrorAndResultOp>(op)) {
      lowerGetByRefResults(b, cache, setByRefResults);
    } else if (auto resume = dyn_cast<CO::ResumeOp>(op)) {
      lowerCoroutineResumeAsync(b, cache, resume);
    } else if (auto destroy = dyn_cast<DestroyOp>(op)) {
      lowerCoroutineDestroyAsync(b, cache, destroy);
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  // The presence of `co.handle` inside a function indicates
  // that it is a coroutine.
  if (handles.empty()) {
    // If we saw any `co.suspend` operations not inside a coroutine,
    // that likely means an `__await__` function was not marked as
    // `always_inline`.
    if (!awaits.empty()) {
      Operation *op = awaits.front();
      return mlir::emitError(op->getLoc(), "coroutine await operation is not "
                                           "contained inside an async function")
                 .attachNote(op->getParentOfType<LLVMFuncOp>().getLoc())
             << "should this function be marked @always_inline?";
    }
    return success();
  }

  FailureOr<CoroutineInfo> coro =
      createAsyncCoroutine(symtab, func, handles.front().getTypes(), b, cache,
                           getCoroEndFn(), awaits.empty());
  if (failed(coro))
    return failure();

  // The coroutine handle is now the first argument of the coroutine function.
  for (HandleOp op : handles) {
    b.setLoc(op.getLoc());
    b.setInsertionPoint(op);
    Value contextPtr = b.create<GEPOp>(
        cache.opaquePtr, coro->getHdlType(), coro->getHdlValue(),
        GEPArg(-coro->contextBaseSize), /*inbounds=*/true);
    op.replaceAllUsesWith(contextPtr);
    op.erase();
  }
  for (auto [i, op] : llvm::enumerate(awaits)) {
    if (failed(lowerCoroutineSuspendAsync(symtab, b, *coro, cache,
                                          getCoroProjFn(), op, i)))
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
  ImplicitLocOpBuilder opBuilder(getOperation().getLoc(), &getContext());
  LLVMBuilder b(opBuilder, target);

  // Initialize the type cache.
  TypeAttrCache cache = {b.getI1Type(),
                         b.getI8Type(),
                         b.getI32Type(),
                         b.getI64Type(),
                         b.getIntegerType(b.getIndexTypeBitwidth()),
                         LLVMPointerType::get(b.getContext()),
                         b.getType<LLVMTokenType>(),
                         LLVMPointerType::get(b.getContext())};

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
