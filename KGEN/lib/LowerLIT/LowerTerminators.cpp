//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/HLCFDialect/HLCFOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace M;
using namespace KGEN;
using namespace POP;

//===----------------------------------------------------------------------===//
// Coroutine Machinery
//===----------------------------------------------------------------------===//

static StringLiteral completeFnName = "KGEN_CompilerRT_LLCL_Complete";
static StringLiteral initializeFnName =
    "KGEN_CompilerRT_LLCL_InitializeContext";

/// Generate the code to get the coroutine promise, checking the coroutine
/// throws.
static Value getCoroutinePromise(ImplicitLocOpBuilder &b, DeclRefType errType,
                                 TypedValue<CoroutineType> hdl) {
  StructType promiseType;
  if (hdl.getType().getSignature().isThrows())
    promiseType = StructType::get(
        VariantType::get({errType, hdl.getType().getResultTypes().front()}));
  else
    promiseType = b.getType<StructType>(hdl.getType().getResultTypes());
  return b.create<CoroutinePromiseOp>(PointerType::get(promiseType), hdl);
}

/// Generate the code to store the results of the coroutine into the coroutine
/// promise. This code is inserted at every return site.
static void createCoroutineFinalize(ImplicitLocOpBuilder &b,
                                    DeclRefType errType, Value hdl,
                                    ValueRange results) {
  Value promise = getCoroutinePromise(b, errType, hdl);
  for (auto [idx, result] : llvm::enumerate(results))
    b.create<StoreOp>(result, b.create<POP::StructGEPOp>(promise, idx));

  // Insert the runtime call to indicate that the coroutine is complete.
  Value ctxPtr =
      b.create<OffsetOp>(promise, b.create<mlir::index::ConstantOp>(1));
  Value ctx =
      b.create<PointerBitcastOp>(PointerType::get(b.getI8Type()), ctxPtr);
  b.create<ExternalCallOp>(completeFnName, ctx);
}

/// Generate the code to propagate the runtime pointer in the async context and
/// runtime call to initialize the async chain.
static void createCoroutineInitialize(Operation *call, DeclRefType errType,
                                      Value curHdl, Value hdl) {
  ImplicitLocOpBuilder b(call->getLoc(), call->getContext());
  b.setInsertionPointAfter(call);
  Value one = b.create<mlir::index::ConstantOp>(1);
  auto getAsyncCtx = [&](Value hdl) {
    Value promise = getCoroutinePromise(b, errType, hdl);
    Value ctxPtr = b.create<OffsetOp>(promise, one);
    return b.create<PointerBitcastOp>(
        PointerType::get(StructType::get({b.getIndexType(), b.getI8Type()})),
        ctxPtr);
  };
  Value curCtx = getAsyncCtx(curHdl);
  Value ctx = getAsyncCtx(hdl);
  b.create<StoreOp>(b.create<LoadOp>(b.create<POP::StructGEPOp>(curCtx, 1)),
                    b.create<POP::StructGEPOp>(ctx, 1));
  b.create<ExternalCallOp>(
      initializeFnName,
      b.create<PointerBitcastOp>(PointerType::get(b.getI8Type()), ctx)
          .getResult());
}

/// Retrieve the results of a coroutine.
static SmallVector<Value> getCoroutineResults(ImplicitLocOpBuilder &b,
                                              SymbolConstantAttr resumeFnRef,
                                              DeclRefType errType, Value curHdl,
                                              LIT::AsyncAwaitOp op) {
  Value promise = getCoroutinePromise(b, errType, op.getCoroutine());
  Value one = b.create<mlir::index::ConstantOp>(1);
  Value ctxPtr = b.create<OffsetOp>(promise, one);
  Value ctx =
      b.create<PointerBitcastOp>(PointerType::get(b.getI8Type()), ctxPtr);
  Value resumeFn =
      b.create<AddressOfOp>(resumeFnRef.getType().getValues(), resumeFnRef,
                            ParamDeclArrayAttr::get(b.getContext(), {}));
  auto awaitOp = b.create<CoroutineAwaitOp>();
  auto toOpaqueHdl = [&](Value hdl) {
    // FIXME: We need an op to convert to opaque pointer.
    return b
        .create<mlir::UnrealizedConversionCastOp>(
            PointerType::get(b.getI8Type()), hdl)
        .getResult(0);
  };
  {
    OpBuilder::InsertionGuard guard(b);
    b.createBlock(&awaitOp.getBody());
    b.create<ExternalCallOp>("KGEN_CompilerRT_LLCL_ExecuteAndResume",
                             ValueRange{resumeFn,
                                        toOpaqueHdl(op.getCoroutine()), ctx,
                                        toOpaqueHdl(curHdl)});
  }

  SmallVector<Value> results;
  for (auto [idx, resultType] :
       llvm::enumerate(op.getCoroutine().getType().getResultTypes()))
    results.push_back(
        b.create<LoadOp>(b.create<POP::StructGEPOp>(promise, idx)));
  return results;
}

/// Given the result of a throwable call, generate the code to check if the
/// result type is an error, and if so, propagate the error.
static Value createUnwrapOrPropagate(ImplicitLocOpBuilder &b, LIT::FuncOp func,
                                     Value errOr, DeclRefType errType,
                                     Type type, Value coroHdl) {
  auto ifOp =
      b.create<HLCF::IfOp>(type, b.create<POP::VariantIsOp>(errOr, errType));

  b.createBlock(&ifOp.getElseRegion());
  Value value = b.create<POP::VariantGetOp>(type, errOr);
  b.create<HLCF::YieldOp>(b.getLoc(), value);

  b.createBlock(&ifOp.getThenRegion());
  Value err = b.create<POP::VariantGetOp>(errType, errOr);
  if (auto tryOp = ifOp->getParentOfType<LIT::TryOp>();
      tryOp && tryOp.getTryRegion().findAncestorOpInRegion(*ifOp)) {
    b.create<LIT::TryRaiseOp>(err);
  } else {
    Value result = b.create<POP::VariantCreateOp>(
        POP::VariantType::get({errType, func.getResultType()}), err);
    if (func.isAsync()) {
      createCoroutineFinalize(b, errType, coroHdl, result);
      result = coroHdl;
    }
    b.create<HLCF::ReturnOp>(result);
  }
  return ifOp.getResult(0);
}

//===----------------------------------------------------------------------===//
// lowerLexicalTerminators
//===----------------------------------------------------------------------===//

/// Lower all lexical terminators in the function and remove dead code.
static LogicalResult lowerLexicalTerminators(DeclRefType errType,
                                             SymbolConstantAttr resumeFnRef,
                                             LIT::FuncOp func) {
  if (func.getIsInterface())
    return success();
  if (func.isThrows() && !errType)
    return func.emitError("function throws but no 'Error' type was found");

  // If this is an async function, insert a coroutine handle.
  Value coroHdl;
  if (func.isAsync()) {
    auto b = OpBuilder::atBlockBegin(func.getBody());
    coroHdl = b.create<CoroutineHandleOp>(
        func.getLoc(), CoroutineType::get(func.getSignature()));
  }

  // Collect all the terminators first to avoid iterator invalidation.
  SmallVector<Operation *> terminators;
  // In a pre-order walk, we cannot erase as we walk.
  SmallVector<Operation *> toErase;
  WalkResult result = func.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (op != func && isa<LIT::FuncOp>(op)) {
      return WalkResult::skip();

    } else if (isa<LIT::ReturnOp, LIT::RaiseOp, LIT::BreakOp, LIT::ContinueOp>(
                   op)) {
      terminators.push_back(op);

    } else if (auto call = dyn_cast<KGENCallOpInterface>(op)) {
      if (!cast<SignatureType>(call.getCallee().getType()).isThrows())
        return WalkResult::advance();
      // FIXME: `kgen.addressof` returns a `FunctionType` (function pointer),
      // which is an important feature to have, but that means we can't globally
      // see calls to a throwing function pointer.
      if (isa<AddressOfOp>(call)) {
        call.emitError("FIXME: cannot take address of throwing function");
        return WalkResult::interrupt();
      }

      // We need to update the result types of the function.
      ImplicitLocOpBuilder b(call.getLoc(), OpBuilder(call->getNextNode()));
      Operation *newCall = b.clone(*call);
      Type resultType = call->getResultTypes().front();
      newCall->getResult(0).setType(VariantType::get({errType, resultType}));
      call->replaceAllUsesWith(ArrayRef(createUnwrapOrPropagate(
          b, func, newCall->getResult(0), errType, resultType, coroHdl)));
      toErase.push_back(call);

    } else if (auto call = dyn_cast<LIT::AsyncCallOp>(op)) {
      // If we see a coroutine call from within another coroutine, insert the
      // initialization machinery.
      if (func.isAsync())
        createCoroutineInitialize(op, errType, coroHdl, call.getResult());
      // Replace the async call with a `call_param`.
      ImplicitLocOpBuilder b(call.getLoc(), OpBuilder(call));
      auto newCall = b.create<CallParamOp>(
          call.getLoc(), call->getResultTypes(), call.getCallee(),
          call.getParamDeclsAttr(), call.getOperands());
      Value result = newCall.getResult(0);
      call.replaceAllUsesWith(result);
      toErase.push_back(call);

    } else if (auto await = dyn_cast<LIT::AsyncAwaitOp>(op)) {
      ImplicitLocOpBuilder b(await.getLoc(), OpBuilder(await));
      SmallVector<Value> results =
          getCoroutineResults(b, resumeFnRef, errType, coroHdl, await);
      auto coroSig = await.getCoroutine().getType().getSignature();
      if (coroSig.isThrows()) {
        results.assign(1, createUnwrapOrPropagate(
                              b, func, results.front(), errType,
                              coroSig.getValueResults().front(), coroHdl));
      }
      await.replaceAllUsesWith(results);
      toErase.push_back(await);
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  for (Operation *op : toErase)
    op->erase();

  // Lower all the terminators as they are encountered.
  auto errorOr = [&] {
    return VariantType::get({errType, func.getResultType()});
  };
  LIT::ReturnOp firstResultParamsReturn;
  ParameterExprArrayAttr resultParams;
  SmallVector<Block *> deadBlocks;
  for (Operation *op : terminators) {
    // Ignore dead operations.
    if (op->getBlock() != &op->getParentRegion()->front())
      continue;

    ImplicitLocOpBuilder b(op->getLoc(), OpBuilder(op));
    auto createReturn = [&](ArrayRef<TypedAttr> params, ValueRange operands) {
      // In a coroutine, materialize the finalize machinery.
      if (func.isAsync()) {
        createCoroutineFinalize(b, errType, coroHdl, operands);
        operands = coroHdl;
      }
      if (op->getParentOp() == func)
        b.create<KGEN::ReturnOp>(params, operands);
      else
        b.create<HLCF::ReturnOp>(operands);
    };
    if (auto returnOp = dyn_cast<LIT::ReturnOp>(op)) {
      if (resultParams && returnOp.getParametersAttr() != resultParams) {
        return returnOp
                   .emitError("function return defines different result "
                              "meta-parameters than previous return statement")
                   .attachNote(firstResultParamsReturn.getLoc())
               << "see conflicting result meta-parameters here";
      }
      firstResultParamsReturn = returnOp;
      resultParams = returnOp.getParametersAttr();

      ValueRange operands = returnOp.getOperands();
      Value result;
      if (func.isThrows()) {
        result = b.create<VariantCreateOp>(errorOr(), operands.front());
        operands = result;
      }
      createReturn(resultParams, operands);

    } else if (auto raiseOp = dyn_cast<LIT::RaiseOp>(op)) {
      auto tryOp = raiseOp->getParentOfType<LIT::TryOp>();
      if (tryOp &&
          tryOp.getTryRegion().isAncestor(raiseOp->getBlock()->getParent())) {
        b.create<LIT::TryRaiseOp>(raiseOp.getError());
      } else {
        // TODO(#6449): Can't have result parameters in a function that
        // raises.
        Value err = b.create<VariantCreateOp>(errorOr(), raiseOp.getError());
        createReturn({}, err);
      }

    } else if (auto breakOp = dyn_cast<LIT::BreakOp>(op)) {
      b.create<HLCF::BreakOp>();
    } else {
      assert(isa<LIT::ContinueOp>(op) && "unknown terminator");
      b.create<HLCF::ContinueOp>();
    }

    // Check and warn about dead code.
    if (!op->getNextNode()->hasTrait<OpTrait::IsTerminator>())
      op->getNextNode()->emitWarning("unreachable code after ")
          << op->getName().stripDialect() << " statement";

    // Mark all subsequent operations as dead.
    deadBlocks.push_back(op->getBlock()->splitBlock(op));
  }

  // Remove all dead code.
  for (Block *block : deadBlocks)
    block->erase();

  // Check if the function lacks a top-level terminator. If the function
  // nominally returns `!lit.none`, then insert one. Otherwise, emit an error.
  Operation *terminator = func.getBody()->getTerminator();
  if (!isa<LIT::EndFuncOp>(terminator))
    return success();
  if (func.getNumResults() != 1 || !isa<LIT::NoneType>(func.getResultType()) ||
      !func.getResultParamTypes().empty())
    return terminator->emitError(
        "return expected at end of function with results");

  ImplicitLocOpBuilder b(func.getLoc(), OpBuilder(terminator));
  Value none = b.create<ParamConstantOp>(b.getAttr<LIT::NoneAttr>());
  if (func.isThrows())
    none = b.create<VariantCreateOp>(errorOr(), none);
  if (func.isAsync()) {
    createCoroutineFinalize(b, errType, coroHdl, none);
    none = coroHdl;
  }
  b.create<KGEN::ReturnOp>(ArrayRef<TypedAttr>(), none);
  terminator->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// lowerThrowsAndAsync
//===----------------------------------------------------------------------===//

/// Lower `<...>(...) throws -> T` to `<...>(...) -> ErrorOr<T>`, and then
/// `<...>(...) async -> T` to `<...>(...) -> Task<T>` in that order.
/// So, an `async|throws` function will become `Task<ErrorOr<T>>`. Update all
/// callsites to signature types to reflect this change.
static void lowerThrowsAndAsync(DeclRefType errType, Operation *op) {
  // Replace every throwing signature type with a variant result type and
  // every async signature type with a coroutine handle result type.
  Builder b(op->getContext());
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](SignatureType sigType) {
    if (!bitEnumContainsAny(sigType.getFnEffects(),
                            FnEffects::Throws | FnEffects::Async))
      return sigType;
    // Wrap the result type with the appropriate type.
    Type type = sigType.getValueResults().front();
    if (sigType.isThrows())
      type = VariantType::get({errType, type});
    if (sigType.isAsync())
      type = CoroutineType::get(type);
    // Clear the `throws` and `async` bits.
    return SignatureType::get(
        sigType.getInputParams(), sigType.getResultParamTypes(),
        b.getFunctionType(sigType.getValueInputs(), type),
        b.getAttr<ConventionsAttr>(
            sigType.getValueInputConventions(),
            bitEnumClear(sigType.getFnEffects(),
                         FnEffects::Throws | FnEffects::Async)));
  });
  replacer.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERLITTERMINATORS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerTerminatorsPass
    : impl::LowerLITTerminatorsBase<LowerTerminatorsPass> {
  using LowerLITTerminatorsBase::LowerLITTerminatorsBase;

  void runOnOperation() override {
    // FIXME: We need to generate a resume function to pass to the LLCL shim.
    // It would be better to construct and pass a closure.
    ImplicitLocOpBuilder b(getOperation().getLoc(), &getContext());
    auto i8Ptr = PointerType::get(b.getI8Type());
    auto resumeFn = b.create<GeneratorOp>(
        "__kgen_coro_resume",
        SignatureType::get(&getContext(), PointerType::get(b.getI8Type()), {}),
        ArrayRef<ConstraintAttr>(), nullptr);
    b.createBlock(&resumeFn.getBodyRegion());
    auto opaqueHdl = b.create<mlir::UnrealizedConversionCastOp>(
        CoroutineType::get(NoneType::get(&getContext())),
        resumeFn.getBody()->addArgument(i8Ptr, b.getLoc()));
    b.create<CoroutineResumeOp>(opaqueHdl.getResult(0));
    b.create<KGEN::ReturnOp>(ArrayRef<TypedAttr>(), ValueRange());
    StringAttr resumeFnName =
        getAnalysis<SymbolTableAnalysis>().getTopLevelSymbolTable().insert(
            resumeFn);
    if (resumeFnName != resumeFn.getSymNameAttr())
      resumeFn.setSymNameAttr(resumeFnName);
    auto resumeFnRef = SymbolConstantAttr::get(
        FlatSymbolRefAttr::get(resumeFnName), resumeFn.getSignature());

    // Look for an error type declaration.
    DeclRefType errType;
    getOperation().walk([&](StructDeclOp decl) {
      if (decl.getName() != "Error" || !decl.getInputParamDecls().empty())
        return;
      // Reconstruct the full symbol reference.
      errType = DeclRefType::get(
          LIT::getFullyResolvedSymbolRef(cast<mlir::SymbolOpInterface>(*decl)));
    });
    // Walk all functions.
    WalkResult result = getOperation().walk([&](LIT::FuncOp func) {
      if (failed(lowerLexicalTerminators(errType, resumeFnRef, func)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return signalPassFailure();

    // Lower all functions that throw.
    if (!errType)
      return;
    lowerThrowsAndAsync(errType, getOperation());
  }
};
} // namespace
