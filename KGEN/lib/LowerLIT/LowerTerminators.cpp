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
  auto isAsyncCtx = [&](Value hdl) {
    Value promise = getCoroutinePromise(b, errType, hdl);
    Value ctxPtr = b.create<OffsetOp>(promise, one);
    return b.create<PointerBitcastOp>(
        PointerType::get(StructType::get({b.getIndexType(), b.getI8Type()})),
        ctxPtr);
  };
  Value curCtx = isAsyncCtx(curHdl);
  Value ctx = isAsyncCtx(hdl);
  b.create<StoreOp>(b.create<LoadOp>(b.create<POP::StructGEPOp>(curCtx, 1)),
                    b.create<POP::StructGEPOp>(ctx, 1));
  b.create<ExternalCallOp>(
      initializeFnName,
      b.create<PointerBitcastOp>(PointerType::get(b.getI8Type()), ctx)
          .getResult());
}

//===----------------------------------------------------------------------===//
// lowerLexicalTerminators
//===----------------------------------------------------------------------===//

/// Lower all lexical terminators in the function and remove dead code.
static LogicalResult lowerLexicalTerminators(DeclRefType errType,
                                             LIT::FuncOp func) {
  if (func.getBodyRegion().empty())
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
  func.walk([&](Operation *op) {
    if (isa<LIT::ReturnOp, LIT::RaiseOp, LIT::BreakOp, LIT::ContinueOp>(op))
      terminators.push_back(op);

    // If we see a coroutine call from within another coroutine, insert the
    // initialization machinery.
    if (auto call = dyn_cast<LIT::AsyncCallOp>(op); call && func.isAsync())
      createCoroutineInitialize(op, errType, coroHdl, call.getResult());
  });

  // Lower all the terminators as they are encountered.
  auto getErrorOr = [&] {
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
        result = b.create<VariantCreateOp>(getErrorOr(), operands.front());
        operands = result;
      }
      createReturn(resultParams, operands);

    } else if (auto raiseOp = dyn_cast<LIT::RaiseOp>(op)) {
      auto tryOp = raiseOp->getParentOfType<LIT::TryOp>();
      if (tryOp &&
          tryOp.getTryRegion().isAncestor(raiseOp->getBlock()->getParent())) {
        b.create<LIT::TryRaiseOp>(raiseOp.getError());
      } else {
        // TODO(#6449): Can't have result parameters in a function that raises.
        Value err = b.create<VariantCreateOp>(getErrorOr(), raiseOp.getError());
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
    none = b.create<VariantCreateOp>(getErrorOr(), none);
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
static LogicalResult lowerThrowsAndAsync(DeclRefType errType, Operation *op) {
  // Find every throwing and async call. Track async calls that also throw.
  SmallVector<KGENCallOpInterface> throwingCalls;
  SmallVector<LIT::AsyncCallOp> asyncCalls;
  op->walk([&](Operation *op) {
    if (auto call = dyn_cast<KGENCallOpInterface>(op)) {
      if (isa<GeneratorInterfaceOp>(*call))
        return;
      if (cast<SignatureType>(call.getCallee().getType()).isThrows())
        throwingCalls.push_back(call);
    } else if (auto call = dyn_cast<LIT::AsyncCallOp>(op)) {
      asyncCalls.push_back(call);
    }
  });

  // Replace every throwing signature type with a variant result type and every
  // async signature type with a coroutine handle result type.
  OpBuilder b(op);
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
  replacer.recursivelyReplaceElementsIn(
      op, /*replaceAttrs=*/true, /*replaceLocs=*/false, /*replaceTypes=*/true);

  // Process async calls. Async calls implicitly wrap the function result type
  // in a coroutine, but now that the signatures are rewritten, they aren't
  // needed anymore.
  for (LIT::AsyncCallOp call : asyncCalls) {
    // Replace the async call with a `call_param`.
    b.setInsertionPoint(call);
    auto newCall = b.create<CallParamOp>(
        call.getLoc(), call->getResultTypes(), call.getCallee(),
        call.getParamDeclsAttr(), call.getOperands());
    Value result = newCall.getResult(0);
    call.replaceAllUsesWith(result);
    call.erase();
  }

  // Process throwing calls.
  for (KGENCallOpInterface call : throwingCalls) {
    // FIXME: `kgen.addressof` returns a `FunctionType` (function pointer),
    // which is an important feature to have, but that means we can't globally
    // see calls to a throwing function pointer.
    if (isa<AddressOfOp>(call))
      return call.emitError("FIXME: cannot take address of throwing function");

    // We need to update the result types of the function.
    Type resultType = call->getResultTypes().front();
    call->getResult(0).setType(VariantType::get({errType, resultType}));
    b.setInsertionPointAfter(call);
    auto unwrap = b.create<LIT::UnwrapOrPropagateOp>(call.getLoc(), resultType,
                                                     call->getResult(0));
    call->getResult(0).replaceAllUsesExcept(unwrap, unwrap);
  }
  return success();
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
    // Look for an error type declaration.
    DeclRefType errType;
    getOperation()->walk([&](StructDeclOp decl) {
      if (decl.getName() != "Error" || !decl.getInputParamDecls().empty())
        return;
      // Reconstruct the full symbol reference.
      errType = DeclRefType::get(
          LIT::getFullyResolvedSymbolRef(cast<mlir::SymbolOpInterface>(*decl)));
    });
    // Walk all top-level functions.
    WalkResult result =
        getOperation()->walk<mlir::WalkOrder::PreOrder>([&](LIT::FuncOp func) {
          if (failed(lowerLexicalTerminators(errType, func)))
            return WalkResult::interrupt();
          return WalkResult::skip();
        });
    if (result.wasInterrupted())
      return signalPassFailure();

    // Lower all functions that throw.
    if (!errType)
      return;
    if (failed(lowerThrowsAndAsync(errType, getOperation())))
      return signalPassFailure();
  }
};
} // namespace
