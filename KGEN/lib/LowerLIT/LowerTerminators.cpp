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
#include "mlir/IR/PatternMatch.h"

using namespace M;
using namespace KGEN;
using namespace POP;

//===----------------------------------------------------------------------===//
// Coroutine Machinery
//===----------------------------------------------------------------------===//

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
  Value promise =
      getCoroutinePromise(b, errType, cast<TypedValue<CoroutineType>>(hdl));
  for (auto [idx, result] : llvm::enumerate(results))
    b.create<StoreOp>(result, b.create<POP::StructGEPOp>(promise, idx));
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
                                             LIT::FuncOp func) {
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
      // Replace the async call with a `call_param`.
      ImplicitLocOpBuilder b(call.getLoc(), OpBuilder(call));
      auto newCall = b.create<CallParamOp>(
          call.getLoc(), call->getResultTypes(), call.getCallee(),
          call.getParamDeclsAttr(), call.getOperands());
      Value result = newCall.getResult(0);
      call.replaceAllUsesWith(result);
      toErase.push_back(call);
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
      !func.getResultParams().empty())
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
        sigType.getInputParams(), sigType.getResultParams(),
        b.getFunctionType(sigType.getValueInputs(), type),
        b.getAttr<MetadataAttr>(
            sigType.getValueInputConventions(), sigType.getDefaultArguments(),
            bitEnumClear(sigType.getFnEffects(),
                         FnEffects::Throws | FnEffects::Async)));
  });
  replacer.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);
}

//===----------------------------------------------------------------------===//
// lowerNestedFunctions
//===----------------------------------------------------------------------===//

/// Get a top-level function, lower all functions nested inside that function.
static void lowerNestedFunctions(LIT::FuncOp topLevelFunc,
                                 mlir::SymbolTableAnalysis &analysis) {
  auto getNestedFuncRef = [&](SymbolRefAttr ref) -> LIT::FuncOp {
    // Perform the symbol lookup and check if the referenced symbol lives inside
    // this function.
    SmallVector<Operation *> symbols;
    if (failed(analysis.getSymbolTables().lookupSymbolIn(
            analysis.getTopLevelOp<ModuleOp>(), ref, symbols)) ||
        !llvm::is_contained(symbols, topLevelFunc))
      return {};
    auto lastSymbol = cast<LIT::FuncOp>(symbols.back());
    if (lastSymbol == topLevelFunc)
      return {};
    return lastSymbol;
  };

  // Demote direct calls to nested functions to `call_param` so the callee can
  // be rewritten.
  topLevelFunc.walk([&](CallOp call) {
    if (!getNestedFuncRef(call.getCalleeSymbol()))
      return;
    mlir::IRRewriter b{OpBuilder(call)};
    b.replaceOpWithNewOp<CallParamOp>(
        call, call.getResultTypes(), call.getCallee(), call.getParamDeclsAttr(),
        call.getOperands());
  });
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](SymbolConstantAttr ref) -> Attribute {
    LIT::FuncOp nestedFunc = getNestedFuncRef(ref.getSymbol());
    if (!nestedFunc)
      return ref;
    // Take the leaf reference as the parameter name. The parser guarntees it
    // has no collisions with parameters.
    TypedAttr newRef = ParamDeclRefAttr::get(ref.getSymbol().getLeafReference(),
                                             nestedFunc.getSignature());
    if (ref.getParamValues().empty())
      return newRef;
    // If the symbol constant had bindings, create a `bind_signature`.
    SmallVector<TypedAttr> operands;
    operands.push_back(newRef);
    for (ParamBindAttr bind : ref.getParamValues())
      operands.push_back(bind.getValue());
    return ParamOperatorAttr::get(POC::BindSignature, operands);
  });
  replacer.recursivelyReplaceElementsIn(topLevelFunc, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);

  topLevelFunc.walk([&](LIT::FuncOp func) {
    if (func == topLevelFunc)
      return;
    // Process a nested function by lowering it straight to a
    // `kgen.param.declare.region`. We need to replace all the symbol
    // references within the function. The parser ensures that the symbol name
    // is unique with parameters.
    ImplicitLocOpBuilder b(func.getLoc(), OpBuilder(func));
    auto region = b.create<ParamDeclareRegionOp>(
        ParamDeclAttr::get(func.getSymNameAttr(), func.getSignature()),
        func.getSignature(), ArrayRef<ConstraintAttr>(), /*isolated=*/false,
        func.getAlwaysInlineLevel());
    region.getBodyRegion().takeBody(func.getBodyRegion());
    func.erase();
  });
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
    getOperation().walk([&](LIT::StructDeclOp decl) {
      if (decl.getName() != "Error" || !decl.getInputParamDecls().empty())
        return;
      // Reconstruct the full symbol reference.
      errType = DeclRefType::get(
          LIT::getFullyResolvedSymbolRef(cast<mlir::SymbolOpInterface>(*decl)));
    });
    // Walk all functions.
    WalkResult result = getOperation().walk([&](LIT::FuncOp func) {
      if (failed(lowerLexicalTerminators(errType, func)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return signalPassFailure();

    // Lower all functions that throw.
    if (errType)
      lowerThrowsAndAsync(errType, getOperation());

    // Lower nested functions by converting them to region declarations. Walk
    // all top-level functions and gather nested functions.
    auto &analysis = getAnalysis<mlir::SymbolTableAnalysis>();
    getOperation()->walk<mlir::WalkOrder::PreOrder>([&](LIT::FuncOp func) {
      lowerNestedFunctions(func, analysis);
      return WalkResult::skip();
    });
  }
};
} // namespace
