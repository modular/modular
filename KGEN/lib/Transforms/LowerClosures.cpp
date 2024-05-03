//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CODialect/COOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/Threading/Shared.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/RWMutex.h"

using namespace M;
using namespace KGEN;
using namespace POP;

//===----------------------------------------------------------------------===//
// liftClosureRegion
//===----------------------------------------------------------------------===//

/// Isolate a closure region from above by replacing uses of captured SSA values
/// in the region with block arguments. Certain zero-cost operations, like
/// constants, should be cloned into the region instead of passed as a capture,
/// since the latter has additional overhead.
///
/// The captured values, excluding the cloned values, are populate into
/// `captures`.
static void liftClosureRegion(Region &body, SmallVectorImpl<Value> &captures,
                              mlir::DominanceInfo &domInfo,
                              bool formTransitiveClosure = false) {
  // Isolate the region from above.
  llvm::SetVector<Value> captureSet;
  mlir::getUsedValuesDefinedAbove(body, captureSet);
  bool sortCaptureSet = false;

  if (formTransitiveClosure) {
    // Note: The size of `captureSet` is changing.
    for (unsigned i = 0; i < captureSet.size(); ++i) {
      Value capture = captureSet[i];
      Operation *capturingOp = capture.getDefiningOp();
      if (!capturingOp)
        continue;
      for (Value operand : capturingOp->getOperands()) {
        if (!captureSet.insert(operand)) {
          // Found an operand that is already in the captureSet.
          // The captureSet will need to be sorted for proper dominance order to
          // clone and replace in the region.
          sortCaptureSet = true;
        }
      }
    }
  }

  llvm::SmallVector<Value> captureValues = captureSet.takeVector();
  if (sortCaptureSet) {
    // Sort the captureSet in the right order for dominance if .
    std::stable_sort(captureValues.begin(), captureValues.end(),
                     [&](Value v1, Value v2) {
                       if (!v2.getDefiningOp())
                         return false;
                       return !domInfo.dominates(v1, v2.getDefiningOp());
                     });
  }

  for (Value capture : captureValues) {
    Operation *capturingOp = capture.getDefiningOp();
    // Clone ConstantLike operations into the region.
    if (capturingOp && (formTransitiveClosure ||
                        capturingOp->hasTrait<OpTrait::ConstantLike>())) {
      auto b = OpBuilder::atBlockBegin(&body.front());
      Operation *cloned = b.clone(*capturingOp);
      // We update the location of the cloned constant, as if it was inlined
      // into the region.
      cloned->setLoc(mlir::CallSiteLoc::get(capturingOp->getLoc(),
                                            body.getParentOp()->getLoc()));

      for (auto [orig, replacement] :
           llvm::zip(capturingOp->getResults(), cloned->getResults()))
        replaceAllUsesInRegionWith(orig, replacement, body);
    } else {
      // Otherwise these are captured variables and we need to pass them as
      // arguments to the block body.
      BlockArgument arg = body.addArgument(capture.getType(), capture.getLoc());
      mlir::replaceAllUsesInRegionWith(capture, arg, body);
      captures.push_back(capture);
    }
  }
}

//===----------------------------------------------------------------------===//
// lowerAsyncExecute
//===----------------------------------------------------------------------===//

/// Generate the code to store the results of the coroutine into the coroutine
/// promise. This code is inserted at every return site.
static void createCoroutineFinalize(ImplicitLocOpBuilder &b, Value hdl,
                                    Operation *ret) {
  b.setLoc(ret->getLoc());
  b.setInsertionPoint(ret);
  Value promise = b.create<CO::CoroutinePromiseOp>(hdl);
  for (auto [idx, result] : llvm::enumerate(ret->getOperands()))
    b.create<StoreOp>(result, b.create<KGEN::StructGEPOp>(promise, idx));
}

/// Lower an async execute by making it isolated from above and hoisting it into
/// a function. The conversion is done post-order, so there should be no nested
/// `lit.async.execute` operations nested beneath this one when the function
/// gets called.
static void lowerAsyncExecute(FuncOp parent, LIT::AsyncExecuteOp op,
                              Shared<SymbolTable &> &sharedTable,
                              size_t &nameCounter,
                              mlir::DominanceInfo &domInfo) {
  // Gather location info from encoded CallLoc, and set the op's location to the
  // unencoded location so that inlined body ops get the right callsite loc.
  LocationAttr callLoc = op.getCallLocAttr();
  Location unencodedLoc = op.getLocNoInlined();
  op.getOperation()->setLoc(unencodedLoc);

  SmallVector<Value> captures;
  Region &body = op.getBodyRegion();
  liftClosureRegion(body, captures, domInfo);

  // Insert the coroutine handle.
  ImplicitLocOpBuilder b(op.getLocNoInlined(),
                         OpBuilder::atBlockBegin(&body.front()));
  Value coroHdl = b.create<CO::CoroutineHandleOp>(op.getType());

  // Replace all returns.
  op.walk([&](ReturnOp ret) {
    createCoroutineFinalize(b, coroHdl, ret);
    b.create<ReturnOp>(coroHdl);
    ret.erase();
  });

  // Move the body into a function. The function is not valid to inline.
  b.clearInsertionPoint();
  b.setLoc(op.getLoc());
  StringAttr name = b.getStringAttr(parent.getSymName() + "_async_closure_" +
                                    Twine(nameCounter++));
  // TODO: What conventions do we use for captures.
  SmallVector<ArgConvention> conventions(body.getArgumentTypes().size(),
                                         ArgConvention::None);
  auto sig = SignatureType::get(
      b.getFunctionType(body.getArgumentTypes(), op.getType()), conventions);
  auto lifted = b.create<FuncOp>(name, sig);
  lifted.getBodyRegion().takeBody(body);

  // Insert the function into the symbol table. Lock the symbol table, which
  // also locks the linked list of operations in the module block.
  name = sharedTable.modify(
      [lifted, it = parent->getIterator()](SymbolTable &symtab) {
        return symtab.insert(lifted, it);
      });

  // Create the call with a dummy callee.
  b.setInsertionPoint(op);
  if (callLoc)
    b.setLoc(callLoc);
  auto call = b.create<CallOp>(
      op.getType(), SymbolConstantAttr::get(FlatSymbolRefAttr::get(name), sig),
      captures);
  op.replaceAllUsesWith(call);
  op.erase();

  if (auto scope = lifted.getSubprogramScope()) {
    DebugInfo::updateSubprogram(
        lifted, lifted.getSymNameAttr(),
        DebugInfo::SourceNameAttr::get(
            "async_closure." + Twine(nameCounter - 1), scope.getName()));
  }
}

//===----------------------------------------------------------------------===//
// lowerStageClosure
//===----------------------------------------------------------------------===//

/// Lower a closure by closing the region over its captured SSA values and
/// lifting it into a top-level function.
static void lowerStageClosure(FuncOp parent, StageClosureOp op,
                              Shared<SymbolTable &> &sharedTable,
                              size_t &nameCounter,
                              mlir::DominanceInfo &domInfo) {
  // Gather location info from encoded CallLoc, and set the op's location to the
  // unencoded location so that inlined body ops get the right callsite loc.
  LocationAttr callLoc = op.getCallLocAttr();
  Location unencodedLoc = op.getLocNoInlined();
  op.getOperation()->setLoc(unencodedLoc);

  Region &body = op.getBodyRegion();
  unsigned numArgs = body.getNumArguments();
  SmallVector<Value> captures;
  // If the `stage_closure` is not capturing, then this is an inline (?)
  // function pointer. Force the transitive closure of operations to be cloned
  // into the body to isolate it.
  liftClosureRegion(body, captures, domInfo, !op.getType().isCapturing());
  // Add the captured arguments to the front so they can be partially applied by
  // `kgen.create_closure`.
  std::rotate(body.getArguments().begin(),
              body.getArguments().begin() + numArgs, body.getArguments().end());

  // We need to ensure we have conventions for each argument.
  // TODO: what convention do we use for the captures?
  SignatureType oldSig = op.getType();
  SmallVector<ArgConvention> newConventions(
      body.getArguments().size() - numArgs, ArgConvention::None);
  ArrayRef<ArgConvention> oldConventions = oldSig.getArgConventions();
  assert(oldConventions.size() == numArgs);
  newConventions.append(oldConventions.begin(), oldConventions.end());

  // Construct the signature of the lifted body.
  MLIRContext *ctx = op.getContext();
  ImplicitLocOpBuilder b(op.getLoc(), ctx);
  FunctionType functionType =
      b.getFunctionType(body.getArgumentTypes(), oldSig.getResults());
  auto sig = SignatureType::get(functionType, /*inputParamTypes=*/{},
                                /*resultParamTypes=*/{}, newConventions,
                                oldSig.getFnEffects());

  // Create the lifted function. Make sure it doesn't get inlined back.
  StringAttr name;
  if (auto nameMaybe = op->getAttrOfType<StringAttr>("name"))
    name = nameMaybe;
  else
    name = b.getStringAttr(parent.getSymName() + "_closure_" +
                           Twine(nameCounter++));
  auto lifted = b.create<FuncOp>(op.getLoc(), name, sig);
  lifted.getBodyRegion().takeBody(body);

  // Insert the function into the symbol table. Lock the symbol table, which
  // also locks the linked list of operations in the module block.
  name = sharedTable.modify(
      [lifted, it = parent->getIterator()](SymbolTable &symtab) {
        return symtab.insert(lifted, it);
      });

  b.setInsertionPoint(op);
  if (callLoc)
    b.setLoc(callLoc);
  auto create = b.create<CreateClosureOp>(
      op.getType(), SymbolConstantAttr::get(FlatSymbolRefAttr::get(name), sig),
      captures);
  op.replaceAllUsesWith(create.getResult());
  op.erase();

  if (auto scope = lifted.getSubprogramScope()) {
    DebugInfo::updateSubprogram(
        lifted, lifted.getSymNameAttr(),
        DebugInfo::SourceNameAttr::get("closure." + Twine(nameCounter - 1),
                                       scope.getName()));
  }
}

//===----------------------------------------------------------------------===//
// lowerAsyncFunction
//===----------------------------------------------------------------------===//

/// To lower an async function, we stick a `co.handle` operation in
/// it, marshall results through a `co.promise`, and return the
/// handle directly.
static LogicalResult lowerAsyncFunction(FuncOp func,
                                        Shared<SymbolTable &> &sharedTable,
                                        mlir::DominanceInfo &domInfo) {
  size_t closureNameCounter = 0;
  Value coroHdl;
  ImplicitLocOpBuilder b(func.getLoc(),
                         OpBuilder::atBlockBegin(func.getBody()));
  bool isAsyncFn = func.isAsync();
  if (isAsyncFn) {
    // Create the coroutine handle. The coroutine result types are the
    // function result types.
    auto coroType =
        CO::CoroutineType::get(func.getContext(), func.getResultTypes());
    coroHdl = b.create<CO::CoroutineHandleOp>(coroType);

    // Update the function result type.
    SignatureType origSig = func.getSignature();
    func.setSignature(
        SignatureType::get(b.getFunctionType(origSig.getArguments(), coroType),
                           origSig.getArgConventions()));
  }

  WalkResult result = func.walk([&](Operation *op) -> WalkResult {
    // Replace async calls with a simple `kgen.call`.
    if (auto call = dyn_cast<LIT::AsyncCallOp>(op)) {
      b.setLoc(call.getLoc());
      b.setInsertionPoint(call);
      // Be defensive about pass ordering.
      auto callee = dyn_cast<SymbolConstantAttr>(call.getCallee());
      if (LLVM_UNLIKELY(!callee)) {
        return op->emitOpError("callee is not a symbol constant, did you "
                               "forget to run `elaborate-generators`?");
      }

      SignatureType origSig = callee.getType();
      auto coroType =
          CO::CoroutineType::get(op->getContext(), origSig.getResults());

      auto asyncSig = SignatureType::get(
          b.getFunctionType(origSig.getArguments(), coroType),
          origSig.getArgConventions());
      auto newCall = b.create<CallOp>(
          call.getType(), SymbolConstantAttr::get(callee.getSymbol(), asyncSig),
          call.getOperands());
      call.replaceAllUsesWith(newCall);
      call.erase();

    } else if (auto exec = dyn_cast<LIT::AsyncExecuteOp>(op)) {
      lowerAsyncExecute(func, exec, sharedTable, closureNameCounter, domInfo);
    } else if (auto closure = dyn_cast<StageClosureOp>(op)) {
      lowerStageClosure(func, closure, sharedTable, closureNameCounter,
                        domInfo);
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  // If the surrounding function is an async function, go rewrite all the return
  // sites now. Do this after nested `lit.async.execute` ops are lifted to not
  // clobber their returns.
  if (isAsyncFn) {
    func.getBodyRegion().walk([&](ReturnOp ret) {
      createCoroutineFinalize(b, coroHdl, ret);
      ret->setOperands(coroHdl);
    });
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERCLOSURES
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerClosuresPass : impl::LowerClosuresBase<LowerClosuresPass> {
  using LowerClosuresBase::LowerClosuresBase;

  void runOnOperation() override {
    SymbolTable &symtab =
        getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
    Shared<SymbolTable &> sharedTable(symtab);

    auto &domInfo = getAnalysis<mlir::DominanceInfo>();

    auto eachFn = [&](FuncOp func) {
      return lowerAsyncFunction(func, sharedTable, domInfo);
    };
    std::vector<FuncOp> funcs;
    llvm::append_range(funcs, getOperation().getOps<FuncOp>());
    if (failed(mlir::failableParallelForEach(&getContext(), funcs, eachFn)))
      return signalPassFailure();
  }
};
} // namespace
