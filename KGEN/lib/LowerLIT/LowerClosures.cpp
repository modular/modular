//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/Threading/Shared.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
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

/// Isolate a closure region from above by replacing uses of capture SSA values
/// in the region with block arguments. Certain zero-cost operations, like
/// constants, should be cloned into the region instead of passed as a capture,
/// since the latter has additional overhead.
///
/// The captured values, excluding the cloned values, are populate into
/// `captures`.
static void liftClosureRegion(Region &body, SmallVectorImpl<Value> &captures) {
  // Isolate the region from above.
  llvm::SetVector<Value> captureSet;
  mlir::getUsedValuesDefinedAbove(body, captureSet);
  for (Value capture : captureSet) {
    Operation *capturingOp = capture.getDefiningOp();
    // Clone ConstantLike operations into the region.
    if (capturingOp && capturingOp->hasTrait<OpTrait::ConstantLike>()) {
      ImplicitLocOpBuilder b(capturingOp->getLoc(),
                             OpBuilder::atBlockBegin(&body.front()));
      Operation *cloned = b.clone(*capturingOp);
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
  Value promise = b.create<CoroutinePromiseOp>(hdl);
  for (auto [idx, result] : llvm::enumerate(ret->getOperands()))
    b.create<StoreOp>(result, b.create<POP::StructGEPOp>(promise, idx));
}

/// Lower an async execute by making it isolated from above and hoisting it into
/// a function. The conversion is done post-order, so there should be no nested
/// `lit.async.execute` operations nested beneath this one when the function
/// gets called.
static void lowerAsyncExecute(FuncOp parent, LIT::AsyncExecuteOp op,
                              Shared<SymbolTable &> &sharedTable) {
  SmallVector<Value> captures;
  Region &body = op.getBodyRegion();
  liftClosureRegion(body, captures);

  // Insert the coroutine handle.
  ImplicitLocOpBuilder b(op.getLoc(), OpBuilder::atBlockBegin(&body.front()));
  Value coroHdl = b.create<CoroutineHandleOp>(op.getType());

  // Replace all returns.
  op.walk([&](LIT::AsyncReturnOp ret) {
    createCoroutineFinalize(b, coroHdl, ret);
    b.create<ReturnOp>(coroHdl);
    ret.erase();
  });

  // Move the body into a function.
  b.clearInsertionPoint();
  b.setLoc(op.getLoc());
  StringAttr name = b.getStringAttr(parent.getSymName() + "_async_closure");
  auto sig = SignatureType::get(
      b.getFunctionType(body.getArgumentTypes(), op.getType()));
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
  auto call = b.create<CallOp>(
      op.getType(), SymbolConstantAttr::get(FlatSymbolRefAttr::get(name), sig),
      ArrayRef<ParamDeclAttr>(), captures);
  op.replaceAllUsesWith(call);
  op.erase();

  DebugInfo::updateSubprogram(lifted, name, name);
}

//===----------------------------------------------------------------------===//
// lowerStageClosure
//===----------------------------------------------------------------------===//

/// Lower a closure by closing the region over its captured SSA values and
/// lifting it into a top-level function.
static void lowerStageClosure(FuncOp parent, StageClosureOp op,
                              Shared<SymbolTable &> &sharedTable) {
  Region &body = op.getBodyRegion();
  unsigned numArgs = body.getNumArguments();
  SmallVector<Value> captures;
  liftClosureRegion(body, captures);
  // Add the capture arguments to the front so they can be partially applied by
  // `kgen.create_closure`.
  std::rotate(body.getArguments().begin(),
              body.getArguments().begin() + numArgs, body.getArguments().end());

  // Construct the signature of the lifted body.
  OpBuilder b(op.getContext());
  auto functionType = b.getFunctionType(body.getArgumentTypes(),
                                        op.getType().getValueResults());
  auto none = TypeArrayAttr::get(op.getContext(), {});
  auto metadata = MetadataAttr::get(op.getContext(), body.getNumArguments(),
                                    FnEffects::Capturing);
  auto sig = SignatureType::get(none, none, functionType, metadata);

  // Create the lifted function.
  StringAttr name;
  if (auto nameMaybe = op->getAttrOfType<StringAttr>("name"))
    name = nameMaybe;
  else
    name = b.getStringAttr(parent.getSymName() + "_closure");
  auto lifted = b.create<FuncOp>(op->getLoc(), name, sig);
  lifted.getBodyRegion().takeBody(body);

  // Insert the function into the symbol table. Lock the symbol table, which
  // also locks the linked list of operations in the module block.
  name = sharedTable.modify(
      [lifted, it = parent->getIterator()](SymbolTable &symtab) {
        return symtab.insert(lifted, it);
      });

  b.setInsertionPoint(op);
  auto create = b.create<CreateClosureOp>(
      op.getLoc(), op.getType(),
      SymbolConstantAttr::get(FlatSymbolRefAttr::get(name), sig), captures);
  op.replaceAllUsesWith(create.getResult());
  op.erase();

  DebugInfo::updateSubprogram(lifted, name, name);
}

//===----------------------------------------------------------------------===//
// lowerAsyncFunction
//===----------------------------------------------------------------------===//

/// To lower an async function, we stick a `pop.coroutine.handle` operation in
/// it, marshall results through a `pop.coroutine.promise`, and return the
/// handle directly.
static LogicalResult lowerAsyncFunction(FuncOp func,
                                        Shared<SymbolTable &> &sharedTable) {
  Value coroHdl;
  ImplicitLocOpBuilder b(func.getLoc(),
                         OpBuilder::atBlockBegin(func.getBody()));
  bool isAsyncFn = func.isAsync();
  if (isAsyncFn) {
    // Create the coroutine handle. The coroutine result types are the
    // function result types.
    auto coroType = CoroutineType::get(
        SignatureType::get(b.getFunctionType({}, func.getResultTypes())));
    coroHdl = b.create<CoroutineHandleOp>(coroType);

    // Update the function result type.
    SignatureType origSig = func.getSignature();
    func.setSignature(SignatureType::get(
        b.getFunctionType(origSig.getValueInputs(), coroType)));
  }

  WalkResult result = func.walk([&](Operation *op) -> WalkResult {
    // If this is an async function, update the return sites.
    if (isAsyncFn) {
      if (auto ret = dyn_cast<ReturnOp>(op)) {
        createCoroutineFinalize(b, coroHdl, ret);
        ret->setOperands(coroHdl);
        return WalkResult::advance();
      }
    }
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
      auto asyncSig = SignatureType::get(b.getFunctionType(
          origSig.getValueInputs(),
          CoroutineType::get(SignatureType::get(
              b.getFunctionType({}, origSig.getValueResults())))));
      auto newCall = b.create<CallOp>(
          call.getType(), SymbolConstantAttr::get(callee.getSymbol(), asyncSig),
          ArrayRef<ParamDeclAttr>(), call.getOperands());
      call.replaceAllUsesWith(newCall);
      call.erase();

    } else if (auto exec = dyn_cast<LIT::AsyncExecuteOp>(op)) {
      lowerAsyncExecute(func, exec, sharedTable);
    } else if (auto closure = dyn_cast<StageClosureOp>(op)) {
      lowerStageClosure(func, closure, sharedTable);
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
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

    auto eachFn = [&](FuncOp func) {
      return lowerAsyncFunction(func, sharedTable);
    };
    std::vector<FuncOp> funcs;
    llvm::append_range(funcs, getOperation().getOps<FuncOp>());
    if (failed(mlir::failableParallelForEach(&getContext(), funcs, eachFn)))
      return signalPassFailure();
  }
};
} // namespace
