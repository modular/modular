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
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/RWMutex.h"

using namespace M;
using namespace KGEN;
using namespace POP;

//===----------------------------------------------------------------------===//
// lowerAsyncFunction
//===----------------------------------------------------------------------===//

namespace {
/// A shared symbol table.
struct LockedSymbolTable {
  SymbolTable &symtab;
  llvm::sys::SmartRWMutex<true> mutex;
};
} // namespace

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
                              LockedSymbolTable &sharedTable) {
  // Isolate the region from above.
  llvm::SetVector<Value> allCaptures;
  SmallVector<Value> captures;
  mlir::getUsedValuesDefinedAbove(op->getRegions(), allCaptures);
  Region &body = op.getBodyRegion();
  for (Value capture : allCaptures) {
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
      mlir::replaceAllUsesInRegionWith(capture, arg, op.getBodyRegion());
      captures.push_back(capture);
    }
  }

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
  auto lifted = b.create<FuncOp>(name, sig, AlwaysInlineLevel::Disabled);
  lifted.getBodyRegion().takeBody(body);

  // Insert the function into the symbol table. Lock the symbol table, which
  // also locks the linked list of operations in the module block.
  {
    llvm::sys::SmartScopedWriter<true> lock(sharedTable.mutex);
    name = sharedTable.symtab.insert(lifted, parent->getIterator());
  }

  // Create the call with a dummy callee.
  b.setInsertionPoint(op);
  auto call = b.create<CallOp>(
      op.getType(), SymbolConstantAttr::get(FlatSymbolRefAttr::get(name), sig),
      ArrayRef<ParamDeclAttr>(), captures);
  op.replaceAllUsesWith(call);
  op.erase();
}

/// To lower an async function, we stick a `pop.coroutine.handle` operation in
/// it, marshall results through a `pop.coroutine.promise`, and return the
/// handle directly.
static LogicalResult lowerAsyncFunction(FuncOp func,
                                        LockedSymbolTable &sharedTable) {
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
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERASYNCFUNCTIONS
#include "KGEN/KGENPasses.h.inc"
#define GEN_PASS_DEF_RUNTIMECLOSURES
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerAsyncFunctionsPass
    : impl::LowerAsyncFunctionsBase<LowerAsyncFunctionsPass> {
  using LowerAsyncFunctionsBase::LowerAsyncFunctionsBase;

  void runOnOperation() override {
    SymbolTable &symtab =
        getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
    LockedSymbolTable sharedTable{symtab, {}};

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

namespace {
struct RuntimeClosuresPass : impl::RuntimeClosuresBase<RuntimeClosuresPass> {
  void runOnOperation() override;
};
} // namespace

static SymbolConstantAttr callSymbolOfLiftedRegion(StageClosureOp opWithRegion,
                                                   SmallVector<Value> &captures,
                                                   SymbolTable &symtab) {
  assert(opWithRegion->getNumRegions() == 1);
  OpBuilder builder(opWithRegion->getParentOfType<ModuleOp>());

  llvm::SetVector<Value> captureSet;
  Region &sourceRegion = opWithRegion->getRegion(0);
  operationIsIsolatedFromAbove(opWithRegion, &captureSet);
  for (Value capture : captureSet) {
    Operation *capturingOp = capture.getDefiningOp();
    // Clone ConstantLike operations into the region.
    if (capturingOp && capturingOp->hasTrait<OpTrait::ConstantLike>()) {
      ImplicitLocOpBuilder b(capturingOp->getLoc(),
                             OpBuilder::atBlockBegin(&sourceRegion.front()));
      Operation *cloned = b.clone(*capturingOp);
      for (auto [orig, replacement] :
           llvm::zip(capturingOp->getResults(), cloned->getResults()))
        replaceAllUsesInRegionWith(orig, replacement, sourceRegion);
    } else {
      // Otherwise these are captured variables and we need to pass them as
      // arguments to the block body.
      captures.emplace_back(capture);
    }
  }

  SignatureType signatureType = opWithRegion.getType();
  // Lift the body by making source region isolated from above
  // add captures in reverse so they appear the same order in
  // the parameter list as they do in the capture order
  for (int i = captures.size() - 1; i >= 0; --i) {
    Value from = captures[i];
    BlockArgument newArg =
        sourceRegion.insertArgument((unsigned)0, from.getType(), from.getLoc());
    replaceAllUsesInRegionWith(from, newArg, sourceRegion);
  }
  auto liftedValueSignature =
      FunctionType::get(builder.getContext(), sourceRegion.getArgumentTypes(),
                        signatureType.getValueResults());

  auto noTypes = TypeArrayAttr::get(signatureType.getContext(), {});
  auto liftedSignature = SignatureType::get(
      noTypes, noTypes, liftedValueSignature,
      MetadataAttr::get(signatureType.getContext(),
                        sourceRegion.getNumArguments(), FnEffects::Capturing));
  builder.setInsertionPoint(opWithRegion->getParentOfType<FuncOp>());

  std::string name = "stage_closure";
  if (opWithRegion->hasAttr("name")) {
    auto nameMaybe =
        dyn_cast_or_null<StringAttr>(opWithRegion->getAttr("name"));
    if (nameMaybe)
      name = nameMaybe.str();
  }

  auto lifted = builder.create<FuncOp>(
      opWithRegion->getLoc(), StringAttr::get(builder.getContext(), name),
      liftedSignature, AlwaysInlineLevel::Disabled);
  symtab.insert(lifted);
  auto liftedSymbol = SymbolConstantAttr::get(
      SymbolRefAttr::get(lifted.getSymNameAttr()), liftedSignature);
  IRMapping mapper;
  sourceRegion.cloneInto(&lifted.getBodyRegion(), mapper);
  return liftedSymbol;
}

void RuntimeClosuresPass::runOnOperation() {
  ModuleOp theModule = getOperation();
  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

  // Lift nested functions
  mlir::IRRewriter rewriter{OpBuilder(theModule)};
  theModule->walk([&](StageClosureOp stageClosure) {
    SmallVector<Value> captures;
    SymbolConstantAttr symbol =
        callSymbolOfLiftedRegion(stageClosure, captures, symtab);
    rewriter.setInsertionPoint(stageClosure);
    rewriter.replaceOpWithNewOp<CreateClosureOp>(
        stageClosure, stageClosure.getType(), symbol, captures);
  });
}
