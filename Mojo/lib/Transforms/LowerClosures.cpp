//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "Mojo/CODialect/COOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
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

/// Isolate a closure region from above by replacing uses of captured SSA values
/// in the region with block arguments. Constant-like operations are cloned into
/// the region instead of passed as a capture, since the latter has additional
/// overhead.
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

/// Lower an async execute by making it isolated from above and hoisting it into
/// a function. The conversion is done post-order, so there should be no nested
/// `co.execute` operations nested beneath this one when the function
/// gets called.
static void lowerAsyncExecute(FuncOp parent, CO::ExecuteOp op,
                              Shared<SymbolTable &> &sharedTable,
                              size_t &nameCounter) {
  // Gather location info from encoded CallLoc, and set the op's location to the
  // unencoded location so that inlined body ops get the right callsite loc.
  LocationAttr callLoc = op.getCallLocAttr();
  Location unencodedLoc = op.getLocNoInlined();
  op.getOperation()->setLoc(unencodedLoc);

  // Before we do anything with the captures, insert the coroutine handle and
  // replace the byref arguments.
  Region &body = op.getBodyRegion();
  unsigned numByRefResults = body.getNumArguments();

  SmallVector<Value> captures;
  liftClosureRegion(body, captures);

  // We know the byref arguments got pushed to the beginning. Move them back to
  // the end.
  MutableArrayRef<BlockArgument> args = body.front().getArguments();
  std::rotate(args.begin(), args.begin() + numByRefResults, args.end());

  // Move the body into a function. The function is not valid to inline.
  ImplicitLocOpBuilder b{op.getLoc(), OpBuilder(op.getContext())};
  StringAttr name = b.getStringAttr(parent.getSymName() + "_async_closure_" +
                                    Twine(nameCounter++));
  // TODO: What conventions do we use for captures.
  SmallVector<ArgConvention> conventions(captures.size(),
                                         ArgConvention::ImmReg);
  // Add the appropriate byref conventions for the result slots.
  if (numByRefResults == 2)
    conventions.push_back(ArgConvention::ByRefError);
  if (numByRefResults)
    conventions.push_back(ArgConvention::ByRefResult);

  auto sig = FuncType::get(
      b.getFunctionType(body.getArgumentTypes(), op.getTypes()), conventions,
      FnEffects().setAsync().setThrows(numByRefResults == 2));
  auto lifted = FuncOp::create(b, name, sig);
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
  Value call = CO::InvokeOp::create(
      b, op.getType(),
      SymbolConstantAttr::get(name, GeneratorType::get({}, sig)), captures);
  op.replaceAllUsesWith(call);
  op.erase();

  if (auto scope = lifted.getSubprogramScope()) {
    DebugInfo::updateSubprogram(
        lifted, lifted.getSymNameAttr(),
        DebugInfo::SourceNameAttr::get(
            "async_closure." + Twine(nameCounter - 1), scope.getSourceName()));
  }
}

//===----------------------------------------------------------------------===//
// lowerAwait
//===----------------------------------------------------------------------===//

/// Codegen `co.await` into its `co` dialect constituents:
static void lowerAwait(CO::AwaitOp op) {
  MLIRContext *ctx = op.getContext();
  ImplicitLocOpBuilder b(op.getLoc(), OpBuilder(op));
  if (op.getNumOperands() > 1)
    CO::SetByRefErrorAndResultOp::create(b, TypeRange(), op->getOperands());
  auto suspend = CO::SuspendOp::create(b);
  Block *body = b.createBlock(&suspend.getBody());
  Value parent = body->addArgument(op.getCoroutine().getType(), op.getLoc());

  auto coroutineType = CO::CoroutineType::get(ctx);
  auto signatureType = FuncTypeGeneratorType::get(
      /*inputParamTypes=*/{}, b.getFunctionType({coroutineType}, {}));
  auto callbackType =
      PointerType::get(StructType::get({signatureType, coroutineType}));
  Value callback =
      CO::GetCallbackPtrOp::create(b, callbackType, op.getCoroutine());
  Value resumeFnPtr = StructGEPOp::create(b, callback, 0);
  Value parentPtr = StructGEPOp::create(b, callback, 1);
  Value resumeFn = CO::ResumeOp::create(b, signatureType, parent);
  POP::StoreOp::create(b, resumeFn, resumeFnPtr);
  POP::StoreOp::create(b, parent, parentPtr);
  Value curResume = CO::ResumeOp::create(b, signatureType, op.getCoroutine());
  CallIndirectOp::create(b, TypeRange(), curResume, op.getCoroutine());
  CO::SuspendEndOp::create(b);

  b.setInsertionPointAfter(suspend);
  if (op.getNumResults()) {
    auto results =
        CO::GetResultsOp::create(b, op.getResultTypes(), op.getCoroutine());
    op.replaceAllUsesWith(results);
  }
  CO::DestroyOp::create(b, op.getCoroutine());
  op.erase();
}

//===----------------------------------------------------------------------===//
// lowerClosures
//===----------------------------------------------------------------------===//

/// To lower an async function, we stick a `co.handle` operation in
/// it, marshall results through a `co.promise`, and return the
/// handle directly.
static LogicalResult lowerClosures(FuncOp func,
                                   Shared<SymbolTable &> &sharedTable) {
  size_t closureNameCounter = 0;
  WalkResult result = func.walk([&](Operation *op) -> WalkResult {
    if (auto exec = dyn_cast<CO::ExecuteOp>(op)) {
      lowerAsyncExecute(func, exec, sharedTable, closureNameCounter);
    } else if (auto await = dyn_cast<CO::AwaitOp>(op)) {
      lowerAwait(await);
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
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerClosuresPass : impl::LowerClosuresBase<LowerClosuresPass> {
  using LowerClosuresBase::LowerClosuresBase;

  void runOnOperation() override {
    SymbolTable &symtab =
        getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
    Shared<SymbolTable &> sharedTable(symtab);

    auto eachFn = [&](FuncOp func) { return lowerClosures(func, sharedTable); };
    std::vector<FuncOp> funcs;
    llvm::append_range(funcs, getOperation().getOps<FuncOp>());
    if (failed(mlir::failableParallelForEach(&getContext(), funcs, eachFn)))
      return signalPassFailure();
  }
};
} // namespace
