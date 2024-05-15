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
#include "Support/MDialect/MAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/BinaryFormat/Dwarf.h"

using namespace M;
using namespace KGEN;
using namespace CO;
using namespace mlir::LLVM;

//===----------------------------------------------------------------------===//
// LowerSuspensionPoints
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERSUSPENSIONPOINTS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerSuspensionPointsPass
    : public KGEN::impl::LowerSuspensionPointsBase<LowerSuspensionPointsPass> {
  using LowerSuspensionPointsBase::LowerSuspensionPointsBase;

  void runOnOperation() override;
  LogicalResult initialize(MLIRContext *ctx) override {
    coroAttrName = StringAttr::get(ctx, "coro");
    return success();
  }
  StringAttr coroAttrName;
};
} // namespace

enum ContinuationFields { State = 0, ClosureState = 1, ClosureFunction = 2 };

struct BuildContext {
  BuildContext(LLVMBuilder &builder) : builder(builder) {
    // Continuation type: !llvm.struct<(i8, ptr, ptr)>
    Type ptrType = LLVMPointerType::get(builder.getContext());
    continuationType = LLVMStructType::getLiteral(
        builder.getContext(),
        ArrayRef<Type>({builder.getI32Type(), ptrType, ptrType}));
  }
  void emitUpdateState(Value continuation) {
    GEPOp slot = builder.create<GEPOp>(
        /*resultType=*/LLVMPointerType::get(builder.getContext()),
        /*elementType=*/continuationType,
        /*basePtr=*/continuation, ArrayRef<GEPArg>({0, State}));
    Value state = builder.create<LoadOp>(builder.getI32Type(), slot);
    Value one = builder.create<ConstantOp>(
        IntegerType::get(builder.getContext(), 32), 1);
    Value newState = builder.create<AddOp>(state, one);
    builder.create<StoreOp>(newState, slot);
  }

  Value getContinuationField(Value operand, ContinuationFields fieldIndex) {
    Type type;
    switch (fieldIndex) {
    case State:
      type = builder.getI32Type();
      break;
    case ClosureFunction:
    case ClosureState:
      type = LLVMPointerType::get(builder.getContext());
      break;
    }
    GEPOp slot = builder.create<GEPOp>(
        /*resultType=*/LLVMPointerType::get(builder.getContext()),
        /*elementType=*/continuationType,
        /*basePtr=*/operand, ArrayRef<GEPArg>({0, fieldIndex}));
    return builder.create<LoadOp>(type, slot);
  }

  LLVMBuilder &builder;
  SmallVector<Block *> blockList;
  Type continuationType;
};

static void addSuspensionPoint(CoroutineAwaitOp await, Block *currentBlock,
                               BuildContext &buildContext) {
  LLVMFuncOp func = cast<LLVMFuncOp>(currentBlock->getParent()->getParentOp());
  Value continuation = func.getBody().getArgument(0);
  buildContext.builder.setInsertionPoint(await);
  buildContext.emitUpdateState(continuation);
  // Copy operations from await region. They represent code to execute after
  // update state but before return.
  Operation *current = &await.getRegion().front().front();
  while (current) {
    Operation *next = current->getNextNode();
    if (auto nestedAwait = dyn_cast<CoroutineAwaitOp>(current))
      assert(false && "cannot have an await nested inside an await");
    else if (auto awaitEnd = dyn_cast<CoroutineAwaitEndOp>(current))
      break;
    else
      current->moveBefore(await);
    current = next;
  }
  buildContext.builder.create<ReturnOp>(ValueRange({}));
  Block *newBlock = currentBlock->splitBlock(await);
  buildContext.blockList.push_back(newBlock);
}

static LogicalResult lowerSuspensionPoints(Operation *func,
                                           StringAttr coroAttrName) {
  LLVMFuncOp funcOp = cast<LLVMFuncOp>(func);
  if (!funcOp->hasAttr(coroAttrName))
    return success();
  TargetInfoAttr target = lookupTargetInfo(funcOp);
  if (!target) {
    mlir::emitError(func->getLoc(),
                    "could not find an enclosing target specification");
    return failure();
  }
  ImplicitLocOpBuilder opBuilder(func->getLoc(), func->getContext());
  LLVMBuilder b(opBuilder, target);

  // Find all suspension points. Create a new block for each suspension point.
  BuildContext buildContext(b);
  SmallVector<Block *> exitPaths;
  Block *block = &*(func->getRegion(0).begin());
  while (block) {
    Block *nextBlock = block->getNextNode();
    bool continueInResume = false;
    b.setInsertionPointToStart(block);
    Operation *current = &block->getOperations().front();
    while (current) {
      if (isa<ReturnOp>(current))
        exitPaths.push_back(continueInResume ? buildContext.blockList.back()
                                             : block);
      if (auto await = dyn_cast<CoroutineAwaitOp>(current)) {
        current = await->getNextNode();
        Block *b = continueInResume ? buildContext.blockList.back() : block;
        addSuspensionPoint(await, b, buildContext);
        await->erase();
        continueInResume = true;
        continue;
      }
      current = current->getNextNode();
    }
    block = nextBlock;
  }

  DenseSet<Block *> visited;
  bool hasSuspensionPoints = !buildContext.blockList.empty();
  if (hasSuspensionPoints) {
    // Create the initial switch to direct to the correct resume point.
    Block &initialBlock = funcOp.getBody().front();
    Block *controlBlock =
        b.createBlock(&funcOp.getRegion(), funcOp->getRegion(0).begin());
    for (Value arg : initialBlock.getArguments())
      controlBlock->addArgument(arg.getType(), arg.getLoc());
    initialBlock.getArgument(0).replaceAllUsesWith(
        controlBlock->getArgument(0));
    initialBlock.eraseArgument(0);

    b.setInsertionPoint(controlBlock, controlBlock->begin());
    SmallVector<int32_t> values;
    for (size_t i = 1; i <= buildContext.blockList.size(); i++)
      values.push_back(i);

    Value state = buildContext.getContinuationField(
        funcOp.getBody().getArgument(0), ContinuationFields::State);
    values.push_back(0);
    buildContext.blockList.push_back(&initialBlock);
    SmallVector<ValueRange> operands(buildContext.blockList.size());
    b.create<SwitchOp>(
        state,
        /*defaultDestination=*/&initialBlock,
        /*defaultOperands=*/ValueRange(),
        /*caseValues=*/
        DenseIntElementsAttr::get(
            VectorType::get({(int32_t)values.size()}, b.getI32Type()), values),
        /*caseDestinations=*/buildContext.blockList,
        /*caseOperands=*/operands);
  }

  // Invoke callback in final block before return.
  Type ptrType = LLVMPointerType::get(buildContext.builder.getContext());
  Value continuation = funcOp.getArgument(0);
  for (Block *current : exitPaths) {
    Operation *terminator = current->getTerminator();
    b.setInsertionPoint(terminator);
    Value callbackFnPtr = buildContext.getContinuationField(
        continuation, ContinuationFields::ClosureFunction);
    Value parent = buildContext.getContinuationField(
        continuation, ContinuationFields::ClosureState);
    SmallVector<Type> params;
    b.create<CallOp>(LLVMFunctionType::get(b.getContext(), ptrType, params, 0),
                     ValueRange({callbackFnPtr, parent}));
  }
  return success();
}

void LowerSuspensionPointsPass::runOnOperation() {
  if (failed(lowerSuspensionPoints(getOperation(), coroAttrName)))
    return signalPassFailure();
}
