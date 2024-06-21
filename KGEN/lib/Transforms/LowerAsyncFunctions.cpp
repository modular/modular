//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CODialect/CODialect.h"
#include "KGEN/CODialect/COOps.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/HLCFDialect/HLCFUtils.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/TransformUtils/AsyncUtils.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;
using namespace POP;
using namespace CO;

//===----------------------------------------------------------------------===//
// Lower Async Functions
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERASYNCFUNCTIONS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerAsyncFunctionsPass
    : impl::LowerAsyncFunctionsBase<LowerAsyncFunctionsPass> {
public:
  using LowerAsyncFunctionsBase::LowerAsyncFunctionsBase;
  void runOnOperation() override;
};
} // namespace

/// Frame Data stores any metadata necessary to transform the async function
/// into a suspendable procedure. This includes indexing information into the
/// frame type so that we can generate loads and stores and state information so
/// we can reuse loaded frame variables when legal.
struct FrameData {
  /// Error value and result value are excluded from the frame
  FrameData(FuncOp originalFunction, mlir::DominanceInfo &domInfo,
            Value errorValue, Value resultValue,
            function_ref<void(DenseMap<Operation *, int> &, FuncOp)> transform);
  FrameData() {}

  /// Given a value, determine the state of its defining op or block argument.
  int getDefinitionStateForValue(Value operand) const;

  SmallVector<Type> frameTypes;
  DenseMap<Value, unsigned> valueToIndexInFrame;
  DenseMap<Operation *, unsigned> operationToIndexInFrame;
  DenseMap<Operation *, int> opToState;
};

struct COTypes {
  Type typeForField(AsyncContinuationField field) {
    switch (field) {
    case State:
      return IntegerType::get(cxt, 32);
    case ResumeFunction:
      return resumeSignatureType;
    case CallbackFn:
      return callbackSignature;
    case Promise:
      return promiseType;
    case ClosureState:
    case ErrorSlot:
    case ResultSlot:
      return opaquePointerType;
    case Frame:
      return StructType::get(cxt, frameData.frameTypes);
    }
    llvm_unreachable("invalid AsyncContinuationField value");
  }
  COTypes(MLIRContext *cxt, FrameData &frameData, StructType promiseType);
  COTypes(const COTypes &) = delete;
  COTypes &operator=(const COTypes &) = delete;

public:
  Type getContinuationType() const { return continuationType; }
  FrameData &getFrameData() const { return frameData; }
  StructType getHeaderType() const { return headerType; }

private:
  Type continuationType;
  Type resumeSignatureType;
  Type opaquePointerType;
  Type callbackSignature;
  StructType headerType;
  MLIRContext *cxt;
  FrameData &frameData;
  Type promiseType;
};

struct LoweredAsyncFunction {
  LoweredAsyncFunction(FuncOp ramp, FuncOp resume)
      : ramp(ramp), resume(resume) {}
  FuncOp ramp;
  FuncOp resume;
};

using VirtualBlock = Operation *;

/// Frame Variables is a Cache of extracted frame variables. We may for example
/// reference a frame variable multiple times within a virtual block. We should
/// only extract that variable once for that state.
class FrameVariables {
public:
  FrameVariables(ImplicitLocOpBuilder &builder, FrameData &frameData,
                 Value errorValue, Value resultValue)
      : builder(builder), frameData(frameData), errorValue(errorValue),
        resultValue(resultValue) {}

  /// Given the original operand, return the value extracted from the frame. Use
  /// a previously extracted value if available.
  Value getFrameValueForOperand(Value continuation, Value operand,
                                Operation *opWithUse, int useState);

  /// Overwrite the cached value for the variable in this state. A state can
  /// contain nested control flow, resulting in frame variables extracted in
  /// nested blocks. Sometime we could optimize this so that frame variables
  /// used multiple times in the same state are extracted in the first shared
  /// parent block, thus removing the need to overwrite state.
  void overwriteValue(int state, Value value);

private:
  ImplicitLocOpBuilder &builder;
  FrameData &frameData;
  DenseMap<Value, DenseMap<int, Value>> frameVariables;
  Value errorValue;
  Value resultValue;
};

/// The LowerAsyncBuildContext is responsible for transforming an async function
/// into a ramp function and resume function.
struct LowerAsyncBuildContext {
  LowerAsyncBuildContext(
      Shared<SymbolTable &> &sharedTable,
      DenseMap<SymbolConstantAttr, std::pair<SymbolConstantAttr, Type>>
          &asyncFuncToRampFunctions,
      ImplicitLocOpBuilder &builder, TargetInfoAttr targetInfoAttr)
      : sharedTable(sharedTable),
        asyncFuncToRampFunctions(asyncFuncToRampFunctions), builder(builder),
        targetInfoAttr(targetInfoAttr) {}

  /// Given an async function and its frame types, create a ramp function and a
  /// resume function.
  LoweredAsyncFunction lowerAsyncFunction(FuncOp funcOp,
                                          mlir::DominanceInfo &domInfo,
                                          COTypes &coTypes, Value errorValue,
                                          Value memoryResultValue);

private:
  /// Given the resume function, the empty ramp function, the original async
  /// function, and coroutine types, populate the ramp function.
  void populateRampFunction(FuncOp rampFunction, FuncOp resumeFunction,
                            FuncOp funcOp, COTypes &coTypes);
  /// Given the empty resume function, the original async function, and the
  /// coroutine types, populate the resume function.
  void populateResumeFunction(FuncOp resumeFunction, FuncOp funcOp,
                              COTypes &coTypes, Value errorValue,
                              Value memoryResultValue);
  Shared<SymbolTable &> &sharedTable;
  DenseMap<SymbolConstantAttr, std::pair<SymbolConstantAttr, Type>>
      &asyncFuncToRampFunctions;
  ImplicitLocOpBuilder &builder;
  TargetInfoAttr targetInfoAttr;
};

//===----------------------------------------------------------------------===//
// LowerAsyncBuildContext
//===----------------------------------------------------------------------===//

LoweredAsyncFunction LowerAsyncBuildContext::lowerAsyncFunction(
    FuncOp funcOp, mlir::DominanceInfo &domInfo, COTypes &coTypes,
    Value errorValue, Value memoryResultValue) {
  MLIRContext *cxt = funcOp.getContext();
  StructType headerType(coTypes.getHeaderType());

  // Create Ramp Function.
  StringAttr rampName = builder.getStringAttr(funcOp.getSymName() + "_ramp");
  unsigned end = funcOp.getNumArguments();
  if (funcOp.isThrows())
    --end;
  if (funcOp.getSignature().hasMemoryOnlyResult())
    --end;
  SmallVector<Type> args;
  for (unsigned i = 0; i < end; ++i)
    args.push_back(funcOp.getArgument(i).getType());
  FunctionType rampFunctionType =
      builder.getFunctionType(args, PointerType::get(headerType));
  auto rampSignature = SignatureType::get(rampFunctionType);
  builder.setInsertionPoint(funcOp);
  FuncOp rampFunction = builder.create<FuncOp>(rampName, rampSignature);
  rampName = sharedTable.modify(
      [rampFunction, it = funcOp->getIterator()](SymbolTable &symtab) {
        return symtab.insert(rampFunction, it);
      });

  // Create resume function.
  StringAttr resumeName =
      builder.getStringAttr(funcOp.getSymName() + "_resume");
  Type opaquePointerType = PointerType::get(KGEN::NoneType::get(cxt));
  SmallVector<Type> inputs;
  SmallVector<Type> results;
  inputs.push_back(opaquePointerType);
  FunctionType resumeFunctionType = FunctionType::get(cxt, inputs, results);
  auto resumeSignature = SignatureType::get(resumeFunctionType);
  FuncOp resumeFunction = builder.create<FuncOp>(
      funcOp->getParentOp()->getLoc(), resumeName, resumeSignature);
  resumeFunction.setCoroutineTypeAttr(
      mlir::TypeAttr::get(coTypes.getContinuationType()));
  resumeName = sharedTable.modify(
      [resumeFunction, it = rampFunction->getIterator()](SymbolTable &symtab) {
        return symtab.insert(resumeFunction, it);
      });

  populateRampFunction(rampFunction, resumeFunction, funcOp, coTypes);
  populateResumeFunction(resumeFunction, funcOp, coTypes, errorValue,
                         memoryResultValue);

  // Update map.
  SymbolConstantAttr key = SymbolConstantAttr::get(
      SymbolRefAttr::get(funcOp.getContext(), funcOp.getSymName()),
      funcOp.getSignature());
  SymbolConstantAttr value = SymbolConstantAttr::get(
      SymbolRefAttr::get(cxt, rampFunction.getSymName()),
      rampFunction.getSignature());
  asyncFuncToRampFunctions[key] = {value, coTypes.getContinuationType()};
  funcOp.erase();
  return {rampFunction, resumeFunction};
}

void LowerAsyncBuildContext::populateResumeFunction(FuncOp resumeFunction,
                                                    FuncOp funcOp,
                                                    COTypes &coTypes,
                                                    Value errorValue,
                                                    Value memoryResultValue) {
  Type continuationType = coTypes.getContinuationType();
  FrameData &frameData = coTypes.getFrameData();
  // Take the body of the original function.
  resumeFunction.getBodyRegion().takeBody(funcOp.getBodyRegion());
  resumeFunction.getBodyRegion().insertArgument(
      (unsigned)0, resumeFunction.getSignature().getArguments()[0],
      resumeFunction->getLoc());
  builder.setInsertionPointToStart(&resumeFunction.getBodyRegion().front());
  // Extract arguments from continuation's frame.
  Value continuation = builder.create<PointerBitcastOp>(
      PointerType::get(continuationType), resumeFunction.getArgument(0));

  // For each new operand, extract operand from frame if it was defined in
  // previous state. For each op, store in frame if it is used downstream
  // across a suspension point. For each block, store arguments if they are
  // accessed across suspnsion points.
  FrameVariables frameVariables(builder, frameData, errorValue,
                                memoryResultValue);
  SmallVector<std::pair<Region *, Block::iterator>> regionsToProcess;
  regionsToProcess.push_back({&resumeFunction.getBodyRegion(),
                              resumeFunction.getBodyRegion().front().begin()});
  SmallVector<Operation *> opsToDelete;
  while (!regionsToProcess.empty()) {
    auto [parentRegion, begin] = regionsToProcess.back();
    regionsToProcess.pop_back();
    // Process the ops of a region.
    Operation *current = &*begin;
    while (current) {
      Operation *op = current;
      current = op->getNextNode();
      if (isa<StackAllocLifetimeEndOp, StackAllocLifetimeStartOp>(op)) {
        int index = 0;
        for (Value value : op->getOperands()) {
          auto entry =
              frameData.operationToIndexInFrame.find(value.getDefiningOp());
          if (entry != frameData.operationToIndexInFrame.end())
            op->eraseOperand(index);
          else
            index++;
        }
        if (op->getNumOperands() == 0)
          op->erase();
        continue;
      }

      // Store op in frame if needed.
      auto entry = frameData.operationToIndexInFrame.find(op);
      if (entry != frameData.operationToIndexInFrame.end()) {
        if (isa<StackAllocationOp>(op)) {
          opsToDelete.push_back(op);
          continue;
        }
        builder.setInsertionPointAfter(op);
        Type frameEntryType = frameData.frameTypes[entry->getSecond()];
        assert(frameEntryType == op->getResultTypes().front() &&
               "The frame type slot does not match the value");
        assert(op->getNumResults() == 1 && "TODO: support multiple results");
        Value dataSlot = builder.create<StructGEPOp>(
            continuation, Frame + entry->getSecond());
        builder.create<StoreOp>(op->getResult(0), dataSlot);
      }

      // Extract arguments from operands in needed.
      int useState = frameData.opToState[op];
      for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
        auto entry = frameData.valueToIndexInFrame.find(operand);
        if (entry != frameData.valueToIndexInFrame.end() ||
            operand == errorValue || operand == memoryResultValue) {
          // Only extract the value out of the frame if the def was in another
          // state. Block arguments have been cached in frameVariables because
          // region block arguments are processed before body ops.
          int defState = -1;
          Operation *definingOp = operand.getDefiningOp();
          if (definingOp)
            defState = frameData.opToState[definingOp];
          // Stack allocated variables are an exception. They are pulled from
          // the frame regardless of state status because the stack allocation
          // is replaced with a frame allocation.
          if (defState == useState &&
              (!(definingOp && isa<StackAllocationOp>(definingOp))))
            continue;
          Value image = frameVariables.getFrameValueForOperand(
              continuation, operand, op, useState);
          op->setOperand(index, image);
        }
      }

      // Store arguments of block if needed.
      for (Region &region : op->getRegions()) {
        // Start processing at the first op. Blocks cannot be empty because
        // they must be terminated.
        Operation *firstOp = &*region.front().begin();
        regionsToProcess.push_back({&region, firstOp->getIterator()});
        if (region.front().getNumArguments() == 0)
          continue;
        builder.setInsertionPointToStart(&region.front());
        int frameValueState = frameData.opToState[firstOp];
        for (BlockArgument argument : region.front().getArguments()) {
          auto entry = frameData.valueToIndexInFrame.find(argument);
          if (entry == frameData.valueToIndexInFrame.end())
            continue;
          Value dataSlot = builder.create<StructGEPOp>(
              continuation, Frame + entry->getSecond());
          builder.create<StoreOp>(argument, dataSlot);
          frameVariables.overwriteValue(frameValueState, argument);
        }
      }
    }
  }

  llvm::BitVector args(resumeFunction.getBodyRegion().front().getNumArguments(),
                       true);
  args.reset(0);
  resumeFunction.getBodyRegion().front().eraseArguments(args);

  resumeFunction.walk([&](Operation *op) {
    if (auto returnOp = dyn_cast<ReturnOp>(op);
        returnOp && returnOp.getNumOperands()) {
      // Replace ReturnOps with set result.
      builder.setInsertionPoint(returnOp);
      Value promiseSlot = builder.create<StructGEPOp>(continuation, Promise);
      for (auto [idx, value] : llvm::enumerate(returnOp.getOperands())) {
        builder.create<StoreOp>(value,
                                builder.create<StructGEPOp>(promiseSlot, idx));
      }
      builder.create<ReturnOp>();
      returnOp->erase();
    } else if (auto suspend = dyn_cast<SuspendOp>(op)) {
      // Replace uses of the suspend argument with the continuation.
      Region &body = suspend.getBody();
      if (!body.getArgument(0).use_empty()) {
        builder.setInsertionPoint(suspend);
        Value header = builder.create<PointerBitcastOp>(
            PointerType::get(coTypes.getHeaderType()),
            resumeFunction.getArgument(0));
        body.getArgument(0).replaceAllUsesWith(header);
      }
      body.eraseArgument(0);
    }
  });
  for (auto op : opsToDelete)
    op->erase();
}
void LowerAsyncBuildContext::populateRampFunction(FuncOp rampFunction,
                                                  FuncOp resumeFunction,
                                                  FuncOp funcOp,
                                                  COTypes &coTypes) {
  Type continuationType = coTypes.getContinuationType();
  FrameData &frameData = coTypes.getFrameData();
  builder.setInsertionPointToStart(
      &rampFunction.getBodyRegion().emplaceBlock());
  for (Type argument : rampFunction.getSignature().getArguments())
    rampFunction.getBodyRegion().addArgument(argument, rampFunction.getLoc());
  // Allocate memory for continuation.
  std::optional<int64_t> size =
      DataLayoutInterface::getTypeStoreSize(targetInfoAttr, continuationType);
  std::optional<int64_t> align =
      DataLayoutInterface::getTypeABIAlign(targetInfoAttr, continuationType);
  Value sizeOf = builder.create<mlir::index::ConstantOp>(size.value());
  Value alignOf = builder.create<mlir::index::ConstantOp>(align.value());

  Value continuation = builder.create<AlignedAllocOp>(
      PointerType::get(continuationType), ValueRange{alignOf, sizeOf});

  // Initialize state to 0.
  Value zero = builder.create<ParamConstantOp>(builder.getI32IntegerAttr(0));
  Value stateSlot = builder.create<StructGEPOp>(continuation, State);
  builder.create<StoreOp>(zero, stateSlot);

  // Store resume function.
  Value resumeFunctionSlot =
      builder.create<StructGEPOp>(continuation, ResumeFunction);
  Value functionPointer =
      builder.create<CreateClosureOp>(SymbolConstantAttr::get(
          SymbolRefAttr::get(builder.getContext(), resumeFunction.getSymName()),
          resumeFunction.getSignature()));
  builder.create<StoreOp>(functionPointer, resumeFunctionSlot);

  // Store arguments in frame.
  for (auto [index, image] : llvm::enumerate(rampFunction.getArguments())) {
    Value arg = funcOp.getArgument(index);
    // If the argument is not in the frame metadata it is unused. Do not
    // store in frame.
    if (!frameData.valueToIndexInFrame.contains(arg))
      continue;
    unsigned argSlot = frameData.valueToIndexInFrame[arg];
    Value slot = builder.create<StructGEPOp>(continuation, Frame + argSlot);
    builder.create<StoreOp>(image, slot);
  }
  Value headerTypedContinuation = builder.create<PointerBitcastOp>(
      PointerType::get(coTypes.getHeaderType()), continuation);
  builder.create<ReturnOp>(headerTypedContinuation);
}

//===----------------------------------------------------------------------===//
// CoTypes
//===----------------------------------------------------------------------===//

COTypes::COTypes(MLIRContext *cxt, FrameData &frameData, StructType promiseType)
    : cxt(cxt), frameData(frameData), promiseType(promiseType) {
  opaquePointerType = PointerType::get(KGEN::NoneType::get(cxt));
  SmallVector<Type> inputs;
  SmallVector<Type> results;
  inputs.push_back(opaquePointerType);
  FunctionType resumeFunctionType = FunctionType::get(cxt, inputs, results);
  resumeSignatureType = SignatureType::get(resumeFunctionType);
  FunctionType callbackFunctionType =
      FunctionType::get(cxt, opaquePointerType, KGEN::NoneType::get(cxt));
  callbackSignature = SignatureType::get(callbackFunctionType);

  // Build Continuation Type.
  size_t size = Promise;
  SmallVector<Type> types(size);
  types[State] = typeForField(State);
  types[ResumeFunction] = typeForField(ResumeFunction);
  types[CallbackFn] = typeForField(CallbackFn);
  types[ClosureState] = typeForField(ClosureState);
  types[ErrorSlot] = typeForField(ErrorSlot);
  types[ResultSlot] = typeForField(ResultSlot);

  // Header type omits the variable sized frame.
  headerType = StructType::get(types);
  types.push_back(typeForField(Promise));
  for (auto [index, frameVariableType] : llvm::enumerate(frameData.frameTypes))
    types.push_back(frameVariableType);
  continuationType = StructType::get(types);
}

//===----------------------------------------------------------------------===//
// FrameVariables
//===----------------------------------------------------------------------===//

Value FrameVariables::getFrameValueForOperand(Value continuation, Value operand,
                                              Operation *opWithUse,
                                              int useState) {
  auto entry = frameData.valueToIndexInFrame.find(operand);
  if (entry == frameData.valueToIndexInFrame.end() && operand != errorValue &&
      operand != resultValue)
    return {};
  DenseMap<int, Value> &frameVariablesForValue =
      frameVariables.try_emplace(operand).first->getSecond();

  // Reuse existing extracted value if possible.
  Value image;
  auto existingImage = frameVariablesForValue.find(useState);
  bool wasExtractedInThisState = existingImage != frameVariablesForValue.end();
  // TODO: can this be parent region also?
  bool wasExtractedInThisRegion =
      wasExtractedInThisState && existingImage->getSecond().getParentRegion() ==
                                     opWithUse->getParentRegion();
  if (wasExtractedInThisRegion) {
    image = existingImage->getSecond();
  } else {
    builder.setInsertionPoint(opWithUse);
    if (operand == errorValue) {
      Value dataSlot = builder.create<StructGEPOp>(continuation, ErrorSlot);
      Value ptr = builder.create<LoadOp>(dataSlot);
      image = builder.create<PointerBitcastOp>(errorValue.getType(), ptr);
    } else if (operand == resultValue) {
      Value dataSlot = builder.create<StructGEPOp>(continuation, ResultSlot);
      Value ptr = builder.create<LoadOp>(dataSlot);
      image = builder.create<PointerBitcastOp>(resultValue.getType(), ptr);
    } else {
      unsigned frameIndex = entry->getSecond();
      Value dataSlot =
          builder.create<StructGEPOp>(continuation, Frame + frameIndex);
      if (operand.getDefiningOp() &&
          isa<StackAllocationOp>(operand.getDefiningOp())) {
        auto stackAlloc = dyn_cast<StackAllocationOp>(operand.getDefiningOp());
        if (cast<IntegerAttr>(stackAlloc.getCount()).getInt() == 1) {
          image = dataSlot;
        } else {
          image =
              builder.create<PointerBitcastOp>(stackAlloc.getType(), dataSlot);
        }
      } else {
        image = builder.create<LoadOp>(dataSlot);
      }
    }
    if (wasExtractedInThisState)
      frameVariablesForValue.erase(existingImage);
    frameVariablesForValue.insert({useState, image});
  }
  return image;
}

void FrameVariables::overwriteValue(int state, Value value) {
  DenseMap<int, Value> &frameVariablesForValue =
      frameVariables.try_emplace(value).first->getSecond();
  auto existing = frameVariablesForValue.find(state);
  if (existing != frameVariablesForValue.end())
    frameVariablesForValue.erase(existing);
  frameVariablesForValue.insert({state, value});
}

//===----------------------------------------------------------------------===//
// FrameData
//===----------------------------------------------------------------------===//

int FrameData::getDefinitionStateForValue(Value operand) const {
  Operation *definingOp = operand.getDefiningOp();
  // Initialize state to the entry state.
  int defState = -1;
  if (!definingOp) {
    BlockArgument blockArgument = cast<BlockArgument>(operand);
    // We always store the function arguments in the frame because they
    // originate in the ramp function.
    Operation *parentOp = blockArgument.getOwner()->getParentOp();
    if (isa<FuncOp>(parentOp))
      return defState;

    Operation *firstOp = &*blockArgument.getOwner()->begin();
    defState = opToState.at(firstOp);
    // In a loop we want to compare state before entering loop to the use
    // inside the body.
    if (isa<HLCF::LoopOp>(parentOp))
      defState = opToState.at(parentOp);
    return defState;
  }
  return opToState.at(definingOp);
}

FrameData::FrameData(
    FuncOp originalFunction, mlir::DominanceInfo &domInfo, Value errorValue,
    Value resultValue,
    function_ref<void(DenseMap<Operation *, int> &, FuncOp)> transform) {
  // Calculate Control Flow Graph.
  // We need to know the predecessors of each region so that
  // we don't process a region until all its predecessors have
  // been processed.
  DenseMap<VirtualBlock, SmallVector<VirtualBlock>> predecessors;
  {
    SmallVector<Region *> regions;
    DenseSet<Region *> visited;
    regions.push_back(&originalFunction.getBodyRegion());

    auto pushSuccessors =
        [&](SmallVector<HLCF::ControlFlowTarget> const &targets,
            Operation *controlFlowNode, Operation *controlFlowParent,
            VirtualBlock current) {
          for (HLCF::ControlFlowTarget target : targets) {
            VirtualBlock succ;
            if (target.index.has_value()) {
              Region *succRegion =
                  &controlFlowParent->getRegion(target.index.value());
              succ = &*succRegion->front().begin();
              regions.push_back(succRegion);
            } else {
              succ = controlFlowParent->getNextNode();
            }
            predecessors[succ].push_back(current);
          }
        };

    // There are three types of ops that form virtual block boundaries within a
    // region: control flow nodes, control flow terminators, and coroutine
    // awaits.
    while (!regions.empty()) {
      Region *region = regions.back();
      regions.pop_back();
      if (visited.contains(region))
        continue;
      visited.insert(region);
      Operation *lastControlFlowNode = nullptr;
      Operation *lastAwait = nullptr;
      for (Operation &op : region->front().getOperations()) {
        if (isa<ReturnOp, UnreachableOp>(op))
          continue;
        if (auto controlFlowTerminator =
                dyn_cast<HLCF::ControlFlowTerminator>(op)) {
          SmallVector<HLCF::ControlFlowTarget> targets;
          SmallVector<Attribute> controlFlowTerminatorOperands(
              controlFlowTerminator->getNumOperands(), Attribute());
          controlFlowTerminator.getBranchTargets(controlFlowTerminatorOperands,
                                                 targets);
          Operation *curr =
              lastControlFlowNode
                  ? lastControlFlowNode->getNextNode()
                  : &*controlFlowTerminator->getParentRegion()->front().begin();
          pushSuccessors(targets, controlFlowTerminator,
                         getParentNode(controlFlowTerminator), curr);
        }
        if (auto controlFlowNode = dyn_cast<HLCF::ControlFlowNode>(op)) {
          lastControlFlowNode = controlFlowNode;
          SmallVector<HLCF::ControlFlowTarget> targets;
          SmallVector<Attribute> controlFlowNodeOperands(
              controlFlowNode->getNumOperands(), Attribute());
          controlFlowNode.getEntryTargets(controlFlowNodeOperands, targets);
          pushSuccessors(targets, controlFlowNode, controlFlowNode,
                         &*controlFlowNode->getParentRegion()->front().begin());
        }
        if (auto suspend = dyn_cast<CO::SuspendOp>(op)) {
          Operation *next = suspend->getNextNode();
          if (!next)
            continue;
          Operation *nodeStart =
              lastAwait ? lastAwait->getNextNode() : &*region->front().begin();
          lastAwait = suspend;
          VirtualBlock awaitVirtualBlock = &suspend.getBody().front().front();
          predecessors[awaitVirtualBlock].push_back(nodeStart);
          predecessors[next].push_back(awaitVirtualBlock);
          regions.push_back(&suspend.getBody());
        }
      }
    }
  }

  // Calculate the state of each op.
  {
    SmallVector<std::pair<VirtualBlock, int>> paths;
    paths.push_back({&*originalFunction.getBodyRegion().front().begin(), 0});
    DenseMap<VirtualBlock, int> stateOfVirtualBlock;
    DenseSet<VirtualBlock> visited;
    auto pushSuccessors =
        [&](SmallVector<HLCF::ControlFlowTarget> const &targets,
            Operation *controlFlowVirtualBlock, int currentState,
            Operation *controlFlowParent) {
          for (HLCF::ControlFlowTarget target : targets) {
            if (target.index.has_value())
              paths.push_back(
                  {&*controlFlowParent->getRegion(target.index.value())
                         .front()
                         .begin(),
                   currentState});
            else
              paths.push_back({controlFlowParent->getNextNode(), currentState});
          }
        };

    // The state of a virtual block is the max of the states
    // of its predecessors.
    // Dry runs are needed for loops. Because the predecessor of a virtual block
    // does not dominate it, we first calculate all other predecessors and then
    // process the target in a dry run to compute the state. Then we process the
    // target a second time, considering the result of the first pass in its
    // state computation.
    SmallVector<VirtualBlock> dryRuns;
    VirtualBlock poppedDryRun = nullptr;
    while (!paths.empty()) {
      auto [virtualBlock, stateAtPred] = paths.back();
      paths.pop_back();
      if (visited.contains(virtualBlock))
        continue;

      bool allPredsHaveProcessed = true;
      int currentState = stateAtPred;
      bool needsDryRun;
      for (VirtualBlock predecessor : predecessors[virtualBlock]) {
        auto predMaybe = stateOfVirtualBlock.find(predecessor);
        // A dry run is needed if the predecessor does not dominate the
        // successor, which is the case in a loop.
        needsDryRun =
            virtualBlock->getParentRegion() == predecessor->getParentRegion() &&
            domInfo.dominates(virtualBlock, predecessor);
        bool predIsProcessed = predMaybe != stateOfVirtualBlock.end();

        // The needed dry run is already complete. Remove record of it so it's
        // results can be recorded.
        if (needsDryRun && predIsProcessed && dryRuns.back() == virtualBlock) {
          poppedDryRun = dryRuns.back();
          dryRuns.pop_back();
        }
        if (!predIsProcessed) {
          allPredsHaveProcessed = false;
          if (needsDryRun) {
            dryRuns.push_back(virtualBlock);
            paths.push_back({virtualBlock, stateAtPred});
          }
          break;
        }
        int stateFromPred = predMaybe->getSecond();
        if (stateFromPred > currentState)
          currentState = stateFromPred;
      }
      // Another predecessor of this region needs to process before we can
      // process this region.
      if (!allPredsHaveProcessed && !needsDryRun)
        continue;
      // Iterate through each op in this virtual block to register its state.
      // The boundaries of a node are defined by awaits, control
      // flow nodes, and control flow terminators.
      Operation *current = virtualBlock;
      while (current) {
        Operation *op = current;
        current = op->getNextNode();
        if (auto suspend = dyn_cast<CO::SuspendOp>(op)) {
          opToState[op] = currentState;
          // If the virtual block is equal to the poppedDryRun then this will
          // result in the coroutine suspend assuming it's second iteration
          // state instead of the original. Despite this we still want to update
          // the state to divide states within the loop. Otherwise we cannot
          // cache frame variable extractions.
          int nextState = currentState + 1;
          if (Operation *next = op->getNextNode())
            paths.push_back({next, nextState});
          paths.push_back({&*suspend.getBody().front().begin(), currentState});
          break;
        }
        opToState[op] = currentState;
        if (isa<ReturnOp, UnreachableOp>(op))
          continue;
        if (auto controlFlowNode = dyn_cast<HLCF::ControlFlowNode>(op)) {
          SmallVector<HLCF::ControlFlowTarget> targets;
          SmallVector<Attribute> controlFlowNodeOperands(
              controlFlowNode->getNumOperands(), Attribute());
          controlFlowNode.getEntryTargets(controlFlowNodeOperands, targets);
          pushSuccessors(targets, controlFlowNode, currentState,
                         controlFlowNode);
          break;
        }
        if (auto controlFlowTerminator =
                dyn_cast<HLCF::ControlFlowTerminator>(op)) {
          SmallVector<HLCF::ControlFlowTarget> targets;
          SmallVector<Attribute> controlFlowTerminatorOperands(
              controlFlowTerminator->getNumOperands(), Attribute());
          controlFlowTerminator.getBranchTargets(controlFlowTerminatorOperands,
                                                 targets);
          pushSuccessors(targets, controlFlowTerminator, currentState,
                         getParentNode(controlFlowTerminator));
          break;
        }
      }
      stateOfVirtualBlock.insert({virtualBlock, currentState});
      if (dryRuns.empty()) {
        visited.insert(virtualBlock);
      } else if (virtualBlock == poppedDryRun) {
        poppedDryRun = nullptr;
        visited.insert(virtualBlock);
      }
    }
  }
  // Give caller chance to transform before frame calculation.
  transform(opToState, originalFunction);

  // Calculate the frame. Whenever there is a use whose def lives in another
  // state, it must be added to the frame. We need to store the location of the
  // value in the frame so that when we generate the resume function the frame
  // values can be extracted.
  auto addToFrame = [&](Type frameVariableType, Operation *definingOp) {
    unsigned index = frameTypes.size();
    frameTypes.push_back(frameVariableType);
    operationToIndexInFrame.insert({definingOp, index});
  };
  auto stackAllocationFrameType =
      [](StackAllocationOp stackAllocation) -> Type {
    int64_t count = cast<IntegerAttr>(stackAllocation.getCount()).getInt();
    if (count == 1) {
      return stackAllocation.getType().getElementType();
    } else {
      return POP::ArrayType::get(stackAllocation.getCount(),
                                 stackAllocation.getType().getElementType());
    }
  };
  originalFunction.walk([&](Operation *operation) {
    int useState = opToState[operation];
    if (StackAllocationOp stackAllocationOp =
            dyn_cast<StackAllocationOp>(operation)) {
      if (!stackAllocationOp.getMarkedLifetimes()) {
        int endState = opToState
            [stackAllocationOp->getParentRegion()->front().getTerminator()];
        if (endState > useState)
          addToFrame(stackAllocationFrameType(stackAllocationOp),
                     stackAllocationOp);
      }
    }

    for (Value operand : operation->getOperands()) {
      // Results and Errors are stored in the header of the continuation, not in
      // the frame.
      if (operand == errorValue || operand == resultValue)
        continue;
      if (valueToIndexInFrame.contains(operand))
        continue;

      // Add to frame if the value was defined in a previous state.
      int defState = getDefinitionStateForValue(operand);
      if (defState < useState) {
        unsigned index = frameTypes.size();
        if (Operation *definitingOp = operand.getDefiningOp()) {
          if (auto stackAllocation =
                  dyn_cast<StackAllocationOp>(definitingOp)) {
            // Stack allocations without marked lifetimes are checked for frame
            // membership at definition site.
            if (stackAllocation.getMarkedLifetimes()) {
              addToFrame(stackAllocationFrameType(stackAllocation),
                         definitingOp);
            } else {
              valueToIndexInFrame.insert(
                  {operand, operationToIndexInFrame[definitingOp]});
              continue;
            }
          } else {
            addToFrame(operand.getType(), definitingOp);
          }
        } else {
          frameTypes.push_back(operand.getType());
        }
        valueToIndexInFrame.insert({operand, index});
      }
    }
  });
}

//===----------------------------------------------------------------------===//
// LowerAsyncFunctionsPass
//===----------------------------------------------------------------------===//

void LowerAsyncFunctionsPass::runOnOperation() {
  ModuleOp module = getOperation();
  TargetInfoAttr targetInfo = lookupTargetInfo(module);
  if (!targetInfo) {
    mlir::emitError(module.getLoc(),
                    "could not find an enclosing target specification");
    return signalPassFailure();
  }

  SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
  Shared<SymbolTable &> sharedTable(symtab);
  DenseMap<SymbolConstantAttr, std::pair<SymbolConstantAttr, Type>>
      asyncFuncToRampFunctions;

  // Convert async functions.
  ImplicitLocOpBuilder b(module->getLoc(), module);
  LowerAsyncBuildContext buildContext(sharedTable, asyncFuncToRampFunctions, b,
                                      targetInfo);
  auto &domInfo = getAnalysis<mlir::DominanceInfo>();
  for (auto funcOp : llvm::make_early_inc_range(module.getOps<FuncOp>())) {
    if (!funcOp.isAsync())
      continue;
    Value errorValue;
    Value memoryResultValue;
    if (funcOp.isThrows() || funcOp.getSignature().hasMemoryOnlyResult()) {
      int errorIndex = -1, resultIndex = -1;
      for (auto [i, convention] :
           llvm::enumerate(funcOp.getSignature().getArgConventions())) {
        if (convention == M::KGEN::ArgConvention::ByRefError)
          errorIndex = i;
        else if (convention == M::KGEN::ArgConvention::ByRefResult)
          resultIndex = i;
      }
      if (errorIndex > -1)
        errorValue = funcOp.getArgument(errorIndex);
      if (resultIndex > -1)
        memoryResultValue = funcOp.getArgument(resultIndex);
    }
    // Preprocess the function to move stack allocation ops as close to
    // lifetime start as possible.
    DenseMap<StackAllocationOp, Operation *> closestCommonParent;
    funcOp.walk([&](StackAllocLifetimeStartOp startOp) {
      for (Value value : startOp.getValues()) {
        StackAllocationOp stackAlloc =
            cast<StackAllocationOp>(value.getDefiningOp());
        auto entry = closestCommonParent.find(stackAlloc);
        if (entry == closestCommonParent.end()) {
          closestCommonParent[stackAlloc] = startOp;
        } else {
          Operation *opInCommonRegion = entry->second;
          Region *currentRegion = startOp->getParentRegion();
          while (!opInCommonRegion->getParentRegion()->isProperAncestor(
                     currentRegion) &&
                 opInCommonRegion->getParentRegion() != currentRegion) {
            opInCommonRegion = opInCommonRegion->getParentOp();
          }
          closestCommonParent[stackAlloc] = opInCommonRegion;
        }
      }
    });
    for (auto [stackAllocation, closestCommonParentOp] : closestCommonParent)
      stackAllocation->moveBefore(closestCommonParentOp);
    auto transform = [&](DenseMap<Operation *, int> &opToState, FuncOp target) {
      // Now that we have a state label for every op, clone stack allocations
      // that have multiple lifespans
      SmallVector<Block *> blocks;
      blocks.push_back(&target.getBodyRegion().front());
      while (!blocks.empty()) {
        Block *current = blocks.pop_back_val();
        for (Operation &op : llvm::make_early_inc_range(*current)) {
          int useState = opToState[&op];
          if (auto startOp = dyn_cast<StackAllocLifetimeStartOp>(op)) {
            // Clone the original stack alloc and replace all uses of the
            // original that appear after this start lifetime marker.
            auto oldPoint = b.saveInsertionPoint();
            b.setInsertionPoint(startOp);
            for (auto [index, value] : llvm::enumerate(startOp.getValues())) {
              auto stackAllocationOp =
                  cast<StackAllocationOp>(value.getDefiningOp());
              int defState = opToState[stackAllocationOp];
              if (defState == useState)
                continue;
              // Otherwise, this is a second lifetime. We need a clone.
              auto clone = cast<StackAllocationOp>(b.clone(*stackAllocationOp));
              opToState[clone] = useState;
              startOp->setOperand(index, clone.getResult());
              stackAllocationOp->replaceUsesWithIf(
                  clone, [&](OpOperand &operand) -> bool {
                    return domInfo.dominates(startOp, operand.getOwner());
                  });
            }
            b.restoreInsertionPoint(oldPoint);
          }
          for (Region &region : op.getRegions())
            blocks.push_back(&region.front());
        }
      }
    };
    FrameData frameData(funcOp, domInfo, errorValue, memoryResultValue,
                        transform);
    COTypes cotypes(
        module.getContext(), frameData,
        StructType::get(funcOp.getContext(), funcOp.getResultTypes()));
    buildContext.lowerAsyncFunction(funcOp, domInfo, cotypes, errorValue,
                                    memoryResultValue);
  }

  // Apply all other CO lowerings.
  mlir::IRRewriter rewriter(b);
  FrameData empty;
  COTypes opaqueCoroutineTypes(module.getContext(), empty, /*promiseType=*/{});
  mlir::AttrTypeReplacer replacer;
  Type headerType = PointerType::get(opaqueCoroutineTypes.getHeaderType());
  replacer.addReplacement([&](CoroutineType type) { return headerType; });
  replacer.recursivelyReplaceElementsIn(module, true, true, true);
  module.walk([&](Operation *op) {
    if (auto invokeOp = dyn_cast<InvokeOp>(op)) {
      auto symbol = cast<SymbolConstantAttr>(invokeOp.getCallee());
      auto newSymbolPtr = asyncFuncToRampFunctions.find(symbol);
      if (newSymbolPtr != asyncFuncToRampFunctions.end()) {
        auto [newSymbol, continuationType] = newSymbolPtr->getSecond();
        rewriter.setInsertionPoint(op);
        auto callOp = rewriter.create<CallOp>(
            invokeOp->getLoc(),
            PointerType::get(opaqueCoroutineTypes.getHeaderType()), newSymbol,
            invokeOp.getOperands());
        rewriter.replaceOp(invokeOp, callOp);
      }
    } else if (auto setErrorResultOp = dyn_cast<SetByRefErrorAndResultOp>(op)) {
      rewriter.setInsertionPoint(op);
      Value continuation = setErrorResultOp.getOperand(0);
      auto setByRefArgument = [&](Value argument, unsigned index) {
        Value slot =
            rewriter.create<StructGEPOp>(op->getLoc(), continuation, index);
        Value typedSlot = rewriter.create<PointerBitcastOp>(
            op->getLoc(), KGEN::PointerType::get(argument.getType()), slot);
        rewriter.create<StoreOp>(op->getLoc(), argument, typedSlot);
      };
      if (Value error = setErrorResultOp.getError())
        setByRefArgument(error, AsyncContinuationField::ErrorSlot);
      if (!isa<KGEN::NoneType>(
              setErrorResultOp.getResult().getType().getElementType())) {
        Value result = setErrorResultOp.getResult();
        setByRefArgument(result, AsyncContinuationField::ResultSlot);
      }
      op->erase();
    } else if (auto resumeOp = dyn_cast<ResumeOp>(op)) {
      rewriter.setInsertionPoint(op);
      Value continuation = resumeOp.getOperand();
      Value slot = rewriter.create<StructGEPOp>(op->getLoc(), continuation,
                                                ResumeFunction);
      Value typed = rewriter.create<PointerBitcastOp>(
          op->getLoc(), PointerType::get(resumeOp.getType()), slot);
      Value load = rewriter.create<LoadOp>(op->getLoc(), typed);
      resumeOp.replaceAllUsesWith(load);
      resumeOp->erase();
    } else if (auto callbackOp = dyn_cast<GetCallbackPtrOp>(op)) {
      rewriter.setInsertionPoint(op);
      Value continuation = callbackOp.getOperand();
      Value slot =
          rewriter.create<StructGEPOp>(op->getLoc(), continuation, CallbackFn);
      Value slotCast = rewriter.create<PointerBitcastOp>(
          op->getLoc(), callbackOp.getType(), slot);
      callbackOp.replaceAllUsesWith(slotCast);
      callbackOp->erase();
    } else if (auto destroyOp = dyn_cast<DestroyOp>(op)) {
      rewriter.setInsertionPoint(op);
      Value continuation = destroyOp.getOperand();
      rewriter.create<AlignedFreeOp>(destroyOp->getLoc(), continuation);
      destroyOp->erase();
    } else if (auto getResults = dyn_cast<GetResultsOp>(op)) {
      rewriter.setInsertionPoint(op);
      Value continuation = getResults.getOperand();
      StructType headerType = opaqueCoroutineTypes.getHeaderType();
      SmallVector<Type> headerPlusPromiseTypes(headerType.getElementTypes());
      headerPlusPromiseTypes.push_back(StructType::get(
          op->getContext(), llvm::to_vector(getResults.getResultTypes())));
      Value promiseContinuation = rewriter.create<PointerBitcastOp>(
          op->getLoc(),
          PointerType::get(StructType::get(headerPlusPromiseTypes)),
          continuation);
      Value promiseSlot = rewriter.create<StructGEPOp>(
          op->getLoc(), promiseContinuation, Promise);
      for (auto [idx, result] : llvm::enumerate(getResults.getResults())) {
        rewriter.replaceAllUsesWith(
            result, rewriter.create<LoadOp>(
                        op->getLoc(), rewriter.create<StructGEPOp>(
                                          op->getLoc(), promiseSlot, idx)));
      }
      getResults->erase();
    }
  });
}
