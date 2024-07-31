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
            function_ref<void(FuncOp, DenseMap<Operation *, int> &)> transform,
            bool isHot);
  FrameData() {}

  /// Given a value, determine the state of its defining op or block argument.
  int getDefinitionStateForValue(Value operand, bool isHot) const;

  /// Update the ops in this virtual block.
  void updateVirtualBlock(Operation *virtualBlock, int newState);

  SmallVector<Type> frameTypes;
  DenseMap<Value, unsigned> valueToIndexInFrame;
  DenseMap<Operation *, unsigned> operationToIndexInFrame;
  DenseMap<Operation *, int> opToState;
  SmallVector<Operation *> virtualBlocksFirstState;
};

static bool needsStateClone(Operation *operation) {
  return operation->hasTrait<OpTrait::ConstantLike>() ||
         isa<KGEN::StructGEPOp, POP::OffsetOp>(operation);
}

struct CloneFrameArgs {
  CloneFrameArgs(ImplicitLocOpBuilder &b, DenseMap<Operation *, int> &opToState,
                 mlir::DominanceInfo &dominanceInfo)
      : builder(b), opToState(opToState), dominanceInfo(dominanceInfo) {}
  void cloneFrameArgsOf(Operation *user) {
    int useState = opToState[user];
    for (auto [index, operand] : llvm::enumerate(user->getOperands())) {
      Operation *definingOp = operand.getDefiningOp();
      if (!definingOp)
        continue;
      if (!needsStateClone(definingOp))
        continue;
      int defState = opToState[definingOp];
      if (defState == useState)
        continue;

      auto existing = constantToStateSpecific.find(definingOp);
      if (existing == constantToStateSpecific.end())
        existing = constantToStateSpecific.try_emplace(definingOp).first;
      auto existingClone = existing->second.find(useState);
      Operation *clonedDefOp;
      if (existingClone == existing->second.end()) {
        builder.setInsertionPoint(user);
        clonedDefOp = builder.clone(*definingOp);
        opToState[clonedDefOp] = useState;
        existing->second.insert({useState, clonedDefOp});
        if (clonedDefOp->getNumOperands() > 0)
          cloneFrameArgsOf(clonedDefOp);
      } else {
        clonedDefOp = existingClone->second;
        if (!dominanceInfo.dominates(clonedDefOp, user)) {
          // We have two uses in the same state where one does not dominate the
          // other. This implies that the first instance of the usage is in a
          // nested region.
          Operation *parent = clonedDefOp->getParentOp();
          while (!dominanceInfo.dominates(parent, user))
            parent = parent->getParentOp();
          clonedDefOp->moveBefore(parent);
        }
      }
      user->setOperand(index, clonedDefOp->getResult(0));
    }
  }
  ImplicitLocOpBuilder &builder;
  DenseMap<Operation *, int> &opToState;
  DenseMap<Operation *, DenseMap<int, Operation *>> constantToStateSpecific;
  mlir::DominanceInfo &dominanceInfo;
};

static void cloneFrameArgs(FuncOp funcOp, ImplicitLocOpBuilder &b,
                           mlir::DominanceInfo &domInfo,
                           FuncOp originalFunction,
                           DenseMap<Operation *, int> &opToState) {
  auto insertPoint = b.saveInsertionPoint();
  CloneFrameArgs cloner(b, opToState, domInfo);
  funcOp.walk([&](Operation *user) {
    if (user->getNumOperands() > 0)
      cloner.cloneFrameArgsOf(user);
  });
  b.restoreInsertionPoint(insertPoint);
}

struct COTypes {
  Type typeForField(AsyncContinuationField field) {
    switch (field) {
    case State:
      return IntegerType::get(cxt, 32);
    case CallbackFn:
      return callbackSignature;
    case Promise:
      return promiseType;
    case ResumeFunction:
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
  Type getResumeSignatureType() const { return resumeSignatureType; }

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

/// The original clone will become the hot ramp if needed.
/// The resume function is shared between hot and cold.
struct AsyncFunctionInfo {
  AsyncFunctionInfo(FuncOp originalFunction, FuncOp resume,
                    SmallVector<Operation *> &&firstStateVirtualBlocks)
      : originalClone(originalFunction), resume(resume),
        firstStateVirtualBlocks(std::move(firstStateVirtualBlocks)) {}
  AsyncFunctionInfo() {}
  FuncOp originalClone;
  FuncOp resume;
  SmallVector<Operation *> firstStateVirtualBlocks;
};

/// The LowerAsyncBuildContext is responsible for transforming an async function
/// into a ramp function and resume function.
struct LowerAsyncBuildContext {
  LowerAsyncBuildContext(
      Shared<SymbolTable &> &sharedTable,
      DenseMap<SymbolConstantAttr, std::pair<SymbolConstantAttr, Type>>
          &asyncFuncToRampFunctions,
      ImplicitLocOpBuilder &builder, mlir::DominanceInfo &domInfo,
      TargetInfoAttr targetInfoAttr)
      : sharedTable(sharedTable),
        asyncFuncToRampFunctions(asyncFuncToRampFunctions), builder(builder),
        targetInfoAttr(targetInfoAttr), dominanceInfo(domInfo) {}

  /// Given an async function and its frame types, create a ramp function and a
  /// resume function.
  LoweredAsyncFunction lowerAsyncFunction(FuncOp funcOp, COTypes &coTypes,
                                          Value errorValue,
                                          Value memoryResultValue);

  /// Given an async function and its frame types, create a ramp function and a
  /// resume function.
  LoweredAsyncFunction
  lowerAsyncFunctionHot(AsyncFunctionInfo &asyncFunctionInfo);

  /// Store async functions that may need to transformed into hot ramp/resumes.
  DenseMap<SymbolConstantAttr, AsyncFunctionInfo> symbolToClone;

private:
  /// Create the continuation and initialize the state and resume function.
  Value initializeContinuation(FuncOp rampFunction, FuncOp resumeFunction,
                               FuncOp funcOp, COTypes &coTypes,
                               unsigned initialState);
  /// Given the resume function, the empty ramp function, the original async
  /// function, and coroutine types, populate the ramp function.
  void populateRampFunction(FuncOp rampFunction, FuncOp resumeFunction,
                            FuncOp funcOp, COTypes &coTypes);
  /// Given the empty resume function, the original async function, and the
  /// coroutine types, populate the resume function.
  void populateResumeFunction(FuncOp resumeFunction, FuncOp funcOp,
                              COTypes &coTypes, Value errorValue,
                              Value memoryResultValue);
  /// Given a resume function and the original async function, create a hot
  /// ramp function. A hot ramp function creates the continuation and executes
  /// the first state of the coroutine.
  FuncOp createHotRamp(FuncOp resumeFunction, FuncOp funcOp, COTypes &coTypes);

  /// Given an async function, populate a function with the paths to
  /// the first suspension point. If there is a path with no suspension point,
  /// the callback is invoked.
  std::pair<Value, Value>
  addSlicedFirstStateTo(FuncOp hotRamp, FuncOp fromFuncOp, COTypes &types);
  /// Replace all `return x` with `store x, y` where y is the address of the
  /// result slot in the frame.
  ReturnOp lowerReturn(ReturnOp returnOp, Value continuation);
  /// Replace handle argument with local coroutine
  void lowerSuspensionPoint(CO::SuspendOp suspend, Value continuation,
                            COTypes &coTypes);
  /// Lower a hot invoke into a suspension point + call to hot ramp. This
  /// triggers the generation of the hot ramp
  void lowerHotInvoke(Value continuation, HotInvokeOp hotInvoke,
                      COTypes &coTypes);
  /// Store block arguments in the frame if they are used across a suspension
  /// point.
  void storeBlockArgumentsInFrame(Block &block, Operation *key,
                                  FrameData &frameData, Value continuation,
                                  FrameVariables &frameVariables);
  /// Store the given op in the frame if it is used across suspension points.
  void storeOpInFrameIfNeeded(FrameData &frameData, Operation *op,
                              Value continuation,
                              SmallVector<Operation *> &opsToDelete);

  Shared<SymbolTable &> &sharedTable;
  DenseMap<SymbolConstantAttr, std::pair<SymbolConstantAttr, Type>>
      &asyncFuncToRampFunctions;
  DenseMap<SymbolConstantAttr, SymbolConstantAttr> fromOriginalToHotRamp;
  ImplicitLocOpBuilder &builder;
  TargetInfoAttr targetInfoAttr;
  mlir::DominanceInfo &dominanceInfo;
};

//===----------------------------------------------------------------------===//
// LowerAsyncBuildContext
//===----------------------------------------------------------------------===//

enum class VisitedState { SUS, NOSUS, SUS_AND_NOSUS };

std::pair<Value, Value>
LowerAsyncBuildContext::addSlicedFirstStateTo(FuncOp hotRamp, FuncOp fromFuncOp,
                                              COTypes &cotypes) {
  /// Augment the hot ramp function signature. We will clone ops from the funcOp
  /// into the hot ramp function.
  SmallVector<Type> inputs;
  Type closureType = cotypes.typeForField(M::KGEN::ClosureState);
  Type callbackType = cotypes.typeForField(M::KGEN::CallbackFn);
  inputs.push_back(callbackType);
  inputs.push_back(closureType);

  for (BlockArgument oldArg : fromFuncOp.getArguments())
    inputs.push_back(oldArg.getType());

  hotRamp.setSignature(SignatureType::get(
      hotRamp.getContext(), inputs, PointerType::get(cotypes.getHeaderType())));
  hotRamp.getBodyRegion().takeBody(fromFuncOp.getBodyRegion());
  Value callback = hotRamp.getBodyRegion().front().insertArgument(
      (unsigned)0, callbackType, hotRamp->getLoc());
  Value closureState = hotRamp.getBodyRegion().front().insertArgument(
      1, closureType, hotRamp->getLoc());

  /// Clone from funcOp into hot ramp until first suspension point.
  SmallVector<std::pair<Operation *, bool>> paths;
  auto addTargets = [&](ArrayRef<HLCF::ControlFlowTarget> targets,
                        Operation *cfn, bool hitSus) {
    for (HLCF::ControlFlowTarget target : targets) {
      if (target.index.has_value()) {
        unsigned index = target.index.value();
        Block &sourceBlock = cfn->getRegion(index).front();
        paths.push_back({&*sourceBlock.begin(), hitSus});
      } else {
        paths.push_back({cfn->getNextNode(), hitSus});
      }
    }
  };
  DenseMap<Operation *, VisitedState> visited;
  DenseSet<Operation *> reachable;
  paths.push_back({&hotRamp.getBodyRegion().front().front(), false});
  while (!paths.empty()) {
    std::pair<Operation *, bool> c = paths.pop_back_val();
    Operation *current = c.first;
    bool hitSus = c.second;
    bool needsEdit = true;
    auto ptr = visited.find(current);
    if (ptr != visited.end()) {
      needsEdit = false;
      VisitedState state = ptr->getSecond();
      if ((state == VisitedState::SUS && !hitSus) ||
          (state == VisitedState::NOSUS && hitSus))
        visited[current] = VisitedState::SUS_AND_NOSUS;
      else
        continue;
    } else {
      visited.insert(
          {current, hitSus ? VisitedState::SUS : VisitedState::NOSUS});
    }
    for (Operation *operation = current; operation != nullptr;
         operation = operation->getNextNode()) {
      if (!hitSus)
        reachable.insert(operation);

      if (auto suspendOp = dyn_cast<SuspendOp>(operation)) {
        Operation *end = suspendOp;
        if (needsEdit) {
          builder.setInsertionPointAfter(suspendOp);
          // keep suspension points around for now.
          for (auto &op : suspendOp.getBody().front().getOperations())
            reachable.insert(&op);
          end = builder.create<KGEN::ReturnOp>();
          reachable.insert(end);
          if (end->getNextNode())
            paths.push_back({end->getNextNode(), true});
        }
        break;
      }
      if (auto cfn = dyn_cast<HLCF::ControlFlowNode>(operation)) {
        SmallVector<HLCF::ControlFlowTarget> targets;
        SmallVector<Attribute> controlFlowOperands(cfn->getNumOperands(),
                                                   Attribute());
        cfn.getEntryTargets(controlFlowOperands, targets);
        addTargets(targets, cfn, hitSus);
        break;
      }
      if (isa<ReturnOp>(operation) && needsEdit) {
        // TODO: enable musttail at the kgen level. MOCO-987
        builder.setInsertionPoint(operation);
        SignatureType signatureType = cast<SignatureType>(callback.getType());
        auto callIndirect = builder.create<CallIndirectOp>(
            signatureType.getResults(), callback, closureState);
        reachable.insert(callIndirect);
        break;
      }
      if (isa<UnreachableOp, SuspendEndOp>(operation))
        break;
      if (auto terminator = dyn_cast<HLCF::ControlFlowTerminator>(operation)) {
        SmallVector<HLCF::ControlFlowTarget> targets;
        SmallVector<Attribute> controlFlowTerminatorOperands(
            terminator->getNumOperands(), Attribute());
        terminator.getBranchTargets(controlFlowTerminatorOperands, targets);
        addTargets(targets, HLCF::getParentNode(terminator), hitSus);
        break;
      }
    }
  }
  SmallVector<Operation *> deleteMe;
  SmallVector<Region *> removalRegions;
  SmallVector<Region *> regions;
  regions.push_back(&hotRamp.getBodyRegion());
  while (!regions.empty()) {
    Region *region = regions.front();
    regions.erase(regions.begin());
    removalRegions.push_back(region);
    for (Operation &op : region->front().getOperations()) {
      for (Region &r : op.getRegions())
        regions.push_back(&r);
    }
  }
  for (auto i = removalRegions.rbegin(); i != removalRegions.rend(); i++) {
    Region *region = *i;
    for (auto opIter = region->front().getOperations().rbegin();
         opIter != region->front().rend();) {
      Operation &op = *opIter;
      opIter++;
      if (!reachable.contains(&op))
        op.erase();
    }
  }
  return {callback, closureState};
}

LoweredAsyncFunction LowerAsyncBuildContext::lowerAsyncFunctionHot(
    AsyncFunctionInfo &asyncFunctionInfo) {
  FuncOp funcOp = asyncFunctionInfo.originalClone;
  FuncOp resume = asyncFunctionInfo.resume;
  builder.setInsertionPoint(funcOp->getParentOp());
  std::function<void(FuncOp, DenseMap<Operation *, int> &)> transform =
      [&](FuncOp originalFunction, DenseMap<Operation *, int> &opToState) {
        cloneFrameArgs(funcOp, builder, dominanceInfo, originalFunction,
                       opToState);
      };

  int index = funcOp.getNumArguments() - 1;
  Value memoryResultValue;
  if (funcOp.getSignature().hasMemoryOnlyResult())
    memoryResultValue = funcOp.getArgument(index--);
  Value errorValue;
  if (funcOp.isThrows())
    errorValue = funcOp.getArgument(index);
  FrameData frameData(funcOp, dominanceInfo, errorValue, memoryResultValue,
                      transform, /*isHot=*/true);
  COTypes coTypes(
      funcOp.getContext(), frameData,
      StructType::get(funcOp.getContext(), funcOp.getResultTypes()));

  FuncOp hotRamp = createHotRamp(resume, funcOp, coTypes);

  // Resume function is shared. Replace coroutine type of resume with hot frame
  // type.
  Type coldContType = resume.getCoroutineType().value();
  Type hotContType = coTypes.getContinuationType();
  unsigned hotContSize = cast<StructType>(hotContType).getElementTypes().size();
  mlir::AttrTypeReplacer walker;
  walker.addReplacement([coldContType, hotContType](Type type) {
    if (type == coldContType)
      return hotContType;
    return type;
  });
  walker.recursivelyReplaceElementsIn(resume, true, true, true);

  // Replace arg0 with local variable for first state
  builder.setInsertionPointToStart(&resume.getBodyRegion().front());
  Value hotStartContinuation = resume.getArgument(0);
  auto pointerBitcast = builder.create<PointerBitcastOp>(
      PointerType::get(coldContType), hotStartContinuation);
  Value coldStartContinuation = pointerBitcast.getResult();
  for (VirtualBlock virtualBlock : asyncFunctionInfo.firstStateVirtualBlocks) {
    // The virtualBlock should point to the first operation of the block. We may
    // have made insertions since the recording so find the head.
    Operation *current = virtualBlock;
    while (current->getPrevNode())
      current = current->getPrevNode();
    while (current) {
      auto gep = dyn_cast<StructGEPOp>(current);
      if (gep && gep.getContainer() == hotStartContinuation) {
        if (gep.getIndex() >= hotContSize)
          current->setOperand(0, coldStartContinuation);
      }
      Operation *next = current->getNextNode();
      if (next && isa<HLCF::ControlFlowNode, HLCF::ControlFlowTerminator,
                      CO::SuspendEndOp>(next))
        break;
      current = next;
    }
  }
  funcOp.erase();
  return {hotRamp, resume};
}

LoweredAsyncFunction
LowerAsyncBuildContext::lowerAsyncFunction(FuncOp funcOp, COTypes &coTypes,
                                           Value errorValue,
                                           Value memoryResultValue) {
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
  auto resumeSignature = SignatureType::get(
      cxt, PointerType::get(coTypes.getContinuationType()), {});
  FuncOp resumeFunction = builder.create<FuncOp>(
      funcOp->getParentOp()->getLoc(), resumeName, resumeSignature);
  resumeFunction.setCoroutineTypeAttr(
      TypeAttr::get(coTypes.getContinuationType()));
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
  // Take the body of the original function.
  resumeFunction.getBodyRegion().takeBody(funcOp.getBodyRegion());
  FrameData &frameData = coTypes.getFrameData();
  resumeFunction.getBodyRegion().insertArgument(
      (unsigned)0, PointerType::get(coTypes.getContinuationType()),
      resumeFunction->getLoc());
  builder.setInsertionPointToStart(&resumeFunction.getBodyRegion().front());
  Value continuation = resumeFunction.getArgument(0);
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

      storeOpInFrameIfNeeded(frameData, op, continuation, opsToDelete);

      // Extract arguments from operands if needed. Hot start ramp should never
      // load from frame.
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
        storeBlockArgumentsInFrame(region.front(), firstOp, frameData,
                                   continuation, frameVariables);
      }
    }
  }

  llvm::BitVector args(resumeFunction.getBodyRegion().front().getNumArguments(),
                       true);
  args.reset(0);
  resumeFunction.getBodyRegion().front().eraseArguments(args);
  resumeFunction.walk([&](Operation *op) {
    if (auto returnOp = dyn_cast<ReturnOp>(op)) {
      lowerReturn(returnOp, continuation);
    } else if (auto suspend = dyn_cast<SuspendOp>(op)) {
      lowerSuspensionPoint(suspend, continuation, coTypes);
    } else if (auto hotInvoke = dyn_cast<HotInvokeOp>(op)) {
      lowerHotInvoke(continuation, hotInvoke, coTypes);
    }
  });
  for (auto op : opsToDelete)
    op->erase();
}

Value LowerAsyncBuildContext::initializeContinuation(FuncOp rampFunction,
                                                     FuncOp resumeFunction,
                                                     FuncOp funcOp,
                                                     COTypes &coTypes,
                                                     unsigned initialState) {
  Type continuationType = coTypes.getContinuationType();
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
  Value zero =
      builder.create<ParamConstantOp>(builder.getI32IntegerAttr(initialState));
  Value stateSlot = builder.create<StructGEPOp>(continuation, State);
  builder.create<StoreOp>(zero, stateSlot);

  // Store resume function.
  Value resumeFunctionSlot =
      builder.create<StructGEPOp>(continuation, ResumeFunction);
  Value functionPointer =
      builder.create<CreateClosureOp>(SymbolConstantAttr::get(
          SymbolRefAttr::get(builder.getContext(), resumeFunction.getSymName()),
          resumeFunction.getSignature()));
  functionPointer = builder.create<PointerBitcastOp>(
      coTypes.typeForField(ResumeFunction), functionPointer);
  builder.create<StoreOp>(functionPointer, resumeFunctionSlot);
  return continuation;
}

ReturnOp LowerAsyncBuildContext::lowerReturn(ReturnOp returnOp,
                                             Value continuation) {
  builder.setInsertionPoint(returnOp);
  if (returnOp->getNumOperands()) {
    // Replace ReturnOps with set result.
    Value promiseSlot = builder.create<StructGEPOp>(continuation, Promise);
    for (auto [idx, value] : llvm::enumerate(returnOp.getOperands())) {
      builder.create<StoreOp>(value,
                              builder.create<StructGEPOp>(promiseSlot, idx));
    }
    auto result = builder.create<ReturnOp>();
    returnOp->erase();
    return result;
  }
  return returnOp;
}

void LowerAsyncBuildContext::lowerSuspensionPoint(CO::SuspendOp suspend,
                                                  Value continuation,
                                                  COTypes &coTypes) {
  // Replace uses of the suspend argument with the continuation.
  Region &body = suspend.getBody();
  if (!body.getArgument(0).use_empty()) {
    builder.setInsertionPointToStart(&suspend.getBody().front());
    Value header = builder.create<PointerBitcastOp>(
        PointerType::get(coTypes.getHeaderType()), continuation);
    body.getArgument(0).replaceAllUsesWith(header);
  }
  body.eraseArgument(0);
}

void LowerAsyncBuildContext::storeBlockArgumentsInFrame(
    Block &block, Operation *key, FrameData &frameData, Value continuation,
    FrameVariables &frameVariables) {
  if (block.getNumArguments() == 0)
    return;
  builder.setInsertionPointToStart(&block);
  int frameValueState = frameData.opToState[key];
  for (BlockArgument argument : block.getArguments()) {
    auto entry = frameData.valueToIndexInFrame.find(
        key->getParentRegion()->getArgument(argument.getArgNumber()));
    if (entry == frameData.valueToIndexInFrame.end())
      continue;
    Value dataSlot =
        builder.create<StructGEPOp>(continuation, Frame + entry->getSecond());
    builder.create<StoreOp>(argument, dataSlot);
    frameVariables.overwriteValue(frameValueState, argument);
  }
}

void LowerAsyncBuildContext::storeOpInFrameIfNeeded(
    FrameData &frameData, Operation *op, Value continuation,
    SmallVector<Operation *> &opsToDelete) {
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
      opsToDelete.push_back(op);
    return;
  }
  auto entry = frameData.operationToIndexInFrame.find(op);
  if (entry != frameData.operationToIndexInFrame.end()) {
    if (isa<StackAllocationOp>(op)) {
      opsToDelete.push_back(op);
      return;
    }
    builder.setInsertionPointAfter(op);
    [[maybe_unused]] Type frameEntryType =
        frameData.frameTypes[entry->getSecond()];
    assert(frameEntryType == op->getResultTypes().front() &&
           "The frame type slot does not match the value");
    assert(op->getNumResults() == 1 && "TODO: support multiple results");
    Value dataSlot =
        builder.create<StructGEPOp>(continuation, Frame + entry->getSecond());
    builder.create<StoreOp>(op->getResult(0), dataSlot);
  }
}

void LowerAsyncBuildContext::lowerHotInvoke(Value continuation,
                                            HotInvokeOp hotInvoke,
                                            COTypes &coTypes) {
  builder.setInsertionPoint(hotInvoke);
  auto suspension = builder.create<SuspendOp>();

  auto symbol = cast<SymbolConstantAttr>(hotInvoke.getCallee());
  auto cloneMaybe = symbolToClone.find(symbol);
  SymbolConstantAttr rampSymbol;
  if (cloneMaybe != symbolToClone.end()) {
    AsyncFunctionInfo &asyncFunctionInfo = symbolToClone[symbol];
    auto [ramp, _] = lowerAsyncFunctionHot(asyncFunctionInfo);
    symbolToClone.erase(symbol);
    rampSymbol = SymbolConstantAttr::get(
        SymbolRefAttr::get(builder.getContext(), ramp.getSymName()),
        ramp.getSignature());
    fromOriginalToHotRamp.insert({symbol, rampSymbol});
  } else {
    assert(fromOriginalToHotRamp.contains(symbol) &&
           "If the symbol was not found in the clones then it must be in the "
           "hot ramp map");
    rampSymbol = fromOriginalToHotRamp[symbol];
  }
  Block &body = suspension.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);
  // callback and closure state are arguments 0 and 1. Then add the arguments
  // from hotInvoke.
  SmallVector<Value> operands;
  Value resumeFunction = builder.create<LoadOp>(
      builder.create<StructGEPOp>(continuation, ResumeFunction));
  operands.push_back(builder.create<PointerBitcastOp>(
      coTypes.getResumeSignatureType(), resumeFunction));
  operands.push_back(builder.create<PointerBitcastOp>(
      coTypes.typeForField(M::KGEN::ClosureState), continuation));
  llvm::append_range(operands, hotInvoke->getOperands());
  auto callOp = builder.create<CallOp>(
      PointerType::get(coTypes.getHeaderType()), rampSymbol, operands);
  hotInvoke.replaceAllUsesWith(callOp);
  hotInvoke->erase();
  builder.create<SuspendEndOp>();
}

FuncOp LowerAsyncBuildContext::createHotRamp(FuncOp resumeFunction,
                                             FuncOp funcOp, COTypes &coTypes) {
  // Create Hot Ramp. We initialize it with the callback function and closures
  // state.
  builder.setInsertionPoint(funcOp);
  StringAttr rampName =
      builder.getStringAttr(funcOp.getSymName() + "_hot_ramp");

  // Create empty function.
  auto hotRamp = builder.create<FuncOp>(
      rampName, SignatureType::get(builder.getContext(), {}, {}));
  rampName = sharedTable.modify(
      [hotRamp, it = funcOp->getIterator()](SymbolTable &symtab) {
        return symtab.insert(hotRamp, it);
      });
  auto [callback, closureState] =
      addSlicedFirstStateTo(hotRamp, funcOp, coTypes);
  // Save starting point of sliced state.
  Operation *start = &hotRamp.getBodyRegion().front().front();

  // Introduce continuation.
  FrameData &frameData = coTypes.getFrameData();
  builder.setInsertionPointToStart(&hotRamp.getBodyRegion().front());
  Value continuation =
      initializeContinuation(hotRamp, resumeFunction, funcOp, coTypes, 1);

  // Store continuation and closure.
  Value closureSlot = builder.create<StructGEPOp>(continuation, ClosureState);
  builder.create<StoreOp>(closureState, closureSlot);
  Value callbackSlot = builder.create<StructGEPOp>(continuation, CallbackFn);
  builder.create<StoreOp>(callback, callbackSlot);

  // Store arguments in frame if used across suspension points.
  for (auto [index, image] :
       llvm::enumerate(hotRamp.getRegion().front().getArguments().slice(2))) {
    // Value arg = funcOp.getArgument(index);
    //  If the argument is not in the frame metadata it is unused. Do not
    //  store in frame.
    if (!frameData.valueToIndexInFrame.contains(image))
      continue;
    for (Operation *user : image.getUsers()) {
      int useState = frameData.opToState[user];
      if (useState > 0) {
        unsigned argSlot = frameData.valueToIndexInFrame[image];
        Value slot = builder.create<StructGEPOp>(continuation, Frame + argSlot);
        builder.create<StoreOp>(image, slot);
        break;
      }
    }
  }

  // Store results/error
  auto setByRefArgument = [&](Value argument, unsigned index) {
    Value slot = builder.create<StructGEPOp>(continuation, index);
    Value typedSlot = builder.create<PointerBitcastOp>(
        KGEN::PointerType::get(argument.getType()), slot);
    builder.create<StoreOp>(argument, typedSlot);
  };
  Value errorValue;
  Value resultValue;
  if (funcOp.isThrows()) {
    errorValue = hotRamp.getArgument(hotRamp.getNumArguments() - 2);
    setByRefArgument(errorValue, AsyncContinuationField::ErrorSlot);
  }
  if (funcOp.getSignature().hasMemoryOnlyResult()) {
    resultValue = hotRamp.getArgument(hotRamp.getNumArguments() - 1);
    setByRefArgument(resultValue, AsyncContinuationField::ResultSlot);
  }

  // Insert stores of variables that are used in later states.
  FrameVariables frameVariables(builder, frameData, errorValue, resultValue);
  SmallVector<Operation *> regionsToProcess;
  regionsToProcess.push_back(start);
  SmallVector<Operation *> opsToDelete;
  while (!regionsToProcess.empty()) {
    Operation *current = regionsToProcess.pop_back_val();
    while (current) {
      Operation *op = current;
      current = op->getNextNode();
      storeOpInFrameIfNeeded(frameData, op, continuation, opsToDelete);

      // Replace operands with frame variables if stack allocation.
      int useState = frameData.opToState[op];
      for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
        auto entry = frameData.valueToIndexInFrame.find(operand);
        if (entry == frameData.valueToIndexInFrame.end())
          continue;
        Operation *definingOp = operand.getDefiningOp();
        if (definingOp && isa<StackAllocationOp>(definingOp)) {
          Value image = frameVariables.getFrameValueForOperand(
              continuation, operand, op, useState);
          op->setOperand(index, image);
        }
      }

      // Store arguments of block if needed and terminate unterminated blocks.
      for (Region &region : op->getRegions()) {
        if (region.empty()) {
          Block &block = region.emplaceBlock();
          builder.setInsertionPointToEnd(&block);
          builder.create<UnreachableOp>();
          continue;
        }
        Block &block = region.front();
        if (!block.mightHaveTerminator()) {
          builder.setInsertionPointToEnd(&block);
          builder.create<UnreachableOp>();
        }
        Operation *firstOp = &*region.front().begin();
        regionsToProcess.push_back(firstOp);
        storeBlockArgumentsInFrame(region.front(), firstOp, frameData,
                                   continuation, frameVariables);
      }
    }
  }

  hotRamp.walk([&](Operation *op) {
    if (auto returnOp = dyn_cast<ReturnOp>(op)) {
      ReturnOp newReturnOp = lowerReturn(returnOp, continuation);
      builder.setInsertionPoint(newReturnOp);
      Value headerTypedContinuation = builder.create<PointerBitcastOp>(
          PointerType::get(coTypes.getHeaderType()), continuation);
      builder.create<ReturnOp>(headerTypedContinuation);
      newReturnOp->erase();
    } else if (auto suspend = dyn_cast<SuspendOp>(op)) {
      lowerSuspensionPoint(suspend, continuation, coTypes);
      for (Operation *operation = &suspend.getBody().front().front(); operation;
           operation = operation->getNextNode()) {
        operation->moveBefore(suspend);
        if (isa<SuspendEndOp>(operation)) {
          operation->erase();
          break;
        }
      }
      suspend->erase();
    } else if (auto hotInvoke = dyn_cast<HotInvokeOp>(op)) {
      lowerHotInvoke(continuation, hotInvoke, coTypes);
    }
  });
  for (auto op : opsToDelete)
    op->erase();

  // Check parent region for termination.
  if (!hotRamp.getBodyRegion().front().mightHaveTerminator()) {
    builder.setInsertionPointToEnd(&hotRamp.getBodyRegion().front());
    builder.create<UnreachableOp>();
  }
  return hotRamp;
}

void LowerAsyncBuildContext::populateRampFunction(FuncOp rampFunction,
                                                  FuncOp resumeFunction,
                                                  FuncOp funcOp,
                                                  COTypes &coTypes) {
  builder.setInsertionPointToStart(
      &rampFunction.getBodyRegion().emplaceBlock());
  for (Type argument : rampFunction.getSignature().getArguments())
    rampFunction.getBodyRegion().addArgument(argument, rampFunction.getLoc());
  Value continuation =
      initializeContinuation(rampFunction, resumeFunction, funcOp, coTypes, 0);
  // Store arguments in frame.
  FrameData &frameData = coTypes.getFrameData();
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
      FunctionType::get(cxt, opaquePointerType, results);
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

  // Header type omits the variable sized frame and promise.
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

int FrameData::getDefinitionStateForValue(Value operand, bool isHot) const {
  Operation *definingOp = operand.getDefiningOp();
  // Initialize state to the entry state.
  int defState = isHot ? 0 : -1;
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

void FrameData::updateVirtualBlock(VirtualBlock virtualBlock, int newState) {
  Operation *op = virtualBlock;
  int state = newState;
  while (op) {
    if (isa<SuspendEndOp>(op))
      ++state;
    int &oldState = opToState[op];
    if (oldState < state)
      oldState = state;
    op = op->getNextNode();
    int &nextOldState = opToState[op];
    // respect control flow boundary
    if (op && isa<HLCF::ControlFlowNode>(op)) {
      if (nextOldState < state)
        nextOldState = state;
      break;
    }
  }
}

using PathContainer = SmallVector<Operation *>;
struct PathInfo {
  PathInfo(VirtualBlock v) : virtualBlock(v), state(0) {}
  PathInfo(VirtualBlock v, int state, PathContainer const &parent)
      : virtualBlock(v), path(parent), state(state) {}
  int existsAt(VirtualBlock virtualBlock) const;
  VirtualBlock virtualBlock;
  PathContainer path;
  int state;
};

int PathInfo::existsAt(VirtualBlock virtualBlock) const {
  for (auto [i, node] : llvm::enumerate(path)) {
    if (node == virtualBlock)
      return i;
  }
  return -1;
}

FrameData::FrameData(
    FuncOp originalFunction, mlir::DominanceInfo &domInfo, Value errorValue,
    Value resultValue,
    function_ref<void(FuncOp, DenseMap<Operation *, int> &)> transform,
    bool isHot) {
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
            Operation *predecessor) {
          // For the first op of the region of each target, add the control flow
          // node as a predecessor
          for (HLCF::ControlFlowTarget target : targets) {
            VirtualBlock successor;
            if (target.index.has_value()) {
              Region *succRegion =
                  &controlFlowParent->getRegion(target.index.value());
              successor = &*succRegion->front().begin();
              regions.push_back(succRegion);
            } else {
              successor = controlFlowParent->getNextNode();
            }
            predecessors[successor].push_back(controlFlowNode);
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
      CO::SuspendOp lastAwait = nullptr;
      for (Operation &op : region->front().getOperations()) {
        if (isa<ReturnOp, UnreachableOp>(op))
          continue;

        // add the control flow terminator as a predecessor to the first op of a
        // target block.
        if (auto controlFlowTerminator =
                dyn_cast<HLCF::ControlFlowTerminator>(op)) {
          SmallVector<HLCF::ControlFlowTarget> targets;
          SmallVector<Attribute> controlFlowTerminatorOperands(
              controlFlowTerminator->getNumOperands(), Attribute());
          controlFlowTerminator.getBranchTargets(controlFlowTerminatorOperands,
                                                 targets);
          Operation *predecessor =
              lastControlFlowNode
                  ? lastControlFlowNode->getNextNode()
                  : &*controlFlowTerminator->getParentRegion()->front().begin();
          if (lastAwait) {
            if (domInfo.dominates(predecessor, lastAwait))
              predecessor = lastAwait->getNextNode();
          }
          pushSuccessors(targets, controlFlowTerminator,
                         getParentNode(controlFlowTerminator), predecessor);
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
          lastAwait = suspend;
          predecessors[&*suspend.getBody().front().begin()].push_back(suspend);
          // Terminator is used because that will trigger updated state.
          predecessors[next].push_back(
              suspend.getBody().front().getTerminator());
          regions.push_back(&suspend.getBody());
        }
      }
    }
  }
  // Calculate the state of each op.
  {
    SmallVector<PathInfo> paths;
    VirtualBlock initial = &*originalFunction.getBodyRegion().front().begin();
    paths.push_back({initial});
    auto pushSuccessors =
        [&](SmallVector<HLCF::ControlFlowTarget> const &targets,
            Operation *controlFlowVirtualBlock, Operation *controlFlowParent,
            PathContainer &path, int state) {
          for (HLCF::ControlFlowTarget target : targets) {
            if (target.index.has_value()) {
              auto o = &*controlFlowParent->getRegion(target.index.value())
                             .front()
                             .begin();
              paths.push_back({o, state, path});
            } else {
              paths.push_back({controlFlowParent->getNextNode(), state, path});
            }
          }
        };

    int j = 0;
    while (!paths.empty()) {
      if (j > 10000)
        llvm_unreachable("infinite loop");

      j++;
      VirtualBlock virtualBlock = paths.back().virtualBlock;
      int indexOfMe = paths.back().existsAt(virtualBlock);
      PathContainer path = std::move(paths.back().path);
      int state = paths.back().state;
      paths.pop_back();

      auto recordedStatePtr = opToState.find(virtualBlock);
      if (recordedStatePtr != opToState.end() &&
          state <= recordedStatePtr->getSecond())
        continue;

      for (auto pred : predecessors[virtualBlock]) {
        if (domInfo.dominates(virtualBlock, pred))
          continue;
        auto predPtr = opToState.find(pred);
        if (predPtr != opToState.end()) {
          if (predPtr->getSecond() > state)
            state = predPtr->getSecond();
        }
      }

      // We have reached a cycle. Terminate.
      if (indexOfMe > -1)
        continue;
      path.push_back(virtualBlock);

      // Iterate through each op in this virtual block to register its state.
      // The boundaries of a node are defined by awaits, control
      // flow nodes, and control flow terminators.
      Operation *current = virtualBlock;
      while (current) {
        Operation *op = current;
        current = op->getNextNode();
        if (auto susEnd = dyn_cast<SuspendEndOp>(op)) {
          Operation *next = susEnd->getParentOp()->getNextNode();
          ++state;
          if (next)
            paths.push_back({next, state, path});
          opToState[op] = state;
          break;
        }
        opToState[op] = state;

        if (auto suspend = dyn_cast<CO::SuspendOp>(op)) {
          paths.push_back({&*suspend.getBody().front().begin(), state, path});
          break;
        }
        if (isa<ReturnOp, UnreachableOp>(op))
          continue;
        if (auto controlFlowNode = dyn_cast<HLCF::ControlFlowNode>(op)) {
          SmallVector<HLCF::ControlFlowTarget> targets;
          SmallVector<Attribute> controlFlowNodeOperands(
              controlFlowNode->getNumOperands(), Attribute());
          controlFlowNode.getEntryTargets(controlFlowNodeOperands, targets);
          pushSuccessors(targets, controlFlowNode, controlFlowNode, path,
                         state);
          break;
        }
        if (auto controlFlowTerminator =
                dyn_cast<HLCF::ControlFlowTerminator>(op)) {
          SmallVector<HLCF::ControlFlowTarget> targets;
          SmallVector<Attribute> controlFlowTerminatorOperands(
              controlFlowTerminator->getNumOperands(), Attribute());
          controlFlowTerminator.getBranchTargets(controlFlowTerminatorOperands,
                                                 targets);
          pushSuccessors(targets, controlFlowTerminator,
                         getParentNode(controlFlowTerminator), path, state);
          break;
        }
      }
    }
  }

  auto propagateChildSuspoints = [&]() -> bool {
    bool wasChange = false;
    for (auto [virtualBlock, preds] : predecessors) {
      int initialState = opToState[virtualBlock];
      int postState = initialState;
      int stateAtContinue = 0;
      int smallestPredState = initialState;
      for (Operation *pred : preds) {
        int predState = opToState[pred];
        if (domInfo.dominates(virtualBlock, pred)) {
          if (predState > stateAtContinue && predState > postState)
            stateAtContinue = predState;
          continue;
        }
        if (predState > postState)
          postState = predState;
        if (predState < smallestPredState)
          smallestPredState = predState;
      }
      wasChange = wasChange || (initialState != postState);
      if (initialState != postState)
        updateVirtualBlock(virtualBlock, postState);

      // Insert a new state at the parent because a child with a suspension
      // point branches to it.
      if (stateAtContinue > 0 && smallestPredState == initialState) {
        int newState = initialState + 1;
        for (auto &[_, state] : opToState) {
          if (state >= newState)
            ++state;
        }
        wasChange = true;
        updateVirtualBlock(virtualBlock, newState);
      }
    }
    return wasChange;
  };

  // FIXED POINT:
  // Update until for every path:
  // (1) if A -> B and A dominates B then state(A) >= state(B)
  // (2) if B -> A and A dominates B and state(A) < state(B), then if there is
  // a predecessor C of A unreachable from B then state(C) < state(A)
  // Condition (2) is achieved by
  // inserting a parent state. Note that intuitively this corresponds to a case
  // of a child cycle with a suspension point branching to a parent cycle.
  bool stateChanged = true;
  while (stateChanged)
    stateChanged = propagateChildSuspoints();
  // Remember the virtual ops in the first state for hot start resume
  // generation.
  for (auto &virtualBlockAndList : predecessors) {
    VirtualBlock virtualBlock = virtualBlockAndList.first;
    if (opToState[virtualBlock] == 0)
      virtualBlocksFirstState.push_back(virtualBlock);
  }
  transform(originalFunction, opToState);

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
  SmallVector<Value> coldCoroTypes;
  originalFunction.walk([&](Operation *operation) {
    int useState = opToState[operation];
    if (StackAllocationOp stackAllocationOp =
            dyn_cast<StackAllocationOp>(operation)) {
      if (!stackAllocationOp.getMarkedLifetimes()) {
        Operation *terminator =
            stackAllocationOp->getParentRegion()->front().getTerminator();
        int endState = isa<SuspendEndOp>(terminator) ? opToState[terminator] - 1
                                                     : opToState[terminator];
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
      int defState = getDefinitionStateForValue(operand, isHot);
      if (defState != useState) {
        bool isArgument = false;
        if (auto blockArg = dyn_cast<BlockArgument>(operand))
          isArgument =
              blockArg.getParentBlock()->getParentOp() == originalFunction;

        unsigned index = isArgument ? coldCoroTypes.size() : frameTypes.size();
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
          if (isArgument)
            coldCoroTypes.push_back(operand);
          else
            frameTypes.push_back(operand.getType());
        }
        valueToIndexInFrame.insert({operand, index});
      }
    }
  });

  // Insert used arguments at end of frame.
  unsigned offset = frameTypes.size();
  for (auto [index, functionArg] : llvm::enumerate(coldCoroTypes)) {
    frameTypes.push_back(functionArg.getType());
    unsigned newIndex = offset + index;
    valueToIndexInFrame[functionArg] = newIndex;
  }
}

//===----------------------------------------------------------------------===//
// LowerAsyncFunctionsPass
//===----------------------------------------------------------------------===//

static Operation *findNearestCommonAncestor(mlir::DominanceInfo &domInfo,
                                            Operation *lhs, Operation *rhs) {
  auto findOpInCommonRegion = [](Operation *lhs, Operation *rhs) {
    Region *currentRegion = rhs->getParentRegion();
    while (!lhs->getParentRegion()->isAncestor(currentRegion))
      lhs = lhs->getParentOp();
    return lhs;
  };
  Operation *lhsCommon = findOpInCommonRegion(lhs, rhs);
  Operation *rhsCommon = findOpInCommonRegion(rhs, lhs);
  return domInfo.dominates(lhsCommon, rhsCommon) ? lhsCommon : rhsCommon;
}

static std::pair<Value, Value>
preprocessAsyncFunction(FuncOp funcOp, mlir::DominanceInfo &domInfo) {
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

  // Preprocess the function to move stack allocation ops as close to their
  // first use as possible.
  SmallVector<StackAllocationOp> allocs;
  funcOp.walk([&](StackAllocationOp alloc) { allocs.push_back(alloc); });
  for (StackAllocationOp alloc : allocs) {
    if (alloc->use_empty())
      continue;
    Operation *ancestor = *alloc->user_begin();
    for (Operation *user : llvm::drop_begin(alloc->getUsers())) {
      if (domInfo.dominates(ancestor, user))
        continue;
      if (domInfo.dominates(user, ancestor)) {
        ancestor = user;
        continue;
      }
      // `ancestor` and `user` live in sibling regions. We need to find a
      // common ancestor.
      ancestor = findNearestCommonAncestor(domInfo, ancestor, user);
    }
    alloc->moveBefore(ancestor);
  }
  return {errorValue, memoryResultValue};
}

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
  // Save a clone of the original async function for the purpose of generating a
  // hot ramp/resume. Key the clone off the symbol of the original so we can
  // look it up.
  ImplicitLocOpBuilder b(module->getLoc(), module);
  auto &domInfo = getAnalysis<mlir::DominanceInfo>();
  LowerAsyncBuildContext buildContext(sharedTable, asyncFuncToRampFunctions, b,
                                      domInfo, targetInfo);

  for (auto funcOp : llvm::make_early_inc_range(module.getOps<FuncOp>())) {
    if (!funcOp.isAsync())
      continue;
    // calculate the frame of the original, unmodified resume.
    auto [errorValue, memoryResultValue] =
        preprocessAsyncFunction(funcOp, domInfo);

    // Store a clone of the function so we can generate the hot ramp/resume.
    SymbolRefAttr symbolRefAttr =
        SymbolRefAttr::get(funcOp.getContext(), funcOp.getSymName());
    SymbolConstantAttr key =
        SymbolConstantAttr::get(symbolRefAttr, funcOp.getSignature());
    b.setInsertionPoint(funcOp);
    FuncOp clone = cast<FuncOp>(b.clone(*funcOp));

    // The transform function clones ops whose values should not be stored in
    // the frame. This includes constants and pointer offsets.
    std::function<void(FuncOp, DenseMap<Operation *, int> &)> transform =
        [&](FuncOp originalFunction, DenseMap<Operation *, int> &opToState) {
          cloneFrameArgs(funcOp, b, domInfo, originalFunction, opToState);
        };
    FrameData frameData(funcOp, domInfo, errorValue, memoryResultValue,
                        transform, /*isHot=*/false);
    COTypes cotypes(
        module.getContext(), frameData,
        StructType::get(funcOp.getContext(), funcOp.getResultTypes()));
    auto [ramp, resume] = buildContext.lowerAsyncFunction(
        funcOp, cotypes, errorValue, memoryResultValue);
    buildContext.symbolToClone.insert(
        {key, {clone, resume, std::move(frameData.virtualBlocksFirstState)}});
  }

  // Apply all other CO lowerings.
  mlir::IRRewriter rewriter(b);
  FrameData empty;
  COTypes opaqueCoroutineTypes(module.getContext(), empty, /*promiseType=*/{});
  mlir::AttrTypeReplacer replacer;
  Type headerType = PointerType::get(opaqueCoroutineTypes.getHeaderType());
  replacer.addReplacement([&](CoroutineType type) { return headerType; });
  replacer.recursivelyReplaceElementsIn(module, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/true,
                                        /*replaceTypes=*/true);
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
  // Any remaining clones can be deleted: they were never hot called.
  for (auto [symbol, clone] : buildContext.symbolToClone) {
    clone.originalClone->erase();
  }
}
