//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/Interpreter/InterpreterState.h"
#include "KGEN/Interpreter/ParametricInterpreterState.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/DebugStringHelper.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// ParamApplyOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess ParamApplyOp::interpret(ArrayRef<Attribute> operands,
                                           InterpreterState &state) {
  llvm_unreachable("kgen.param.apply interpret undefined");
}

ErrorTreeOrSuccess
ParamApplyOp::parametric_interpret(ArrayRef<Attribute> operands,
                                   ParametricInterpreterState &state) {
  auto ops = state.getReboundAttribute(getOperandsAttr());
  auto callee = state.getReboundAttribute(getCalleeAttr());
  auto calleeAttr = cast<SymbolConstantAttr>(callee);

  auto operandsAttr = cast<ParameterExprArrayAttr>(ops);
  SmallVector<Attribute> arguments(operandsAttr.size());
  for (auto [idx, attr] : llvm::enumerate(operandsAttr.getValue()))
    arguments[idx] = attr;

  ErrorTreeOr<TypedAttr> result = state.interpretGenerator(
      calleeAttr, calleeAttr.getParamValues(), arguments, getLoc());

  if (result.isError())
    return result.takeError();

  state.appendParamValues({result.getValue()}, 0, this->getOperation());
  bool overwrite =
      state.overwriteDeclBinding(getParamDecl(), result.getValue());
  if (overwrite)
    state.clearParameterCache();

  return success();
}

template <typename Payload>
static ErrorOrSuccess populateContainsPtrPayload(Attribute value,
                                                 Payload &payload) {
  mlir::AttrTypeWalker walker;
  walker.addWalk(
      [](MemRefAttr memref) -> WalkResult { return WalkResult::interrupt(); });
  payload.value = value;
  payload.containsPtr = walker.walk(value).wasInterrupted();
  return success();
}

template <typename Payload>
ErrorOrSuccess interpretIfContainsPtr(const Payload &payload,
                                      InterpreterState &state) {
  if (payload.containsPtr) {
    SmallVector<Attribute> attributes;
    attributes.push_back(payload.value);
    if (ErrorOrSuccess err = state.internalizeMemory(attributes); err.isError())
      return err.takeError();
    state.mapResults(attributes.front());
  } else {
    state.mapResults(payload.value);
  }
  return success();
}

template <typename Payload>
ErrorOrSuccess
parametricInterpretIfContainsPtr(const Payload &payload,
                                 ParametricInterpreterState &state) {
  Attribute attr = state.getReboundAttribute(payload.value);
  if (payload.containsPtr) {
    SmallVector<Attribute> attributes;
    attributes.push_back(payload.value);
    if (ErrorOrSuccess err = state.internalizeMemory(attributes); err.isError())
      return err.takeError();
    state.mapResults(attributes.front());
  } else {
    state.mapResults(attr);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ParamConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ParamConstantOp::fold(FoldAdaptor adaptor) {
  return getValueAttr();
}

ErrorOrSuccess ParamConstantOp::compile(Payload &payload,
                                        TargetInfoAttr target) {
  return populateContainsPtrPayload(getValue(), payload);
}

ErrorOrSuccess
ParamConstantOp::parametric_compile(Payload &payload, TargetInfoAttr target,
                                    ArrayRef<Attribute> operands,
                                    ParametricInterpreterState &state) {
  return populateContainsPtrPayload(state.getReboundAttribute(getValue()),
                                    payload);
}

ErrorTreeOrSuccess ParamConstantOp::interpret(ArrayRef<Attribute> operands,
                                              const Payload &payload,
                                              InterpreterState &state) {
  if (ErrorOrSuccess err = interpretIfContainsPtr(payload, state);
      err.isError())
    return ErrorTree(getLoc(), err.takeError());
  return success();
}

ErrorTreeOrSuccess
ParamConstantOp::parametric_interpret(ArrayRef<Attribute> operands,
                                      const Payload &payload,
                                      ParametricInterpreterState &state) {
  if (ErrorOrSuccess err = parametricInterpretIfContainsPtr(payload, state);
      err.isError())
    return ErrorTree(getLoc(), err.takeError());
  return success();
}

//===----------------------------------------------------------------------===//
// ParamDeclareOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess ParamDeclareOp::interpret(ArrayRef<Attribute> operands,
                                             InterpreterState &state) {
  llvm_unreachable("kgen.param.declare interpret undefined");
}

ErrorTreeOrSuccess
ParamDeclareOp::parametric_interpret(ArrayRef<Attribute> operands,
                                     ParametricInterpreterState &state) {
  TypedAttr value = state.getReboundAttribute(getValue());
  if (!value)
    return ErrorTree(getLoc(), "cannot interpret kgen.param.declare");

  state.appendParamValues({value}, 1, this->getOperation());
  state.overwriteDeclBinding(getParamDecl(), value);
  return success();
}

//===----------------------------------------------------------------------===//
// ParamMaterializeOp
//===----------------------------------------------------------------------===//

LogicalResult ParamMaterializeOp::canonicalize(ParamMaterializeOp op,
                                               PatternRewriter &rewriter) {
  // Decay to a constant if the parameter value is a constant value with no
  // memory references.
  if (!ParameterAttr::isSimpleConstant(op.getValue()))
    return rewriter.notifyMatchFailure(op, "value is not a simple constant");

  mlir::AttrTypeWalker walker;
  walker.addWalk([&](MemRefAttr ref) {
    for (MemoryBlobAttr blob : ref.getModel().getMemory())
      if (blob.getKind() != MemoryKind::ConstGlobal)
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (walker.walk(op.getValue()).wasInterrupted())
    return rewriter.notifyMatchFailure(op, "value has memory references");

  rewriter.replaceOpWithNewOp<ParamConstantOp>(op, op.getValue());
  return success();
}

ErrorOrSuccess ParamMaterializeOp::compile(Payload &payload,
                                           TargetInfoAttr target) {
  return populateContainsPtrPayload(getValue(), payload);
}

ErrorOrSuccess
ParamMaterializeOp::parametric_compile(Payload &payload, TargetInfoAttr target,
                                       ArrayRef<Attribute> operands,
                                       ParametricInterpreterState &state) {
  auto value = state.getReboundAttribute(getValue());
  return populateContainsPtrPayload(value, payload);
}

ErrorTreeOrSuccess ParamMaterializeOp::interpret(ArrayRef<Attribute> operands,
                                                 const Payload &payload,
                                                 InterpreterState &state) {
  if (ErrorOrSuccess err = interpretIfContainsPtr(payload, state);
      err.isError())
    return ErrorTree(getLoc(), err.takeError());
  return success();
}

ErrorTreeOrSuccess
ParamMaterializeOp::parametric_interpret(ArrayRef<Attribute> operands,
                                         const Payload &payload,
                                         ParametricInterpreterState &state) {

  if (ErrorOrSuccess err = parametricInterpretIfContainsPtr(payload, state);
      err.isError())
    return ErrorTree(getLoc(), err.takeError());
  return success();
}

//===----------------------------------------------------------------------===//
// RebindOp
//===----------------------------------------------------------------------===//
/// Fold away the rebind if the input and output types are the same.
OpFoldResult RebindOp::fold(FoldAdaptor adaptor) {
  if (getInput().getType() == getType()) {
    if (Attribute input = adaptor.getInput())
      return input;
    return getInput();
  }

  // If the input is a rebindop(x) from some other type then change this op to
  // rebind "x" instead of the result of rebind "x".  Even if the types differ,
  // they will all need to elaborate to the same type, so we might as well
  // simplify ourselves.
  bool foldedRebind = false;
  while (auto srcRebind = getInput().getDefiningOp<RebindOp>()) {
    setOperand(srcRebind.getInput());
    foldedRebind = true;
  }
  if (foldedRebind)
    return getResult();

  return {};
}

ErrorTreeOrSuccess RebindOp::interpret(ArrayRef<Attribute> operands,
                                       InterpreterState &state) {
  return state.interpretOpWithFolder(this->getOperation(), operands);
}

ErrorTreeOrSuccess
RebindOp::parametric_interpret(ArrayRef<Attribute> operands,
                               ParametricInterpreterState &state) {
  Type type = state.getReboundType(getType());
  if (TypedAttr attr = llvm::dyn_cast_if_present<TypedAttr>(operands.front())) {
    if (attr.getType() == type) {
      state.mapResults(attr);
      return success();
    }
  }

  return ErrorTree(getLoc(), "type mismatch");
}

//===----------------------------------------------------------------------===//
// ParamAssertOp
//===----------------------------------------------------------------------===//

LogicalResult ParamAssertOp::canonicalize(ParamAssertOp op,
                                          PatternRewriter &rewriter) {
  // If the condition is statically true then we can just remove this op.
  auto cond = op.getCond();
  if (auto intCond = dyn_cast<IntegerAttr>(cond)) {
    // Leave failing conditions, they must be diagnosed at elaboration time.
    if (intCond.getValue().isZero())
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
  return failure();
}

ErrorTreeOrSuccess ParamAssertOp::interpret(ArrayRef<Attribute> operands,
                                            InterpreterState &state) {
  llvm_unreachable("kgen.param.assert interpret undefined");
}

ErrorTreeOrSuccess
ParamAssertOp::parametric_interpret(ArrayRef<Attribute> operands,
                                    ParametricInterpreterState &state) {
  Attribute cond = state.getFailableReboundAttribute(getCond());
  if (!cond)
    return ErrorTree(getLoc(), "evaluate kgen.param.assert's cond failed");

  auto resultInt = cast<IntegerAttr>(cond);
  if (resultInt.getValue().isZero()) {
    Attribute message = state.getReboundAttribute(getMessage());
    return ErrorTree(getLoc(), "constraint failed: " +
                                   cast<StringAttr>(message).getValue());
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ParamForOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess ParamForOp::interpret(ArrayRef<Attribute> operands,
                                         InterpreterState &state) {
  llvm_unreachable("kgen.param.for interpret undefined");
}

ErrorTreeOrSuccess
ParamForOp::parametric_interpret(ArrayRef<Attribute> operands,
                                 ParametricInterpreterState &state) {
  SmallVector<Type> resultTypes;
  Attribute hasNext = state.getReboundAttribute(getHasNext());
  Attribute getNext = state.getReboundAttribute(getGetNextIter());
  for (Type type : getResultTypes()) {
    resultTypes.push_back(state.getReboundType(type));
  }

  auto hasNextCall = cast<SymbolConstantAttr>(hasNext);
  auto iter = state.currOpSideEffectState().find(this->getOperation());
  bool firstIteration =
      (iter == state.currOpSideEffectState().end() || !iter->second.iterator);

  // Can probably cache this for each iteration.
  SmallVector<TypedAttr> paramValues;
  for (auto pv : hasNextCall.getParamValues()) {
    paramValues.push_back(state.getReboundAttribute(pv));
  }

  ErrorOr<Type> hasNextTypeResult =
      state.lookupFuncTypeGenerator(hasNextCall.getSymbol());
  if (hasNextTypeResult.isError()) {
    return ErrorTree(getLoc(), hasNextTypeResult.takeError());
  }

  FuncType hasNextType =
      cast<FuncTypeGeneratorType>(*hasNextTypeResult).getBody();

  // Push an empty slot to paramValues count to mark this is the boundary
  // of a ParamFor so that we know how much to pop once hitting
  // kgen.param.for.break or kgen.param.for.continue
  // state.pushParamValues({}, false, this->getOperation());
  Attribute initial = state.getReboundAttribute(getInitial());
  TypedAttr iterator =
      cast<TypedAttr>(firstIteration ? initial : iter->second.iterator);

  TypedAttr hasNextInput = iterator;
  if (hasAddress(hasNextType.getArgConvention(0)))
    hasNextInput = StoreToMemAttr::get(iterator, hasNextType.getArguments()[0]);

  ErrorTreeOr<TypedAttr> hasNextResult =
      state.interpretGenerator(hasNextCall, paramValues, iterator, getLoc());
  if (hasNextResult.isError()) {
    return hasNextResult.takeError();
  }

  if (!cast<BoolAttr>(*hasNextResult).getValue()) {
    // Go to else region
    ArrayRef<Attribute> arguments =
        firstIteration ? operands : iter->second.operands;
    state.currOpSideEffectState().erase(this->getOperation());
    state.transferControlFlowTo(this->getOperation(), arguments);

  } else {
    state.overwriteDeclBinding(getParamDecl(), iterator);
    iterator =
        StoreToMemAttr::get(iterator, PointerType::get(iterator.getType()));

    auto getNextCall = cast<SymbolConstantAttr>(getNext);
    paramValues.clear();
    for (auto pv : getNextCall.getParamValues()) {
      paramValues.push_back(state.getReboundAttribute(pv));
    }

    ErrorTreeOr<TypedAttr> getNextResult =
        state.interpretGeneratorWithResultSlot(getNextCall, paramValues,
                                               iterator, getLoc());
    if (getNextResult.isError())
      return getNextResult.takeError();

    if (firstIteration) {
      state.currOpSideEffectState()[this->getOperation()] = {
          {}, {}, *getNextResult};
    } else {
      // Clear up iterator in case function returns in the body of the ParamFor
      // so that the iterator value doesn't carry over to another round of
      // interpreting this ParamFor by mistake.
      iter->second.iterator = {};
      // Set nextIterator value so that kgen.param.for.continue can set the
      // iterator value correctly for the next iteration.
      iter->second.nextIterator = *getNextResult;
    }

    ArrayRef<Attribute> arguments =
        firstIteration ? operands : iter->second.operands;

    state.pushParamValues({iterator}, false, this->getOperation());
    state.pushEvalFrame(getOperation(), &getBody(), {}, 5);
    state.transferControlFlowTo(getBody(), arguments);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ParamForBreakOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess ParamForBreakOp::interpret(ArrayRef<Attribute> operands,
                                              InterpreterState &state) {
  llvm_unreachable("kgen.param.for.break interpret undefined");
}

ErrorTreeOrSuccess
ParamForBreakOp::parametric_interpret(ArrayRef<Attribute> operands,
                                      ParametricInterpreterState &state) {
  auto parent = this->getOperation()->getParentOfType<ParamForOp>();
  state.popEvalFrame();
  state.popParamValues(false, this->getOperation(), parent);
  state.transferControlFlowTo(parent, operands);
  return success();
}

//===----------------------------------------------------------------------===//
// ParamForContinueOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess ParamForContinueOp::interpret(ArrayRef<Attribute> operands,
                                                 InterpreterState &state) {
  llvm_unreachable("kgen.param.for.continue interpret undefined");
}

ErrorTreeOrSuccess
ParamForContinueOp::parametric_interpret(ArrayRef<Attribute> operands,
                                         ParametricInterpreterState &state) {
  if (auto parent = this->getOperation()->getParentOfType<KGEN::ParamForOp>()) {
    state.popEvalFrame();
    state.popParamValues(false, this->getOperation(), parent);
    state.transferControlFlowToParent(parent, operands);
    auto iter = state.currOpSideEffectState().find(parent.getOperation());
    assert(iter != state.currOpSideEffectState().end() &&
           "kgen.param.for.continue has broken state");
    iter->second.operands = SmallVector<Attribute>(operands);
    iter->second.iterator = iter->second.nextIterator;
    return success();
  }
  return ErrorTree(getLoc(), "INTERNAL ERROR: cannot find parent ParamForOp");
}

//===----------------------------------------------------------------------===//
// ParamIfOp
//===----------------------------------------------------------------------===//

LogicalResult ParamIfOp::canonicalize(ParamIfOp op, PatternRewriter &b) {
  Block &ifBranch = op->getRegion(0).front();
  Block &elseBranch = op->getRegion(1).front();
  Operation *ifTerm = ifBranch.getTerminator();
  Operation *elseTerm = elseBranch.getTerminator();

  // Simple patterns to handle the case of branches containing just terminator
  // ops.
  if (ifTerm == &ifBranch.front() && elseTerm == &elseBranch.front() &&
      op->getNumResults() == 0) {
    // If both sides are yielding, we can delete the op.
    if (isa<ParamYieldOp>(ifTerm) && isa<ParamYieldOp>(elseTerm)) {
      b.eraseOp(op);
      return success();
    }

    // If one branch yields and another breaks we can delete the op if the op is
    // immediately preceding another break. The terminators can't have any
    // returns.
    if (ifTerm->getNumOperands() == 0 && elseTerm->getNumOperands() == 0 &&
        isa<ParamYieldOp, HLCF::BreakOp>(ifTerm) &&
        isa<ParamYieldOp, HLCF::BreakOp>(elseTerm) &&
        isa<HLCF::BreakOp>(op->getNextNode())) {
      b.eraseOp(op);
      return success();
    }
  }

  auto condAttr = dyn_cast<BoolAttr>(op.getCond());
  if (!condAttr)
    return b.notifyMatchFailure(op.getLoc(), "condition is not a constant");

  // We can't fold away the op entirely, because it defines a parameter scope
  // and this could create param decl conflicts. Instead, purge the dead region
  // and insert a `kgen.unreachable`.
  Block &deadBlock = op->getRegion(condAttr.getValue()).front();

  // Don't match again if the dead block is already purged.
  if (isa<UnreachableOp>(deadBlock.front()))
    return b.notifyMatchFailure(op.getLoc(), "dead block already purged");

  // Hoist all the non parameter defining ops out of the live region.
  Block &liveBlock = op->getRegion(!condAttr.getValue()).front();
  while (!liveBlock.front().hasTrait<OpTrait::IsTerminator>()) {
    // Stop if we hit an operation defining a parameter. We don't hoist these as
    // the parameter regions could conflict.
    if (auto paramOp = dyn_cast<ParamOpInterface>(liveBlock.front())) {
      bool hasParam = false;
      paramOp.walkDeclarations([&](ParamDeclAttr attr) { hasParam = true; });
      if (hasParam)
        break;
    }

    // Otherwise, hoist the operation above the 'if'.
    b.moveOpBefore(&liveBlock.front(), op);
  }

  // If we got down to a terminator that we can handle, eliminate the 'if'.
  Operation &liveFront = liveBlock.front();
  // If the live block is now trivial, we can remove the whole
  // operation. Replace the results with the operands to the yield.
  if (auto yield = dyn_cast<ParamYieldOp>(liveFront)) {
    b.replaceOp(op, yield.getOperands());
    return success();
  }

  // If we are ending control flow we can hoist it out but we have to delete
  // all following ops to retain legality.
  if (isa<KGEN::UnreachableOp, HLCF::BreakOp, HLCF::ContinueOp>(liveFront)) {
    Block *block = op->getBlock();
    // Delete things bottom-up so we delete uses before defs.
    while (&block->back() != op)
      b.eraseOp(&block->back());
    // Move the terminator out of the 'if' and remove the 'if'.
    b.moveOpBefore(&liveFront, op);
    b.eraseOp(op);
    return success();
  }

  // Otherwise, we have a parameter defining op (which we need the scope for)
  // or control flow we don't know about.
  for (Operation &subOp : llvm::make_early_inc_range(llvm::reverse(deadBlock)))
    b.eraseOp(&subOp);
  b.setInsertionPointToStart(&deadBlock);
  UnreachableOp::create(b, op.getLoc());
  return success();
}

ErrorTreeOrSuccess ParamIfOp::interpret(ArrayRef<Attribute> operands,
                                        InterpreterState &state) {
  llvm_unreachable("kgen.param.if interpret undefined");
}

ErrorTreeOrSuccess
ParamIfOp::parametric_interpret(ArrayRef<Attribute> operands,
                                ParametricInterpreterState &state) {
  Attribute cond = state.getReboundAttribute(getCond());
  unsigned regionId = 2;
  if (auto result = dyn_cast<BoolAttr>(cond)) {
    regionId = result.getValue() == 0 ? 1 : 0;
  } else if (auto result = dyn_cast<IntegerAttr>(cond)) {
    regionId = result.getValue() == 0 ? 1 : 0;
  }

  if (regionId < 2) {
    Region &target = getRegion(regionId);
    state.pushParamValues({}, false);
    state.pushEvalFrame(getOperation(), &target, {}, 6);
    state.transferControlFlowTo(target, {});
    return success();
  }

  return ErrorTree(getLoc(), "wrong param if condition");
}

//===----------------------------------------------------------------------===//
// ParamYieldOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess ParamYieldOp::interpret(ArrayRef<Attribute> operands,
                                           InterpreterState &state) {
  llvm_unreachable("kgen.param.yield interpret undefined");
}

ErrorTreeOrSuccess
ParamYieldOp::parametric_interpret(ArrayRef<Attribute> operands,
                                   ParametricInterpreterState &state) {
  state.popEvalFrame();
  state.popParamValues(false, this->getOperation());
  state.transferControlFlowTo((*this)->getParentOp(), operands);
  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess CallOp::interpret(ArrayRef<Attribute> operands,
                                     InterpreterState &state) {
  auto bodyOr = state.lookupFunctionBody(getCalleeSymbol());
  if (bodyOr.isError())
    return ErrorTree(getLoc(), bodyOr.takeError());
  Region &body = **bodyOr;

  if (auto err = state.callFunctionBody(body, operands))
    return err.takeError();
  return success();
}

ErrorTreeOrSuccess
CallOp::parametric_interpret(ArrayRef<Attribute> operands,
                             ParametricInterpreterState &state) {
  auto callee =
      dyn_cast<SymbolConstantAttr>(state.getReboundAttribute(getCallee()));

  if (!callee)
    return ErrorTree(getLoc(), "cannot find callee");

  auto bodyOr = state.lookupParametricFunctionBody(callee.getSymbol());
  if (bodyOr.isError())
    return ErrorTree(getLoc(), bodyOr.takeError());
  Region &body = (*bodyOr->first);

  state.pushParamValues(callee.getParamValues(), true);
  state.pushEvalFrame(bodyOr->second, bodyOr->first, callee.getParamValues(),
                      7);
  if (auto err = state.callFunctionBody(body, operands))
    return err.takeError();

  state.setDeclBindings(bodyOr->second, callee.getParamValues());

  return success();
}

//===----------------------------------------------------------------------===//
// CallParamOp
//===----------------------------------------------------------------------===//

LogicalResult CallParamOp::canonicalize(CallParamOp op,
                                        PatternRewriter &rewriter) {
  // If the condition is a known symbol, then replace this with a kgen.call.
  auto callee = dyn_cast<SymbolConstantAttr>(op.getCallee());
  if (!callee)
    return failure();

  rewriter.replaceOpWithNewOp<CallOp>(op, op.getResultTypes(), callee,
                                      op.getOperands());
  return success();
}

ErrorTreeOrSuccess CallParamOp::interpret(ArrayRef<Attribute> operands,
                                          InterpreterState &state) {
  llvm_unreachable("kgen.call_param interpret undefined");
}

ErrorTreeOrSuccess
CallParamOp::parametric_interpret(ArrayRef<Attribute> operands,
                                  ParametricInterpreterState &state) {
  auto callee =
      dyn_cast<SymbolConstantAttr>(state.getReboundAttribute(getCallee()));
  if (!callee)
    return ErrorTree(getLoc(), "cannot find callee");

  auto bodyOr = state.lookupParametricFunctionBody(callee.getSymbol());

  if (bodyOr.isError())
    return ErrorTree(getLoc(), bodyOr.takeError());
  Region &body = *bodyOr->first;

  state.pushParamValues(callee.getParamValues(), true);
  state.pushEvalFrame(bodyOr->second, bodyOr->first, callee.getParamValues(),
                      8);
  if (auto err = state.callFunctionBody(body, operands))
    return err.takeError();

  state.setDeclBindings(bodyOr->second, callee.getParamValues());
  return success();
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

LogicalResult CallIndirectOp::canonicalize(CallIndirectOp op,
                                           PatternRewriter &b) {
  auto create = op.getCallee().getDefiningOp<CreateClosureOp>();
  if (!create)
    return b.notifyMatchFailure(op.getLoc(), "callee op is not create closure");
  // Replace this with a direct call.
  SmallVector<Value> args = llvm::to_vector(create.getCaptures());
  llvm::append_range(args, op.getArguments());
  b.replaceOpWithNewOp<CallParamOp>(op, op.getResultTypes(), create.getCallee(),
                                    args);
  return success();
}

/// CallIndirectOp cannot conform to CallOpInterface, but is very similar since
/// we know the callee at elaboration time.
ErrorTreeOrSuccess CallIndirectOp::interpret(ArrayRef<Attribute> operands,
                                             InterpreterState &state) {
  auto callee = dyn_cast<SymbolConstantAttr>(operands[0]);
  if (!callee)
    return ErrorTree(getLoc(), "couldn't resolve kgen.call_indirect callee");

  auto bodyOr = state.lookupFunctionBody(callee.getSymbol());
  if (bodyOr.isError())
    return ErrorTree(getLoc(), bodyOr.takeError());

  Region &body = **bodyOr;
  if (auto err = state.callFunctionBody(body, operands.drop_front()))
    return err.takeError();
  return success();
}

ErrorTreeOrSuccess
CallIndirectOp::parametric_interpret(ArrayRef<Attribute> operands,
                                     ParametricInterpreterState &state) {
  auto callee = dyn_cast<SymbolConstantAttr>(operands[0]);
  if (!callee)
    return ErrorTree(getLoc(), "couldn't resolve kgen.call_indirect callee");

  auto calleeAttr = cast<SymbolConstantAttr>(state.getReboundAttribute(callee));

  auto bodyOr = state.lookupParametricFunctionBody(calleeAttr.getSymbol());
  // calleeAttr.getParamValues());
  if (bodyOr.isError())
    return ErrorTree(getLoc(), bodyOr.takeError());

  Region &body = *bodyOr->first;
  state.pushParamValues(callee.getParamValues(), true);
  state.pushEvalFrame(bodyOr->second, bodyOr->first, callee.getParamValues(),
                      9);
  if (auto err = state.callFunctionBody(body, operands.drop_front()))
    return err.takeError();

  state.setDeclBindings(bodyOr->second, callee.getParamValues());
  return success();
}

//===----------------------------------------------------------------------===//
// CreateClosureOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess CreateClosureOp::interpret(ArrayRef<Attribute> operands,
                                              InterpreterState &state) {
  // We have no representation for closing over runtime values.
  if (!operands.empty())
    return ErrorTree(getLoc(), "TODO: cannot form a closure at compile time");

  state.mapResults(getCallee());
  return success();
}

ErrorTreeOrSuccess
CreateClosureOp::parametric_interpret(ArrayRef<Attribute> operands,
                                      ParametricInterpreterState &state) {
  // We have no representation for closing over runtime values.
  if (!operands.empty())
    return ErrorTree(getLoc(), "TODO: cannot form a closure at compile time");

  state.mapResults(state.getReboundAttribute(getCallee()));
  return success();
}

//===----------------------------------------------------------------------===//
// CostOfOp
//===----------------------------------------------------------------------===//

/// Compute cost of the given function.
static ErrorTreeOrSuccess computeCost(SymbolConstantAttr func, Location loc,
                                      InterpreterState &state, int64_t &loads,
                                      int64_t &stores,
                                      MutableArrayRef<int64_t> compute,
                                      size_t depth) {
  ErrorOr<Region *> body = state.lookupFunctionBody(func.getSymbol());

  if (body.isError())
    return ErrorTree(loc, body.takeError());

  // Count the number of ops in the body, including parents of regions.
  ErrorTreeOrSuccess walkOutcome;

  body.get()->walk([&](Operation *op) -> WalkResult {
    // Don't count constants, terminators, and debug ops.
    if (op->hasTrait<OpTrait::ConstantLike>() ||
        op->hasTrait<OpTrait::IsTerminator>() ||
        llvm::isa_and_present<DebugInfo::DebugInfoDialect>(op->getDialect()))
      return WalkResult::advance();

    // Compute the cost of the function call descending into the function
    // upto 'maxDepth'. Currently, 'maxDepth' is set to 2, which is sufficient
    // to count pop-level operations for exponentiation.
    constexpr size_t maxDepth = 2;
    if (auto call = dyn_cast<CallOp>(op)) {
      if (depth < maxDepth) {
        auto result = computeCost(call.getCallee(), call.getLoc(), state, loads,
                                  stores, compute, depth + 1);
        if (result.isError()) {
          walkOutcome = result.takeError();
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
    }

    // Count memory operations.
    if (auto memOp = dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
      if (memOp.hasEffect<mlir::MemoryEffects::Read>()) {
        ++loads;
        return WalkResult::advance();
      }
      if (memOp.hasEffect<mlir::MemoryEffects::Write>()) {
        ++stores;
        return WalkResult::advance();
      }
    }

    // Count compute operations.
    ComputeKind kind = ComputeKind::Other;
    if (auto computeOp = dyn_cast<ComputeOpInterface>(op))
      kind = computeOp.getComputeKind();

    ++(compute[static_cast<int>(kind)]);

    return WalkResult::advance();
  });

  return walkOutcome;
}

ErrorTreeOrSuccess CostOfOp::interpret(ArrayRef<Attribute> operands,
                                       InterpreterState &state) {
  int64_t loads = 0, stores = 0;
  std::array<int64_t, getMaxEnumValForComputeKind() + 1> compute{};
  auto callee = dyn_cast<SymbolConstantAttr>(getCallee());
  if (!callee)
    return ErrorTree(getLoc(), "callee is not concrete");

  ErrorTreeOrSuccess result =
      computeCost(callee, getLoc(), state, loads, stores, compute, /*depth=*/0);
  if (result.isError())
    return result;

  Builder builder(getContext());
  auto getComputeOpsAttr = [&builder, &compute](ComputeKind kind) {
    return builder.getIndexAttr(compute[static_cast<int>(kind)]);
  };

  state.mapResults({builder.getIndexAttr(loads), builder.getIndexAttr(stores),
                    getComputeOpsAttr(ComputeKind::Addition),
                    getComputeOpsAttr(ComputeKind::Comparison),
                    getComputeOpsAttr(ComputeKind::Division),
                    getComputeOpsAttr(ComputeKind::Multiplication),
                    getComputeOpsAttr(ComputeKind::MultiplyAdd),
                    getComputeOpsAttr(ComputeKind::Other)});
  return success();
}

ErrorTreeOrSuccess
CostOfOp::parametric_interpret(ArrayRef<Attribute> operands,
                               ParametricInterpreterState &state) {
  int64_t loads = 0, stores = 0;
  std::array<int64_t, getMaxEnumValForComputeKind() + 1> compute{};
  auto callee =
      dyn_cast<SymbolConstantAttr>(state.getReboundAttribute(getCallee()));

  if (!callee)
    return ErrorTree(getLoc(), "callee is not found");

  ErrorTreeOrSuccess result =
      computeCost(callee, getLoc(), state, loads, stores, compute, /*depth=*/0);
  if (result.isError())
    return result;

  Builder builder(getContext());
  auto getComputeOpsAttr = [&builder, &compute](ComputeKind kind) {
    return builder.getIndexAttr(compute[static_cast<int>(kind)]);
  };

  state.mapResults({builder.getIndexAttr(loads), builder.getIndexAttr(stores),
                    getComputeOpsAttr(ComputeKind::Addition),
                    getComputeOpsAttr(ComputeKind::Comparison),
                    getComputeOpsAttr(ComputeKind::Division),
                    getComputeOpsAttr(ComputeKind::Multiplication),
                    getComputeOpsAttr(ComputeKind::MultiplyAdd),
                    getComputeOpsAttr(ComputeKind::Other)});
  return success();
}

//===----------------------------------------------------------------------===//
// IsCompileTimeOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess IsCompileTimeOp::interpret(ArrayRef<Attribute> operands,
                                              InterpreterState &state) {
  // Always return true during interpreting time.
  Builder builder(getContext());
  state.mapResults(builder.getBoolAttr(true));
  return success();
}

ErrorTreeOrSuccess
IsCompileTimeOp::parametric_interpret(ArrayRef<Attribute> operands,
                                      ParametricInterpreterState &state) {
  // Always return true during interpreting time.
  Builder builder(getContext());
  state.mapResults(builder.getBoolAttr(true));
  return success();
}

//===----------------------------------------------------------------------===//
// SourceLocOp
//===----------------------------------------------------------------------===//

/// Resolve negative inlineCounts by inspecting location.
/// Non-negative inlineCounts can be optionally handled by providing `state`.
/// On success, pushes the three result attributes into the result vector.
template <typename ResultList>
static LogicalResult sourceLocOpHelper(int64_t inlineCount, MLIRContext *ctx,
                                       Location loc, InterpreterState *state,
                                       ResultList &results) {
  LocationAttr targetLocation;
  StringRef errorLocMsg;
  if (inlineCount >= 0) {
    if (!state)
      return failure();

    // Need to fetch upwards in the call stack. Requires `state`.
    // Note that "0" inline count means 1 level up.
    Operation *ancestorCallOp = state->getOrigin(inlineCount);
    if (ancestorCallOp)
      targetLocation = ancestorCallOp->getLoc();
    else
      errorLocMsg = "<unknown location in parameter context>";
  } else {
    // Need to fetch downwards in the inlined call stack. Inspect the location's
    // callsite history. Since "0" inlineCount means 1 level up, -1 inlineCount
    // means 0 levels down (i.e. outermost caller loc).
    int64_t remaining = -inlineCount;
    DebugInfo::walkLocation(loc, DebugInfo::LocWalkPolicy::CallerPriority,
                            [&](Location loc) -> WalkResult {
                              if (isa<mlir::CallSiteLoc>(loc))
                                return WalkResult::advance();
                              if (!--remaining) {
                                // If after decrementing, we get to 0, this is
                                // the location to stop at.
                                targetLocation = loc;
                                return WalkResult::interrupt();
                              }
                              return WalkResult::skip();
                            });
    if (!targetLocation)
      errorLocMsg = "<unknown inlined location>";
  }

  OpBuilder b(ctx);
  auto strType = b.getType<StringType>();
  if (!targetLocation) {
    auto zero = b.getIndexAttr(0);
    results.insert(results.begin(),
                   {zero, zero, StringAttr::get(errorLocMsg, strType)});
    return success();
  }

  FileLineColLoc fileLoc = DebugInfo::extractSourceLoc(targetLocation);
  results.insert(results.begin(),
                 {b.getIndexAttr(fileLoc.getLine()),
                  b.getIndexAttr(fileLoc.getColumn()),
                  StringAttr::get(fileLoc.getFilename().getValue(), strType)});
  return success();
}

LogicalResult SourceLocOp::fold(FoldAdaptor adaptor,
                                SmallVectorImpl<OpFoldResult> &results) {
  auto inlineCountIntAttr = dyn_cast<IntegerAttr>(getInlineCount());
  if (!inlineCountIntAttr)
    return failure();

  return sourceLocOpHelper(inlineCountIntAttr.getInt(), getContext(), getLoc(),
                           nullptr, results);
}

ErrorTreeOrSuccess SourceLocOp::interpret(ArrayRef<Attribute> operands,
                                          InterpreterState &state) {
  // The inline count must be an immediate at interpretation time.
  auto inlineCountIntAttr = dyn_cast<IntegerAttr>(getInlineCount());
  if (!inlineCountIntAttr)
    return ErrorTree(getLoc(), Error("inlineCount must be an "
                                     "integer immediate"));

  SmallVector<Attribute> results;
  (void)sourceLocOpHelper(inlineCountIntAttr.getInt(), getContext(), getLoc(),
                          &state, results);
  state.mapResults(results);
  return success();
}

ErrorTreeOrSuccess
SourceLocOp::parametric_interpret(ArrayRef<Attribute> operands,
                                  ParametricInterpreterState &state) {
  // The inline count must be an immediate at interpretation time.
  Attribute inlineCount = state.getReboundAttribute(getInlineCount());
  auto inlineCountIntAttr = dyn_cast<IntegerAttr>(inlineCount);
  if (!inlineCountIntAttr)
    return ErrorTree(getLoc(), Error("inlineCount must be an "
                                     "integer immediate"));

  SmallVector<Attribute> results;
  (void)sourceLocOpHelper(inlineCountIntAttr.getInt(), getContext(), getLoc(),
                          &state, results);
  state.mapResults(results);
  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess ReturnOp::interpret(ArrayRef<Attribute> operands,
                                       InterpreterState &state) {
  state.returnFromFunction(operands);
  return success();
}

ErrorTreeOrSuccess
ReturnOp::parametric_interpret(ArrayRef<Attribute> operands,
                               ParametricInterpreterState &state) {
  state.returnFromFunction(operands);
  return success();
}

//===----------------------------------------------------------------------===//
// PackCreateOp
//===----------------------------------------------------------------------===//

OpFoldResult PackCreateOp::fold(FoldAdaptor adaptor) {
  SmallVector<TypedAttr> values;
  values.reserve(adaptor.getOperands().size());
  for (Attribute operand : adaptor.getOperands()) {
    auto value = llvm::cast_if_present<TypedAttr>(operand);
    if (!value)
      return {};
    values.push_back(value);
  }
  return PackAttr::get(values, getType());
}

ErrorTreeOrSuccess PackCreateOp::interpret(ArrayRef<Attribute> operands,
                                           InterpreterState &state) {
  return state.interpretOpWithFolder(this->getOperation(), operands);
}

ErrorTreeOrSuccess
PackCreateOp::parametric_interpret(ArrayRef<Attribute> operands,
                                   ParametricInterpreterState &state) {
  SmallVector<TypedAttr> values;
  values.reserve(operands.size());
  for (Attribute operand : operands) {
    auto value = llvm::cast_if_present<TypedAttr>(operand);
    if (!value)
      return ErrorTree(getLoc(), "input type ill formed");
    values.push_back(value);
  }
  state.mapResults(
      PackAttr::get(values, cast<PackType>(state.getReboundType(getType()))));
  return success();
}

//===----------------------------------------------------------------------===//
// PackExtractOp
//===----------------------------------------------------------------------===//

OpFoldResult PackExtractOp::fold(FoldAdaptor adaptor) {
  auto index = dyn_cast_or_null<IntegerAttr>(adaptor.getIndexAttr());
  if (!index)
    return {};

  if (auto pack = dyn_cast_or_null<PackAttr>(adaptor.getPack()))
    return pack.getValues()[index.getInt()];

  // Canonicalize `get(create(x)) -> x`.
  if (auto create = getPack().getDefiningOp<PackCreateOp>())
    return create.getOperands()[index.getInt()];

  return {};
}

ErrorTreeOrSuccess PackExtractOp::interpret(ArrayRef<Attribute> operands,
                                            InterpreterState &state) {
  return state.interpretOpWithFolder(this->getOperation(), operands);
}

ErrorTreeOrSuccess
PackExtractOp::parametric_interpret(ArrayRef<Attribute> operands,
                                    ParametricInterpreterState &state) {
  auto index =
      dyn_cast_or_null<IntegerAttr>(state.getReboundAttribute(getIndexAttr()));
  if (!index) {
    return ErrorTree(getLoc(), "cannot get PackExtractOp index");
  }

  if (auto pack = dyn_cast_or_null<PackAttr>(operands.front())) {
    state.mapResults(pack.getValues()[index.getInt()]);
    return success();
  }

  return ErrorTree(getLoc(), "failed to interpret PackExtractOp");
}

//===----------------------------------------------------------------------===//
// PackSizeOp
//===----------------------------------------------------------------------===//

OpFoldResult PackSizeOp::fold(FoldAdaptor adaptor) {
  if (auto pack = dyn_cast_if_present<PackAttr>(adaptor.getOperand()))
    return IntegerAttr::get(IndexType::get(getContext()),
                            pack.getValues().size());
  return {};
}

//===----------------------------------------------------------------------===//
// PackGEPOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess PackGEPOp::interpret(ArrayRef<Attribute> operands,
                                        InterpreterState &state) {
  auto ptr = dyn_cast_if_present<PointerAttr>(operands[0]);
  auto idxAttr = dyn_cast_if_present<IntegerAttr>(getIndex());
  if (!ptr || !idxAttr)
    return ErrorTree(getLoc(), "non-constant inputs");

  int64_t offset = 0;
  auto variadic =
      getPack().getType().getElementAs<PackType>().getVariadicIfResolved();
  if (!variadic)
    return ErrorTree(getLoc(), "unknown type list");

  ArrayRef<TypedAttr> typeElts = variadic.getValues();

  // Move the address over the elements before the one we are reading.
  unsigned index = idxAttr.getInt();
  for (unsigned i = 0; i != index; ++i) {
    auto eltType = cast<TypeParamAttr>(typeElts[i]).getMlirType();
    auto dl = cast<DataLayoutInterface>(eltType);
    offset = llvm::alignTo(offset, *dl.getTypeAlign(state.getTarget()));
    offset += *dl.getTypeSize(state.getTarget());
  }

  // Align the address to the target element.
  Type targetType = cast<TypeParamAttr>(typeElts[index]).getMlirType();
  offset = llvm::alignTo(
      offset,
      *cast<DataLayoutInterface>(targetType).getTypeAlign(state.getTarget()));
  state.mapResults(
      PointerAttr::get(ptr.getAddr() + offset, PointerType::get(targetType)));
  return success();
}

ErrorTreeOrSuccess
PackGEPOp::parametric_interpret(ArrayRef<Attribute> operands,
                                ParametricInterpreterState &state) {
  auto ptr = dyn_cast_if_present<PointerAttr>(operands[0]);
  auto idxAttr =
      dyn_cast_if_present<IntegerAttr>(state.getReboundAttribute(getIndex()));
  if (!ptr || !idxAttr)
    return ErrorTree(getLoc(), "non-constant inputs");

  int64_t offset = 0;
  auto variadic = cast<PointerType>(ptr.getType())
                      .getElementAs<PackType>()
                      .getVariadicIfResolved();
  if (!variadic)
    return ErrorTree(getLoc(), "unknown type list");

  ArrayRef<TypedAttr> typeElts = variadic.getValues();

  // Move the address over the elements before the one we are reading.
  unsigned index = idxAttr.getInt();
  for (unsigned i = 0; i != index; ++i) {
    auto eltType = cast<TypeParamAttr>(typeElts[i]).getMlirType();
    auto dl = cast<DataLayoutInterface>(eltType);
    offset = llvm::alignTo(offset, *dl.getTypeAlign(state.getTarget()));
    offset += *dl.getTypeSize(state.getTarget());
  }

  // Align the address to the target element.
  Type targetType = cast<TypeParamAttr>(typeElts[index]).getMlirType();
  offset = llvm::alignTo(
      offset,
      *cast<DataLayoutInterface>(targetType).getTypeAlign(state.getTarget()));
  state.mapResults(
      PointerAttr::get(ptr.getAddr() + offset, PointerType::get(targetType)));
  return success();
}

//===----------------------------------------------------------------------===//
// PackLoadOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess PackLoadOp::interpret(ArrayRef<Attribute> operands,
                                         InterpreterState &state) {
  if (auto pack = dyn_cast<PackAttr>(operands[0])) {
    auto variadic = getType().getVariadicIfResolved();
    if (!variadic)
      return ErrorTree(getLoc(), "unknown type list");
    ArrayRef<TypedAttr> typeElts = variadic.getValues();

    SmallVector<TypedAttr> values;
    for (auto [ptr, type] : llvm::zip(pack.getValues(), typeElts)) {
      ErrorOr<Attribute> result = state.readAttributeFromPointer(
          ptr, cast<TypeParamAttr>(type).getMlirType());
      if (result.isError())
        return ErrorTree(getLoc(), result.takeError());
      values.push_back(cast<TypedAttr>(result.takeValue()));
    }
    state.mapResults(PackAttr::get(values, getType()));
    return success();
  }
  return ErrorTree(getLoc(), "non-constant inputs");
}

ErrorTreeOrSuccess
PackLoadOp::parametric_interpret(ArrayRef<Attribute> operands,
                                 ParametricInterpreterState &state) {
  if (auto pack = dyn_cast<PackAttr>(operands[0])) {
    auto packType = cast<PackType>(state.getReboundType(getType()));
    auto variadic = packType.getVariadicIfResolved();

    if (!variadic)
      return ErrorTree(getLoc(), "unknown type list");
    ArrayRef<TypedAttr> typeElts = variadic.getValues();

    SmallVector<TypedAttr> values;
    for (auto [ptr, type] : llvm::zip(pack.getValues(), typeElts)) {
      ErrorOr<Attribute> result = state.readAttributeFromPointer(
          ptr, cast<TypeParamAttr>(type).getMlirType());
      if (result.isError())
        return ErrorTree(getLoc(), result.takeError());
      values.push_back(cast<TypedAttr>(result.takeValue()));
    }
    state.mapResults(PackAttr::get(values, packType));
    return success();
  }
  return ErrorTree(getLoc(), "non-constant inputs");
}

//===----------------------------------------------------------------------===//
// VariantCreateOp
//===----------------------------------------------------------------------===//

OpFoldResult VariantCreateOp::fold(FoldAdaptor adaptor) {
  if (auto value = llvm::cast_if_present<TypedAttr>(adaptor.getOperand()))
    if (getOperand().getType() == value.getType())
      return VariantAttr::get(value, getIndex(), getType());

  // Canonicalize `kgen.variant.create(kgen.variant.get(x, n), n) -> x`
  auto takeOp = getOperand().getDefiningOp<VariantGetOp>();
  if (takeOp && takeOp.getIndex() == getIndex() &&
      takeOp.getOperand().getType() == getType())
    return takeOp.getOperand();

  return {};
}

ErrorTreeOrSuccess VariantCreateOp::interpret(ArrayRef<Attribute> operands,
                                              InterpreterState &state) {
  return state.interpretOpWithFolder(this->getOperation(), operands);
}

ErrorTreeOrSuccess
VariantCreateOp::parametric_interpret(ArrayRef<Attribute> operands,
                                      ParametricInterpreterState &state) {
  if (auto value = llvm::cast_if_present<TypedAttr>(operands.front())) {
    auto type = state.getReboundType(cast<TypedAttr>(operands[0]).getType());
    auto resultType = cast<VariantType>(state.getReboundType(getType()));
    if (type == value.getType()) {
      state.mapResults(VariantAttr::get(value, getIndex(), resultType));
      return success();
    }
  }

  return ErrorTree(getLoc(), "cannot interpret kgen.variant.create");
}

//===----------------------------------------------------------------------===//
// VariantIsOp
//===----------------------------------------------------------------------===//

OpFoldResult VariantIsOp::fold(FoldAdaptor adaptor) {
  if (auto variant = dyn_cast_if_present<VariantAttr>(adaptor.getVariant()))
    return BoolAttr::get(getContext(), variant.getIndex() == getIndex());

  if (auto createOp = getOperand().getDefiningOp<VariantCreateOp>())
    return BoolAttr::get(getContext(), createOp.getIndex() == getIndex());

  return {};
}

//===----------------------------------------------------------------------===//
// VariantGetOp
//===----------------------------------------------------------------------===//

OpFoldResult VariantGetOp::fold(FoldAdaptor adaptor) {
  if (auto variant = dyn_cast_if_present<VariantAttr>(adaptor.getVariant())) {
    // If the variant value type is not equal to the result type, this is
    // undefined behaviour.
    if (variant.getValue().getType() != getType())
      return {};
    return variant.getValue();
  }

  // Canonicalize `kgen.variant.get(kgen.variant.create(x)) -> x`.
  auto create = getVariant().getDefiningOp<VariantCreateOp>();
  if (!create || create.getOperand().getType() != getType() ||
      create.getIndex() != getIndex())
    return {};
  return create.getOperand();
}

ErrorTreeOrSuccess VariantGetOp::interpret(ArrayRef<Attribute> operands,
                                           InterpreterState &state) {
  return state.interpretOpWithFolder(this->getOperation(), operands);
}

ErrorTreeOrSuccess
VariantGetOp::parametric_interpret(ArrayRef<Attribute> operands,
                                   ParametricInterpreterState &state) {
  if (auto variant = dyn_cast_if_present<VariantAttr>(operands.front())) {
    //// If the variant value type is not equal to the result type, this is
    //// undefined behaviour.
    // if (variant.getValue().getType() != state.getReboundType(getType()))
    state.mapResults(variant.getValue());
    return success();
  }
  return ErrorTree(getLoc(), "input ill formed");
}

//===----------------------------------------------------------------------===//
// StructCreateOp
//===----------------------------------------------------------------------===//

static StructAttr foldStructCreateConstant(StructType resultType,
                                           ArrayRef<Attribute> operands) {
  SmallVector<TypedAttr> values;
  values.reserve(operands.size());
  for (Attribute operand : operands) {
    auto value = llvm::cast_if_present<TypedAttr>(operand);
    if (!value)
      return {};
    values.push_back(value);
  }
  return StructAttr::get(values, resultType);
}

static Value foldTrivialStructCopy(StructCreateOp op) {
  // Fold `create(%x[0], %x[1], %x[2]) -> %x` where `%x` has the same type.
  // An empty create would have been folded above.
  auto getSourceContainer = [&](unsigned idx, Value operand) -> Value {
    auto extract = operand.getDefiningOp<StructExtractOp>();
    if (extract && extract.getIndex() == idx &&
        extract.getContainer().getType() == op.getType())
      return extract.getContainer();
    return {};
  };
  Value container = getSourceContainer(0, op.getOperands().front());
  if (!container)
    return {};
  for (auto [idx, operand] :
       llvm::enumerate(llvm::drop_begin(op.getOperands())))
    if (getSourceContainer(idx + 1, operand) != container)
      return {};
  return container;
}

OpFoldResult StructCreateOp::fold(FoldAdaptor adaptor) {
  if (StructAttr cst =
          foldStructCreateConstant(getType(), adaptor.getOperands()))
    return cst;
  if (Value container = foldTrivialStructCopy(*this))
    return container;

  return {};
}

ErrorTreeOrSuccess StructCreateOp::interpret(ArrayRef<Attribute> operands,
                                             InterpreterState &state) {
  return state.interpretOpWithFolder(this->getOperation(), operands);
}

ErrorTreeOrSuccess
StructCreateOp::parametric_interpret(ArrayRef<Attribute> operands,
                                     ParametricInterpreterState &state) {
  if (StructAttr cst = foldStructCreateConstant(
          cast<StructType>(state.getReboundType(getType())), operands)) {
    state.mapResults(cst);
    return success();
  }

  return ErrorTree(getLoc(), "can't interpret POP::StructCreateOp");
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

OpFoldResult StructExtractOp::fold(FoldAdaptor adaptor) {
  if (auto container = adaptor.getContainer())
    return StructExtractAttr::get(cast<TypedAttr>(container),
                                  getIndexAttr().getInt());
  if (auto structCreate = getOperand().getDefiningOp<StructCreateOp>())
    return structCreate.getOperand(adaptor.getIndex());
  return {};
}

ErrorTreeOrSuccess StructExtractOp::interpret(ArrayRef<Attribute> operands,
                                              InterpreterState &state) {
  return state.interpretOpWithFolder(this->getOperation(), operands);
}

ErrorTreeOrSuccess
StructExtractOp::parametric_interpret(ArrayRef<Attribute> operands,
                                      ParametricInterpreterState &state) {
  auto result = StructExtractAttr::get(cast<TypedAttr>(operands.front()),
                                       getIndexAttr().getInt());
  state.mapResults(result);
  return success();
}

//===----------------------------------------------------------------------===//
// StructReplaceOp
//===----------------------------------------------------------------------===//

OpFoldResult StructReplaceOp::fold(FoldAdaptor adaptor) {
  auto value = llvm::cast_if_present<TypedAttr>(adaptor.getValue());
  auto container = dyn_cast_if_present<StructAttr>(adaptor.getContainer());
  if (!value || !container)
    return {};
  SmallVector<TypedAttr> values(container.getValues());
  values[getIndexAttr().getInt()] = value;
  return StructAttr::get(values, getType());
}

ErrorTreeOrSuccess StructReplaceOp::interpret(ArrayRef<Attribute> operands,
                                              InterpreterState &state) {
  return state.interpretOpWithFolder(this->getOperation(), operands);
}

ErrorTreeOrSuccess
StructReplaceOp::parametric_interpret(ArrayRef<Attribute> operands,
                                      ParametricInterpreterState &state) {
  auto value = llvm::cast_if_present<TypedAttr>(operands[0]);
  auto container = dyn_cast_if_present<StructAttr>(operands[1]);
  if (!value || !container)
    return ErrorTree(getLoc(), "input ill formed");
  SmallVector<TypedAttr> values(container.getValues());
  values[getIndexAttr().getInt()] = value;
  state.mapResults(StructAttr::get(
      values, cast<StructType>(state.getReboundType(getType()))));
  return success();
}

//===----------------------------------------------------------------------===//
// StructGEPOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess StructGEPOp::interpret(ArrayRef<Attribute> operands,
                                          InterpreterState &state) {
  auto ptr = dyn_cast_if_present<PointerAttr>(operands.front());
  if (!ptr)
    return ErrorTree(getLoc(), "non-constant inputs");

  int64_t offset = 0;
  auto structType = getContainer().getType().getElementAs<StructType>();

  // Move the address over the elements before the one we are reading.
  unsigned index = getIndexAttr().getInt();
  for (unsigned i = 0; i != index; ++i) {
    auto dl = cast<DataLayoutInterface>(structType.getElementTypes()[i]);
    offset = llvm::alignTo(offset, *dl.getTypeAlign(state.getTarget()));
    offset += *dl.getTypeSize(state.getTarget());
  }

  // Align the address to the target element.
  Type targetType = structType.getElementTypes()[index];
  offset = llvm::alignTo(
      offset,
      *cast<DataLayoutInterface>(targetType).getTypeAlign(state.getTarget()));
  state.mapResults(PointerAttr::get(ptr.getAddr() + offset, getType()));
  return success();
}

ErrorTreeOrSuccess
StructGEPOp::parametric_interpret(ArrayRef<Attribute> operands,
                                  ParametricInterpreterState &state) {
  auto ptr = dyn_cast_if_present<PointerAttr>(operands.front());
  if (!ptr)
    return ErrorTree(getLoc(), "non-constant inputs");

  int64_t offset = 0;
  auto structType =
      cast<KGEN::PointerType>(cast<TypedAttr>(operands[0]).getType())
          .getElementAs<StructType>();

  // Move the address over the elements before the one we are reading.
  unsigned index = getIndexAttr().getInt();
  for (unsigned i = 0; i != index; ++i) {
    auto dl = cast<DataLayoutInterface>(structType.getElementTypes()[i]);
    offset = llvm::alignTo(offset, *dl.getTypeAlign(state.getTarget()));
    offset += *dl.getTypeSize(state.getTarget());
  }

  // Align the address to the target element.
  Type targetType = structType.getElementTypes()[index];
  offset = llvm::alignTo(
      offset,
      *cast<DataLayoutInterface>(targetType).getTypeAlign(state.getTarget()));
  state.mapResults(PointerAttr::get(ptr.getAddr() + offset,
                                    state.getReboundType(getType())));
  return success();
}
