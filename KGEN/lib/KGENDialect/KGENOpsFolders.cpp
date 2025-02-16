//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/Interpreter/InterpreterState.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/DebugStringHelper.h"

using namespace M;
using namespace KGEN;

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

//===----------------------------------------------------------------------===//
// ParamConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ParamConstantOp::fold(FoldAdaptor adaptor) {
  return getValueAttr();
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

ErrorTreeOrSuccess ParamMaterializeOp::interpret(ArrayRef<Attribute> operands,
                                                 const Payload &payload,
                                                 InterpreterState &state) {
  if (ErrorOrSuccess err = interpretIfContainsPtr(payload, state);
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
  if (auto ptr = dyn_cast_or_null<SymbolicPointerAttr>(adaptor.getInput()))
    return SymbolicPointerAttr::get(ptr.getSlot(), getType());

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
  b.create<UnreachableOp>(op.getLoc());
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

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess ReturnOp::interpret(ArrayRef<Attribute> operands,
                                       InterpreterState &state) {
  state.returnFromFunction(operands);
  return success();
}

//===----------------------------------------------------------------------===//
// IntLiteralCmp
//===----------------------------------------------------------------------===//

OpFoldResult IntLiteralCmp::fold(FoldAdaptor adaptor) {
  IntLiteralAttr lAttr = dyn_cast_or_null<IntLiteralAttr>(adaptor.getLhs());
  IntLiteralAttr rAttr = dyn_cast_or_null<IntLiteralAttr>(adaptor.getRhs());
  IntLiteralCmpPred pred = adaptor.getPred();
  if (!lAttr || !rAttr)
    return {};
  IPInt l = lAttr.getValue();
  IPInt r = rAttr.getValue();

  switch (pred) {
  case IntLiteralCmpPred::Eq:
    return BoolAttr::get(lAttr.getContext(), l == r);
  case IntLiteralCmpPred::Ne:
    return BoolAttr::get(lAttr.getContext(), l != r);
  case IntLiteralCmpPred::Lt:
    return BoolAttr::get(lAttr.getContext(), l < r);
  case IntLiteralCmpPred::Le:
    return BoolAttr::get(lAttr.getContext(), l <= r);
  case IntLiteralCmpPred::Gt:
    return BoolAttr::get(lAttr.getContext(), l > r);
  case IntLiteralCmpPred::Ge:
    return BoolAttr::get(lAttr.getContext(), l >= r);
  }
  llvm_unreachable("invalid cmp predicate");
}

//===----------------------------------------------------------------------===//
// IntLiteralBinop
//===----------------------------------------------------------------------===//

OpFoldResult IntLiteralBinop::fold(FoldAdaptor adaptor) {
  IntLiteralAttr lAttr = dyn_cast_or_null<IntLiteralAttr>(adaptor.getLhs());
  IntLiteralAttr rAttr = dyn_cast_or_null<IntLiteralAttr>(adaptor.getRhs());
  IntLiteralBinopKind o = adaptor.getOper();
  if (!lAttr || !rAttr)
    return {};
  IPInt l = lAttr.getValue();
  IPInt r = rAttr.getValue();
  IPInt zero(0);
  IPInt one(1);

  IPInt result;
  switch (o) {
  case IntLiteralBinopKind::Add:
    result = l + r;
    break;
  case IntLiteralBinopKind::Sub:
    result = l - r;
    break;
  case IntLiteralBinopKind::Mul:
    result = l * r;
    break;
  case IntLiteralBinopKind::FloorDiv:
    if ((l >= zero) == (r >= zero) || l % r == zero)
      result = l / r;
    else
      result = (l / r) - one;
    break;
  case IntLiteralBinopKind::Mod:
    // Python's mod:
    // The result sign matches the RHS sign.
    // If the signs match, the value is the same as: sign(abs(l) % abs(r)),
    // where sign is determined by the RHS sign. If the signs don't match, the
    // value is the same as: sign((abs(r) - (abs(l) % abs(r))) % abs(r)).
    {
      bool signMatch = (l >= zero) == (r >= zero);
      IPInt lAbs = l.abs();
      IPInt rAbs = r.abs();
      result = (lAbs % rAbs).abs();
      if (!signMatch && result != zero)
        result = rAbs - result;
      if (r < zero)
        result = zero - result;
    }
    break;
  case IntLiteralBinopKind::Lshift:
    result = l << r;
    break;
  case IntLiteralBinopKind::Rshift:
    result = l >> r;
    break;
  case IntLiteralBinopKind::And:
    result = l & r;
    break;
  case IntLiteralBinopKind::Or:
    result = l | r;
    break;
  case IntLiteralBinopKind::Xor:
    result = l ^ r;
    break;
  }

  return IntLiteralAttr::get(lAttr.getContext(), IPInt(result));
}

//===----------------------------------------------------------------------===//
// IntLiteralConvertOp
//===----------------------------------------------------------------------===//

OpFoldResult IntLiteralConvertOp::fold(FoldAdaptor adaptor) {
  if (auto inAttr = dyn_cast_if_present<TypedAttr>(adaptor.getInput()))
    return IntLiteralConvertAttr::get(getType(), inAttr,
                                      getTreatIndexAsUnsigned());
  return {};
}

//===----------------------------------------------------------------------===//
// IntLiteralBitWidth
//===----------------------------------------------------------------------===//

OpFoldResult IntLiteralBitWidthOp::fold(FoldAdaptor adaptor) {
  auto in = dyn_cast_if_present<IntLiteralAttr>(adaptor.getInput());
  if (!in)
    return {};
  unsigned bits = in.getValue().getAPInt().getBitWidth();
  return IntLiteralAttr::get(in.getContext(), IPInt(bits));
}

//===----------------------------------------------------------------------===//
// IntLiteralToFloatLiteral
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess
IntLiteralToFloatLiteralOp::interpret(ArrayRef<Attribute> operands,
                                      InterpreterState &state) {
  assert(!operands.empty() &&
         "IntLiteralToFloatLiteralOp must have an operand");
  auto inval = ::dyn_cast<IntLiteralAttr>(operands[0]);
  if (!inval)
    return ErrorTree(getLoc(), Error("input must be IntLiteralAttr"));
  FloatLiteralAttr attrResult = FloatLiteralAttr::get(
      inval.getContext(),
      FloatLiteralSpecialValuesAttr::get(inval.getContext(),
                                         FloatLiteralSpecialValues::Normal),
      IPRational(inval.getValue(), IPInt(1)));
  state.mapResults(attrResult);
  return success();
}

OpFoldResult IntLiteralToFloatLiteralOp::fold(FoldAdaptor adaptor) {
  auto in = dyn_cast_if_present<IntLiteralAttr>(adaptor.getInput());
  if (!in)
    return {};
  return FloatLiteralAttr::get(
      in.getContext(),
      FloatLiteralSpecialValuesAttr::get(in.getContext(),
                                         FloatLiteralSpecialValues::Normal),
      IPRational(in.getValue(), IPInt(1)));
}

//===----------------------------------------------------------------------===//
// FloatLiteralIsa
//===----------------------------------------------------------------------===//

OpFoldResult FloatLiteralIsa::fold(FoldAdaptor adaptor) {
  if (auto input = dyn_cast_or_null<FloatLiteralAttr>(adaptor.getInput())) {
    return BoolAttr::get(input.getContext(),
                         input.getSpecial().getValue() == adaptor.getSpecial());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// FloatLiteralCmp
//===----------------------------------------------------------------------===//

static bool isNan(FloatLiteralSpecialValues v) {
  return v == FloatLiteralSpecialValues::Nan;
}
static bool isNegZero(FloatLiteralSpecialValues v) {
  return v == FloatLiteralSpecialValues::NegZero;
}
static bool isInf(FloatLiteralSpecialValues v) {
  return v == FloatLiteralSpecialValues::Inf;
}
static bool isNegInf(FloatLiteralSpecialValues v) {
  return v == FloatLiteralSpecialValues::NegInf;
}
static bool isNormal(FloatLiteralSpecialValues v) {
  return v == FloatLiteralSpecialValues::Normal;
}

/// Helper for float literal comparison.  The lhs/rhs values are only meaningful
/// when lSpecial/rSpecial are normal.
static bool floatLiteralCmpHelper(const FloatLiteralCmpPred &pred,
                                  const FloatLiteralSpecialValues &lSpecial,
                                  const FloatLiteralSpecialValues &rSpecial,
                                  const IPRational &lhs,
                                  const IPRational &rhs) {
  switch (pred) {
  case FloatLiteralCmpPred::Eq:
    if (lSpecial == rSpecial) {
      if (isNormal(lSpecial))
        return lhs == rhs;
      return !isNan(lSpecial);
    }
    // Python treats -0 and 0 as equal.
    if (isNegZero(lSpecial) && isNormal(rSpecial) && rhs == 0)
      return true;
    if (isNegZero(rSpecial) && isNormal(lSpecial) && lhs == 0)
      return true;
    return false;
  case FloatLiteralCmpPred::Ne:
    return !floatLiteralCmpHelper(FloatLiteralCmpPred::Eq, lSpecial, rSpecial,
                                  lhs, rhs);
  case FloatLiteralCmpPred::Lt:
    switch (lSpecial) {
    case FloatLiteralSpecialValues::Normal:
      switch (rSpecial) {
      case FloatLiteralSpecialValues::Normal:
        return lhs < rhs;
      case FloatLiteralSpecialValues::Inf:
        return true;
      case FloatLiteralSpecialValues::NegZero:
        return lhs < 0;
      default:
        return false;
      }
    case FloatLiteralSpecialValues::NegZero:
      switch (rSpecial) {
      case FloatLiteralSpecialValues::Normal:
        // This would be <=, but Python treats -0 as equal to 0, so the RHS
        // needs to be strictly greater than positive zero.
        return IPRational(0) < rhs;
      case FloatLiteralSpecialValues::Inf:
        return true;
      default:
        return false;
      }
    case FloatLiteralSpecialValues::Inf:
    case FloatLiteralSpecialValues::Nan:
      return false;
    case FloatLiteralSpecialValues::NegInf:
      return !isNan(rSpecial) && !isNegInf(rSpecial);
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralCmpPred::Le:
    return floatLiteralCmpHelper(FloatLiteralCmpPred::Lt, lSpecial, rSpecial,
                                 lhs, rhs) ||
           floatLiteralCmpHelper(FloatLiteralCmpPred::Eq, lSpecial, rSpecial,
                                 lhs, rhs);
  case FloatLiteralCmpPred::Gt:
    if (isNan(lSpecial) || isNan(rSpecial))
      return false;
    return !floatLiteralCmpHelper(FloatLiteralCmpPred::Le, lSpecial, rSpecial,
                                  lhs, rhs);
  case FloatLiteralCmpPred::Ge:
    return floatLiteralCmpHelper(FloatLiteralCmpPred::Gt, lSpecial, rSpecial,
                                 lhs, rhs) ||
           floatLiteralCmpHelper(FloatLiteralCmpPred::Eq, lSpecial, rSpecial,
                                 lhs, rhs);
  }
  llvm_unreachable("invalid cmp predicate");
}

OpFoldResult FloatLiteralCmp::fold(FoldAdaptor adaptor) {
  auto lAttr = dyn_cast_or_null<FloatLiteralAttr>(adaptor.getLhs());
  auto rAttr = dyn_cast_or_null<FloatLiteralAttr>(adaptor.getRhs());
  if (!lAttr || !rAttr)
    return {};
  FloatLiteralSpecialValues lSpecial = lAttr.getSpecial().getValue();
  FloatLiteralSpecialValues rSpecial = rAttr.getSpecial().getValue();
  IPRational lhs;
  IPRational rhs;
  if (isNormal(lSpecial)) {
    assert(lAttr.getRational().has_value() &&
           "rational does not have a value when special value is normal");
    lhs = lAttr.getRational().value();
  }
  if (isNormal(rSpecial)) {
    assert(rAttr.getRational().has_value() &&
           "rational does not have a value when special value is normal");
    rhs = rAttr.getRational().value();
  }
  return BoolAttr::get(
      lAttr.getContext(),
      floatLiteralCmpHelper(adaptor.getPred(), lSpecial, rSpecial, lhs, rhs));
}

//===----------------------------------------------------------------------===//
// FloatLiteralBinop
//===----------------------------------------------------------------------===//

static std::tuple<FloatLiteralSpecialValues, IPRational>
floatLiteralAdd(FloatLiteralSpecialValues lSpecial,
                FloatLiteralSpecialValues rSpecial, IPRational lhs,
                IPRational rhs) {
  switch (lSpecial) {
  case FloatLiteralSpecialValues::NegZero:
    if (isNegZero(rSpecial))
      return {FloatLiteralSpecialValues::Normal, 0};
    return {rSpecial, rhs};
  case FloatLiteralSpecialValues::Inf:
    if (isNegInf(rSpecial) || isNan(rSpecial))
      return {FloatLiteralSpecialValues::Nan, 0};
    return {FloatLiteralSpecialValues::Inf, 0};
  case FloatLiteralSpecialValues::NegInf:
    if (isInf(rSpecial) || isNan(rSpecial))
      return {FloatLiteralSpecialValues::Nan, 0};
    return {FloatLiteralSpecialValues::NegInf, 0};
  case FloatLiteralSpecialValues::Nan:
    return {FloatLiteralSpecialValues::Nan, 0};
  case FloatLiteralSpecialValues::Normal:
    if (isNormal(rSpecial))
      return {FloatLiteralSpecialValues::Normal, lhs + rhs};
    return floatLiteralAdd(rSpecial, lSpecial, rhs, lhs);
  }
  llvm_unreachable("unknown FloatLiteral special type");
}

static std::tuple<FloatLiteralSpecialValues, IPRational>
floatLiteralSub(FloatLiteralSpecialValues lSpecial,
                FloatLiteralSpecialValues rSpecial, const IPRational &lhs,
                const IPRational &rhs) {
  switch (lSpecial) {
  case FloatLiteralSpecialValues::NegZero:
    // When adding zeroes, the signs are basically XORed, like with
    // multiplication.
    if (isNegZero(rSpecial))
      return {FloatLiteralSpecialValues::Normal, 0};
    if (isNormal(rSpecial) && rhs == 0)
      return {FloatLiteralSpecialValues::NegZero, 0};
    return floatLiteralSub(FloatLiteralSpecialValues::Normal, rSpecial, 0, rhs);
  case FloatLiteralSpecialValues::Inf:
    if (isInf(rSpecial) || isNan(rSpecial))
      return {FloatLiteralSpecialValues::Nan, 0};
    return {FloatLiteralSpecialValues::Inf, 0};
  case FloatLiteralSpecialValues::NegInf:
    if (isNegInf(rSpecial) || isNan(rSpecial))
      return {FloatLiteralSpecialValues::Nan, 0};
    return {FloatLiteralSpecialValues::NegInf, 0};
  case FloatLiteralSpecialValues::Nan:
    return {FloatLiteralSpecialValues::Nan, 0};
  case FloatLiteralSpecialValues::Normal:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::NegZero:
      return {lSpecial, lhs};
    case FloatLiteralSpecialValues::Inf:
      return {FloatLiteralSpecialValues::NegInf, 0};
    case FloatLiteralSpecialValues::NegInf:
      return {FloatLiteralSpecialValues::Inf, 0};
    case FloatLiteralSpecialValues::Nan:
      return {FloatLiteralSpecialValues::Nan, 0};
    case FloatLiteralSpecialValues::Normal:
      return {FloatLiteralSpecialValues::Normal, lhs - rhs};
    }
  }
  llvm_unreachable("unknown FloatLiteral special type");
}

/// Helper for multiplication, to keep the special case matching table separate.
/// Assumes that at least one of lSpecial and rSpecial is non-normal.
static FloatLiteralSpecialValues
floatLiteralMulSpecialCases(const FloatLiteralSpecialValues &lSpecial,
                            const FloatLiteralSpecialValues &rSpecial,
                            const IPRational &lhs, const IPRational &rhs) {
  switch (lSpecial) {
  case FloatLiteralSpecialValues::NegZero:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
    case FloatLiteralSpecialValues::Inf:
    case FloatLiteralSpecialValues::NegInf:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::NegZero:
      return FloatLiteralSpecialValues::Normal;
    case FloatLiteralSpecialValues::Normal:
      if (rhs < 0)
        return FloatLiteralSpecialValues::Normal;
      return FloatLiteralSpecialValues::NegZero;
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralSpecialValues::Inf:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
    case FloatLiteralSpecialValues::NegZero:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::NegInf:
      return FloatLiteralSpecialValues::NegInf;
    case FloatLiteralSpecialValues::Inf:
      return FloatLiteralSpecialValues::Inf;
    case FloatLiteralSpecialValues::Normal:
      if (rhs == 0)
        return FloatLiteralSpecialValues::Nan;
      if (rhs < 0)
        return FloatLiteralSpecialValues::NegInf;
      return FloatLiteralSpecialValues::Inf;
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralSpecialValues::NegInf:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
    case FloatLiteralSpecialValues::NegZero:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::NegInf:
      return FloatLiteralSpecialValues::Inf;
    case FloatLiteralSpecialValues::Inf:
      return FloatLiteralSpecialValues::NegInf;
    case FloatLiteralSpecialValues::Normal:
      if (rhs == 0)
        return FloatLiteralSpecialValues::Nan;
      if (rhs < 0)
        return FloatLiteralSpecialValues::Inf;
      return FloatLiteralSpecialValues::NegInf;
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralSpecialValues::Nan:
    return FloatLiteralSpecialValues::Nan;
  case FloatLiteralSpecialValues::Normal:
    // The case of both being normal is handled up front, so we don't worry
    // about it here.  Instead just recur with flipped operand order to handle
    // the case that LHS is normal.
    return floatLiteralMulSpecialCases(rSpecial, lSpecial, rhs, lhs);
  }
  llvm_unreachable("unknown FloatLiteral special type");
}

static std::tuple<FloatLiteralSpecialValues, IPRational>
floatLiteralMul(FloatLiteralSpecialValues lSpecial,
                FloatLiteralSpecialValues rSpecial, IPRational lhs,
                IPRational rhs) {
  if (isNormal(lSpecial) && isNormal(rSpecial)) {
    IPRational ratResult = lhs * rhs;
    if (ratResult == 0 && ((lhs < 0) || (rhs < 0)))
      return {FloatLiteralSpecialValues::NegZero, 0};
    return {FloatLiteralSpecialValues::Normal, ratResult};
  }
  return {floatLiteralMulSpecialCases(lSpecial, rSpecial, lhs, rhs), 0};
}

/// Helper to separate the special case logic for division.  Assumes that at
/// least one of lSpecial and rSpecial is non-normal.
static FloatLiteralSpecialValues
floatLiteralDivSpecialCases(const FloatLiteralSpecialValues &lSpecial,
                            const FloatLiteralSpecialValues &rSpecial,
                            const IPRational &lhs, const IPRational &rhs) {
  switch (lSpecial) {
  case FloatLiteralSpecialValues::NegZero:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::Inf:
      return FloatLiteralSpecialValues::NegZero;
    case FloatLiteralSpecialValues::NegInf:
      return FloatLiteralSpecialValues::Normal;
    case FloatLiteralSpecialValues::NegZero:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::Normal:
      if (rhs == 0)
        return FloatLiteralSpecialValues::Nan;
      if (rhs < 0)
        return FloatLiteralSpecialValues::Normal;
      return FloatLiteralSpecialValues::NegZero;
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralSpecialValues::Inf:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
    case FloatLiteralSpecialValues::NegZero:
    case FloatLiteralSpecialValues::NegInf:
    case FloatLiteralSpecialValues::Inf:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::Normal:
      if (rhs == 0)
        return FloatLiteralSpecialValues::Nan;
      if (rhs < 0)
        return FloatLiteralSpecialValues::NegInf;
      return FloatLiteralSpecialValues::Inf;
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralSpecialValues::NegInf:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
    case FloatLiteralSpecialValues::NegZero:
    case FloatLiteralSpecialValues::NegInf:
    case FloatLiteralSpecialValues::Inf:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::Normal:
      if (rhs == 0)
        return FloatLiteralSpecialValues::Nan;
      if (rhs < 0)
        return FloatLiteralSpecialValues::Inf;
      return FloatLiteralSpecialValues::NegInf;
    }
    llvm_unreachable("all specials covered");
  case FloatLiteralSpecialValues::Nan:
    return FloatLiteralSpecialValues::Nan;
  case FloatLiteralSpecialValues::Normal:
    switch (rSpecial) {
    case FloatLiteralSpecialValues::Nan:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::Inf:
      if (lhs < 0)
        return FloatLiteralSpecialValues::NegZero;
      return FloatLiteralSpecialValues::Normal;
    case FloatLiteralSpecialValues::NegInf:
      if (lhs < 0)
        return FloatLiteralSpecialValues::Normal;
      return FloatLiteralSpecialValues::NegZero;
    case FloatLiteralSpecialValues::NegZero:
      return FloatLiteralSpecialValues::Nan;
    case FloatLiteralSpecialValues::Normal:
      llvm_unreachable("double normal case handled above");
    }
  }
  llvm_unreachable("unknown FloatLiteral special type");
}

static std::tuple<FloatLiteralSpecialValues, IPRational>
floatLiteralDiv(FloatLiteralSpecialValues lSpecial,
                FloatLiteralSpecialValues rSpecial, IPRational lhs,
                IPRational rhs) {
  if (isNormal(lSpecial) && isNormal(rSpecial)) {
    if (rhs == 0)
      return {FloatLiteralSpecialValues::Nan, 0};
    IPRational ratResult = lhs / rhs;
    if (lhs == 0 && rhs < 0)
      return {FloatLiteralSpecialValues::NegZero, 0};
    return {FloatLiteralSpecialValues::Normal, ratResult};
  };
  return {floatLiteralDivSpecialCases(lSpecial, rSpecial, lhs, rhs), 0};
}

OpFoldResult FloatLiteralBinop::fold(FoldAdaptor adaptor) {
  FloatLiteralAttr lAttr = dyn_cast_or_null<FloatLiteralAttr>(adaptor.getLhs());
  FloatLiteralAttr rAttr = dyn_cast_or_null<FloatLiteralAttr>(adaptor.getRhs());
  FloatLiteralBinopKind oper = adaptor.getOper();
  if (!lAttr || !rAttr)
    return {};
  FloatLiteralSpecialValues lSpecial = lAttr.getSpecial().getValue();
  FloatLiteralSpecialValues rSpecial = rAttr.getSpecial().getValue();
  IPRational lhs;
  IPRational rhs;
  if (isNormal(lSpecial)) {
    assert(lAttr.getRational().has_value() &&
           "rational has value when special value is normal");
    lhs = lAttr.getRational().value();
  }
  if (isNormal(rSpecial)) {
    assert(rAttr.getRational().has_value() &&
           "rational has value when special value is normal");
    rhs = rAttr.getRational().value();
  }

  auto mkAttr = [&](FloatLiteralSpecialValues resultSpecial,
                    IPRational rational) -> FloatLiteralAttr {
    return FloatLiteralAttr::get(
        lAttr.getContext(),
        FloatLiteralSpecialValuesAttr::get(lAttr.getContext(), resultSpecial),
        rational);
  };

  switch (oper) {
  case FloatLiteralBinopKind::Add: {
    auto [resultSpecial, rational] =
        floatLiteralAdd(lSpecial, rSpecial, lhs, rhs);
    return mkAttr(resultSpecial, rational);
  } break;
  case FloatLiteralBinopKind::Sub: {
    auto [resultSpecial, rational] =
        floatLiteralSub(lSpecial, rSpecial, lhs, rhs);
    return mkAttr(resultSpecial, rational);
  } break;
  case FloatLiteralBinopKind::Mul: {
    auto [resultSpecial, rational] =
        floatLiteralMul(lSpecial, rSpecial, lhs, rhs);
    return mkAttr(resultSpecial, rational);
  } break;
  case FloatLiteralBinopKind::TrueDiv: {
    auto [resultSpecial, rational] =
        floatLiteralDiv(lSpecial, rSpecial, lhs, rhs);
    return mkAttr(resultSpecial, rational);
  } break;
  }
  llvm_unreachable("unknown FloatLiteralBinop type");
}

//===----------------------------------------------------------------------===//
// FloatLiteralConvertOp
//===----------------------------------------------------------------------===//

/// Take an IPRational along with a specification for an output float type and
/// return the IEEE-style float bit string as an APInt.
static APInt floatLiteralConvertGetBitstring(IPRational input,
                                             unsigned totalLength,
                                             unsigned exponentLength,
                                             unsigned bias) {
  // Throughout this function I use “significand” to mean the float value
  // including the digit before the decimal, and “mantissa” to mean just the
  // part after the decimal, IE the bit pattern that is actually present in the
  // float value.  That's not technically correct, but it was helpful for me to
  // distinguish the two.

  unsigned mantissaLength = totalLength - exponentLength - 1;
  IPInt maxExponentZeroBias = (IPInt(1) << exponentLength) - 1;
  IPInt maxExponent = maxExponentZeroBias - bias;
  IPInt minExponent = IPInt(-1) * IPInt(bias - 1);

  // The maxSignificandIPIntLength is longer than the float mantissa bit width
  // to allow for:
  // * leading 0 in IPInt format
  // * most significant 1 bit that is removed in final encoding
  // * extra precision bits to ensure correct rounding
  unsigned maxSignificandIPIntRoundedLength = mantissaLength + 2;
  static const unsigned kSignificandRoundingLength = 3;
  unsigned maxSignificandIPIntLength =
      maxSignificandIPIntRoundedLength + kSignificandRoundingLength;

  // To support subnormal numbers (IE numbers with minimum exponent that have an
  // implicit leading 0 instead of implicit leading 1), we need to support lower
  // exponents during calculation.
  IPInt minCalculationExponent = minExponent - mantissaLength;

  if (input.getNumerator() == 0)
    return APInt(totalLength, 0);

  bool negativeSign = input.getNumerator() < 0;
  APInt signBits = APInt(totalLength, negativeSign ? 1 : 0);
  signBits = signBits << (totalLength - 1);

  IPInt initialNumerator = input.getNumerator().abs();
  const IPInt &denominator = input.getDenominator();
  IPInt significand = initialNumerator / denominator;
  IPInt remainder = initialNumerator % denominator;
  IPInt exponent = 0;
  bool exponentFinalized = false;
  if (significand > 0) {
    // The IPInt encoding of the number will have a leading 0 bit (because it is
    // positive), and the exponent when treating the most significant one bit is
    // one less than the number of bits representing the number with no leading
    // zeroes.
    exponent = significand.getAPInt().getBitWidth() - 2;
    exponentFinalized = true;
  }

  auto keepDoingLongDivision = [&]() -> bool {
    if (remainder == 0)
      return false;
    if (exponent < minCalculationExponent || exponent > maxExponent)
      return false;
    if (significand.getAPInt().getBitWidth() > maxSignificandIPIntLength)
      return false;
    return true;
  };

  // Do long division loop.
  while (keepDoingLongDivision()) {
    unsigned nBitsToShift = denominator.getAPInt().getBitWidth() -
                            remainder.getAPInt().getBitWidth();
    if (nBitsToShift == 0)
      nBitsToShift = 1;
    IPInt nCur = remainder << nBitsToShift;
    if (!exponentFinalized) {
      exponent = exponent - nBitsToShift;
    }
    IPInt quotient = nCur / denominator;
    remainder = nCur % denominator;
    if (quotient > 0)
      exponentFinalized = true;
    significand = (significand << nBitsToShift) + quotient;
  }

  // If we finished long division with “enough” rounding bits, but the remainder
  // is still not zero, it means that eventually there will be another 1 bit,
  // which would break a rounding tie.  Appending any further 1 bit will have
  // the same effect on rounding (no effect other than tie breaking), so we just
  // add the next one.
  if (remainder != 0)
    significand = (significand << 1) + 1;

  // Early return for obvious zero case because our later logic requires a
  // non-zero significand.
  if (significand == 0)
    return signBits;

  // Pad to mantissa length before performing rounding, etc.
  if (significand.getAPInt().getBitWidth() < maxSignificandIPIntLength) {
    significand = significand << (maxSignificandIPIntLength -
                                  significand.getAPInt().getBitWidth());
  }

  auto performRounding = [](IPInt &significand, IPInt &exponent,
                            unsigned maxSignificandIPIntRoundedLength) {
    APInt roundingBits = significand.getAPInt().extractBits(
        /*numBits=*/significand.getAPInt().getBitWidth() -
            maxSignificandIPIntRoundedLength,
        /*bitPosition=*/0);
    unsigned roundingBitsActualLength = roundingBits.getBitWidth();
    APInt roundingMidpoint = APInt(roundingBitsActualLength, 1)
                             << (roundingBitsActualLength - 1);
    // Truncate bits first.
    significand = significand >> roundingBitsActualLength;
    // Now that we've truncated, rounding either means doing nothing (for
    // round toward zero) or adding one to the significand representation
    // (for rounding away from zero). The default rounding mode for IEEE
    // floats is “round to nearest, ties to even”. It might be good to take
    // an option to do other rounding modes, but for now we just support the
    // default.
    if (roundingBits.ugt(roundingMidpoint))
      significand = significand + 1;
    else if (roundingBits == roundingMidpoint && significand % 2 == 1)
      significand = significand + 1;
    // If rounding up increased digit count, we need to convert that into a
    // larger exponent and re-truncate.
    if (significand.getAPInt().getBitWidth() >
        maxSignificandIPIntRoundedLength) {
      exponent = exponent + 1;
      significand = significand >> 1;
    }
  };

  // Do rounding now unless we are dealing with a subnormal number, which needs
  // some extra handling before rounding.
  if (exponent >= minExponent)
    performRounding(significand, exponent, maxSignificandIPIntRoundedLength);

  if (exponent > maxExponent) {
    // Return +/- infinity.
    APInt exponentOnes = APInt::getAllOnes(exponentLength);
    APInt exponentBits = APInt(totalLength, 0);
    exponentBits.insertBits(exponentOnes, mantissaLength);
    // Mantissa for infinity is zero.
    return signBits | exponentBits;
  }

  // Handle subnormal numbers, including zero valuess.  (I'm not sure whether
  // zero counts technically as a subnormal number, but it fits the subnormal
  // encoding.)
  if (exponent < minExponent) {
    // Below the minExponent we can still convert to subnormal numbers.
    // The subnormal range is tagged with minExponent - 1, but the exponent
    // value is effectively the same as minExponent. However, instead of an
    // implicit leading 1 before the decimal, there is a leading 0. So subnormal
    // numbers cover down to minExponent - (mantissaWidth - 1) exponent, but
    // losing one bit of mantissa precision for each exponent lowering.
    IPInt minSubnormalExponent = minExponent - (mantissaLength - 1);
    if (exponent < minSubnormalExponent) {
      // We could let this fall through and be handled by the shifting and bit
      // mangling, but at this point we know that every bit is zero except
      // (maybe) the sign.
      return signBits;
    }
    IPInt shiftBits = minExponent - exponent;
    IPInt shiftTag = IPInt(1) << (IPInt(significand.getAPInt().getBitWidth()) -
                                  IPInt(2) + shiftBits);
    // The significand is now
    // `01<correct-bit-pattern><at-least-one-extra-bit>`.
    significand = shiftTag + significand;
    exponent = minExponent - 1;
    // If rounding increases the exponent and carries to a new high bit, then we
    // end up at 1000... for the significand with minExponent, and thus the
    // right number.  Cool.
    performRounding(significand, exponent, maxSignificandIPIntRoundedLength);
  }

  // Whether or not the value was subnormal, the significand now has the bit
  // pattern `01<correct-bit-pattern><maybe-extra-bit-due-to-rounding>`.  So we
  // drop the leading 2 bits and the trailing extra bits to arrive at the final
  // bit pattern for the mantissa.

  unsigned extraSignificandBits =
      significand.getAPInt().getBitWidth() - (mantissaLength + 2);
  significand = significand >> extraSignificandBits;
  assert(significand.getAPInt().getBitWidth() == mantissaLength + 2 &&
         "proper mantissa bit length");
  APInt mantissaLowBits = significand.getAPInt().extractBits(
      /*numBits=*/mantissaLength,
      /*bitPosition=*/0);
  APInt mantissaBits = APInt(totalLength, 0);
  mantissaBits.insertBits(mantissaLowBits, /*bitPosition=*/0);

  // Floating point numbers encode the exponent as `bias + exponent`, so that
  // the result is always a natural number, where `bias + exponent = 0`
  // signifies subnormal (including zero) numbers, and all ones is the
  // exponent for infinity and the NAN values.
  exponent = exponent + bias;
  // Place the bits into an APInt at the appropriate place.
  APInt exponentBits = APInt(totalLength, 0);
  exponentBits.insertBits(exponent.getAPInt(), mantissaLength);

  // Combine pieces to get final bit string: <sign><exponent><mantissa>.
  return signBits | exponentBits | mantissaBits;
}

static ErrorTreeOr<FloatAttr>
floatLiteralConvertOpHelper(FloatLiteralSpecialValues special,
                            std::optional<IPRational> inRat, Type outType,
                            Location loc) {
  unsigned totalLength = 0;
  unsigned exponentLength = 0;
  unsigned bias = 0;
  llvm::APFloatBase::Semantics semantics = llvm::APFloatBase::S_IEEEhalf;

#define DECLARE_FLOAT(SHORT_NAME, LONG_NAME, M_TYPE, MLIR_TYPE, CXX_TYPE,      \
                      BITCOUNT, APFLOAT_TYPE, LLVM_SEMANTICS,                  \
                      SIGNIFICAND_PRECISION, EXPONENT_LENGTH, BIAS)            \
  if (isa<MLIR_TYPE>(outType)) {                                               \
    totalLength = BITCOUNT;                                                    \
    exponentLength = EXPONENT_LENGTH;                                          \
    bias = BIAS;                                                               \
    semantics = LLVM_SEMANTICS;                                                \
  }

#include "Support/ML/FloatTypes.def"

#undef DECLARE_FLOAT

  if (bias == 0) {
    return ErrorTree(
        loc, Error("float literal conversion: unsupported output type"));
  }

  const llvm::fltSemantics &fltSemantics =
      llvm::APFloatBase::EnumToSemantics(semantics);
  APFloat resultValue(fltSemantics, APFloat::uninitialized);
  switch (special) {
  case FloatLiteralSpecialValues::Nan:
    // Set the payload to uint64_t::max to make the NaN fill all the low bits
    // to 1. This makes the NaN value aligned with the NaN values generated by
    // CUDA libraries.
    resultValue = APFloat::getNaN(
        fltSemantics,
        /*Negative=*/false, /*payload=*/std::numeric_limits<uint64_t>::max());
    break;
  case FloatLiteralSpecialValues::Inf:
    resultValue = APFloat::getInf(fltSemantics, /*negative=*/false);
    break;
  case FloatLiteralSpecialValues::NegInf:
    resultValue = APFloat::getInf(fltSemantics, /*negative=*/true);
    break;
  case FloatLiteralSpecialValues::NegZero:
    resultValue = APFloat::getZero(fltSemantics, /*negative=*/true);
    break;
  case FloatLiteralSpecialValues::Normal: {
    assert(inRat.has_value() && "normal FloatLiteral values have a rational");
    APInt floatBits = floatLiteralConvertGetBitstring(
        inRat.value(), totalLength, exponentLength, bias);
    resultValue = APFloat(fltSemantics, floatBits);
    break;
  }
  }
  return FloatAttr::get(outType, resultValue);
}

ErrorTreeOrSuccess
FloatLiteralConvertOp::interpret(ArrayRef<Attribute> operands,
                                 InterpreterState &state) {
  assert(!operands.empty() && "FloatLiteralConvertOp must have an operand");
  auto inval = ::dyn_cast<FloatLiteralAttr>(operands[0]);
  ErrorTreeOr<FloatAttr> errOrAttr = floatLiteralConvertOpHelper(
      inval.getSpecial().getValue(), inval.getRational(), getType(), getLoc());
  if (errOrAttr.hasValue())
    state.mapResults(errOrAttr.getValue());
  else
    return errOrAttr.takeError();
  return success();
}

OpFoldResult FloatLiteralConvertOp::fold(FoldAdaptor adaptor) {
  auto in = dyn_cast_if_present<FloatLiteralAttr>(adaptor.getInput());
  if (!in)
    return {};
  ErrorTreeOr<FloatAttr> errOrAttr = floatLiteralConvertOpHelper(
      in.getSpecial().getValue(), in.getRational(), getType(), getLoc());
  if (errOrAttr.hasValue())
    return errOrAttr.getValue();
  return {};
}

//===----------------------------------------------------------------------===//
// FloatLiteralToIntLiteral
//===----------------------------------------------------------------------===//

static IntLiteralAttr FloatLiteralToIntLiteralOpHelper(FloatLiteralAttr fattr) {
  IPInt result;
  switch (fattr.getSpecial().getValue()) {
  case FloatLiteralSpecialValues::Nan:
  case FloatLiteralSpecialValues::Inf:
  case FloatLiteralSpecialValues::NegInf:
  case FloatLiteralSpecialValues::NegZero:
    result = 0;
    break;
  case FloatLiteralSpecialValues::Normal:
    assert(fattr.getRational().has_value() &&
           "normal FloatLiterals have rational");
    result = fattr.getRational()->getNumerator() /
             fattr.getRational()->getDenominator();
    break;
  }
  return IntLiteralAttr::get(fattr.getContext(), result);
}

ErrorTreeOrSuccess
FloatLiteralToIntLiteralOp::interpret(ArrayRef<Attribute> operands,
                                      InterpreterState &state) {
  assert(!operands.empty() &&
         "FloatLiteralToIntLiteralOp must have an operand");
  auto inval = ::dyn_cast<FloatLiteralAttr>(operands[0]);
  IntLiteralAttr attrResult = FloatLiteralToIntLiteralOpHelper(inval);
  state.mapResults(attrResult);
  return success();
}

OpFoldResult FloatLiteralToIntLiteralOp::fold(FoldAdaptor adaptor) {
  auto in = dyn_cast_if_present<FloatLiteralAttr>(adaptor.getInput());
  if (!in)
    return {};
  return FloatLiteralToIntLiteralOpHelper(in);
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

//===----------------------------------------------------------------------===//
// StructCreateOp
//===----------------------------------------------------------------------===//

static StructAttr foldStructCreateConstant(StructCreateOp op,
                                           ArrayRef<Attribute> operands) {
  SmallVector<TypedAttr> values;
  values.reserve(operands.size());
  for (Attribute operand : operands) {
    auto value = llvm::cast_if_present<TypedAttr>(operand);
    if (!value)
      return {};
    values.push_back(value);
  }
  return StructAttr::get(values, op.getType());
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
  if (StructAttr cst = foldStructCreateConstant(*this, adaptor.getOperands()))
    return cst;
  if (Value container = foldTrivialStructCopy(*this))
    return container;

  return {};
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

//===----------------------------------------------------------------------===//
// ParamConstantOp
//===----------------------------------------------------------------------===//

ErrorOrSuccess ParamConstantOp::compile(Payload &payload,
                                        TargetInfoAttr target) {
  return populateContainsPtrPayload(getValue(), payload);
}

ErrorTreeOrSuccess ParamConstantOp::interpret(ArrayRef<Attribute> operands,
                                              const Payload &payload,
                                              InterpreterState &state) {
  if (ErrorOrSuccess err = interpretIfContainsPtr(payload, state);
      err.isError())
    return ErrorTree(getLoc(), err.takeError());
  return success();
}
