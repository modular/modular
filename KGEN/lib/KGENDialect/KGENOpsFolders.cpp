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

OpFoldResult IntLiteralCmpOp::fold(FoldAdaptor adaptor) {
  if (auto lhs = dyn_cast_or_null<TypedAttr>(adaptor.getLhs()))
    if (auto rhs = dyn_cast_or_null<TypedAttr>(adaptor.getRhs()))
      return IntLiteralCmpAttr::get(lhs.getContext(), getPredAttr(), lhs, rhs);
  return {};
}

//===----------------------------------------------------------------------===//
// IntLiteralBinop
//===----------------------------------------------------------------------===//

OpFoldResult IntLiteralBinOp::fold(FoldAdaptor adaptor) {
  if (auto lhs = dyn_cast_or_null<TypedAttr>(adaptor.getLhs()))
    if (auto rhs = dyn_cast_or_null<TypedAttr>(adaptor.getRhs()))
      return IntLiteralBinAttr::get(lhs.getContext(), lhs, rhs, getOperAttr());
  return {};
}

//===----------------------------------------------------------------------===//
// IntLiteralConvertOp
//===----------------------------------------------------------------------===//

OpFoldResult IntLiteralConvertOp::fold(FoldAdaptor adaptor) {
  if (auto inAttr = dyn_cast_if_present<TypedAttr>(adaptor.getInput()))
    return IntLiteralConvertAttr::get(getType().getContext(), getType(), inAttr,
                                      getTreatIndexAsUnsigned());
  return {};
}

//===----------------------------------------------------------------------===//
// FloatLiteralIsa
//===----------------------------------------------------------------------===//

OpFoldResult FloatLiteralIsa::fold(FoldAdaptor adaptor) {
  if (auto input = dyn_cast_or_null<TypedAttr>(adaptor.getInput()))
    return FloatLiteralIsaAttr::get(input.getContext(), getSpecialAttr(),
                                    input);
  return {};
}

//===----------------------------------------------------------------------===//
// FloatLiteralCmpOp
//===----------------------------------------------------------------------===//

OpFoldResult FloatLiteralCmpOp::fold(FoldAdaptor adaptor) {
  if (auto lAttr = dyn_cast_or_null<TypedAttr>(adaptor.getLhs()))
    if (auto rAttr = dyn_cast_or_null<TypedAttr>(adaptor.getRhs()))
      return FloatLiteralCmpAttr::get(lAttr.getContext(), adaptor.getPredAttr(),
                                      lAttr, rAttr);
  return {};
}

//===----------------------------------------------------------------------===//
// FloatLiteralBinOp
//===----------------------------------------------------------------------===//

OpFoldResult FloatLiteralBinOp::fold(FoldAdaptor adaptor) {
  if (auto lhs = dyn_cast_or_null<TypedAttr>(adaptor.getLhs()))
    if (auto rhs = dyn_cast_or_null<TypedAttr>(adaptor.getRhs()))
      return FloatLiteralBinAttr::get(lhs.getContext(), lhs, rhs,
                                      getOperAttr());
  return {};
}

//===----------------------------------------------------------------------===//
// FloatLiteralConvertOp
//===----------------------------------------------------------------------===//

OpFoldResult FloatLiteralConvertOp::fold(FoldAdaptor adaptor) {
  if (auto in = dyn_cast_if_present<TypedAttr>(adaptor.getInput()))
    return FloatLiteralConvertAttr::get(in.getContext(), getType(), in);
  return {};
}

//===----------------------------------------------------------------------===//
// IntLiteralToFloatLiteral
//===----------------------------------------------------------------------===//

OpFoldResult IntToFloatLiteralOp::fold(FoldAdaptor adaptor) {
  if (auto in = dyn_cast_if_present<TypedAttr>(adaptor.getInput()))
    return IntToFloatLiteralAttr::get(in.getContext(), in);
  return {};
}

//===----------------------------------------------------------------------===//
// FloatToIntLiteral
//===----------------------------------------------------------------------===//

OpFoldResult FloatToIntLiteralOp::fold(FoldAdaptor adaptor) {
  if (auto in = dyn_cast_if_present<TypedAttr>(adaptor.getInput()))
    return FloatToIntLiteralAttr::get(in.getContext(), in);
  return {};
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
