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

#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/Interpreter/InterpreterState.h"
#include "Mojo/Interpreter/ParametricInterpreterState.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/DebugStringHelper.h"

using namespace M;
using namespace KGEN;

namespace {
constexpr llvm::StringLiteral kStructElemLayoutError =
    "failed to get size/alignment for struct element type";
constexpr llvm::StringLiteral kStructTargetAlignError =
    "failed to get alignment for struct target element type";

/// Returns the offset after aligning to the type's alignment and advancing by
/// its size. Returns an error containing `errorMessage` if the size or
/// alignment cannot be determined (e.g. for unresolved parametric types).
ErrorOr<int64_t> advanceOffset(int64_t offset, DataLayoutInterface dl,
                               TargetInfoAttr target,
                               llvm::StringLiteral errorMessage) {
  std::optional<int64_t> align = dl.getTypeAlign(target);
  std::optional<int64_t> size = dl.getTypeSize(target);
  if (!align || !size)
    return Error(errorMessage);
  return llvm::alignTo(offset, *align) + *size;
}
} // namespace

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
ErrorTreeOrSuccess interpretIfContainsPtr(const Payload &payload,
                                          InterpreterState &state,
                                          Location loc) {
  if (!payload.containsPtr)
    return state.mapResults(payload.value);
  SmallVector<Attribute> attributes;
  attributes.push_back(payload.value);
  if (ErrorOrSuccess err = state.internalizeMemory(attributes); err.isError())
    return ErrorTree(loc, err.takeError());
  return state.mapResults(attributes.front());
}

template <typename Payload>
ErrorTreeOrSuccess parametricInterpretIfContainsPtr(
    const Payload &payload, ParametricInterpreterState &state, Location loc) {
  Attribute attr = state.getReboundAttribute(payload.value);
  if (!payload.containsPtr)
    return state.mapResults(attr);
  SmallVector<Attribute> attributes;
  attributes.push_back(payload.value);
  if (ErrorOrSuccess err = state.internalizeMemory(attributes); err.isError())
    return ErrorTree(loc, err.takeError());
  return state.mapResults(attributes.front());
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
  return interpretIfContainsPtr(payload, state, getLoc());
}

ErrorTreeOrSuccess
ParamConstantOp::parametric_interpret(ArrayRef<Attribute> operands,
                                      const Payload &payload,
                                      ParametricInterpreterState &state) {
  return parametricInterpretIfContainsPtr(payload, state, getLoc());
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
  return interpretIfContainsPtr(payload, state, getLoc());
}

ErrorTreeOrSuccess
ParamMaterializeOp::parametric_interpret(ArrayRef<Attribute> operands,
                                         const Payload &payload,
                                         ParametricInterpreterState &state) {

  return parametricInterpretIfContainsPtr(payload, state, getLoc());
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
      return state.mapResults(attr);
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
  if (auto boolCond = sugarDynCast<SIMDAttr>(cond)) {
    // Leave failing conditions, they must be diagnosed at elaboration time.
    if (!boolCond.getAsBool())
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

  auto resultBool = cast<SIMDAttr>(cond);
  if (!resultBool.getAsBool()) {
    Attribute message = state.getReboundAttribute(getMessage());
    return ErrorTree(getLoc(), "constraint failed: " +
                                   cast<StringAttr>(message).getValue());
  }

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
                                      op.getOperands(), op.getTailKindAttr());
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
  auto callable = dyn_cast<CallableSymbolAttrInterface>(operands[0]);
  if (!callable)
    return ErrorTree(getLoc(), "couldn't resolve kgen.call_indirect callee");

  auto bodyOr = state.lookupFunctionBody(callable.getSymbol());
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
  auto callable = dyn_cast<CallableSymbolAttrInterface>(operands[0]);
  if (!callable)
    return ErrorTree(getLoc(), "couldn't resolve kgen.call_indirect callee");

  auto bodyOr = state.lookupParametricFunctionBody(callable.getSymbol());
  if (bodyOr.isError())
    return ErrorTree(getLoc(), bodyOr.takeError());

  Region &body = *bodyOr->first;
  state.pushParamValues(callable.getParamValues(), true);
  state.pushEvalFrame(bodyOr->second, bodyOr->first, callable.getParamValues(),
                      9);
  if (auto err = state.callFunctionBody(body, operands.drop_front()))
    return err.takeError();

  state.setDeclBindings(bodyOr->second, callable.getParamValues());
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

  return state.mapResults(getCallee());
}

ErrorTreeOrSuccess
CreateClosureOp::parametric_interpret(ArrayRef<Attribute> operands,
                                      ParametricInterpreterState &state) {
  // We have no representation for closing over runtime values.
  if (!operands.empty())
    return ErrorTree(getLoc(), "TODO: cannot form a closure at compile time");

  return state.mapResults(state.getReboundAttribute(getCallee()));
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

  return state.mapResults({builder.getIndexAttr(loads),
                           builder.getIndexAttr(stores),
                           getComputeOpsAttr(ComputeKind::Addition),
                           getComputeOpsAttr(ComputeKind::Comparison),
                           getComputeOpsAttr(ComputeKind::Division),
                           getComputeOpsAttr(ComputeKind::Multiplication),
                           getComputeOpsAttr(ComputeKind::MultiplyAdd),
                           getComputeOpsAttr(ComputeKind::Other)});
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

  return state.mapResults({builder.getIndexAttr(loads),
                           builder.getIndexAttr(stores),
                           getComputeOpsAttr(ComputeKind::Addition),
                           getComputeOpsAttr(ComputeKind::Comparison),
                           getComputeOpsAttr(ComputeKind::Division),
                           getComputeOpsAttr(ComputeKind::Multiplication),
                           getComputeOpsAttr(ComputeKind::MultiplyAdd),
                           getComputeOpsAttr(ComputeKind::Other)});
}

//===----------------------------------------------------------------------===//
// IsCompileTimeOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess
IsRunInComptimeInterpreterOp::interpret(ArrayRef<Attribute> operands,
                                        InterpreterState &state) {
  // Always return true during interpreting time.
  return state.mapResults(SIMDAttr::getScalarBool(getContext(), true));
}

ErrorTreeOrSuccess IsRunInComptimeInterpreterOp::parametric_interpret(
    ArrayRef<Attribute> operands, ParametricInterpreterState &state) {
  // Always return true during interpreting time.
  return state.mapResults(SIMDAttr::getScalarBool(getContext(), true));
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
  return state.mapResults(results);
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
  return state.mapResults(results);
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess ReturnOp::interpret(ArrayRef<Attribute> operands,
                                       InterpreterState &state) {
  return state.returnFromFunction(operands);
}

ErrorTreeOrSuccess
ReturnOp::parametric_interpret(ArrayRef<Attribute> operands,
                               ParametricInterpreterState &state) {
  return state.returnFromFunction(operands);
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
      return state.mapResults(VariantAttr::get(value, getIndex(), resultType));
    }
  }

  return ErrorTree(getLoc(), "cannot interpret kgen.variant.create");
}

//===----------------------------------------------------------------------===//
// VariantIsOp
//===----------------------------------------------------------------------===//

OpFoldResult VariantIsOp::fold(FoldAdaptor adaptor) {
  if (auto variant = dyn_cast_if_present<VariantAttr>(adaptor.getVariant()))
    return SIMDAttr::getScalarBool(getContext(),
                                   variant.getIndex() == getIndex());

  if (auto createOp = getOperand().getDefiningOp<VariantCreateOp>())
    return SIMDAttr::getScalarBool(getContext(),
                                   createOp.getIndex() == getIndex());

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
    return state.mapResults(variant.getValue());
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
    if (!extract)
      return {};
    auto indexAttr = dyn_cast<IntegerAttr>(extract.getIndexAttr());
    if (!indexAttr)
      return {}; // Parametric index, can't fold.
    if (indexAttr.getInt() == idx &&
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
    return state.mapResults(cst);
  }

  return ErrorTree(getLoc(), "can't interpret POP::StructCreateOp");
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

OpFoldResult StructExtractOp::fold(FoldAdaptor adaptor) {
  auto index = dyn_cast_or_null<IntegerAttr>(adaptor.getIndexAttr());
  if (!index)
    return {};

  if (auto container = adaptor.getContainer())
    return StructExtractAttr::get(cast<TypedAttr>(container), index.getInt());
  if (auto structCreate = getOperand().getDefiningOp<StructCreateOp>())
    return structCreate.getOperand(index.getInt());
  return {};
}

ErrorTreeOrSuccess StructExtractOp::interpret(ArrayRef<Attribute> operands,
                                              InterpreterState &state) {
  return state.interpretOpWithFolder(this->getOperation(), operands);
}

ErrorTreeOrSuccess
StructExtractOp::parametric_interpret(ArrayRef<Attribute> operands,
                                      ParametricInterpreterState &state) {
  auto index =
      dyn_cast_or_null<IntegerAttr>(state.getReboundAttribute(getIndexAttr()));
  if (!index)
    return ErrorTree(getLoc(), "cannot resolve parametric index '" +
                                   debugString(getIndexAttr()) +
                                   "' to a constant");

  auto structValue = cast<TypedAttr>(operands.front());
  auto structType = cast<StructType>(structValue.getType());
  auto elementTypes = structType.getElementTypes();
  if (!elementTypes)
    return ErrorTree(getLoc(), "cannot extract from unresolved struct type");

  size_t idx = index.getInt();
  if (idx >= elementTypes->size())
    return ErrorTree(getLoc(), "struct extract index " + Twine(idx).str() +
                                   " is out of bounds for struct with " +
                                   Twine(elementTypes->size()).str() +
                                   " elements");

  auto result = StructExtractAttr::get(structValue, idx);
  return state.mapResults(result);
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
  return state.mapResults(StructAttr::get(
      values, cast<StructType>(state.getReboundType(getType()))));
}

//===----------------------------------------------------------------------===//
// StructGEPOp
//===----------------------------------------------------------------------===//

OpFoldResult StructGEPOp::fold(FoldAdaptor adaptor) {
  // Fold identity case: when container and result types are the same.
  // This can happen for single-element structs that get flattened during
  // type lowering - the struct type is replaced with its element type,
  // making the GEP a no-op.
  if (getContainer().getType() == getType())
    return getContainer();
  return {};
}

ErrorTreeOrSuccess StructGEPOp::interpret(ArrayRef<Attribute> operands,
                                          InterpreterState &state) {
  auto ptr = dyn_cast_if_present<PointerAttr>(operands.front());
  auto idxAttr = dyn_cast_if_present<IntegerAttr>(getIndex());
  if (!ptr || !idxAttr)
    return ErrorTree(getLoc(), "non-constant inputs");

  int64_t offset = 0;
  // Move the address over the elements before the one we are reading.
  unsigned index = idxAttr.getInt();
  SmallVector<Type> elementTypes;
  if (!llvm::isa<StructType>(getContainer().getType().getElementType())) {
    // Might be a single element struct which got flattened during lower-lit :(
    if (index != 0)
      return ErrorTree(getLoc(), "struct field index out of bounds");

    elementTypes.push_back(getContainer().getType().getElementType());
  } else {
    StructType structType = getContainer().getType().getElementAs<StructType>();
    std::optional<size_t> numElements = structType.getNumElements();
    if (!numElements || index >= *numElements)
      return ErrorTree(getLoc(), "struct field index out of bounds");

    std::optional<SmallVector<Type>> eltTypesOr = structType.getElementTypes();
    if (!eltTypesOr)
      return ErrorTree(getLoc(), "struct element types not resolved");
    elementTypes = std::move(*eltTypesOr);
  }

  for (unsigned i = 0; i != index; ++i) {
    ErrorOr<int64_t> next =
        advanceOffset(offset, cast<DataLayoutInterface>(elementTypes[i]),
                      state.getTarget(), kStructElemLayoutError);
    if (next.isError())
      return ErrorTree(getLoc(), next.takeError());
    offset = next.takeValue();
  }

  // Align the address to the target element.
  Type targetType = elementTypes[index];
  std::optional<int64_t> targetAlign =
      cast<DataLayoutInterface>(targetType).getTypeAlign(state.getTarget());
  if (!targetAlign)
    return ErrorTree(getLoc(), kStructTargetAlignError);
  offset = llvm::alignTo(offset, *targetAlign);
  return state.mapResults(PointerAttr::get(ptr.getAddr() + offset, getType()));
}

ErrorTreeOrSuccess
StructGEPOp::parametric_interpret(ArrayRef<Attribute> operands,
                                  ParametricInterpreterState &state) {
  auto ptr = dyn_cast_if_present<PointerAttr>(operands.front());
  auto idxAttr =
      dyn_cast_if_present<IntegerAttr>(state.getReboundAttribute(getIndex()));
  if (!ptr || !idxAttr)
    return ErrorTree(getLoc(), "non-constant inputs");

  int64_t offset = 0;
  auto structType =
      cast<KGEN::PointerType>(cast<TypedAttr>(operands[0]).getType())
          .getElementAs<StructType>();

  // Move the address over the elements before the one we are reading.
  unsigned index = idxAttr.getInt();
  auto numElements = structType.getNumElements();
  if (!numElements || index >= *numElements)
    return ErrorTree(getLoc(), "struct field index out of bounds");

  auto elementTypes = structType.getElementTypes();
  if (!elementTypes)
    return ErrorTree(getLoc(), "struct element types not resolved");

  for (unsigned i = 0; i != index; ++i) {
    ErrorOr<int64_t> next =
        advanceOffset(offset, cast<DataLayoutInterface>((*elementTypes)[i]),
                      state.getTarget(), kStructElemLayoutError);
    if (next.isError())
      return ErrorTree(getLoc(), next.takeError());
    offset = next.takeValue();
  }

  // Align the address to the target element.
  Type targetType = (*elementTypes)[index];
  std::optional<int64_t> targetAlign =
      cast<DataLayoutInterface>(targetType).getTypeAlign(state.getTarget());
  if (!targetAlign)
    return ErrorTree(getLoc(), kStructTargetAlignError);
  offset = llvm::alignTo(offset, *targetAlign);
  return state.mapResults(PointerAttr::get(ptr.getAddr() + offset,
                                           state.getReboundType(getType())));
}

//===----------------------------------------------------------------------===//
// StructLoadIndirectOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess StructLoadIndirectOp::interpret(ArrayRef<Attribute> operands,
                                                   InterpreterState &state) {
  if (auto structValue = dyn_cast<StructAttr>(operands[0])) {
    auto variadic = getType().getVariadicIfResolved();
    if (!variadic)
      return ErrorTree(getLoc(), "unknown type list");
    ArrayRef<TypedAttr> typeElts = variadic.getValues();

    SmallVector<TypedAttr> values;
    for (auto [ptr, type] : llvm::zip(structValue.getValues(), typeElts)) {
      ErrorOr<Attribute> result = state.readAttributeFromPointer(
          ptr, cast<TypeParamAttr>(type).getMlirType());
      if (result.isError())
        return ErrorTree(getLoc(), result.takeError());
      values.push_back(cast<TypedAttr>(result.takeValue()));
    }
    return state.mapResults(StructAttr::get(values, getType()));
  }
  return ErrorTree(getLoc(), "non-constant inputs");
}

ErrorTreeOrSuccess
StructLoadIndirectOp::parametric_interpret(ArrayRef<Attribute> operands,
                                           ParametricInterpreterState &state) {
  if (auto structValue = dyn_cast<StructAttr>(operands[0])) {
    auto structType = cast<StructType>(state.getReboundType(getType()));
    auto variadic = structType.getVariadicIfResolved();

    if (!variadic)
      return ErrorTree(getLoc(), "unknown type list");
    ArrayRef<TypedAttr> typeElts = variadic.getValues();

    SmallVector<TypedAttr> values;
    for (auto [ptr, type] : llvm::zip(structValue.getValues(), typeElts)) {
      ErrorOr<Attribute> result = state.readAttributeFromPointer(
          ptr, cast<TypeParamAttr>(type).getMlirType());
      if (result.isError())
        return ErrorTree(getLoc(), result.takeError());
      values.push_back(cast<TypedAttr>(result.takeValue()));
    }
    return state.mapResults(StructAttr::get(values, structType));
  }
  return ErrorTree(getLoc(), "non-constant inputs");
}

//===----------------------------------------------------------------------===//
// CodeGenReachableOp
//===----------------------------------------------------------------------===//

LogicalResult CodeGenReachableOp::canonicalize(CodeGenReachableOp op,
                                               PatternRewriter &rewriter) {
  auto cond = op.getCond();
  if (auto intCond = dyn_cast<IntegerAttr>(cond)) {
    if (intCond.getValue().isZero())
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
  return failure();
}

ErrorTreeOrSuccess CodeGenReachableOp::interpret(ArrayRef<Attribute> operands,
                                                 InterpreterState &state) {
  return success();
}

ErrorTreeOrSuccess
CodeGenReachableOp::parametric_interpret(ArrayRef<Attribute> operands,
                                         ParametricInterpreterState &state) {
  return success();
}
