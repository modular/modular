//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Interfaces/LoweringState.h"
#include "GraphCompiler/Interfaces/EmitMojoState.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPUtils.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/MLIRDType.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"

#include <numeric>

using namespace M;
using namespace M::MGP;

static Value popScalarToBuiltinType(OpBuilder &builder, Location loc, Value v) {
  auto type = cast<KGEN::SIMDType>(v.getType());
  auto maybeDtype = type.getResolvedDType();
  assert(maybeDtype && "unable to resolve dtype");

  auto dtype = *maybeDtype;

  MLIRContext *ctx = builder.getContext();

  auto resultType =
      dtype.isIndex()
          ? (Type)IndexType::get(ctx)
          : (Type)M::getEquivalentIntegerType(ctx, DType(dtype.getValue()));

  return builder.createOrFold<KGEN::POP::CastToBuiltinOp>(loc, resultType, v);
}

//===----------------------------------------------------------------------===//
// MOGraphLoweringState
//===----------------------------------------------------------------------===//

ErrorOr<ElementsAttr>
MOGraphLoweringState::lowerElementsAttr(ElementsAttr elementsAttr) {
  MLIRContext *ctx = getExecuteOpBuilder().getContext();

  uint64_t alignment = static_cast<uint64_t>(getMinimumCpuAlignment());

  auto hasAlignedAttr = dyn_cast<HasAlignedBytesInterface>(elementsAttr);
  if (hasAlignedAttr == nullptr)
    return Error("unsupported constant attribute type");
  auto origType = hasAlignedAttr.getAlignedBytesType();

  alignment = std::max(alignment, origType.getAlign());

  auto finalType = AlignedBytesType::get(ctx, origType.getSize(), alignment);

  if (auto arrayAttr = dyn_cast<ArrayElementsAttr>(elementsAttr)) {
    // Reuse the data as is; the aligned_bytes type conveys the MEF alignment.
    static constexpr size_t kLargeDataThreshold = 512;
    if (arrayAttr.getRawData().size() < kLargeDataThreshold) {
      return AlignedBytesAttr::get(ctx, arrayAttr.getRawData(), alignment);
    } else {
      std::string name = "const_" + std::to_string(getExecuteNextConstId());
      ArrayRef<char> data(
          reinterpret_cast<const char *>(arrayAttr.getRawData().data()),
          arrayAttr.getRawData().size());
      return DenseResourceElementsAttr::get(
          finalType,
          DenseResourceElementsHandle::getManagerInterface(ctx).insert(
              name, mlir::UnmanagedAsmResourceBlob::allocateWithAlign(
                        data, alignment)));
    }
  }

  if (auto resourceAttr = dyn_cast<DenseResourceElementsAttr>(elementsAttr)) {
    // Reuse the blob; the type carries the alignment.
    return DenseResourceElementsAttr::get(finalType,
                                          resourceAttr.getRawHandle());
  }

  if (auto alignedAttr = dyn_cast<AlignedBytesAttr>(elementsAttr))
    return AlignedBytesAttr::get(ctx, alignedAttr.getData(), alignment);

  llvm_unreachable(
      "missing case for implementation of HasAlignedBytesInterface");
}

TypedValue<MGP::TensorSpecType>
MOGraphLoweringState::getShape(Location loc, Value loweredTensor,
                               DeviceRefAttr device) {
  OpBuilder &builder = getExecuteOpBuilder();
  return getBufferBuilder(device).tensorExtractTensorSpec(builder, loc,
                                                          loweredTensor);
}

MGP::TensorSpecGetDimOp MOGraphLoweringState::getDim(Location loc,
                                                     Value loweredTensor,
                                                     uint64_t axis,
                                                     DeviceRefAttr device) {
  OpBuilder &builder = getExecuteOpBuilder();
  // Not getShape: callers usually also need the buffer, i.e. a full
  // tensor.extract anyway.
  TypedValue<MGP::TensorSpecType> tensorSpec =
      getBufferBuilder(device).tensorExtractTensorSpec(builder, loc,
                                                       loweredTensor);
  return MGP::TensorSpecGetDimOp::create(builder, loc, /*spec=*/tensorSpec,
                                         /*axis=*/axis);
}

SmallVector<Value> MOGraphLoweringState::getDynamicDims(Location loc,
                                                        Value loweredTensor,
                                                        DeviceRefAttr device) {
  OpBuilder &builder = getExecuteOpBuilder();
  // Not getShape: callers usually also need the buffer, i.e. a full
  // tensor.extract anyway.
  TypedValue<MGP::TensorSpecType> tensorSpec =
      getBufferBuilder(device).tensorExtractTensorSpec(builder, loc,
                                                       loweredTensor);
  TensorSpecType specType = tensorSpec.getType();
  SmallVector<Value> dynDims;

  for (size_t i = 0; i < specType.getRank(); ++i)
    if (specType.isDynamicDim(i))
      dynDims.push_back(
          MGP::TensorSpecGetDimOp::create(builder, loc, tensorSpec, i));

  return dynDims;
}

static Value assertSymbolicDimensionMatchesPreviousValue(
    OpBuilder &builder, Location loc, Value previousValue, Value newValue,
    KGEN::ParamDeclRefAttr symbolicDimension, unsigned argumentIndex) {

  FailureOr<Type> targetType = previousValue.getType();
  // TODO: cast?
  Value isEqual = KGEN::POP::CmpOp::create(builder, loc, KGEN::CmpPredicate::EQ,
                                           previousValue, newValue);

  isEqual = popScalarToBuiltinType(builder, loc, isEqual);

  auto errorMessage = llvm::formatv(
      "symbolic dimension '{}' for input {} does not match prior uses "
      "of that dimension",
      symbolicDimension.getName().strref(), argumentIndex);

  return MGP::AssertOp::create(builder, loc, isEqual,
                               builder.getStringAttr(errorMessage));
}

ErrorOr<MOGraphLoweringState::ArgLoweringResult>
MOGraphLoweringState::lowerArg(Region *parentRegion, BlockArgument inArg,
                               BlockArgument outArg) {
  bool hasParams = false;
  Location loc = inArg.getLoc();

  auto isSubgraphRegion = [](Region *region) {
    if (auto graph = region->getParentOfType<MO::GraphOp>())
      return graph.getIsSubgraph();
    return false;
  };

  Value loweredArg = outArg;
  if (auto argType = MO::dynCastFromTensorLike(inArg.getType())) {
    DeviceRefAttr argDevice = MO::getDeviceOrDefault(argType);
    if (argType.hasStaticShape()) {
      // No parameters to deal with.
    } else if (auto inShapeParam =
                   dyn_cast<KGEN::ParamDeclRefAttr>(argType.getShape())) {
      // A whole-shape parameter maps to the runtime MGP shape (a tensor_spec
      // of unknown dtype), on its first occurrence only.
      hasParams = true;
      if (!params.lookupLowered(parentRegion, inShapeParam))
        params.emplaceLowered(parentRegion, inShapeParam,
                              getShape(loc, outArg, argDevice));
    } else {
      // Map each dim parameter to its runtime dimension on first occurrence.
      // Dims are extracted as a group regardless; DCE cleans up if all were
      // already mapped.
      hasParams = true;

      SmallVector<Value> dynDims = getDynamicDims(loc, outArg, argDevice);
      // Arbitrary expressions in the MO shape signature are unsupported.
      SmallVector<KGEN::ParamDeclRefAttr> inDimParams;
      for (TypedAttr inDim : argType.getDims()) {
        if (auto inDimParam = dyn_cast<KGEN::ParamDeclRefAttr>(inDim))
          inDimParams.emplace_back(inDimParam);
      }
      if (dynDims.size() != inDimParams.size()) {
        // TODO(#14531) handle param expr
        return Error("Expressions in parameter arguments are not supported");
      }

      OpBuilder &builder = getExecuteOpBuilder();
      SmallVector<Value> assertChains;
      for (auto [inDimParam, outDim] : llvm::zip_equal(inDimParams, dynDims)) {
        auto existingValue = params.lookupLowered(parentRegion, inDimParam);
        auto loweredValue = indexToDimension(builder, loc, outDim);

        if (existingValue && !isSubgraphRegion(parentRegion)) {
          // Subgraphs inherit this check from their caller.
          auto assert = assertSymbolicDimensionMatchesPreviousValue(
              builder, loc, existingValue, loweredValue, inDimParam,
              inArg.getArgNumber());

          assertChains.push_back(assert);
        } else {
          params.emplaceLowered(parentRegion, inDimParam, loweredValue);
        }
      }

      if (!assertChains.empty()) {
        auto consolidatedChains =
            GRT::ChainCreateOp::create(builder, loc, assertChains);

        auto guarded =
            MGP::GuardOp::create(builder, loc, consolidatedChains, outArg);

        loweredArg = guarded.getResult(0);
      }
    }
  }
  return ArgLoweringResult{hasParams, loweredArg};
}

ErrorOr<Value> MOGraphLoweringState::lowerExpression(Region *parentRegion,
                                                     Location loc,
                                                     TypedAttr attr) {
  auto newValOr = params.lowerExpression(
      getExecuteOpBuilder(), loc, parentRegion, attr,
      [this](auto &&...args) { return this->lowerExpressionImpl(args...); });

  if (newValOr.isError())
    return newValOr.takeError();
  auto newVal = std::move(*newValOr);

  // Remember placeholders in case an SSA value maps to one later.
  if (auto placeholder = newVal.getDefiningOp<ParamPlaceholderOp>())
    placeholderMap.try_emplace(placeholder, SmallVector<Value>{});
  return newVal;
}

ErrorOrSuccess MOGraphLoweringState::lowerParamDecl(
    Region *parentRegion, KGEN::ParamDeclAttr declAttr, Value value) {
  Value maybePlaceholder = params.lowerParamDecl(parentRegion, declAttr, value);

  if (maybePlaceholder) {
    // A resolved placeholder must never be mapped to again.
    if (auto it = placeholderMap.find(maybePlaceholder);
        it != placeholderMap.end()) {
      for (Value key : it->second) {
        execute.loweredValues.erase(key);
        execute.loweredValues.try_emplace(key, value);
      }
      placeholderMap.erase(it);
    }

    maybePlaceholder.replaceAllUsesWith(value);
    maybePlaceholder.getDefiningOp<ParamPlaceholderOp>().erase();
  }
  return success();
}

ErrorOr<SmallVector<Value>> MOGraphLoweringState::lowerShapeAttrDynDims(
    Region *parentRegion, MOSH::ShapeAttr shapeAttr, Location loc) {
  SmallVector<Value> runtimeDims;
  for (TypedAttr dim : shapeAttr.getValues()) {
    if (!MOSH::ShapeAttr::getStaticDim(dim).has_value()) {
      auto vOr = lowerDimensionExpression(params, getExecuteOpBuilder(), loc,
                                          parentRegion, dim);
      if (vOr.isError())
        return vOr.takeError();
      runtimeDims.emplace_back(vOr.takeValue());
    }
  }
  return runtimeDims;
}

/// Reduces `values` with binary ops, lowering variadic KGEN::POC ops.
template <typename Op>
static ErrorOr<Value> buildVariadicOp(OpBuilder &builder, Location loc,
                                      ValueRange operands, Type resultType) {
  if (operands.empty())
    return Error("Variadic op without operands");

  Value start = operands.front();
  ValueRange rest = operands.drop_front();

  auto createBinop = [&](Value lhs, Value rhs) {
    return Op::create(builder, loc, resultType, lhs, rhs);
  };

  return std::accumulate(rest.begin(), rest.end(), start, createBinop);
}

/// Convert a KGEN::POC operation on !kgen.scalar values to the POP dialect.
static ErrorOr<Value> pocToMopInteger(KGEN::POC poc, OpBuilder &builder,
                                      Location loc, ValueRange operands,
                                      Type resultType) {
  auto getCmpPredicate = [](KGEN::POC poc) -> KGEN::CmpPredicate {
    switch (poc) {
    case KGEN::POC::EQ:
      return KGEN::CmpPredicate::EQ;
    case KGEN::POC::LT:
      return KGEN::CmpPredicate::LT;
    case KGEN::POC::LE:
      return KGEN::CmpPredicate::LE;
    default:
      llvm_unreachable("unrecognized predicate");
    }
  };

  switch (poc) {
  // All comparisons lower to `pop.cmp`
  case KGEN::POC::EQ:
  case KGEN::POC::LT:
  case KGEN::POC::LE:
    return KGEN::POP::CmpOp::create(builder, loc, resultType,
                                    getCmpPredicate(poc), operands[0],
                                    operands[1]);

  // All potentially variadic ops
  case KGEN::POC::Add:
    return buildVariadicOp<KGEN::POP::AddOp>(builder, loc, operands,
                                             resultType);
  case KGEN::POC::Div:
    return buildVariadicOp<KGEN::POP::DivOp>(builder, loc, operands,
                                             resultType);
  case KGEN::POC::Max:
    return buildVariadicOp<KGEN::POP::MaxOp>(builder, loc, operands,
                                             resultType);
  case KGEN::POC::Min:
    return buildVariadicOp<KGEN::POP::MinOp>(builder, loc, operands,
                                             resultType);
  case KGEN::POC::Mul:
  case KGEN::POC::MulNoWrap:
    return buildVariadicOp<KGEN::POP::MulOp>(builder, loc, operands,
                                             resultType);
  case KGEN::POC::And:
    return buildVariadicOp<KGEN::POP::SIMDAndOp>(builder, loc, operands,
                                                 resultType);
  case KGEN::POC::Or:
    return buildVariadicOp<KGEN::POP::SIMDOrOp>(builder, loc, operands,
                                                resultType);
  case KGEN::POC::Xor:
    return buildVariadicOp<KGEN::POP::SIMDXOrOp>(builder, loc, operands,
                                                 resultType);

  // Remaining ops
  case KGEN::POC::Shl:
    return KGEN::POP::ShlOp::create(builder, loc, resultType, operands);
  case KGEN::POC::Shr:
    return KGEN::POP::ShrOp::create(builder, loc, resultType, operands);
  case KGEN::POC::Mod:
    return KGEN::POP::RemOp::create(builder, loc, resultType, operands);
  case KGEN::POC::Cond: {
    auto condition = operands[0];
    auto trueValue = operands[1];
    auto falseValue = operands[2];
    return KGEN::POP::SIMDSelectOp::create(builder, loc, resultType, condition,
                                           trueValue, falseValue);
  }
  default:
    return Error(Twine("unsupported POC '") + stringifyPOC(poc) + "'");
  }
}

ErrorOr<Value>
MOGraphLoweringState::lowerParamOperator(OpBuilder &builder, Location loc,
                                         KGEN::ParamOperatorAttr opAttr,
                                         ValueRange operands) {
  auto resultType = opAttr.getType();

  ASSERT_STREAM(isa<KGEN::SIMDType>(resultType),
                "must be a simd parameter expression");

  SmallVector<Value> popOperands;
  popOperands.reserve(operands.size());

  for (auto operand : operands) {
    ASSERT_STREAM(isa<KGEN::SIMDType>(operand.getType()),
                  "must be a simd parameter expression");
    popOperands.push_back(operand);
  }

  auto maybeResult = pocToMopInteger(opAttr.getOpcode(), builder, loc,
                                     popOperands, resultType);

  if (maybeResult.isError())
    return maybeResult;

  return maybeResult.takeValue();
}

Operation *MOGraphLoweringState::lowerKGENConstantAttr(OpBuilder &builder,
                                                       Location loc,
                                                       TypedAttr value) {
  return KGEN::ParamConstantOp::create(builder, loc, value);
}

/// Sequenced assert lowering helpers.
namespace {

bool followOperands(Operation *op) {
  // POP::And/Or/XOrOp rather than index::*, to handle more than IndexType.
  return isa<KGEN::POP::POPDialect>(op->getDialect());
}

DenseSet<Operation *> getAncestors(Operation *root) {

  DenseSet<Operation *> ancestors;
  SmallVector<Operation *> worklist;

  worklist.push_back(root);

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();

    if (!op || !followOperands(op))
      continue;

    ancestors.insert(op);

    for (Value val : op->getOperands()) {
      Operation *defOp = val.getDefiningOp();

      if (!ancestors.contains(defOp))
        worklist.push_back(defOp);
    }
  }

  return ancestors;
}

SmallVector<OpOperand *>
getBoundaryOpOperands(Operation *root,
                      const llvm::DenseSet<Operation *> &prevAncestors) {

  DenseSet<OpOperand *> boundaryOpOperands;
  SmallVector<Operation *> worklist;

  worklist.push_back(root);

  while (!worklist.empty()) {

    Operation *op = worklist.pop_back_val();

    if (!op || !followOperands(op) || prevAncestors.contains(op))
      continue;

    for (OpOperand &opOperand : op->getOpOperands()) {
      auto defOp = opOperand.get().getDefiningOp();
      if (!defOp)
        continue;

      const bool isTrivial = defOp->hasTrait<OpTrait::ConstantLike>() ||
                             isa<mlir::CastOpInterface>(defOp);

      if (prevAncestors.contains(defOp) && !isTrivial)
        boundaryOpOperands.insert(&opOperand);
      else
        worklist.push_back(defOp);
    }
  }

  return {boundaryOpOperands.begin(), boundaryOpOperands.end()};
}

/// Returns the OpOperands crossing from the previous assertion's condition tree
/// into the current assertion's.
SmallVector<OpOperand *> getSequencedAssertBoundaryOperands(
    Value cond, const SmallVector<Operation *> &parentAsserts) {

  DenseSet<Operation *> allAncestors;

  for (auto parentAssert : parentAsserts) {
    auto prevAssertCondRoot =
        cast<MGP::AssertOp>(parentAssert).getCond().getDefiningOp();

    auto prevAncestors = getAncestors(prevAssertCondRoot);

    allAncestors.insert(prevAncestors.begin(), prevAncestors.end());
  }

  return getBoundaryOpOperands(cond.getDefiningOp(), allAncestors);
}

SmallVector<Operation *>
getParentAsserts(Operation *currentAssert,
                 const DenseMap<Operation *, Operation *> &parentMap) {
  SmallVector<Operation *> parents = {currentAssert};

  auto it = parentMap.find(currentAssert);
  bool hasParent = it != parentMap.end();

  while (hasParent) {
    currentAssert = it->second;
    parents.push_back(currentAssert);
    it = parentMap.find(currentAssert);
    hasParent = it != parentMap.end();
  }

  return parents;
}

} // namespace

ErrorOr<Value> MOGraphLoweringState::lowerDimensionExpression(
    ParameterLoweringState &state, OpBuilder &builder, Location loc,
    Region *parentRegion, TypedAttr dim) {
  auto dimValOr = state.lowerExpression(
      builder, loc, parentRegion, dim,
      [this](auto &&...args) { return this->lowerExpressionImpl(args...); });

  if (failed(dimValOr))
    return dimValOr;

  return dimensionToIndex(builder, loc, dimValOr.takeValue());
}

ErrorOr<Value> MOGraphLoweringState::lowerExpressionImpl(
    ParameterLoweringState &state, OpBuilder &builder, Location loc,
    Region *parentRegion, TypedAttr attr, ValueRange operands) {
  Operation *newOp;
  if (isa<IntegerAttr, KGEN::SIMDAttr>(attr)) {
    newOp = lowerKGENConstantAttr(builder, loc, attr);
  } else if (auto cast = dyn_cast<KGEN::CastFromBuiltinAttr>(attr)) {
    ErrorOr<Value> castFrom = lowerExpression(parentRegion, loc, cast.getArg());
    if (failed(castFrom))
      return castFrom;
    newOp = KGEN::POP::CastFromBuiltinOp::create(builder, loc, cast.getType(),
                                                 castFrom.takeValue());
  } else if (auto opAttr = dyn_cast<KGEN::ParamOperatorAttr>(attr)) {
    return lowerParamOperator(builder, loc, opAttr, operands);
  } else if (auto identical = dyn_cast<KGEN::ParamIdenticalAttr>(attr)) {
    // Scalar int-like `eq` canonicalizes to `identical`, which lowers the same
    // way: a compare of the two operands.
    ASSERT_STREAM(identical.getNumOperands() == 2,
                  "expected binary identity from scalar eq");
    return pocToMopInteger(KGEN::POC::EQ, builder, loc, operands,
                           identical.getType());
  } else if (auto opAttr = dyn_cast<KGEN::MLIROpAttr>(attr)) {
    StringAttr opName = opAttr.getName();
    if (opName != MO::AssertOp::getOperationName()) {
      return Error(Twine("cannot lower mlir operation '") + opName.strref() +
                   "' in parameter expression");
    }

    auto [cond, maybeParentAssert] = [&]() -> std::pair<Value, Operation *> {
      bool hasChain = isa<MEF::ChainType>(operands[0].getType());

      if (hasChain) {
        Value chain = operands[0];
        Value cond = operands[1];

        ASSERT_STREAM(
            isa<MGP::AssertOp>(chain.getDefiningOp()),
            << "Encountered a mo.assert with a chain that was not produced by "
               "another mo.assert!");

        SmallVector<Operation *> parentAsserts =
            getParentAsserts(chain.getDefiningOp(), execute.parentAssert);

        auto bOperands =
            getSequencedAssertBoundaryOperands(cond, parentAsserts);

        // Guard only shared values: mgp.guard has no Mojo implementation, so
        // it blocks sequence fusion.
        if (!bOperands.empty()) {
          SmallVector<Value> v;
          for (OpOperand *opOperand : bOperands) {
            v.push_back(opOperand->get());
          }

          ValueRange guardedValRange{v};

          auto guardOp = MGP::GuardOp::create(
              builder, loc, guardedValRange.getTypes(), chain, guardedValRange);

          for (auto &&[guarded, original] :
               llvm::zip(guardOp.getResults(), bOperands)) {
            original->assign(guarded);
          }
        }

        return {cond, chain.getDefiningOp()};
      } else {
        return {operands[0], nullptr};
      }
    }();

    if (cond.getType() != builder.getIntegerType(1)) {
      assert(KGEN::isScalarOf<KGEN::KGENDType::kBool>(cond.getType()));
      cond = builder.createOrFold<KGEN::POP::CastToBuiltinOp>(
          loc, builder.getIntegerType(1), cond);
    }
    // An assert. The dictionary comes from an arbitrary op, so split the
    // inherent `message` into properties by name.
    MGP::AssertOp::Properties properties;
    SmallVector<NamedAttribute> discardableAttributes;
    StringRef messageName = MGP::AssertOp::getAttributeNames().front();
    for (NamedAttribute namedAttr : opAttr.getAttrs()) {
      if (namedAttr.getName() == messageName)
        properties.setMessage(::M::cast<StringAttr>(namedAttr.getValue()));
      else
        discardableAttributes.push_back(namedAttr);
    }

    MGP::AssertOp assertOp = MGP::AssertOp::create(
        builder, loc, ValueRange{cond}, properties, discardableAttributes);

    newOp = assertOp;
    if (maybeParentAssert)
      execute.parentAssert[newOp] = maybeParentAssert;
  } else if (auto shapeAttr = dyn_cast<MOSH::ShapeAttr>(attr)) {
    SmallVector<Value> dimVals;
    for (TypedAttr dim : shapeAttr.getValues()) {
      auto dimValOr =
          lowerDimensionExpression(state, builder, loc, parentRegion, dim);

      if (dimValOr.isError())
        return dimValOr.takeError();
      dimVals.emplace_back(std::move(*dimValOr));
    }

    // MGP has no runtime shape type; use a fully dynamic spec.
    SmallVector<std::optional<uint64_t>> dummyShape(dimVals.size());
    auto specTy =
        MGP::TensorSpecType::get(loc.getContext(), dummyShape, DType::si64);
    auto specOp =
        MGP::TensorSpecCreateOp::create(builder, loc, specTy, dimVals);
    return specOp.getResult();
  } else {
    return Error("cannot lower parameter expression");
  }

  return newOp->getResult(0);
}

/// Returns `T` for `!mogg.async<T>`, else `type`.
static Type getAsyncElementOrSelf(Type type) {
  if (auto asyncType = dyn_cast<MOGG::AsyncType>(type))
    return asyncType.getElementType();
  return type;
}

/// Rewires consumers of `asyncValue` to its synchronous `projection`, except
/// the projection itself and `mogg.async.unpack` ops.
static void rewireAsyncConsumersToProjection(Value asyncValue,
                                             Value projection) {
  Operation *projectionOp = projection.getDefiningOp();
  asyncValue.replaceUsesWithIf(projection, [&](mlir::OpOperand &operand) {
    Operation *owner = operand.getOwner();
    return owner != projectionOp && !isa<MOGG::AsyncUnpackOp>(owner);
  });
}

void MOGraphLoweringState::lowerMOGGABI(MOGG::AsyncRoutineOp kernel) {
  auto builder = OpBuilder::atBlockBegin(&kernel.getBodyBlock());

  auto loc = kernel->getLoc();
  MOGG::OutputOp terminator = kernel.getBodyTerminator();

  // Chain inputs should never be used; replace them with a dummy chain.
  for (BlockArgument arg : kernel.getBodyBlock().getArguments()) {
    if (isa<MEF::ChainType>(getAsyncElementOrSelf(arg.getType()))) {
      arg.replaceAllUsesWith(
          MOGG::ChainDummyOp::create(builder, loc).getResult());
    }
  }

  // Unpack inputs before packing outputs, which inspects the unpacked body
  // operands.
  builder.setInsertionPointToStart(&kernel.getBody().front());
  for (auto v : kernel.getBody().getArguments()) {
    if (isa<MEF::ChainType>(getAsyncElementOrSelf(v.getType())))
      continue;

    auto unpackOp = MOGG::AsyncUnpackOp::create(builder, loc, v);
    rewireAsyncConsumersToProjection(v, unpackOp);
  }

  builder.setInsertionPoint(terminator);
  // Chain AsyncReadyOps go after all other packs so consumers wait for every
  // output.
  SmallVector<std::pair<size_t, Value>> deferredChainOps;

  for (auto [i, value] : llvm::enumerate(terminator->getOperands())) {
    if (isa<MEF::ChainType>(value.getType())) {
      deferredChainOps.emplace_back(i, value);
      continue;
    }

    // Non-chain outputs carry their own storage handle, so all take the owned
    // pack path.
    Value newValue = MOGG::AsyncPackOp::create(
        builder, loc, MOGG::AsyncType::get(value.getType()), value);

    value.replaceUsesWithIf(
        newValue, [&](OpOperand &use) { return use.getOwner() == terminator; });

    // Sink chains so a sole chain output isn't DCE-ed.
    if (isa<MEF::ChainType>(value.getType()) &&
        !isa<MOGG::AsyncReadyOp>(value.getDefiningOp()))
      MOGG::ChainSinkOp::create(builder, loc, value);
  }

  for (auto [i, value] : deferredChainOps) {
    Value newValue = MOGG::AsyncReadyOp::create(builder, loc);
    value.replaceUsesWithIf(
        newValue, [&](OpOperand &use) { return use.getOwner() == terminator; });

    // Sink chains so a sole chain output isn't DCE-ed.
    if (!isa<MOGG::AsyncReadyOp>(value.getDefiningOp()))
      MOGG::ChainSinkOp::create(builder, loc, value);
  }
}

//===----------------------------------------------------------------------===//
// MOOpLoweringState
//===----------------------------------------------------------------------===//

SmallVector<Value> MOOpLoweringState::getLoweredOperands() {
  SmallVector<Value> loweredOperands;
  for (Value operand : opBeingLowered->getOperands()) {
    Value loweredInput = loweringState.lookupExecuteLoweredValue(operand);

    if (isa<MO::TensorType>(operand.getType()) ||
        isa<MO::BufferType>(operand.getType())) {
      BufferizedTensor bufferizedInput = getBufferizedInput(operand);
      loweredOperands.push_back(bufferizedInput.buffer);
    } else {
      loweredOperands.push_back(loweredInput);
    }
  }
  return loweredOperands;
}

Value MOOpLoweringState::getGuardChain(Value chain) {
  SmallVector<Value> inputChains;
  for (Value operand : opBeingLowered->getOperands()) {
    if (Value c = loweringState.getChainGuardingValueOrNull(operand))
      inputChains.push_back(c);
  }
  if (!inputChains.empty()) {
    inputChains.push_back(chain);
    chain = create<GRT::ChainCreateOp>(inputChains).getResult();
  }
  return chain;
}

FailureOr<SmallVector<MOOutputStorage>>
MOOpLoweringState::createOutputStorage(Value inChain) {
  FailureOr<SmallVector<TypedValue<MGP::TensorSpecType>>> outSpecsOr =
      createOutputTensorSpecs();
  if (failed(outSpecsOr))
    return failure();
  return createOutputStorage(inChain, outSpecsOr.value());
}

FailureOr<SmallVector<MOOutputStorage>> MOOpLoweringState::createOutputStorage(
    Value inChain, ArrayRef<TypedValue<MGP::TensorSpecType>> outSpecs) {
  SmallVector<BufferizedTensor> bufferizedTensors;
  bufferizedTensors.reserve(outSpecs.size());

  // In-place allocations get no mgp storage.
  DenseSet<int> inplaceAllocIndicesSet;

  SmallVector<Value> results;
  for (auto [idx, res] : llvm::enumerate(opBeingLowered->getResults())) {
    if (!inplaceAllocIndicesSet.contains(idx) &&
        !isa<MO::ChainType, MO::OpaqueType>(res.getType()))
      results.push_back(opBeingLowered->getResult(idx));
  }

  for (auto [spec, result] : llvm::zip(outSpecs, results)) {
    auto bufOr = createBuffer(inChain, spec, result.getType());
    if (failed(bufOr))
      return failure();
    bufferizedTensors.emplace_back(*bufOr);
  }
  return bufferizedTensors;
}

void MOOpLoweringState::createOutputs(
    Value inChain, SmallVector<MOOutputStorage> &loweredOutputsStorage) {
  for (const auto &[idx, outputStorage] :
       llvm::enumerate(loweredOutputsStorage)) {
    createOutputTensor(opBeingLowered->getResult(idx), inChain,
                       outputStorage.buffer, outputStorage.tensorSpec);
  }
}

BufferizedTensor MOOpLoweringState::getBufferizedInput(Value moTensor) {
  Value loweredTensor = loweringState.lookupExecuteLoweredValue(moTensor);
  ASSERT_STREAM(loweredTensor, << "referenced SSA value has not been lowered");

  MO::TensorType ty = MO::castFromTensorLike(moTensor.getType());
  MGPBufferBuilder &bufferBuilder = getStorageBufferBuilder(ty);
  auto buffer = bufferBuilder.tensorExtractBuffer(builder, loc, loweredTensor);
  auto tensorSpec =
      bufferBuilder.tensorExtractTensorSpec(builder, loc, loweredTensor);

  // The buffer shares the tensor's chain.
  loweringState.guardValueWithChain(
      buffer, loweringState.getChainGuardingValueOrNull(loweredTensor));

  return BufferizedTensor(buffer, tensorSpec);
}

FailureOr<TypedValue<MGP::TensorSpecType>>
MOOpLoweringState::createTensorSpec(MO::TensorType moTensorType) {
  MGP::TensorSpecType tensorSpecType = MGP::createTensorSpecType(moTensorType);
  if (tensorSpecType.getSizeInBytes()) {
    return MGP::TensorSpecCreateOp::create(builder, loc,
                                           /*spec=*/tensorSpecType)
        .getResult();
  }

  auto runtimeDimsOr = loweringState.lowerShapeAttrDynDims(
      opBeingLowered->getParentRegion(), moTensorType.getShape(), loc);
  if (runtimeDimsOr)
    return opBeingLowered->emitOpError(runtimeDimsOr.getError());
  return MGP::TensorSpecCreateOp::create(builder, loc,
                                         /*spec=*/tensorSpecType,
                                         /*runtimeDims=*/*runtimeDimsOr)
      .getResult();
}

FailureOr<SmallVector<TypedValue<MGP::TensorSpecType>>>
MOOpLoweringState::createOutputTensorSpecs() {
  SmallVector<TypedValue<MGP::TensorSpecType>> tensorSpecs;
  tensorSpecs.reserve(opBeingLowered->getNumResults());

  DenseSet<int> inplaceAllocIndicesSet;

  for (auto [idx, type] :
       llvm::enumerate(opBeingLowered->getResults().getType())) {
    if (inplaceAllocIndicesSet.contains(idx))
      continue;

    // The kernel lowering handles these separately.
    if (isa<MO::ChainType, MO::OpaqueType, MOGG::ShapeFuncResultType>(type))
      continue;

    auto moTensorType = dyn_cast<MO::TensorType>(type);
    ASSERT_STREAM(moTensorType, << "result of op is not a !mo.tensor");
    FailureOr<TypedValue<MGP::TensorSpecType>> tensorSpecOr =
        createTensorSpec(moTensorType);
    if (failed(tensorSpecOr))
      return failure();
    tensorSpecs.emplace_back(*tensorSpecOr);
  }
  return tensorSpecs;
}

FailureOr<BufferizedTensor> MOOpLoweringState::createBuffer(
    Value inChain, TypedValue<TensorSpecType> tensorSpec, Type tensorType) {
  MGPBufferBuilder &bufferBuilder = getStorageBufferBuilder(tensorType);
  MO::TensorType moTensorType = MO::dynCastFromTensorLike(tensorType);
  ASSERT_STREAM(moTensorType, << "input is not a !mo.tensor or !mo.buffer");

  Value numBytes = bufferSizeBytes(tensorSpec, moTensorType);

  auto bufferDevice = bufferBuilder.getDevice();
  auto preferredAlignment =
      M::Driver::getPreferredAlignment(bufferDevice.getLabel().str());
  if (bufferDevice.getLabel() == M::kCPULabel)
    preferredAlignment = std::gcd(preferredAlignment, minimumCpuAlignment);
  auto deviceContext = loweringState.getExecuteDevice(bufferDevice);
  return BufferizedTensor(
      bufferBuilder.bufferAlloc(builder, loc, context, inChain, numBytes,
                                deviceContext, preferredAlignment),
      tensorSpec);
}

void MOOpLoweringState::createOutputTensor(
    Value moTensor, Value inChain, Value buffer,
    TypedValue<MGP::TensorSpecType> tensorSpec) {
  bool isMut = isa<MO::BufferType>(moTensor.getType());

  Value loweredTensor =
      getStorageBufferBuilder(moTensor.getType())
          .tensorCreate(builder, loc, context, inChain, buffer, tensorSpec,
                        /*mut=*/isMut);
  loweringState.emplaceExecuteLoweredBufferValue(moTensor, loweredTensor,
                                                 inChain);
}

void MOOpLoweringState::createOutputTensors(
    Value inChain, SmallVector<Value> &buffers,
    SmallVector<TypedValue<MGP::TensorSpecType>> &tensorSpecs) {
  for (const auto &[idx, buffer, tensorSpec] :
       llvm::enumerate(buffers, tensorSpecs))
    createOutputTensor(opBeingLowered->getResult(idx), inChain, buffer,
                       tensorSpec);
}

MGPBufferBuilder &
MOOpLoweringState::getStorageBufferBuilder(Type moTensorType) {
  auto type = MO::dynCastFromTensorLike(moTensorType);
  ASSERT_STREAM(type, << "expecting a !mo.tensor or !mo.buffer type");
  return loweringState.getBufferBuilder(MO::getDeviceOrDefault(type));
}
