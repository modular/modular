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

#include "GraphCompiler/MODialect/Analysis/ParametricTopologyAnalysis.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/ShapeFunctions.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "GraphCompiler/MOSHDialect/MOSHOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/MLIRDType.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"

#include <deque>

using namespace M;
using namespace MO;

namespace M::MO {

#define GEN_PASS_DEF_PROPAGATESHAPES

#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

} // namespace M::MO

// Collects the param decls `ty`'s shape references, for a rebind's
// outputParamDecls.
static SmallVector<KGEN::ParamDeclAttr> collectDeclsFromShape(TensorType ty) {
  llvm::SmallSetVector<KGEN::ParamDeclAttr, 4> decls;
  auto shape = ty.getShape();
  for (TypedAttr dim : shape.getValues())
    if (auto ref = dyn_cast<KGEN::ParamDeclRefAttr>(dim))
      decls.insert(KGEN::ParamDeclAttr::get(ref));
  return decls.takeVector();
}

// Normalize mo.call operands to match the callee signature by inserting
// mo.rebind where shape symbols differ across the boundary.
static void normalizeCallOperandsToCallee(GraphOp graphOp) {
  ModuleOp moduleOp = graphOp->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return;

  SymbolTable symbolTable(moduleOp);

  // Prevents redeclaring a parameter needed by multiple call sites.
  llvm::DenseSet<StringAttr> declaredInRegion;

  for (KGEN::ParamDeclAttr decl : graphOp.getInputParameters())
    declaredInRegion.insert(decl.getName());

  for (KGEN::ParamDeclAttr decl : graphOp.getResultParameters())
    declaredInRegion.insert(decl.getName());

  graphOp.walk([&](KGEN::ParamOpInterface paramOp) {
    paramOp.walkDeclarations([&](KGEN::ParamDeclAttr paramDecl) {
      declaredInRegion.insert(paramDecl.getName());
    });
  });

  graphOp.walk([&](CallOp call) {
    GraphOp callee = symbolTable.lookup<GraphOp>(call.getCallee());
    if (!callee)
      return WalkResult::advance();

    auto calleeInputs = callee.getFunctionType().getInputs();
    if (calleeInputs.size() != call->getNumOperands())
      return WalkResult::advance();

    OpBuilder b(call);
    Location loc = call.getLoc();
    for (auto [idx, operand] : llvm::enumerate(call.getOperands())) {
      auto expected = dyn_cast<TensorType>(calleeInputs[idx]);
      auto actual = dyn_cast<TensorType>(operand.getType());
      if (!expected || !actual)
        continue;

      if (expected.cloneWithoutLayout() == actual.cloneWithoutLayout())
        continue;

      // Only rebind if dtype/device agree; otherwise let verifier report.
      if ((expected.getDtype() != actual.getDtype()) ||
          (expected.getDeviceRef() != actual.getDeviceRef()))
        continue;

      SmallVector<KGEN::ParamDeclAttr> expectedDecls =
          collectDeclsFromShape(expected);
      SmallVector<KGEN::ParamDeclAttr> filteredDecls;
      for (auto decl : expectedDecls) {
        if (!declaredInRegion.contains(decl.getName()))
          filteredDecls.push_back(decl);
      }

      Value reb = RebindOp::create(b, loc, expected, operand, filteredDecls);
      call->setOperand(idx, reb);

      for (KGEN::ParamDeclAttr decl : filteredDecls)
        declaredInRegion.insert(decl.getName());
    }
    return WalkResult::advance();
  });
}

static unsigned getStorageBitWidth(Type type) {
  type = getElementTypeOrSelf(type);
  if (auto simdType = dyn_cast<KGEN::SIMDType>(type)) {
    assert(simdType.getResolvedSize() == 1 && "expected scalar SIMD type");
    type = M::getEquivalentIntegerType(type.getContext(),
                                       simdType.getResolvedDType().value());
  }
  return mlir::ConstantIntRanges::getStorageBitwidth(type);
}

namespace {

bool isParamOperatorPure(KGEN::ParamOperatorAttr paramOperator) {
  return paramOperator.getOpcode() != KGEN::POC::Apply;
}

bool isParamDeclOpPure(KGEN::ParamDeclareOp paramDeclOp) {
  if (auto paramOperatorAttr =
          dyn_cast<KGEN::ParamOperatorAttr>(paramDeclOp.getValue()))
    return isParamOperatorPure(paramOperatorAttr);
  return true;
}

// Aggressive DCE of `kgen.param.declare` ops. `useDefGraph` must be built for
// `graphOp`.
void removeDeadParamDeclares(GraphOp graphOp,
                             const KGEN::ParameterUseDefGraph &useDefGraph) {
  // Invariants:
  // Any op that has been on `worklist`
  // 1. is in `processed`
  // 2. must be alive
  DenseSet<Operation *> processed;
  SmallVector<Operation *> worklist;
  auto pushOntoWorklist = [&](Operation *op) {
    if (auto [_, inserted] = processed.insert(op); inserted)
      worklist.push_back(op);
  };

  graphOp.walk([&](Operation *op) {
    // Seed with every op except pure `kgen.param.declare`s, which may be dead.
    auto paramDeclOp = dyn_cast<KGEN::ParamDeclareOp>(op);
    if (paramDeclOp && isParamDeclOpPure(paramDeclOp))
      return;
    else
      pushOntoWorklist(op);
  });

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();

    // A live op keeps the definitions of the parameters it uses alive.
    for (KGEN::ParamDeclRefAttr usedParamDeclRef :
         useDefGraph.opUses.lookup(op)) {
      const KGEN::ParamDefinition &usedParamDef =
          useDefGraph.defs.at(usedParamDeclRef.getName());
      pushOntoWorklist(usedParamDef.defOp);
    }
  }

  graphOp.walk([&](KGEN::ParamDeclareOp paramDeclOp) {
    if (!processed.contains(paramDeclOp))
      paramDeclOp.erase();
  });
}

/// Binds graph result parameters to the output op's binding parameters. Result
/// params are declared in the signature and never materialize a
/// kgen.param.declare, so bindings are checked manually. Assumes shape
/// propagation already ran on the whole `mo.graph`.
void handleOutputBindings(GraphOp graphOp,
                          KGEN::ParameterEvaluator &evaluator) {
  SmallPtrSet<StringAttr, 4> resDeclNames;
  for (KGEN::ParamDeclAttr decl : graphOp.getResultParameters())
    resDeclNames.insert(decl.getName());
  SmallPtrSet<StringAttr, 4> argDeclNames;
  for (KGEN::ParamDeclAttr decl : graphOp.getInputParameters())
    argDeclNames.insert(decl.getName());

  auto isValidOutputDim = [&](Attribute dimAttr) {
    // Don't allow any reference to parameters declared inside the graph.
    if (auto declRef = dyn_cast<KGEN::ParamDeclRefAttr>(dimAttr))
      return argDeclNames.contains(declRef.getName());

    if (isStaticDimAttr(cast<TypedAttr>(dimAttr)))
      return true;

    return false;
  };

  auto isValidOutputShape = [&](Attribute attr) {
    if (auto declRef = dyn_cast<KGEN::ParamDeclRefAttr>(attr))
      return argDeclNames.contains(declRef.getName());

    if (auto shapeAttr = dyn_cast<MOSH::ShapeAttr>(attr))
      return llvm::all_of(shapeAttr.getValues(), isValidOutputDim);

    return false;
  };

  auto outputOp = graphOp.getOutputOp();
  for (auto [bindingParam, resultParam] :
       llvm::zip(outputOp.getParameters(), graphOp.getResultParameters())) {
    // The binding parameter is already rebound and may be a shape or dim.
    if ((isa<ShapeType>(bindingParam.getType()) &&
         isValidOutputShape(bindingParam)) ||
        isValidOutputDim(bindingParam)) {
      evaluator.setDeclBinding(resultParam.getName(), bindingParam);
    }
    // Otherwise the evaluator maps the result parameter to itself.
  }
}

/// Update the signature's result declarations and output bindings to account
/// for any output parameter simplification (e.g., maybe an old output
/// parameter `D1` is deduced to be the same as an input parameter `D0`).
void updateGraphSignature(GraphOp graphOp) {
  SmallPtrSet<StringAttr, 4> resParamNames;
  for (Type resTy : graphOp.getResultTypes())
    if (auto resTensorTy = dyn_cast<MO::TensorType>(resTy))
      llvm::set_union(resParamNames, resTensorTy.getUsedParamNames());

  auto outputOp = graphOp.getOutputOp();
  SmallVector<KGEN::ParamDeclAttr> newResultParams{};
  SmallVector<TypedAttr> newBindingParams{};
  for (auto [bindingParam, resParamDecl] :
       llvm::zip(outputOp.getParameters(), graphOp.getResultParameters())) {
    if (resParamNames.contains(resParamDecl.getName())) {
      newResultParams.emplace_back(resParamDecl);
      newBindingParams.emplace_back(bindingParam);
    }
  }

  outputOp.setParameters(newBindingParams);

  auto newResultParamsAttr =
      KGEN::ParamDeclArrayAttr::get(graphOp->getContext(), newResultParams);
  NamedAttrList attrs = graphOp->getAttrDictionary();
  attrs.set(graphOp.getResultParametersAttrName(), newResultParamsAttr);

  FuncTypeGeneratorType signature =
      KGEN::FuncTypeGeneratorType::remapToFuncTypeGenerator(
          graphOp.getInputParameters(), graphOp.getFunctionType());
  attrs.set(graphOp.getSignatureAttrName(), TypeAttr::get(signature));
  graphOp->setAttrs(attrs.getDictionary(graphOp->getContext()));
}

/// A variant class that contains result of staticizing a series of computation
/// on shapes tensors.
/// - Success (with parameter bindings and result attribute).
/// - NonStaticizable (when inputs are too dynamic).
/// - Error (when a fatal error has been detected).
class StaticizeResult {
public:
  struct Success {
    // Bindings modeling `kgen.param.declare`s of expressions like
    // `D2 = <add(D0, D1)>` or asserts like `Chain = <...assert(cond, msg)...>`,
    // which shape propagation substitutes, folds, and checks.
    SmallVector<ParamBinding> bindings;

    // One of:
    // - `IntegerAttr` for a constant integer or boolean.
    // - `ParamDeclRefAttr` for an integer or boolean parameter.
    // - `ParamOperatorAttr` for an integer or boolean parameter expression.
    // - `ShapeAttr` which is a list where each item could be any of the above.
    TypedAttr resultAttr;

    Success() = default;
    Success(SmallVector<ParamBinding> &&bindings, TypedAttr resultAttr)
        : bindings(std::move(bindings)), resultAttr(resultAttr) {}

    Success(const Success &) = delete;
    Success(Success &&) = delete;
    Success &operator=(const Success &) = delete;
    Success &operator=(Success &&) = delete;
  };
  struct NonStaticizable {};
  struct Error {
    std::string msg;

    Error(const Twine &msg) : msg(msg.str()) {}

    Error(const Error &) = delete;
    Error(Error &&) = delete;
    Error &operator=(const Error &) = delete;
    Error &operator=(Error &&) = delete;
  };

  bool isSuccess() { return std::holds_alternative<Success>(result); }
  bool isNonStaticizable() {
    return std::holds_alternative<NonStaticizable>(result);
  }
  bool isError() { return std::holds_alternative<Error>(result); }

  Success &getSuccess() { return std::get<Success>(result); }
  Error &getError() { return std::get<Error>(result); }

  static StaticizeResult success(SmallVector<ParamBinding> &&bindings,
                                 TypedAttr resultAttr) {
    return StaticizeResult(std::move(bindings), resultAttr);
  }
  static StaticizeResult nonStaticizable() { return StaticizeResult(); }
  static StaticizeResult error(const Twine &errMsg) {
    return StaticizeResult(errMsg);
  }

private:
  StaticizeResult(SmallVector<ParamBinding> &&bindings, TypedAttr resultAttr) {
    result.emplace<Success>(std::move(bindings), resultAttr);
  }
  StaticizeResult() { result = NonStaticizable{}; }
  StaticizeResult(const Twine &errMsg) { result.emplace<Error>(errMsg.str()); }
  std::variant<Success, NonStaticizable, Error> result;
};

} // namespace

namespace {

/// Try converting `val` to either
/// - a symbolic int/bool (if `val` is a scalar tensor) or
/// - a list of the above (if `val` is a 1D tensor)
///
/// `b` is used to build up and track parameter expressions, if needed.
static ErrorOr<TypedAttr> tryStaticizeImpl(Value val, ParamExprBuilder &b) {
  // Base case -- source of shape tensor.
  if (auto shapeOfOp = val.getDefiningOp<ShapeOfOp>())
    return shapeOfOp.getInput().getType().getShape();

  if (auto paramToValue = val.getDefiningOp<MOSH::ParamToValueOp>())
    if (isa<IntegerType, KGEN::SIMDType, MOSH::ShapeType>(
            paramToValue.getType()))
      return paramToValue.getValue();

  auto tensorVal = dyn_cast<TensorValue>(val);
  if (!tensorVal)
    return TypedAttr{};

  TensorType tensorTy = tensorVal.getType();
  if (!tensorTy.hasRank() || tensorTy.getRank() > 2)
    return TypedAttr{};

  if (!isa<IntegerType>(tensorTy.getElementType()))
    return TypedAttr{};

  // TODO replace `tryGetDimAttr` with `Staticization` implementation of
  // relevant ops. Examples: `D0`
  if (auto dimAttr = tryGetDimAttr(val))
    return dimAttr;

  // TODO replace `tryGetShapeAttr` with `Staticization` implementation of
  // relevant ops. Examples: `[D0, 10, ...]`
  if (auto shapeAttr = tryGetShapeAttr(val))
    return shapeAttr;

  if (auto staticization = val.getDefiningOp<Staticization>()) {
    SmallVector<TypedAttr> staticizedInputs;
    for (Value input : staticization->getOperands()) {
      ErrorOr<TypedAttr> staticizedInputOr = tryStaticizeImpl(input, b);

      if (staticizedInputOr.isError())
        return Error(staticizedInputOr.getError());

      TypedAttr staticizedInput = staticizedInputOr.get();
      if (!staticizedInput)
        return TypedAttr{};

      staticizedInputs.push_back(staticizedInput);
    }
    return staticization.tryStaticize(staticizedInputs, b);
  }

  return TypedAttr{};
}

static TypedAttr tryFoldMoshOp(Value val) {
  auto op = val.getDefiningOp();
  if (!op || !isa<M::MOSH::MOSHDialect>(op->getDialect()))
    return TypedAttr{};

  auto attr = MOSH::tryFoldOp(op);
  if (failed(attr))
    return TypedAttr{};

  return *attr;
}

/// Like `tryStaticizeImpl`, but tries the cheaper MOSH folding first.
static StaticizeResult tryStaticize(Value val) {
  if (TypedAttr folded = tryFoldMoshOp(val))
    return StaticizeResult::success({}, folded);

  SmallVector<ParamBinding> bindings;
  auto graphOp = val.getParentRegion()->getParentOfType<GraphOp>();
  ParamExprBuilder b(graphOp, val.getLoc(), &bindings);
  ErrorOr<TypedAttr> resOr = tryStaticizeImpl(val, b);
  if (resOr.isError())
    return StaticizeResult::error(resOr.getError());

  if (TypedAttr res = resOr.get())
    return StaticizeResult::success(std::move(bindings), res);

  return StaticizeResult::nonStaticizable();
}

/// Shape-infers `op` from its input shapes or staticized input data. Returns
/// an error on fatal failure (e.g. a failed shape assertion), the decomposed
/// ops (ordered by SSA and parametric dependencies), or an empty vector if the
/// shape logic can't be materialized.
static ErrorOr<SmallVector<KGEN::ParamDeclareOp>>
shapeInferOpImpl(ShapeMaterialization op, ArrayRef<TypedAttr> inputAttrs) {

  ShapeMaterializeResult result = op.materializeShapeDefs(inputAttrs);

  if (result.isNonMaterializable())
    return SmallVector<KGEN::ParamDeclareOp>();

  if (result.isError())
    return Error(result.getError().msg);

  SmallVector<ParamBinding> bindings;
  if (result.isSuccess()) {
    auto &success = result.getSuccess();
    llvm::append_range(bindings, success.bindings);

    // outputDecls is always one of
    // - Sh2: !mosh.ape
    // - [D0, D1, ...]
    ArrayRef<KGEN::ParamDeclAttr> outputDecls = op.getOutputParamDecls();

    if (outputDecls.size() == 1 &&
        isa<ShapeType>(outputDecls.front().getType())) {
      // `outputShape` has the form `[N0, N1, ...]`
      ASSERT_STREAM(success.resultShapes.size() == 1,
                    << "expect a single output");
      ShapeAttr outputShape = success.resultShapes.front();

      // Output decl is a shape:
      //   mo.reshape<() -> Sh0: !mosh.ape> : ... -> Sh0
      // binds
      //   kgen.param.decare Sh0 = <[N0, N1, ...]>
      bindings.push_back(ParamBinding(outputDecls.front(), outputShape));
    } else {
      llvm::DenseSet<ParamDeclAttr> outputDeclSet(outputDecls.begin(),
                                                  outputDecls.end());
      llvm::DenseSet<KGEN::ParamDeclRefAttr> processedDeclRefs;
      for (auto [i, res, outputShape] :
           llvm::enumerate(op->getResults(), success.resultShapes)) {
        // Output decls are dimensions:
        //   mo.reshape<() -> D1, D2> : ... -> [D1, D2]
        //   mo.reshape<() -> D1> : ... -> [10, D1]
        // bind
        //   kgen.param.decare D1 = <N0>
        //   kgen.param.decare D2 = <add(N1, 10)>
        ShapeAttr annotatedOutShape =
            cast<TensorType>(res.getType()).getShape();
        size_t returnedRank = outputShape.getRank();
        size_t annotatedRank = annotatedOutShape.getRank();
        if (returnedRank != annotatedRank) {
          std::stringstream ss;
          ss << "shape function result #" << i << " returned shape of rank "
             << returnedRank << ", but the type annotation has shape of rank "
             << annotatedRank;
          return Error(ss.str());
        }
        // Given `outputShape` for
        //   mo.reshape<() -> D1> : ... -> [10, D1]
        // Generate
        //   D1 = <outputShape[1]>
        for (auto [axis, dim] :
             llvm::enumerate(annotatedOutShape.getValues())) {
          auto paramDim = dyn_cast<KGEN::ParamDeclRefAttr>(dim);
          if (!paramDim)
            continue;

          if (auto [_, inserted] = processedDeclRefs.insert(paramDim);
              !inserted)
            continue;

          // Skip if `dim` is defined elsewhere.
          auto paramDecl = KGEN::ParamDeclAttr::get(paramDim);
          if (!outputDeclSet.contains(paramDecl))
            continue;

          bindings.push_back(ParamBinding(paramDecl, outputShape[axis]));
        }
      }
    }
  }

  OpBuilder builder(op.getContext());
  SmallVector<KGEN::ParamDeclareOp> ops;
  Location loc = op->getLoc();
  for (const ParamBinding &binding : bindings) {
    auto declOp =
        KGEN::ParamDeclareOp::create(builder, loc, binding.decl, binding.value);
    ops.push_back(declOp);
  }
  return ops;
}

struct PropagateShapes {
  /// For each op inside `graphOp` (including nested blocks within), make sure
  /// any depended ops (SSA or parameter dependencies) appear before the op.
  LogicalResult topoSortWithParameters(GraphOp graphOp) {
    ParametricTopologyAnalysis analysis(graphOp);
    return analysis.topologicallySortAllOpsWithin(&graphOp.getRegion());
  }

  /// Return the depth of a KGEN param operator, where
  /// - <add(D0, 1)> has depth 1
  /// - <add(mul(D0, 2), 1)> has depth 2
  size_t getParamAttrDepth(KGEN::ParamOperatorAttr paramOperator) {
    if (!paramAttrToDepth.contains(paramOperator)) {
      size_t depth = 0;
      for (TypedAttr operand : paramOperator.getOperands()) {
        if (auto nestedParamOperator =
                dyn_cast<KGEN::ParamOperatorAttr>(operand)) {
          depth = std::max(depth, getParamAttrDepth(nestedParamOperator));
        }
      }
      depth += 1;
      paramAttrToDepth[paramOperator] = depth;
    }
    return paramAttrToDepth.at(paramOperator);
  }

  bool canParamOperatorBeNested(KGEN::ParamOperatorAttr paramOperator) {
    if (!isParamOperatorPure(paramOperator))
      return false;

    // Bounds unfoldable expression nesting; practical models need at most
    // this depth.
    constexpr size_t kMaxDepth = 4;
    auto size = getParamAttrDepth(paramOperator);
    if (size > kMaxDepth)
      return false;

    return true;
  }

  /// Track `attr` as definitely non-negative.
  void markDefinitelyNonNegative(TypedAttr attr) {
    unsigned bitwidth = getStorageBitWidth(attr.getType());

    if (isSignedType(attr.getType()))
      ranges.try_emplace(attr, mlir::ConstantIntRanges::fromSigned(
                                   APInt::getZero(bitwidth),
                                   APInt::getSignedMaxValue(bitwidth)));
    else
      ranges.try_emplace(
          attr, mlir::ConstantIntRanges::fromUnsigned(
                    APInt::getZero(bitwidth), APInt::getMaxValue(bitwidth)));
  }

  static bool isSignedType(Type type) {
    if (auto integer = dyn_cast<IntegerType>(type))
      return integer.getSignedness() !=
             IntegerType::SignednessSemantics::Unsigned;
    if (auto simdType = dyn_cast<KGEN::SIMDType>(type)) {
      if (std::optional<KGEN::KGENDType> dtype = simdType.getResolvedDType())
        return dtype->isSInt();
    }
    return isa<IndexType>(type);
  }

  static mlir::ConstantIntRanges getMaxRange(Type type) {
    unsigned width = getStorageBitWidth(type);
    APInt umin = APInt::getMinValue(width);
    APInt umax = APInt::getMaxValue(width);
    APInt smin = width != 0 ? APInt::getSignedMinValue(width) : umin;
    APInt smax = width != 0 ? APInt::getSignedMaxValue(width) : umax;
    return mlir::ConstantIntRanges{umin, umax, smin, smax};
  }

  mlir::ConstantIntRanges
  getRange(KGEN::POC opcode, ArrayRef<TypedAttr> operands, Type resultType) {
    auto inputs = llvm::map_to_vector(
        operands, [&](TypedAttr arg) { return getRange(arg); });

    auto evaluateCmp = [&](mlir::intrange::CmpPredicate pred) {
      const mlir::ConstantIntRanges &lhs = inputs[0], &rhs = inputs[1];

      APInt min = APInt::getZero(1);
      APInt max = APInt::getAllOnes(1);

      std::optional<bool> truthValue =
          mlir::intrange::evaluatePred(pred, lhs, rhs);
      if (truthValue.has_value() && *truthValue)
        min = max;
      else if (truthValue.has_value() && !(*truthValue))
        max = min;

      return mlir::ConstantIntRanges::fromUnsigned(min, max);
    };

    switch (opcode) {
    case KGEN::POC::Add:
      return mlir::intrange::inferAdd(inputs);
    case KGEN::POC::Mul:
      return mlir::intrange::inferMul(inputs,
                                      mlir::intrange::OverflowFlags::None);
    case KGEN::POC::MulNoWrap:
      return mlir::intrange::inferMul(inputs,
                                      mlir::intrange::OverflowFlags::Nuw);
    case KGEN::POC::And:
      return mlir::intrange::inferAnd(inputs);
    case KGEN::POC::Or:
      return mlir::intrange::inferOr(inputs);
    case KGEN::POC::Xor:
      return mlir::intrange::inferXor(inputs);
    case KGEN::POC::Max:
      return isSignedType(resultType) ? mlir::intrange::inferMaxS(inputs)
                                      : mlir::intrange::inferMaxU(inputs);
    case KGEN::POC::Min:
      return isSignedType(resultType) ? mlir::intrange::inferMinS(inputs)
                                      : mlir::intrange::inferMinU(inputs);
    case KGEN::POC::Shl:
      return mlir::intrange::inferShl(inputs);
    case KGEN::POC::Shr:
      return isSignedType(resultType) ? mlir::intrange::inferShrS(inputs)
                                      : mlir::intrange::inferShrU(inputs);
    case KGEN::POC::Div:
      return isSignedType(resultType) ? mlir::intrange::inferDivS(inputs)
                                      : mlir::intrange::inferDivU(inputs);
    case KGEN::POC::Mod:
      return isSignedType(resultType) ? mlir::intrange::inferRemS(inputs)
                                      : mlir::intrange::inferRemU(inputs);
    case KGEN::POC::EQ:
      return evaluateCmp(mlir::intrange::CmpPredicate::eq);
    case KGEN::POC::LT:
      return isSignedType(operands[0].getType())
                 ? evaluateCmp(mlir::intrange::CmpPredicate::slt)
                 : evaluateCmp(mlir::intrange::CmpPredicate::ult);
    case KGEN::POC::LE:
      return isSignedType(operands[0].getType())
                 ? evaluateCmp(mlir::intrange::CmpPredicate::sle)
                 : evaluateCmp(mlir::intrange::CmpPredicate::ule);
    case KGEN::POC::Cond: {
      auto constant = inputs[0].getConstantValue();
      auto &trueCase = inputs[1];
      auto &falseCase = inputs[2];

      if (constant)
        return constant->isZero() ? falseCase : trueCase;

      return falseCase.rangeUnion(trueCase);
    }
    default:
      return getMaxRange(resultType);
    };
  }

  mlir::ConstantIntRanges getRange(TypedAttr attr) {
    if (auto it = ranges.find(attr); it != ranges.end())
      return it->second;

    // Constants have trivially inferrable ranges (IntegerAttr or scalar
    // SIMDAttr).
    if (auto cst = tryGetConstantInt(attr)) {
      unsigned bitwidth = getStorageBitWidth(attr.getType());
      return mlir::ConstantIntRanges::constant(
          APInt(bitwidth, *cst, /*isSigned=*/bitwidth != 1)); // i1 is signless
    }

    if (!isa<IntegerType, IndexType, KGEN::SIMDType>(attr.getType()))
      return mlir::ConstantIntRanges::maxRange(1);

    if (auto identical = dyn_cast<KGEN::ParamIdenticalAttr>(attr)) {
      if (identical.getNumOperands() == 2)
        return getRange(KGEN::POC::EQ, identical.getOperands(),
                        identical.getType());
      return getMaxRange(attr.getType());
    }

    // Bail out for attributes we don't recognize.
    auto exprAttr = dyn_cast<KGEN::ParamOperatorAttr>(attr);
    if (!exprAttr)
      return getMaxRange(attr.getType());

    return getRange(exprAttr.getOpcode(), exprAttr.getOperands(),
                    exprAttr.getType());
  }

  static APInt getZero(Type type) {
    unsigned bitwidth = getStorageBitWidth(type);
    return APInt::getZero(bitwidth);
  }

  static mlir::ConstantIntRanges getZeroRange(TypedAttr attr) {
    return mlir::ConstantIntRanges::constant(getZero(attr.getType()));
  }

  bool isDefinitelyGreater(TypedAttr lhs, TypedAttr rhs) {
    auto pred = isSignedType(lhs.getType()) ? mlir::intrange::CmpPredicate::sge
                                            : mlir::intrange::CmpPredicate::uge;

    mlir::ConstantIntRanges lhsRange = getRange(lhs);
    mlir::ConstantIntRanges rhsRange = getRange(rhs);

    if (auto cmpResult =
            mlir::intrange::evaluatePred(pred, lhsRange, rhsRange)) {
      if (cmpResult.value())
        return true;
    }

    return false;
  }

  /// Returns true iff `attr` is provably non-negative.
  bool isDefinitelyNonNegative(TypedAttr attr) {
    auto pred = isSignedType(attr.getType())
                    ? mlir::intrange::CmpPredicate::sge
                    : mlir::intrange::CmpPredicate::uge;

    mlir::ConstantIntRanges range = getRange(attr);
    mlir::ConstantIntRanges zero = getZeroRange(attr);

    if (std::optional<bool> cmpResult =
            mlir::intrange::evaluatePred(pred, range, zero)) {
      if (cmpResult.value())
        return true;
    }

    return false;
  }

  /// Range-based extension of the KGEN folder for `cond(operands)`, with
  /// `operands` already folded. Returns null if nothing applies.
  TypedAttr tryFoldCond(ArrayRef<TypedAttr> operands) {
    // Cond is a scalar<bool>, `getRange` need a builtin type.
    if (auto constant = getRange(operands[0]).getConstantValue())
      return constant->isZero() ? operands[2] : operands[1];
    return {};
  }

  /// Range-based extension of the KGEN folders for `opcode(operands)`, with
  /// `operands` already folded. Returns null if nothing applies.
  TypedAttr tryFoldExprAttr(KGEN::POC opcode, ArrayRef<TypedAttr> operands) {
    if (opcode == KGEN::POC::Cond)
      return tryFoldCond(operands);

    // Only fold certain binary expression for now.
    if (operands.size() != 2)
      return {};

    // If we have an integer operand, it'll be `rhs` because the order is
    // canonicalized by attribute type.
    TypedAttr lhs = operands[0];
    TypedAttr rhs = operands[1];

    if (opcode == KGEN::POC::Max) {
      // TODO: Use a pattern matcher.
      // <max(add(A, B), B)>
      auto maxAddOptimization = [&](TypedAttr maxLHS,
                                    TypedAttr maxRHS) -> TypedAttr {
        // Assume add operand is on the lhs.
        auto lhsOp = dyn_cast<M::KGEN::ParamOperatorAttr>(maxLHS);
        if (!lhsOp || lhsOp.getOpcode() != KGEN::POC::Add)
          return {};

        if (lhsOp.getNumOperands() != 2)
          return {};

        TypedAttr addLHS = lhsOp.getOperand(0);
        TypedAttr addRHS = lhsOp.getOperand(1);

        // If both params `A` and `B` are non negative, then `add(A, B) >= B`
        // and the max of both sides is `add(A, B)`
        if (!isDefinitelyNonNegative(addLHS) ||
            !isDefinitelyNonNegative(addRHS))
          return {};

        // Trick to handle `<max(add(A, B), B)>` and `<max(add(A, B), A)>`
        if (addLHS == rhs || addRHS == maxRHS)
          return maxLHS;

        return {};
      };
      if (auto maybeMaxAddOptimization = maxAddOptimization(lhs, rhs))
        return maybeMaxAddOptimization;
      if (auto maybeMaxAddOptimization = maxAddOptimization(rhs, lhs))
        return maybeMaxAddOptimization;

      if (isDefinitelyGreater(lhs, rhs))
        return lhs;

      if (isDefinitelyGreater(rhs, lhs))
        return rhs;
    } else if (opcode == KGEN::POC::Min) {
      if (isDefinitelyGreater(lhs, rhs))
        return rhs;

      if (isDefinitelyGreater(rhs, lhs))
        return lhs;
    } else if (opcode == KGEN::POC::LT || opcode == KGEN::POC::LE) {
      // Slice produces lt(add(X, Y), X); rewrite lhs < rhs as lhs - rhs < 0,
      // which symbolic add simplification can fold.
      auto zero = splatIntLiteralToSIMD(0, cast<KGEN::SIMDType>(lhs.getType()));
      auto diff = KGEN::ParamOperatorAttr::getSub(lhs, rhs);
      auto i1Type = IntegerType::get(lhs.getContext(), 1);

      auto cmp = getRange(opcode, {diff, zero}, i1Type);

      if (auto truthValue = cmp.getConstantValue())
        return KGEN::splatIntLiteralToSIMD(
            !truthValue->isZero(),
            KGEN::SIMDType::get(lhs.getContext(), 1, KGEN::KGENDType::kBool));
    }

    return {};
  }

  TypedAttr tryRangeBasedFolding(TypedAttr attr) {
    auto integerType = dyn_cast<IntegerType>(attr.getType());
    if (!integerType)
      return {};

    std::optional<APInt> constant = getRange(attr).getConstantValue();
    if (!constant)
      return {};

    return IntegerAttr::get(integerType, constant.value());
  }

  /// Applies the range-based folding recursively to every expression nested
  /// in `attr`.
  TypedAttr tryFoldExprAttr(TypedAttr attr) {
    if (auto identical = dyn_cast<KGEN::ParamIdenticalAttr>(attr)) {
      SmallVector<TypedAttr> foldedOperands;
      for (TypedAttr operand : identical.getOperands())
        foldedOperands.push_back(tryFoldExprAttr(operand));
      return KGEN::ParamIdenticalAttr::get(foldedOperands);
    }

    auto exprAttr = dyn_cast<KGEN::ParamOperatorAttr>(attr);
    if (!exprAttr)
      return attr;

    if (auto value = tryRangeBasedFolding(attr))
      return value;

    KGEN::POC opcode = exprAttr.getOpcode();
    SmallVector<TypedAttr> foldedOperands;
    for (TypedAttr operand : exprAttr.getOperands())
      foldedOperands.push_back(tryFoldExprAttr(operand));

    if (TypedAttr foldedAttr = tryFoldExprAttr(opcode, foldedOperands))
      return foldedAttr;

    // Otherwise fallback to KGEN's folding hooks.
    return KGEN::ParamOperatorAttr::get(opcode, foldedOperands);
  }

  /// Shape-infers `op`. On success, replaces it with the generated ops and
  /// pushes them onto `worklist` to be processed next.
  static ErrorOrSuccess shapeInferOp(ShapeMaterialization op,
                                     std::deque<Operation *> &worklist) {
    // Data-dependent inputs are staticized; the rest contribute their shape.
    SmallVector<ParamBinding> staticizationBindings;
    DenseSet<int32_t> dataDepInputIdxs;
    for (int32_t idx : op.getDataDependentInputIndices()) {
      ASSERT_STREAM(0 <= idx && uint32_t(idx) < op->getNumOperands(),
                    << "data dependent input index must be in bound");
      dataDepInputIdxs.insert(idx);
    }

    SmallVector<TypedAttr> inputAttrs;
    for (auto [idx, input] : llvm::enumerate(op->getOperands())) {
      if (dataDepInputIdxs.contains(idx)) {
        StaticizeResult res = tryStaticize(input);

        if (res.isNonStaticizable())
          return success();

        if (res.isError())
          return Error(res.getError().msg);

        auto &success = res.getSuccess();
        llvm::append_range(staticizationBindings, success.bindings);
        inputAttrs.push_back(success.resultAttr);

      } else {
        // Not expected, but bail if a non-data-dependent input has no shape.
        TensorValue tensorInput = dyn_cast<TensorValue>(input);
        if (!tensorInput)
          return success();

        inputAttrs.push_back(tensorInput.getType().getShape());
      }
    }

    ErrorOr<SmallVector<KGEN::ParamDeclareOp>> paramDeclOpsOr =
        shapeInferOpImpl(op, inputAttrs);
    if (paramDeclOpsOr.isError())
      return Error(paramDeclOpsOr.getError());

    SmallVector<KGEN::ParamDeclareOp> &paramDeclOps = paramDeclOpsOr.get();
    if (paramDeclOps.empty())
      return success();

    // Some ops (e.g. `mo.slice`) still consume the data-dependent input after
    // shape inference, so materialize the staticized value:
    //   %tmp = mosh.param.to_value: !mosh.ape = <[10, D0]>
    //   %input = mo.shape.to_tensor(%tmp) : !mo.tensor<[2], si32>
    // rather than
    //   %shape = mo.shape_of(%x)
    //   %d0 = mo.gather(%shape)
    //   %input = mo.concat(%ten, %d0) : ... -> !mo.tensor<[2], si32>
    OpBuilder builder(op);
    Location loc = op.getLoc();
    SmallVector<Operation *> newOps;
    for (ParamBinding &binding : staticizationBindings)
      newOps.push_back(binding.toOp(loc, builder));
    for (auto [idx, input] : llvm::enumerate(op->getOperands())) {
      // Skip inputs that aren't data-dependent or that another op sharing them
      // already materialized.
      if (!dataDepInputIdxs.contains(idx) ||
          !input.getDefiningOp<Staticization>())
        continue;

      TensorType inputTensorTy = cast<TensorType>(input.getType());
      DType inputDtype = inputTensorTy.getDtype();
      // Only si32/si64 for now; easy to extend if perf needs it.
      if (!inputDtype.isInt() ||
          !llvm::is_contained({32, 64}, inputDtype.getWidthInBits()))
        continue;

      // Create
      //   %tmp   = mosh.param.to_value: !mosh.ape = <[10, D0]>
      //   %input = mo.shape.to_tensor(%tmp) : !mo.tensor<...>
      TypedAttr inputAttr = inputAttrs[idx];
      auto paramToValOp =
          MOSH::ParamToValueOp::create(builder, loc, inputAttrs[idx]);
      Value newInputVal;
      if (isa<ShapeType>(inputAttr.getType())) {
        newInputVal = Shape::ToTensorOp::create(builder, loc, inputTensorTy,
                                                paramToValOp.getResult());
      } else {
        newInputVal = IndexToTensorOp::create(builder, loc, inputTensorTy,
                                              paramToValOp.getResult());
      }
      newOps.push_back(paramToValOp);
      newOps.push_back(newInputVal.getDefiningOp());

      input.replaceAllUsesWith(newInputVal);
    }

    // Staticization ops were already inserted by `builder`.
    for (Operation *declOp : paramDeclOps)
      builder.insert(declOp);

    ASSERT_STREAM(!isa<InvokeShapeFuncOp>(op),
                  << "We do not expect InvokeShapeFuncOp to be in the IR "
                     "before PropagateShape!");
    op.setOutputParamDecls({});

    // Push in reverse so the first new op is processed next. `op` is revisited
    // after them, once there is enough information to update its output shape.
    llvm::append_range(newOps, paramDeclOps);
    worklist.push_back(op);
    for (Operation *newOp : llvm::reverse(newOps))
      worklist.push_back(newOp);

    return success();
  }

  LogicalResult propagateShape(GraphOp graphOp) {
    if (failed(topoSortWithParameters(graphOp)))
      return failure();

    // These parameters map to themselves so the evaluator can look them up:
    // 1. Graph input parameters, e.g. `mo.graph<D0, Sh1 -> ...>`.
    // 2. Op output param decls shape inference can't resolve, e.g.
    //    `mo.reshape<() -> D0>`.
    // 3. Selected expressions, to avoid huge unfoldable ones like
    //    `max(D42, max(D772, 42, add(...)), ...))`.
    //    TODO this limits folding as well, we need a better approach.
    // (1) and (2) are the evaluator's default; (3) is handled in the loop
    // below.
    KGEN::ParameterEvaluator evaluator;
    auto markIntParamDeclAsNonNegative = [&](KGEN::ParamDeclAttr paramDecl) {
      if (MOSH::ShapeAttr::isDimType(paramDecl.getType()))
        markDefinitelyNonNegative(KGEN::ParamDeclRefAttr::get(paramDecl));
    };
    // Integer graph input parameters are dynamic dims, hence non-negative.
    llvm::for_each(graphOp.getInputParameters(), markIntParamDeclAsNonNegative);

    // TODO(#9008) inline inputs to `mo.assert`
    // TODO(#26115) replace manually via `evaluator.getReboundAttribute` per op,
    // since only parts of an op matter, and to avoid caching in both the
    // replacer and the evaluator.
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([&](KGEN::ParamDeclRefAttr declRef) -> Attribute {
      return evaluator.getReboundAttribute(declRef);
    });

    // The back op is processed next.
    std::deque<Operation *> worklist;
    graphOp.walk([&](Operation *op) {
      // The graph op itself is handled after its body.
      if (!isa<GraphOp>(op))
        worklist.push_front(op);
    });

    while (!worklist.empty()) {
      Operation *op = worklist.back();
      worklist.pop_back();

      if (auto shapeMaterialization = dyn_cast<ShapeMaterialization>(op)) {
        // Ops without output param decls fall through to the replacer below.
        if (!shapeMaterialization.getOutputParamDecls().empty()) {
          ErrorOrSuccess result = shapeInferOp(shapeMaterialization, worklist);
          if (result.isError()) {
            op->emitError(result.getError());
            return failure();
          }
          continue;
        }

      } else if (auto shapeOfOp = dyn_cast<ShapeOfOp>(op)) {
        // A still dynamically ranked input here means a real dynamic rank,
        // which isn't supported.
        if (auto inputShape = shapeOfOp.getInput().getType().getShape()) {
          // Change
          //   mo.shape_of<() -> D1> : [N1, N2] -> [D1]
          // to
          //   kgen.param.declare D1 = <2>
          //   mo.shape_of : [N1, N2] -> [D1]
          ArrayRef<KGEN::ParamDeclAttr> outParamDecls =
              shapeOfOp.getOutputParamDecls();
          if (!outParamDecls.empty()) {
            KGEN::ParamDeclAttr decl = outParamDecls.front();
            ASSERT_STREAM(outParamDecls.size() == 1 &&
                              ShapeAttr::isDimType(decl.getType()),
                          << "expect only 1 output dimension parameter");
            ImplicitLocOpBuilder builder(op->getLoc(), op);
            auto rankAttr =
                ShapeAttr::getDim(builder.getContext(), inputShape.getRank());
            auto declOp = KGEN::ParamDeclareOp::create(builder, decl, rankAttr);
            shapeOfOp.setOutputParamDecls({});
            worklist.push_back(op);
            worklist.push_back(declOp);
            continue;
          }
        }
      }

      if (auto declOp = dyn_cast<KGEN::ParamDeclareOp>(op)) {
        KGEN::ParamDeclAttr declAttr = declOp.getParamDecl();
        auto foldedRhsAttr = evaluator.getReboundAttribute(declOp.getValue());

        foldedRhsAttr = tryFoldExprAttr(foldedRhsAttr);
        declOp.setValueAttr(foldedRhsAttr);

        auto paramOpAttr = dyn_cast<KGEN::ParamOperatorAttr>(foldedRhsAttr);
        // An un-nestable expression's decl maps to itself or to an earlier
        // decl with the same expression, maximizing textual equality for later
        // simplification.
        if (paramOpAttr && !canParamOperatorBeNested(paramOpAttr)) {
          // On-the-fly CSE keeps `D0` and `D1` textually equal here:
          //   kgen.param.declare D0 = <...nested_expr_e1...>
          //   kgen.param.declare D1 = <...nested_expr_e1...>
          //   static.broadcast_to(...) : <[D0], f32> -> <[D1], f32>
          //
          // It doesn't CSE across nested regions:
          //   if {
          //     D0 = expr
          //     if {
          //       D1 = expr // we'd like this to become `D1 = D0`
          //     }
          //   }
          Region *opParentRegion = op->getParentRegion();
          if (exprToRegionToDeclRef.contains(paramOpAttr) &&
              exprToRegionToDeclRef[paramOpAttr].contains(opParentRegion)) {
            KGEN::ParamDeclRefAttr existingDeclRef =
                exprToRegionToDeclRef[paramOpAttr][opParentRegion];
            evaluator.setDeclBinding(declAttr, existingDeclRef);
            declOp.setValueAttr(existingDeclRef);

            // Otherwise record the (pure) expression and keep it un-inlined.
          } else {
            auto declRefAttr = KGEN::ParamDeclRefAttr::get(declAttr);
            if (isParamOperatorPure(paramOpAttr))
              exprToRegionToDeclRef[paramOpAttr][opParentRegion] = declRefAttr;
            evaluator.setDeclBinding(declAttr.getName(), declRefAttr);
          }

        } else {
          evaluator.setDeclBinding(declAttr, foldedRhsAttr);
        }

        // Emit compile time error for any `mo.assert false`
        if (StringAttr errMsg = tryGetAssertFalseErrMsg(foldedRhsAttr)) {
          op->emitError(errMsg.strref());
          return failure();
        }
      }

      if (auto paramFromValue = dyn_cast<MOSH::ParamFromValueOp>(op)) {
        StaticizeResult input = tryStaticize(paramFromValue.getValue());
        if (input.isError()) {
          op->emitError(input.getError().msg);
          return failure();
        }

        if (input.isSuccess()) {
          StaticizeResult::Success &success = input.getSuccess();

          OpBuilder builder(op);
          Location loc = op->getLoc();
          SmallVector<Operation *> newOps;
          for (ParamBinding &binding : success.bindings)
            newOps.push_back(binding.toOp(loc, builder));

          llvm::append_range(worklist, llvm::reverse(newOps));

          auto result = evaluator.getReboundAttribute(success.resultAttr);
          evaluator.setDeclBinding(paramFromValue.getParamDecl(), result);
        }
      }

      // Rewrite `op` with the bindings accumulated so far, letting DCE drop the
      // decls of simplified expressions.
      replacer.replaceElementsIn(op, /*replaceAttrs=*/true,
                                 /*replaceLocs=*/true, /*replaceTypes=*/true);
    }

    handleOutputBindings(graphOp, evaluator);
    replacer.replaceElementsIn(graphOp, /*replaceAttrs=*/true,
                               /*replaceLocs=*/true, /*replaceTypes=*/true);
    updateGraphSignature(graphOp);

    // DCE on `kgen.param.declare`s for better readability/debuggability.
    KGEN::ParameterUseDefGraph useDefGraph(graphOp.getRegion());
    useDefGraph.calculate(paramCache);
    removeDeadParamDeclares(graphOp, useDefGraph);

    // Convert
    //   mo.shape_of(%input) : [N1, 32] -> ...
    // to
    //   const_shape = mosh.param.to_value <N, 32>
    //   shape       = mo.shape.to_tensor(const_shape) : ...
    // to drop the false data dependency of the op and its users on `%input`.
    // In the worklist loop it would enable no further propagation, and as a
    // `ShapeOfOp` canonicalization it would fire too widely, obscuring frontend
    // lowering output when debugging.
    graphOp.walk([&](ShapeOfOp op) {
      ImplicitLocOpBuilder builder(op.getLoc(), op);
      TypedAttr inputShape = op.getInput().getType().getShape();
      auto shConst = MOSH::ParamToValueOp::create(builder, inputShape);
      auto toTensor = Shape::ToTensorOp::create(builder, op.getType(), shConst);
      op.replaceAllUsesWith(toTensor.getResult());
      op.erase();
      return WalkResult::advance();
    });

    normalizeCallOperandsToCallee(graphOp);

    return success();
  }

  PropagateShapes(KGEN::ParameterCollector::Analysis &paramCache)
      : paramCache{paramCache} {}

  KGEN::ParameterCollector::Analysis &paramCache;

  DenseMap<KGEN::ParamOperatorAttr, size_t> paramAttrToDepth{};
  DenseMap<TypedAttr, mlir::ConstantIntRanges> ranges;
  // Un-inlined expressions (kept as `kgen.param.declare X = <expr>`), mapped
  // per enclosing region to `X`.
  DenseMap<KGEN::ParamOperatorAttr, DenseMap<Region *, KGEN::ParamDeclRefAttr>>
      exprToRegionToDeclRef{};
};

} // namespace

struct PropagateShapesPass
    : public MO::impl::PropagateShapesBase<PropagateShapesPass> {
  using Base::Base;

  void runOnOperation() override {
    auto &paramCache = getAnalysis<KGEN::ParameterCollector::Analysis>();

    PropagateShapes propagator(paramCache);
    if (failed(propagator.propagateShape(getOperation())))
      signalPassFailure();
  }
};
