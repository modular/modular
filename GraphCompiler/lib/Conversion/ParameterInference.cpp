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

#include "GraphCompiler/Conversion/ParameterInference.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MojoParameters.h"
#include "GraphCompiler/MOGGDialect/Support/MojoIdentifiers.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENPogUtils.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/LITDialect/LITUtils.h"

using namespace mlir;

namespace M {

Attribute normalizeKernelParameter(Attribute value) {
  if (auto vectorAttr = dyn_cast<DenseIntElementsAttr>(value)) {
    SmallVector<Attribute> intAttrs;
    for (const APInt &intVal : vectorAttr.getValues<APInt>())
      intAttrs.push_back(IntegerAttr::get(
          IntegerType::get(vectorAttr.getContext(), intVal.getBitWidth()),
          intVal));
    return ArrayAttr::get(vectorAttr.getContext(), intAttrs);
  }
  if (auto typeAttr = dyn_cast<TypeAttr>(value))
    return KGEN::DTypeConstantAttr::get(
        typeAttr.getContext(), MO::getDTypeOrInvalid(typeAttr.getValue()));
  // An enum attribute is not a `TypedAttr`; kernels take the raw value.
  if (auto coord = dyn_cast<MO::CoordinateTransformModeAttr>(value))
    return IntegerAttr::get(IndexType::get(coord.getContext()),
                            static_cast<int64_t>(coord.getValue()));
  return value;
}

// Already the kernel's parameters: nothing to select, only spellings to fix.
static DictionaryAttr normalizeParameterDict(DictionaryAttr params,
                                             OpBuilder &builder) {
  if (!params)
    return builder.getDictionaryAttr({});
  SmallVector<NamedAttribute> normalized;
  for (NamedAttribute attr : params)
    normalized.push_back(NamedAttribute(
        attr.getName(), normalizeKernelParameter(attr.getValue())));
  return builder.getDictionaryAttr(normalized);
}

DictionaryAttr inferParameters(Operation *op, OpBuilder &builder) {
  if (auto custom = dyn_cast<MO::CustomOp>(op))
    return normalizeParameterDict(custom.getParameters(), builder);

  // `kgenParams` is not a `TypedAttr`, so the scrape below would drop it.
  if (auto layoutTransform = dyn_cast<MO::LayoutTransformOp>(op))
    return normalizeParameterDict(layoutTransform.getKgenParamsAttr(), builder);

  SmallVector<NamedAttribute> params;
  for (NamedAttribute attr : op->getAttrs()) {
    Attribute value = attr.getValue();
    if (isa<M::DeviceRefAttr, KGEN::ParamDeclArrayAttr>(value))
      continue;
    // An all-integer `ArrayAttr` (e.g. a fused slice's `static_starts`) is not
    // a `TypedAttr`, but is a valid comptime `IntTuple` kernel parameter.
    if (auto arrayAttr = dyn_cast<ArrayAttr>(value)) {
      if (llvm::all_of(arrayAttr,
                       [](Attribute a) { return isa<IntegerAttr>(a); }))
        params.push_back(NamedAttribute(attr.getName(), value));
      continue;
    }
    // Not `TypedAttr`s, but the normalizer knows how to spell them.
    if (!isa<TypedAttr, TypeAttr, MO::CoordinateTransformModeAttr>(value))
      continue;
    params.push_back(
        NamedAttribute(attr.getName(), normalizeKernelParameter(value)));
  }
  appendStaticOperandParameters(op, builder, params);
  return builder.getDictionaryAttr(params);
}

// TODO(GEX-2177): Remove this by adding an opt-in in the implementation and
// generically supporting static values in kernels.
void appendStaticOperandParameters(Operation *op, OpBuilder &builder,
                                   SmallVectorImpl<NamedAttribute> &params) {
  auto extractConstantValues = [&](Value ssaValue, StringLiteral paramName) {
    SmallVector<int64_t> values;
    auto asTensor = cast<MO::TensorValue>(ssaValue);
    if (auto maybeIntVec = tryGetConstantIntVector(asTensor)) {
      values = std::move(*maybeIntVec);
    } else if (auto maybeIntScalar = tryGetConstantInt(asTensor)) {
      values.push_back(*maybeIntScalar);
    } else {
      MO::TensorType t = asTensor.getType();
      if (t.hasStaticShape()) {
        int64_t numElems = *t.getNumElements();
        values.append(numElems, ShapedType::kDynamic);
      } else {
        values.push_back(ShapedType::kDynamic);
      }
    }

    params.push_back(NamedAttribute(builder.getStringAttr(paramName),
                                    builder.getI64ArrayAttr(values)));
  };
  if (auto convOp = dyn_cast<MO::ConvOp>(op)) {
    extractConstantValues(convOp.getStrides(),
                          MOGG::PARAMETER_CONV_STATIC_STRIDES);
    extractConstantValues(convOp.getDilations(),
                          MOGG::PARAMETER_CONV_STATIC_DILATIONS);
    extractConstantValues(convOp.getPaddings(),
                          MOGG::PARAMETER_CONV_STATIC_PADDING);
  } else if (auto transposeOp = dyn_cast<MO::TransposeOp>(op)) {
    extractConstantValues(transposeOp.getPerm(),
                          MOGG::PARAMETER_TRANSPOSE_STATIC_PERM);
  } else if (auto sliceOp = dyn_cast<MO::SliceOp>(op)) {
    extractConstantValues(sliceOp.getStart(),
                          MOGG::PARAMETER_SLICE_STATIC_STARTS);
    extractConstantValues(sliceOp.getStep(),
                          MOGG::PARAMETER_SLICE_STATIC_STEPS);
  }
}

} // namespace M

namespace M::MOGG {

/// Resolves kernel parameters (P0, P1, ...) from the opaque type parameters
/// (S0, S1, ...) on operands and results, via the kernel's annotations:
///   mogg.value_params: [{S0 = #kgen.param.decl.ref<P0>, ...}, ...]
///   mogg.value_result_params: [{S0 = #kgen.param.decl.ref<P0>, ...}]
void inferOpaqueParameters(SmallVectorImpl<NamedAttribute> &forwardedParameters,
                           Operation *moOp,
                           const KernelDeclFunctionAdaptor &function,
                           mlir::ArrayAttr inputParams) {
  auto inputParamsAttr = function.attr.getArgParams();
  auto resultParamsAttr = function.attr.getResultParams();

  // Explicitly declared parameters only (e.g. "P0" from "[P0: Int]"); a
  // backtick marks an inferred one (e.g. "S1`2x1" from a "_" placeholder).
  llvm::DenseSet<StringRef> declaredParams;
  if (inputParams) {
    for (auto paramAttr : inputParams) {
      if (auto param = dyn_cast<KGEN::ParamDeclAttr>(paramAttr)) {
        StringRef paramName = param.getName();
        if (!paramName.contains('`'))
          declaredParams.insert(paramName);
      }
    }
  }

  auto inferFromOpaque = [&](DictionaryAttr kernelParamDict,
                             MO::OpaqueType opaqueType) {
    if (!kernelParamDict || !opaqueType)
      return;

    DictionaryAttr opaqueParams = opaqueType.getParameters();
    if (!opaqueParams)
      return;

    for (NamedAttribute kernelParam : kernelParamDict) {
      auto declRef = dyn_cast<KGEN::ParamDeclRefAttr>(kernelParam.getValue());
      if (!declRef)
        continue;

      StringRef paramName = declRef.getName();
      if (!declaredParams.contains(paramName))
        continue;

      // Same key as in the opaque type (e.g. "S0").
      auto opaqueValue = opaqueParams.getNamed(kernelParam.getName());
      if (!opaqueValue)
        continue;

      auto typedValue = dyn_cast<TypedAttr>(opaqueValue->getValue());
      if (!typedValue)
        continue;

      forwardedParameters.push_back(
          NamedAttribute(declRef.getName(), typedValue));
    }
  };

  // Kernel arguments list DPS outputs first, then inputs; MO operands are
  // only the inputs.
  if (inputParamsAttr) {
    unsigned numDpsOutputs =
        static_cast<unsigned>(function.outputArguments.size());

    for (auto [idx, operand] : llvm::enumerate(moOp->getOperands())) {
      unsigned kernelArgIdx = idx + numDpsOutputs;
      if (kernelArgIdx >= inputParamsAttr.size())
        break;

      auto opaqueType = dyn_cast<MO::OpaqueType>(operand.getType());
      if (!opaqueType)
        continue;

      auto kernelParamDict =
          dyn_cast<DictionaryAttr>(inputParamsAttr[kernelArgIdx]);
      inferFromOpaque(kernelParamDict, opaqueType);
    }
  }

  if (!resultParamsAttr)
    return;

  for (auto [idx, result] : llvm::enumerate(moOp->getResults())) {
    if (idx >= resultParamsAttr.size())
      break;

    auto opaqueType = dyn_cast<MO::OpaqueType>(result.getType());
    if (!opaqueType)
      continue;

    auto kernelParamDict = dyn_cast<DictionaryAttr>(resultParamsAttr[idx]);
    inferFromOpaque(kernelParamDict, opaqueType);
  }
}

/// Removes parameters from inferByInputs that appear in the function's result
/// type but not in its input argument types.
static void removeResultOnlyParams(KGEN::LIT::FnTypeGeneratorType sig,
                                   llvm::DenseSet<StringRef> &inferByInputs) {
  SmallVector<KGEN::ParamDeclRefAttr> resultParamRefs;
  bool hasConstExpr = false;
  size_t requiredSignatureDepth = 0;
  Type userResultType = sig.getUserResultType();
  KGEN::collectParameterReferences(userResultType, resultParamRefs,
                                   hasConstExpr, requiredSignatureDepth);
  llvm::DenseSet<StringRef> resultParams;
  for (auto paramRef : resultParamRefs)
    resultParams.insert(paramRef.getName().getValue());

  // Excludes the byref result slot.
  ArrayRef<Type> argTypes = sig.getArguments();
  if (sig.hasMemoryOnlyResult())
    argTypes = argTypes.drop_back();
  if (sig.isThrows())
    argTypes = argTypes.drop_back();

  llvm::DenseSet<StringRef> inputParams;
  for (Type argType : argTypes) {
    SmallVector<KGEN::ParamDeclRefAttr> argParamRefs;
    KGEN::collectParameterReferences(argType, argParamRefs, hasConstExpr,
                                     requiredSignatureDepth);
    for (auto paramRef : argParamRefs)
      inputParams.insert(paramRef.getName().getValue());
  }

  SmallVector<StringRef> toRemove;
  for (StringRef paramName : inferByInputs) {
    if (resultParams.contains(paramName) && !inputParams.contains(paramName))
      toRemove.push_back(paramName);
  }
  for (StringRef paramName : toRemove)
    inferByInputs.erase(paramName);
}

/// Shared by all inferKernelParameters overloads.
static ErrorOr<InferredParameters>
inferParameters(ArrayRef<mlir::Attribute> fnLevelParams,
                ArrayRef<mlir::Attribute> structLevelParams,
                mlir::ArrayAttr inputParamMappings,
                mlir::DictionaryAttr defaultAttrs,
                KGEN::LIT::FnTypeGeneratorType sig,
                std::function<ErrorOr<MojoValue>(StringRef)> &inferCallback) {

  // A param is inferable from inputs when an argument type references it, as
  // a `ParamDeclRefAttr` or a depth-0 `ParamIndexRefAttr` (elaborated
  // bytecode's form). An IndexRef `*(0, K)` indexes the generator's full
  // parameter list, laid out as `structLevelParams + fnLevelParams`, so look
  // it up in the merged list; `fnLevelParams` alone would name the wrong
  // param.
  llvm::SmallVector<mlir::Attribute> mergedParams;
  mergedParams.reserve(structLevelParams.size() + fnLevelParams.size());
  mergedParams.append(structLevelParams.begin(), structLevelParams.end());
  mergedParams.append(fnLevelParams.begin(), fnLevelParams.end());

  llvm::DenseSet<StringRef> inferByInputs;
  std::function<void(mlir::Attribute)> walkAttr;
  std::function<void(mlir::Type)> walkType;
  walkAttr = [&](mlir::Attribute attr) {
    if (!attr)
      return;
    if (auto sugar = dyn_cast<KGEN::SugarAttr>(attr)) {
      walkAttr(sugar.getCanonical());
      return;
    }
    if (auto declRef = dyn_cast<KGEN::ParamDeclRefAttr>(attr)) {
      inferByInputs.insert(declRef.getName().getValue());
      return;
    }
    if (auto indexRef = dyn_cast<KGEN::ParamIndexRefAttr>(attr)) {
      if (indexRef.getDepth() == 0 &&
          indexRef.getIndex() < mergedParams.size()) {
        auto pd = cast<KGEN::ParamDeclAttr>(mergedParams[indexRef.getIndex()]);
        inferByInputs.insert(pd.getName().getValue());
      }
      return;
    }
    attr.walkImmediateSubElements([&](mlir::Attribute a) { walkAttr(a); },
                                  [&](mlir::Type t) { walkType(t); });
  };
  walkType = [&](mlir::Type type) {
    if (!type)
      return;
    type.walkImmediateSubElements([&](mlir::Attribute a) { walkAttr(a); },
                                  [&](mlir::Type t) { walkType(t); });
  };
  for (size_t i = 0, n = sig.getNumArguments(); i < n; ++i)
    walkType(sig.getArgument(i));

  removeResultOnlyParams(sig, inferByInputs);

  // Infer-only parameters (declared before `//`) are left for Mojo to infer
  // at the call. The walk above can miss one: a `TypeList` that a
  // `xs: TensorTuple[*Ts]` argument unpacks is referenced only through its
  // values.
  llvm::DenseSet<StringRef> inferOnly;
  if (KGEN::PogListAttr pogs = sig.getParamListAttrs())
    for (size_t i = 0, n = pogs.size(); i < n; ++i)
      if (pogs.getPassingKind(i) == KGEN::PassingKind::Inferred)
        inferOnly.insert(pogs.getName(i).getValue());

  auto isIOSpecType = [](Type ty) {
    auto structType = dyn_cast<KGEN::LIT::StructType>(ty);
    return structType &&
           isMojoIOSpecTypeName(structType.getSymbol().getNestedReferences());
  };

  auto inferParametersInner =
      [&](ArrayRef<mlir::Attribute> params,
          SmallVector<MojoValue> &inferredParams) -> ErrorOrSuccess {
    for (auto paramAttr : params) {
      auto param = cast<KGEN::ParamDeclAttr>(paramAttr);
      StringRef paramName = param.getName().getValue();

      if (isa<KGEN::LIT::OriginType>(param.getType()) ||
          isIOSpecType(param.getType()))
        continue;
      if (paramName.contains("."))
        continue;

      ErrorOr<MojoValue> maybeVal = inferCallback(paramName);
      if ((inferByInputs.contains(paramName) ||
           inferOnly.contains(paramName)) &&
          maybeVal.isError())
        continue;

      if (maybeVal.isError()) {
        if (defaultAttrs && defaultAttrs.contains(paramName))
          continue;
        return maybeVal.takeError();
      }
      inferredParams.push_back(MojoValue(paramName).equal(*maybeVal));
    }
    return success();
  };

  InferredParameters ret;
  auto errOr = inferParametersInner(fnLevelParams, ret.functionParameters);
  if (errOr.isError())
    return errOr.takeError();
  errOr = inferParametersInner(structLevelParams, ret.structParameters);
  if (errOr.isError())
    return errOr.takeError();
  return ret;
}

// {param-name -> UnitAttr} for fn-level params with a declared default, from
// the function's `PogListAttr`; such params may fail to infer.
static mlir::DictionaryAttr
computeDefaultParamsDict(KernelDeclFunctionAdaptor *fnAdaptor) {
  if (!fnAdaptor || !fnAdaptor->attr.getParams() ||
      fnAdaptor->attr.getParams().empty())
    return nullptr;
  auto fnParams = fnAdaptor->attr.getParams().getValue();
  KGEN::PogListAttr pogs = fnAdaptor->attr.getSignature().getParamListAttrs();
  size_t numImplicit =
      fnAdaptor->attr.getSignature().getNumImplicitOriginDecls();
  size_t totalPogs = pogs.getPogs().size();
  // Fn-level params are at the end of the merged pog list (struct-level
  // params come first).
  size_t fnStart =
      totalPogs >= fnParams.size() ? totalPogs - fnParams.size() : 0;
  size_t fnEnd =
      fnParams.size() > numImplicit ? fnParams.size() - numImplicit : 0;

  mlir::Builder builder(fnAdaptor->attr.getSignature().getContext());
  mlir::NamedAttrList defaultNames;
  for (size_t fnIdx = 0; fnIdx < fnEnd; ++fnIdx) {
    size_t pogIdx = fnStart + fnIdx;
    if (pogIdx >= totalPogs || !pogs.getDefault(pogIdx))
      continue;
    auto paramDecl = mlir::cast<KGEN::ParamDeclAttr>(fnParams[fnIdx]);
    defaultNames.append(
        mlir::StringAttr::get(
            fnAdaptor->attr.getSignature().getContext(),
            KGEN::demangleParameterName(paramDecl.getName().strref())),
        builder.getUnitAttr());
  }
  if (defaultNames.empty())
    return nullptr;
  return mlir::DictionaryAttr::get(fnAdaptor->attr.getSignature().getContext(),
                                   defaultNames.getAttrs());
}

ErrorOr<InferredParameters> inferKernelParameters(
    KernelDeclAdaptor &adaptor, KernelFunctionKind kind,
    std::function<ErrorOr<MojoValue>(StringRef)> &&inferCallback) {
  KernelDeclFunctionAdaptor *fnAdaptor =
      const_cast<KernelDeclFunctionAdaptor *>(adaptor.getFunction(kind));
  ASSERT_STREAM(fnAdaptor, "Kernel has no '"
                               << stringifyKernelFunctionKind(kind)
                               << "' function");

  // Each variant (execute / elementwise / shape / build_graph) has its own
  // param list, since signatures differ (e.g. elementwise omits
  // `target`/`_trace_name`).
  ArrayRef<mlir::Attribute> fnParams;
  if (fnAdaptor->attr.getParams())
    fnParams = fnAdaptor->attr.getParams().getValue();

  ArrayRef<mlir::Attribute> structParams;
  if (adaptor.structParams)
    structParams = adaptor.structParams.getValue();

  // Defaults are declared on the execute signature (`execute_device`'s, for a
  // declarative kernel), except that `execute_configuration` declares its own.
  KernelDeclFunctionAdaptor *defaultsFn =
      kind == KernelFunctionKind::ExecuteConfiguration
          ? fnAdaptor
          : (adaptor.executeFunction ? &*adaptor.executeFunction : nullptr);
  mlir::DictionaryAttr defaultParams = computeDefaultParamsDict(defaultsFn);
  return inferParameters(fnParams, structParams, fnAdaptor->attr.getArgParams(),
                         defaultParams, fnAdaptor->attr.getSignature(),
                         inferCallback);
}

ErrorOr<InferredParameters> inferKernelParameters(
    PrimitiveDeclOp primDecl,
    std::function<ErrorOr<MojoValue>(StringRef)> &&inferCallback) {
  MOGG::KernelFunctionAttr fn = primDecl.getFunction();
  return inferParameters(fn.getParams().getValue(),
                         /*structLevelParams=*/{}, fn.getArgParams(),
                         /*defaultAttrs=*/nullptr, fn.getSignature(),
                         inferCallback);
}

ErrorOr<InferredParameters> inferKernelParameters(
    ShapeFunctionDeclOp shapeDecl,
    std::function<ErrorOr<MojoValue>(StringRef)> &&inferCallback) {
  MOGG::KernelFunctionAttr fn = shapeDecl.getFunction();
  return inferParameters(fn.getParams().getValue(),
                         /*structLevelParams=*/{}, fn.getArgParams(),
                         /*defaultAttrs=*/nullptr, fn.getSignature(),
                         inferCallback);
}

} // namespace M::MOGG
