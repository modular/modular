//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringSet.h"
#include <KGEN/LITDialect/LITUtils.h>

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/MOGGPreElab/MOGGDecorators.h"
#include "KGEN/MOGGPreElab/MOGGUtils.h"
#include "KGEN/MOGGPreElab/Passes.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/include/KGEN/MOGGPreElab/MOGGDecorators.h"
#include "Support/AssertStream.h"
#include "mlir/Pass/Pass.h"

#include "Helpers.h"

using namespace M;
using namespace KGEN;
using namespace MOGGPreElab;

namespace M::KGEN::MOGGPreElab {
#define GEN_PASS_DEF_MOGGANNOTATE
#include "KGEN/MOGGPreElab/MOGGPreElabPasses.h.inc"
} // namespace M::KGEN::MOGGPreElab

static constexpr llvm::StringLiteral kExecuteFuncName = "execute";
static constexpr llvm::StringLiteral kShapeFuncName = "shape";
static constexpr llvm::StringLiteral kPyTorchFallbackFuncName =
    "pytorch_fallback";

static constexpr std::array<StringLiteral, 3> kIOSpec = {"tensor_internal",
                                                         "io_spec", "IOSpec"};

static constexpr std::array<StringLiteral, 3> kMaxManagedTensorSlice = {
    "tensor_internal", "managed_tensor_slice", "ManagedTensorSlice"};
static constexpr std::array<StringLiteral, 4> kMaxSIMD = {"stdlib", "builtin",
                                                          "simd", "SIMD"};
static constexpr std::array<StringLiteral, 3> kMaxStaticTuple = {
    "tensor_internal", "managed_tensor_slice", "VariadicTensors"};
static constexpr std::array<StringLiteral, 4> kMaxList = {
    "stdlib", "collections", "vector", "InlinedFixedVector"};

// Define the ordered parameter info - pairs of (name, uses_inferred_params)
// TODO(GEX-1822): Should be able to query this information from
// ManagedTensorSlice's lit.struct.decl op directly.
static constexpr std::array<std::pair<StringLiteral, bool>, 6>
    kManagedTensorSliceParams = {{
        {"mut", false},
        {"input", false},
        {"type", false},
        {"rank", false},
        {"io_spec", true}, // Only ioSpec uses inferred parameters
        {"static_spec", false},
    }};

// Check if the decorator correspond to a function call:
// eg @foo(arg)
//
// Also check that foo is marked with the attribute `expectedFuncAttr`.
// Returns arg (or nullptr if invalid).
static std::optional<SmallVector<TypedAttr>>
getDecoratorLambdaArgument(ModuleOp mod, TypedAttr decorator,
                           StringRef expectedFuncAttr,
                           ArrayRef<size_t> indicesToFetch) {
  auto apply = dyn_cast<ParamOperatorAttr>(decorator);
  if (!apply)
    return std::nullopt;

  auto sym = dyn_cast<SymbolConstantAttr>(apply.getOperand(0));
  if (!sym)
    return std::nullopt;

  auto decoratorFunc = mod.lookupSymbol<LIT::FnOp>(sym.getSymbol());
  if (!decoratorFunc || !decoratorFunc->hasAttr(expectedFuncAttr))
    return std::nullopt;

  SmallVector<TypedAttr> answer;
  for (size_t i : indicesToFetch) {
    if (i >= apply.getNumOperands())
      return std::nullopt;
    answer.push_back(apply.getOperand(i));
  }
  return answer;
}

static bool hasEnforceIODecorator(LIT::FnOp func) {
  auto isEnforceIO = [](TypedAttr attr) {
    auto sym = dyn_cast<SymbolConstantAttr>(attr);
    if (!sym)
      return false;
    return sym.getSymbol().getLeafReference().strref().starts_with(
        M::KGEN::MOGGPreElab::Decorators::ENFORCE_IO_PARAM);
  };

  return llvm::any_of(func.getDecorators(), isEnforceIO);
}

/// Look through ref types to get underlaying decl ref type if needed.
LIT::StructType getAsDeclRefOrNull(Type t) {
  auto asLitRef = dyn_cast<LIT::RefType>(t);
  if (asLitRef)
    return dyn_cast<LIT::StructType>(asLitRef.getElementType());
  return dyn_cast<LIT::StructType>(t);
}

/// Check if the function is a DPS kernel with by-ref tensor arguments.
static LogicalResult checkByRefTensorArgs(LIT::FnOp func) {
  LIT::FnTypeGeneratorType signature = func.getFuncTypeGenerator();
  for (auto [index, litType] : llvm::enumerate(signature.getArguments())) {
    if (LIT::StructType asDeclRef = getAsDeclRefOrNull(litType)) {
      if (isDPSKernel(func) && isDPSTensor(asDeclRef) &&
          isa<LIT::RefType>(litType)) {
        return func.emitError()
               << " Only the borrowed argument (read) convention is supported "
                  "for tensor arguments ("
               << signature.getArgName(index) << " is mutable here).";
      }
    }
  }

  return success();
}

static LogicalResult annotateTypes(LIT::FnOp func) {
  // Anything taking a tensor needs the annotation.
  bool takesTensor = false;
  for (Type litType : func.getArgumentTypes()) {
    if (LIT::StructType asDeclRef = getAsDeclRefOrNull(litType)) {
      takesTensor |= isExtensibilityTensor(asDeclRef);
      takesTensor |= isDPSTensor(asDeclRef);
    }
  }

  if (!isKernel(func) && !isV1ShapeFunc(func) && !isDPSKernel(func) &&
      !takesTensor)
    return success();

  if (failed(checkByRefTensorArgs(func)))
    return failure();

  OpBuilder builder{func.getContext()};

  SmallVector<Attribute> observedParams, typeNames, sourceName;
  observedParams.reserve(func.getNumArguments());

  Attribute emptyAttr = builder.getUnitAttr();

  // Extract the source name any of the lit argument.
  auto litTypeToSourceName = [&](Type litType) -> Attribute {
    LIT::StructType asDeclRef = getAsDeclRefOrNull(litType);
    if (!asDeclRef)
      return emptyAttr;

    // We can't lower the symbol as it may become illegal at some point in IR so
    // we combine it into ROOT::LEAF;
    std::string combinedName =
        Twine(asDeclRef.getSymbol().getRootReference().strref())
            .concat("::")
            .concat(asDeclRef.getSymbol().getLeafReference().strref())
            .str();
    return builder.getStringAttr(combinedName);
  };

  // Extract the used parameters from the lit type.
  auto litTypeToParams = [&](Type litType) -> Attribute {
    LIT::StructType asDeclRef = getAsDeclRefOrNull(litType);

    // We still need to have one entry per argument even if it is empty.
    if (!asDeclRef || asDeclRef.getParamValues().empty())
      return emptyAttr;

    SmallVector<Attribute> attrs;
    for (TypedAttr param : asDeclRef.getParamValues())
      attrs.push_back(param);
    return builder.getArrayAttr(attrs);
  };

  for (auto [i, litType] : llvm::enumerate(func.getArgumentTypes())) {
    observedParams.push_back(litTypeToParams(litType));
    typeNames.push_back(litTypeToSourceName(litType));

    sourceName.push_back(func.getFuncTypeGenerator().getArgName(i));
  }

  // Attach the parameter mapping information to the kernel.
  if (!observedParams.empty()) {
    func->setDiscardableAttr(MOGG_ARG_PARAMS,
                             builder.getArrayAttr(observedParams));
  }

  // Add the result type.
  Type resultType = func.getResultTypes()[0];
  if (!isa<KGEN::NoneType>(resultType)) {
    func->setDiscardableAttr(MOGG_RESULT_PARAMS, litTypeToParams(resultType));
    func->setDiscardableAttr(MOGG_RESULT_TYPE_NAME,
                             litTypeToSourceName(resultType));
  }

  if (!typeNames.empty()) {
    func->setDiscardableAttr(MOGG_ARG_TYPE_NAMES,
                             builder.getArrayAttr(typeNames));
  }

  if (!sourceName.empty()) {
    func->setDiscardableAttr(MOGG_ARG_SRC_NAMES,
                             builder.getArrayAttr(sourceName));
  }

  return success();
}

// Extract the used parameters from the lit type. Returns null for fields which
// are unbound or already populated.
static SmallVector<KGEN::ParamDeclRefAttr>
litTypeToParams(LIT::StructType structType) {
  SmallVector<KGEN::ParamDeclRefAttr> attrs;
  for (TypedAttr param : structType.getParamValues()) {
    auto declRefAttr = dyn_cast<KGEN::ParamDeclRefAttr>(param);
    attrs.push_back(declRefAttr);
  }

  return attrs;
};

/// Return a set of named attributes mapping all unbound parameters in the
/// tensor type struct
static SmallVector<NamedAttribute>
getUnboundParametersForTensor(LIT::StructType &structType, Builder &builder) {
  enum ParameterIndexes {
    kMutIndex,
    kInputIndex,
    kDTypeIndex,
    kRankIndex,
    kIOSpec,
    kStaticSpecsIndex,
    kNumParameters
  };

  auto allParameters = litTypeToParams(structType);
  ASSERT_STREAM(allParameters.size() >= kNumParameters,
                << "Expected at least " << kNumParameters
                << " parameters on the tensor type");

  auto mut = allParameters[kMutIndex];
  auto input = allParameters[kInputIndex];
  auto dtype = allParameters[kDTypeIndex];
  auto rank = allParameters[kRankIndex];
  auto spec = allParameters[kStaticSpecsIndex];

  SmallVector<NamedAttribute> tensorSpecNamedAttrs;
  // Sometimes, dtype or ranks are not present because the user expects
  // specific values for those parameters (ex: dtype=float32 or rank=2).
  if (dtype)
    tensorSpecNamedAttrs.emplace_back(builder.getStringAttr(kParameterDType),
                                      dtype);

  if (rank)
    tensorSpecNamedAttrs.emplace_back(builder.getStringAttr(kParameterRank),
                                      rank);

  if (spec)
    tensorSpecNamedAttrs.emplace_back(builder.getStringAttr(kStaticSpec), spec);

  if (mut)
    tensorSpecNamedAttrs.emplace_back(builder.getStringAttr(kParameterMut),
                                      mut);

  if (input)
    tensorSpecNamedAttrs.emplace_back(builder.getStringAttr(kParameterInput),
                                      input);

  return tensorSpecNamedAttrs;
}

static SmallVector<NamedAttribute>
getUnboundParametersForSIMD(LIT::StructType structType, Builder &builder) {
  static constexpr unsigned kDTypeIndex = 0;
  static constexpr unsigned kSizeIndex = 1;
  auto allParameters = litTypeToParams(structType);

  ASSERT_STREAM(allParameters.size() >= 2,
                << "Expected at least two parameters on the SIMD type");
  auto dtype = allParameters[kDTypeIndex];
  auto size = allParameters[kSizeIndex];

  SmallVector<NamedAttribute> tensorSpecNamedAttrs;
  // Sometimes, dtype or size are not present because the user expects
  // specific values for those parameters (ex: dtype=float32 or size=1).
  if (dtype)
    tensorSpecNamedAttrs.push_back(
        NamedAttribute{builder.getStringAttr(kParameterDType), dtype});
  if (size)
    tensorSpecNamedAttrs.push_back(
        NamedAttribute{builder.getStringAttr(kParameterSize), size});

  return tensorSpecNamedAttrs;
}

// Returns a dictionary attribute containing all parameters for simple types
// (i.e: tensor or SIMD).
static DictionaryAttr getParametersForSimpleType(LIT::StructType structType,
                                                 Builder &builder) {
  SmallVector<NamedAttribute> attrs;
  auto paramValues = structType.getParamValues();

  // Add non-inferred parameters to the dictionary
  for (auto [param, paramInfo] :
       llvm::zip(paramValues, kManagedTensorSliceParams)) {
    auto [paramName, usesInferred] = paramInfo;
    // Skip parameters that use inferred values
    // This is because slice-mogg-funcs uses the values gathered here to
    // parameterize lambda fusion hooks, but the KGEN drops any
    // parameters composed of inferred parameters.
    if (usesInferred)
      continue;

    attrs.push_back(NamedAttribute{builder.getStringAttr(paramName), param});
  }

  return builder.getDictionaryAttr(attrs);
}

/// Return a set of named attributes mapping all unbound parameters in the tuple
/// of tensor struct
///
//  mut: Bool,
//  input: IO, //,
//  type: DType,
//  rank: Int,
//  size: Int,
//  ioSpec: IOSpec[mut, input],
//  *,
//  static_specs: StaticTuple[StaticTensorSpec[type, rank], size],
static SmallVector<NamedAttribute>
getUnboundParametersForVariadicTensors(LIT::StructType structType,
                                       Builder &builder) {
  enum ParameterIndexes {
    kMut,
    kInput,
    kDTypeIndex,
    kRankIndex,
    kSizeIndex,
    kIOSpec,
    kStaticSpecsIndex,
    kNumParameters
  };

  SmallVector<KGEN::ParamDeclRefAttr> allParameters =
      litTypeToParams(structType);

  ASSERT_STREAM(allParameters.size() >= kNumParameters,
                << "Expected at least " << kNumParameters
                << " parameters on the tuple-of-tensors type");

  auto mut = allParameters[kMut];
  auto input = allParameters[kInput];
  auto type = allParameters[kDTypeIndex];
  auto rank = allParameters[kRankIndex];
  auto size = allParameters[kSizeIndex];
  auto spec = allParameters[kStaticSpecsIndex];

  SmallVector<NamedAttribute> namedAttrs;
  if (mut)
    namedAttrs.emplace_back(builder.getStringAttr(kParameterMut), mut);

  if (input)
    namedAttrs.emplace_back(builder.getStringAttr(kParameterInput), input);

  if (type)
    namedAttrs.emplace_back(builder.getStringAttr(kParameterDType), type);

  if (rank)
    namedAttrs.emplace_back(builder.getStringAttr(kParameterRank), rank);

  if (size)
    namedAttrs.emplace_back(builder.getStringAttr(kParameterSize), size);

  if (spec)
    namedAttrs.emplace_back(builder.getStringAttr(kStaticSpecs), spec);

  return namedAttrs;
}

/// Return a set of named attributes mapping all unbound parameters in the list
/// of tensor struct
static std::optional<SmallVector<NamedAttribute>>
getUnboundParametersForTensorList(LIT::StructType &structType,
                                  Builder &builder) {
  // TODO(GEX-1126): consider a tuple which only contains tensors to
  // simplify this
  static constexpr unsigned kElementType = 0;
  auto allParameters = litTypeToParams(structType);

  ASSERT_STREAM(
      allParameters.size() >= 2,
      << "Expected at least two parameters on the list-of-tensor type");
  [[maybe_unused]] auto elementType = allParameters[kElementType];
  ASSERT_STREAM(!elementType,
                << "Element type must be defined and be equal to tensor");
  SmallVector<NamedAttribute> listNamedAttrs;

  auto elementTypeAttr =
      cast<KGEN::TypeParamAttr>(structType.getParamValues()[0]);
  auto elementTypeStruct =
      cast<LIT::StructType>(elementTypeAttr.getTypeValue());

  if (!symbolMatches(elementTypeStruct.getSymbol(), kMaxManagedTensorSlice))
    return std::nullopt;

  auto elementTypeParams =
      getUnboundParametersForTensor(elementTypeStruct, builder);

  listNamedAttrs.append(elementTypeParams);
  return listNamedAttrs;
}

static void labelTensorParamsInKernel(LIT::FnOp funcOp) {
  Builder builder{funcOp.getContext()};

  if (!isDPSKernel(funcOp))
    return;

  // Look through ref types to get underlying decl ref type if needed.
  auto getAsStructType = [](Type t) {
    auto asLitRef = dyn_cast<LIT::RefType>(t);
    if (asLitRef)
      return dyn_cast<LIT::StructType>(asLitRef.getElementType());
    return dyn_cast<LIT::StructType>(t);
  };

  SmallVector<Attribute> tensorSpecs;
  SmallVector<Attribute> tensorArgsParams;
  Attribute emptyAttr = builder.getUnitAttr();

  for (auto [i, litType] : llvm::enumerate(funcOp.getArgumentTypes())) {
    auto asStructType = getAsStructType(litType);

    if (!asStructType) {
      tensorSpecs.push_back(emptyAttr);
      tensorArgsParams.push_back(emptyAttr);
      continue;
    }

    if (symbolMatches(asStructType.getSymbol(), kMaxManagedTensorSlice)) {
      SmallVector<NamedAttribute> tensorSpecNamedAttrs =
          getUnboundParametersForTensor(asStructType, builder);
      tensorSpecs.push_back(
          DictionaryAttr::get(funcOp.getContext(), tensorSpecNamedAttrs));
      tensorArgsParams.push_back(
          getParametersForSimpleType(asStructType, builder));
    } else if (symbolMatches(asStructType.getSymbol(), kMaxSIMD)) {
      SmallVector<NamedAttribute> tensorSpecNamedAttrs =
          getUnboundParametersForSIMD(asStructType, builder);
      tensorSpecs.push_back(
          DictionaryAttr::get(funcOp.getContext(), tensorSpecNamedAttrs));
      tensorArgsParams.push_back(
          getParametersForSimpleType(asStructType, builder));
    } else if (symbolMatches(asStructType.getSymbol(), kMaxStaticTuple)) {
      auto tensorSpecNamedAttrs =
          getUnboundParametersForVariadicTensors(asStructType, builder);
      tensorSpecs.push_back(
          DictionaryAttr::get(funcOp.getContext(), tensorSpecNamedAttrs));
      tensorArgsParams.push_back(tensorSpecs.back());
    } else if (symbolMatches(asStructType.getSymbol(), kMaxList)) {
      auto tensorSpecNamedAttrs =
          getUnboundParametersForTensorList(asStructType, builder);
      if (!tensorSpecNamedAttrs) {
        tensorSpecs.push_back(emptyAttr);
        tensorArgsParams.push_back(emptyAttr);
      } else {
        tensorSpecs.push_back(
            DictionaryAttr::get(funcOp.getContext(), *tensorSpecNamedAttrs));
        tensorArgsParams.push_back(emptyAttr);
      }
    } else {
      // Unsupported type, can ignore
      tensorSpecs.push_back(emptyAttr);
      tensorArgsParams.push_back(emptyAttr);
    }
  }
  funcOp->setDiscardableAttr(kKernelValueParameterAttrName,
                             builder.getArrayAttr(tensorSpecs));
  funcOp->setDiscardableAttr(MOGG_TENSOR_ARG_PARAMS,
                             builder.getArrayAttr(tensorArgsParams));
}

namespace {

// Important metadata about the structs under the extensibility API
struct ExtensibilityAPIStructInfo {
  // Whether the operation is marked as an elementwise kernel
  bool isElementwiseKernel = false;

  // Whether the operation is marked as a view kernel.
  bool isViewKernel = false;

  // The number of Destination Passing Style result operands to expect.
  IntegerAttr numDPSOperands{};

  // The name of the operation this struct is registered to
  StringAttr registrationName{};
};

// Returns whether this is a struct under the extensibility API.
// Along the way, populate metadata in `registrationInfo.`
//
// Also known as ExtensibilityV3 or KernelAPI.
ErrorOr<bool>
isExtensibilityAPIStruct(LIT::StructDeclOp structDeclOp, ModuleOp moduleOp,
                         ExtensibilityAPIStructInfo &registrationInfo) {
  auto decorators = structDeclOp.getDecorators();

  // Iterate over the decorators and to find max.compiler.register.
  for (auto decorator : decorators) {
    // Handle elementwise annotation
    if (auto directSym = dyn_cast<SymbolConstantAttr>(decorator)) {
      auto decoratorFunc =
          moduleOp.lookupSymbol<LIT::FnOp>(directSym.getSymbol());

      if (decoratorFunc && decoratorFunc->hasAttr(MOGG_INTRINSIC_ELEMENTWISE)) {
        if (registrationInfo.isElementwiseKernel)
          return Error("Kernel has multiple elementwise annotations");
        registrationInfo.isElementwiseKernel = true;
        continue;
      }

      if (decoratorFunc && decoratorFunc->hasAttr(MOGG_INTRINSIC_VIEW_KERNEL)) {
        if (registrationInfo.isViewKernel)
          return Error("Kernel has multiple view annotations");
        registrationInfo.isViewKernel = true;
        continue;
      }
    }

    std::optional<SmallVector<TypedAttr>> registerOperand =
        getDecoratorLambdaArgument(moduleOp, decorator, MOGG_INTRINSIC_REGISTER,
                                   SmallVector<size_t>{1, 2});
    if (!registerOperand.has_value())
      continue;

    auto [_, nameAttr] = cast<LIT::LITStructAttr>(registerOperand.value()[0])
                             .getValues()
                             .front();
    auto name = dyn_cast<StringAttr>(nameAttr);
    ASSERT_STREAM(name, << "Expected a StringAttr as the registration name");
    if (registrationInfo.registrationName) {
      return Error("Only one op can be registered per kernel");
    }
    registrationInfo.registrationName = name;

    auto [__, numDPSOperandsAttr] =
        cast<LIT::LITStructAttr>(registerOperand.value()[1])
            .getValues()
            .front();
    auto numDPSOperands = dyn_cast<IntegerAttr>(numDPSOperandsAttr);
    ASSERT_STREAM(numDPSOperands,
                  << "Expected an IntegerAttr as the number of DPS operands");
    registrationInfo.numDPSOperands = numDPSOperands;
  }

  return registrationInfo.registrationName != nullptr;
}

// Run standard checks and mutations on a function. emits an
// error and returns false if an issue was present.
//
// `structDeclOp` the parent struct the function being process lives in
// `registrationInfo` the metadata about registration info
// `func` the func being processed
// `annotation` the annotation name to attach to this function to be used in
//    later lowering.
// `builder` builder for ops
LogicalResult processStructFuncCommon(
    LIT::StructDeclOp structDeclOp, ExtensibilityAPIStructInfo registrationInfo,
    LIT::FnOp func, StringLiteral annotation, Builder &builder) {
  if (!func.getIsStatic()) {
    func->emitError("This function must be static");
    return failure();
  }

  func->setAttr(builder.getStringAttr(annotation),
                registrationInfo.registrationName);
  func.setExported();
  if (failed(annotateTypes(func)))
    return failure();

  return success();
}

enum class IOSpec {
  Input,       // Input tensor, read-only
  Output,      // Output tensor, write-only
  MutableInput // Input tensor that can be modified
};

std::optional<IOSpec> parseTensorIOSpec(LIT::StructType tensorStruct,
                                        Location loc, StringRef argName) {
  // Get the parameter values
  auto params = tensorStruct.getParamValues();
  if (params.size() != 2) {
    emitError(loc, "Error for argument '" + argName + "': " + kIOSpec.back() +
                       " must have exactly 2 parameters");
    return std::nullopt;
  }

  // First parameter should be a bool struct for mut
  auto mutParam = dyn_cast<LIT::LITStructAttr>(params[0]);
  if (!mutParam) {
    emitError(loc, "Error for argument '" + argName +
                       "': 'mut' inferred parameter must be set");
    return std::nullopt;
  }

  auto [_, mutValueAttr] = mutParam.getValues().front();
  auto mutIntAttr = dyn_cast<IntegerAttr>(mutValueAttr);
  if (!mutIntAttr) {
    emitError(loc, "Error for argument '" + argName +
                       "': Expected integer attribute for mut parameter value");
    return std::nullopt;
  }
  bool isMut = mutIntAttr.getValue().getBoolValue();

  // Second parameter should be an IO struct with an Int value
  auto inputParam = dyn_cast<LIT::LITStructAttr>(params[1]);
  if (!inputParam) {
    emitError(loc, "Error for argument '" + argName +
                       "': 'input' inferred parameter must be set");
    return std::nullopt;
  }

  auto [__, inputValueAttr] = inputParam.getValues().front();
  // The input value is now wrapped in another struct
  auto inputStructAttr = dyn_cast<LIT::LITStructAttr>(inputValueAttr);
  if (!inputStructAttr || inputStructAttr.getValues().empty()) {
    emitError(
        loc, "Error for argument '" + argName +
                 "': Expected struct attribute with value for input parameter");
    return std::nullopt;
  }

  auto [___, inputIntValueAttr] = inputStructAttr.getValues().front();
  auto inputIntAttr = dyn_cast<IntegerAttr>(inputIntValueAttr);
  if (!inputIntAttr) {
    emitError(loc,
              "Error for argument '" + argName +
                  "': Expected integer attribute for input parameter value");
    return std::nullopt;
  }
  auto inputVal = inputIntAttr.getValue().getSExtValue();

  if (isMut && inputVal == 0)
    return IOSpec::Output;
  else if (!isMut && inputVal == 1)
    return IOSpec::Input;
  else if (isMut && inputVal == 1)
    return IOSpec::MutableInput;

  emitError(loc, "Error for argument '" + argName + "': Invalid " +
                     kIOSpec.back() +
                     " param. Valid configs are: [False,True]=Input, "
                     "[True,False]=Output, [True,True]=MutableInput");
  return std::nullopt;
}

std::optional<IOSpec> findIOSpec(LIT::StructType tensorStruct, Location loc,
                                 StringRef argName) {
  for (auto param : tensorStruct.getParamValues()) {
    auto declRef = dyn_cast<KGEN::ParamDeclRefAttr>(param);

    if (declRef &&
        LIT::demangleParameterName(declRef.getName()) != kParameterIOSpec)
      continue;

    auto maybeStructType = declRef ? declRef.getType() : param.getType();

    auto structType = dyn_cast<LIT::StructType>(maybeStructType);
    if (!structType)
      continue;

    if (!symbolMatches(structType.getSymbol(), kIOSpec))
      continue;

    return parseTensorIOSpec(structType, loc, argName);
  }

  emitError(loc, "Error for argument '" + argName + "': No valid " +
                     kIOSpec.back() + " found for tensor");
  return std::nullopt;
}

static std::optional<SmallVector<std::pair<size_t, IOSpec>>>
processIOSpecs(LIT::FnOp func) {
  SmallVector<std::pair<size_t, IOSpec>> specs;

  bool error = false;
  bool foundNonOutputTensor = false;

  for (auto &&[argIdx, argType] : llvm::enumerate(func.getArgumentTypes())) {
    auto structType = getAsDeclRefOrNull(argType);

    if (!structType || !isDPSTensor(structType)) {
      foundNonOutputTensor = true;
      continue;
    }

    auto argName = func.getFuncTypeGenerator().getArgName(argIdx);
    auto loc = func.getBodyRegion().getArgument(argIdx).getLoc();
    auto ioSpec = findIOSpec(structType, loc, argName);
    if (!ioSpec) {
      error = true;
      continue;
    }

    if (*ioSpec != IOSpec::Output)
      foundNonOutputTensor = true;

    if (*ioSpec == IOSpec::Output && foundNonOutputTensor) {
      emitError(loc, "Output tensor argument '" + argName.strref() +
                         "' must come before other non-output tensor "
                         "arguments");
      continue;
    }

    specs.push_back({argIdx, *ioSpec});
  }

  if (error)
    return std::nullopt;

  return specs;
}

// Run checks and mutations on an execute function. emits an
// error and returns false if an issue was present.
//
// `moduleOp` the overarching module containing the code
// `registrationInfo` whether the struct is annotated as elementwise
// `structDeclOp` the parent struct the function being process lives in
// `registrationName` the op the struct is registered for
// `func` the func being processed
// `annotation` the annotation name to attach to this function to be used in
//    later lowering.
// `builder` builder for ops
bool processStructExecuteFunc(ModuleOp moduleOp,
                              ExtensibilityAPIStructInfo registrationInfo,
                              LIT::StructDeclOp structDeclOp, LIT::FnOp func,
                              StringLiteral annotation, Builder &builder) {
  if (failed(processStructFuncCommon(structDeclOp, registrationInfo, func,
                                     annotation, builder)))
    return false;

  // Handle extra parameters we allow on execute:
  for (auto param : func.getInputParams()) {
    if (param.getName() == kMOGGSynchronousParameterName) {
      func->setDiscardableAttr(builder.getStringAttr(kMOGGSynchronousLabel),
                               param);
    } else if (param.getName() == kMOGGTargetParameterName) {
      func->setDiscardableAttr(builder.getStringAttr(kMOGGTargetLabel), param);
    } else if (param.getName() == kMOGGLambdasHaveFusionParameterName) {
      func->setDiscardableAttr(
          builder.getStringAttr(kMOGGLambdasHaveFusionLabel), param);
    }
  }

  // Handle fusion if needed
  if (registrationInfo.isElementwiseKernel)
    func->setAttr(kMOGGElementFunction, UnitAttr::get(func->getContext()));
  if (registrationInfo.isViewKernel)
    func->setAttr(kMOGGViewKernel, UnitAttr::get(func->getContext()));

  bool enforceIOParamUsage = hasEnforceIODecorator(func);
  SmallVector<std::pair<size_t, IOSpec>> ioSpecs;
  if (enforceIOParamUsage) {
    auto result = processIOSpecs(func);
    if (!result)
      return false;
    ioSpecs = std::move(*result);
  }

  if (enforceIOParamUsage) {
    // Set mogg.num_dps_outputs
    auto numOutputs = llvm::count_if(
        ioSpecs, [](auto &&elem) { return elem.second == IOSpec::Output; });
    // TODO: should we emit an error if numDPSOperands on the register decorator
    // was set to something other than the default?
    func->setDiscardableAttr(kMOGGNumDPSOutputs,
                             builder.getIndexAttr(numOutputs));

    // Set mogg.buffer_args
    SmallVector<Attribute> mutableIdxs;
    for (auto [idx, spec] : ioSpecs) {
      if (spec == IOSpec::MutableInput) {
        mutableIdxs.push_back(builder.getIndexAttr(idx - numOutputs));
      }
    }
    if (!mutableIdxs.empty())
      func->setAttr(kMOGGBufferArgs, builder.getArrayAttr(mutableIdxs));
  } else {
    func->setDiscardableAttr(kMOGGNumDPSOutputs,
                             registrationInfo.numDPSOperands);
  }

  // Iterate over the decorators to find enable_fusion_for/mutable
  for (auto decorator : func.getDecorators()) {
    std::optional<SmallVector<TypedAttr>> mutableOperands =
        getDecoratorLambdaArgument(moduleOp, decorator, MOGG_INTRINSIC_MUTABLE,
                                   SmallVector<size_t>{1});

    std::optional<SmallVector<TypedAttr>> enableFusionOperand =
        getDecoratorLambdaArgument(moduleOp, decorator,
                                   MOGG_INTRINSIC_ENABLE_FUSION_FOR,
                                   SmallVector<size_t>{1});

    if (mutableOperands.has_value() && enforceIOParamUsage) {
      emitError(func->getLoc(),
                "Using the mutable decorator and enforce_io_param decorator at "
                "the same time is not permitted");
      return false;
    }

    if (!enableFusionOperand.has_value() && !mutableOperands.has_value())
      continue;

    ArrayAttr argSrcNamesAttr =
        dyn_cast_or_null<ArrayAttr>(func->getAttr(MOGG_ARG_SRC_NAMES));
    if (!argSrcNamesAttr)
      continue;

    SmallVector<Attribute> argIdxsAttrs;

    auto nameArgs = [&]() {
      if (enableFusionOperand)
        return cast<KGEN::VariadicAttr>(enableFusionOperand.value()[0]);

      else if (mutableOperands)
        return cast<KGEN::VariadicAttr>(mutableOperands.value()[0]);

      llvm_unreachable("Unsupported decorator!");
    }();

    for (TypedAttr operandAttr : nameArgs.getValues()) {
      auto [_, nameAttr] =
          cast<LIT::LITStructAttr>(operandAttr).getValues().front();
      StringRef argName = cast<StringAttr>(nameAttr).getValue();

      auto argIt =
          llvm::find_if(argSrcNamesAttr.getValue(), [&](Attribute attr) {
            return cast<StringAttr>(attr).getValue() == argName;
          });

      if (argIt == argSrcNamesAttr.getValue().end()) {
        StringRef decoratorName =
            enableFusionOperand ? "enable_fusion_for" : "mutable";

        emitError(func->getLoc(),
                  llvm::formatv("{0} decorator: '{1}' does not name any of the "
                                "arguments of {2}::{3}",
                                decoratorName, argName,
                                structDeclOp.getDeclName(),
                                func.getDeclName()));
        return false;
      }

      auto idx = std::distance(argSrcNamesAttr.getValue().begin(), argIt);

      if (mutableOperands) {
        idx -= registrationInfo.numDPSOperands.getInt();
      }

      argIdxsAttrs.push_back(builder.getIndexAttr(idx));
    }

    if (!argIdxsAttrs.empty() && enableFusionOperand) {
      func->setAttr(kMOGGFusableArgs, builder.getArrayAttr(argIdxsAttrs));
    }

    if (!argIdxsAttrs.empty() && mutableOperands) {
      func->setAttr(kMOGGBufferArgs, builder.getArrayAttr(argIdxsAttrs));
    }
  }
  return true;
}

class MOGGAnnotatePass
    : public M::KGEN::MOGGPreElab::impl::MOGGAnnotateBase<MOGGAnnotatePass> {
public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder{moduleOp.getContext()};

    // Do a first walk through the IR to strip the decorators and add
    // attributes. Mostly used for older extensibility API iterations
    moduleOp->walk([](Operation *operation) {
      if (auto func = dyn_cast<LIT::FnOp>(operation)) {
        stripDecorators(func);
        if (failed(annotateTypes(func)))
          return WalkResult::interrupt();
      } else if (auto structDeclOp = dyn_cast<LIT::StructDeclOp>(operation)) {
        stripDecorators(structDeclOp);
      }

      return WalkResult::advance();
    });

    // Walk to process struct-based extensibility kernels.
    auto walker = [&](LIT::StructDeclOp structDeclOp) {
      auto decorators = structDeclOp.getDecorators();
      if (decorators.empty())
        return WalkResult::advance();

      ExtensibilityAPIStructInfo registrationInfo;
      ErrorOr<bool> isExtensibilityStruct =
          isExtensibilityAPIStruct(structDeclOp, moduleOp, registrationInfo);
      if (isExtensibilityStruct.isError()) {
        structDeclOp->emitError(isExtensibilityStruct.getError());
        return WalkResult::interrupt();
      }

      // Is not extensibility struct, but maybe some regular mojo object
      if (!isExtensibilityStruct.takeValue())
        return WalkResult::advance();
      LIT::FnOp executeOp;
      for (auto &curOp : structDeclOp.getFields().front()) {
        auto func = dyn_cast<LIT::FnOp>(curOp);
        if (!func)
          continue;

        if (func.getSourceName() == kExecuteFuncName) {
          if (!processStructExecuteFunc(moduleOp, registrationInfo,
                                        structDeclOp, func,
                                        kMOGGExecuteFunctionLabel, builder))
            return WalkResult::interrupt();
          executeOp = func;
        } else if (func.getSourceName() == kShapeFuncName) {
          if (failed(processStructFuncCommon(structDeclOp, registrationInfo,
                                             func, kMOGGShapeFunctionLabel,
                                             builder)))
            return WalkResult::interrupt();
        } else if (func.getSourceName() == kPyTorchFallbackFuncName) {
          if (failed(processStructFuncCommon(
                  structDeclOp, registrationInfo, func,
                  kMOGGPyTorchFallbackFunctionLabel, builder)))
            return WalkResult::interrupt();
        }
      }

      // Some struct verifiers
      if (!executeOp) {
        structDeclOp.emitError(llvm::formatv(
            "The kernel must have an entry point named {0}", kExecuteFuncName));
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    };

    if (moduleOp.walk(walker).wasInterrupted())
      signalPassFailure();

    moduleOp.walk([](LIT::FnOp funcOp) { labelTensorParamsInKernel(funcOp); });
  }
};
} // namespace
