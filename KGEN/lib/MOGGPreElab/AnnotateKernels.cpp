//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MOGGPreElab/MOGGPreElabDecorators.h"
#include "KGEN/MOGGPreElab/MOGGPreElabHelpers.h"
#include "KGEN/MOGGPreElab/MOGGPreElabUtils.h"
#include "KGEN/MOGGPreElab/Passes.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/AssertStream.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;
using namespace MOGGPreElab;

namespace M::KGEN::MOGGPreElab {
#define GEN_PASS_DEF_ANNOTATEKERNELS
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
static constexpr std::array<StringLiteral, 3> kMaxVariadicTensors = {
    "tensor_internal", "managed_tensor_slice", "VariadicTensors"};
static constexpr std::array<StringLiteral, 4> kMaxList = {
    "stdlib", "collections", "list", "List"};
static constexpr std::array<StringLiteral, 4> kPythonObject = {
    "stdlib", "python", "python_object", "PythonObject"};

// TODO(GEX-1822): Should be able to query this information from
// The lit.struct.decl ops for each of these types rather than hard-coding them.
enum class ManagedTensorSliceParams {
  kRead,
  kWrite,
  kDType,
  kRank,
  kIOSpec,
  kStaticSpec,
  kNumParams
};

enum class VariadicTensorsParams {
  kRead,
  kWrite,
  kDType,
  kRank,
  kSize,
  kIOSpec,
  kStaticSpecs,
  kNumParams
};

enum class SIMDParams { kDType, kSize, kNumParams };

// Parameter indexes for List struct
enum class ListParams { kElementType, kNumParams };

// Helper function to convert enum class to underlying index
template <typename T>
constexpr int toIndex(T value) {
  return static_cast<int>(value);
}

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

  if (!isKernel(func) && !isDPSKernel(func) && !takesTensor)
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

/// Extract the read and write fields from a ManagedTensorSlice struct type
/// Returns a pair of (read, write) TypedAttrs
static std::pair<TypedAttr, TypedAttr>
extractReadWriteFromTensorStruct(LIT::StructType structType) {
  auto allParameters = structType.getParamValues();
  ASSERT_STREAM(
      allParameters.size() >= toIndex(ManagedTensorSliceParams::kNumParams),
      << "Expected at least " << toIndex(ManagedTensorSliceParams::kNumParams)
      << " parameters on the tensor type");

  auto read = allParameters[toIndex(ManagedTensorSliceParams::kRead)];
  auto write = allParameters[toIndex(ManagedTensorSliceParams::kWrite)];

  return std::make_pair(read, write);
}

/// Return a set of named attributes mapping all unbound parameters in the
/// tensor type struct
static SmallVector<NamedAttribute>
getUnboundParametersForTensor(LIT::StructType &structType, Builder &builder) {
  auto allParameters = structType.getParamValues();
  ASSERT_STREAM(
      allParameters.size() >= toIndex(ManagedTensorSliceParams::kNumParams),
      << "Expected at least " << toIndex(ManagedTensorSliceParams::kNumParams)
      << " parameters on the tensor type");

  auto read = allParameters[toIndex(ManagedTensorSliceParams::kRead)];
  auto write = allParameters[toIndex(ManagedTensorSliceParams::kWrite)];
  auto dtype = allParameters[toIndex(ManagedTensorSliceParams::kDType)];
  auto rank = allParameters[toIndex(ManagedTensorSliceParams::kRank)];
  auto spec = allParameters[toIndex(ManagedTensorSliceParams::kStaticSpec)];

  SmallVector<NamedAttribute> tensorSpecNamedAttrs;
  if (dtype)
    tensorSpecNamedAttrs.emplace_back(builder.getStringAttr(kParameterDType),
                                      dtype);

  if (rank)
    tensorSpecNamedAttrs.emplace_back(builder.getStringAttr(kParameterRank),
                                      rank);

  if (spec)
    tensorSpecNamedAttrs.emplace_back(
        builder.getStringAttr(kParameterStaticSpec), spec);

  if (read)
    tensorSpecNamedAttrs.emplace_back(builder.getStringAttr(kParameterRead),
                                      read);

  if (write)
    tensorSpecNamedAttrs.emplace_back(builder.getStringAttr(kParameterWrite),
                                      write);

  return tensorSpecNamedAttrs;
}

static SmallVector<NamedAttribute>
getUnboundParametersForSIMD(LIT::StructType structType, Builder &builder) {
  auto allParameters = structType.getParamValues();
  ASSERT_STREAM(allParameters.size() >= toIndex(SIMDParams::kNumParams),
                << "Expected at least " << toIndex(SIMDParams::kNumParams)
                << " parameters on the SIMD type");

  auto dtype = allParameters[toIndex(SIMDParams::kDType)];
  auto size = allParameters[toIndex(SIMDParams::kSize)];

  SmallVector<NamedAttribute> tensorSpecNamedAttrs;
  if (dtype)
    tensorSpecNamedAttrs.push_back(
        NamedAttribute{builder.getStringAttr(kParameterDType), dtype});
  if (size)
    tensorSpecNamedAttrs.push_back(
        NamedAttribute{builder.getStringAttr(kParameterSize), size});

  return tensorSpecNamedAttrs;
}

/// Return a set of named attributes mapping all unbound parameters in the tuple
/// of tensor struct
///
//  read: Bool,
//  write: Bool, //,
//  type: DType,
//  rank: Int,
//  size: Int,
//  ioSpec: IOSpec[read, write],
//  *,
//  static_specs: StaticTuple[StaticTensorSpec[type, rank], size],
static SmallVector<NamedAttribute>
getUnboundParametersForVariadicTensors(LIT::StructType structType,
                                       Builder &builder) {
  auto allParameters = structType.getParamValues();
  ASSERT_STREAM(
      allParameters.size() >= toIndex(VariadicTensorsParams::kNumParams),
      << "Expected at least " << toIndex(VariadicTensorsParams::kNumParams)
      << " parameters on the tuple-of-tensors type");

  auto read = allParameters[toIndex(VariadicTensorsParams::kRead)];
  auto write = allParameters[toIndex(VariadicTensorsParams::kWrite)];
  auto type = allParameters[toIndex(VariadicTensorsParams::kDType)];
  auto rank = allParameters[toIndex(VariadicTensorsParams::kRank)];
  auto size = allParameters[toIndex(VariadicTensorsParams::kSize)];
  auto spec = allParameters[toIndex(VariadicTensorsParams::kStaticSpecs)];

  SmallVector<NamedAttribute> namedAttrs;
  if (read)
    namedAttrs.emplace_back(builder.getStringAttr(kParameterRead), read);

  if (write)
    namedAttrs.emplace_back(builder.getStringAttr(kParameterWrite), write);

  if (type)
    namedAttrs.emplace_back(builder.getStringAttr(kParameterDType), type);

  if (rank)
    namedAttrs.emplace_back(builder.getStringAttr(kParameterRank), rank);

  if (size)
    namedAttrs.emplace_back(builder.getStringAttr(kParameterSize), size);

  if (spec)
    namedAttrs.emplace_back(builder.getStringAttr(kParameterStaticSpecs), spec);

  return namedAttrs;
}

/// Return a set of named attributes mapping all unbound parameters in the list
/// of tensor struct
static std::optional<SmallVector<NamedAttribute>>
getUnboundParametersForTensorList(LIT::StructType structType,
                                  Builder &builder) {
  // TODO(GEX-1126): consider a tuple which only contains tensors to
  // simplify this
  [[maybe_unused]] auto allParameters = structType.getParamValues();

  ASSERT_STREAM(allParameters.size() >= toIndex(ListParams::kNumParams),
                << "Expected at least " << toIndex(ListParams::kNumParams)
                << " parameters on the list-of-tensor type");
  SmallVector<NamedAttribute> listNamedAttrs;

  auto elementTypeAttr = cast<KGEN::TypeParamAttr>(
      structType.getParamValues()[toIndex(ListParams::kElementType)]);
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
  Attribute emptyAttr = builder.getUnitAttr();

  for (auto [i, litType] : llvm::enumerate(funcOp.getArgumentTypes())) {
    auto asStructType = getAsStructType(litType);

    if (!asStructType) {
      tensorSpecs.push_back(emptyAttr);
      continue;
    }

    if (symbolMatches(asStructType.getSymbol(), kMaxManagedTensorSlice)) {
      SmallVector<NamedAttribute> tensorSpecNamedAttrs =
          getUnboundParametersForTensor(asStructType, builder);
      tensorSpecs.push_back(builder.getDictionaryAttr(tensorSpecNamedAttrs));
    } else if (symbolMatches(asStructType.getSymbol(), kMaxSIMD)) {
      SmallVector<NamedAttribute> tensorSpecNamedAttrs =
          getUnboundParametersForSIMD(asStructType, builder);
      tensorSpecs.push_back(builder.getDictionaryAttr(tensorSpecNamedAttrs));
    } else if (symbolMatches(asStructType.getSymbol(), kMaxVariadicTensors)) {
      auto tensorSpecNamedAttrs =
          getUnboundParametersForVariadicTensors(asStructType, builder);
      tensorSpecs.push_back(builder.getDictionaryAttr(tensorSpecNamedAttrs));
    } else if (symbolMatches(asStructType.getSymbol(), kMaxList)) {
      auto tensorSpecNamedAttrs =
          getUnboundParametersForTensorList(asStructType, builder);
      if (!tensorSpecNamedAttrs) {
        tensorSpecs.push_back(emptyAttr);
      } else {
        tensorSpecs.push_back(builder.getDictionaryAttr(*tensorSpecNamedAttrs));
      }
    } else {
      // Unsupported type, can ignore
      tensorSpecs.push_back(emptyAttr);
    }
  }
  funcOp->setDiscardableAttr(kKernelValueParameterAttrName,
                             builder.getArrayAttr(tensorSpecs));
}

namespace {

// Important metadata about the structs under the extensibility API
struct ExtensibilityAPIStructInfo {
  // Whether the operation is marked as an elementwise kernel
  bool isElementwiseKernel = false;

  // Whether the operation is marked as a view kernel.
  bool isViewKernel = false;

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
                                   SmallVector<size_t>{1});
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
  }

  return registrationInfo.registrationName != nullptr;
}

/// Extract the read and write fields from various tensor-related struct types
/// Returns a pair of (read, write) TypedAttrs
static std::pair<TypedAttr, TypedAttr>
extractIOSpecSubFields(LIT::StructType structType) {
  // Check if this is a ManagedTensorSlice
  if (symbolMatches(structType.getSymbol(), kMaxManagedTensorSlice)) {
    return extractReadWriteFromTensorStruct(structType);
  }

  // Check if this is a VariadicTensors
  if (symbolMatches(structType.getSymbol(), kMaxVariadicTensors)) {
    auto allParameters = structType.getParamValues();
    ASSERT_STREAM(
        allParameters.size() >= 7,
        << "Expected at least 7 parameters on the variadic tensors type");

    auto read = allParameters[toIndex(VariadicTensorsParams::kRead)];
    auto write = allParameters[toIndex(VariadicTensorsParams::kWrite)];
    return std::make_pair(read, write);
  }

  // Check if this is a list of tensors
  if (symbolMatches(structType.getSymbol(), kMaxList)) {
    static constexpr unsigned kElementType = 0;
    [[maybe_unused]] auto allParameters = structType.getParamValues();

    ASSERT_STREAM(allParameters.size() >= 2,
                  << "Expected at least two parameters on the list type");

    auto elementTypeAttr =
        cast<KGEN::TypeParamAttr>(structType.getParamValues()[kElementType]);
    auto elementTypeStruct =
        cast<LIT::StructType>(elementTypeAttr.getTypeValue());

    // Only handle lists of ManagedTensorSlice
    if (!symbolMatches(elementTypeStruct.getSymbol(), kMaxManagedTensorSlice)) {
      return std::make_pair(TypedAttr(), TypedAttr());
    }

    return extractReadWriteFromTensorStruct(elementTypeStruct);
  }

  // Unsupported type
  return std::make_pair(TypedAttr(), TypedAttr());
}

static std::optional<IOSpec> maybeGetIOSpec(TypedAttr readAttr,
                                            TypedAttr writeAttr, Location loc,
                                            StringRef argName,
                                            bool isShapeFunc) {
  auto processAttribute =
      [&](TypedAttr attr,
          llvm::StringLiteral paramName) -> std::optional<bool> {
    auto structAttr = dyn_cast<LIT::LITStructAttr>(attr);
    if (!structAttr) {
      if (!isShapeFunc) {
        emitError(loc, "Error for argument '" + argName + "': '" +
                           paramName.str() +
                           "' inferred parameter must be set");
      }
      return std::nullopt;
    }

    auto [_, valueAttr] = structAttr.getValues().front();
    auto intAttr = dyn_cast<IntegerAttr>(valueAttr);
    ASSERT_STREAM(intAttr, << "Error for argument '" << argName
                           << "': Expected integer attribute for "
                           << paramName.str() << " parameter value");
    return intAttr.getValue().getBoolValue();
  };

  auto readValue = processAttribute(readAttr, kParameterRead);
  auto writeValue = processAttribute(writeAttr, kParameterWrite);

  if (!readValue || !writeValue) {
    return std::nullopt;
  }

  auto read = readValue.value();
  auto write = writeValue.value();

  if (!read && write)
    return IOSpec::OutputTensor;
  else if (read && !write)
    return IOSpec::InputTensor;
  else if (read && write)
    return IOSpec::MutableInputTensor;

  emitError(loc, "Error for argument '" + argName + "': Invalid " +
                     kIOSpec.back() +
                     " param. Valid configs are: [True,False]=Input, "
                     "[False,True]=Output, [True,True]=MutableInput");

  return std::nullopt;
}

static std::optional<SmallVector<std::pair<size_t, IOSpec>>>
processIOSpecs(LIT::FnOp func, bool isShapeFunc = false) {
  SmallVector<std::pair<size_t, IOSpec>> specs;

  bool error = false;
  bool foundNonOutputOperand = false;

  for (auto &&[argIdx, argType] : llvm::enumerate(func.getArgumentTypes())) {
    auto structType = getAsDeclRefOrNull(argType);

    if (!structType) {
      foundNonOutputOperand = true;
      continue;
    }

    auto [read, write] = extractIOSpecSubFields(structType);

    if (!read && !write)
      continue;

    auto argName = func.getFuncTypeGenerator().getArgName(argIdx);
    auto loc = func.getBodyRegion().getArgument(argIdx).getLoc();
    auto ioSpec = maybeGetIOSpec(read, write, loc, argName, isShapeFunc);

    auto hasIOSpec = ioSpec.has_value();

    if (isShapeFunc && (!hasIOSpec || *ioSpec != IOSpec::InputTensor)) {
      emitError(loc, "Error for argument '" + argName.strref() +
                         "': Tensor arguments to shape functions must be "
                         "'InputTensor'");
      error = true;
      continue;
    }

    if (!hasIOSpec) {
      error = true;
      continue;
    }

    if (symbolMatches(structType.getSymbol(), kMaxList) &&
        ioSpec != IOSpec::InputTensor) {
      // TODO(GEX_1882)
      emitError(loc, "Only input tensors are allowed as the element type for "
                     "list arguments at the moment.");
    }

    if (*ioSpec != IOSpec::OutputTensor)
      foundNonOutputOperand = true;

    if (*ioSpec == IOSpec::OutputTensor && foundNonOutputOperand) {
      emitError(loc,
                "Output tensor argument '" +
                    func.getFuncTypeGenerator().getArgName(argIdx).strref() +
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

LogicalResult allTypesArePythonObject(LIT::FnOp pytorch_fallback) {

  auto isPythonObject = [](Type type) {
    auto structType = getAsDeclRefOrNull(type);

    if (!structType)
      return false;

    return symbolMatches(structType.getSymbol(), kPythonObject);
  };

  bool error = false;
  // Check argument types
  for (auto &&[argIdx, argType] :
       llvm::enumerate(pytorch_fallback.getArgumentTypes())) {

    if (isPythonObject(argType))
      continue;

    auto argName = pytorch_fallback.getFuncTypeGenerator().getArgName(argIdx);
    auto loc = pytorch_fallback.getBodyRegion().getArgument(argIdx).getLoc();

    emitError(loc, "Error for argument '" + argName.strref() +
                       "' all arguments to 'pytorch_fallback' functions must "
                       "have type 'PythonObject'");

    error = true;
  }

  auto resultType = pytorch_fallback.getFuncTypeGenerator().getUserResultType();
  if (!isPythonObject(resultType)) {
    pytorch_fallback.emitError(
        "Error for result type: the only permitted return type for "
        "'pytorch_fallback' functions is 'PythonObject'");

    error = true;
  }

  return failure(error);
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

  if (annotation == kMOGGShapeFunctionLabel) {
    auto result = processIOSpecs(func, /*isShapeFunc=*/true);
    if (!result.has_value())
      return failure();
  }

  if (annotation == kMOGGPyTorchFallbackFunctionLabel) {
    if (failed(allTypesArePythonObject(func)))
      return failure();
  }

  func->setAttr(builder.getStringAttr(annotation),
                registrationInfo.registrationName);
  func.setExported();
  if (failed(annotateTypes(func)))
    return failure();

  return success();
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
    } else if (param.getName() == kMOGGTraceNameParameterName) {
      func->setDiscardableAttr(builder.getStringAttr(kMOGGTraceNameLabel),
                               param);
    }
  }

  // Handle fusion if needed
  if (registrationInfo.isElementwiseKernel)
    func->setAttr(kMOGGElementFunction, UnitAttr::get(func->getContext()));
  if (registrationInfo.isViewKernel)
    func->setAttr(kMOGGViewKernel, UnitAttr::get(func->getContext()));

  SmallVector<std::pair<size_t, IOSpec>> ioSpecs;
  auto result = processIOSpecs(func);
  if (!result)
    return false;
  ioSpecs = std::move(*result);

  auto numOutputs = llvm::count_if(
      ioSpecs, [](auto &&elem) { return elem.second == IOSpec::OutputTensor; });
  func->setDiscardableAttr(kMOGGNumDPSOutputs,
                           builder.getIndexAttr(numOutputs));

  // Set mogg.buffer_args
  SmallVector<Attribute> mutableIdxs;
  for (auto [idx, spec] : ioSpecs) {
    if (spec == IOSpec::MutableInputTensor) {
      mutableIdxs.push_back(builder.getIndexAttr(idx - numOutputs));
    }
  }
  if (!mutableIdxs.empty())
    func->setAttr(kMOGGBufferArgs, builder.getArrayAttr(mutableIdxs));

  // Iterate over the decorators to find enable_fusion_for/mutable
  for (auto decorator : func.getDecorators()) {
    std::optional<SmallVector<TypedAttr>> enableFusionOperand =
        getDecoratorLambdaArgument(moduleOp, decorator,
                                   MOGG_INTRINSIC_ENABLE_FUSION_FOR,
                                   SmallVector<size_t>{1});

    if (!enableFusionOperand.has_value())
      continue;

    ArrayAttr argSrcNamesAttr =
        dyn_cast_or_null<ArrayAttr>(func->getAttr(MOGG_ARG_SRC_NAMES));
    if (!argSrcNamesAttr)
      continue;

    SmallVector<Attribute> argIdxsAttrs;

    auto nameArgs = [&]() {
      if (enableFusionOperand)
        return cast<KGEN::VariadicAttr>(enableFusionOperand.value()[0]);

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
        StringRef decoratorName = "enable_fusion_for";

        emitError(func->getLoc(),
                  llvm::formatv("{0} decorator: '{1}' does not name any of the "
                                "arguments of {2}::{3}",
                                decoratorName, argName,
                                structDeclOp.getDeclName(),
                                func.getDeclName()));
        return false;
      }

      auto idx = std::distance(argSrcNamesAttr.getValue().begin(), argIt);

      argIdxsAttrs.push_back(builder.getIndexAttr(idx));
    }

    if (!argIdxsAttrs.empty() && enableFusionOperand) {
      func->setAttr(kMOGGFusableArgs, builder.getArrayAttr(argIdxsAttrs));
    }
  }
  return true;
}

class AnnotateKernelsPass
    : public M::KGEN::MOGGPreElab::impl::AnnotateKernelsBase<
          AnnotateKernelsPass> {
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
