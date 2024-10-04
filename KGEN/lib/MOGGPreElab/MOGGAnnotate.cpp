//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringSet.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/MOGGPreElab/MOGGDecorators.h"
#include "KGEN/MOGGPreElab/MOGGUtils.h"
#include "KGEN/MOGGPreElab/Passes.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPOps.h"
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
static constexpr llvm::StringLiteral kInitializeOutputFuncName =
    "initialize_output";
static constexpr llvm::StringLiteral kPyTorchFallbackFuncName =
    "pytorch_fallback";
static constexpr std::array<StringLiteral, 3> kMaxManagedTensorSlice = {
    "tensor_utils", "managed_tensor_slice", "ManagedTensorSlice"};

// Check if the decorator correspond to a function call:
// eg @foo(arg)
// Also check that foo is marked with the attribute `expectedFuncAttr`.
// Returns arg (or nullptr if invalid).
static TypedAttr getDecoratorLambdaArgument(ModuleOp mod, TypedAttr decorator,
                                            StringRef expectedFuncAttr) {
  auto apply = dyn_cast<ParamOperatorAttr>(decorator);
  if (!apply || apply.getNumOperands() != 2)
    return nullptr;

  auto sym = dyn_cast<SymbolConstantAttr>(apply.getOperand(0));
  if (!sym)
    return nullptr;

  auto decoratorFunc = mod.lookupSymbol<LIT::FuncOp>(sym.getSymbol());
  if (!decoratorFunc || !decoratorFunc->hasAttr(expectedFuncAttr))
    return nullptr;

  return apply.getOperand(1);
}

static void annotateTypes(LIT::FuncOp func) {
  // Look through ref types to get underlaying decl ref type if needed.
  auto getAsDeclRefOrNull = [&](Type t) {
    auto asLitRef = dyn_cast<LIT::RefType>(t);
    if (asLitRef)
      return dyn_cast<LIT::StructType>(asLitRef.getElementType());
    return dyn_cast<LIT::StructType>(t);
  };

  // Anything taking a tensor needs the annotation.
  bool takesTensor = false;
  for (Type litType : func.getArgumentTypes()) {
    if (LIT::StructType asDeclRef = getAsDeclRefOrNull(litType)) {
      takesTensor |= isMOGGTensor(asDeclRef);
      takesTensor |= isExtensibilityTensor(asDeclRef);
      takesTensor |= isDPSTensor(asDeclRef);
    }
  }

  if (!isKernel(func) && !isV1ShapeFunc(func) && !isDPSKernel(func) &&
      !takesTensor)
    return;

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

    sourceName.push_back(func.getSignature().getArgName(i));
  }

  // Attach the parameter mapping infomation to the kernel.
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
}

static void labelTensorParamsInKernel(LIT::FuncOp funcOp) {
  OpBuilder builder{funcOp.getContext()};

  if (!isDPSKernel(funcOp))
    return;

  // Look through ref types to get underlying decl ref type if needed.
  auto getAsStructType = [](Type t) {
    auto asLitRef = dyn_cast<LIT::RefType>(t);
    if (asLitRef)
      return dyn_cast<LIT::StructType>(asLitRef.getElementType());
    return dyn_cast<LIT::StructType>(t);
  };

  // Extract the used parameters from the lit type.
  auto litTypeToParams = [](LIT::StructType structType) {
    SmallVector<KGEN::ParamDeclRefAttr> attrs;
    for (TypedAttr param : structType.getParamValues()) {
      auto declRefAttr = dyn_cast<KGEN::ParamDeclRefAttr>(param);
      attrs.push_back(declRefAttr);
    }

    return attrs;
  };

  SmallVector<Attribute> tensorSpecs;
  Attribute emptyAttr = builder.getUnitAttr();
  for (auto [i, litType] : llvm::enumerate(funcOp.getArgumentTypes())) {
    auto asStructType = getAsStructType(litType);
    if (!asStructType ||
        !symbolMatches(asStructType.getSymbol(), kMaxManagedTensorSlice)) {
      tensorSpecs.push_back(emptyAttr);
      continue;
    }

    constexpr unsigned kDTypeIndex = 0;
    constexpr unsigned kRankIndex = 1;
    auto allParameters = litTypeToParams(asStructType);
    assert(allParameters.size() >= 2);
    auto dtype = allParameters[kDTypeIndex];
    auto rank = allParameters[kRankIndex];

    SmallVector<NamedAttribute> tensorSpecNamedAttrs;
    // Sometimes, dtype or ranks are not present because the user expects
    // specific values for those parameters (ex: dtype=float32 or rank=2).
    if (dtype)
      tensorSpecNamedAttrs.push_back(
          NamedAttribute{builder.getStringAttr("dtype"), dtype});
    if (rank)
      tensorSpecNamedAttrs.push_back(
          NamedAttribute{builder.getStringAttr("rank"), rank});

    tensorSpecs.push_back(
        DictionaryAttr::get(funcOp.getContext(), tensorSpecNamedAttrs));
  }
  funcOp->setDiscardableAttr(kKernelTensorParameterAttrName,
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
          moduleOp.lookupSymbol<LIT::FuncOp>(directSym.getSymbol());

      if (decoratorFunc && decoratorFunc->hasAttr(MOGG_INTRINSIC_ELEMENTWISE)) {
        if (registrationInfo.isElementwiseKernel)
          return Error("Op has multiple elementwise annotations");
        registrationInfo.isElementwiseKernel = true;
        continue;
      }

      if (decoratorFunc && decoratorFunc->hasAttr(MOGG_INTRINSIC_VIEW_KERNEL)) {
        if (registrationInfo.isViewKernel)
          return Error("Op has multiple view annotations");
        registrationInfo.isViewKernel = true;
        continue;
      }
    }

    TypedAttr registerOperand = getDecoratorLambdaArgument(
        moduleOp, decorator, MOGG_INTRINSIC_REGISTER);
    if (!registerOperand)
      continue;

    auto [_, nameAttr] =
        cast<LIT::LITStructAttr>(registerOperand).getValues().front();
    auto name = dyn_cast<StringAttr>(nameAttr);
    assert(name);
    if (registrationInfo.registrationName) {
      return Error("Only one op can be registered per kernel struct");
    }
    registrationInfo.registrationName = name;
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
bool processStructFuncCommon(LIT::StructDeclOp structDeclOp,
                             ExtensibilityAPIStructInfo registrationInfo,
                             LIT::FuncOp func, StringLiteral annotation,
                             OpBuilder &builder) {
  if (!func.getIsStatic()) {
    func->emitError("Function is not static");
    return false;
  }

  func->setAttr(builder.getStringAttr(annotation),
                registrationInfo.registrationName);
  func.setExported();
  annotateTypes(func);

  return true;
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
                              LIT::StructDeclOp structDeclOp, LIT::FuncOp func,
                              StringLiteral annotation, OpBuilder &builder) {
  if (!processStructFuncCommon(structDeclOp, registrationInfo, func, annotation,
                               builder))
    return false;

  // Handle extra parameters we allow on execute:
  for (auto param : func.getInputParams()) {
    if (param.getName() == kMOGGSynchronousParameterName) {
      func->setDiscardableAttr(builder.getStringAttr(kMOGGSynchronousLabel),
                               param);
    } else if (param.getName() == kMOGGTargetParameterName) {
      func->setAttr(builder.getStringAttr(kMOGGTargetLabel), param);
    }
  }

  // Handle fusion if needed
  if (registrationInfo.isElementwiseKernel)
    func->setAttr(kMOGGElementFunction, UnitAttr::get(func->getContext()));
  if (registrationInfo.isViewKernel)
    func->setAttr(kMOGGViewKernel, UnitAttr::get(func->getContext()));

  // Iterate over the decorators to find enable_fusion_for
  for (auto decorator : func.getDecorators()) {
    TypedAttr enableFusionOperand = getDecoratorLambdaArgument(
        moduleOp, decorator, MOGG_INTRINSIC_ENABLE_FUSION_FOR);
    if (!enableFusionOperand)
      continue;

    ArrayAttr argSrcNamesAttr =
        dyn_cast_or_null<ArrayAttr>(func->getAttr(MOGG_ARG_SRC_NAMES));
    if (!argSrcNamesAttr)
      continue;

    SmallVector<Attribute> argIdxsAttrs;
    auto nameArgs = cast<KGEN::VariadicAttr>(enableFusionOperand);
    for (TypedAttr operandAttr : nameArgs.getValues()) {
      auto [_, nameAttr] =
          cast<LIT::LITStructAttr>(operandAttr).getValues().front();
      StringRef argName = cast<StringAttr>(nameAttr).getValue();

      auto argIt =
          llvm::find_if(argSrcNamesAttr.getValue(), [&](Attribute attr) {
            return cast<StringAttr>(attr).getValue() == argName;
          });
      if (argIt == argSrcNamesAttr.getValue().end()) {
        emitError(structDeclOp->getLoc(),
                  "enable_fusion_for decorator: invalid argument "
                  "name");
        return false;
      }
      argIdxsAttrs.push_back(builder.getIndexAttr(
          std::distance(argSrcNamesAttr.getValue().begin(), argIt)));
    }

    if (!argIdxsAttrs.empty()) {
      func->setAttr(kMOGGFusableArgs, builder.getArrayAttr(argIdxsAttrs));
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
      if (auto func = dyn_cast<LIT::FuncOp>(operation)) {
        stripDecorators(func);
        annotateTypes(func);
      } else if (auto structDeclOp = dyn_cast<LIT::StructDeclOp>(operation)) {
        stripDecorators(structDeclOp);
      }
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

      LIT::FuncOp executeOp, shapeOp, initializeOutputOp;
      for (auto &curOp : structDeclOp.getFields().front()) {
        auto func = dyn_cast<LIT::FuncOp>(curOp);
        if (!func)
          continue;

        if (func.getSourceName() == kExecuteFuncName) {
          if (!processStructExecuteFunc(moduleOp, registrationInfo,
                                        structDeclOp, func,
                                        kMOGGExecuteFunctionLabel, builder))
            return WalkResult::interrupt();
          executeOp = func;
        } else if (func.getSourceName() == kShapeFuncName) {
          if (!processStructFuncCommon(structDeclOp, registrationInfo, func,
                                       kMOGGShapeFunctionLabel, builder))
            return WalkResult::interrupt();
          shapeOp = func;
        } else if (func.getSourceName() == kInitializeOutputFuncName) {
          if (!processStructFuncCommon(structDeclOp, registrationInfo, func,
                                       kMOGGInitializeOutputFunctionLabel,
                                       builder))
            return WalkResult::interrupt();
          initializeOutputOp = func;
        } else if (func.getSourceName() == kPyTorchFallbackFuncName) {
          if (!processStructFuncCommon(structDeclOp, registrationInfo, func,
                                       kMOGGPyTorchFallbackFunctionLabel,
                                       builder))
            return WalkResult::interrupt();
        }
      }

      // Some struct verifiers
      if (!executeOp && !initializeOutputOp) {
        structDeclOp.emitError("Struct based extensibility needs execute or "
                               "initialize_output!");
        return WalkResult::interrupt();
      }

      if (executeOp && initializeOutputOp) {
        structDeclOp.emitError("Struct based extensibility cannot have "
                               "execute and initialize_output op!");
        return WalkResult::interrupt();
      }

      if (initializeOutputOp && shapeOp) {
        structDeclOp.emitError("Struct based extensibility cannot have "
                               "initialize_output and shape op!");
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    };

    if (moduleOp.walk(walker).wasInterrupted())
      signalPassFailure();

    moduleOp.walk(
        [](LIT::FuncOp funcOp) { labelTensorParamsInKernel(funcOp); });
  }
};
} // namespace
