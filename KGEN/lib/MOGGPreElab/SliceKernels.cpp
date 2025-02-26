//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MOGGPreElab/MOGGPreElabDecorators.h"
#include "KGEN/MOGGPreElab/MOGGPreElabHelpers.h"
#include "KGEN/MOGGPreElab/Passes.h"
#include "Support/DebugInfoDialect/Transforms/StripDebugInfo.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

#include "Support/AssertStream.h"

using namespace M;
using namespace KGEN;
using namespace MOGGPreElab;

namespace {
struct NameUniquer {
public:
  NameUniquer(MLIRContext *ctx) : ctx{ctx} {}

  void collectReservedNames(Operation *op);

  template <typename AttrOrType>
  void collectReservedNames(AttrOrType attrOrType);

  StringAttr newName();

private:
  MLIRContext *ctx;
  DenseSet<StringAttr> reservedNames;
  unsigned counter = 0;
};

template <typename AttrOrType>
void NameUniquer::collectReservedNames(AttrOrType type) {
  mlir::AttrTypeWalker walker;
  walker.addWalk(
      [&](ParamDeclRefAttr attr) { reservedNames.insert(attr.getName()); });

  walker.addWalk(
      [&](ParamDeclAttr attr) { reservedNames.insert(attr.getName()); });
}

void NameUniquer::collectReservedNames(Operation *op) {
  op->walk([&](Operation *op) {
    collectReservedNames(op->getAttrDictionary());
    for (Type type : op->getResultTypes())
      collectReservedNames(type);
    for (Region &region : op->getRegions())
      for (Type type : region.getArgumentTypes())
        collectReservedNames(type);
  });
}

StringAttr NameUniquer::newName() {
  while (true) {
    auto name = StringAttr::get(ctx, Twine("P") + Twine(counter++));

    if (!reservedNames.contains(name))
      return name;
  }
}

struct SpliceResult {
  TypedAttr getNewTensorSpec() { return newTensorSpec; }

  KGEN::ParamDeclRefAttr getNewLambdaRef() {
    auto declAttr = newLambda.getParamDecl();
    return KGEN::ParamDeclRefAttr::get(declAttr);
  }

  bool contains(Operation *op) { return op == newLambda; }

  ParamDeclareRegionOp newLambda;
  TypedAttr newTensorSpec;
};

/// We have a special mojo hook which show us what the canonical lambda
/// looks like and a call which tells us the resulting type with the lambda
/// applied.
struct LambdaTemplate {
  LambdaTemplate() = default;

  /// Scan the hook for the properties that we know exist.
  LambdaTemplate(GeneratorOp hook) : templateOp(hook) {
    auto regions = hook.getOps<ParamDeclareRegionOp>();
    ASSERT_STREAM(
        llvm::hasSingleElement(regions),
        << "There must be exactly one region in the I/O lambda intrinsic");
    canonicalLambda = *regions.begin();

    auto calls = hook.getOps<CallOp>();
    ASSERT_STREAM(
        llvm::hasSingleElement(calls),
        << "There must be exactly one call op in the I/O lambda intrinsic");
    extractTensorSpecCall = *calls.begin();
  }

  /// Splice the template into the location pointed to by the given builder.
  /// The newly spliced input/output lambda will be named newLambdaName.
  ///
  /// The return value is the new tensor spec attribute with any parameter
  /// references remapped.
  SpliceResult splice(OpBuilder &builder, NameUniquer &uniquer,
                      ValueRange operands, DictionaryAttr params,
                      StringRef newLambdaName);

  /// The op we are pulling this info from.
  GeneratorOp templateOp;

  /// This the the template lambda we will clone as the input or output
  /// lambda.
  ParamDeclareRegionOp canonicalLambda;

  /// The call operation which will hold the updated StaticTensorSpec
  /// with a new input or output lambda.
  CallOp extractTensorSpecCall;
};

/// What is going on here. We need tooling to do the following.
/// We have the set of parameters from a ManagedTensorSlice:
///
///     ManagedTensorSlice[
///         type,
///         rank + 1,
///         static_spec = StaticTensorSpec[type, rank]()
///     ]
///
/// And we need to replace each parameter of the input hook function with
/// these parameters in a consistent way.
///
/// fn _input_hook_fn[type, rank, static_spec]():
///     use[type, rank, static_spec]()
///
class SafeRenamer {
public:
  SafeRenamer(NameUniquer &uniquer) : uniquer{uniquer} {}

  SafeRenamer(NameUniquer &uniquer, ArrayRef<ParamDeclAttr> oldNames,
              ArrayRef<TypedAttr> newValues)
      : uniquer{uniquer} {
    for (auto [name, value] : llvm::zip(oldNames, newValues))
      setParameterValue(name.getName(), value);
  }

  void setParameterValue(StringAttr name, Attribute attr) {
    mlir::AttrTypeWalker walker;
    walker.addWalk([&](ParamDeclRefAttr attr) {
      if (forwardRenaming.contains(attr.getName()))
        return;

      auto newName = uniquer.newName();
      auto newAttr = ParamDeclRefAttr::get(newName, attr.getType());
      forwardRenaming.insert({attr.getName(), newAttr});
      inverseRenaming.insert({newAttr.getName(), attr});
    });

    walker.walk(attr);

    renaming.insert({name, doReplace(attr, forwardRenaming)});
  }

  Type doReplace(Type attrOrType, const DenseMap<StringAttr, Attribute> &map) {
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([&](ParamDeclRefAttr attr) -> Attribute {
      if (auto it = map.find(attr.getName()); it != map.end())
        return it->second;
      return attr;
    });
    return replacer.replace(attrOrType);
  }

  Attribute doReplace(Attribute attrOrType,
                      const DenseMap<StringAttr, Attribute> &map) {
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([&](ParamDeclRefAttr attr) -> Attribute {
      if (auto it = map.find(attr.getName()); it != map.end())
        return it->second;
      return attr;
    });
    return replacer.replace(attrOrType);
  }

  Type replace(Type attrOrType) {
    return doReplace(doReplace(attrOrType, renaming), inverseRenaming);
  }

  Attribute replace(Attribute attrOrType) {
    return doReplace(doReplace(attrOrType, renaming), inverseRenaming);
  }

  void replaceElementsIn(Operation *op, bool replaceAttrs, bool replaceLocs,
                         bool replaceTypes) {
    // Functor that replaces the given element if the new value is different,
    // otherwise returns nullptr.
    auto replaceIfDifferent = [&](auto element) {
      auto replacement = replace(element);
      return (replacement && replacement != element) ? replacement : nullptr;
    };

    // Update the attribute dictionary.
    if (replaceAttrs) {
      if (auto newAttrs = replaceIfDifferent(op->getAttrDictionary()))
        op->setAttrs(cast<DictionaryAttr>(newAttrs));
    }

    // If we aren't updating locations or types, we're done.
    if (!replaceTypes && !replaceLocs)
      return;

    // Update the location.
    if (replaceLocs) {
      if (Attribute newLoc = replaceIfDifferent(op->getLoc()))
        op->setLoc(cast<LocationAttr>(newLoc));
    }

    // Update the result types.
    if (replaceTypes) {
      for (OpResult result : op->getResults())
        if (Type newType = replaceIfDifferent(result.getType()))
          result.setType(newType);
    }

    // Update any nested block arguments.
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument &arg : block.getArguments()) {
          if (replaceLocs) {
            if (Attribute newLoc = replaceIfDifferent(arg.getLoc()))
              arg.setLoc(cast<LocationAttr>(newLoc));
          }

          if (replaceTypes) {
            if (Type newType = replaceIfDifferent(arg.getType()))
              arg.setType(newType);
          }
        }
      }
    }
  }

  void recursivelyReplaceElementsIn(Operation *op, bool replaceAttrs,
                                    bool replaceLocs, bool replaceTypes) {
    op->walk([&](Operation *nestedOp) {
      replaceElementsIn(nestedOp, replaceAttrs, replaceLocs, replaceTypes);
    });
  }

private:
  NameUniquer &uniquer;

  DenseMap<StringAttr, Attribute> forwardRenaming;
  DenseMap<StringAttr, Attribute> inverseRenaming;
  DenseMap<StringAttr, Attribute> renaming;
};

/// Given the input parameters to the hook function, create the array of
/// remapped parameters using the captured parameter args dictionary.
static SmallVector<TypedAttr>
getRemappedParameters(DictionaryAttr argsParamsDict,
                      ArrayRef<ParamDeclAttr> inputParams) {
  SmallVector<TypedAttr> result;
  result.reserve(inputParams.size());

  for (ParamDeclAttr paramDecl : inputParams) {
    auto paramName = paramDecl.getName();
    ASSERT_STREAM(paramName, "Parameter must have a name");

    StringRef demangledName = LIT::demangleParameterName(paramName.getValue());

    auto paramValue = argsParamsDict.get(demangledName);
    if (!paramValue)
      continue;

    ASSERT_STREAM(paramValue, "Missing parameter '"
                                  << demangledName
                                  << "' in arguments dictionary");

    auto typedValue = dyn_cast<TypedAttr>(paramValue);
    ASSERT_STREAM(typedValue, "Parameter value must be a TypedAttr");

    result.push_back(typedValue);
  }

  return result;
}

SpliceResult LambdaTemplate::splice(OpBuilder &builder, NameUniquer &uniquer,
                                    ValueRange operands, DictionaryAttr params,
                                    StringRef newLambdaName) {
  ASSERT_STREAM(templateOp != nullptr, "missing lambda");

  // Instead of referring to the `self` argument of the wrapper function
  // which contains the canonical lambda we remap onto the argument of
  // this function which invoked the enable fusion method.
  IRMapping mapper;
  mapper.map(templateOp.getArguments(), operands);

  // Copy the param region into the body.
  auto newLambda =
      cast<ParamDeclareRegionOp>(builder.clone(*canonicalLambda, mapper));

  auto remappedParams =
      getRemappedParameters(params, templateOp.getInputParams());

  // Rebind the parameters of the lambda from the `self` argument in the
  // method onto the specific parameters of the tensor being used at the
  // callsite.
  SafeRenamer safeRenamer(uniquer, templateOp.getInputParams(), remappedParams);

  safeRenamer.recursivelyReplaceElementsIn(newLambda, /*replaceAttrs=*/true,
                                           /*replaceLocs=*/false,
                                           /*replaceTypes=*/true);

  newLambda.setParamDeclAttr(
      ParamDeclAttr::get(newLambdaName, newLambda.getParamDecl().getType()));

  // Remap the name of the canonical lambda to newLambda.
  auto lambdaRef = ParamDeclRefAttr::get(newLambda.getParamDecl());
  safeRenamer.setParameterValue(this->canonicalLambda.getParamDecl().getName(),
                                lambdaRef);

  OwningOpRef<CallOp> newTensorSpecCall = extractTensorSpecCall.clone();
  safeRenamer.recursivelyReplaceElementsIn(*newTensorSpecCall,
                                           /*replaceAttrs=*/true,
                                           /*replaceLocs=*/false,
                                           /*replaceTypes=*/true);

  TypedAttr newTensorSpecAttr = newTensorSpecCall->getParamValues().back();
  return {newLambda, newTensorSpecAttr};
}

/// Given the array of parameters for an argument from
/// kKernelValueParameterAttrName, return the parameter ref which corresponds to
/// the static tensor spec parameter. This is tightly coupled on the exact
/// order of parameter to the tensor type.
static ParamDeclRefAttr getTensorSpecParamRef(DictionaryAttr argParams) {
  return cast<ParamDeclRefAttr>(
      argParams.get(MOGGPreElab::kParameterStaticSpec));
}

} // end namespace

namespace M::KGEN::MOGGPreElab {
#define GEN_PASS_DEF_SLICEKERNELS
#include "KGEN/MOGGPreElab/MOGGPreElabPasses.h.inc"
} // namespace M::KGEN::MOGGPreElab

namespace {
class SliceKernelsPass
    : public MOGGPreElab::impl::SliceKernelsBase<SliceKernelsPass> {
private:
  /// The reference input and output lambdas we should use for materializing the
  /// input/output fusion.
  LambdaTemplate inLambdaTemplate, outLambdaTemplate;

  void attachMetadataToGenerator(GeneratorOp gen,
                                 ArrayRef<std::string> inputLambdaNames,
                                 ArrayRef<std::string> outputLambdaNames) {
    // The new attributes on the generator.
    SmallVector<NamedAttribute> newAttrs;

    // Add all the old attributes.
    llvm::append_range(newAttrs, gen->getAttrs());

    Builder b{&getContext()};

    // Mark this as a sliced function so MOGG lowering can identify it.
    newAttrs.push_back(
        NamedAttribute{b.getStringAttr(SLICED_ATTR), b.getUnitAttr()});

    // Attach the lambda metadata.
    SmallVector<StringRef> inNames =
        llvm::to_vector_of<StringRef>(inputLambdaNames);
    newAttrs.push_back(NamedAttribute{b.getStringAttr(kMOGGInputLambdas),
                                      b.getStrArrayAttr(inNames)});

    SmallVector<StringRef> outNames =
        llvm::to_vector_of<StringRef>(outputLambdaNames);
    newAttrs.push_back(NamedAttribute{b.getStringAttr(kMOGGOutputLambdas),
                                      b.getStrArrayAttr(outNames)});
    gen->setAttrs(newAttrs);
  }

  /// We enable fusion for I/Os tensors that have opted in.
  /// Enabling fusion involves materializing a call to the input/output lambda
  /// within the body of the function and building a new tensor spec parameter
  /// with the lambda replaced.
  void enableFusion(GeneratorOp gen,
                    MutableArrayRef<std::string> inputLambdaNames,
                    MutableArrayRef<std::string> outputLambdaNames,
                    ArrayRef<unsigned> fusedOperands) {
    Block *computeBlock = gen.getBody();

    ArrayAttr argumentTypeNames =
        cast<ArrayAttr>(gen->getAttr(MOGG_ARG_TYPE_NAMES));

    auto argsParams =
        gen->getAttrOfType<ArrayAttr>(kKernelValueParameterAttrName);

    if (!argsParams)
      return;

    NameUniquer uniquer(gen.getContext());

    uniquer.collectReservedNames(gen);
    uniquer.collectReservedNames(inLambdaTemplate.templateOp);
    uniquer.collectReservedNames(outLambdaTemplate.templateOp);

    for (unsigned idx : fusedOperands) {
      BlockArgument tensorFusionEnabledOn = gen.getBody()->getArgument(idx);
      DictionaryAttr paramsForTensor = cast<DictionaryAttr>(argsParams[idx]);

      auto typeNameAttr =
          dyn_cast<StringAttr>(argumentTypeNames.getValue()[idx]);
      bool isVariadic =
          typeNameAttr && typeNameAttr.getValue() == MOJO_VARIADIC_TENSORS_NAME;

      // TODO(GEX-1591): Re-enable fusion support for variadic kernels
      if (isVariadic) {
        gen.emitWarning() << "Fusion requested on variadic argument " << idx
                          << " but variadic fusion is not supported";
        continue;
      }

      auto argTensorSpecRef = getTensorSpecParamRef(paramsForTensor);
      if (!argTensorSpecRef) {
        gen.emitWarning() << "Fusion requested on argument " << idx
                          << " but tensor spec parameter cannot be found";
        continue;
      }

      ASSERT_STREAM(inLambdaTemplate.templateOp && outLambdaTemplate.templateOp,
                    "intrinsic I/O fusion hooks not found");

      // Determine whether it is a input/output fusion interface.
      bool isInput = idx >= outputLambdaNames.size();
      LambdaTemplate *lambda = isInput ? &inLambdaTemplate : &outLambdaTemplate;
      std::string newLambdaName = isInput
                                      ? "input_" + std::to_string(idx) + "_fn"
                                      : "output_" + std::to_string(idx) + "_fn";
      if (isInput)
        inputLambdaNames[idx - outputLambdaNames.size()] = newLambdaName;
      else
        outputLambdaNames[idx] = newLambdaName;

      OpBuilder builder{computeBlock, computeBlock->begin()};
      SpliceResult spliceResult =
          lambda->splice(builder, uniquer, {tensorFusionEnabledOn},
                         paramsForTensor, newLambdaName);

      ASSERT_STREAM(
          argTensorSpecRef.getType() == spliceResult.newTensorSpec.getType(),
          << "invalid type of new tensor spec");

      SafeRenamer renamer(uniquer);
      renamer.setParameterValue(argTensorSpecRef.getName(),
                                spliceResult.getNewTensorSpec());

      for (Operation &op : gen.getBody()->getOperations()) {
        if (!spliceResult.contains(&op)) {
          renamer.recursivelyReplaceElementsIn(&op, /*replaceAttrs=*/true,
                                               /*replaceLocs=*/false,
                                               /*replaceTypes=*/true);
        }
      }
    }
  }

public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    auto &analysis = getAnalysis<mlir::SymbolTableAnalysis>();
    SymbolTable &symTab = analysis.getTopLevelSymbolTable();

    // Scan the generators to find the global helper functions we will need to
    // call or inspect.
    for (GeneratorOp func : mod.getOps<GeneratorOp>()) {
      if (func->hasAttr(MOGG_INTRINSIC_INPUT_FUSION_HOOK))
        inLambdaTemplate = LambdaTemplate{func};
      else if (func->hasAttr(MOGG_INTRINSIC_OUTPUT_FUSION_HOOK))
        outLambdaTemplate = LambdaTemplate{func};
    }

    for (GeneratorOp userKernel :
         llvm::make_early_inc_range(mod.getOps<GeneratorOp>())) {
      // Only look at the DPS execute functions.
      if (!userKernel->hasAttr(kMOGGExecuteFunctionLabel))
        continue;

      // Don't process again functions that we already sliced.
      if (userKernel->hasAttr(SLICED_ATTR))
        continue;

      // Dummy execute function without operands (mo.reshape).
      /// TODO: this should not be allowed for users.
      auto argumentTypeNames =
          userKernel->getAttrOfType<ArrayAttr>(MOGG_ARG_TYPE_NAMES);
      if (!argumentTypeNames)
        continue;

      // Slice out a new compute kernel. This replaces the old kernel as the
      // entry point for the thing we are going to execute.
      GeneratorOp slicedComputeFunction = userKernel.clone();
      std::string name = (Twine(userKernel.getSymName()) + "_COMPUTE").str();
      slicedComputeFunction.setSymName(name);

      // Figure out the number of tensor inputs for the kernel.
      // The first operand is always the output.
      // And there might be non-tensor arguments (eg MojoCallContext).
      unsigned kernelInputsCount = 0;
      /// TODO: GEX-1046: We should have markers in Mojo for what is an input
      /// and what is an output (ex: mo.top_k).
      /// kMOGGNumDPSOutputs might not be accurate, but should be better than 1
      /// and it works on allreduce.
      unsigned kernelOutputsCount = 1;
      auto numDpsOut = userKernel->getAttr(MOGGPreElab::kMOGGNumDPSOutputs);
      if (numDpsOut)
        kernelOutputsCount = cast<IntegerAttr>(numDpsOut).getInt();

      for (size_t i = kernelOutputsCount,
                  e = argumentTypeNames.getValue().size();
           i < e; i++) {
        auto nameAttr = dyn_cast<StringAttr>(argumentTypeNames.getValue()[i]);
        if (nameAttr &&
            (nameAttr.getValue() == MOJO_INTERNAL_DPS_TENSOR_TYPE_NAME ||
             nameAttr.getValue() == MOJO_VARIADIC_TENSORS_NAME)) {
          ++kernelInputsCount;
        }
      }

      // Figure out which kernel input / outputs are fused.
      SmallVector<unsigned> fusedOperands;
      if (userKernel->hasAttr(kMOGGElementFunction)) {
        // For elementwise kernel, fuse everything.
        for (unsigned i = 0, e = kernelInputsCount + kernelOutputsCount; i < e;
             ++i)
          fusedOperands.push_back(i);
      } else if (auto fusedOperandsAttr = dyn_cast_or_null<ArrayAttr>(
                     userKernel->getAttr(kMOGGFusableArgs))) {
        for (Attribute attr : fusedOperandsAttr.getValue())
          fusedOperands.push_back(
              cast<IntegerAttr>(attr).getValue().getZExtValue());
      }

      // Resize the input and output lambda metadata to encapsulate all inputs
      // and outputs. Each one is marked off. As we detect in / out lambdas we
      // will populate these.
      SmallVector<std::string> inputLambdaNames, outputLambdaNames;
      inputLambdaNames.resize(kernelInputsCount, "");
      outputLambdaNames.resize(kernelOutputsCount, "");

      enableFusion(slicedComputeFunction, inputLambdaNames, outputLambdaNames,
                   fusedOperands);

      // Strip all debug info. Its too annoying to maintain and there is no
      // way to actually debug the sliced kernel directly. Users would debug
      // the base kernel.
      DebugInfo::stripDebugInfo(slicedComputeFunction,
                                /*preserveLineTables=*/true);

      // Add compute function part to the module, i.e the kernel sans
      // allocation.
      symTab.insert(slicedComputeFunction);

      // Add info for mogg to read off the kernel.
      attachMetadataToGenerator(slicedComputeFunction, inputLambdaNames,
                                outputLambdaNames);

      // Remove attributes that would cause MojoLibraryAnalysis to treat the
      // sliced function as a registered kernel.
      userKernel->removeAttr(kernelRegistrationAttr);
      userKernel->removeAttr(kMOGGExecuteFunctionLabel);
      userKernel->removeAttr(shapeFuncRegistrationAttr);
      userKernel->removeAttr(kMOGGShapeFunctionLabel);
      userKernel->removeAttr(kMOGGPyTorchFallbackFunctionLabel);
    }
  }
};
} // namespace
