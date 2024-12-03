//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/MOGGPreElab/MOGGDecorators.h"
#include "KGEN/MOGGPreElab/Passes.h"
#include "KGEN/TransformUtils/CallGraphUtils.h"
#include "Support/AssertStream.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "mogg-autospecialize"

using namespace M;
using namespace KGEN;
using namespace MOGGPreElab;

namespace M::KGEN::MOGGPreElab {
#define GEN_PASS_DEF_MOGGAUTOSPECIALIZE
#include "KGEN/MOGGPreElab/MOGGPreElabPasses.h.inc"
} // namespace M::KGEN::MOGGPreElab

static constexpr llvm::StringLiteral SPEC_PREFIX_STR = "__MOGG_SPEC";
static constexpr llvm::StringLiteral TENSOR_SPEC_NONE = "TENSOR_SPEC_NONE";

static bool isTensorType(Attribute typeName) {
  // For variadic arguments, the registered type will be a unit attr (e.g.
  // empty attr)
  if (!isa<StringAttr>(typeName))
    return false;
  auto typeNameStr = cast<StringAttr>(typeName).strref();
  return typeNameStr == MOJO_DPS_TENSOR_TYPE_NAME ||
         typeNameStr == MOJO_INTERNAL_DPS_TENSOR_TYPE_NAME;
}

namespace {
/// Information pertaining to the the tensor spec as represented in KGEN.
struct TensorSpecKGEN {
  void pullMetadataFromFunc(GeneratorOp);

  Type getType() { return noneSpecExample.getParamDecl().getType(); }

  operator bool() { return noneSpecExample != nullptr; }

  TypedAttr getValue() { return noneSpecExample.getValue(); }

  /// The intrinsic function which retrieves the tensor spec.
  GeneratorOp getterFunc;

  /// The KGEN parameter for a none spec which contains the expression we can
  /// use to instantiate a none spec.
  ParamDeclareOp noneSpecExample;
};

void TensorSpecKGEN::pullMetadataFromFunc(GeneratorOp gen) {
  getterFunc = gen;
  for (ParamDeclareOp decl : gen.getOps<ParamDeclareOp>()) {
    if (decl.getParamDecl().getName().strref().starts_with(TENSOR_SPEC_NONE)) {
      noneSpecExample = decl;
      break;
    }
  }
}

struct CallGraphNode
    : public CallGraphNodeBase<CallGraphNode, GeneratorOp, CallOp> {
  using CallGraphNodeBase::CallGraphNodeBase;

  /// 'true' at any index where the argument requires a spec.
  llvm::BitVector argsNeedingSpec;

  /// The spec parameter is accessed via an intrinsic getter function in the
  /// parameter domain. This tracks calls to those getter functions in a given
  /// function.
  SmallVector<ParamOperatorAttr> getterFunctions;

  /// If this node has been specialized then we track it here.
  GeneratorOp specialization = nullptr;

  /// If we've already evaluated this node
  bool hasBeenProcessed = false;
};
} // namespace

/// Find the intrinsic functions which are used to access the spec.
static void identifyGetterFunctions(CallGraphNode *node, SymbolTable &symTab) {
  mlir::AttrTypeWalker walker;
  GeneratorOp gen = node->func;
  ArrayAttr argNames = gen->getAttrOfType<ArrayAttr>(MOGG_ARG_SRC_NAMES);
  llvm::BitVector &argsToSpec = node->argsNeedingSpec;

  // Walk the attribute operators to identify an operator refering to the
  // 'getSpec' function. This function is used to materialize the parameter
  // static spec info for a given tensor within a function.
  walker.addWalk([&](ParamOperatorAttr attr) {
    // Our intrinsic should always be Apply(intrinicFunc, "name_of_tensor")
    if (attr.getOpcode() != KGEN::POC::Apply || attr.getOperands().size() != 2)
      return;
    auto sym = dyn_cast<SymbolConstantAttr>(attr.getOperands()[0]);
    if (!sym)
      return;

    // Pull the name of the function being targeted off of the parameter
    // expression.
    auto asStr = cast<FlatSymbolRefAttr>(sym.getSymbol()).getValue();
    auto invokedFunc =
        llvm::dyn_cast_if_present<GeneratorOp>(symTab.lookup(asStr));
    auto name = dyn_cast<StringAttr>(attr.getOperands()[1]);
    if (!invokedFunc || !name)
      return;

    // Check if the targetted function is the known intrinsic function and
    // attach that as metadata for the function if so.
    if (invokedFunc->hasAttr(MOGGPreElab::MOGG_INTRINSIC_TENSOR_SPEC_HOOK) ||
        invokedFunc->hasAttr(
            MOGGPreElab::MOGG_INTRINSIC_TENSOR_SPEC_TUPLE_HOOK)) {
      node->getterFunctions.push_back(attr);

      // Identify which tensor is getting the spec.
      StringAttr tensorName = cast<StringAttr>(attr.getOperand(1));
      for (auto [i, argName] : llvm::enumerate(argNames)) {
        if (cast<StringAttr>(argName).strref() == tensorName.strref())
          argsToSpec[i] = true;
      }
    }
  });

  // Walk all of the generator.
  gen.walk([&](Operation *op) {
    for (const NamedAttribute &attr : op->getAttrs())
      walker.walk(attr.getValue());
    for (Type type : op->getOperandTypes())
      walker.walk(type);
    for (Region &region : op->getRegions()) {
      for (Type type : region.getArgumentTypes())
        walker.walk(type);
    }
  });
}

namespace {
struct CallGraph : public CallGraphBase<CallGraph, CallGraphNode> {
  explicit CallGraph(const SymbolTable &symtab) : symtab(symtab) {}

  bool shouldAddToGraph(KGENCallOpInterface call, CallGraphNode *node) {
    // If the function doesn't take any tensors we can eagerly exclude it from
    // the analysis.
    llvm::sys::SmartScopedWriter<true> lock(node->mutex);
    if (!node->func->hasAttr(MOGG_ARG_SRC_NAMES)) {
      node->hasBeenProcessed = true;
    } else {
      node->argsNeedingSpec =
          llvm::BitVector(node->func.getNumArguments(), false);
    }
    return true;
  }

  const SymbolTable &symtab;
  llvm::sys::SmartRWMutex<true> mutex;
};

using CallGraphImpl = CallGraph;

} // namespace

namespace {

class Specializer {
public:
  Specializer(ArrayRef<StringAttr> argNames, llvm::BitVector &argsToSpec)
      : argsToSpec{argsToSpec} {

    // Reset the argsToSpec and prepare for optional parameters.
    // We'll only add parameters if needed (specsof or call to another kernel).
    for (auto index : llvm::seq(argNames.size())) {
      StringRef argName = argNames[index].strref();
      argNameToSpecParam[argName] = {nullptr, index};
      argsToSpec[index] = false;
    }
  }

  /// Get parameter info for the given parameter name.
  std::pair<ParamDeclAttr, size_t> *getParamInfo(StringRef paramName) {
    auto itr = argNameToSpecParam.find(paramName);
    if (itr == argNameToSpecParam.end())
      return nullptr;
    return &itr->second;
  }

  /// Find uses of the specsof operation and replace that with a reference
  /// to the parameter declaration for the StaticTensorSpec struct.
  ErrorOrSuccess
  replaceSpecsOfCalls(GeneratorOp func,
                      ArrayRef<ParamOperatorAttr> getterFunctions) {
    ErrorOrSuccess result = success();

    // Apply the replacement on every op in the function.
    mlir::AttrTypeReplacer walker;
    walker.addReplacement(
        [&](ParamOperatorAttr attr) -> std::optional<TypedAttr> {
          // Is this one of the specsof getter functions?
          if (!llvm::is_contained(getterFunctions, attr))
            return std::nullopt;

          StringAttr tensorName = cast<StringAttr>(attr.getOperand(1));
          std::pair<ParamDeclAttr, size_t> *paramInfo =
              getParamInfo(tensorName.strref());

          if (!paramInfo) {
            result =
                Error(Twine("Unable to resolve specsof for variable named '") +
                      tensorName.strref() + "'");
            return nullptr;
          }

          if (paramInfo->first == nullptr) {
            // First time we encounter a getter function for this argument.
            // Define and add a specs parameter to the function.
            size_t argIdx = paramInfo->second;
            std::string paramName =
                (SPEC_PREFIX_STR + std::to_string(argIdx)).str();
            paramInfo->first = ParamDeclAttr::get(paramName, attr.getType());
            argsToSpec[argIdx] = true;

            LLVM_DEBUG(llvm::dbgs() << "Add param (specsof) for "
                                    << tensorName.getValue() << " (input #"
                                    << argIdx << ").\n";);
          }

          return ParamDeclRefAttr::get(paramInfo->first);
        });

    // Apply the replacement on every op in the function.
    walker.recursivelyReplaceElementsIn(func, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/true,
                                        /*replaceTypes=*/true);

    return result;
  }

private:
  /// Mapping of each argument name to the its index in the argument list
  /// and the parameter declaration holding the tensor spec corresponding
  /// to that argument (if any).
  llvm::StringMap<std::pair<ParamDeclAttr, size_t>> argNameToSpecParam;

  /// Bitmask determining which arguments need a tensor spec.
  llvm::BitVector &argsToSpec;
};

} // namespace

/// Create a specialzied version of the function now accepting a tensor spec
/// parameter.
static FailureOr<OwningOpRef<GeneratorOp>>
specializeOnSpec(CallGraphNode *node, TensorSpecKGEN &specTemplate,
                 const CallGraph &cg) {
  GeneratorOp gen = node->func;
  SmallVector<StringAttr> argNames =
      llvm::to_vector(gen->getAttrOfType<ArrayAttr>(MOGG_ARG_SRC_NAMES)
                          .getAsRange<StringAttr>());

  ArrayAttr argParams = gen->getAttrOfType<ArrayAttr>(MOGG_ARG_PARAMS);
  llvm::BitVector &argsToSpec = node->argsNeedingSpec;

  LLVM_DEBUG(llvm::dbgs() << "BEGIN specializeOnSpec for " << gen.getSymName()
                          << "\n");

  OwningOpRef<GeneratorOp> cloned = gen.clone();
  Specializer specializer(argNames, argsToSpec);

  auto result = specializer.replaceSpecsOfCalls(*cloned, node->getterFunctions);
  if (result.isError())
    return gen.emitError() << result.takeError().get();

  // The Spec type from the callee refers to parameter of the callee.
  // Replace it to use parameters of the caller.
  auto fixupCalleeSpecType = [&argParams](Type specType, GeneratorOp callee,
                                          size_t calleeArgIdx,
                                          size_t callerArgIdx) -> Type {
    ArrayAttr callerArgParams =
        cast<ArrayAttr>(argParams.getValue()[callerArgIdx]);
    ArrayAttr calleeParams = callee->getAttrOfType<ArrayAttr>(MOGG_ARG_PARAMS);
    ArrayAttr calleeArgParams =
        cast<ArrayAttr>(calleeParams.getValue()[calleeArgIdx]);

    struct Substitutor : IndexParameterReplacer<Substitutor> {
      Type tryReplace(Type, size_t) { return nullptr; }
      Attribute tryReplace(Attribute attr, size_t depth) {

        // Look for a parameter reference.
        StringRef paramName;

        if (auto paramRef = dyn_cast<ParamDeclRefAttr>(attr))
          paramName = paramRef.getName();

        if (auto indexRef = dyn_cast<ParamIndexRefAttr>(attr)) {
          // For index, manually find the reference.
          if (indexRef.getDepth() != depth || indexRef.getIsResult())
            return nullptr;

          paramName = callee.getInputParams()[indexRef.getIndex()].getName();
        }

        if (paramName.empty())
          return nullptr;

        // Look at the callee parameters to see if we can find it.
        auto it =
            llvm::find_if(calleeArgParams, [paramName](Attribute paramAttr) {
              return cast<ParamDeclRefAttr>(paramAttr).getName() == paramName;
            });
        ASSERT_STREAM(it != calleeArgParams.end(),
                      << "failed to specialize spec type: unknown parameter "
                      << paramName << " for " << attr);

        // Since the list of parameter match, we can get the new parameter from
        // the caller list.
        auto replacedParam =
            callerArgParams[std::distance(calleeArgParams.begin(), it)];

        LLVM_DEBUG(llvm::dbgs()
                   << "specType: replaced " << attr << " (" << paramName
                   << ") with " << replacedParam << "\n");
        return replacedParam;
      }

      GeneratorOp callee;
      ArrayAttr callerArgParams;
      ArrayAttr calleeArgParams;
    } substitutor;
    substitutor.callee = callee;
    substitutor.callerArgParams = callerArgParams;
    substitutor.calleeArgParams = calleeArgParams;
    return substitutor.replace(specType);
  };

  const SymbolTable &symTab = cg.symtab;

  // Find calls to detect other params
  cloned->walk([&](CallOp oldCall) {
    // Identify the call
    auto calledSym = oldCall.getCalleeSymbol().getValue();
    auto calledFunc = dyn_cast_or_null<GeneratorOp>(symTab.lookup(calledSym));
    // Skip any non generators.
    if (!calledFunc)
      return;

    // We now have unreachable calls because we are dealing with the cloned
    // calls.
    const CallGraphNode *calledFuncNode = cg.lookup(calledFunc);
    if (!calledFuncNode)
      return;

    // Specialize the node.
    if (!calledFuncNode->specialization)
      return;

    GeneratorOp specialized = calledFuncNode->specialization;

    size_t calleeParamIdx = oldCall.getCallee().getParamValues().size();

    for (auto [calleeIdx, val] : llvm::enumerate(oldCall->getOperands())) {
      if (!calledFuncNode->argsNeedingSpec[calleeIdx])
        continue;
      ++calleeParamIdx;

      auto blockArg = dyn_cast<BlockArgument>(val);
      if (!blockArg)
        continue;
      auto callerIdx = blockArg.getArgNumber();

      StringAttr tensorName = argNames[callerIdx];
      auto paramInfo = specializer.getParamInfo(tensorName);
      if (!paramInfo)
        continue;

      if (paramInfo->first == nullptr) {
        // First time we encounter this arguments.
        // Define and add a specs parameter to the function.
        std::string paramName =
            (SPEC_PREFIX_STR + std::to_string(callerIdx)).str();

        Type calleeParamTy =
            specialized.getSignature().getInputParamTypes()[calleeParamIdx - 1];
        paramInfo->first = ParamDeclAttr::get(
            paramName, fixupCalleeSpecType(calleeParamTy, specialized,
                                           calleeIdx, callerIdx));
        argsToSpec[callerIdx] = true;

        LLVM_DEBUG(llvm::dbgs()
                   << "Add param (call) for " << tensorName << " (#"
                   << callerIdx << "): callee input #" << calleeIdx
                   << " (param #" << (calleeParamIdx - 1) << ") "
                   << "@" << oldCall.getCalleeSymbol() << ".\n");
      }
    }
  });

  // Get the existing params, these won't change.
  SmallVector<ParamDeclAttr> params(cloned->getInputParams());

  // Let's add the parameters now.
  // We need to add them at the end to ensure they follow the arguments order.
  SmallVector<Attribute> tensorSpecParamNames;
  LLVM_DEBUG(llvm::dbgs() << "Original parameters count: " << params.size()
                          << "\n");
  for (size_t argIdx = 0, e = argsToSpec.size(); argIdx < e; ++argIdx) {
    if (argsToSpec[argIdx]) {
      std::pair<ParamDeclAttr, size_t> *paramInfos =
          specializer.getParamInfo(cast<StringAttr>(argNames[argIdx]).strref());
      ASSERT_STREAM(paramInfos && paramInfos->first != nullptr,
                    << "parameter undefined");
      params.push_back(paramInfos->first);
      tensorSpecParamNames.push_back(paramInfos->first);
    } else {
      tensorSpecParamNames.push_back(mlir::UnitAttr::get(gen->getContext()));
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "Final parameters count: " << params.size()
                          << "\n");

  // Update the signature to add the new tensor types.
  SignatureType oldSig = cloned->getSignature();
  FnEffects effects = oldSig.getFnEffects();
  effects.setCapturing(true);
  cloned->setSignature(SignatureType::remapToSignature(
      params, {}, cloned->getFunctionType(),
      /*argConventions=*/oldSig.getArgConventions(),
      /*fnEffects=*/effects,
      /*metadata=*/oldSig.getMetadata()));

  // Remove the old params from the function.
  cloned->setInputParams(params);
  cloned->getOperation()->setAttr(
      kKernelTensorSpecParameterAttrName,
      ArrayAttr::get(gen->getContext(), tensorSpecParamNames));

  node->hasBeenProcessed = true;
  node->specialization = *cloned;

  // Remove the attribute that identifies the operation name.
  // It ensures MojoLibraryAnalysis will chose the specialized kernel (and not
  // the original one).
  gen->removeAttr(MOGGPreElab::kMOGGExecuteFunctionLabel);
  gen->removeAttr(MOGGPreElab::kMOGGShapeFunctionLabel);

  LLVM_DEBUG(llvm::dbgs() << "END specializeOnSpec for " << gen.getSymName()
                          << "\n\n");

  return cloned;
}

namespace {
class MOGGAutospecializePass
    : public M::KGEN::MOGGPreElab::impl::MOGGAutospecializeBase<
          MOGGAutospecializePass> {
public:
  static TensorSpecKGEN getTensorSpecTemplate(ModuleOp mod) {
    TensorSpecKGEN specTemplate;
    for (GeneratorOp gen : mod.getOps<GeneratorOp>()) {
      if (gen->hasAttr(MOGG_INTRINSIC_TENSOR_SPEC_HOOK)) {
        specTemplate.pullMetadataFromFunc(gen);
        break;
      }
    }

    return specTemplate;
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = mod.getContext();
    OpBuilder builder{ctx};

    auto &analysis = getAnalysis<mlir::SymbolTableAnalysis>();
    SymbolTable &symTab = analysis.getTopLevelSymbolTable();

    TensorSpecKGEN specTemplate = getTensorSpecTemplate(mod);

    // Pass relies on this, if we could find the function it should be there.
    // It's not a hard fail for users as this is sensitive to IR changes.
    if (!specTemplate) {
      if (specTemplate.getterFunc) {
        mod.emitError() << "Sample none spec parameter could not be "
                           "found in spec get function.";
        signalPassFailure();
      }
      return;
    }

    // Build a callgraph of all calls so we have something to traverse
    // through.
    CallGraph cg{symTab};
    cg.build(mod, symTab);

    SmallVector<CallGraphNode *> worklist;
    // Collect all kernels which need to be autospecialized with tensor spec.
    for (GeneratorOp gen : mod.getOps<GeneratorOp>()) {
      // If this is not an DPS kernel skip.
      if (!isDPSKernel(gen))
        continue;
      CallGraphNode *node = cg.lookup(gen);

      // Kernels always need a spec for all of their tensor arguments.
      node->argsNeedingSpec =
          llvm::BitVector(node->func.getNumArguments(), false);
      ArrayAttr argTypeNames =
          gen->getAttrOfType<ArrayAttr>(MOGG_ARG_TYPE_NAMES);
      if (!argTypeNames)
        continue;

      for (auto [idx, type] : llvm::enumerate(argTypeNames)) {
        if (isTensorType(type))
          node->argsNeedingSpec[idx] = true;
      }

      // Add it to the worklist to start processing.
      worklist.push_back(node);
    }

    // New parameters to put on the call, reuse same allocation each time to
    // limit memory allocs.
    SmallVector<TypedAttr> newParamBuffer;

    // The set of all tensors which need to have a spec added.
    llvm::SetVector<Value> tensorsNeedingSpecs;

    // Mapping between a tensor and the spec for that tensor.
    DenseMap<Value, ParamDeclRefAttr> tensorsToSpecs;

    while (!worklist.empty()) {
      tensorsNeedingSpecs.clear();
      tensorsToSpecs.clear();
      CallGraphNode *node = worklist.back();

      if (node->hasBeenProcessed) {
        worklist.pop_back();
        continue;
      }
      GeneratorOp gen = node->func;
      bool anyPendingDeps = false;

      for (CallGraphNode::EdgeT edge : node->callsites) {
        auto calledSym =
            cast<FlatSymbolRefAttr>(edge.call.getCalleeSymbol()).getValue();
        auto g2 = cast<GeneratorOp>(symTab.lookup(calledSym));

        CallGraphNode *calleeNode = cg.lookup(g2);
        if (calleeNode->hasBeenProcessed) {
          // If the caller has tensor arguments needing specs add those
          // tensors to the set of tensor being tracked.
          llvm::BitVector &argsToSpec = calleeNode->argsNeedingSpec;
          for (size_t argIdx = 0, e = argsToSpec.size(); argIdx < e; ++argIdx) {
            if (argsToSpec[argIdx])
              tensorsNeedingSpecs.insert(edge.call.getOperand(argIdx));
          }

          // Skip since this has already been processed.
          continue;
        }

        // Otherwise add to the worklist.
        worklist.push_back(calleeNode);
        anyPendingDeps = true;
      }

      // If there are no dependencies waiting then we can pop, otherwise wait
      // for them to be resolved.
      if (anyPendingDeps)
        continue;

      // All the children have been scheduled so we are ready to specialize
      // this one.
      worklist.pop_back();

      // Add the needing spec bit mask if this node doesn't already have it.
      if (node->argsNeedingSpec.empty())
        node->argsNeedingSpec = llvm::BitVector(gen.getNumArguments(), false);

      ArrayAttr argTypeNames =
          gen->getAttrOfType<ArrayAttr>(MOGG_ARG_TYPE_NAMES);

      for (auto [idx, type] : llvm::enumerate(argTypeNames)) {
        if (!isTensorType(type))
          continue;

        Value arg = gen.getArgument(idx);

        // If this tensor needs a spec track the argument index so the
        // specializer will add it.
        if (tensorsNeedingSpecs.contains(arg))
          node->argsNeedingSpec[idx] = true;
      }

      // Check to see if there's any use of the special spec replacing
      // intrinsic within this function.
      identifyGetterFunctions(node, symTab);

      // We can now safely confirm that this node has no tensors needing a
      // spec.
      if (node->argsNeedingSpec.none()) {
        node->hasBeenProcessed = true;
        continue;
      }

      // Specialize the node on the argument specs.
      FailureOr<OwningOpRef<GeneratorOp>> maybeNewGen =
          specializeOnSpec(node, specTemplate, cg);
      if (failed(maybeNewGen)) {
        signalPassFailure();
        return;
      }

      GeneratorOp newGen = maybeNewGen->release();
      symTab.insert(newGen);

      // Each new param is added after the existing params.
      size_t numAddedParams = 0;
      size_t numOldParams = gen.getInputParams().size();
      for (auto [idx, arg] : llvm::enumerate(newGen.getArguments())) {
        if (node->argsNeedingSpec[idx]) {
          tensorsToSpecs[arg] = ParamDeclRefAttr::get(
              newGen.getInputParams()[numOldParams + numAddedParams]);
          ++numAddedParams;
        }
      }

      newGen.walk([&](CallOp oldCall) {
        // Identify the call
        auto calledSym =
            cast<FlatSymbolRefAttr>(oldCall.getCalleeSymbol()).getValue();
        auto calledFunc =
            dyn_cast_or_null<GeneratorOp>(symTab.lookup(calledSym));
        // Skip any non generators.
        if (!calledFunc)
          return;

        // We now have unreachable calls because we are dealing with the
        // cloned calls.
        CallGraphNode *node = cg.lookup(calledFunc);
        if (!node)
          return;

        // Specialize the node.
        CallGraphNode *calledFuncNode = cg.lookup(calledFunc);
        if (calledFuncNode->specialization) {
          // If there's a specialized version then we should call it with the
          // new specs.
          GeneratorOp specialized = calledFuncNode->specialization;

          // Add the old unchanged parameters.
          newParamBuffer.clear();
          for (TypedAttr inParam : oldCall.getCallee().getParamValues())
            newParamBuffer.push_back(inParam);

          // If we need a parameter and we have that parameter add it,
          // otherwise add the none spec.
          for (auto [i, val] : llvm::enumerate(oldCall->getOperands())) {
            if (calledFuncNode->argsNeedingSpec[i]) {
              auto itr = tensorsToSpecs.find(val);
              LLVM_DEBUG(llvm::dbgs() << "Fixup call: add parameter for input #"
                                      << i << " " << val << " (@"
                                      << oldCall.getCalleeSymbol() << ").\n");
              if (itr == tensorsToSpecs.end())
                newParamBuffer.push_back(specTemplate.getValue());
              else
                newParamBuffer.push_back(itr->second);
            }
          }

          auto flatSym = FlatSymbolRefAttr::get(ctx, specialized.getSymName());
          auto specializedSig =
              specialized.getSignature().getSpecializedSignature(
                  newParamBuffer, oldCall.getLoc());
          auto symbol =
              SymbolConstantAttr::get(flatSym, specializedSig, newParamBuffer);

          // Call it.
          builder.setInsertionPoint(oldCall);
          auto newCall = builder.create<CallOp>(oldCall.getLoc(), symbol,
                                                oldCall->getOperands());

          // Delete the old one.
          oldCall.replaceAllUsesWith(newCall);
          oldCall.erase();
        }
      });
    }
  }
};

} // namespace
