//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/MOGGPreElab/MOGGDecorators.h"
#include "KGEN/MOGGPreElab/Passes.h"
#include "KGEN/TransformUtils/CallGraphUtils.h"
#include "Support/AssertStream.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallBitVector.h"

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
  auto typeNameStr = dyn_cast<StringAttr>(typeName);
  if (!typeNameStr)
    return false;
  return typeNameStr == MOJO_DPS_TENSOR_TYPE_NAME ||
         typeNameStr == MOJO_INTERNAL_DPS_TENSOR_TYPE_NAME;
}

/// Tensor specs are assigned a name based on their argument index.
static std::string getTensorSpecParamName(unsigned argIndex) {
  return (SPEC_PREFIX_STR + std::to_string(argIndex)).str();
}

namespace {

struct CallGraphNode
    : public CallGraphNodeBase<CallGraphNode, GeneratorOp, CallOp> {
  CallGraphNode(GeneratorOp func)
      : CallGraphNodeBase(func),
        argsNeedingSpec{func.getNumArguments(), false} {
    argTypeNames = func->getAttrOfType<ArrayAttr>(MOGG_ARG_TYPE_NAMES);

    if (!argTypeNames || llvm::none_of(argTypeNames, isTensorType))
      return;

    auto argSrcNames = func->getAttrOfType<ArrayAttr>(MOGG_ARG_SRC_NAMES)
                           .getAsRange<StringAttr>();

    argParams = func->getAttrOfType<ArrayAttr>(MOGG_ARG_PARAMS);

    for (auto [index, argName] : llvm::enumerate(argSrcNames)) {
      argNameToIndex[argName] = index;
      paramInfos.push_back(nullptr);
    }
  }

  CallGraphNode(CallGraphNode &&) = default;

  /// Does the callgraph node have any tensor arguments?
  bool hasTensorArgs() const {
    return argTypeNames && llvm::any_of(argTypeNames, isTensorType);
  }

  /// Query if the function for the node has an argument with the given name.
  bool hasArgumentOfName(StringRef tensorName) const {
    return argNameToIndex.contains(tensorName);
  }

  /// Update the ParamInfo of the corresponding tensor if it has not already
  /// been set.
  void setParamInfoIfNeeded(unsigned argIndex, Type paramType) {
    // Already set.
    if (argsNeedingSpec.test(argIndex))
      return;

    ParamDeclAttr paramInfo = paramInfos[argIndex];
    ASSERT_STREAM(!paramInfo,
                  << "bitvector out of sync with parameter information");

    std::string paramName = getTensorSpecParamName(argIndex);
    paramInfos[argIndex] = ParamDeclAttr::get(paramName, paramType);
    argsNeedingSpec.set(argIndex);
  }

  /// Update the ParamInfo of the corresponding tensor if it has not already
  /// been set.
  void setParamInfoIfNeeded(StringRef tensorName, Type paramType) {
    auto argIndex = argNameToIndex.at(tensorName);
    return setParamInfoIfNeeded(argIndex, paramType);
  }

  ParamDeclAttr getParamDecl(StringRef tensorName) {
    auto itr = argNameToIndex.find(tensorName);
    ASSERT_STREAM(itr != argNameToIndex.end(), << "unknown argument name");
    return getParamDecl(itr->second);
  }

  ParamDeclAttr getParamDecl(unsigned argIndex) {
    ASSERT_STREAM(argIndex < paramInfos.size(), << "argument out of bounds");
    return paramInfos[argIndex];
  }

  /// If this node has been specialized then we track it here.
  GeneratorOp specialization{nullptr};

  /// The spec parameter is accessed via an intrinsic getter function in the
  /// parameter domain. This tracks calls to those getter functions in a given
  /// function.
  SmallVector<ParamOperatorAttr> getterFunctions{};

  /// The names of the argument types for each argument.
  ArrayAttr argTypeNames{nullptr};

  /// Array of parameters for the generator being specialized.
  ArrayAttr argParams{nullptr};

  /// 'true' at any index where the argument requires a spec.
  llvm::SmallBitVector argsNeedingSpec;

private:
  /// Stores the spec parameter for arguments which require a spec.
  llvm::SmallVector<ParamDeclAttr> paramInfos;

  /// Mapping of each argument name to the its index in the argument list.
  /// The StringRef keys are safe to use as all the come from StringAttrs.
  llvm::DenseMap<StringRef, unsigned> argNameToIndex;
};

struct CallGraph : public CallGraphBase<CallGraph, CallGraphNode> {
  explicit CallGraph(SymbolTable &symtab) : symtab(symtab) {}

  /// No point analyzing nodes which do not have tensor arguments.
  bool shouldAddToGraph(CallOp call, CallGraphNode *node) {
    return node->hasTensorArgs();
  }

  SymbolTable &symtab;
};
} // namespace

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

} // namespace

/// Find the intrinsic functions which are used to access the spec.
/// This function is responsible for validating the uses of compiler.specsof
/// as well and will produce an error when the given node contains a specsof
/// call which does not correspond to any tensor variable.
static ErrorOrSuccess identifyGetterFunctions(CallGraphNode *node,
                                              const SymbolTable &symTab) {
  ErrorOrSuccess result = success();

  mlir::AttrTypeWalker walker;

  // Walk the attribute operators to identify an operator refering to the
  // 'getSpec' function. This function is used to materialize the parameter
  // static spec info for a given tensor within a function.
  walker.addWalk([&](ParamOperatorAttr attr) -> WalkResult {
    // Our intrinsic should always be Apply(intrinsicFunc, "name_of_tensor")
    if (attr.getOpcode() != KGEN::POC::Apply || attr.getOperands().size() != 2)
      return WalkResult::advance();

    auto sym = dyn_cast<SymbolConstantAttr>(attr.getOperands()[0]);
    if (!sym)
      return WalkResult::advance();

    // Pull the name of the function being targeted off of the parameter
    // expression.
    auto asStr = cast<FlatSymbolRefAttr>(sym.getSymbol()).getValue();
    auto invokedFunc = symTab.lookup<GeneratorOp>(asStr);
    auto name = dyn_cast<StringAttr>(attr.getOperands()[1]);
    if (!invokedFunc || !name)
      return WalkResult::advance();

    // Check if the targetted function is the known intrinsic function and
    // attach that as metadata for the function if so.
    if (invokedFunc->hasAttr(MOGGPreElab::MOGG_INTRINSIC_TENSOR_SPEC_HOOK) ||
        invokedFunc->hasAttr(
            MOGGPreElab::MOGG_INTRINSIC_TENSOR_SPEC_TUPLE_HOOK)) {
      if (!llvm::is_contained(node->getterFunctions, attr))
        node->getterFunctions.push_back(attr);

      // Identify which tensor is getting the spec.
      StringAttr tensorName = cast<StringAttr>(attr.getOperand(1));

      if (!node->hasArgumentOfName(tensorName)) {
        result = Error(Twine("Unable to resolve specsof for variable named '") +
                       tensorName.strref() + "'");
        return WalkResult::interrupt();
      }

      node->setParamInfoIfNeeded(tensorName, attr.getType());
    }

    return WalkResult::advance();
  });

  // Walk all of the generator.
  node->func.walk([&](Operation *op) {
    if (walker.walk(op->getAttrDictionary()).wasInterrupted())
      return WalkResult::interrupt();

    for (Type type : op->getOperandTypes())
      if (walker.walk(type).wasInterrupted())
        return WalkResult::interrupt();

    for (Type type : op->getResultTypes())
      if (walker.walk(type).wasInterrupted())
        return WalkResult::interrupt();

    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument &arg : block.getArguments()) {
          if (walker.walk(arg.getType()).wasInterrupted())
            return WalkResult::interrupt();
        }
      }
    }

    return WalkResult::advance();
  });

  return result;
}

namespace {

/// Fix the spec type for the callee.
///
/// When specializing a callsite, the given `specType` contains parameter
/// references to parameters bound in the callee.
/// Rebind those references to refer to parameters bound in the caller.
Type fixupCalleeSpecType(ArrayAttr argParams, Type specType, GeneratorOp callee,
                         size_t calleeArgIdx, size_t callerArgIdx) {
  ArrayAttr callerArgParams = cast<ArrayAttr>(argParams[callerArgIdx]);
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
}

} // namespace

/// Given the program's callgraph, initialize each node's state to the default
/// values before performing the analysis and return the initial worklist.
///
/// The state initialization involves initialization the bitsets to the
/// the appropriate size and identifying all uses of the getter functions
/// (compiler.specsof calls), which will never change throughout the analysis.
static FailureOr<SmallVector<CallGraphNode *>>
initializeAnalysis(CallGraph &cg) {
  SmallVector<CallGraphNode *> worklist;
  worklist.reserve(cg.nodes.size());

  // Add all nodes in the callgraph to the worklist which have tensor types.
  for (auto &[func, node] : cg.nodes) {
    if (!node.hasTensorArgs())
      continue;

    auto maybeError = identifyGetterFunctions(&node, cg.symtab);
    if (maybeError.isError())
      return func.emitError() << maybeError.takeError().get();

    worklist.push_back(&node);
  }

  return worklist;
}

/// Analyze a single callsite with the `current` node.
/// This function looks for parameters which were added to the callee since
/// the last time it was analyzed. When a new parameter is found, that
/// parameter is associated with a parameter to the `current` generator by
/// associating it to one of the block arguments.
static void analyzeCallsite(CallGraphNode *current,
                            const CallGraph::EdgeT &edge) {
  KGEN::CallOp callsite = edge.call;
  CallGraphNode *calleeNode = edge.node;

  for (auto calleeIdx : calleeNode->argsNeedingSpec.set_bits()) {
    auto blockArg = dyn_cast<BlockArgument>(callsite.getOperand(calleeIdx));
    if (!blockArg)
      continue;

    auto callerIdx = blockArg.getArgNumber();

    // No need to re-analyze arguments that have already been processed
    // for the current node.
    if (current->argsNeedingSpec.test(callerIdx))
      continue;

    // Extract callsite information for the callee.
    ParamDeclAttr calleeParamDecl = calleeNode->getParamDecl(calleeIdx);
    ASSERT_STREAM(calleeParamDecl != nullptr,
                  << "unable to find parameter declaration");

    Type calleeParamTy = calleeParamDecl.getType();

    Type callerParamTy =
        fixupCalleeSpecType(current->argParams, calleeParamTy, calleeNode->func,
                            calleeIdx, callerIdx);

    current->setParamInfoIfNeeded(callerIdx, callerParamTy);
  }
}

static LogicalResult analyzeCallgraph(CallGraph &cg) {
  FailureOr<SmallVector<CallGraphNode *>> maybeWorklist =
      initializeAnalysis(cg);

  if (failed(maybeWorklist))
    return failure();

  SmallVector<CallGraphNode *> worklist = std::move(*maybeWorklist);

  // Perform fixpoint iteration until nothing changes
  while (!worklist.empty()) {
    CallGraphNode *current = worklist.pop_back_val();
    ASSERT_STREAM(
        current->argsNeedingSpec.size() == current->func.getNumArguments(),
        << "node was not properly initialized");

    // The count of nodes needing a spec before the update step.
    // This is used to determine if new spec arguments are needed.
    unsigned beforeNumArgsNeedingSpec = current->argsNeedingSpec.count();

    // Evaluate the transfer function for the current node.
    if (!current->hasTensorArgs())
      continue;

    // Go through each callsite and identify the tensor values which require
    // spects at the callsite.
    for (const CallGraph::EdgeT &edge : current->callsites)
      analyzeCallsite(current, edge);

    unsigned afterNumArgsNeedingSpec = current->argsNeedingSpec.count();
    ASSERT_STREAM(afterNumArgsNeedingSpec >= beforeNumArgsNeedingSpec,
                  << "number of arguments requiring a spec should be "
                     "monotonically increasing during analysis.");

    // If the argsNeedingSpec set changed during the update phase, queue all
    // the dependent (i.e. nodes calling the current function) in the graph.
    if (afterNumArgsNeedingSpec != beforeNumArgsNeedingSpec)
      worklist.append(current->callers.begin(), current->callers.end());
  }

  return success();
}

static void replaceSpecsOfCalls(CallGraphNode *node) {
  auto getterFunctions = node->getterFunctions;

  mlir::AttrTypeReplacer walker;
  walker.addReplacement(
      [&](ParamOperatorAttr attr) -> std::optional<TypedAttr> {
        // Is this one of the specsof getter functions?
        if (!llvm::is_contained(getterFunctions, attr))
          return std::nullopt;

        StringAttr tensorName = cast<StringAttr>(attr.getOperand(1));
        auto paramDecl = node->getParamDecl(tensorName.strref());

        // If parameter information is not found corresponding to the
        // specsof string argument, then the specsof operation is malformed.
        // Validation of the input should have occurred already.
        ASSERT_STREAM(paramDecl, << "unable to resolve specsof variable");

        return ParamDeclRefAttr::get(paramDecl);
      });

  walker.recursivelyReplaceElementsIn(node->specialization,
                                      /*replaceAttrs=*/true,
                                      /*replaceLocs=*/true,
                                      /*replaceTypes=*/true);
}

/// Add the spec parameters to the specialization of the given function
/// node.
static void addSpecParametersToSpecialization(CallGraphNode *node) {
  GeneratorOp func = node->func;
  GeneratorOp specialization = node->specialization;

  // Create the new parameter list for the specialization
  SmallVector<ParamDeclAttr> newParams(specialization.getInputParams());
  for (unsigned index : node->argsNeedingSpec.set_bits())
    newParams.push_back(node->getParamDecl(index));

  // Create the new array for tensor spec names.
  SmallVector<Attribute> tensorSpecParamNames(
      node->argsNeedingSpec.size(), UnitAttr::get(func->getContext()));

  for (unsigned index : node->argsNeedingSpec.set_bits())
    tensorSpecParamNames[index] = node->getParamDecl(index);

  // Update the signature to add the new tensor types.
  SignatureType oldSig = specialization.getSignature();
  FnEffects effects = oldSig.getFnEffects();
  effects.setCapturing(true);
  specialization.setSignature(SignatureType::remapToSignature(
      newParams, {}, specialization.getFunctionType(),
      /*argConventions=*/oldSig.getArgConventions(),
      /*fnEffects=*/effects,
      /*metadata=*/oldSig.getMetadata()));

  // Remove the old params from the function.
  specialization.setInputParams(newParams);
  specialization.getOperation()->setAttr(
      kKernelTensorSpecParameterAttrName,
      ArrayAttr::get(func->getContext(), tensorSpecParamNames));

  // Remove the attribute that identifies the operation name.
  // It ensures MojoLibraryAnalysis will chose the specialized kernel (and not
  // the original one).
  func->removeAttr(MOGGPreElab::kMOGGExecuteFunctionLabel);
  func->removeAttr(MOGGPreElab::kMOGGShapeFunctionLabel);
}

/// Update each CallOp in the specialization of `node` to call the specialized
/// version of the callee (if it was specialized).
static void updateCallOps(CallGraphNode *node, CallGraph &cg,
                          TensorSpecKGEN &specTemplate) {
  GeneratorOp specialization = node->specialization;

  if (!specialization)
    return;

  auto &symtab = cg.symtab;
  auto func = node->func;

  llvm::SmallDenseMap<Value, ParamDeclRefAttr, 8> tensorsToSpecs;

  size_t numOldParams = func.getInputParams().size();
  ArrayRef<ParamDeclAttr> newParams =
      specialization.getInputParams().drop_front(numOldParams);

  for (auto [offset, argIndex] :
       llvm::enumerate(node->argsNeedingSpec.set_bits())) {
    auto arg = specialization.getArgument(argIndex);
    tensorsToSpecs[arg] = ParamDeclRefAttr::get(newParams[offset]);
  }

  specialization.walk([&](CallOp oldCall) {
    // Identify the call
    auto calledSym = oldCall.getCalleeSymbol().getValue();
    auto calledFunc = symtab.lookup<GeneratorOp>(calledSym);
    // Skip any non generators.
    if (!calledFunc)
      return;

    // Specialize the node.
    CallGraphNode *calledFuncNode = cg.lookup(calledFunc);
    if (!calledFuncNode || !calledFuncNode->specialization)
      return;

    // If there's a specialized version then we should call it with the
    // new specs.
    GeneratorOp specialized = calledFuncNode->specialization;

    // Add the old unchanged parameters.
    auto calleeParams = oldCall.getCallee().getParamValues();

    // New parameters to put on the call.
    auto newParamBuffer = llvm::to_vector_of<TypedAttr>(calleeParams);

    // If we need a parameter and we have that parameter add it,
    // otherwise add the none spec.
    for (unsigned index : calledFuncNode->argsNeedingSpec.set_bits()) {
      auto val = oldCall->getOperand(index);
      auto itr = tensorsToSpecs.find(val);
      LLVM_DEBUG(llvm::dbgs()
                 << "Fixup call: add parameter for input #" << index << " "
                 << val << " (@" << oldCall.getCalleeSymbol() << ").\n");
      if (itr == tensorsToSpecs.end()) {
        newParamBuffer.push_back(specTemplate.getValue());
      } else {
        newParamBuffer.push_back(itr->second);
      }
    }

    MLIRContext *ctx = oldCall.getContext();
    auto flatSym = FlatSymbolRefAttr::get(ctx, specialized.getSymName());
    auto specializedSig = specialized.getSignature().getSpecializedSignature(
        newParamBuffer, oldCall.getLoc());
    auto symbol =
        SymbolConstantAttr::get(flatSym, specializedSig, newParamBuffer);

    // Replace the call with a new call operator.
    oldCall.setCalleeAttr(symbol);
  });
}

/// Clone each generator requiring a tensor spec to serve as the
/// specialized version of the generator.
static void createSpecializations(ModuleOp module, CallGraph &cg,
                                  TensorSpecKGEN &specTemplate) {
  SymbolTable &symtab = cg.symtab;

  // Walk the graph in dfs post order to compute the set of new parameters for
  // each function. This computation relies on visiting callees before callers.
  for (GeneratorOp func : module.getOps<GeneratorOp>()) {
    CallGraphNode *node = cg.lookup(func);
    if (!node)
      continue;

    // Functions which are not DPS kernels do not require specialization if
    // none of the arguments need specialization.
    if (!isDPSKernel(func) && node->argsNeedingSpec.none())
      continue;

    // Initialize the specialized version of the function.
    // This will be mutated into its final form and then all calls will be
    // updated to point to the specialization.
    node->specialization = node->func.clone();
    symtab.insert(node->specialization);

    // 1. Find all the uses of compiler.specsof and create new parameters
    //    which replace the compiler.specsof calls.
    replaceSpecsOfCalls(node);

    // 2. Update the specialized function's signature to account for the
    //    new parameters.
    addSpecParametersToSpecialization(node);
  }

  // Now that all the specialized variants of the functions have been created
  // and their signatures are finalized, visit every CallOp with tensor spec
  // parameters and point the CallOp to call the specialized variant of the
  // old function.
  for (GeneratorOp func : module.getOps<GeneratorOp>()) {
    // Specialized functions are now in the module, which will not have
    // callgraph nodes associated with them.
    if (CallGraphNode *node = cg.lookup(func))
      updateCallOps(node, cg, specTemplate);
  }
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

    if (failed(analyzeCallgraph(cg)))
      return signalPassFailure();

    createSpecializations(mod, cg, specTemplate);
  }
};

} // namespace
