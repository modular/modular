//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ConstraintSet.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/HLCFDialect/HLCFOps.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERLIT
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
class SignatureUnifier {
public:
  SignatureUnifier(GeneratorOp generatorOp, GeneratorInterfaceOp interfaceOp);

  /// Add the constraints already on the generator to the constraint set,
  /// returning failure if a contradiction was detected.
  LogicalResult checkExistingConstraints();

  ParseResult addEqualityConstraintFn(ParamDeclRefAttr param, TypedAttr value);
  ParseResult addEquivalenceConstraint(ParamDeclRefAttr lhs,
                                       ParamDeclRefAttr rhs);

  ParseResult tryUnifyingTypes(Type itfArgTy, Type genArgTy);
  ParseResult tryUnifyingTypeParameters(Attribute itfParam, Attribute genParam);
  ParseResult checkArgumentType(size_t argNo, Type itfArgTy, Type genArgTy,
                                Location loc);
  ParseResult checkResultType(size_t argNo, Type itfResultTy, Type genResultTy,
                              Location loc);

  // Now that we've inferred parameters, we may have inferred new input
  // parameters.  Check to see that whatever we have is a complete covering of
  // the interfaces expectations.
  LogicalResult verifyInputParameters();

  void reinstallConstraints();

public:
  GeneratorOp generatorOp;
  GeneratorInterfaceOp interfaceOp;

  ConstraintSet constraints;

  /// This string is set to information indicating context about in inferred
  /// constraint or diagnostic, e.g. that this is happening with argument #0.
  std::string inferenceContext;
  Location inferenceLoc;
};
} // namespace

SignatureUnifier::SignatureUnifier(GeneratorOp generatorOp,
                                   GeneratorInterfaceOp interfaceOp)
    : generatorOp(generatorOp), interfaceOp(interfaceOp),
      constraints(generatorOp),
      inferenceLoc(UnknownLoc::get(generatorOp.getContext())) {}

/// Add the constraints already on the generator to the constraint set,
/// returning failure if a contradiction was detected.
LogicalResult SignatureUnifier::checkExistingConstraints() {
  for (ConstraintAttr constraint : generatorOp.getConstraints())
    if (failed(constraints.addConstraint(constraint)))
      return failure();

  return success();
}

/// Now that we've inferred parameters, we may have inferred new input
/// parameters.  Check to see that whatever we have is a complete covering of
/// the interface's expectations.
LogicalResult SignatureUnifier::verifyInputParameters() {
  // The lit.func may have additional input parameters that are
  // disallowed, and may be missing parameters.  We may have inferred some or
  // all of the missing parameters, but if not, we need to reject.
  ArrayRef<ParamDeclAttr> inputParamDecls = generatorOp.getInputParamDecls();
  SmallPtrSet<Attribute, 8> inputParams(inputParamDecls.begin(),
                                        inputParamDecls.end());
  // Add the parameter decls that were inferred.
  for (ParamDeclRefAttr declRef :
       constraints.getPotentiallyInferredParameters()) {
    // Convert ParamDeclRefAttr -> ParamDeclAttr.
    inputParams.insert(
        ParamDeclAttr::get(declRef.getName(), declRef.getType()));
  }

  // Ok, now that we have all the input parameters, validate that they match up.
  // We do this by checking the set for everything that should be there and
  // deleting them as we go.  By the end, the set should be empty.
  for (ParamDeclAttr itfParam : interfaceOp.getInputParamDecls()) {
    // In the normal case, the
    if (inputParams.erase(itfParam))
      continue;

    // Well we have a problem to diagnose.  It could be because the parameter is
    // missing or the type doesn't match.  Scan for a matching name.
    for (Attribute genParamC : inputParams) {
      ParamDeclAttr genParam = genParamC.cast<ParamDeclAttr>();
      if (genParam.getName() == itfParam.getName()) {
        // Ok, found matching names but the types don't match.
        auto diag = generatorOp.emitError("input parameter ")
                    << genParam.getName() << " has type " << genParam.getType()
                    << " but interface expects " << itfParam.getType();
        diag.attachNote(interfaceOp.getLoc()) << "interface defined here";
        return failure();
      }
    }

    // If no match is found then it is missing.
    auto diag = generatorOp.emitError("missing interface input parameter ")
                << itfParam.getName() << " of type " << itfParam.getType();
    diag.attachNote(interfaceOp.getLoc()) << "interface defined here";
    return failure();
  }

  // If we have left over entries in `inputParams` then we have extra
  // parameters.
  if (!inputParams.empty()) {
    auto badParam = (*inputParams.begin()).cast<ParamDeclAttr>();
    auto diag = generatorOp.emitError("input parameter ")
                << badParam.getName() << " is unexpected by interface";
    diag.attachNote(interfaceOp.getLoc()) << "interface defined here";
    return failure();
  }

  // Finally after all this checking, we know the generator has the same
  // input parameters as the interface so we can just take it directly!
  generatorOp.setSignature(SignatureType::get(
      interfaceOp.getInputParamDeclsAttr(), generatorOp.getResultParamsAttr(),
      generatorOp.getFunctionType(), generatorOp.getConventions()));
  return success();
}

/// When we're done checking the conformance, this method reinstalls the
/// (possibly updated) constraint information on the generator declaration.
void SignatureUnifier::reinstallConstraints() {
  generatorOp.setConstraintsAttr(constraints.getConstraintsSpec());
}

ParseResult SignatureUnifier::addEqualityConstraintFn(ParamDeclRefAttr param,
                                                      TypedAttr value) {
  auto message = StringAttr::get(value.getContext(),
                                 Twine(inferenceContext) + " specifies '" +
                                     param.getName().getValue() +
                                     "' = " + getParamAsString(value));
  auto constraintValue =
      PointwiseValue::getSingleValue(value, message, inferenceLoc);
  return constraints.addPointwiseParamConstraint(param, constraintValue);
}

ParseResult SignatureUnifier::addEquivalenceConstraint(ParamDeclRefAttr lhs,
                                                       ParamDeclRefAttr rhs) {
  auto message = StringAttr::get(lhs.getContext(),
                                 Twine(inferenceContext) + " specifies '" +
                                     lhs.getName().getValue() + "' = '" +
                                     rhs.getName().getValue() + "'");
  auto constraintValue =
      PointwiseValue::getParamEquivalence(rhs, message, inferenceLoc);
  return constraints.addPointwiseParamConstraint(lhs, constraintValue);
}

ParseResult SignatureUnifier::tryUnifyingTypeParameters(Attribute itfParam,
                                                        Attribute genParam) {
  // If these attributes are (recursively) identical, then they match.
  if (itfParam == genParam)
    return success();

  // If the interface requires something but the generator is ? then the
  // generator is more flexible than it needs to be.
  if (isa<UnknownAttr>(genParam))
    return success();

  // If the interface is ? but the generator is more specific, then we cannot
  // support this: we cannot impose a constraint on a ?.
  if (isa<UnknownAttr>(itfParam)) {
    // TODO: It is possible to add inferred dynamic constraints when we have an
    // error handling model.
    auto diag = emitError(inferenceLoc, inferenceContext)
                << ": dynamic `?` value cannot have static constraint: '"
                << genParam << "'";
    diag.attachNote(interfaceOp->getLoc()) << "interface declared here";
    return failure();
  }

  if (auto decl = dyn_cast<ParamDeclRefAttr>(itfParam)) {
    // If one of these is a parameter, and one is concrete, then that infers a
    // value for the parameter.
    if (ParameterAttr::isSimpleConstant(genParam))
      return addEqualityConstraintFn(decl, genParam);
    // If the other is a parameter, then that infers an equivalence constraint.
    if (auto genDecl = dyn_cast<ParamDeclRefAttr>(genParam))
      return addEquivalenceConstraint(decl, genDecl);
  }

  // Otherwise we don't know how to unify this.
  // TODO: Could handle node-wise merging of expressions to find constraints
  // like "x+1" and "y+1" --> "x == y".

  // If both parameters are type expressions, try to unify the contained types.
  if (auto itfType = dyn_cast<TypeConstantAttr>(itfParam))
    if (auto genType = dyn_cast<TypeConstantAttr>(genParam))
      return tryUnifyingTypes(itfType.getValue(), genType.getValue());

  auto itfElems = dyn_cast<mlir::SubElementAttrInterface>(itfParam);
  if (!itfElems) {
    auto diag = emitError(inferenceLoc, inferenceContext)
                << ": cannot unify : '" << genParam << "'";
    diag.attachNote(interfaceOp->getLoc()) << "interface declared here";
    return failure();
  }
  auto genElems = cast<mlir::SubElementAttrInterface>(genParam);

  SmallVector<Attribute> itfParams, genParams;
  SmallVector<Type> itfTypes, genTypes;
  itfElems.walkImmediateSubElements(
      [&](Attribute attr) { itfParams.push_back(attr); },
      [&](Type type) { itfTypes.push_back(type); });
  genElems.walkImmediateSubElements(
      [&](Attribute attr) { genParams.push_back(attr); },
      [&](Type type) { genTypes.push_back(type); });
  assert(itfParams.size() == genParams.size() &&
         itfTypes.size() == genTypes.size());

  // Unify each expression.
  for (auto [itfParam, genParam] : llvm::zip(itfParams, genParams))
    if (failed(tryUnifyingTypeParameters(itfParam, genParam)))
      return failure();
  for (auto [itfType, genType] : llvm::zip(itfTypes, genTypes))
    if (failed(tryUnifyingTypes(itfType, genType)))
      return failure();
  return success();
}

/// Check to see if the specified types can be merged, where the 'itfArgTy' is
/// the argument type from the interface and 'genArgTy' is the actual argument
/// from the generator.  On failure, this generates a failure but does not emit
/// an error message.
ParseResult SignatureUnifier::tryUnifyingTypes(Type itfArgTy, Type genArgTy) {
  // If the types are identical then of course they match.
  if (itfArgTy == genArgTy)
    return success();

  // If the interface type is a parameter reference, try to unify them.
  if (auto itfParamRef = dyn_cast<ParamRefType>(itfArgTy))
    return tryUnifyingTypeParameters(itfParamRef.getParam(),
                                     TypeConstantAttr::get(genArgTy));

  // If they don't match, then reject them.
  if (itfArgTy.getTypeID() != genArgTy.getTypeID()) {
    auto diag = emitError(inferenceLoc, inferenceContext)
                << " has type " << genArgTy << " but interface expected type "
                << itfArgTy;
    diag.attachNote(interfaceOp->getLoc()) << "interface declared here";
    return failure();
  }

  // Try to unify their nested parameter expressions.
  auto itfElems = dyn_cast<mlir::SubElementTypeInterface>(itfArgTy);
  if (!itfElems) {
    return emitError(inferenceLoc, inferenceContext)
           << " has type " << genArgTy << " not equal to interface type "
           << itfArgTy << " but does not implement SubElementTypeInterface";
  }
  auto genElems = genArgTy.cast<mlir::SubElementTypeInterface>();

  SmallVector<Attribute> itfParams, genParams;
  SmallVector<Type> itfTypes, genTypes;
  itfElems.walkImmediateSubElements(
      [&](Attribute attr) { itfParams.push_back(attr); },
      [&](Type type) { itfTypes.push_back(type); });
  genElems.walkImmediateSubElements(
      [&](Attribute attr) { genParams.push_back(attr); },
      [&](Type type) { genTypes.push_back(type); });
  assert(itfParams.size() == genParams.size() &&
         itfTypes.size() == genTypes.size());

  // Unify each expression.
  for (auto [itfParam, genParam] : llvm::zip(itfParams, genParams))
    if (failed(tryUnifyingTypeParameters(itfParam, genParam)))
      return failure();
  for (auto [itfType, genType] : llvm::zip(itfTypes, genTypes))
    if (failed(tryUnifyingTypes(itfType, genType)))
      return failure();
  return success();
}

ParseResult SignatureUnifier::checkArgumentType(size_t argNo, Type itfArgTy,
                                                Type genArgTy, Location loc) {
  inferenceContext = "argument #" + std::to_string(argNo);
  inferenceLoc = loc;

  // Try unifying the types.  If this successed, then the signature types match.
  return tryUnifyingTypes(itfArgTy, genArgTy);
}

ParseResult SignatureUnifier::checkResultType(size_t argNo, Type itfResultTy,
                                              Type genResultTy, Location loc) {
  inferenceContext = "result #" + std::to_string(argNo);
  inferenceLoc = loc;

  // Try unifying the types.  If this successed, then the signature types match.
  return tryUnifyingTypes(itfResultTy, genResultTy);
}

/// Insert a cast of 'arg' to 'type' for an argument/result conversion when
/// generating a generator thunk (if needed).
static Value insertRebindOp(Value arg, Type type, ImplicitLocOpBuilder &b) {
  if (arg.getType() == type)
    return arg;
  return b.create<RebindOp>(type, arg);
}

/// If this generator is implementing an interface, check its conformance,
/// diagnose any conflicts, and infer constraints.  Note that 'itf' may be null
/// if this generator is not implementing an interface.
static LogicalResult
checkInterfaceConformance(GeneratorOp gen, GeneratorInterfaceOp itf,
                          SymbolTable &symbolTable,
                          FlatSymbolRefAttr implementsAttr) {
  SignatureUnifier unifier(gen, itf);

  // Verify that the constraints already imposed on the generator are
  // satisfiable.
  if (failed(unifier.checkExistingConstraints()))
    return failure();

  // If this generator is not actually implementing an interface, just return
  // after successfully checking the existing constraints for contradictions.
  if (!itf) {
    unifier.reinstallConstraints();
    return success();
  }

  // If the generator and the interface have differing signatures, we need to
  // synthesize a forwarding thunk.
  bool needsForwardingThunk = false;

  // Match up the argument types with the generator's.  These are allowed to
  // be more specialized, in which case they imply argument constraints.
  auto itfArgs = itf.getArgumentTypes();
  auto genArgs = gen.getArguments();
  if (itfArgs.size() != genArgs.size()) {
    auto diag = gen.emitOpError()
                << "generator has " << genArgs.size() << " argument"
                << (genArgs.size() != 1 ? "s" : "") << " but interface expects "
                << itfArgs.size();
    diag.attachNote(itf->getLoc()) << "interface declared here";
    return failure();
  }
  size_t itemNo = 0;
  for (auto [itfArgTy, genArg] : llvm::zip(itfArgs, genArgs)) {
    if (failed(unifier.checkArgumentType(itemNo, itfArgTy, genArg.getType(),
                                         genArg.getLoc())))
      return failure();
    needsForwardingThunk |= itfArgTy != genArg.getType();
    ++itemNo;
  }

  // Check and integrate the result types.
  auto itfResTys = itf.getResultTypes();
  auto genResTys = gen.getResultTypes();
  if (itfResTys.size() != genResTys.size()) {
    auto diag = gen.emitOpError()
                << "generator has " << genResTys.size() << " result"
                << (genResTys.size() != 1 ? "s" : "")
                << " but interface expects " << itfResTys.size();
    diag.attachNote(itf->getLoc()) << "interface declared here";
    return failure();
  }
  itemNo = 0;
  for (auto [itfResTy, genResTy] : llvm::zip(itfResTys, genResTys)) {
    // TODO: We don't have per-result location info.
    auto resultLoc = gen.getReturnOp().getLoc();
    if (failed(unifier.checkResultType(itemNo, itfResTy, genResTy, resultLoc)))
      return failure();
    needsForwardingThunk |= itfResTy != genResTy;
    ++itemNo;
  }

  // Now that we've inferred parameters, we may have inferred new input
  // parameters.  Check to see that whatever we have is a complete covering of
  // the interfaces expectations.
  if (failed(unifier.verifyInputParameters()))
    return failure();

  // Now that we have successfully completed inference, reinstall updated
  // constraint attrs.
  unifier.reinstallConstraints();

  // If the generator has a different (i.e., more specific) signature than the
  // interface requires, then it cannot directly fulfill the interface at the
  // kgen level - we need to generate a thunk.
  if (needsForwardingThunk) {
    ImplicitLocOpBuilder b(gen.getLoc(), gen);
    auto thunk =
        b.create<GeneratorOp>(b.getStringAttr(gen.getSymName() + "_thunk"),
                              // Take the signature from the interface.
                              itf.getSignatureAttr(),
                              // Take the constraints from the generator.
                              gen.getConstraintsAttr(), implementsAttr);
    // The thunk implements the interface, not the original generator.
    gen.removeImplementsAttr();

    // Have the symbol table unique our provisional name.
    symbolTable.insert(thunk);

    // Set up the body.
    Block *body = thunk.addEntryBlock();
    b.setInsertionPoint(body, body->end());

    // Set up the argument list for the call.
    SmallVector<Value> castedArgs;
    for (auto [bodyArg, genArg] :
         llvm::zip(body->getArguments(), gen.getBody()->getArguments())) {
      // The thunk argument locations should be the locations of the generator
      // arguments.
      bodyArg.setLoc(genArg.getLoc());
      b.setLoc(genArg.getLoc());

      // Insert a cast from the more general interface argument type to the more
      // specific type implemented by the generator.
      castedArgs.push_back(insertRebindOp(bodyArg, genArg.getType(), b));
    }

    // The call will need to passes on all the input parameters unmodified.
    SmallVector<ParamBindAttr> callInputParams;
    for (ParamDeclAttr inParam : gen.getInputParamDecls()) {
      auto value = ParamDeclRefAttr::get(inParam.getName(), inParam.getType());
      callInputParams.push_back(ParamBindAttr::get(inParam.getName(), value));
    }

    // It also captures the result parameters and returns them from the
    // kgen.output for the thunk.
    SmallVector<ParamDeclAttr> callResultParams; // <StringAttr name, Type type>
    SmallVector<TypedAttr> returnParams;

    unsigned paramNo = 0;
    for (ParamDeclAttr resultParam : gen.getResultParams()) {
      auto paramName = b.getStringAttr("resultParam" + Twine(paramNo++));

      // The call returns the same thing as the generator.
      callResultParams.push_back(
          ParamDeclAttr::get(paramName, resultParam.getType()));

      // The output binds each result from the call into the return value of the
      // generator thunk.
      returnParams.push_back(
          ParamDeclRefAttr::get(paramName, resultParam.getType()));
    }

    // Create the call.
    b.setLoc(gen.getLoc());
    auto callOp = b.create<CallOp>(
        gen.getResultTypes(),
        SymbolConstantAttr::get(
            FlatSymbolRefAttr::get(gen.getNameAttr()),
            ParamBindArrayAttr::get(gen.getContext(), callInputParams),
            gen.getSignature().dropParamValues()),
        callResultParams, castedArgs);

    // Create any rebind's for the results.
    SmallVector<Value> results;
    for (auto [result, resultTy] : llvm::zip(callOp.getResults(), itfResTys))
      results.push_back(insertRebindOp(result, resultTy, b));

    b.create<KGEN::ReturnOp>(b.getAttr<ParameterExprArrayAttr>(returnParams),
                             results);

    // The thunk is required because there could be direct callers of the
    // original generator, which expect the original signature.  If there
    // aren't, then we can just inline it away.
    // TODO: Inline these away if/when they have no additional callers.
  }

  return success();
}

static void buildDebugInfoValue(Operation *insertPt, Location loc,
                                StringRef varName,
                                DebugInfo::DIFileAttr fileAttr, Value value,
                                Type type) {
  auto fileLoc = loc->findInstanceOf<FileLineColLoc>();
  auto varScope = DebugInfo::extractScope<DebugInfo::DILocalScopeAttr>(loc);
  if (!fileLoc || !varScope)
    return;

  auto varAttr = DebugInfo::DILocalVariableAttr::get(
      varScope, varName, fileAttr, fileLoc.getLine(), /*arg=*/0,
      /*alignInBits=*/0, DebugInfo::DIUnresolvedMLIRType::get(type));
  OpBuilder(insertPt).create<DebugInfo::ValueOp>(loc, value, varAttr);
}

/// Flatten the given symbol reference, collapsing all nested scopes into one
/// mangled name.
static FlatSymbolRefAttr flattenSymbolRefAttr(SymbolRefAttr ref) {
  // If the symbol is already flat, there is nothing to do.
  if (auto flatSym = dyn_cast<FlatSymbolRefAttr>(ref))
    return flatSym;

  // Flatten the symbol name into a single string.
  SmallString<32> name = ref.getRootReference().getValue();
  llvm::raw_svector_ostream nameOS(name);
  for (FlatSymbolRefAttr sym : ref.getNestedReferences())
    nameOS << "::" << sym.getValue();
  return SymbolRefAttr::get(ref.getContext(), nameOS.str());
}

static void lowerLITOps(LIT::FuncOp func,
                        DebugInfo::DISubprogramAttr funcSpAttr) {
  // Check if we are building debug info for source variables.
  bool buildingDebugVars =
      funcSpAttr && funcSpAttr.getCompileUnit().getEmissionKind() ==
                        DebugInfo::EmissionKind::Full;
  SmallDenseMap<StringAttr, ParamDeclRefAttr> nestedFuncRenames;
  func.walk([&](Operation *op) {
    mlir::IRRewriter b{OpBuilder(op)};
    if (isa<AliasForwardDeclOp>(op)) {
      // lit.alias.fwd.decl is used internally by the frontend, but is not
      // needed by lowering at all.
      op->erase();
    } else if (auto letDecl = dyn_cast<LIT::LetDeclOp>(op)) {
      // Build information for this decl if necessary.
      if (buildingDebugVars) {
        buildDebugInfoValue(letDecl, letDecl.getLoc(), letDecl.getName(),
                            funcSpAttr.getFile(), letDecl.getOperand(),
                            letDecl.getType());
      }

      b.replaceOp(letDecl, letDecl.getOperand());
    } else if (auto varDecl = dyn_cast<LIT::VarDeclOp>(op)) {
      StringAttr varName = varDecl.getNameAttr();
      auto varType = varDecl.getType();

      // Lower a lit.var.decl to pop.stack_allocation.
      auto allocOp =
          b.replaceOpWithNewOp<POP::StackAllocationOp>(varDecl, varType, 1);

      // Build information for this variable if necessary.
      if (buildingDebugVars) {
        // TODO: Mark the value op as describing the "address" of the
        // variable, instead of claiming to describe the variable itself.
        buildDebugInfoValue(allocOp->getNextNode(), allocOp.getLoc(), varName,
                            funcSpAttr.getFile(), allocOp, varType);
      }
    } else if (auto nestedFunc = dyn_cast<LIT::FuncOp>(op);
               nestedFunc && nestedFunc != func) {
      // Process a nested function by lowering it straight to a
      // `kgen.param.declare.region`. We need to replace all the symbol
      // references within the function. The parser ensures that the symbol name
      // is unique with parameters.
      auto region = b.create<ParamDeclareRegionOp>(
          op->getLoc(), ParamDeclAttr::get(nestedFunc.getSymNameAttr(),
                                           nestedFunc.getSignature()));
      b.createBlock(&region.getBody());
      auto body = b.create<RegionBodyOp>(
          op->getLoc(), nestedFunc.getSignature(), ArrayRef<ConstraintAttr>());
      body.getBodyRegion().takeBody(nestedFunc.getBodyRegion());
      nestedFuncRenames.try_emplace(
          b.getStringAttr(func.getName() + "::" + nestedFunc.getName()),
          ParamDeclRefAttr::get(nestedFunc.getSymNameAttr(),
                                nestedFunc.getSignature()));
      b.eraseOp(nestedFunc);
    }
  });

  // Demote direct calls to nested functions to `call_param` so the callee can
  // be rewritten.
  func.walk([&](CallOp call) {
    if (!nestedFuncRenames.lookup(
            flattenSymbolRefAttr(call.getCallee().getSymbol()).getAttr()))
      return;
    mlir::IRRewriter b{OpBuilder(call)};
    b.replaceOpWithNewOp<CallParamOp>(
        call, call.getResultTypes(), call.getCallee(), call.getParamDeclsAttr(),
        call.getOperands());
  });
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](SymbolConstantAttr ref) -> Attribute {
    ParamDeclRefAttr newRef = nestedFuncRenames.lookup(
        flattenSymbolRefAttr(ref.getSymbol()).getAttr());
    if (!newRef)
      return ref;
    if (ref.getParamValues().empty())
      return newRef;
    // If the symbol constant had bindings, create a `bind_signature`.
    SmallVector<TypedAttr> operands;
    operands.push_back(newRef);
    for (ParamBindAttr bind : ref.getParamValues())
      operands.push_back(bind.getValue());
    return ParamOperatorAttr::get(POC::BindSignature, operands);
  });
  replacer.recursivelyReplaceElementsIn(func, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);
}

/// Flatten the name of the given symbol operation and insert it in the given
/// symbol table with that flattened name. Returns the flattened symbol name.
template <typename T>
static StringAttr flattenAndRenameSymbol(T op, const Twine &parentPrefix,
                                         SymbolTable &symbolTable,
                                         Block::iterator symbolTableIt) {
  StringAttr name = op.getSymNameAttr();
  if (parentPrefix.isTriviallyEmpty())
    return name;

  // Remove the operation in preparation for re-insertion. This gets handled
  // differently depending on if we are already tracking this op in the symbol
  // table.
  if (op->getParentOp() == symbolTable.getOp())
    symbolTable.remove(op);
  else
    op->remove();

  StringAttr newName =
      StringAttr::get(name.getContext(), parentPrefix + name.getValue());
  op.setSymNameAttr(newName);
  symbolTable.insert(op, symbolTableIt);
  return newName;
}

/// Lower an lit.func to kgen.generator.
static LogicalResult
lowerLITFunc(LIT::FuncOp gen, SymbolTable &symbolTable,
             Block::iterator symTableIt, const Twine &parentPrefix,
             ArrayRef<ParamDeclAttr> parentInputParams = {}) {
  auto funcSpAttr = DebugInfo::extractScope<DebugInfo::DISubprogramAttr>(gen);

  // Update the function name, incorporating the parent prefix.
  if (!parentPrefix.isTriviallyEmpty()) {
    StringAttr newName =
        flattenAndRenameSymbol(gen, parentPrefix, symbolTable, symTableIt);

    // If this function has a subprogram attached, update its information to
    // account for the new name.
    if (funcSpAttr) {
      auto newSpAttr = DebugInfo::DISubprogramAttr::get(
          funcSpAttr.getContext(), funcSpAttr.getCompileUnit(),
          funcSpAttr.getScope(), funcSpAttr.getName(), newName,
          funcSpAttr.getFile(), funcSpAttr.getLine(), funcSpAttr.getScopeLine(),
          funcSpAttr.getSubprogramFlags(), funcSpAttr.getType());

      DebugInfo::DIAttrTypeReplacer replacer;
      replacer.addReplacement([&](DebugInfo::DISubprogramAttr attr) {
        return attr == funcSpAttr ? newSpAttr : attr;
      });
      replacer.recursivelyReplaceElementsIn(gen);
      funcSpAttr = newSpAttr;
    }
  }

  // Prepend the parameters from the parent decl if present.
  if (!parentInputParams.empty()) {
    SmallVector<ParamDeclAttr> paramDecls;
    ArrayRef<ParamDeclAttr> genParamDecls = gen.getInputParamDecls();
    paramDecls.reserve(parentInputParams.size() + genParamDecls.size());
    llvm::append_range(paramDecls, parentInputParams);
    llvm::append_range(paramDecls, genParamDecls);

    gen.setSignature(SignatureType::get(
        ParamDeclArrayAttr::get(gen.getContext(), paramDecls),
        gen.getResultParamsAttr(), gen.getSignature().getValues(),
        gen.getConventions()));
  }

  lowerLITOps(gen, funcSpAttr);
  OpBuilder b(gen);

  // Is a LITFuncOp with empy body representing an interface?
  if (gen.getIsInterface()) {
    SymbolConstantAttr evaluator;
    if (gen.getEvaluator().has_value())
      evaluator = gen.getEvaluatorAttr();
    auto result = b.create<GeneratorInterfaceOp>(
        gen.getLoc(), gen.getSymNameAttr(), gen.getSignatureAttr(),
        gen.getConstraintsAttr(), evaluator, nullptr);
    // Move over the symbol.
    symbolTable.erase(gen);
    symbolTable.insert(result);
    return success();
  }

  // Flatten the implements reference if present.
  FlatSymbolRefAttr implementsAttr;
  if (auto fullImplementsAttr = gen.getImplementsAttr())
    implementsAttr = flattenSymbolRefAttr(fullImplementsAttr);

  // Directly lower since these operations are exactly identical right now.
  auto result = b.create<GeneratorOp>(gen.getLoc(), gen.getSymNameAttr(),
                                      gen.getSignatureAttr(),
                                      gen.getConstraintsAttr(), implementsAttr);

  // Move over the body.
  auto *bodyBlock = gen.getBody();
  gen.getBodyRegion().getBlocks().remove(bodyBlock);
  result.getBodyRegion().push_back(bodyBlock);

  // Move over the symbol.
  symbolTable.erase(gen);
  gen = LIT::FuncOp(); // The line above also erases 'gen'.
  symbolTable.insert(result);

  // If the generator implemented an interface, infer additional constraints
  // and check the signature.
  GeneratorInterfaceOp itf;
  if (implementsAttr) {
    // Check that the callee attribute was specified.
    itf = dyn_cast_if_present<GeneratorInterfaceOp>(
        symbolTable.lookup(implementsAttr.getAttr()));
    if (!itf) {
      return result.emitError("could not find implemented interface: ")
             << implementsAttr.getValue();
    }
  }

  return checkInterfaceConformance(result, itf, symbolTable, implementsAttr);
}

/// Lower nested structures in lit.struct.decl away.
static LogicalResult lowerStructDecl(StructDeclOp structDecl,
                                     SymbolTable &symbolTable,
                                     Block::iterator symTableIt,
                                     const Twine &parentPrefix) {
  // Update the name of this struct, incorporating any parent prefix.
  StringAttr structName =
      flattenAndRenameSymbol(structDecl, parentPrefix, symbolTable, symTableIt);

  ArrayRef<ParamDeclAttr> structInputParams = structDecl.getInputParamDecls();
  SmallVector<LIT::VarDeclOp> opsToErase;
  for (Operation &member : llvm::make_early_inc_range(
           structDecl.getFields().front().getOperations())) {
    if (isa<StructFieldOp>(member))
      continue; // Already lowered field.

    if (auto varDecl = dyn_cast<LIT::VarDeclOp>(member)) {
      Type elemType = ParamRefType::get(varDecl.getType().getElementType());
      OpBuilder b(&member);
      b.create<StructFieldOp>(member.getLoc(), varDecl.getName(), elemType);
      varDecl->erase();
      continue;
    } else if (auto paramDeclare = dyn_cast<KGEN::ParamDeclareOp>(member)) {
      paramDeclare.erase();
      continue;
    }
    auto func = dyn_cast<LIT::FuncOp>(member);
    if (!func)
      return member.emitError("unsupported op in lit lowering");

    // Lower renamed function as usual.
    if (failed(lowerLITFunc(func, symbolTable, structDecl->getIterator(),
                            structName.getValue() + "::", structInputParams)))
      return failure();
  }
  return success();
}

static void lowerAttributesAndTypes(Operation *op) {
  mlir::AttrTypeReplacer replacer;

  // Member functions are reference with nested symbol references. After
  // lowering, the symbol tree will be flat. Concatenate all nested symbol
  // references in symbol constants.
  replacer.addReplacement(
      [](SymbolRefAttr ref) { return flattenSymbolRefAttr(ref); });

  // Lower `!lit.none` to `list<i1[0]>`, which will eventually become nothing.
  auto emptyList = ListType::get(IntegerType::get(op->getContext(), 1), 0);
  replacer.addReplacement([&](KGEN::LIT::NoneType type) { return emptyList; });
  // Lower `#lit.none` to `[]`.
  replacer.addReplacement([&](LIT::NoneAttr attr) {
    return ListAttr::get(attr.getContext(), {}, emptyList);
  });

  // Remove all input conventions, but keep function effects.
  replacer.addReplacement([](ConventionsAttr conventions) {
    return ConventionsAttr::get(conventions.getContext(),
                                conventions.getInputConventions().size(),
                                conventions.getFnEffects());
  });

  replacer.recursivelyReplaceElementsIn(
      op, /*replaceAttrs=*/true, /*replaceLocs=*/true, /*replaceTypes=*/true);
}

/// Lower the constructs within the body of a module decl.
static LogicalResult lowerModuleDecl(Block *moduleBody,
                                     SymbolTable &symbolTable,
                                     Block::iterator symTableIt = {},
                                     const Twine &parentPrefix = {}) {
  bool isTopLevel = symTableIt == Block::iterator();
  for (Operation &op : llvm::make_early_inc_range(*moduleBody)) {
    // If we are already in the symbol table, use the the operations iterator.
    auto opSymTableIt = isTopLevel ? op.getIterator() : symTableIt;

    if (auto exportOp = dyn_cast<ExportOp>(op)) {
      OpBuilder(&op).create<KGEN::ExportOp>(exportOp.getLoc(),
                                            exportOp.getExportsAttr());
      exportOp->erase();
    } else if (auto func = dyn_cast<LIT::FuncOp>(op)) {
      if (failed(lowerLITFunc(func, symbolTable, opSymTableIt, parentPrefix)))
        return failure();
    } else if (auto structDecl = dyn_cast<StructDeclOp>(op)) {
      if (failed(lowerStructDecl(structDecl, symbolTable, opSymTableIt,
                                 parentPrefix)))
        return failure();
    } else if (auto fileDecl = dyn_cast<LIT::FileModuleOp>(op)) {
      // Lower the constructs within the body.
      Block *fileBody = fileDecl.getBody();
      if (failed(lowerModuleDecl(fileBody, symbolTable, opSymTableIt,
                                 parentPrefix + fileDecl.getName() + "::")))
        return failure();

      // Inline the remaining body of the file into the parent.
      fileDecl->getBlock()->getOperations().splice(
          fileDecl->getIterator(), fileBody->getOperations(), fileBody->begin(),
          fileBody->end());
      fileDecl->erase();
    } else if (auto paramDeclare = dyn_cast<KGEN::ParamDeclareOp>(op)) {
      op.erase();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pass boilerplate.
//===----------------------------------------------------------------------===//

namespace {
struct LowerLITPass : public impl::LowerLITBase<LowerLITPass> {
  void runOnOperation() override {
    // TODO: This has to be a module pass because this mutates the body of
    // the module, but we could trivially parallelize this within the pass.
    ModuleOp module = getOperation();
    SymbolTable &symbolTable =
        getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
    if (failed(lowerModuleDecl(module.getBody(), symbolTable)))
      return signalPassFailure();
    lowerAttributesAndTypes(module);
  }
};

} // namespace
