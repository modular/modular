//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENPasses.h"

#include "ConstraintSet.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;

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
  ArrayRef<ParamDeclAttr> inputParamDecls = generatorOp.getParamDecls();
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
  for (ParamDeclAttr itfParam : interfaceOp.getParamDecls()) {
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
  generatorOp.setParamDeclsAttr(interfaceOp.getParamDeclsAttr());
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
    if (isSimpleConstant(genParam))
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
static LogicalResult checkInterfaceConformance(GeneratorOp gen,
                                               GeneratorInterfaceOp itf,
                                               SymbolTable &symbolTable) {
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
    auto thunk = b.create<GeneratorOp>(
        b.getStringAttr(gen.getSymName() + "_thunk"),
        // Take the signature from the interface.
        itf.getFunctionTypeAttr(), itf.getParamDeclsAttr(),
        itf.getResultParamTypesAttr(),
        // Take the constraints from the generator.
        gen.getConstraintsAttr(), gen.getImplementsAttr());
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
    for (Type resultParamType : gen.getResultParamTypes()) {
      auto paramName = b.getStringAttr("resultParam" + Twine(paramNo++));

      // The call returns the same thing as the generator.
      callResultParams.push_back(
          ParamDeclAttr::get(paramName, resultParamType));

      // The output binds each result from the call into the return value of the
      // generator thunk.
      returnParams.push_back(ParamDeclRefAttr::get(paramName, resultParamType));
    }

    // Create the call.
    b.setLoc(gen.getLoc());
    auto callOp = b.create<CallOp>(
        gen.getResultTypes(), FlatSymbolRefAttr::get(gen.getNameAttr()),
        callInputParams, callResultParams, castedArgs, /*no regions*/ 0);

    // Create any rebind's for the results.
    SmallVector<Value> results;
    for (auto [result, resultTy] : llvm::zip(callOp.getResults(), itfResTys))
      results.push_back(insertRebindOp(result, resultTy, b));

    b.create<ReturnOp>(b.getAttr<ParameterExprArrayAttr>(returnParams),
                       results);

    // The thunk is required because there could be direct callers of the
    // original generator, which expect the original signature.  If there
    // aren't, then we can just inline it away.
    // TODO: Inline these away if/when they have no additional callers.
  }

  return success();
}

/// Lower a lit.var.decl to pop.stack_allocation.
static void lowerLITVarDecl(LITFuncOp func) {
  func.walk([&](KGEN::VarDeclOp varDecl) -> void {
    OpBuilder b(varDecl);
    Value stackAlloc = b.create<POP::StackAllocationOp>(varDecl.getLoc(),
                                                        varDecl.getType(), 1);
    varDecl.replaceAllUsesWith(stackAlloc);
    varDecl->erase();
  });
}

/// Lower a lit.struct.gep to kgen.struct.gep.
static void lowerLITStructGEP(LITFuncOp func) {
  func.walk([&](LITStructGEPOp gepOp) -> void {
    OpBuilder b(gepOp);
    Value kgenGepOP =
        b.create<StructGEPOp>(gepOp.getLoc(), gepOp.getType(), gepOp.getField(),
                              gepOp.getContainer());
    gepOp.replaceAllUsesWith(kgenGepOP);
    gepOp->erase();
  });
}

/// Lower an lit.func to kgen.generator.
static LogicalResult lowerLITFunc(LITFuncOp gen, SymbolTable &symbolTable) {
  lowerLITVarDecl(gen);
  lowerLITStructGEP(gen);
  OpBuilder b(gen);

  // Is a LITFuncOp with empy body representing an interface?
  if (gen.getIsInterface()) {
    auto result = b.create<GeneratorInterfaceOp>(
        gen.getLoc(), gen.getSymNameAttr(), gen.getFunctionTypeAttr(),
        gen.getParamDeclsAttr(), gen.getResultParamTypesAttr(),
        gen.getConstraintsAttr(), nullptr, nullptr);
    // Move over the symbol.
    symbolTable.erase(gen);
    symbolTable.insert(result);
    return success();
  }
  // Directly lower since these operations are exactly identical right now.
  auto result = b.create<GeneratorOp>(
      gen.getLoc(), gen.getSymNameAttr(), gen.getFunctionTypeAttr(),
      gen.getParamDeclsAttr(), gen.getResultParamTypesAttr(),
      gen.getConstraintsAttr(), gen.getImplementsAttr());

  // Move over the body.
  auto *bodyBlock = gen.getBody();
  gen.getBodyRegion().getBlocks().remove(bodyBlock);
  result.getBodyRegion().push_back(bodyBlock);

  // Move over the symbol.
  symbolTable.erase(gen);
  gen = LITFuncOp(); // The line above also erases 'gen'.
  symbolTable.insert(result);

  // If the generator implemented an interface, infer additional constraints and
  // check the signature.
  GeneratorInterfaceOp itf;
  if (auto interfaceName = result.getImplements()) {
    if (!interfaceName)
      return success();

    // Check that the callee attribute was specified.
    itf = dyn_cast_if_present<GeneratorInterfaceOp>(
        symbolTable.lookup(interfaceName.value()));
    if (!itf)
      return gen.emitError("could not find implemented interface");
  }

  return checkInterfaceConformance(result, itf, symbolTable);
}

/// Lower nested structures in kgen.struct.decl away.
static LogicalResult lowerStructDecl(StructDeclOp structDecl,
                                     SymbolTable &symbolTable) {
  SmallVector<VarDeclOp> opsToErase;
  for (Operation &member : llvm::make_early_inc_range(
           structDecl.getFields().front().getOperations())) {
    if (isa<KGEN::StructFieldOp>(member))
      continue; // Already lowered field.

    if (auto varDecl = dyn_cast<KGEN::VarDeclOp>(member)) {
      Type elemType = ParamRefType::get(varDecl.getType().getElementType());
      OpBuilder b(&member);
      b.create<KGEN::StructFieldOp>(member.getLoc(), varDecl.getName(),
                                    elemType);
      varDecl->erase();
      continue;
    }

    auto func = dyn_cast<KGEN::LITFuncOp>(member);
    if (!func)
      return member.emitError("unsupported op in lit lowering");
    // Move and rename the function from a field position inside the struct
    // to freestanding global function.
    func->remove();
    auto genOpNewName = StringAttr::get(
        structDecl.getContext(), Twine(structDecl.getSymName()) +
                                     "::" + func.getSymNameAttr().getValue());
    func.setSymName(genOpNewName);
    symbolTable.insert(func, Block::iterator(structDecl));

    // Prepend the parameters from the struct decl.
    SmallVector<ParamDeclAttr> paramDecls;
    paramDecls.reserve(structDecl.getParamDecls().size() +
                       func.getParamDecls().size());
    llvm::append_range(paramDecls, structDecl.getParamDecls());
    llvm::append_range(paramDecls, func.getParamDecls());
    func.setParamDecls(paramDecls);

    // Lower renamed function as usual.
    if (failed(lowerLITFunc(func, symbolTable)))
      return failure();
  }
  return success();
}

/// Member functions are reference with nested symbol references. After
/// lowering, the symbol tree will be flat. Concatenate all nested symbol
/// references in symbol constants.
static void renameSymbolReferences(Operation *op) {
  auto trimSymbolConstant = [](SymbolConstantAttr attr) -> Attribute {
    // If this attribute is already flat, there is nothing to do.
    if (isa<FlatSymbolRefAttr>(attr.getSymbol()))
      return attr;

    SymbolRefAttr symRef = attr.getSymbol();
    SmallString<64> qualifiedName(symRef.getRootReference().getValue().str());
    for (FlatSymbolRefAttr symRefAttr : symRef.getNestedReferences()) {
      qualifiedName.append("::");
      qualifiedName.append(symRefAttr.getValue());
    }
    return SymbolConstantAttr::get(
        FlatSymbolRefAttr::get(attr.getContext(), qualifiedName),
        attr.getParamValues(), attr.getType());
  };
  auto trimInType = [&](Type type) {
    if (auto itf = dyn_cast<mlir::SubElementTypeInterface>(type))
      return itf.replaceSubElements(trimSymbolConstant);
    return type;
  };

  op->walk([&](Operation *op) {
    // Search in operation attributes, result types, and argument types.
    op->setAttrs(cast<DictionaryAttr>(
        op->getAttrDictionary().replaceSubElements(trimSymbolConstant)));
    for (OpResult result : op->getOpResults())
      result.setType(trimInType(result.getType()));
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          arg.setType(trimInType(arg.getType()));
  });
}

namespace {
// Convenience wrapper instead of a std::pair.
struct NameAndType {
  StringAttr name;
  Type type;
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass boilerplate.
//===----------------------------------------------------------------------===//

namespace {

struct LowerLITPass : public impl::LowerLITBase<LowerLITPass> {
  void runOnOperation() override {
    // TODO: This has to be a module pass because this mutates the body of the
    // module, but we could trivially parallelize this within the pass.
    ModuleOp module = getOperation();
    SymbolTable symbolTable(module);
    for (auto &op : llvm::make_early_inc_range(module.getOps())) {
      if (auto func = dyn_cast<KGEN::LITFuncOp>(op)) {
        if (failed(lowerLITFunc(func, symbolTable)))
          return signalPassFailure();
      } else if (auto structDecl = dyn_cast<KGEN::StructDeclOp>(op)) {
        if (failed(lowerStructDecl(structDecl, symbolTable)))
          return signalPassFailure();
      }
    }
    renameSymbolReferences(module);
  }
};

} // namespace
