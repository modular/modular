//===- LowerHLKGEN.cpp ----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENPasses.h"

#include "ConstraintSet.h"
#include "KGEN/HLKGENDialect/HLKGENOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/MetaDialect/MetaOps.h"
#include "KGEN/MetaDialect/MetaTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;

namespace {
class SignatureUnifier {
public:
  SignatureUnifier(GeneratorOp generatorOp, GeneratorInterfaceOp interfaceOp);

  /// Add the constraints already on the generator to the constraint set,
  /// returning failure if a contradiction was detected.
  LogicalResult checkExistingConstraints();

  ParseResult addEqualityConstraintFn(ParamDeclRefAttr param, TypedAttr value);

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
  // The hlkgen.generator may have additional input parameters that are
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
                                     param.getName().str() +
                                     "' = " + getParamAsString(value));
  auto constraintValue =
      PointwiseValue::getSingleValue(value, message, inferenceLoc);
  return constraints.addPointwiseParamConstraint(param, constraintValue);
}

ParseResult SignatureUnifier::tryUnifyingTypeParameters(Attribute itfParam,
                                                        Attribute genParam) {
  // If these attributes are (recursively) identical, then they match.
  if (itfParam == genParam)
    return success();

  // If the interface requires something but the generator is ? then the
  // generator is more flexible than it needs to be.
  if (!genParam)
    return success();

  // If the interface is ? but the generator is more specific, then we cannot
  // support this: we cannot impose a constraint on a ?.
  if (!itfParam) {
    // TODO: It is possible to add inferred dynamic constraints when we have an
    // error handling model.
    auto diag = emitError(inferenceLoc, inferenceContext)
                << ": dynamic `?` value cannot have static constraint: '"
                << genParam << "'";
    diag.attachNote(interfaceOp->getLoc()) << "interface declared here";
    return failure();
  }

  // If one of these is a parameter, and one is concrete, then that infers a
  // value for the parameter.
  if (auto decl = itfParam.dyn_cast<ParamDeclRefAttr>())
    return addEqualityConstraintFn(decl, genParam);

  // Otherwise we don't know how to unify this.
  // TODO: Could handle node-wise merging of expressions to find constraints
  // like "x+1" and "y+1" --> "x == y".

  // TODO: It is possible to add inferred dynamic constraints when we have an
  // error handling model.
  auto diag = emitError(inferenceLoc, inferenceContext)
              << ": cannot unify : '" << genParam << "'";
  diag.attachNote(interfaceOp->getLoc()) << "interface declared here";
  return failure();
}

/// Check to see if the specified types can be merged, where the 'itfArgTy' is
/// the argument type from the interface and 'genArgTy' is the actual argument
/// from the generator.  On failure, this generates a failure but does not emit
/// an error message.
ParseResult SignatureUnifier::tryUnifyingTypes(Type itfArgTy, Type genArgTy) {
  // If the types are identical then of course they match.
  if (itfArgTy == genArgTy)
    return success();

  // If both types are scalar types, try to unify them.
  if (auto itfScalarTy = itfArgTy.dyn_cast<ScalarType>())
    if (auto genScalarTy = genArgTy.dyn_cast<ScalarType>())
      return tryUnifyingTypeParameters(itfScalarTy.getDType(),
                                       genScalarTy.getDType());

  // If both types are PointerType's, try to unify them.
  if (auto itfPointerTy = itfArgTy.dyn_cast<PointerType>())
    if (auto genPointerTy = genArgTy.dyn_cast<PointerType>())
      return tryUnifyingTypeParameters(itfPointerTy.getElementType(),
                                       genPointerTy.getElementType());

  // If both types are SIMDType's, try to unify them.
  if (auto itfSIMDTy = itfArgTy.dyn_cast<SIMDType>())
    if (auto genSIMDTy = genArgTy.dyn_cast<SIMDType>()) {
      return failure(
          tryUnifyingTypeParameters(itfSIMDTy.getDType(),
                                    genSIMDTy.getDType()) ||
          tryUnifyingTypeParameters(itfSIMDTy.getSize(), genSIMDTy.getSize()));
    }

  // If both types are BufferType's, try to unify them.
  if (auto itfBufferTy = itfArgTy.dyn_cast<BufferType>())
    if (auto genBufferTy = genArgTy.dyn_cast<BufferType>()) {
      return failure(tryUnifyingTypeParameters(itfBufferTy.getDType(),
                                               genBufferTy.getDType()) ||
                     tryUnifyingTypeParameters(itfBufferTy.getSize(),
                                               genBufferTy.getSize()));
    }

  // If they don't match, then reject them.
  auto diag = emitError(inferenceLoc, inferenceContext)
              << " has type " << genArgTy << " but interface expected type "
              << itfArgTy;
  diag.attachNote(interfaceOp->getLoc()) << "interface declared here";
  return failure();
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

  // This only needs to handle the types we're capable of unifying.
  if (type.isa<ScalarType>())
    return b.create<ScalarRebindOp>(type, arg);
  if (type.isa<PointerType>())
    return b.create<PointerRebindOp>(type, arg);
  if (type.isa<SIMDType>())
    return b.create<SIMDRebindOp>(type, arg);
  if (type.isa<BufferType>())
    return b.create<BufferRebindOp>(type, arg);

  llvm_unreachable("unknown type to unify");
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
        gen.getSymVisibilityAttr(),
        // Take the signature from the interface.
        itf.getFunctionTypeAttr(), itf.getParamDeclsAttr(),
        itf.getResultParamDeclsAttr(),
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
         llvm::zip(body->getArguments(), gen.getBodyBlock()->getArguments())) {
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
    SmallVector<ParamBindAttr> returnParams;
    for (ParamDeclAttr resultParam : gen.getResultParamDecls()) {
      // The call returns the same thing as the generator.
      callResultParams.push_back(resultParam);
      // The output binds each result from the call into the return value of the
      // generator thunk.

      auto value =
          ParamDeclRefAttr::get(resultParam.getName(), resultParam.getType());
      returnParams.push_back(ParamBindAttr::get(resultParam.getName(), value));
    }

    // Create the call.
    b.setLoc(gen.getLoc());
    auto callOp =
        b.create<CallOp>(gen.getResultTypes(), gen.getNameAttr(),
                         callInputParams, callResultParams, castedArgs);

    // Create any rebind's for the results.
    SmallVector<Value> results;
    for (auto [result, resultTy] : llvm::zip(callOp.getResults(), itfResTys))
      results.push_back(insertRebindOp(result, resultTy, b));

    b.create<ReturnOp>(b.getAttr<ParamBindArrayAttr>(returnParams), results);

    // The thunk is required because there could be direct callers of the
    // original generator, which expect the original signature.  If there
    // aren't, then we can just inline it away.
    // TODO: Inline these away if/when they have no additional callers.
  }

  return success();
}

/// Lower an hlkgen.generator to kgen.generator.
static LogicalResult lowerHLGenerator(HLGeneratorOp gen,
                                      SymbolTable &symbolTable) {
  OpBuilder b(gen);

  // Directly lower since these operations are exactly identical right now.
  auto result = b.create<GeneratorOp>(
      gen.getLoc(), gen.getSymNameAttr(), gen.getSymVisibilityAttr(),
      gen.getFunctionTypeAttr(), gen.getParamDeclsAttr(),
      gen.getResultParamDeclsAttr(), gen.getConstraintsAttr(),
      gen.getImplementsAttr());

  // Move over the body unmodified.
  auto *bodyBlock = gen.getBodyBlock();
  gen.getBody().getBlocks().remove(bodyBlock);
  result.getBody().push_back(bodyBlock);

  // Move over the symbol.
  symbolTable.erase(gen);
  gen = HLGeneratorOp(); // The line above also erases 'gen'.
  symbolTable.insert(result);

  // If the generator implemented an interface, infer additional constraints and
  // check the signature.
  GeneratorInterfaceOp itf;
  if (auto interfaceName = result.getImplements()) {
    if (!interfaceName)
      return success();

    // Check that the callee attribute was specified.
    itf = dyn_cast_or_null<GeneratorInterfaceOp>(
        symbolTable.lookup(interfaceName.value()));
    if (!itf)
      return gen.emitError("could not find implemented interface");
  }

  return checkInterfaceConformance(result, itf, symbolTable);
}

//===----------------------------------------------------------------------===//
// Pass boilerplate.
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_CLASSES
#include "KGEN/KGENPasses.h.inc"

class LowerHLKGENPass : public LowerHLKGENBase<LowerHLKGENPass> {
public:
  void runOnOperation() override {
    // TODO: This has to be a module pass because this mutates the body of the
    // module, but we could trivially parallelize this within the pass.
    ModuleOp module = getOperation();
    SymbolTable symbolTable(module);
    for (auto hlGenerator :
         llvm::make_early_inc_range(module.getOps<KGEN::HLGeneratorOp>())) {
      if (failed(lowerHLGenerator(hlGenerator, symbolTable)))
        signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> M::KGEN::createLowerHLKGENPass() {
  return std::make_unique<LowerHLKGENPass>();
}
