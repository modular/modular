//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "IREvaluator.h"
#include "Elaborator.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/Support/NameMangling.h"
#include "Support/MDialect/MTypeInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/DebugStringHelper.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// IR Interpreter
//===----------------------------------------------------------------------===//

ErrorOr<Region *> IREvaluator::lookupFunctionBody(SymbolRefAttr symbol) {
  StringAttr name = cast<FlatSymbolRefAttr>(symbol).getAttr();
  FuncOp func =
      elaborator->getSymbolTable().read([name](const SymbolTable &symtab) {
        return symtab.lookup<FuncOp>(name);
      });

  // Now we can return the function body.
  return &func.getBodyRegion();
}

ErrorTreeOr<TypedAttr>
IREvaluator::evaluateFunction(FuncOp func, ArrayRef<TypedAttr> inputs) {
  // Evaluate the function body.
  SmallVector<Attribute> arguments;
  for (TypedAttr input : inputs)
    arguments.push_back(input);
  ErrorTreeOr<SmallVector<Attribute>> result =
      executeRegion(func.getBodyRegion(), arguments);

  // Report an error if evaluation fails.
  if (result.isError()) {
    return ErrorTree(*errorLoc, "failed to evaluate 'apply'",
                     result.takeError());
  }

  // Apply operators only return one result.
  return cast<TypedAttr>(result.getValue().front());
}

ErrorTreeOr<TypedAttr>
IREvaluator::evaluateFunctionWithResultSlot(FuncOp func,
                                            ArrayRef<TypedAttr> inputs) {
  // Evaluate the function body.
  SmallVector<Attribute> arguments;
  for (TypedAttr input : inputs)
    arguments.push_back(input);
  auto ptr = dyn_cast<PointerType>(func.getArgument(0).getType());
  if (!ptr)
    return ErrorTree(func.getLoc(), "first argument is not a pointer");
  ErrorTreeOr<TypedAttr> result = executeRegionWithResultSlot(
      ptr.getElementType(), func.getBodyRegion(), arguments);

  // Report an error if evaluation fails.
  if (result.isError()) {
    return ErrorTree(*errorLoc, "failed to evaluate 'apply'",
                     result.takeError());
  }

  // Apply operators only return one result.
  return result.takeValue();
}

//===----------------------------------------------------------------------===//
// Expression Evaluation
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> IREvaluator::evaluateExpression(ParamOperatorAttr op) {
  // Don't try to evaluate a parameter operator that still contains parametric
  // things in it, since it may be transitory.
  struct IndexRefFinder : IndexParameterReplacer<IndexRefFinder> {
    Attribute tryReplace(Attribute attr, size_t depth) {
      if (auto ref = dyn_cast<ParamIndexRefAttr>(attr)) {
        if (ref.getDepth() >= depth) {
          escapingReference = true;
          return attr;
        }
      }
      return nullptr;
    }
    Type tryReplace(Type, size_t) { return nullptr; }

    bool escapingReference = false;
  } finder;
  finder.replace(op);
  if (finder.escapingReference)
    return {op};

  // Try to narrow this operator to an expression we can evaluate. We only need
  // to emit an error during the evaluation attempt.
  switch (op.getOpcode()) {
  case POC::CurrentTarget:
    // Retrieve the contextual compilation target info.
    return {TargetParamAttr::get(elaborator->getTarget())};
  case POC::GetEnv:
    return evaluateGetEnv(op);
  case POC::Apply:
    return evaluateApplyLike(op, /*withResultSlot=*/false);
  case POC::ApplyResultSlot:
    return evaluateApplyLike(op, /*withResultSlot=*/true);
  case POC::Rebind:
    // Catch unfolded rebinds to emit a nicer error message.
    emitError(ErrorTree(
        *errorLoc, "error: rebind input type '" +
                       mlir::debugString(op.getOperands().front().getType()) +
                       "' does not match result type '" +
                       mlir::debugString(op.getType()) + "'"));
    return failure();
  case POC::GetAllImpls:
    return evaluateGetAllImpls(op);
  case POC::CompileAssembly:
    return evaluateCompileAssembly(op);
  case POC::GetLinkageName:
    return evaluateGetLinkageName(op);
  default:
    return failure();
  }
}

FailureOr<TypedAttr> IREvaluator::evaluateGetAllImpls(ParamOperatorAttr op) {
  auto symbol = dyn_cast<SymbolConstantAttr>(op.getOperand(0));
  if (!symbol)
    return failure();
  std::vector<FuncOp> funcs;
  std::optional<ErrorTreeOrSuccess> err = elaborator->getAllConcreteFunctions(
      parent, *errorLoc, cast<FlatSymbolRefAttr>(symbol.getSymbol()),
      symbol.getParamValues(), funcs);
  if (!err)
    return TypedAttr();
  if (err->isError()) {
    emitError(err->takeError());
    return TypedAttr();
  }

  std::vector<TypedAttr> refs;
  refs.reserve(funcs.size());
  for (FuncOp f : funcs)
    refs.emplace_back(SymbolConstantAttr::get(
        SymbolRefAttr::get(f.getSymNameAttr()), f.getSignature()));

  return {VariadicAttr::get(refs, cast<VariadicType>(op.getType()))};
}

ErrorTreeOr<SymbolConstantAttr>
IREvaluator::concretizeFunctionSymbol(TypedAttr symbolMaybe,
                                      Location location) {
  ErrorTreeOr<Attribute> symMaybe =
      concretizeParameterExpr(parent, location, symbolMaybe);
  if (symMaybe.isError())
    return symMaybe.takeError();
  SymbolConstantAttr symbol = dyn_cast<SymbolConstantAttr>(symMaybe.getValue());
  if (!symbol)
    return symbol;
  if (!symbol.getParamValues().empty()) {
    auto ref = dyn_cast<FlatSymbolRefAttr>(symbol.getSymbol());
    std::optional<ErrorTreeOr<FuncOp>> maybeFuncOrError =
        elaborator->getConcreteFunction(parent, location, ref,
                                        symbol.getParamValues());
    if (!maybeFuncOrError.has_value())
      return SymbolConstantAttr();
    if (maybeFuncOrError->isError())
      return maybeFuncOrError->takeError();

    FuncOp newCalleeFunc = maybeFuncOrError->getValue();
    return SymbolConstantAttr::get(
        FlatSymbolRefAttr::get(newCalleeFunc.getNameAttr()),
        newCalleeFunc.getSignature());
  }
  return symbol;
}

FailureOr<TypedAttr> IREvaluator::evaluateApplyLike(ParamOperatorAttr op,
                                                    bool withResultSlot) {
  auto symbol = dyn_cast<SymbolConstantAttr>(op.getOperand(0));
  if (!symbol || !symbol.getType().getResultParamTypes().empty())
    return failure();
  ArrayRef<TypedAttr> operands = op.getOperands().drop_front();
  auto ref = dyn_cast<FlatSymbolRefAttr>(symbol.getSymbol());
  if (!llvm::all_of(operands, ParameterAttr::isSimpleConstant) || !ref)
    return failure();

  // Lookup the symbol reference and resolve it.
  std::optional<ErrorTreeOr<FuncOp>> func = elaborator->getConcreteFunction(
      parent, *errorLoc, ref, symbol.getParamValues());
  if (!func)
    return TypedAttr();
  if (func->isError()) {
    emitError(func->takeError());
    return TypedAttr();
  }

  ErrorTreeOr<TypedAttr> result =
      withResultSlot ? evaluateFunctionWithResultSlot(**func, operands)
                     : evaluateFunction(**func, operands);
  if (TypedAttr value = result.tryGetValue())
    return value;
  emitError(result.takeError());
  return TypedAttr();
}

FailureOr<TypedAttr> IREvaluator::evaluateGetEnv(ParamOperatorAttr op) {
  // Grab the module from the elaborator. This is a read operation of memory
  // that is not modified during elaboration, so no synchronization is needed.
  auto module = cast<ModuleOp>(elaborator->getSymbolTable().get().getOp());
  auto env = module->getAttrOfType<EnvAttr>(EnvAttr::getEnvAttrName());
  assert(env && "expected an environment attribute on the module");

  auto name = dyn_cast<StringAttr>(op.getOperands().front());
  if (!name) {
    emitError({*errorLoc, "'get_env' name did not narrow to a constant"});
    return failure();
  }

  // Get the `StringRef` out of the `StringAttr` because the latter comes with
  // a `StringType` type that makes pointer comparisons fails.
  Attribute value = env.getValues().get(name.getValue());
  if (isa<IndexType, StringType>(op.getType()) && !value) {
    emitError({*errorLoc, "define '" + name.getValue() +
                              "' does not exist, please provide it via -D"});
    return failure();
  }

  if (isa<IndexType>(op.getType())) {
    if (auto intVal = dyn_cast<IntegerAttr>(value))
      return {intVal};
    emitError({*errorLoc, "define '" + name.getValue() +
                              "' is not an integer, got " +
                              mlir::debugString(value)});
    return failure();
  }

  if (isa<StringType>(op.getType())) {
    if (auto strVal = dyn_cast<StringAttr>(value))
      return {strVal};
    emitError({*errorLoc, "define '" + name.getValue() +
                              "' is not a string, got " +
                              mlir::debugString(value)});
    return failure();
  }

  // This must be an `i1` type. Return true or false based on whether the
  // environment variable is present.
  assert(cast<IntegerType>(op.getType()).isSignlessInteger(1));
  return {BoolAttr::get(op.getContext(), static_cast<bool>(value))};
}

/// Compute the expected mangled name of a generator, assuming it has one
/// successful implementation. If it doesn't, elaboration will fail anyways.
static StringAttr getExpectedMangledName(GeneratorOp func,
                                         ArrayRef<TypedAttr> params,
                                         bool sanitize) {
  auto baseName =
      StringAttr::get(func.getContext(), mangleParameterValues(func, params));
  if (sanitize)
    baseName = sanitizeSymbolToAlnum(baseName);
  return baseName;
}

FailureOr<TypedAttr>
IREvaluator::evaluateCompileAssembly(ParamOperatorAttr op) {
  // Cheeky copy. The state of the symbol table right at this moment is
  // sufficient to produce a standalone object for the generator being JIT'd.
  SymbolTable symtabCopy = elaborator->getSymbolTable().read(
      [](const SymbolTable &symtab) -> SymbolTable { return symtab; });

  // Slice out a stanalone module to re-elaborate with the new target.
  TargetInfoAttr target = cast<TargetParamAttr>(op.getOperand(0)).getTarget();
  auto emissionKind =
      (EmissionKind)cast<IntegerAttr>(op.getOperand(1)).getInt();
  auto symbol = dyn_cast<SymbolConstantAttr>(op.getOperand(2));
  if (!symbol || !symbol.getType().isConcrete()) {
    emitError({*errorLoc, "'compile_assembly' function is not concrete"});
    return failure();
  }

  auto func = symtabCopy.lookup<GeneratorOp>(
      cast<FlatSymbolRefAttr>(symbol.getSymbol()).getAttr());
  assert(func && "expected a valid generator reference");

  // Specialize the generator with another target by slicing it and its
  // transitive dependencies out of the IR and re-invoking the elaborator. If it
  // turns out that the specialization has more than one implementation, then
  // the elaborator invocation will fail due to multiple implementations of a
  // primary generator, and the functor will return an error.
  StringAttr name =
      getExpectedMangledName(func, symbol.getParamValues(), /*sanitize=*/false);
  ErrorOr<CrossDeviceFunction> closure = elaborator->getCompileAsmFn()(
      func, symbol, name, symtabCopy, target, emissionKind);
  if (closure.isError()) {
    emitError({*errorLoc, closure.takeError()});
    return failure();
  }
  auto populate = cast<FuncOp>(closure->populateCapturesFn.release());
  elaborator->addDeferredFunction(populate);

  Builder b(op.getContext());
  SmallVector<TypedAttr> fieldValues{
      closure->contents, b.getIndexAttr(closure->numCaptures),
      SymbolConstantAttr::get(FlatSymbolRefAttr::get(populate.getSymNameAttr()),
                              populate.getSignature())};
  SmallVector<Type> fieldTypes =
      llvm::map_to_vector(fieldValues, [](TypedAttr v) { return v.getType(); });
  auto structType = StructType::get(b.getContext(), fieldTypes);
  return {StructAttr::get(fieldValues, structType)};
}

FailureOr<TypedAttr> IREvaluator::evaluateGetLinkageName(ParamOperatorAttr op) {
  // This only supports generators with an empty set of parameters, otherwise we
  // need to resolve the symbol name after elaboration.
  TargetInfoAttr target = cast<TargetParamAttr>(op.getOperand(0)).getTarget();
  auto symbol = dyn_cast<SymbolConstantAttr>(op.getOperand(1));
  if (!symbol || !symbol.getType().isConcrete()) {
    emitError({*errorLoc, "'get_linkage_name' function is not concrete"});
    return failure();
  }
  auto genOp = elaborator->getSymbolTable().read(
      [name = symbol.getSymbol().getRootReference()](const SymbolTable &symtab)
          -> GeneratorOp { return symtab.lookup<GeneratorOp>(name); });
  assert(genOp && "expected a valid generator reference");

  // HACK HACK HACK: Our current name mangling scheme is not compatible with the
  // NVPTX backend.
  StringAttr name = getExpectedMangledName(
      genOp, symbol.getParamValues(),
      llvm::is_contained({llvm::Triple::nvptx, llvm::Triple::nvptx64},
                         target.getTriple().getArch()));
  return {StringAttr::get(name.getValue(), op.getType())};
}

//===----------------------------------------------------------------------===//
// IREvaluator
//===----------------------------------------------------------------------===//

IREvaluator::IREvaluator(Elaborator &elaborator,
                         DenseMap<StringAttr, Attribute> paramValues)
    : ParameterEvaluator(std::move(paramValues)),
      InterpreterState(elaborator.getTarget()), elaborator(&elaborator),
      parent(nullptr) {}

/// Given a generic parameter expression, simplify it by folding the
/// expression according to known parameter values.  This returns an error if
/// the expression cannot be folded for one reason or another.
ErrorTreeOr<Attribute> IREvaluator::concretizeParameterExpr(ImplNode *parent,
                                                            Location loc,
                                                            Attribute expr) {
  // FIXME: Refactor ParameterEvaluator for better error propagation.
  this->parent = parent;
  errorLoc = loc;
  std::optional<ErrorTree> error;
  emitError = [&](ErrorTree err) { error = std::move(err); };

  Attribute result = getReboundAttribute(expr);
  if (error)
    return std::move(*error);

  if (!result)
    return Attribute();

  // If we can fold this to a simple constant result, do.
  if (ParameterAttr::isSimpleConstant(result))
    return result;

  // If this was an unfoldable operator expression, error.  This can happen for
  // things like 'index' arithmetic that has target-specific results.
  if (auto oper = dyn_cast<ParamOperatorAttr>(result))
    return ErrorTree(loc,
                     "could not simplify operator " + getParamAsString(result));
  return result;
}

ErrorTreeOr<Type> IREvaluator::concretizeParameterExpr(ImplNode *parent,
                                                       Location loc,
                                                       Type expr) {
  // FIXME: Refactor ParameterEvaluator for better error propagation.
  this->parent = parent;
  errorLoc = loc;
  std::optional<ErrorTree> error;
  emitError = [&](ErrorTree err) { error = std::move(err); };

  Type result = getReboundType(expr);
  if (error)
    return std::move(*error);
  return result;
}

//===----------------------------------------------------------------------===//
// evaluateConstraints implementation.
//===----------------------------------------------------------------------===//

/// Given a generator or interface declaration operation, evaluate any
/// constraints against inputParamValues.  If the constraints are met, return
/// success, otherwise return why they aren't.
std::optional<ErrorTreeOrSuccess>
KGEN::evaluateConstraints(ImplNode *parent,
                          ArrayRef<ConstraintAttr> constraints,
                          IREvaluator &evaluator) {
  // Each constraint must be foldable, and must fold to true.
  for (ConstraintAttr constraint : constraints) {
    Location loc = constraint.getLoc();
    ErrorTreeOr<Attribute> result =
        evaluator.concretizeParameterExpr(parent, loc, constraint.getExpr());
    if (result.isError())
      return ErrorTree(loc, "constraint evaluation failure",
                       result.takeError());
    if (!*result)
      return std::nullopt;

    auto resultInt = dyn_cast<IntegerAttr>(result.takeValue());
    if (!resultInt || resultInt.getValue().getBitWidth() != 1)
      return ErrorTree(loc,
                       "constraint evaluation didn't return true or false");

    // If this failed, indicate why.
    if (resultInt.getValue().isZero())
      return ErrorTree(loc, "constraint failed: " +
                                constraint.getMessage().getValue());
  }

  // If we made it this far, then everything folded to true.
  return success();
}

void IREvaluator::setParent(ImplNode *impl) { parent = impl; }

ImplNode::ImplNode(FuncOp func, ParamNode *parent, ParameterUseDefGraph &&graph,
                   std::string &&baseName, IREvaluator evaluator)
    : func(func), parent(parent), paramGraph(std::move(graph)),
      baseName(std::move(baseName)), evaluator(std::move(evaluator)) {
  this->evaluator.setParent(this);
}
