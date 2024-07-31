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
#include "KGEN/TransformUtils/ManglingUtils.h"
#include "Support/Compiler/DiagnosticHandler.h"
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
  FuncOp func = elaborator->lookupConcreteFunction(symbol);

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
    return ErrorTree(*errorLoc, "failed to compile-time evaluate function call",
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

  // True if InitSelf, false if ByRefResult.
  bool isInitSelf = func.getSignature().hasInitSelfArg();
  auto resultArg =
      isInitSelf ? func.getArgument(0) : func.getArguments().back();
  auto ptr = dyn_cast<PointerType>(resultArg.getType());
  if (!ptr)
    return ErrorTree(func.getLoc(), "result argument is not a pointer");
  ErrorTreeOr<TypedAttr> result = executeRegionWithResultSlot(
      func.getBodyRegion(), arguments, isInitSelf, ptr.getElementType());

  // Report an error if evaluation fails.
  if (result.isError()) {
    return ErrorTree(*errorLoc, "failed to compile-time evaluate function call",
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
  case POC::CompileAssembly:
    return evaluateCompileAssembly(op);
  case POC::GetLinkageName:
    return evaluateGetLinkageName(op);
  case POC::LoadFromMem:
    if (auto memref = dyn_cast<MemRefAttr>(op.getOperands().front())) {
      ErrorOr<TypedAttr> value = loadAttributeFromMemRef(memref, op.getType());
      if (value.isError()) {
        emitError({*errorLoc, value.takeError()});
        return failure();
      }
      return value.takeValue();
    }
    return failure();
  default:
    return failure();
  }
}

FailureOr<TypedAttr> IREvaluator::evaluateApplyLike(ParamOperatorAttr op,
                                                    bool withResultSlot) {
  // Attempt to concretize the function first.
  ErrorTreeOr<FuncOp> funcOr = elaborator->getConcreteFunction(
      parent, *errorLoc, cast<SymbolConstantAttr>(op.getOperands().front()));
  if (funcOr.isError()) {
    emitError(funcOr.takeError());
    return failure();
  }
  FuncOp func = funcOr.takeValue();
  if (!func)
    return TypedAttr();

  // Attempt to lookup a cached value. This returns a thread local cached value.
  auto operandsAttr = ParameterExprArrayAttr::get(
      op.getContext(), op.getOperands().drop_front());
  TypedAttr &cached =
      elaborator->lookupCachedInterpretation(func, operandsAttr);
  if (cached)
    return cached;

  // Concretize symbols within the operands before invoking the function.
  ErrorTreeOr<Attribute> operandsOr =
      elaborator->concretizeSymbolsWithin(operandsAttr, parent, *errorLoc);
  if (operandsOr.isError()) {
    emitError(operandsOr.takeError());
    return failure();
  }
  operandsAttr = cast_or_null<ParameterExprArrayAttr>(operandsOr.takeValue());
  if (!operandsAttr)
    return TypedAttr();

  // Now invoke the interpreter.
  ErrorTreeOr<TypedAttr> result =
      withResultSlot ? evaluateFunctionWithResultSlot(func, operandsAttr)
                     : evaluateFunction(func, operandsAttr);

  // If we had a value, write it back.
  if ((cached = result.tryGetValue())) {
    elaborator->writeGlobalCachedInterpretation(func, operandsAttr, cached);
    return cached;
  }
  emitError(result.takeError());
  return TypedAttr();
}

FailureOr<TypedAttr> IREvaluator::evaluateGetEnv(ParamOperatorAttr op) {
  // Grab the module from the elaborator. This is a read operation of memory
  // that is not modified during elaboration, so no synchronization is needed.
  EnvAttr env = elaborator->getCompilationEnvAttr();
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

static void emitDiagnosticToStream(raw_ostream &os, Diagnostic &diag) {
  os << "\n" << diag.getLocation() << ": " << diag;
  for (Diagnostic &note : diag.getNotes())
    emitDiagnosticToStream(os, note);
}

FailureOr<TypedAttr>
IREvaluator::evaluateCompileAssembly(ParamOperatorAttr op) {
  // Cheeky copy. The state of the symbol table right at this moment is
  // sufficient to produce a standalone object for the generator being JIT'd.
  SymbolTable symtabCopy = elaborator->getSliceSymTab();

  // Slice out a stanalone module to re-elaborate with the new target.
  TargetInfoAttr target = cast<TargetParamAttr>(op.getOperand(0)).getTarget();
  auto emissionKind =
      (EmissionKind)cast<IntegerAttr>(op.getOperand(1)).getInt();
  bool propagateError = cast<IntegerAttr>(op.getOperand(2)).getInt();
  auto symbol = dyn_cast<SymbolConstantAttr>(op.getOperand(3));
  if (!symbol || !symbol.getType().isConcrete()) {
    emitError({*errorLoc, "'compile_assembly' function is not concrete"});
    return failure();
  }

  auto func = symtabCopy.lookup<GeneratorOp>(
      cast<FlatSymbolRefAttr>(symbol.getSymbol()).getAttr());
  assert(func && "expected a valid generator reference");

  // Construct the expected result type.
  MLIRContext *ctx = op.getContext();
  Builder b(op.getContext());
  auto noneType = KGEN::NoneType::get(ctx);
  auto populateFnType = SignatureType::get(
      b.getFunctionType(PointerType::get(noneType), noneType), {}, {},
      {ArgConvention::BorrowedInReg}, FnEffects().setCapturing());

  // Specialize the generator with another target by slicing it and its
  // transitive dependencies out of the IR and re-invoking the elaborator. If it
  // turns out that the specialization has more than one implementation, then
  // the elaborator invocation will fail due to multiple implementations of a
  // primary generator, and the functor will return an error.
  StringAttr name =
      getExpectedMangledName(func, symbol.getParamValues(), /*sanitize=*/false);

  // Capture the diagnostics that may be emitted.
  DiagnosticHandler handler(ctx);
  ErrorOr<CrossDeviceFunction> closure = elaborator->getCompileAsmFn()(
      func, symbol, name, symtabCopy, target, emissionKind);
  handler.release();

  if (closure.isError()) {
    // Emit all the errors now.
    if (!propagateError) {
      handler.emitDiagnostics([&](Diagnostic &diag) {
        ctx->getDiagEngine().emit(std::move(diag));
      });
      emitError({*errorLoc, closure.takeError()});
      return failure();
    }
    // Concat all the errors together and return it as a variant.
    std::string error;
    llvm::raw_string_ostream os(error);
    os << closure.getError();
    handler.emitDiagnostics(
        [&](Diagnostic &diag) { emitDiagnosticToStream(os, diag); });
    // Note: return -1 to indicate an error state.
    return {StructAttr::get({StringAttr::get(os.str(), StringType::get(ctx)),
                             b.getIndexAttr(-1),
                             UnknownAttr::get(populateFnType)})};
  }

  auto populate = cast<FuncOp>(closure->populateCapturesFn.release());
  auto populateFnRef = SymbolConstantAttr::get(
      FlatSymbolRefAttr::get(populate.getSymNameAttr()), populateFnType);
  elaborator->addDeferredFunction(populate);
  return {
      StructAttr::get({closure->contents, b.getIndexAttr(closure->numCaptures),
                       populateFnRef})};
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
  auto genOp = elaborator->getSliceSymTab().lookup<GeneratorOp>(
      symbol.getSymbol().getRootReference());
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

IREvaluator::IREvaluator(Elaborator &elaborator, ImplNode *parent)
    : InterpreterState(elaborator.getTarget()), elaborator(&elaborator),
      parent(parent) {}

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
