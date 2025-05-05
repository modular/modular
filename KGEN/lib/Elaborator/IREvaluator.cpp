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
#include "KGEN/TransformUtils/ManglingUtils.h"
#include "Support/Compiler/DiagnosticHandler.h"
#include "Support/MDialect/MTypeInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/ScopeExit.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// IR Interpreter
//===----------------------------------------------------------------------===//

ErrorOr<Region *> IREvaluator::lookupFunctionBody(SymbolRefAttr symbol) {
  InstantiatedOpInterface inst = elaborator->lookupInstantiatedOp(symbol);
  FuncOp func = cast<FuncOp>(inst);

  // Now we can return the function body.
  return &func.getBodyRegion();
}

ErrorTreeOr<TypedAttr>
IREvaluator::evaluateFunction(FuncOp func, ArrayRef<TypedAttr> inputs) {
  if constexpr (KGEN::kIsTracingEnabled)
    auto ts = InterpreterTimeTraceScope("Launch interpreter",
                                        mlir::debugString(errorLoc));

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
  if constexpr (KGEN::kIsTracingEnabled)
    auto ts = InterpreterTimeTraceScope("Launch interpreter",
                                        mlir::debugString(errorLoc));

  // Evaluate the function body.
  SmallVector<Attribute> arguments;
  for (TypedAttr input : inputs)
    arguments.push_back(input);

  auto resultArg = func.getArguments().back();
  auto ptr = dyn_cast<PointerType>(resultArg.getType());
  if (!ptr)
    return ErrorTree(func.getLoc(), "result argument is not a pointer");
  ErrorTreeOr<TypedAttr> result = executeRegionWithResultSlot(
      func.getBodyRegion(), arguments, ptr.getElementType());

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

FailureOr<TypedAttr>
IREvaluator::evaluateExpression(EvaluatableAttrInterface attr) {
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
  finder.replace(attr);
  if (finder.escapingReference)
    return cast<TypedAttr>(attr);

  if (auto genref = dyn_cast<TypeGeneratorRefAttr>(attr)) {
    // Attempt to concretize the function first.
    ErrorTreeOr<TypeInstanceRefAttr> symOr =
        elaborator->getConcreteStructTypeReference(parent, *errorLoc, genref);
    if (symOr.isError()) {
      emitError(symOr.takeError());
      return failure();
    }
    return cast<TypedAttr>(symOr.takeValue());
  }

  if (auto getWitnessEntry = dyn_cast<GetWitnessAttr>(attr)) {
    // Find the node for the instantiated type ref.
    SymbolRefAttr instanceRef =
        getWitnessEntry.getTypeInstanceRef().getSymbol();
    ParamNode *genNode = elaborator->lookupImplNode(instanceRef)->parent;
    // Always look up witness tables from the StructGeneratorOp, since the
    // StructInstanceOp is undergoing elaboration, and we should not block on
    // the instance's completion.
    StructGeneratorOp gen = cast<StructGeneratorOp>(genNode->gen);
    SymbolTable symtab(gen);
    ConformanceOp witnessTable =
        symtab.lookup<ConformanceOp>(getWitnessEntry.getTraitName());
    if (!witnessTable) {
      emitError({*errorLoc, "instantiated struct type " +
                                mlir::debugString(instanceRef) +
                                " does not have witness table for trait " +
                                getWitnessEntry.getTraitName().getValue()});
      return failure();
    }

    IREvaluator nestedEvaluator(*elaborator, parent);
    nestedEvaluator.setErrorLoc(*errorLoc);
    // If the struct generator has input params, we need to provide an
    // IREvaluator for concretizing the witness entries.
    for (auto [param, value] :
         llvm::zip(gen.getInputParams(), genNode->inputParams))
      nestedEvaluator.setParameterValue(param, value);
    FailureOr<TypedAttr> simplified =
        getWitnessEntry.simplify(witnessTable, &nestedEvaluator);
    if (failed(simplified)) {
      emitError(
          {*errorLoc, "failed to locate witness entry for " + gen.getSymName() +
                          ", " + getWitnessEntry.getTraitName().getValue() +
                          ", " + getWitnessEntry.getWitnessName().getValue()});
      return failure();
    }
    return simplified;
  }

  // Must be a parameter operator then.
  auto op = dyn_cast<ParamOperatorAttr>(attr);
  assert(op && "unknown attribute with EvaluatableAttrInterface");

  // Try to narrow this operator to an expression we can evaluate. We only need
  // to emit an error during the evaluation attempt.
  switch (op.getOpcode()) {
  case POC::CurrentTarget:
    // Retrieve the contextual compilation target info.
    return {TargetParamAttr::get(elaborator->getTarget())};
  case POC::AcceleratorArch:
    return {StringAttr::get(elaborator->options.targetAccelerator,
                            StringType::get(op.getContext()))};
  case POC::CrossCompilation:
    return {
        BoolAttr::get(op.getContext(), elaborator->options.isCrossCompilation)};
  case POC::GetEnv:
    return evaluateGetEnv(op);
  case POC::Apply:
    return evaluateApplyLike(op, /*withResultSlot=*/false);
  case POC::ApplyResultSlot:
    return evaluateApplyLike(op, /*withResultSlot=*/true);
  case POC::AttrToStr:
    return {StringAttr::get(mlir::debugString(op.getOperands().front()),
                            StringType::get(op.getContext()))};
  case POC::DataToStr:
    return evaluateDataToStr(op);
  case POC::StringAddress:
    return evaluateStringAddress(op);
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
  case POC::CompileOffloadClosure:
    return evaluateCompileOffloadClosure(op);
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
    return elaborator->writeGlobalCachedInterpretation(func, operandsAttr,
                                                       cached);
  }
  emitError(result.takeError());
  return TypedAttr();
}

// See if we can decode the first 'numBytes' of the memory blob into a
// StringAttr.
static StringAttr getBytesOf(MemoryBlobAttr value, size_t numBytes) {
  // We don't bother handling these.
  if (!value.getPointerRegions().empty() || !value.getSymbolRegions().empty())
    return {};

  if (numBytes <= value.getHandle().getSize()) {
    return StringAttr::get(StringRef(value.getHandle().getData(), numBytes),
                           StringType::get(value.getContext()));
  }
  return {};
}

/// Extract a value of type `struct<(pointer<none>, index)>` into a StringAttr.
FailureOr<StringAttr> IREvaluator::evaluateStringPart(TypedAttr part) {
  // Get the two parts of the struct, StructExtract will fold.
  auto lengthAttr = dyn_cast<IntegerAttr>(StructExtractAttr::get(part, 1));
  if (!lengthAttr) {
    emitError({*errorLoc, "'data_to_str' length didn't resolve to a constant"});
    return failure();
  }
  size_t numBytes = lengthAttr.getInt();
  if (!numBytes)
    return {StringAttr::get("", StringType::get(getContext()))};

  MemRefAttr pointerAttr =
      dyn_cast<MemRefAttr>(StructExtractAttr::get(part, 0));
  if (!pointerAttr) {
    emitError({*errorLoc, "'data_to_str' did not narrow to a constant"});
    return failure();
  }

  // Check to see if we have a memref(interp.memory_handle(...)) because
  // we can just immediately fold it in common cases without materializing the
  // memory.
  // We don't handle index/offset yet.
  if (auto result =
          getBytesOf(pointerAttr.getModel().getMemory()[pointerAttr.getIndex()],
                     numBytes)) {
    if (pointerAttr.getOffset() == 0)
      return result;
  }

  // Reset memory upon exit.
  auto resetState = llvm::make_scope_exit([&] { reset(); });

  if (ErrorOrSuccess err = internalizeMemory(pointerAttr)) {
    emitError({*errorLoc, "'data_to_str' failed to read data"});
    return failure();
  }

  size_t address = cast<PointerAttr>(pointerAttr).getAddr();
  Type byteType = IntegerType::get(getContext(), 8);

  // Read each of the bytes into 'result' one at a time.  If any fail,
  // just bail out.
  std::string result;
  while (numBytes) {
    ErrorOr<TypedAttr> attrOr = readAttributeFromMemory(address, byteType);
    if (attrOr.isError() || !isa<IntegerAttr>(attrOr.get())) {
      emitError({*errorLoc, "'data_to_str' failed to read data"});
      return failure();
    }
    result.push_back((char)cast<IntegerAttr>(attrOr.get()).getInt());
    ++address;
    --numBytes;
  }

  // Success!
  return {StringAttr::get(result, StringType::get(getContext()))};
}

/// Evaluate POC::DataToStr "data_to_str" operator.
FailureOr<TypedAttr> IREvaluator::evaluateDataToStr(ParamOperatorAttr op) {
  FailureOr<StringAttr> result = evaluateStringPart(op.getOperand(0));
  if (failed(result))
    return failure();

  // Extra string parts, which will be a VariadicAttr of type
  // !kgen.variadic<>
  VariadicAttr extrasAttr = dyn_cast<VariadicAttr>(op.getOperand(1));
  if (!extrasAttr) {
    emitError(
        {*errorLoc, "'data_to_str' did not narrow to a variadic constant"});
    return failure();
  }

  // If there are no extra parts then we're done.
  if (extrasAttr.getValues().empty())
    return TypedAttr(*result);

  // Otherwise, we need to evaluate the extra parts and concatenate them.
  std::string concatStr = result->str();
  for (TypedAttr extra : extrasAttr.getValues()) {
    FailureOr<StringAttr> extraResult = evaluateStringPart(extra);
    if (failed(extraResult))
      return failure();
    concatStr += extraResult->str();
  }
  return TypedAttr(StringAttr::get(concatStr, StringType::get(getContext())));
}

FailureOr<TypedAttr> IREvaluator::evaluateStringAddress(ParamOperatorAttr op) {
  // Ensure the string is null-terminated. This is safe because `StringAttr`
  // always stores a null terminator.
  auto value = dyn_cast<StringAttr>(op.getOperand(0));
  if (!value) {
    emitError({*errorLoc, "argument is not a concrete string"});
    return failure();
  }

  StringRef str(value.data(), value.size() + 1);
  if (value.getValue().empty())
    str = "\0";

  MemoryHandleAttr hdl = MemoryHandleAttr::get(getContext(), str);
  ErrorOr<int64_t> addr = mapConstGlobalMemory(hdl);
  if (addr.isError()) {
    emitError({*errorLoc, addr.takeError()});
    return failure();
  }

  auto ptr = PointerAttr::get(getContext(), addr.takeValue(), op.getType());
  if (ErrorOrSuccess err = externalizeMemory(ptr)) {
    emitError({*errorLoc, err.takeError()});
    return failure();
  }
  return {ptr};
}

FailureOr<TypedAttr> IREvaluator::evaluateGetEnv(ParamOperatorAttr op) {
  // Grab the module from the elaborator. This is a read operation of memory
  // that is not modified during elaboration, so no synchronization is needed.
  auto name = dyn_cast<StringAttr>(op.getOperands().front());
  if (!name) {
    emitError({*errorLoc, "'get_env' name did not narrow to a constant"});
    return failure();
  }

  // Get the `StringRef` out of the `StringAttr` because the latter comes with
  // a `StringType` type that makes pointer comparisons fails.
  Attribute value = elaborator->env.getValues().get(name.getValue());
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

static void emitDiagnosticToStream(raw_ostream &os, Diagnostic &diag) {
  os << "\n" << diag.getLocation() << ": " << diag;
  for (Diagnostic &note : diag.getNotes())
    emitDiagnosticToStream(os, note);
}

FailureOr<TypedAttr>
IREvaluator::evaluateCompileAssembly(ParamOperatorAttr op) {
  // Cheeky copy. The state of the symbol table right at this moment is
  // sufficient to produce a standalone object for the generator being JIT'd.
  SymbolTable symtabCopy = elaborator->oldSymTab;

  // Slice out a stanalone module to re-elaborate with the new target.
  TargetInfoAttr target = cast<TargetParamAttr>(op.getOperand(0)).getTarget();
  EmitAs emissionKind = cast<EmitAsAttr>(op.getOperand(1)).getValue();
  StringRef emissionOptionsStr = cast<StringAttr>(op.getOperand(2)).getValue();
  bool propagateError = cast<IntegerAttr>(op.getOperand(3)).getInt();
  SymbolConstantAttr symbol = dyn_cast<SymbolConstantAttr>(op.getOperand(4));
  ErrorTreeOr<std::pair<StringAttr, GeneratorOp>> pairOrError =
      elaborator->getExpectedMangledName(*errorLoc, "compile_assembly", symbol,
                                         /*allowParametric=*/false,
                                         /*sanitize=*/false);
  if (pairOrError.isError()) {
    parent->setToError(pairOrError.takeError());
    return failure();
  }
  StringAttr name;
  GeneratorOp func;
  std::tie(name, func) = pairOrError.takeValue();

  // Construct the expected result type.
  MLIRContext *ctx = op.getContext();
  Builder b(ctx);
  auto noneType = KGEN::NoneType::get(ctx);
  auto populateFnType = FuncTypeGeneratorType::get(
      /*inputParamTypes=*/{},
      b.getFunctionType(PointerType::get(noneType), noneType),
      {ArgConvention::ReadReg}, FnEffects().setCapturing());

  // Specialize the generator with another target by slicing it and its
  // transitive dependencies out of the IR and re-invoking the elaborator. If it
  // turns out that the specialization has more than one implementation, then
  // the elaborator invocation will fail due to multiple implementations of a
  // primary generator, and the functor will return an error.

  // Parse the emission options from a comma separated list of values.
  SmallVector<StringRef> emissionOptions;
  emissionOptionsStr.split(emissionOptions, /*Separator=*/",",
                           /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  // Capture the diagnostics that may be emitted.
  DiagnosticHandler handler(ctx);
  ErrorOr<CrossDeviceFunction> closure = elaborator->compileAsmFn(
      func, symbol, name, symtabCopy, target, emissionKind, emissionOptions,
      elaborator->options, elaborator->getOptions(), handler.getHandlerID());
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
                             UninitMemAttr::get(populateFnType)})};
  }

  auto populate = cast<FuncOp>(closure->populateCapturesFn.release());
  auto populateFnRef = SymbolConstantAttr::get(populate);
  elaborator->addDeferredFunction(populate);
  return {
      StructAttr::get({closure->contents, b.getIndexAttr(closure->numCaptures),
                       populateFnRef})};
}

FailureOr<TypedAttr>
IREvaluator::evaluateCompileOffloadClosure(ParamOperatorAttr op) {
  // Create the signature and an empty body of the populate capture for offload
  // closures as part of elaboration step.
  // We currently only support capturing closure as a parameter. So this closure
  // has to be created during elaboration time as a compile time constant.
  // However, bundling GPU and other offload compilation means
  // that the actual compilation of the offload functions will happen later
  // once all of them are seen and collected, and the actual body of this
  // closure will not be known until the offload function is compiled
  // (so that we know what needs to be captured).
  // We will generated the actual body of this closure later.

  // Slice out a standalone module to re-elaborate with the new target later.
  ErrorTreeOr<std::pair<StringAttr, GeneratorOp>> pairOrError =
      elaborator->getExpectedMangledName(
          *errorLoc, "compile_offload_closure", op.getOperand(0),
          /*allowParametric=*/false, /*sanitize=*/false);
  if (pairOrError.isError()) {
    emitError(pairOrError.takeError());
    return failure();
  }
  StringAttr name = pairOrError.takeValue().first;

  // Construct the expected result type.
  MLIRContext *ctx = op.getContext();
  auto noneType = KGEN::NoneType::get(ctx);

  // The location to use for generated code. Remove all debuginfo from it.
  Location loc = DebugInfo::stripDebugScopesRecursively(*errorLoc);

  // The expected signature is `fn(Pointer[None]) capturing -> None`.
  ImplicitLocOpBuilder bb(loc, ctx);
  auto nonePtr = PointerType::get(noneType);
  auto sig = FuncType::get(bb.getFunctionType(nonePtr, noneType),
                           ArgConvention::ReadReg, FnEffects().setCapturing());

  OwningOpRef<FuncOp> populateFunc = bb.create<FuncOp>(
      bb.getStringAttr(name.getValue() + "_populate_captures"), sig,
      InlineLevel::Always);

  auto populate = cast<FuncOp>(populateFunc.get());
  auto populateFnRef = SymbolConstantAttr::get(populate);
  elaborator->addDeferredFunction(std::move(populateFunc));
  return {populateFnRef};
}

FailureOr<TypedAttr> IREvaluator::evaluateGetLinkageName(ParamOperatorAttr op) {
  // This only supports generators with an empty set of parameters, otherwise we
  // need to resolve the symbol name after elaboration.
  TargetInfoAttr target = cast<TargetParamAttr>(op.getOperand(0)).getTarget();
  // HACK HACK HACK: Our current name mangling scheme is not compatible with the
  // GPU backends.
  ErrorTreeOr<std::pair<StringAttr, GeneratorOp>> pairOrError =
      elaborator->getExpectedMangledName(
          *errorLoc, "get_linkage_name", op.getOperand(1),
          /*allowParametric=*/true, /*sanitize=*/target.isGPU());
  if (pairOrError.isError()) {
    emitError(pairOrError.takeError());
    return failure();
  }
  StringAttr name = pairOrError.takeValue().first;
  return {StringAttr::get(name.getValue(), op.getType())};
}

//===----------------------------------------------------------------------===//
// IREvaluator
//===----------------------------------------------------------------------===//

IREvaluator::IREvaluator(Elaborator &elaborator, ImplNode *parent)
    : BytecodeInterpreter(elaborator.getTarget(), &elaborator),
      elaborator(&elaborator), parent(parent) {}

IREvaluator::IREvaluator(const IREvaluator &other)
    : ParameterEvaluator(other),
      BytecodeInterpreter(other.getTarget(), other.elaborator),
      elaborator(other.elaborator), parent(other.parent) {}

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

  // Otherwise we had an error folding the expression tree or we just have a
  // some foreign attribute that doesn't participate in the parameter system.
  // Walk the attribute tree postorder - if we see any attribute that has
  // all-simple-constant leaves, then we check to see if it is errorneous so we
  // can report the error.  We do this in postorder because you could have:
  //    add(4, div(8000000000, 4))
  // and the problem is that div isn't target invariant.  The problem isn't the
  // add outside it.
  result.walk<mlir::WalkOrder::PostOrder>(
      [&](Attribute attr) -> mlir::WalkResult {
        bool allSimple = true;
        attr.walkImmediateSubElements(
            [&](Attribute sub) {
              if (allSimple)
                allSimple = ParameterAttr::isSimpleConstant(sub);
            },
            [&](Type T) {});

        // If this is an attribute with simple operands that refused to fold,
        // see if we're able to get a custom error message from it to explain
        // what is going on.
        if (allSimple) {
          if (auto itf = ::dyn_cast<ParameterAttr>(attr)) {
            auto errorMessage = itf.validateForElaborator();
            if (errorMessage.isError()) {
              emitError(ErrorTree(loc, errorMessage.takeError()));
              return WalkResult::interrupt();
            }
          }
        }
        return WalkResult::advance();
      });

  if (error)
    return std::move(*error);

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

void IREvaluator::addCustomReplacementsToLiftStore(
    mlir::AttrTypeReplacer &liftStore) {
  liftStore.addReplacement([](VTableAttr vtable) {
    return std::make_pair(vtable, WalkResult::skip());
  });
}
