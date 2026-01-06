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
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/POPDialect/POPUtils.h"
#include "KGEN/TransformUtils/ManglingUtils.h"
#include "KGEN/lib/Elaborator/IREvaluatorContext.h"
#include "Support/Compiler/DiagnosticHandler.h"
#include "Support/MDialect/MTypeInterfaces.h"
#include "Support/StringExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/ScopeExit.h"

#include <regex>

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

int IREvaluator::getErrorLimit() {
  return elaborator->options.elaborationErrorLimit;
}

bool IREvaluator::getElabErrorIncludePrelude() {
  return elaborator->options.elaborationErrorIncludePrelude;
}

//===----------------------------------------------------------------------===//
// Expression Evaluation
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr>
IREvaluator::evaluateExpression(ContextuallyEvaluatedAttrInterface attr) {
  // Don't try to evaluate a parameter operator that still contains parametric
  // things in it, since it may be transitory.
  struct IndexRefFinder : IndexParameterReplacer<IndexRefFinder> {
    Attribute tryReplace(Attribute attr, size_t depth) {
      if (auto ref = dyn_cast<ParamIndexRefAttr>(attr)) {
        // This check means it's referring to a param-decl *outside* `attr`, see
        // PSTIAIRAID.
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

  if (auto getWitnessEntry = dyn_cast<GetWitnessAttr>(attr))
    return evaluateGetWitnessAttr(getWitnessEntry);
  if (auto getLinkageNameAttr = dyn_cast<GetLinkageNameAttr>(attr))
    return evaluateGetLinkageNameAttr(getLinkageNameAttr);
  if (auto getSourceNameAttr = dyn_cast<GetSourceNameAttr>(attr))
    return evaluateGetSourceNameAttr(getSourceNameAttr);
  if (auto getTypeNameAttr = dyn_cast<GetTypeNameAttr>(attr))
    return evaluateGetTypeNameAttr(getTypeNameAttr);
  if (auto structFieldTypesAttr = dyn_cast<StructFieldTypesAttr>(attr))
    return evaluateStructFieldTypesAttr(structFieldTypesAttr);
  if (auto structFieldNamesAttr = dyn_cast<StructFieldNamesAttr>(attr))
    return evaluateStructFieldNamesAttr(structFieldNamesAttr);
  if (auto structFieldIndexByNameAttr =
          dyn_cast<StructFieldIndexByNameAttr>(attr))
    return evaluateStructFieldIndexByNameAttr(structFieldIndexByNameAttr);
  if (auto structFieldTypeByNameAttr =
          dyn_cast<StructFieldTypeByNameAttr>(attr))
    return evaluateStructFieldTypeByNameAttr(structFieldTypeByNameAttr);
  if (auto typeConformToTraitAttr = dyn_cast<TypeConformsToTraitAttr>(attr))
    return evaluateTypeConformToTraitAttr(typeConformToTraitAttr);
  if (auto compileOffloadClosureAttr =
          dyn_cast<CompileOffloadClosureAttr>(attr))
    return evaluateCompileOffloadClosureAttr(compileOffloadClosureAttr);
  if (auto compileAssemblyAttr = dyn_cast<CompileAssemblyAttr>(attr))
    return evaluateCompileAssemblyAttr(compileAssemblyAttr);
  if (auto variadicReduceAttr = dyn_cast<VariadicReduceAttr>(attr))
    return variadicReduceAttr.evaluateWith(this);
  if (auto variadicSizeAttr = dyn_cast<VariadicSizeAttr>(attr))
    return evaluateVariadicSizeAttr(variadicSizeAttr);

  if (auto castAttr = dyn_cast<POP::CastAttr>(attr)) {
    auto outType = cast<POP::SIMDType>(castAttr.getType());
    auto inType = cast<POP::SIMDType>(castAttr.getArg().getType());
    if (auto fold =
            POP::foldCast({castAttr.getArg()}, outType, inType, outType,
                          elaborator->getTarget().resolveIndexBitWidth())) {
      return cast<TypedAttr>(cast<Attribute>(fold));
    }
    emitError(ErrorTree(*errorLoc, "Unable to evaluate #pop.cast attribute"));
    return failure();
  }

  // Must be a parameter operator then.
  auto op = dyn_cast<ParamOperatorAttr>(attr);
  assert(op && "unknown attribute with ContextuallyEvaluatedAttrInterface");

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
    return evaluateDataToStr(op, /*reset=*/true);
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
  // The callee may not be a SymbolConstantAttr if it contains unevaluated
  // parameter expressions (e.g., from constructor calls in type parameters
  // accessed via struct_field_types).
  // See https://github.com/modular/modular/issues/5732.
  auto callee = dyn_cast<SymbolConstantAttr>(op.getOperands().front());
  if (!callee) {
    emitError({*errorLoc,
               "callee could not be resolved to a concrete symbol; "
               "this can occur when using reflection on types with unevaluated "
               "constructor calls in their parameters"});
    return failure();
  }

  // Attempt to concretize the function first.
  ErrorTreeOr<FuncOp> funcOr =
      elaborator->getConcreteFunction(parent, *errorLoc, callee);
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

  auto resetState = llvm::make_scope_exit([&] { reset(); });

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

FailureOr<TypedAttr>
IREvaluator::evaluateGetWitnessAttr(GetWitnessAttr getWitnessEntry) {
  // Find the node for the instantiated type ref.
  TypedAttr resolved = getWitnessEntry.getTypeRefIfResolved();
  if (!resolved) {
    emitError({*errorLoc, "no instantiation for trait " +
                              getWitnessEntry.getTraitName().getValue() +
                              ", get witness table failed"});
    return failure();
  }

  // The resolved type may not be a TypeInstanceRefAttr if it contains
  // unevaluated parameter expressions (e.g., apply/apply_result_slot from
  // constructor calls in type parameters accessed via struct_field_types).
  // See https://github.com/modular/modular/issues/5732.
  auto instanceRefAttr = dyn_cast<TypeInstanceRefAttr>(resolved);
  if (!instanceRefAttr) {
    // Return failure to allow graceful handling (e.g., printing <unprintable>)
    return failure();
  }
  SymbolRefAttr instanceRef = instanceRefAttr.getSymbol();
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
    nestedEvaluator.setDeclBinding(param, value);
  FailureOr<TypedAttr> simplified =
      getWitnessEntry.simplify(witnessTable, &nestedEvaluator);
  if (failed(simplified)) {
    emitError({*errorLoc, "failed to locate witness entry for " +
                              gen.getSymName() + ", " +
                              getWitnessEntry.getTraitName().getValue() + ", " +
                              getWitnessEntry.getWitnessName().getValue()});
    return failure();
  }
  return simplified;
}

//===----------------------------------------------------------------------===//
// Struct Field Reflection Helpers
//===----------------------------------------------------------------------===//

FailureOr<StructInstanceType>
IREvaluator::resolveStructInstanceType(TypedAttr typeRef, StringRef funcName) {
  // Resolve type reference to TypeInstanceRefAttr
  if (!isa<TypeInstanceRefAttr>(typeRef)) {
    if (auto typeParam = dyn_cast<TypeParamAttr>(typeRef))
      typeRef = cast<TypeValueType>(typeParam.getTypeValue()).getTypeValue();
    else {
      emitError(
          {*errorLoc, Twine(funcName) + " requires a concrete struct type"});
      return failure();
    }
  }
  // The type may still not be a TypeInstanceRefAttr after extraction from
  // TypeParamAttr if it contains unevaluated parameter expressions.
  // See https://github.com/modular/modular/issues/5732.
  auto instanceRef = dyn_cast<TypeInstanceRefAttr>(typeRef);
  if (!instanceRef) {
    emitError(
        {*errorLoc, Twine(funcName) + " requires a concrete struct type"});
    return failure();
  }

  // Look up the impl node and get the StructGeneratorOp
  SymbolRefAttr instanceSymbol = instanceRef.getSymbol();
  ParamNode *genNode = elaborator->lookupImplNode(instanceSymbol)->parent;
  if (!isa<StructGeneratorOp>(genNode->gen)) {
    emitError({*errorLoc, Twine(funcName) + " requires a struct type"});
    return failure();
  }
  StructGeneratorOp structGen = cast<StructGeneratorOp>(genNode->gen);

  auto structType =
      dyn_cast<StructInstanceType>(structGen.getValueDomainType());
  if (!structType) {
    emitError(
        {*errorLoc, Twine(funcName) + " requires a struct type, got " +
                        mlir::debugString(structGen.getValueDomainType())});
    return failure();
  }

  return structType;
}

//===----------------------------------------------------------------------===//
// Struct Field Reflection Evaluators
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr>
IREvaluator::evaluateStructFieldTypesAttr(StructFieldTypesAttr attr) {
  FailureOr<StructInstanceType> structTypeOr =
      resolveStructInstanceType(attr.getTypeValue(), "struct_field_types");
  if (failed(structTypeOr))
    return failure();
  StructInstanceType structType = *structTypeOr;

  // Build variadic of field types
  MLIRContext *ctx = attr.getContext();
  SmallVector<TypedAttr> fieldTypes;
  for (StructDefFieldAttr field : structType.getFields()) {
    // Wrap field type as a TypeParamAttr
    fieldTypes.push_back(TypeParamAttr::get(field.getType(), field.getType(),
                                            TypeType::get(ctx)));
  }

  return {cast<TypedAttr>(VariadicAttr::get(fieldTypes, attr.getType()))};
}

FailureOr<TypedAttr>
IREvaluator::evaluateStructFieldNamesAttr(StructFieldNamesAttr attr) {
  FailureOr<StructInstanceType> structTypeOr =
      resolveStructInstanceType(attr.getTypeValue(), "struct_field_names");
  if (failed(structTypeOr))
    return failure();
  StructInstanceType structType = *structTypeOr;

  // Build variadic of field names as StringAttrs
  MLIRContext *ctx = attr.getContext();
  SmallVector<TypedAttr> fieldNames;
  for (StructDefFieldAttr field : structType.getFields()) {
    // StringAttr with StringType wrapping
    fieldNames.push_back(
        StringAttr::get(field.getName().getValue(), StringType::get(ctx)));
  }

  return {cast<TypedAttr>(VariadicAttr::get(fieldNames, attr.getType()))};
}

FailureOr<TypedAttr> IREvaluator::evaluateStructFieldIndexByNameAttr(
    StructFieldIndexByNameAttr attr) {
  FailureOr<StructInstanceType> structTypeOr = resolveStructInstanceType(
      attr.getTypeValue(), "struct_field_index_by_name");
  if (failed(structTypeOr))
    return failure();
  StructInstanceType structType = *structTypeOr;

  // The field name should be a StringAttr after parameter evaluation
  auto fieldNameAttr = dyn_cast<StringAttr>(attr.getFieldName());
  if (!fieldNameAttr) {
    emitError({*errorLoc,
               "struct_field_index_by_name requires a string literal for field "
               "name, got " +
                   mlir::debugString(attr.getFieldName())});
    return failure();
  }
  StringRef fieldName = fieldNameAttr.getValue();

  // Find field by name
  size_t index = 0;
  for (StructDefFieldAttr field : structType.getFields()) {
    if (field.getName().getValue() == fieldName) {
      // Return the index as an IntegerAttr with index type
      return {cast<TypedAttr>(Builder(attr.getContext()).getIndexAttr(index))};
    }
    ++index;
  }

  // Field not found - emit compile error
  emitError({*errorLoc, "struct '" + mlir::debugString(structType) +
                            "' has no field named '" + fieldName + "'"});
  return failure();
}

FailureOr<TypedAttr>
IREvaluator::evaluateStructFieldTypeByNameAttr(StructFieldTypeByNameAttr attr) {
  FailureOr<StructInstanceType> structTypeOr = resolveStructInstanceType(
      attr.getTypeValue(), "struct_field_type_by_name");
  if (failed(structTypeOr))
    return failure();
  StructInstanceType structType = *structTypeOr;

  // The field name should be a StringAttr after parameter evaluation
  auto fieldNameAttr = dyn_cast<StringAttr>(attr.getFieldName());
  if (!fieldNameAttr) {
    emitError({*errorLoc,
               "struct_field_type_by_name requires a string literal for field "
               "name, got " +
                   mlir::debugString(attr.getFieldName())});
    return failure();
  }
  StringRef fieldName = fieldNameAttr.getValue();

  // Find field by name
  MLIRContext *ctx = attr.getContext();
  for (StructDefFieldAttr field : structType.getFields()) {
    if (field.getName().getValue() == fieldName) {
      // Return the field type wrapped in TypeParamAttr
      return {cast<TypedAttr>(TypeParamAttr::get(
          field.getType(), field.getType(), TypeType::get(ctx)))};
    }
  }

  // Field not found - emit compile error
  emitError({*errorLoc, "struct '" + mlir::debugString(structType) +
                            "' has no field named '" + fieldName + "'"});
  return failure();
}

FailureOr<TypedAttr>
IREvaluator::evaluateVariadicSizeAttr(VariadicSizeAttr attr) {
  // If the inner variadic is a concrete VariadicAttr, return its size.
  // This enables folding after nested attributes like StructFieldTypesAttr
  // have been evaluated to concrete variadics.
  auto vaAttr = sugarDynCast<VariadicAttr>(attr.getVariadic());
  if (!vaAttr)
    return failure();

  return {cast<TypedAttr>(
      Builder(attr.getContext()).getIndexAttr(vaAttr.getValues().size()))};
}

ParamNodeBase *IREvaluator::lookupParamNodeBase(SymbolRefAttr symbol) {
  return elaborator->lookupImplNode(symbol)->parent;
}

ErrorTreeOr<std::pair<StringAttr, GeneratorOp>>
IREvaluator::getExpectedMangledName(
    Location errorLoc, StringRef errorContext, TypedAttr symCst,
    bool allowParametric, bool sanitize,
    function_ref<std::string(StringRef)> getPrefix) {
  return elaborator->getExpectedMangledName(
      errorLoc, errorContext, symCst, allowParametric, sanitize, getPrefix);
}

GeneratorOp IREvaluator::getGenerator(SymbolRefAttr symbol) {
  return elaborator->oldSymTab.lookup<GeneratorOp>(
      cast<FlatSymbolRefAttr>(symbol).getAttr());
}

ErrorOr<CrossDeviceFunction>
IREvaluator::compileAsm(MLIRContext *ctx, GeneratorOp func,
                        SymbolConstantAttr symbol, StringAttr name,
                        TargetInfoAttr target, EmitAs emissionKind,
                        EmissionOptions emissionOptions) {
  SymbolTable symtabCopy = elaborator->oldSymTab;
  ErrorOr<CrossDeviceFunction> closure = elaborator->compileAsmFn(
      func, symbol, name, symtabCopy, target, emissionKind, emissionOptions,
      elaborator->options, elaborator->getOptions());
  return closure;
}

void IREvaluator::addDeferredFunction(OwningOpRef<FuncOp> func) {
  elaborator->addDeferredFunction(std::move(func));
}

ImplNodeBase *IREvaluator::getParentNode() { return parent; }
//===----------------------------------------------------------------------===//
// IREvaluator
//===----------------------------------------------------------------------===//

IREvaluator::IREvaluator(Elaborator &elaborator, ImplNode *parent)
    : IREvaluatorContext(elaborator.env, elaborator.getTarget().getContext(),
                         this),
      BytecodeInterpreter(elaborator.getTarget(), &elaborator),
      elaborator(&elaborator), parent(parent) {
  setEvaluationContext(this);
}

IREvaluator::IREvaluator(const IREvaluator &other)
    : ParameterEvaluator(other),
      IREvaluatorContext(other.elaborator->env, other.getTarget().getContext(),
                         this),
      BytecodeInterpreter(other.getTarget(), other.elaborator),
      elaborator(other.elaborator), parent(other.parent) {
  setEvaluationContext(this);
}

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
  // all-simple-constant leaves, then we check to see if it is erroneous so we
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
