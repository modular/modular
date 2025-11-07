//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ParametricIREvaluator.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "KGEN/TransformUtils/ManglingUtils.h"
#include "KGEN/lib/Elaborator/IREvaluatorContext.h"
#include "ParametricElaborator.h"
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

ErrorOr<std::pair<Region *, Operation *>>
ParametricIREvaluator::lookupParametricFunctionBody(SymbolRefAttr symbol) {
  StringAttr name = cast<FlatSymbolRefAttr>(symbol).getAttr();
  if (GeneratorOpInterface gen = elaborator->lookupGeneratorOp(name)) {
    return std::make_pair(&gen.getBodyRegion(), gen.getOperation());
  }

  InstantiatedOpInterface inst = elaborator->lookupInstantiatedOp(name);
  FuncOp func = cast<FuncOp>(inst);

  // Now we can return the function body.
  return std::make_pair(&func.getBodyRegion(), nullptr);
}

ErrorOr<Region *>
ParametricIREvaluator::lookupFunctionBody(SymbolRefAttr symbol) {
  ErrorOr<std::pair<Region *, Operation *>> result =
      lookupParametricFunctionBody(symbol);
  if (result.isError())
    return result.takeError();

  return result->first;
}

ErrorOr<Type>
ParametricIREvaluator::lookupFuncTypeGenerator(SymbolRefAttr symbol) {
  StringAttr name = cast<FlatSymbolRefAttr>(symbol).getAttr();
  if (GeneratorOpInterface itf = elaborator->lookupGeneratorOp(name)) {
    if (auto gen = dyn_cast<GeneratorOp>(itf.getOperation())) {
      return gen.getFuncTypeGenerator();
    }
    return Error("cannot find FuncTypeGeneratorType");
  }

  InstantiatedOpInterface inst = elaborator->lookupInstantiatedOp(name);
  FuncOp func = cast<FuncOp>(inst);
  return func.getFuncTypeGenerator();
}

ErrorTreeOr<TypedAttr> ParametricIREvaluator::interpretGeneratorWithResultSlot(
    Attribute calleeAttr, llvm::ArrayRef<TypedAttr> paramValues,
    ArrayRef<Attribute> arguments, Location loc) {

  ParametricIREvaluator nestedEvaluator(*this);
  nestedEvaluator.setErrorLoc(loc);
  auto callee = cast<SymbolConstantAttr>(calleeAttr);
  ErrorOr<std::pair<Region *, Operation *>> bodyOr =
      nestedEvaluator.lookupParametricFunctionBody(callee.getSymbol());

  if (bodyOr.isError())
    return ErrorTree(loc, bodyOr.takeError());
  Region &body = *bodyOr->first;
  Operation *op = bodyOr->second;

  SmallVector<TypedAttr> operands(arguments.size());
  for (auto [idx, arg] : llvm::enumerate(arguments))
    operands[idx] = cast<TypedAttr>(arg);

  auto operandsAttr =
      ParameterExprArrayAttr::get(calleeAttr.getContext(), operands);

  TypedAttr cached =
      elaborator->lookupCachedInterpretation(op, operandsAttr, callee);
  if (!cached) {
    nestedEvaluator.pushParamValues(paramValues, true);
    nestedEvaluator.setDeclBindings(op, paramValues);

    Type resultPtrType =
        nestedEvaluator.getReboundType(body.getArguments().back().getType());
    auto ptr = dyn_cast<PointerType>(resultPtrType);
    Type elemType = ptr.getElementType();

    // Evaluate the iterator function body.
    auto result = nestedEvaluator.executeRegionWithResultSlot(
        body, arguments, elemType, resultPtrType);
    if (result.isError())
      return result.takeError();

    cached = result.takeValue();

    (void)elaborator->writeGlobalCachedInterpretation(op, operandsAttr, callee,
                                                      cached);
  }
  return cached;
}

ErrorTreeOr<TypedAttr> ParametricIREvaluator::interpretGenerator(
    Attribute calleeAttr, llvm::ArrayRef<TypedAttr> paramValues,
    ArrayRef<Attribute> arguments, Location loc) {

  ParametricIREvaluator nestedEvaluator(*this);
  nestedEvaluator.setErrorLoc(loc);

  auto callee = cast<SymbolConstantAttr>(calleeAttr);
  auto bodyOr =
      nestedEvaluator.lookupParametricFunctionBody(callee.getSymbol());
  if (bodyOr.isError())
    return ErrorTree(loc, bodyOr.takeError());
  Region &body = *bodyOr->first;
  Operation *op = bodyOr->second;

  SmallVector<TypedAttr> operands(arguments.size());
  for (auto [idx, arg] : llvm::enumerate(arguments))
    operands[idx] = cast<TypedAttr>(arg);

  auto operandsAttr =
      ParameterExprArrayAttr::get(calleeAttr.getContext(), operands);

  TypedAttr cached =
      elaborator->lookupCachedInterpretation(op, operandsAttr, callee);

  if (!cached) {
    nestedEvaluator.pushParamValues(paramValues, true);
    nestedEvaluator.setDeclBindings(bodyOr->second, paramValues);

    auto result = nestedEvaluator.executeRegion(body, arguments);
    if (result.isError())
      return result.takeError();
    cached = cast<TypedAttr>(result.getValue().front());
    (void)elaborator->writeGlobalCachedInterpretation(op, operandsAttr, callee,
                                                      cached);
  }

  return cached;
}

void ParametricIREvaluator::setDeclBindings(Operation *op,
                                            ArrayRef<TypedAttr> paramValues) {
  if (auto gen = dyn_cast<GeneratorOpInterface>(op)) {
    for (auto [decl, attr] : llvm::zip(gen.getInputParams(), paramValues)) {
      getCurrentParamEval().overwriteDeclBinding(decl, attr);
    }
  }
}

void ParametricIREvaluator::clearParameterCache() {}

void ParametricIREvaluator::pushEvalFrame(Operation *op, Region *region,
                                          llvm::ArrayRef<TypedAttr> paramValues,
                                          int id) {
  ParametricParameterEvaluator curr(paramEvaluators.back());
  curr.setEvaluationContext(this);

  ParameterExprArrayAttr paramValueAttr;
  SmallVector<TypedAttr> pValues;

  if (!frameParamInfos.empty()) {
    paramValueAttr = ParameterExprArrayAttr::get(
        op->getContext(), frameParamInfos.back().paramValues);
    pValues = frameParamInfos.back().paramValues;
  }

  curr.cachedOpKey = op;
  curr.cachedRegionKey = region;
  curr.cachedAttrKey = paramValueAttr;

  auto tlIter = elaborator->tlParamInterpCache->find({region, paramValueAttr});

  if (tlIter != elaborator->tlParamInterpCache->end()) {
    curr.setRewritten(tlIter->second);
    curr.foundCached = true;
  } else {
    std::optional<DenseMap<std::pair<size_t, const void *>, const void *>>
        result = elaborator->paramInterpCache.read(
            [region, paramValueAttr](auto &map)
                -> std::optional<
                    DenseMap<std::pair<size_t, const void *>, const void *>> {
              auto result = map.find({region, paramValueAttr});
              if (result != map.end()) {
                return result->second;
              }
              return std::nullopt;
            });

    if (result) {
      curr.setRewritten(std::move(*result));
      curr.foundCached = true;
    } else {
      bool clearCache = !isa<ParamIfOp>(op);
      if (auto gen = dyn_cast<GeneratorOpInterface>(op)) {
        FunctionParameterUseDefGraph &g = *elaborator->knownGraphs.get()[gen];
        clearCache = g.hasParams || !gen.getInputParams().empty();
      }
      if (clearCache)
        curr.clearCache();
    }
  }

  paramEvaluators.push_back(std::move(curr));
}

void ParametricIREvaluator::popEvalFrame() {
  ParametricParameterEvaluator &back = paramEvaluators.back();
  if (!back.foundCached && back.cachedRegionKey) {
    // set the cache
    auto &rewritten = back.getRewritten();
    if (!rewritten.empty()) {
      (*elaborator
            ->tlParamInterpCache)[{back.cachedRegionKey, back.cachedAttrKey}] =
          rewritten;
      elaborator->paramInterpCache.modify([back, rewritten](auto &map) {
        map.insert({{back.cachedRegionKey, back.cachedAttrKey}, rewritten});
      });
    }
  }

  paramEvaluators.pop_back();
}

void ParametricIREvaluator::popEvalFrame(size_t size) {
  assert(paramEvaluators.size() >= size && "popEvalFrame failed!");
  paramEvaluators.erase(paramEvaluators.begin() + size - 1,
                        paramEvaluators.end());
}

void ParametricIREvaluator::pushParamValues(llvm::ArrayRef<TypedAttr> values,
                                            bool pushFrame, Operation *op) {
  if (pushFrame)
    frameParamInfos.emplace_back(FrameParamInfo{});

  FrameParamInfo &currFrame = frameParamInfos.back();

  currFrame.numParamsPerScope.emplace_back(std::make_pair(op, values.size()));
  currFrame.paramValues.append(values.begin(), values.end());
}

DenseSet<Operation *> *ParametricIREvaluator::getParamOps(Operation *op,
                                                          std::string &name) {
  auto &map = elaborator->knownGraphs.get();
  if (auto gen = dyn_cast<GeneratorOpInterface>(op)) {
    assert(map.contains(gen));
    return &map[gen]->paramOpsSet;
  }
  return nullptr;
}

void ParametricIREvaluator::setIsCurrOpParam(Operation *op) {
  isCurrOpParam = isa<ParamOpInterface>(op) || !stack.back().paramOps ||
                  stack.back().paramOps->contains(op);
}

void ParametricIREvaluator::popParamValues(bool popFrame, Operation *op,
                                           Operation *tillOp) {
  if (popFrame) {
    if (!frameParamInfos.empty())
      frameParamInfos.pop_back();
    return;
  }

  FrameParamInfo &currFrame = frameParamInfos.back();

  Operation *currOp = nullptr;

  do {
    assert(currFrame.numParamsPerScope.size() > 0 &&
           "popParamValues has wrong number of param values to pop");
    size_t numValuesToPop = currFrame.numParamsPerScope.back().second;
    currOp = currFrame.numParamsPerScope.back().first;

    assert(currFrame.paramValues.size() >= numValuesToPop &&
           "popParamValues has not enough param values to pop");

    currFrame.paramValues.erase(currFrame.paramValues.end() - numValuesToPop,
                                currFrame.paramValues.end());

    currFrame.numParamsPerScope.pop_back();
  } while (tillOp && currOp != tillOp);
}

void ParametricIREvaluator::appendParamValues(llvm::ArrayRef<TypedAttr> values,
                                              int id, Operation *op) {
  assert(frameParamInfos.size() > 0 && "no more frames to append to");
  FrameParamInfo &currFrame = frameParamInfos.back();

  currFrame.paramValues.append(values.begin(), values.end());
  if (currFrame.numParamsPerScope.empty())
    currFrame.numParamsPerScope.emplace_back(std::make_pair(op, values.size()));
  else
    currFrame.numParamsPerScope.back().second += values.size();
}

ErrorTreeOr<TypedAttr>
ParametricIREvaluator::evaluateFunction(FuncOp func,
                                        ArrayRef<TypedAttr> inputs) {
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
ParametricIREvaluator::evaluateGenerator(GeneratorOp func,
                                         ArrayRef<TypedAttr> inputs) {
  // Evaluate the function body.
  SmallVector<Attribute> arguments;
  for (TypedAttr input : inputs)
    arguments.push_back(input);

  std::optional<ErrorTree> error;
  if (!emitError)
    emitError = [&](ErrorTree err) { error = std::move(err); };

  ErrorTreeOr<SmallVector<Attribute>> result =
      executeRegion(func.getBodyRegion(), arguments);

  // Report an error if evaluation fails.
  if (result.isError()) {
    return ErrorTree(*errorLoc, "failed to compile-time evaluate function call",
                     result.takeError());
  } else if (error) {
    return ErrorTree(*errorLoc, "failed to compile-time evaluate function call",
                     std::move(*error));
  }

  // Apply operators only return one result.
  return cast<TypedAttr>(result.getValue().front());
}

ErrorTreeOr<TypedAttr> ParametricIREvaluator::evaluateGeneratorWithResultSlot(
    GeneratorOp func, ArrayRef<TypedAttr> inputs) {
  if constexpr (KGEN::kIsTracingEnabled)
    auto ts = InterpreterTimeTraceScope("Launch interpreter",
                                        mlir::debugString(errorLoc));

  // Evaluate the function body.
  SmallVector<Attribute> arguments;
  for (TypedAttr input : inputs)
    arguments.push_back(input);

  std::optional<ErrorTree> error;
  if (!emitError)
    emitError = [&](ErrorTree err) { error = std::move(err); };

  auto resultArg = func.getArguments().back();
  auto ptr = dyn_cast<PointerType>(resultArg.getType());
  if (!ptr)
    return ErrorTree(func.getLoc(), "result argument is not a pointer");

  Type elemType = getReboundType(ptr.getElementType());
  Type resultPtrType =
      getReboundType(func.getBodyRegion().getArguments().back().getType());

  ErrorTreeOr<TypedAttr> result = executeRegionWithResultSlot(
      func.getBodyRegion(), arguments, elemType, resultPtrType);

  // Report an error if evaluation fails.
  if (result.isError()) {
    return ErrorTree(
        *errorLoc,
        "failed to compile-time evaluate generator with resultslot call",
        result.takeError());
  } else if (error) {
    return ErrorTree(
        *errorLoc,
        "failed to compile-time evaluate generator with resultslot call",
        std::move(*error));
  }

  // Apply operators only return one result.
  return result.takeValue();
}

ErrorTreeOr<TypedAttr> ParametricIREvaluator::evaluateFunctionWithResultSlot(
    FuncOp func, ArrayRef<TypedAttr> inputs) {
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
      func.getBodyRegion(), arguments, ptr.getElementType(),
      func.getBodyRegion().getArguments().back().getType());

  // Report an error if evaluation fails.
  if (result.isError()) {
    return ErrorTree(
        *errorLoc,
        "failed to compile-time evaluate function with resultslot call",
        result.takeError());
  }

  // Apply operators only return one result.
  return result.takeValue();
}

int ParametricIREvaluator::getErrorLimit() {
  return elaborator->options.elabErrorLimit;
}

bool ParametricIREvaluator::getElabErrorIncludePrelude() {
  return elaborator->options.elabErrorIncludePrelude;
}

//===----------------------------------------------------------------------===//
// Expression Evaluation
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> ParametricIREvaluator::evaluateExpression(
    ContextuallyEvaluatedAttrInterface attr) {
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
  if (auto typeConformToTraitAttr = dyn_cast<TypeConformsToTraitAttr>(attr))
    return evaluateTypeConformToTraitAttr(typeConformToTraitAttr);

  if (auto compileOffloadClosureAttr =
          dyn_cast<CompileOffloadClosureAttr>(attr))
    return evaluateCompileOffloadClosureAttr(compileOffloadClosureAttr);
  if (auto compileAssemblyAttr = dyn_cast<CompileAssemblyAttr>(attr))
    return evaluateCompileAssemblyAttr(compileAssemblyAttr);

  // Must be a parameter operator then.
  auto op = dyn_cast<ParamOperatorAttr>(attr);
  if (!op)
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
    return evaluateDataToStr(op, /*reset=*/false);
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

FailureOr<TypedAttr>
ParametricIREvaluator::evaluateApplyLike(ParamOperatorAttr op,
                                         bool withResultSlot) {
  auto symbol =
      cast<SymbolConstantAttr>(getReboundAttribute(op.getOperands().front()));
  StringAttr name = cast<FlatSymbolRefAttr>(symbol.getSymbol()).getAttr();
  auto gen = elaborator->oldSymTab.lookup<GeneratorOp>(name);

  // Attempt to lookup a cached value. This returns a thread local cached value.
  auto operandsAttr = cast<ParameterExprArrayAttr>(
      getReboundAttribute(ParameterExprArrayAttr::get(
          op.getContext(), op.getOperands().drop_front())));

  FuncOp func;
  if (!gen) {
    // if this is not a generator, get a concreteFunction
    ErrorTreeOr<FuncOp> funcOr =
        elaborator->getConcreteFunction(parent, *errorLoc, symbol);
    if (funcOr.isError()) {
      emitError(funcOr.takeError());
      return failure();
    }
    func = funcOr.takeValue();
    if (!func)
      return TypedAttr();
  }

  Operation *cacheOp = (!gen) ? func : gen;

  TypedAttr cached =
      elaborator->lookupCachedInterpretation(cacheOp, operandsAttr, symbol);
  if (cached) {
    return cached;
  }

  ParametricIREvaluator nestedEvaluator(*this);

  // Now invoke the interpreter.
  ErrorTreeOr<TypedAttr> result = [&]() -> ErrorTreeOr<TypedAttr> {
    if (!gen) {
      nestedEvaluator.pushEvalFrame(func, &func.getBodyRegion(), {}, 3);
      return withResultSlot
                 ? nestedEvaluator.evaluateFunctionWithResultSlot(func,
                                                                  operandsAttr)
                 : nestedEvaluator.evaluateFunction(func, operandsAttr);
    } else {
      nestedEvaluator.pushParamValues(symbol.getParamValues(), true);
      nestedEvaluator.pushEvalFrame(gen.getOperation(), &gen.getBodyRegion(),
                                    symbol.getParamValues(), 4);

      for (auto [decl, attr] :
           llvm::zip(gen.getInputParams(), symbol.getParamValues())) {
        nestedEvaluator.overwriteDeclBinding(decl, attr);
      }
      return withResultSlot
                 ? nestedEvaluator.evaluateGeneratorWithResultSlot(gen,
                                                                   operandsAttr)
                 : nestedEvaluator.evaluateGenerator(gen, operandsAttr);
    }
  }();

  // If we had a value, write it back.
  if (auto cached = result.tryGetValue()) {
    auto res = elaborator->writeGlobalCachedInterpretation(
        cacheOp, operandsAttr, symbol, cached);
    return res.first;
  }

  result.takeError().emit([](Location loc) { return mlir::emitError(loc); },
                          "interpreter failed.",
                          elaborator->options.elabErrorIncludePrelude);

  return TypedAttr();
}

FailureOr<TypedAttr>
ParametricIREvaluator::evaluateStringAddress(ParamOperatorAttr op) {
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

  ParametricIREvaluator nestedEvaluator(*this);

  MemoryHandleAttr hdl = MemoryHandleAttr::get(getContext(), str);
  ErrorOr<int64_t> addr = nestedEvaluator.mapConstGlobalMemory(hdl);
  if (addr.isError()) {
    emitError({*errorLoc, addr.takeError()});
    return failure();
  }

  auto ptr = PointerAttr::get(getContext(), addr.takeValue(), op.getType());
  if (ErrorOrSuccess err = nestedEvaluator.externalizeMemory(ptr)) {
    emitError({*errorLoc, err.takeError()});
    return failure();
  }
  return {ptr};
}

FailureOr<TypedAttr>
ParametricIREvaluator::evaluateGetWitnessAttr(GetWitnessAttr getWitnessEntry) {
  // Find the node for the instantiated type ref.
  TypedAttr resolved = getWitnessEntry.getTypeRefIfResolved();
  if (!resolved) {
    emitError({*errorLoc, "no instantiation for trait " +
                              getWitnessEntry.getTraitName().getValue() +
                              ", get witness table failed"});
    return failure();
  }

  SymbolRefAttr instanceRef = cast<TypeInstanceRefAttr>(resolved).getSymbol();
  PParamNode *genNode = elaborator->lookupImplNode(instanceRef)->parent;
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

  ParametricIREvaluator nestedEvaluator(*elaborator, parent);
  nestedEvaluator.setErrorLoc(*errorLoc);
  // If the struct generator has input params, we need to provide an
  // IREvaluator for concretizing the witness entries.
  nestedEvaluator.pushParamValues(genNode->inputParams, true);
  for (auto [param, value] :
       llvm::zip(gen.getInputParams(), genNode->inputParams)) {
    nestedEvaluator.overwriteDeclBinding(param, value);
  }

  FailureOr<TypedAttr> simplified = getWitnessEntry.simplify(
      witnessTable, &nestedEvaluator.getCurrentParamEval());

  if (failed(simplified)) {
    emitError({*errorLoc, "failed to locate witness entry for " +
                              gen.getSymName() + ", " +
                              getWitnessEntry.getTraitName().getValue() + ", " +
                              getWitnessEntry.getWitnessName().getValue()});
    return failure();
  }
  return simplified;
}

static void emitDiagnosticToStream(raw_ostream &os, Diagnostic &diag) {
  os << "\n" << diag.getLocation() << ": " << diag;
  for (Diagnostic &note : diag.getNotes())
    emitDiagnosticToStream(os, note);
}

FailureOr<TypedAttr>
ParametricIREvaluator::evaluateCompileAssemblyAttr(CompileAssemblyAttr attr) {
  // Cheeky copy. The state of the symbol table right at this moment is
  // sufficient to produce a standalone object for the generator being JIT'd.
  SymbolTable symtabCopy = elaborator->oldSymTab;

  // Slice out a standalone module to re-elaborate with the new target.
  TargetInfoAttr target = cast<TargetParamAttr>(attr.getTarget()).getTarget();
  EmitAs emissionKind = cast<EmitAsAttr>(attr.getEmissionKind()).getValue();
  StringRef emissionOptionsStr =
      cast<StringAttr>(attr.getEmissionOptions()).getValue();
  bool propagateError = cast<BoolAttr>(attr.getPropagateError()).getValue();
  SymbolConstantAttr symbol = dyn_cast<SymbolConstantAttr>(attr.getFunc());
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
  MLIRContext *ctx = attr.getContext();
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
      elaborator->options, elaborator->getOptions());
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

FailureOr<TypedAttr> ParametricIREvaluator::evaluateCompileOffloadClosureAttr(
    CompileOffloadClosureAttr compileOffloadClosureAttr) {
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

  TargetInfoAttr target =
      cast<TargetParamAttr>(compileOffloadClosureAttr.getTarget()).getTarget();
  // Add "_" prefix to GPU kernel name if it starts with a number, otherwise ptx
  // compiler will fail.
  ErrorTreeOr<std::pair<StringAttr, GeneratorOp>> pairOrError =
      elaborator->getExpectedMangledName(
          *errorLoc, "compile_offload_closure",
          compileOffloadClosureAttr.getFunc(),
          /*allowParametric=*/false, /*sanitize=*/false,
          [isGPU = target.isGPU()](StringRef name) {
            return (isGPU && llvm::isDigit(name.front())) ? "_" : "";
          });

  if (pairOrError.isError()) {
    emitError(pairOrError.takeError());
    return failure();
  }
  StringAttr name = pairOrError.takeValue().first;

  // Construct the expected result type.
  MLIRContext *ctx = compileOffloadClosureAttr.getContext();
  auto noneType = KGEN::NoneType::get(ctx);

  // The location to use for generated code. Remove all debuginfo from it.
  Location loc = DebugInfo::stripDebugScopesRecursively(*errorLoc);

  // The expected signature is `fn(Pointer[None]) capturing -> None`.
  ImplicitLocOpBuilder bb(loc, ctx);
  auto nonePtr = PointerType::get(noneType);
  auto sig = FuncType::get(bb.getFunctionType(nonePtr, noneType),
                           ArgConvention::ReadReg, FnEffects().setCapturing());

  OwningOpRef<FuncOp> populateFunc = FuncOp::create(
      bb, bb.getStringAttr(name.getValue() + "_populate_captures"), sig,
      InlineLevel::Always);

  auto populate = cast<FuncOp>(populateFunc.get());
  auto populateFnRef = SymbolConstantAttr::get(populate);
  elaborator->addDeferredFunction(std::move(populateFunc));
  return {populateFnRef};
}

FailureOr<TypedAttr> ParametricIREvaluator::evaluateGetLinkageNameAttr(
    GetLinkageNameAttr getLinkageNameAttr) {
  // This only supports generators with an empty set of parameters, otherwise we
  // need to resolve the symbol name after elaboration.
  TargetInfoAttr target =
      cast<TargetParamAttr>(getLinkageNameAttr.getTarget()).getTarget();
  // HACK HACK HACK: Our current name mangling scheme is not compatible with the
  // GPU backends.

  // Add "_" prefix to GPU kernel name if it starts with a number, otherwise ptx
  // compiler will fail.
  ErrorTreeOr<std::pair<StringAttr, GeneratorOp>> pairOrError =
      elaborator->getExpectedMangledName(
          *errorLoc, "get_linkage_name", getLinkageNameAttr.getFunc(),
          /*allowParametric=*/true, /*sanitize=*/target.isGPU(),
          [isGPU = target.isGPU()](StringRef name) {
            return (isGPU && llvm::isDigit(name.front())) ? "_" : "";
          });
  if (pairOrError.isError()) {
    emitError(pairOrError.takeError());
    return failure();
  }
  StringAttr name = pairOrError.takeValue().first;
  return {StringAttr::get(name.getValue(), getLinkageNameAttr.getType())};
}

FailureOr<TypedAttr> ParametricIREvaluator::evaluateGetSourceNameAttr(
    GetSourceNameAttr getSourceNameAttr) {

  auto symbol = dyn_cast<SymbolConstantAttr>(getSourceNameAttr.getFunc());
  if (!symbol) {
    emitError({*errorLoc, "'get_source_name' function argument did not resolve "
                          "to a concrete function"});
    return failure();
  }
  auto func = elaborator->oldSymTab.lookup<GeneratorOp>(
      cast<FlatSymbolRefAttr>(symbol.getSymbol()).getAttr());
  std::optional<StringRef> sourceName = func.getSourceName();
  if (!sourceName) {
    emitError({*errorLoc, "function '" +
                              symbol.getSymbol().getLeafReference().getValue() +
                              "' has no source name"});
    return failure();
  }
  return {StringAttr::get(*sourceName, getSourceNameAttr.getType())};
}

ParamNodeBase *
ParametricIREvaluator::lookupParamNodeBase(SymbolRefAttr symbol) {
  return elaborator->lookupImplNode(symbol)->parent;
}

FailureOr<TypedAttr> ParametricIREvaluator::evaluateGetTypeNameAttr(
    GetTypeNameAttr getTypeNameAttr) {
  auto qualifiedBuiltins =
      dyn_cast<BoolAttr>(getTypeNameAttr.getQualifiedBuiltins());
  if (!qualifiedBuiltins) {
    emitError({*errorLoc, "'get_type_name' name did not narrow to a constant"});
    return failure();
  }

  // Find the struct generator for the instantiated type ref.
  TypedAttr typeRef = getTypeNameAttr.getTypeValue();
  if (!isa<TypeInstanceRefAttr>(typeRef)) {
    typeRef = cast<TypeValueType>(cast<TypeParamAttr>(typeRef).getTypeValue())
                  .getTypeValue();
  }

  TypeInstanceRefAttr instanceRef = cast<TypeInstanceRefAttr>(typeRef);
  return {StringAttr::get(
      stringifyTypeInstanceRef(instanceRef, qualifiedBuiltins.getValue()),
      getTypeNameAttr.getType())};
}

FailureOr<TypedAttr> ParametricIREvaluator::evaluateTypeConformToTraitAttr(
    TypeConformsToTraitAttr typeConformToTraitAttr) {
  // This is the list of trait names (`alias T = T1 & T2 & ....`) we need to
  // check.
  auto traitNames =
      dyn_cast<VariadicAttr>(typeConformToTraitAttr.getTraitNames());
  if (!traitNames) {
    emitError({*errorLoc, "'" + TypeConformsToTraitAttr::name + "'" +
                              " did not narrow to concrete trait names"});
    return failure();
  }

  // Find the struct generator for the instantiated type ref.
  TypedAttr typeRef = typeConformToTraitAttr.getTypeValue();
  if (!isa<TypeInstanceRefAttr>(typeRef)) {
    typeRef = cast<TypeValueType>(cast<TypeParamAttr>(typeRef).getTypeValue())
                  .getTypeValue();
  }
  TypeInstanceRefAttr instanceRef = cast<TypeInstanceRefAttr>(typeRef);
  PParamNode *genNode =
      elaborator->lookupImplNode(instanceRef.getSymbol())->parent;
  StructGeneratorOp genOp = cast<StructGeneratorOp>(genNode->gen);

  // Check when the struct conforms to all the traits.
  for (auto toCheck : traitNames.getValues())
    if (!genOp.lookupSymbol(cast<StringAttr>(toCheck).getValue()))
      return {BoolAttr::get(getContext(), false)};

  return {BoolAttr::get(getContext(), true)};
}

//===----------------------------------------------------------------------===//
// ParametricIREvaluator
//===----------------------------------------------------------------------===//

ParametricIREvaluator::ParametricIREvaluator(ParametricElaborator &elaborator,
                                             PImplNode *parent)
    : IREvaluatorContext(elaborator.env, elaborator.getTarget().getContext(),
                         this),
      ParametricParameterEvaluator(),
      ParametricIRInterpreter(elaborator.config.maxDepth,
                              elaborator.getTarget()),
      elaborator(&elaborator), parent(parent) {
  setEvaluationContext(this);
  paramEvaluators.emplace_back(*this);
  paramEvaluators.back().setEvaluationContext(this);
}

ParametricIREvaluator::ParametricIREvaluator(const ParametricIREvaluator &other)
    : IREvaluatorContext(other.elaborator->env, other.getTarget().getContext(),
                         this),
      ParametricParameterEvaluator(other),
      ParametricIRInterpreter(other.maxDepth, other.getTarget()),
      elaborator(other.elaborator), parent(other.parent) {
  setEvaluationContext(this);
  // This is weird, should move this initialization somewhere else.
  nestedStackDepth = other.nestedStackDepth + other.stack.size();
  this->errorLoc = other.errorLoc;
  this->emitError = other.emitError;
  if (!other.paramEvaluators.empty()) {
    const ParametricParameterEvaluator &eval = other.paramEvaluators.back();
    this->paramEvaluators.push_back(ParametricParameterEvaluator(
        eval.getDeclBindings(), eval.getIndexBindings(),
        eval.getNumIndexBindings(), eval.inputDepth));
    this->paramEvaluators.back().setEvaluationContext(this);
  }
}

/// Given a generic parameter expression, simplify it by folding the
/// expression according to known parameter values.  This returns an error if
/// the expression cannot be folded for one reason or another.
ErrorTreeOr<Attribute>
ParametricIREvaluator::concretizeParameterExpr(PImplNode *parent, Location loc,
                                               Attribute expr) {
  // FIXME: Refactor ParametricParameterEvaluator for better error propagation.
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

ErrorTreeOr<Type>
ParametricIREvaluator::concretizeParameterExpr(PImplNode *parent, Location loc,
                                               Type expr) {
  // FIXME: Refactor ParametricParameterEvaluator for better error propagation.
  this->parent = parent;
  errorLoc = loc;
  std::optional<ErrorTree> error;
  emitError = [&](ErrorTree err) { error = std::move(err); };

  Type result = getReboundType(expr);
  if (error)
    return std::move(*error);
  return result;
}
