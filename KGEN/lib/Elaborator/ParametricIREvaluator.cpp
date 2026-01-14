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
#include "KGEN/POPDialect/POPUtils.h"
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
  return elaborator->options.elaborationErrorLimit;
}

bool ParametricIREvaluator::getElabErrorIncludePrelude() {
  return elaborator->options.elaborationErrorIncludePrelude;
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
                          elaborator->options.elaborationErrorIncludePrelude);

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

  // If the struct generator has input params, we need to provide an
  // IREvaluator for concretizing the witness entries.
  ParametricIREvaluator nestedEvaluator = createNestedEvaluator(genNode);
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

//===----------------------------------------------------------------------===//
// Struct Field Reflection Evaluators
//===----------------------------------------------------------------------===//
//
// NOTE: These evaluators differ from the ones in IREvaluator.cpp (main
// elaborator) in two key ways:
//
// 1. The main elaborator waits for a struct instance to be fully concretized
//    before doing reflection (via getConcreteStructTypeInstance), which may
//    return null if the instance is not yet ready. The parametric elaborator
//    does not wait and directly accesses the StructGeneratorOp.
//
// 2. The parametric evaluators use getReboundType() on field types to handle
//    nested parametric types correctly. The main elaborator works with already
//    concrete types so this is not needed.
//
//===----------------------------------------------------------------------===//

FailureOr<std::pair<StructInstanceType, PParamNode *>>
ParametricIREvaluator::resolveStructInstanceType(TypedAttr typeRef,
                                                 StringRef funcName) {
  // Unwrap the type reference to get to the underlying TypeInstanceRefAttr.
  typeRef = getTypeRefForTypeValueIfResolved(typeRef);
  auto instanceRef = dyn_cast_if_present<TypeInstanceRefAttr>(typeRef);
  if (!instanceRef) {
    emitError({*errorLoc, Twine(funcName) + " requires a struct type"});
    return failure();
  }

  PParamNode *genNode =
      elaborator->lookupImplNode(instanceRef.getSymbol())->parent;
  // Always look up witness tables from the StructGeneratorOp, since we cannot
  // block on the instance's completion in the parametric elaborator.
  StructGeneratorOp gen = cast<StructGeneratorOp>(genNode->gen);

  // Unlike IREvaluator::resolveStructInstanceType, this always returns a valid
  // StructInstanceType since we don't wait for concretization. The generator's
  // value domain type is always available.
  auto structType = cast<StructInstanceType>(gen.getValueDomainType());
  return {std::make_pair(structType, genNode)};
}

ParametricIREvaluator
ParametricIREvaluator::createNestedEvaluator(PParamNode *genNode) {
  ParametricIREvaluator nestedEvaluator(*elaborator, parent);
  nestedEvaluator.setErrorLoc(*errorLoc);
  nestedEvaluator.pushParamValues(genNode->inputParams, true);
  for (auto [param, value] :
       llvm::zip(genNode->gen.getInputParams(), genNode->inputParams)) {
    nestedEvaluator.overwriteDeclBinding(param, value);
  }
  return nestedEvaluator;
}

FailureOr<TypedAttr>
ParametricIREvaluator::evaluateStructFieldTypesAttr(StructFieldTypesAttr attr) {
  FailureOr<std::pair<StructInstanceType, PParamNode *>> structTypeOr =
      resolveStructInstanceType(attr.getTypeValue(), "struct_field_types");
  if (failed(structTypeOr))
    return failure();
  StructInstanceType structType = structTypeOr->first;
  PParamNode *genNode = structTypeOr->second;

  // Build variadic of field types
  SmallVector<TypedAttr> fieldTypes;
  ParametricIREvaluator nestedEvaluator = createNestedEvaluator(genNode);
  for (StructDefFieldAttr field : structType.getFields()) {
    // Wrap field type as a TypeParamAttr.
    TypedAttr fieldType =
        nestedEvaluator.getReboundAttribute(field.getTypeValue());
    fieldTypes.push_back(fieldType);
  }

  return {VariadicAttr::get(fieldTypes, attr.getType())};
}

FailureOr<TypedAttr>
ParametricIREvaluator::evaluateStructFieldNamesAttr(StructFieldNamesAttr attr) {
  FailureOr<std::pair<StructInstanceType, PParamNode *>> structTypeOr =
      resolveStructInstanceType(attr.getTypeValue(), "struct_field_names");
  if (failed(structTypeOr))
    return failure();
  StructInstanceType structType = structTypeOr->first;

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

FailureOr<TypedAttr> ParametricIREvaluator::evaluateStructFieldIndexByNameAttr(
    StructFieldIndexByNameAttr attr) {
  FailureOr<std::pair<StructInstanceType, PParamNode *>> structTypeOr =
      resolveStructInstanceType(attr.getTypeValue(),
                                "struct_field_index_by_name");
  if (failed(structTypeOr))
    return failure();
  StructInstanceType structType = structTypeOr->first;

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

FailureOr<TypedAttr> ParametricIREvaluator::evaluateStructFieldTypeByNameAttr(
    StructFieldTypeByNameAttr attr) {
  FailureOr<std::pair<StructInstanceType, PParamNode *>> structTypeOr =
      resolveStructInstanceType(attr.getTypeValue(),
                                "struct_field_type_by_name");
  if (failed(structTypeOr))
    return failure();
  StructInstanceType structType = structTypeOr->first;
  PParamNode *genNode = structTypeOr->second;

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
  ParametricIREvaluator nestedEvaluator = createNestedEvaluator(genNode);
  for (StructDefFieldAttr field : structType.getFields())
    if (field.getName().getValue() == fieldName)
      return nestedEvaluator.getReboundAttribute(field.getTypeValue());

  // Field not found - emit compile error
  emitError({*errorLoc, "struct '" + mlir::debugString(structType) +
                            "' has no field named '" + fieldName + "'"});
  return failure();
}

ParamNodeBase *
ParametricIREvaluator::lookupParamNodeBase(SymbolRefAttr symbol) {
  return elaborator->lookupImplNode(symbol)->parent;
}

GeneratorOp ParametricIREvaluator::getGenerator(SymbolRefAttr symbol) {
  return elaborator->oldSymTab.lookup<GeneratorOp>(
      cast<FlatSymbolRefAttr>(symbol).getAttr());
}

ErrorTreeOr<std::pair<StringAttr, GeneratorOp>>
ParametricIREvaluator::getExpectedMangledName(
    Location errorLoc, StringRef errorContext, TypedAttr symCst,
    bool allowParametric, bool sanitize,
    function_ref<std::string(StringRef)> getPrefix) {
  return elaborator->getExpectedMangledName(
      errorLoc, errorContext, symCst, allowParametric, sanitize, getPrefix);
}

ErrorOr<CrossDeviceFunction>
ParametricIREvaluator::compileAsm(MLIRContext *ctx, GeneratorOp func,
                                  SymbolConstantAttr symbol, StringAttr name,
                                  TargetInfoAttr target, EmitAs emissionKind,
                                  EmissionOptions emissionOptions) {
  SymbolTable symtabCopy = elaborator->oldSymTab;
  return elaborator->compileAsmFn(
      func, symbol, name, symtabCopy, target, emissionKind, emissionOptions,
      elaborator->options, elaborator->getOptions());
}

void ParametricIREvaluator::addDeferredFunction(OwningOpRef<FuncOp> func) {
  elaborator->addDeferredFunction(std::move(func));
}

ImplNodeBase *ParametricIREvaluator::getParentNode() { return parent; }

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
        eval.getDeclBindings(), eval.getIndexBindings(), eval.inputDepth));
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
