//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LIT dialect operations.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/HLCFDialect/HLCFOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/KGENDialect/ParameterReplacer.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/LITDialect/SpecialFunctions.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/Compiler/Properties.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// Given an insertion point in a block, scan up the parent hierarchy to see if
/// this block is nested under the TryOp region that will handle a 'raise'd
/// error, or if this is in a function that is allowed to raise.  This returns
/// the TryOp or FuncOp if found, or null if raise is not valid.
Operation *LIT::findOpProcessingRaise(Block *currentBlock) {
  Operation *parentOp;
  while (currentBlock && (parentOp = currentBlock->getParentOp())) {
    // If we find a throwing function, return it.
    if (auto funcOp = dyn_cast<LIT::FuncOp>(parentOp))
      return funcOp.isThrows() ? funcOp : nullptr;

    if (auto tryOp = dyn_cast<TryOp>(parentOp)) {
      Region &tryBody = tryOp.getTryRegion();
      if (!tryBody.empty() && &tryBody.front() == currentBlock) {
        // If the except region has an UnreachableOp in it, then this is not
        // allowed to raise.  This must be for a 'with' or something else that
        // needs a finally but isn't itself in a throwing region.
        auto &exceptRegion = tryOp.getExceptRegion();
        if (exceptRegion.empty() ||
            !isa<UnreachableOp>(exceptRegion.front().front()))
          return tryOp;
      }
    }
    currentBlock = parentOp->getBlock();
  }
  return nullptr;
}

SymbolRefAttr LIT::getFullyResolvedSymbolRef(mlir::SymbolOpInterface op) {
  SmallVector<FlatSymbolRefAttr> symbols;
  do {
    symbols.push_back(FlatSymbolRefAttr::get(op.getNameAttr()));
  } while ((op = dyn_cast<mlir::SymbolOpInterface>(op->getParentOp())));

  // Form a reference from the symbols we collected.
  if (symbols.size() == 1)
    return symbols.front();
  std::reverse(symbols.begin(), symbols.end());
  return SymbolRefAttr::get(symbols[0].getAttr(),
                            ArrayRef(symbols).drop_front());
}

Type LIT::getSignatureUserResultType(SignatureType sigType,
                                     ArrayRef<Type> argTypes, Type resultType) {
  // If this function is a memory only type, return the by-ref result.
  if (sigType.hasMemoryOnlyResult())
    return cast<RefType>(argTypes.front()).getElementType();

  // Otherwise it is the normal result.
  if (sigType.isThrows())
    return cast<VariantType>(resultType).getType(1);
  return resultType;
}

std::pair<LITSignatureType, ParameterExprArrayAttr>
LIT::getUnboundSpecializedSignature(LITSignatureType type,
                                    ParameterExprArrayAttr bindings) {
  if (bindings.empty())
    return {type, bindings};

  // KGEN expects different bindings types than Lit can provide. Rebind the
  // parameters to the expected types.
  SmallVector<TypedAttr> unboundBindings;
  ParameterEvaluator evaluator;
  for (auto [binding, type] : llvm::zip(bindings, type.getInputParamTypes())) {
    TypedAttr value = binding;
    Type unboundType = evaluator.getReboundType(type);
    if (unboundType != value.getType())
      value = ParamOperatorAttr::get(POC::Rebind, value, unboundType);
    evaluator.addInputValue(value);
    unboundBindings.push_back(value);
  }
  type = type.getSpecializedSignature(
      unboundBindings, [&]() -> InFlightDiagnostic {
        return mlir::emitError(UnknownLoc::get(type.getContext()));
      });
  assert(type && "bad bindings specified");
  return {type,
          ParameterExprArrayAttr::get(type.getContext(), unboundBindings)};
}

//===----------------------------------------------------------------------===//
// FileModuleOp
//===----------------------------------------------------------------------===//

void FileModuleOp::build(OpBuilder &builder, OperationState &state,
                         StringAttr name, StringAttr sourceName) {
  state.addAttribute(getSymNameAttrName(state.name), name);
  state.addAttribute(getSourceNameAttrName(state.name), sourceName);
  state.addRegion()->push_back(new Block());
}

/// Modules don't have input parameters but do define a parameter scope.
ArrayRef<ParamDeclAttr> FileModuleOp::getInputParams() { return {}; }

/// Modules don't have result parameters.
ArrayRef<ParamDeclAttr> FileModuleOp::getResultParams() { return {}; }

//===----------------------------------------------------------------------===//
// PackageOp
//===----------------------------------------------------------------------===//

void PackageOp::build(OpBuilder &builder, OperationState &state,
                      StringAttr name, StringAttr sourceName) {
  state.addAttribute(getSymNameAttrName(state.name), name);
  state.addAttribute(getSourceNameAttrName(state.name), sourceName);
  state.addAttribute(getArchivesAttrName(state.name),
                     PackageArchiveArrayAttr::get(name.getContext(), {}));
  state.addRegion()->push_back(new Block());
}

/// Packages don't have input parameters but do define a parameter scope.
ArrayRef<ParamDeclAttr> PackageOp::getInputParams() { return {}; }

/// Packages don't have result parameters.
ArrayRef<ParamDeclAttr> PackageOp::getResultParams() { return {}; }

LogicalResult PackageOp::verify() {
  for (Operation &op : *getBody()) {
    if (!isa<FileModuleOp, PackageOp, UnresolvedImportOp,
             UnresolvedWildcardImportOp>(op)) {
      return emitOpError("expected only `lit.file_module`, `lit.package`, "
                         "`lit.unresolved_import`, or "
                         "`lit.unresolved_wildcard_import` in its body")
          .attachNote(op.getLoc())
          .append("see operation defined here");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

static ParseResult
parseLifetimeParams(AsmParser &p, ParameterExprArrayAttr &implicitLifetimes) {
  SmallVector<TypedAttr> values;
  auto lifetimeType = LifetimeType::get(p.getContext());
  if (p.parseCommaSeparatedList(
          AsmParser::Delimiter::OptionalSquare, [&]() -> ParseResult {
            return parseParamValue(p, values.emplace_back(), lifetimeType);
          }))
    return failure();
  implicitLifetimes = ParameterExprArrayAttr::get(p.getContext(), values);
  return success();
}

static void printLifetimeParams(AsmPrinter &p, Operation *op,
                                ParameterExprArrayAttr implicitLifetimes) {
  if (implicitLifetimes.empty())
    return;
  p << '[';
  llvm::interleaveComma(implicitLifetimes, p,
                        [&](TypedAttr value) { printParamValue(p, value); });
  p << ']';
}

/// Infer call operation operand and result types from the signature,
/// substituting implicit lifetime parameters.
template <typename CalleeT>
static ParseResult
parseCallOpTypes(AsmParser &p, SmallVectorImpl<Type> &operandTypes,
                 SmallVectorImpl<Type> &resultTypes, CalleeT callee,
                 ArrayRef<TypedAttr> implicitLifetimes) {
  SignatureType calleeType;
  if constexpr (std::is_same_v<Type, CalleeT>)
    calleeType = cast<SignatureType>(callee);
  else
    calleeType = cast<SignatureType>(callee.getType());

  FunctionType values;
  if (implicitLifetimes.empty()) {
    values = calleeType.getValues();
  } else {
    auto calleeLITType = cast<LITSignatureType>(calleeType);
    if (calleeLITType.getNumImplicitLifetimeDecls() != implicitLifetimes.size())
      return p.emitError(p.getNameLoc())
             << implicitLifetimes.size()
             << " lifetimes specified, but signature expected "
             << calleeLITType.getNumImplicitLifetimeDecls();

    values = calleeLITType.substituteImplicitLifetimesIntoValues(
        implicitLifetimes, [&] { return p.emitError(p.getNameLoc()); });
    if (!values)
      return failure();
  }

  llvm::append_range(operandTypes, values.getInputs());
  llvm::append_range(resultTypes, values.getResults());
  return success();
}

/// Nothing to do on print.
template <typename CalleeT>
static void printCallOpTypes(AsmPrinter &, Operation *, TypeRange, TypeRange,
                             CalleeT, ArrayRef<TypedAttr>) {}

static ParseResult
parseCallOp(OpAsmParser &p, SymbolConstantAttr &calleeCst,
            ParameterExprArrayAttr &implicitLifetimes,
            SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
            SmallVectorImpl<Type> &operandTypes,
            SmallVectorImpl<Type> &resultTypes) {
  SymbolRefAttr callee;
  ParameterExprArrayAttr paramValues;
  ParamDeclArrayAttr paramDecls;
  if (p.parseAttribute(callee) || parseLifetimeParams(p, implicitLifetimes),
      parseCallOpParams(p, paramValues, paramDecls) ||
          p.parseOperandList(operands, AsmParser::Delimiter::Paren) ||
          p.parseColon())
    return failure();
  if (!paramDecls.empty()) {
    return p.emitError(p.getCurrentLocation(),
                       "result parameters are not supported");
  }

  SignatureType signature;
  FunctionType functionType;
  if (parseKGENSignature(p, paramDecls, functionType, signature))
    return failure();
  calleeCst = SymbolConstantAttr::get(callee, paramValues, signature);
  if (failed(parseCallOpTypes(p, operandTypes, resultTypes, calleeCst,
                              implicitLifetimes)))
    return failure();
  return success();
}

static void printCallOp(OpAsmPrinter &p, Operation *op,
                        SymbolConstantAttr calleeCst,
                        ParameterExprArrayAttr implicitLifetimes,
                        ValueRange operands, TypeRange operandTypes,
                        TypeRange resultTypes) {
  p << calleeCst.getSymbol();
  printLifetimeParams(p, op, implicitLifetimes);
  printCallOpParams(p, op, calleeCst.getParamValues(), /*resultDecls=*/{},
                    calleeCst.getType().getResultParamTypes());
  p << '(';
  p.printOperands(operands);
  p << ") : ";
  printSignatureValues(
      p, FunctionType::get(op->getContext(), operandTypes, resultTypes),
      calleeCst.getType());
}

template <typename OpT>
static LogicalResult verifyLifetimeParams(OpT op, LITSignatureType sig) {
  size_t numImplicit = sig.getMetadata().getNumImplicitLifetimeDecls();
  size_t numParams = op.getImplicitLifetimes().size();
  if (numParams == numImplicit)
    return success();
  return op->emitOpError("operation has ")
         << numParams
         << " bindings for implicit lifetime parameters, but callee "
            "expected "
         << numImplicit;
}

template <typename OpT>
static LogicalResult verifyCallOp(OpT op, LITSignatureType sig,
                                  ValueRange operands, TypeRange results) {
  FunctionType values = sig.substituteImplicitLifetimesIntoValues(
      op.getImplicitLifetimes(), [&] { return op.emitOpError(); });
  if (!values)
    return failure();

  auto verifyTypes = [&](StringRef kind, TypeRange types,
                         TypeRange expected) -> LogicalResult {
    for (auto [i, type, exp] : llvm::enumerate(types, expected)) {
      if (type == exp)
        continue;
      return op.emitOpError("callee expected call ")
             << kind << " #" << i << " to be " << exp << " but got " << type;
    }
    return success();
  };

  if (failed(verifyTypes("argument", operands, values.getInputs())) ||
      failed(verifyTypes("result", results, values.getResults())))
    return failure();
  return success();
}

LogicalResult LIT::CallOp::verify() {
  auto sig = dyn_cast<LITSignatureType>(getCallee().getType());
  if (!sig)
    return emitOpError("callee type must be a `!lit.signature`");
  if (failed(verifyLifetimeParams(*this, sig)))
    return failure();
  return verifyCallOp(*this, sig, getOperands(), getResultTypes());
}

void LIT::CallOp::setCalleeAttr(TypedAttr callee) {
  setCalleeAttr(cast<SymbolConstantAttr>(callee));
}

OperandRange LIT::CallOp::getArgOperands() { return getOperands(); }

MutableOperandRange LIT::CallOp::getArgOperandsMutable() {
  return getOperandsMutable();
}

mlir::CallInterfaceCallable LIT::CallOp::getCallableForCallee() {
  return getCallee().getSymbol();
}

void LIT::CallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  auto symbol = callee.get<SymbolRefAttr>();
  setCalleeAttr(SymbolConstantAttr::get(symbol, getCallee().getType()));
}

//===----------------------------------------------------------------------===//
// CallParamOp
//===----------------------------------------------------------------------===//

static ParseResult parseCallee(OpAsmParser &p, TypedAttr &callee) {
  ParamDeclArrayAttr paramDecls;
  if (parseParametricCallee(p, callee, paramDecls))
    return failure();
  if (!paramDecls.empty()) {
    return p.emitError(p.getCurrentLocation(),
                       "operation does not support result parameters");
  }
  return success();
}

static void printCallee(OpAsmPrinter &p, Operation *op, TypedAttr callee) {
  printParametricCallee(p, op, callee, /*paramDecls=*/{});
}

LogicalResult LIT::CallParamOp::verify() {
  auto sig = cast<LITSignatureType>(getCallee().getType());
  if (failed(verifyLifetimeParams(*this, sig)))
    return failure();
  return verifyCallOp(*this, sig, getOperands(), getResultTypes());
}

//===----------------------------------------------------------------------===//
// CallSignatureOp
//===----------------------------------------------------------------------===//

LogicalResult LIT::CallSignatureOp::verify() {
  auto sig = cast<LITSignatureType>(getCallee().getType());
  if (failed(verifyLifetimeParams(*this, sig)))
    return failure();
  return verifyCallOp(*this, sig, getArguments(), getResultTypes());
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

/// If this is a special function like __init__ return the enum that
/// identifies it, otherwise return kNormal.
SpecialFunctionKind SpecialFunctionInfo::getKind(StringRef name) {
  if (name.size() < 5 || !name.starts_with("__") || !name.ends_with("__"))
    return SpecialFunctionKind::kNormal;

#define SF(ENUM, NAME, MINOPERANDS, MAXOPERANDS, EXPRNODE, FLAGS)              \
  if (name == (NAME))                                                          \
    return SpecialFunctionKind::ENUM;
#include "KGEN/LITDialect/SpecialFunctions.def"

  // Otherwise, this declaration isn't known.
  return SpecialFunctionKind::kNormal;
}

/// If this is a special function like __init__ return the enum that
/// identifies it, otherwise return kNormal.
const SpecialFunctionInfo &SpecialFunctionInfo::get(SpecialFunctionKind kind) {
  static const SpecialFunctionInfo infos[] = {
      {nullptr, SpecialFunctionKind::kNormal, /*minNumArguments=*/0,
       /*maxNumArguments=*/-1, /*flags=*/0},
#define SF(ENUM, NAME, MINOPERANDS, MAXOPERANDS, EXPRNODE, FLAGS)              \
  {NAME, SpecialFunctionKind::ENUM, (MINOPERANDS), (MAXOPERANDS), (FLAGS)},
#include "KGEN/LITDialect/SpecialFunctions.def"
  };

  assert(unsigned(kind) < sizeof(infos) / sizeof(infos[0]));
  return infos[unsigned(kind)];
}

/// Return the SpecialFunctionKind ID that indicates if this is a special
/// function like __init__ or __radd__.
SpecialFunctionKind LIT::FuncOp::getSpecialFunctionKind() {
  return (SpecialFunctionKind)getSpecialFnKind();
}
const SpecialFunctionInfo &LIT::FuncOp::getSpecialFunctionInfo() {
  return SpecialFunctionInfo::get(getSpecialFunctionKind());
}

/// Returns the user-defined result type, looking through implicit memory
/// results and stripping off the variant from error throwing results if needed.
Type LIT::FuncOp::getUserResultType() {
  return LIT::getSignatureUserResultType(getSignature(), getArgumentTypes(),
                                         getMLIRResultType());
}

TypedAttr LIT::FuncOp::getBoundReference(ParameterExprArrayAttr bindings) {
  if (!bindings) // We allow null for convenience.
    bindings = ParameterExprArrayAttr::get(getContext(), {});

  // SymbolConstantAttr provides a type for the SymbolRefAttr with the
  // parameters substituted in.  The function reference binds any parameter
  // bindings present on the access (in bindings), which typically concretizes
  // the signature.
  LITSignatureType resultType;
  std::tie(resultType, bindings) =
      getUnboundSpecializedSignature(getFullSignature(), bindings);

  if (ParamDeclAttr decl = getParamDeclAttr()) {
    SmallVector<TypedAttr> bindOperands{ParamDeclRefAttr::get(decl)};
    for (TypedAttr binding : bindings)
      bindOperands.push_back(binding);
    return ParamOperatorAttr::get(POC::BindSignature, bindOperands);
  }

  return SymbolConstantAttr::get(getFullyResolvedSymbolRef(*this), bindings,
                                 resultType);
}

SymbolConstantAttr
LIT::FuncOp::getBoundSymbolRef(ParameterExprArrayAttr bindings) {
  return cast<SymbolConstantAttr>(getBoundReference());
}

bool LIT::FuncOp::isSynthetic() { return getIsSynthetic(); }

// These FuncOp attributes are disallowed while parsing since they can
// be inferred. Likewise while printing we ignore them.
static StringRef disallowedAttrNames[] = {
    "sym_name",    "exportKind",     "isCExported",  "constraints",
    "implements",  "signature",      "functionType", "sym_name",
    "argNames",    "paramNames",     "evaluator",    "defaultImpl",
    "inlineLevel", "paramDecl",      "inputParams",  "resultParams",
    "decorators",  "argPassingKinds"};

static ParseResult parseLITFunctionSignature(
    OpAsmParser &p, SmallVectorImpl<OpAsmParser::Argument> &args,
    ParamDeclArrayAttr &inputParams, ParamDeclArrayAttr &resultParams,
    FunctionType &functionType, LITSignatureType &signature) {
  llvm::SMLoc startLoc = p.getCurrentLocation();

  SmallVector<ParamDeclAttr> lifetimeDecls;
  auto lifetimeType = LifetimeType::get(p.getContext());
  auto parseLifetimeDecl = [&]() -> ParseResult {
    StringAttr name;
    if (parseParamName(p, name))
      return failure();
    lifetimeDecls.push_back(ParamDeclAttr::get(name, lifetimeType));
    return success();
  };
  if (p.parseCommaSeparatedList(AsmParser::Delimiter::OptionalSquare,
                                parseLifetimeDecl))
    return failure();

  SmallVector<StringAttr> paramNames;
  SmallVector<TypedAttr> defaultPosParams;
  SmallVector<TypedAttr> defaultKwOnlyParams;
  SmallVector<PassingKind> paramPassingKinds;
  if (parseOptionalParameterSpec(p, inputParams, resultParams, paramNames,
                                 paramPassingKinds, defaultPosParams,
                                 defaultKwOnlyParams))
    return failure();

  SmallVector<StringAttr> argNames;
  SmallVector<TypedAttr> defaultPosArgs;
  SmallVector<TypedAttr> defaultKwOnlyArgs;
  SmallVector<ValueInputConvention> inputConventions;

  PassingKindParser passingKindParser(p);
  auto parseArg = [&](SmallVectorImpl<Type> &argTypes) -> ParseResult {
    if (OptionalParseResult res = passingKindParser.parseOptionalStarSlash();
        res.has_value())
      return res.value();

    // Parse the ssa name first.
    OpAsmParser::Argument &arg = args.emplace_back();
    StringAttr &argName = argNames.emplace_back();
    if (p.parseOperand(arg.ssaName, /*allowResultNumber=*/false))
      return failure();
    // A user defined name might follow in brackets, e.g. `%arg0[someName]`; if
    // omitted, we just use the SSA name.
    if (succeeded(p.parseOptionalLSquare())) {
      // The user defined names might be escaped, since we allow arbitrary
      // identifiers, e.g.: `%arg1[*"!415weirdname"]`.
      if (parseParamName(p, argName) || p.parseRSquare())
        return failure();
    } else {
      // The parsed SSA name comes prepended with '%', so drop it.
      argName = p.getBuilder().getStringAttr((arg.ssaName.name.drop_front()));
    }

    // A colon and type should come next, followed by an optional location and
    // input convention.
    if (p.parseColonType(arg.type) ||
        p.parseOptionalLocationSpecifier(arg.sourceLoc) ||
        parseInputConvention(p, inputConventions.emplace_back(),
                             ValueInputConvention::OwnedInReg))
      return failure();

    // Parse an optional default value.
    TypedAttr defaultVal;
    if (failed(parseOptionalDefaultValue(
            p, defaultVal, arg.type,
            SignatureType::hasAddress(inputConventions.back()))))
      return failure();
    if (defaultVal) {
      if (passingKindParser.isCurrentKwOnly())
        defaultKwOnlyArgs.emplace_back(defaultVal);
      else
        defaultPosArgs.emplace_back(defaultVal);
    }

    argTypes.push_back(arg.type);
    return success();
  };

  FnEffects effects;
  if (failed(parseSignatureValues(p, parseArg, functionType, effects,
                                  /*optionalResultList=*/true)))
    return failure();

  SmallVector<PassingKind> argPassingKinds;
  passingKindParser.populatePassingKinds(argPassingKinds);

  auto metadata = FnMetadataAttr::get(
      p.getContext(), argNames, argPassingKinds, paramNames, paramPassingKinds,
      defaultPosArgs, defaultPosParams, defaultKwOnlyArgs, defaultKwOnlyParams,
      lifetimeDecls.size());
  signature = SignatureType::remapToSignature(
      inputParams, resultParams, functionType, inputConventions, effects,
      metadata, [&] { return p.emitError(startLoc); });
  if (!signature)
    return failure();

  // Replace named implicit lifetime parameter references with index-based
  // references in the signature.
  signature = signature.replaceImplicitLifetimesWithIndexes(lifetimeDecls);

  // Prepend the lifetime declarations.
  llvm::append_range(lifetimeDecls, inputParams);
  inputParams = ParamDeclArrayAttr::get(p.getContext(), lifetimeDecls);
  return success();
}

static void printLITFunctionSignature(OpAsmPrinter &p, Region *region,
                                      ArrayRef<StringAttr> argNames,
                                      ArrayRef<ParamDeclAttr> inputParams,
                                      ArrayRef<ParamDeclAttr> resultParams,
                                      FunctionType functionType,
                                      LITSignatureType signature) {
  ArrayRef<ParamDeclAttr> lifetimeDecls =
      inputParams.drop_back(signature.getNumInputParams());
  if (!lifetimeDecls.empty()) {
    p << '[';
    llvm::interleaveComma(lifetimeDecls, p, [&](ParamDeclAttr decl) {
      printParamName(p, decl.getName());
    });
    p << ']';
  }

  ParameterEvaluator evaluator;
  printOptionalParameterSpec(p, inputParams.drop_front(lifetimeDecls.size()),
                             resultParams, signature.getParamNames(),
                             signature.getParamPassingKinds(),
                             signature.getDefaultPosParams(),
                             signature.getDefaultKwOnlyParams(), evaluator);

  // Substitute input and result parameters when printing default arguments.
  ArrayRef<TypedAttr> defaultPosArgs = signature.getDefaultPosArgs();
  ArrayRef<PassingKind> argPassingKinds = signature.getArgPassingKinds();
  size_t numInputs = signature.getNumInputs();
  size_t defaultPosEnd = countNumPositional(argPassingKinds);
  size_t defaultPosStart = defaultPosEnd - defaultPosArgs.size();

  ArrayRef<TypedAttr> defaultKwOnlyArgs = signature.getDefaultKwOnlyArgs();
  size_t defaultKwOnlyEnd = numInputs - countNumImplicitKinds(argPassingKinds);
  size_t defaultKwOnlyStart = defaultKwOnlyEnd - defaultKwOnlyArgs.size();

  PassingKindPrinter passingKindPrinter(p, numInputs, '|');
  auto printElt = [&](unsigned i) {
    passingKindPrinter.printOptionalStarSlash(argPassingKinds[i], i);

    // Print the SSA name first, which might have been automatically uniqued.
    BlockArgument arg = region->getArgument(i);
    std::string ssaName;
    llvm::raw_string_ostream ss(ssaName);
    p.printOperand(arg, ss);
    p << ssaName;

    // If different from the SSA name (e.g. because it was uniqued, or because
    // it contains characters that need escaping), we also print the
    // user-defined argument name in brackets.
    StringAttr argName = argNames[i];
    if (StringRef(ssaName).drop_front() != argName) {
      p << "[";
      printParamName(p, argName);
      p << "]";
    }

    // Finally, we print the type after a colon.
    p << ": ";
    p.printType(arg.getType());

    // Then we print the optional location before and input convention.
    p.printOptionalLocationSpecifier(arg.getLoc());
    printInputConvention(p, signature.getInputConvention(i),
                         ValueInputConvention::OwnedInReg);

    if (i >= defaultPosStart && i < defaultPosEnd) {
      p << " = ";
      printParamValue(p, cast<TypedAttr>(evaluator.getReboundAttribute(
                             defaultPosArgs[i - defaultPosStart])));
    } else if (i >= defaultKwOnlyStart && i < defaultKwOnlyEnd) {
      p << " = ";
      printParamValue(p, cast<TypedAttr>(evaluator.getReboundAttribute(
                             defaultKwOnlyArgs[i - defaultKwOnlyStart])));
    }

    // Check if we are at the end; if so, we might still have to print a '/'.
    passingKindPrinter.printOptionalTrailingSlash(i);
  };
  printSignatureValues(p, printElt, functionType, signature,
                       /*optionalResultList=*/true);
}

/// Parses a LIT Generator.
ParseResult LIT::FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  ExportKindAttr exportKind;
  if (parseSymbolExport(parser, exportKind))
    return failure();
  result.addAttribute(getExportKindAttrName(result.name), exportKind);

  // Parse the name as a symbol or a parameter declaration.
  StringAttr nameAttr;
  bool isParamDecl = false;
  if (parser.parseOptionalSymbolName(nameAttr)) {
    if (parseParamName(parser, nameAttr))
      return failure();
    isParamDecl = true;
  }
  result.addAttribute(getSymNameAttrName(result.name), nameAttr);

  // Parse the function signature.
  SmallVector<OpAsmParser::Argument> entryArgs;
  ParamDeclArrayAttr inputParams, resultParams;
  FunctionType functionType;
  LITSignatureType signature;
  if (parseLITFunctionSignature(parser, entryArgs, inputParams, resultParams,
                                functionType, signature))
    return failure();

  // Parse additional function attributes.
  ConstraintArrayAttr constraints;
  InlineLevelAttr inlineLevel;
  DecoratorsAttr decorators;
  if (parseOptionalInline(parser, inlineLevel) ||
      parseOptionalConstraints(parser, constraints) ||
      parseOptionalDecorators(parser, decorators))
    return failure();
  result.addAttribute(getInlineLevelAttrName(result.name), inlineLevel);
  result.addAttribute(getConstraintsAttrName(result.name), constraints);
  result.addAttribute(getDecoratorsAttrName(result.name), decorators);
  result.addAttribute(getInputParamsAttrName(result.name), inputParams);
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(functionType));
  if (isParamDecl)
    result.addAttribute(getParamDeclAttrName(result.name),
                        ParamDeclAttr::get(nameAttr, signature));

  // If function attributes are present, parse them.
  NamedAttrList parsedAttributes;
  llvm::SMLoc attributeDictLocation = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(parsedAttributes))
    return failure();

  result.addAttribute(getSignatureAttrName(result.name),
                      TypeAttr::get(signature));

  // Disallow attributes that are inferred from elsewhere in the attribute
  // dictionary.
  for (StringRef disallowed : disallowedAttrNames) {
    if (parsedAttributes.get(disallowed))
      return parser.emitError(attributeDictLocation, "'")
             << disallowed
             << "' is an inferred attribute and should not be specified in the "
                "explicit attribute dictionary";
  }
  result.attributes.append(parsedAttributes);

  // Parse the required function body.
  Region *region = result.addRegion();
  if (parser.parseRegion(*region, entryArgs))
    return failure();

  return success();
}

// Print the LIT::FuncOp using the shared printing logic.
void LIT::FuncOp::print(OpAsmPrinter &p) {
  using namespace mlir::function_interface_impl;

  // Print the operation and the function name.
  printSymbolExport(p, *this, getExportKindAttr());
  p << ' ';
  if (ParamDeclAttr decl = getParamDeclAttr())
    printParamName(p, decl.getName());
  else
    p.printSymbolName(getSymName());

  // Print the function arguments. Here we need all the use defined names.
  printLITFunctionSignature(p, &getBodyRegion(), getSignature().getArgNames(),
                            getInputParams(), getResultParams(),
                            getFunctionType(), getSignature());
  printOptionalInline(p, getInlineLevel());
  printOptionalConstraints(p, *this, getConstraints());
  printOptionalDecorators(p, *this, getDecorators());

  // Don't print the following in lit.func.
  SmallVector<StringRef> ignoredAttrNames(
      (ArrayRef<StringRef>(disallowedAttrNames)));
  if (getLLVMMetadata().empty())
    ignoredAttrNames.push_back(getLLVMMetadataAttrName());

  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                     ignoredAttrNames);

  p << ' ';
  p.printRegion(getBodyRegion(), /*printEntryBlockArgs=*/false);
}

// Name the arguments of the region with the argument names.
void LIT::FuncOp::getAsmBlockArgumentNames(
    Region &region, llvm::function_ref<void(Value, StringRef)> setNameFn) {
  if (region.empty())
    return;

  // Set a name for each argument.
  auto resName = StringAttr::get(getContext(), "__result__");
  for (auto [arg, name] :
       llvm::zip(getBody()->getArguments(), getSignature().getArgNames())) {
    // If the user defined name is short and simple, we use it for the SSA names
    // to make testing a bit easier. Otherwise we use 'arg' and let the
    // interface unique the name.
    bool shouldSugarSSA = name == resName || name.size() <= 5;
    setNameFn(arg, shouldSugarSSA ? name.strref() : "arg");
  }
}

LogicalResult LIT::FuncOp::verify() {
  // Check that the number of argument labels matches the number of argument
  // types.
  if (getSignature().getMetadata().getArgNames().size() !=
      getFunctionType().getNumInputs())
    return emitOpError("incorrect number of value parameter labels");

  if (isExternal()) {
    if (!llvm::hasSingleElement(*getBody()) ||
        !isa<LIT::ExternFuncOp>(&getBody()->front()))
      return emitOpError("expected external function body to contain a single "
                         "`lit.extern_func`");
    if (!getPreElaborationNameAttr())
      return emitOpError(
          "external function requires attribute 'preElaborationName'");
  }
  // Verify order of positional-only, pos-or-kw, and keyword-only args.
  PassingKind prevPassingKind = PassingKind::PosOnly;
  for (PassingKind passingKind : getSignature().getArgPassingKinds()) {
    if (prevPassingKind != passingKind) {
      if (prevPassingKind == PassingKind::KwOnly) {
        return emitOpError(
            "keyword-only argument must follow all other arguments");
      }
      if (prevPassingKind == PassingKind::PosOrKw &&
          passingKind == PassingKind::PosOnly) {
        return emitOpError(
            "positional-only argument cannot follow positional-or-keyword");
      }
    }
  }

  // Verify the correct number of input parameters.
  if (getSignature().getNumInputParams() +
          getSignature().getNumImplicitLifetimeDecls() !=
      getInputParams().size())
    return emitOpError("incorrect number of input params: have ")
           << getInputParams().size() << ", but expected "
           << getSignature().getNumImplicitLifetimeDecls()
           << " implicit lifetimes and " << getSignature().getNumInputParams()
           << " input params";

  return success();
}

void LIT::FuncOp::walkDeclarations(function_ref<void(ParamDeclAttr)> walkDecl) {
  if (auto decl = getParamDeclAttr())
    walkDecl(decl);
}

void LIT::FuncOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  if (auto decl = getParamDeclAttr())
    walkDef(decl, &getBodyRegion());
}

void LIT::FuncOp::renameDeclarations(ArrayRef<ParamDeclAttr> decls) {
  if (getParamDecl()) {
    assert(decls.size() == 1);
    setParamDeclAttr(decls.front());
  } else {
    assert(decls.empty());
  }
}

/// This operation has no uses to collect in its current scope.
void LIT::FuncOp::collectParameterUses(function_ref<void(Attribute)> scanAttr,
                                       function_ref<void(Type)> scanType) {}

/// If the specified operation is non-null and contains parameters, collect
/// them into the specified array.
static void collectContextParameters(Operation *op,
                                     SmallVector<ParamDeclAttr> &params) {
  auto decl = dyn_cast_or_null<DeclInterface>(op);
  if (!decl || isa<FuncInterface>(*decl))
    return;
  collectContextParameters(op->getParentOp(), params);
  llvm::append_range(params, decl.getInputParams());
}

SmallVector<ParamDeclAttr>
LIT::FuncOp::collectAllInputParams(bool includeImplLifetimes) {
  SmallVector<ParamDeclAttr> result;
  collectContextParameters(getOperation()->getParentOp(), result);

  auto inputParams = getInputParams();
  if (!includeImplLifetimes)
    inputParams =
        inputParams.drop_front(getSignature().getNumImplicitLifetimeDecls());
  llvm::append_range(result, inputParams);
  return result;
}

LITSignatureType LIT::FuncOp::getFullSignature() {
  LITSignatureType signature = getSignature();

  // Collect contextual params, if there are none, the full signature is the
  // same as the local signature.
  SmallVector<ParamDeclAttr> inputParams;
  collectContextParameters(getOperation()->getParentOp(), inputParams);
  if (inputParams.empty())
    return signature;

  return SignatureType::prependParams(signature, inputParams);
}

void LIT::FuncOp::build(OpBuilder &builder, OperationState &result) {
  MLIRContext *ctx = builder.getContext();

  // Before resolution, we treat the function as having type ()->Error,
  // because parse or other errors forming the signature won't update the
  // representation.  This makes sure that the error case doesn't break
  // invariants (that functions always have a single result).
  auto errorType = builder.getType<TypeCheckErrorType>();
  auto signatureType = LITSignatureType::get(ctx, ArrayRef<Type>(), {errorType},
                                             /*numImplicitLifetimeDecls=*/0);

  // NOTE: We set an attribute named 'sym_namex' here instead of setting
  // 'sym_name' because we don't /know/ the symbol name on construction and need
  // to set it during signature resolution phase of the parser.
  //
  // Unfortunately, we cannot set it to null because that causes the SymbolTable
  // logic to be extremely cranky and breaks other MLIR invariants.
  //
  // We also cannot completely omit the symbol, because ODS is doing some clever
  // stuff to speed up attribute lookup.  That clever stuff requires that a slot
  // is filled in the attr dict, so we set this thing and remove it when the
  // real name is set.
  result.addAttribute("sym_namex", StringArrayAttr::get(ctx, {}));

  result.addAttribute(getExportKindAttrName(result.name),
                      ExportKindAttr::get(ctx, ExportKind::NotExported));
  result.addAttribute(getSignatureAttrName(result.name),
                      TypeAttr::get(signatureType));
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(signatureType.getValues()));
  result.addAttribute(getInputParamsAttrName(result.name),
                      ParamDeclArrayAttr::get(ctx, {}));
  result.addAttribute(getConstraintsAttrName(result.name),
                      ConstraintArrayAttr::get(ctx, {}));
  result.addAttribute(getDecoratorsAttrName(result.name),
                      DecoratorsAttr::get(ctx, {}));
  result.addAttribute(getSpecialFnKindAttrName(result.name),
                      builder.getI8IntegerAttr(0));
  result.addAttribute(getInlineLevelAttrName(result.name),
                      InlineLevelAttr::get(ctx, InlineLevel::Automatic));
  result.addAttribute(getLLVMMetadataAttrName(result.name),
                      DictionaryAttr::get(ctx));

  result.addRegion()->push_back(new Block());
}

/// Build a function in a default configuration, used by member synthesization.
void LIT::FuncOp::build(OpBuilder &builder, OperationState &result,
                        StringAttr name, StringAttr sourceName,
                        SignatureType signature,
                        SpecialFunctionKind specialFnKind) {
  MLIRContext *ctx = builder.getContext();
  mlir::UnitAttr none;
  build(builder, result, name, ParamDeclAttr(), TypeAttr::get(signature),
        TypeAttr::get(signature.getValues()),
        /*inputParams=*/ParamDeclArrayAttr::get(ctx, {}),
        ConstraintArrayAttr::get(ctx, {}), DecoratorsAttr::get(ctx, {}),
        /*isStatic=*/none, /*isParametric=*/none, /*isDef=*/none,
        /*isInherited=*/none, /*isSynthetic=*/none,
        ExportKindAttr::get(ctx, ExportKind::NotExported),
        InlineLevelAttr::get(ctx, InlineLevel::Automatic),
        builder.getI8IntegerAttr(uint8_t(specialFnKind)), FlatSymbolRefAttr(),
        StringAttr(), StringAttr(), sourceName, DocStringAttr(),
        DictionaryAttr::get(ctx));
  result.regions[0]->push_back(new Block());
}

//===----------------------------------------------------------------------===//
// StructDeclOp
//===----------------------------------------------------------------------===//

static ParseResult parseTypeConvention(AsmParser &p, TypeConvention &value) {
  StringRef str;
  value = TypeConvention::MemoryOnly;
  if (succeeded(p.parseOptionalKeyword(
          &str, {stringifyEnum(TypeConvention::MemoryOnly),
                 stringifyEnum(TypeConvention::RegisterPassable),
                 stringifyEnum(TypeConvention::RegisterPassableTrivial)})))
    value = *symbolizeTypeConvention(str);
  return success();
}

static void printTypeConvention(AsmPrinter &p, Operation *op,
                                TypeConvention value) {
  if (value != TypeConvention::MemoryOnly)
    p << stringifyTypeConvention(value);
}

static ParseResult parseStructParameterSpec(AsmParser &p,
                                            ParamDeclArrayAttr &inputParams,
                                            TypeAttr &signature,
                                            TypeLineageArrayAttr &parentTypes) {
  SmallVector<TypedAttr> defaultPosParams;
  SmallVector<TypedAttr> defaultKwOnlyParams;
  SmallVector<StringAttr> paramNames;
  SmallVector<PassingKind> paramPassingKinds;
  ParamDeclArrayAttr resultParams;
  llvm::SMLoc loc = p.getCurrentLocation();
  if (parseOptionalParameterSpec(p, inputParams, resultParams, paramNames,
                                 paramPassingKinds, defaultPosParams,
                                 defaultKwOnlyParams))
    return failure();
  if (!resultParams.empty())
    return p.emitError(loc, "expected no result parameters");
  bool paramVarArg = succeeded(p.parseOptionalKeyword("param_vararg"));

  SmallVector<TypeLineageAttr> parentTypeExprs;
  auto parseTypeAndLineage = [&]() -> ParseResult {
    Type type;
    SmallVector<Type> lineage;
    if (parseParamType(p, type) ||
        p.parseCommaSeparatedList(AsmParser::Delimiter::OptionalSquare, [&] {
          return parseParamType(p, lineage.emplace_back());
        }))
      return failure();
    parentTypeExprs.push_back(TypeLineageAttr::get(type, lineage));
    return success();
  };
  if (p.parseCommaSeparatedList(AsmParser::Delimiter::OptionalParen,
                                parseTypeAndLineage))
    return failure();
  parentTypes = TypeLineageArrayAttr::get(p.getContext(), parentTypeExprs);

  auto sig = TypeSignatureType::remapToSignature(
      [&] { return p.emitError(loc); }, inputParams, paramNames,
      paramPassingKinds, defaultPosParams, defaultKwOnlyParams, paramVarArg);
  if (!sig)
    return failure();
  signature = TypeAttr::get(sig);
  return success();
}

static void printStructParameterSpec(AsmPrinter &p, Operation *op,
                                     ArrayRef<ParamDeclAttr> inputParamDecls,
                                     TypeAttr signature,
                                     ArrayRef<TypeLineageAttr> parentTypes) {
  auto sig = cast<TypeSignatureType>(signature.getValue());
  ParameterEvaluator evaluator;
  printOptionalParameterSpec(
      p, inputParamDecls, {}, sig.getParamNames(), sig.getParamPassingKinds(),
      sig.getDefaultPosParams(), sig.getDefaultKwOnlyParams(), evaluator);
  if (sig.getParamVarArg())
    p << " param_vararg";
  if (!parentTypes.empty()) {
    p << '(';
    llvm::interleaveComma(parentTypes, p, [&](TypeLineageAttr type) {
      printParamType(p, type.getType());
      if (!type.getInheritedFrom().empty()) {
        p << '[';
        llvm::interleaveComma(type.getInheritedFrom(), p,
                              [&](Type type) { printParamType(p, type); });
        p << ']';
      }
    });
    p << ')';
  }
}

bool StructDeclOp::isSynthetic() { return getIsSynthetic(); }

DeclRefType StructDeclOp::bindReference(ArrayRef<TypedAttr> paramValues) {
  SymbolRefAttr symbol = getFullyResolvedSymbolRef(*this);
  TypeSignatureType sig = getSignature();

  if (paramValues.empty()) {
    // Create a fully unbound reference to the type.
    SmallVector<TypedAttr> unbound;
    for (Type type : sig.getInputParamTypes())
      unbound.push_back(UnboundAttr::get(type));
    return DeclRefType::get(symbol, unbound,
                            MetaTypeType::get(symbol, unbound, sig));
  }

  // Compute the resultant signature.
  ArrayRef<PassingKind> passingKinds = sig.getParamPassingKinds();
  ArrayRef<TypedAttr> defaultsPos = sig.getDefaultPosParams();
  size_t numInputs = sig.getNumInputParams();
  size_t defaultPosEnd = countNumPositional(passingKinds);
  size_t defaultPosStart = defaultPosEnd - defaultsPos.size();

  ArrayRef<TypedAttr> defaultsKwOnly = sig.getDefaultKwOnlyParams();
  size_t defaultKwOnlyEnd = numInputs - countNumImplicitKinds(passingKinds);
  size_t defaultKwOnlyStart = defaultKwOnlyEnd - defaultsKwOnly.size();

  SmallVector<Type> newParamTypes;
  SmallVector<StringAttr> newParamNames;
  SmallVector<PassingKind> newPassingKinds;
  SmallVector<TypedAttr> newPosDefaults;
  SmallVector<TypedAttr> newKwOnlyDefaults;
  bool paramVarArg = false;
  for (auto [i, value, type, name, kind] :
       llvm::enumerate(paramValues, sig.getInputParamTypes(),
                       sig.getParamNames(), sig.getParamPassingKinds())) {
    if (::isa<UnboundAttr>(value)) {
      newParamTypes.push_back(type);
      newParamNames.push_back(name);
      newPassingKinds.push_back(kind);

      if (i >= defaultPosStart && i < defaultPosEnd)
        newPosDefaults.push_back(defaultsPos[i - defaultPosStart]);
      else if (i >= defaultKwOnlyStart && i < defaultKwOnlyEnd)
        newKwOnlyDefaults.push_back(defaultsKwOnly[i - defaultKwOnlyStart]);

      if (sig.isVarArg(i))
        paramVarArg = true;
    }
  }
  auto newSig = TypeSignatureType::get(
      getContext(), newParamTypes, newParamNames, newPassingKinds,
      newPosDefaults, newKwOnlyDefaults, paramVarArg);
  return DeclRefType::get(symbol, paramValues,
                          MetaTypeType::get(symbol, paramValues, newSig));
}

/// Verify the debuginfo scope of an op that must be a top-level declaration.
static LogicalResult verifyTopLevelLocScope(Operation *op) {
  Location loc = op->getLoc();

  // If the decl doesn't contain a location scope, we don't verify it.
  auto fusedLoc = dyn_cast<mlir::FusedLocWith<DebugInfo::DIScopeAttr>>(loc);
  if (!fusedLoc)
    return success();

  DebugInfo::DIScopeAttr scope = fusedLoc.getMetadata();
  auto funcScope = dyn_cast<DebugInfo::DIFileAttr>(scope);
  if (funcScope)
    return success();
  return op->emitOpError("must have file scope in location, but got ") << scope;
}

/// Return the debuginfo scope of an op that must be a top-level declaration.
static DebugInfo::DIFileAttr getTopLevelScope(Operation *op) {
  if (auto fusedLoc =
          dyn_cast<mlir::FusedLocWith<DebugInfo::DIFileAttr>>(op->getLoc()))
    return fusedLoc.getMetadata();
  return {};
}

LogicalResult StructDeclOp::verify() {
  if (getFields().getNumArguments())
    return emitOpError("expected declaration body to have no arguments");
  return verifyTopLevelLocScope(*this);
}

DebugInfo::DIScopeAttr StructDeclOp::getLocScope() {
  return getTopLevelScope(*this);
}

/// Verify that there are no duplicate field names.
LogicalResult StructDeclOp::verifyRegions() {
  SmallDenseMap<StringAttr, StructFieldOp, 8> seenFields;
  for (Operation &op : getFields().front()) {
    auto field = dyn_cast<StructFieldOp>(&op);
    if (!field)
      continue;
    auto [it, inserted] = seenFields.try_emplace(field.getNameAttr(), field);
    if (!inserted) {
      return (field.emitError("duplicate struct field ") << field.getNameAttr())
                 .attachNote(it->second.getLoc())
             << "see previous declaration here";
    }
  }
  return success();
}

void StructDeclOp::build(OpBuilder &builder, OperationState &result,
                         StringAttr name) {
  MLIRContext *ctx = builder.getContext();
  build(builder, result, name, TypeAttr::get(TypeSignatureType::get(ctx)),
        ParamDeclArrayAttr::get(ctx, {}), DecoratorsAttr::get(ctx, {}),
        TypeLineageArrayAttr::get(ctx, {}), /*isSynthetic=*/nullptr,
        /*nonmaterializableTarget=*/nullptr,
        /*destructor=*/nullptr, /*moveInit=*/nullptr, /*copyInit=*/nullptr,
        /*closureSignature=*/nullptr, /*docString=*/nullptr);
  result.regions[0]->push_back(new Block());
}

//===----------------------------------------------------------------------===//
// StructFieldOp
//===----------------------------------------------------------------------===//

/// Parse the struct field name as a keyword literal.
static ParseResult parseKeywordAsString(OpAsmParser &p, StringAttr &name) {
  StringRef value;
  if (p.parseKeyword(&value))
    return failure();
  name = p.getBuilder().getStringAttr(value);
  return success();
}

/// Print the struct field name as a keyword literal.
static void printKeywordAsString(OpAsmPrinter &p, Operation *op,
                                 StringAttr name) {
  p << name.getValue();
}

Type StructFieldOp::getReboundType(DeclRefType structSelfType) {
  if (structSelfType.getParamValues().empty())
    return getType();
  ParameterEvaluator evaluator(getParentOp().getInputParams(),
                               structSelfType.getParamValues());
  return evaluator.getReboundType(getType());
}

void StructFieldOp::build(OpBuilder &builder, OperationState &odsState,
                          StringAttr name, Type type) {
  build(builder, odsState, name, type, nullptr);
}

void StructFieldOp::build(OpBuilder &builder, OperationState &odsState,
                          const Twine &name, Type type) {
  build(builder, odsState, builder.getStringAttr(name), type);
}

//===----------------------------------------------------------------------===//
// StructCreateOp
//===----------------------------------------------------------------------===//

/// Lookup the declaration for the struct. When checking field types, we can't
/// directly compare operation types to the struct field types because they are
/// parameterized under different domains. We have to rebind them.
static std::pair<StructDeclOp, ParameterEvaluator>
lookupStructDecl(SymbolTableCollection &symbolTable, Operation *user,
                 DeclRefType ref) {
  auto module = KGENModule::from(user, symbolTable);
  auto decl = module.lookup<StructDeclOp>(ref.getSymbol());
  if (!decl) {
    user->emitOpError("expected to find a struct decl for ") << ref;
    return {};
  }
  ParameterEvaluator evaluator(decl.getInputParams(), ref.getParamValues());
  return {decl, std::move(evaluator)};
}

/// Verify the reference struct type.
LogicalResult
LIT::StructCreateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verify the types of the fields in the operands match those in the
  // struct declaration.
  auto [structDecl, evaluator] =
      lookupStructDecl(symbolTable, *this, getType());
  if (!structDecl)
    return emitOpError("expected to find a struct decl for ") << getType();
  auto fields = structDecl.getFieldDecls();
  unsigned numFields = std::distance(fields.begin(), fields.end());
  if (numFields != getNumOperands())
    return emitOpError("expected ")
           << numFields << " operands but got " << getNumOperands();
  if (getFieldsAttr().size() != numFields)
    return emitOpError("expected ")
           << numFields << " based on the declaration, but got "
           << getFieldsAttr().size();

  for (auto [fieldDecl, fieldAttrInOp, operand, i] :
       llvm::zip(fields, getFieldsAttr(), getOperands(),
                 llvm::seq<unsigned>(0, numFields))) {
    StringAttr nameInDecl = fieldDecl.getNameAttr();
    StringAttr nameInOp = fieldAttrInOp;
    if (nameInDecl != nameInOp) {
      return emitOpError("the field name ")
             << nameInOp << " at the position #" << i
             << " did not match the name " << nameInDecl
             << " in the op declaration.";
    }

    Type reboundType = evaluator.getReboundType(fieldDecl.getType());
    if (reboundType != operand.getType()) {
      return emitOpError("operand #")
             << i << " has type " << operand.getType()
             << " but corresponding struct field " << fieldDecl.getNameAttr()
             << " expected " << reboundType;
    }
  }
  return success();
}

/// Parse a sequence of "field_name=operand" entries.
static ParseResult
parseOperandsAndFields(OpAsmParser &p,
                       SmallVector<OpAsmParser::UnresolvedOperand, 4> &operands,
                       StringArrayAttr &fields) {
  SmallVector<StringAttr> fieldNames;
  if (p.parseCommaSeparatedList(
          OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
            std::string fieldNameStr;
            if (p.parseKeywordOrString(&fieldNameStr) || p.parseEqual() ||
                p.parseOperand(operands.emplace_back()))
              return failure();
            fieldNames.push_back(StringAttr::get(p.getContext(), fieldNameStr));
            return success();
          }))
    return failure();

  fields = StringArrayAttr::get(p.getContext(), fieldNames);
  return success();
}

/// Print a sequence of "field_name=operand" entries.
static void printOperandsAndFields(OpAsmPrinter &p, Operation *op,
                                   OperandRange operands,
                                   StringArrayAttr fields) {
  p << "(";
  llvm::interleaveComma(llvm::zip(fields.getValue(), op->getOperands()), p,
                        [&](const std::tuple<StringAttr, Value> &val) {
                          auto &[fieldName, operand] = val;
                          p << fieldName.getValue() << "=" << operand;
                        });
  p << ")";
}

OpFoldResult LIT::StructCreateOp::fold(FoldAdaptor adaptor) {
  SmallVector<std::tuple<StringAttr, TypedAttr>> values;
  for (auto [name, value] : llvm::zip(getFields(), adaptor.getOperands())) {
    if (!value)
      return {};
    values.emplace_back(name, cast<TypedAttr>(value));
  }
  return LITStructAttr::get(getContext(), values, getType());
}

//===----------------------------------------------------------------------===//
// StructInsertOp
//===----------------------------------------------------------------------===//

LogicalResult
StructInsertOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto [structDecl, evaluator] =
      lookupStructDecl(symbolTable, *this, getType());
  if (!structDecl)
    return emitOpError("expected to find a struct decl for ") << getType();

  for (StructFieldOp fieldDecl : structDecl.getFieldDecls()) {
    if (fieldDecl.getName() != getFieldAttr())
      continue;
    Type reboundType = evaluator.getReboundType(fieldDecl.getType());
    if (reboundType != getValue().getType())
      return emitOpError("cannot insert value of type ")
             << getValue().getType() << " into struct field " << getFieldAttr()
             << " which expected " << reboundType;
    return success();
  }

  return emitOpError("struct ")
         << getType().getSymbol() << " has no field named " << getFieldAttr();
}

OpFoldResult StructInsertOp::fold(FoldAdaptor adaptor) {
  auto value = dyn_cast_if_present<LITStructAttr>(adaptor.getContainer());
  if (!value || !adaptor.getValue())
    return {};
  auto it = llvm::find_if(value.getValues(), [&](const auto &p) {
    return std::get<0>(p) == getFieldAttr();
  });
  if (it == value.getValues().end())
    return {};
  SmallVector<std::tuple<StringAttr, TypedAttr>> values(value.getValues());
  std::get<1>(values[std::distance(value.getValues().begin(), it)]) =
      cast<TypedAttr>(adaptor.getValue());
  return LITStructAttr::get(getContext(), values, getType());
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

static LogicalResult
verifyStructFieldAndType(SymbolTableCollection &symbolTable, Operation *op,
                         DeclRefType ref, StringAttr fieldName, Type type) {
  auto [structDecl, evaluator] = lookupStructDecl(symbolTable, op, ref);
  if (!structDecl)
    return op->emitOpError("struct ") << ref.getSymbol() << " cannot be found";

  for (StructFieldOp fieldDecl : structDecl.getFieldDecls()) {
    if (fieldDecl.getName() != fieldName)
      continue;
    Type reboundType = evaluator.getReboundType(fieldDecl.getType());
    if (reboundType != type)
      return op->emitOpError("cannot extract value of type ")
             << type << " from struct field " << fieldName << " which has type "
             << reboundType;
    return success();
  }

  return op->emitOpError("struct ")
         << ref.getSymbol() << " has no field named " << fieldName;
}

LogicalResult
LIT::StructExtractOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyStructFieldAndType(symbolTable, *this, getContainer().getType(),
                                  getFieldAttr(), getValue().getType());
}

void LIT::StructExtractOp::build(OpBuilder &builder, OperationState &result,
                                 Value structBase, StructFieldOp field) {
  auto structType = cast<DeclRefType>(structBase.getType());
  build(builder, result, field.getReboundType(structType), structBase,
        field.getNameAttr());
}

OpFoldResult LIT::StructExtractOp::fold(FoldAdaptor adaptor) {
  if (auto value = adaptor.getContainer())
    return StructExtractAttr::get(cast<TypedAttr>(value), getFieldAttr(),
                                  getType());

  // Fold
  //  %S = lit.struct.create(a=%a, b=%b)
  //  %x = lit.struct.extract %S[a]
  // into %a.
  if (auto create = getContainer().getDefiningOp<LIT::StructCreateOp>()) {
    for (size_t i = 0, e = create->getNumOperands(); i < e; i++) {
      if (create.getFieldsAttr()[i] == getFieldAttr())
        return create.getOperand(i);
    }
    // A field referred to in the struct.extract op didn't appear in the
    // previous struct.create op - the IR is probably malformed, do not fold
    // anything.
    return {};
  }
  // Fold
  //    %S = lit.struct.insert %x, %S0[a]
  //    %y = lit.struct.extract %S[a]
  // into %x
  if (auto insert = getContainer().getDefiningOp<StructInsertOp>()) {
    if (insert.getFieldAttr() == getFieldAttr())
      return insert.getOperand(0);
  }
  return {};
}

//===----------------------------------------------------------------------===//
// RefStructGEROp
//===----------------------------------------------------------------------===//

LogicalResult
RefStructGEROp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Type structType = getContainer().getType().getElementType();
  return verifyStructFieldAndType(symbolTable, *this,
                                  cast<DeclRefType>(structType), getFieldAttr(),
                                  getResult().getType().getElementType());
}

void RefStructGEROp::build(OpBuilder &builder, OperationState &result,
                           Value structBasePtr, StructFieldOp field) {
  auto refType = cast<RefType>(structBasePtr.getType());
  auto eltType = cast<DeclRefType>(refType.getElementType());
  build(builder, result, refType.getWithElement(field.getReboundType(eltType)),
        field.getNameAttr(), structBasePtr);
}

static ParseResult parseStructGERTypes(AsmParser &p, Type &fieldType,
                                       Type &containerType) {
  llvm::SMLoc loc = p.getCurrentLocation();
  // parse: 'type' from 'type'
  fieldType = RefType::parse(p);
  if (!fieldType || p.parseKeyword("from") || parseParamType(p, containerType))
    return failure();
  auto fieldRefType = dyn_cast<RefType>(fieldType);
  if (!fieldRefType)
    return p.emitError(loc, "expected '!lit.ref' type in !lit.struct.ger");

  // The container type gets wrapped with the same mutability and lifetime as
  // the result element.
  containerType = fieldRefType.getWithElement(containerType);
  return success();
}

static void printStructGERTypes(AsmPrinter &p, Operation *, RefType fieldType,
                                RefType containerType) {
  fieldType.print(p);
  p << " from ";
  if (auto refType = dyn_cast<RefType>(containerType))
    printParamType(p, refType.getElementType());
  else {
    p << "<<ERROR NOT REF CONTAINER>>";
    p.printType(containerType);
  }
}

//===----------------------------------------------------------------------===//
// RefImmutOp
//===----------------------------------------------------------------------===//

OpFoldResult RefImmutOp::fold(RefImmutOp::FoldAdaptor adaptor) {
  // If the operand is already known to be immutable then this is a noop.
  if (getRef().getType().isMutableKnown(false))
    return getRef();
  return {};
}

//===----------------------------------------------------------------------===//
// RefFromPointerOp
//===----------------------------------------------------------------------===//

void RefFromPointerOp::build(OpBuilder &builder, OperationState &result,
                             bool isMut, Value pointer, TypedAttr lifetime,
                             bool startsUninit, bool endsUninit) {
  auto ptr = cast<PointerType>(pointer.getType());
  auto refType = RefType::get(isMut, ptr.getElementType(), lifetime,
                              ptr.getAddressSpace());
  build(builder, result, refType, pointer, startsUninit, endsUninit);
}

//===----------------------------------------------------------------------===//
// TraitDeclOp
//===----------------------------------------------------------------------===//

DebugInfo::DIScopeAttr TraitDeclOp::getLocScope() {
  return getTopLevelScope(*this);
}

void TraitDeclOp::build(OpBuilder &builder, OperationState &result,
                        StringAttr name) {
  MLIRContext *ctx = builder.getContext();
  build(builder, result, name, TypeAttr::get(TypeSignatureType::get(ctx)),
        ParamDeclArrayAttr::get(ctx, {}), TypeLineageArrayAttr::get(ctx, {}),
        /*dtorSig=*/nullptr, /*docString=*/nullptr);
  result.regions[0]->push_back(new Block());
}

TraitType TraitDeclOp::bindReference() {
  return TraitType::get(getFullyResolvedSymbolRef(*this));
}

//===----------------------------------------------------------------------===//
// TryOp
//===----------------------------------------------------------------------===//

static ParseResult parseExceptRegion(OpAsmParser &p, Region &region) {
  SmallVector<OpAsmParser::Argument> args;
  if (p.parseArgumentList(args, AsmParser::Delimiter::Paren,
                          /*allowType=*/true) ||
      p.parseRegion(region, args))
    return failure();
  return success();
}

static void printExceptRegion(OpAsmPrinter &p, Operation *op, Region &region) {
  p << '(';
  llvm::interleaveComma(region.getArguments(), p,
                        [&](BlockArgument arg) { p.printRegionArgument(arg); });
  p << ") ";
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

LogicalResult TryOp::verify() {
  if (getExceptRegion().getNumArguments() < 1)
    return emitOpError("expected except region to have at least one argument");
  return success();
}

void TryOp::getEntryTargets(ArrayRef<Attribute> operands,
                            SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.empty());
  targets.emplace_back(0, getTryRegion().getArguments());
}

ValueRange TryOp::getEntryArguments(std::optional<unsigned> target) {
  if (!target)
    return getResults();
  return getRegion(*target).getArguments();
}

bool TryOp::hasTrivialFinally() {
  Block &finally = getFinallyRegion().front();
  return llvm::hasSingleElement(finally) &&
         isa<TryYieldOp>(finally.getTerminator());
}

//===----------------------------------------------------------------------===//
// TryYieldOp
//===----------------------------------------------------------------------===//

bool TryYieldOp::isParentNode(Operation *op) { return isa<TryOp>(op); }

void TryYieldOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  Region *region = (*this)->getParentRegion();
  // Figure out which region this yield is in.
  if (!isa<TryOp>(region->getParentOp()))
    region = region->getParentRegion();

  // The region indices of the try operation.
  enum { TRY, EXCEPT, ELSE, FINALLY };
  switch (region->getRegionNumber()) {
  case TRY:
    // Yield from the 'try' region branches to the 'else' region.
    targets.emplace_back(ELSE, getOperands());
    break;
  case EXCEPT:
  case ELSE:
    // Yield from either the 'except' or 'else' regions branches back to the
    // parent operation.
    targets.emplace_back(std::nullopt, getOperands());
    break;
  case FINALLY:
    // The finally region is a no-op according to HLCF.
    break;
  default:
    llvm_unreachable("unknown lit.try region");
  }
}

//===----------------------------------------------------------------------===//
// TryRaiseOp
//===----------------------------------------------------------------------===//

bool TryRaiseOp::isParentNode(Operation *op) {
  if (auto tryOp = dyn_cast<TryOp>(op))
    return tryOp.getTryRegion().isAncestor((*this)->getParentRegion());
  return false;
}

void TryRaiseOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  targets.emplace_back(1, getOperands());
}

//===----------------------------------------------------------------------===//
// AliasDeclOp
//===----------------------------------------------------------------------===//

static ParseResult parseAliasDeclOpValue(OpAsmParser &p,
                                         ParamDeclAttr &paramDecl,
                                         TypedAttr &value) {
  return parseParamDeclaration(p, paramDecl, value);
}

static void printAliasDeclOpValue(OpAsmPrinter &p, Operation *,
                                  ParamDeclAttr paramDecl, TypedAttr value) {
  return printParamDeclaration(p, paramDecl, value);
}

void AliasDeclOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  walkDef(getParamDecl(), getValue());
}

LogicalResult AliasDeclOp::verify() {
  if (getParamDecl().getType() == getValue().getType())
    return success();
  return emitOpError("declares a parameter with type ")
         << getParamDecl().getType() << " but parameter expression has type "
         << getValue().getType();
}

//===----------------------------------------------------------------------===//
// LetRegDeclOp
//===----------------------------------------------------------------------===//

void LetRegDeclOp::build(OpBuilder &builder, OperationState &state,
                         Type resultType, StringAttr name) {
  state.addAttribute(getNameAttrName(state.name), name);
  state.addTypes(resultType);
}

void LetRegDeclOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), getName());
}

OpFoldResult LetRegDeclOp::fold(LetRegDeclOp::FoldAdaptor adaptor) {
  return adaptor.getValue();
}

//===----------------------------------------------------------------------===//
// VarLetDeclOp
//===----------------------------------------------------------------------===//

static ParseResult parseVarLetDeclType(AsmParser &p, Type &resultType,
                                       ParamDeclAttr &lifetimeDecl) {
  if (p.parseType(resultType))
    return failure();
  auto refType = dyn_cast<RefType>(resultType);
  if (!refType || !refType.isMutableKnown(true))
    return p.emitError(p.getNameLoc(),
                       "expected a mutable !lit.ref<> result type");
  // The lifetime must be a simple name, which becomes the name we are
  // declaring.
  auto lifetime = dyn_cast<ParamDeclRefAttr>(refType.getLifetime());
  if (!lifetime)
    return p.emitError(p.getNameLoc(),
                       "expected a !lit.ref<> with named lifetime");
  lifetimeDecl = ParamDeclAttr::get(lifetime);
  return success();
}

static void printVarLetDeclType(AsmPrinter &p, Operation *op, Type resultType,
                                ParamDeclAttr decl) {
  p.printType(resultType);
}

void VarLetDeclOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), getName());
}

void VarLetDeclOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  walkDef(getParamDecl(), ParamDefValue());
}

void VarLetDeclOp::build(OpBuilder &b, OperationState &state, Type elementType,
                         StringRef name, StringRef lifetimeName,
                         VarLetDeclKind kind) {
  auto lifetimeType = b.getType<LifetimeType>();
  auto lifetimeNameAttr = b.getAttr<StringAttr>(lifetimeName);
  auto lifetimeDecl = ParamDeclAttr::get(lifetimeNameAttr, lifetimeType);
  // Lets are mutable because they may be lazy initialized.
  auto resultType = RefType::get(
      /*isMutable=*/true, elementType,
      ParamDeclRefAttr::get(lifetimeNameAttr, lifetimeType));
  build(b, state, resultType, name, kind, lifetimeDecl, /*docString=*/{});
}

bool VarLetDeclOp::isSynthetic() {
  return getKind() == VarLetDeclKind::Synthesized;
}

//===----------------------------------------------------------------------===//
// GlobalVarDeclOp
//===----------------------------------------------------------------------===//

static ParseResult parseNoArgRegion(OpAsmParser &p, Region &region) {
  if (p.parseRegion(region, {}))
    return failure();
  if (region.empty())
    region.push_back(new Block);
  return success();
}

static void printNoArgRegion(OpAsmPrinter &p, Operation *op, Region &region) {
  p.printRegion(region);
}

LogicalResult GlobalVarDeclOp::verify() {
  if (getCtor().getNumArguments())
    return emitOpError() << "constructor region should have zero arguments";
  if (getDtor().getNumArguments())
    return emitOpError() << "destructor region should have zero arguments";
  return verifyTopLevelLocScope(*this);
}

DebugInfo::DIScopeAttr GlobalVarDeclOp::getLocScope() {
  return getTopLevelScope(*this);
}

//===----------------------------------------------------------------------===//
// GlobalVarRefOp
//===----------------------------------------------------------------------===//

void GlobalVarRefOp::build(OpBuilder &builder, OperationState &state,
                           GlobalVarDeclOp op) {
  build(builder, state, RefType::getImmortal(/*isMut=*/true, op.getType()),
        getFullyResolvedSymbolRef(op));
}

LogicalResult GlobalVarRefOp::verifySymbolUses(SymbolTableCollection &symtab) {
  auto global = symtab.lookupSymbolIn<GlobalVarDeclOp>(
      (*this)->getParentOfType<ModuleOp>(), getGlobal());
  if (!global || global.getType() != getResult().getType().getElementType())
    return emitOpError() << "does not refer to a global variable declaration "
                            "of the right type";
  return success();
}

//===----------------------------------------------------------------------===//
// AsyncCallOp
//===----------------------------------------------------------------------===//

/// Use the result types to form the coroutine type, inheriting the throws bit.
static ParseResult
parseAsyncCallOpTypes(AsmParser &p, SmallVectorImpl<Type> &operandTypes,
                      Type &coroutineType, TypedAttr callee,
                      ArrayRef<TypedAttr> implicitLifetimes) {
  SmallVector<Type> resultTypes;
  if (failed(parseCallOpTypes(p, operandTypes, resultTypes, callee,
                              implicitLifetimes)))
    return failure();
  coroutineType =
      POP::CoroutineType::get(p.getContext(), resultTypes,
                              cast<SignatureType>(callee.getType()).isThrows());
  return success();
}

/// Nothing to do on print.
static void printAsyncCallOpTypes(AsmPrinter &, Operation *, TypeRange, Type,
                                  TypedAttr, ArrayRef<TypedAttr>) {}

LogicalResult AsyncCallOp::verify() {
  auto sig = cast<SignatureType>(getCallee().getType());
  if (!sig.isAsync())
    return emitOpError("callable must be 'async'");
  SignatureType resultSig = getResult().getType().getSignature();
  if (sig.isThrows() != resultSig.isThrows())
    return emitOpError() << "'throws' of resultant coroutine must match callee";

  if (auto litSig = dyn_cast<LITSignatureType>(sig)) {
    if (failed(verifyLifetimeParams(*this, litSig)) ||
        failed(verifyCallOp(*this, litSig, getOperands(),
                            resultSig.getValueResults())))
      return failure();
  }
  return success();
}

void AsyncCallOp::concretizeCallee(mlir::IRRewriter &b,
                                   SymbolConstantAttr callee) {
  setCalleeAttr(callee);
}

bool AsyncCallOp::isImplicitlyParametric() { return true; }

void AsyncCallOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {}

//===----------------------------------------------------------------------===//
// AsyncExecuteOp
//===----------------------------------------------------------------------===//

/// The results of a `lit.async.execute` when treated like a function, although
/// an async one, are the results of the coroutine.
ArrayRef<Type> AsyncExecuteOp::getResultTypes() {
  return getType().getResultTypes();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult LIT::ReturnOp::verify() {
  auto func = (*this)->getParentOfType<LIT::FuncOp>();
  if (!func)
    return emitOpError("expected to be nested inside a `lit.func` operation");
  return checkOperandTypes(*this, func.getResultTypes());
}

ErrorTreeOrSuccess LIT::ReturnOp::interpret(ArrayRef<Attribute> operands,
                                            InterpreterState &state) {
  // Manually implement the return hook for this operation; it does not
  // implement `ReturnLike`. Pop the current frame and transfer control flow
  // back to the call operation, using the operands of the return as the results
  // of the call.
  Operation *call = state.popFrame();
  state.setReturnValues(operands);
  state.transferControlFlowTo(call);
  return success();
}

//===----------------------------------------------------------------------===//
// RaiseOp
//===----------------------------------------------------------------------===//

LogicalResult RaiseOp::verify() {
  Operation *op = *this;

  // Scan for an enclosing try block (where we're in the try part, not the
  // except) or a throwing function.
  while (Operation *parentOp = op->getParentOp()) {
    if (auto tryOp = dyn_cast<TryOp>(parentOp)) {
      if (&tryOp.getTryRegion().front() == op->getBlock())
        return success();
    }

    if (auto funcOp = dyn_cast<LIT::FuncOp>(parentOp)) {
      if (funcOp.isThrows())
        return success();
    }
    op = parentOp;
  }

  return emitOpError("must be nested inside the 'try' region of a `lit.try` "
                     "operation or a throwing function");
}

//===----------------------------------------------------------------------===//
// BreakOp / ContinueOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyBreakOrContinueOp(Operation *op) {
  if (auto loop = op->getParentOfType<LIT::LoopOp>();
      loop && op->getBlock() == &loop.getElseRegion().front()) {
    if (!loop->getParentOfType<LIT::LoopOp>())
      return op->emitOpError(
          "A loop with continue or break in its else region should "
          "have a parent loop.");
  }

  if (op->getParentOfType<LIT::LoopOp>())
    return success();

  return op->emitOpError("must be nested within a `lit.loop` operation");
}

LogicalResult BreakOp::verify() { return verifyBreakOrContinueOp(*this); }
LogicalResult ContinueOp::verify() { return verifyBreakOrContinueOp(*this); }

//===----------------------------------------------------------------------===//
// UnboundRegionOp
//===----------------------------------------------------------------------===//

LogicalResult UnboundRegionOp::verify() {
  return emitOpError("is never valid. Was it not erased by the parser?");
}

//===----------------------------------------------------------------------===//
// HandleVariantOp
//===----------------------------------------------------------------------===//

/// Return the range of values that should be mapped onto incoming values.
ValueRange HandleVariantOp::getEntryArguments(std::optional<unsigned> target) {
  // If there are no targets, then the target region is the region directly
  // after this operation and the results of this op are the outgoing values to
  // be bound to the incoming arguments of the subsequent region
  if (!target)
    return getResults();
  assert(*target == 0 || *target == 1);
  return {};
}

LogicalResult HandleVariantOp::verify() {
  if (getVariant().getType().getNumTypes() != 2)
    return emitOpError("expected the variant to have two types: a success type "
                       "and an error type");
  if (!getSuccessRegion().getArguments().empty())
    return emitOpError("expected success region to have zero arguments");
  if (!getErrorRegion().getArguments().empty())
    return emitOpError("expected error region to have zero arguments");
  return success();
}

/// The condition that determines which region is entered is dynamic; check both
/// regions.
void HandleVariantOp::getEntryTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  // TODO: Check for VariantAttr presence to prune targets.
  targets.emplace_back(0);
  targets.emplace_back(1);
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

bool YieldOp::isParentNode(Operation *op) { return isa<HandleVariantOp>(op); }

void YieldOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  // Branch to after the parent operation.
  targets.emplace_back(std::nullopt, getOperands());
}

//===----------------------------------------------------------------------===//
// ErrorReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ErrorReturnOp::verify() {
  if (getVariant().getType().getNumTypes() != 2)
    return emitOpError(
        "expected two types in the variant: an error type and a success type.");
  return success();
}

bool ErrorReturnOp::isParentNode(Operation *op) { return isa<LIT::FuncOp>(op); }

void ErrorReturnOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.size() == 1);
  targets.emplace_back(std::nullopt, getVariant());
}

//===----------------------------------------------------------------------===//
// ExternFuncOp
//===----------------------------------------------------------------------===//

LogicalResult LIT::ExternFuncOp::verify() {
  if (getParentOp().isExternal())
    return success();
  return emitOpError("expected an external parent function");
}

//===----------------------------------------------------------------------===//
// TraitFuncOp
//===----------------------------------------------------------------------===//

LogicalResult LIT::TraitFuncOp::verify() {
  if (llvm::isa_and_present<TraitDeclOp>(getParentOp()->getParentOp()))
    return success();

  return emitOpError("expected a parent function in a trait");
}

//===----------------------------------------------------------------------===//
// TransferMemOwnershipOp
//===----------------------------------------------------------------------===//

void TransferMemOwnershipOp::build(OpBuilder &b, OperationState &state,
                                   Value srcValue, StringAttr lifetimeName) {
  auto lifetimeType = b.getType<LifetimeType>();
  auto lifetimeDecl = ParamDeclAttr::get(lifetimeName, lifetimeType);
  // Lets are mutable because they may be lazy initialized.
  auto resultType = cast<RefType>(srcValue.getType())
                        .getWithLifetime(ParamDeclRefAttr::get(lifetimeDecl));
  build(b, state, resultType, srcValue, lifetimeDecl);
}

void TransferMemOwnershipOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  // Set the name of the SSA value to follow the lifetime name since it
  // indicates where the value came from.
  StringRef name = getParamDecl().getName().strref();
  if (!name.empty() && name[0] == '`')
    name = name.drop_front();

  setNameFn(getResult(), name);
}

void TransferMemOwnershipOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  walkDef(getParamDecl(), ParamDefValue());
}

//===----------------------------------------------------------------------===//
// UnresolvedImportOp
//===----------------------------------------------------------------------===//

LogicalResult LIT::UnresolvedImportOp::verify() {
  if (getDeclNameLoc().has_value() && !getDeclName().has_value())
    return emitOpError("specified `declNameLoc` without `declName`");
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "KGEN/LITDialect/LIT.cpp.inc"
