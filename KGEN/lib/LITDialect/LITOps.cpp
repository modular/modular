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
#include "KGEN/CODialect/COUtils.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/KGENDialect/ParameterReplacer.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/LITDialect/SpecialFunctions.h"
#include "Support/Compiler/Properties.h"
#include "Support/Compiler/VerifyUtils.h"

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
    if (auto funcOp = dyn_cast<FnOp>(parentOp))
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

FnTypeGeneratorType LIT::getCalleeType(Operation *op) {
  if (auto call = dyn_cast<LIT::CallOp>(op))
    return call.getCalleeType();
  return cast<LIT::CallIndirectOp>(op).getCalleeType();
}

ValueRange LIT::getCalleeArguments(Operation *op) {
  if (auto call = dyn_cast<LIT::CallOp>(op))
    return call.getOperands();
  return cast<LIT::CallIndirectOp>(op).getArguments();
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

/// Collect ancestor ops whose parameters are relevant, and create a
/// concatenated list of their parameters.
static std::pair<SmallVector<Operation *>, SmallVector<ParamDeclAttr>>
collectParametricAncestors(Operation *op) {
  std::pair<SmallVector<Operation *>, SmallVector<ParamDeclAttr>> res;
  auto &[ancestors, params] = res;

  auto isRelevantAncestor = [](Operation *op) {
    auto decl = dyn_cast_or_null<DeclInterface>(op);
    return decl && !isa<FuncInterface, ParamForOp>(*decl);
  };
  while (isRelevantAncestor(op)) {
    ancestors.push_back(op);
    op = op->getParentOp();
  }
  for (Operation *op : ancestors)
    llvm::append_range(params, cast<DeclInterface>(op).getInputParams());
  return res;
}

FnTypeGeneratorType LIT::getFullSignature(Operation *container,
                                          FnTypeGeneratorType signature) {
  // Collect contextual params, if there are none, the full signature is the
  // same as the local signature.
  auto [ancestors, params] = collectParametricAncestors(container);
  if (params.empty())
    return signature;
  return FnTypeGeneratorType::prependParams(
      signature, params, getContextualVariadicMask(ancestors));
}

//===----------------------------------------------------------------------===//
// FileModuleOp
//===----------------------------------------------------------------------===//

void FileModuleOp::build(OpBuilder &builder, OperationState &state,
                         StringAttr name) {
  state.addAttribute(getSymNameAttrName(state.name), name);
  state.addRegion()->push_back(new Block());
}

/// Modules don't have input parameters but do define a parameter scope.
ArrayRef<ParamDeclAttr> FileModuleOp::getInputParams() { return {}; }

//===----------------------------------------------------------------------===//
// PackageOp
//===----------------------------------------------------------------------===//

void PackageOp::build(OpBuilder &builder, OperationState &state,
                      StringAttr name) {
  state.addAttribute(getSymNameAttrName(state.name), name);
  state.addRegion()->push_back(new Block());
}

/// Packages don't have input parameters but do define a parameter scope.
ArrayRef<ParamDeclAttr> PackageOp::getInputParams() { return {}; }

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

static ParseResult parseOriginParams(AsmParser &p,
                                     ParameterExprArrayAttr &implicitOrigins) {
  SmallVector<TypedAttr> values;
  if (p.parseCommaSeparatedList(
          AsmParser::Delimiter::OptionalSquare, [&]() -> ParseResult {
            return parseOriginParamValue(p, values.emplace_back());
          }))
    return failure();
  implicitOrigins = ParameterExprArrayAttr::get(p.getContext(), values);
  return success();
}

static void printOriginParams(AsmPrinter &p, Operation *op,
                              ParameterExprArrayAttr implicitOrigins) {
  if (implicitOrigins.empty())
    return;
  p << '[';
  llvm::interleaveComma(implicitOrigins, p, [&](TypedAttr value) {
    printOriginParamValue(p, value);
  });
  p << ']';
}

/// Infer call operation operand and result types from the signature,
/// substituting implicit origin parameters.
template <typename CalleeT>
static ParseResult
parseCallOpTypes(AsmParser &p, SmallVectorImpl<Type> &operandTypes,
                 SmallVectorImpl<Type> &resultTypes, CalleeT callee,
                 ArrayRef<TypedAttr> implicitOrigins) {
  FuncTypeGeneratorType calleeType;
  if constexpr (std::is_same_v<Type, CalleeT>)
    calleeType = cast<FuncTypeGeneratorType>(callee);
  else
    calleeType = cast<FuncTypeGeneratorType>(callee.getType());

  FunctionType values;
  if (implicitOrigins.empty()) {
    values = calleeType.getBody().getValues();
  } else {
    auto calleeLITTypeGen = dyn_cast<FnTypeGeneratorType>(calleeType);
    if (!calleeLITTypeGen)
      return p.emitError(p.getCurrentLocation(),
                         "expected a FnTypeGeneratorType");
    FnType calleeLITType = calleeLITTypeGen.getBody();
    if (calleeLITType.getNumImplicitOriginDecls() != implicitOrigins.size())
      return p.emitError(p.getNameLoc())
             << implicitOrigins.size()
             << " origins specified, but signature expected "
             << calleeLITType.getNumImplicitOriginDecls();

    values = calleeLITType.substituteImplicitOriginsIntoValues(
        implicitOrigins, [&] { return p.emitError(p.getNameLoc()); });
    if (!values)
      return failure();
  }

  // Async calls don't provide result slots.
  llvm::append_range(operandTypes,
                     values.getInputs().drop_back(
                         calleeType.getBody().getNumAsyncReturnSlots()));
  llvm::append_range(resultTypes, values.getResults());
  return success();
}

/// Nothing to do on print.
template <typename CalleeT>
static void printCallOpTypes(AsmPrinter &, Operation *, TypeRange, TypeRange,
                             CalleeT, ArrayRef<TypedAttr>) {}

static ParseResult
parseCallOp(OpAsmParser &p, TypedAttr &calleeAttr,
            ParameterExprArrayAttr &implicitOrigins,
            SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
            SmallVectorImpl<Type> &operandTypes,
            SmallVectorImpl<Type> &resultTypes) {
  SymbolRefAttr callee;
  // Optionally parse the direct call syntax: `lit.call @abc`.
  OptionalParseResult optResult = p.parseOptionalAttribute(callee);
  if (!optResult.has_value()) {
    // Otherwise, parse the parametric call syntax `lit.call[...: @abc]`
    if (parseParametricCallee(p, calleeAttr))
      return failure();
  } else if (failed(*optResult)) {
    return failure();
  }

  ParameterExprArrayAttr paramValues;
  if (parseOriginParams(p, implicitOrigins))
    return failure();
  if (callee && parseParameterValues(p, paramValues))
    return failure();
  if (p.parseOperandList(operands, AsmParser::Delimiter::Paren))
    return failure();

  if (callee) {
    FuncTypeGeneratorType signature;
    FunctionType functionType;
    if (p.parseColon() ||
        parseKGENFuncTypeGenerator(p, functionType, signature))
      return failure();
    calleeAttr = SymbolConstantAttr::get(callee, signature, paramValues);
  }
  if (failed(parseCallOpTypes(p, operandTypes, resultTypes, calleeAttr,
                              implicitOrigins)))
    return failure();
  return success();
}

static void printCallOp(OpAsmPrinter &p, Operation *op, TypedAttr calleeAttr,
                        ParameterExprArrayAttr implicitOrigins,
                        ValueRange operands, TypeRange operandTypes,
                        TypeRange resultTypes) {
  auto symbolCst = dyn_cast<SymbolConstantAttr>(calleeAttr);
  // Optionally print the direct call syntax. Otherwise, print the parametric
  // call syntax.
  if (symbolCst)
    p << ' ' << symbolCst.getSymbol();
  else
    printParametricCallee(p, op, calleeAttr);
  printOriginParams(p, op, implicitOrigins);
  if (symbolCst)
    printParameterValues(p, symbolCst.getParamValues());
  p << '(';
  p.printOperands(operands);
  p << ')';
  if (symbolCst) {
    p << " : ";
    printSignatureValues(
        p, FunctionType::get(op->getContext(), operandTypes, resultTypes),
        symbolCst.getType());
  }
}

template <typename OpT>
static LogicalResult verifyOriginParams(OpT op, FnType sig) {
  size_t numImplicit = sig.getMetadata().getNumImplicitOriginDecls();
  size_t numParams = op.getImplicitOrigins().size();
  if (numParams == numImplicit)
    return success();
  return op->emitOpError("operation has ")
         << numParams
         << " bindings for implicit origin parameters, but callee "
            "expected "
         << numImplicit;
}

template <typename OpT>
static LogicalResult verifyCallOp(OpT op, FnType sig, ValueRange operands,
                                  std::optional<TypeRange> results) {
  FunctionType values = sig.substituteImplicitOriginsIntoValues(
      op.getImplicitOrigins(), [&] { return op.emitOpError(); });
  if (!values)
    return failure();

  auto verifyTypes = [&](StringRef kind, TypeRange types,
                         TypeRange expected) -> LogicalResult {
    if (types.size() != expected.size()) {
      return op.emitOpError("callee expected ")
             << expected.size() << " " << kind << "s but got " << types.size();
    }
    for (auto [i, type, exp] : llvm::enumerate(types, expected)) {
      if (type == exp)
        continue;
      return op.emitOpError("callee expected call ")
             << kind << " #" << i << " to be " << exp << " but got " << type;
    }
    return success();
  };

  // Async calls don't provide result slots.
  if (failed(verifyTypes(
          "argument", operands,
          values.getInputs().drop_back(sig.getNumAsyncReturnSlots()))) ||
      (results && failed(verifyTypes("result", *results, values.getResults()))))
    return failure();
  return success();
}

LogicalResult LIT::CallOp::verify() {
  auto sig = dyn_cast<FnTypeGeneratorType>(getCallee().getType());
  if (!sig)
    return emitOpError("callee type must be a FnTypeGeneratorType");
  if (failed(verifyOriginParams(*this, sig.getBody())))
    return failure();
  return verifyCallOp(*this, sig.getBody(), getOperands(), getResultTypes());
}

SymbolRefAttr LIT::CallOp::getDirectCallee() {
  if (auto symbolCst = dyn_cast<SymbolConstantAttr>(getCallee()))
    return symbolCst.getSymbol();
  return {};
}

FailureOr<InlineResult> LIT::CallOp::prepInline(mlir::RewriterBase &b) {
  // Inlining not supported for this op
  return failure();
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

LogicalResult LIT::CallIndirectOp::verify() {
  auto sig = cast<FnTypeGeneratorType>(getCallee().getType());
  if (failed(verifyOriginParams(*this, sig.getBody())))
    return failure();
  return verifyCallOp(*this, sig.getBody(), getArguments(), getResultTypes());
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
SpecialFunctionKind FnOp::getSpecialFunctionKind() {
  return (SpecialFunctionKind)getSpecialFnKind();
}
const SpecialFunctionInfo &FnOp::getSpecialFunctionInfo() {
  return SpecialFunctionInfo::get(getSpecialFunctionKind());
}

/// Returns the user-defined result type, looking through implicit memory
/// results and stripping off the variant from error throwing results if needed.
Type FnOp::getUserResultType() {
  return LIT::getSignatureUserResultType(
      getFuncTypeGenerator(), getArgumentTypes(), getMLIRResultType());
}

TypedAttr FnOp::getBoundReference(ParameterExprArrayAttr bindings) {
  if (!bindings) // We allow null for convenience.
    bindings = ParameterExprArrayAttr::get(getContext(), {});

  // SymbolConstantAttr provides a type for the SymbolRefAttr with the
  // parameters substituted in.  The function reference binds any parameter
  // bindings present on the access (in bindings), which typically concretizes
  // the signature.
  FnTypeGeneratorType resultType;
  std::tie(resultType, bindings) =
      getUnboundSpecializedSignature(getFullSignature(), bindings);

  if (ParamDeclAttr decl = getParamDeclAttr())
    return BindParamsAttr::get(ParamDeclRefAttr::get(decl), bindings);

  return SymbolConstantAttr::get(getFullyResolvedSymbolRef(*this), resultType,
                                 bindings);
}

SymbolConstantAttr FnOp::getBoundSymbolRef(ParameterExprArrayAttr bindings) {
  return cast<SymbolConstantAttr>(getBoundReference(bindings));
}

bool FnOp::isSynthetic() { return getIsSynthetic(); }

/// Parse a fixed mutability specifier that occurs for implicit Origins.
// Implicit origin params are always known immut or mut, never parametric.
static ParseResult parseImplicitOriginMutability(AsmParser &p,
                                                 bool &isMutable) {
  llvm::SMLoc loc;
  StringRef mutability;
  if (p.getCurrentLocation(&loc) || p.parseKeyword(&mutability))
    return failure();
  if (mutability != "mut" && mutability != "imm")
    return p.emitError(loc, "expected 'mut' or 'imm' to indicate mutability");
  isMutable = mutability == "mut";
  return success();
}

static void printImplicitOriginMutability(AsmPrinter &p, OriginType type) {
  assert((type.isMutableKnown(true) || type.isMutableKnown(false)) &&
         "Implicit Origins are always known mut or imm");
  p << (type.isMutableKnown(true) ? "mut " : "imm ");
}

// These FuncOp attributes are disallowed while parsing since they can
// be inferred. Likewise while printing we ignore them.
static StringRef disallowedAttrNames[] = {
    "sym_name",       "exportKind",        "isCExported",  "constraints",
    "implements",     "funcTypeGenerator", "functionType", "sym_name",
    "argNames",       "paramNames",        "evaluator",    "defaultImpl",
    "inlineLevel",    "paramDecl",         "params",       "decorators",
    "argPassingKinds"};

static ParseResult parseLITFunctionSignature(
    OpAsmParser &p, SmallVectorImpl<OpAsmParser::Argument> &args,
    ParamDeclArrayAttr &params, FunctionType &functionType,
    FnTypeGeneratorType &signature) {
  llvm::SMLoc startLoc = p.getCurrentLocation();

  TypedAttr captureOrigins;
  auto originSet = OriginSetType::get(p.getContext());
  if (succeeded(p.parseOptionalColon())) {
    if (parseParamValue(p, captureOrigins, originSet) || p.parseColon())
      return failure();
  } else {
    captureOrigins = OriginSetAttr::get({}, originSet);
  }
  bool isNestedOriginExclusivityCheckingDisabled =
      succeeded(p.parseOptionalKeyword("no_nested_origin_exclusivity"));

  SmallVector<ParamDeclAttr> originDecls;
  auto parseOriginDecl = [&]() -> ParseResult {
    bool isMutable = false;
    StringAttr name;
    if (parseImplicitOriginMutability(p, isMutable) || parseParamName(p, name))
      return failure();
    originDecls.push_back(
        ParamDeclAttr::get(name, OriginType::get(p.getContext(), isMutable)));
    return success();
  };

  PogListAttr paramListAttr;
  if (parseOptionalParameterSpec(p, params, paramListAttr))
    return failure();

  // Parse implicit origin decls.
  if (p.parseCommaSeparatedList(AsmParser::Delimiter::OptionalSquare,
                                parseOriginDecl))
    return failure();

  SmallVector<StringAttr> argNames;
  SmallVector<TypedAttr> defaultPosArgs;
  SmallVector<TypedAttr> defaultKwOnlyArgs;
  SmallVector<ArgConvention> argConventions;
  SmallVector<size_t> argVariadicIndices;
  ssize_t argPackIndex = -1;
  std::optional<ArgConvention> origArgPackConvention;

  PassingKindParser passingKindParser(p);
  size_t idx = 0;
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

    // A colon and type should come next, followed by an optional location,
    // input convention, and variadicness.
    if (p.parseColonType(arg.type) ||
        p.parseOptionalLocationSpecifier(arg.sourceLoc) ||
        parseConventionAndVariadicness(p, argConventions.emplace_back(),
                                       argVariadicIndices, argPackIndex,
                                       origArgPackConvention, idx++))
      return failure();

    // Parse an optional default value.
    TypedAttr defaultVal;
    if (failed(parseOptionalDefaultValue(p, defaultVal, arg.type,
                                         hasAddress(argConventions.back()))))
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
      PogListAttr::get(p.getContext(), argNames, argPassingKinds,
                       defaultPosArgs, defaultKwOnlyArgs, argVariadicIndices,
                       argPackIndex, origArgPackConvention),
      originDecls.size(), captureOrigins,
      isNestedOriginExclusivityCheckingDisabled);
  signature = FuncTypeGeneratorType::remapToFuncTypeGenerator(
      params, functionType, argConventions, effects, metadata, paramListAttr,
      [&] { return p.emitError(startLoc); });
  if (!signature)
    return failure();

  // Replace named implicit origin parameter references with index-based
  // references in the signature.
  signature = signature.replaceImplicitOriginsWithIndexes(originDecls);

  // The formal params are the declared params + the implicit origin decls.
  SmallVector<ParamDeclAttr> allParams;
  allParams.reserve(params.size() + originDecls.size());
  llvm::append_range(allParams, params);
  llvm::append_range(allParams, originDecls);
  params = ParamDeclArrayAttr::get(p.getContext(), allParams);
  return success();
}

static void printLITFunctionSignature(OpAsmPrinter &p, Region *region,
                                      ArrayRef<ParamDeclAttr> params,
                                      FunctionType functionType,
                                      FnTypeGeneratorType signature) {
  ArrayRef<ParamDeclAttr> originDecls =
      params.drop_front(signature.getInputParamTypes().size());

  if (!isEmptyOriginSet(signature.getCaptureOrigins())) {
    p << ':';
    printParamValue(p, signature.getCaptureOrigins());
    p << ':';
  }
  if (signature.getIsNestedOriginExclusivityCheckingDisabled())
    p << "no_nested_origin_exclusivity";

  ParameterEvaluator evaluator;
  printOptionalParameterSpec(p, params.drop_back(originDecls.size()),
                             signature.getParamListAttrs(), evaluator);

  if (!originDecls.empty()) {
    p << '[';
    llvm::interleaveComma(originDecls, p, [&](ParamDeclAttr decl) {
      printImplicitOriginMutability(p, cast<OriginType>(decl.getType()));
      printParamName(p, decl.getName());
    });
    p << ']';
  }

  PogListAttr argListAttr = signature.getArgListAttrs();
  SmallVector<Variadicness> variadicness = getVariadicness(argListAttr);
  DefaultValueHandler defaultHandler(argListAttr);
  PassingKindPrinter passingKindPrinter(p, argListAttr, '|');
  auto printElt = [&](unsigned i) {
    passingKindPrinter.printOptionalStarSlash(i);

    // Print the SSA name first, which might have been automatically uniqued.
    BlockArgument arg = region->getArgument(i);
    std::string ssaName;
    llvm::raw_string_ostream ss(ssaName);
    p.printOperand(arg, ss);
    p << ssaName;

    // If different from the SSA name (e.g. because it was uniqued, or because
    // it contains characters that need escaping), we also print the
    // user-defined argument name in brackets.
    StringAttr argName = signature.getArgName(i);
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
    auto argConv = signature.getArgConvention(i);

    if (variadicness[i] == Variadicness::kPack) {
      assert(argConv == ArgConvention::ReadMem ||
             argConv == ArgConvention::OwnedMem ||
             argConv == ArgConvention::OwnedReg);
      argConv = signature.getPackVarArgConvention(i);
    }
    printConventionAndVariadicness(p, argConv, variadicness[i]);

    if (TypedAttr defaultOr = defaultHandler.getDefault(i)) {
      p << " = ";
      printParamValue(p, evaluator.getReboundAttribute(defaultOr));
    }

    // Check if we are at the end; if so, we might still have to print a '/'.
    passingKindPrinter.printOptionalTrailingSlash(i);
  };
  printSignatureValues(p, printElt, functionType, signature.getArgConventions(),
                       signature.getFnEffects(),
                       /*optionalResultList=*/true);
}

/// Parses a LIT Generator.
ParseResult FnOp::parse(OpAsmParser &parser, OperationState &result) {
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
  if (!isParamDecl)
    result.addAttribute(getSymNameAttrName(result.name), nameAttr);

  // Parse the function signature.
  SmallVector<OpAsmParser::Argument> entryArgs;
  ParamDeclArrayAttr params;
  FunctionType functionType;
  FnTypeGeneratorType signature;
  if (parseLITFunctionSignature(parser, entryArgs, params, functionType,
                                signature))
    return failure();

  // Parse additional function attributes.
  InlineLevelAttr inlineLevel;
  DecoratorsAttr decorators;
  if (parseOptionalInline(parser, inlineLevel) ||
      parseOptionalDecorators(parser, decorators))
    return failure();
  result.addAttribute(getInlineLevelAttrName(result.name), inlineLevel);
  result.addAttribute(getDecoratorsAttrName(result.name), decorators);
  result.addAttribute(getParamsAttrName(result.name), params);
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

  result.addAttribute(getFuncTypeGeneratorAttrName(result.name),
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

// Print the FnOp using the shared printing logic.
void FnOp::print(OpAsmPrinter &p) {
  using namespace mlir::function_interface_impl;

  // Print the operation and the function name.
  printSymbolExport(p, *this, getExportKindAttr());
  p << ' ';
  if (ParamDeclAttr decl = getParamDeclAttr())
    printParamName(p, decl.getName());
  else
    p.printSymbolName(*getSymName());

  // Print the function arguments. Here we need all the use defined names.
  printLITFunctionSignature(p, &getBodyRegion(), getParams(), getFunctionType(),
                            getFuncTypeGenerator());
  printOptionalInline(p, getInlineLevel());
  printOptionalDecorators(p, *this, getDecorators());

  // Don't print the following in lit.fn.
  SmallVector<StringRef> ignoredAttrNames(
      (ArrayRef<StringRef>(disallowedAttrNames)));
  if (getLLVMMetadataArray().empty())
    ignoredAttrNames.push_back(getLLVMMetadataArrayAttrName());
  if (getLLVMArgMetadataArray().empty())
    ignoredAttrNames.push_back(getLLVMArgMetadataArrayAttrName());

  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                     ignoredAttrNames);

  p << ' ';
  p.printRegion(getBodyRegion(), /*printEntryBlockArgs=*/false);
}

// Name the arguments of the region with the argument names.
void FnOp::getAsmBlockArgumentNames(
    Region &region, llvm::function_ref<void(Value, StringRef)> setNameFn) {
  if (region.empty())
    return;

  // Set a name for each argument.
  for (auto [idx, arg] : llvm::enumerate(getBody()->getArguments()))
    setNameFn(arg, getFuncTypeGenerator().getArgName(idx).strref());
}

LogicalResult FnOp::verify() {
  if ((getLLVMMetadataArray().size() & 1) != 0)
    return emitOpError("expected an even number elements in LLVMMetadataArray");
  if (ArrayAttr argsArray = getLLVMArgMetadataArray();
      !argsArray.empty() && argsArray.size() != getNumArguments()) {
    return emitOpError("LLVMArgMetadataArray size does not equal number of "
                       "arguments, got ")
           << argsArray.size();
  }

  // Check that the number of argument labels matches the number of argument
  // types.
  if (getFuncTypeGenerator().getBody().getMetadata().getNumArgs() !=
      getFunctionType().getNumInputs())
    return emitOpError("incorrect number of value parameter labels");

  // Verify the correct number of parameters.
  if (getFuncTypeGenerator().getInputParamTypes().size() +
          getFuncTypeGenerator().getNumImplicitOriginDecls() !=
      getInputParams().size()) {
    return emitOpError("incorrect number of input params: have ")
           << getParams().size() << ", but expected "
           << getFuncTypeGenerator().getNumImplicitOriginDecls()
           << " implicit origins and "
           << getFuncTypeGenerator().getInputParamTypes().size()
           << " input params";
  }

  return success();
}

void FnOp::walkDeclarations(function_ref<void(ParamDeclAttr)> walkDecl) {
  if (auto decl = getParamDeclAttr())
    walkDecl(decl);
}

void FnOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  if (auto decl = getParamDeclAttr())
    walkDef(decl, &getBodyRegion());
}

void FnOp::renameDeclarations(ArrayRef<ParamDeclAttr> decls) {
  if (getParamDecl()) {
    assert(decls.size() == 1);
    setParamDeclAttr(decls.front());
  } else {
    assert(decls.empty());
  }
}

/// This operation has no uses to collect in its current scope.
void FnOp::collectParameterUses(function_ref<void(Attribute)> scanAttr,
                                function_ref<void(Type)> scanType) {}

SmallVector<ParamDeclAttr> FnOp::collectAllParams(bool includeImplOrigins) {
  auto [_, result] = collectParametricAncestors(getOperation()->getParentOp());

  auto params = getParams();
  if (!includeImplOrigins)
    params =
        params.drop_back(getFuncTypeGenerator().getNumImplicitOriginDecls());
  llvm::append_range(result, params);
  return result;
}

FnTypeGeneratorType FnOp::getFullSignature() {
  return LIT::getFullSignature((*this)->getParentOp(), getFuncTypeGenerator());
}

/// Build a function in a default configuration, used by member synthesis.
void FnOp::build(OpBuilder &builder, OperationState &result, StringAttr name,
                 StringAttr sourceName, FuncTypeGeneratorType signature) {
  MLIRContext *ctx = builder.getContext();
  UnitAttr none;
  build(builder, result, name, ParamDeclAttr(), TypeAttr::get(signature),
        TypeAttr::get(signature.getBody().getValues()),
        ParamDeclArrayAttr::get(ctx, {}), DecoratorsAttr::get(ctx, {}),
        /*isStatic=*/none, /*isDef=*/none,
        /*isInherited=*/none, /*isSynthetic=*/none,
        /*isImplicitConversion=*/none,
        ExportKindAttr::get(ctx, ExportKind::NotExported),
        InlineLevelAttr::get(ctx, InlineLevel::Automatic),
        builder.getI8IntegerAttr(uint8_t(SpecialFunctionKind::kNormal)),
        StringAttr(), sourceName, StringAttr(), DocStringAttr(), StringAttr(),
        ArrayAttr::get(ctx, {}), ArrayAttr::get(ctx, {}), Attribute());
  result.regions[0]->push_back(new Block());
}

//===----------------------------------------------------------------------===//
// StructDeclOp
//===----------------------------------------------------------------------===//

static ParseResult parseSymbol(AsmParser &p, SymbolConstantAttr &symbol) {
  TypedAttr value;
  if (parseColonTypeParamValue(p, value))
    return failure();
  symbol = cast<SymbolConstantAttr>(value);
  return success();
}

static void printSymbol(AsmPrinter &p, Operation *, SymbolConstantAttr symbol) {
  printColonTypeParamValue(p, symbol);
}

static ParseResult parseTypeConvention(AsmParser &p, TypeConvention &value) {
  StringRef str;
  value = TypeConvention::MemoryOnly;
  if (succeeded(p.parseOptionalKeyword(
          &str, {stringifyEnum(TypeConvention::MemoryOnly),
                 stringifyEnum(TypeConvention::RegisterPassable),
                 stringifyEnum(TypeConvention::RegisterPassableTrivial),
                 stringifyEnum(TypeConvention::Unspecified)})))
    value = *symbolizeTypeConvention(str);
  return success();
}

static void printTypeConvention(AsmPrinter &p, Operation *op,
                                TypeConvention value) {
  if (value != TypeConvention::MemoryOnly)
    p << ' ' << stringifyTypeConvention(value);
}

static ParseResult parseStructParameterSpec(AsmParser &p,
                                            ParamDeclArrayAttr &params,
                                            TypeAttr &signature,
                                            TypeAttr &canonicalTraitAttr) {
  llvm::SMLoc startLoc = p.getCurrentLocation();
  PogListAttr paramListAttr;
  if (parseOptionalParameterSpec(p, params, paramListAttr))
    return failure();

  TraitType canonicalTrait;
  if (succeeded(p.parseOptionalLParen())) {
    if (parseParamType(p, canonicalTrait) || p.parseRParen())
      return failure();
  } else {
    canonicalTrait = TraitType::get(p.getContext(), {});
  }
  canonicalTraitAttr = TypeAttr::get(canonicalTrait);

  auto sig = TypeSignatureType::remapToSignature(
      [&] { return p.emitError(startLoc); }, params, paramListAttr);
  if (!sig)
    return failure();
  signature = TypeAttr::get(sig);
  return success();
}

static void printStructParameterSpec(AsmPrinter &p, Operation *op,
                                     ArrayRef<ParamDeclAttr> params,
                                     TypeAttr signature,
                                     TypeAttr canonicalTraitAttr) {
  auto sig = cast<TypeSignatureType>(signature.getValue());
  ParameterEvaluator evaluator;
  printOptionalParameterSpec(p, params, sig.getParamListAttrs(), evaluator);

  TraitType canonicalTrait = cast<TraitType>(canonicalTraitAttr.getValue());
  if (!canonicalTrait.getSymbols().empty()) {
    p << '(';
    printParamType(p, canonicalTrait);
    p << ')';
  }
}

bool StructDeclOp::isSynthetic() { return getIsSynthetic(); }

LIT::StructType StructDeclOp::bindReference(ArrayRef<TypedAttr> paramValues) {
  SymbolRefAttr symbol = getFullyResolvedSymbolRef(*this);
  TypeSignatureType sig = getSignature();

  if (paramValues.empty()) {
    // Create a fully unbound reference to the type.
    SmallVector<TypedAttr> unbound;
    ParameterEvaluator evaluator;
    for (Type type : sig.getParamTypes()) {
      unbound.push_back(UnboundAttr::get(evaluator.getReboundType(type)));
      evaluator.addInputValue(unbound.back());
    }
    return LIT::StructType::get(symbol, unbound, sig);
  }

  // Compute the resultant signature.
  auto newSig = sig.bind(paramValues);
  return LIT::StructType::get(symbol, paramValues, newSig);
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
        TypeAttr::get(TraitType::get(ctx, {})),
        /*isSynthetic=*/{},
        /*nonmaterializableTarget=*/{}, /*destructor=*/{}, /*moveInit=*/{},
        /*copyInit=*/{}, /*linearTypeErrorMsg*/ {}, /*closureSignature=*/{},
        /*docString=*/{}, /*deprecationWarning=*/{}, /*sourceName=*/{},
        /*convention=*/{});
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

Type StructFieldOp::getReboundType(StructType structSelfType) {
  if (structSelfType.getParamValues().empty())
    return getType();
  ParameterEvaluator evaluator(getParentOp().getParams(),
                               structSelfType.getParamValues());
  return evaluator.getReboundType(getType());
}

void StructFieldOp::build(OpBuilder &builder, OperationState &odsState,
                          StringAttr name, Type type) {
  build(builder, odsState, name, type, /*docString=*/{});
}

void StructFieldOp::build(OpBuilder &builder, OperationState &odsState,
                          const Twine &name, Type type) {
  build(builder, odsState, builder.getStringAttr(name), type);
}

//===----------------------------------------------------------------------===//
// StructInsertOp
//===----------------------------------------------------------------------===//

/// Lookup the declaration for the struct. When checking field types, we can't
/// directly compare operation types to the struct field types because they are
/// parameterized under different domains. We have to rebind them.
static std::pair<StructDeclOp, ParameterEvaluator>
lookupStructDecl(SymbolTableCollection &symbolTable, Operation *user,
                 LIT::StructType ref) {
  auto module = KGENModule::from(user, symbolTable);
  auto decl = module.lookup<StructDeclOp>(ref.getSymbol());
  if (!decl) {
    user->emitOpError("expected to find a struct decl for ") << ref;
    return {};
  }
  ParameterEvaluator evaluator(decl.getParams(), ref.getParamValues());
  return {decl, std::move(evaluator)};
}

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
  return LITStructAttr::get(values, getType());
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

static LogicalResult
verifyStructFieldAndType(SymbolTableCollection &symbolTable, Operation *op,
                         LIT::StructType ref, StringAttr fieldName, Type type) {
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
  auto structType = cast<StructType>(structBase.getType());
  build(builder, result, field.getReboundType(structType), structBase,
        field.getNameAttr());
}

OpFoldResult LIT::StructExtractOp::fold(FoldAdaptor adaptor) {
  if (auto value = adaptor.getContainer())
    return StructExtractAttr::get(cast<TypedAttr>(value), getFieldAttr(),
                                  getType());

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

/// Given a reference to a struct, return the reference type to the
/// specified field, maintaining origin and mutability, assuming the type
/// is already rebound to its final type.
RefType RefStructGEROp::getReboundFieldType(RefType structRefTy,
                                            StringAttr fieldName,
                                            Type reboundType) {
  // The origin of the struct reference incorporates field sensitivity.
  auto fieldOrigin = OriginFieldAttr::get(structRefTy.getOrigin(), fieldName);
  return RefType::get(reboundType, fieldOrigin, structRefTy.getAddressSpace());
}

RefType RefStructGEROp::getFieldType(RefType structRefTy, StructFieldOp field) {
  auto structTy = cast<StructType>(structRefTy.getElementType());
  return getReboundFieldType(structRefTy, field.getNameAttr(),
                             field.getReboundType(structTy));
}

LogicalResult
RefStructGEROp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Type structType = getContainer().getType().getElementType();
  return verifyStructFieldAndType(symbolTable, *this,
                                  cast<StructType>(structType), getFieldAttr(),
                                  getResult().getType().getElementType());
}

void RefStructGEROp::build(OpBuilder &builder, OperationState &result,
                           Value structBaseRef, StructFieldOp field) {
  auto resultType = getFieldType(cast<RefType>(structBaseRef.getType()), field);
  build(builder, result, resultType, field.getNameAttr(), structBaseRef);
}

LogicalResult RefStructGEROp::verify() {
  if (getType() != getReboundFieldType(getContainer().getType(), getFieldAttr(),
                                       getType().getElementType()))
    return emitOpError("invalid origin or address space");
  return success();
}

static ParseResult parseStructGERTypes(AsmParser &p, Type &containerType,
                                       Type &fieldRefType,
                                       StringAttr fieldName) {
  llvm::SMLoc loc = p.getCurrentLocation();
  Type fieldType;
  // parse: 'type' `->` 'type'
  containerType = RefType::parse(p);
  if (!containerType || p.parseArrow() || parseParamType(p, fieldType))
    return failure();
  auto containerRefType = dyn_cast<RefType>(containerType);
  if (!containerRefType)
    return p.emitError(loc, "expected '!lit.ref' type in lit.ref.struct.ger");

  // The field type gets wrapped with the same mutability and origin as
  // the result element.
  fieldRefType = RefStructGEROp::getReboundFieldType(containerRefType,
                                                     fieldName, fieldType);
  return success();
}

static void printStructGERTypes(AsmPrinter &p, Operation *,
                                RefType containerType, RefType fieldType,
                                StringAttr fieldName) {
  containerType.print(p);
  p << " -> ";
  if (auto refType = dyn_cast<RefType>(fieldType))
    printParamType(p, refType.getElementType());
  else {
    p << "<<ERROR NOT REF CONTAINER>>";
    p.printType(fieldType);
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
                             Value pointer, TypedAttr origin, bool startsUninit,
                             bool endsUninit) {
  auto ptr = cast<PointerType>(pointer.getType());
  auto refType =
      RefType::get(ptr.getElementType(), origin, ptr.getAddressSpace());
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
        ParamDeclArrayAttr::get(ctx, {}),
        TypeAttr::get(TraitType::get(ctx, {})),
        /*convention=*/TypeConvention::Unspecified,
        /*dtorSig=*/{}, /*docString=*/{}, /*deprecationWarning=*/{},
        /*linearTypeErrorMsg*/ {});
  result.regions[0]->push_back(new Block());
}

TraitType TraitDeclOp::bindReference() {
  return TraitType::get(getFullyResolvedSymbolRef(*this));
}

//===----------------------------------------------------------------------===//
// TryOp
//===----------------------------------------------------------------------===//

void TryOp::getEntryTargets(ArrayRef<Attribute> operands,
                            SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
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
  if (parseParamDecl(p, paramDecl))
    return failure();

  if (failed(p.parseOptionalEqual())) {
    // This is actually valid; an alias declaration in a trait is an associated
    // alias.
    return success();
  }

  if (p.parseLess() || parseParamValue(p, value, paramDecl.getType()) ||
      p.parseGreater())
    return failure();

  return success();
}

static void printAliasDeclOpValue(OpAsmPrinter &p, Operation *,
                                  ParamDeclAttr paramDecl, TypedAttr value) {
  printParamDecl(p, paramDecl);
  // Traits' alias declarations need no value, in which case they're associated
  // types.
  if (value) {
    p << " = <";
    printParamValue(p, value);
    p << ">";
  }
}

void AliasDeclOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  if (TypedAttr value = getValueAttr()) {
    walkDef(getParamDecl(), value);
  } else {
    // This could happen if we're in a trait's associated alias declaration.
    walkDef(getParamDecl(), ParamDefValue());
  }
}

LogicalResult AliasDeclOp::verify() {
  // Associated types in traits need no value.
  if (TypedAttr value = getValueAttr()) {
    if (getParamDecl().getType() != value.getType()) {
      return emitOpError("declares a parameter with type ")
             << getParamDecl().getType()
             << " but parameter expression has type " << value.getType();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// VarDeclOp
//===----------------------------------------------------------------------===//

static ParseResult parseVarDeclType(AsmParser &p, Type &resultType,
                                    ParamDeclAttr &originDecl) {
  if (p.parseType(resultType))
    return failure();
  auto refType = dyn_cast<RefType>(resultType);
  if (!refType || !refType.isMutableKnown(true))
    return p.emitError(p.getNameLoc(),
                       "expected a mutable !lit.ref<> result type");
  // The origin must be a simple name, which becomes the name we are
  // declaring.
  auto origin = dyn_cast<ParamDeclRefAttr>(refType.getOrigin());
  if (!origin)
    return p.emitError(p.getNameLoc(),
                       "expected a !lit.ref<> with named origin");
  originDecl = ParamDeclAttr::get(origin);
  return success();
}

static void printVarDeclType(AsmPrinter &p, Operation *op, Type resultType,
                             ParamDeclAttr decl) {
  p.printType(resultType);
}

void VarDeclOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), getName());
}

void VarDeclOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  walkDef(getParamDecl(), ParamDefValue());
}

void VarDeclOp::build(OpBuilder &b, OperationState &state, Type elementType,
                      StringRef name, StringRef originName, VarDeclKind kind) {
  auto originType = b.getType<OriginType>(/*isMutable=*/true);
  auto originNameAttr = b.getAttr<StringAttr>(originName);
  auto originDecl = ParamDeclAttr::get(originNameAttr, originType);
  auto resultType = RefType::get(
      elementType, ParamDeclRefAttr::get(originNameAttr, originType));
  build(b, state, resultType, name, kind, originDecl, /*argShadowIndex=*/{});
}

bool VarDeclOp::isSynthetic() { return getKind() == VarDeclKind::Synthesized; }

/// Return true if this is non-synthetic variable, if its name starts with
/// something other than an underscore, and is not an argument shadow.
bool VarDeclOp::shouldWarnAboutUnused() {
  auto kind = getKind();
  // Don't warn about synthesized VarDecls, they aren't user-declared.
  return kind != VarDeclKind::Synthesized && kind != VarDeclKind::Arg &&
         kind != VarDeclKind::InitOutArg &&
         // Don't warn about things like _x, because this silences the warning.
         !getName().starts_with("_");
}

LogicalResult VarDeclOp::verify() {
  if (getArgShadowIndex().has_value() && getKind() != VarDeclKind::Arg)
    return emitOpError() << "cannot have arg index unless is arg kind";
  return success();
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
  build(builder, state, RefType::getAnyOrigin(op.getType(), /*isMut=*/true),
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
static ParseResult parseAsyncCallOpTypes(AsmParser &p,
                                         SmallVectorImpl<Type> &operandTypes,
                                         TypedAttr callee,
                                         ArrayRef<TypedAttr> implicitOrigins) {
  SmallVector<Type> resultTypes;
  return parseCallOpTypes(p, operandTypes, resultTypes, callee,
                          implicitOrigins);
}

/// Nothing to do on print.
static void printAsyncCallOpTypes(AsmPrinter &, Operation *, TypeRange,
                                  TypedAttr, ArrayRef<TypedAttr>) {}

LogicalResult AsyncCallOp::verify() {
  auto sig = cast<FuncTypeGeneratorType>(getCallee().getType()).getBody();
  if (!sig.isAsync())
    return emitOpError("callable must be 'async'");
  if (auto litSigGen = dyn_cast<FnTypeGeneratorType>(sig)) {
    if (failed(verifyOriginParams(*this, litSigGen.getBody())) ||
        failed(verifyCallOp(*this, litSigGen.getBody(), getOperands(),
                            /*results=*/{})))
      return failure();
  }
  return success();
}

FailureOr<InlineResult> LIT::AsyncCallOp::prepInline(mlir::RewriterBase &b) {
  // Inlining not supported for this op
  return failure();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult LIT::ReturnOp::verify() {
  auto func = (*this)->getParentOfType<FnOp>();
  if (!func)
    return emitOpError("expected to be nested inside a `lit.fn` operation");
  return checkOperandTypes(*this, func.getResultTypes());
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

    if (auto funcOp = dyn_cast<FnOp>(parentOp)) {
      if (funcOp.isThrows())
        return success();
    }
    op = parentOp;
  }

  return emitOpError("must be nested inside the 'try' region of a `lit.try` "
                     "operation or a throwing function");
}

//===----------------------------------------------------------------------===//
// UnboundRegionOp
//===----------------------------------------------------------------------===//

LogicalResult UnboundRegionOp::verify() {
  return emitOpError("is never valid. Was it not erased by the parser?");
}

//===----------------------------------------------------------------------===//
// ErrorReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ErrorReturnOp::verify() {
  auto func = (*this)->getParentOfType<FnOp>();
  if (!func)
    return emitOpError("expected to be nested inside a `lit.fn` operation");
  return checkOperandTypes(*this, func.getResultTypes());
}

bool ErrorReturnOp::isParentNode(Operation *op) { return isa<FnOp>(op); }

void ErrorReturnOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.size() == 1);
  targets.emplace_back(std::nullopt, getResult());
}

//===----------------------------------------------------------------------===//
// TraitFnOp
//===----------------------------------------------------------------------===//

LogicalResult LIT::TraitFnOp::verify() {
  if (llvm::isa_and_present<TraitDeclOp>(getParentOp()->getParentOp()))
    return success();

  return emitOpError("expected a parent function in a trait");
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
// RefPackCreateOp
//===----------------------------------------------------------------------===//

/// Parses a kgen.pack.create op.
///
/// operation ::=
///   `lit.ref.pack.create` `(` operands `)` attr-dict `:` result-type
///
/// This is custom because we need to match operands at each index to the
/// resulting pack type element at that index.
static ParseResult parseRefPackCreateType(AsmParser &p, Type &resultType,
                                          SmallVectorImpl<Type> &elementTypes) {
  llvm::SMLoc loc = p.getCurrentLocation();
  if (p.parseType(resultType))
    return failure();
  auto type = dyn_cast<RefPackType>(resultType);
  if (!type)
    return p.emitError(loc, "expected a !lit.ref.pack type");

  auto variadic = type.getVariadicIfResolved();
  if (!variadic) {
    // We can only infer if we know the elements of the pack type (i.e.: it is
    // backed by a variadic attribute).
    return p.emitError(loc) << "operand types cannot be "
                               "inferred for resulting pack type "
                            << type;
  }

  // The operands have the same type as the elements but wrapped in a reference
  // with the specified origin and addr space.
  ArrayRef<TypedAttr> values = variadic.getValues();
  for (TypedAttr value : values) {
    Type eltType = type.getElementRefTypeFor(ParamType::get(value));
    elementTypes.push_back(eltType);
  }
  return success();
}

static void printRefPackCreateType(OpAsmPrinter &p, Operation *op,
                                   Type resultType, TypeRange elementTypes) {
  p << resultType;
}

LogicalResult RefPackCreateOp::verify() {
  RefPackType packType = getType();
  VariadicAttr elementTypesAttr = packType.getVariadicIfResolved();
  if (!elementTypesAttr)
    return emitOpError() << "cannot create pack with parametric element types";
  ArrayRef<TypedAttr> elementTypes = elementTypesAttr.getValues();
  if (elementTypes.size() != getNumOperands()) {
    return emitOpError() << "expected " << elementTypes.size()
                         << " operands, but got " << getNumOperands();
  }
  for (auto [i, expected, provided] :
       llvm::enumerate(elementTypes, getOperandTypes())) {
    Type type = packType.getElementRefTypeFor(ParamType::get(expected));
    if (type == provided)
      continue;
    return emitOpError() << "operand #" << i << " should have type " << type
                         << " but got " << provided;
  }
  return success();
}

/// Given an argument to a function that takes a VariadicPack argument, dig
/// out the RefPackCreateOp (or ParamConstantOp) that formed it.  This is
/// guaranteed to succeed immediately during/after the parser, not later.
Value RefPackCreateOp::findRefPackCreate(Value val) {
  /// This code grovels through the IR, looking for the standard pattern of:
  ///
  ///   %1 = lit.ref.pack.create(...)
  ///   %anonymous2A_0 = lit.var.decl "anonymous*"
  ///   lit.call VariadicPack::__init__(%anonymous2A_0, %1, ...)
  ///   %4 = lit.load.consume / lit.ref.load %anonymous2A_0  <<= we are here.
  ///
  /// This happens because we're passing the VariadicPack to the callee, and
  /// it has a memory-style init.
  Value loadOperand;

  // VariadicPack is a @register_passable type so it often is immediately
  // available.  However, it gets passed by-ref to function calls.
  // If the operand is already a reference to a pack, then use it.  Otherwise
  // we must have a register pack.  Figure out how it is formed.
  if (::isa<RefType>(val.getType())) {
    loadOperand = val;
    if (auto immOp = val.getDefiningOp<RefImmutOp>())
      loadOperand = immOp.getOperand();
  } else {
    if (auto load = val.getDefiningOp<RefLoadOp>())
      loadOperand = load.getOperand();
    else if (auto load = val.getDefiningOp<LoadConsumeOp>())
      loadOperand = load.getOperand();
    else
      return {};
  }

  auto varDecl = loadOperand.getDefiningOp<VarDeclOp>();
  if (!varDecl)
    return {};

  for (Operation *user : varDecl.getResult().getUsers()) {
    // Find the store to the pack.
    auto refStore = ::dyn_cast<RefStoreOp>(user);
    if (!refStore || refStore.getDest() != varDecl.getResult())
      continue;

    auto call = refStore.getValue().getDefiningOp<LIT::CallOp>();
    if (!call || call.getNumOperands() != 1)
      continue;

    // Make sure any change to the API forces this code to get updated.
    return call.getOperand(0);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// RefPackExtractOp
//===----------------------------------------------------------------------===//

LogicalResult RefPackExtractOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto emitError = [&](const Twine &msg) -> LogicalResult {
    return mlir::emitOptionalError(loc, msg);
  };
  if (operands.size() != 1 || !isa<RefPackType>(operands[0].getType()))
    return emitError("expected 1 operand");

  auto indexAttr = dyn_cast_if_present<TypedAttr>(attrs.get("index"));
  if (!indexAttr || !indexAttr.getType().isIndex())
    return emitError("expected an index attribute");

  auto refPackTy = cast<RefPackType>(operands[0].getType());

  // The result type is a !lit.ref wrapping the type extracted from the
  // type list.  Extract the element from the type list.
  auto typeAttr = ParamOperatorAttr::get(POC::VariadicGet,
                                         refPackTy.getVariadic(), indexAttr);
  Type type = ParamType::get(typeAttr);
  inferredReturnTypes.push_back(refPackTy.getElementRefTypeFor(type));
  return success();
}

//===----------------------------------------------------------------------===//
// ClosureInitOp
//===----------------------------------------------------------------------===//

void LIT::ClosureInitOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  walkDef(getParamDecl(), ParamDefValue());
}

static ParseResult parseRegionOnly(OpAsmParser &p,
                                   ParamDeclArrayAttr &inputParams,
                                   TypeAttr &funcType, TypeAttr &type,
                                   InlineLevelAttr &inlineLevel, Region &body) {
  FnTypeGeneratorType sigGenType;
  llvm::SMLoc bodyLoc;
  SmallVector<OpAsmParser::Argument> args;
  FunctionType functionType;
  if (parseLITFunctionSignature(p, args, inputParams, functionType,
                                sigGenType) ||
      parseOptionalInline(p, inlineLevel) || p.getCurrentLocation(&bodyLoc) ||
      p.parseRegion(body, args))
    return failure();

  SmallVector<Type> argTypes;
  for (const OpAsmParser::Argument &arg : args)
    argTypes.push_back(arg.type);
  funcType = TypeAttr::get(functionType);
  type = TypeAttr::get(sigGenType);
  return success();
}

static ParseResult parseClosureInitOpValue(
    OpAsmParser &p, TypeAttr &funcTypeGenerator, TypeAttr &functionType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &captures,
    ArrayAttr &moveOrCopyCaptureSymbols, TypedAttr &captureOrigins,
    KGEN::ParamDeclArrayAttr &inputParams, KGEN::InlineLevelAttr &inlineLevel,
    Region &bodyRegion, SmallVectorImpl<Type> &captureTypes, Type &resultType,
    KGEN::ParamDeclAttr &paramDecl) {
  if (p.parseLParen())
    return failure();

  // Collect the captures and symbols, if provided.
  SmallVector<TypedAttr> origins;
  SmallVector<Attribute> copyOrMoveSymbols;
  LogicalResult result = success();
  if (failed(p.parseOptionalRParen())) {
    do {
      OpAsmParser::UnresolvedOperand capture;
      if (p.parseOperand(capture))
        return failure();
      captures.push_back(capture);

      // Parse captures and how they should be captured.
      // There are three possibilities:
      // (1) capture by value. This is a register passable type.
      // (2) capture by reference. This uses a keyword "ref" and expects a
      // lifetime. (3) capture by copy/move. There is no keyword, only a symbol
      // indicating the method to call.
      if (p.parseOptionalLSquare()) {
        // No attribute specified; capture by value.
        copyOrMoveSymbols.push_back(BoolAttr::get(p.getContext(), false));
      } else {
        if (succeeded(p.parseOptionalKeyword("ref"))) {
          // capture by reference, parse the origin.
          if (p.parseColon() ||
              parseOriginParamValue(p, origins.emplace_back()) ||
              p.parseRSquare())
            return failure();
          copyOrMoveSymbols.push_back(BoolAttr::get(p.getContext(), true));
        } else {
          // capture by copy/move, parse the symbol.
          SymbolRefAttr callee;
          ParameterExprArrayAttr paramValues;
          int numOrigins = 0;
          if (p.parseAttribute(callee) || p.parseLSquare() ||
              p.parseInteger(numOrigins) || p.parseRSquare() ||
              parseParameterValues(p, paramValues))
            return failure();

          Type existingType;
          Type selfType;
          if (p.parseLParen() || p.parseType(existingType) || p.parseComma() ||
              p.parseType(selfType) || p.parseRParen() || p.parseRSquare())
            return failure();

          // TODO MOCO-1721: Name metadata is discarded when inferring the type
          // of the symbol. Parameters and lifetimes are preserved. Lifetimes
          // should not be needed since the function call is not generated until
          // after lower-lit. I am uncertain what relies on the symbol type
          // contained the same metadata as the function type on the function
          // the symbol refers to. Revisit this TODO once the frontend pipeline
          // is built out.
          FnType funType =
              FnType::get(p.getContext(), {existingType, selfType}, {},
                          /*numImplicitOriginDecls=*/numOrigins);
          FnTypeGeneratorType signatureType = LITGeneratorType::get(
              /*inputParamTypes=*/llvm::to_vector(
                  llvm::map_range(paramValues.getValue(),
                                  [](TypedAttr in) { return in.getType(); })),
              funType,
              PogListAttr::get(
                  p.getContext(),
                  llvm::to_vector(llvm::map_range(
                      paramValues.getValue(), [&](TypedAttr param) {
                        StringAttr name = StringAttr::get(p.getContext(), "");
                        return PogMetadataAttr::get(
                            name, PassingKind::PosOnly,
                            isa<VariadicType>(param.getType()));
                        ;
                      }))));
          copyOrMoveSymbols.push_back(
              SymbolConstantAttr::get(callee, signatureType, paramValues));
        }
      }

      result = p.parseOptionalComma();
    } while (succeeded(result));
    if (p.parseRParen())
      return failure();
  }
  // Parse the function signature and the body.
  if (parseRegionOnly(p, inputParams, functionType, funcTypeGenerator,
                      inlineLevel, bodyRegion))
    return failure();
  if (p.parseColon() || p.parseLParen())
    return failure();
  if (p.parseOptionalRParen()) {
    if (p.parseTypeList(captureTypes) || p.parseRParen())
      return failure();
  }
  if (p.parseComma() || parseVarDeclType(p, resultType, paramDecl))
    return failure();

  if (captureTypes.size() != copyOrMoveSymbols.size())
    return p.emitError(p.getCurrentLocation(),
                       "expected symbols to match number of capture types");
  moveOrCopyCaptureSymbols = ArrayAttr::get(p.getContext(), copyOrMoveSymbols);
  captureOrigins = OriginSetAttr::get(p.getContext(), origins);
  return success();
}

static void printClosureInitOpValue(
    OpAsmPrinter &p, Operation *op, TypeAttr funcTypeGenerator,
    TypeAttr functionType, ValueRange captures,
    ArrayAttr moveOrCopyCaptureSymbols, TypedAttr captureOrigins,
    KGEN::ParamDeclArrayAttr inputParams, KGEN::InlineLevelAttr inlineLevel,
    Region &bodyRegion, TypeRange captureTypes, Type resultType,
    KGEN::ParamDeclAttr paramDecl) {
  p << "(";
  int i = 0;
  int j = 0;
  ArrayRef<TypedAttr> origins =
      cast<OriginSetAttr>(captureOrigins).getOperands();
  int n = captures.size();
  for (auto [capture, symbol] : llvm::zip(captures, moveOrCopyCaptureSymbols)) {
    p << capture;
    if (SymbolConstantAttr callee = dyn_cast<SymbolConstantAttr>(symbol)) {
      p << "[";
      p << callee.getSymbol();
      p << "[";
      FnTypeGeneratorType fnTypeGen =
          cast<FnTypeGeneratorType>(callee.getType());
      p << fnTypeGen.getBody().getNumImplicitOriginDecls();
      p << "]";
      printParameterValues(p, callee.getParamValues());
      p << "(";
      p.printType(fnTypeGen.getValues().getInput(0));
      p << ", ";
      p.printType(fnTypeGen.getValues().getInput(1));
      p << ")";
      p << "]";
    } else if (BoolAttr hasLifetime = dyn_cast<BoolAttr>(symbol)) {
      if (hasLifetime.getValue()) {
        p << "[ref: ";
        printOriginParamValue(p, op, origins[j++]);
        p << "]";
      }
    }
    if (++i < n)
      p << ", ";
  }
  p << ")";
  printLITFunctionSignature(
      p, &bodyRegion, inputParams, cast<FunctionType>(functionType.getValue()),
      cast<FnTypeGeneratorType>(funcTypeGenerator.getValue()));
  printOptionalInline(p, inlineLevel.getValue());
  p << ' ';
  p.printRegion(bodyRegion, /*printEntryBlockArgs=*/false);
  p << " : ";
  p << '(';
  llvm::interleaveComma(captureTypes, p, [&](Type type) { p.printType(type); });
  p << ')';
  p << ", ";
  printVarDeclType(p, op, resultType, paramDecl);
}

LogicalResult LIT::ClosureInitOp::verify() {
  if (getMoveOrCopyCaptureSymbols().size() != getCaptures().size())
    return emitOpError(
        "expected move or copy capture symbols to match number of captures");
  return success();
}

//===----------------------------------------------------------------------===//
// EndFnOp
//===----------------------------------------------------------------------===//

LogicalResult EndFnOp::verify() {
  auto func = (*this)->getParentOfType<KGEN::FunctionLike>();
  if (!func)
    return emitOpError("expected to be nested inside a function");
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "KGEN/LITDialect/LIT.cpp.inc"
