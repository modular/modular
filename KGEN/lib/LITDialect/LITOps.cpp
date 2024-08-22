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
#include "KGEN/Interpreter/InterpreterState.h"
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

LITSignatureType LIT::getCalleeType(Operation *op) {
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

LITSignatureType LIT::getFullSignature(Operation *container,
                                       LITSignatureType signature) {
  // Collect contextual params, if there are none, the full signature is the
  // same as the local signature.
  auto [ancestors, params] = collectParametricAncestors(container);
  if (params.empty())
    return signature;
  return LITSignatureType::prependParams(signature, params,
                                         getContextualVariadicMask(ancestors));
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

/// Modules don't have result parameters.
ArrayRef<ParamDeclAttr> FileModuleOp::getResultParams() { return {}; }

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
  if (p.parseCommaSeparatedList(
          AsmParser::Delimiter::OptionalSquare, [&]() -> ParseResult {
            return parseLifetimeParamValue(p, values.emplace_back());
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
  llvm::interleaveComma(implicitLifetimes, p, [&](TypedAttr value) {
    printLifetimeParamValue(p, value);
  });
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
    auto calleeLITType = dyn_cast<LITSignatureType>(calleeType);
    if (!calleeLITType)
      return p.emitError(p.getCurrentLocation(), "expected a `!lit.signature`");
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

  // Async calls don't provide result slots.
  llvm::append_range(operandTypes, values.getInputs().drop_back(
                                       calleeType.getNumAsyncReturnSlots()));
  llvm::append_range(resultTypes, values.getResults());
  return success();
}

/// Nothing to do on print.
template <typename CalleeT>
static void printCallOpTypes(AsmPrinter &, Operation *, TypeRange, TypeRange,
                             CalleeT, ArrayRef<TypedAttr>) {}

static ParseResult
parseCallOp(OpAsmParser &p, TypedAttr &calleeAttr,
            ParameterExprArrayAttr &implicitLifetimes,
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
  if (parseLifetimeParams(p, implicitLifetimes))
    return failure();
  if (callee && parseParameterValues(p, paramValues))
    return failure();
  if (p.parseOperandList(operands, AsmParser::Delimiter::Paren))
    return failure();

  if (callee) {
    SignatureType signature;
    FunctionType functionType;
    if (p.parseColon() || parseKGENSignature(p, functionType, signature))
      return failure();
    calleeAttr = SymbolConstantAttr::get(callee, paramValues, signature);
  }
  if (failed(parseCallOpTypes(p, operandTypes, resultTypes, calleeAttr,
                              implicitLifetimes)))
    return failure();
  return success();
}

static void printCallOp(OpAsmPrinter &p, Operation *op, TypedAttr calleeAttr,
                        ParameterExprArrayAttr implicitLifetimes,
                        ValueRange operands, TypeRange operandTypes,
                        TypeRange resultTypes) {
  auto symbolCst = dyn_cast<SymbolConstantAttr>(calleeAttr);
  // Optionally print the direct call syntax. Otherwise, print the parametric
  // call syntax.
  if (symbolCst)
    p << ' ' << symbolCst.getSymbol();
  else
    printParametricCallee(p, op, calleeAttr);
  printLifetimeParams(p, op, implicitLifetimes);
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
                                  ValueRange operands,
                                  std::optional<TypeRange> results) {
  FunctionType values = sig.substituteImplicitLifetimesIntoValues(
      op.getImplicitLifetimes(), [&] { return op.emitOpError(); });
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
  auto sig = dyn_cast<LITSignatureType>(getCallee().getType());
  if (!sig)
    return emitOpError("callee type must be a `!lit.signature`");
  if (failed(verifyLifetimeParams(*this, sig)))
    return failure();
  return verifyCallOp(*this, sig, getOperands(), getResultTypes());
}

SymbolRefAttr LIT::CallOp::getDirectCallee() {
  if (auto symbolCst = dyn_cast<SymbolConstantAttr>(getCallee()))
    return symbolCst.getSymbol();
  return {};
}

ErrorTreeOrSuccess LIT::CallOp::interpret(ArrayRef<Attribute> operands,
                                          InterpreterState &state) {
  SymbolRefAttr callee = getDirectCallee();
  if (!callee)
    return ErrorTree(getLoc(), "cannot interpret a parametric call");

  auto bodyOr = state.lookupFunctionBody(callee);
  if (bodyOr.isError())
    return ErrorTree(getLoc(), bodyOr.takeError());
  Region &body = **bodyOr;

  state.callFunctionBody(body, operands);
  return success();
}

FailureOr<InlineResult> LIT::CallOp::prepInline(mlir::RewriterBase &b) {
  // Inlining not supported for this op
  return failure();
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

LogicalResult LIT::CallIndirectOp::verify() {
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

/// Parse a fixed mutability specifier that occurs for implicit lifetimes.
// Implicit lifetime params are always known immut or mut, never parametric.
static ParseResult parseImplicitLifetimeMutability(AsmParser &p,
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

static void printImplicitLifetimeMutability(AsmPrinter &p, LifetimeType type) {
  assert((type.isMutableKnown(true) || type.isMutableKnown(false)) &&
         "Implicit lifetimes are always known mut or imm");
  p << (type.isMutableKnown(true) ? "mut " : "imm ");
}

// These FuncOp attributes are disallowed while parsing since they can
// be inferred. Likewise while printing we ignore them.
static StringRef disallowedAttrNames[] = {
    "sym_name",   "exportKind",     "isCExported", "constraints", "implements",
    "signature",  "functionType",   "sym_name",    "argNames",    "paramNames",
    "evaluator",  "defaultImpl",    "inlineLevel", "paramDecl",   "params",
    "decorators", "argPassingKinds"};

static ParseResult parseLITFunctionSignature(
    OpAsmParser &p, SmallVectorImpl<OpAsmParser::Argument> &args,
    ParamDeclArrayAttr &params, FunctionType &functionType,
    LITSignatureType &signature) {
  llvm::SMLoc startLoc = p.getCurrentLocation();

  SmallVector<ParamDeclAttr> lifetimeDecls;
  auto parseLifetimeDecl = [&]() -> ParseResult {
    bool isMutable = false;
    StringAttr name;
    if (parseImplicitLifetimeMutability(p, isMutable) ||
        parseParamName(p, name))
      return failure();
    lifetimeDecls.push_back(
        ParamDeclAttr::get(name, LifetimeType::get(p.getContext(), isMutable)));
    return success();
  };

  PogListAttr paramListAttr;
  if (parseOptionalParameterSpec(p, params, paramListAttr))
    return failure();

  // Parse implicit lifetime decls.
  if (p.parseCommaSeparatedList(AsmParser::Delimiter::OptionalSquare,
                                parseLifetimeDecl))
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
    if (failed(parseOptionalDefaultValue(
            p, defaultVal, arg.type,
            SignatureType::hasAddress(argConventions.back()))))
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
      paramListAttr, lifetimeDecls.size());
  signature = SignatureType::remapToSignature(
      params, /*resultParams=*/{}, functionType, argConventions, effects,
      metadata, [&] { return p.emitError(startLoc); });
  if (!signature)
    return failure();

  // Replace named implicit lifetime parameter references with index-based
  // references in the signature.
  signature = signature.replaceImplicitLifetimesWithIndexes(lifetimeDecls);

  // The formal params are the declared params + the implicit lifetime decls.
  SmallVector<ParamDeclAttr> allParams;
  allParams.reserve(params.size() + lifetimeDecls.size());
  llvm::append_range(allParams, params);
  llvm::append_range(allParams, lifetimeDecls);
  params = ParamDeclArrayAttr::get(p.getContext(), allParams);
  return success();
}

static void printLITFunctionSignature(OpAsmPrinter &p, Region *region,
                                      ArrayRef<ParamDeclAttr> params,
                                      FunctionType functionType,
                                      LITSignatureType signature) {
  ArrayRef<ParamDeclAttr> lifetimeDecls =
      params.drop_front(signature.getNumParams());

  ParameterEvaluator evaluator;
  printOptionalParameterSpec(p, params.drop_back(lifetimeDecls.size()),
                             signature.getParamListAttrs(), evaluator);

  if (!lifetimeDecls.empty()) {
    p << '[';
    llvm::interleaveComma(lifetimeDecls, p, [&](ParamDeclAttr decl) {
      printImplicitLifetimeMutability(p, cast<LifetimeType>(decl.getType()));
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
      assert(argConv == ArgConvention::BorrowedInReg ||
             argConv == ArgConvention::OwnedInReg);
      argConv = signature.getPackVarArgConvention(i);
    }
    printConventionAndVariadicness(p, argConv, variadicness[i]);

    if (TypedAttr defaultOr = defaultHandler.getDefault(i)) {
      p << " = ";
      printParamValue(
          p, cast<TypedAttr>(evaluator.getReboundAttribute(defaultOr)));
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
  if (!isParamDecl)
    result.addAttribute(getSymNameAttrName(result.name), nameAttr);

  // Parse the function signature.
  SmallVector<OpAsmParser::Argument> entryArgs;
  ParamDeclArrayAttr params;
  FunctionType functionType;
  LITSignatureType signature;
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
    p.printSymbolName(*getSymName());

  // Print the function arguments. Here we need all the use defined names.
  printLITFunctionSignature(p, &getBodyRegion(), getParams(), getFunctionType(),
                            getSignature());
  printOptionalInline(p, getInlineLevel());
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
  for (auto [idx, arg] : llvm::enumerate(getBody()->getArguments()))
    setNameFn(arg, getSignature().getArgName(idx).strref());
}

LogicalResult LIT::FuncOp::verify() {
  // Check that the number of argument labels matches the number of argument
  // types.
  if (getSignature().getMetadata().getNumArgs() !=
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

  // Verify the correct number of parameters.
  if (getSignature().getNumParams() +
          getSignature().getNumImplicitLifetimeDecls() !=
      getInputParams().size())
    return emitOpError("incorrect number of input params: have ")
           << getParams().size() << ", but expected "
           << getSignature().getNumImplicitLifetimeDecls()
           << " implicit lifetimes and " << getSignature().getNumParams()
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

SmallVector<ParamDeclAttr>
LIT::FuncOp::collectAllParams(bool includeImplLifetimes) {
  auto [_, result] = collectParametricAncestors(getOperation()->getParentOp());

  auto params = getParams();
  if (!includeImplLifetimes)
    params = params.drop_back(getSignature().getNumImplicitLifetimeDecls());
  llvm::append_range(result, params);
  return result;
}

LITSignatureType LIT::FuncOp::getFullSignature() {
  return LIT::getFullSignature((*this)->getParentOp(), getSignature());
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
  result.addAttribute(getParamsAttrName(result.name),
                      ParamDeclArrayAttr::get(ctx, {}));
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

void LIT::FuncOp::build(OpBuilder &b, OperationState &state,
                        StringAttr declName, StringRef sourceName,
                        FunctionType funcType,
                        ArrayRef<ParamDeclAttr> paramDecls, FnEffects effects,
                        InlineLevel inlineLevel) {
  MLIRContext *ctx = b.getContext();
  mlir::UnitAttr none;
  SmallVector<ArgConvention> convs(funcType.getNumInputs());
  auto sig = LITSignatureType::remapToSignature(
      paramDecls, {}, funcType, convs, effects,
      FnMetadataAttr::get(ctx, paramDecls.size(), funcType.getNumInputs()));
  build(b, state, StringAttr(), ParamDeclAttr::get(declName, sig),
        TypeAttr::get(sig), TypeAttr::get(funcType),
        ParamDeclArrayAttr::get(ctx, paramDecls), DecoratorsAttr::get(ctx, {}),
        /*isStatic=*/none, /*isDef=*/none, /*isInherited=*/none,
        /*isSynthetic=*/none, ExportKindAttr::get(ctx, ExportKind::NotExported),
        InlineLevelAttr::get(ctx, inlineLevel), b.getI8IntegerAttr(0),
        FlatSymbolRefAttr(), StringAttr(), StringAttr(),
        b.getStringAttr(sourceName), StringAttr(), DocStringAttr(),
        StringAttr(), DictionaryAttr::get(ctx));
  state.regions[0]->push_back(new Block());
}

/// Build a function in a default configuration, used by member synthesization.
void LIT::FuncOp::build(OpBuilder &builder, OperationState &result,
                        StringAttr name, StringAttr sourceName,
                        SignatureType signature,
                        SpecialFunctionKind specialFnKind) {
  MLIRContext *ctx = builder.getContext();
  mlir::UnitAttr none;
  build(builder, result, name, ParamDeclAttr(), TypeAttr::get(signature),
        TypeAttr::get(signature.getValues()), ParamDeclArrayAttr::get(ctx, {}),
        DecoratorsAttr::get(ctx, {}), /*isStatic=*/none, /*isDef=*/none,
        /*isInherited=*/none, /*isSynthetic=*/none,
        ExportKindAttr::get(ctx, ExportKind::NotExported),
        InlineLevelAttr::get(ctx, InlineLevel::Automatic),
        builder.getI8IntegerAttr(uint8_t(specialFnKind)), FlatSymbolRefAttr(),
        StringAttr(), StringAttr(), sourceName, StringAttr(), DocStringAttr(),
        StringAttr(), DictionaryAttr::get(ctx));
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
                 stringifyEnum(TypeConvention::RegisterPassableTrivial)})))
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
                                            TypeLineageArrayAttr &parentTypes) {
  llvm::SMLoc startLoc = p.getCurrentLocation();
  PogListAttr paramListAttr;
  if (parseOptionalParameterSpec(p, params, paramListAttr))
    return failure();

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
      [&] { return p.emitError(startLoc); }, params, paramListAttr);
  if (!sig)
    return failure();
  signature = TypeAttr::get(sig);
  return success();
}

static void printStructParameterSpec(AsmPrinter &p, Operation *op,
                                     ArrayRef<ParamDeclAttr> params,
                                     TypeAttr signature,
                                     ArrayRef<TypeLineageAttr> parentTypes) {
  auto sig = cast<TypeSignatureType>(signature.getValue());
  ParameterEvaluator evaluator;
  printOptionalParameterSpec(p, params, sig.getParamListAttrs(), evaluator);
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
    return LIT::StructType::get(symbol, unbound,
                                AnyStructType::get(symbol, unbound, sig));
  }

  // Compute the resultant signature.
  SmallVector<Type> newParamTypes;
  SmallVector<PogMetadataAttr> newPogs;
  SmallVector<TypedAttr> newPosDefaults;
  SmallVector<TypedAttr> newKwOnlyDefaults;

  PogListAttr paramListAttr = sig.getParamListAttrs();
  DefaultValueHandler defaultHandler(paramListAttr);

  for (auto [i, value, type, pogAttr] : llvm::enumerate(
           paramValues, sig.getParamTypes(), paramListAttr.getPogs())) {
    if (::isa<UnboundAttr>(value)) {
      newParamTypes.push_back(type);
      newPogs.push_back(pogAttr);

      if (TypedAttr defaultOr = defaultHandler.getPosDefault(i))
        newPosDefaults.push_back(defaultOr);
      else if (TypedAttr defaultOr = defaultHandler.getKwOnlyDefault(i))
        newKwOnlyDefaults.push_back(defaultOr);
    }
  }

  MLIRContext *ctx = getContext();
  auto newParamListAttr =
      PogListAttr::get(ctx, newPogs, newPosDefaults, newKwOnlyDefaults);
  auto newSig = TypeSignatureType::get(ctx, newParamTypes, newParamListAttr);
  return LIT::StructType::get(symbol, paramValues,
                              AnyStructType::get(symbol, paramValues, newSig));
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
        TypeLineageArrayAttr::get(ctx, {}), /*isSynthetic=*/{},
        /*nonmaterializableTarget=*/{}, /*destructor=*/{}, /*moveInit=*/{},
        /*copyInit=*/{}, /*closureSignature=*/{}, /*docString=*/{},
        /*deprecationWarning=*/{}, /*sourceName=*/{}, /*convention*/ {});
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
// StructCreateOp
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
  return LITStructAttr::get(values, getType());
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

RefType RefStructGEROp::getFieldType(RefType structRefTy, StructFieldOp field) {
  auto structTy = cast<StructType>(structRefTy.getElementType());
  return structRefTy.getWithElement(field.getReboundType(structTy));
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

OpFoldResult RefStructGEROp::fold(FoldAdaptor adaptor) {
  auto value = cast_or_null<TypedAttr>(adaptor.getContainer());
  if (!isa_and_nonnull<SymbolicPointerAttr, StructGERAttr>(value))
    return {};
  return StructGERAttr::get(value, getFieldAttr(), getType());
}

//===----------------------------------------------------------------------===//
// RefLoadOp
//===----------------------------------------------------------------------===//

static ErrorTreeOr<TypedAttr>
interpretSymbolicLoadOp(Location loc, TypedAttr arg, InterpreterState &state) {
  if (auto ptr = dyn_cast<SymbolicPointerAttr>(arg)) {
    // Base case: load the value.
    ErrorOr<TypedAttr &> mem = state.getSymbolicMemory(ptr.getSlot());
    if (mem.isError())
      return ErrorTree(loc, mem.takeError());
    return *mem;
  }
  // If this is a GER, recurse by loading the base value and then extracting the
  // requested element.
  auto ger = cast<StructGERAttr>(arg);
  ErrorTreeOr<TypedAttr> value =
      interpretSymbolicLoadOp(loc, ger.getValue(), state);
  if (value.isError())
    return value.takeError();

  auto structAttr = cast<LITStructAttr>(*value);
  for (auto [name, value] : structAttr.getValues()) {
    if (name == ger.getField())
      return value;
  }
  llvm_unreachable("should have found a matching field name");
}

ErrorTreeOrSuccess RefLoadOp::interpret(ArrayRef<Attribute> operands,
                                        InterpreterState &state) {
  ErrorTreeOr<TypedAttr> result = interpretSymbolicLoadOp(
      getLoc(), cast<TypedAttr>(operands.front()), state);
  if (result.isError())
    return result.takeError();
  state.mapResults(*result);
  return success();
}

//===----------------------------------------------------------------------===//
// RefStoreOp
//===----------------------------------------------------------------------===//

static ErrorTreeOrSuccess interpretSymbolicStoreOp(Location loc,
                                                   TypedAttr value,
                                                   TypedAttr target,
                                                   InterpreterState &state) {
  // Build the set of field accesses up to the base value. If `target` is
  // already a full object reference, then `fields` will be empty.
  SmallVector<StringAttr, 1> fields;
  for (StructGERAttr ger; (ger = dyn_cast<StructGERAttr>(target));
       target = ger.getValue())
    fields.push_back(ger.getField());

  // `target` must be a full object reference. Read the whole struct.
  auto ptr = cast<SymbolicPointerAttr>(target);
  ErrorOr<TypedAttr &> mem = state.getSymbolicMemory(ptr.getSlot());
  if (mem.isError())
    return ErrorTree(loc, mem.takeError());

  // Build the chain of accessed element values, starting with the full object.
  SmallVector<std::pair<TypedAttr, int>> values;
  values.emplace_back(*mem, -1);
  for (StringAttr field : llvm::reverse(fields)) {
    auto attr = cast<LITStructAttr>(values.back().first);
    int i = 0;
    for (auto [name, value] : attr.getValues()) {
      if (name == field) {
        values.emplace_back(value, i);
        break;
      }
      ++i;
    }
  }
  assert(values.size() == fields.size() + 1 && "invalid field attribute name");

  // Now overwrite the leaf element and reconstruct the full object.
  values.back().first = value;
  while (values.size() != 1) {
    auto [value, i] = values.back();
    values.pop_back();
    auto attr = cast<LITStructAttr>(values.back().first);
    SmallVector<std::tuple<StringAttr, TypedAttr>> elements =
        llvm::to_vector(attr.getValues());
    std::get<1>(elements[i]) = value;
    values.back().first = LITStructAttr::get(elements, attr.getType());
  }

  // There should be a single value in the vector now. Overwrite the
  // whole-object value in symbolic memory.
  *mem = values.back().first;
  return success();
}

ErrorTreeOrSuccess RefStoreOp::interpret(ArrayRef<Attribute> operands,
                                         InterpreterState &state) {
  return interpretSymbolicStoreOp(getLoc(), cast<TypedAttr>(operands.front()),
                                  cast<TypedAttr>(operands.back()), state);
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
                             Value pointer, TypedAttr lifetime,
                             bool startsUninit, bool endsUninit) {
  auto ptr = cast<PointerType>(pointer.getType());
  auto refType =
      RefType::get(ptr.getElementType(), lifetime, ptr.getAddressSpace());
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
        /*dtorSig=*/{}, /*docString=*/{}, /*deprecationWarning=*/{});
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
// VarDeclOp
//===----------------------------------------------------------------------===//

static ParseResult parseVarDeclType(AsmParser &p, Type &resultType,
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
                      StringRef name, StringRef lifetimeName,
                      VarDeclKind kind) {
  auto lifetimeType = b.getType<LifetimeType>(/*isMutable=*/true);
  auto lifetimeNameAttr = b.getAttr<StringAttr>(lifetimeName);
  auto lifetimeDecl = ParamDeclAttr::get(lifetimeNameAttr, lifetimeType);
  auto resultType = RefType::get(
      elementType, ParamDeclRefAttr::get(lifetimeNameAttr, lifetimeType));
  build(b, state, resultType, name, kind, lifetimeDecl, /*argShadowIndex=*/{});
}

bool VarDeclOp::isSynthetic() { return getKind() == VarDeclKind::Synthesized; }

LogicalResult VarDeclOp::verify() {
  if (getArgShadowIndex().has_value() && getKind() != VarDeclKind::Arg)
    return emitOpError() << "cannot have arg index unless is arg kind";
  return success();
}

ErrorTreeOrSuccess VarDeclOp::interpret(ArrayRef<Attribute> operands,
                                        InterpreterState &state) {
  ErrorOr<TypedAttr> value =
      createUninitializedValueOf(getType().getElementType(), state);
  if (value.isError())
    return ErrorTree(getLoc(), value.takeError());

  uint64_t result = state.allocateSymbolicMemory(*value);
  state.mapResults(SymbolicPointerAttr::get(result, getType()));
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
  build(builder, state, RefType::getImmortal(op.getType(), /*isMut=*/true),
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
                      TypedAttr callee, ArrayRef<TypedAttr> implicitLifetimes) {
  SmallVector<Type> resultTypes;
  return parseCallOpTypes(p, operandTypes, resultTypes, callee,
                          implicitLifetimes);
}

/// Nothing to do on print.
static void printAsyncCallOpTypes(AsmPrinter &, Operation *, TypeRange,
                                  TypedAttr, ArrayRef<TypedAttr>) {}

LogicalResult AsyncCallOp::verify() {
  auto sig = cast<SignatureType>(getCallee().getType());
  if (!sig.isAsync())
    return emitOpError("callable must be 'async'");
  if (auto litSig = dyn_cast<LITSignatureType>(sig)) {
    if (failed(verifyLifetimeParams(*this, litSig)) ||
        failed(verifyCallOp(*this, litSig, getOperands(), /*results=*/{})))
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
  auto func = (*this)->getParentOfType<LIT::FuncOp>();
  if (!func)
    return emitOpError("expected to be nested inside a `lit.func` operation");
  return checkOperandTypes(*this, func.getResultTypes());
}

ErrorTreeOrSuccess LIT::ReturnOp::interpret(ArrayRef<Attribute> operands,
                                            InterpreterState &state) {
  state.returnFromFunction(operands);
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
// UnboundRegionOp
//===----------------------------------------------------------------------===//

LogicalResult UnboundRegionOp::verify() {
  return emitOpError("is never valid. Was it not erased by the parser?");
}

//===----------------------------------------------------------------------===//
// LoadConsumeOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess LoadConsumeOp::interpret(ArrayRef<Attribute> operands,
                                            InterpreterState &state) {
  ErrorTreeOr<TypedAttr> result = interpretSymbolicLoadOp(
      getLoc(), cast<TypedAttr>(operands.front()), state);
  if (result.isError())
    return result.takeError();
  state.mapResults(*result);
  return success();
}

//===----------------------------------------------------------------------===//
// ErrorReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ErrorReturnOp::verify() {
  auto func = (*this)->getParentOfType<LIT::FuncOp>();
  if (!func)
    return emitOpError("expected to be nested inside a `lit.func` operation");
  return checkOperandTypes(*this, func.getResultTypes());
}

bool ErrorReturnOp::isParentNode(Operation *op) { return isa<LIT::FuncOp>(op); }

void ErrorReturnOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.size() == 1);
  targets.emplace_back(std::nullopt, getResult());
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
  // with the specified lifetime and addr space.
  ArrayRef<TypedAttr> values = variadic.getValues();
  for (TypedAttr value : values) {
    Type eltType = type.getElementRefTypeFor(ParamRefType::get(value));
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
    Type type = packType.getElementRefTypeFor(ParamRefType::get(expected));
    if (type == provided)
      continue;
    return emitOpError() << "operand #" << i << " should have type " << type
                         << " but got " << provided;
  }
  return success();
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
  Type type = ParamRefType::get(typeAttr);
  inferredReturnTypes.push_back(refPackTy.getElementRefTypeFor(type));
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "KGEN/LITDialect/LIT.cpp.inc"
