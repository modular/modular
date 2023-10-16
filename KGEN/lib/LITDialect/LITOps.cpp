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
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/LITDialect/SpecialFunctions.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

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

bool LIT::findTryBlock(Block *currentBlock) {
  Operation *parentOp;
  while (currentBlock && (parentOp = currentBlock->getParentOp())) {
    if (isa<LIT::FuncOp>(parentOp))
      break;
    TryOp tryOp = dyn_cast<TryOp>(parentOp);
    if (tryOp)
      if (&tryOp.getTryRegion().front() == currentBlock)
        return true;
    currentBlock = parentOp->getBlock();
  }
  return false;
}

//===----------------------------------------------------------------------===//
// LIT::MangledSymbol
//===----------------------------------------------------------------------===//

LIT::MangledSymbol LIT::MangledSymbol::mangle(mlir::SymbolOpInterface op) {
  MangledSymbol out;
  // The parser mangles the argument types into the symbol name.
  size_t firstParen = op.getName().find('(');
  if (firstParen == std::string::npos)
    firstParen = op.getName().size();
  // Get the name of the func.
  out.symName =
      StringAttr::get(op.getContext(), op.getName().take_front(firstParen));
  out.identifier = StringAttr::get(
      op->getContext(),
      op.getName().take_front(op.getName().find_first_of("[(")));

  auto signatureStr =
      StringAttr::get(op.getContext(), op.getName().drop_front(firstParen));
  // If the operation is function-like, we can get its signature. However, using
  // it for name mangling breaks a lot of things right now.
  // TODO(10920): We have to re-evaluate if we want to have the parser doing
  //   some of this, or if we want to mangle it here.
  if (auto funcLike = dyn_cast<FuncInterface>(op.getOperation()))
    out.signature = funcLike.getFunctionType();
  else
    out.signature = nullptr;

  // Grab parent structs/modules/etc., add them in order from in -> out (they'll
  // be added to the name from out->in).
  Operation *parentOp = op;
  while ((parentOp = parentOp->getParentOp())) {
    TypeSwitch<Operation *>(parentOp)
        .Case([&](StructDeclOp op) {
          out.structNames.push_back(op.getNameAttr());
        })
        .Case<FileModuleOp, PackageOp>(
            [&](auto op) { out.moduleNames.push_back(op.getNameAttr()); });
  }
  std::reverse(out.structNames.begin(), out.structNames.end());
  std::reverse(out.moduleNames.begin(), out.moduleNames.end());

  std::string mangledName;
  llvm::raw_string_ostream nameStream(mangledName);
  // Emit the parent module and struct names. Module names are prefixed with `$`
  // - which provides a signal for what's a module vs struct when demangling.
  for (auto name : llvm::concat<StringAttr>(out.moduleNames, out.structNames))
    nameStream << name.getValue() << "::";
  // Finally, function name and argument types. Use the string coming out of the
  // parser rather than the actual function type.
  nameStream << out.symName.getValue() << signatureStr.getValue();

  out.mangled = StringAttr::get(op.getContext(), mangledName);
  return out;
}

/// Parse a mangled signature from `typeStr`. Expects a signature that looks
/// like `(type1,type2)rtype1,rtype2`.
static FailureOr<FunctionType> parseMangledSignature(MLIRContext *ctx,
                                                     StringRef typeStr) {
  SmallVector<Type> inputTypes, resultTypes;
  SmallVector<Type> *typeVec = &inputTypes;
  if (typeStr.empty())
    return FunctionType{};

  // Drop the first '(' if there is one.
  if (typeStr.starts_with("("))
    typeStr = typeStr.drop_front();

  // If the first thing in the string is the closing paren, move straight to
  // result types.
  if (typeStr.starts_with(")")) {
    typeStr = typeStr.drop_front();
    typeVec = &resultTypes;
  }

  // Now, parse the type string.
  while (!typeStr.empty()) {
    size_t numBytes = 0;
    Type t = mlir::parseType(typeStr, ctx, &numBytes);
    if (!t)
      return failure();

    typeVec->push_back(t);
    typeStr = typeStr.drop_front(numBytes);
    // Drop the comma.
    if (typeStr.starts_with(","))
      typeStr = typeStr.drop_front();

    // If we have reached the closing paren, then skip it and parse any
    // leftovers into the result types.
    if (typeStr.starts_with(")")) {
      typeStr = typeStr.drop_front();
      typeVec = &resultTypes;
    }
  }

  return FunctionType::get(ctx, inputTypes, resultTypes);
}

FailureOr<LIT::MangledSymbol>
LIT::MangledSymbol::demangle(StringAttr mangled, bool parseSignature) {
  MangledSymbol out;
  out.mangled = mangled;
  StringRef m = mangled.getValue();
  // We'll first tokenize the owning module and structs.
  size_t separator = m.find("::");
  size_t firstOpen = m.find_first_of("([");
  for (; separator != std::string::npos && separator < firstOpen;
       separator = m.find("::"), firstOpen = m.find_first_of("([")) {
    StringRef current = m.take_front(separator);
    // Drop until the separator.
    m = m.drop_front(separator);
    // Skip past the separator as well (if it exists).
    m.consume_front("::");
    // It's a module name if it starts with a leading `$`.
    if (current.starts_with("$"))
      out.moduleNames.push_back(
          StringAttr::get(mangled.getContext(), current.drop_front()));
    else
      out.structNames.push_back(StringAttr::get(mangled.getContext(), current));
  }
  // Get the name of the func and the types of its arguments.
  StringRef nameWithParameters = m.take_front(m.find('('));
  StringRef nameWithoutParameters = m.take_front(firstOpen);

  out.symName = StringAttr::get(mangled.getContext(), nameWithParameters);
  out.identifier = StringAttr::get(mangled.getContext(), nameWithoutParameters);

  size_t firstParen = m.find('(');
  if (firstParen == std::string::npos)
    firstParen = m.size();

  // If there's no parenthesis here, don't even parse out the signature.
  if (firstParen == m.size()) {
    out.signature = nullptr;
    return out;
  }

  // If there are more mangled symbols, then there are Mojo types we cannot
  // parse in general.
  if (separator != std::string::npos)
    return out;

  if (!parseSignature)
    return out;

  // If we *have* a signature, parse it out.
  FailureOr<FunctionType> sigOr =
      parseMangledSignature(mangled.getContext(), m.drop_front(firstParen));
  if (failed(sigOr))
    return failure();

  out.signature = *sigOr;
  return out;
}

llvm::raw_ostream &LIT::operator<<(raw_ostream &os,
                                   const LIT::MangledSymbol &ms) {
  os << "Mangled: \"";
  // Need to escape the mangled string, it might have some characters that
  // terminals don't like.
  llvm::printEscapedString(ms.mangled.getValue(), os);
  os << "\" - ";
  os << "Modules: [";
  llvm::interleaveComma(ms.moduleNames, os);
  os << "], Structs: [";
  llvm::interleaveComma(ms.structNames, os);
  os << "], Symbol: " << ms.symName;
  os << ", Identifier: " << ms.identifier;
  os << ", Signature: ";
  if (ms.signature)
    os << ms.signature;
  else
    os << "(none)";
  return os;
}

//===----------------------------------------------------------------------===//
// FileModuleOp
//===----------------------------------------------------------------------===//

void FileModuleOp::build(OpBuilder &odsBuilder, OperationState &state,
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

void PackageOp::build(OpBuilder &odsBuilder, OperationState &state,
                      StringAttr name) {
  state.addAttribute(getSymNameAttrName(state.name), name);
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
// FuncOp
//===----------------------------------------------------------------------===//

/// If this is a special function like __init__ return the enum that
/// identifies it, otherwise return kNormal.
SpecialFunctionKind SpecialFunctionInfo::getKind(StringRef name) {
  if (name.size() < 5 || !name.startswith("__") || !name.endswith("__"))
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

Type LIT::getSignatureUserResultType(SignatureType sigType,
                                     ArrayRef<Type> argTypes, Type resultType) {
  // If this function is a memory only type, return the by-ref result.
  if (sigType.hasMemoryOnlyResult())
    return cast<PointerType>(argTypes.front()).getElementAsType();

  // Otherwise it is the normal result.
  if (sigType.isThrows())
    return cast<POP::VariantType>(resultType).getType(1);
  return resultType;
}

/// Returns the user-defined result type, looking through implicit memory
/// results and stripping off the variant from error throwing results if needed.
Type LIT::FuncOp::getUserResultType() {
  return LIT::getSignatureUserResultType(getSignature(), getArgumentTypes(),
                                         getMLIRResultType());
}

/// Return a SymbolConstantAttr for this function, optionally bound to a set
/// of parameter bindings.
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

// These FuncOp attributes are disallowed while parsing since they can
// be inferred. Likewise while printing we ignore them.
static StringRef disallowedAttrNames[] = {
    "exportKind",     "isCExported",  "constraints",  "implements",
    "signature",      "functionType", "sym_name",     "argNames",
    "paramNames",     "evaluator",    "defaultImpl",  "inlineLevel",
    "paramDecl",      "inputParams",  "resultParams", "decorators",
    "argPassingKinds"};

static ParseResult parseLITFunctionSignature(
    OpAsmParser &p, SmallVectorImpl<OpAsmParser::Argument> &args,
    ParamDeclArrayAttr &inputParams, ParamDeclArrayAttr &resultParams,
    FunctionType &functionType, LITSignatureType &signature) {
  llvm::SMLoc loc = p.getCurrentLocation();

  SmallVector<StringAttr> paramNames;
  SmallVector<TypedAttr> defaultParams;
  if (parseOptionalParameterSpec(p, inputParams, resultParams, paramNames,
                                 defaultParams))
    return failure();

  SmallVector<StringAttr> argNames;
  SmallVector<TypedAttr> defaults;
  SmallVector<ValueInputConvention> inputConventions;

  StarSlashParser ssParser(p, loc);
  auto parseArg = [&](SmallVectorImpl<Type> &argTypes) -> ParseResult {
    if (OptionalParseResult res = ssParser.parseOptionalStarSlash();
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
        parseInputConvention(p, inputConventions.emplace_back()))
      return failure();

    // Parse an optional default value.
    TypedAttr defaultVal;
    if (failed(parseOptionalDefaultValue(
            p, defaultVal, arg.type,
            SignatureType::hasAddress(inputConventions.back()))))
      return failure();
    if (defaultVal)
      defaults.emplace_back(defaultVal);

    argTypes.push_back(arg.type);
    return success();
  };

  FnEffects effects;
  if (failed(parseSignatureValues(p, parseArg, functionType, effects,
                                  /*optionalResultList=*/true)))
    return failure();

  auto [numPosOnly, numPosOrKw, numKwOnly] = ssParser.getNumPassingKinds();
  SmallVector<PassingKind> argPassingKinds(numPosOnly, PassingKind::PosOnly);
  argPassingKinds.append(numPosOrKw, PassingKind::PosOrKw);
  argPassingKinds.append(numKwOnly, PassingKind::KwOnly);

  signature = IndexRefRemapper::remapToSignature(
      inputParams, resultParams, functionType, inputConventions, effects,
      FnMetadataAttr::get(p.getContext(), argNames, argPassingKinds, paramNames,
                          defaults, defaultParams),
      [&] { return p.emitError(loc); });
  return success(!!signature);
}

static void printLITFunctionSignature(OpAsmPrinter &p, Region *region,
                                      ArrayRef<StringAttr> argNames,
                                      ArrayRef<ParamDeclAttr> inputParams,
                                      ArrayRef<ParamDeclAttr> resultParams,
                                      FunctionType functionType,
                                      LITSignatureType signature) {
  ParameterEvaluator evaluator;
  printOptionalParameterSpec(p, inputParams, resultParams,
                             signature.getParamNames(),
                             signature.getDefaultParameters(), evaluator);

  // Substitute input and result parameters when printing default arguments.
  ArrayRef<TypedAttr> defaultArgs = signature.getDefaultArguments();
  size_t numInputs = signature.getNumInputs();
  size_t defaultStartIndex = numInputs - defaultArgs.size();

  StarSlashPrinter ssPrinter(p, numInputs, '|');
  auto printElt = [&](unsigned i) {
    ssPrinter.printOptionalStarSlash(signature.getArgPassingKinds()[i], i);

    // Print the SSA name first, followed by the user-defined argument name in
    // brackets, and the type.
    BlockArgument arg = region->getArgument(i);
    p.printOperand(arg);
    p << "[";
    printParamName(p, argNames[i]);
    p << "]: ";
    p.printType(arg.getType());

    // Then we print the optional location before and input convention.
    p.printOptionalLocationSpecifier(arg.getLoc());
    printInputConvention(p, signature.getInputConvention(i));

    if (i >= defaultStartIndex) {
      p << " = ";
      printParamValue(p, cast<TypedAttr>(evaluator.getReboundAttribute(
                             defaultArgs[i - defaultStartIndex])));
    }

    // Check if we are at the end; if so, we might still have to print a '/'.
    ssPrinter.printOptionalTrailingSlash(i);
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
  result.addAttribute(getResultParamsAttrName(result.name), resultParams);
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
  ignoredAttrNames.emplace_back(mlir::SymbolTable::getSymbolAttrName());

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

LITSignatureType LIT::FuncOp::getFullSignature() {
  LITSignatureType signature = getSignature();

  // Collect contextual params, if there are none, the full signature is the
  // same as the local signature.
  SmallVector<ParamDeclAttr> inputParams;
  collectContextParameters(getOperation()->getParentOp(), inputParams);
  if (inputParams.empty())
    return signature;

  return IndexRefRemapper::prependParams(signature, inputParams);
}

void LIT::FuncOp::build(OpBuilder &builder, OperationState &result) {
  auto context = builder.getContext();

  // Before resolution, we treat the function as having type ()->Error,
  // because parse or other errors forming the signature won't update the
  // representation.  This makes sure that the error case doesn't break
  // invariants (that functions always have a single result).
  auto errorType = builder.getType<TypeCheckErrorType>();
  auto signatureType =
      LITSignatureType::get(context, ArrayRef<Type>(), {errorType});

  auto emptyParamNames = StringArrayAttr::get(context, {});
  auto emptyParamDecls = ParamDeclArrayAttr::get(context, {});

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
  result.addAttribute("sym_namex", emptyParamNames);

  result.addAttribute(getExportKindAttrName(result.name),
                      ExportKindAttr::get(context, ExportKind::NotExported));
  result.addAttribute(getSignatureAttrName(result.name),
                      TypeAttr::get(signatureType));
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(signatureType.getValues()));
  result.addAttribute(getInputParamsAttrName(result.name), emptyParamDecls);
  result.addAttribute(getResultParamsAttrName(result.name), emptyParamDecls);
  result.addAttribute(getConstraintsAttrName(result.name),
                      ConstraintArrayAttr::get(context, {}));
  result.addAttribute(getDecoratorsAttrName(result.name),
                      DecoratorsAttr::get(context, {}));
  result.addAttribute(getSpecialFnKindAttrName(result.name),
                      builder.getI8IntegerAttr(0));
  result.addAttribute(getInlineLevelAttrName(result.name),
                      InlineLevelAttr::get(context, InlineLevel::Automatic));

  result.addRegion()->push_back(new Block());
}

/// Build a function in a default configuration, used by member synthesization.
void LIT::FuncOp::build(OpBuilder &builder, OperationState &result,
                        StringAttr name, SignatureType signature,
                        SpecialFunctionKind specialFnKind) {
  MLIRContext *ctx = builder.getContext();
  build(builder, result, name, ParamDeclAttr(), TypeAttr::get(signature),
        TypeAttr::get(signature.getValues()),
        /*inputParams=*/ParamDeclArrayAttr::get(ctx, {}),
        /*resultParams=*/ParamDeclArrayAttr::get(ctx, {}),
        ConstraintArrayAttr::get(ctx, {}), DecoratorsAttr::get(ctx, {}),
        /*isStatic=*/mlir::UnitAttr(), /*isAdaptive=*/mlir::UnitAttr(),
        /*isParametric=*/mlir::UnitAttr(), /*isDef=*/mlir::UnitAttr(),
        ExportKindAttr::get(ctx, ExportKind::NotExported),
        InlineLevelAttr::get(ctx, InlineLevel::Automatic),
        builder.getI8IntegerAttr(uint8_t(specialFnKind)), FlatSymbolRefAttr(),
        StringAttr(), StringAttr(), DocStringAttr());

  result.regions[0]->push_back(new Block());
}

//===----------------------------------------------------------------------===//
// StructDeclOp
//===----------------------------------------------------------------------===//

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

LogicalResult
StructDeclOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (!getTraitsAttr())
    return success();

  KGENModule module = KGENModule::from(*this, symbolTable);
  for (SymbolRefAttr trait : getTraitsAttr()) {
    auto traitDeclOp = module.lookup<TraitDeclOp>(trait);
    if (!traitDeclOp)
      return emitOpError("expected to find a trait decl of ")
             << trait << " for struct";
  }
  return success();
}

void StructDeclOp::build(OpBuilder &builder, OperationState &result,
                         StringAttr name) {
  MLIRContext *ctx = builder.getContext();
  build(builder, result, name, ParamDeclArrayAttr::get(ctx, {}),
        StringArrayAttr::get(ctx, {}), DecoratorsAttr::get(ctx, {}),
        /*paramVarArgs=*/false,
        /*defaultParameters=*/ParameterExprArrayAttr::get(ctx, {}),
        /*registerPassable=*/0,
        /*traits=*/nullptr,
        /*nonmaterializableTarget=*/nullptr,
        /*destructor=*/nullptr, /*moveInit=*/nullptr,
        /*copyInit=*/nullptr, /*closureSignature=*/nullptr,
        /*docString=*/nullptr);
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
  ParameterEvaluator evaluator(structSelfType.getParamValues());
  return evaluator.getReboundType(getType());
}

void StructFieldOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                          StringAttr name, Type type) {
  build(odsBuilder, odsState, name, type, nullptr);
}

void StructFieldOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                          const Twine &name, Type type) {
  build(odsBuilder, odsState, odsBuilder.getStringAttr(name), type);
}

//===----------------------------------------------------------------------===//
// StructCreateOp
//===----------------------------------------------------------------------===//

/// Lookup the declaration for the struct. When checking field types, we can't
/// directly compare operation types to the struct field types because they are
/// parameterized under different domains. We have to rebind them.
static StructDeclOp lookupStructDecl(SymbolTableCollection &symbolTable,
                                     Operation *user, DeclRefType ref) {
  auto module = KGENModule::from(user, symbolTable);
  return module.lookup<StructDeclOp>(ref.getSymbol());
}

/// Verify the reference struct type.
LogicalResult
StructCreateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verify the types of the fields in the operands match those in the
  // struct declaration.
  ParameterEvaluator evaluator(getType().getParamValues());
  StructDeclOp structDecl = lookupStructDecl(symbolTable, *this, getType());
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

OpFoldResult StructCreateOp::fold(FoldAdaptor adaptor) {
  SmallVector<std::tuple<StringAttr, TypedAttr>> values;
  for (auto [name, value] : llvm::zip(getFields(), adaptor.getOperands())) {
    if (!value)
      return {};
    values.emplace_back(name, cast<TypedAttr>(value));
  }
  return StructAttr::get(getContext(), values, getType());
}

//===----------------------------------------------------------------------===//
// StructInsertOp
//===----------------------------------------------------------------------===//

LogicalResult
StructInsertOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  ParameterEvaluator evaluator(getType().getParamValues());
  StructDeclOp structDecl = lookupStructDecl(symbolTable, *this, getType());

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
  auto value = dyn_cast_if_present<StructAttr>(adaptor.getContainer());
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
  return StructAttr::get(getContext(), values, getType());
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

static LogicalResult
verifyStructFieldAndType(SymbolTableCollection &symbolTable, Operation *op,
                         DeclRefType ref, StringAttr fieldName, Type type) {
  ParameterEvaluator evaluator(ref.getParamValues());
  StructDeclOp structDecl = lookupStructDecl(symbolTable, op, ref);
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
StructExtractOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyStructFieldAndType(symbolTable, *this, getContainer().getType(),
                                  getFieldAttr(), getValue().getType());
}

void StructExtractOp::build(OpBuilder &builder, OperationState &result,
                            Value structBase, StructFieldOp field) {
  auto structType = cast<DeclRefType>(structBase.getType());
  build(builder, result, field.getReboundType(structType), structBase,
        field.getNameAttr());
}

OpFoldResult StructExtractOp::fold(FoldAdaptor adaptor) {
  if (auto value = adaptor.getContainer())
    return StructExtractAttr::get(cast<TypedAttr>(value), getFieldAttr(),
                                  getType());

  // Fold
  //  %S = lit.struct.create(a=%a, b=%b)
  //  %x = lit.struct.extract %S[a]
  // into %a.
  if (auto create = getContainer().getDefiningOp<StructCreateOp>()) {
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
// StructGEPOp
//===----------------------------------------------------------------------===//

LogicalResult
StructGEPOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Type structType = getContainer().getType().getElementAsType();
  return verifyStructFieldAndType(symbolTable, *this,
                                  cast<DeclRefType>(structType), getFieldAttr(),
                                  getResult().getType().getElementAsType());
}

void StructGEPOp::build(OpBuilder &builder, OperationState &result,
                        Value structBasePtr, StructFieldOp field) {
  Type eltType = cast<PointerType>(structBasePtr.getType()).getElementAsType();
  auto structType = field.getReboundType(cast<DeclRefType>(eltType));
  build(builder, result, PointerType::get(structType), field.getNameAttr(),
        structBasePtr);
}

//===----------------------------------------------------------------------===//
// RefStructGEROp
//===----------------------------------------------------------------------===//

LogicalResult
RefStructGEROp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Type structType = getContainer().getType().getElementAsType();
  return verifyStructFieldAndType(symbolTable, *this,
                                  cast<DeclRefType>(structType), getFieldAttr(),
                                  getResult().getType().getElementAsType());
}

void RefStructGEROp::build(OpBuilder &builder, OperationState &result,
                           Value structBasePtr, StructFieldOp field) {
  auto refType = cast<RefType>(structBasePtr.getType());
  Type eltType = refType.getElementAsType();
  auto structType = field.getReboundType(cast<DeclRefType>(eltType));
  build(builder, result,
        RefType::get(refType.getIsMutable(), structType, refType.getLifetime()),
        field.getNameAttr(), structBasePtr);
}

//===----------------------------------------------------------------------===//
// TraitDeclOp
//===----------------------------------------------------------------------===//

DebugInfo::DIScopeAttr TraitDeclOp::getLocScope() {
  return getTopLevelScope(*this);
}

void TraitDeclOp::build(OpBuilder &builder, OperationState &result,
                        StringAttr name) {
  build(builder, result, name, /*docString=*/nullptr);
  result.regions[0]->push_back(new Block());
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
  if (!refType || !refType.getIsMutable())
    return p.emitError(p.getNameLoc(),
                       "expected a mutable !lit.ref<> result type");
  // The lifetime must be a simple name, which becomes the name we are
  // declaring.
  auto lifetime = dyn_cast<ParamDeclRefAttr>(refType.getLifetime());
  if (!lifetime)
    return p.emitError(p.getNameLoc(),
                       "expected a !lit.ref<> with named lifetime");

  lifetimeDecl = ParamDeclAttr::get(lifetime.getName(), lifetime.getType());
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
                         StringRef name, StringRef lifetimeName, bool isVar,
                         bool isSynth) {
  auto lifetimeType = b.getType<LifetimeType>();
  auto lifetimeNameAttr = b.getAttr<StringAttr>(lifetimeName);
  auto lifetimeDecl = ParamDeclAttr::get(lifetimeNameAttr, lifetimeType);
  auto resultType = RefType::get(
      /*mutable=*/true, elementType,
      ParamDeclRefAttr::get(lifetimeNameAttr, lifetimeType));
  build(b, state, resultType, name, isVar, isSynth, lifetimeDecl,
        /*docString=*/{});
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

LogicalResult GlobalVarRefOp::verifySymbolUses(SymbolTableCollection &symtab) {
  auto global = symtab.lookupSymbolIn<GlobalVarDeclOp>(
      (*this)->getParentOfType<ModuleOp>(), getGlobal());
  if (!global || global.getType() != getResult().getType().getElementAsType())
    return emitOpError() << "does not refer to a global variable declaration "
                            "of the right type";
  return success();
}

//===----------------------------------------------------------------------===//
// AsyncCallOp
//===----------------------------------------------------------------------===//

static POP::CoroutineType getCoroutineOfResultTypes(Type type) {
  return POP::CoroutineType::get(cast<SignatureType>(type));
}

LogicalResult AsyncCallOp::verify() {
  if (cast<SignatureType>(getCallee().getType()).isAsync())
    return success();
  return emitOpError("callable must be 'async'");
}

void AsyncCallOp::concretizeCallee(mlir::IRRewriter &b,
                                   SymbolConstantAttr callee) {
  setCalleeAttr(callee);
  setParamDecls({});
}

//===----------------------------------------------------------------------===//
// AsyncExecuteOp
//===----------------------------------------------------------------------===//

/// The results of a `lit.async.execute` when treated like a function, although
/// an async one, are the results of the coroutine.
ArrayRef<Type> AsyncExecuteOp::getResultTypes() {
  return getType().getResultTypes();
}

//===----------------------------------------------------------------------===//
// ParamReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ParamReturnOp::verify() {
  auto func = (*this)->getParentOfType<LIT::FuncOp>();
  if (!func)
    return emitOpError("expected to be nested inside a `lit.func` operation");
  return checkResultParameterTypes(*this, getParameters(), func);
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
  if (op->getParentOfType<HLCF::LoopOp>())
    return success();
  return op->emitOpError("must be nested within an `hlcf.loop` operation");
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
  // TODO: Check for POP::VariantAttr presence to prune targets.
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
