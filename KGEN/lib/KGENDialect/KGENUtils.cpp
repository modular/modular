//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements utility functions primarily for parsing, printing and
// verifying KGEN related operations and types.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/KGENDType.h"
#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENTypeInterfaces.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "Support/Compiler/ParserUtils.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/ML/DType.h"
#include "Support/Preprocessor.h"
#include "Support/STLExtras.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Parameter Type and Value Printing and Parsing
//===----------------------------------------------------------------------===//

std::string KGEN::getParamAsString(Attribute value, bool forDiag) {
  SmallVector<char, 128> result;
  {
    llvm::raw_svector_ostream os(result);
    if (auto ta = dyn_cast<TypedAttr>(value)) {
      StreamAsmPrinter p(os);
      printParamValue(p, ta, {}, forDiag);
    } else {
      os << value;
    }
  }
  return std::string(result.data(), result.size());
}

StringAttr KGEN::getParamTypeAsString(TypedAttr value) {
  std::string str;
  llvm::raw_string_ostream os(str);
  StreamAsmPrinter p(os);
  printColonTypeParamValue(p, value);
  return StringAttr::get(value.getContext(), str);
}

StringAttr KGEN::getTypeAsString(Type type) {
  std::string str;
  llvm::raw_string_ostream os(str);
  StreamAsmPrinter p(os);
  printKGENType(p, type);
  return StringAttr::get(type.getContext(), str);
}

/// Parse a parameter of type kgen.string.
ParseResult KGEN::parseStringParam(AsmParser &p, TypedAttr &value) {
  return parseParamValue(p, value, KGEN::StringType::get(p.getContext()));
}

/// Print a parameter of type kgen.string.
void KGEN::printStringParam(AsmPrinter &p, Operation *op, Attribute value) {
  return printParamValue(p, cast<TypedAttr>(value));
}

/// Parse a non-empty parameter list without the surrounding braces.
static ParseResult parseParameterSpec(AsmParser &parser,
                                      ParamDeclArrayAttr &inputParamDecls,
                                      ParamDeclArrayAttr &resultParamDecls,
                                      ParamDeclParseHookTy parseInputElt) {
  // Parse the input list.
  if (parseParamDecls(parser, inputParamDecls, parseInputElt))
    return failure();

  // Check to see if we have results and parse them if so.
  if (succeeded(parser.parseOptionalArrow())) {
    if (parseParamDecls(parser, resultParamDecls))
      return failure();
  } else {
    resultParamDecls = ParamDeclArrayAttr::get(parser.getContext(), {});
  }
  return success();
}

/// Parse an argument or type list with optional metadata. This is an optional
/// parse, which allows the KGEN type parser to check if it is parsing a
/// signature. The provided parseArg hook is responsible for parsing an
/// individual argument and adding its type to the provided array.
static OptionalParseResult parseOptionalSignatureValues(
    AsmParser &p, function_ref<ParseResult(SmallVectorImpl<Type> &)> parseArg,
    FunctionType &values, FnEffects &effects, bool optionalResultList) {
  SmallVector<Type> argTypes, resTypes;

  if (failed(p.parseOptionalLParen()))
    return std::nullopt;
  if (failed(p.parseOptionalRParen())) {
    if (p.parseCommaSeparatedList([&]() { return parseArg(argTypes); }) ||
        p.parseRParen())
      return failure();
  }

  // Parse the function effects. Check for each case to disambiguate the syntax
  // for interfaces.
  auto effectsValue = impl::FnEffects::None;
  StringRef kw;
  while (succeeded(p.parseOptionalKeyword(
      &kw, {"throws", "async", "capturing", "escaping", "refresult"}))) {
    effectsValue |= *impl::symbolizeFnEffects(kw);

    // No vertical bar? We're done. It's not a parse error, but it does mean we
    // can't specify more effects.
    if (failed(p.parseOptionalVerticalBar()))
      break;
  }

  if (optionalResultList ? p.parseOptionalArrowTypeList(resTypes)
                         : p.parseArrowTypeList(resTypes))
    return failure();

  effects = FnEffects(effectsValue);
  values = p.getBuilder().getFunctionType(argTypes, resTypes);
  return mlir::success();
}

/// Parse and print an operand and result type list with metadata for a plain
/// (i.e. non-lit) signature.
static OptionalParseResult parseOptionalKGENSignature(AsmParser &p,
                                                      Type &signature) {
  llvm::SMLoc loc = p.getCurrentLocation();
  SmallVector<Type> inputParamTypes, resultParamTypes;
  if (failed(parseOptionalParamSignature(p, inputParamTypes, resultParamTypes)))
    return failure();

  SmallVector<ArgConvention> argConventions;
  auto parseArg = [&](SmallVectorImpl<Type> &argTypes) -> ParseResult {
    // Parse the argument type and its input convention.
    if (p.parseType(argTypes.emplace_back()) ||
        parseArgConvention(p, argConventions.emplace_back()))
      return failure();
    return success();
  };

  FunctionType functionType;
  FnEffects effects;
  OptionalParseResult result = parseOptionalSignatureValues(
      p, parseArg, functionType, effects, /*optionalResultList=*/false);
  if (result.has_value() && succeeded(*result)) {
    signature = SignatureType::getChecked(
        [&] { return p.emitError(loc); }, functionType, inputParamTypes,
        resultParamTypes, argConventions, effects, {});
    if (!signature)
      return failure();
  }
  return result;
}

/// Parse a type in a KGEN context, handling sugar like "dtype" for
/// "!kgen.dtype" etc.
OptionalParseResult KGEN::parseOptionalKGENType(AsmParser &p, Type &type) {
  // Check for sugared types before parsing standard ones. We need to check for
  // each keyword individually, since builtin types are also keywords.
  auto *dialect = p.getContext()->getLoadedDialect<KGENDialect>();
  assert(dialect && "cannot parse KGEN type without KGEN dialect");
  for (auto &[keyword, parseFn] : dialect->typeParseFns) {
    if (p.parseOptionalKeyword(keyword))
      continue;
    type = parseFn(p);
    return failure(!type);
  }

  // Parse symbol references as decl reference types.
  if (dialect->symbolTypeParser) {
    SymbolRefAttr symbol;
    OptionalParseResult result = p.parseOptionalAttribute(symbol);
    if (result.has_value()) {
      if (failed(*result))
        return failure();
      FailureOr<Type> symbolResult = dialect->symbolTypeParser(p, symbol);
      if (failed(symbolResult))
        return failure();
      type = *symbolResult;
      return LogicalResult::success();
    }
  }

  // Try to parse an optional signature. Signatures can begin with `<` or `(`.
  {
    SignatureType signature;
    OptionalParseResult result = parseOptionalKGENSignature(p, signature);
    if (result.has_value()) {
      if (failed(*result))
        return failure();
      type = signature;
      return LogicalResult::success();
    }
  }

  return p.parseOptionalType(type);
}

ParseResult KGEN::parseKGENType(AsmParser &p, Type &type) {
  OptionalParseResult result = parseOptionalKGENType(p, type);
  if (result.has_value())
    return result.value();
  return p.emitError(p.getCurrentLocation(), "expected a KGEN type");
}

void KGEN::printKGENType(raw_ostream &os, Type type) {
  StreamAsmPrinter p(os);
  printKGENType(p, type);
}

void KGEN::printKGENType(AsmPrinter &p, Type type) {
  // Always print an alias if available.
  if (succeeded(p.printAlias(type)))
    return;

  // Handle other special cases for parameters here.  These each are sugar for a
  // kgen type.
  auto *dialect = type.getContext()->getLoadedDialect<KGENDialect>();
  assert(dialect && "cannot print KGEN type without KGEN dialect");
  if (auto it = dialect->typePrintFns.find(type.getTypeID());
      it != dialect->typePrintFns.end()) {
    it->second(p, type);
  } else if (auto ref = dyn_cast<StructTypeInterface>(type)) {
    ref.printSymbol(p);
  } else if (auto signature = dyn_cast<SignatureType>(type)) {
    // Otherwise print it as "p1, p2 -> r3, () -> ())"
    printSignature(p, signature);
  } else {
    p << type;
  }
}

static OptionalParseResult parseOptionalColonType(AsmParser &parser,
                                                  Type &type) {
  if (failed(parser.parseOptionalColon()))
    return std::nullopt;
  return OptionalParseResult(parseKGENType(parser, type));
}

/// Parse a "colon type" production if present or default to index if not.  This
/// is commonly used in our parameter representation.
ParseResult KGEN::parseColonTypeOrIndex(AsmParser &parser, Type &type) {
  auto result = parseOptionalColonType(parser, type);
  if (!result.has_value()) {
    type = parser.getBuilder().getIndexType();
    return success();
  }
  return result.value();
}

/// print `: <type>` or elide it entirely if type is an `index` type.
void KGEN::printColonTypeOrIndex(AsmPrinter &p, Type type) {
  // Index type is the default so it doesn't print.
  if (type.isIndex())
    return;
  p << ": ";
  printKGENType(p, type);
}

/// print `:<type> ` or elide it entirely if type is an `index` type.
static void printColonTypeOrIndexPrefix(AsmPrinter &p, Type type) {
  // Index type is the default so it doesn't print.
  if (type.isIndex())
    return;
  p << ':';
  printKGENType(p, type);
  p << ' ';
}

/// Parse ":type 42" or "42" and default to index type.
ParseResult KGEN::parseParamValueDefaultingToIndex(AsmParser &p,
                                                   TypedAttr &value) {
  Type type = p.getBuilder().getIndexType();
  mlir::OptionalParseResult typePresent = parseOptionalColonType(p, type);
  if (typePresent.has_value() && failed(typePresent.value()))
    return failure();
  return parseParamValue(p, value, type);
}

/// Print a parameter value that is known to have `dtype` type.
void KGEN::printDTypeParamValue(AsmPrinter &p, Attribute value) {
  printParamValue(p, cast<TypedAttr>(value));
}

/// Parse a parameter value that is known to have `dtype` type.
ParseResult KGEN::parseDTypeParamValue(AsmParser &p, TypedAttr &value) {
  return parseParamValue(p, value, DTypeType::get(p.getContext()));
}

void KGEN::printTypeParamValue(AsmPrinter &p, TypedAttr value) {
  if (!isa<TypeType>(value.getType()))
    printColonTypeOrIndexPrefix(p, value.getType());
  printParamValue(p, value);
}

ParseResult KGEN::parseTypeParamValue(AsmParser &p, TypedAttr &value) {
  Type type;
  if (succeeded(p.parseOptionalColon())) {
    if (parseKGENType(p, type))
      return failure();
  } else {
    type = TypeType::get(p.getContext());
  }
  return parseParamValue(p, value, type);
}

ParseResult KGEN::parseParamType(AsmParser &p, Type &type) {
  TypedAttr typeParam;
  if (parseTypeParamValue(p, typeParam))
    return failure();
  type = ParamRefType::get(typeParam);
  return success();
}

void KGEN::printParamType(AsmPrinter &p, Type type) {
  printTypeParamValue(
      p, TypeConstantAttr::get(type, TypeType::get(type.getContext())));
}

ParseResult KGEN::parseParamTypes(AsmParser &p, SmallVectorImpl<Type> &types) {
  return p.parseCommaSeparatedList(
      [&] { return parseParamType(p, types.emplace_back()); });
}

void KGEN::printParamTypes(AsmPrinter &p, ArrayRef<Type> types) {
  llvm::interleaveComma(types, p, [&](Type type) { printParamType(p, type); });
}

void KGEN::printTypeParamValues(AsmPrinter &p, ArrayRef<TypedAttr> values) {
  llvm::interleaveComma(
      values, p, [&](TypedAttr value) { printTypeParamValue(p, value); });
}

ParseResult KGEN::parseTypeParamValues(AsmParser &p,
                                       SmallVector<TypedAttr> &values) {
  return p.parseCommaSeparatedList(
      [&] { return parseTypeParamValue(p, values.emplace_back()); });
}

void KGEN::printTypeValueBody(
    AsmPrinter &p, TypeConstantAttr type,
    llvm::function_ref<void(AsmPrinter &, Type)> typePrinter) {
  typePrinter(p, type.getTypeValue());
  if (type.getMlirType() != type.getTypeValue()) {
    p << ", ";
    typePrinter(p, type.getMlirType());
  }
  VTableAttr vtable = type.getVTable();
  if (!vtable.getEntries().empty()) {
    p << ", {";
    p.printStrippedAttrOrType(vtable);
    p << "}";
  }
}

OptionalParseResult KGEN::parseTypeValueBody(
    AsmParser &p, TypedAttr &value, Type type,
    llvm::function_ref<OptionalParseResult(AsmParser &, Type &)> typeParser,
    bool knownIdenticalRepresentation) {
  Type typeValue, mlirType;
  auto vtable = VTableAttr::get(p.getContext(), {});

  OptionalParseResult result = typeParser(p, typeValue);
  if (!result.has_value())
    return {}; // Not a type-value at all.

  if (failed(*result))
    return failure();

  if (knownIdenticalRepresentation || failed(p.parseOptionalComma())) {
    // This type-value has identical type/value representation. Stop here.
    value = TypeConstantAttr::get(typeValue, typeValue, type, vtable);
    return mlir::success();
  }

  // Parse the mlirType if a vtable is not seen immediately.
  bool seenVTable = succeeded(p.parseOptionalLBrace());
  if (seenVTable) {
    // mlirType is identical to typeValue.
    mlirType = typeValue;
  } else {
    OptionalParseResult result = typeParser(p, mlirType);
    if (!result.has_value())
      return p.emitError(p.getCurrentLocation(), "expected a type");
    if (failed(*result))
      return failure();

    if (failed(p.parseOptionalComma())) {
      // No vtable.
      value = TypeConstantAttr::get(typeValue, mlirType, type, vtable);
      return mlir::success();
    }

    seenVTable = succeeded(p.parseOptionalLBrace());
  }

  // Parse the vtable if a '{' was seen.
  if (seenVTable) {
    if (p.parseOptionalRBrace() &&
        (!(vtable = cast_or_null<VTableAttr>(VTableAttr::parse(p, {}))) ||
         p.parseRBrace()))
      return failure();
  }

  value = TypeConstantAttr::get(typeValue, mlirType, type, vtable);
  return mlir::success();
}

LogicalResult KGEN::printSugaredTypeValue(
    AsmPrinter &p, TypedAttr value,
    llvm::function_ref<void(AsmPrinter &, Type)> typePrinter) {
  auto type = dyn_cast<TypeConstantAttr>(value);
  if (!type)
    return failure();

  if (succeeded(p.printAlias(type)))
    return success();

  const bool nonTrivial = !type.hasIdenticalRepresentation();
  if (nonTrivial)
    p << '[';

  printTypeValueBody(p, type, typePrinter);

  if (nonTrivial)
    p << "]";
  return success();
}

OptionalParseResult KGEN::parseSugaredTypeValue(
    AsmParser &p, TypedAttr &value, Type type,
    llvm::function_ref<OptionalParseResult(AsmParser &, Type &)> typeParser) {
  bool nonTrivial = succeeded(p.parseOptionalLSquare());

  OptionalParseResult bodyParseResult = parseTypeValueBody(
      p, value, type, typeParser, /*knownIdenticalRepresentation=*/!nonTrivial);
  if (!bodyParseResult.has_value()) {
    // If a '[' was seen, require a type to be present.
    if (nonTrivial)
      return p.emitError(p.getCurrentLocation(), "expected a type");
    return {};
  }
  if (failed(*bodyParseResult))
    return failure();

  if (nonTrivial && failed(p.parseRSquare()))
    return failure();
  return mlir::success();
}

/// Print/Parse an attribute value that is known to have index type.
void KGEN::printIndexParamValue(AsmPrinter &p, Operation *op, Attribute value) {
  printParamValue(p, cast<TypedAttr>(value));
}
void KGEN::printIndexParamValue(AsmPrinter &p, Attribute value) {
  printParamValue(p, cast<TypedAttr>(value));
}
ParseResult KGEN::parseIndexParamValue(AsmParser &p, TypedAttr &value) {
  return parseParamValue(p, value, p.getBuilder().getIndexType());
}

/// Print/Parse an attribute value that is known to have i1 type.
void KGEN::printI1ParamValue(AsmPrinter &p, Operation *op, Attribute value) {
  printParamValue(p, cast<TypedAttr>(value));
}
void KGEN::printI1ParamValue(AsmPrinter &p, Attribute value) {
  printParamValue(p, cast<TypedAttr>(value));
}
ParseResult KGEN::parseI1ParamValue(AsmParser &p, TypedAttr &value) {
  return parseParamValue(p, value, p.getBuilder().getI1Type());
}

ParseResult KGEN::parseColonTypeParamValue(AsmParser &p, TypedAttr &value) {
  Type type;
  if (parseColonTypeOrIndex(p, type) || parseParamValue(p, value, type))
    return failure();

  return success();
}

void KGEN::printColonTypeParamValue(AsmPrinter &p, TypedAttr value) {
  printColonTypeOrIndexPrefix(p, value.getType());
  printParamValue(p, value);
}

ParseResult KGEN::parseParamDecl(AsmParser &p, ParamDeclAttr &result) {
  StringAttr name;
  Type type;
  if (parseParamName(p, name) || parseColonTypeOrIndex(p, type))
    return failure();
  result = ParamDeclAttr::get(name, type);
  return success();
}

void KGEN::printParamDecl(AsmPrinter &p, ParamDeclAttr decl) {
  printParamName(p, decl.getName());
  printColonTypeOrIndex(p, decl.getType());
}

/// Parse a parameter declaration list if present.
///
///   parameter-decl   ::= identifier (`:` type)?
///   parameter-decl-list  ::= parameter-decl (`,` parameter-decl)* | `(` `)`
ParseResult KGEN::parseParamDecls(AsmParser &p, ParamDeclArrayAttr &result,
                                  ParamDeclParseHookTy parseElt) {
  auto defaultParseElt = [&](SmallVectorImpl<ParamDeclAttr> &decls) {
    return parseParamDecl(p, decls.emplace_back(ParamDeclAttr()));
  };
  if (!parseElt)
    parseElt = std::move(defaultParseElt);

  // Parse each of the decls.
  SmallVector<ParamDeclAttr> decls;

  // Check to see if we have the () syntax instead of arguments.
  if (succeeded(p.parseOptionalLParen())) {
    if (p.parseRParen())
      return failure();
  } else {
    if (p.parseCommaSeparatedList([&]() { return parseElt(decls); }))
      return failure();
  }

  result = ParamDeclArrayAttr::get(p.getContext(), decls);
  return success();
}

void KGEN::printParamDecls(AsmPrinter &p, ArrayRef<ParamDeclAttr> decls,
                           ParamDeclPrintHookTy printElt) {
  auto defaultPrintElt = [&](ParamDeclAttr decl) { printParamDecl(p, decl); };
  if (!printElt)
    printElt = defaultPrintElt;

  if (decls.empty())
    p << "()";
  else
    llvm::interleaveComma(decls, p, printElt);
}

/// Parse a parameter spec if present, including input and result parameter
/// declarations.
/// parameter-decl   ::= identifier (`:` type)?
/// parameter-list   ::= parameter-decl (`,` parameter-decl)* | `(` `)`
/// parameter-spec   ::= `<` parameter-list (`->` parameter-list)? `>`
ParseResult KGEN::parseOptionalParameterSpec(
    AsmParser &parser, ParamDeclArrayAttr &inputParamDecls,
    ParamDeclArrayAttr &resultParamDecls, ParamDeclParseHookTy parseInputElt) {
  // If there is no parameter list, or if it is empty, we're done.
  if (failed(parser.parseOptionalLess()) ||
      succeeded(parser.parseOptionalGreater())) {
    inputParamDecls = ParamDeclArrayAttr::get(parser.getContext(), {});
    resultParamDecls = ParamDeclArrayAttr::get(parser.getContext(), {});
  } else {
    if (parseParameterSpec(parser, inputParamDecls, resultParamDecls,
                           parseInputElt) ||
        parser.parseGreater())
      return failure();
  }
  return success();
}

void KGEN::printOptionalParameterSpec(AsmPrinter &p,
                                      ArrayRef<ParamDeclAttr> inputParamDecls,
                                      ArrayRef<ParamDeclAttr> resultParams,
                                      ParamDeclPrintHookTy printInputElt) {
  if (inputParamDecls.empty() && resultParams.empty())
    return;

  p << '<';
  printParamDecls(p, inputParamDecls, printInputElt);

  if (!resultParams.empty()) {
    p << " -> ";
    printParamDecls(p, resultParams);
  }
  p << '>';
}

//===----------------------------------------------------------------------===//
// "Pretty" parameter printing and parsing
//===----------------------------------------------------------------------===//

// Parameters are complex nested expressions.  While they have a generic
// printing syntax that is supported in full generality, they often appear in
// tightly controlled situations, e.g. in return operations, in types, or when
// invoking a generator. In these cases we can use a much nicer and more compact
// syntax so we as compiler engineers don't go bonkers looking at IR dumps.

enum class POCAliases : uint32_t {
  // The builtin opcodes have 0...127.
  FIRST_PSEUDO = 128,
  NEG, // negation
  SUB, // subtraction
  NOT,
  NE, // !(==)
  GT, // !(<)
  GE, // !(<=)
  NOT_IN,
  // This is an unknown opcode name.
  kInvalid,
};

/// Returns true if the given string can be represented as a bare identifier.
static bool isLegalMLIRIdentifier(StringRef name) {
  // By making this unsigned, the value passed in to isalnum will always be
  // in the range 0-255. This is important when building with MSVC because
  // its implementation will assert. This situation can arise when dealing
  // with UTF-8 multibyte characters.
  if (name.empty() || (!isalpha(name[0]) && name[0] != '_'))
    return false;
  return llvm::all_of(name.drop_front(), [](unsigned char c) {
    return isalnum(c) || c == '_' || c == '$' || c == '.';
  });
}

/// Returns true if the given string could be an MLIR builtin type.
/// TODO: Can't interact directly with the MLIR AsmParser.
static bool isMLIRBuiltinType(StringRef name) {
  // Check for a keyword type.
  static const char *keywordTypes[] = {"bf16", "f16",  "f32",   "f64",
                                       "f80",  "f128", "index", "none"};
  if (auto it = llvm::find(keywordTypes, name); it != std::end(keywordTypes))
    return true;
  // Check for an integral type: (s|u)*i[0-9]+
  if (name.front() == 's' || name.front() == 'u')
    name = name.drop_front();
  if (name.size() <= 1 || name.front() != 'i')
    return false;
  return llvm::all_of(name.drop_front(), isdigit);
}

ParseResult KGEN::parseParamName(AsmParser &p, StringAttr &name) {
  // If this is a '*'-prefixed double quoted string, then this is an escaped
  // parameter name.
  if (succeeded(p.parseOptionalStar())) {
    std::string value;
    if (failed(p.parseString(&value)))
      return failure();
    name = StringAttr::get(p.getContext(), value);
  } else {
    // Barewords / MLIR keywords are param names otherwise.
    StringRef keyword;
    if (failed(p.parseKeyword(&keyword)))
      return failure();
    name = StringAttr::get(p.getContext(), keyword);
  }
  return success();
}

/// Print a parameter name correctly, using a double quoted syntax if it
/// conflicts with an MLIR or KGEN keyword, or a bareword otherwise.
void KGEN::printParamName(AsmPrinter &p, StringAttr name, bool isRef) {
  // If this will conflict with a reserved keyword then we need a '*' prefix and
  // double quotes.
  auto isSugaredType = [&] {
    return name.getContext()
        ->getLoadedDialect<KGENDialect>()
        ->typeParseFns.contains(name);
  };
  bool needsQuotes = !isLegalMLIRIdentifier(name) ||
                     (isRef && (succeeded(DType::getFromString(name)) ||
                                isMLIRBuiltinType(name) || isSugaredType()));
  if (needsQuotes)
    p << "*\"";
  llvm::printEscapedString(name, p.getStream());
  if (needsQuotes)
    p << '"';
}

/// Parse operator expression operands with operator-specific syntax.
static ParseResult parseOperatorOperands(AsmParser &p, uint32_t opcode,
                                         SmallVectorImpl<TypedAttr> &operands,
                                         Type type) {
  switch (opcode) {
  default:
    // operand-list ::= expr (`,` expr)*
    return p.parseCommaSeparatedList(
        [&] { return parseParamValue(p, operands.emplace_back(), type); });
  case (uint32_t)POC::In:
  case (uint32_t)POCAliases::NOT_IN:
    // operand-list ::= expr `,` `[` (expr (`,` expr)*)? `]`
    if (parseParamValue(p, operands.emplace_back(), type) || p.parseComma() ||
        p.parseCommaSeparatedList(AsmParser::Delimiter::OptionalSquare, [&] {
          return parseParamValue(p, operands.emplace_back(), type);
        }))
      return failure();
    return success();
    if (parseParamValue(p, operands.emplace_back(), type) || p.parseComma() ||
        parseIndexParamValue(p, operands.emplace_back()))
      return failure();
    return success();
  case (uint32_t)POC::TargetHasFeature:
  case (uint32_t)POC::TargetGetField:
    // Parse TargetHasFeature, and TargetGetField -- the first operand is a
    // TargetType, the second a StringType.
    if (parseParamValue(p, operands.emplace_back(),
                        TargetType::get(p.getContext())) ||
        p.parseComma() ||
        parseParamValue(p, operands.emplace_back(),
                        StringType::get(p.getContext())))
      return failure();
    return success();
  case (uint32_t)POC::GetSizeOf:
  case (uint32_t)POC::GetAlignOf:
    if (parseParamValue(p, operands.emplace_back(),
                        TypeType::get(p.getContext())) ||
        p.parseComma() ||
        parseParamValue(p, operands.emplace_back(),
                        TargetType::get(p.getContext())))
      return failure();
    return success();
  case (uint32_t)POC::BindSignature: {
    auto sig = dyn_cast_or_null<SignatureType>(type);
    if (!sig)
      return p.emitError(p.getCurrentLocation(),
                         "expected a signature type for 'bind_signature'");
    if (parseParamValue(p, operands.emplace_back(), sig))
      return failure();
    // Parse each operand, inferring its type from the signature type. Bound
    // parameters are allowed to refine the types of subsequent parameters, so
    // specialize the types as we go.
    ParameterEvaluator evaluator;
    for (Type type : sig.getInputParamTypes()) {
      if (p.parseComma() || parseParamValue(p, operands.emplace_back(),
                                            evaluator.getReboundType(type)))
        return failure();
      evaluator.addInputValue(operands.back());
    }
    return success();
  }
  case (uint32_t)POC::Apply: {
    auto sig = dyn_cast_or_null<SignatureType>(type);
    if (!sig)
      return p.emitError(p.getCurrentLocation(),
                         "expected a signature type for 'apply'");
    if (parseParamValue(p, operands.emplace_back(), sig))
      return failure();
    // Parse each operand, inferring its type from the signature type.
    IndexDepthAdjuster adjuster(/*adjustDepth=*/-1);
    for (Type type : sig.getArguments())
      if (p.parseComma() ||
          parseParamValue(p, operands.emplace_back(), adjuster.replace(type)))
        return failure();
    return success();
  }
  case (uint32_t)POC::ApplyResultSlot: {
    auto sig = dyn_cast_or_null<SignatureType>(type);
    if (!sig)
      return p.emitError(p.getCurrentLocation(),
                         "expected a signature type for 'apply_result_slot'");
    if (parseParamValue(p, operands.emplace_back(), sig))
      return failure();
    if (sig.getNumArguments() < 1)
      return p.emitError(
          p.getCurrentLocation(),
          "'apply_result_slot' callee must have at least one result");
    // Parse each operand besides the result slot.
    auto argTypes = sig.getArguments()
                        .drop_front(sig.hasInitSelfArg())
                        .drop_back(sig.hasMemoryOnlyResult());
    for (Type type : argTypes)
      if (p.parseComma() || parseParamValue(p, operands.emplace_back(), type))
        return failure();
    return success();
  }
  case (uint32_t)POC::VariadicGet: {
    if (parseParamValue(p, operands.emplace_back(), type) || p.parseComma() ||
        parseIndexParamValue(p, operands.emplace_back()))
      return failure();
    return success();
  }
  case (uint32_t)POC::Cond:
    if (parseParamValue(p, operands.emplace_back(),
                        IntegerType::get(p.getContext(), 1)) ||
        p.parseComma() || parseParamValue(p, operands.emplace_back(), type) ||
        p.parseComma() || parseParamValue(p, operands.emplace_back(), type))
      return failure();
    return success();
  case (uint32_t)POC::GetEnv:
    return parseParamValue(p, operands.emplace_back(),
                           StringType::get(p.getContext()));
  case (uint32_t)POC::CompileAssembly: {
    StringRef emissionKind;
    if (parseParamValue(p, operands.emplace_back(),
                        TargetType::get(p.getContext())) ||
        p.parseComma() || p.parseKeyword(&emissionKind))
      return failure();

    if (!llvm::is_contained({"llvm", "asm"}, emissionKind))
      return p.emitError(p.getCurrentLocation(),
                         "the emission kind must be either llvm or asm");

    EmissionKind emissionKindEnum =
        emissionKind == "llvm" ? EmissionKind::LLVM : EmissionKind::ASM;
    operands.emplace_back(
        p.getBuilder().getIndexAttr(to_underlying(emissionKindEnum)));

    if (p.parseComma() ||
        parseParamValue(p, operands.emplace_back(),
                        p.getBuilder().getI1Type()) ||
        p.parseComma() || parseColonTypeParamValue(p, operands.emplace_back()))
      return failure();

    return success();
  }
  case (uint32_t)POC::GetLinkageName:
    if (parseParamValue(p, operands.emplace_back(),
                        TargetType::get(p.getContext())) ||
        p.parseComma() || parseColonTypeParamValue(p, operands.emplace_back()))
      return failure();
    return success();
  case (uint32_t)POC::GetTypeMethod:
    if (!type)
      type = TypeType::get(p.getContext());
    if (parseParamValue(p, operands.emplace_back(), type) || p.parseComma() ||
        parseParamValue(p, operands.emplace_back(),
                        StringType::get(p.getContext())))
      return failure();
    return success();
  case (uint32_t)POC::VariadicPtrMap:
    if (parseParamValue(p, operands.emplace_back(), type) || p.parseComma() ||
        parseParamValue(p, operands.emplace_back(),
                        IndexType::get(p.getContext())))
      return failure();
    return success();
  }
  llvm_unreachable("unknown operator");
}

static uint32_t getOpcodeFromString(StringRef keyword) {
  // All the valid and builtin opcodes are legal.
  auto opcode = symbolizePOC(keyword);
  if (opcode.has_value())
    return (uint32_t)*opcode;

  if (keyword == "neg")
    return (uint32_t)POCAliases::NEG;
  if (keyword == "sub")
    return (uint32_t)POCAliases::SUB;
  if (keyword == "not")
    return (uint32_t)POCAliases::NOT;
  if (keyword == "ne")
    return (uint32_t)POCAliases::NE;
  if (keyword == "gt")
    return (uint32_t)POCAliases::GT;
  if (keyword == "ge")
    return (uint32_t)POCAliases::GE;
  if (keyword == "not_in")
    return (uint32_t)POCAliases::NOT_IN;

  return (uint32_t)POCAliases::kInvalid;
}

/// When in a context that knows it is dealing with a parameter specifically,
/// utilize syntactic shortcuts to make the parsed syntax easier to grok.
ParseResult KGEN::parseParamValue(AsmParser &p, TypedAttr &value, Type type) {
  assert(type && "always have a contextual type");
  llvm::SMLoc loc = p.getCurrentLocation();

  // If the type provides a pretty parsing hook, use it.
  if (auto typeItf = dyn_cast<ParameterTypeInterface>(type)) {
    OptionalParseResult result = typeItf.parseValue(p, value);
    if (result.has_value())
      return *result;
  }

  // If this is a '*'-prefixed double quoted string, then this is a simple
  // parameter reference.
  if (succeeded(p.parseOptionalStar())) {
    if (succeeded(p.parseOptionalLParen())) {
      // Try to parse *("...") as a SourceStruct parameter name reference.
      std::string name;
      if (succeeded(p.parseOptionalString(&name))) {
        if (p.parseRParen())
          return failure();
        value = StructDefParamRefAttr::get(
            StringAttr::get(p.getContext(), name), type);
        return success();
      }

      // Try to parse *(0,0) as an index reference.
      size_t depth, index;
      if (p.parseInteger(depth) || p.parseComma() || p.parseInteger(index) ||
          p.parseRParen())
        return failure();
      bool isResult = succeeded(p.parseOptionalStar());
      value = ParamIndexRefAttr::get(depth, isResult, index, type);
      return success();
    }

    // Try to parse '*?' as an undef value.
    if (succeeded(p.parseOptionalQuestion())) {
      value = UnknownAttr::get(type);
      return success();
    }

    std::string name;
    if (failed(p.parseString(&name)))
      return failure();
    value = ParamDeclRefAttr::get(name, type);
    return success();
  }

  // A '?' represents an unknown parameter.
  if (succeeded(p.parseOptionalQuestion())) {
    value = UnboundAttr::get(type);
    return success();
  }

  // Barewords / MLIR keywords are implicitly parameter declaration references
  // or the start of a expression in function form.
  StringRef keyword;
  if (succeeded(p.parseOptionalKeyword(&keyword))) {
    // Check to see if we're parsing a dtype name like 'f32'.
    if (isa<DTypeType>(type)) {
      auto dtype = KGENDType::getFromString(keyword);
      if (succeeded(dtype)) {
        value = DTypeConstantAttr::get(p.getContext(), *dtype);
        return success();
      }
    }

    // A bareword or string with no trailing `(` must be a parameter reference.
    if (failed(p.parseOptionalLParen())) {
      value = ParamDeclRefAttr::get(keyword, type);
      return success();
    }

    // Otherwise it's a function expression.  If this has an explicit operand
    // type, parse it.
    Type operandType;
    OptionalParseResult typePresent = parseOptionalColonType(p, operandType);
    if (typePresent.has_value() && failed(typePresent.value()))
      return failure();

    // Decode the name as an operation code.
    auto opcode = getOpcodeFromString(keyword);
    if (opcode == (uint32_t)POCAliases::kInvalid)
      return p.emitError(loc, "unknown expression ") << keyword;
    // If it is a known opcode, parse the operand list.
    SmallVector<TypedAttr> operands;

    // If there was no specified element type, then pick a default based on the
    // opcode in question.
    if (!operandType) {
      switch (opcode) {
      case (uint32_t)POCAliases::NEG:
      case (uint32_t)POCAliases::SUB:
      case (uint32_t)POC::EQ:
      case (uint32_t)POC::LT:
      case (uint32_t)POC::LE:
      case (uint32_t)POCAliases::NE:
      case (uint32_t)POCAliases::GE:
      case (uint32_t)POCAliases::GT:
      case (uint32_t)POCAliases::NOT_IN:
      case (uint32_t)POC::In:
        // Comparisons default to index type for their operand, since their
        // result is always `i1`.
        operandType = p.getBuilder().getIndexType();
        break;
      case (uint32_t)POC::GetTypeMethod:
        operandType = TypeType::get(p.getContext());
        break;
      default:
        // Other operators default to the same operand type as the result type.
        operandType = type;
        break;
      }
    }

    // Parse the remaining operands.
    if (failed(p.parseOptionalRParen())) {
      if (parseOperatorOperands(p, opcode, operands, operandType) ||
          p.parseRParen())
        return failure();
    }

    // Desugar the negation operator from `neg(a)` to `mul(a, -1)`
    if (opcode == (uint32_t)POCAliases::NEG) {
      if (operands.size() != 1)
        return p.emitError(loc, "neg operator expects a single operand");
      operands.emplace_back(p.getBuilder().getIndexAttr(-1));
      opcode = (uint32_t)POC::Mul;
    }

    // Desugar the subtract operator from `sub(a, b)` to `add(a, mul(b, -1))`
    if (opcode == (uint32_t)POCAliases::SUB) {
      if (operands.size() != 2)
        return p.emitError(loc, "sub operator expects two operands");
      operands[1] = ParamOperatorAttr::get(
          POC::Mul, {operands[1], p.getBuilder().getIndexAttr(-1)});
      opcode = (uint32_t)POC::Add;
    }

    // If these are aliases for inverted i1 value, build the correct nodes.
    bool needsInvert = false;
    switch (opcode) {
    case (uint32_t)POCAliases::NE:
      opcode = (uint32_t)POC::EQ;
      needsInvert = true;
      break;
    case (uint32_t)POCAliases::GE:
      opcode = (uint32_t)POC::LT;
      needsInvert = true;
      break;
    case (uint32_t)POCAliases::GT:
      opcode = (uint32_t)POC::LE;
      needsInvert = true;
      break;
    case (uint32_t)POCAliases::NOT_IN:
      opcode = (uint32_t)POC::In;
      needsInvert = true;
      break;
    case (uint32_t)POCAliases::NOT:
      if (operands.size() != 1 || !operands[0].getType().isSignlessInteger(1))
        return p.emitError(loc, "not operator returns a single i1 operand");
      value = ParamOperatorAttr::getNot(operands[0]);
      return success();
    }

    // Okay, we parsed the operands, see if this is a valid expression.
    if (failed(ParamOperatorAttr::verify(
            [&]() -> mlir::InFlightDiagnostic { return p.emitError(loc); },
            (POC)opcode, operands, type)))
      return failure();
    // All is good, let's move!
    value =
        ParamOperatorAttr::get(type.getContext(), (POC)opcode, operands, type);

    // If we need to invert this, do so.
    if (needsInvert)
      value = ParamOperatorAttr::getNot(value);

    return success();
  }

  // Otherwise, we support other typed attributes as well, including dialect
  // define attributes, integers, strings, etc.
  return p.parseAttribute(value, type);
}

static void printOperatorOperands(AsmPrinter &p, POC opcode,
                                  ArrayRef<TypedAttr> operands) {
  // If the elements are not index type, print the type explicitly.
  if (llvm::is_contained({POC::In, POC::EQ, POC::LT, POC::LE, POC::Rebind},
                         opcode))
    printColonTypeOrIndexPrefix(p, operands[0].getType());

  switch (opcode) {
  default:
    // operand-list ::= expr (`,` expr)*
    llvm::interleaveComma(
        operands, p, [&](TypedAttr operand) { printParamValue(p, operand); });
    break;
  case POC::In:
    // operand-list ::= expr `,` `[` (expr (`,` expr)*)? `]`
    printParamValue(p, operands[0]);
    p << ", [";
    llvm::interleaveComma(operands.drop_front(), p, [&](TypedAttr operand) {
      printParamValue(p, operand);
    });
    p << "]";
    break;

  case POC::Apply:
  case POC::BindSignature:
    // Print the signature operand with a type. Print all other operands without
    // types.
    printColonTypeOrIndexPrefix(p, operands.front().getType());
    printParamValue(p, operands.front());
    for (TypedAttr operand : operands.drop_front()) {
      p << ", ";
      printParamValue(p, operand);
    }
    break;

  case POC::ApplyResultSlot:
    // Print the signature operand with a type. Print all other operands without
    // types.
    printColonTypeOrIndexPrefix(p, operands.front().getType());
    printParamValue(p, operands.front());
    for (TypedAttr operand : operands.drop_front()) {
      p << ", ";
      printParamValue(p, operand);
    }
    break;

  case POC::VariadicGet:
    p << ':';
    printKGENType(p, operands.front().getType());
    p << ' ';
    printParamValue(p, operands.front());
    p << ", ";
    printIndexParamValue(p, operands.back());
    break;

  case POC::Cond:
    printParamValue(p, operands[0]);
    p << ", ";
    printParamValue(p, operands[1]);
    p << ", ";
    printParamValue(p, operands[2]);
    break;

  case POC::CompileAssembly: {
    printParamValue(p, operands[0]);
    p << ", ";
    EmissionKind emissionKind =
        (EmissionKind)cast<IntegerAttr>(operands[1]).getInt();
    if (emissionKind == EmissionKind::ASM)
      p << "asm";
    else if (emissionKind == EmissionKind::LLVM)
      p << "llvm";
    p << ", ";
    printParamValue(p, operands[2]);
    p << ", ";
    printColonTypeParamValue(p, operands[3]);
    break;
  }
  case POC::GetLinkageName:
    printParamValue(p, operands[0]);
    p << ", ";
    printColonTypeParamValue(p, operands[1]);
    break;
  case POC::GetTypeMethod:
    if (!isa<TypeType>(operands[0].getType())) {
      p << ':';
      printKGENType(p, operands[0].getType());
      p << ' ';
    }
    printParamValue(p, operands[0]);
    p << ", ";
    printParamValue(p, operands[1]);
    break;
  case POC::PtrBitcast:
  case POC::LoadFromMem:
    printColonTypeParamValue(p, operands.front());
    break;
  case POC::VariadicPtrMap:
    // Type is of the list, but the index type doesn't need it.
    printColonTypeParamValue(p, operands[0]);
    p << ", ";
    printParamValue(p, operands[1]);
    break;
  case POC::VariadicPtrRemoveMap:
    // Include the type of the list.
    printColonTypeParamValue(p, operands[0]);
    break;
  }
}

void KGEN::printParamValue(AsmPrinter &p, TypedAttr value, Type type,
                           bool forDiag) {
  // If the attribute's type provides a pretty printing hook, try to use it.
  if (auto typeItf = dyn_cast<ParameterTypeInterface>(value.getType()))
    if (succeeded(typeItf.printValue(p, value)))
      return;

  if (isa<UnknownAttr>(value)) {
    p << "*?";
    return;
  }

  if (isa<UnboundAttr>(value)) {
    p << '?';
    return;
  }

  if (auto declRef = dyn_cast<ParamDeclRefAttr>(value)) {
    bool isRef = isTypeExpr(value);
    if (auto type = dyn_cast<ParameterTypeInterface>(value.getType()))
      isRef |= type.isMetaType();
    if (forDiag)
      llvm::printEscapedString(declRef.getName(), p.getStream());
    else
      printParamName(p, declRef.getName(), isRef);
    return;
  }
  if (auto indexRef = dyn_cast<ParamIndexRefAttr>(value)) {
    p << "*(" << indexRef.getDepth() << ',' << indexRef.getIndex() << ")";
    if (indexRef.getIsResult())
      p << '*';
    return;
  }
  if (auto sourceStructParamRef = dyn_cast<StructDefParamRefAttr>(value)) {
    p << "*(" << sourceStructParamRef.getName() << ")";
    return;
  }

  // If this is a dtype constant with simple syntax, we can print it as a
  // keyword.
  if (auto dtypeConstant = dyn_cast<DTypeConstantAttr>(value)) {
    auto eltType = dtypeConstant.getDType();
    std::string stringRep = eltType.getAsString();
    // Don't allow things like complex<f64>.  We can extend this in the future
    // if there is a reason to of course.
    if (!StringRef(stringRep).contains('<')) {
      p << stringRep;
      return;
    }
  }

  // Handle expressions.
  if (auto expr = dyn_cast<ParamOperatorAttr>(value)) {
    auto printExpr = [&](StringRef opcode, ArrayRef<TypedAttr> operands) {
      p << opcode << '(';
      printOperatorOperands(p, expr.getOpcode(), operands);
      p << ')';
    };

    // If this is a inverted boolean sugar, handle it.
    if (expr.getOpcode() == POC::Xor && expr.getType().isSignlessInteger(1) &&
        expr.getNumOperands() == 2 && isa<IntegerAttr>(expr.getOperand(1))) {
      if (auto invertedExpr = dyn_cast<ParamOperatorAttr>(expr.getOperand(0))) {
        if (invertedExpr.getOpcode() == POC::EQ) {
          expr = invertedExpr;
          return printExpr("ne", expr.getOperands());
        }
        if (invertedExpr.getOpcode() == POC::LT) {
          expr = invertedExpr;
          return printExpr("ge", expr.getOperands());
        }
        if (invertedExpr.getOpcode() == POC::LE) {
          expr = invertedExpr;
          return printExpr("gt", expr.getOperands());
        }
      }

      // Otherwise, print as a generic "not".
      return printExpr("not", expr.getOperand(0));
    }

    return printExpr(stringifyEnum(expr.getOpcode()), expr.getOperands());
  }

  // If this is an i1 integer attr, print it as zero or one; not true/false
  // keywords.  This simplifies the keyword processing logic.
  if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
    if (intAttr.getType().isSignlessInteger(1)) {
      p << (intAttr.getValue().isZero() ? 0 : 1);
      return;
    }
  }

  p.printAttributeWithoutType(value);
}

void KGEN::printParamValue(AsmPrinter &p, Operation *op, TypedAttr value,
                           Type type) {
  printParamValue(p, value, type);
}

ParseResult KGEN::parseParamDeclaration(OpAsmParser &p,
                                        ParamDeclAttr &paramDecl,
                                        TypedAttr &value) {
  StringAttr name;
  Type resultType;
  if (parseParamName(p, name) || parseColonTypeOrIndex(p, resultType) ||
      p.parseEqual() || p.parseLess() ||
      parseParamValue(p, value, resultType) || p.parseGreater())
    return failure();

  paramDecl = ParamDeclAttr::get(name, value.getType());
  return success();
}

void KGEN::printParamDeclaration(OpAsmPrinter &p, ParamDeclAttr paramDecl,
                                 TypedAttr value) {
  printParamName(p, paramDecl.getName());
  printColonTypeOrIndex(p, value.getType());
  p << " = <";
  printParamValue(p, value);
  p << ">";
}

bool KGEN::isTypeExprType(Type type) { return isa<TypeType>(type); }

bool KGEN::isTypeExpr(TypedAttr attr) { return isTypeExprType(attr.getType()); }

KGEN::EnvAttr KGEN::getModularEnvAttr(MLIRContext *ctx) {
  NamedAttrList envAttrs;

#ifdef MODULAR_PRODUCTION
  envAttrs.set("MODULAR_PRODUCTION", IntegerAttr::get(IndexType::get(ctx), 1));
#endif // MODULAR_PRODUCTION

#ifdef MODULAR_PARANOID
  envAttrs.set("MODULAR_PARANOID", IntegerAttr::get(IndexType::get(ctx), 1));
#endif // MODULAR_PARANOID

  envAttrs.set("BUILD_TYPE", StringAttr::get(STRINGIFY(BUILD_TYPE),
                                             KGEN::StringType::get(ctx)));
  envAttrs.set("KERNELS_BUILD_TYPE",
               StringAttr::get(STRINGIFY(KERNELS_BUILD_TYPE),
                               KGEN::StringType::get(ctx)));
  envAttrs.set("MODULAR_ASYNCRT_MAX_PROFILING_LEVEL",
               IntegerAttr::get(IndexType::get(ctx),
                                MODULAR_ASYNCRT_MAX_PROFILING_LEVEL));

  return KGEN::EnvAttr::get(envAttrs.getDictionary(ctx));
}

static KGEN::EnvAttr getModuleEnvAttr(ModuleOp moduleOp) {
  if (moduleOp->hasAttrOfType<KGEN::EnvAttr>(KGEN::EnvAttr::getEnvAttrName()))
    return moduleOp->getAttrOfType<KGEN::EnvAttr>(
        KGEN::EnvAttr::getEnvAttrName());

  return EnvAttr::get(DictionaryAttr::get(moduleOp.getContext()));
}

void KGEN::extendWithModularEnvAttr(ModuleOp moduleOp) {
  moduleOp->setAttr(KGEN::EnvAttr::getEnvAttrName(),
                    KGEN::getModularEnvAttr(moduleOp.getContext())
                        .extend(getModuleEnvAttr(moduleOp)));
}

void KGEN::printIsMemoryOnly(AsmPrinter &p, bool isMemoryOnly) {
  if (isMemoryOnly)
    p << " memoryOnly";
}

ParseResult KGEN::parseIsMemoryOnly(AsmParser &p, bool &isMemoryOnly) {
  if (succeeded(p.parseOptionalKeyword("memoryOnly")))
    isMemoryOnly = true;
  return success();
}

//===----------------------------------------------------------------------===//
// Logic shared between funcs, generators, and generator interfaces
//===----------------------------------------------------------------------===//

ParseResult KGEN::parseArgConvention(AsmParser &p, ArgConvention &convention) {
  StringRef effectStr;
  llvm::SMLoc loc = p.getCurrentLocation();
  // Parse an optional input convention specifier.
  convention = ArgConvention::BorrowedInReg;
  if (succeeded(p.parseOptionalKeyword(&effectStr))) {
    if (std::optional<ArgConvention> conv = symbolizeArgConvention(effectStr)) {
      convention = *conv;
    } else {
      return p.emitError(loc, "expected a valid input convention");
    }
  }
  return success();
}

void KGEN::printArgConvention(AsmPrinter &p, ArgConvention convention) {
  if (convention != ArgConvention::BorrowedInReg)
    p << ' ' << stringifyArgConvention(convention);
}

ParseResult KGEN::parseSignatureValues(
    AsmParser &p, function_ref<ParseResult(SmallVectorImpl<Type> &)> parseArg,
    FunctionType &values, FnEffects &effects, bool optionalResultList) {
  OptionalParseResult result = parseOptionalSignatureValues(
      p, parseArg, values, effects, optionalResultList);
  if (result.has_value())
    return *result;
  return p.emitError(p.getCurrentLocation(), "expected '(' to begin signature");
}

/// Print an argument or type list with optional metadata.
void KGEN::printSignatureValues(AsmPrinter &p,
                                function_ref<void(unsigned)> printElt,
                                FunctionType functionType,
                                SignatureType signature,
                                bool optionalResultList) {
  p << '(';
  llvm::interleaveComma(
      llvm::seq<unsigned>(0, signature.getArgConventions().size()), p,
      printElt);
  p << ')';

  // Print the function effects.
  impl::FnEffects effects = signature.getFnEffects().getImpl();
  if (effects != impl::FnEffects::None)
    p << ' ' << impl::stringifyFnEffects(effects);

  if (optionalResultList)
    p.printOptionalArrowTypeList(functionType.getResults());
  else
    p.printArrowTypeList(functionType.getResults());
}

ParseResult KGEN::parseFunctionSignature(
    OpAsmParser &p, SmallVectorImpl<OpAsmParser::Argument> &args,
    ParamDeclArrayAttr &inputParams, ParamDeclArrayAttr &resultParams,
    FunctionType &functionType, SignatureType &signature) {
  llvm::SMLoc loc = p.getCurrentLocation();
  if (parseOptionalParameterSpec(p, inputParams, resultParams))
    return failure();

  SmallVector<ArgConvention> argConventions;
  auto parseArg = [&](SmallVectorImpl<Type> &argTypes) -> ParseResult {
    // Parse the argument type and its input convention.
    OpAsmParser::Argument &arg = args.emplace_back();
    OptionalParseResult result =
        p.parseOptionalArgument(arg, /*allowType=*/true);
    if (result.has_value() && failed(*result))
      return failure();
    if (!result.has_value() && p.parseType(arg.type))
      return failure();

    if (parseArgConvention(p, argConventions.emplace_back()))
      return failure();

    argTypes.push_back(arg.type);
    return success();
  };

  FnEffects effects;
  if (failed(parseSignatureValues(p, parseArg, functionType, effects,
                                  /*optionalResultList=*/true)))
    return failure();

  signature = SignatureType::remapToSignature(
      inputParams, resultParams, functionType, argConventions, effects, {},
      [&] { return p.emitError(loc); });
  return success(!!signature);
}

void KGEN::printFunctionSignature(OpAsmPrinter &p, Region *region,
                                  ArrayRef<ParamDeclAttr> inputParams,
                                  ArrayRef<ParamDeclAttr> resultParams,
                                  FunctionType functionType,
                                  SignatureType signature) {
  // Print the function arguments.
  auto printElt = [&](unsigned i) {
    if (!region)
      p << functionType.getInput(i);
    else
      p.printRegionArgument(region->getArgument(i));

    printArgConvention(p, signature.getArgConvention(i));
  };

  printOptionalParameterSpec(p, inputParams, resultParams);
  printSignatureValues(p, printElt, functionType, signature,
                       /*optionalResultList=*/true);
}

ParseResult KGEN::parseOptionalParamSignature(
    AsmParser &p, SmallVectorImpl<Type> &inputParamTypes,
    SmallVectorImpl<Type> &resultParamTypes,
    function_ref<ParseResult(SmallVectorImpl<Type> &)> parseInputTy) {
  if (failed(p.parseOptionalLess()) || succeeded(p.parseOptionalGreater()))
    return success();

  auto defaultParseInputTy = [&](SmallVectorImpl<Type> &inputs) {
    return parseKGENType(p, inputs.emplace_back());
  };
  if (!parseInputTy)
    parseInputTy = defaultParseInputTy;

  // Parse the input parameter types.
  auto parseIn = [&]() { return parseInputTy(inputParamTypes); };
  if (succeeded(p.parseOptionalLSquare())) {
    if (p.parseRSquare())
      return failure();
  } else if (p.parseCommaSeparatedList(parseIn)) {
    return failure();
  }

  // Parse the result parameter types.
  auto parseRes = [&]() {
    return parseKGENType(p, resultParamTypes.emplace_back());
  };
  if (succeeded(p.parseOptionalArrow()) && p.parseCommaSeparatedList(parseRes))
    return failure();

  if (p.parseGreater())
    return failure();
  return success();
}

void KGEN::printOptionalParamSignature(AsmPrinter &p,
                                       ArrayRef<Type> inputParamTypes,
                                       ArrayRef<Type> resultParamTypes,
                                       function_ref<void(Type)> printInputTy) {
  if (inputParamTypes.empty() && resultParamTypes.empty())
    return;

  auto defaultPrintInputTy = [&](Type type) { printKGENType(p, type); };
  if (!printInputTy)
    printInputTy = defaultPrintInputTy;

  p << '<';
  if (inputParamTypes.empty())
    p << "[]";
  llvm::interleaveComma(inputParamTypes, p, printInputTy);
  if (!resultParamTypes.empty()) {
    p << " -> ";
    llvm::interleaveComma(resultParamTypes, p,
                          [&](Type type) { printKGENType(p, type); });
  }
  p << '>';
}

ParseResult KGEN::parseSignature(AsmParser &p, Type &signature) {
  OptionalParseResult result = parseOptionalKGENSignature(p, signature);
  if (result.has_value())
    return *result;
  result = p.parseOptionalType(signature);
  if (!result.has_value())
    return p.emitError(p.getCurrentLocation(),
                       "expected '<' or '(' to begin a signature");
  if (failed(*result))
    return failure();
  if (!isa<SignatureType>(signature))
    return p.emitError(p.getCurrentLocation(), "expected a signature type");
  return success();
}

ParseResult KGEN::parseSignature(AsmParser &p, TypeAttr &signature) {
  SignatureType type;
  if (parseSignature(p, type))
    return failure();
  signature = TypeAttr::get(type);
  return success();
}

void KGEN::printSignature(AsmPrinter &p, Type signatureType) {
  auto signature = cast<SignatureType>(signatureType);

  // If the signature has metadata, ask its dialect to print the signature.
  if (FnMetadataAttrInterface metadata = signature.getMetadata()) {
    metadata.printSignature(p, signature);
    return;
  }

  printOptionalParamSignature(p, signature.getInputParamTypes(),
                              signature.getResultParamTypes());

  auto printElt = [&](unsigned i) {
    p << signature.getArguments()[i];
    printArgConvention(p, signature.getArgConvention(i));
  };

  printSignatureValues(p, printElt, signature.getValues(), signature,
                       /*optionalResultList=*/false);
}

void KGEN::printSignature(AsmPrinter &p, Operation *op, TypeAttr signature) {
  printSignature(p, cast<SignatureType>(signature.getValue()));
}

ParseResult KGEN::parseKGENSignature(AsmParser &p, FunctionType &functionType,
                                     SignatureType &signature) {
  llvm::SMLoc loc = p.getCurrentLocation();

  SmallVector<StringAttr> argNames;
  SmallVector<ArgConvention> argConventions;
  auto parseArg = [&](SmallVectorImpl<Type> &argTypes) -> ParseResult {
    if (p.parseType(argTypes.emplace_back()) ||
        parseArgConvention(p, argConventions.emplace_back()))
      return failure();
    return success();
  };

  FnEffects effects;
  OptionalParseResult result =
      parseOptionalSignatureValues(p, parseArg, functionType, effects,
                                   /*optionalResultList=*/false);
  if (result.has_value() && failed(*result))
    return failure();
  // Try to parse a signature alias.
  if (!result.has_value()) {
    result = p.parseOptionalType(signature);
    if (result.has_value() && failed(*result))
      return failure();
    if (!result.has_value())
      return p.emitError(loc, "expected a KGEN signature");
    functionType = signature.getValues();
    signature = SignatureType::remapToSignature(
        {}, {}, functionType, signature.getArgConventions(),
        signature.getFnEffects(), signature.getMetadata(),
        [&] { return p.emitError(loc); });
    return success();
  }

  signature = SignatureType::remapToSignature({}, {}, functionType,
                                              argConventions, effects, {},
                                              [&] { return p.emitError(loc); });
  return success(!!signature);
}

void KGEN::printSignatureValues(AsmPrinter &p, FunctionType functionType,
                                SignatureType signature) {
  // If the signature has metadata, ask its dialect to print the signature.
  if (FnMetadataAttrInterface metadata = signature.getMetadata()) {
    metadata.printSignature(p, signature);
    return;
  }

  auto printElt = [&](unsigned i) {
    p << functionType.getInput(i);
    printArgConvention(p, signature.getArgConvention(i));
  };

  printSignatureValues(p, printElt, functionType, signature,
                       /*optionalResultList=*/false);
}

ParseResult KGEN::parseOptionalDecorators(AsmParser &p,
                                          DecoratorsAttr &decorators) {
  SmallVector<TypedAttr> decoVals;
  if (succeeded(p.parseOptionalKeyword("decorators"))) {
    if (p.parseCommaSeparatedList(AsmParser::Delimiter::LessGreater, [&] {
          return parseColonTypeParamValue(p, decoVals.emplace_back());
        }))
      return failure();
  }
  decorators = DecoratorsAttr::get(p.getContext(), decoVals);
  return success();
}

void KGEN::printOptionalDecorators(OpAsmPrinter &p, Operation *op,
                                   ArrayRef<TypedAttr> decorators) {
  if (decorators.empty())
    return;
  p.printNewline();
  p << "  decorators <";
  llvm::interleaveComma(decorators, p, [&](TypedAttr decorator) {
    if (decorators.size() > 1) {
      p.printNewline();
      p << "    ";
    }
    printColonTypeParamValue(p, decorator);
  });
  p << ">";
}

/// Parse the always_inline related keywords if present.
ParseResult KGEN::parseOptionalInline(OpAsmParser &parser,
                                      InlineLevelAttr &attr) {
  // Handle always_inline.
  InlineLevel inlineLevel;
  if (succeeded(parser.parseOptionalKeyword("always_inline")))
    inlineLevel = InlineLevel::Always;
  else if (succeeded(parser.parseOptionalKeyword("always_inline_no_debug")))
    inlineLevel = InlineLevel::AlwaysNoDebug;
  else if (succeeded(parser.parseOptionalKeyword("no_inline")))
    inlineLevel = InlineLevel::Never;
  else
    inlineLevel = InlineLevel::Automatic;
  attr = InlineLevelAttr::get(parser.getContext(), inlineLevel);
  return success();
}

void KGEN::printOptionalInline(AsmPrinter &p, InlineLevel level) {
  if (level == InlineLevel::Always)
    p << " always_inline";
  else if (level == InlineLevel::AlwaysNoDebug)
    p << " always_inline_no_debug";
  else if (level == InlineLevel::Never)
    p << " no_inline";
}

ParseResult KGEN::parseSymbolExport(AsmParser &p, ExportKindAttr &exportKind) {
  ExportKind value = ExportKind::NotExported;
  if (succeeded(p.parseOptionalKeyword("export"))) {
    value = ExportKind::Exported;
    if (succeeded(p.parseOptionalKeyword("C")))
      value = ExportKind::CExported;
    else if (succeeded(p.parseOptionalKeyword("package")))
      value = ExportKind::PackageExported;
  }
  exportKind = ExportKindAttr::get(p.getContext(), value);
  return success();
}

void KGEN::printSymbolExport(AsmPrinter &p, Operation *op,
                             ExportKindAttr exportKind) {
  if (exportKind.getValue() != ExportKind::NotExported) {
    p << " export";
    switch (exportKind.getValue()) {
    case ExportKind::CExported:
      p << " C";
      break;
    case ExportKind::PackageExported:
      p << " package";
      break;
    default:
      break;
    }
  }
}

ParseResult KGEN::parseParameterValues(AsmParser &p,
                                       ParameterExprArrayAttr &values) {
  SmallVector<TypedAttr> elts;
  if (parseParameterValues(p, elts))
    return failure();
  values = ParameterExprArrayAttr::get(p.getContext(), elts);
  return success();
}

ParseResult KGEN::parseParameterValues(AsmParser &p,
                                       SmallVectorImpl<TypedAttr> &values) {
  return p.parseCommaSeparatedList(
      OpAsmParser::Delimiter::OptionalLessGreater, [&]() -> ParseResult {
        TypedAttr value;
        if (parseParamValueDefaultingToIndex(p, value))
          return failure();
        values.push_back(value);
        return success();
      });
}

void KGEN::printParameterValues(OpAsmPrinter &p, Operation *op,
                                ParameterExprArrayAttr values) {
  printParameterValues(p, values);
}

void KGEN::printParameterValues(AsmPrinter &p, ArrayRef<TypedAttr> values) {
  if (values.empty())
    return;
  p << '<';
  llvm::interleaveComma(values, p, [&](TypedAttr value) {
    auto valType = value.getType();
    if (!valType.isIndex()) {
      p << ":";
      printKGENType(p, valType);
      p << " ";
    }
    printParamValue(p, value);
  });
  p << '>';
}

ParseResult KGEN::parseParametricCallee(OpAsmParser &p, TypedAttr &callee) {
  Type type;
  llvm::SMLoc loc = p.getCurrentLocation();
  if (p.parseLSquare() || parseKGENType(p, type) || p.parseColon() ||
      parseParamValue(p, callee, type) || p.parseRSquare())
    return failure();

  if (!isa<ParamRefType, SignatureType>(callee.getType()))
    return p.emitError(loc, "callee parameter type must be a signature type");
  return success();
}

void KGEN::printParametricCallee(OpAsmPrinter &p, Operation *,
                                 TypedAttr callee) {
  p << "[";
  printKGENType(p, callee.getType());
  p << ": ";
  printParamValue(p, callee);
  p << "]";
}

/// Compare a range of values from an "originator" to a corresponding range of
/// values from a "target".  If the two mismatch, emit an error that tries to
/// explain the issue in a nice way.
template <typename TargetRange, typename OriginatorRange>
static ParseResult verifyMatchingLists(
    const OriginatorRange &originatorRange, const TargetRange &targetRange,
    StringRef originatorName, Location originatorLoc, StringRef targetName,
    Location targetLoc, StringRef itemName, StringRef propertyName) {
  // Check that the ranges have the same size.  If not, diagnose this.
  size_t numOriginator =
      std::distance(originatorRange.begin(), originatorRange.end());
  size_t numTarget = std::distance(targetRange.begin(), targetRange.end());
  if (numOriginator != numTarget) {
    auto diag = emitError(originatorLoc, originatorName)
                << " has " << numOriginator << " " << itemName
                << (numOriginator != 1 ? "s" : "") << " but @" << targetName
                << " expects " << numTarget;
    if (originatorLoc != targetLoc)
      diag.attachNote(targetLoc) << "@" << targetName << " declared here";
    return failure();
  }

  // If they have the same sizes, diagnose any mismatches between their
  // elements.

  // NOTE: llvm::zip doesn't work with LLVM mapped iterators.
  auto targetIt = targetRange.begin();
  auto originatorIt = originatorRange.begin();
  for (size_t itemNum = 0; itemNum != numTarget; ++itemNum) {
    auto targetVal = *targetIt++;
    auto originatorVal = *originatorIt++;
    if (originatorVal == targetVal)
      continue;

    auto diag = emitError(originatorLoc, originatorName)
                << ' ' << itemName << " #" << itemNum << " has " << propertyName
                << ' ' << originatorVal << " but @" << targetName
                << " expected " << propertyName << ' ' << targetVal;
    if (originatorLoc != targetLoc)
      diag.attachNote(targetLoc) << "@" << targetName << " declared here";
    return failure();
  }

  return success();
}

/// Check that the specified declaration signatures match, checking the
/// parameter and value type information.
LogicalResult
KGEN::verifyDeclSignaturesMatch(StringRef lhsName, SignatureType lhsSig,
                                Location lhsLoc, StringRef rhsName,
                                SignatureType rhsSig, Location rhsLoc) {
  VerboseCompilerTimeTraceScope traceScope("verifyDeclSignaturesMatch");

  FunctionType lhsType = lhsSig.getValues();
  FunctionType rhsType = rhsSig.getValues();

  /// Verify that a list of parameter declarations from a generator or func
  /// matches those of an interface.  This produces an error diagnostic and
  /// returns failure when a problem is detected, or returns true if
  /// everything is ok.
  if (failed(verifyMatchingLists(lhsSig.getInputParamTypes(),
                                 rhsSig.getInputParamTypes(), lhsName, lhsLoc,
                                 rhsName, rhsLoc, "input parameter", "type")) ||
      failed(verifyMatchingLists(
          lhsSig.getResultParamTypes(), rhsSig.getResultParamTypes(), lhsName,
          lhsLoc, rhsName, rhsLoc, "result parameter", "type")) ||
      verifyMatchingLists(lhsType.getInputs(), rhsType.getInputs(), lhsName,
                          lhsLoc, rhsName, rhsLoc, "argument", "type") ||
      verifyMatchingLists(lhsType.getResults(), rhsType.getResults(), lhsName,
                          lhsLoc, rhsName, rhsLoc, "result", "type") ||
      verifyMatchingLists(lhsSig.getArgConventions(),
                          rhsSig.getArgConventions(), lhsName, lhsLoc, rhsName,
                          rhsLoc, "argument", "convention"))
    return failure();

  if (lhsSig.getFnEffects() != rhsSig.getFnEffects()) {
    auto diag = emitError(lhsLoc, lhsName)
                << " function effects are " << lhsSig.getFnEffects() << " but @"
                << rhsName << " expected " << rhsSig.getFnEffects();
    if (lhsLoc != rhsLoc)
      diag.attachNote(rhsLoc) << rhsName << " declared here";
    return failure();
  }

  if (lhsSig.getMetadata() != rhsSig.getMetadata()) {
    auto diag = emitError(lhsLoc, lhsName)
                << " metadata is " << lhsSig.getMetadata() << " but @"
                << rhsName << " expected " << rhsSig.getMetadata();
    if (lhsLoc != rhsLoc)
      diag.attachNote(rhsLoc) << rhsName << " declared here";
    return failure();
  }

  return success();
}

LogicalResult
KGEN::verifyParamDeclsMatch(StringRef paramKind, StringRef originatorName,
                            ArrayRef<TypedAttr> paramValues,
                            Location originatorLoc, StringRef targetName,
                            ArrayRef<ParamDeclAttr> decls, Location targetLoc) {
  using llvm::map_range;
  auto getType = [](auto attr) -> Type { return attr.getType(); };
  return verifyMatchingLists(
      map_range(paramValues, getType), map_range(decls, getType),
      originatorName, originatorLoc, targetName, targetLoc, paramKind, "type");
}

LogicalResult KGEN::checkResultParameterTypes(Operation *op,
                                              ArrayRef<TypedAttr> resultParams,
                                              DeclInterface decl) {
  // Check the parameters match up.
  ArrayRef<ParamDeclAttr> paramResults = decl.getResultParams();
  if (resultParams.size() != paramResults.size())
    return op->emitOpError("expected ")
           << paramResults.size() << " parameters for enclosing op";

  for (size_t i = 0, e = paramResults.size(); i != e; ++i) {
    Type expectedTy = paramResults[i].getType();
    Type actualTy = cast<TypedAttr>(resultParams[i]).getType();
    if (actualTy != expectedTy)
      return op->emitOpError("parameter #") << i << " has type " << actualTy
                                            << " but should be " << expectedTy;
  }
  return success();
}

LogicalResult KGEN::checkResultArgumentTypes(Operation *op,
                                             ArrayRef<TypedAttr> resultParams,
                                             FuncInterface func) {
  if (failed(checkResultParameterTypes(
          op, resultParams, cast<DeclInterface>(func.getOperation()))))
    return failure();
  return checkOperandTypes(op, func.getResultTypes());
}

ExportMap KGEN::getExportedSymbols(ModuleOp module) {
  llvm::MapVector<StringAttr, ExportedSymbol> exportedSymbols;
  for (auto op : module.getOps<ExportInterface>()) {
    if (op.isExported())
      exportedSymbols.insert(
          {op.getLinkageNameAttr(),
           ExportedSymbol(op.getExportKind(), isa<GlobalOp>(*op))});
  }
  return exportedSymbols;
}

/// Return if the given decorator matches an annotation, whose scopes are split
/// into the given parts.
static bool isDecorator(TypedAttr decorator,
                        ArrayRef<StringRef> annotationParts) {
  if (auto apply = dyn_cast<KGEN::ParamOperatorAttr>(decorator))
    decorator = apply.getOperand(0);

  auto sym = dyn_cast<KGEN::SymbolConstantAttr>(decorator);
  if (!sym)
    return false;
  SymbolRefAttr symRef = sym.getSymbol();
  ArrayRef<FlatSymbolRefAttr> nestedRefs = symRef.getNestedReferences();

  // Check the root reference.
  if (symRef.getRootReference() != annotationParts.front() ||
      nestedRefs.size() != annotationParts.size() - 1)
    return false;
  // Check the middle references.
  for (int i = 0, e = annotationParts.size() - 2; i < e; ++i)
    if (nestedRefs[i].getValue() != annotationParts[i + 1])
      return false;
  // Check the leaf reference.
  return nestedRefs.back().getValue().starts_with(annotationParts.back());
}

bool KGEN::hasDecorator(ArrayRef<TypedAttr> decorators, StringRef annotation) {
  SmallVector<StringRef> parts;
  annotation.split(parts, "::");
  return llvm::any_of(decorators, [&](TypedAttr decorator) {
    return isDecorator(decorator, parts);
  });
}

bool KGEN::hasAnyDecorator(ArrayRef<TypedAttr> decorators,
                           ArrayRef<StringLiteral> annotations) {
  return llvm::any_of(annotations, [&](const StringLiteral &annot) {
    return hasDecorator(decorators, annot);
  });
}

ParseResult KGEN::parseRegionWithArgs(OpAsmParser &p, Region &region) {
  SmallVector<OpAsmParser::Argument> args;
  if (p.parseArgumentList(args, AsmParser::Delimiter::OptionalParen,
                          /*allowType=*/true) ||
      p.parseRegion(region, args))
    return failure();
  return success();
}

void KGEN::printRegionWithArgs(OpAsmPrinter &p, Operation *op, Region &region) {
  if (!region.getArguments().empty()) {
    p << '(';
    llvm::interleaveComma(region.getArguments(), p, [&](BlockArgument arg) {
      p.printRegionArgument(arg);
    });
    p << ") ";
  }
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}
