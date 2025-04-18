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
#include "AsyncRT/CompilerSupport/Context.h"
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
                                      ParamDeclParseHookTy parseDeclElt) {
  // Parse the input list.
  if (parseParamDecls(parser, inputParamDecls, parseDeclElt))
    return failure();

  // Check to see if we have results and parse them if so.
  if (succeeded(parser.parseOptionalArrow())) {
    if (parseParamDecls(parser, resultParamDecls, parseDeclElt))
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
OptionalParseResult KGEN::parseOptionalSignatureValues(
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

/// Parse an operand and result type list with metadata for a plain (i.e.
/// non-lit) signature.
static OptionalParseResult parseOptionalNewKGENSignature(AsmParser &p,
                                                         Type &signature) {
  llvm::SMLoc loc = p.getCurrentLocation();
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
    signature =
        FuncType::getChecked([&] { return p.emitError(loc); }, functionType,
                             argConventions, effects, /*metadata=*/{});
    if (!signature)
      return failure();
  }
  return result;
}

/// Parse a plain (i.e. non-LIT) generator type.
static OptionalParseResult parseOptionalKGENGenerator(AsmParser &p,
                                                      Type &generator) {
  SmallVector<Type> paramTypes;
  Type body;

  bool sawParamList = false;
  if (succeeded(p.parseOptionalLess())) {
    sawParamList = true;
    // A failure is if the param list is not empty, and param type parsing
    // failed.
    if (failed(p.parseOptionalGreater()) &&
        (parseParamTypes(p, paramTypes) || p.parseGreater())) {
      return failure();
    }
  }

  // Try to parse an optional FuncType immediately here because we do not
  // allow standalone FuncTypes yet.
  OptionalParseResult optionalSigBody = parseOptionalNewKGENSignature(p, body);
  if (optionalSigBody.has_value()) {
    if (failed(*optionalSigBody))
      return failure();
    generator = GeneratorType::get(paramTypes, body);
    return mlir::success();
  }

  // For anything that's not a func type generator, require a param list.
  if (!sawParamList)
    return std::nullopt;

  if (parseParamType(p, body))
    return failure();

  generator = GeneratorType::get(paramTypes, body);
  return mlir::success();
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

  // Try to parse an optional generator. Generators begin with `<`.
  {
    GeneratorType generator;
    OptionalParseResult result = parseOptionalKGENGenerator(p, generator);
    if (result.has_value()) {
      if (failed(*result))
        return failure();
      type = generator;
      return LogicalResult::success();
    }
  }

  // Try to parse an optional FuncType. FuncTypes begin with `(`.
  // For now we parse all standalone FuncTypes as FuncType generator types for
  // back-compat.
  {
    FuncType signature;
    OptionalParseResult result = parseOptionalNewKGENSignature(p, signature);
    if (result.has_value()) {
      if (failed(*result))
        return failure();
      type = GeneratorType::get(/*inputParamTypes=*/{}, signature);
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
  } else if (auto signature = dyn_cast<FuncType>(type)) {
    printFuncType(p, signature);
  } else if (auto generator = dyn_cast<GeneratorType>(type)) {
    printGenerator(p, generator);
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

/// Parse a "colon type" production if present or default to `defaultType` type
/// if not.
ParseResult KGEN::parseColonTypeOrDefault(AsmParser &parser, Type &type,
                                          Type defaultType) {
  auto result = parseOptionalColonType(parser, type);
  if (!result.has_value()) {
    type = defaultType;
    return success();
  }
  return result.value();
}

/// Parse a "colon type" production if present or default to index if not.  This
/// is commonly used in our parameter representation.
ParseResult KGEN::parseColonTypeOrIndex(AsmParser &parser, Type &type) {
  return parseColonTypeOrDefault(parser, type,
                                 parser.getBuilder().getIndexType());
}

/// print `: <type>` or elide it entirely if type is an `index` type.
void KGEN::printColonTypeOrDefault(AsmPrinter &p, Type type, Type defaultType) {
  // Index type is the default so it doesn't print.
  if (type == defaultType)
    return;
  p << ": ";
  printKGENType(p, type);
}

/// print `: <type>` or elide it entirely if type is an `index` type.
void KGEN::printColonTypeOrIndex(AsmPrinter &p, Type type) {
  return printColonTypeOrDefault(p, type, IndexType::get(type.getContext()));
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
  OptionalParseResult result = parseOptionalKGENType(p, type);
  if (result.has_value())
    return success();

  // If not a mlir Type, it's a parameter in the type-domain. Parse as a
  // parameter and wrap with ParamType.
  TypedAttr typeParam;
  if (parseTypeParamValue(p, typeParam))
    return failure();
  type = ParamType::get(typeParam);
  return success();
}

void KGEN::printParamType(AsmPrinter &p, Type type) {
  // A "ParamType" is either:
  // 1. An actual mlir Type,
  // 2. A type-value in the type domain (i.e. wrapped with ParamType), or
  // 3. A type-value in the value domain (i.e. wrapped with TypeValueType).
  //
  // For case 2, the ParamType wrapper around the internal parameter is
  // omitted for simplicity. The internal parameter is printed directly (with
  // an optional colon type prefix).
  // For case 3, the TypeValueType is NOT omitted to differentiate with case 2.
  if (auto paramRef = dyn_cast<ParamType>(type))
    printTypeParamValue(p, paramRef.getParam());
  else
    printKGENType(p, type);
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
    AsmPrinter &p, TypeParamAttr type,
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
    value = TypeParamAttr::get(typeValue, typeValue, type, vtable);
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
      value = TypeParamAttr::get(typeValue, mlirType, type, vtable);
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

  value = TypeParamAttr::get(typeValue, mlirType, type, vtable);
  return mlir::success();
}

LogicalResult KGEN::printSugaredTypeValue(
    AsmPrinter &p, TypedAttr value,
    llvm::function_ref<void(AsmPrinter &, Type)> typePrinter) {
  auto type = dyn_cast<TypeParamAttr>(value);
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

ParseResult KGEN::parseParamDeclAttrs(AsmParser &p,
                                      SmallVector<ParamDeclAttr> &decls) {
  return p.parseCommaSeparatedList([&]() {
    decls.emplace_back();
    return parseParamDecl(p, decls.back());
  });
}

void KGEN::printParamDeclAttrs(AsmPrinter &p, ArrayRef<ParamDeclAttr> decls) {
  llvm::interleaveComma(decls, p,
                        [&](ParamDeclAttr decl) { printParamDecl(p, decl); });
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
                                      ParamDeclPrintHookTy printInputElt,
                                      ParamDeclPrintHookTy printResultElt) {
  if (inputParamDecls.empty() && resultParams.empty())
    return;

  p << '<';
  printParamDecls(p, inputParamDecls, printInputElt);

  if (!resultParams.empty()) {
    p << " -> ";
    printParamDecls(p, resultParams, printResultElt);
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
  static const char *keywordTypes[] = {
      "f8e5m2", "f8e4m3fn", "f8e3m4", "f8e5m2fnuz", "f8e4m3fnuz", "bf16", "f16",
      "f32",    "f64",      "f80",    "f128",       "index",      "none"};
  if (llvm::is_contained(keywordTypes, name))
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

ParseResult KGEN::parseBindParams(AsmParser &p, TypedAttr &generator,
                                  SmallVectorImpl<TypedAttr> &paramValues,
                                  Type preParsedGeneratorType) {
  if (!preParsedGeneratorType &&
      parseColonTypeOrIndex(p, preParsedGeneratorType))
    return failure();

  if (parseParamValue(p, generator, preParsedGeneratorType))
    return failure();

  auto genType = cast<GeneratorType>(preParsedGeneratorType);
  // Parse each operand, inferring its type from the signature type. Bound
  // parameters are allowed to refine the types of subsequent parameters, so
  // specialize the types as we go.
  ParameterEvaluator evaluator;
  for (Type type : genType.getInputParamTypes()) {
    if (failed(p.parseOptionalComma()))
      break;
    if (parseParamValue(p, paramValues.emplace_back(),
                        evaluator.getReboundType(type)))
      return failure();
    evaluator.addInputValue(paramValues.back());
  }
  return success();
}

void KGEN::printBindParams(AsmPrinter &p, TypedAttr generator,
                           ArrayRef<TypedAttr> paramValues) {
  printColonTypeParamValue(p, generator);
  for (TypedAttr value : paramValues) {
    p << ", ";
    printParamValue(p, value);
  }
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
  printAsMojoStringLiteral(name, p.getStream());
  if (needsQuotes)
    p << '"';
}

ParseResult KGEN::parseParamNames(AsmParser &p,
                                  SmallVector<StringAttr> &names) {
  return p.parseCommaSeparatedList(
      [&] { return parseParamName(p, names.emplace_back()); });
}

void KGEN::printParamNames(AsmPrinter &p, ArrayRef<StringAttr> names,
                           bool isRef) {
  llvm::interleaveComma(
      names, p, [&](StringAttr name) { printParamName(p, name, isRef); });
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
  case (uint32_t)POC::Apply: {
    auto sigGen = dyn_cast_or_null<FuncTypeGeneratorType>(type);
    if (!sigGen)
      return p.emitError(p.getCurrentLocation(),
                         "expected a func type generator type for 'apply'");

    if (parseParamValue(p, operands.emplace_back(), sigGen))
      return failure();
    // Parse each operand, inferring its type from the signature type.
    IndexDepthAdjuster adjuster(/*adjustDepth=*/-1);
    for (Type type : sigGen.getBody().getArguments())
      if (p.parseComma() ||
          parseParamValue(p, operands.emplace_back(), adjuster.replace(type)))
        return failure();
    return success();
  }
  case (uint32_t)POC::ApplyResultSlot: {
    auto sigGen = dyn_cast_or_null<FuncTypeGeneratorType>(type);
    if (!sigGen)
      return p.emitError(
          p.getCurrentLocation(),
          "expected a func type generator type for 'apply_result_slot'");
    FuncType sig = sigGen.getBody();

    if (parseParamValue(p, operands.emplace_back(), sigGen))
      return failure();
    if (sig.getNumArguments() < 1)
      return p.emitError(
          p.getCurrentLocation(),
          "'apply_result_slot' callee must have at least one result");
    // Parse each operand besides the result slot.
    auto argTypes = sig.getArguments().drop_back(sig.hasMemoryOnlyResult());
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
    // Parse the target.
    if (parseParamValue(p, operands.emplace_back(),
                        TargetType::get(p.getContext())) ||
        p.parseComma())
      return failure();

    // Parse the emission kind.
    if (succeeded(p.parseOptionalEqual())) {
      StringRef emissionKind;
      if (p.parseKeyword(&emissionKind))
        return failure();
      std::optional<EmitAs> kind = symbolizeEmitAs(emissionKind);
      if (!kind) {
        return p.emitError(p.getCurrentLocation(),
                           "the immediate emission kind must be either "
                           "'=llvm', '=asm', '=llvm-opt', or '=object'");
      }
      operands.emplace_back(EmitAsAttr::get(p.getContext(), *kind));
    } else if (parseParamValue(p, operands.emplace_back(),
                               p.getBuilder().getIndexType())) {
      return failure();
    }

    // Parse the emission options.
    if (p.parseComma() || parseParamValue(p, operands.emplace_back(),
                                          StringType::get(p.getContext())))
      return failure();

    // Parse the fallibility option.
    if (p.parseComma() ||
        parseParamValue(p, operands.emplace_back(), p.getBuilder().getI1Type()))
      return failure();

    // Parse the type.
    if (p.parseComma() || parseColonTypeParamValue(p, operands.emplace_back()))
      return failure();

    return success();
  }
  case (uint32_t)POC::GetLinkageName:
    if (parseParamValue(p, operands.emplace_back(),
                        TargetType::get(p.getContext())) ||
        p.parseComma() || parseColonTypeParamValue(p, operands.emplace_back()))
      return failure();
    return success();

  case (uint32_t)POC::CompileOffloadClosure: {
    // Parse the type.
    // Use type with parseParamValue here instead of
    // using parseColonTypeParamValue to get the type because the parser
    // should already parsed the function type here if it's the first operand.
    if (parseParamValue(p, operands.emplace_back(), type))
      return failure();

    return success();
  }

  case (uint32_t)POC::GetVTableEntry:
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
  case (uint32_t)POC::DataToStr:
    if (parseParamValue(p, operands.emplace_back(), type) || p.parseComma() ||
        parseParamValue(p, operands.emplace_back(), VariadicType::get(type)))
      return failure();

    return success();

  case (uint32_t)POC::StringAddress:
    return parseParamValue(p, operands.emplace_back(),
                           StringType::get(type.getContext()));
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
      // Try to parse *(0,0) as an index reference.
      size_t depth, index;
      if (p.parseInteger(depth) || p.parseComma() || p.parseInteger(index) ||
          p.parseRParen())
        return failure();
      value = ParamIndexRefAttr::get(depth, index, type);
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

    // Handle other expressions with the same syntax as ParamOperatorAttr
    // TODO: Could turn this into a trait and push all this logic into the
    // attrs, which would also be nice for LIT attrs.
    if (opcode == (uint32_t)POCAliases::kInvalid) {
      if (keyword == "upcast" && operandType) {
        TypedAttr operand;
        VTableAttr vtable;
        if (parseParamValue(p, operand, operandType))
          return failure();
        if (succeeded(p.parseOptionalComma())) {
          vtable = cast_or_null<VTableAttr>(VTableAttr::parse(p, {}));
          if (!vtable)
            return failure();
        } else {
          vtable = VTableAttr::get(type.getContext(), {});
        }
        if (p.parseRParen())
          return failure();
        value = UpcastAttr::get(type, operand, vtable);
        return success();
      }

      if (keyword == "bind_params" && operandType) {
        TypedAttr generator;
        SmallVector<TypedAttr> paramValues;
        if (parseBindParams(p, generator, paramValues, operandType))
          return failure();
        if (p.parseRParen())
          return failure();
        value =
            BindParamsAttr::get(p.getContext(), generator, paramValues, type);
        return success();
      }

      return p.emitError(loc, "unknown expression ") << keyword;
    }

    // Otherwise it is a ParamOperatorAttr.  Parse the operand list.
    SmallVector<TypedAttr> operands;

    // If there was no specified element type, then pick a default based on the
    // opcode in question.
    if (!operandType) {
      switch (opcode) {
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
      case (uint32_t)POC::GetVTableEntry:
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
      operands.emplace_back(
          p.getBuilder().getIntegerAttr(operands[0].getType(), -1));
      opcode = (uint32_t)POC::Mul;
    }

    // Desugar the subtract operator from `sub(a, b)` to `add(a, mul(b, -1))`
    if (opcode == (uint32_t)POCAliases::SUB) {
      if (operands.size() != 2)
        return p.emitError(loc, "sub operator expects two operands");
      operands[1] = ParamOperatorAttr::getNeg(operands[1]);
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
    printParamValue(p, operands.back());
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
    // '=' is used to disambiguate the string form.
    if (auto emitAsAttr = dyn_cast<EmitAsAttr>(operands[1]))
      p << '=' << stringifyEmitAs(emitAsAttr.getValue());
    else
      printParamValue(p, operands[1]);
    p << ", ";
    printParamValue(p, operands[2]);
    p << ", ";
    printParamValue(p, operands[3]);
    p << ", ";
    printColonTypeParamValue(p, operands[4]);
    break;
  }
  case POC::GetLinkageName:
    printParamValue(p, operands[0]);
    p << ", ";
    printColonTypeParamValue(p, operands[1]);
    break;

  case POC::CompileOffloadClosure:
    printColonTypeParamValue(p, operands[0]);
    break;

  case POC::GetVTableEntry:
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
  case POC::DataToStr:
    p << ':';
    printKGENType(p, operands[0].getType());
    p << ' ';
    printParamValue(p, operands[0]);
    p << ", ";
    printParamValue(p, operands[1]);
    break;
  }
}

void KGEN::printAsMojoStringLiteral(StringRef name, raw_ostream &out) {
  for (unsigned char c : name) {
    switch (c) {
    case '\\':
      out << "\\\\";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\t':
      out << "\\t";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\a':
      out << "\\a";
      break;
    case '\b':
      out << "\\b";
      break;
    case '\f':
      out << "\\f";
      break;
    case '\v':
      out << "\\v";
      break;
    default:
      if (llvm::isPrint(c) && c != '"')
        out << c;
      else
        out << '\\' << llvm::hexdigit(c >> 4) << llvm::hexdigit(c & 0x0F);
      break;
    }
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

  if (auto bindParams = dyn_cast<BindParamsAttr>(value)) {
    p << "bind_params(";
    printBindParams(p, bindParams.getGenerator(), bindParams.getParamValues());
    p << ')';
    return;
  }

  if (auto declRef = dyn_cast<ParamDeclRefAttr>(value)) {
    bool isRef = isTypeExpr(value);
    if (auto type = dyn_cast<ParameterTypeInterface>(value.getType()))
      isRef |= type.isMetaType();
    if (forDiag)
      printAsMojoStringLiteral(declRef.getName(), p.getStream());
    else
      printParamName(p, declRef.getName(), isRef);
    return;
  }
  if (auto indexRef = dyn_cast<ParamIndexRefAttr>(value)) {
    p << "*(" << indexRef.getDepth() << ',' << indexRef.getIndex() << ")";
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

  // Handle other expressions with the same syntax as ParamOperatorAttr
  // TODO: Could turn this into a trait like ParameterTypeInterface and push all
  // this logic into the attrs, which would also be nice for LIT attrs.
  if (auto upcast = dyn_cast<UpcastAttr>(value)) {
    p << "upcast(:";
    printKGENType(p, upcast.getInputTypeValue().getType());
    p << ' ';
    printParamValue(p, upcast.getInputTypeValue());
    if (!upcast.getVTable().getEntries().empty()) {
      p << ", ";
      p.printStrippedAttrOrType(upcast.getVTable());
    }
    p << ')';
    return;
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

bool KGEN::isTypeExprType(Type type) { return isa<TypeType>(type); }

bool KGEN::isTypeExpr(TypedAttr attr) { return isTypeExprType(attr.getType()); }

KGEN::EnvAttr KGEN::getModularEnvAttr(MLIRContext *ctx,
                                      CompilationContext *compileCtx) {
  NamedAttrList envAttrs;

#ifdef MODULAR_PRODUCTION
  envAttrs.set("MODULAR_PRODUCTION", IntegerAttr::get(IndexType::get(ctx), 1));
#endif // MODULAR_PRODUCTION

#ifdef MODULAR_PARANOID
  envAttrs.set("MODULAR_PARANOID", IntegerAttr::get(IndexType::get(ctx), 1));
#endif // MODULAR_PARANOID

#ifdef MODULAR_ENABLE_GPU_PROFILING
  envAttrs.set("MODULAR_ENABLE_GPU_PROFILING",
               IntegerAttr::get(IndexType::get(ctx), 1));
#endif // MODULAR_ENABLE_GPU_PROFILING

#ifdef MODULAR_ENABLE_GPU_PROFILING_DETAILED
  envAttrs.set("MODULAR_ENABLE_GPU_PROFILING_DETAILED",
               IntegerAttr::get(IndexType::get(ctx), 1));
#endif // MODULAR_ENABLE_GPU_PROFILING_DETAILED

  envAttrs.set("BUILD_TYPE", StringAttr::get(STRINGIFY(BUILD_TYPE),
                                             KGEN::StringType::get(ctx)));
  envAttrs.set("MODULAR_ASYNCRT_MAX_PROFILING_LEVEL",
               IntegerAttr::get(IndexType::get(ctx),
                                MODULAR_ASYNCRT_MAX_PROFILING_LEVEL));

  if (compileCtx) {
    for (auto entry : compileCtx->mojoDefines) {
      auto k = entry.first;
#ifdef MODULAR_PRODUCTION
      // This is an end users release build. Pretend that the
      // `MODULAR_PRODUCTION` flag does not exist. This protects us from end
      // users trying to leak internal details.
      if (k == "MODULAR_PRODUCTION")
        continue;
#endif // MODULAR_PRODUCTION

      std::visit(
          [&](auto &&v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, bool>) {
              envAttrs.set(k, BoolAttr::get(ctx, v));
            } else if constexpr (std::is_same_v<T, int>) {
              envAttrs.set(k, IntegerAttr::get(IndexType::get(ctx), v));
            } else if constexpr (std::is_same_v<T, std::string>) {
              envAttrs.set(k, StringAttr::get(v, KGEN::StringType::get(ctx)));
            } else {
              // NOTE: This should be a static_assert, but that breaks in torch
              // compile tests on some mac devices.
              assert("non-exhaustive visitor!");
            }
          },
          entry.second);
    }
  }

  return KGEN::EnvAttr::get(envAttrs.getDictionary(ctx));
}

KGEN::EnvAttr KGEN::getModuleEnvAttr(ModuleOp moduleOp) {
  if (moduleOp->hasAttrOfType<KGEN::EnvAttr>(KGEN::EnvAttr::getEnvAttrName()))
    return moduleOp->getAttrOfType<KGEN::EnvAttr>(
        KGEN::EnvAttr::getEnvAttrName());

  return EnvAttr::get(DictionaryAttr::get(moduleOp.getContext()));
}

void KGEN::extendWithModularEnvAttr(ModuleOp moduleOp,
                                    CompilationContext *compileCtx) {
  moduleOp->setAttr(KGEN::EnvAttr::getEnvAttrName(),
                    KGEN::getModularEnvAttr(moduleOp.getContext(), compileCtx)
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

ParseResult
KGEN::parseStructDefFields(AsmParser &p,
                           SmallVector<StructDefFieldAttr> &fields) {
  MLIRContext *ctx = p.getContext();
  return p.parseCommaSeparatedList([&]() {
    StringAttr name;
    Type type;
    if (parseParamName(p, name) || p.parseColon() || parseKGENType(p, type))
      return failure();
    fields.push_back(StructDefFieldAttr::get(ctx, name, type));
    return mlir::success();
  });
}

void KGEN::printStructDefFields(AsmPrinter &p,
                                ArrayRef<StructDefFieldAttr> fields) {
  llvm::interleaveComma(fields, p, [&](StructDefFieldAttr field) {
    printParamName(p, field.getName());
    p << ": ";
    printKGENType(p, field.getType());
  });
}

//===----------------------------------------------------------------------===//
// Logic shared between funcs, generators, and generator interfaces
//===----------------------------------------------------------------------===//

ParseResult KGEN::parseArgConvention(AsmParser &p, ArgConvention &convention) {
  StringRef effectStr;
  llvm::SMLoc loc = p.getCurrentLocation();
  // Parse an optional input convention specifier.
  convention = ArgConvention::ReadReg;
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
  if (convention != ArgConvention::ReadReg)
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

void KGEN::printSignatureValues(AsmPrinter &p,
                                function_ref<void(unsigned)> printElt,
                                FunctionType functionType,
                                ArrayRef<ArgConvention> argConvs,
                                FnEffects fnEffects, bool optionalResultList) {
  p << '(';
  llvm::interleaveComma(llvm::seq<unsigned>(0, argConvs.size()), p, printElt);
  p << ')';

  // Print the function effects.
  impl::FnEffects effects = fnEffects.getImpl();
  if (effects != impl::FnEffects::None)
    p << ' ' << impl::stringifyFnEffects(effects);

  if (optionalResultList)
    p.printOptionalArrowTypeList(functionType.getResults());
  else
    p.printArrowTypeList(functionType.getResults());
}

ParseResult KGEN::parseFunctionFuncTypeGenerator(
    OpAsmParser &p, SmallVectorImpl<OpAsmParser::Argument> &args,
    ParamDeclArrayAttr &inputParams, ParamDeclArrayAttr &resultParams,
    FunctionType &functionType, FuncTypeGeneratorType &signature,
    ParamDeclParseHookTy parseDeclElt) {
  llvm::SMLoc loc = p.getCurrentLocation();
  SmallVector<ArgConvention> argConventions;
  FnEffects effects;
  if (parseOptionalParameterSpec(p, inputParams, resultParams, parseDeclElt))
    return failure();

  auto parseArg = [&](SmallVectorImpl<Type> &argTypes) -> ParseResult {
    // Parse the argument type and its input convention.
    OpAsmParser::Argument &arg = args.emplace_back();
    OptionalParseResult result =
        p.parseOptionalArgument(arg, /*allowType=*/true);

    // An SSA name is present and resulted in a parsing error.
    if (result.has_value() && failed(*result))
      return failure();

    // An SSA name is not present, try parsing just the type.
    if (!result.has_value()) {
      // Failed to parse the type as well.
      if (p.parseType(arg.type))
        return failure();

      // Without an SSA name, the location information will not be set for
      // the argument, use the current parser location.
      arg.ssaName.location = p.getCurrentLocation();
    }

    if (parseArgConvention(p, argConventions.emplace_back()))
      return failure();

    argTypes.push_back(arg.type);
    return success();
  };

  if (failed(parseSignatureValues(p, parseArg, functionType, effects,
                                  /*optionalResultList=*/true)))
    return failure();

  signature = FuncTypeGeneratorType::remapToFuncTypeGenerator(
      inputParams, functionType, argConventions, effects, {}, {},
      [&] { return p.emitError(loc); });
  return success(!!signature);
}

void KGEN::printFunctionFuncTypeGenerator(OpAsmPrinter &p, Region *region,
                                          ArrayRef<ParamDeclAttr> inputParams,
                                          ArrayRef<ParamDeclAttr> resultParams,
                                          FunctionType functionType,
                                          FuncTypeGeneratorType signature,
                                          ParamDeclPrintHookTy printInputElt,
                                          ParamDeclPrintHookTy printResultElt) {
  // Print the function arguments.
  FuncType sigBase = signature.getBody();
  auto printElt = [&](unsigned i) {
    if (!region)
      p << functionType.getInput(i);
    else
      p.printRegionArgument(region->getArgument(i));

    printArgConvention(p, sigBase.getArgConvention(i));
  };

  printOptionalParameterSpec(p, inputParams, resultParams, printInputElt,
                             printResultElt);
  printSignatureValues(p, printElt, functionType, sigBase.getArgConventions(),
                       sigBase.getFnEffects(),
                       /*optionalResultList=*/true);
}

ParseResult KGEN::parseOptionalParamSignature(
    AsmParser &p, SmallVectorImpl<Type> &inputParamTypes,
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
  if (p.parseCommaSeparatedList(parseIn))
    return failure();

  if (p.parseGreater())
    return failure();
  return success();
}

void KGEN::printOptionalParamSignature(AsmPrinter &p,
                                       ArrayRef<Type> inputParamTypes,
                                       function_ref<void(Type)> printInputTy) {
  if (inputParamTypes.empty())
    return;

  auto defaultPrintInputTy = [&](Type type) { printKGENType(p, type); };
  if (!printInputTy)
    printInputTy = defaultPrintInputTy;

  p << '<';
  llvm::interleaveComma(inputParamTypes, p, printInputTy);
  p << '>';
}

ParseResult KGEN::parseFuncType(AsmParser &p, Type &signature) {
  OptionalParseResult result = parseOptionalNewKGENSignature(p, signature);
  if (result.has_value())
    return *result;
  result = p.parseOptionalType(signature);
  if (!result.has_value())
    return p.emitError(p.getCurrentLocation(),
                       "expected '<' or '(' to begin a signature");
  if (failed(*result))
    return failure();
  if (!isa<FuncType>(signature))
    return p.emitError(p.getCurrentLocation(), "expected a signature type");
  return success();
}

void KGEN::printFuncType(AsmPrinter &p, FuncType signature) {
  // If the signature has metadata, ask its dialect to print the signature.
  if (FnMetadataAttrInterface metadata = signature.getMetadata()) {
    metadata.printFuncType(p, signature);
    return;
  }

  auto printElt = [&](unsigned i) {
    p << signature.getArguments()[i];
    printArgConvention(p, signature.getArgConvention(i));
  };

  printSignatureValues(p, printElt, signature.getValues(),
                       signature.getArgConventions(), signature.getFnEffects(),
                       /*optionalResultList=*/false);
}

ParseResult KGEN::parseKGENFuncTypeGenerator(AsmParser &p,
                                             FunctionType &functionType,
                                             FuncTypeGeneratorType &generator) {
  Type type;
  if (parseGenerator(p, type))
    return failure();
  generator = dyn_cast<FuncTypeGeneratorType>(type);
  if (!generator)
    return failure();
  functionType = generator.getBody().getValues();
  return success();
}

ParseResult KGEN::parseGenerator(AsmParser &p, Type &generator) {
  // Try parsing as a plain KGEN generator first (no metadata);
  OptionalParseResult result = parseOptionalKGENGenerator(p, generator);
  if (result.has_value())
    return *result;

  result = p.parseOptionalType(generator);
  if (!result.has_value())
    return p.emitError(p.getCurrentLocation(),
                       "expected '<' to begin a generator");
  if (failed(*result))
    return failure();
  if (!isa<GeneratorType>(generator))
    return p.emitError(p.getCurrentLocation(), "expected a generator type");
  return success();
}

void KGEN::printGenerator(AsmPrinter &p, GeneratorType generator) {
  if (GeneratorMetadataAttrInterface metadata = generator.getMetadata()) {
    metadata.printGenerator(p, generator);
    return;
  }

  // For maximum textual IR back-compat, skip printing the empty angle brackets
  // for func type generators. We should remove this sugar after the migration.
  if (!isa<FuncType>(generator.getBody()) ||
      !generator.getInputParamTypes().empty()) {
    p << '<';
    printParamTypes(p, generator.getInputParamTypes());
    p << '>';
  }
  printParamType(p, generator.getBody());
}

void KGEN::printSignatureValues(AsmPrinter &p, FunctionType functionType,
                                FuncTypeGeneratorType sigGen) {
  // If the signature has metadata, ask its dialect to print the signature.
  if (GeneratorMetadataAttrInterface metadata = sigGen.getMetadata()) {
    metadata.printGenerator(p, sigGen);
    return;
  }

  FuncType signature = sigGen.getBody();
  auto printElt = [&](unsigned i) {
    p << functionType.getInput(i);
    printArgConvention(p, signature.getArgConvention(i));
  };

  printSignatureValues(p, printElt, functionType, signature.getArgConventions(),
                       signature.getFnEffects(),
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
  else if (succeeded(parser.parseOptionalKeyword("always_inline_builtin")))
    inlineLevel = InlineLevel::AlwaysBuiltin;
  else if (succeeded(parser.parseOptionalKeyword("no_inline")))
    inlineLevel = InlineLevel::Never;
  else
    inlineLevel = InlineLevel::Automatic;
  attr = InlineLevelAttr::get(parser.getContext(), inlineLevel);
  return success();
}

void KGEN::printOptionalInline(AsmPrinter &p, InlineLevel level) {
  switch (level) {
  case InlineLevel::Automatic:
    break;
  case InlineLevel::Always:
    p << " always_inline";
    break;
  case InlineLevel::AlwaysNoDebug:
    p << " always_inline_no_debug";
    break;
  case InlineLevel::AlwaysBuiltin:
    p << " always_inline_builtin";
    break;
  case InlineLevel::Never:
    p << " no_inline";
    break;
  }
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

  if (!isa<ParamType, FuncTypeGeneratorType>(callee.getType()))
    return p.emitError(
               loc,
               "callee parameter type must be a func type generator type. Got ")
           << callee.getType();
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
LogicalResult KGEN::verifyDeclSignaturesMatch(
    StringRef lhsName, FuncTypeGeneratorType lhsSigGen, Location lhsLoc,
    StringRef rhsName, FuncTypeGeneratorType rhsSigGen, Location rhsLoc) {
  VerboseCompilerTimeTraceScope traceScope("verifyDeclSignaturesMatch");

  FuncType lhsSig = lhsSigGen.getBody();
  FuncType rhsSig = rhsSigGen.getBody();

  FunctionType lhsType = lhsSig.getValues();
  FunctionType rhsType = rhsSig.getValues();

  /// Verify that a list of parameter declarations from a generator or func
  /// matches those of an interface.  This produces an error diagnostic and
  /// returns failure when a problem is detected, or returns true if
  /// everything is ok.
  if (failed(verifyMatchingLists(
          lhsSigGen.getInputParamTypes(), rhsSigGen.getInputParamTypes(),
          lhsName, lhsLoc, rhsName, rhsLoc, "input parameter", "type")) ||
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

LogicalResult KGEN::verifyCallOperands(Operation *op, ValueRange args,
                                       FuncType callee, bool ignoreByRef) {
  unsigned numByRef = ignoreByRef * callee.getNumAsyncReturnSlots();
  if (args.size() != callee.getNumArguments() - numByRef) {
    return op->emitOpError("callee expected ")
           << callee.getNumArguments() << " arguments but operation only has "
           << args.size();
  }
  for (auto [i, arg, type] :
       llvm::enumerate(args, callee.getArguments().drop_back(numByRef))) {
    if (arg.getType() != type) {
      return op->emitOpError("callee argument #")
             << i << " expected type " << type
             << " but operation argument has type " << arg.getType();
    }
  }
  return success();
}

LogicalResult KGEN::verifyCallResults(Operation *op, ValueRange results,
                                      FuncType callee) {
  if (results.size() != callee.getNumResults()) {
    return op->emitOpError("callee expected ")
           << callee.getNumArguments() << " results but operation only has "
           << results.size();
  }
  for (auto [i, res, type] : llvm::enumerate(results, callee.getResults())) {
    if (res.getType() != type) {
      return op->emitOpError("callee result #")
             << i << " expected type " << type
             << " but operation result has type " << res.getType();
    }
  }
  return success();
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

ParseResult KGEN::parseMemSymbolParts(AsmParser &p,
                                      MemSymbolTripleParts &parts) {
  SmallVector<SymbolParts> symbols;
  do {
    SymbolRefAttr callee;
    ParameterExprArrayAttr paramValues;
    if (p.parseAttribute(callee) || parseParameterValues(p, paramValues))
      return failure();
    symbols.push_back(SymbolParts{callee, paramValues});
  } while (succeeded(p.parseOptionalComma()));
  switch (symbols.size()) {
  case 2:
    parts = MemSymbolTripleParts{{}, symbols[0], symbols[1]};
    break;
  case 3:
    parts = MemSymbolTripleParts{symbols[0], symbols[1], symbols[2]};
    break;
  default:
    return p.emitError(p.getCurrentLocation(), "expected 2 or 3 symbols");
  }
  return success();
}

SymbolConstantAttr KGEN::makeSymbol(Type type, SymbolRefAttr symbol,
                                    ParameterExprArrayAttr paramValues,
                                    bool isConstructor) {
  SmallVector<Type> inputs;
  if (isConstructor) {
    inputs = {type, type};
  } else {
    inputs = {type};
  }
  return SymbolConstantAttr::get(
      symbol,
      FuncTypeGeneratorType::get(
          {}, FunctionType::get(type.getContext(), inputs, {}), {}, {}, {}, {}),
      paramValues);
}

void KGEN::printMemSymbolTripleAttrWithoutType(AsmPrinter &p,
                                               SymbolConstantAttr copy,
                                               SymbolConstantAttr move,
                                               SymbolConstantAttr del) {
  if (copy) {
    p << copy.getSymbol();
    printParameterValues(p, copy.getParamValues());
    p << ", ";
  }
  if (move) {
    p << move.getSymbol();
    printParameterValues(p, move.getParamValues());
    p << ", ";
  }
  if (del) {
    p << del.getSymbol();
    printParameterValues(p, del.getParamValues());
  }
}
