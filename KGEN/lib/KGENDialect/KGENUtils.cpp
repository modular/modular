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
#include "KGEN/KGENDialect/KGENTypeInterfaces.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/Compiler/ParserUtils.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/ML/DType.h"
#include "Support/Profiling/TimeProfiler.h"
#include "Support/STLExtras.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Parameter Type and Value Printing and Parsing
//===----------------------------------------------------------------------===//

/// Return the string form for an attribute value that is printed in a <>
/// context in the .mlir file.
std::string KGEN::getParamAsString(Attribute value) {
  SmallVector<char, 128> result;
  {
    llvm::raw_svector_ostream os(result);
    if (auto ta = dyn_cast<TypedAttr>(value)) {
      StreamAsmPrinter p(os);
      printParamValue(p, ta);
    } else {
      os << value;
    }
  }
  return std::string(result.data(), result.size());
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
                                      ParamDeclArrayAttr &resultParamDecls) {
  // Parse the input list.
  if (parseParamDecls(parser, inputParamDecls))
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

/// Parse a type in a KGEN context, handling sugar like "dtype" for
/// "!kgen.dtype" etc.
OptionalParseResult KGEN::parseOptionalKGENType(AsmParser &parser, Type &type) {
  // Check for sugared types before parsing standard ones. We need to check for
  // each keyword individually, since builtin types are also keywords.
  auto *dialect = parser.getContext()->getLoadedDialect<KGENDialect>();
  assert(dialect && "cannot parse KGEN type without KGEN dialect");
  for (auto &[keyword, parseFn] : dialect->typeParseFns) {
    if (parser.parseOptionalKeyword(keyword))
      continue;
    type = parseFn(parser);
    return failure(!type);
  }

  // Parse symbol references as decl reference types.
  SymbolRefAttr symbol;
  OptionalParseResult result = parser.parseOptionalAttribute(symbol);
  if (result.has_value()) {
    if (failed(*result))
      return failure();
    ParamBindArrayAttr values;
    if (parseOptionalParamBindSpec(parser, values))
      return failure();
    type = DeclRefType::get(symbol, values);
    return LogicalResult::success();
  }

  // Try to parse an optional signature. Signatures can begin with `<` or `(`.
  {
    SignatureType signature;
    OptionalParseResult result = parseOptionalSignature(parser, signature);
    if (result.has_value()) {
      if (failed(*result))
        return failure();
      type = signature;
      return LogicalResult::success();
    }
  }

  return parser.parseOptionalType(type);
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
  // Handle other special cases for parameters here.  These each are sugar for a
  // kgen type.
  auto *dialect = type.getContext()->getLoadedDialect<KGENDialect>();
  assert(dialect && "cannot print KGEN type without KGEN dialect");
  if (auto it = dialect->typePrintFns.find(type.getTypeID());
      it != dialect->typePrintFns.end()) {
    it->second(p, type);
  } else if (auto ref = dyn_cast<DeclRefType>(type)) {
    // Use the alias printer if suitable.
    if (ref.getAliasName()) {
      p.printType(ref);
    } else {
      p << ref.getSymbol();
      printOptionalParamBindSpec(p, ref.getParamValues());
    }
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

/// Print a parameter value that is known to have `type` type.
void KGEN::printTypeParamValue(AsmPrinter &p, Attribute value) {
  printParamValue(p, cast<TypedAttr>(value));
}

/// Parse a parameter value that is known to have `type` type.
ParseResult KGEN::parseTypeParamValue(AsmParser &p, TypedAttr &value) {
  return parseParamValue(p, value, MLIRTypeType::get(p.getContext()));
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

/// Print an attribute value that is known to have index type.
void KGEN::printIndexParamValue(AsmPrinter &p, Operation *op, Attribute value) {
  printParamValue(p, cast<TypedAttr>(value));
}

void KGEN::printIndexParamValue(AsmPrinter &p, Attribute value) {
  printParamValue(p, cast<TypedAttr>(value));
}

/// Parse a parameter value that is known to be an index type.
ParseResult KGEN::parseIndexParamValue(AsmParser &p, TypedAttr &value) {
  return parseParamValue(p, value, p.getBuilder().getIndexType());
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
ParseResult KGEN::parseParamDecls(AsmParser &p, ParamDeclArrayAttr &result) {
  // Parse each of the decls.
  SmallVector<ParamDeclAttr> decls;

  // Check to see if we have the () syntax instead of arguments.
  if (succeeded(p.parseOptionalLParen())) {
    if (p.parseRParen())
      return failure();
  } else {
    if (p.parseCommaSeparatedList([&]() {
          return parseParamDecl(p, decls.emplace_back(ParamDeclAttr()));
        }))
      return failure();
  }

  result = ParamDeclArrayAttr::get(p.getContext(), decls);
  return success();
}

/// Print a comma separated parameter declaration list.
void KGEN::printParamDecls(AsmPrinter &p, ArrayRef<ParamDeclAttr> decls) {
  if (decls.empty()) {
    p << "()";
  } else {
    llvm::interleaveComma(decls, p,
                          [&](ParamDeclAttr decl) { printParamDecl(p, decl); });
  }
}

/// Parse a parameter spec if present, including input and result parameter
/// declarations.
/// parameter-decl   ::= identifier (`:` type)?
/// parameter-list   ::= parameter-decl (`,` parameter-decl)* | `(` `)`
/// parameter-spec   ::= `<` parameter-list (`->` parameter-list)? `>`
ParseResult
KGEN::parseOptionalParameterSpec(AsmParser &parser,
                                 ParamDeclArrayAttr &inputParamDecls,
                                 ParamDeclArrayAttr &resultParamDecls) {
  // If there is no parameter list, or if it is empty, we're done.
  if (failed(parser.parseOptionalLess()) ||
      succeeded(parser.parseOptionalGreater())) {
    inputParamDecls = ParamDeclArrayAttr::get(parser.getContext(), {});
    resultParamDecls = ParamDeclArrayAttr::get(parser.getContext(), {});
  } else {
    if (parseParameterSpec(parser, inputParamDecls, resultParamDecls) ||
        parser.parseGreater())
      return failure();
  }
  return success();
}

/// Parse a parameter specification as a SignatureType.
ParseResult
KGEN::parseOptionalParameterSpec(AsmParser &parser,
                                 ParamDeclArrayAttr &inputParamDecls) {
  ParamDeclArrayAttr resultParams;
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parseOptionalParameterSpec(parser, inputParamDecls, resultParams))
    return failure();
  if (!resultParams.empty())
    return parser.emitError(loc, "expected no result parameters");
  return success();
}

/// Print a parameter list for a generator, func or interface.
void KGEN::printOptionalParameterSpec(AsmPrinter &p,
                                      ArrayRef<ParamDeclAttr> inputParamDecls,
                                      ArrayRef<ParamDeclAttr> resultParams) {
  if (inputParamDecls.empty() && resultParams.empty())
    return;

  p << '<';
  printParamDecls(p, inputParamDecls);

  if (!resultParams.empty()) {
    p << " -> ";
    llvm::interleaveComma(resultParams, p, [&](ParamDeclAttr param) {
      printParamDecl(p, param);
    });
  }
  p << '>';
}

void KGEN::printOptionalParameterSpec(AsmPrinter &p, Operation *op,
                                      ArrayRef<ParamDeclAttr> inputParamDecls) {
  printOptionalParameterSpec(p, inputParamDecls, {});
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
  case (uint32_t)POC::BuildInfoGetField:
    // Parse the BuildInfoGetField -- the first operand is a BuildInfoType, the
    // second a StringType.
    if (parseParamValue(p, operands.emplace_back(),
                        BuildInfoType::get(p.getContext())) ||
        p.parseComma() ||
        parseParamValue(p, operands.emplace_back(),
                        StringType::get(p.getContext())))
      return failure();
    return success();
  case (uint32_t)POC::GetSizeOf:
  case (uint32_t)POC::GetAlignOf:
    if (parseParamValue(p, operands.emplace_back(),
                        MLIRTypeType::get(p.getContext())) ||
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
    for (Type type : sig.getValueInputs())
      if (p.parseComma() || parseParamValue(p, operands.emplace_back(), type))
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
    if (sig.getNumInputs() < 1)
      return p.emitError(
          p.getCurrentLocation(),
          "'apply_result_slot' callee must have at least one result");
    // Parse each operand besides the result slot.
    for (Type type : llvm::drop_begin(sig.getValueInputs()))
      if (p.parseComma() || parseParamValue(p, operands.emplace_back(), type))
        return failure();
    return success();
  }
  case (uint32_t)POC::GetAllImpls: {
    auto varTy = dyn_cast_or_null<VariadicType>(type);
    if (!varTy)
      return p.emitError(p.getCurrentLocation(),
                         "expected a variadic type for 'get_all_impls'");
    auto sigTy = dyn_cast<SignatureType>(varTy.getElementAsType());
    if (!sigTy)
      return p.emitError(
          p.getCurrentLocation(),
          "expected a variadic of signatures type for 'get_all_impls'");
    if (parseParamValue(p, operands.emplace_back(), sigTy))
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
    // Try to parse *(0,0) as an index reference.
    size_t depth, index;
    if (succeeded(p.parseOptionalLParen())) {
      if (p.parseInteger(depth) || p.parseComma() || p.parseInteger(index) ||
          p.parseRParen())
        return failure();
      bool isResult = succeeded(p.parseOptionalStar());
      value = ParamIndexRefAttr::get(depth, isResult, index, type);
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
    value = UnknownAttr::get(type);
    return success();
  }

  // Barewords / MLIR keywords are implicitly parameter declaration references
  // or the start of a expression in function form.
  StringRef keyword;
  if (succeeded(p.parseOptionalKeyword(&keyword))) {
    // Check to see if we're parsing a dtype name like 'f32'.
    if (type.isa<DTypeType>()) {
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
  // If this is a comparison and the elements are not index type, print the
  // type explicitly.
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
  }
}

/// Convert a parameter value to a string when in a context that knows it is
/// dealing with a parameter specifically.  This utilize syntactic shortcuts to
/// make the printed syntax easier to grok.
void KGEN::printParamValue(AsmPrinter &p, TypedAttr value, Type type) {
  // If the attribute's type provides a pretty printing hook, try to use it.
  if (auto typeItf = dyn_cast<ParameterTypeInterface>(value.getType()))
    if (succeeded(typeItf.printValue(p, value)))
      return;

  if (isa<UnknownAttr>(value)) {
    p << '?';
    return;
  }

  if (auto declRef = dyn_cast<ParamDeclRefAttr>(value)) {
    printParamName(p, declRef.getName(), isa<MLIRTypeType>(value.getType()));
    return;
  }
  if (auto indexRef = dyn_cast<ParamIndexRefAttr>(value)) {
    p << "*(" << indexRef.getDepth() << ',' << indexRef.getIndex() << ")";
    if (indexRef.getIsResult())
      p << '*';
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
        expr.getNumOperands() == 2 && expr.getOperand(1).isa<IntegerAttr>()) {
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

//===----------------------------------------------------------------------===//
// Logic shared between funcs, generators, and generator interfaces
//===----------------------------------------------------------------------===//

/// Parse an argument or type list with optional metadata. This is an optional
/// parse, which allows the KGEN type parser to check if it is parsing a
/// signature.
template <bool optionalResultList>
static OptionalParseResult parseOptionalSignatureValues(
    AsmParser &p,
    function_ref<FailureOr<std::pair<StringAttr, Type>>()> parseNameElt,
    FunctionType &values,
    SmallVectorImpl<ValueInputConvention> &inputConventions, FnEffects &effects,
    FnMetadataAttr &metadata) {
  SmallVector<StringAttr> argNames;
  SmallVector<TypedAttr> defaults;
  SmallVector<Type> argTypes, resTypes;

  // Parse an element list with input effects and default values.
  auto parseArg = [&]() -> ParseResult {
    FailureOr<std::pair<StringAttr, Type>> nameElt = parseNameElt();
    if (failed(nameElt))
      return failure();
    argNames.push_back(nameElt->first);
    argTypes.push_back(nameElt->second);

    StringRef effectStr;
    llvm::SMLoc loc = p.getCurrentLocation();
    // Parse an optional input convention specifier.
    auto convention = ValueInputConvention::OwnedInReg;
    if (succeeded(p.parseOptionalKeyword(&effectStr))) {
      if (std::optional<ValueInputConvention> conv =
              symbolizeValueInputConvention(effectStr)) {
        convention = *conv;
      } else {
        return p.emitError(loc, "expected a valid input convention");
      }
    }
    inputConventions.push_back(convention);

    if (succeeded(p.parseOptionalEqual())) {
      TypedAttr value;
      if (parseParamValue(p, value, nameElt->second))
        return failure();
      defaults.push_back(value);
    }
    return success();
  };

  llvm::SMLoc loc = p.getCurrentLocation();
  if (failed(p.parseOptionalLParen()))
    return std::nullopt;
  if (failed(p.parseOptionalRParen())) {
    if (p.parseCommaSeparatedList(parseArg) || p.parseRParen())
      return failure();
  }

  // Parse the function effects. Check for each case to disambiguate the syntax
  // for interfaces.
  auto effectsValue = impl::FnEffects::None;
  StringRef kw;
  while (succeeded(p.parseOptionalKeyword(
      &kw, {"throws", "async", "vararg", "packvararg", "kwvararg",
            "param_vararg", "capturing", "ownedresult", "escaping"}))) {
    effectsValue |= *impl::symbolizeFnEffects(kw);

    // No vertical bar? We're done. It's not a parse error, but it does mean we
    // can't specify more effects.
    if (failed(p.parseOptionalVerticalBar()))
      break;
  }

  if (optionalResultList ? p.parseOptionalArrowTypeList(resTypes)
                         : p.parseArrowTypeList(resTypes))
    return failure();
  auto emitError = [&] { return p.emitError(loc); };

  // FIXME: Force C++ to select the derived class getter, not the storage
  // uniquer getter, which won't compile outside of `KGENAttrs.cpp`.
  using GetCheckedT = FnMetadataAttr (*)(
      function_ref<InFlightDiagnostic()>, MLIRContext *, ArrayRef<StringAttr>,
      ArrayRef<TypedAttr>, ArrayRef<TypedAttr>);
  effects = FnEffects(effectsValue);
  metadata = ((GetCheckedT)&FnMetadataAttr::getChecked)(
      emitError, p.getContext(), argNames, defaults, ArrayRef<TypedAttr>{});
  if (!metadata)
    return failure();
  values = p.getBuilder().getFunctionType(argTypes, resTypes);
  return mlir::success();
}

template <bool optionalResultList>
static ParseResult parseSignatureValuesElt(
    AsmParser &p,
    function_ref<FailureOr<std::pair<StringAttr, Type>>()> parseElt,
    FunctionType &values,
    SmallVectorImpl<ValueInputConvention> &inputConventions, FnEffects &effects,
    FnMetadataAttr &metadata) {
  OptionalParseResult result = parseOptionalSignatureValues<optionalResultList>(
      p, parseElt, values, inputConventions, effects, metadata);
  if (result.has_value())
    return *result;
  return p.emitError(p.getCurrentLocation(), "expected '(' to begin signature");
}

/// Print an argument or type list with optional metadata.
template <bool optionalResultList>
static void
printSignatureValuesElt(AsmPrinter &p, function_ref<void(unsigned)> printElt,
                        FunctionType functionType, SignatureType signature) {
  p << '(';
  FnMetadataAttr metadata = signature.getMetadata();
  ArrayRef<TypedAttr> defaults = metadata.getDefaultArguments();
  llvm::interleaveComma(
      llvm::seq<unsigned>(0, signature.getInputConventions().size()), p,
      [&](unsigned i) {
        printElt(i);
        ValueInputConvention conv = signature.getInputConventions()[i];
        if (conv != ValueInputConvention::OwnedInReg)
          p << ' ' << stringifyValueInputConvention(conv);

        // If a default argument value has been provided for the argument at
        // this index, print an `=`, followed by the value.
        size_t defaultIndex =
            signature.getInputConventions().size() - defaults.size();
        if (i >= defaultIndex) {
          p << " = ";
          printParamValue(p, defaults[i - defaultIndex]);
        }
      });
  p << ')';

  // Print the function effects.
  impl::FnEffects effects = signature.getFnEffects().getImpl();
  if (effects != impl::FnEffects::None)
    p << ' ' << impl::stringifyFnEffects(effects);

  if constexpr (optionalResultList)
    p.printOptionalArrowTypeList(functionType.getResults());
  else
    p.printArrowTypeList(functionType.getResults());
}

ParseResult KGEN::parseFunctionSignature(
    OpAsmParser &p, SmallVectorImpl<OpAsmParser::Argument> &args,
    ParamDeclArrayAttr &inputParams, ParamDeclArrayAttr &resultParams,
    FunctionType &functionType, SignatureType &signature, bool parseNames) {
  llvm::SMLoc loc = p.getCurrentLocation();
  if (parseOptionalParameterSpec(p, inputParams, resultParams))
    return failure();

  // Parse the argument list with input effects.
  auto parseElt = [&]() -> FailureOr<std::pair<StringAttr, Type>> {
    OpAsmParser::Argument arg;
    OptionalParseResult res = p.parseOptionalArgument(arg, /*allowType=*/true);
    if (res.has_value()) {
      if (failed(*res))
        return failure();
    } else if (p.parseType(arg.type)) {
      return failure();
    }
    args.emplace_back(arg);

    if (!parseNames)
      return std::make_pair(StringAttr::get(p.getContext()), arg.type);

    StringRef argName = arg.ssaName.name;
    if (!argName.empty())
      argName = argName.drop_front();
    return std::make_pair(StringAttr::get(p.getContext(), argName), arg.type);
  };

  SmallVector<ValueInputConvention> inputConventions;
  FnEffects effects;
  FnMetadataAttr metadata;
  if (failed(parseSignatureValuesElt</*optionalResultList=*/true>(
          p, parseElt, functionType, inputConventions, effects, metadata)))
    return failure();

  signature = IndexRefRemapper::remapToSignature(
      inputParams, resultParams, functionType, inputConventions, effects,
      metadata, [&] { return p.emitError(loc); });
  return success(!!signature);
}

void KGEN::printFunctionSignature(OpAsmPrinter &p,
                                  function_ref<void(unsigned i)> printElt,
                                  ArrayRef<ParamDeclAttr> inputParams,
                                  ArrayRef<ParamDeclAttr> resultParams,
                                  FunctionType functionType,
                                  SignatureType signature) {
  printOptionalParameterSpec(p, inputParams, resultParams);
  printSignatureValuesElt</*optionalResultList=*/true>(p, printElt,
                                                       functionType, signature);
}

void KGEN::printFunctionSignature(OpAsmPrinter &p, Region *region,
                                  ArrayRef<ParamDeclAttr> inputParams,
                                  ArrayRef<ParamDeclAttr> resultParams,
                                  FunctionType functionType,
                                  SignatureType signature) {
  // Print the function arguments.
  ArrayRef<StringAttr> argNames = signature.getArgNames();
  auto printElt = [&](unsigned i) {
    if (!region) {
      if (argNames[i].size())
        p << "%" + argNames[i].getValue() + ": ";
      p << functionType.getInput(i);
    } else {
      p.printRegionArgument(region->getArgument(i));
    }
  };

  printFunctionSignature(p, printElt, inputParams, resultParams, functionType,
                         signature);
}

OptionalParseResult KGEN::parseOptionalSignature(AsmParser &p,
                                                 Type &signature) {
  llvm::SMLoc loc = p.getCurrentLocation();
  SmallVector<Type> inputParamTypes, resultParamTypes;
  if (succeeded(p.parseOptionalLess())) {
    if (p.parseOptionalGreater()) {
      if (succeeded(p.parseOptionalLSquare())) {
        if (p.parseRSquare())
          return failure();
      } else if (p.parseCommaSeparatedList([&] {
                   return parseKGENType(p, inputParamTypes.emplace_back());
                 })) {
        return failure();
      }
      if (succeeded(p.parseOptionalArrow())) {
        if (p.parseCommaSeparatedList([&] {
              return parseKGENType(p, resultParamTypes.emplace_back());
            }))
          return failure();
      }
      if (p.parseGreater())
        return failure();
    }
  }

  auto parseElt = [&]() -> FailureOr<std::pair<StringAttr, Type>> {
    std::string argName;
    if (succeeded(p.parseOptionalString(&argName))) {
      if (failed(p.parseColon()))
        return failure();
    } else {
      argName = "";
    }
    Type type;
    if (failed(p.parseType(type)))
      return failure();
    return std::make_pair(StringAttr::get(p.getContext(), argName), type);
  };
  FunctionType functionType;
  SmallVector<ValueInputConvention> inputConventions;
  FnEffects effects;
  FnMetadataAttr metadata;
  OptionalParseResult result =
      parseOptionalSignatureValues</*optionalResultList=*/false>(
          p, parseElt, functionType, inputConventions, effects, metadata);
  if (result.has_value() && succeeded(*result)) {
    signature = SignatureType::getChecked(
        [&] { return p.emitError(loc); }, functionType,
        TypeArrayAttr::get(p.getContext(), inputParamTypes),
        TypeArrayAttr::get(p.getContext(), resultParamTypes), inputConventions,
        effects, metadata);
    if (!signature)
      return failure();
  }
  return result;
}

ParseResult KGEN::parseSignature(AsmParser &p, Type &signature) {
  OptionalParseResult result = parseOptionalSignature(p, signature);
  if (result.has_value())
    return *result;
  return p.emitError(p.getCurrentLocation(),
                     "expected '<' or '(' to begin a signature");
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
  if (!signature.getInputParamTypes().empty() ||
      !signature.getResultParamTypes().empty()) {
    p << '<';
    if (signature.getInputParamTypes().empty())
      p << "[]";
    llvm::interleaveComma(signature.getInputParamTypes(), p,
                          [&](Type type) { printKGENType(p, type); });
    if (!signature.getResultParamTypes().empty()) {
      p << " -> ";
      llvm::interleaveComma(signature.getResultParamTypes(), p,
                            [&](Type type) { printKGENType(p, type); });
    }
    p << '>';
  }

  auto printElt = [&](unsigned i) {
    StringAttr argName = signature.getArgName(i);
    if (argName.size())
      p << "\"" << argName.strref() << "\": ";
    p << signature.getValueInputs()[i];
  };
  printSignatureValuesElt</*optionalResultList=*/false>(
      p, printElt, signature.getValues(), signature);
}

void KGEN::printSignature(AsmPrinter &p, Operation *op, TypeAttr signature) {
  printSignature(p, cast<SignatureType>(signature.getValue()));
}

ParseResult KGEN::parseSignatureValues(AsmParser &p,
                                       ParamDeclArrayAttr resultParamDecls,
                                       FunctionType &functionType,
                                       SignatureType &signature) {
  llvm::SMLoc loc = p.getCurrentLocation();
  auto parseElt = [&]() -> FailureOr<std::pair<StringAttr, Type>> {
    std::string argName;
    if (succeeded(p.parseOptionalString(&argName))) {
      if (failed(p.parseColon()))
        return failure();
    } else {
      argName = "";
    }
    Type type;
    if (failed(p.parseType(type)))
      return failure();
    return std::make_pair(StringAttr::get(p.getContext(), argName), type);
  };
  SmallVector<ValueInputConvention> inputConventions;
  FnEffects effects;
  FnMetadataAttr metadata;
  if (parseSignatureValuesElt</*optionalResultList=*/false>(
          p, parseElt, functionType, inputConventions, effects, metadata))
    return failure();
  signature = IndexRefRemapper::remapToSignature(
      {}, resultParamDecls, functionType, inputConventions, effects, metadata,
      [&] { return p.emitError(loc); });
  return success(!!signature);
}

void KGEN::printSignatureValues(AsmPrinter &p, FunctionType functionType,
                                SignatureType signature) {
  auto printElt = [&](unsigned i) {
    StringAttr argName = signature.getArgName(i);
    if (argName.size())
      p << "\"" << argName.strref() << "\": ";
    p << functionType.getInput(i);
  };
  printSignatureValuesElt</*optionalResultList=*/false>(
      p, printElt, functionType, signature);
}

/// Parse a constraint specification if
/// present. constraints-spec ::=
///    `constraints` `<` attribute-value
///    (`,` attribute-value)? `>`
ParseResult KGEN::parseOptionalConstraints(OpAsmParser &parser,
                                           ConstraintArrayAttr &result) {
  SmallVector<ConstraintAttr> constraints;

  if (succeeded(parser.parseOptionalKeyword("constraints"))) {
    auto parseConstraint = [&]() -> ParseResult {
      ConstraintAttr constraint;
      if (parser.parseCustomAttributeWithFallback(constraint))
        return failure();
      constraints.push_back(constraint);
      return success();
    };

    if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::LessGreater,
                                       parseConstraint))
      return failure();
  }
  result = ConstraintArrayAttr::get(parser.getContext(), constraints);
  return success();
}

/// Print a constraint list for a generator or interface.
void KGEN::printOptionalConstraints(OpAsmPrinter &p, Operation *op,
                                    ArrayRef<ConstraintAttr> constraints) {
  if (constraints.empty())
    return;

  p.printNewline();
  p << "  constraints <";
  llvm::interleaveComma(constraints, p, [&](ConstraintAttr constraint) {
    if (constraints.size() > 1) {
      p.printNewline();
      p << "    ";
    }
    constraint.print(p);
  });
  p << ">";
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
  }
  exportKind = ExportKindAttr::get(p.getContext(), value);
  return success();
}

void KGEN::printSymbolExport(AsmPrinter &p, Operation *op,
                             ExportKindAttr exportKind) {
  if (exportKind.getValue() != ExportKind::NotExported) {
    p << " export";
    if (exportKind.getValue() == ExportKind::CExported)
      p << " C";
  }
}

/// Parse either a kgen.generator or kgen.func declaration, depending on what
/// `isGenerator` is set to.
ParseResult KGEN::parseGeneratorOrFunc(OpAsmParser &parser,
                                       OperationState &result,
                                       GeneratorOrFuncKind opKind) {
  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<Type> resultTypes;

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  SignatureType signature;
  FunctionType functionType;
  ParamDeclArrayAttr inputParams, resultParams;
  llvm::SMLoc sigLoc = parser.getCurrentLocation();
  if (parseFunctionSignature(parser, entryArgs, inputParams, resultParams,
                             functionType, signature))
    return failure();

  InlineLevelAttr inlineLevel;
  if (parseOptionalInline(parser, inlineLevel))
    return failure();
  result.addAttribute("inlineLevel", inlineLevel);

  // Funcs cannot have constraint specifications.
  if (opKind != GeneratorOrFuncKind::func) {
    ConstraintArrayAttr constraints;
    if (parseOptionalConstraints(parser, constraints))
      return failure();
    result.addAttribute("constraints", constraints);
  }

  DecoratorsAttr decorators;
  if (parseOptionalDecorators(parser, decorators))
    return failure();
  result.addAttribute("decorators", decorators);

  if (opKind == GeneratorOrFuncKind::generator) {
    result.addAttribute("functionType", TypeAttr::get(functionType));
    result.addAttribute("inputParams", inputParams);
    result.addAttribute("resultParams", resultParams);
  } else {
    // Concrete functions are not allowed to have input parameter lists.
    if (!inputParams.empty() || !resultParams.empty())
      return parser.emitError(
          sigLoc, "concrete functions cannot have input or result parameters");
  }
  result.addAttribute("signature", TypeAttr::get(signature));

  // If function attributes are present, parse them.
  NamedAttrList parsedAttributes;
  llvm::SMLoc attributeDictLocation = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(parsedAttributes))
    return failure();

  // Disallow attributes that are inferred from elsewhere in the attribute
  // dictionary.
  for (StringRef disallowed : GeneratorOp::getAttributeNames()) {
    if (parsedAttributes.get(disallowed))
      return parser.emitError(attributeDictLocation, "'")
             << disallowed
             << "' is an inferred attribute and should not be specified in the "
                "explicit attribute dictionary";
  }
  result.attributes.append(parsedAttributes);

  // Parse the required function body.
  Region *region = result.addRegion();

  return parser.parseRegion(*region, entryArgs, /*enableNameShadowing=*/true);
}

void KGEN::printGeneratorOrFunc(OpAsmPrinter &p, FuncInterface op) {
  auto func = cast<mlir::FunctionOpInterface>(*op);

  // Print the operation and the function name.
  StringRef funcName = func.getName();
  p << ' ';

  p.printSymbolName(funcName);
  auto decl = cast<DeclInterface>(*op);
  printFunctionSignature(p, &func.getFunctionBody(), decl.getInputParams(),
                         decl.getResultParams(), op.getFunctionType(),
                         op.getSignature());

  printOptionalInline(p, op.getInlineLevel());

  SmallVector<StringRef> ignoredAttrNames(
      GeneratorOp::getAttributeNames().begin(),
      GeneratorOp::getAttributeNames().end());

  // Print out function attributes, if present.
  SmallVector<StringRef, 8> ignoredAttrs = {SymbolTable::getSymbolAttrName()};
  ignoredAttrs.append(ignoredAttrNames.begin(), ignoredAttrNames.end());
  p.printOptionalAttrDictWithKeyword(op->getAttrs(), ignoredAttrs);

  printOptionalConstraints(p, op, cast<DeclInterface>(*op).getConstraints());
  printOptionalDecorators(p, op, op.getDecorators());

  p << ' ';
  if (!func.isExternal())
    p.printRegion(func.getFunctionBody(), /*printEntryBlockArgs=*/false);
}

/// Parse a parameter binding list if present.
///
///   parameter-bind   ::= identifier (`:` type)? `=` attribute-value
///   parameter-bind-list ::= parameter-bind (`,` parameter-bind)* | `(` `)`
ParseResult KGEN::parseParamBinds(AsmParser &p,
                                  ParamBindArrayAttr &paramBinds) {
  // Check to see if we have the () syntax instead of arguments.
  if (succeeded(p.parseOptionalLParen())) {
    if (p.parseRParen())
      return failure();
    paramBinds = ParamBindArrayAttr::get(p.getContext(), {});
    return success();
  }

  SmallVector<ParamBindAttr> values;
  auto parseParamBind = [&]() -> ParseResult {
    StringAttr name;
    Type type;
    TypedAttr value;

    if (parseParamName(p, name) || parseColonTypeOrIndex(p, type) ||
        p.parseEqual() || parseParamValue(p, value, type))
      return failure();
    values.push_back(ParamBindAttr::get(name, value));
    return success();
  };

  if (p.parseCommaSeparatedList(OpAsmParser::Delimiter::None, parseParamBind))
    return failure();

  paramBinds = ParamBindArrayAttr::get(p.getContext(), values);
  return success();
}

void KGEN::printParamBinds(AsmPrinter &p, ArrayRef<ParamBindAttr> paramBinds) {
  if (paramBinds.empty()) {
    p << "()";
  } else {
    llvm::interleaveComma(paramBinds, p, [&](ParamBindAttr bind) {
      printParamName(p, bind.getName());
      printColonTypeOrIndex(p, bind.getType());
      p << " = ";
      printParamValue(p, bind.getValue());
    });
  }
}

/// Parse a list of parameter bindings without result parameters in <>'s
ParseResult KGEN::parseOptionalParamBindSpec(AsmParser &p,
                                             ParamBindArrayAttr &paramValues) {
  // If there are no parameter declarations, return an empty array.
  if (p.parseOptionalLess()) {
    paramValues = ParamBindArrayAttr::get(p.getContext(), {});
    return success();
  }

  if (parseParamBinds(p, paramValues))
    return failure();
  return p.parseGreater();
}

void KGEN::printOptionalParamBindSpec(AsmPrinter &p,
                                      ParamBindArrayAttr paramValues) {
  if (paramValues.empty())
    return;
  p << '<';
  printParamBinds(p, paramValues);
  p << '>';
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

ParseResult KGEN::parseParametricCallee(OpAsmParser &p, TypedAttr &callee,
                                        ParamDeclArrayAttr &paramDecls) {
  Type type;
  llvm::SMLoc loc = p.getCurrentLocation();
  if (p.parseLSquare() || parseKGENType(p, type) || p.parseColon() ||
      parseParamValue(p, callee, type) || p.parseRSquare())
    return failure();
  if (succeeded(p.parseOptionalLess())) {
    if (p.parseLParen() || p.parseRParen() || p.parseArrow() ||
        parseParamDecls(p, paramDecls) || p.parseGreater())
      return failure();
  } else {
    paramDecls = ParamDeclArrayAttr::get(p.getContext(), {});
  }

  if (!isa<ParamRefType, SignatureType>(callee.getType()))
    return p.emitError(loc, "callee parameter type must be a signature type");
  return success();
}

void KGEN::printParametricCallee(OpAsmPrinter &p, Operation *, TypedAttr callee,
                                 ParamDeclArrayAttr paramDecls) {
  p << "[";
  printKGENType(p, callee.getType());
  p << ": ";
  printParamValue(p, callee);
  p << "]";
  if (!paramDecls.empty()) {
    p << "<() -> ";
    printParamDecls(p, paramDecls);
    p << '>';
  }
}

/// Parse an address space parameter if present.
void KGEN::printOptionalAddressSpaceParamValue(AsmPrinter &p, Operation *op,
                                               TypedAttr addressSpace) {
  if (!addressSpace)
    return;

  // If the address space is an integer and zero, then we can skip since that's
  // the default address space.
  if (auto addressSpaceInt = dyn_cast<IntegerAttr>(addressSpace);
      addressSpaceInt && addressSpaceInt.getValue().isZero())
    return;

  p << " address_space ";
  printParamValue(p, addressSpace);
  p << " ";
}

/// Parse a parameter value that is known to be an address space type.
ParseResult KGEN::parseOptionalAddressSpaceParamValue(AsmParser &p,
                                                      TypedAttr &result) {
  if (p.parseOptionalKeyword("address_space")) {
    // The default address space is 0.
    result = p.getBuilder().getIndexAttr(0);
    return success();
  }

  return parseIndexParamValue(p, result);
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
  TimeTraceScope<> traceScope("verifyDeclSignaturesMatch");

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
      verifyMatchingLists(lhsSig.getInputConventions(),
                          rhsSig.getInputConventions(), lhsName, lhsLoc,
                          rhsName, rhsLoc, "argument", "convention"))
    return failure();

  if (lhsSig.getFnEffects() != rhsSig.getFnEffects()) {
    auto diag = emitError(lhsLoc, lhsName)
                << " function effects are " << lhsSig.getFnEffects() << " but @"
                << rhsName << " expected " << rhsSig.getFnEffects();
    if (lhsLoc != rhsLoc)
      diag.attachNote(rhsLoc) << rhsName << " declared here";
    return failure();
  }

  // Check the metadata matches up: input argument conventions and function
  // effects ought to match.
  if (lhsSig.getMetadata() != rhsSig.getMetadata()) {
    if (rhsSig.getMetadata() == lhsSig.getMetadata())
      return success();
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
                            ArrayRef<ParamDeclAttr> originatorParamDecls,
                            Location originatorLoc, StringRef targetName,
                            ArrayRef<ParamDeclAttr> targetParamDecls,
                            Location targetLoc) {
  using llvm::map_range;
  auto getType = [](auto attr) -> Type { return attr.getType(); };
  auto getName = [](auto attr) -> StringAttr { return attr.getName(); };

  if (verifyMatchingLists(map_range(originatorParamDecls, getName),
                          map_range(targetParamDecls, getName), originatorName,
                          originatorLoc, targetName, targetLoc, paramKind,
                          "name") ||
      verifyMatchingLists(map_range(originatorParamDecls, getType),
                          map_range(targetParamDecls, getType), originatorName,
                          originatorLoc, targetName, targetLoc, paramKind,
                          "type"))
    return failure();
  return success();
}

LogicalResult
KGEN::verifyParamDeclsMatch(StringRef paramKind, StringRef originatorName,
                            ArrayRef<ParamBindAttr> binds,
                            Location originatorLoc, StringRef targetName,
                            ArrayRef<ParamDeclAttr> decls, Location targetLoc) {
  using llvm::map_range;
  auto getType = [](auto attr) -> Type { return attr.getType(); };
  auto getName = [](auto attr) -> StringAttr { return attr.getName(); };

  if (verifyMatchingLists(map_range(binds, getName), map_range(decls, getName),
                          originatorName, originatorLoc, targetName, targetLoc,
                          paramKind, "name") ||
      verifyMatchingLists(map_range(binds, getType), map_range(decls, getType),
                          originatorName, originatorLoc, targetName, targetLoc,
                          paramKind, "type"))
    return failure();
  return success();
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
    Type actualTy = resultParams[i].cast<TypedAttr>().getType();
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

llvm::MapVector<StringAttr, ExportedSymbol>
KGEN::getExportedSymbols(ModuleOp module) {
  llvm::MapVector<StringAttr, ExportedSymbol> exportedSymbols;
  for (auto op : module.getOps<ExportInterface>()) {
    if (op.isExported())
      exportedSymbols.insert(
          {op.getLinkageNameAttr(), ExportedSymbol(op.isCExported())});
  }
  return exportedSymbols;
}
