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
#include "Cache/CacheDialect/CacheOps.h"
#include "KGEN/KGENDialect/KGENDType.h"
#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENTypeInterfaces.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/ML/DType.h"
#include "Support/STLExtras.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// StreamAsmPrinter
//===----------------------------------------------------------------------===//

/// Returns true if the given string can be represented as a bare identifier
/// compatible with the MLIR lexer.
static bool isBareIdentifier(StringRef name) {
  if (name.empty() || (!isalpha(name[0]) && name[0] != '_'))
    return false;
  return llvm::all_of(name.drop_front(), [](unsigned char c) {
    return isalnum(c) || c == '_' || c == '$' || c == '.';
  });
}

namespace {
/// This is an AsmPrinter implementation that just outputs to an external output
/// stream.
class StreamAsmPrinter : public AsmPrinter {
public:
  explicit StreamAsmPrinter(raw_ostream &os) : os(os) {}

  /// Implement all the virtual hooks.

  raw_ostream &getStream() const override { return os; }

  /// Trivial hooks

  void printType(Type type) override { os << type; }
  void printAttribute(Attribute attr) override { os << attr; }
  void printAttributeWithoutType(Attribute attr) override {
    attr.print(os, /*elideType=*/true);
  }
  LogicalResult printAlias(Attribute attr) override { return failure(); }
  LogicalResult printAlias(Type type) override { return failure(); }

  /// Less trivial hooks.

  void printKeywordOrString(StringRef keyword) override {
    if (isBareIdentifier(keyword)) {
      os << keyword;
      return;
    }
    os << "\"";
    printEscapedString(keyword, os);
    os << '"';
  }
  void printSymbolName(StringRef symbolRef) override {
    os << '@';
    printKeywordOrString(symbolRef);
  }
  void
  printResourceHandle(const mlir::AsmDialectResourceHandle &resource) override {
    auto *interface = cast<OpAsmDialectInterface>(resource.getDialect());
    os << interface->getResourceKey(resource);
  }

  /// Print floats like MLIR does.
  void printFloat(const APFloat &value) override {
    if (!value.isInfinity() && !value.isNaN()) {
      SmallString<128> strValue;
      value.toString(strValue, /*FormatPrecision=*/6, /*FormatMaxPadding=*/0,
                     /*TruncateZero=*/false);
      if (APFloat(value.getSemantics(), strValue).bitwiseIsEqual(value)) {
        os << strValue;
        return;
      }
      strValue.clear();
      value.toString(strValue);
      if (strValue.str().contains('.')) {
        os << strValue;
        return;
      }
    }
    SmallVector<char, 16> str;
    APInt apInt = value.bitcastToAPInt();
    apInt.toString(str, /*Radix=*/16, /*Signed=*/false,
                   /*formatAsCLiteral=*/true);
    os << str;
  }

private:
  /// The stream to output to.
  raw_ostream &os;
};
} // namespace

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

/// Parse a non-empty parameter list without the surrounding braces.
static ParseResult parseParameterSpec(AsmParser &parser,
                                      ParamDeclArrayAttr &inputParamDecls,
                                      TypeArrayAttr &resultParamTypesAttr) {
  // Parse the input list.
  if (parseParamDecls(parser, inputParamDecls))
    return failure();

  // Check to see if we have results and parse them if so.
  SmallVector<Type> resultParamTypes;
  if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseCommaSeparatedList([&]() -> ParseResult {
          return parseKGENType(parser, resultParamTypes.emplace_back(Type()));
        }))
      return failure();
  }
  resultParamTypesAttr =
      TypeArrayAttr::get(parser.getContext(), resultParamTypes);
  return success();
}

/// Parse a type in a KGEN context, handling sugar like "dtype" for
/// "!kgen.dtype" etc.
ParseResult KGEN::parseKGENType(AsmParser &parser, Type &type) {
  llvm::SMLoc typeLoc = parser.getCurrentLocation();

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

  // Helper for building (and checking) a Signature type.
  auto returnSignatureType = [&](ParamDeclArrayAttr inputParams,
                                 TypeArrayAttr resultParamTypes,
                                 FunctionType valuesType,
                                 ConventionsAttr conventions) -> LogicalResult {
    type = SignatureType::getChecked([&] { return parser.emitError(typeLoc); },
                                     parser.getContext(), inputParams,
                                     resultParamTypes, valuesType, conventions);
    return success(!!type);
  };

  if (succeeded(parser.parseOptionalLess())) {
    // Signature for values and parameters.
    ParamDeclArrayAttr inputParams;
    TypeArrayAttr resultParamTypes;
    if (succeeded(parser.parseOptionalGreater())) {
      inputParams = ParamDeclArrayAttr::get(parser.getContext(), {});
      resultParamTypes = TypeArrayAttr::get(parser.getContext(), {});
    } else if (parseParameterSpec(parser, inputParams, resultParamTypes) ||
               parser.parseGreater()) {
      return failure();
    }
    SmallVector<Type> inputs, outputs;
    ConventionsAttr conventions;
    if (parseTypesWithConventions(parser, inputs, outputs, conventions))
      return failure();
    return returnSignatureType(
        inputParams, resultParamTypes,
        parser.getBuilder().getFunctionType(inputs, outputs), conventions);
  }

  if (failed(parser.parseType(type)))
    return failure();

  // We accept function type syntax as sugar for a SignatureType without
  // parameters.
  if (auto valuesType = dyn_cast<FunctionType>(type)) {
    // Default to empty input/result parameters and no conventions.
    auto noInputParams = ParamDeclArrayAttr::get(parser.getContext(), {});
    auto noResultParams = TypeArrayAttr::get(parser.getContext(), {});
    return returnSignatureType(
        noInputParams, noResultParams, valuesType,
        ConventionsAttr::get(parser.getContext(), valuesType.getNumInputs()));
  }

  return success();
}

void KGEN::printKGENType(AsmPrinter &p, Type type) {
  // Handle other special cases for parameters here.  These each are sugar for a
  // kgen type.
  auto *dialect = type.getContext()->getLoadedDialect<KGENDialect>();
  assert(dialect && "cannot print KGEN type without KGEN dialect");
  if (auto it = dialect->typePrintFns.find(type.getTypeID());
      it != dialect->typePrintFns.end()) {
    it->second(p, type);
  } else if (auto signature = dyn_cast<SignatureType>(type)) {
    // If there are no parameters and no effects, print a SignatureType as a
    // function type to keep things concise.
    if (signature.getInputParams().empty() &&
        signature.getResultParamTypes().empty()) {
      if (signature.getConventions().isDefault()) {
        p << signature.getValues();
        return;
      }
      // If there are effects but no parameters, print "<>" to disambiguate the
      // syntax.
      p << "<>";
    }
    // Otherwise print it as "p1, p2 -> r3, () -> ())"
    printOptionalParameterSpec(p, signature.getInputParams(),
                               signature.getResultParamTypes());
    printTypesWithConventions(p, signature.getValueInputs(),
                              signature.getValueResults(),
                              signature.getConventions());
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
  printParamValue(p, value);
}

/// Parse a parameter value that is known to have `dtype` type.
ParseResult KGEN::parseDTypeParamValue(AsmParser &p,
                                       FailureOr<TypedAttr> &value) {
  TypedAttr result;
  if (parseParamValue(p, result, DTypeType::get(p.getContext())))
    return failure();
  value = result;
  return success();
}

/// Print a parameter value that is known to have `type` type.
void KGEN::printTypeParamValue(AsmPrinter &p, Attribute value) {
  printParamValue(p, value);
}

/// Parse a parameter value that is known to have `type` type.
ParseResult KGEN::parseTypeParamValue(AsmParser &p,
                                      FailureOr<TypedAttr> &value) {
  TypedAttr result;
  if (parseParamValue(p, result, MLIRTypeType::get(p.getContext())))
    return failure();
  value = result;
  return success();
}

/// Print an attribute value that is known to have index type.
void KGEN::printIndexParamValue(AsmPrinter &p, Operation *op, Attribute value) {
  printParamValue(p, value);
}

void KGEN::printIndexParamValue(AsmPrinter &p, Attribute value) {
  printParamValue(p, value);
}

/// Parse a parameter value that is known to be an index type.
ParseResult KGEN::parseIndexParamValue(AsmParser &p, TypedAttr &value) {
  if (parseParamValue(p, value, p.getBuilder().getIndexType()))
    return failure();
  return success();
}

ParseResult KGEN::parseIndexParamValue(AsmParser &p,
                                       FailureOr<TypedAttr> &value) {
  TypedAttr result;
  if (parseIndexParamValue(p, result))
    return failure();
  value = result;
  return success();
}

ParseResult KGEN::parseColonTypeParamValue(AsmParser &p,
                                           FailureOr<TypedAttr> &value) {
  Type type;
  if (parseColonTypeOrIndex(p, type) || parseParamValue(p, value, type))
    return failure();

  return success();
}

void KGEN::printColonTypeParamValue(AsmPrinter &p, TypedAttr value) {
  printColonTypeOrIndexPrefix(p, value.getType());
  printParamValue(p, value);
}

/// We need this for an ODS reason, it doesn't know that ParamDeclAttr is
/// nullable or something :-/.
ParseResult KGEN::parseParamDecl(AsmParser &p,
                                 FailureOr<ParamDeclAttr> &result) {
  ParamDeclAttr pResult;
  if (failed(parseParamDecl(p, pResult)))
    return failure();
  result = pResult;
  return success();
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
  printParamName(p, decl.getName().getValue());
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

/// Parse an parameter list if present, and return it as a SignatureType.
/// parameter-decl   ::= identifier (`:` type)?
/// parameter-list   ::= parameter-decl (`,` parameter-decl)* | `(` `)`
/// parameter-spec   ::= `<` parameter-list (`->` type-list)? `>`
ParseResult
KGEN::parseOptionalParameterSpec(AsmParser &parser,
                                 ParamDeclArrayAttr &inputParamDecls,
                                 TypeArrayAttr &resultParamTypes) {
  // If there is no parameter list, or if it is empty, we're done.
  if (failed(parser.parseOptionalLess()) ||
      succeeded(parser.parseOptionalGreater())) {
    inputParamDecls = ParamDeclArrayAttr::get(parser.getContext(), {});
    resultParamTypes = TypeArrayAttr::get(parser.getContext(), {});
  } else {
    if (parseParameterSpec(parser, inputParamDecls, resultParamTypes) ||
        parser.parseGreater())
      return failure();
  }
  return success();
}

/// Parse a parameter specification as a SignatureType.
ParseResult
KGEN::parseOptionalParameterSpec(AsmParser &parser,
                                 ParamDeclArrayAttr &inputParamDecls) {
  TypeArrayAttr resultParamTypes;
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parseOptionalParameterSpec(parser, inputParamDecls, resultParamTypes))
    return failure();
  if (!resultParamTypes.empty())
    return parser.emitError(loc, "expected no result parameters");
  return success();
}

/// Print a parameter list for a generator, func or interface.
void KGEN::printOptionalParameterSpec(AsmPrinter &p,
                                      ArrayRef<ParamDeclAttr> inputParamDecls,
                                      ArrayRef<Type> resultParamTypes) {
  if (inputParamDecls.empty() && resultParamTypes.empty())
    return;

  p << '<';
  printParamDecls(p, inputParamDecls);

  if (!resultParamTypes.empty()) {
    p << " -> ";
    llvm::interleaveComma(resultParamTypes, p,
                          [&](Type type) { printKGENType(p, type); });
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

ParseResult KGEN::parseParamName(AsmParser &p, FailureOr<StringAttr> &name) {
  name.emplace();
  return parseParamName(p, *name);
}

/// Print a parameter name correctly, using a double quoted syntax if it
/// conflicts with an MLIR or KGEN keyword, or a bareword otherwise.
void KGEN::printParamName(AsmPrinter &p, StringRef name) {
  // If this will conflict with a reserved keyword then we need a '*' prefix and
  // double quotes.
  bool needsQuotes = succeeded(DType::getFromString(name)) ||
                     !isLegalMLIRIdentifier(name) || isMLIRBuiltinType(name);
  if (needsQuotes)
    p << "*\"";
  p << name;
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
  case (uint32_t)POC::GetListElement:
    if (!isa_and_nonnull<ListType>(type))
      return p.emitError(p.getCurrentLocation(),
                         "expected a list type for 'get_list_element'");
    if (parseParamValue(p, operands.emplace_back(), type) || p.parseComma() ||
        parseIndexParamValue(p, operands.emplace_back()))
      return failure();
    return success();
  case (uint32_t)POC::TargetHasFeature:
  case (uint32_t)POC::TargetIsArch:
  case (uint32_t)POC::TargetGetField:
    // Parse TargetHasFeature, TargetIsArch, and TargetGetField --
    // the first operand is a TargetType, the second a StringType.
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
    for (ParamDeclAttr decl : sig.getInputParams()) {
      if (p.parseComma() ||
          parseParamValue(p, operands.emplace_back(),
                          evaluator.getReboundType(decl.getType())))
        return failure();
      evaluator.setParameterValue(decl, operands.back());
    }
    return success();
  }
  case (uint32_t)POC::Apply:
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
        value = DTypeConstantAttr::getChecked(p.getEncodedSourceLoc(loc),
                                              p.getContext(), *dtype, type);
        return success(value != Attribute());
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
      case (uint32_t)POC::TargetEq:
        operandType = TargetType::get(p.getContext());
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
    value = ParamOperatorAttr::get((POC)opcode, operands);

    // If we need to invert this, do so.
    if (needsInvert)
      value = ParamOperatorAttr::getNot(value);

    return success();
  }

  // Otherwise, we support other typed attributes as well, including dialect
  // define attributes, integers, strings, etc.
  return p.parseAttribute(value, type);
}

ParseResult KGEN::parseParamValue(AsmParser &p, FailureOr<TypedAttr> &result,
                                  Type type) {
  result.emplace();
  if (parseParamValue(p, *result, type))
    return failure();
  return success();
}

static void printOperatorOperands(AsmPrinter &p, POC opcode,
                                  ArrayRef<TypedAttr> operands) {
  // If this is a comparison and the elements are not index type, print the
  // type explicitly.
  if (llvm::is_contained({POC::In, POC::EQ, POC::LT, POC::LE, POC::TargetEq},
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

  case POC::GetListElement:
    // Print types on all operands.
    llvm::interleaveComma(operands, p, [&](TypedAttr operand) {
      printColonTypeOrIndexPrefix(p, operand.getType());
      printParamValue(p, operand);
    });
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
    printParamName(p, declRef.getName());
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

//===----------------------------------------------------------------------===//
// Logic shared between funcs, generators, and generator interfaces
//===----------------------------------------------------------------------===//

/// Parse an argument or type list with optional conventions.
static ParseResult
parseElementsWithConventions(AsmParser &p, function_ref<ParseResult()> parseElt,
                             ConventionsAttr &conventions) {
  // Parse an element list with input effects.
  SmallVector<ValueInputConvention> inputConventions;
  auto parseArg = [&]() -> ParseResult {
    if (parseElt())
      return failure();
    StringRef effectStr;
    llvm::SMLoc loc = p.getCurrentLocation();
    if (succeeded(p.parseOptionalKeyword(&effectStr))) {
      if (std::optional<ValueInputConvention> effect =
              symbolizeValueInputConvention(effectStr))
        inputConventions.push_back(*effect);
      else
        return p.emitError(loc, "expected 'byval' or 'byref' for input effect");
    } else {
      inputConventions.push_back(ValueInputConvention::ByVal);
    }
    return success();
  };
  if (p.parseCommaSeparatedList(AsmParser::Delimiter::Paren, parseArg))
    return failure();

  // Parse the function effects. Check for each case to disambiguate the syntax
  // for interfaces.
  auto effect = FnEffects::None;
  StringRef kw;
  while (succeeded(p.parseOptionalKeyword(
      &kw, {"throws", "none", "force_inline", "async"}))) {
    if (kw == "throws")
      effect = effect | FnEffects::Throws;
    else if (kw == "force_inline")
      effect = effect | FnEffects::ForceInline;
    else if (kw == "async")
      effect = effect | FnEffects::Async;
    else if (kw == "none")
      ; // Swallow this keyword

    // No vertical bar? We're done. It's not a parse error, but it does mean we
    // can't specify more effects.
    if (failed(p.parseOptionalVerticalBar()))
      break;
  }

  conventions = ConventionsAttr::get(p.getContext(), inputConventions, effect);
  return success();
}

/// Print an argument or type list with optional conventions.
static void printElementsWithConventions(AsmPrinter &p,
                                         function_ref<void(unsigned)> printElt,
                                         ConventionsAttr conventions) {
  p << '(';
  llvm::interleaveComma(
      llvm::enumerate(conventions.getInputConventions()), p, [&](auto it) {
        printElt(it.index());
        if (it.value() != ValueInputConvention::ByVal)
          p << ' ' << stringifyValueInputConvention(it.value());
      });
  p << ')';

  // Print the function effects.
  if (conventions.getFnEffects() != FnEffects::None)
    p << ' ' << stringifyFnEffects(conventions.getFnEffects());
}

ParseResult KGEN::parseFunctionSignature(
    OpAsmParser &p, SmallVectorImpl<OpAsmParser::Argument> &args,
    SmallVectorImpl<Type> &resultTypes, ConventionsAttr &conventions) {
  // Parse the argument list with input effects.
  auto parseArg = [&]() -> ParseResult {
    OptionalParseResult result =
        p.parseOptionalArgument(args.emplace_back(), /*allowType=*/true);
    if (result.has_value())
      return *result;
    return p.parseType(args.back().type);
  };

  if (parseElementsWithConventions(p, parseArg, conventions) ||
      p.parseOptionalArrowTypeList(resultTypes))
    return failure();
  return success();
}

void KGEN::printFunctionSignature(OpAsmPrinter &p, Region &region,
                                  TypeRange argTypes, TypeRange resultTypes,
                                  ConventionsAttr conventions,
                                  StringArrayAttr valueParamNames) {
  // Print the function arguments.
  auto printElt = [&](unsigned i) {
    if (region.empty())
      p << (valueParamNames ? "%" + valueParamNames[i].getValue() + ": " : "")
        << argTypes[i];
    else
      p.printRegionArgument(region.getArgument(i));
  };
  printElementsWithConventions(p, printElt, conventions);

  // Print the function results.
  if (resultTypes.empty())
    return;
  p << " -> ";
  if (resultTypes.size() == 1 && !isa<FunctionType>(resultTypes.front())) {
    p << resultTypes.front();
    return;
  }
  p << '(';
  llvm::interleaveComma(resultTypes, p);
  p << ')';
}

ParseResult KGEN::parseTypesWithConventions(AsmParser &p,
                                            SmallVectorImpl<Type> &operandTypes,
                                            SmallVectorImpl<Type> &resultTypes,
                                            ConventionsAttr &conventions) {
  auto parseElt = [&] { return p.parseType(operandTypes.emplace_back()); };
  if (parseElementsWithConventions(p, parseElt, conventions) || p.parseArrow())
    return failure();
  if (failed(p.parseOptionalLParen()))
    return p.parseType(resultTypes.emplace_back());
  if (succeeded(p.parseOptionalRParen()))
    return success();
  if (p.parseTypeList(resultTypes) || p.parseRParen())
    return failure();
  return success();
}

void KGEN::printTypesWithConventions(AsmPrinter &p, TypeRange operandTypes,
                                     TypeRange resultTypes,
                                     ConventionsAttr conventions) {
  auto printElt = [&](unsigned i) { p << operandTypes[i]; };
  printElementsWithConventions(p, printElt, conventions);
  p << " -> ";
  if (resultTypes.size() == 1 && !isa<FunctionType>(resultTypes.front())) {
    p << resultTypes.front();
    return;
  }
  p << '(';
  llvm::interleaveComma(resultTypes, p);
  p << ')';
}

/// Parse a constraint specification if present.
/// constraints-spec ::=
///    `constraints` `<` attribute-value (`,` attribute-value)? `>`
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

void KGEN::printOptionalConstraints(OpAsmPrinter &p, Operation *op,
                                    ConstraintArrayAttr constraints) {
  return printOptionalConstraints(p, op, constraints.getValue());
}

/// Parse either a kgen.generator or kgen.func declaration, depending on what
/// `isGenerator` is set to.
ParseResult KGEN::parseGeneratorOrFunc(OpAsmParser &parser,
                                       OperationState &result,
                                       GeneratorOrFuncKind opKind) {
  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<Type> resultTypes;
  Builder &builder = parser.getBuilder();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  ParamDeclArrayAttr inputParamDecls;
  TypeArrayAttr resultParamTypes;
  ConventionsAttr conventions;
  llvm::SMLoc sigLoc;
  if (parseOptionalParameterSpec(parser, inputParamDecls, resultParamTypes) ||
      parser.getCurrentLocation(&sigLoc) ||
      parseFunctionSignature(parser, entryArgs, resultTypes, conventions))
    return failure();

  // Funcs cannot have constraint specifications.
  if (opKind != GeneratorOrFuncKind::func) {
    ConstraintArrayAttr constraints;
    if (parseOptionalConstraints(parser, constraints))
      return failure();
    result.addAttribute("constraints", constraints);
  }

  SmallVector<Type> argTypes;
  argTypes.reserve(entryArgs.size());
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);
  FunctionType type = builder.getFunctionType(argTypes, resultTypes);
  auto signature =
      parser.getChecked<SignatureType>(parser.getContext(), inputParamDecls,
                                       resultParamTypes, type, conventions);
  if (!signature)
    return failure();

  result.addAttribute("signature", TypeAttr::get(signature));

  // If function attributes are present, parse them.
  NamedAttrList parsedAttributes;
  llvm::SMLoc attributeDictLocation = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(parsedAttributes))
    return failure();

  // If this is a generator, see if it is an implementation of a generator
  // interface.
  if (opKind == GeneratorOrFuncKind::generator &&
      succeeded(parser.parseOptionalKeyword("implements"))) {
    SymbolRefAttr implementsAttr;
    if (parser.parseAttribute(implementsAttr,
                              parser.getBuilder().getType<::mlir::NoneType>(),
                              "implements", result.attributes))
      return failure();
  }

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

  // If this is a generator interface, no body block is allowed.
  if (opKind == GeneratorOrFuncKind::interface ||
      dyn_cast_or_null<mlir::UnitAttr>(parsedAttributes.get("isInterface")))
    return success();

  // If this is cached, no body block is allowed.
  if (parsedAttributes.get(Cache::getRegionHashAttrName()))
    return success();

  return parser.parseRegion(*region, entryArgs, /*enableNameShadowing=*/true);
}

void KGEN::printGeneratorOrFunc(OpAsmPrinter &p, FuncInterface op) {
  auto func = cast<mlir::FunctionOpInterface>(*op);

  // Print the operation and the function name.
  StringRef funcName = func.getName();
  p << ' ';

  p.printSymbolName(funcName);
  printOptionalParameterSpec(p, op.getInputParamDeclsAttr(),
                             op.getResultParamTypesAttr());

  ArrayRef<Type> argTypes = op.getArgumentTypes();
  printFunctionSignature(p, func.getFunctionBody(), argTypes,
                         op.getResultTypes(), op.getConventions());

  SmallVector<StringRef> ignoredAttrNames(
      GeneratorOp::getAttributeNames().begin(),
      GeneratorOp::getAttributeNames().end());
  // Don't print evaluator in kgen.generator.interface.
  ignoredAttrNames.push_back("evaluator");
  // Don't print the default_impl in kgen.generator.interface.
  ignoredAttrNames.push_back("defaultImpl");

  // Print out function attributes, if present.
  SmallVector<StringRef, 8> ignoredAttrs = {SymbolTable::getSymbolAttrName()};
  ignoredAttrs.append(ignoredAttrNames.begin(), ignoredAttrNames.end());
  p.printOptionalAttrDictWithKeyword(op->getAttrs(), ignoredAttrs);

  printOptionalConstraints(p, func, cast<DeclInterface>(*op).getConstraints());

  // If this is a generator implementing a generator.interface, include the
  // symbol for the generator interface.
  if (auto gen = dyn_cast<GeneratorOp>(*op)) {
    if (auto itf = gen.getImplementsAttr()) {
      p.printNewline();
      p << "  implements " << itf;
    }
  }

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
    ParamDeclAttr decl;
    TypedAttr value;

    if (parseParamDecl(p, decl) || p.parseEqual() ||
        parseParamValue(p, value, decl.getType()))
      return failure();
    values.push_back(ParamBindAttr::get(decl, value));
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
      printParamDecl(p, bind.getDecl());
      p << " = ";
      printParamValue(p, bind.getValue());
    });
  }
}

/// Parse a list of parameter bindings without result parameters in <>'s
ParseResult
KGEN::parseOptionalParamBindSpec(AsmParser &p,
                                 FailureOr<ParamBindArrayAttr> &paramValues) {
  // If there are no parameter declarations, return an empty array.
  if (p.parseOptionalLess()) {
    paramValues = ParamBindArrayAttr::get(p.getContext(), {});
    return success();
  }

  ParamBindArrayAttr result;
  if (parseParamBinds(p, result))
    return failure();
  paramValues = result;
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

ParseResult KGEN::parseParameterValues(OpAsmParser &p,
                                       ParameterExprArrayAttr &value) {
  SmallVector<TypedAttr> elts;
  if (p.parseCommaSeparatedList(
          OpAsmParser::Delimiter::OptionalLessGreater, [&]() -> ParseResult {
            TypedAttr value;
            if (parseParamValueDefaultingToIndex(p, value))
              return failure();
            elts.push_back(value);
            return success();
          }))
    return failure();

  value = ParameterExprArrayAttr::get(p.getContext(), elts);
  return success();
}

void KGEN::printParameterValues(OpAsmPrinter &p, Operation *op,
                                ParameterExprArrayAttr value) {
  if (value.empty())
    return;
  p << '<';
  llvm::interleaveComma(value, p, [&](TypedAttr value) {
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

  if (!isa<SignatureType>(callee.getType()))
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

/// Parse an align parameter if present.
void KGEN::printOptionalAlignmentParamValue(AsmPrinter &p, Operation *op,
                                            TypedAttr alignment) {
  if (!alignment)
    return;
  p << " align ";
  printParamValue(p, alignment);
  p << " ";
}

/// Parse a parameter value that is known to be an alignment type.
ParseResult KGEN::parseOptionalAlignmentParamValue(AsmParser &p,
                                                   TypedAttr &result) {
  if (p.parseOptionalKeyword("align")) {
    result = TypedAttr();
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
    const char *originatorName, Location originatorLoc, const char *targetName,
    Location targetLoc, const char *itemName, const char *propertyName) {
  // Check that the ranges have the same size.  If not, diagnose this.
  size_t numOriginator =
      std::distance(originatorRange.begin(), originatorRange.end());
  size_t numTarget = std::distance(targetRange.begin(), targetRange.end());
  if (numOriginator != numTarget) {
    auto diag = emitError(originatorLoc, originatorName)
                << " has " << numOriginator << " " << itemName
                << (numOriginator != 1 ? "s" : "") << " but " << targetName
                << " expects " << numTarget;
    if (originatorLoc != targetLoc)
      diag.attachNote(targetLoc) << targetName << " declared here";
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
                << ' ' << originatorVal << " but " << targetName << " expected "
                << propertyName << ' ' << targetVal;
    if (originatorLoc != targetLoc)
      diag.attachNote(targetLoc) << targetName << " declared here";
    return failure();
  }

  return success();
}

/// Check that the specified declaration signatures match, checking the
/// parameter and value type information.
LogicalResult KGEN::verifyDeclSignaturesMatch(const char *originatorName,
                                              SignatureType originatorSignature,
                                              Location originatorLoc,
                                              const char *targetName,
                                              SignatureType targetSignature,
                                              Location targetLoc) {
  FunctionType originatorType = originatorSignature.getValues();
  FunctionType targetType = targetSignature.getValues();
  ParamDeclArrayAttr originatorParamDecls =
      originatorSignature.getInputParams();
  ParamDeclArrayAttr targetParamDecls = targetSignature.getInputParams();

  /// Verify that a list of parameter declarations from a generator or func
  /// matches those of an interface.  This produces an error diagnostic and
  /// returns failure when a problem is detected, or returns true if
  /// everything is ok.
  if (failed(verifyParamDeclsMatch(
          originatorName, originatorParamDecls.getValue(), originatorLoc,
          targetName, targetParamDecls.getValue(), targetLoc)) ||
      verifyMatchingLists(originatorSignature.getResultParamTypes(),
                          targetSignature.getResultParamTypes(), originatorName,
                          originatorLoc, targetName, targetLoc,
                          "result parameter", "type") ||
      verifyMatchingLists(originatorType.getInputs(), targetType.getInputs(),
                          originatorName, originatorLoc, targetName, targetLoc,
                          "argument", "type") ||
      verifyMatchingLists(originatorType.getResults(), targetType.getResults(),
                          originatorName, originatorLoc, targetName, targetLoc,
                          "result", "type"))
    return failure();

  // Check the conventions match up.
  if (originatorSignature.getConventions() !=
      targetSignature.getConventions()) {
    auto diag = emitError(originatorLoc, originatorName)
                << " conventions are " << originatorSignature.getConventions()
                << " but " << targetName << " expected "
                << targetSignature.getConventions();
    if (originatorLoc != targetLoc)
      diag.attachNote(targetLoc) << targetName << " declared here";
    return failure();
  }

  return success();
}

LogicalResult KGEN::verifyParamDeclsMatch(
    const char *originatorName, ArrayRef<ParamDeclAttr> originatorParamDecls,
    Location originatorLoc, const char *targetName,
    ArrayRef<ParamDeclAttr> targetParamDecls, Location targetLoc) {
  using llvm::map_range;
  auto getType = [](auto attr) -> Type { return attr.getType(); };
  auto getName = [](auto attr) -> StringAttr { return attr.getName(); };

  if (verifyMatchingLists(map_range(originatorParamDecls, getName),
                          map_range(targetParamDecls, getName), originatorName,
                          originatorLoc, targetName, targetLoc,
                          "input parameter", "name") ||
      verifyMatchingLists(map_range(originatorParamDecls, getType),
                          map_range(targetParamDecls, getType), originatorName,
                          originatorLoc, targetName, targetLoc,
                          "input parameter", "type"))
    return failure();
  return success();
}

/// Check that the specified generator/interfaces matches signature
/// information with the other interface.
LogicalResult KGEN::verifyDeclMatchesInterface(
    const char *originatorName, FuncInterface originatorDecl,
    const char *interfaceName, GeneratorInterfaceOp interfaceDecl) {

  return verifyDeclSignaturesMatch(
      originatorName, originatorDecl.getSignature(), originatorDecl.getLoc(),
      interfaceName, interfaceDecl.getSignature(), interfaceDecl.getLoc());
}

/// If the specified operation is non-null and contains parameters, collect
/// them into the specified array.
static void collectContextParameters(Operation *op,
                                     SmallVector<ParamDeclAttr> &params) {
  auto decl = dyn_cast_or_null<DeclInterface>(op);
  if (!decl || isa<FuncInterface>(*decl))
    return;
  collectContextParameters(op->getParentOp(), params);
  llvm::append_range(params, decl.getInputParamDecls());
}

/// Return the full signature of this declaration, including parameters from
/// enclosing struct declarations.
SignatureType KGEN::getFullSignature(FuncInterface decl) {
  SignatureType signature = decl.getSignature();

  // Collect contextual params, if there are none, the full signature is the
  // same as the local signature.
  SmallVector<ParamDeclAttr> inputParams;
  collectContextParameters(decl.getOperation()->getParentOp(), inputParams);
  if (inputParams.empty())
    return signature;

  llvm::append_range(inputParams, signature.getInputParams());

  return SignatureType::get(
      ParamDeclArrayAttr::get(signature.getContext(), inputParams),
      signature.getResultParamTypes(), signature.getValues(),
      signature.getConventions());
}

/// Verify that the provided operation has exactly one block in its body
/// region, or that region was cached.
LogicalResult KGEN::verifyOneBlockOrCached(Operation *op) {
  size_t numBlocks = op->getRegion(0).getBlocks().size();
  if (numBlocks == 0) {
    if (!op->hasAttr(Cache::getRegionHashAttrName()))
      return op->emitError()
             << "must have a body region or it must be elided into the cache";
  }

  if (numBlocks > 1)
    return op->emitError() << "does not support > 1 block in its body";

  return success();
}

LogicalResult
KGEN::checkResultArgumentTypes(Operation *op, ArrayRef<TypedAttr> resultParams,
                               ArrayRef<Type> paramResultTypes,
                               std::optional<TypeRange> resultTypes) {
  // Check the parameters match up.
  if (resultParams.size() != paramResultTypes.size())
    return op->emitOpError("expected ")
           << paramResultTypes.size() << " parameters for enclosing op";

  for (size_t i = 0, e = paramResultTypes.size(); i != e; ++i) {
    auto expectedTy = paramResultTypes[i];
    auto actualTy = resultParams[i].cast<TypedAttr>().getType();
    if (actualTy != expectedTy)
      return op->emitOpError("parameter #") << i << " has type " << actualTy
                                            << " but should be " << expectedTy;
  }

  // Verify the result types if they were provided.
  if (!resultTypes)
    return success();

  return checkResultTypes(op, *resultTypes);
}
