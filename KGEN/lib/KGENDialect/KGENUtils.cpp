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
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENDeclInterface.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "Support/ML/DType.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace M;
using namespace KGEN;
using mlir::OptionalParseResult;

/// Return the string form for an attribute value that is printed in a <>
/// context in the .mlir file.
std::string KGEN::getParamAsString(Attribute value) {
  SmallVector<char, 128> result;
  {
    llvm::raw_svector_ostream os(result);
    if (auto ta = dyn_cast<TypedAttr>(value))
      printParamValue(ta, os);
    else
      os << value;
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
  // Check for sugared types before parsing standard ones.
  if (succeeded(parser.parseOptionalKeyword("type"))) {
    type = parser.getBuilder().getType<MLIRTypeType>();
    return success();
  }

  if (succeeded(parser.parseOptionalKeyword("dtype"))) {
    type = parser.getBuilder().getType<DTypeType>();
    return success();
  }

  if (succeeded(parser.parseOptionalKeyword("string"))) {
    type = parser.getBuilder().getType<StringType>();
    return success();
  }

  if (succeeded(parser.parseOptionalKeyword("list"))) {
    FailureOr<TypedAttr> elementType, numElements;
    if (parser.parseLess() || parseTypeParamValue(parser, elementType) ||
        parser.parseLSquare() || parseIndexParamValue(parser, numElements) ||
        parser.parseRSquare() || parser.parseGreater())
      return failure();
    type = ListType::get(*elementType, *numElements);
    return success();
  }

  if (succeeded(parser.parseOptionalKeyword("target"))) {
    type = parser.getBuilder().getType<TargetType>();
    return LogicalResult::success();
  }

  // Helper for building (and checking) a Signature type.
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  auto returnSignatureType = [&](ParamDeclArrayAttr inputParams,
                                 TypeArrayAttr resultParamTypes,
                                 FunctionType valuesType) -> LogicalResult {
    auto sigTy = SignatureType::get(inputParams, resultParamTypes, valuesType);

    // Signature types can fail to parse when they reference parameters that
    // are not defined in their input list.  Handle this by reporting the error
    // correctly through the parser and returning a failure.
    auto isSelfContained = sigTy.checkSelfContained();
    if (isSelfContained.isError())
      return parser.emitError(typeLoc, isSelfContained.takeError().get());
    type = sigTy;
    return success();
  };

  if (succeeded(parser.parseOptionalLess())) {
    // signature for values and parameters.
    ParamDeclArrayAttr inputParams;
    TypeArrayAttr resultParamTypes;
    if (succeeded(parser.parseOptionalGreater())) {
      inputParams = ParamDeclArrayAttr::get(parser.getContext(), {});
      resultParamTypes = TypeArrayAttr::get(parser.getContext(), {});
    } else if (parseParameterSpec(parser, inputParams, resultParamTypes) ||
               parser.parseGreater()) {
      return failure();
    }
    FunctionType valuesType;
    if (parser.parseType(valuesType))
      return failure();
    return returnSignatureType(inputParams, resultParamTypes, valuesType);
  }

  if (failed(parser.parseType(type)))
    return failure();

  // We accept function type syntax as sugar for a SignatureType without
  // parameters.
  if (auto valuesType = dyn_cast<FunctionType>(type)) {
    // Default to empty input/result parameters.
    auto noInputParams = ParamDeclArrayAttr::get(parser.getContext(), {});
    auto noResultParams = TypeArrayAttr::get(parser.getContext(), {});
    return returnSignatureType(noInputParams, noResultParams, valuesType);
  }

  return success();
}

void KGEN::printKGENType(raw_ostream &os, Type type) {
  // Handle other special cases for parameters here.  These each are sugar for a
  // kgen type.
  if (isa<MLIRTypeType>(type)) {
    os << "type";
  } else if (isa<DTypeType>(type)) {
    os << "dtype";
  } else if (isa<StringType>(type)) {
    os << "string";
  } else if (isa<TargetType>(type)) {
    os << "target";
  } else if (auto list = dyn_cast<ListType>(type)) {
    os << "list<";
    printParamValue(list.getElementType(), os);
    os << '[';
    printParamValue(list.getLength(), os);
    os << "]>";
  } else if (auto signature = dyn_cast<SignatureType>(type)) {
    // If there are no parameters, print a SignatureType as a function type to
    // keep things concise.
    if (signature.getInputParams().empty() &&
        signature.getResultParamTypes().empty())
      os << signature.getValues();
    else { // Otherwise print it as "p1, p2 -> r3, () -> ())"
      printOptionalParameterSpec(signature.getInputParams(),
                                 signature.getResultParamTypes(), os);
      os << signature.getValues();
    }
  } else {
    os << type;
  }
}

static OptionalParseResult parseOptionalColonType(AsmParser &parser,
                                                  Type &type) {
  if (failed(parser.parseOptionalColon()))
    return None;
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
void KGEN::printColonTypeOrIndex(raw_ostream &os, Type type) {
  // Index type is the default so it doesn't print.
  if (type.isIndex())
    return;
  os << ": ";
  printKGENType(os, type);
}

/// print `:<type> ` or elide it entirely if type is an `index` type.
static void printColonTypeOrIndexPrefix(raw_ostream &os, Type type) {
  // Index type is the default so it doesn't print.
  if (type.isIndex())
    return;
  os << ':';
  printKGENType(os, type);
  os << ' ';
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
  printParamDecl(decl, p.getStream());
}

void KGEN::printParamDecl(ParamDeclAttr decl, raw_ostream &os) {
  printParamName(decl.getName().getValue(), os);
  printColonTypeOrIndex(os, decl.getType());
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
void KGEN::printParamDecls(ParamDeclArrayAttr decls, raw_ostream &os) {
  if (decls.empty()) {
    os << "()";
  } else {
    llvm::interleaveComma(
        decls, os, [&](ParamDeclAttr decl) { printParamDecl(decl, os); });
  }
}

/// Parse an parameter list if present.
/// parameter-decl   ::= identifier (`:` type)?
/// parameter-list   ::= parameter-decl (`,` parameter-decl)* | `(` `)`
/// parameter-spec   ::= `<` parameter-list (`->` type-list)? `>`
ParseResult
KGEN::parseOptionalParameterSpec(AsmParser &parser,
                                 ParamDeclArrayAttr &inputParamDecls,
                                 TypeArrayAttr &resultParamTypesAttr) {
  // If there is no parameter list, or if it is empty, we're done.
  if (failed(parser.parseOptionalLess()) ||
      succeeded(parser.parseOptionalGreater())) {
    inputParamDecls = ParamDeclArrayAttr::get(parser.getContext(), {});
    resultParamTypesAttr = TypeArrayAttr::get(parser.getContext(), {});
    return success();
  }

  if (parseParameterSpec(parser, inputParamDecls, resultParamTypesAttr) ||
      parser.parseGreater())
    return failure();
  return success();
}

/// Print a parameter list for a generator, func or interface.
void KGEN::printOptionalParameterSpec(ParamDeclArrayAttr inputParamDecls,
                                      TypeArrayAttr resultParamTypes,
                                      raw_ostream &os) {
  if (inputParamDecls.empty() && resultParamTypes.empty())
    return;

  os << '<';
  printParamDecls(inputParamDecls, os);

  if (!resultParamTypes.empty()) {
    os << " -> ";
    llvm::interleaveComma(resultParamTypes.getValue(), os,
                          [&](Type type) { printKGENType(os, type); });
  }
  os << '>';
}

void KGEN::printOptionalParameterSpec(AsmPrinter &p, Operation *op,
                                      ParamDeclArrayAttr paramDecls,
                                      TypeArrayAttr resultParamTypes) {
  printOptionalParameterSpec(paramDecls, resultParamTypes, p.getStream());
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

void KGEN::printParamName(StringRef name, raw_ostream &os) {
  // If this will conflict with a reserved keyword then we need a '*' prefix and
  // double quotes.
  bool needsQuotes = succeeded(DType::getFromString(name)) ||
                     !isLegalMLIRIdentifier(name) || isMLIRBuiltinType(name) ||
                     name == "region";
  if (needsQuotes)
    os << "*\"";
  os << name;
  if (needsQuotes)
    os << '"';
}

/// Print a parameter name correctly, using a double quoted syntax if it
/// conflicts with an MLIR or KGEN keyword, or a bareword otherwise.
void KGEN::printParamName(AsmPrinter &p, StringRef name) {
  printParamName(name, p.getStream());
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
  case (uint32_t)POC::BindSignature:
  case (uint32_t)POC::Apply:
    if (!isa_and_nonnull<SignatureType>(type))
      return p.emitError(p.getCurrentLocation(),
                         "expected a signature type for operator");
    // Parse each operand with a type.  TODO: We could do better here by using
    // the signature to infer the types of the parameters.
    if (parseParamValue(p, operands.emplace_back(), type))
      return failure();
    if (failed(p.parseOptionalComma()))
      return success();

    return p.parseCommaSeparatedList([&]() -> LogicalResult {
      if (parseColonTypeOrIndex(p, type) ||
          parseParamValue(p, operands.emplace_back(), type))
        return failure();
      return success();
    });

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

  // If this is a '*'-prefixed double quoted string, then this is a simple
  // parameter reference.
  if (succeeded(p.parseOptionalStar())) {
    std::string name;
    if (failed(p.parseString(&name)))
      return failure();
    value = ParamDeclRefAttr::get(name, type);
    return success();
  }

  // If this parameter is a type, parse it as such here to catch MLIR builtin
  // types that look like keywords.
  if (type.isa<MLIRTypeType>()) {
    Type result;
    OptionalParseResult parseResult = p.parseOptionalType(result);
    if (parseResult.has_value()) {
      if (failed(parseResult.value()))
        return failure();
      // We always parse this as a parameterized type, but the builder will form
      // a concrete type if there are no type parameters in it.  We could add
      // specific syntax to differentiate them if there is a reason to.
      value = TypeConstantAttr::get(result);
      return success();
    }
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
        value = DTypeConstantAttr::getChecked(
            p.getEncodedSourceLoc(loc), p.getContext(), dtype.value(), type);
        return success(value != Attribute());
      }
    }

    /// The region keyword is a token that specifies that a signature value will
    /// be provided by a region on a kgen.call{_param} operation.
    if (keyword == "region" && type.isa<SignatureType>()) {
      value = ParamCallRegionRefAttr::get(p.getContext(),
                                          cast<SignatureType>(type));
      return success();
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
        // The subtraction operation defaults to index type for its operands.
        operandType = p.getBuilder().getIndexType();
        break;
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
      case (uint32_t)POC::GetDType:
      case (uint32_t)POC::GetSizeOf:
      case (uint32_t)POC::GetAlignOf:
        // The `get_dtype` and `get_sizeof` operand is always an MLIR type.
        operandType = MLIRTypeType::get(p.getContext());
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

  // If this is a SignatureType, we expect a symbol name or a
  // SymbolConstantAttr.  We need special parsing logic here because
  // FlatSymbolRefAttr isn't a TypedAttr.
  if (auto sigType = dyn_cast<SignatureType>(type)) {
    Attribute attr;
    if (p.parseAttribute(attr, type))
      return failure();

    if (auto symbol = dyn_cast<SymbolRefAttr>(attr)) {
      // Parse any trailing parameter bindings.
      FailureOr<ParamBindArrayAttr> paramValues;
      if (parseOptionalParamBindSpec(p, paramValues))
        return failure();
      value = SymbolConstantAttr::get(symbol, paramValues.value(), sigType);
      return success();
    }

    if (auto typedAttr = dyn_cast<TypedAttr>(attr)) {
      value = typedAttr;
      return success();
    }
    return p.emitError(loc, "invalid signature parameter attribute");
  }

  // If this is a list type, parse a comma-separated list of parameter values of
  // the element type surrounded by square brackets.
  if (auto list = dyn_cast<ListType>(type)) {
    Optional<int64_t> length = list.getResolvedLength();
    llvm::SMLoc loc = p.getCurrentLocation();
    if (!length)
      return p.emitError(
          loc, "cannot parse a list constant for a list with unknown size");

    if (p.parseLSquare())
      return failure();
    if (succeeded(p.parseOptionalRSquare())) {
      value = ListAttr::get(p.getContext(), {}, list);
      return success();
    }
    SmallVector<TypedAttr> values;
    auto type = ParamRefType::get(list.getElementType());
    if (p.parseCommaSeparatedList(
            [&] { return parseParamValue(p, values.emplace_back(), type); }) ||
        p.parseRSquare())
      return failure();
    value = ListAttr::get(p.getContext(), values, list);

    int64_t numParsedElements = cast<ListAttr>(value).getValues().size();
    if (numParsedElements != *length)
      return p.emitError(loc, "expected ")
             << *length << " list elements but got " << numParsedElements;
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

static void printOperatorOperands(raw_ostream &os, POC opcode,
                                  ArrayRef<TypedAttr> operands) {
  // If this is a comparison and the elements are not index type, print the
  // type explicitly.
  if (llvm::is_contained(
          {POC::In, POC::EQ, POC::LT, POC::LE, POC::TargetSupports}, opcode))
    printColonTypeOrIndexPrefix(os, operands[0].getType());

  switch (opcode) {
  default:
    // operand-list ::= expr (`,` expr)*
    llvm::interleaveComma(
        operands, os, [&](TypedAttr operand) { printParamValue(operand, os); });
    break;
  case POC::In:
    // operand-list ::= expr `,` `[` (expr (`,` expr)*)? `]`
    printParamValue(operands[0], os);
    os << ", [";
    llvm::interleaveComma(operands.drop_front(), os, [&](TypedAttr operand) {
      printParamValue(operand, os);
    });
    os << "]";
    break;

  case POC::GetListElement:
  case POC::Apply:
  case POC::BindSignature:
    // Print types on all operands.
    llvm::interleaveComma(operands, os, [&](TypedAttr operand) {
      printColonTypeOrIndexPrefix(os, operand.getType());
      printParamValue(operand, os);
    });
    break;
  }
}

/// Convert a parameter value to a string when in a context that knows it is
/// dealing with a parameter specifically.  This utilize syntactic shortcuts to
/// make the printed syntax easier to grok.
void KGEN::printParamValue(TypedAttr value, raw_ostream &os) {
  if (isa<UnknownAttr>(value)) {
    os << '?';
    return;
  }

  if (auto declRef = dyn_cast<ParamDeclRefAttr>(value)) {
    printParamName(declRef.getName(), os);
    return;
  }

  // If this is a type constant, print it as a bare type.
  if (auto typeConstant = dyn_cast<TypeConstantAttr>(value)) {
    if (auto paramRef = dyn_cast<ParamRefType>(typeConstant.getValue()))
      printParamValue(paramRef.getParam(), os);
    else
      os << typeConstant.getValue();
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
      os << stringRep;
      return;
    }
  }

  // Symbol constants print as just the symbol followed by parameter bindings.
  if (auto symbolConstant = dyn_cast<SymbolConstantAttr>(value)) {
    os << symbolConstant.getSymbol();
    printOptionalParamBindSpec(symbolConstant.getParamValues(), os);
    return;
  }

  // A ParamCallRegionRefAttr is always printed as "region" in an argument list.
  if (value.isa<ParamCallRegionRefAttr>()) {
    os << "region";
    return;
  }

  // Handle expressions.
  if (auto expr = dyn_cast<ParamOperatorAttr>(value)) {
    auto printExpr = [&](StringRef opcode, ArrayRef<TypedAttr> operands) {
      os << opcode << '(';
      printOperatorOperands(os, expr.getOpcode(), operands);
      os << ')';
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
      os << (intAttr.getValue().isZero() ? 0 : 1);
      return;
    }
  }

  if (auto list = dyn_cast<ListAttr>(value)) {
    os << '[';
    llvm::interleaveComma(list.getValues(), os,
                          [&](TypedAttr value) { printParamValue(value, os); });
    os << ']';
    return;
  }

  value.print(os, /*elideType=*/true);
}

/// When in a context that knows it is dealing with a parameter specifically,
/// utilize syntactic shortcuts to make the printed syntax easier to grok.
void KGEN::printParamValue(AsmPrinter &p, TypedAttr value, Type type) {
  printParamValue(value, p.getStream());
}

//===----------------------------------------------------------------------===//
// Logic shared between funcs, generators, and generator interfaces
//===----------------------------------------------------------------------===//

/// Parse a constraint specification if present.
/// constraints-spec ::=
///    `constraints` `<` attribute-value (`,` attribute-value)? `>`
static ParseResult parseOptionalConstraints(OpAsmParser &parser,
                                            OperationState &result,
                                            GeneratorOrFuncKind opKind) {
  // Funcs cannot have constraint specifications.
  if (opKind == GeneratorOrFuncKind::func)
    return success();
  ConstraintArrayAttr constraints;
  if (parseOptionalConstraints(parser, constraints))
    return failure();
  result.addAttribute("constraints", constraints);
  return success();
}

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
                                    ConstraintArrayAttr constraints) {
  if (!constraints || constraints.empty())
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

/// Parse either a kgen.generator or kgen.func declaration, depending on what
/// `isGenerator` is set to.
ParseResult KGEN::parseGeneratorOrFunc(OpAsmParser &parser,
                                       OperationState &result,
                                       GeneratorOrFuncKind opKind) {
  using namespace mlir::function_interface_impl;

  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<DictionaryAttr> resultAttrs;
  SmallVector<Type> resultTypes;
  auto &builder = parser.getBuilder();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  bool isVariadic = false;
  ParamDeclArrayAttr inputParamDecls;
  TypeArrayAttr resultParamTypes;
  if (parseOptionalParameterSpec(parser, inputParamDecls, resultParamTypes) ||
      parseFunctionSignature(parser, /*allowVariadic=*/false, entryArgs,
                             isVariadic, resultTypes, resultAttrs) ||
      ::parseOptionalConstraints(parser, result, opKind))
    return failure();

  result.addAttribute("paramDecls", inputParamDecls);
  result.addAttribute("resultParamTypes", resultParamTypes);

  SmallVector<Type> argTypes;
  argTypes.reserve(entryArgs.size());
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);
  Type type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));

  // If this is a litfunc, handle keyword argument names.
  if (opKind == GeneratorOrFuncKind::litfunc) {
    SmallVector<StringAttr> names;
    for (auto &arg : entryArgs) {
      StringRef spelling;
      assert(arg.ssaName.name.size() >= 2 && "Should have % and one letter");
      if (isdigit(arg.ssaName.name[1])) // %42 -> no name.
        spelling = "";
      else
        spelling = arg.ssaName.name.drop_front();
      names.push_back(builder.getStringAttr(spelling));
    }

    result.addAttribute("valueParamNames",
                        StringArrayAttr::get(builder.getContext(), names));
  }

  // If function attributes are present, parse them.
  NamedAttrList parsedAttributes;
  llvm::SMLoc attributeDictLocation = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(parsedAttributes))
    return failure();

  // If this is a generator, see if it is an implementation of a generator
  // interface.
  if ((opKind == GeneratorOrFuncKind::generator ||
       opKind == GeneratorOrFuncKind::litfunc) &&
      succeeded(parser.parseOptionalKeyword("implements"))) {
    FlatSymbolRefAttr implementsAttr;
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

  // Add the attributes to the function arguments.
  assert(resultAttrs.size() == resultTypes.size());
  addArgAndResultAttrs(builder, result, entryArgs, resultAttrs);

  // Parse the required function body.
  auto *region = result.addRegion();

  // If this is a generator interface, no body block is allowed.
  if (opKind == GeneratorOrFuncKind::interface)
    return success();

  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseRegion(*region, entryArgs,
                         /*enableNameShadowing=*/false))
    return failure();

  if (region->empty()) {
    if (opKind != GeneratorOrFuncKind::litfunc)
      return parser.emitError(loc, "expected non-empty function body");
    region->push_back(new Block());
  }

  // Function body was parsed, make sure it's not empty.
  Attribute isInterface = parsedAttributes.get("isInterface");
  Block &body = region->back();
  if (!isInterface && body.empty())
    return parser.emitError(loc, "expected non-empty function body");
  if (isInterface && !body.empty())
    return parser.emitError(loc, "expected empty function body");

  return success();
}

void KGEN::printGeneratorOrFunc(OpAsmPrinter &p, mlir::FunctionOpInterface op) {
  using namespace mlir::function_interface_impl;

  // TODO: KGENDeclInterface should inherit from FunctionOpInterface.
  auto opDecl = cast<KGENDeclInterface>((Operation *)op);

  // Print the operation and the function name.
  auto funcName =
      op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  p << ' ';

  p.printSymbolName(funcName);
  printOptionalParameterSpec(opDecl.getParamDeclsAttr(),
                             opDecl.getResultParamTypesAttr(), p.getStream());

  ArrayRef<Type> argTypes = op.getArgumentTypes();
  ArrayRef<Type> resultTypes = op.getResultTypes();
  printFunctionSignature(p, op, argTypes, /*isVariadic=*/false, resultTypes);

  SmallVector<StringRef> ignoredAttrNames(
      GeneratorOp::getAttributeNames().begin(),
      GeneratorOp::getAttributeNames().end());
  // Don't print valueParamNames in lit.func.
  ignoredAttrNames.push_back(StringRef("valueParamNames"));
  // Don't print evaluator in kgen.generator.interface.
  ignoredAttrNames.push_back(StringRef("evaluator"));
  // Don't print the default_impl in kgen.generator.interface.
  ignoredAttrNames.push_back(StringRef("defaultImpl"));

  printFunctionAttributes(p, op, argTypes.size(), resultTypes.size(),
                          ignoredAttrNames);
  printOptionalConstraints(p, opDecl, opDecl.getConstraintsAttr());

  // If this is a generator implementing a generator.interface, include the
  // symbol for the generator interface.
  if (auto implementsAttr =
          op->getAttrOfType<FlatSymbolRefAttr>("implements")) {
    p.printNewline();
    p << "  implements " << implementsAttr;
  }

  p << ' ';
  if (!op.getFunctionBody().empty()) {
    p.printRegion(op.getFunctionBody(), /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
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

void KGEN::printParamBinds(ParamBindArrayAttr paramBinds, raw_ostream &os) {
  if (paramBinds.empty()) {
    os << "()";
  } else {
    llvm::interleaveComma(paramBinds, os, [&](ParamBindAttr bind) {
      printParamDecl(bind.getDecl(), os);
      os << " = ";
      printParamValue(bind.getValue(), os);
    });
  }
}

void KGEN::printParamBinds(AsmPrinter &p, ParamBindArrayAttr paramBinds) {
  printParamBinds(paramBinds, p.getStream());
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

void KGEN::printOptionalParamBindSpec(ParamBindArrayAttr paramValues,
                                      raw_ostream &os) {
  if (paramValues.empty())
    return;
  os << '<';
  printParamBinds(paramValues, os);
  os << '>';
}

void KGEN::printOptionalParamBindSpec(AsmPrinter &p,
                                      ParamBindArrayAttr paramValues) {
  printOptionalParamBindSpec(paramValues, p.getStream());
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
  /// returns failure when a problem is detected, or returns true if everything
  /// is ok.
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

/// Check that the specified generator/interfaces matches signature information
/// with the other interface.
LogicalResult KGEN::verifyDeclMatchesInterface(
    const char *originatorName, KGENDeclInterface originatorDecl,
    const char *interfaceName, GeneratorInterfaceOp interfaceDecl) {

  return verifyDeclSignaturesMatch(
      originatorName, originatorDecl.getSignature(), originatorDecl.getLoc(),
      interfaceName, interfaceDecl.getSignature(), interfaceDecl.getLoc());
}
