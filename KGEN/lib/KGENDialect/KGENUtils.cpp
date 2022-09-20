//===- KGENUtils.cpp ------------------------------------------------------===//
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
#include "KGENVerifyHelper.h"
#include "Support/ML/DType.h"
#include "mlir/IR/FunctionImplementation.h"
using namespace M;
using namespace KGEN;
using mlir::OptionalParseResult;

static OptionalParseResult parseOptionalColonType(AsmParser &parser,
                                                  Type &type) {
  if (failed(parser.parseOptionalColon()))
    return None;
  return OptionalParseResult(parseKGENType(parser, type));
}

/// Return the string form for an attribute value that is printed in a <>
/// context in the .mlir file.
std::string KGEN::getParamAsString(Attribute value) {
  SmallVector<char, 128> result;
  {
    llvm::raw_svector_ostream os(result);
    if (auto ta = value.dyn_cast<TypedAttr>())
      printParamValue(ta, os);
    else
      os << value;
  }
  return std::string(result.data(), result.size());
}

/// Parse a type in a KGEN context, handling sugar like "dtype" for
/// "!kgen.dtype" etc.
ParseResult KGEN::parseKGENType(AsmParser &parser, Type &type) {
  // Check for sugared types before parsing standard ones.
  if (succeeded(parser.parseOptionalKeyword("type"))) {
    type = parser.getBuilder().getType<MLIRTypeType>();
    return LogicalResult::success();
  }

  if (succeeded(parser.parseOptionalKeyword("dtype"))) {
    type = parser.getBuilder().getType<DTypeType>();
    return LogicalResult::success();
  }

  if (succeeded(parser.parseOptionalKeyword("string"))) {
    type = parser.getBuilder().getType<StringType>();
    return LogicalResult::success();
  }

  // Helper for building (and checking) a Signature type.
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  auto returnSignatureType = [&](ParamDeclArrayAttr inputParams,
                                 TypeArrayAttr resultParamTypes,
                                 FunctionType valuesType) -> LogicalResult {
    // Signature types can fail to parse when they reference parameters that
    // are not defined in their input list.  Handle this by reporting the error
    // correctly through the parser and returning a failure.
    type = SignatureType::getChecked(
        [&]() -> InFlightDiagnostic { return parser.emitError(typeLoc); },
        inputParams.getContext(), inputParams, resultParamTypes, valuesType);
    return LogicalResult::success(type != Type());
  };

  if (succeeded(parser.parseOptionalKeyword("signature"))) {
    // signature for values and parameters.
    ParamDeclArrayAttr inputParams;
    TypeArrayAttr resultParamTypes;
    FunctionType valuesType;
    if (parser.parseLess() ||
        parseOptionalParameterSpec(parser, inputParams, resultParamTypes) ||
        parser.parseType(valuesType) || parser.parseGreater())
      return failure();
    return returnSignatureType(inputParams, resultParamTypes, valuesType);
  }

  if (failed(parser.parseType(type)))
    return LogicalResult::failure();

  // We accept function type syntax as sugar for a SignatureType without
  // parameters.
  if (auto valuesType = type.dyn_cast<FunctionType>()) {
    // Default to empty input/result parameters.
    auto noInputParams = ParamDeclArrayAttr::get(parser.getContext(), {});
    auto noResultParams = TypeArrayAttr::get(parser.getContext(), {});
    return returnSignatureType(noInputParams, noResultParams, valuesType);
  }

  return LogicalResult::success();
}

void KGEN::printKGENType(raw_ostream &os, Type type) {
  // Handle other special cases for parameters here.  These each are sugar for a
  // kgen type.
  if (type.isa<MLIRTypeType>())
    os << "type";
  else if (type.isa<DTypeType>())
    os << "dtype";
  else if (type.isa<StringType>())
    os << "string";
  else if (auto signature = type.dyn_cast<SignatureType>()) {
    // If there are no parameters, print a SignatureType as a function type to
    // keep things concise.
    if (signature.getInputParams().empty() &&
        signature.getResultParamTypes().empty())
      os << signature.getValues();
    else { // Otherwise print it as "signature<p1, p2 -> r3, () -> ())>"
      os << "signature<";
      printOptionalParameterSpec(os, signature.getInputParams(),
                                 signature.getResultParamTypes());
      os << signature.getValues() << ">";
    }
  } else
    os << type;
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

/// Print a parameter value that either has an index type or is null (which
/// prints as a `?`).
void KGEN::printOptionalIndexParamValue(AsmPrinter &p, Attribute value) {
  if (!value)
    p << '?';
  else
    printIndexParamValue(p, value);
}

/// Parse a parameter value that is known to be an index type or a `?` which
/// results in a null attribute.
ParseResult KGEN::parseOptionalIndexParamValue(AsmParser &p,
                                               FailureOr<TypedAttr> &result) {
  if (succeeded(p.parseOptionalQuestion())) {
    result = TypedAttr();
    return success();
  }
  return parseIndexParamValue(p, result);
}

/// Print a parameter value that either has `dtype` type or is null (which
/// prints as a `?`).
void KGEN::printOptionalDTypeParamValue(AsmPrinter &p, Attribute value) {
  if (!value)
    p << '?';
  else
    printDTypeParamValue(p, value);
}

/// Parse a parameter value that is known to be an index type or a `?` which
/// results in a null attribute.
ParseResult KGEN::parseOptionalDTypeParamValue(AsmParser &p,
                                               FailureOr<TypedAttr> &result) {
  if (succeeded(p.parseOptionalQuestion())) {
    result = TypedAttr();
    return success();
  }
  return parseDTypeParamValue(p, result);
}

/// Print a parameter value that either has `type` type or is null (which
/// prints as a `?`).
void KGEN::printOptionalTypeParamValue(AsmPrinter &p, TypedAttr value) {
  if (!value)
    p << '?';
  else
    printTypeParamValue(p, value);
}

/// Parse a parameter value that is known to be a `type` type or a `?` which
/// results in a null attribute.
ParseResult KGEN::parseOptionalTypeParamValue(AsmParser &p,
                                              FailureOr<TypedAttr> &result) {
  if (succeeded(p.parseOptionalQuestion())) {
    result = TypedAttr();
    return success();
  }
  return parseTypeParamValue(p, result);
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
    auto parseParamDecl = [&]() -> ParseResult {
      StringAttr name;
      Type type;
      if (parseParamName(p, name) || parseColonTypeOrIndex(p, type))
        return failure();
      decls.push_back(ParamDeclAttr::get(name, type));
      return success();
    };
    if (p.parseCommaSeparatedList(OpAsmParser::Delimiter::None, parseParamDecl))
      return failure();
  }

  result = ParamDeclArrayAttr::get(p.getContext(), decls);
  return success();
}

/// Print a comma separated parameter declaration list.
void KGEN::printParamDecls(raw_ostream &os, ParamDeclArrayAttr decls) {
  if (decls.empty()) {
    os << "()";
  } else {
    llvm::interleaveComma(decls, os, [&](ParamDeclAttr ref) {
      printParamName(ref.getName().getValue(), os);
      printColonTypeOrIndex(os, ref.getType());
    });
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

  return parser.parseGreater();
}

/// Print a parameter list for a generator, func or interface.
void KGEN::printOptionalParameterSpec(raw_ostream &os,
                                      ParamDeclArrayAttr inputParamDecls,
                                      TypeArrayAttr resultParamTypes) {
  if (inputParamDecls.empty() && resultParamTypes.empty())
    return;

  os << '<';
  printParamDecls(os, inputParamDecls);

  if (!resultParamTypes.empty()) {
    os << " -> ";
    llvm::interleaveComma(resultParamTypes.getValue(), os,
                          [&](Type type) { printKGENType(os, type); });
  }
  os << '>';
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
  // If this will conflict with a DType keyword or isn't a legal MLIR name,
  // then we need a '*' prefix and double quotes.
  bool needsQuotes =
      succeeded(DType::getFromString(name)) || !isLegalMLIRIdentifier(name);
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
  case (uint32_t)POC::IN:
  case (uint32_t)POCAliases::NOT_IN:
    // operand-list ::= expr `,` `[` (expr (`,` expr)*)? `]`
    if (parseParamValue(p, operands.emplace_back(), type) || p.parseComma() ||
        p.parseCommaSeparatedList(AsmParser::Delimiter::OptionalSquare, [&] {
          return parseParamValue(p, operands.emplace_back(), type);
        }))
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

  // Barewords / MLIR keywords are implicitly parameter declaration references
  // or the start of a expression in function form.
  StringRef keyword;
  if (succeeded(p.parseOptionalKeyword(&keyword))) {
    // Check to see if we're parsing a dtype name like 'f32'.
    if (type.isa<DTypeType>()) {
      auto dtype = DType::getFromString(keyword);
      if (succeeded(dtype)) {
        value = DTypeConstantAttr::getChecked(
            p.getEncodedSourceLoc(loc), p.getContext(), dtype.value(), type);
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
      case (uint32_t)POC::EQ:
      case (uint32_t)POC::LT:
      case (uint32_t)POC::LE:
      case (uint32_t)POCAliases::NE:
      case (uint32_t)POCAliases::GE:
      case (uint32_t)POCAliases::GT:
      case (uint32_t)POCAliases::NOT_IN:
      case (uint32_t)POC::IN:
        // Comparisons default to index type for their operand, since their
        // result is always `i1`.
        operandType = p.getBuilder().getIndexType();
        break;
      case (uint32_t)POC::GET_DTYPE:
        // The `dtype` operator always has an `mlirtype` operand.
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
      opcode = (uint32_t)POC::IN;
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

  // If this parameter is a type, parse it as such.
  if (type.isa<MLIRTypeType>()) {
    Type result;
    if (p.parseType(result))
      return failure();
    // We always parse this as a parameterized type, but the builder will form
    // a concrete type if there are no type parameters in it.  We could add
    // specific syntax to differentiate them if there is a reason to.
    value = ParameterizedTypeConstantAttr::get(result);
    return success();
  }

  // If this is a SignatureType, we expect a symbol name.  We need special
  // parsing logic here because FlatSymbolRefAttr isn't a TypedAttr.
  if (auto signatureType = type.dyn_cast<SignatureType>()) {
    FlatSymbolRefAttr symbol;
    if (p.parseAttribute(symbol, Type()))
      return failure();
    value = SymbolConstantAttr::get(symbol, signatureType);
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
  if (opcode == POC::IN || opcode == POC::EQ || opcode == POC::LT ||
      opcode == POC::LE)
    printColonTypeOrIndexPrefix(os, operands[0].getType());

  switch (opcode) {
  default:
    // operand-list ::= expr (`,` expr)*
    llvm::interleaveComma(
        operands, os, [&](TypedAttr operand) { printParamValue(operand, os); });
    break;
  case POC::IN:
    // operand-list ::= expr `,` `[` (expr (`,` expr)*)? `]`
    printParamValue(operands[0], os);
    os << ", [";
    llvm::interleaveComma(operands.drop_front(), os, [&](TypedAttr operand) {
      printParamValue(operand, os);
    });
    os << "]";
    break;
  }
}

/// Convert a parameter value to a string when in a context that knows it is
/// dealing with a parameter specifically.  This utilize syntactic shortcuts to
/// make the printed syntax easier to grok.
void KGEN::printParamValue(TypedAttr value, raw_ostream &os) {
  if (auto declRef = value.dyn_cast<ParamDeclRefAttr>()) {
    printParamName(declRef.getName(), os);
    return;
  }

  // If this is a type constant, print it as a bare type.
  if (auto typeConstant = value.dyn_cast<TypeConstantAttr>()) {
    os << typeConstant.getValue();
    return;
  }

  // If this is a dtype constant with simple syntax, we can print it as a
  // keyword.
  if (auto dtypeConstant = value.dyn_cast<DTypeConstantAttr>()) {
    auto eltType = dtypeConstant.getDType();
    std::string stringRep = eltType.getAsString();
    // Don't allow things like complex<f64>.  We can extend this in the future
    // if there is a reason to of course.
    if (!StringRef(stringRep).contains('<')) {
      os << stringRep;
      return;
    }
  }

  // Symbol constants print as just the symbol.
  if (auto symbolConstant = value.dyn_cast<SymbolConstantAttr>()) {
    os << symbolConstant.getSymbol();
    return;
  }

  // Handle expressions.
  if (auto expr = value.dyn_cast<ParamOperatorAttr>()) {
    auto printExpr = [&](StringRef opcode, ArrayRef<TypedAttr> operands) {
      os << opcode << '(';
      printOperatorOperands(os, expr.getOpcode(), operands);
      os << ')';
    };

    // If this is a inverted boolean sugar, handle it.
    if (expr.getOpcode() == POC::Xor && expr.getType().isSignlessInteger(1) &&
        expr.getNumOperands() == 2 && expr.getOperand(1).isa<IntegerAttr>()) {
      if (auto invertedExpr =
              expr.getOperand(0).dyn_cast<ParamOperatorAttr>()) {
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
  if (auto intAttr = value.dyn_cast<IntegerAttr>())
    if (intAttr.getType().isSignlessInteger(1)) {
      os << (intAttr.getValue().isZero() ? 0 : 1);
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
  result.addAttribute("constraints", ConstraintArrayAttr::get(
                                         parser.getContext(), constraints));
  return success();
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

  // Parse visibility. If none is provided, use private by default.
  if (failed(mlir::impl::parseOptionalVisibilityKeyword(parser,
                                                        result.attributes)))
    result.addAttribute(SymbolTable::getVisibilityAttrName(),
                        parser.getBuilder().getStringAttr("private"));

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
      parseOptionalConstraints(parser, result, opKind))
    return failure();

  result.addAttribute("paramDecls", inputParamDecls);
  result.addAttribute("resultParamTypes", resultParamTypes);

  SmallVector<Type> argTypes;
  argTypes.reserve(entryArgs.size());
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);
  Type type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));

  // If function attributes are present, parse them.
  NamedAttrList parsedAttributes;
  llvm::SMLoc attributeDictLocation = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(parsedAttributes))
    return failure();

  // If this is a generator, see if it is an implementation of a generator
  // interface.
  if ((opKind == GeneratorOrFuncKind::generator ||
       opKind == GeneratorOrFuncKind::hlgenerator) &&
      succeeded(parser.parseOptionalKeyword("implements"))) {
    ::mlir::FlatSymbolRefAttr implementsAttr;
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
  auto *body = result.addRegion();

  // If this is a generator interface, no body block is allowed.
  if (opKind == GeneratorOrFuncKind::interface)
    return success();

  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseRegion(*body, entryArgs,
                         /*enableNameShadowing=*/false))
    return failure();

  // Function body was parsed, make sure its not empty.
  if (body->empty())
    return parser.emitError(loc, "expected non-empty function body");

  return success();
}

/// Print a constraint list for a generator or interface.
static void printConstraints(KGENDeclInterface decl, OpAsmPrinter &p) {
  ArrayRef<ConstraintAttr> constraints = decl.getConstraints();
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

void KGEN::printGeneratorOrFunc(OpAsmPrinter &p, mlir::FunctionOpInterface op) {
  using namespace mlir::function_interface_impl;

  // TODO: KGENDeclInterface should inherit from FunctionOpInterface.
  auto opDecl = cast<KGENDeclInterface>((Operation *)op);

  // Print the operation and the function name.
  auto funcName =
      op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  p << ' ';

  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();
  if (auto visibility = op->getAttrOfType<StringAttr>(visibilityAttrName))
    if (visibility.getValue() != "private")
      p << visibility.getValue() << ' ';
  p.printSymbolName(funcName);
  printOptionalParameterSpec(p.getStream(), opDecl.getParamDeclsAttr(),
                             opDecl.getResultParamTypesAttr());

  ArrayRef<Type> argTypes = op.getArgumentTypes();
  ArrayRef<Type> resultTypes = op.getResultTypes();
  printFunctionSignature(p, op, argTypes, /*isVariadic=*/false, resultTypes);
  printFunctionAttributes(p, op, argTypes.size(), resultTypes.size(),
                          GeneratorOp::getAttributeNames());
  printConstraints(opDecl, p);

  // If this is a generator implementing a generator.interface, include the
  // symbol for the generator interface.
  if (auto implementsAttr =
          op->getAttrOfType<FlatSymbolRefAttr>("implements")) {
    p.printNewline();
    p << "  implements " << implementsAttr;
  }

  p << ' ';
  if (!op.getBody().empty()) {
    p.printRegion(op.getBody(), /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

/// Verify that a list of parameter declarations from a generator or func
/// matches those of an interface.  This produces an error diagnostic and
/// returns failure when a problem is detected, or returns true if everything is
/// ok.
static ParseResult verifyParameterList(ParamDeclArrayAttr originatorParamDecls,
                                       ParamDeclArrayAttr targetParamDecls,
                                       const char *originatorName,
                                       Location originatorLoc,
                                       const char *targetName,
                                       Location targetLoc,
                                       const char *parameterKind) {

  auto getParamDeclName = [](ParamDeclArrayAttr decls) {
    return llvm::map_range(decls.getValue(), [](Attribute value) -> StringAttr {
      return value.cast<ParamDeclAttr>().getName();
    });
  };
  auto getParamDeclType = [](ParamDeclArrayAttr decls) {
    return llvm::map_range(decls.getValue(), [](Attribute value) -> Type {
      return value.cast<ParamDeclAttr>().getType();
    });
  };

  if (verifyMatchingLists(getParamDeclName(originatorParamDecls),
                          getParamDeclName(targetParamDecls), originatorName,
                          originatorLoc, targetName, targetLoc, parameterKind,
                          "name") ||
      verifyMatchingLists(getParamDeclType(originatorParamDecls),
                          getParamDeclType(targetParamDecls), originatorName,
                          originatorLoc, targetName, targetLoc, parameterKind,
                          "type"))
    return failure();

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
  if (verifyMatchingLists(originatorType.getInputs(), targetType.getInputs(),
                          originatorName, originatorLoc, targetName, targetLoc,
                          "argument", "type") ||
      verifyMatchingLists(originatorType.getResults(), targetType.getResults(),
                          originatorName, originatorLoc, targetName, targetLoc,
                          "result", "type") ||
      verifyParameterList(originatorSignature.getInputParams(),
                          targetSignature.getInputParams(), originatorName,
                          originatorLoc, targetName, targetLoc,
                          "input parameter") ||
      verifyMatchingLists(originatorSignature.getResultParamTypes(),
                          targetSignature.getResultParamTypes(), originatorName,
                          originatorLoc, targetName, targetLoc,
                          "result parameter", "type"))
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
