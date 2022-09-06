//===- KGENAttrs.cpp ------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/ML/DType.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SubElementInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace M::KGEN;
using mlir::OptionalParseResult;

// Provide implementations for the enums we use.
#include "KGEN/KGENDialect/KGENEnums.cpp.inc"

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

static OptionalParseResult parseOptionalColonType(AsmParser &parser,
                                                  Type &type) {
  if (failed(parser.parseOptionalColon()))
    return None;

  // In addition to standard types, we support 'type' as a sugared form of
  // !kgen.type.
  if (succeeded(parser.parseOptionalKeyword("type"))) {
    type = parser.getBuilder().getType<MLIRTypeType>();
    return OptionalParseResult(LogicalResult::success());
  }

  // In addition to standard types, we support 'dtype' as a sugared form of
  // !kgen.dtype.
  if (succeeded(parser.parseOptionalKeyword("dtype"))) {
    type = parser.getBuilder().getType<DTypeType>();
    return OptionalParseResult(LogicalResult::success());
  }

  return parser.parseType(type);
}

static void printColonTypeOrIndexPrefix(raw_ostream &os, Type type);

//===----------------------------------------------------------------------===//
// ODS Boilerplate
//===----------------------------------------------------------------------===//

namespace mlir {
/// Parse an opcode.
template <>
struct FieldParser<POC> {
  static FailureOr<POC> parse(AsmParser &parser) {
    StringRef value;
    if (parser.parseKeyword(&value))
      return failure();
    auto result = symbolizePOC(value);
    if (result.has_value())
      return *result;
    return failure();
  }
};

/// Parse a dtype.
template <>
struct FieldParser<DType> {
  static FailureOr<DType> parse(AsmParser &parser) {
    StringRef value;
    if (parser.parseKeyword(&value))
      return failure();
    return DType::getFromString(value);
  }
};

} // namespace mlir

//===----------------------------------------------------------------------===//
// KGENDialect attribute support
//===----------------------------------------------------------------------===//

void KGENDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "KGEN/KGENDialect/KGENAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Attribute implementations
//===----------------------------------------------------------------------===//

void ConcreteTypeConstantAttr::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkTypesFn(getValue());
}

Attribute ConcreteTypeConstantAttr::replaceImmediateSubElements(
    ArrayRef<Attribute> replAttrs, ArrayRef<Type> replTypes) const {
  assert(replAttrs.empty() && replTypes.size() == 1);
  return get(replTypes[0]);
}

void ParameterizedTypeConstantAttr::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkTypesFn(getValue());
}

Attribute ParameterizedTypeConstantAttr::replaceImmediateSubElements(
    ArrayRef<Attribute> replAttrs, ArrayRef<Type> replTypes) const {
  assert(replAttrs.empty() && replTypes.size() == 1);
  // NOTE: This will automatically convert to ConcreteTypeConstantAttr if the
  // subtype is non-parametric.
  return get(replTypes[0]);
}

void ParamDeclArrayAttr::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  for (ParamDeclAttr value : getValue())
    walkAttrsFn(value);
}

Attribute ParamDeclArrayAttr::replaceImmediateSubElements(
    ArrayRef<Attribute> replAttrs, ArrayRef<Type> replTypes) const {
  return get(getContext(),
             {reinterpret_cast<const ParamDeclAttr *>(replAttrs.begin()),
              replAttrs.size()});
}

void ParamBindArrayAttr::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  for (ParamBindAttr value : getValue())
    walkAttrsFn(value);
}

Attribute ParamBindArrayAttr::replaceImmediateSubElements(
    ArrayRef<Attribute> replAttrs, ArrayRef<Type> replTypes) const {
  return get(getContext(),
             {reinterpret_cast<const ParamBindAttr *>(replAttrs.begin()),
              replAttrs.size()});
}

void ConstraintArrayAttr::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  for (ConstraintAttr value : getValue())
    walkAttrsFn(value);
}

Attribute ConstraintArrayAttr::replaceImmediateSubElements(
    ArrayRef<Attribute> replAttrs, ArrayRef<Type> replTypes) const {
  return get(getContext(),
             {reinterpret_cast<const ConstraintAttr *>(replAttrs.begin()),
              replAttrs.size()});
}

//===----------------------------------------------------------------------===//
// "Pretty" parameter printing and parsing
//===----------------------------------------------------------------------===//

// Parameters are complex nested expressions.  While they have a generic
// printing syntax that is supported in full generality, they often appear in
// tightly controlled situations, e.g. in return operations, in types, or when
// invoking a generator. In these cases we can use a much nicer and more compact
// syntax so we as compiler engineers don't go bonkers looking at IR dumps.
/// Print a parameter value that is known to be an index type.

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
  std::string value;
  if (p.parseKeywordOrString(&value))
    return failure();
  name = StringAttr::get(p.getContext(), value);
  return success();
}

ParseResult KGEN::parseParamName(AsmParser &p, FailureOr<StringAttr> &name) {
  name.emplace();
  return parseParamName(p, *name);
}

void KGEN::printParamName(StringRef name, raw_ostream &os) {
  // If this will conflict with a DType keyword or isn't a legal MLIR name,
  // then we need quotes.
  bool needsQuotes =
      succeeded(DType::getFromString(name)) || !isLegalMLIRIdentifier(name);
  if (needsQuotes)
    os << '"';
  os << name;
  if (needsQuotes)
    os << '"';
}

/// Print a parameter name correctly, using a double quoted syntax if it
/// conflicts with an MLIR or KGEN keyword, or a bareword otherwise.
void KGEN::printParamName(AsmPrinter &p, StringRef name) {
  printParamName(name, p.getStream());
}

/// Parse a bareword (a keyword in MLIR terminology) or double quoted string
/// into "result", returning "isBareword=true" in the former case and false in
/// the later case.
static ParseResult parseOptionalParamKeywordOrString(AsmParser &p,
                                                     std::string *result,
                                                     bool &isBareWord) {
  StringRef keyword;
  if (succeeded(p.parseOptionalKeyword(&keyword))) {
    isBareWord = true;
    *result = keyword.str();
    return success();
  }
  isBareWord = false;
  return p.parseOptionalString(result);
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

  // keyword are implicitly parameter declaration references or the start of
  // a expression in function form.
  std::string keyword;
  bool isBareword;
  if (succeeded(parseOptionalParamKeywordOrString(p, &keyword, isBareword))) {
    // If this is a KGEN keyword (a bareword with a known identifier), process
    // it.
    if (isBareword && type.isa<DTypeType>()) {
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

  // Otherwise, we support other typed attributes as well, including dialect
  // define attributes, integers, etc.
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

/// Print a floating point value in a way that the parser will be able to
/// round-trip losslessly.
static void printFloatValue(const APFloat &apValue, raw_ostream &os) {
  // We would like to output the FP constant value in exponential notation,
  // but we cannot do this if doing so will lose precision.  Check here to
  // make sure that we only output it in exponential format if we can parse
  // the value back and get the same value.
  bool isInf = apValue.isInfinity();
  bool isNaN = apValue.isNaN();
  if (!isInf && !isNaN) {
    SmallString<128> strValue;
    apValue.toString(strValue, /*FormatPrecision=*/6, /*FormatMaxPadding=*/0,
                     /*TruncateZero=*/false);

    // Check to make sure that the stringized number is not some string like
    // "Inf" or NaN, that atof will accept, but the lexer will not.  Check
    // that the string matches the "[-+]?[0-9]" regex.
    assert(((strValue[0] >= '0' && strValue[0] <= '9') ||
            ((strValue[0] == '-' || strValue[0] == '+') &&
             (strValue[1] >= '0' && strValue[1] <= '9'))) &&
           "[-+]?[0-9] regex does not match!");

    // Parse back the stringized version and check that the value is equal
    // (i.e., there is no precision loss).
    if (APFloat(apValue.getSemantics(), strValue).bitwiseIsEqual(apValue)) {
      os << strValue;
      return;
    }

    // If it is not, use the default format of APFloat instead of the
    // exponential notation.
    strValue.clear();
    apValue.toString(strValue);

    // Make sure that we can parse the default form as a float.
    if (strValue.str().contains('.')) {
      os << strValue;
      return;
    }
  }

  // Print special values in hexadecimal format. The sign bit should be included
  // in the literal.
  SmallVector<char, 16> str;
  APInt apInt = apValue.bitcastToAPInt();
  apInt.toString(str, /*Radix=*/16, /*Signed=*/false,
                 /*formatAsCLiteral=*/true);
  os << str;
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
  if (auto typeConstant = value.dyn_cast<ConcreteTypeConstantAttr>()) {
    os << typeConstant.getValue();
    return;
  }
  if (auto typeConstant = value.dyn_cast<ParameterizedTypeConstantAttr>()) {
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
  if (auto intAttr = value.dyn_cast<IntegerAttr>()) {
    // Make sure 'index' values are printed as signed.
    bool isUnsigned = intAttr.getType().isUnsignedInteger() ||
                      intAttr.getType().isSignlessInteger(1);
    intAttr.getValue().print(os, !isUnsigned);
    return;
  }

  if (auto floatAttr = value.dyn_cast<FloatAttr>()) {
    printFloatValue(floatAttr.getValue(), os);
    return;
  }

  os << value;
}

/// Return the string form for an attribute value that is printed in a <>
/// context in the .mlir file.
std::string KGEN::getParamAsString(TypedAttr value) {
  SmallVector<char, 128> result;
  {
    llvm::raw_svector_ostream os(result);
    printParamValue(value, os);
  }
  return std::string(result.data(), result.size());
}

/// When in a context that knows it is dealing with a parameter specifically,
/// utilize syntactic shortcuts to make the printed syntax easier to grok.
void KGEN::printParamValue(AsmPrinter &p, TypedAttr value, Type type) {
  printParamValue(value, p.getStream());
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

  // Handle other special cases for parameters here.
  if (type.isa<MLIRTypeType>())
    p << "type"; // `type` => `!kgen.type`.
  else if (type.isa<DTypeType>())
    p << "dtype"; // `dtype` => `!kgen.dtype`.
  else
    p << type;
}

/// print `:<type> ` or elide it entirely if type is an `index` type.
static void printColonTypeOrIndexPrefix(raw_ostream &os, Type type) {
  // Index type is the default so it doesn't print.
  if (type.isIndex())
    return;
  os << ':';

  // Handle other special cases for parameters here.
  if (type.isa<MLIRTypeType>())
    os << "type"; // `type` => `!kgen.type`.
  else if (type.isa<DTypeType>())
    os << "dtype"; // `dtype` => `!kgen.dtype`.
  else
    os << type;
  os << ' ';
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
void KGEN::printIndexParamValue(AsmPrinter &p, Attribute value) {
  printParamValue(p, value);
}

/// Parse a parameter value that is known to be an index type.
ParseResult KGEN::parseIndexParamValue(AsmParser &p,
                                       FailureOr<TypedAttr> &value) {
  TypedAttr result;
  if (parseParamValue(p, result, p.getBuilder().getIndexType()))
    return failure();
  value = result;
  return success();
}

/// Print a parameter value that either has an index type or is null (which
/// prints as a `?`).
void KGEN::printOptionalIndexParamValue(AsmPrinter &p, Attribute value) {
  if (!value) {
    p << '?';
    return;
  }
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

//===----------------------------------------------------------------------===//
// ParamDeclRefAttr
//===----------------------------------------------------------------------===//

Attribute ParamDeclRefAttr::parse(AsmParser &p, Type type) {
  if (!type) {
    p.emitError(p.getNameLoc(), "parameter reference requires a type");
    return {};
  }

  std::string name;
  if (p.parseLess() || p.parseKeywordOrString(&name) || p.parseGreater())
    return {};

  return ParamDeclRefAttr::get(name, type);
}

void ParamDeclRefAttr::print(AsmPrinter &p) const {
  p << "<";
  printParamName(p, getName());
  p << ">";
}

//===----------------------------------------------------------------------===//
// ParamBindAttr
//===----------------------------------------------------------------------===//

void ParamBindAttr::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getName());
  walkAttrsFn(getValue());
}

Attribute
ParamBindAttr::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                           ArrayRef<Type> replTypes) const {
  assert(replAttrs.size() == 2 && replTypes.empty());
  return ParamBindAttr::get(replAttrs[0].cast<StringAttr>(), replAttrs[1]);
}

//===----------------------------------------------------------------------===//
// ParamOperatorAttr
//===----------------------------------------------------------------------===//

LogicalResult ParamOperatorAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, POC opcode,
    ArrayRef<TypedAttr> operands, Type type) {
  // All the operand types must match.
  if (!llvm::all_of(operands, [&](auto operand) {
        return operand.getType() == operands.front().getType();
      }))
    return emitError() << "operand type mismatch";

  // Check invariants on the expression.
  switch (opcode) {
  case POC::Add:
  case POC::Mul:
  case POC::And:
  case POC::Or:
  case POC::Xor:
    if (operands.empty())
      return emitError() << stringifyEnum(opcode)
                         << " operator must have at least one operand";
    if (type != operands[0].getType())
      return emitError() << "result type should match operand types";
    // Check the types that are supported.
    if (type.isIndex())
      break; // Index type supported for all of these.
    if (opcode == POC::Xor && type.isSignlessInteger(1))
      break; // i1 types only support xor.
    // TODO: Can support signful fixed width types as needed.
    return emitError() << "operator requires an index type";

  // Binary expressions.
  case POC::Shl:
  case POC::Shr:
  case POC::Div:
  case POC::Mod:
    if (operands.size() != 2)
      return emitError() << stringifyEnum(opcode) << " must have two operands";
    if (type != operands[0].getType())
      return emitError() << "result type should match operand types";
    if (!operands[0].getType().isIndex())
      return emitError() << "operator requires an index type";
    break;
  case POC::EQ:
  case POC::LT:
  case POC::LE:
    if (operands.size() != 2)
      return emitError() << "comparison operators must have two operands";
    if (!operands[0].getType().isIndex() &&
        !operands[0].getType().isa<DTypeType>()) {
      return emitError() << "unsupported comparison type "
                         << operands[0].getType();
    }
    if (!type.isInteger(1))
      return emitError() << "comparisons return i1";

    // Relational operations don't work for dtypes.
    if (opcode != POC::EQ && operands[0].getType().isa<DTypeType>())
      return emitError() << "relational comparisons aren't allowed on dtype's";

    break;
  case POC::IN:
    if (operands.empty())
      return emitError() << "operator requires at least one operand";
    if (!operands[0].getType().isIndex() &&
        !operands[0].getType().isa<DTypeType>()) {
      return emitError() << "unsupported set comparison type "
                         << operands[0].getType();
    }
    if (!type.isInteger(1))
      return emitError() << "comparisons return i1";
    break;
  }
  return success();
}

/// If the specified attribute is a ParamOperatorAttr with the specified opcode,
/// return it.  Otherwise return null.
static ParamOperatorAttr dyn_castPE(POC opcode, Attribute value) {
  if (auto expr = value.dyn_cast<ParamOperatorAttr>())
    if (expr.getOpcode() == opcode)
      return expr;
  return {};
}

/// This implements a < comparison for two operands to an associative operation
/// imposing an ordering upon them.
///
/// The ordering provided puts more complex things to the start of the list,
/// from left to right:
///    expressions :: decl.refs :: constant
///
static bool paramExprOperandSortPredicate(Attribute lhs, Attribute rhs) {
  // Simplify the code below - we never have to care about exactly equal values.
  if (lhs == rhs)
    return false;

  // All non-constant expressions are "less than" a constant, since they appear
  // on the right. We handle all simple constants consistently here: they can
  // never occur in the same expression since they have different types.
  if (isSimpleConstant(rhs)) {
    if (auto intRhs = rhs.dyn_cast<IntegerAttr>()) {
      auto intLhs = lhs.dyn_cast<IntegerAttr>();
      return !intLhs || intLhs.getValue().slt(intRhs.getValue());
    }
    if (auto dtypeRhs = rhs.dyn_cast<DTypeConstantAttr>()) {
      auto dtypeLhs = lhs.dyn_cast<DTypeConstantAttr>();
      return !dtypeLhs ||
             dtypeLhs.getDType().getValue() < dtypeRhs.getDType().getValue();
    }
    if (auto strRhs = rhs.dyn_cast<StringAttr>()) {
      auto strLhs = lhs.dyn_cast<StringAttr>();
      return !strLhs || strLhs.getValue() < strRhs.getValue();
    }
    auto fltRhs = rhs.cast<FloatAttr>();
    auto fltLhs = lhs.dyn_cast<FloatAttr>();
    return !fltLhs || fltLhs.getValue() < fltRhs.getValue();
  }
  if (isSimpleConstant(lhs))
    return false;

  // Next up are named parameters.
  if (auto rhsParam = rhs.dyn_cast<ParamDeclRefAttr>()) {
    // Parameters are sorted lexically w.r.t. each other.
    if (auto lhsParam = lhs.dyn_cast<ParamDeclRefAttr>())
      return lhsParam.getName().getValue() < rhsParam.getName().getValue();
    // They otherwise appear on the right of other things.
    return true;
  }
  if (lhs.isa<ParamDeclRefAttr>())
    return false;

  // The only thing left are nested expressions.
  auto lhsExpr = lhs.cast<ParamOperatorAttr>(),
       rhsExpr = rhs.cast<ParamOperatorAttr>();
  // Sort by the string form of the opcode, e.g. add, .. mul,... then xor.
  if (lhsExpr.getOpcode() != rhsExpr.getOpcode())
    return stringifyPOC(lhsExpr.getOpcode()) <
           stringifyPOC(rhsExpr.getOpcode());

  // If they are the same opcode, then sort by arity: more complex to the left.
  ArrayRef<TypedAttr> lhsOperands = lhsExpr.getOperands(),
                      rhsOperands = rhsExpr.getOperands();
  if (lhsOperands.size() != rhsOperands.size())
    return lhsOperands.size() > rhsOperands.size();

  // We know the two subexpressions are different (they'd otherwise be pointer
  // equivalent) so just go compare all of the elements.
  for (size_t i = 0, e = lhsOperands.size(); i != e; ++i) {
    if (paramExprOperandSortPredicate(lhsOperands[i], rhsOperands[i]))
      return true;
    if (paramExprOperandSortPredicate(rhsOperands[i], lhsOperands[i]))
      return false;
  }

  llvm_unreachable("expressions should never be equivalent");
  return false;
}

/// Given a function_ref from `(APInt,APInt)->T` and two APInt's, compute the
/// result value T and return it.
///
/// Note that this function has special behavior when 'valueTy' (the MLIR type
/// of the two operand values) is 'index' type. In this case, it does extra work
/// to make sure that a 32-bit and 64-bit target will compute the same result
/// using the same approach as the index dialect.  If they differ, this refuses
/// to fold the operation, returning a null IntegerAttr.
template <typename ResultTy>
static IntegerAttr foldBinaryValues(
    const llvm::function_ref<ResultTy(const APInt &, const APInt &)>
        &calculateFn,
    const APInt &lhs, const APInt &rhs, Type valueTy, Type resultTy = {}) {

  // Clients can specify resultTy if it differs from valueTy (e.g. for
  // compares), but not specifying it defaults to the result being the same type
  // as the operands.
  if (!resultTy)
    resultTy = valueTy;

  auto result1 = calculateFn(lhs, rhs);
  if (!valueTy.isa<IndexType>())
    return IntegerAttr::get(resultTy, result1);

  // If this is an index computation, then we just did the 64-bit computation,
  // see what would happen on a 32-bit host.
  assert(lhs.getBitWidth() == 64);

  // We require that the computation satisfy the invariant that:
  //   trunc(f(a, b)) = f(trunc(a), trunc(b))
  auto result2 = calculateFn(lhs.trunc(32), rhs.trunc(32));

  // If not bool result (e.g. a compare), truncate the LHS for our check.
  auto result1test = result1;
  if constexpr (!std::is_same_v<bool, ResultTy>) {
    result1test = result1.trunc(result2.getBitWidth());
  }

  // We can use the full 64-bit folded result if they match, otherwise leave
  // unfolded.
  return result1test == result2 ? IntegerAttr::get(resultTy, result1)
                                : IntegerAttr();
}

/// Given a fully associative variadic integer operation, constant fold any
/// constant operands and move them to the right.  If the whole expression is
/// constant, then return that, otherwise update the operands list.
static Attribute simplifyAssocOp(
    POC opcode, SmallVectorImpl<TypedAttr> &operands,
    llvm::function_ref<APInt(const APInt &, const APInt &)> calculateFn,
    llvm::function_ref<bool(const APInt &)> identityConstantFn,
    llvm::function_ref<bool(const APInt &)> destructiveConstantFn = {}) {
  auto type = operands[0].getType();
  if (operands.size() == 1)
    return operands[0];

  // Flatten any of the same operation into the operand list:
  // `(add x, (add y, z))` => `(add x, y, z)`.
  for (size_t i = 0, e = operands.size(); i != e; ++i) {
    if (auto subexpr = dyn_castPE(opcode, operands[i])) {
      std::swap(operands[i], operands.back());
      operands.pop_back();
      --e;
      --i;
      operands.append(subexpr.getOperands().begin(),
                      subexpr.getOperands().end());
    }
  }

  // Impose an ordering on the operands, pushing subexpressions to the left and
  // constants to the right, with ParamRefs in the middle - but predictably
  // ordered w.r.t. each other.
  llvm::stable_sort(operands, paramExprOperandSortPredicate);

  // Merge any constants, they will appear at the back of the operand list now.
  if (operands.back().isa<IntegerAttr>()) {
    while (operands.size() >= 2 &&
           operands[operands.size() - 2].isa<IntegerAttr>()) {
      APInt c1 = operands[operands.size() - 2].cast<IntegerAttr>().getValue();
      APInt c2 = operands.back().cast<IntegerAttr>().getValue();
      if (auto resultConstant = foldBinaryValues(calculateFn, c1, c2, type)) {
        operands.pop_back();
        operands.pop_back();
        operands.push_back(resultConstant);
      } else {
        // If we couldn't fold the two values, bail.
        break;
      }
    }

    auto resultCst = operands.back().cast<IntegerAttr>();

    // If the resulting constant is the destructive constant (e.g. `x*0`), then
    // return it.
    if (destructiveConstantFn && destructiveConstantFn(resultCst.getValue()))
      return resultCst;

    // Remove the constant back to our operand list if it is the identity
    // constant for this operator (e.g. `x*1`) and there are other operands.
    if (identityConstantFn(resultCst.getValue()) && operands.size() != 1)
      operands.pop_back();
  }

  return operands.size() == 1 ? operands[0] : Attribute();
}

/// Analyze an operand to an add.  If it is a multiplication by a constant (e.g.
/// `(a*b*42)` then split it into the non-constant and the constant portions
/// (e.g. `a*b` and `42`).  Otherwise return the operand as the first value and
/// null as the second (standin for "multiplication by 1").
static std::pair<TypedAttr, TypedAttr> decomposeAddend(TypedAttr operand) {
  if (auto mul = dyn_castPE(POC::Mul, operand))
    if (auto cst = mul.getOperands().back().dyn_cast<IntegerAttr>()) {
      auto nonCst =
          ParamOperatorAttr::get(POC::Mul, mul.getOperands().drop_back());
      return {nonCst, cst};
    }
  return {operand, TypedAttr()};
}

static Attribute getOneOfType(Type type) {
  size_t width = type.isIndex() ? 64 : type.getIntOrFloatBitWidth();
  return IntegerAttr::get(type, APInt(width, 1));
}

static Attribute simplifyAdd(SmallVectorImpl<TypedAttr> &operands) {
  if (auto result = simplifyAssocOp(
          POC::Add, operands, [](auto a, auto b) { return a + b; },
          /*identityCst*/ [](auto cst) { return cst.isZero(); }))
    return result;

  // Canonicalize the add by splitting all addends into their variable and
  // constant factors.
  SmallVector<std::pair<TypedAttr, TypedAttr>> decomposedOperands;
  llvm::SmallDenseSet<TypedAttr> nonConstantParts;
  for (auto &op : operands) {
    decomposedOperands.push_back(decomposeAddend(op));

    // Keep track of non-constant parts we've already seen.  If we see multiple
    // uses of the same value, then we can fold them together with a multiply.
    // This handles things like `(a+b+a)` => `(a*2 + b)` and `(a*2 + b + a)` =>
    // `(a*3 + b)`.
    if (!nonConstantParts.insert(decomposedOperands.back().first).second) {
      // The thing we multiply will be the common expression.
      TypedAttr mulOperand = decomposedOperands.back().first;

      // Find the index of the first occurrence.
      size_t i = 0;
      while (decomposedOperands[i].first != mulOperand)
        ++i;
      // Remove both occurrences from the operand list.
      operands.erase(operands.begin() + (&op - &operands[0]));
      operands.erase(operands.begin() + i);

      auto type = mulOperand.getType();
      auto c1 = decomposedOperands[i].second,
           c2 = decomposedOperands.back().second;
      // Fill in missing constant multiplicands with 1.
      if (!c1)
        c1 = getOneOfType(type);
      if (!c2)
        c2 = getOneOfType(type);
      // Re-add the "a"*(c1+c2) expression to the operand list and
      // re-canonicalize.
      auto constant = ParamOperatorAttr::get(POC::Add, c1, c2);
      auto mulCst = ParamOperatorAttr::get(POC::Mul, mulOperand, constant);
      operands.push_back(mulCst);
      return ParamOperatorAttr::get(POC::Add, operands);
    }
  }

  return {};
}

static Attribute simplifyMul(SmallVectorImpl<TypedAttr> &operands) {
  if (auto result = simplifyAssocOp(
          POC::Mul, operands, [](auto a, auto b) { return a * b; },
          /*identityCst*/ [](auto cst) { return cst.isOne(); },
          /*destructiveCst*/ [](auto cst) { return cst.isZero(); }))
    return result;

  // We always build a sum-of-products representation, so if we see an addition
  // as a subexpr, we need to pull it out: (a+b)*c*d ==> (a*c*d + b*c*d).
  for (size_t i = 0, e = operands.size(); i != e; ++i) {
    if (auto addSubExpr = dyn_castPE(POC::Add, operands[i])) {
      // Pull the `c*d` operands out - it is whatever operands remain after
      // removing the `(a+b)` term.
      operands.erase(operands.begin() + i);

      // Build each add operand.
      SmallVector<TypedAttr> addOperands;
      for (auto addOperand : addSubExpr.getOperands()) {
        operands.push_back(addOperand);
        addOperands.push_back(ParamOperatorAttr::get(POC::Mul, operands));
        operands.pop_back();
      }
      // Canonicalize and form the add expression.
      return ParamOperatorAttr::get(POC::Add, addOperands);
    }
  }

  return {};
}

static Attribute simplifyAnd(SmallVectorImpl<TypedAttr> &operands) {
  return simplifyAssocOp(
      POC::And, operands, [](auto a, auto b) { return a & b; },
      /*identityCst*/ [](auto cst) { return cst.isAllOnes(); },
      /*destructiveCst*/ [](auto cst) { return cst.isZero(); });
}

static Attribute simplifyOr(SmallVectorImpl<TypedAttr> &operands) {
  return simplifyAssocOp(
      POC::Or, operands, [](auto a, auto b) { return a | b; },
      /*identityCst*/ [](auto cst) { return cst.isZero(); },
      /*destructiveCst*/ [](auto cst) { return cst.isAllOnes(); });
}

static Attribute simplifyXor(SmallVectorImpl<TypedAttr> &operands) {
  return simplifyAssocOp(
      POC::Xor, operands, [](auto a, auto b) { return a ^ b; },
      /*identityCst*/ [](auto cst) { return cst.isZero(); });
}

/// Given a binary function, if the two operands are known constant integers,
/// use the specified fold functions to compute the result.
static Attribute
foldBinaryOp(ArrayRef<TypedAttr> operands,
             llvm::function_ref<APInt(const APInt &, const APInt &)> unsignedfn,
             llvm::function_ref<APInt(const APInt &, const APInt &)> signedFn) {
  assert(operands.size() == 2 && "binary operator always has two operands");
  if (auto lhs = operands[0].dyn_cast<IntegerAttr>())
    if (auto rhs = operands[1].dyn_cast<IntegerAttr>()) {
      const auto &fn =
          (lhs.getType().isSignedInteger() ? signedFn : unsignedfn);
      if (auto resultConstant = foldBinaryValues(fn, lhs.getValue(),
                                                 rhs.getValue(), lhs.getType()))
        return resultConstant;
    }
  return {};
}

/// Folds constants given a comparison function that returns bool.  The client
/// must handle signedness etc.
static Attribute foldCompareOp(
    ArrayRef<TypedAttr> operands,
    llvm::function_ref<bool(const APInt &, const APInt &)> compareFn) {
  assert(operands.size() == 2 && "compare operator always has two operands");
  if (auto lhs = operands[0].dyn_cast<IntegerAttr>())
    if (auto rhs = operands[1].dyn_cast<IntegerAttr>()) {
      if (auto resultConstant = foldBinaryValues(
              compareFn, lhs.getValue(), rhs.getValue(), lhs.getType(),
              IntegerType::get(rhs.getContext(), 1)))
        return resultConstant;
    }
  return {};
}

static Attribute simplifyShl(SmallVectorImpl<TypedAttr> &operands) {
  // Canonicalize `x << cst` => `x * (1<<cst)` to compose correctly with
  // add/mul canonicalization (also handles constant folding).
  if (auto rhs = operands[1].dyn_cast<IntegerAttr>()) {
    // NOTE: This is correct even for index types because an overlong shift will
    // turn the result to zero.
    // FIXME: getOneBitSet asserts the shift amount should be in-range.  We need
    // to check this.
    auto rhsCst = APInt::getOneBitSet(rhs.getValue().getBitWidth(),
                                      rhs.getValue().getZExtValue());
    return ParamOperatorAttr::get(POC::Mul, operands[0],
                                  IntegerAttr::get(rhs.getType(), rhsCst));
  }
  return {};
}

static Attribute simplifyShr(SmallVectorImpl<TypedAttr> &operands) {
  if (auto rhs = operands[1].dyn_cast<IntegerAttr>())
    if (rhs.getValue().isZero())
      return operands[0]; // `x >> 0 = x`.
  // TODO: 0 >> x, -1 >>> x

  // FIXME: Must care about high bits.
  return foldBinaryOp(
      operands, [](auto a, auto b) { return a.lshr(b); },
      [](auto a, auto b) { return a.ashr(b); });
}

static Attribute simplifyDiv(SmallVectorImpl<TypedAttr> &operands) {
  // Implement support for identities like `x/1 = x`.
  if (auto rhs = operands[1].dyn_cast<IntegerAttr>())
    if (rhs.getValue().isOne())
      return operands[0];

  return foldBinaryOp(
      operands, [](auto a, auto b) { return a.udiv(b); },
      [](auto a, auto b) { return a.sdiv(b); });
}

static Attribute simplifyMod(SmallVectorImpl<TypedAttr> &operands) {
  // Implement support for identities like `x%1 = 0`.
  if (auto rhs = operands[1].dyn_cast<IntegerAttr>())
    if (rhs.getValue().isOne())
      return IntegerAttr::get(rhs.getType(), 0);

  return foldBinaryOp(
      operands, [](auto a, auto b) { return a.urem(b); },
      [](auto a, auto b) { return a.srem(b); });
}

static Attribute simplifyEQ(SmallVectorImpl<TypedAttr> &operands) {
  // Make sure parameters are ordered correctly.
  llvm::stable_sort(operands, paramExprOperandSortPredicate);

  if (auto lhs = operands[0].dyn_cast<DTypeConstantAttr>())
    if (auto rhs = operands[1].dyn_cast<DTypeConstantAttr>())
      return BoolAttr::get(rhs.getContext(), lhs == rhs);

  return foldCompareOp(operands, [](auto a, auto b) { return a == b; });
}

// Simplify the < and <= operations.
static Attribute
simplifyRelationalCompare(POC opcode, SmallVectorImpl<TypedAttr> &operands) {
  // We only support signed arithmetic so far.
  assert(operands[0].getType().isIndex());

  if (auto rhs = operands[1].dyn_cast<IntegerAttr>()) {
    // If this is a `(le x, RHS)` and RHS is a constant, canonicalize to `lt`.
    if (opcode == POC::LE) {
      if (rhs.getValue().isMaxSignedValue()) // x <=s 127 --> TRUE.
        return BoolAttr::get(rhs.getContext(), true);
      return ParamOperatorAttr::get(
          POC::LT, operands[0],
          IntegerAttr::get(rhs.getType(), rhs.getValue() + 1));
    }
    // If this is (x < MAXCST) canonicalize to (x != MAXCST).
    if (rhs.getValue().isMaxSignedValue())
      return ParamOperatorAttr::getNE(operands[0], rhs);
  }

  if (auto lhs = operands[0].dyn_cast<IntegerAttr>()) {
    // (le cst, x) -> !(lt x, cst)
    if (opcode == POC::LE)
      return ParamOperatorAttr::getNot(
          ParamOperatorAttr::get(POC::LT, operands[1], operands[0]));
    // (lt cst, x) -> !(le x, cst)
    return ParamOperatorAttr::getNot(
        ParamOperatorAttr::get(POC::LE, operands[1], operands[0]));
  }

  if (opcode == POC::LT)
    return foldCompareOp(operands, [](auto a, auto b) { return a.slt(b); });
  assert(opcode == POC::LE);
  return foldCompareOp(operands, [](auto a, auto b) { return a.sle(b); });
}

/// Simplifies an `in` (also `in:dtype`) operator.  We know the all the
/// operands have the same type.
static Attribute simplifyIN(SmallVectorImpl<TypedAttr> &operands) {
  TypedAttr lhs = operands[0];
  MutableArrayRef<TypedAttr> trailing =
      llvm::makeMutableArrayRef(operands).drop_front();

  Builder b(lhs.getContext());

  // If there are no trailing operands, fold to false.
  if (trailing.empty())
    return b.getBoolAttr(false);

  // If there is only one trailing operand, canonicalize to an `eq` operator.
  if (trailing.size() == 1)
    return ParamOperatorAttr::get(POC::EQ, operands);

  auto isConst = [](Attribute attr) -> bool {
    return attr.isa<IntegerAttr>() || attr.isa<DTypeConstantAttr>();
  };

  bool allConst = true;
  for (TypedAttr operand : trailing) {
    // Fold to true if a match was found.
    if (lhs == operand)
      return b.getBoolAttr(true);
    allConst &= isConst(operand);
  }
  // If all operands are constants and a match was not found, then
  // definitively fold to false.
  if (allConst && isConst(lhs))
    return b.getBoolAttr(false);

  // Sort and unique the trailing operands.
  llvm::stable_sort(trailing, paramExprOperandSortPredicate);
  SmallVector<TypedAttr> newOperands;
  newOperands.reserve(operands.size());
  newOperands.push_back(lhs);
  SmallPtrSet<Attribute, 4> seenTrailing;
  for (TypedAttr operand : trailing)
    if (seenTrailing.insert(operand).second)
      newOperands.push_back(operand);
  if (newOperands == operands)
    return {};
  return ParamOperatorAttr::get(POC::IN, newOperands);
}

TypedAttr ParamOperatorAttr::get(MLIRContext *context, POC opcode,
                                 ArrayRef<TypedAttr> operandsIn, Type type) {
  auto result = get(opcode, operandsIn);
  assert((!type || type == result.getType()) && "unexpected type");
  return result;
}

TypedAttr
ParamOperatorAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                              MLIRContext *context, POC opcode,
                              ArrayRef<TypedAttr> operandsIn, Type type) {
  if (failed(verify(emitError, opcode, operandsIn, type)))
    return {};
  return get(context, opcode, operandsIn, type);
}

/// Return (not x) which is the same as (xor x, true).  The `operand` value
/// must have type `i1`.
TypedAttr ParamOperatorAttr::getNot(TypedAttr operand) {
  TypedAttr one = BoolAttr::get(operand.getContext(), true);
  return ParamOperatorAttr::get(POC::Xor, {operand, one});
}

TypedAttr ParamOperatorAttr::get(POC opcode, ArrayRef<TypedAttr> operandsIn) {
  assert(!operandsIn.empty() && "Cannot have expr with no operands");
  // All operands must have the same type.  The result type is usually the
  // same as the operands, but is i1 for comparisons (overridden below).
  auto resultType = operandsIn.front().getType();
  assert(llvm::all_of(operandsIn.drop_front(),
                      [&](auto op) { return op.getType() == resultType; }));

  SmallVector<TypedAttr, 4> operands(operandsIn.begin(), operandsIn.end());

  auto *context = operandsIn[0].getContext();

  // Verify and canonicalize parameter expressions.
  Attribute result;
  switch (opcode) {
  case POC::Add:
    result = simplifyAdd(operands);
    break;
  case POC::Mul:
    result = simplifyMul(operands);
    break;
  case POC::And:
    result = simplifyAnd(operands);
    break;
  case POC::Or:
    result = simplifyOr(operands);
    break;
  case POC::Xor:
    result = simplifyXor(operands);
    break;
  case POC::Shl:
    result = simplifyShl(operands);
    break;
  case POC::Shr:
    result = simplifyShr(operands);
    break;
  case POC::Div:
    result = simplifyDiv(operands);
    break;
  case POC::Mod:
    result = simplifyMod(operands);
    break;
  case POC::EQ:
    result = simplifyEQ(operands);
    resultType = IntegerType::get(context, 1);
    break;
  case POC::LT:
  case POC::LE:
    result = simplifyRelationalCompare(opcode, operands);
    resultType = IntegerType::get(context, 1);
    break;
  case POC::IN:
    result = simplifyIN(operands);
    resultType = IntegerType::get(context, 1);
    break;
  }

  // If we folded to an operand, return it.
  if (result)
    return result;

  return Base::get(context, opcode, operands, resultType);
}

void ParamOperatorAttr::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  for (auto operand : getOperands())
    walkAttrsFn(operand);
}

Attribute
ParamOperatorAttr::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                               ArrayRef<Type> replTypes) const {
  assert(!replAttrs.empty() && replTypes.empty());
  if (!llvm::all_of(replAttrs,
                    [](Attribute attr) { return attr.isa<TypedAttr>(); }))
    return nullptr;
  return ParamOperatorAttr::get(
      getOpcode(), {reinterpret_cast<const TypedAttr *>(replAttrs.data()),
                    replAttrs.size()});
}

//===----------------------------------------------------------------------===//
// ConcreteTypeConstantAttr
//===----------------------------------------------------------------------===//

ConcreteTypeConstantAttr ConcreteTypeConstantAttr::get(Type type) {
  auto *ctx = type.getContext();
  assert(!isParameterizedType(type) &&
         "Cannot create a ConcreteTypeConstantAttr with parameterized type");
  return Base::get(ctx, type, MLIRTypeType::get(ctx));
}

//===----------------------------------------------------------------------===//
// ParameterizedTypeConstantAttr
//===----------------------------------------------------------------------===//

TypedAttr ParameterizedTypeConstantAttr::get(Type type) {
  auto *ctx = type.getContext();
  auto typeType = MLIRTypeType::get(ctx);

  if (isParameterizedType(type))
    return Base::get(ctx, type, typeType);
  return ConcreteTypeConstantAttr::Base::get(ctx, type, typeType);
}

//===----------------------------------------------------------------------===//
// DTypeConstantAttr
//===----------------------------------------------------------------------===//

DTypeConstantAttr DTypeConstantAttr::get(MLIRContext *ctx, DType dtype) {
  return get(ctx, dtype, DTypeType::get(ctx));
}

LogicalResult
DTypeConstantAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                          DType dtype, Type type) {
  if (!type || !type.isa<DTypeType>())
    return emitError() << "kgen.dtype.constant requires !kgen.dtype type";
  return success();
}

/// Checks if the DType constant is compatible with the MLIR type.
bool DTypeConstantAttr::isCompatibleWith(Type type) {
  if (!type.isa<IntegerType, FloatType>())
    return false;

  auto eltTy = getDType();
  auto builtinWidth = type.getIntOrFloatBitWidth();
  auto eltWidth = eltTy.getWidthInBits();
  if (eltTy.isBool())
    return type.isa<IntegerType>() && (builtinWidth == 1);
  if (eltTy.isInt())
    return type.isa<IntegerType>() && (builtinWidth == eltWidth);

  switch (eltTy.getValue()) {
  default:
    return type.isa<FloatType>() && builtinWidth == eltWidth;
  // Special cases for bf16, fp16, and tf32.
  case DType::bf16:
    return type.isa<BFloat16Type>();
  case DType::f16:
    return type.isa<Float16Type>();
  case DType::tf32:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// Parameter Helper Functions
//===----------------------------------------------------------------------===//

/// Return the `paramDecls` array of ParamDeclAttr values if the specified
/// operation has it, or an empty array otherwise.
ArrayRef<ParamDeclAttr> KGEN::getParamDecls(Operation *op) {
  if (auto paramDeclsArray =
          op->getAttrOfType<ParamDeclArrayAttr>("paramDecls"))
    return paramDeclsArray.getValue();
  return {};
}

//===----------------------------------------------------------------------===//
// ConstraintAttr
//===----------------------------------------------------------------------===//

/// Parse an optional location or use the current location of the parser.
static ParseResult parseConstraintLoc(AsmParser &parser,
                                      FailureOr<Location> &loc) {
  if (succeeded(parser.parseOptionalComma())) {
    mlir::LocationAttr locAttr;
    if (parser.parseAttribute(locAttr))
      return failure();
    loc.emplace(locAttr);
  } else {
    loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  }
  return success();
}

/// Always print the location.
static void printConstraintLoc(AsmPrinter &printer, Location loc) {
  printer << ", ";
  printer.printAttribute(loc);
}

void ConstraintAttr::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getExpr());
  walkAttrsFn(getMessage());
  walkAttrsFn(getLoc());
}

Attribute
ConstraintAttr::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                            ArrayRef<Type> replTypes) const {
  assert(replTypes.empty() && "constraint has no types");
  assert(replAttrs.size() == 3 && "expected 3 sub-elements");
  return get(replAttrs[0].cast<TypedAttr>(), replAttrs[1].cast<StringAttr>(),
             replAttrs[2].cast<mlir::LocationAttr>());
}

//===----------------------------------------------------------------------===//
// HistogramParam
//===----------------------------------------------------------------------===//

void M::KGEN::printHistogram(AsmPrinter &p,
                             ArrayRef<std::pair<size_t, uint64_t>> histogram) {
  p << "[";
  llvm::interleaveComma(histogram, p, [&](auto elt) {
    p << "(" << elt.first << ", " << elt.second << ")";
  });
  p << "]";
}

FailureOr<SmallVector<std::pair<size_t, uint64_t>>>
M::KGEN::parseHistogram(AsmParser &p) {
  SmallVector<std::pair<size_t, uint64_t>> out;
  auto elementParser = [&]() -> ParseResult {
    size_t size;
    uint64_t weight;
    if (p.parseLParen() || p.parseInteger(size) || p.parseComma() ||
        p.parseInteger(weight) || p.parseRParen())
      return failure();

    out.emplace_back(size, weight);
    return success();
  };

  if (p.parseLSquare() || p.parseCommaSeparatedList(elementParser) ||
      p.parseRSquare())
    return failure();

  return out;
}

//===----------------------------------------------------------------------===//
// InputGenKind(Attr)
//===----------------------------------------------------------------------===//

namespace mlir {
template <>
struct FieldParser<InputGenKind> {
  static FailureOr<InputGenKind> parse(AsmParser &p) {
    // Stash the current location for caret diagnostics.
    llvm::SMLoc currentLoc = p.getCurrentLocation();

    StringRef kw;
    if (p.parseKeyword(&kw))
      return failure();

    auto inputGenKindOr = symbolizeInputGenKind(kw);
    if (!inputGenKindOr)
      return p.emitError(currentLoc) << "unknown InputGenKind '" << kw << "'";

    return *inputGenKindOr;
  }
};
} // namespace mlir

//===----------------------------------------------------------------------===//
// EvalConfigurationAttr
//===----------------------------------------------------------------------===//

void EvalConfigurationAttr::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  for (EvalArgConfigurationAttr v : getValue())
    walkAttrsFn(v);
}

Attribute EvalConfigurationAttr::replaceImmediateSubElements(
    ArrayRef<Attribute> replAttrs, ArrayRef<Type> replTypes) const {
  assert(replTypes.empty() && "constraint has no types");
  assert(replAttrs.size() == getValue().size());
  if (!llvm::all_of(replAttrs, [](Attribute attr) {
        return attr.isa<EvalConfigurationAttr>();
      }))
    return {};

  return get(
      {reinterpret_cast<const EvalArgConfigurationAttr *>(replAttrs.data()),
       replAttrs.size()});
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "KGEN/KGENDialect/KGENAttrs.cpp.inc"
