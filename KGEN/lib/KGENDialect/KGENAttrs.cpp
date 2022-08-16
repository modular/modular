//===- KGENAttrs.cpp - Implement KGEN attributes --------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "Support/DType.h"
#include "Support/LLVMCompilerForwardDecls.h"
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

/// Given an arbitrary MLIR operation, classify it into a declaration kind or
/// return None if unknown.
Optional<GeneratorOrKernelKind> KGEN::classifyDecl(Operation *op) {
  if (isa<KernelOp>(op))
    return GeneratorOrKernelKind::kernel;
  if (isa<GeneratorOp>(op))
    return GeneratorOrKernelKind::generator;

  if (isa<GeneratorInterfaceOp>(op))
    return GeneratorOrKernelKind::interface;
  // Classify hlkgen.generator even though kgen cannot depend on hlkgen libs.
  if (op->getName().getStringRef() == "hlkgen.generator")
    return GeneratorOrKernelKind::hlgenerator;
  return {};
}

static OptionalParseResult parseOptionalColonType(AsmParser &parser,
                                                  Type &type) {
  if (failed(parser.parseOptionalColon()))
    return None;

  // In addition to standard types, we support 'dtype' as a sugared form of
  // !kgen.dtype.
  if (succeeded(parser.parseOptionalKeyword("dtype"))) {
    type = parser.getBuilder().getType<DTypeType>();
    return OptionalParseResult(LogicalResult::success());
  }

  return parser.parseType(type);
}
static void printColonTypeOrIndexPrefix(AsmPrinter &p, Type type);

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

/// Parse a parameter name. This is the corresponding parser to
/// `printParamName`.
static ParseResult parseParamName(AsmParser &p, FailureOr<StringAttr> &result) {
  std::string name;
  if (p.parseKeywordOrString(&name))
    return failure();
  result = StringAttr::get(p.getContext(), name);
  return success();
}

#define GET_ATTRDEF_CLASSES
#include "KGEN/KGENDialect/KGENAttrs.cpp.inc"

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
// ArrayOfAttrsAttr
//===----------------------------------------------------------------------===//

void ParamDeclArrayAttr::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  for (ParamDeclAttr value : getValue())
    walkAttrsFn(value);
}

void ParamBindArrayAttr::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  for (ParamBindAttr value : getValue())
    walkAttrsFn(value);
}

Attribute ParamDeclArrayAttr::replaceImmediateSubElements(
    ArrayRef<Attribute> replAttrs, ArrayRef<Type> replTypes) const {
  return get(getContext(),
             {reinterpret_cast<const ParamDeclAttr *>(replAttrs.begin()),
              replAttrs.size()});
}

Attribute ParamBindArrayAttr::replaceImmediateSubElements(
    ArrayRef<Attribute> replAttrs, ArrayRef<Type> replTypes) const {
  return get(getContext(),
             {reinterpret_cast<const ParamBindAttr *>(replAttrs.begin()),
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

/// Print a parameter name correctly, using a double quoted syntax if it
/// conflicts with an MLIR or KGEN keyword, or a bareword otherwise.
void KGEN::printParamName(AsmPrinter &p, StringRef name) {
  // If this will conflict with a DType keyword, rename it.
  if (succeeded(DType::getFromString(name))) {
    p << '"' << name << '"';
    return;
  }

  // Otherwise, allow MLIR to decide if the name will conflict with its keywords
  // and avoid it if so.
  p.printKeywordOrString(name);
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

enum class POCAliases : uint32_t {
  // The builtin opcodes have 0...127.
  FIRST_PSEUDO = 128,
  NOT,
  NE, // !(==)
  GT, // !(<)
  GE, // !(<=)

  // This is an unknown opcode name.
  kInvalid,
};

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
    if (isBareword) {
      auto dtype = DType::getFromString(keyword);
      if (succeeded(dtype)) {
        value = DTypeConstantAttr::getChecked(
            p.getEncodedSourceLoc(loc), p.getContext(), dtype.value(), type);
        return success(value != Attribute());
      }
    }

    // Just a bareword with no trailing `(`?  Must be a parameter reference.
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
  // define attributes, integers, etc.
  return p.parseAttribute(value, type);
}

static void printOperatorOperands(AsmPrinter &p, POC opcode,
                                  ArrayRef<TypedAttr> operands) {
  // If this is a comparison and the elements are not index type, print the
  // type explicitly.
  if (opcode == POC::IN || opcode == POC::EQ || opcode == POC::LT ||
      opcode == POC::LE)
    printColonTypeOrIndexPrefix(p, operands[0].getType());

  switch (opcode) {
  default:
    // operand-list ::= expr (`,` expr)*
    llvm::interleaveComma(
        operands, p, [&](TypedAttr operand) { printParamValue(p, operand); });
    break;
  case POC::IN:
    // operand-list ::= expr `,` `[` (expr (`,` expr)*)? `]`
    printParamValue(p, operands[0]);
    p << ", [";
    llvm::interleaveComma(operands.drop_front(), p, [&](TypedAttr operand) {
      printParamValue(p, operand);
    });
    p << "]";
    break;
  }
}

/// When in a context that knows it is dealing with a parameter specifically,
/// utilize syntactic shortcuts to make the printed syntax easier to grok.
void KGEN::printParamValue(AsmPrinter &p, TypedAttr value) {
  if (auto declRef = value.dyn_cast<ParamDeclRefAttr>()) {
    printParamName(p, declRef.getName());
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
      p << stringRep;
      return;
    }
  }

  // Handle expressions.
  if (auto expr = value.dyn_cast<ParamOperatorAttr>()) {
    auto printExpr = [&](StringRef opcode, ArrayRef<TypedAttr> operands) {
      p << opcode << '(';
      printOperatorOperands(p, expr.getOpcode(), operands);
      p << ')';
    };

    // If this is a inverted boolean sugar, handle it.
    if (expr.getOpcode() == POC::Xor && expr.getType().isSignlessInteger(1) &&
        expr.getOperands().size() == 2 &&
        expr.getOperands()[1].isa<IntegerAttr>()) {
      if (auto invertedExpr =
              expr.getOperands()[0].dyn_cast<ParamOperatorAttr>()) {
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
      return printExpr("not", expr.getOperands()[0]);
    }

    return printExpr(stringifyEnum(expr.getOpcode()), expr.getOperands());
  }

  // If this is an i1 integer attr, print it as zero or one; not true/false
  // keywords.  This simplifies the keyword processing logic.
  if (auto intAttr = value.dyn_cast<IntegerAttr>()) {
    if (intAttr.getValue().getBitWidth() == 1) {
      p << (int)intAttr.getValue().getZExtValue();
      return;
    }
  }

  p.printAttributeWithoutType(value);
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

  // Handle other special cases for parameters here: we support `dtype` as a
  // bareword for `!kgen.dtype`.
  if (type.isa<DTypeType>())
    p << "dtype";
  else
    p << type;
}

/// print `:<type> ` or elide it entirely if type is an `index` type.
static void printColonTypeOrIndexPrefix(AsmPrinter &p, Type type) {
  // Index type is the default so it doesn't print.
  if (type.isIndex())
    return;
  p << ':';

  // Handle other special cases for parameters here: we support `dtype` as a
  // bareword for `!kgen.dtype`.
  if (type.isa<DTypeType>())
    p << "dtype";
  else
    p << type;
  p << ' ';
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

  // All expressions are "less than" a constant, since they appear on the right.
  // We handle integers and dtypes consistently here, they can never occur in
  // the same expression, since they have different types.
  if (auto intRhs = rhs.dyn_cast<IntegerAttr>()) {
    auto intLhs = lhs.dyn_cast<IntegerAttr>();
    return !intLhs || intLhs.getValue().slt(intRhs.getValue());
  } else if (auto dtypeRhs = rhs.dyn_cast<DTypeConstantAttr>()) {
    auto dtypeLhs = lhs.dyn_cast<DTypeConstantAttr>();
    return !dtypeLhs ||
           dtypeLhs.getDType().getValue() < dtypeRhs.getDType().getValue();
  }
  if (lhs.isa<IntegerAttr, DTypeConstantAttr>())
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
      APInt c1 = operands.pop_back_val().cast<IntegerAttr>().getValue();
      APInt c2 = operands.pop_back_val().cast<IntegerAttr>().getValue();
      auto resultConstant = IntegerAttr::get(type, calculateFn(c1, c2));
      operands.push_back(resultConstant);
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
      auto result = (lhs.getType().isSignedInteger() ? signedFn : unsignedfn)(
          lhs.getValue(), rhs.getValue());
      return IntegerAttr::get(lhs.getType(), result);
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
      bool result = compareFn(lhs.getValue(), rhs.getValue());
      return BoolAttr::get(rhs.getContext(), result);
    }
  return {};
}

static Attribute simplifyShl(SmallVectorImpl<TypedAttr> &operands) {
  // Canonicalize `x << cst` => `x * (1<<cst)` to compose correctly with
  // add/mul canonicalization (also handles constant folding).
  if (auto rhs = operands[1].dyn_cast<IntegerAttr>()) {
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

/// Simplifies an `in` (also `in:dtype`) operator.  We know the all the operands
/// have the same type.
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
  // If all operands are constants and a match was not found, then definitively
  // fold to false.
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
  // All operands must have the same type.  The result type is usually the same
  // as the operands, but is i1 for comparisons (overridden below).
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
// DTypeConstantAttr
//===----------------------------------------------------------------------===//

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
  llvm::report_fatal_error("unhandled DType");
}

//===----------------------------------------------------------------------===//
// Parameter Helper Functions
//===----------------------------------------------------------------------===//

/// Return true if the attribute is a valid parameter expression.
bool KGEN::isValidParameterExpr(Attribute value) {
  // Leaf constants and references to parameter declarations are valid.
  if (value.isa<IntegerAttr, FloatAttr, DTypeConstantAttr, ParamDeclRefAttr>())
    return true;

  // Expressions composed of other expressions are valid.
  if (auto expr = value.dyn_cast<ParamOperatorAttr>()) {
    return llvm::all_of(expr.getOperands(), [](Attribute operand) -> bool {
      return isValidParameterExpr(operand);
    });
  }

  // Nothing else is.
  return false;
}

/// Return the `paramDecls` array of ParamDeclAttr values if the specified
/// operation has it, or an empty array otherwise.
ArrayRef<ParamDeclAttr> KGEN::getParamDecls(Operation *op) {
  if (auto paramDeclsArray =
          op->getAttrOfType<ParamDeclArrayAttr>("paramDecls"))
    return paramDeclsArray.getValue();
  return {};
}

/// Given a kernel, generator, or generator interface operation, return an array
/// of `ParamDeclAttr`s for the inputs and the array of `ParamDeclAttr`s for the
/// result parameters.  A concrete kernel will always never have input params.
std::pair<ArrayRef<ParamDeclAttr>, ArrayRef<ParamDeclAttr>>
KGEN::getDeclParameterInfo(Operation *decl) {
  assert(classifyDecl(decl).has_value() && "unknown declaration");
  ArrayRef<ParamDeclAttr> declParams;
  ArrayRef<ParamDeclAttr> resultParams;
  // Kernels never have input parameters, but they can have output parameters.
  if (!isa<KernelOp>(decl))
    declParams = getParamDecls(decl);
  if (auto resultAttr =
          decl->getAttrOfType<ParamDeclArrayAttr>("resultParamDecls"))
    resultParams = resultAttr.getValue();
  return std::make_pair(declParams, resultParams);
}

SmallVector<std::pair<Attribute, StringAttr>>
KGEN::getDeclConstraints(Operation *decl) {
  SmallVector<std::pair<Attribute, StringAttr>> result;
  // Kernels never have constraints.
  if (isa<KernelOp>(decl))
    return result;

  // Must be a generator or interface.
  assert(classifyDecl(decl).has_value() && "unknown declaration");
  auto exprs = decl->getAttrOfType<ArrayAttr>("constraints").getValue();
  auto messages = decl->getAttrOfType<ArrayAttr>("constraintMessages")
                      .getAsRange<StringAttr>();
  for (auto [expr, message] : llvm::zip(exprs, messages))
    result.push_back({expr, message});
  return result;
}

//===----------------------------------------------------------------------===//
// DTypeConstantAttr
//===----------------------------------------------------------------------===//

DTypeConstantAttr DTypeConstantAttr::get(MLIRContext *ctx, DType dtype) {
  return get(ctx, dtype, DTypeType::get(ctx));
}
