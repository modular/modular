//===- KGENAttrs.cpp - Implement KGEN attributes --------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "GenericML/Support/TensorEltType.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SubElementInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace M::KGEN;

// Provide implementations for the enums we use.
#include "KGEN/KGENDialect/KGENEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// ODS Boilerplate
//===----------------------------------------------------------------------===//

namespace mlir {
/// Parse an attribute.
template <>
struct FieldParser<PEO> {
  static FailureOr<PEO> parse(AsmParser &parser) {
    StringRef value;
    if (parser.parseKeyword(&value))
      return failure();
    auto result = symbolizePEO(value);
    if (result.hasValue())
      return *result;
    return failure();
  }
};
} // namespace mlir

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
  // If this will conflict with a TensorEltType keyword, rename it.
  if (succeeded(TensorEltType::getFromString(name))) {
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

/// When in a context that knows it is dealing with a parameter specifically,
/// utilize syntactic shortcuts to make the parsed syntax easier to grok.
ParseResult KGEN::parseParamValue(AsmParser &p, Attribute &value, Type type) {
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
      auto dtype = TensorEltType::getFromString(keyword);
      if (succeeded(dtype)) {
        auto cst = p.getBuilder().getI8IntegerAttr(dtype.getValue().getValue());

        value = DTypeConstantAttr::getChecked(p.getEncodedSourceLoc(loc),
                                              cst.getContext(), cst, type);
        return success(value != Attribute());
      }
    }

    // Just a bareword with no trailing `(`?  Must be a parameter reference.
    if (failed(p.parseOptionalLParen())) {
      value = ParamDeclRefAttr::get(keyword, type);
      return success();
    }

    // Otherwise it's a function expression, decode the name as an operation
    // code.
    auto opcode = symbolizePEO(keyword);
    if (!opcode.hasValue())
      return p.emitError(loc, "unknown expression ") << keyword;
    // If it is a known opcode, parse the operand list.
    SmallVector<Attribute> operands;

    // The element type of a function is currently always the same type
    // as the expression itself.
    Type operandType = type;
    if (failed(p.parseOptionalRParen())) {
      if (p.parseCommaSeparatedList([&]() -> ParseResult {
            return parseParamValue(p, operands.emplace_back(Attribute()),
                                   operandType);
          }) ||
          p.parseRParen())
        return failure();
    }

    // Okay, we parsed the operands, see if this is a valid expression.
    if (failed(ParamExprAttr::verify(
            [&]() -> mlir::InFlightDiagnostic { return p.emitError(loc); },
            *opcode, operands, type)))
      return failure();
    // all is good, lets move!
    value = ParamExprAttr::get(*opcode, operands);
    return success();
  }

  // Otherwise, we support other typed attributes as well, including dialect
  // define attributes, integers, etc.
  return p.parseAttribute(value, type);
}

/// When in a context that knows it is dealing with a parameter specifically,
/// utilize syntactic shortcuts to make the printed syntax easier to grok.
void KGEN::printParamValue(AsmPrinter &p, Attribute value, Type type) {
  assert(type && "parameter's should always have a contextual type!");
  if (auto declRef = value.dyn_cast<ParamDeclRefAttr>()) {
    assert(type == declRef.getType() && "type mismatch in emission?");
    printParamName(p, declRef.getName());
    return;
  }

  // If this is a dtype constant with simple syntax, we can print it as a
  // keyword.
  if (auto dtypeConstant = value.dyn_cast<DTypeConstantAttr>()) {
    auto eltType = dtypeConstant.getTensorEltType();
    std::string stringRep = eltType.getAsString();
    // Don't allow things like complex<f64>.  We can extend this in the future
    // if there is a reason to of course.
    if (!StringRef(stringRep).contains('<')) {
      p << stringRep;
      return;
    }
  }

  if (auto expr = value.dyn_cast<ParamExprAttr>()) {
    p << stringifyEnum(expr.getOpcode()) << '(';
    llvm::interleaveComma(expr.getOperands(), p, [&](Attribute operand) {
      printParamValue(p, operand, type);
    });
    p << ')';
    return;
  }

  p.printAttributeWithoutType(value);
}

/// Parse a "colon type" production if present or default to index if not.  This
/// is commonly used in our parameter representation.
ParseResult KGEN::parseColonTypeOrIndex(AsmParser &parser, Type &type) {
  if (succeeded(parser.parseOptionalColon())) {
    // In addition to standard types, we support 'dtype' as a sugared form of
    // !kgen.dtype.
    if (succeeded(parser.parseOptionalKeyword("dtype"))) {
      type = parser.getBuilder().getType<DTypeType>();
      return success();
    }

    return parser.parseType(type);
  }

  type = parser.getBuilder().getIndexType();
  return success();
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

/// Print an attribute value that is known to have index type.
void KGEN::printIndexParamValue(AsmPrinter &p, Attribute value) {
  printParamValue(p, value, IndexType::get(value.getContext()));
}

/// Parse a parameter value that is known to be an index type.
ParseResult KGEN::parseIndexParamValue(AsmParser &p,
                                       FailureOr<Attribute> &result) {
  Attribute value;
  if (parseParamValue(p, value, p.getBuilder().getIndexType()))
    return failure();
  result = value;
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
                                               FailureOr<Attribute> &result) {
  if (succeeded(p.parseOptionalQuestion())) {
    result = Attribute();
    return success();
  }
  return parseIndexParamValue(p, result);
}

//===----------------------------------------------------------------------===//
// ParamDeclAttr
//===----------------------------------------------------------------------===//

Attribute ParamDeclAttr::parse(AsmParser &p, Type type) {
  if (!type) {
    p.emitError(p.getNameLoc(), "parameter declaration requires a type");
    return {};
  }

  std::string name;
  if (p.parseLess() || p.parseKeywordOrString(&name) || p.parseGreater())
    return {};

  return ParamDeclAttr::get(name, type);
}

void ParamDeclAttr::print(AsmPrinter &p) const {
  p << "<";
  printParamName(p, getName());
  p << ">";
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

Attribute ParamBindAttr::parse(AsmParser &p, Type type) {
  std::string name;
  Attribute value;
  if (p.parseLess() || p.parseKeywordOrString(&name) ||
      parseColonTypeOrIndex(p, type) || p.parseEqual() ||
      p.parseAttribute(value, type) || p.parseGreater())
    return {};

  return ParamBindAttr::get(name, type, value);
}

void ParamBindAttr::print(AsmPrinter &p) const {
  p << "<" << getName();
  printColonTypeOrIndex(p, getType());
  p << " = ";
  p.printAttributeWithoutType(getValue());
  p << ">";
}

void ParamBindAttr::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  walkAttrsFn(getName());
  walkAttrsFn(getValue());
}

//===----------------------------------------------------------------------===//
// ParamExprAttr
//===----------------------------------------------------------------------===//

LogicalResult
ParamExprAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                      PEO opcode, ArrayRef<Attribute> operands, Type type) {
  // All the operand types and result type must match.
  if (!llvm::all_of(operands, [&](auto op) {
        return op.getType() == operands.front().getType();
      }))
    return emitError() << "operand type mismatch";

  // Check invariants on the expression.
  switch (opcode) {
  case PEO::Add:
  case PEO::Mul:
  case PEO::And:
  case PEO::Or:
  case PEO::Xor:
    if (operands.size() < 1)
      return emitError()
             << "associative operator must have at least one operand";
    type = operands[0].getType();
    if (!type.isIndex())
      return emitError() << "operator requires an index type";
    break;

  // Binary expressions.
  case PEO::Shl:
  case PEO::Shr:
  case PEO::Div:
  case PEO::Mod:
    if (operands.size() != 2)
      return emitError() << "binary operators must have two operands";
    type = operands[0].getType();
    if (!type.isIndex())
      return emitError() << "operator requires an index type";
    break;
  }
  return success();
}

/// If the specified attribute is a ParamExprAttr with the specified opcode,
/// return it.  Otherwise return null.
static ParamExprAttr dyn_castPE(PEO opcode, Attribute value) {
  if (auto expr = value.dyn_cast<ParamExprAttr>())
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
  if (rhs.isa<IntegerAttr>()) {
    // We don't bother to order constants w.r.t. each other since they will be
    // folded - they can all compare equal.
    return !lhs.isa<IntegerAttr>();
  }
  if (lhs.isa<IntegerAttr>())
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
  auto lhsExpr = lhs.cast<ParamExprAttr>(), rhsExpr = rhs.cast<ParamExprAttr>();
  // Sort by the string form of the opcode, e.g. add, .. mul,... then xor.
  if (lhsExpr.getOpcode() != rhsExpr.getOpcode())
    return stringifyPEO(lhsExpr.getOpcode()) <
           stringifyPEO(rhsExpr.getOpcode());

  // If they are the same opcode, then sort by arity: more complex to the left.
  ArrayRef<Attribute> lhsOperands = lhsExpr.getOperands(),
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
    PEO opcode, SmallVector<Attribute, 4> &operands,
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
static std::pair<Attribute, Attribute> decomposeAddend(Attribute operand) {
  if (auto mul = dyn_castPE(PEO::Mul, operand))
    if (auto cst = mul.getOperands().back().dyn_cast<IntegerAttr>()) {
      auto nonCst = ParamExprAttr::get(PEO::Mul, mul.getOperands().drop_back());
      return {nonCst, cst};
    }
  return {operand, Attribute()};
}

static Attribute getOneOfType(Type type) {
  size_t width = type.isIndex() ? 64 : type.getIntOrFloatBitWidth();
  return IntegerAttr::get(type, APInt(width, 1));
}

static Attribute simplifyAdd(SmallVector<Attribute, 4> &operands) {
  if (auto result = simplifyAssocOp(
          PEO::Add, operands, [](auto a, auto b) { return a + b; },
          /*identityCst*/ [](auto cst) { return cst.isZero(); }))
    return result;

  // Canonicalize the add by splitting all addends into their variable and
  // constant factors.
  SmallVector<std::pair<Attribute, Attribute>> decomposedOperands;
  llvm::SmallDenseSet<Attribute> nonConstantParts;
  for (auto &op : operands) {
    decomposedOperands.push_back(decomposeAddend(op));

    // Keep track of non-constant parts we've already seen.  If we see multiple
    // uses of the same value, then we can fold them together with a multiply.
    // This handles things like `(a+b+a)` => `(a*2 + b)` and `(a*2 + b + a)` =>
    // `(a*3 + b)`.
    if (!nonConstantParts.insert(decomposedOperands.back().first).second) {
      // The thing we multiply will be the common expression.
      Attribute mulOperand = decomposedOperands.back().first;

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
      auto constant = ParamExprAttr::get(PEO::Add, c1, c2);
      auto mulCst = ParamExprAttr::get(PEO::Mul, mulOperand, constant);
      operands.push_back(mulCst);
      return ParamExprAttr::get(PEO::Add, operands);
    }
  }

  return {};
}

static Attribute simplifyMul(SmallVector<Attribute, 4> &operands) {
  if (auto result = simplifyAssocOp(
          PEO::Mul, operands, [](auto a, auto b) { return a * b; },
          /*identityCst*/ [](auto cst) { return cst.isOne(); },
          /*destructiveCst*/ [](auto cst) { return cst.isZero(); }))
    return result;

  // We always build a sum-of-products representation, so if we see an addition
  // as a subexpr, we need to pull it out: (a+b)*c*d ==> (a*c*d + b*c*d).
  for (size_t i = 0, e = operands.size(); i != e; ++i) {
    if (auto addSubExpr = dyn_castPE(PEO::Add, operands[i])) {
      // Pull the `c*d` operands out - it is whatever operands remain after
      // removing the `(a+b)` term.
      operands.erase(operands.begin() + i);

      // Build each add operand.
      SmallVector<Attribute> addOperands;
      for (auto addOperand : addSubExpr.getOperands()) {
        operands.push_back(addOperand);
        addOperands.push_back(ParamExprAttr::get(PEO::Mul, operands));
        operands.pop_back();
      }
      // Canonicalize and form the add expression.
      return ParamExprAttr::get(PEO::Add, addOperands);
    }
  }

  return {};
}

static Attribute simplifyAnd(SmallVector<Attribute, 4> &operands) {
  return simplifyAssocOp(
      PEO::And, operands, [](auto a, auto b) { return a & b; },
      /*identityCst*/ [](auto cst) { return cst.isAllOnes(); },
      /*destructiveCst*/ [](auto cst) { return cst.isZero(); });
}

static Attribute simplifyOr(SmallVector<Attribute, 4> &operands) {
  return simplifyAssocOp(
      PEO::Or, operands, [](auto a, auto b) { return a | b; },
      /*identityCst*/ [](auto cst) { return cst.isZero(); },
      /*destructiveCst*/ [](auto cst) { return cst.isAllOnes(); });
}

static Attribute simplifyXor(SmallVector<Attribute, 4> &operands) {
  return simplifyAssocOp(
      PEO::Xor, operands, [](auto a, auto b) { return a ^ b; },
      /*identityCst*/ [](auto cst) { return cst.isZero(); });
}

/// Given a binary function, if the two operands are known constant integers,
/// use the specified fold functions to compute the result.
static Attribute
foldBinaryOp(ArrayRef<Attribute> operands,
             llvm::function_ref<APInt(const APInt &, const APInt &)> unsignedfn,
             llvm::function_ref<APInt(const APInt &, const APInt &)> signedFn) {
  assert(operands.size() == 2 && "binary operator always has two operands");
  if (auto lhs = operands[0].dyn_cast<IntegerAttr>())
    if (auto rhs = operands[1].dyn_cast<IntegerAttr>()) {
      auto result = (lhs.getType().isSignedInteger() ? signedFn : unsignedfn)(
          lhs.getValue(), rhs.getValue());
      return IntegerAttr::get(lhs.getType(), std::move(result));
    }
  return {};
}

static Attribute simplifyShl(SmallVector<Attribute, 4> &operands) {
  // Canonicalize `x << cst` => `x * (1<<cst)` to compose correctly with
  // add/mul canonicalization (also handles constant folding).
  if (auto rhs = operands[1].dyn_cast<IntegerAttr>()) {
    auto rhsCst = APInt::getOneBitSet(rhs.getValue().getBitWidth(),
                                      rhs.getValue().getZExtValue());
    return ParamExprAttr::get(PEO::Mul, operands[0],
                              IntegerAttr::get(rhs.getType(), rhsCst));
  }
  return {};
}

static Attribute simplifyShr(SmallVector<Attribute, 4> &operands) {
  if (auto rhs = operands[1].dyn_cast<IntegerAttr>())
    if (rhs.getValue().isZero())
      return operands[0]; // `x >> 0 = x`.
  // TODO: 0 >> x, -1 >>> x

  return foldBinaryOp(
      operands, [](auto a, auto b) { return a.lshr(b); },
      [](auto a, auto b) { return a.ashr(b); });
}

static Attribute simplifyDiv(SmallVector<Attribute, 4> &operands) {
  // Implement support for identities like `x/1 = x`.
  if (auto rhs = operands[1].dyn_cast<IntegerAttr>())
    if (rhs.getValue().isOne())
      return operands[0];

  return foldBinaryOp(
      operands, [](auto a, auto b) { return a.udiv(b); },
      [](auto a, auto b) { return a.sdiv(b); });
}

static Attribute simplifyMod(SmallVector<Attribute, 4> &operands) {
  // Implement support for identities like `x%1 = 0`.
  if (auto rhs = operands[1].dyn_cast<IntegerAttr>())
    if (rhs.getValue().isOne())
      return IntegerAttr::get(rhs.getType(), 0);

  return foldBinaryOp(
      operands, [](auto a, auto b) { return a.urem(b); },
      [](auto a, auto b) { return a.srem(b); });
}

/// Build a parameter expression.  This automatically canonicalizes and
/// folds, so it may not necessarily return a ParamExprAttr.
Attribute ParamExprAttr::get(PEO opcode, ArrayRef<Attribute> operandsIn) {
  assert(!operandsIn.empty() && "Cannot have expr with no operands");
  // All operands must have the same type, which is the type of the result.
  auto type = operandsIn.front().getType();
  assert(llvm::all_of(operandsIn.drop_front(),
                      [&](auto op) { return op.getType() == type; }));

  SmallVector<Attribute, 4> operands(operandsIn.begin(), operandsIn.end());

  // Verify and canonicalize parameter expressions.
  Attribute result;
  switch (opcode) {
  case PEO::Add:
    result = simplifyAdd(operands);
    break;
  case PEO::Mul:
    result = simplifyMul(operands);
    break;
  case PEO::And:
    result = simplifyAnd(operands);
    break;
  case PEO::Or:
    result = simplifyOr(operands);
    break;
  case PEO::Xor:
    result = simplifyXor(operands);
    break;
  case PEO::Shl:
    result = simplifyShl(operands);
    break;
  case PEO::Shr:
    result = simplifyShr(operands);
    break;
  case PEO::Div:
    result = simplifyDiv(operands);
    break;
  case PEO::Mod:
    result = simplifyMod(operands);
    break;
  }

  // If we folded to an operand, return it.
  if (result)
    return result;

  return Base::get(operandsIn[0].getContext(), opcode, operands, type);
}

/// Builder used by the generic parser.
Attribute ParamExprAttr::get(MLIRContext *ctx, PEO opcode,
                             ArrayRef<Attribute> operands, Type type) {
  auto result = get(opcode, operands);
  assert((!type || result.getType() == type) && "unexpected types");
  assert(ctx == result.getContext());
  return result;
}

void ParamExprAttr::walkImmediateSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) const {
  for (auto operand : getOperands())
    walkAttrsFn(operand);
}

//===----------------------------------------------------------------------===//
// DTypeConstantAttr
//===----------------------------------------------------------------------===//

LogicalResult DTypeConstantAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, IntegerAttr value,
    Type type) {
  if (!value.getType().isSignlessInteger(8))
    return emitError() << "kgen.dtype.constant requires i8 value";
  if (!type || !type.isa<DTypeType>())
    return emitError() << "kgen.dtype.constant requires !kgen.dtype type";
  return success();
}

/// Return the TensorEltType for the value we contain.
TensorEltType DTypeConstantAttr::getTensorEltType() {
  return TensorEltType(getValue().getValue().getZExtValue());
}

//===----------------------------------------------------------------------===//
// Parameter Helper Functions
//===----------------------------------------------------------------------===//

/// Return true if the attribute is a valid parameter expression.
bool KGEN::isValidParameterExpr(Attribute value) {
  // Leaf constants and references to parameter declarations are valid.
  if (value.isa<IntegerAttr, DTypeConstantAttr, ParamDeclRefAttr>())
    return true;

  // Expressions composed of other expressions are valid.
  if (auto expr = value.dyn_cast<ParamExprAttr>()) {
    return llvm::all_of(expr.getOperands(), [](Attribute operand) -> bool {
      return isValidParameterExpr(operand);
    });
  }

  // Nothing else is.
  return false;
}

/// Given a kernel, generator, or generator interface operation, return an array
/// of `ParamDeclAttr`s for the inputs and the array of `ParamDeclAttr`s for the
/// result parameters.  A concrete kernel will always return empty arrays.
std::pair<ArrayRef<Attribute>, ArrayRef<Attribute>>
KGEN::getCalleeParameterInfo(Operation *callee) {
  // Fully defined kernels never have parameters.
  if (isa<KernelOp>(callee))
    return std::make_pair(ArrayRef<Attribute>(), ArrayRef<Attribute>());

  assert((isa<GeneratorOp, GeneratorInterfaceOp>(callee)) && "unknown callee");
  ArrayRef<Attribute> calleeParams =
      callee->getAttrOfType<ArrayAttr>("paramDecls").getValue();
  size_t calleeNumInputParams =
      callee->getAttrOfType<IntegerAttr>("numInputParameters")
          .getValue()
          .getZExtValue();
  assert(calleeNumInputParams <= calleeParams.size());
  return std::make_pair(calleeParams.take_front(calleeNumInputParams),
                        calleeParams.drop_front(calleeNumInputParams));
}

//===----------------------------------------------------------------------===//
// Parameter Verification
//===----------------------------------------------------------------------===//

namespace {
struct ParameterVerifier final {

  /// Walk the operation and all the operations in its body to find the
  /// definitions and uses of parameters.  This diagnoses and rejects parameter
  /// definitions in invalid positions as well.
  LogicalResult collectParameterDefsAndUses(Operation *topLevelOp);

  /// Once all the defs and uses of parameters are collected, verify that the
  /// uses are correct.
  LogicalResult checkParameterUses();

private:
  /// Scan the specified attribute and its recursive uses, diagnosing incorrect
  /// parameter declarations and collecting parameter uses.
  void collectParameterUsesFromAttr(Attribute attr, Operation *op);

  /// Scan the specified type and its recursive uses, diagnosing incorrect
  /// parameter declarations and collecting parameter uses.
  void collectParameterUsesFromType(Type type, Operation *op);

  /// This is set to true if we find a problem during the collect phase.
  bool hadError = false;

  // Parameter definitions, if any are present, should all be in a single
  // `paramDecls` attribute on an operation.  We restrict where declarations
  // can be found to make them easier to identify and work with.  Keep track of
  // all the parameters we find by their name, this allows detecting
  // redefinitions with different types.
  SmallDenseMap<StringAttr, std::pair<Operation *, ParamDeclAttr>> paramDecls;

  // Parameter uses can occur in any attribute and even in in types.  We collect
  // all the uses we see by their operation.  Remember that attributes are
  // uniqued, so the same ParamDeclRefAttr can be used by multiple operations,
  // or even multiple times in the same operation.
  SmallVector<std::pair<ParamDeclRefAttr, Operation *>> parameterUses;

  // Attributes and types are memoized and exist in tree structures with reuse:
  // naively scanning them can lead to exponential compile time behavior.  As
  // such, we memoize the attributes and types we've already checked that we
  // know have no parameters in them.
  llvm::SmallDenseSet<Attribute> parameterLessAttrs;
  llvm::SmallDenseSet<Type> parameterLessTypes;
};
} // end anonymous namespace.

/// Walk the operation and all the operations in its body to find the
/// definitions and uses of parameters.  This diagnoses and rejects parameter
/// definitions in invalid positions as well.
LogicalResult
ParameterVerifier::collectParameterDefsAndUses(Operation *topLevelOp) {
  // TODO: We probably shouldn't walk into IsolatedFromAbove operations.  This
  // walk may need to be adjusted if we have any.
  topLevelOp->walk<mlir::WalkOrder::PreOrder>([&](Operation *bodyOp) {
    Attribute paramDeclsAttr;
    // Scan all the attributes and types to look for uses of parameters.  We let
    // the walker scan the region hierarchy.
    for (const NamedAttribute &namedAttr : bodyOp->getAttrs()) {
      // We handle paramDecls below specially, just remember it for then.
      if (namedAttr.getName().strref() == "paramDecls") {
        paramDeclsAttr = namedAttr.getValue();
        continue;
      }
      // Scan the attribute tree looking or parameter uses and reject unexpected
      // parameter definitions.
      collectParameterUsesFromAttr(namedAttr.getValue(), bodyOp);
    }

    // Check the types of results to find any parameters embedded in their
    // types.  We don't have to check operands because they are always checked
    // when being defined.
    for (Type type : bodyOp->getResultTypes())
      collectParameterUsesFromType(type, bodyOp);

    // Scan the region list if present.  The walker will automatically recurse
    // for us, but we have to check the block arguments.
    if (bodyOp->getNumRegions()) { // Microoptimization: getRegions() is slow.
      for (auto &region : bodyOp->getRegions()) {
        for (auto &block : region)
          for (Value arg : block.getArguments())
            collectParameterUsesFromType(arg.getType(), bodyOp);
      }
    }

    // Ok, check parameter declarations if present.
    if (!paramDeclsAttr)
      return;

    auto arrayAttr = paramDeclsAttr.dyn_cast<ArrayAttr>();
    if (!arrayAttr) {
      bodyOp->emitError("paramDecls attribute should be an array ")
          << paramDeclsAttr;
      hadError = true;
      return;
    }

    for (Attribute attr : arrayAttr) {
      // All the members of this array must be ParamDeclAttr's.
      auto param = attr.dyn_cast<ParamDeclAttr>();
      if (!param) {
        bodyOp->emitError("unknown attribute kind in paramDecls list ") << attr;
        hadError = true;
        return;
      }

      // We cannot have any redefinitions.
      auto &opAndDeclAttr = paramDecls[param.getName()];
      if (opAndDeclAttr.first) {
        auto diag = bodyOp->emitError("redeclaration of parameter ")
                    << param.getName();
        diag.attachNote(opAndDeclAttr.first->getLoc())
            << "previous declaration here";
        hadError = true;
        return;
      }
      opAndDeclAttr = {bodyOp, param};
    }
  });

  return failure(hadError);
}

/// Scan the specified attribute and its recursive uses, diagnosing incorrect
/// parameter declarations and collecting parameter uses.
void ParameterVerifier::collectParameterUsesFromAttr(Attribute attr,
                                                     Operation *op) {

  // Collect parameter references.
  if (auto paramRef = attr.dyn_cast<ParamDeclRefAttr>()) {
    parameterUses.push_back({paramRef, op});
    return;
  }

  // If this attribute has no sub-attributes or we have already scanned it an
  // know that it has no parameters in it, return early.
  if (attr.isa<IntegerAttr, FloatAttr, StringAttr, SymbolRefAttr>() ||
      parameterLessAttrs.contains(attr))
    return;

  // Reject errant parameter decls.
  if (auto paramDecl = attr.dyn_cast<ParamDeclAttr>()) {
    op->emitError("invalid ParamDeclAttr outside of paramDecls attribute ")
        << paramDecl;
    return;
  }

  size_t oldSize = parameterUses.size();

  // Otherwise we haven't processed this, check the attribute's type.
  collectParameterUsesFromType(attr.getType(), op);

  // Recursively check for any nested types/attributes, e.g. the elements of an
  // array attribute.
  if (auto itf = attr.dyn_cast<mlir::SubElementAttrInterface>()) {
    itf.walkSubElements(
        [&](Attribute attr) { collectParameterUsesFromAttr(attr, op); },
        [&](Type type) { collectParameterUsesFromType(type, op); });
  } else if (attr.isa<DTypeConstantAttr>()) {
    // This attribute doesn't participate with SubElementAttrInterface but we
    // know it doesn't have any subelements.
  } else {
    // Conservatively reject unknown attributes, we don't want someone to forget
    // to conform to SubElementAttrInterface.
    op->emitError("unknown attribute for parameterization scan: ") << attr;
  }

  // If the attribute had no uses, remember that so we don't have to re-scan it
  // in the future.
  if (oldSize == parameterUses.size())
    parameterLessAttrs.insert(attr);
}

/// Scan the specified type and its recursive uses, diagnosing incorrect
/// parameter declarations and collecting parameter uses.
void ParameterVerifier::collectParameterUsesFromType(Type type, Operation *op) {
  // Ignore common trivial types we know are never parameterized, and types we
  // have already scanned.
  if (parameterLessTypes.count(type))
    return;

  size_t oldSize = parameterUses.size();

  // Recursively check for any nested types, e.g. the input/outputs of a
  // function type.  This also handles types like !meta.scalar etc.
  if (auto itf = type.dyn_cast<mlir::SubElementTypeInterface>()) {
    itf.walkSubElements(
        [&](Attribute attr) { collectParameterUsesFromAttr(attr, op); },
        [&](Type type) { collectParameterUsesFromType(type, op); });
  } else {
    // These are known leaf types that don't participate with
    // SubElementTypeInterface.
    if (!type.isa<IntegerType, FloatType, NoneType, IndexType, DTypeType>()) {
      // Conservatively reject unknown types, we don't want someone to forget to
      // conform to SubElementTypeInterface.
      op->emitError("unknown type for parameterization scan: ") << type;
    }
  }

  // If the attribute had no uses, remember that so we don't have to re-scan it
  // in the future.
  if (oldSize == parameterUses.size())
    parameterLessTypes.insert(type);
}

/// Once all the defs and uses of parameters are collected, verify that the
/// uses are correct.
LogicalResult ParameterVerifier::checkParameterUses() {
  for (auto &[paramRefAttr, usingOp] : parameterUses) {
    // Check the use is referring to a parameter that was defined.
    auto decl = paramDecls[paramRefAttr.getName()];
    if (!decl.first) {
      usingOp->emitError("invalid use of parameter with no declaration ")
          << paramRefAttr.getName();
      return failure();
    }

    // Check that the types of the uses match the defs.
    if (decl.second.getType() != paramRefAttr.getType()) {
      auto diag = usingOp->emitError("reference to parameter ")
                  << paramRefAttr.getName() << " with incorrect type "
                  << paramRefAttr.getType();
      diag.attachNote(decl.first->getLoc())
          << "parameter defined with type " << decl.second.getType();
      return failure();
    }

    // FIXME: Check partial ordering.
  }
  return success();
}

/// Scan the body of the specified operation checking invariants on
/// parameters, diagnosing errors and returning failure if so.  This is used
/// by verifiers for ops with bodies, like kgen.generator.
LogicalResult KGEN::checkParametersInOpBody(Operation *topLevelOp) {
  ParameterVerifier verifier;

  // Start by doing a pass over the operation and all the operations in its body
  // to find the definitions and uses of parameters.
  if (failed(verifier.collectParameterDefsAndUses(topLevelOp)))
    return failure();

  // Ok, now that we know the set of parameters we have to process, verify that
  // the uses match up and that we have a proper partial order relationship
  // between of definitions and uses.
  return verifier.checkParameterUses();
}
