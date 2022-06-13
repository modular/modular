//===- KGENAttrs.cpp - Implement KGEN attributes --------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace M::KGEN;

// Provide implementations for the enums we use.
#include "KGEN/KGENDialect/KGENEnums.cpp.inc"

/// Parse a "colon type" production if present or default to si64 if not.  This
/// is commonly used in our parameter representation.
ParseResult KGEN::parseColonTypeOrSI64(OpAsmParser &parser, Type &type) {
  if (succeeded(parser.parseOptionalColon()))
    return parser.parseType(type);

  type = parser.getBuilder().getIntegerType(64, /*isSigned=*/true);
  return success();
}

/// print `: <type>` or elide it entirely if type is an si64.
void KGEN::printColonTypeOrSI64(OpAsmPrinter &p, Type type) {
  if (!type.isSignedInteger(64))
    p << ": " << type;
}

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

Attribute KGENDialect::parseAttribute(DialectAsmParser &p, Type type) const {
  StringRef attrName;
  Attribute attr;
  if (p.parseKeyword(&attrName))
    return Attribute();
  auto parseResult = generatedAttributeParser(p, attrName, type, attr);
  if (parseResult.hasValue())
    return attr;

  p.emitError(p.getNameLoc(), "Unexpected kgen attribute '" + attrName + "'");
  return {};
}

void KGENDialect::printAttribute(Attribute attr, DialectAsmPrinter &p) const {
  if (succeeded(generatedAttributePrinter(attr, p)))
    return;
  llvm_unreachable("Unexpected attribute");
}

//===----------------------------------------------------------------------===//
// "Pretty" parameter printing and parsing
//===----------------------------------------------------------------------===//

// Parameters are complex nested expressions.  While they have a generic
// printing syntax that is supported in full generality, they often appear in
// tightly controlled situations, e.g. in return operations, in types, or when
// invoking a generator. In these cases we can use a much nicer and more compact
// syntax so we as compiler engineers don't go bonkers looking at IR dumps.

/// When in a context that knows it is dealing with a parameter specifically,
/// utilize syntactic shortcuts to make the parsed syntax easier to grok.
ParseResult KGEN::parseParamValue(OpAsmParser &p, Attribute &value, Type type) {
  assert(type && "always have a contextual type");
  std::string bareword;
  // barewords are implicitly parameter declaration references or the start of
  // a expression in function form.
  if (succeeded(p.parseOptionalKeywordOrString(&bareword))) {
    // Just a bareword with no trailing `(`?  Must be a parameter reference.
    if (failed(p.parseOptionalLParen())) {
      value = ParamDeclRefAttr::get(bareword, type);
      return success();
    }

    // Otherwise it's a function expression, decode the name as an operation
    // code.
    auto opcode = symbolizePEO(bareword);
    auto loc = p.getCurrentLocation();
    if (!opcode.hasValue())
      return p.emitError(loc, "unknown expression ") << bareword;
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
void KGEN::printParamValue(OpAsmPrinter &p, Attribute value, Type type) {
  assert(type && "parameter's should always have a contextual type!");
  if (auto declRef = value.dyn_cast<ParamDeclRefAttr>()) {
    assert(type == declRef.getType() && "type mismatch in emission?");
    p.printKeywordOrString(declRef.getName());
    return;
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
  p.printKeywordOrString(getName());
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
  p.printKeywordOrString(getName());
  p << ">";
}

//===----------------------------------------------------------------------===//
// ParamBindAttr
//===----------------------------------------------------------------------===//

Attribute ParamBindAttr::parse(AsmParser &p, Type type) {
  std::string name;
  Attribute value;
  if (p.parseLess() || p.parseKeywordOrString(&name) ||
      p.parseColonType(type) || p.parseEqual() ||
      p.parseAttribute(value, type) || p.parseGreater())
    return {};

  return ParamBindAttr::get(name, type, value);
}

void ParamBindAttr::print(AsmPrinter &p) const {
  p << "<" << getName() << ": " << getType() << " = ";
  p.printAttributeWithoutType(getValue());
  p << ">";
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
    if (!type.isSignedInteger() && !type.isUnsignedInteger())
      return emitError() << "operator requires a signful integer type";
    break;

  // Binary expressions.
  case PEO::Shl:
  case PEO::Shr:
  case PEO::Div:
  case PEO::Mod:
    if (operands.size() != 2)
      return emitError() << "binary operators must have two operands";
    type = operands[0].getType();
    if (!type.isSignedInteger() && !type.isUnsignedInteger())
      return emitError() << "operator requires a signful integer type";
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
  return IntegerAttr::get(type, APInt(type.getIntOrFloatBitWidth(), 1));
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

//===----------------------------------------------------------------------===//
// Parameter Verification
//===----------------------------------------------------------------------===//

/// Scan the specified attribute and its recursive uses, diagnosing incorrect
/// parameter declarations and collecting parameter uses.
static LogicalResult collectParameterUses(
    Attribute attr, Operation *op,
    SmallVectorImpl<std::pair<ParamDeclRefAttr, Operation *>> &parameterUses,
    llvm::SmallDenseSet<Attribute> &parameterLessAttrs) {

  // Reject errant parameter decls.
  if (auto paramDecl = attr.dyn_cast<ParamDeclAttr>()) {
    op->emitError("invalid ParamDeclAttr outside of paramDecls attribute ")
        << paramDecl;
    return failure();
  }

  // Collect parameter references.
  if (auto paramRef = attr.dyn_cast<ParamDeclRefAttr>()) {
    parameterUses.push_back({paramRef, op});
    return success();
  }

  // If this attribute has no sub-attributes or we have already scanned it an
  // know that it has no parameters in it, return early.
  if (attr.isa<IntegerAttr, FloatAttr, StringAttr, SymbolRefAttr, TypeAttr>() ||
      // TODO: Handle TypeAttr for parameterized types.
      parameterLessAttrs.count(attr))
    return success();

  // Otherwise we need to recursively process attributes that we know about.
  size_t oldSize = parameterUses.size();
  if (auto array = attr.dyn_cast<ArrayAttr>()) {
    for (auto elt : array) {
      if (failed(
              collectParameterUses(elt, op, parameterUses, parameterLessAttrs)))
        return failure();
    }
  } else if (auto bind = attr.dyn_cast<ParamBindAttr>()) {
    if (failed(collectParameterUses(bind.getValue(), op, parameterUses,
                                    parameterLessAttrs)))
      return failure();
  } else if (auto expr = attr.dyn_cast<ParamExprAttr>()) {
    for (auto operand : expr.getOperands()) {
      if (failed(collectParameterUses(operand, op, parameterUses,
                                      parameterLessAttrs)))
        return failure();
    }
  } else {
    // FIXME: hard coding specific attributes is really problematic, doesn't
    // MLIR have a generic way to walk sub-attributes?
    return op->emitError("unknown attribute for parameterization: ") << attr;
    return failure();
  }

  // If the attribute had no uses, remember that so we don't have to re-scan it
  // in the future.
  if (oldSize == parameterUses.size())
    parameterLessAttrs.insert(attr);

  return success();
}

/// Scan the body of the specified operation checking invariants on
/// parameters, diagnosing errors and returning failure if so.  This is used
/// by verifiers for ops with bodies, like kgen.generator.
LogicalResult KGEN::checkParametersInOpBody(Operation *topLevelOp) {
  // Start by doing a pass over the operation and all the operations in its body
  // to find the definitions and uses of parameters.

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

  // This is slow and expensive so we need to memoize the attributes and types
  // we've already checked.
  llvm::SmallDenseSet<Attribute> parameterLessAttrs;
  // TODO: parameterLessTypes.

  bool hadError = false;
  topLevelOp->walk<mlir::WalkOrder::PreOrder>([&](Operation *bodyOp) {
    // Scan all the attributes and types to look for uses of parameters.  We let
    // the walker scan the region hierarchy.
    for (const NamedAttribute &namedAttr : bodyOp->getAttrs()) {
      // We handle paramDecls below specially.
      if (namedAttr.getName().strref() == "paramDecls")
        continue;
      // Scan the attribute tree looking or parameter uses and reject unexpected
      // parameter definitions.
      if (failed(collectParameterUses(namedAttr.getValue(), bodyOp,
                                      parameterUses, parameterLessAttrs))) {
        hadError = true;
        break;
      }

      // TODO: Look into types when we support parameterized types.
    }

    // Ok, check for parameter declarations as well.
    auto arrayAttr = bodyOp->getAttrOfType<ArrayAttr>("paramDecls");
    if (!arrayAttr)
      return;

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

  if (hadError)
    return failure();

  // Ok, now that we know the set of parameters we have to process, verify that
  // the uses match up and that we have a proper partial order relationship
  // between of definitions and uses.
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
      auto diag = usingOp->emitError("invalid reference to parameter ")
                  << paramRefAttr;
      diag.attachNote(decl.first->getLoc())
          << "parameter defined as " << decl.second;
      return failure();
    }

    // FIXME: Check partial ordering.
  }

  return success();
}
