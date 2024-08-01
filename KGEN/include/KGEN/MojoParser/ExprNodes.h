//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides declarations for various expression nodes when in
// syntactic (not yet type checked) form.
//
// Expressions are parsed with a two-phase approach.  The first phase pulls out
// the syntactic structure of the expression, whereas the second pass does type
// checking and IR generation.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_EXPRNODES_H
#define KGEN_MOJOPARSER_EXPRNODES_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/LITDialect/LITAttrs.h"
#include "KGEN/MojoParser/ExprNode.h"
#include "KGEN/MojoParser/IRValues.h"
#include "Support/Compiler/Diags.h"
#include "llvm/ADT/StringExtras.h"

namespace M::KGEN {
class SignatureType;
} // namespace M::KGEN

namespace M::KGEN::LIT {
struct ParsedArgument;
class SRValue;

/// The ExprEmitter depends on ExprNode to provide a location and emit IR for
/// its value. In the case of synthetic code, there is a source sequence that
/// triggered the generation but not necessarily a value associated with the
/// synthetic code. To solve this, we use the synthetic node which will vend a
/// location to the emitter but has no value.
struct SyntheticNode final : public ExprNode {
  // If SyntheticNode is created with a value, then emitIR will produce that
  // value.  If not, emitIR will abort.
  SyntheticNode(SMLoc loc, AnyValue irValue = AnyValue())
      : ExprNode(kSynthetic), location(loc), irValue(irValue) {}

  const SMLoc location;

  // If null, emitIR will explode, otherwise it will produce this.
  AnyValue irValue;

  static bool classof(const ExprNode *node) { return node->kind == kSynthetic; }
  SMLoc getLoc() const override { return location; }
  SourceRange getRange() const override { return {getLoc(), getLoc()}; }
  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;

  operator ExprNode *() { return this; }
};

/// This returns an SMLoc from a StringRef that points into the source buffer.
inline SMLoc getSMLocFromStringRef(StringRef bufferRef,
                                   uint8_t startOffset = 0) {
  return SMLoc::getFromPointer(bufferRef.data() - startOffset);
}

struct IntLiteralNode final : public ExprNode {
  IntLiteralNode(StringRef spelling)
      : ExprNode(kIntLiteral), spelling(spelling) {}

  const StringRef spelling;

  static bool classof(const ExprNode *node) {
    return node->kind == kIntLiteral;
  }
  SMLoc getLoc() const override { return getSMLocFromStringRef(spelling); }
  SourceRange getRange() const override { return {getLoc(), getLoc()}; }

  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;
};

struct FloatLiteralNode final : public ExprNode {
  FloatLiteralNode(StringRef spelling)
      : ExprNode(kFloatLiteral), spelling(spelling) {}

  const StringRef spelling;

  static bool classof(const ExprNode *node) {
    return node->kind == kFloatLiteral;
  }
  SMLoc getLoc() const override { return getSMLocFromStringRef(spelling); }
  SourceRange getRange() const override { return {getLoc(), getLoc()}; }
  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;
};

struct BoolLiteralNode final : public ExprNode {
  BoolLiteralNode(SMLoc loc, bool value)
      : ExprNode(kBoolLiteral), loc(loc), value(value) {}

  const SMLoc loc;
  const bool value;

  static bool classof(const ExprNode *node) {
    return node->kind == kBoolLiteral;
  }

  SMLoc getLoc() const override { return loc; }
  SourceRange getRange() const override { return {getLoc(), getLoc()}; }
  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;
};

// This node is used for things like 'Self', '_', 'None' expressions etc.
struct SimpleLiteralNode final : public ExprNode {
  SimpleLiteralNode(Kind kind, SMLoc loc) : ExprNode(kind), loc(loc) {
    assert(classof(kind) && "invalid expr kind for this node");
  }

  const SMLoc loc;

  static bool classof(const ExprNode *node) { return classof(node->kind); }

  static bool classof(Kind kind) {
    return kind == kSelfLiteral || kind == kNoneLiteral ||
           kind == kDiscardLiteral;
  }

  SMLoc getLoc() const override { return loc; }
  SourceRange getRange() const override { return {getLoc(), getLoc()}; }
  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;
};

/// String literal nodes like "foo".  String literals support implicit
/// concatenation, so `"foo" "bar"` is treated as one expression node.
struct StringLiteralNode final : public ExprNode {
  StringLiteralNode(ArrayRef<StringRef> spellings)
      : ExprNode(kStringLiteral), spellings(spellings) {}

  const ArrayRef<StringRef> spellings;

  /// Return the contents of the string without the quotes and after
  /// concatenation.
  std::string getValue() const;

  static bool classof(const ExprNode *node) {
    return node->kind == kStringLiteral;
  }
  SMLoc getLoc() const override {
    return getSMLocFromStringRef(spellings.front());
  }
  SourceRange getRange() const override {
    return {getLoc(), getSMLocFromStringRef(spellings.back())};
  }
  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;
};

struct Identifier {
  Identifier(StringRef spelling, bool isEscaped)
      : spelling(spelling), isEscaped(isEscaped) {}

  const StringRef spelling;
  /// Needed to emit correct location if an identifier was escaped.
  bool isEscaped;

  /// Return the identifier's location with the offset taken into account.
  SMLoc getIdentifierLoc() const {
    return getSMLocFromStringRef(spelling, /*startOffset=*/isEscaped);
  }

  /// Return the identifier's range with the offset taken into account.
  SourceRange getIdentifierRange() const {
    SMLoc start = getIdentifierLoc();
    if (!isEscaped)
      return {start, start};
    auto end = SMLoc::getFromPointer(start.getPointer() + spelling.size() + 2);
    return SourceRange::getByteLevel(start, end);
  }
};

struct DeclRefNode final : public ExprNode, Identifier {
  DeclRefNode(StringRef spelling, bool isEscapedIdentifier = false)
      : ExprNode(kDeclRef), Identifier(spelling, isEscapedIdentifier) {}

  static bool classof(const ExprNode *node) { return node->kind == kDeclRef; }
  SMLoc getLoc() const override { return getIdentifierLoc(); }
  SourceRange getRange() const override { return getIdentifierRange(); }
  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;
};

struct AttributeRefNode final : public ExprNode, Identifier {
  AttributeRefNode(ExprNode *base, SMLoc dotLoc, StringRef spelling,
                   bool isEscapedIdentifier = false)
      : ExprNode(kAttributeRef), Identifier(spelling, isEscapedIdentifier),
        base(base), dotLoc(dotLoc) {}

  ExprNode *const base;
  const SMLoc dotLoc;

  static bool classof(const ExprNode *node) {
    return node->kind == kAttributeRef;
  }
  SMLoc getLoc() const override { return dotLoc; }
  SourceRange getAttributeNameRange() const { return getIdentifierRange(); }
  SourceRange getRange() const override {
    return {base->getRangeStart(), getIdentifierLoc()};
  }
  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;

  /// Emit a reference to a stored field with a base that is known not to be a
  /// dynamic lvalue.
  static CValue emitStoredFieldRef(ASTExprAnd<CValue> base,
                                   StructFieldOp fieldOp, const ExprNode *expr,
                                   ValueDest &dest, ExprEmitter &emitter);
};

/// Struct to represent an expression passed as a parameter or argument operand,
/// along with metatadata to help overload resolution and call emission.
struct Operand {
  /// This specifies how the operand is passed, these are always present
  /// in a specific order, and any of these may be missing.
  enum PassKind {
    kPositional, ///< Positional operand like foo(x)
    kStar,       ///< Splat list of positional values like: foo(*x)
    kKeyword,    ///< Keyword operand: foo(arg=x)
    kStarStar,   ///< Splat list of keyword values like: foo(**x)
  };

  Operand(ExprNode *expr, SMLoc startLoc, PassKind passKind,
          StringAttr name = StringAttr())
      : expr(expr), startLoc(startLoc), passKind(passKind), name(name) {
    assert(passKind != kKeyword || name);
  }

  /// This is the expression for the operand value.
  ExprNode *expr;

  /// The location where the keyword (if given) or the value starts.
  const SMLoc startLoc;

  const PassKind passKind;

  /// This is the name of a keyword operand when kind=kKeyword, else null.
  const StringAttr name;

  SMLoc getLoc() const { return startLoc; }

  /// Return true if this is a positional operand.
  bool isPositional() const { return passKind == kPositional; }

  /// Return true if this is a keyword operand.
  bool isKeyword() const { return passKind == kKeyword; }

  /// Return true if this is an unpacked keyword operand.
  bool isUnpackedKeyword() const { return passKind == kStarStar; }

  /// Return true if this is a keyword or keyword pack operand.
  bool isKeywordOrUnpackedKeyword() const {
    return isKeyword() || isUnpackedKeyword();
  }

  /// Return true if this is an unpacked (positional or keyword) operand.
  bool isUnpacked() const { return passKind == kStar || passKind == kStarStar; }

  /// Return true if this is a positional operand with a string literal
  /// containing the specified string.
  bool isPositionalStringLiteral(StringRef str) const;
};

struct CallNode final : public ExprNode {
  CallNode(const ExprNode *callee, SMLoc lparenLoc, ArrayRef<Operand> operands,
           SMLoc rparenLoc)
      : ExprNode(kCall), callee(callee), lparenLoc(lparenLoc),
        operands(operands), rparenLoc(rparenLoc) {}

  const ExprNode *const callee;
  const SMLoc lparenLoc;
  const ArrayRef<Operand> operands;
  const SMLoc rparenLoc;

  static bool classof(const ExprNode *node) { return node->kind == kCall; }
  SMLoc getLoc() const override { return lparenLoc; }
  SourceRange getRange() const override {
    return {callee->getRangeStart(), rparenLoc};
  }
  SourceRange getParenRange() const { return {lparenLoc, rparenLoc}; }
  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;
};

/// This represents `A[i,j]`.  In the case of slices (e.g. `A[i, ::]`), the
/// slice will be represented with a subexpression.
struct SubscriptNode final : public ExprNode {
  SubscriptNode(const ExprNode *base, SMLoc lsquareLoc,
                ArrayRef<Operand> operands, SMLoc rsquareLoc)
      : ExprNode(kSubscript), base(base), lsquareLoc(lsquareLoc),
        operands(operands), rsquareLoc(rsquareLoc) {}

  const ExprNode *const base;
  const SMLoc lsquareLoc;
  const ArrayRef<Operand> operands;
  const SMLoc rsquareLoc;

  static bool classof(const ExprNode *node) { return node->kind == kSubscript; }
  SMLoc getLoc() const override { return lsquareLoc; }
  SourceRange getRange() const override {
    return {base->getRangeStart(), rsquareLoc};
  }
  /// Return a source range from '[' to ']'.
  SourceRange getIndexRange() const { return {lsquareLoc, rsquareLoc}; }

  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;
};

/// This is an expression that produces a slice value in a SubscriptNode index
/// expression.  These have at least one colon in them, and one, two, or three
/// expressions, e.g. `:`, `: :`, `a:b`, `a::b` etc.
///
/// All the elements of the syntax are optional (and thus may be null!) except
/// for the first colon.
struct SliceNode final : public ExprNode {
  SliceNode(ExprNode *lower, SMLoc colon1Loc, ExprNode *upper, SMLoc colon2Loc,
            ExprNode *stride)
      : ExprNode(kSlice), lower(lower), colon1Loc(colon1Loc), upper(upper),
        colon2Loc(colon2Loc), stride(stride) {}

  ExprNode *const lower;
  SMLoc colon1Loc;
  ExprNode *const upper;
  SMLoc colon2Loc;
  ExprNode *const stride;

  static bool classof(const ExprNode *node) { return node->kind == kSlice; }
  SMLoc getLoc() const override { return colon1Loc; }

  SourceRange getRange() const override {
    auto startLoc = lower ? lower->getRangeStart() : colon1Loc;
    if (stride)
      return {startLoc, stride->getRangeEnd()};
    if (colon2Loc.isValid())
      return {startLoc, colon2Loc};
    if (upper)
      return {startLoc, upper->getRangeEnd()};
    return {startLoc, colon1Loc};
  }

  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;
};

struct ParenNode final : public ExprNode {
  ParenNode(SMLoc lparenLoc, ExprNode *subExpr, SMLoc rparenLoc)
      : ExprNode(kParen), lparenLoc(lparenLoc), subExpr(subExpr),
        rparenLoc(rparenLoc) {}

  const SMLoc lparenLoc;
  ExprNode *const subExpr;
  const SMLoc rparenLoc;

  static bool classof(const ExprNode *node) { return node->kind == kParen; }
  SMLoc getLoc() const override { return lparenLoc; }
  SourceRange getRange() const override { return {lparenLoc, rparenLoc}; }
  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;
};

/// `a, b, c` and `a,`.  TupleNode does not carry parens, but is often nested
/// in a ParenNode.
///
/// Note that an empty tuple `()` is represented as a TupleNode no exprs,
/// and the firstCommaLoc is at the `(`.  It is then wrapped with a ParenNode.
struct TupleNode final : public ExprNode {
  TupleNode(SMLoc firstCommaLoc, ArrayRef<ExprNode *> exprs)
      : ExprNode(kTuple), firstCommaLoc(firstCommaLoc), exprs(exprs) {}

  const SMLoc firstCommaLoc;
  ArrayRef<ExprNode *> exprs;

  static bool classof(const ExprNode *node) { return node->kind == kTuple; }
  SMLoc getLoc() const override { return firstCommaLoc; }
  SourceRange getRange() const override {
    if (exprs.empty())
      return {firstCommaLoc, firstCommaLoc};
    return {exprs.front()->getRangeStart(), exprs.back()->getRangeEnd()};
  }
  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;
};

/// [a, b, c]
struct ListNode final : public ExprNode {
  ListNode(SMLoc lsquareLoc, ArrayRef<ExprNode *> exprs, SMLoc rsquareLoc)
      : ExprNode(kList), lsquareLoc(lsquareLoc), exprs(exprs),
        rsquareLoc(rsquareLoc) {}

  const SMLoc lsquareLoc;
  ArrayRef<ExprNode *> exprs;
  const SMLoc rsquareLoc;

  static bool classof(const ExprNode *node) { return node->kind == kList; }
  SMLoc getLoc() const override { return lsquareLoc; }
  SourceRange getRange() const override { return {lsquareLoc, rsquareLoc}; }
  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;
};

/// This represents `{key1: value1, key2: value2, **dictunpack}` expressions.
/// The dictionary unpacking syntax is represented with a null key and with the
/// unpack expression as the value.
struct DictionaryNode final : public ExprNode {
  DictionaryNode(SMLoc lbraceLoc,
                 ArrayRef<std::pair<ExprNode *, ExprNode *>> values,
                 SMLoc rbraceLoc)
      : ExprNode(kDictionary), lbraceLoc(lbraceLoc), values(values),
        rbraceLoc(rbraceLoc) {}

  const SMLoc lbraceLoc;
  const ArrayRef<std::pair<ExprNode *, ExprNode *>> values;
  const SMLoc rbraceLoc;

  static bool classof(const ExprNode *node) {
    return node->kind == kDictionary;
  }
  SMLoc getLoc() const override { return lbraceLoc; }
  SourceRange getRange() const override { return {lbraceLoc, rbraceLoc}; }

  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;
};

/// This represents `expr{x:y, **unpack}` using DictionaryNode as the storage.
struct DictSubscriptNode final : public ExprNode {
  DictSubscriptNode(ExprNode *base, DictionaryNode *indices)
      : ExprNode(kDictSubscript), base(base), indices(indices) {}

  ExprNode *const base;
  DictionaryNode *const indices;

  static bool classof(const ExprNode *node) {
    return node->kind == kDictSubscript;
  }
  SMLoc getLoc() const override { return indices->lbraceLoc; }
  SourceRange getRange() const override {
    return {base->getRangeStart(), indices->rbraceLoc};
  }

  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;
  AnyValue emitTypeSubscriptIR(ASTType initType, ValueDest &dest,
                               ExprEmitter &emitter) const;
};

// trueExpr 'if' condition 'else' falseExpr
struct IfElseOpNode final : public ExprNode {
  IfElseOpNode(ExprNode *trueExpr, SMLoc ifLoc, ExprNode *condExpr,
               SMLoc elseLoc, ExprNode *falseExpr)
      : ExprNode(kIfElse), trueExpr(trueExpr), ifLoc(ifLoc), condExpr(condExpr),
        elseLoc(elseLoc), falseExpr(falseExpr) {}

  ExprNode *const trueExpr;
  const SMLoc ifLoc;
  ExprNode *const condExpr;
  const SMLoc elseLoc;
  ExprNode *const falseExpr;

  static bool classof(const ExprNode *node) { return node->kind == kIfElse; }

  SMLoc getLoc() const override { return ifLoc; }
  SourceRange getRange() const override {
    return {trueExpr->getRangeStart(), falseExpr->getRangeEnd()};
  }
  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;
};

struct BinOpNode final : public ExprNode {
  BinOpNode(Kind kind, ExprNode *lhs, SMLoc opLoc, ExprNode *rhs)
      : ExprNode(kind), lhs(lhs), opLoc(opLoc), rhs(rhs) {}

  ExprNode *const lhs;
  const SMLoc opLoc;
  ExprNode *const rhs;

  static bool classof(const ExprNode *node) {
    return node->kind >= kFirstBinOp && node->kind <= kLastBinOp;
  }

  /// Return true if this is an "assignment stmt" node like =, +=, or *=.
  bool isAssignmentStmt() const {
    return kind >= kFirstAssignStmt && kind <= kLastAssignStmt;
  }

  SMLoc getLoc() const override { return opLoc; }
  SourceRange getRange() const override {
    return {lhs->getRangeStart(), rhs->getRangeEnd()};
  }

  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;

private:
  AnyValue emitAndOr(ValueDest &dest, ExprEmitter &emitter) const;
  AnyValue emitAssign(ValueDest &dest, ExprEmitter &emitter) const;
  AnyValue emitInplace(ValueDest &dest, ExprEmitter &emitter) const;
};

struct UnaryOpNode final : public ExprNode {
  UnaryOpNode(Kind kind, SMLoc opLoc, ExprNode *subExpr)
      : ExprNode(kind), opLoc(opLoc), subExpr(subExpr) {}

  const SMLoc opLoc;
  ExprNode *const subExpr;

  static bool classof(const ExprNode *node) {
    return node->kind >= kFirstUnaryOp && node->kind <= kLastUnaryOp;
  }
  SMLoc getLoc() const override { return opLoc; }
  SourceRange getRange() const override {
    return {opLoc, subExpr->getRangeEnd()};
  }
  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;
  AnyValue emitTransfer(AnyValue argValue, ValueDest &dest,
                        ExprEmitter &emitter) const;

  /// Emit a unary arithmetic operation.
  static AnyValue emitArith(Kind kind, const ExprNode *expr,
                            ASTExprAnd<AnyValue> value, ValueDest &dest,
                            ExprEmitter &emitter);
};

/// This represents a chained comparison expression (ex. a < b <= c).
/// exprs stores all the expressions in the comparison (ex. a, b, c), while
/// ops stores the ops in between pairs of expressions (ex. <, <=).
/// Chained expressions are evaluated left to right and each expression is
/// valuated at most once: a < b <= c is equivalent to a < b and b <= c, but
/// b is evaluated only once.
struct ChainedCmpOpNode final : public ExprNode {
  ChainedCmpOpNode(ArrayRef<ExprNode *> exprs, ArrayRef<ExprNode::Kind> ops,
                   SMLoc opLoc)
      : ExprNode(ExprNode::Kind::kChainedCmp), exprs(exprs), ops(ops),
        opLoc(opLoc) {}

  const ArrayRef<ExprNode *> exprs;
  const ArrayRef<ExprNode::Kind> ops;
  const SMLoc opLoc;

  SMLoc getLoc() const override { return opLoc; }
  SourceRange getRange() const override {
    return {exprs.front()->getRangeStart(), exprs.back()->getRangeEnd()};
  }

  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;
  RValue emitNextCmp(ExprEmitter &emitter, size_t opIdx, RValue lastCmp,
                     RValue lastExpr, bool hasPrevIfOp, ValueDest &dest) const;
};

struct FunctionTypeNode final : public ExprNode {
  FunctionTypeNode(SMLoc baseLoc, ArrayRef<ParsedArgument> parsedParams,
                   ArrayRef<ParsedArgument> parsedArgs,
                   ArrayRef<ParsedArgument> resultArgs, FnEffects effects,
                   SMLoc endLoc, bool isDef)
      : ExprNode(kFunctionType), baseLoc(baseLoc), parsedParams(parsedParams),
        parsedArgs(parsedArgs), resultArgs(resultArgs), effects(effects),
        endLoc(endLoc), isDef(isDef) {}

  SMLoc baseLoc;
  ArrayRef<ParsedArgument> parsedParams; // Parameter list
  ArrayRef<ParsedArgument> parsedArgs;   // Argument list
  ArrayRef<ParsedArgument> resultArgs;   // Result list
  FnEffects effects;
  SMLoc endLoc;
  bool isDef;

  static bool classof(const ExprNode *node) {
    return node->kind == kFunctionType;
  }
  SMLoc getLoc() const override { return baseLoc; }
  SourceRange getRange() const override { return {baseLoc, endLoc}; }
  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;
};

/// __get_value_from_rvalue(some_ref)      # returns LValue or BValue
/// __get_address_as_owned_value(some_ptr) # returns RValue
/// __lifetime_of(decl)                    # returns !lit.lifetime<mut>
struct MagicFunctionNode final : public ExprNode {
  MagicFunctionNode(ExprNode::Kind kind, SMLoc baseLoc,
                    ArrayRef<ExprNode *> subExprs, SMLoc rparenLoc)
      : ExprNode(kind), baseLoc(baseLoc), subExprs(subExprs),
        rparenLoc(rparenLoc) {
    assert(classof(this) && "Kind is wrong");
  }

  const SMLoc baseLoc;
  const ArrayRef<ExprNode *> subExprs;
  const SMLoc rparenLoc;

  static bool classof(const ExprNode *node) {
    return node->kind >= kFirstMagicFunction &&
           node->kind <= kLastMagicFunction;
  }
  SMLoc getLoc() const override { return baseLoc; }
  SourceRange getRange() const override { return {baseLoc, rparenLoc}; }
  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;

  AnyValue emitLifetimeOf(ValueDest &dest, ExprEmitter &emitter) const;
  AnyValue emitTypeOf(ValueDest &dest, ExprEmitter &emitter) const;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_EXPRNODES_H
