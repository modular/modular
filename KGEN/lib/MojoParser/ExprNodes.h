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

#ifndef EXPRNODES_H
#define EXPRNODES_H

#include "Diags.h"
#include "ExprNode.h"
#include "IRValues.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "llvm/ADT/StringExtras.h"

namespace M::KGEN {
class SignatureType;
} // namespace M::KGEN

namespace M::KGEN::LIT {
struct ParsedArgument;
class SRValue;

/// This returns an SMLoc from a StringRef that points into the source buffer.
inline SMLoc getSMLocFromStringRef(StringRef bufferRef) {
  return SMLoc::getFromPointer(bufferRef.data());
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

struct DeclRefNode final : public ExprNode {
  DeclRefNode(StringRef spelling) : ExprNode(kDeclRef), spelling(spelling) {}

  const StringRef spelling;

  static bool classof(const ExprNode *node) { return node->kind == kDeclRef; }
  SMLoc getLoc() const override { return getSMLocFromStringRef(spelling); }
  SourceRange getRange() const override { return {getLoc(), getLoc()}; }
  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;
};

/// The ExprEmitter depends on ExprNode to provide a location and emit IR for
/// its value. In the case of synthetic code, there is a source sequence that
/// triggered the generation but not necessarily a value associated with the
/// synthetic code. To solve this, we use the synthetic node which will vend a
/// location to the emitter but has no value.
struct SyntheticNode final : public ExprNode {
  SyntheticNode(SMLoc loc) : ExprNode(kSynthetic), location(loc) {}

  const SMLoc location;

  static bool classof(const ExprNode *node) { return node->kind == kSynthetic; }
  SMLoc getLoc() const override { return location; }
  SourceRange getRange() const override { return {getLoc(), getLoc()}; }
  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;
};

struct AttributeRefNode final : public ExprNode {
  AttributeRefNode(ExprNode *base, SMLoc dotLoc, StringRef attrSpelling)
      : ExprNode(kAttributeRef), base(base), dotLoc(dotLoc),
        attrSpelling(attrSpelling) {}

  ExprNode *const base;
  const SMLoc dotLoc;
  const StringRef attrSpelling;

  static bool classof(const ExprNode *node) {
    return node->kind == kAttributeRef;
  }
  SMLoc getLoc() const override { return dotLoc; }
  SMLoc getAttributeNameLoc() const {
    return getSMLocFromStringRef(attrSpelling);
  }
  SourceRange getRange() const override {
    return {base->getRangeStart(), getAttributeNameLoc()};
  }
  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;

  /// Emit a reference to a stored field with a base that is known not to be a
  /// dynamic lvalue.
  static CValue emitStoredFieldRef(ASTExprAnd<CValue> base,
                                   StructFieldOp fieldOp, const ExprNode *expr,
                                   ValueDest &dest, ExprEmitter &emitter);
};

struct CallArgument {
  /// This specifies what "kind" of argument this is, these are always present
  /// in a specific order, and any of these may be missing.
  enum Kind {
    kPositional, ///< Positional argument like foo(x)
    kStar,       ///< Splat list of positional values like: foo(*x)
    kKeyword,    ///< Keyword argument: foo(arg=x)
    kStarStar,   ///< Splat list of keywrod values like: foo(**x)
  } kind = kPositional;

  /// This is the expression for the value, and is always present.
  ExprNode *expr = nullptr;

  /// This is the name of a keyword argument when kind=kKeyword, else null.
  StringAttr name;

  SMLoc getLoc() const { return expr->getLoc(); }

  /// Return true if this is a positional argument with a string literal
  /// containing the specified string.
  bool isPositionalStringLiteral(StringRef str) const;

  template <typename N>
  bool isPositionalIntLiteral(N &value, unsigned base = 0) const {
    auto *intExpr = dyn_cast<IntLiteralNode>(expr);
    if (kind == kPositional && intExpr) {
      return llvm::to_integer(intExpr->spelling, value, base);
    }
    return false;
  }
};

struct CallNode final : public ExprNode {
  CallNode(ExprNode *callee, SMLoc lparenLoc, ArrayRef<CallArgument> args,
           SMLoc rparenLoc)
      : ExprNode(kCall), callee(callee), lparenLoc(lparenLoc), args(args),
        rparenLoc(rparenLoc) {}

  ExprNode *const callee;
  const SMLoc lparenLoc;
  const ArrayRef<CallArgument> args;
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
  SubscriptNode(ExprNode *base, SMLoc lsquareLoc, ArrayRef<ExprNode *> indices,
                SMLoc rsquareLoc)
      : ExprNode(kSubscript), base(base), lsquareLoc(lsquareLoc),
        indices(indices), rsquareLoc(rsquareLoc) {}

  ExprNode *const base;
  const SMLoc lsquareLoc;
  const ArrayRef<ExprNode *> indices;
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

/// This represents `A[i,j -> a, b]`.  In the case of slices (e.g. `A[i, ::]`),
/// the slice will be represented with a subexpression.
struct SubscriptArrowNode final : public ExprNode {
  SubscriptArrowNode(ExprNode *base, SMLoc lsquareLoc,
                     ArrayRef<ExprNode *> indices, SMLoc arrowLoc,
                     ArrayRef<ExprNode *> arrowExprs, SMLoc rsquareLoc)
      : ExprNode(kSubscriptArrow), base(base), lsquareLoc(lsquareLoc),
        indices(indices), arrowLoc(arrowLoc), arrowExprs(arrowExprs),
        rsquareLoc(rsquareLoc) {}

  ExprNode *const base;
  const SMLoc lsquareLoc;
  ArrayRef<ExprNode *> indices;
  const SMLoc arrowLoc;
  ArrayRef<ExprNode *> arrowExprs;
  const SMLoc rsquareLoc;

  static bool classof(const ExprNode *node) {
    return node->kind == kSubscriptArrow;
  }
  SMLoc getLoc() const override { return lsquareLoc; }
  SourceRange getRange() const override { return {lsquareLoc, rsquareLoc}; }
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
};

/// `borrowed[lt] Type` and related ownership type specifiers.
struct OwnershipOpNode final : public ExprNode {
  OwnershipOpNode(SMLoc keywordLoc, bool isMutable, ExprNode *lifetime,
                  ExprNode *subExpr)
      : ExprNode(kRef), isMutable(isMutable), keywordLoc(keywordLoc),
        lifetime(lifetime), subExpr(subExpr) {}

  bool isMutable;
  const SMLoc keywordLoc;
  // NOTE: We don't keep track of the [] locations.
  ExprNode *const lifetime;
  ExprNode *const subExpr;

  static bool classof(const ExprNode *node) { return node->kind == kRef; }
  SMLoc getLoc() const override { return keywordLoc; }
  SourceRange getRange() const override {
    return {keywordLoc, subExpr->getRangeEnd()};
  }
  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;
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
  AnyValue emitNextCmp(ExprEmitter &emitter, size_t opIdx, SRValue lastCmp,
                       SRValue lastExpr) const;
};

struct FunctionTypeNode final : public ExprNode {
  FunctionTypeNode(SMLoc baseLoc, ArrayRef<ParsedArgument> inputParams,
                   ArrayRef<ParsedArgument> resultParams,
                   ArrayRef<ParsedArgument> arguments,
                   const ExprNode *resultTypeExpr, FnEffects effects,
                   SMLoc endLoc, bool isDef, SMLoc resultLoc)
      : ExprNode(kFunctionType), baseLoc(baseLoc), inputParams(inputParams),
        resultParams(resultParams), arguments(arguments),
        resultTypeExpr(resultTypeExpr), effects(effects), endLoc(endLoc),
        isDef(isDef), resultLoc(resultLoc) {}

  SMLoc baseLoc;
  ArrayRef<ParsedArgument> inputParams;
  ArrayRef<ParsedArgument> resultParams;
  ArrayRef<ParsedArgument> arguments;
  const ExprNode *resultTypeExpr;
  FnEffects effects;
  SMLoc endLoc;
  bool isDef;
  SMLoc resultLoc;

  static bool classof(const ExprNode *node) {
    return node->kind == kFunctionType;
  }
  SMLoc getLoc() const override { return baseLoc; }
  SourceRange getRange() const override { return {baseLoc, endLoc}; }
  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;
};

/// __get_lvalue_as_address(someSLValue)        # returns pop.pointer
/// __get_address_as_lvalue(pop_pointer)        # returns SLValue
/// __get_address_as_owned_value(pop_pointer) # returns RValue
struct AddressConvertNode final : public ExprNode {
  AddressConvertNode(ExprNode::Kind kind, SMLoc baseLoc, ExprNode *subExpr,
                     SMLoc rparenLoc)
      : ExprNode(kind), baseLoc(baseLoc), subExpr(subExpr),
        rparenLoc(rparenLoc) {
    assert(classof(this) && "Kind is wrong");
  }

  const SMLoc baseLoc;
  ExprNode *const subExpr;
  const SMLoc rparenLoc;

  static bool classof(const ExprNode *node) {
    return node->kind >= kFirstAddressConvert &&
           node->kind <= kLastAddressConvert;
  }
  SMLoc getLoc() const override { return baseLoc; }
  SourceRange getRange() const override { return {baseLoc, rparenLoc}; }
  AnyValue emitIR(ValueDest &dest, ExprEmitter &emitter) const override;
};

} // namespace M::KGEN::LIT

#endif // EXPRNODES_H
