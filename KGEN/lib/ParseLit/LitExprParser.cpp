//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Expression parsing in Lightning is done in with a 2-phase approach where we
// parse one or more expressions into an AST-like representation in a first
// pass, then type check and generate operations or a type for it in a second
// pass.  This enables a number of features:
//
//   1) Non-lexical variable references: `[x.strip().upper() for x in flags]`.
//      Where we can only type check the expression after the 'for x' is type
//      checked and resolved.
//   2) Weird order of evaluations: `foo() if cond() else bar()`
//   3) Parser ambiguity of the LHS of an assignment, which we don't know if it
//      is a target until we see the equals: `x[foo()] = bar()`
//
// We handle this by having an expression parser distinct from the main parser
// that builds this tree.
//
//===----------------------------------------------------------------------===//

#include "LitExprNodes.h"
#include "LitParserBase.h"
#include "llvm/Support/SaveAndRestore.h"
using namespace M::KGEN::LIT;
using namespace M;

//===----------------------------------------------------------------------===//
// Expression Parsing
//===----------------------------------------------------------------------===//

namespace M::KGEN::LIT {
/// This class implements the ExprParser interface, implemented with the pImpl
/// idiom.
class ExprParser : public LitParserBase {
public:
  ExprParser(LitLexer &lexer, Optional<size_t> stmtIndent)
      : LitParserBase(lexer), stmtIndent(stmtIndent) {}

  ~ExprParser() {}

  // Expressions.  These methods always return a non-null ExprNode, but it may
  // be (or include) an Error node if parsing failed.
  ParseResult parseExpressionList(SmallVectorImpl<ExprNode *> &results);
  ParseResult parseExpression(ExprNode *&result);

private:
  /// Return true if the current token is the start of another statement, false
  /// if it is part of this one.
  bool isTokenStartOfNextStatement();

  ParseResult parsePrimary(ExprNode *&result);
  ParseResult parseCallSuffix(ExprNode *&result, SMLoc lparenLoc);
  ParseResult parseSubscriptSuffix(ExprNode *&result, SMLoc lsquareLoc);

  enum class Precedence {
    kInvalid, // Not a binary operator token.
    kLowest,  // Lowest precedence (most loosely bound).
    kAdd,
    kMul, // Highest precedence (most tightly bound).
  };
  std::pair<Precedence, ExprNode::Kind> getBinOpTokenPrecedenceAndKind() const;

  ParseResult parseBinOpRHS(ExprNode *&lhs, Precedence minPrec);

  /// Return an error node at the specified location.
  ExprNode *getErrorAtToken() { return alloc<ErrorNode>(getToken().getLoc()); };

  /// Allocate an expression node into the expression bump pointer allocator.
  template <typename T, typename... Args>
  T *alloc(Args &&...args);

  /// memcpy the specified ArrayRef into the expression allocator and return a
  /// pointer to the new data.  This cannot be used with things that have
  /// non-trivial copyctors/dtors because the expression allocator does run
  /// destructors.
  template <typename T>
  ArrayRef<T> copyArrayRef(ArrayRef<T> elements);

  /// This specifies the indentation level of the start of the statement that
  /// contains this expression if the expression can exist at the end of the
  /// line.  This allows the expression parser to know when to keep parsing the
  /// expression on the next line - when it is more indented than the start of
  /// the current statement.  This is None when there is a trailing punctuator
  /// that naturally terminates the expression.
  Optional<size_t> stmtIndent;
};
} // namespace M::KGEN::LIT

//===----------------------------------------------------------------------===//
// Mechanics
//===----------------------------------------------------------------------===//

/// Return true if the current token is the start of another statement, false
/// if it is part of this one.
bool ExprParser::isTokenStartOfNextStatement() {
  // If the current token is on the same line as the last or if we should always
  // eat tokens, then keep going.
  auto tokIndent = getToken().getIndentation();
  if (!tokIndent.has_value() || !stmtIndent.has_value())
    return false;

  // If this token is on its own line and we care, then it is a new statement if
  // it is as indented (or less) than the statement.
  return tokIndent <= stmtIndent;
}

/// Allocate an expression node into the expression bump pointer allocator.
template <typename T, typename... Args>
T *ExprParser::alloc(Args &&...args) {
  auto &allocator = getSharedState().persistentAllocator;
  void *node = allocator.Allocate(sizeof(T), llvm::Align::Of<T>());
  return new (node) T(std::forward<Args>(args)...);
}

/// memcpy the specified ArrayRef into the expression allocator and return a
/// pointer to the new data.  This cannot be used with things that have
/// non-trivial copyctors/dtors because the expression allocator does run
/// destructors.
template <typename T>
ArrayRef<T> ExprParser::copyArrayRef(ArrayRef<T> elements) {
  if (elements.empty())
    return elements;

  size_t dataSize = sizeof(T) * elements.size();
  auto &allocator = getSharedState().persistentAllocator;
  T *result =
      static_cast<T *>(allocator.Allocate(dataSize, llvm::Align::Of<T>()));
  memcpy(result, elements.data(), dataSize);
  return ArrayRef<T>(result, elements.size());
}

//===----------------------------------------------------------------------===//
// Parsing rules
//===----------------------------------------------------------------------===//

/// expression_list ::= expression ("," expression)* [","]
ParseResult
ExprParser::parseExpressionList(SmallVectorImpl<ExprNode *> &results) {
  // TODO: Support trailing comma for singleton tuple.
  return parseCommaSeparatedList([&]() -> ParseResult {
    return parseExpression(results.emplace_back(nullptr));
  });
}

/// expression ::=
///
///
ParseResult ExprParser::parseExpression(ExprNode *&expr) {
  if (parsePrimary(expr) || parseBinOpRHS(expr, Precedence::kLowest))
    return failure();
  return success();
}

/// Return the operator precedence for the specified token or
std::pair<ExprParser::Precedence, ExprNode::Kind>
ExprParser::getBinOpTokenPrecedenceAndKind() const {
  switch (getToken().getKind()) {
  default:
    return {Precedence::kInvalid, ExprNode::kError};
  case LitToken::plus:
    return {Precedence::kAdd, ExprNode::kAdd};
  case LitToken::star:
    return {Precedence::kMul, ExprNode::kMul};
  }
}

/// primary ::=  atom | attributeref | subscription | slicing | call
///
/// atom    ::= identifier | literal | enclosure [TODO]
/// call    ::=  primary "(" [argument_list [","] | comprehension] ")"
///
/// enclosure ::= parenth_form | list_display | dict_display | set_display
///             | generator_expression | yield_atom
/// parenth_form ::= "(" [starred_expression] ")"
///
/// literal ::= [TODO]
///     stringliteral | bytesliteral | integer | floatnumber | imagnumber
///
ParseResult ExprParser::parsePrimary(ExprNode *&result) {
  switch (getToken().getKind()) {
  case LitToken::identifier: // primary -> atom -> identifier
    result = alloc<DeclRefNode>(getToken().getSpelling());
    consumeToken(LitToken::identifier);
    break;
  case LitToken::integer: // primary -> literal -> integer
    result = alloc<IntLiteralNode>(getToken().getSpelling());
    consumeToken(LitToken::integer);
    break;
  case LitToken::float_num: // primary -> literal -> floatnumber
    result = alloc<FloatLiteralNode>(getToken().getSpelling());
    consumeToken(LitToken::float_num);
    break;
  case LitToken::string: // primary -> literal -> stringliteral
    result = alloc<StringLiteralNode>(getToken().getSpelling());
    consumeToken(LitToken::string);
    break;
  case LitToken::l_paren: { // primary -> atom -> enclosure -> parenth_form
    auto lpLoc = consumeToken(LitToken::l_paren).getLoc();
    if (parseExpression(result))
      return failure();
    auto rpLoc = getToken().getLoc();
    // FIXME: This is terrible error recovery.
    if (parseToken(LitToken::r_paren,
                   "expected ')' in parenthesized expression"))
      return failure();
    result = alloc<ParenExprNode>(lpLoc, result, rpLoc);
    break;
  }

  default:
    emitError("unexpected token in expression");
    result = getErrorAtToken();
    return failure();
  }

  // Parse postfix productions so long as they aren't the start of the next
  // statement.
  while (!isTokenStartOfNextStatement()) {
    auto loc = getToken().getLoc();

    // Handle calls.
    if (consumeIf(LitToken::l_paren)) {
      if (parseCallSuffix(result, loc))
        return failure();
      continue;
    }

    // Handle "subscription" and "slicing" array subscripts and slicing.
    if (consumeIf(LitToken::l_square)) {
      if (parseSubscriptSuffix(result, loc))
        return failure();
      continue;
    }

    break;
  }

  return success();
}

/// call ::=  primary "(" [argument_list [","] | comprehension] ")"
///
/// argument_list        ::=  positional_arguments ["," starred_and_keywords]
///                             ["," keywords_arguments]
///                           | starred_and_keywords ["," keywords_arguments]
///                           | keywords_arguments
/// positional_arguments ::=  positional_item ("," positional_item)*
/// positional_item      ::=  assignment_expression | "*" expression
/// starred_and_keywords ::=  ("*" expression | keyword_item)
///                           ("," "*" expression | "," keyword_item)*
/// keywords_arguments   ::=  (keyword_item | "**" expression)
///                           ("," keyword_item | "," "**" expression)*
/// keyword_item         ::=  identifier "=" expression
ParseResult ExprParser::parseCallSuffix(ExprNode *&result, SMLoc lparenLoc) {
  SmallVector<ExprNode *> args;
  // TODO: Handle comprehension arguments, stars, etc.
  if (!consumeIf(LitToken::r_paren)) {
    // Expressions continue maximally because we are within ()'s.
    llvm::SaveAndRestore<Optional<size_t>> X(stmtIndent, None);
    if (parseExpressionList(args) ||
        parseToken(LitToken::r_paren, "expected ')' in call argument list")) {
      return failure();
    }
  }
  result = alloc<CallNode>(result, lparenLoc, copyArrayRef<ExprNode *>(args));
  return success();
}

/// subscription ::=  primary "[" expression_list "]"
///
/// slicing      ::=  primary "[" slice_list "]"  [TODO]
/// slice_list   ::=  slice_item ("," slice_item)* [","]
/// slice_item   ::=  expression | proper_slice
/// proper_slice ::=  [lower_bound] ":" [upper_bound] [ ":" [stride] ]
/// lower_bound  ::=  expression
/// upper_bound  ::=  expression
/// stride       ::=  expression
ParseResult ExprParser::parseSubscriptSuffix(ExprNode *&result,
                                             SMLoc lsquareLoc) {
  // Expressions continue maximally because we are within []'s.
  llvm::SaveAndRestore<Optional<size_t>> X(stmtIndent, None);

  // TODO: Add support for slices.
  SmallVector<ExprNode *> indices;
  if (parseExpressionList(indices) ||
      parseToken(LitToken::r_square, "expected ']' in call argument list"))
    return failure();

  result = alloc<SubscriptNode>(result, lsquareLoc,
                                copyArrayRef<ExprNode *>(indices));
  return success();
}

/// Parse any binary operators that have precedence of at least `minPrec`.  This
/// stop if the current token isn't a binary operator or if it binds more
/// loosely than the specified precedence level.
ParseResult ExprParser::parseBinOpRHS(ExprNode *&expr, Precedence minPrec) {
  // Process operators unless they are at the start of the next statement.
  while (!isTokenStartOfNextStatement()) {
    auto [tokPrec, binOpKind] = getBinOpTokenPrecedenceAndKind();

    // If the next token is lower precedence than we are allowed to eat, return
    // successfully with what we ate already.  This also handles invalid tokens,
    // since they are treated as lower precedence than we ever allow.
    if (unsigned(tokPrec) < unsigned(minPrec))
      return success();

    SMLoc opLoc = getToken().getLoc();
    consumeToken();

    // Eat the next primary expression.
    ExprNode *rhs = nullptr;
    if (parsePrimary(rhs))
      return failure();

    // If the operator we parse bind looser with the RHS than the operator after
    // the RHS, then give the RHS primary to the RHS.
    auto [nextTokPrec, nextBinOpKind] = getBinOpTokenPrecedenceAndKind();
    if (unsigned(tokPrec) < unsigned(nextTokPrec)) {
      if (parseBinOpRHS(rhs, Precedence(unsigned(tokPrec) + 1)))
        return failure();
    }

    // Merge LHS and RHS according to operator.
    expr = alloc<BinOpNode>(binOpKind, expr, opLoc, rhs);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ExprParser implementation
//===----------------------------------------------------------------------===//

ParseResult
LitParserBase::parseExpressionList(SmallVectorImpl<ExprNode *> &results,
                                   Optional<size_t> stmtIndent) {
  return ExprParser(getLexer(), stmtIndent).parseExpressionList(results);
}

ParseResult LitParserBase::parseExpression(ExprNode *&result,
                                           Optional<size_t> stmtIndent) {
  return ExprParser(getLexer(), stmtIndent).parseExpression(result);
}

ParseResult LitParserBase::parseType(Type &result, Scope &scope,
                                     Optional<size_t> stmtIndent) {
  ExprNode *expr = nullptr;
  if (parseExpression(expr, stmtIndent))
    return failure();
  result = ExprEmitter(getSharedState(), scope, None, nullptr).emitType(expr);
  return success();
}
