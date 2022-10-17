//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This implements the expression parser.
//
//===----------------------------------------------------------------------===//

#include "LitExprNodes.h"
#include "LitParserBase.h"
using namespace M::KGEN::LIT;
using namespace M;

//===----------------------------------------------------------------------===//
// Expression Parsing
//===----------------------------------------------------------------------===//

namespace M::KGEN::LIT {
/// This class implements the ExprParser interface, implemented with the pImpl
/// idiom.
class ExprParserImpl : public LitParserBase {
public:
  ExprParserImpl(LitParserBase &existing)
      : LitParserBase(existing.lexer, existing.sharedParserState) {
    // Only a single expression parser can be active at a time, because we clear
    // the bump pointer allocator when done.
    assert(!sharedParserState->hasExprParser &&
           "Cannot create multiple expr parsers at once");
    sharedParserState->hasExprParser = true;
  }

  ~ExprParserImpl() {
    assert(sharedParserState->hasExprParser);
    sharedParserState->hasExprParser = false;
    /// Free all the expression nodes.
    sharedParserState->exprAllocator.Reset();
  }

  // Expressions.  These methods always return a non-null ExprNode, but it may
  // be (or include) an Error node if parsing failed.
  void parseExpressionList(SmallVectorImpl<ExprNode *> &results);
  ExprNode *parseExpression();

private:
  ExprNode *parsePrimary();

  enum class Precedence {
    kInvalid, // Not a binary operator token.
    kLowest,  // Lowest precedence (most loosely bound).
    kAdd,
    kMul, // Highest precedence (most tightly bound).
  };
  std::pair<Precedence, ExprNode::Kind> getBinOpTokenPrecedenceAndKind() const;

  ExprNode *parseBinOpRHS(ExprNode *lhs, Precedence minPrec);

private:
  /// Allocate an expression node into the expression bump pointer allocator.
  template <typename T, typename... Args>
  T *alloc(Args &&...args) {
    void *node = sharedParserState->exprAllocator.Allocate(
        sizeof(T), llvm::Align::Of<T>());
    return new (node) T(std::forward<Args>(args)...);
  }

  /// Return an error node at the specified location.
  ExprNode *getErrorAtToken() { return alloc<ErrorNode>(getToken().getLoc()); };

  /// memcpy the specified ArrayRef into the expression allocator and return a
  /// pointer to the new data.  This cannot be used with things that have
  /// non-trivial copyctors/dtors because the expression allocator does run
  /// destructors.
  template <typename T>
  ArrayRef<T> copyArrayRef(ArrayRef<T> elements) {
    if (elements.empty())
      return elements;

    size_t dataSize = sizeof(T) * elements.size();
    T *result = static_cast<T *>(sharedParserState->exprAllocator.Allocate(
        dataSize, llvm::Align::Of<T>()));
    memcpy(result, elements.data(), dataSize);
    return ArrayRef<T>(result, elements.size());
  }
};
} // namespace M::KGEN::LIT

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

/// expression_list ::= expression ("," expression)* [","]
void ExprParserImpl::parseExpressionList(SmallVectorImpl<ExprNode *> &results) {
  // TODO: Support trailing comma for singleton tuple.
  (void)parseCommaSeparatedList([&]() -> ParseResult {
    results.push_back(parseExpression());
    return success();
  });
}

/// expression ::=
///
///
ExprNode *ExprParserImpl::parseExpression() {
  return parseBinOpRHS(parsePrimary(), Precedence::kLowest);
}

/// Return the operator precedence for the specified token or
std::pair<ExprParserImpl::Precedence, ExprNode::Kind>
ExprParserImpl::getBinOpTokenPrecedenceAndKind() const {
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
ExprNode *ExprParserImpl::parsePrimary() {
  ExprNode *result;
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
    ExprNode *subExpr = parseExpression();
    auto rpLoc = getToken().getLoc();
    // FIXME: This is terrible error recovery.
    if (parseToken(LitToken::r_paren,
                   "expected ')' in parenthesized expression"))
      return getErrorAtToken();
    result = alloc<ParenExprNode>(lpLoc, subExpr, rpLoc);
    break;
  }

  default:
    emitError("unexpected token in expression");
    result = getErrorAtToken();

    // TODO: Probably shouldn't consume this token in all cases, this could be
    // the introducer of another statement etc.  We should check to see what it
    // looks like and be smarter about this: consuming to end of paren, or to
    // introducer keyword.
    consumeToken();
    break;
  }

  // Parse postfix productions.
  while (1) {
    auto loc = getToken().getLoc();

    // Handle calls.
    if (consumeIf(LitToken::l_paren)) {
      SmallVector<ExprNode *> argExprs;
      // TODO: Handle comprehension arguments.
      if (!consumeIf(LitToken::r_paren)) {
        parseExpressionList(argExprs);
        if (parseToken(LitToken::r_paren, "expected ')' in call argument list"))
          return getErrorAtToken();
      }

      result = alloc<CallNode>(result, loc, copyArrayRef<ExprNode *>(argExprs));
      continue;
    }
    break;
  }

  return result;
}

/// Parse any binary operators that have precedence of at least `minPrec`.  This
/// stop if the current token isn't a binary operator or if it binds more
/// loosely than the specified precedence level.
ExprNode *ExprParserImpl::parseBinOpRHS(ExprNode *lhs, Precedence minPrec) {
  while (true) {
    auto [tokPrec, binOpKind] = getBinOpTokenPrecedenceAndKind();

    // If the next token is lower precedence than we are allowed to eat, return
    // successfully with what we ate already.  This also handles invalid tokens,
    // since they are treated as lower precedence than we ever allow.
    if (unsigned(tokPrec) < unsigned(minPrec))
      return lhs;

    SMLoc opLoc = getToken().getLoc();
    consumeToken();

    // Eat the next primary expression.
    // TODO: Need to decide how to handle syntactic errors, should propagate up
    // to the caller?
    ExprNode *rhs = parsePrimary();

    // If the operator we parse bind looser with the RHS than the operator after
    // the RHS, then give the RHS primary to the RHS.
    auto [nextTokPrec, nextBinOpKind] = getBinOpTokenPrecedenceAndKind();
    if (unsigned(tokPrec) < unsigned(nextTokPrec))
      rhs = parseBinOpRHS(rhs, Precedence(unsigned(tokPrec) + 1));

    // Merge LHS and RHS according to operator.
    lhs = alloc<BinOpNode>(binOpKind, lhs, opLoc, rhs);
  }
}

//===----------------------------------------------------------------------===//
// ExprParser implementation
//===----------------------------------------------------------------------===//

ExprParser::ExprParser(LitParserBase &existing)
    : pImpl(std::make_unique<ExprParserImpl>(existing)) {}

ExprParser::~ExprParser() {}

/// Parse an expression to check for syntactic validity, but throw it away
/// immediately.  Record the starting position for the expression in the
/// specified cursor.
ParseResult ExprParser::parseOverExpression(LitParserBase &p,
                                            Optional<LitLexerCursor> &cursor) {
  cursor = p.getLexer().getCursor();
  ExprParser exprParser(p);
  if (exprParser.parseExpression())
    return success();
  return failure();
}

void ExprParser::parseExpressionList(SmallVectorImpl<ExprNode *> &results) {
  return pImpl->parseExpressionList(results);
}

ExprNode *ExprParser::parseExpression() { return pImpl->parseExpression(); }
