//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Expression parsing in Mojo is done in with a 2-phase approach where we
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

#include "Signatures.h"

#include "KGEN/MojoParser/ExprNodes.h"
#include "KGEN/MojoParser/Lexer.h"
#include "KGEN/MojoParser/ParserBase.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace M::KGEN::LIT;
using namespace M;

//===----------------------------------------------------------------------===//
// Expression Parsing
//===----------------------------------------------------------------------===//

// See https://docs.python.org/3/reference/expressions.html#operator-precedence
enum class Precedence {
  kInvalid, // No precedence

  kUnpack,     // prefix: * or **
  kAssignExpr, // infix: := (walrus)
  kIfElse,     // infix: if - else
  kBoolOr,     // infix: or
  kBoolAnd,    // infix: and
  kBoolNot,    // prefix: not
  kComparison, // infix: in, not in, is, is not, <, <=, >, >=, !=, ==
  kOr,         // infix: |
  kXor,        // infix: ^
  kAnd,        // infix: &
  kShift,      // infix: <<, >>
  kSum,        // infix: +, -
  kTerm,       // infix: *, @, /, //, %
  kFactor,     // prefix: +, -, ~
  kPower,      // infix: **
  kAwait,      // prefix: await
  kPrimary,    // prefix: foo, "123", 123, 1.23, True, False, foo(1),
               //         foo.bar, foo[bar], lambda

  kExpression = kIfElse, // "expression" precedence is if/else.
  kHighest = kPrimary
};

namespace M::KGEN::LIT {
/// This class implements the ExprParser interface, implemented with the pImpl
/// idiom.
class ExprParser : public ParserBase {
public:
  ExprParser(SharedState &shared, Lexer &lexer,
             std::optional<size_t> stmtIndent)
      : ParserBase(shared, lexer), stmtIndent(stmtIndent) {}

  ~ExprParser() = default;

  // Expressions.
  ParseResult parseStarredList(SmallVectorImpl<ExprNode *> &results,
                               ArrayRef<Token::Kind> terminators,
                               SMLoc *firstCommaLoc = nullptr);
  ParseResult parseStarredListAsTuple(ExprNode *&result,
                                      ArrayRef<Token::Kind> terminators);

  ParseResult parseExpression(ExprNode *&result,
                              Precedence minPrec = Precedence::kExpression);
  ParseResult parseStarredItem(ExprNode *&result);

  template <typename T, typename... Args>
  T *alloc(Args &&...args) {
    return shared.allocPersistent<T>(std::forward<Args>(args)...);
  }

  template <typename T>
  ArrayRef<T> copyArrayRef(ArrayRef<T> elements) {
    return shared.getPersistentCopy(elements);
  }

  /// Return true if the current token is part of the current statement, false
  /// if it is the start of a new one.
  bool isTokenInCurrentStatement() const {
    return ParserBase::isTokenInCurrentStatement(stmtIndent);
  }

private:
  ParseResult parsePrimaryExpr(ExprNode *&result);
  ParseResult parsePrefixLParen(ExprNode *&result, SMLoc lparenLoc);
  ParseResult parsePrefixLSquare(ExprNode *&result, SMLoc lsquareLoc);
  ParseResult parsePrefixLBrace(DictionaryNode *&result, SMLoc lbraceLoc,
                                bool isSubscript);
  ParseResult parseAttributeRefSuffix(ExprNode *&result, SMLoc dotLoc);
  FailureOr<Operand> parseOperand(
      function_ref<ParseResult(ExprNode *&, Precedence)> parseOperandValue);
  ParseResult parseCallSuffix(ExprNode *&result, SMLoc lparenLoc);
  ParseResult parseExprOrSlice(ExprNode *&result);
  ParseResult parseSubscriptSuffix(ExprNode *&result, SMLoc lsquareLoc);
  ParseResult parseComparisonExpr(ExprNode *&result, ExprNode *rhs,
                                  ExprNode::Kind kind, SMLoc loc);
  ParseResult parseFunctionType(ExprNode *&result);
  ParseResult parseLambda(ExprNode *&result);
  ParseResult parseMagicFunction(ExprNode *&result);

  /// Check if the given operands (e.g. in a `(...)` call or `[...]` subscript)
  /// adhere to the Python grammar. Positional operands cannot appear after
  /// keyword operands, and duplicate keyword operands are not allowed. If the
  /// `isArgument` flag is true, operands are checked as if dynamic runtime
  /// operands (and explicitly unbound packs, i.e. `*_` are not allowed).
  /// Otherwise, operands are considered parameters.
  ParseResult checkOperands(ArrayRef<Operand>, bool isArgument);

  /// This specifies the indentation level of the start of the statement that
  /// contains this expression if the expression can exist at the end of the
  /// line.  This allows the expression parser to know when to keep parsing the
  /// expression on the next line - when it is more indented than the start of
  /// the current statement.  This is None when there is a trailing punctuator
  /// that naturally terminates the expression.
  std::optional<size_t> stmtIndent;
};
} // namespace M::KGEN::LIT

//===----------------------------------------------------------------------===//
// Parsing rules
//===----------------------------------------------------------------------===//

/// starred_list       ::=  starred_item ("," starred_item)* [","]
/// starred_item       ::=  assignment_expression | "*" or_expr
ParseResult ExprParser::parseStarredList(SmallVectorImpl<ExprNode *> &results,
                                         ArrayRef<Token::Kind> terminators,
                                         SMLoc *firstCommaLoc) {
  auto parseItem = [&]() -> ParseResult {
    return parseStarredItem(results.emplace_back(nullptr));
  };

  return parseCommaSeparatedList(parseItem, terminators, stmtIndent,
                                 firstCommaLoc);
}

/// Parse a starred_list, forming a single TupleExpr if a comma is present.
ParseResult
ExprParser::parseStarredListAsTuple(ExprNode *&result,
                                    ArrayRef<Token::Kind> terminators) {
  SmallVector<ExprNode *> exprs;
  SMLoc firstCommaLoc;
  if (parseStarredList(exprs, terminators, &firstCommaLoc))
    return failure();

  // If there was a tuple inside the parens, form it.
  if (firstCommaLoc.isValid())
    result = alloc<TupleNode>(firstCommaLoc, copyArrayRef<ExprNode *>(exprs));
  else {
    assert(exprs.size() == 1);
    result = exprs[0];
  }
  return success();
}

namespace {
/// This struct bundles up information related to infix binary operations.
struct InfixInfo {
  Precedence precedence;
  ExprNode::Kind nodeKind;

  // True when this operator is right associative:
  //   https://en.wikipedia.org/wiki/Operator_associativity
  // This matters when operators are at the same precedence level.  Consider
  // 7 op 4 op 2. The result could be either `(7 op 4) op 2` or `7 op (4 op 2)`.
  // The former result corresponds to the case the operators are
  // left-associative, the latter to when they are right-associative.
  //
  // Almost all operators in Python/Mojo are left associative.  Exceptions are
  // the power operator and assignment operator `=`.
  bool isRightAssociative;

  /// Classify a token for an infix operator.
  static InfixInfo get(Token::Kind tokKind, ParserBase &p) {
    // Helper to reduce boilerplate with isRightAssociative.
    auto get = [](Precedence precedence, ExprNode::Kind nodeKind,
                  bool isRightAssociative = false) -> InfixInfo {
      return {precedence, nodeKind, isRightAssociative};
    };

    switch (tokKind) {
    default:
      return get(Precedence::kInvalid, ExprNode::kLastBinOp);
    case Token::plus:
      return get(Precedence::kSum, ExprNode::kAdd);
    case Token::minus:
      return get(Precedence::kSum, ExprNode::kSub);
    case Token::star:
      return get(Precedence::kTerm, ExprNode::kMul);
    case Token::at:
      return get(Precedence::kTerm, ExprNode::kMatMul);
    case Token::slash:
      return get(Precedence::kTerm, ExprNode::kTrueDiv);
    case Token::slash_slash:
      return get(Precedence::kTerm, ExprNode::kFloorDiv);
    case Token::percent:
      return get(Precedence::kTerm, ExprNode::kMod);
    case Token::kw_or:
      return get(Precedence::kBoolOr, ExprNode::kBoolOr);
    case Token::kw_and:
      return get(Precedence::kBoolAnd, ExprNode::kBoolAnd);
    case Token::kw_not: {
      LexerCursor c;
      std::ignore = p.getCursor(c);
      p.consumeToken();
      Token next = p.getToken();
      c.restore(p.lexer);
      if (next.getKind() == Token::kw_in)
        return get(Precedence::kComparison, ExprNode::kCmpNotIn);
      // `not` by itself is not an infix operator, it is prefix.
      return get(Precedence::kInvalid, ExprNode::kLastBinOp);
    }
    case Token::kw_in:
      return get(Precedence::kComparison, ExprNode::kCmpIn);
    case Token::kw_is:
      return get(Precedence::kComparison, ExprNode::kCmpIs);
    case Token::less:
      return get(Precedence::kComparison, ExprNode::kCmpLT);
    case Token::less_equal:
      return get(Precedence::kComparison, ExprNode::kCmpLE);
    case Token::greater:
      return get(Precedence::kComparison, ExprNode::kCmpGT);
    case Token::greater_equal:
      return get(Precedence::kComparison, ExprNode::kCmpGE);
    case Token::exclaim_equal:
      return get(Precedence::kComparison, ExprNode::kCmpNE);
    case Token::equal_equal:
      return get(Precedence::kComparison, ExprNode::kCmpEQ);
    case Token::pipe:
      return get(Precedence::kOr, ExprNode::kOr);
    case Token::caret:
      return get(Precedence::kXor, ExprNode::kXor);
    case Token::amp:
      return get(Precedence::kAnd, ExprNode::kAnd);
    case Token::less_less:
      return get(Precedence::kShift, ExprNode::kLShift);
    case Token::right_right:
      return get(Precedence::kShift, ExprNode::kRShift);
    case Token::kw_if:
      return get(Precedence::kIfElse, ExprNode::kIfElse,
                 /*isRightAssociative=*/true);
    case Token::star_star:
      return get(Precedence::kPower, ExprNode::kPow,
                 /*isRightAssociative=*/true);
    case Token::colon_equal:
      return get(Precedence::kAssignExpr, ExprNode::kWalrus);
    }
  }
};
} // namespace

/// Parse an expression using top-down operator precedence parsing.  minPrec
/// specifies the minimum precedence that binary sub-expression must have to be
/// included.  Anything looser than the specified precedence is left for a
/// parent expression to parse.
ParseResult ExprParser::parseExpression(ExprNode *&result, Precedence minPrec) {
  // Parse any prefix expression like -1.
  if (parsePrimaryExpr(result))
    return failure();

  // Consume infix tokens until we meet a token whose tokPrecedence is equal or
  // lower than minPrec. This means that it collects all tokens that bind
  // together before returning to the operator that called it.
  InfixInfo infixInfo = InfixInfo::get(getToken().getKind(), *this);
  while (isTokenInCurrentStatement() && minPrec <= infixInfo.precedence) {
    Token::Kind tokKind = getToken().getKind();
    auto binOpLoc = consumeToken().getLoc();

    ExprNode *ifElseCond;
    SMLoc elseLoc;
    if (tokKind == Token::Kind::kw_if) {
      // Conditional if - else expression.
      // trueExpr 'if' condition 'else' falseExpr.
      // If/else operator needs special handling because it has an expression in
      // the middle of what can otherwise be parsed like a binary operator.
      if (parseExpression(ifElseCond, Precedence::kBoolOr))
        return failure();
      elseLoc = getToken().getLoc();
      if (parseToken(Token::Kind::kw_else,
                     "expecting an 'else' followed by an expression"))
        return failure();
    }

    // rhs 'is' 'not' lhs -> a is not True.
    if (tokKind == Token::Kind::kw_is && consumeIf(Token::Kind::kw_not))
      infixInfo.nodeKind = ExprNode::Kind::kCmpIsNot;
    // rhs 'not' 'in' lhs -> a not in {1, 2}.
    else if (tokKind == Token::Kind::kw_not && consumeIf(Token::Kind::kw_in)) {
      infixInfo.nodeKind = ExprNode::Kind::kCmpNotIn;
      infixInfo.precedence = Precedence::kComparison;
    }

    // Right associative operations can parse anything at the current operator
    // level on the right side, but left associative operators consume RHS that
    // binds more tightly than the current operator.
    Precedence subExprPrec = infixInfo.precedence;
    if (!infixInfo.isRightAssociative)
      subExprPrec = Precedence(unsigned(infixInfo.precedence) + 1);

    ExprNode *rhs = nullptr;
    if (parseExpression(rhs, subExprPrec))
      return failure();

    if (infixInfo.precedence == Precedence::kComparison) {
      // Comparison operators get special handling to treat 'a < b < c' as a
      // ChainedCmpOpNode.
      if (parseComparisonExpr(result, rhs, infixInfo.nodeKind, binOpLoc))
        return failure();
    } else if (tokKind == Token::Kind::kw_if) {
      result = alloc<IfElseOpNode>(result, binOpLoc, ifElseCond, elseLoc, rhs);
    } else
      result = alloc<BinOpNode>(infixInfo.nodeKind, result, binOpLoc, rhs);
    infixInfo = InfixInfo::get(getToken().getKind(), *this);
  }
  return success();
}

/// starred_item ::= assignment_expression | '*' bitwise_or
ParseResult ExprParser::parseStarredItem(ExprNode *&result) {
  SMLoc starLoc;
  if (consumeIf(Token::star, &starLoc)) {
    if (parseExpression(result, Precedence::kOr))
      return failure();
    result = alloc<UnaryOpNode>(ExprNode::kUnpack, starLoc, result);
    return success();
  }

  return parseExpression(result, Precedence::kAssignExpr);
}

/// Parse a chained comparison expression (ex. a < b < c) starting from the
/// first comparison given as input:
/// expr is the lhs and kind specifies the type of comparison, ex. kCmpLT.
/// This function returns in expr a ChainedCmpOpNode on success.
ParseResult ExprParser::parseComparisonExpr(ExprNode *&result, ExprNode *rhs,
                                            ExprNode::Kind kind, SMLoc loc) {
  SmallVector<ExprNode *> exprs;
  SmallVector<ExprNode::Kind> ops;
  exprs.push_back(result);
  exprs.push_back(rhs);
  ops.push_back(kind);
  InfixInfo infixInfo = InfixInfo::get(getToken().getKind(), *this);
  while (isTokenInCurrentStatement() &&
         infixInfo.precedence == Precedence::kComparison) {
    consumeToken();
    ExprNode *cmpOperand;
    if (parseExpression(cmpOperand, Precedence::kOr))
      return failure();
    exprs.push_back(cmpOperand);
    ops.push_back(infixInfo.nodeKind);
    infixInfo = InfixInfo::get(getToken().getKind(), *this);
  }
  result = alloc<ChainedCmpOpNode>(copyArrayRef<ExprNode *>(exprs),
                                   copyArrayRef<ExprNode::Kind>(ops), loc);
  return success();
}

/// Return true if the specified token kind is the start of a primary
/// expression.
static bool isPrimaryExprToken(Token::Kind tokKind) {
  switch (tokKind) {
  case Token::plus:
  case Token::minus:
  case Token::tilde:
  case Token::star:
  case Token::kw_await:
  case Token::kw_not:
  case Token::identifier:
  case Token::escaped_identifier:
  case Token::integer:
  case Token::kw_False:
  case Token::kw_True:
  case Token::kw_Self:
  case Token::kw__:
  case Token::float_num:
  case Token::string:
  case Token::kw_None:
  case Token::l_paren:
  case Token::l_square:
  case Token::l_brace:
  case Token::kw_async:
  case Token::kw_def:
  case Token::kw_lambda:
  case Token::kw_fn:
  case Token::kw___get_mvalue_as_litref:
  case Token::kw___get_litref_as_mvalue:
  case Token::kw___get_address_as_owned_value:
  case Token::kw___get_address_as_uninit_lvalue:
  case Token::kw___get_nearest_error_slot:
  case Token::kw___lifetime_of:
  case Token::kw___type_of:
    return true;
  default:
    return false;
  }
}

/// Given a token for a unary operator like await or ~, return the ExprNode
/// code to use along with the precedence of the subexpression we should parse.
static std::pair<ExprNode::Kind, Precedence>
getUnaryOpInfo(Token::Kind tokKind) {
  switch (tokKind) {
  default:
    llvm_unreachable("invalid unary token");
  case Token::kw_await:
    return {ExprNode::kAwait, Precedence::kPrimary};
  case Token::kw_not:
    return {ExprNode::kBoolNot, Precedence::kBoolNot};
  case Token::plus:
    return {ExprNode::kPos, Precedence::kFactor};
  case Token::minus:
    return {ExprNode::kNeg, Precedence::kFactor};
  case Token::tilde:
    return {ExprNode::kInvert, Precedence::kFactor};
  case Token::star:
    return {ExprNode::kUnpack, Precedence::kUnpack};
  }
}

/// Parse the expression identified by the current token and provided
/// `precedence`.  Store the resulting expression in `expr`.
/// Prefix expressions supported are:
///
/// primary ::=  atom | attributeref | subscription | slicing | call
///
/// atom    ::= identifier | literal | enclosure [TODO]
/// call    ::=  primary "(" [argument_list [","] | comprehension] ")"
///
/// enclosure ::= parenth_form | list_display | dict_display | set_display
///             | generator_expression | yield_atom
/// literal ::=
///     stringliteral | bytesliteral | integer | floatnumber | imagnumber
///
/// u_expr ::=  power | "-" u_expr | "+" u_expr | "~" u_expr
///
ParseResult ExprParser::parsePrimaryExpr(ExprNode *&result) {
  Token startTok = getToken();
  switch (startTok.getKind()) {
  case Token::plus:
  case Token::star:
  case Token::minus:
  case Token::tilde:
  case Token::kw_await:
  case Token::kw_not: { // u_expr
    consumeToken();
    // Get the kind enum and the precedence of the subexpression.
    auto [unaryKind, subExprPrec] = getUnaryOpInfo(startTok.getKind());
    ExprNode *expr = nullptr;
    if (parseExpression(expr, subExprPrec))
      return failure();
    result = alloc<UnaryOpNode>(unaryKind, startTok.getLoc(), expr);
    break;
  }
  case Token::identifier: // primary -> atom -> identifier
  case Token::escaped_identifier:
    consumeIdentifier();
    result = alloc<DeclRefNode>(startTok.getSpelling(),
                                startTok.is(Token::escaped_identifier));
    break;
  case Token::integer: // primary -> literal -> integer
    consumeToken(Token::integer);
    result = alloc<IntLiteralNode>(startTok.getSpelling());
    break;
  case Token::kw_False:
    consumeToken(Token::kw_False);
    result = alloc<BoolLiteralNode>(startTok.getLoc(), false);
    break;
  case Token::kw_True:
    consumeToken(Token::kw_True);
    result = alloc<BoolLiteralNode>(startTok.getLoc(), true);
    break;
  case Token::kw_Self:
    consumeToken(Token::kw_Self);
    result =
        alloc<SimpleLiteralNode>(ExprNode::kSelfLiteral, startTok.getLoc());
    break;
  case Token::kw__:
    consumeToken(Token::kw__);
    result =
        alloc<SimpleLiteralNode>(ExprNode::kDiscardLiteral, startTok.getLoc());
    break;
  case Token::float_num: // primary -> literal -> floatnumber
    consumeToken(Token::float_num);
    result = alloc<FloatLiteralNode>(startTok.getSpelling());
    break;
  case Token::string: { // primary -> literal -> stringliteral
    SmallVector<StringRef> spellings;
    // Python supports string literal concatenation
    while (getToken().is(Token::string) && isTokenInCurrentStatement()) {
      spellings.push_back(getToken().getSpelling());
      consumeToken(Token::string);
    }
    result = alloc<StringLiteralNode>(copyArrayRef<StringRef>(spellings));
    break;
  }
  case Token::kw_None:
    consumeToken(Token::kw_None);
    result =
        alloc<SimpleLiteralNode>(ExprNode::kNoneLiteral, startTok.getLoc());
    break;
  case Token::l_paren: // primary -> atom -> enclosure -> parenth_form
    consumeToken(Token::l_paren);
    if (parsePrefixLParen(result, startTok.getLoc()))
      return failure();
    break;
  case Token::l_square: // list_display
    consumeToken(Token::l_square);
    if (parsePrefixLSquare(result, startTok.getLoc()))
      return failure();
    break;
  case Token::l_brace: { // dict_display
    consumeToken(Token::l_brace);
    DictionaryNode *dict = nullptr;
    if (parsePrefixLBrace(dict, startTok.getLoc(),
                          /*isSubscript=*/false))
      return failure();
    result = dict;
    break;
  }

  case Token::kw_async:
  case Token::kw_def:
  case Token::kw_fn:
    if (failed(parseFunctionType(result)))
      return failure();
    break;

  case Token::kw_lambda:
    // We parse lambda as part of primary expressions to simplify the grammar.
    // They end on an 'expression' production though, and thus should not / can
    // not / will not consume any postfix attachments - the expression suffix
    // will have handled that, so we can return here.
    return parseLambda(result);

  case Token::kw___get_mvalue_as_litref:
  case Token::kw___get_litref_as_mvalue:
  case Token::kw___get_address_as_owned_value:
  case Token::kw___get_address_as_uninit_lvalue:
  case Token::kw___get_nearest_error_slot:
  case Token::kw___lifetime_of:
  case Token::kw___type_of:
    if (failed(parseMagicFunction(result)))
      return failure();
    break;

  default:
    emitTokenError("unexpected token in expression");
    result = nullptr;
    return failure();
  }

  // Check isPrimaryExprToken agrees with the cases above.
  assert(isPrimaryExprToken(startTok.getKind()) &&
         "isPrimaryExprToken out of sync with grammar above");

  // Parse postfix productions so long as they aren't the start of the next
  // statement.
  while (isTokenInCurrentStatement()) {
    auto loc = getToken().getLoc();

    // Handle "attributeref": x.y
    if (consumeIf(Token::dot)) {
      if (parseAttributeRefSuffix(result, loc))
        return failure();
      continue;
    }

    // Handle calls.
    if (consumeIf(Token::l_paren)) {
      if (parseCallSuffix(result, loc))
        return failure();
      continue;
    }

    // Handle "subscription" and "slicing" array subscripts and slicing.
    if (consumeIf(Token::l_square)) {
      if (parseSubscriptSuffix(result, loc))
        return failure();
      continue;
    }

    // Handle dictionary indexing.
    if (consumeIf(Token::l_brace)) {
      DictionaryNode *dict = nullptr;
      if (parsePrefixLBrace(dict, loc, /*isSubscript=*/true))
        return failure();
      result = alloc<DictSubscriptNode>(result, dict);
      continue;
    }

    // Handle postfix ^.  This is a bit tricky because ^ is also an infix
    // expression.  We handle this by consuming it and backtracking if needed.
    if (getToken().is(Token::caret)) {
      auto cursor = lexer.getCursor();
      auto loc = consumeToken(Token::caret).getLoc();

      // We know this is a binary ^ if there is a primary expression after it.
      if (isPrimaryExprToken(getToken().getKind()) &&
          isTokenInCurrentStatement()) {
        cursor.restore(lexer);
        break;
      }

      result = alloc<UnaryOpNode>(ExprNode::kTransfer, loc, result);
      continue;
    }

    break;
  }

  return success();
}

/// parenth_form ::= "(" [starred_expression] ")"
///
/// If the list contains at least one comma, it yields a tuple.
ParseResult ExprParser::parsePrefixLParen(ExprNode *&result, SMLoc lparenLoc) {
  SMLoc rparenLoc;

  ExprNode *element = nullptr;

  // Empty parens is a tuple.
  if (consumeIf(Token::r_paren, &rparenLoc)) {
    // Empty tuples are represented as ParenNode(TupleNode()) where the tuple
    // has no subexpressions.
    element = alloc<TupleNode>(lparenLoc, ArrayRef<ExprNode *>());
  } else if (parseStarredListAsTuple(element, Token::r_paren) ||
             parseToken(Token::r_paren,
                        "expected ')' in parenthesized expression", &rparenLoc))
    return failure();

  result = alloc<ParenNode>(lparenLoc, element, rparenLoc);
  return success();
}

/// list_display ::=  "[" [starred_list | comprehension [TODO]] "]"
ParseResult ExprParser::parsePrefixLSquare(ExprNode *&result,
                                           SMLoc lsquareLoc) {
  SMLoc rsquareLoc;
  SmallVector<ExprNode *> exprs;
  // Handle empty list: []
  if (consumeIf(Token::r_square, &rsquareLoc)) {
    result = alloc<ListNode>(lsquareLoc, exprs, rsquareLoc);
    return success();
  }

  if (parseStarredList(exprs, Token::r_square) || getLocation(rsquareLoc) ||
      parseToken(Token::r_square, "expected ']' in list expression"))
    return failure();
  result =
      alloc<ListNode>(lsquareLoc, copyArrayRef<ExprNode *>(exprs), rsquareLoc);
  return success();
}

/// dict_display       ::=  "{" [key_datum_list | dict_comprehension] "}"
/// key_datum_list     ::=  key_datum ("," key_datum)* [","]
/// key_datum          ::=  expression ":" expression | "**" or_expr
/// dict_comprehension ::=  expression ":" expression comp_for
ParseResult ExprParser::parsePrefixLBrace(DictionaryNode *&result,
                                          SMLoc lbraceLoc, bool isSubscript) {
  SMLoc rbraceLoc;
  // Handle empty dict: {}
  SmallVector<std::pair<ExprNode *, ExprNode *>> elements;

  /// Parse either a colon or an equal sign.  If we have an equal sign,
  /// diagnose it as a typo error.
  auto parseColonOrEqual = [&]() -> ParseResult {
    auto loc = getToken().getLoc();
    if (consumeIf(Token::equal)) {
      emitTokenError("expected ':' after dictionary key, not '='")
          << FixIt::replaceToken(loc, ":");
      return success();
    }
    return parseToken(Token::colon, "expected ':' in dictionary");
  };

  // Parse all the comma separated elements.
  while (elements.empty() || consumeIf(Token::comma)) {
    // Allow empty initializers and trailing comma in the initializer.
    if (getToken().is(Token::r_brace))
      break;

    ExprNode *key = nullptr, *value = nullptr;
    // Handle normal key:value and dictionary unpacking.  The later has a null
    // key in the DictionaryNode representation.
    if (!consumeIf(Token::star_star)) {
      if (parseExpression(key) || parseColonOrEqual())
        return failure();
    }
    if (parseExpression(value))
      return failure();
    elements.push_back({key, value});
  }

  // Handle dict_comprehension if present
  SMLoc forLoc;
  if (consumeIf(Token::kw_for, &forLoc)) {
    if (elements.size() != 1 || !elements[0].first)
      emitError(
          forLoc,
          "dictionary comprehension must start with single key:value pair");
    else
      emitError(forLoc, "TODO: dictionary comprehension parsing");
    return failure();
  }

  // Otherwise we must be out of elements.
  if (parseToken(Token::r_brace, "expected '}' at end of dictionary",
                 &rbraceLoc))
    return failure();

  result = alloc<DictionaryNode>(
      lbraceLoc, copyArrayRef<std::pair<ExprNode *, ExprNode *>>(elements),
      rbraceLoc);
  return success();
}

/// attributeref ::=  primary "." identifier
ParseResult ExprParser::parseAttributeRefSuffix(ExprNode *&result,
                                                SMLoc dotLoc) {
  Token token = getToken();
  StringRef spelling = token.getSpelling();
  if (parseIdentifier("expected name in attribute reference")) {
    // If we didn't get an identifier, recover by using an empty string.
    // Reuse the spelling buffer to preserve the expected location of the
    // identifier.
    spelling = StringRef(spelling.data(), 0);
  }

  result = alloc<AttributeRefNode>(result, dotLoc, spelling,
                                   token.is(Token::escaped_identifier));
  return success();
}

/// Parses a (subscript or call) operand expression with optional keyword. The
/// given callback is used to parse the operand value expression.
FailureOr<Operand> ExprParser::parseOperand(
    function_ref<ParseResult(ExprNode *&, Precedence)> parseOperandValue) {
  ExprNode *value;
  SMLoc startLoc = getToken().getLoc();
  if (getToken().is(Token::star)) {
    if (failed(parseStarredItem(value)))
      return failure();
    return Operand(value, startLoc, Operand::kStar);
  }
  if (consumeIf(Token::star_star)) {
    if (failed(parseExpression(value)))
      return failure();
    return Operand(value, startLoc, Operand::kStarStar);
  }

  // Check for a keyword argument.  We need look-ahead to determine whether
  // the token after the identifier is an equal sign.
  if (getToken().isIdentifier()) {
    auto cursor = lexer.getCursor();
    StringAttr name;
    (void)parseIdentifier(name, "<<already know this is identifier>>");
    if (consumeIf(Token::equal)) {
      if (failed(parseOperandValue(value, Precedence::kExpression)))
        return failure();
      return Operand(value, startLoc, Operand::kKeyword, name);
    }
    // Otherwise, we consumed the base expression, just pop it back off.
    cursor.restore(lexer);
  }

  // Parse this as an assignment_expression, allowing := operator.
  if (failed(parseOperandValue(value, Precedence::kAssignExpr)))
    return failure();
  return Operand(value, startLoc, Operand::kPositional);
}

/// call ::=  primary "(" [argument_list [","] | comprehension] ")"
/// argument_list ::= argument ("," argument)*
/// argument      ::= assignment_expression
/// argument      ::= "*" expression
/// argument      ::= identifier "=" expression
/// argument      ::= "**" expression
///
/// The official Python grammar is super complicated, but the constraint is
/// just that you can't have position arguments after keyword arguments. This
/// is easier to enforce imperatively than with BNF.
ParseResult ExprParser::parseCallSuffix(ExprNode *&result, SMLoc lparenLoc) {
  SmallVector<Operand> operands;
  SMLoc rparenLoc;
  if (!consumeIf(Token::r_paren, &rparenLoc)) {
    // Expressions continue maximally because we are within ()'s.
    llvm::SaveAndRestore<std::optional<size_t>> X(stmtIndent, std::nullopt);

    // Parse an argument.
    auto parseCallOperand = [&]() -> ParseResult {
      auto parseOperandValue = [&](ExprNode *&result, Precedence minPrec) {
        return parseExpression(result, minPrec);
      };
      FailureOr<Operand> operandOr = parseOperand(parseOperandValue);
      if (failed(operandOr))
        return failure();
      operands.emplace_back(std::move(*operandOr));
      return success();
    };

    // TODO: Handle comprehension argument.
    if (parseCommaSeparatedList(parseCallOperand, Token::r_paren) ||
        parseToken(Token::r_paren, "expected ')' in call argument list",
                   &rparenLoc)) {
      return failure();
    }
  }

  if (checkOperands(operands, /*isArgument=*/true))
    return failure();

  // Otherwise we're good to go.
  result = alloc<CallNode>(result, lparenLoc, copyArrayRef<Operand>(operands),
                           rparenLoc);
  return success();
}

/// Parses a slice or an ordinary expression
ParseResult ExprParser::parseExprOrSlice(ExprNode *&result) {
  // If this has a leading expr it could be an expr only or could be the first
  // (optional) part of a slice.
  if (getToken().isNot(Token::colon)) {
    if (parseExpression(result))
      return failure();
    // If we had an expr with no trailing colon, then we are done with the
    // expr case.
    if (getToken().isNot(Token::colon, Token::equal))
      return success();
  } else {
    // If it starts with a colon, this is a slice without a lower bound.
    result = nullptr;
  }

  /// Consume either a colon or an equal sign.  If we have an equal sign,
  /// diagnose it as a typo error.
  auto consumeColonOrEqual = [&]() -> SMLoc {
    assert(getToken().isAny(Token::colon, Token::equal));
    auto loc = getToken().getLoc();
    if (getToken().is(Token::equal))
      emitTokenError("expected ':' in subscript slice, not '='")
          << FixIt::replaceToken(loc, ":");
    consumeToken();
    return loc;
  };

  // Okay we have at least one colon, so we have a slice.
  SMLoc colon1Loc = consumeColonOrEqual(), colon2Loc;
  ExprNode *secondExpr = nullptr, *thirdExpr = nullptr;

  // Parse the second expr if present.
  if (getToken().isNot(Token::colon, Token::equal, Token::comma,
                       Token::r_square))
    if (parseExpression(secondExpr))
      return failure();

  // Parse a second colon if present and stride expression.
  if (getToken().isAny(Token::colon, Token::equal)) {
    colon2Loc = consumeColonOrEqual();
    if (getToken().isNot(Token::comma, Token::r_square))
      if (parseExpression(thirdExpr))
        return failure();
  }

  result =
      alloc<SliceNode>(result, colon1Loc, secondExpr, colon2Loc, thirdExpr);
  return success();
}

ParseResult ExprParser::checkOperands(ArrayRef<Operand> operands,
                                      bool isArgument) {
  std::string argOrParam = isArgument ? "argument" : "parameter";
  // We keep a map of "name -> operand" so that we can emit better diagnostics.
  llvm::SmallDenseMap<StringAttr, const Operand *> kwOperands;
  for (const Operand &operand : operands) {
    SMLoc loc = operand.getLoc();
    if (operand.isUnpackedKeyword())
      return emitError(loc, "keyword unpacking not supported yet");
    if (operand.isPositional() && !kwOperands.empty()) {
      return emitError(loc, "positional ")
             << argOrParam << " follows keyword " << argOrParam;
    }
    if (operand.isKeyword()) {
      auto [it, addedNew] = kwOperands.try_emplace(operand.name, &operand);
      if (!addedNew) {
        auto diag = emitError(loc, "duplicate keyword ")
                    << argOrParam << " " << operand.name;
        diag.attachNote(it->getSecond()->getLoc())
            << "previously specified here";
        return std::move(diag);
      }
    }
  }
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
  llvm::SaveAndRestore<std::optional<size_t>> X(stmtIndent, std::nullopt);

  // If we have an empty parameter list, we return immediately.
  SMLoc rsquareLoc;
  if (consumeIf(Token::r_square, &rsquareLoc)) {
    result = alloc<SubscriptNode>(result, lsquareLoc, ArrayRef<Operand>(),
                                  rsquareLoc);
    return success();
  }

  // Helper to parse an input or a result parameter.
  auto parseSubscriptOperand =
      [&](SmallVectorImpl<Operand> &parsed) -> ParseResult {
    auto parseOperandValue = [&](ExprNode *&result, Precedence minPrec) {
      // Precedence is ignored here on purpose; we don't allow walrus here.
      return parseExprOrSlice(result);
    };
    FailureOr<Operand> operandOr = parseOperand(parseOperandValue);
    if (failed(operandOr))
      return failure();
    parsed.emplace_back(std::move(*operandOr));
    return success();
  };

  SmallVector<Operand> operands;
  if (parseCommaSeparatedList([&]() { return parseSubscriptOperand(operands); },
                              {Token::r_square, Token::minus_greater}) ||
      getLocation(rsquareLoc))
    return failure();

  if (checkOperands(operands, /*isArgument=*/false))
    return failure();

  if (parseToken(Token::r_square, "expected ']' in call argument list"))
    return failure();
  result = alloc<SubscriptNode>(result, lsquareLoc,
                                copyArrayRef<Operand>(operands), rsquareLoc);
  return success();
}

ParseResult ExprParser::parseFunctionType(ExprNode *&result) {
  SMLoc baseLoc = getToken().getLoc();

  ParsedParamList paramList;
  ParsedArgumentList fnSignature;

  ExprNode *resultTypeExpr = nullptr, *resultRefLifetimeExpr = nullptr;
  bool isDef = false;

  // Parse the function effects from the leading keyword.
  fnSignature.effects.setAsync(consumeIf(Token::kw_async));
  if (consumeToken().is(Token::kw_def)) {
    fnSignature.effects.setThrows();
    isDef = true;
  }

  // Parameter signature, argument list and the function effects next.
  if (paramList.parseOptionalParameters(*this, ArgListKind::kFnTypeParamList) ||
      fnSignature.parseArgumentListAndEffects(*this,
                                              ArgListKind::kFnTypeArgList))
    return failure();

  // Parse the result type.
  SMLoc endLoc = getToken().getEndLoc();
  SMLoc resultLoc = getToken().getLoc();
  if (!isDef || getToken().is(Token::minus_greater)) {
    if (parseToken(Token::minus_greater, "expected '->' in function type"))
      return failure();

    // Parse a result reference if present.
    if (parseRefSpecifier(resultRefLifetimeExpr) ||
        ParserBase::parseExpression(resultTypeExpr, stmtIndent))
      return failure();
  }

  result = alloc<FunctionTypeNode>(
      baseLoc, copyArrayRef<ParsedArgument>(paramList.params),
      copyArrayRef<ParsedArgument>(fnSignature.parsedArgs), resultTypeExpr,
      resultRefLifetimeExpr, fnSignature.effects, endLoc, isDef, resultLoc);
  return success();
}

/// lambda_expr ::= "lambda" [parameter_list] argument_list ":" expression
ParseResult ExprParser::parseLambda(ExprNode *&result) {
  SMLoc lambdaLoc = consumeToken(Token::kw_lambda).getLoc();

  ParsedArgumentList parsedSignature;

  // Mojo supports naked parameters without type annotations for compatibility
  // with Python, but also supports parethesized ones.  We can only support
  // type annotations in parentheses since we'd otherwise have ambiguity with
  // the ":" in the lambda expression.
  if (getToken().is(Token::colon)) {
    // Nothing to parse.
  } else {
    // Parse general parenthesized argument list if a paren is present,
    // otherwise a bare identifier list.
    auto kind = getToken().is(Token::l_paren) ? ArgListKind::kArgList
                                              : ArgListKind::kBareLambdaArgList;
    if (parsedSignature.parseArgumentListAndEffects(*this, kind))
      return failure();
  }

  ExprNode *bodyExpr = nullptr;
  if (parseToken(Token::colon, "expected ':' in lambda expression") ||
      ParserBase::parseExpression(bodyExpr, stmtIndent))
    return failure();

  // Ok, we have a syntactically correct lambda, but we still don't support
  // them yet.
  emitError(lambdaLoc, "Mojo doesn't support lambda expressions yet");
  return failure();
}

ParseResult ExprParser::parseMagicFunction(ExprNode *&result) {
  ExprNode::Kind nodeKind;
  switch (getToken().getKind()) {
  default:
    llvm_unreachable("bad token");
  case Token::kw___get_address_as_uninit_lvalue:
    nodeKind = ExprNode::kGetAddressAsUninitLValue;
    break;
  case Token::kw___get_mvalue_as_litref:
    nodeKind = ExprNode::kGetMValueAsLitRef;
    break;
  case Token::kw___get_litref_as_mvalue:
    nodeKind = ExprNode::kGetLitRefAsMValue;
    break;
  case Token::kw___get_address_as_owned_value:
    nodeKind = ExprNode::kGetAddressAsOwned;
    break;
  case Token::kw___get_nearest_error_slot:
    nodeKind = ExprNode::kGetNearestErrorSlot;
    break;
  case Token::kw___lifetime_of:
    nodeKind = ExprNode::kLifetimeOf;
    break;
  case Token::kw___type_of:
    nodeKind = ExprNode::kTypeOf;
    break;
  }
  SMLoc baseLoc = consumeToken().getLoc();

  SmallVector<ExprNode *> subExprs;
  SMLoc rpLoc;
  // All "magic" functions take an argument.
  if (parseToken(Token::l_paren, "expected '('"))
    return failure();
  if (!consumeIf(Token::r_paren, &rpLoc)) {
    if (parseCommaSeparatedList(
            [&] { return parseExpression(subExprs.emplace_back()); },
            Token::r_paren) ||
        parseToken(Token::r_paren, "expected ')'", &rpLoc))
      return failure();
  }

  result = alloc<MagicFunctionNode>(nodeKind, baseLoc,
                                    copyArrayRef<ExprNode *>(subExprs), rpLoc);
  return success();
}

//===----------------------------------------------------------------------===//
// ExprParser implementation
//===----------------------------------------------------------------------===//

/// Parse an expression_list production, returning a single expression or a
/// tuple expression if there are commas.
ParseResult ParserBase::parseExpressionList(ExprNode *&result,
                                            std::optional<size_t> stmtIndent) {
  ExprParser parser(shared, getLexer(), stmtIndent);
  SmallVector<ExprNode *> exprs;
  auto parseItem = [&]() -> ParseResult {
    return parser.parseExpression(exprs.emplace_back(nullptr),
                                  Precedence::kExpression);
  };

  SMLoc firstCommaLoc;
  if (parser.parseCommaSeparatedList(parseItem, /*terminators=*/{}, stmtIndent,
                                     &firstCommaLoc))
    return failure();

  // If we parsed multiple items or have a comma, then this is actually a tuple.
  // If there was a tuple inside the parens, form it.
  if (exprs.size() != 1 || firstCommaLoc.isValid())
    result = parser.alloc<TupleNode>(firstCommaLoc,
                                     parser.copyArrayRef<ExprNode *>(exprs));
  else
    result = exprs[0];
  return success();
}

ParseResult ParserBase::parseOptionalIdentifier(StringAttr &result,
                                                Token::Kind delimiter,
                                                SMLoc *loc) {
  LexerCursor cursor(lexer);
  result = StringAttr::get(getContext(), getToken().getSpelling());
  if (consumeIf(Token::identifier, loc)) {
    if (loc)
      *loc = getToken().getLoc();
    if (getToken().is(delimiter))
      return success();
  }
  cursor.restore(lexer);
  return failure();
}

/// Expression parsing.  Each of these take a `stmtIndent` specifier that
/// indicates the indentation level of the start of the statement that
/// contains this expression if the expression can exist at the end of the
/// line.  This allows the expression parser to know when to keep parsing the
/// expression on the next line - when it is more indented than the start of
/// the current statement.  This can be passed in as None when there is a
/// trailing punctuator that naturally terminates the expression.
ParseResult ParserBase::parseExpression(ExprNode *&result,
                                        std::optional<size_t> stmtIndent) {
  return ExprParser(shared, getLexer(), stmtIndent)
      .parseExpression(result, Precedence::kExpression);
}
ParseResult
ParserBase::parseAssignExpression(ExprNode *&result,
                                  std::optional<size_t> stmtIndent) {
  return ExprParser(shared, getLexer(), stmtIndent)
      .parseExpression(result, Precedence::kAssignExpr);
}

ParseResult ParserBase::parseStarredItem(ExprNode *&result) {
  return ExprParser(shared, getLexer(), std::nullopt).parseStarredItem(result);
}

/// If the specified token is an '=' or '+=' sort of token, return the
/// expression kind, otherwise return null.
static std::optional<ExprNode::Kind> getAssignmentKind(Token::Kind tokenKind) {
  switch (tokenKind) {
  default:
    return std::nullopt;
  case Token::equal:
    return ExprNode::kAssign;
  case Token::plus_equal:
    return ExprNode::kIAdd;
  case Token::minus_equal:
    return ExprNode::kISub;
  case Token::star_equal:
    return ExprNode::kIMul;
  case Token::at_equal:
    return ExprNode::kIMatMul;
  case Token::slash_equal:
    return ExprNode::kITrueDiv;
  case Token::percent_equal:
    return ExprNode::kIMod;
  case Token::amp_equal:
    return ExprNode::kIAnd;
  case Token::pipe_equal:
    return ExprNode::kIOr;
  case Token::caret_equal:
    return ExprNode::kIXor;
  case Token::less_less_equal:
    return ExprNode::kILShift;
  case Token::right_right_equal:
    return ExprNode::kIRShift;
  case Token::star_star_equal:
    return ExprNode::kIPow;
  case Token::slash_slash_equal:
    return ExprNode::kIFloorDiv;
  }
}

/// Parse a simple_stmt production containing an expression, including
/// expression_stmt and {augmented_|annotated_|}assignment_stmt.
///
/// expression_stmt ::= starred_expression
/// assignment_stmt ::=
///              (expression_list "=")+ (starred_expression | yield_expression)
/// augmented_assignment_stmt ::=
///                        expression "+=" (expression_list | yield_expression)
///
/// NOTE: we do not handle this as part of binary operator parsing because the
/// grammar is so weird and different with yield expressions, expression_list,
/// and starred expression.
ParseResult ParserBase::parseSimpleStmtExprs(ExprNode *&result,
                                             size_t stmtIndent) {
  ExprParser p(shared, getLexer(), stmtIndent);

  // We have three very different grammar productions that all start with an
  // expression, starred_expression, or assignment_expression plus the target
  // stuff in various mixes.  This is all the Python grammar trying to enforce
  // semantic considerations in the grammar, which is unpleasant.  Implement
  // this by parsing the most general thing and sorting out what is valid later.
  ExprNode *expr = nullptr;
  // TODO: Handle yield_expression.
  if (p.parseStarredListAsTuple(expr, /*terminators=*/{}))
    return failure();

  // If that was it, just return the expression.
  std::optional<ExprNode::Kind> assignKind =
      getAssignmentKind(p.getToken().getKind());
  if (!p.isTokenInCurrentStatement() || !assignKind.has_value()) {
    result = expr;
    return success();
  }
  SMLoc assignLoc = p.consumeToken().getLoc();

  // If we have an = or += operator, parse the rest of the statement pieces;
  // assignments are right associative, so we just recurse to handle this.
  ExprNode *rhsExpr = nullptr;
  if (parseSimpleStmtExprs(rhsExpr, stmtIndent))
    return failure();

  result = p.alloc<BinOpNode>(assignKind.value(), expr, assignLoc, rhsExpr);
  return success();
}

ParseResult ParserBase::parseVarInitExpression(ExprNode *&result,
                                               size_t stmtIndent) {
  return ExprParser(shared, getLexer(), stmtIndent)
      .parseStarredListAsTuple(result, /*terminators=*/{});
}
