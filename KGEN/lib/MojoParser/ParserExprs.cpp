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

#include "DeclResolver.h"
#include "ExprNodes.h"
#include "Lexer.h"
#include "ParserBase.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace M::KGEN::LIT;
using namespace M;

//===----------------------------------------------------------------------===//
// Expression Parsing
//===----------------------------------------------------------------------===//

// See https://docs.python.org/3/reference/expressions.html#operator-precedence
enum class Precedence {
  kInvalid, // No precedence

  // infix: =, +=, -=: These are not a Python 'expression', and are not allowed
  // in parens, they are only allowed as a top level statement.
  kAssignStmt,

  kAssignExpr, // "assignment_expression" precedence
  kWalrus,     // infix: := (walrus)
  kExpression, // "expression" precedence
  kIfElse,     // infix: if - else + lambda.
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
               //         foo.bar, foo[bar]
  kHighest = kPrimary
};

namespace M::KGEN::LIT {
/// This class implements the ExprParser interface, implemented with the pImpl
/// idiom.
class ExprParser : public ParserBase {
public:
  ExprParser(Lexer &lexer, std::optional<size_t> stmtIndent)
      : ParserBase(lexer), stmtIndent(stmtIndent) {}

  ~ExprParser() {}

  // Expressions.
  ParseResult parseStarredList(SmallVectorImpl<ExprNode *> &results,
                               ArrayRef<Token::Kind> terminators,
                               SMLoc *firstCommaLoc = nullptr);

  ParseResult parseExpression(ExprNode *&result,
                              Precedence minPrec = Precedence::kExpression);
  ParseResult parseStarredItem(ExprNode *&result);

  ExprNode *getNoneExpr(SMLoc loc) {
    return alloc<SimpleLiteralNode>(ExprNode::kNoneLiteral, loc);
  };

  template <typename T, typename... Args>
  T *alloc(Args &&...args) {
    return shared.allocPersistent<T>(std::forward<Args>(args)...);
  }

  template <typename T>
  ArrayRef<T> copyArrayRef(ArrayRef<T> elements) {
    return shared.getPersistentCopy(elements);
  }

private:
  /// Return true if the current token is the start of another statement, false
  /// if it is part of this one.
  bool isTokenStartOfNextStatement();

  ParseResult parsePrimaryExpr(ExprNode *&result);
  ParseResult parsePrefixLParen(ExprNode *&result, SMLoc lparenLoc);
  ParseResult parsePrefixLSquare(ExprNode *&result, SMLoc lsquareLoc);
  ParseResult parsePrefixLBrace(DictionaryNode *&result, SMLoc lbraceLoc,
                                bool isSubscript);
  ParseResult parseAttributeRefSuffix(ExprNode *&result, SMLoc dotLoc);
  ParseResult parseCallSuffix(ExprNode *&result, SMLoc lparenLoc);
  ParseResult parseSubscriptSuffix(ExprNode *&result, SMLoc lsquareLoc);
  ParseResult parseComparisonExpr(ExprNode *&result, ExprNode *rhs,
                                  ExprNode::Kind kind, SMLoc loc);
  ParseResult parseFunctionType(ExprNode *&result);
  ParseResult parseAddressConvert(ExprNode *&result);

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

  return parseCommaSeparatedList(parseItem, terminators, firstCommaLoc);
}

namespace {
/// This struct bundles up information related to infix binary operations.
struct InfixInfo {
  Precedence precedence;
  ExprNode::Kind nodeKind;
  bool isLeftAssociative;

  /// Classify a token for an infix operator.
  static InfixInfo get(Token::Kind tokKind) {
    switch (tokKind) {
    default:
      return {Precedence::kInvalid, ExprNode::kLastBinOp, false};
    case Token::equal:
      return {Precedence::kAssignExpr, ExprNode::kAssign, false};
    case Token::plus_equal:
      return {Precedence::kAssignExpr, ExprNode::kIAdd, false};
    case Token::minus_equal:
      return {Precedence::kAssignExpr, ExprNode::kISub, false};
    case Token::star_equal:
      return {Precedence::kAssignExpr, ExprNode::kIMul, false};
    case Token::at_equal:
      return {Precedence::kAssignExpr, ExprNode::kIMatMul, false};
    case Token::slash_equal:
      return {Precedence::kAssignExpr, ExprNode::kITrueDiv, false};
    case Token::percent_equal:
      return {Precedence::kAssignExpr, ExprNode::kIMod, false};
    case Token::amp_equal:
      return {Precedence::kAssignExpr, ExprNode::kIAnd, false};
    case Token::pipe_equal:
      return {Precedence::kAssignExpr, ExprNode::kIOr, false};
    case Token::caret_equal:
      return {Precedence::kAssignExpr, ExprNode::kIXor, false};
    case Token::less_less_equal:
      return {Precedence::kAssignExpr, ExprNode::kILShift, false};
    case Token::right_right_equal:
      return {Precedence::kAssignExpr, ExprNode::kIRShift, false};
    case Token::star_star_equal:
      return {Precedence::kAssignExpr, ExprNode::kIPow, false};
    case Token::slash_slash_equal:
      return {Precedence::kAssignExpr, ExprNode::kIFloorDiv, false};
    case Token::plus:
      return {Precedence::kSum, ExprNode::kAdd, false};
    case Token::minus:
      return {Precedence::kSum, ExprNode::kSub, false};
    case Token::star:
      return {Precedence::kTerm, ExprNode::kMul, false};
    case Token::at:
      return {Precedence::kTerm, ExprNode::kMatMul, false};
    case Token::slash:
      return {Precedence::kTerm, ExprNode::kTrueDiv, false};
    case Token::slash_slash:
      return {Precedence::kTerm, ExprNode::kFloorDiv, false};
    case Token::percent:
      return {Precedence::kTerm, ExprNode::kMod, false};
    case Token::kw_or:
      return {Precedence::kBoolOr, ExprNode::kBoolOr, false};
    case Token::kw_and:
      return {Precedence::kBoolAnd, ExprNode::kBoolAnd, false};
    case Token::kw_not:
      return {Precedence::kBoolNot, ExprNode::kBoolNot, false};
    case Token::kw_in:
      return {Precedence::kComparison, ExprNode::kCmpIn, false};
    case Token::kw_is:
      return {Precedence::kComparison, ExprNode::kCmpIs, false};
    case Token::less:
      return {Precedence::kComparison, ExprNode::kCmpLT, false};
    case Token::less_equal:
      return {Precedence::kComparison, ExprNode::kCmpLE, false};
    case Token::greater:
      return {Precedence::kComparison, ExprNode::kCmpGT, false};
    case Token::greater_equal:
      return {Precedence::kComparison, ExprNode::kCmpGE, false};
    case Token::exclaim_equal:
      return {Precedence::kComparison, ExprNode::kCmpNE, false};
    case Token::equal_equal:
      return {Precedence::kComparison, ExprNode::kCmpEQ, false};
    case Token::pipe:
      return {Precedence::kOr, ExprNode::kOr, false};
    case Token::caret:
      return {Precedence::kXor, ExprNode::kXor, false};
    case Token::amp:
      return {Precedence::kAnd, ExprNode::kAnd, false};
    case Token::less_less:
      return {Precedence::kShift, ExprNode::kLShift, false};
    case Token::right_right:
      return {Precedence::kShift, ExprNode::kRShift, false};
    case Token::kw_if:
      return {Precedence::kIfElse, ExprNode::kIfElse, false};
    case Token::star_star:
      return {Precedence::kPower, ExprNode::kPow, true};
    case Token::kw_await:
      return {Precedence::kAwait, ExprNode::kAwait, false};
    case Token::colon_equal:
      return {Precedence::kWalrus, ExprNode::kWalrus, false};
    }
  }
};
} // namespace

/// Parse an expression using top-down operator precedence parsing.
ParseResult ExprParser::parseExpression(ExprNode *&expr, Precedence minPrec) {
  // Parse any prefix expression like -1.
  if (parsePrimaryExpr(expr))
    return failure();

  // Consume infix tokens until we meet a token whose tokPrecedence is equal or
  // lower than minPrec. This means that it collects all tokens that bind
  // together before returning to the operator that called it.
  InfixInfo infixInfo = InfixInfo::get(getToken().getKind());
  while (!isTokenStartOfNextStatement() && minPrec < infixInfo.precedence) {
    Token::Kind tokKind = getToken().getKind();
    auto binOpLoc = consumeToken().getLoc();

    if (tokKind == Token::Kind::kw_if) {
      // Conditional if - else expression.
      // trueExpr 'if' condition 'else' falseExpr.
      ExprNode *cond;
      if (parseExpression(cond, infixInfo.precedence))
        return failure();

      ExprNode *falseExpr;
      auto elseLoc = getToken().getLoc();
      if (parseToken(Token::Kind::kw_else,
                     "expecting an 'else' followed by an expression") ||
          parseExpression(falseExpr, infixInfo.precedence))
        return failure();
      expr = alloc<IfElseOpNode>(expr, binOpLoc, cond, elseLoc, falseExpr);
      infixInfo = InfixInfo::get(getToken().getKind());
      continue;
    }

    // rhs 'is' 'not' lhs -> a is not True.
    if (tokKind == Token::Kind::kw_is && consumeIf(Token::Kind::kw_not))
      infixInfo.nodeKind = ExprNode::Kind::kCmpIsNot;
    // rhs 'not' 'in' lhs -> a not in {1, 2}.
    else if (tokKind == Token::Kind::kw_not && consumeIf(Token::Kind::kw_in)) {
      infixInfo.nodeKind = ExprNode::Kind::kCmpNotIn;
      infixInfo.precedence = Precedence::kComparison;
    }

    // Handle left associative operations.
    if (infixInfo.isLeftAssociative)
      infixInfo.precedence = Precedence(unsigned(infixInfo.precedence) - 1);

    ExprNode *rhs;
    if (parseExpression(rhs, infixInfo.precedence))
      return failure();

    if (infixInfo.precedence != Precedence::kComparison)
      expr = alloc<BinOpNode>(infixInfo.nodeKind, expr, binOpLoc, rhs);
    else if (parseComparisonExpr(expr, rhs, infixInfo.nodeKind, binOpLoc))
      return failure();
    infixInfo = InfixInfo::get(getToken().getKind());
  }
  return success();
}

/// starred_item ::= assignment_expression | '*' bitwise_or
ParseResult ExprParser::parseStarredItem(ExprNode *&expr) {
  SMLoc starLoc;
  if (consumeIf(Token::star, &starLoc)) {
    if (parseExpression(expr, Precedence::kOr))
      return failure();
    expr = alloc<UnaryOpNode>(ExprNode::kUnpack, starLoc, expr);
    return success();
  }

  return parseExpression(expr, Precedence::kAssignExpr);
}

static ExprNode::Kind getUnaryOpKind(Token::Kind tokKind) {
  switch (tokKind) {
  default:
    llvm_unreachable("invalid unary token");
  case Token::kw_await:
    return ExprNode::kAwait;
  case Token::kw_not:
    return ExprNode::kBoolNot;
  case Token::plus:
    return ExprNode::kPos;
  case Token::minus:
    return ExprNode::kNeg;
  case Token::tilde:
    return ExprNode::kInvert;
  }
}
/// Parse a chained comparison expression (ex. a < b < c) starting from the
/// first comparison given as input:
/// expr is the lhs and kind specifies the type of comparison, ex. kCmpLT.
/// This function returns in expr a ChainedCmpOpNode on success.
ParseResult ExprParser::parseComparisonExpr(ExprNode *&expr, ExprNode *rhs,
                                            ExprNode::Kind kind, SMLoc loc) {
  SmallVector<ExprNode *> exprs;
  SmallVector<ExprNode::Kind> ops;
  exprs.push_back(expr);
  exprs.push_back(rhs);
  ops.push_back(kind);
  InfixInfo infixInfo = InfixInfo::get(getToken().getKind());
  while (!isTokenStartOfNextStatement() &&
         infixInfo.precedence == Precedence::kComparison) {
    consumeToken();
    ExprNode *cmpOperand;
    if (parseExpression(cmpOperand, Precedence::kComparison))
      return failure();
    exprs.push_back(cmpOperand);
    ops.push_back(infixInfo.nodeKind);
    infixInfo = InfixInfo::get(getToken().getKind());
  }
  expr = alloc<ChainedCmpOpNode>(copyArrayRef<ExprNode *>(exprs),
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
  case Token::kw_await:
  case Token::kw_not:
  case Token::identifier:
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
  case Token::kw_fn:
  case Token::kw___get_address_as_lvalue:
  case Token::kw___get_lvalue_as_address:
  case Token::kw___get_address_as_owned_value:
  case Token::kw___get_address_as_uninit_lvalue:
    return true;
  default:
    return false;
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
  Token::Kind tokKind = getToken().getKind();
  switch (tokKind) {
  case Token::plus:
  case Token::minus:
  case Token::tilde:
  case Token::kw_await:
  case Token::kw_not: { // u_expr
    auto unaryLoc = consumeToken().getLoc();
    ExprNode *expr;
    Precedence precedence = InfixInfo::get(tokKind).precedence;
    if (parseExpression(expr, precedence))
      return failure();
    result = alloc<UnaryOpNode>(getUnaryOpKind(tokKind), unaryLoc, expr);
    break;
  }
  case Token::identifier: // primary -> atom -> identifier
    result = alloc<DeclRefNode>(getToken().getSpelling());
    consumeToken(Token::identifier);
    break;
  case Token::integer: // primary -> literal -> integer
    result = alloc<IntLiteralNode>(getToken().getSpelling());
    consumeToken(Token::integer);
    break;
  case Token::kw_False:
    result = alloc<BoolLiteralNode>(getToken().getLoc(), false);
    consumeToken(Token::kw_False);
    break;
  case Token::kw_True:
    result = alloc<BoolLiteralNode>(getToken().getLoc(), true);
    consumeToken(Token::kw_True);
    break;
  case Token::kw_Self:
    result =
        alloc<SimpleLiteralNode>(ExprNode::kSelfLiteral, getToken().getLoc());
    consumeToken(Token::kw_Self);
    break;
  case Token::kw__:
    result = alloc<SimpleLiteralNode>(ExprNode::kDiscardLiteral,
                                      getToken().getLoc());
    consumeToken(Token::kw__);
    break;
  case Token::float_num: // primary -> literal -> floatnumber
    result = alloc<FloatLiteralNode>(getToken().getSpelling());
    consumeToken(Token::float_num);
    break;
  case Token::string: { // primary -> literal -> stringliteral
    SmallVector<StringRef> spellings;
    // Python supports string literal concatenation
    while (getToken().is(Token::string) && !isTokenStartOfNextStatement()) {
      spellings.push_back(getToken().getSpelling());
      consumeToken(Token::string);
    }
    result = alloc<StringLiteralNode>(copyArrayRef<StringRef>(spellings));
    break;
  }
  case Token::kw_None:
    result = getNoneExpr(getToken().getLoc());
    consumeToken(Token::kw_None);
    break;
  case Token::l_paren: // primary -> atom -> enclosure -> parenth_form
    if (parsePrefixLParen(result, consumeToken(Token::l_paren).getLoc()))
      return failure();
    break;
  case Token::l_square: // list_display
    if (parsePrefixLSquare(result, consumeToken(Token::l_square).getLoc()))
      return failure();
    break;
  case Token::l_brace: { // dict_display
    DictionaryNode *dict = nullptr;
    if (parsePrefixLBrace(dict, consumeToken(Token::l_brace).getLoc(),
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

  case Token::kw___get_address_as_lvalue:
  case Token::kw___get_lvalue_as_address:
  case Token::kw___get_address_as_owned_value:
  case Token::kw___get_address_as_uninit_lvalue:
    if (failed(parseAddressConvert(result)))
      return failure();
    break;

  default:
    emitTokenError("unexpected token in expression");
    result = nullptr;
    return failure();
  }

  // Check isPrimaryExprToken agrees with the cases above.
  assert(isPrimaryExprToken(tokKind) &&
         "isPrimaryExprToken out of sync with grammar above");

  // Parse postfix productions so long as they aren't the start of the next
  // statement.
  while (!isTokenStartOfNextStatement()) {
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
          !isTokenStartOfNextStatement()) {
        cursor.restore(lexer);
        break;
      }

      result = alloc<UnaryOpNode>(ExprNode::kConsume, loc, result);
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
  SmallVector<ExprNode *> exprs;
  SMLoc firstCommaLoc;

  // Empty parens is a tuple.
  if (consumeIf(Token::r_paren, &rparenLoc)) {
    // Empty tuples are represented as ParenNode(TupleNode()) where the tuple
    // has no subexpressions.
    firstCommaLoc = lparenLoc;
  } else if (parseStarredList(exprs, Token::r_paren, &firstCommaLoc) ||
             parseToken(Token::r_paren,
                        "expected ')' in parenthesized expression", &rparenLoc))
    return failure();

  // If there was a tuple inside the parens, form it.
  if (exprs.size() != 1 || firstCommaLoc.isValid())
    result = alloc<TupleNode>(firstCommaLoc, copyArrayRef<ExprNode *>(exprs));
  else
    result = exprs[0];
  result = alloc<ParenNode>(lparenLoc, result, rparenLoc);
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
  StringRef spelling = getTokenSpelling();
  if (parseToken(Token::identifier, "expected name in attribute reference"))
    return failure();

  result = alloc<AttributeRefNode>(result, dotLoc, spelling);
  return success();
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
  SmallVector<CallArgument> args;
  SMLoc rparenLoc;
  if (!consumeIf(Token::r_paren, &rparenLoc)) {
    // Expressions continue maximally because we are within ()'s.
    llvm::SaveAndRestore<std::optional<size_t>> X(stmtIndent, std::nullopt);

    // Parse an argument.
    auto parseArgument = [&]() -> ParseResult {
      CallArgument &arg = args.emplace_back(CallArgument());

      if (consumeIf(Token::star)) {
        arg.kind = CallArgument::kStar;
        return parseExpression(arg.expr);
      }
      if (consumeIf(Token::star_star)) {
        arg.kind = CallArgument::kStarStar;
        return parseExpression(arg.expr);
      }

      // Check for a keyword argument.  We need look-ahead to determine whether
      // the token after the identifier is an equal sign.
      if (getToken().is(Token::identifier)) {
        auto cursor = lexer.getCursor();
        (void)parseIdentifier(arg.name, "<<already know this is identifier>>");
        if (consumeIf(Token::equal)) {
          arg.kind = CallArgument::kKeyword;
          return parseExpression(arg.expr);
        }
        // Otherwise, we consumed the base expression, just pop it back off.
        cursor.restore(lexer);
      }

      // Parse this as an assignment_expression, allowing := operator.
      return parseExpression(arg.expr, Precedence::kAssignExpr);
    };

    // TODO: Handle comprehension argument.
    if (parseCommaSeparatedList(parseArgument, Token::r_paren) ||
        parseToken(Token::r_paren, "expected ')' in call argument list",
                   &rparenLoc)) {
      return failure();
    }
  }

  // The official Python grammar is super complicated, but the constraint is
  // just that you can't have positional arguments after keyword arguments. This
  // is easier to enforce with a bool than with BNF.
  bool sawKeywordArg = false;
  for (auto &arg : args) {
    // We have a positional / non-keyword argument.  Python syntactically
    // rejects these to reduce ambiguity so we do the same.
    if (arg.kind == CallArgument::kPositional && sawKeywordArg) {
      emitError(arg.getLoc(), "positional argument follows keyword argument");
      return failure();
    }
    if (arg.kind == CallArgument::kKeyword ||
        arg.kind == CallArgument::kStarStar)
      sawKeywordArg = true;
  }

  // Otherwise we're good to go.
  result = alloc<CallNode>(result, lparenLoc, copyArrayRef<CallArgument>(args),
                           rparenLoc);
  return success();
}

/// subscription ::=  primary "[" expression_list ("->" expression_list)?"]"
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

  SmallVector<ExprNode *> indices;

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

  auto parseExprOrSlice = [&]() -> ParseResult {
    ExprNode *firstExpr = nullptr;
    // If this has a leading expr it could be an expr only or could be the first
    // (optional) part of a slice.
    if (getToken().isNot(Token::colon)) {
      if (parseExpression(firstExpr))
        return failure();
      // If we had an expr with no trailing colon, then we are done with the
      // expr case.
      if (getToken().isNot(Token::colon, Token::equal)) {
        indices.push_back(firstExpr);
        return success();
      }
    }

    // Okay we have at least one colon, so we have a slice.
    SMLoc colon1Loc = consumeColonOrEqual(), colon2Loc;
    ExprNode *secondExpr = nullptr, *thirdExpr = nullptr;

    // Parse the second expr if present.
    if (getToken().isNot(Token::colon, Token::equal, Token::comma,
                         Token::r_square)) {
      if (parseExpression(secondExpr))
        return failure();
    }

    // Parse a second colon if present and stride expression.
    if (getToken().isAny(Token::colon, Token::equal)) {
      colon2Loc = consumeColonOrEqual();
      if (getToken().isNot(Token::comma, Token::r_square)) {
        if (parseExpression(thirdExpr))
          return failure();
      }
    }
    indices.push_back(alloc<SliceNode>(firstExpr, colon1Loc, secondExpr,
                                       colon2Loc, thirdExpr));
    return success();
  };

  SMLoc rsquareLoc;
  if (parseCommaSeparatedList(parseExprOrSlice,
                              {Token::r_square, Token::minus_greater}) ||
      getLocation(rsquareLoc))
    return failure();

  // If we have no arrow, handle this as a normal subscript.
  if (!consumeIf(Token::minus_greater)) {
    if (parseToken(Token::r_square, "expected ']' in call argument list"))
      return failure();
    result = alloc<SubscriptNode>(
        result, lsquareLoc, copyArrayRef<ExprNode *>(indices), rsquareLoc);
    return success();
  }

  // Otherwise, parse the arrow production.
  SMLoc arrowLoc = rsquareLoc;
  SmallVector<ExprNode *> arrowExprs;
  std::swap(indices, arrowExprs);
  if (parseCommaSeparatedList(parseExprOrSlice,
                              {Token::r_square, Token::minus_greater}) ||
      getLocation(rsquareLoc) ||
      parseToken(Token::r_square, "expected ']' in call argument list"))
    return failure();

  std::swap(indices, arrowExprs);
  result = alloc<SubscriptArrowNode>(
      result, lsquareLoc, copyArrayRef<ExprNode *>(indices), arrowLoc,
      copyArrayRef<ExprNode *>(arrowExprs), rsquareLoc);
  return success();
}

/// Parse the input and result parameters, if they are present.
static ParseResult
parseOptionalFunctionParameters(ParserBase &p,
                                SmallVectorImpl<ParsedArgument> &inputParams,
                                SmallVectorImpl<ParsedArgument> &resultParams) {
  if (!p.consumeIf(Token::l_square))
    return success();
  // Handle '[]'.
  if (p.consumeIf(Token::r_square))
    return success();

  if (p.consumeIf(Token::l_paren)) {
    if (p.parseToken(Token::r_paren,
                     "expected ')' in empty parameter list; try dropping the "
                     "'(' if you have parameters"))
      return failure();
  } else {
    // Parse an actual parameter list.
    if (ParsedArgument::parseAndResolvePresentArgumentList(
            p, inputParams, /*isParameterList=*/true))
      return failure();
  }

  // Parse result parameters if present.
  if (p.consumeIf(Token::minus_greater)) {
    if (ParsedArgument::parseAndResolvePresentArgumentList(
            p, resultParams, /*isParameterList=*/true))
      return failure();
  }
  return p.parseToken(Token::r_square, "expected ']' for parameter list");
}

ParseResult ExprParser::parseFunctionType(ExprNode *&result) {
  SMLoc baseLoc = getToken().getLoc();
  SmallVector<ParsedArgument> inputParams, resultParams, arguments;
  ExprNode *resultTypeExpr = nullptr;
  FnEffects effects = FnEffects::None;
  bool isDef = false;

  // Parse the function effects from the leading keyword.
  if (consumeIf(Token::kw_async))
    effects = effects | FnEffects::Async;
  if (consumeToken().is(Token::kw_def)) {
    effects = effects | FnEffects::Throws;
    isDef = true;
  }

  // Parameter signature.
  if (parseOptionalFunctionParameters(*this, inputParams, resultParams))
    return failure();

  // Parse the argument list next if present.
  if (parseToken(Token::l_paren, "expected '(' for argument list"))
    return failure();
  if (!consumeIf(Token::r_paren)) {
    if (ParsedArgument::parseAndResolvePresentArgumentList(
            *this, arguments, /*isParameterList=*/false, /*omitNames=*/true) ||
        parseToken(Token::r_paren, "expected ')' in argument list"))
      return failure();
  }

  // Parse other function effects.
  while (getToken().is(Token::identifier)) {
    SMLoc loc = getToken().getLoc();
    if (getToken().getSpelling() == "raises") {
      if (bitEnumContainsAny(effects, FnEffects::Throws))
        emitError(loc, "function effect 'raises' was already specified");
      effects = effects | FnEffects::Throws;
    } else if (getToken().getSpelling() == "capturing") {
      if (bitEnumContainsAny(effects, FnEffects::Capturing))
        emitError(loc, "function effect 'capturing' was already specified");
      effects = effects | FnEffects::Capturing;
    } else {
      emitError(loc, "unknown function effect '")
          << getToken().getSpelling() << "', expected 'raises' or 'capturing'";
    }
    consumeToken();
  }

  // Parse the result type.
  SMLoc endLoc = getToken().getEndLoc();
  SMLoc resultLoc = getToken().getLoc();
  if (!isDef || getToken().is(Token::minus_greater)) {
    if (parseToken(Token::minus_greater, "expected '->' in function type") ||
        ParserBase::parseExpression(resultTypeExpr, stmtIndent))
      return failure();
  }

  result = alloc<FunctionTypeNode>(
      baseLoc, copyArrayRef<ParsedArgument>(inputParams),
      copyArrayRef<ParsedArgument>(resultParams),
      copyArrayRef<ParsedArgument>(arguments), resultTypeExpr, effects, endLoc,
      isDef, resultLoc);
  return success();
}

ParseResult ExprParser::parseAddressConvert(ExprNode *&result) {
  ExprNode::Kind nodeKind;
  switch (getToken().getKind()) {
  default:
    llvm_unreachable("bad token");
  case Token::kw___get_address_as_lvalue:
    nodeKind = ExprNode::kGetAddressAsLValue;
    break;
  case Token::kw___get_address_as_uninit_lvalue:
    nodeKind = ExprNode::kGetAddressAsUninitLValue;
    break;
  case Token::kw___get_lvalue_as_address:
    nodeKind = ExprNode::kGetLValueAsAddress;
    break;
  case Token::kw___get_address_as_owned_value:
    nodeKind = ExprNode::kGetAddressAsOwned;
    break;
  }
  SMLoc baseLoc = consumeToken().getLoc();

  ExprNode *subExpr = nullptr;
  SMLoc rpLoc;
  if (parseToken(Token::l_paren, "expected '('") || parseExpression(subExpr) ||
      parseToken(Token::r_paren, "expected ')'", &rpLoc))
    return failure();

  result = alloc<AddressConvertNode>(nodeKind, baseLoc, subExpr, rpLoc);
  return success();
}

//===----------------------------------------------------------------------===//
// ExprParser implementation
//===----------------------------------------------------------------------===//

/// Parse an expression_list production, returning a single expression or a
/// tuple expression if there are commas.  If 'terminators' is specified,
/// (e.g. in a subscript expression) then parsing ignores indentation and
/// looks for the specified terminator.
ParseResult ParserBase::parseExpressionList(SMLoc emptyLoc, ExprNode *&result,
                                            std::optional<size_t> stmtIndent,
                                            ArrayRef<Token::Kind> terminators) {
  // If this expression_list has no terminator (e.g. not in a subscript list
  // terminated with ], and if it is empty, then produce a None value.  This
  // makes return statements happy.
  if (terminators.empty() && getToken().getIndentation().has_value() &&
      stmtIndent.has_value() && *getToken().getIndentation() <= *stmtIndent) {
    result = getNoneExpr(emptyLoc);
    return success();
  }

  ExprParser parser(getLexer(), stmtIndent);
  SmallVector<ExprNode *> exprs;
  auto parseItem = [&]() -> ParseResult {
    return parser.parseExpression(exprs.emplace_back(nullptr),
                                  Precedence::kExpression);
  };

  SMLoc firstCommaLoc;
  if (parser.parseCommaSeparatedList(parseItem, terminators, &firstCommaLoc))
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

/// Expression parsing.  Each of these take a `stmtIndent` specifier that
/// indicates the indentation level of the start of the statement that
/// contains this expression if the expression can exist at the end of the
/// line.  This allows the expression parser to know when to keep parsing the
/// expression on the next line - when it is more indented than the start of
/// the current statement.  This can be passed in as None when there is a
/// trailing punctuator that naturally terminates the expression.
ParseResult ParserBase::parseExpression(ExprNode *&result,
                                        std::optional<size_t> stmtIndent) {
  return ExprParser(getLexer(), stmtIndent)
      .parseExpression(result, Precedence::kExpression);
}

ParseResult ParserBase::parseStarredItem(ExprNode *&result) {
  return ExprParser(getLexer(), std::nullopt).parseStarredItem(result);
}

/// assignment_stmt ::=
///                 (target_list "=")+ (starred_expression | yield_expression)
/// target_list     ::=  target ("," target)* [","]
/// target ::= identifier
///          | "(" [target_list] ")" | "[" [target_list] "]"
///          | attributeref | subscription | slicing | "*" target
///
/// expression_stmt ::= starred_expression
/// augmented_assignment_stmt ::=
///                         augtarget augop (expression_list |
///                         yield_expression)
/// augtarget ::=  identifier | attributeref | subscription | slicing
/// augop ::=  "+=" | "-=" | "*=" | "@=" | "/=" | "//=" | "%=" | "**="
///            | ">>=" | "<<=" | "&=" | "^=" | "|="
///
/// Parse an expression, allowing `=`, and `+=`.
ParseResult
ParserBase::parseExpressionOrAssignmentStmt(ExprNode *&result,
                                            std::optional<size_t> stmtIndent) {
  return ExprParser(getLexer(), stmtIndent)
      .parseExpression(result, Precedence::kAssignStmt);
}

/// Return an expression node for None at the specified location.
ExprNode *ParserBase::getNoneExpr(SMLoc loc) {
  return ExprParser(getLexer(), 0).getNoneExpr(loc);
}
