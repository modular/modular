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
#include "LitLexer.h"
#include "LitParserBase.h"
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

  kLowestExpr, // Lowest expression precedence (most loosely bound).
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
  kPrimary,    // prefix: foo, "123", 123, 1.23, True, False, foo(1),
               //         foo.bar, foo[bar]
  kHighest = kPrimary
};

namespace M::KGEN::LIT {
/// This class implements the ExprParser interface, implemented with the pImpl
/// idiom.
class ExprParser : public LitParserBase {
public:
  ExprParser(LitLexer &lexer, Optional<size_t> stmtIndent)
      : LitParserBase(lexer), stmtIndent(stmtIndent) {}

  ~ExprParser() {}

  // Expressions.
  ParseResult parseExpressionList(SmallVectorImpl<ExprNode *> &results,
                                  LitToken::Kind terminator,
                                  bool *hadTrailingSep = nullptr);
  ParseResult parseExpression(ExprNode *&result,
                              Precedence minPrec = Precedence::kLowestExpr);

  ExprNode *getNoneExpr(SMLoc loc) { return alloc<NoneLiteralNode>(loc); };

private:
  template <typename T, typename... Args>
  T *alloc(Args &&...args) {
    return getSharedState().allocPersistent<T>(std::forward<Args>(args)...);
  }

  template <typename T>
  ArrayRef<T> copyArrayRef(ArrayRef<T> elements) {
    return getSharedState().getPersistentCopy(elements);
  }

  /// Return true if the current token is the start of another statement, false
  /// if it is part of this one.
  bool isTokenStartOfNextStatement();

  ParseResult parsePrefixExpr(ExprNode *&result);
  ParseResult parseAttributeRefSuffix(ExprNode *&result, SMLoc dotLoc);
  ParseResult parseCallSuffix(ExprNode *&result, SMLoc lparenLoc);
  ParseResult parseSubscriptSuffix(ExprNode *&result, SMLoc lsquareLoc);

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

//===----------------------------------------------------------------------===//
// Parsing rules
//===----------------------------------------------------------------------===//

/// expression_list ::= expression ("," expression)* [","]
ParseResult
ExprParser::parseExpressionList(SmallVectorImpl<ExprNode *> &results,
                                LitToken::Kind terminator,
                                bool *hadTrailingSep) {
  return parseCommaSeparatedList(
      [&]() -> ParseResult {
        return parseExpression(results.emplace_back(nullptr));
      },
      terminator, hadTrailingSep);
}

namespace {
/// This struct bundles up information related to infix binary operations.
struct InfixInfo {
  Precedence precedence;
  ExprNode::Kind nodeKind;
  bool isLeftAssociative;

  /// Classify a token for an infix operator.
  static InfixInfo get(LitToken::Kind tokKind) {
    switch (tokKind) {
    default:
      return {Precedence::kInvalid, ExprNode::kLastBinOp, false};
    case LitToken::equal:
      return {Precedence::kLowestExpr, ExprNode::kAssign, false};
    case LitToken::plus_equal:
      return {Precedence::kLowestExpr, ExprNode::kIAdd, false};
    case LitToken::minus_equal:
      return {Precedence::kLowestExpr, ExprNode::kISub, false};
    case LitToken::star_equal:
      return {Precedence::kLowestExpr, ExprNode::kIMul, false};
    case LitToken::at_equal:
      return {Precedence::kLowestExpr, ExprNode::kIMatMul, false};
    case LitToken::slash_equal:
      return {Precedence::kLowestExpr, ExprNode::kITrueDiv, false};
    case LitToken::percent_equal:
      return {Precedence::kLowestExpr, ExprNode::kIMod, false};
    case LitToken::amp_equal:
      return {Precedence::kLowestExpr, ExprNode::kIAnd, false};
    case LitToken::pipe_equal:
      return {Precedence::kLowestExpr, ExprNode::kIOr, false};
    case LitToken::circumflex_equal:
      return {Precedence::kLowestExpr, ExprNode::kIXor, false};
    case LitToken::less_less_equal:
      return {Precedence::kLowestExpr, ExprNode::kILShift, false};
    case LitToken::right_right_equal:
      return {Precedence::kLowestExpr, ExprNode::kIRShift, false};
    case LitToken::star_star_equal:
      return {Precedence::kLowestExpr, ExprNode::kIPow, false};
    case LitToken::slash_slash_equal:
      return {Precedence::kLowestExpr, ExprNode::kIFloorDiv, false};
    case LitToken::plus:
      return {Precedence::kSum, ExprNode::kAdd, false};
    case LitToken::minus:
      return {Precedence::kSum, ExprNode::kSub, false};
    case LitToken::star:
      return {Precedence::kTerm, ExprNode::kMul, false};
    case LitToken::at:
      return {Precedence::kTerm, ExprNode::kMatMul, false};
    case LitToken::slash:
      return {Precedence::kTerm, ExprNode::kTrueDiv, false};
    case LitToken::slash_slash:
      return {Precedence::kTerm, ExprNode::kFloorDiv, false};
    case LitToken::percent:
      return {Precedence::kTerm, ExprNode::kMod, false};
    case LitToken::kw_or:
      return {Precedence::kBoolOr, ExprNode::kBoolOr, false};
    case LitToken::kw_and:
      return {Precedence::kBoolAnd, ExprNode::kBoolAnd, false};
    case LitToken::kw_not:
      return {Precedence::kBoolNot, ExprNode::kBoolNot, false};
    case LitToken::kw_in:
      return {Precedence::kComparison, ExprNode::kCmpIn, false};
    case LitToken::kw_is:
      return {Precedence::kComparison, ExprNode::kCmpIs, false};
    case LitToken::less:
      return {Precedence::kComparison, ExprNode::kCmpLT, false};
    case LitToken::less_equal:
      return {Precedence::kComparison, ExprNode::kCmpLE, false};
    case LitToken::greater:
      return {Precedence::kComparison, ExprNode::kCmpGT, false};
    case LitToken::greater_equal:
      return {Precedence::kComparison, ExprNode::kCmpGE, false};
    case LitToken::exclaim_equal:
      return {Precedence::kComparison, ExprNode::kCmpNE, false};
    case LitToken::equal_equal:
      return {Precedence::kComparison, ExprNode::kCmpEQ, false};
    case LitToken::pipe:
      return {Precedence::kOr, ExprNode::kOr, false};
    case LitToken::circumflex:
      return {Precedence::kXor, ExprNode::kXor, false};
    case LitToken::amp:
      return {Precedence::kAnd, ExprNode::kAnd, false};
    case LitToken::less_less:
      return {Precedence::kShift, ExprNode::kLShift, false};
    case LitToken::right_right:
      return {Precedence::kShift, ExprNode::kRShift, false};
    case LitToken::kw_if:
      return {Precedence::kIfElse, ExprNode::kIfElse, false};
    case LitToken::star_star:
      return {Precedence::kPower, ExprNode::kPow, true};
    }
  }
};
} // namespace

/// Parse an expression using top-down operator precedence parsing.
ParseResult ExprParser::parseExpression(ExprNode *&expr, Precedence minPrec) {
  // Parse any prefix expression like -1.
  if (parsePrefixExpr(expr))
    return failure();

  // Consume infix tokens until we meet a token whose tokPrecedence is equal or
  // lower than minPrec. This means that it collects all tokens that bind
  // together before returning to the operator that called it.
  InfixInfo infixInfo = InfixInfo::get(getToken().getKind());
  while (!isTokenStartOfNextStatement() &&
         unsigned(minPrec) < unsigned(infixInfo.precedence)) {
    LitToken::Kind tokKind = getToken().getKind();
    auto binOpLoc = consumeToken().getLoc();

    if (tokKind == LitToken::Kind::kw_if) {
      // Conditional if - else expression.
      // trueExpr 'if' condition 'else' falseExpr.
      ExprNode *cond;
      if (parseExpression(cond, infixInfo.precedence))
        return failure();

      ExprNode *falseExpr;
      auto elseLoc = getToken().getLoc();
      if (parseToken(LitToken::Kind::kw_else,
                     "expecting an 'else' followed by an expression") ||
          parseExpression(falseExpr, infixInfo.precedence))
        return failure();
      expr = alloc<IfElseOpNode>(expr, binOpLoc, cond, elseLoc, falseExpr);
      infixInfo = InfixInfo::get(getToken().getKind());
      continue;
    }

    // rhs 'is' 'not' lhs -> a is not True.
    if (tokKind == LitToken::Kind::kw_is && consumeIf(LitToken::Kind::kw_not))
      infixInfo.nodeKind = ExprNode::Kind::kCmpIsNot;
    // rhs 'not' 'in' lhs -> a not in {1, 2}.
    else if (tokKind == LitToken::Kind::kw_not &&
             consumeIf(LitToken::Kind::kw_in)) {
      infixInfo.nodeKind = ExprNode::Kind::kCmpNotIn;
      infixInfo.precedence = Precedence::kComparison;
    }

    // Handle left associative operations.
    if (infixInfo.isLeftAssociative)
      infixInfo.precedence = Precedence(unsigned(infixInfo.precedence) - 1);

    ExprNode *rhs;
    if (parseExpression(rhs, infixInfo.precedence))
      return failure();
    expr = alloc<BinOpNode>(infixInfo.nodeKind, expr, binOpLoc, rhs);
    infixInfo = InfixInfo::get(getToken().getKind());
  }
  return success();
}

static ExprNode::Kind getUnaryOpKind(LitToken::Kind tokKind) {
  switch (tokKind) {
  default:
    llvm_unreachable("invalid unary token");
  case LitToken::kw_not:
    return ExprNode::kBoolNot;
  case LitToken::plus:
    return ExprNode::kPos;
  case LitToken::minus:
    return ExprNode::kNeg;
  case LitToken::tilde:
    return ExprNode::kInvert;
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
/// parenth_form ::= "(" [starred_expression] ")"
/// list_display ::=  "[" [starred_list | comprehension [TODO]] "]"
/// starred_list       ::=  starred_item ("," starred_item)* [","]
/// starred_item       ::=  assignment_expression[TODO] | "*" or_expr [TODO]
/// literal ::=
///     stringliteral | bytesliteral | integer | floatnumber | imagnumber
///
/// u_expr ::=  power | "-" u_expr | "+" u_expr | "~" u_expr
///
ParseResult ExprParser::parsePrefixExpr(ExprNode *&result) {
  LitToken::Kind tokKind = getToken().getKind();
  switch (tokKind) {
  case LitToken::plus:
  case LitToken::minus:
  case LitToken::tilde:
  case LitToken::kw_not: { // u_expr
    auto unaryLoc = consumeToken().getLoc();
    ExprNode *expr;
    Precedence precedence = Precedence::kFactor;
    if (tokKind == LitToken::kw_not) // not expr.
      precedence = Precedence::kBoolNot;
    if (parseExpression(expr, precedence))
      return failure();
    result = alloc<UnaryOpNode>(getUnaryOpKind(tokKind), unaryLoc, expr);
    break;
  }
  case LitToken::identifier: // primary -> atom -> identifier
    result = alloc<DeclRefNode>(getToken().getSpelling());
    consumeToken(LitToken::identifier);
    break;
  case LitToken::integer: // primary -> literal -> integer
    result = alloc<IntLiteralNode>(getToken().getSpelling());
    consumeToken(LitToken::integer);
    break;
  case LitToken::kw_False:
    result = alloc<BoolLiteralNode>(getToken().getLoc(), false);
    consumeToken(LitToken::kw_False);
    break;
  case LitToken::kw_True:
    result = alloc<BoolLiteralNode>(getToken().getLoc(), true);
    consumeToken(LitToken::kw_True);
    break;
  case LitToken::float_num: // primary -> literal -> floatnumber
    result = alloc<FloatLiteralNode>(getToken().getSpelling());
    consumeToken(LitToken::float_num);
    break;
  case LitToken::string: // primary -> literal -> stringliteral
    result = alloc<StringLiteralNode>(getToken().getSpelling());
    consumeToken(LitToken::string);
    break;
  case LitToken::kw_None:
    result = getNoneExpr(getToken().getLoc());
    consumeToken(LitToken::kw_None);
    break;
  case LitToken::l_paren: { // primary -> atom -> enclosure -> parenth_form
    auto lpLoc = consumeToken(LitToken::l_paren).getLoc();
    if (parseExpression(result))
      return failure();
    auto rpLoc = getToken().getLoc();
    if (parseToken(LitToken::r_paren,
                   "expected ')' in parenthesized expression"))
      return failure();
    result = alloc<ParenExprNode>(lpLoc, result, rpLoc);
    break;
  }
  case LitToken::l_square: { // list_display
    SMLoc lsLoc = consumeToken(LitToken::l_square).getLoc();
    SMLoc rsLoc;
    SmallVector<ExprNode *> exprs;
    if (consumeIf(LitToken::r_square, &rsLoc)) {
      // Empty list: []
      result = alloc<ListExprNode>(lsLoc, exprs, rsLoc);
      break;
    }
    if (parseExpressionList(exprs, LitToken::r_square))
      return failure();
    rsLoc = getToken().getLoc();
    if (parseToken(LitToken::r_square, "expected ']' in list expression"))
      return failure();
    result = alloc<ListExprNode>(lsLoc, copyArrayRef<ExprNode *>(exprs), rsLoc);
    break;
  }

  default:
    emitError("unexpected token in expression");
    result = nullptr;
    return failure();
  }

  // Parse postfix productions so long as they aren't the start of the next
  // statement.
  while (!isTokenStartOfNextStatement()) {
    auto loc = getToken().getLoc();

    // Handle "attributeref": x.y
    if (consumeIf(LitToken::dot)) {
      if (parseAttributeRefSuffix(result, loc))
        return failure();
      continue;
    }

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

/// attributeref ::=  primary "." identifier
ParseResult ExprParser::parseAttributeRefSuffix(ExprNode *&result,
                                                SMLoc dotLoc) {
  StringRef spelling = getTokenSpelling();
  if (parseToken(LitToken::identifier, "expected name in attribute reference"))
    return failure();

  result = alloc<AttributeRefNode>(result, dotLoc, spelling);
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
  SMLoc rparenLoc;
  // TODO: Handle comprehension arguments, stars, etc.
  if (!consumeIf(LitToken::r_paren, &rparenLoc)) {
    // Expressions continue maximally because we are within ()'s.
    llvm::SaveAndRestore<Optional<size_t>> X(stmtIndent, std::nullopt);
    if (parseExpressionList(args, LitToken::r_paren) ||
        getLocation(rparenLoc) ||
        parseToken(LitToken::r_paren, "expected ')' in call argument list")) {
      return failure();
    }
  }
  result = alloc<CallNode>(result, lparenLoc, copyArrayRef<ExprNode *>(args),
                           rparenLoc);
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
  llvm::SaveAndRestore<Optional<size_t>> X(stmtIndent, std::nullopt);

  SmallVector<ExprNode *> indices;

  auto parseExprOrSlice = [&]() -> ParseResult {
    ExprNode *firstExpr = nullptr;
    // If this has a leading expr it could be an expr only or could be the first
    // (optional) part of a slice.
    if (getToken().isNot(LitToken::colon)) {
      if (parseExpression(firstExpr))
        return failure();
      // If we had an expr with no trailing colon, then we are done with the
      // expr case.
      if (getToken().isNot(LitToken::colon)) {
        indices.push_back(firstExpr);
        return success();
      }
    }

    // Okay we have at least one colon, so we have a slice.
    SMLoc colon1Loc = consumeToken(LitToken::colon).getLoc(), colon2Loc;
    ExprNode *secondExpr = nullptr, *thirdExpr = nullptr;

    // Parse the second expr if present.
    if (getToken().isNot(LitToken::colon, LitToken::comma,
                         LitToken::r_square)) {
      if (parseExpression(secondExpr))
        return failure();
    }

    // Parse a second colon if present and stride expression.
    if (getToken().is(LitToken::colon)) {
      colon2Loc = consumeToken(LitToken::colon).getLoc();
      if (getToken().isNot(LitToken::comma, LitToken::r_square)) {
        if (parseExpression(thirdExpr))
          return failure();
      }
    }
    indices.push_back(alloc<SliceNode>(firstExpr, colon1Loc, secondExpr,
                                       colon2Loc, thirdExpr));
    return success();
  };

  SMLoc rsquareLoc;
  if (parseCommaSeparatedList(parseExprOrSlice, LitToken::r_square) ||
      getLocation(rsquareLoc) ||
      parseToken(LitToken::r_square, "expected ']' in call argument list"))
    return failure();

  result = alloc<SubscriptNode>(result, lsquareLoc,
                                copyArrayRef<ExprNode *>(indices), rsquareLoc);
  return success();
}

//===----------------------------------------------------------------------===//
// ExprParser implementation
//===----------------------------------------------------------------------===//

ParseResult
LitParserBase::parseExpressionList(SmallVectorImpl<ExprNode *> &results,
                                   Optional<size_t> stmtIndent,
                                   bool *hadTrailingSep) {
  return ExprParser(getLexer(), stmtIndent)
      .parseExpressionList(results, LitToken::Kind::eof, hadTrailingSep);
}

/// Expression parsing.  Each of these take a `stmtIndent` specifier that
/// indicates the indentation level of the start of the statement that
/// contains this expression if the expression can exist at the end of the
/// line.  This allows the expression parser to know when to keep parsing the
/// expression on the next line - when it is more indented than the start of
/// the current statement.  This can be passed in as None when there is a
/// trailing punctuator that naturally terminates the expression.
ParseResult LitParserBase::parseExpression(ExprNode *&result,
                                           Optional<size_t> stmtIndent) {
  return ExprParser(getLexer(), stmtIndent)
      .parseExpression(result, Precedence::kLowestExpr);
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
LitParserBase::parseExpressionOrAssignmentStmt(ExprNode *&result,
                                               Optional<size_t> stmtIndent) {
  return ExprParser(getLexer(), stmtIndent)
      .parseExpression(result, Precedence::kAssignStmt);
}

/// Return an expression node for None at the specified location.
ExprNode *LitParserBase::getNoneExpr(SMLoc loc) {
  return ExprParser(getLexer(), 0).getNoneExpr(loc);
}
