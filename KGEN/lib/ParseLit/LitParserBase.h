//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the base class for Lit file parsers that is common between
// expression and statement parsing.
//
//===----------------------------------------------------------------------===//

#ifndef LIT_PARSER_BASE_H
#define LIT_PARSER_BASE_H

#include "LitLexer.h"
#include "mlir/IR/Diagnostics.h"

namespace M::KGEN::LIT {
class ExprNode;
class ASTDecl;

//===----------------------------------------------------------------------===//
// LitParserBase
//===----------------------------------------------------------------------===//

/// This class implements logic that is common to many parts of the parser, but
/// which is independent of the concrete grammar.
class LitParserBase : public LitSharedStateUser {
public:
  LitParserBase(LitLexer &lexer)
      : LitSharedStateUser(lexer.shared), lexer(lexer) {}

  LitLexer &getLexer() { return lexer; }

  /// Return the current token the parser is inspecting.
  const LitToken &getToken() const { return lexer.getToken(); }
  StringRef getTokenSpelling() const { return getToken().getSpelling(); }

  //===--------------------------------------------------------------------===//
  // Error Handling
  //===--------------------------------------------------------------------===//

  using LitSharedStateUser::emitError;

  /// Emit an error at a specific lexer location.
  LitDiagnostic emitError(llvm::SMLoc loc, const Twine &message = {});

  /// Emit an error at the current token.
  LitDiagnostic emitTokenError(const Twine &message = {}) {
    return emitError(getToken().getLoc(), message);
  }

  //===--------------------------------------------------------------------===//
  // Location Handling
  //===--------------------------------------------------------------------===//

  /// Capture the location of the current token in a convenient way that can be
  /// used in parsing pipelines.
  ParseResult getLocation(SMLoc &result) {
    result = getToken().getLoc();
    return success();
  }

  /// This returns the current lexor cursor and succeeds, so it can be used in a
  /// parser pipeline.
  ParseResult getCursor(LitLexerCursor &cursor) const {
    cursor = lexer.getCursor();
    return success();
  }

  /// Return the location of the current token, or a location at the end of the
  /// previous line if it is on a new line.  This is used when there was a
  /// problem with the previous token to make sure we report the error on that
  /// line.
  SMLoc getTokenLocOrEndOfPreviousLineIfOnNewLine() const {
    SMLoc loc = getToken().getLoc();
    if (getToken().getIndentation().has_value())
      return lexer.findEndOfPreviousLine(loc);
    return loc;
  }

  //===--------------------------------------------------------------------===//
  // Token Parsing
  //===--------------------------------------------------------------------===//

  /// If the current token has the specified kind, consume it and return true.
  /// If not, return false.  If 'tokLoc' is non-null, it is filled in with the
  /// location of the consumed token (on success).
  bool consumeIf(LitToken::Kind kind, llvm::SMLoc *tokLoc = nullptr) {
    if (getToken().isNot(kind))
      return false;
    if (tokLoc)
      *tokLoc = getToken().getLoc();
    consumeToken(kind);
    return true;
  }

  /// Advance the current lexer onto the next token.
  ///
  /// This returns the consumed token.
  LitToken consumeToken() {
    LitToken consumedToken = getToken();
    assert(consumedToken.isNot(LitToken::eof) && "shouldn't advance past EOF");
    lexer.lexToken();
    return consumedToken;
  }

  /// Advance the current lexer onto the next token, asserting what the expected
  /// current token is.  This is preferred to the above method because it leads
  /// to more self-documenting code with better checking.
  ///
  /// This returns the consumed token.
  LitToken consumeToken(LitToken::Kind kind) {
    LitToken consumedToken = getToken();
    assert(consumedToken.is(kind) && "consumed an unexpected token");
    consumeToken();
    return consumedToken;
  }

  /// Consume the specified token if present and return success.  On failure,
  /// output a diagnostic and return failure. If `loc` is set, it is populated
  /// with the source location of the token.
  ParseResult parseToken(LitToken::Kind expectedToken, const Twine &message,
                         SMLoc *loc = nullptr);

  /// Consume an identifier token, binding its name into the specified result
  /// string attribute.
  ParseResult parseIdentifier(StringAttr &result, const Twine &message);

  /// Parse a list of elements, terminated with an arbitrary token.  This does
  /// not consume the stop token.
  ///
  /// list ::= (element)* STOPTOKEN
  ///
  ParseResult parseListUntil(LitToken::Kind stopToken,
                             const std::function<ParseResult()> &parseElement);

  /// Parse a list of elements continued with a separator token, like a comma.
  /// The list ends either with a terminator, which is not consumed, or a new
  /// line. hadTrailingSep is set to true if a trailing separator was found.
  ///
  /// separated_list ::= (element (SEPARATOR element)* [SEPARATOR] TERMINATOR
  ///
  ParseResult
  parseSeparatedList(LitToken::Kind separator,
                     const std::function<ParseResult()> &parseElement,
                     ArrayRef<LitToken::Kind> terminators,
                     bool *hadTrailingSep);
  ParseResult
  parseCommaSeparatedList(const std::function<ParseResult()> &parseElement,
                          ArrayRef<LitToken::Kind> terminators,
                          bool *hadTrailingSep = nullptr) {
    return parseSeparatedList(LitToken::comma, parseElement, terminators,
                              hadTrailingSep);
  }

  /// Skip tokens until we get to a token at start of line that has indentation
  /// that is equal or less than the specified indentation.  This is used for
  /// multiphase parsing.
  ///
  /// When stopOnSemicolon is true this will stop at the first semicolon seen.
  /// This should only be used for statements that can share a line with other
  /// statements with ; separation.
  void skipUntilIndentation(size_t minIndent, bool stopOnSemicolon = false);

  /// Consume tokens until we get to the end of the current line, used for error
  /// recovery.
  /// TODO: we should know the indentation of the current statement so we can
  /// eat trailing components that lack a \.
  void eatToEndOfLine() {
    while (!getToken().getIndentation().has_value() &&
           getToken().isNot(LitToken::eof))
      consumeToken();
  }

  //===--------------------------------------------------------------------===//
  // Integration with parsers for subsets of the grammar.
  //===--------------------------------------------------------------------===//

  /// Parse the follow-on doc string for the given decl if it is present.
  void parseDocString(ASTDecl &decl);

  /// Parse and return a set of decorators for the specified declaration or
  /// statement at the specified indentation level.
  SmallVector<ExprNode *> parseDecorators(ASTDecl &decl);
  SmallVector<ExprNode *> parseDecorators(ssize_t indention);

  /// Expression parsing.  Each of these take a `stmtIndent` specifier that
  /// indicates the indentation level of the start of the statement that
  /// contains this expression if the expression can exist at the end of the
  /// line.  This allows the expression parser to know when to keep parsing the
  /// expression on the next line - when it is more indented than the start of
  /// the current statement.  This can be passed in as None when there is a
  /// trailing punctuator that naturally terminates the expression.
  ParseResult parseExpressionList(SmallVectorImpl<ExprNode *> &results,
                                  std::optional<size_t> stmtIndent,
                                  bool *hadTrailingComma);
  ParseResult parseExpression(ExprNode *&expr,
                              std::optional<size_t> stmtIndent);
  /// Parse an expression, allowing `=`, and `+=`.
  ParseResult parseExpressionOrAssignmentStmt(ExprNode *&expr,
                                              std::optional<size_t> stmtIndent);

  /// Return an expression node for None at the specified location.
  ExprNode *getNoneExpr(SMLoc loc);

  /// Parse a 'suite' production into the declaration specified by `decl`.
  static ParseResult parseSuite(ASTDecl &decl, LitLexer &lexer);

public:
  LitLexer &lexer;

  LitParserBase(const LitParserBase &) = delete;
  void operator=(const LitParserBase &) = delete;
};

} // namespace M::KGEN::LIT

#endif // LIT_PARSER_BASE_H
