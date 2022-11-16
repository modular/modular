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

#include "LitExprs.h"
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
class LitParserBase {
public:
  LitParserBase(LitLexer &lexer) : lexer(lexer) {}

  LitSharedState &getSharedState() const { return lexer.sharedState; }
  MLIRContext *getContext() const { return getSharedState().context; }
  DeclResolver &getDeclResolver() const {
    return *getSharedState().declResolver;
  }

  LitLexer &getLexer() { return lexer; }

  /// Return the current token the parser is inspecting.
  const LitToken &getToken() const { return lexer.getToken(); }
  StringRef getTokenSpelling() const { return getToken().getSpelling(); }

  //===--------------------------------------------------------------------===//
  // Error Handling
  //===--------------------------------------------------------------------===//

  /// Emit an error and notice that so we don't verify the IR at the end of
  /// compilation.
  InFlightDiagnostic emitError(Location loc, const Twine &message = {}) {
    return getSharedState().emitError(loc, message);
  }

  /// Emit an error at the current token.
  InFlightDiagnostic emitError(const Twine &message = {}) {
    return emitError(getToken().getLoc(), message);
  }
  /// Emit an error at a specific lexer location.
  InFlightDiagnostic emitError(llvm::SMLoc loc, const Twine &message = {});

  //===--------------------------------------------------------------------===//
  // Location Handling
  //===--------------------------------------------------------------------===//

  /// Encode the specified source location information into an attribute for
  /// attachment to the IR.
  Location translateLocation(llvm::SMLoc loc) {
    return getSharedState().translateLocation(loc);
  }

  /// Return the location for the current token.
  Location getTokenLocation() { return translateLocation(getToken().getLoc()); }

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

  /// Capture the location of the current token in a convenient way that can be
  /// used in parsing pipelines.
  ParseResult getLocation(SMLoc &result) {
    result = getToken().getLoc();
    return success();
  }

  /// Consume the specified token if present and return success.  On failure,
  /// output a diagnostic and return failure.
  ParseResult parseToken(LitToken::Kind expectedToken, const Twine &message);

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
                     LitToken::Kind terminator, bool *hadTrailingSep);
  ParseResult
  parseCommaSeparatedList(const std::function<ParseResult()> &parseElement,
                          LitToken::Kind terminator,
                          bool *hadTrailingSep = nullptr) {
    return parseSeparatedList(LitToken::comma, parseElement, terminator,
                              hadTrailingSep);
  }

  /// Skip tokens until we get to a token at start of line that has indentation
  /// that is equal or less than the specified indentation.  This is used for
  /// multiphase parsing.
  ///
  /// When stopOnSemicolon is true this will stop at the first semicolon seen.
  /// This should only be used for statements that can share a line with other
  /// statements with ; separation.
  void skipUntilIndentation(size_t minIndent, bool stopOnSemicolon = false) {
    // TODO: This needs to do python style brace matching.
    while (getToken().isNot(LitToken::eof)) {
      if (auto indent = getToken().getIndentation())
        if (indent.value() <= minIndent)
          break;
      if (stopOnSemicolon && getToken().is(LitToken::semi))
        break;
      consumeToken();
    }
  }

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

  /// Expression parsing.  Each of these take a `stmtIndent` specifier that
  /// indicates the indentation level of the start of the statement that
  /// contains this expression if the expression can exist at the end of the
  /// line.  This allows the expression parser to know when to keep parsing the
  /// expression on the next line - when it is more indented than the start of
  /// the current statement.  This can be passed in as None when there is a
  /// trailing punctuator that naturally terminates the expression.
  ParseResult parseExpressionList(SmallVectorImpl<ExprNode *> &results,
                                  Optional<size_t> stmtIndent,
                                  bool *hadTrailingSep);
  ParseResult parseExpression(ExprNode *&expr, Optional<size_t> stmtIndent);
  /// Parse an expression, allowing `=`, and `+=`.
  ParseResult parseExpressionOrAssignmentStmt(ExprNode *&expr,
                                              Optional<size_t> stmtIndent);
  ParseResult parseType(ASTType &result, ASTDecl &declScope,
                        Optional<size_t> stmtIndent);

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
