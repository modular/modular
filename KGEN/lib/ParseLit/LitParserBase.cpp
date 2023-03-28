//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This implements the base class for Lit file parsers, logic that is shared
// between expression and statement parsing in particular.
//
//===----------------------------------------------------------------------===//

#include "LitParserBase.h"
using namespace M::KGEN::LIT;
using namespace M;

LitDiagnostic LitParserBase::emitError(SMLoc loc, const Twine &message) {
  auto diag = shared.emitError(loc, message);

  // If we hit a parse error in response to a lexer error, then the lexer
  // already reported the error.
  if (getToken().is(LitToken::error))
    diag.abandon();
  return diag;
}

/// Consume the specified token if present and return success.  On failure,
/// output a diagnostic and return failure.
ParseResult LitParserBase::parseToken(LitToken::Kind expectedToken,
                                      const Twine &message, SMLoc *loc) {
  if (loc)
    *loc = getToken().getLoc();
  if (consumeIf(expectedToken))
    return success();

  // If the current token is on a new line, report the error on the end of the
  // previous line, this is probably where the punctuation was omitted.
  auto diagLoc = getTokenLocOrEndOfPreviousLineIfOnNewLine();

  // Report the error.
  auto diag = emitError(diagLoc, message);

  // Customize the error if an identifier was expected by a keyword was found.
  if (expectedToken == LitToken::identifier && getToken().isKeyword())
    diag.attachNote(diagLoc) << "escape keyword '" << getToken().getSpelling()
                             << "' with backticks to use it as an identifier";

  return failure();
}

/// Consume an identifier token, binding its name into the specified result
/// string attribute. If `loc` is set, it is populated with the source location
/// of the token.
ParseResult LitParserBase::parseIdentifier(StringAttr &result,
                                           const Twine &message, SMLoc *loc) {
  if (loc)
    *loc = getToken().getLoc();
  result = StringAttr::get(getContext(), getToken().getSpelling());
  return parseToken(LitToken::identifier, message);
}

/// Parse a list of elements, terminated with an arbitrary token.  This does
/// not consume the stop token.
///
/// list ::= (element)* STOPTOKEN
///
ParseResult LitParserBase::parseListUntil(
    LitToken::Kind rightToken,
    const std::function<ParseResult()> &parseElement) {

  while (!consumeIf(rightToken)) {
    if (parseElement())
      return failure();
  }
  return success();
}

/// Parse a list of elements continued with a separator token, like a comma.
/// The list ends either with a terminator, which is not consumed, or a new
/// line. hadTrailingSep is set to true if a trailing separator was found.
///
/// separated_list ::= (element (SEPARATOR element)* [SEPARATOR] TERMINATOR
///
ParseResult LitParserBase::parseSeparatedList(
    LitToken::Kind separator, const std::function<ParseResult()> &parseElement,
    ArrayRef<LitToken::Kind> terminators, bool *hadTrailingSep) {
  if (hadTrailingSep)
    *hadTrailingSep = false;
  if (parseElement())
    return failure();
  while (consumeIf(separator)) {
    // Empty terminators signals no terminator was given as input so check for
    // "new line": if we have indentation it means we are starting a line
    // after the last separator.
    if (getToken().isAny(terminators) ||
        (terminators.empty() && getToken().getIndentation().has_value())) {
      if (hadTrailingSep)
        *hadTrailingSep = true;
      break;
    }
    if (parseElement())
      return failure();
  }
  return success();
}

/// Skip tokens until we get to a token at start of line that has indentation
/// that is equal or less than the specified indentation.  This is used for
/// multiphase parsing.
///
/// When stopOnSemicolon is true this will stop at the first semicolon seen.
/// This should only be used for statements that can share a line with other
/// statements with ; separation.
void LitParserBase::skipUntilIndentation(size_t minIndent,
                                         bool stopOnSemicolon) {
  // This keeps track of open brackets we are inside of.
  SmallVector<LitToken> openBrackets;

  auto handleCloseBracket = [&](LitToken::Kind leftBracket) {
    // If we see the correct closing bracket for the structure we're in, then
    // just pop out of that context and keep going.
    if (!openBrackets.empty() && openBrackets.back().getKind() == leftBracket) {
      openBrackets.pop_back();
      return;
    }

    // Otherwise, we have a parse error: don't diagnose it though, because the
    // non-skipping parse will.  We don't really know how best to recover so we
    // just nuke our scope which will cause us to stop skipping at the
    // indentation level requested.
    openBrackets.clear();
  };

  // We scan until we find the specified indentation at the same expression
  // level as the current token.
  while (getToken().isNot(LitToken::eof)) {
    // If we are outside a bracketed expression, check indentation.
    if (auto indent = getToken().getIndentation())
      if (*indent <= minIndent && openBrackets.empty())
        return;

    // Check to see if this is a bracket that needs special handling.
    switch (getToken().getKind()) {
    default:
      break;
    case LitToken::l_paren:
    case LitToken::l_square:
    case LitToken::l_brace:
      // Remember that we're nested.
      openBrackets.push_back(getToken());
      break;

      // Handle closing brackets.
    case LitToken::r_paren:
      handleCloseBracket(LitToken::l_paren);
      break;
    case LitToken::r_square:
      handleCloseBracket(LitToken::l_square);
      break;
    case LitToken::r_brace:
      handleCloseBracket(LitToken::l_brace);
      break;

      // Stop on semicolons when outside a bracket expression if requested.
    case LitToken::semi:
      if (stopOnSemicolon && openBrackets.empty())
        return;
      break;
    }

    // Otherwise, keep eating.
    consumeToken();
  }
}
