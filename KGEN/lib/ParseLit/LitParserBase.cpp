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

InFlightDiagnostic LitParserBase::emitError(SMLoc loc, const Twine &message) {
  auto diag = emitError(translateLocation(loc), message);

  // If we hit a parse error in response to a lexer error, then the lexer
  // already reported the error.
  if (getToken().is(LitToken::error))
    diag.abandon();
  return diag;
}

/// Consume the specified token if present and return success.  On failure,
/// output a diagnostic and return failure.
ParseResult LitParserBase::parseToken(LitToken::Kind expectedToken,
                                      const Twine &message) {
  if (consumeIf(expectedToken))
    return success();
  return emitError(message);
}

/// Consume an identifier token, binding its name into the specified result
/// string attribute.
ParseResult LitParserBase::parseIdentifier(StringAttr &result,
                                           const Twine &message) {
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
///
/// separated_list ::= (element (SEPARATOR element)*
///
ParseResult LitParserBase::parseSeparatedList(
    LitToken::Kind separator,
    const std::function<ParseResult()> &parseElement) {
  if (parseElement())
    return failure();
  while (consumeIf(separator)) {
    if (parseElement())
      return failure();
  }
  return success();
}
