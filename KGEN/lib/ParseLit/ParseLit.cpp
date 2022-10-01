//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the main entrypoints for the lit parser.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ParseLit.h"
#include "LitLexer.h"

#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/Timing.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

using llvm::SMLoc;
using llvm::SourceMgr;

//===----------------------------------------------------------------------===//
// LitParserBase
//===----------------------------------------------------------------------===//

namespace {
/// This class implements logic that is common to many parts of the parser, but
/// which is independent of the concrete grammar.
struct LitParserBase {
  LitParserBase(LitLexer &lexer, MLIRContext *context)
      : lexer(lexer), context(context) {}

  MLIRContext *getContext() const { return context; }
  LitLexer &getLexer() { return lexer; }

  /// Return the indentation level of the specified token.
  Optional<size_t> getIndentation() const {
    return lexer.getIndentation(getToken());
  }

  /// Return the current token the parser is inspecting.
  const LitToken &getToken() const { return lexer.getToken(); }
  StringRef getTokenSpelling() const { return getToken().getSpelling(); }

  //===--------------------------------------------------------------------===//
  // Error Handling
  //===--------------------------------------------------------------------===//

  /// Emit an error and return failure.
  InFlightDiagnostic emitError(const Twine &message = {}) {
    return emitError(getToken().getLoc(), message);
  }
  InFlightDiagnostic emitError(SMLoc loc, const Twine &message = {});

  //===--------------------------------------------------------------------===//
  // Location Handling
  //===--------------------------------------------------------------------===//

  /// Encode the specified source location information into an attribute for
  /// attachment to the IR.
  Location translateLocation(llvm::SMLoc loc) {
    return lexer.translateLocation(loc);
  }

  //===--------------------------------------------------------------------===//
  // Token Parsing
  //===--------------------------------------------------------------------===//

  /// If the current token has the specified kind, consume it and return true.
  /// If not, return false.
  bool consumeIf(LitToken::Kind kind) {
    if (getToken().isNot(kind))
      return false;
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
  /// output a diagnostic and return failure.
  ParseResult parseToken(LitToken::Kind expectedToken, const Twine &message);

  /// Parse a list of elements, terminated with an arbitrary token.  This does
  /// not consume the stop token.
  ///
  /// list ::= (element)* STOPTOKEN
  ///
  ParseResult parseListUntil(LitToken::Kind stopToken,
                             const std::function<ParseResult()> &parseElement);

  /// Parse a list of elements continued with a separator token, like a comma.
  ///
  /// separated_list ::= (element (SEPARATOR element)*
  ///
  ParseResult
  parseSeparatedList(LitToken::Kind separator,
                     const std::function<ParseResult()> &parseElement);
  ParseResult
  parseCommaSeparatedList(const std::function<ParseResult()> &parseElement) {
    return parseSeparatedList(LitToken::comma, parseElement);
  }

  /// Consume tokens until one of the specified set of token, leaving the
  /// stopToken in the stream.  This produces an error if EOF is encountered.
  ///
  /// NOTE: This shouldn't be used in a real parser, this is just for phasing
  /// things in.
  ParseResult eatUntil(ArrayRef<LitToken::Kind> stopTokens) {
    while (getToken().isNot(LitToken::eof)) {
      // If we found our stop token, we succeeded!
      if (getToken().isAny(stopTokens))
        return success();
      consumeToken();
    }

    return emitError("expected end token");
  }

private:
  LitParserBase(const LitParserBase &) = delete;
  void operator=(const LitParserBase &) = delete;

  LitLexer &lexer;
  MLIRContext *const context;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Error Handling
//===----------------------------------------------------------------------===//

InFlightDiagnostic LitParserBase::emitError(SMLoc loc, const Twine &message) {
  auto diag = mlir::emitError(translateLocation(loc), message);

  // If we hit a parse error in response to a lexer error, then the lexer
  // already reported the error.
  if (getToken().is(LitToken::error))
    diag.abandon();
  return diag;
}

//===----------------------------------------------------------------------===//
// Token Parsing
//===----------------------------------------------------------------------===//

/// Consume the specified token if present and return success.  On failure,
/// output a diagnostic and return failure.
ParseResult LitParserBase::parseToken(LitToken::Kind expectedToken,
                                      const Twine &message) {
  if (consumeIf(expectedToken))
    return success();
  return emitError(message);
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

//===----------------------------------------------------------------------===//
// LitParser
//===----------------------------------------------------------------------===//

/// This class provides the implementation details of the concrete Lit Grammar.
struct LitParser : public LitParserBase {
  LitParser(LitLexer &lexer, ModuleOp module)
      : LitParserBase(lexer, module.getContext()), module(module) {}

  // Statements.
  ParseResult parseFile();
  ParseResult parseSuite(size_t curIndent);
  ParseResult parseStmts(size_t minIndent);
  ParseResult parseStmt(size_t curIndent);
  ParseResult parseSimpleStmt();

  // Compound statements.
  ParseResult parseDefStmt(size_t curIndent);

  // Simple statements.
  ParseResult parseReturnStmt();

private:
  // TODO: Current context to parse into (mutable).
  const ModuleOp module;
};

/// file ::= statements
ParseResult LitParser::parseFile() {
  // TODO: Build IR.
  return parseStmts(/*indent=*/0);
}

/// Parse a suite, which is either a series of comma separated simple_stmt's on
/// one line, or an indented block of statements. curIndent is the containing
/// statement's indentation level.
///
/// suite     ::=  stmt_list NEWLINE | NEWLINE INDENT statement+ DEDENT
/// stmt_list ::=  simple_stmt (";" simple_stmt)* [";"]
ParseResult LitParser::parseSuite(size_t curIndent) {
  auto indent = getIndentation();
  // If there is a newline, then parse a list of statements.
  if (indent.has_value()) {
    if (indent.value() <= curIndent)
      emitError("body should be indented more than containing statement");
    return parseStmts(indent.value());
  }

  // Otherwise, parse a stmt_list.
  do {
    if (parseSimpleStmt())
      return failure();
    // Stop if we see a semicolon at the end of line or a missing semicolon.
  } while (consumeIf(LitToken::semi) && !getIndentation().has_value());

  return success();
}

/// statements ::= statement+
///
/// This parses statements at the current indentation level or greater, it
/// refuses to parse things at lower indentation level.
ParseResult LitParser::parseStmts(size_t minIndent) {
  while (getToken().isNot(LitToken::eof)) {
    auto indent = getIndentation();
    if (!indent.has_value())
      return emitError("statements must start at the beginning of a line");
    if (indent.value() < minIndent)
      break;

    if (parseStmt(indent.value()))
      return failure();
  }
  return success();
}

/// statement ::= compound_stmt | simple_stmt
///
/// compound_stmt ::= if_stmt [TODO]
///                 | while_stmt [TODO]
///                 | for_stmt [TODO]
///                 | try_stmt [TODO]
///                 | with_stmt [TODO]
///                 | match_stmt [TODO]
///                 | funcdef
///                 | classdef [TODO]
///                 | async_with_stmt [TODO]
///                 | async_for_stmt [TODO]
///                 | async_funcdef [TODO]
///
ParseResult LitParser::parseStmt(size_t curIndent) {
  // Handle compound stmts here and chain to simple statements to handle the
  // whole "statement" production.
  switch (getToken().getKind()) {
  case LitToken::kw_def:
    return parseDefStmt(curIndent);
  default:
    // Otherwise must be a simple statement.
    return parseSimpleStmt();
  }
}

/// simple_stmt ::= expression_stmt [TODO]
///               | assert_stmt [TODO]
///               | assignment_stmt [TODO]
///               | augmented_assignment_stmt [TODO]
///               | annotated_assignment_stmt [TODO]
///               | pass_stmt
///               | del_stmt [TODO]
///               | return_stmt
///               | yield_stmt [TODO]
///               | raise_stmt [TODO]
///               | break_stmt [TODO]
///               | continue_stmt [TODO]
///               | import_stmt [TODO]
///               | future_stmt [TODO]
///               | global_stmt [TODO]
///               | nonlocal_stmtParseResult [TODO]
ParseResult LitParser::parseSimpleStmt() {
  switch (getToken().getKind()) {
  case LitToken::kw_pass:
    // pass_stmt ::= "pass"
    consumeToken(LitToken::kw_pass);
    return success();
  case LitToken::kw_return:
    return parseReturnStmt();
  default:
    return emitError("unexpected statement kind");
  }
}

/// funcdef ::=  [decorators] "def" funcname generic_signature?
///              "(" [parameter_list] ")"
///              ["->" expression] ":" suite
ParseResult LitParser::parseDefStmt(size_t curIndent) {
  // TODO: Add support for decorators.
  consumeToken(LitToken::kw_def);

  auto parseParameter = [&]() -> ParseResult {
    // TODO: implement this correctly.
    return eatUntil({LitToken::comma, LitToken::r_paren});
  };

  auto parseGenericSignature = [&]() -> ParseResult {
    // TODO: implement this correctly.
    return eatUntil(LitToken::l_paren);
  };

  StringRef funcName = getToken().getSpelling();
  if (parseToken(LitToken::identifier, "expected function name") ||
      parseGenericSignature() ||
      parseToken(LitToken::l_paren, "expected '(' for parameter list"))
    return failure();

  if (!consumeIf(LitToken::r_paren)) {
    if (parseCommaSeparatedList(parseParameter) ||
        parseToken(LitToken::r_paren, "expected ')' for parameter list"))
      return failure();
  }

  if (consumeIf(LitToken::minus_greater)) {
    // TODO: Parse return type.
    if (eatUntil(LitToken::colon))
      return failure();
  }

  if (parseToken(LitToken::colon, "expected ':' in function definition"))
    return failure();

  if (parseSuite(curIndent))
    return failure();

  // Build something.
  (void)funcName;

  return success();
}

/// return_stmt ::= "return" [expression_list]
ParseResult LitParser::parseReturnStmt() {
  consumeToken(LitToken::kw_return);

  // TODO: Parse expression_list correctly.
  while (getToken().isNot(LitToken::eof) && !getIndentation().has_value())
    consumeToken();

  return success();
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

// Parse the specified .lit file into the specified MLIR context.
OwningOpRef<mlir::ModuleOp> M::importLitFile(SourceMgr &sourceMgr,
                                             MLIRContext *context,
                                             mlir::TimingScope &ts) {
  auto sourceBuf = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());

  context->loadDialect<POP::POPDialect, KGENDialect>();

  // This is the result module we are parsing into.
  mlir::OwningOpRef<ModuleOp> module(ModuleOp::create(
      FileLineColLoc::get(context, sourceBuf->getBufferIdentifier(), /*line=*/0,
                          /*column=*/0)));

  // Parse the file.
  LitLexer lexer(sourceMgr, context);
  if (LitParser(lexer, *module).parseFile())
    return nullptr;

  // TODO: Need to decide on an error recovery policy.  Should probably return
  // failure here when parsing does not succeed.

  // Make sure the parse module has no other structural problems detected by
  // the verifier.
  auto verificationTimer = ts.nest("Verify module");
  if (failed(verify(*module)))
    return {};
  return module;
}
