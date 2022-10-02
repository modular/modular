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

#include "KGEN/HLKGENDialect/HLKGENOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "LLCL/Support/RCRef.h"
#include "LLCL/Support/ReferenceCounted.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/Timing.h"

using namespace M;
using namespace M::LLCL;
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

  /// Emit an error and notice that so we don't verify the IR at the end of
  /// compilation.
  InFlightDiagnostic emitError(Location loc, const Twine &message = {}) {
    errorOccurred = true;
    return mlir::emitError(loc, message);
  }

  /// Emit an error at the current token.
  InFlightDiagnostic emitError(const Twine &message = {}) {
    return emitError(getToken().getLoc(), message);
  }
  /// Emit an error at a specific lexer location.
  InFlightDiagnostic emitError(SMLoc loc, const Twine &message = {});

  /// Return true if we encountered an error during compilation.
  bool hadError() const { return errorOccurred; }

  //===--------------------------------------------------------------------===//
  // Location Handling
  //===--------------------------------------------------------------------===//

  /// Encode the specified source location information into an attribute for
  /// attachment to the IR.
  Location translateLocation(llvm::SMLoc loc) {
    return lexer.translateLocation(loc);
  }

  /// Return the location for the current token.
  Location getTokenLocation() { return translateLocation(getToken().getLoc()); }

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

protected:
  LitLexer &lexer;
  MLIRContext *const context;

private:
  bool errorOccurred = false;

  LitParserBase(const LitParserBase &) = delete;
  void operator=(const LitParserBase &) = delete;
};

} // end anonymous namespace

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
// Scope handling
//===----------------------------------------------------------------------===//

/// Scopes in Lightning work the same way as in Python: scopes are nested and
/// are defined when a builtin, module, class/struct, or function definition is
/// introduced.  Because Lightning (like Python) allows forward references to
/// values before they are defined, the body of declarations is parsed after the
/// signature of its peer declarations are all parsed.
///
/// This means that we can't just use a ScopedHashTable or similar - we need to
/// maintain our scopes until all bodies that refer to them are resolved.  As
/// such, we heap allocate and reference count these.
namespace {
class Scope : public NonAtomicallyReferenceCounted<Scope> {
public:
  Scope(Operation *decl, RCRef<Scope> parentScope)
      : decl(decl), parentScope(std::move(parentScope)) {}

  /// Return the Module, StructDecl, Func/Generator that this scope corresponds
  /// to.
  Operation *getDecl() const { return decl; }
  const RCRef<Scope> &getParentScope() const { return parentScope; }

  OpBuilder getBuilder() {
    return OpBuilder::atBlockEnd(&decl->getRegion(0).front());
  }

  /// Add the specified declaration to the current scope, returning non-null if
  /// a previous operation is already in this scope.
  Operation *addToScope(StringRef name, Operation *newDecl) {
    Operation *&entry = decls[name];
    if (entry)
      return entry;
    entry = newDecl;
    return nullptr;
  }

  /// Perform a lookup in this scope tree, returning the nearest target or null
  /// if nothing is found.
  Operation *lookup(StringRef name) {
    Scope *curScope = this;
    while (curScope) {
      auto it = curScope->decls.find(name);
      if (it != curScope->decls.end())
        return it->second;
      curScope = curScope->parentScope.getPointer();
    }
    return nullptr;
  }

private:
  /// This is the Module, StructDecl, Func/Generator that this scope corresponds
  /// to.
  Operation *decl;
  RCRef<Scope> parentScope;

  // Note: we could unique the identifiers and use a DenseMap.
  llvm::StringMap<Operation *> decls;
};

} // namespace

//===----------------------------------------------------------------------===//
// LitParser
//===----------------------------------------------------------------------===//

/// Declaration bodies are parsed after all the signatures at the current level
/// of the file are parsed.  This keeps track of
struct DeferredDeclBodyToParse {
  /// This is the scope for the declaration, which also contains the declaration
  /// itself.
  RCRef<Scope> declScope;

  /// This is where to start lexing the body from.
  LitLexerCursor lexerCursor;

  /// This is the indentation level of the decl.
  size_t indentLevel;
};

/// This class provides the implementation details of the concrete Lit Grammar.
struct LitParser : public LitParserBase {
  LitParser(LitLexer &lexer, ModuleOp module)
      : LitParserBase(lexer, module.getContext()), module(module) {}

  ParseResult parseFile();
  void finalizeScopeDecl();

private:
  // Statements.
  ParseResult parseSuite(size_t curIndent);
  ParseResult parseStmts(size_t minIndent);
  ParseResult parseStmt(size_t curIndent);
  ParseResult parseSimpleStmt();

  // Compound statements.
  ParseResult parseDefStmt(size_t curIndent);
  void parseDefBody(size_t defIndent, HLGeneratorOp defDecl);

  // Simple statements.
  ParseResult parseReturnStmt();

  // Expressions.
  ParseResult parseExpressionList();
  ParseResult parseExpression();

private:
  const ModuleOp module;

  /// This is the current context that we're parsing into.
  RCRef<Scope> currentScope;

  /// These are deferred declarations that need parsing, which are processed
  /// after other things in a scope have been resolved.
  std::vector<DeferredDeclBodyToParse> deferredDecls;
};

/// file ::= statements
ParseResult LitParser::parseFile() {
  // The outermost scope contains the __builtins__ function definitions.
  // TODO: Add these:
  // https://docs.python.org/3/library/functions.html#built-in-funcs
  // https://docs.python.org/3/reference/executionmodel.html#naming-and-binding
  auto builtinsScope = RCRef<Scope>::create(module, RCRef<Scope>());

  // Create the module scope which will contain all things we parse.  These
  // shadow the builtins module during name lookup.
  currentScope = RCRef<Scope>::create(module, std::move(builtinsScope));

  // We fail either if we have a non-recoverable parse error, or if we emitted
  // an error and then recovered.  In either case, the IR will not be valid and
  // the caller should not verify it.
  if (parseStmts(/*indent=*/0))
    return failure();

  // Finalize the current scope, parsing any deferred declarations in it.
  finalizeScopeDecl();

  if (hadError())
    return failure();

  return success();
}

/// Finalize parsing of a scoped declaration (e.g. module, class, function).
///
/// Once its body is fully parsed, we loop back around to parse the bodies of
/// any nested scopes (e.g. nested functions) that are encountered while parsing
/// this scope.  This ensures that the forward references between peer
/// declarations are handled correctly, for example in mutually recursive
/// functions and code like this:
///
///   def foo():
///     def bar():
///       print(x)
///     x = 42
///     bar()
///   foo()
void LitParser::finalizeScopeDecl() {
  // We're done with the current scope and the declaration we're parsing into.
  currentScope.reset();
  if (deferredDecls.empty())
    return;

  // If we have deferred declarations, process each of them.
  std::vector<DeferredDeclBodyToParse> decls;
  std::swap(deferredDecls, decls);

  for (DeferredDeclBodyToParse &decl : decls) {
    currentScope = std::move(decl.declScope);
    decl.lexerCursor.restore(lexer);

    // Only support def's right now.
    parseDefBody(decl.indentLevel,
                 cast<HLGeneratorOp>(currentScope->getDecl()));
  }
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

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

/// simple_stmt ::= expression_stmt
///               | assert_stmt [TODO]
///               | assignment_stmt
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
    break;
  }

  // Otherwise, we must have a statement that starts with the expression
  // grammar.

  // expression_stmt ::= starred_expression
  // assignment_stmt ::=
  //                 (target_list "=")+ (starred_expression | yield_expression)
  //  target_list     ::=  target ("," target)* [","]
  // target ::= identifier
  //          | "(" [target_list] ")" | "[" [target_list] "]"
  //          | attributeref | subscription | slicing | "*" target
  if (parseExpression())
    return failure();

  // If the expression was followed by a `=` then we have an assignment.
  if (!consumeIf(LitToken::equal))
    return success(); // expression_stmt.

  // Must be assignment_stmt

  // TODO: Check the LHS expression is a `target_list` to reject "x+4=7"
  return parseExpression();
}

/// funcdef ::=  [decorators] "def" funcname generic_signature?
///              "(" [parameter_list] ")"
///              ["->" expression] ":" suite
ParseResult LitParser::parseDefStmt(size_t curIndent) {
  auto loc = getTokenLocation();

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

  auto builder = currentScope->getBuilder();
  auto nameAttr = builder.getStringAttr(funcName);
  auto functionType = builder.getFunctionType({}, {});
  auto symVisibility = builder.getStringAttr("public");
  // TODO: Should have nicer builder.
  auto newFunc = builder.create<HLGeneratorOp>(
      loc, nameAttr, symVisibility, TypeAttr::get(functionType),
      ParamDeclArrayAttr::get(context, {}), TypeArrayAttr::get(context, {}),
      ConstraintArrayAttr::get(context, {}), FlatSymbolRefAttr());
  newFunc.getRegion().push_back(new Block());

  auto prevDecl = currentScope->addToScope(funcName, newFunc);
  if (prevDecl) {
    auto diag = emitError(loc, "invalid redefinition of function ") << nameAttr;
    diag.attachNote(prevDecl->getLoc()) << "previous definition here";
    // Keep parsing even though we failed to add to the scope.  Note that this
    // can cause type errors downstream.
    // TODO: We should mark both declarations erroneous in the symbol table so
    // reference to them get squashed as errors during name lookup, avoiding
    // cascading errors.
  }

  // We cannot parse the current body without having parsed other declarations
  // at the current level, so we defer parsing it.  Remember that we need to
  // do so.
  deferredDecls.push_back({RCRef<Scope>::create(newFunc, currentScope.copy()),
                           lexer.getCursor(), curIndent});

  // Skip the body of this definition: go to a token the starts a line at the
  // same indent level (or less) as the function definition.
  while (getToken().isNot(LitToken::eof)) {
    auto indent = getIndentation();
    if (indent.has_value() && indent.value() <= curIndent)
      break;
    consumeToken();
  }

  return success();
}

/// Parse a deferred 'def' body.
void LitParser::parseDefBody(size_t defIndent, HLGeneratorOp defDecl) {
  (void)parseSuite(defIndent);

  // Add kgen.return so the IR verifies.
  // TODO: Generalize hlkgen.generator.
  auto returnParams = ArrayAttr::get(context, {});
  OpBuilder::atBlockEnd(defDecl.getBody())
      .create<ReturnOp>(defDecl->getLoc(), returnParams, ArrayRef<Value>());

  finalizeScopeDecl();
}

/// return_stmt ::= "return" [expression_list]
ParseResult LitParser::parseReturnStmt() {
  consumeToken(LitToken::kw_return);
  return parseExpressionList();
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

/// expression_list ::= expression ("," expression)* [","]
ParseResult LitParser::parseExpressionList() {
  // TODO: Support trailing comma for singleton tuple.
  return parseCommaSeparatedList(
      [&]() -> ParseResult { return parseExpression(); });
}

/// expression ::= atom | call
///
/// atom    ::= identifier | literal | enclosure [TODO]
/// call    ::=  primary "(" [argument_list [","] | comprehension] ")"
///
/// literal ::= [TODO]
///     stringliteral | bytesliteral | integer | floatnumber | imagnumber
///
ParseResult LitParser::parseExpression() {
  // TODO: Handle precedence.
  switch (getToken().getKind()) {
  case LitToken::identifier: // expression -> atom -> identifier
    if (!currentScope->lookup(getToken().getSpelling())) {
      emitError("use of unknown declaration \"")
          << getToken().getSpelling() << '"';
      // TODO: return an error expression.
    }
    consumeToken(LitToken::identifier);
    break;
  case LitToken::integer: // expression -> literal -> integer
    consumeToken(LitToken::integer);
    break;
  default:
    return emitError("unexpected token in expression");
  }

  // Parse postfix productions.
  while (1) {
    // Handle calls.
    if (consumeIf(LitToken::l_paren)) {
      // TODO: Handle comprehension arguments.
      if (!consumeIf(LitToken::r_paren)) {
        if (parseExpressionList() ||
            parseToken(LitToken::r_paren, "expected ')' in call argument list"))
          return failure();
      }
      continue;
    }
    break;
  }

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

  context->loadDialect<POP::POPDialect, HLKGENDialect, KGENDialect>();

  // This is the result module we are parsing into.
  mlir::OwningOpRef<ModuleOp> module(ModuleOp::create(
      FileLineColLoc::get(context, sourceBuf->getBufferIdentifier(), /*line=*/0,
                          /*column=*/0)));

  // Parse the file.
  LitLexer lexer(sourceMgr, context);
  if (LitParser(lexer, *module).parseFile())
    return nullptr;

  // Make sure the parse module has no other structural problems detected by
  // the verifier.
  auto verificationTimer = ts.nest("Verify module");
  if (failed(verify(*module)))
    return {};
  return module;
}
