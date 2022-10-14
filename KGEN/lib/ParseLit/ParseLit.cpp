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
#include "LitExprNodes.h"
#include "LitLexer.h"
#include "LitScope.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/IndexDialect/IndexDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/Timing.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace M;
using namespace M::LLCL;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

using llvm::SourceMgr;

//===----------------------------------------------------------------------===//
// Scope
//===----------------------------------------------------------------------===//

static Location getLocationFrom(Scope::ScopeValue value) {
  if (std::holds_alternative<VarDeclOp>(value))
    return std::get<VarDeclOp>(value).getLoc();
  return std::get<Scope::MetaParameterValue>(value).loc;
}

/// Add the specified declaration to the current scope, emitting an error on
/// a name collision.
void Scope::addToScope(StringRef name, ScopeValue newValue, bool &hadError) {
  Optional<Scope::ScopeValue> &entry = decls[name];
  if (!entry) {
    entry = newValue;
    return;
  }

  auto diag = emitError(getLocationFrom(newValue), "invalid redefinition of \"")
              << name << '"';
  diag.attachNote(getLocationFrom(entry.value())) << "previous definition here";
  hadError = true;

  // TODO: We should mark both declarations erroneous in the symbol table
  // so reference to them get squashed as errors during name lookup,
  // avoiding cascading errors.
}

//===----------------------------------------------------------------------===//
// SharedParserState
//===----------------------------------------------------------------------===//

namespace {
/// This is state shared across multiple different instances of LitParserBase
/// which are always shared across them.
struct SharedParserState {
  MLIRContext *const context;
  llvm::BumpPtrAllocator exprAllocator;
  bool hasExprParser = false;
  bool errorOccurred = false;

  SharedParserState(MLIRContext *context) : context(context) {}
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// LitParserBase
//===----------------------------------------------------------------------===//

namespace {
/// This class implements logic that is common to many parts of the parser, but
/// which is independent of the concrete grammar.
struct LitParserBase {
  LitParserBase(LitLexer &lexer, SharedParserState *sharedParserState)
      : lexer(lexer), sharedParserState(sharedParserState) {}

  MLIRContext *getContext() const { return sharedParserState->context; }
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
    sharedParserState->errorOccurred = true;
    return mlir::emitError(loc, message);
  }

  /// Emit an error at the current token.
  InFlightDiagnostic emitError(const Twine &message = {}) {
    return emitError(getToken().getLoc(), message);
  }
  /// Emit an error at a specific lexer location.
  InFlightDiagnostic emitError(SMLoc loc, const Twine &message = {});

  /// Return true if we encountered an error during compilation.
  bool hadError() const { return sharedParserState->errorOccurred; }

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
  /// If not, return false.  If 'tokLoc' is non-null, it is filled in with the
  /// location of the consumed token (on success).
  bool consumeIf(LitToken::Kind kind, SMLoc *tokLoc = nullptr) {
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

  /// Skip tokens until we get to a token at start of line that has indentation
  /// that is equal or less than the specified indentation.  This is used for
  /// multiphase parsing.
  void skipUntilIndentation(size_t minIndent) {
    while (getToken().isNot(LitToken::eof)) {
      auto indent = getToken().getIndentation();
      if (indent.has_value() && indent.value() <= minIndent)
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

public:
  LitLexer &lexer;
  SharedParserState *const sharedParserState;

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

//===----------------------------------------------------------------------===//
// Expression Parsing
//===----------------------------------------------------------------------===//

/// Expression parsing in Lightning is done in with a 2-phase approach where we
/// parse one or more expressions into an AST-like representation in a first
/// pass, then type check and generate IR for it in a second pass.  This enables
/// a number of features:
///
///   1) Non-lexical variable references: `[x.strip().upper() for x in flags]`
///   2) Weird order of evaluations: `foo() if cond() else bar()`
///   3) Parser ambiguity of the LHS of an assignment, which we don't know if it
///      is a target until we see the equals: `x[foo()] = bar()`
///   4) Contextually sensitive type checking, e.g. x = 42 where x is known to
///      be Int8 instead of Int.
///
/// We handle this by having an expression parser distinct from the main parser
/// that builds this tree and manages the lifetime of the nodes.  Only one
/// expression parser may be active at a time, which allows us to bump pointer
/// allocate the notes we create for the expression tree.
///
class ExprParser : public LitParserBase {
public:
  ExprParser(LitParserBase &existing, const RCRef<Scope> &currentScope)
      : LitParserBase(existing.lexer, existing.sharedParserState),
        currentScope(currentScope.getPointer()) {
    // Only a single expression parser can be active at a time, because we clear
    // the bump pointer allocator when done.
    assert(!sharedParserState->hasExprParser &&
           "Cannot create multiple expr parsers at once");
    sharedParserState->hasExprParser = true;
  }

  ~ExprParser() {
    assert(sharedParserState->hasExprParser);
    sharedParserState->hasExprParser = false;
    /// Free all the expression nodes.
    sharedParserState->exprAllocator.Reset();
  }

  // Expressions.  These methods always return a non-null ExprNode, but it may
  // be (or include) an Error node if parsing failed.
  void parseExpressionList(SmallVectorImpl<ExprNode *> &results);
  ExprNode *parseExpression();
  ExprNode *parsePrimary();

  enum class Precedence {
    kInvalid, // Not a binary operator token.
    kLowest,  // Lowest precedence (most loosely bound).
    kAdd,
    kMul, // Highest precedence (most tightly bound).
  };
  std::pair<Precedence, ExprNode::Kind> getBinOpTokenPrecedenceAndKind() const;

  ExprNode *parseBinOpRHS(ExprNode *lhs, Precedence minPrec);

  /// Emit the specified expression tree to MLIR in the current context.
  MLIRValueRep emit(ExprNode *node);

  Value emitAsValue(ExprNode *node) {
    auto builder = currentScope->getBuilder();
    return emit(node).getAsValue(translateLocation(node->getLoc()), builder);
  }

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

  Scope *currentScope;
};

/// Emit the specified expression tree to MLIR in the current context.
MLIRValueRep ExprParser::emit(ExprNode *node) {
  // TODO: Need a notion of a current builder that isn't just end of decl.
  auto builder = currentScope->getBuilder();
  EmitterState state{
      builder, currentScope,
      [&](SMLoc loc) -> Location { return translateLocation(loc); },
      [&](SMLoc loc, const Twine &twine) -> InFlightDiagnostic {
        return emitError(loc, twine);
      }};
  return node->emit(state);
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

/// expression_list ::= expression ("," expression)* [","]
void ExprParser::parseExpressionList(SmallVectorImpl<ExprNode *> &results) {
  // TODO: Support trailing comma for singleton tuple.
  (void)parseCommaSeparatedList([&]() -> ParseResult {
    results.push_back(parseExpression());
    return success();
  });
}

/// expression ::=
///
///
ExprNode *ExprParser::parseExpression() {
  return parseBinOpRHS(parsePrimary(), Precedence::kLowest);
}

/// Return the operator precedence for the specified token or
std::pair<ExprParser::Precedence, ExprNode::Kind>
ExprParser::getBinOpTokenPrecedenceAndKind() const {
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
ExprNode *ExprParser::parsePrimary() {
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
ExprNode *ExprParser::parseBinOpRHS(ExprNode *lhs, Precedence minPrec) {
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

  /// This is the location of each input parameter.
  std::vector<Location> inputParamLocs;
};

/// This class provides the implementation details of the concrete Lit Grammar.
struct LitParser : public LitParserBase {
  LitParser(LitLexer &lexer, SharedParserState *sharedParserState,
            ModuleOp module)
      : LitParserBase(lexer, sharedParserState), module(module) {}

  ParseResult parseFile();
  void finalizeScopeDecl();

  const RCRef<Scope> &getCurrentScope() const { return currentScope; }

private:
  // Statements.
  enum class StmtContext {
    normal,     // All normal statements are supported.
    structBody, // Only statements in a struct body supported.
  };
  ParseResult parseSuite(size_t curIndent, StmtContext stmtContext);
  ParseResult parseStmts(size_t minIndent, StmtContext stmtContext);
  ParseResult parseStmt(size_t curIndent, StmtContext stmtContext);
  ParseResult parseSimpleStmt(StmtContext stmtContext);

  // Compound statements.
  ParseResult parseDefStmt(size_t curIndent);
  void parseDefBody(LITFuncOp defDecl, size_t defIndent,
                    ArrayRef<Location> inputParamLocs);
  ParseResult parseStructStmt(size_t curIndent);

  // Simple statements.
  ParseResult parseReturnStmt();
  ParseResult parseAssignmentStmt(ExprParser &exprParser, ExprNode *lhs,
                                  SMLoc equalsLoc);

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
  if (parseStmts(/*indent=*/0, StmtContext::normal))
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

    parseDefBody(cast<LITFuncOp>(currentScope->getDecl()), decl.indentLevel,
                 decl.inputParamLocs);
  }
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

/// Parse a suite, which is either a series of comma separated simple_stmt's on
/// one line, or an indented block of statements. curIndent is the containing
/// statement's indentation level and stmtContext indicates if there are a
/// subset of statements supported.
///
/// suite     ::=  [stmt_list NEWLINE] | NEWLINE INDENT statement+ DEDENT
/// stmt_list ::=  simple_stmt (";" simple_stmt)* [";"]
ParseResult LitParser::parseSuite(size_t curIndent, StmtContext stmtContext) {
  // Ignore empty body at end of file: a `pass` is not required.
  if (getToken().is(LitToken::eof))
    return success();

  // If there is a newline, then parse a list of statements.
  auto indent = getToken().getIndentation();
  if (indent.has_value()) {
    // If the current token is less indented that the source of the suite,
    // then the body is empty.  We don't require a pass.
    if (indent.value() <= curIndent)
      return success();
    return parseStmts(indent.value(), stmtContext);
  }

  // Otherwise, parse a stmt_list.
  do {
    if (parseSimpleStmt(stmtContext))
      return failure();
    // Stop if we see a semicolon at the end of line or a missing semicolon.
  } while (consumeIf(LitToken::semi) &&
           !getToken().getIndentation().has_value());

  return success();
}

/// statements ::= statement+
///
/// This parses statements at the current indentation level or greater, it
/// refuses to parse things at lower indentation level.
ParseResult LitParser::parseStmts(size_t minIndent, StmtContext stmtContext) {
  while (getToken().isNot(LitToken::eof)) {
    auto indent = getToken().getIndentation();
    if (!indent.has_value())
      return emitError("statements must start at the beginning of a line");
    if (indent.value() < minIndent)
      break;

    if (parseStmt(indent.value(), stmtContext))
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
///                 | structdef
///                 | classdef [TODO]
///                 | async_with_stmt [TODO]
///                 | async_for_stmt [TODO]
///                 | async_funcdef [TODO]
///
ParseResult LitParser::parseStmt(size_t curIndent, StmtContext stmtContext) {
  // Handle compound stmts here and chain to simple statements to handle the
  // whole "statement" production.
  switch (getToken().getKind()) {
  case LitToken::kw_def:
    return parseDefStmt(curIndent);
  case LitToken::kw_struct:
    // We don't support structs in structs (yet?).
    if (stmtContext != StmtContext::normal)
      emitError("nested struct not supported here");
    return parseStructStmt(curIndent);

  // NOTE: When adding new cases here, make sure to add them to parseSimpleStmt
  // as well for error recovery.
  default:
    // Otherwise must be a simple statement.
    return parseSimpleStmt(stmtContext);
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
ParseResult LitParser::parseSimpleStmt(StmtContext stmtContext) {
  switch (getToken().getKind()) {
  case LitToken::kw_def:
  case LitToken::kw_struct:
    emitError() << "'" << getToken().getSpelling()
                << "' statement must be on its own line";
    return parseStmt(0, stmtContext);

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
  if (stmtContext != StmtContext::normal)
    emitError("invalid expression in this context");

  // expression_stmt ::= starred_expression
  // assignment_stmt ::=
  //                 (target_list "=")+ (starred_expression | yield_expression)
  ExprParser exprParser(*this, currentScope);
  ExprNode *expr = exprParser.parseExpression();

  // If the expression was followed by a `=` then we have an assignment.  If not
  // then we have an expression_stmt.
  SMLoc equalsLoc;
  if (consumeIf(LitToken::equal, &equalsLoc))
    return parseAssignmentStmt(exprParser, expr, equalsLoc);

  // Materialize the expression statement in our current scope but discard the
  // result on the floor.
  (void)exprParser.emit(expr);
  return success();
}

/// Parse an assignment_stmt after having parsed a leading expression (which
/// we need to resolve into a target_list) and an `=` sign.
///
/// assignment_stmt ::=
///                 (target_list "=")+ (starred_expression | yield_expression)
/// target_list     ::=  target ("," target)* [","]
/// target ::= identifier
///          | "(" [target_list] ")" | "[" [target_list] "]"
///          | attributeref | subscription | slicing | "*" target
///
ParseResult LitParser::parseAssignmentStmt(ExprParser &exprParser,
                                           ExprNode *lhs, SMLoc equalsLoc) {
  // Resolve the parse expression on the LHS into an lvalue that we can store
  // into.
  // TODO: implement support for generalized lvalues / target_list.
  auto dre = dyn_cast<DeclRefNode>(lhs);
  if (!dre) {
    if (!lhs->containsError())
      emitError(lhs->getLoc(), "cannot assign to expression");
    eatToEndOfLine();
    return success();
  }

  auto builder = currentScope->getBuilder();

  // Look up the name being assigned to if it already exists.
  Value lvalue;
  if (Optional<Scope::ScopeValue> decl =
          currentScope->lookupInCurrentScope(dre->spelling)) {
    // Don't allow reassigning to functions and other constant parameters.
    if (std::holds_alternative<VarDeclOp>(decl.value()))
      lvalue = std::get<VarDeclOp>(decl.value());
    else
      emitError(lhs->getLoc(), "this declaration isn't reassignable");
  } else {
    // Otherwise, introduce a new lit.var.decl node.

    // TODO: Add types instead of hard coding to index type!
    auto declType = POP::PointerType::get(builder.getIndexType());

    // TODO: This will emit it on first use, which can be in weird places (e.g.
    // inside of the branch of an if statement).  We want to use dataflow
    // analysis to do definitive analysis of the accesses to the declaration. We
    // could just emit all these in the entry to the enclosing function/module
    // to maintain SSA.
    auto varDecl =
        builder.create<VarDeclOp>(translateLocation(lhs->getLoc()), declType,
                                  builder.getStringAttr(dre->spelling));
    currentScope->addToScope(dre->spelling, varDecl,
                             sharedParserState->errorOccurred);
    lvalue = varDecl;
  }

  ExprNode *rhs = exprParser.parseExpression();

  // Materialize the expression statement in our current scope.
  // TODO: Should pass in contextual type if known from previous declaration.
  auto rhsValue = exprParser.emitAsValue(rhs);

  // If IR generation failed, return success since we have a fine parse.
  if (!lvalue || !rhsValue)
    return success();

  if (!rhsValue.getType().isIndex()) {
    // emitError(rhs->getLoc(), "TODO: don't support non-index types yet");
    return success();
  }

  // If everything worked out, store the resultant value into the lvalue for the
  // destination.  If things didn't work, just drop this on the floor.
  builder.create<POP::StoreOp>(translateLocation(equalsLoc), rhsValue, lvalue,
                               /*alignment*/ None);

  return success();
}

namespace {
/// identifier_opt_type  ::= identifier [":" expression]
/// meta_signature    ::= "[" [meta_param_list] "]"
/// meta_param_list   ::= identifier_opt_type ("," identifier_opt_type)
struct MetaSignatureParser {
  SmallVector<ParamDeclAttr> inputParameters;
  std::vector<Location> inputParamLocs;

  ParseResult parseOptionalMetaSignature(LitParser &p) {
    if (!p.consumeIf(LitToken::l_square) || p.consumeIf(LitToken::r_square))
      return success();

    auto parseMetaParameter = [&]() -> ParseResult {
      inputParamLocs.push_back(p.getTokenLocation());

      StringAttr name;
      if (p.parseIdentifier(name, "expected parameter name"))
        // TODO: Scan ahead for better recovery.
        return failure();

      Type paramType = IndexType::get(p.getContext());
      if (p.consumeIf(LitToken::colon)) {
        ExprParser exprParser(p, p.getCurrentScope());
        ExprNode *typeExpr = exprParser.parseExpression();
        // TODO (types): translate typeExpr into a type.
        (void)typeExpr;
      }
      inputParameters.push_back(ParamDeclAttr::get(name, paramType));
      return success();
    };

    if (p.parseCommaSeparatedList(parseMetaParameter) ||
        p.parseToken(LitToken::r_square, "expected ']' for parameter list"))
      return failure();
    return success();
  };
};
} // namespace

/// funcdef ::=  [decorators] "def" identifier [meta_signature]
///              "(" [value_param_list] ")" ["->" expression] ":" suite
///
/// value_param_list  ::= value_parameter ("," value_parameter)*
/// value_parameter   ::= value_parammarker identifier_opt_type ["=" expression]
/// value_parammarker ::= "/" | "*" | "**"
///
ParseResult LitParser::parseDefStmt(size_t curIndent) {
  auto loc = getTokenLocation();

  // TODO: Add support for decorators.
  consumeToken(LitToken::kw_def);

  // TODO: Implement support for variadic parameter markers:
  // Python's parameter grammar embeds checking for `/` and `*` and `**` into
  // the grammar, we can just check for it using ad-hoc logic for simplicity,
  // according to the following rules:
  //   1) Only one /, *, and ** parameter may exist in the parameter list.
  //   2) They are specified in that order.
  //   3) These do not permit default arguments.
  SmallVector<Location> valueParamLocs;
  SmallVector<StringAttr> valueParamNames;
  SmallVector<Type> valueParamTypes;
  // TODO: Default values.

  auto parseParameter = [&]() -> ParseResult {
    auto loc = getTokenLocation();
    if (parseIdentifier(valueParamNames.emplace_back(StringAttr()),
                        "expected parameter name"))
      // TODO: Scan ahead for better recovery.
      return failure();

    Type paramType = IndexType::get(getContext());
    if (consumeIf(LitToken::colon)) {
      ExprParser exprParser(*this, currentScope);
      ExprNode *typeExpr = exprParser.parseExpression();
      // TODO (types): translate typeExpr into a type.
      (void)typeExpr;
    }
    valueParamLocs.push_back(loc);
    valueParamTypes.push_back(paramType);

    if (consumeIf(LitToken::equal)) {
      ExprParser exprParser(*this, currentScope);
      ExprNode *defaultExpr = exprParser.parseExpression();
      // TODO: add support for default parameter expressions.
      if (!defaultExpr->containsError())
        emitError(defaultExpr->getLoc(),
                  "default parameters not supported yet");
    }
    return success();
  };

  StringAttr funcNameAttr;
  MetaSignatureParser metaSignature;
  if (parseIdentifier(funcNameAttr, "expected function name") ||
      metaSignature.parseOptionalMetaSignature(*this) ||
      parseToken(LitToken::l_paren, "expected '(' for parameter list"))
    return failure();

  if (!consumeIf(LitToken::r_paren)) {
    if (parseCommaSeparatedList(parseParameter) ||
        parseToken(LitToken::r_paren, "expected ')' for parameter list"))
      return failure();
  }

  // Parse the result type if present.
  SmallVector<Type> resultTypes;
  // TODO: This will be one difference between a def and fn: no result type on
  // a def should default to returning a (default initialized) Object, whereas
  // a fn can return void.  We can provide a guaranteed optimization to remove
  // it though.
  if (consumeIf(LitToken::minus_greater)) {
    ExprParser exprParser(*this, currentScope);
    ExprNode *typeExpr = exprParser.parseExpression();
    // TODO (types): translate typeExpr into a type.
    (void)typeExpr;
    resultTypes.push_back(IndexType::get(getContext()));
  }

  if (parseToken(LitToken::colon, "expected ':' in function definition"))
    return failure();

  auto builder = currentScope->getBuilder();
  auto functionType = builder.getFunctionType(valueParamTypes, resultTypes);
  auto linkage = builder.getAttr<LinkageAttr>(Linkage::Public);

  // TODO: Should have nicer builder.
  auto newFunc = builder.create<LITFuncOp>(
      loc, funcNameAttr, StringArrayAttr::get(getContext(), valueParamNames),
      TypeAttr::get(functionType), linkage,
      ParamDeclArrayAttr::get(getContext(), metaSignature.inputParameters),
      TypeArrayAttr::get(getContext(), {}),
      ConstraintArrayAttr::get(getContext(), {}), FlatSymbolRefAttr());
  auto bodyBlock = new Block();
  bodyBlock->addArguments(valueParamTypes, valueParamLocs);
  newFunc.getRegion().push_back(bodyBlock);

  auto newFuncRefAttr = SymbolConstantAttr::get(
      FlatSymbolRefAttr::get(funcNameAttr), newFunc.getSignature());

  currentScope->addToScope(funcNameAttr,
                           Scope::MetaParameterValue{newFuncRefAttr, loc},
                           sharedParserState->errorOccurred);

  // We cannot parse the current body without having parsed other declarations
  // at the current level, so we defer parsing it.  Remember that we need to
  // do so.
  deferredDecls.push_back({RCRef<Scope>::create(newFunc, currentScope.copy()),
                           lexer.getCursor(), curIndent,
                           std::move(metaSignature.inputParamLocs)});

  // Skip the body of this definition: go to a token the starts a line at the
  // same indent level (or less) as the current definition.
  skipUntilIndentation(curIndent);
  return success();
}

/// Parse a deferred 'def' body.
void LitParser::parseDefBody(LITFuncOp defDecl, size_t defIndent,
                             ArrayRef<Location> inputParamLocs) {
  // Add the meta parameters to the symbol table.
  for (auto [param, loc] : llvm::zip(defDecl.getParamDecls(), inputParamLocs)) {
    auto value = ParamDeclRefAttr::get(param.getName(), param.getType());
    currentScope->addToScope(param.getName(),
                             Scope::MetaParameterValue{value, loc},
                             sharedParserState->errorOccurred);
  }

  // Set up the body of the def, creating declarations for the value parameters
  // and adding them to the symbol table.
  auto builder = currentScope->getBuilder();
  for (auto [arg, name] : llvm::zip(defDecl.getBody()->getArguments(),
                                    defDecl.getValueParamNames())) {
    // Create a mutable var.decl that references to the name can load from.
    // TODO: This is the wrong default, reconsider this for 'fn's when we have
    // a notion of immutability.
    auto type = POP::PointerType::get(arg.getType());
    auto varDecl = builder.create<VarDeclOp>(arg.getLoc(), type, name);
    currentScope->addToScope(name, varDecl, sharedParserState->errorOccurred);
    builder.create<POP::StoreOp>(arg.getLoc(), arg, varDecl,
                                 /*alignment*/ None);
  }

  (void)parseSuite(defIndent, StmtContext::normal);

  // Check to see if we have a kgen.return at the end of function.  If not,
  // complain or add one implicitly if we have no results.
  Block *bodyBlock = defDecl.getBody();
  if (bodyBlock->empty() || !isa<ReturnOp>(bodyBlock->back())) {
    if (defDecl.getResultTypes().empty() &&
        defDecl.getResultParamTypes().empty()) {
      // TODO: Generalize lit.func.
      auto returnParams = ArrayAttr::get(getContext(), {});
      OpBuilder::atBlockEnd(bodyBlock).create<ReturnOp>(
          defDecl->getLoc(), returnParams, ArrayRef<Value>());
    } else if (!sharedParserState->errorOccurred) {
      Location endLoc =
          bodyBlock->empty() ? defDecl.getLoc() : bodyBlock->back().getLoc();
      emitError(endLoc, "return expected at end of 'def' with results");
    }
  }

  finalizeScopeDecl();
}

/// return_stmt ::= "return" [expression_list]
ParseResult LitParser::parseReturnStmt() {
  auto loc = consumeToken(LitToken::kw_return).getLoc();

  SmallVector<Value> operandValues;

  // If there is an expression list present, parse it.
  if (!getToken().getIndentation().has_value()) {
    ExprParser exprParser(*this, currentScope);
    SmallVector<ExprNode *> operandExprs;
    exprParser.parseExpressionList(operandExprs);

    // Materialize the expression values into our current scope.
    // TODO: Should pass in contextual type from return value.
    for (auto expr : operandExprs) {
      auto value = exprParser.emitAsValue(expr);
      if (!value)
        return failure();
      operandValues.push_back(value);
    }
  }

  // We don't support formation of tuples / multiple result values yet.
  if (operandValues.size() > 1) {
    emitError(loc, "tuple return not supported yet");
    return success();
  }

  // Check the result values match expected types.
  LITFuncOp decl = dyn_cast<LITFuncOp>(currentScope->getDecl());
  if (!decl) {
    emitError(loc, "cannot return from this context");
    return success();
  }

  if (operandValues.empty() && !decl.getResultTypes().empty()) {
    emitError(loc, "expected a return value from 'def' with return type ")
        << decl.getResultTypes()[0];
    return success();
  }

  if (operandValues.size() == 1 && decl.getResultTypes().empty()) {
    emitError(loc, "extraneous return value from 'def'");
    return success();
  }

  if (operandValues[0].getType() != decl.getResultTypes()[0]) {
    emitError(loc, "returned value has type ")
        << operandValues[0].getType() << " but 'def' expected "
        << decl.getResultTypes()[0];
    return success();
  }

  // TODO: Support result parameters.
  auto returnParams = ArrayAttr::get(getContext(), {});
  currentScope->getBuilder().create<ReturnOp>(translateLocation(loc),
                                              returnParams, operandValues);
  return success();
}

// FIXME(https://reviews.llvm.org/D135940): This is a clone of
// llvm::SaveAndRestore that is updated to work with non-copyable values. Remove
// this when fixed upstream.
namespace {
/// A utility class that uses RAII to save and restore the value of a variable.
template <typename T>
struct SaveAndRestore {
  SaveAndRestore(T &X) : X(X), OldValue(X) {}
  SaveAndRestore(T &X, const T &NewValue) : X(X), OldValue(X) { X = NewValue; }
  SaveAndRestore(T &X, T &&NewValue) : X(X), OldValue(std::move(X)) {
    X = std::move(NewValue);
  }
  ~SaveAndRestore() { X = std::move(OldValue); }
  const T &get() { return OldValue; }

private:
  T &X;
  T OldValue;
};

} // namespace

/// structdef ::=
///   [decorators] "struct" identifier [meta_signature] ":" suite
///
ParseResult LitParser::parseStructStmt(size_t curIndent) {
  auto loc = getTokenLocation();

  // TODO: Add support for decorators.
  consumeToken(LitToken::kw_struct);

  StringAttr nameAttr;
  MetaSignatureParser metaSignature;
  if (parseIdentifier(nameAttr, "expected struct name") ||
      metaSignature.parseOptionalMetaSignature(*this) ||
      parseToken(LitToken::colon, "expected ':' in function definition"))
    return failure();

  auto builder = currentScope->getBuilder();
  // TODO: Should have nicer builder.
  auto newStruct = builder.create<StructDeclOp>(
      loc, nameAttr,
      ParamDeclArrayAttr::get(getContext(), metaSignature.inputParameters),
      TypeArrayAttr::get(getContext(), {}),
      ConstraintArrayAttr::get(getContext(), {}));
  newStruct.getRegion().push_back(new Block());

  auto newRefAttr = SymbolConstantAttr::get(FlatSymbolRefAttr::get(nameAttr),
                                            builder.getType<MLIRTypeType>());

  currentScope->addToScope(nameAttr, Scope::MetaParameterValue{newRefAttr, loc},
                           sharedParserState->errorOccurred);

  // Switch to the struct's scope to parse things into it.
  SaveAndRestore<RCRef<Scope>> scopeSaver(
      currentScope, RCRef<Scope>::create(newStruct, currentScope.copy()));

  // Add the meta parameters to the symbol table.
  for (auto [param, loc] :
       llvm::zip(newStruct.getParamDecls(), metaSignature.inputParamLocs)) {
    auto value = ParamDeclRefAttr::get(param.getName(), param.getType());
    currentScope->addToScope(param.getName(),
                             Scope::MetaParameterValue{value, loc},
                             sharedParserState->errorOccurred);
  }

  (void)parseSuite(curIndent, StmtContext::structBody);

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

  context->loadDialect<POP::POPDialect, LITDialect, index::IndexDialect,
                       KGENDialect>();

  // This is the result module we are parsing into.
  mlir::OwningOpRef<ModuleOp> module(ModuleOp::create(
      FileLineColLoc::get(context, sourceBuf->getBufferIdentifier(), /*line=*/0,
                          /*column=*/0)));

  SharedParserState sharedState(context);

  // Parse the file.
  LitLexer lexer(sourceMgr, context);
  if (LitParser(lexer, &sharedState, *module).parseFile())
    return nullptr;

  // Make sure the parse module has no other structural problems detected by
  // the verifier.
  auto verificationTimer = ts.nest("Verify module");
  if (failed(verify(*module)))
    return {};
  return module;
}
