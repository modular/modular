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
#include "LitParserBase.h"

#include "LitExprNodes.h"
#include "LitScope.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/IndexDialect/IndexAttrs.h"
#include "Support/IndexDialect/IndexDialect.h"
#include "Support/IndexDialect/IndexOps.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/Timing.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace M;
using namespace M::LLCL;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

using llvm::SourceMgr;
namespace scf = mlir::scf;

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

//===----------------------------------------------------------------------===//
// NameBindingContext
//===----------------------------------------------------------------------===//

/// This class is used to perform name binding for a scope after all the
/// declarations within it have been parsed.
class NameBindingContext {
public:
  NameBindingContext(LitParserBase &parser, Scope &scope)
      : parser(parser), scope(scope), declCursors(scope.takeDeclCursors()),
        declsWithExprsToNameBind(scope.takeDeclsWithExprsToNameBind()) {}

  void doNameBinding();

private:
  void ensureOpIsNameBound(Operation *op);

  void nameBind(VarDeclOp op, LitLexerCursor cursor);
  void nameBind(LITStructDeclOp op, LitLexerCursor cursor);

  /// Given a cursor location for a type expression that correctly parsed in the
  /// first pass, reparse it into an expression and resolve it into a type by
  /// performing name lookup and other resolution.  This can produce errors, but
  /// always returns a non-null type.
  Type resolveType(LitLexerCursor cursor);

private:
  LitParserBase &parser;
  Scope &scope;

  /// This records where (lexically) a declaration is that has types that need
  /// to be reparsed.  This allows us to do name binding of types in an
  /// on-demand order, necessary for resolving inter-dependencies between
  /// declarations.
  DenseMap<Operation *, LitLexerCursor> declCursors;

  /// This is a list of operations that have deferred expressions to name bind
  /// and type check in the second pass of parsing.
  std::vector<Operation *> declsWithExprsToNameBind;

  /// Name binding is an recursive process in the general case.  This keeps
  /// track of the declarations currently being name bound so we can diagnose
  /// cyclic dependencies.
  DenseSet<Operation *> declsCurrentlyProcessing;
};

void NameBindingContext::doNameBinding() {
  // Name binding can recursively visit entries that are transitively referenced
  // by other declarations, but we need to make sure that each declaration is
  // visited.  We handle this by using the declsWithExprsToNameBind list as the
  // top level to visit (which also gives us lexical order of top-level
  // visitation, nice for diagnostics coming out in a logical order) but remove
  // declarations from `declCursors` when they are processed.
  for (Operation *op : declsWithExprsToNameBind)
    ensureOpIsNameBound(op);
}

void NameBindingContext::ensureOpIsNameBound(Operation *op) {
  // Find the cursor for this operation, if we don't know any, it is already
  // done.
  auto cursorIt = declCursors.find(op);
  if (cursorIt == declCursors.end())
    return;

  // If we are currently name binding this operation, we found a cycle, reject
  // it with an error.
  if (!declsCurrentlyProcessing.insert(op).second) {
    assert(0 &&
           "FIXME: Diagnose cyclic reference when it is possible to happen");
  }

  // Handle each operation that can be name bound.
  TypeSwitch<Operation *>(op)
      .Case<LITStructDeclOp, VarDeclOp>(
          [&](auto op) { nameBind(op, cursorIt->second); })
      .Default([&](auto attr) {
        op->emitError("do not know how to perform name binding on this op!");
      });

  declsCurrentlyProcessing.erase(op);
}

/// Given a cursor location for a type expression that correctly parsed in the
/// first pass, reparse it into an expression and resolve it into a type by
/// performing name lookup and other resolution.  This can produce errors, but
/// always returns a non-null type.
Type NameBindingContext::resolveType(LitLexerCursor cursor) {
  // Move the cursor to the specified location.
  cursor.restore(parser.getLexer());
  // Re-parse the expression at that location.
  ExprParser exprParser(parser);
  ExprNode *typeExpr = exprParser.parseExpression();
  assert(typeExpr && "We know expr parsing will work");

  auto emitError = [&](const Twine &message) -> Type {
    parser.emitError(typeExpr->getLoc(), message);
    return UnresolvedType::get(parser.getContext());
  };

  // TODO: Make this a recursive walk when we have more interesting types.
  if (auto dre = dyn_cast<DeclRefNode>(typeExpr)) {
    // TODO(types): This is a hack to unblock tests in the interim.
    if (dre->spelling == "index")
      return IndexType::get(parser.getContext());

    // Lookup the identifier.
    Optional<Scope::ScopeValue> lookup = scope.lookup(dre->spelling);
    if (!lookup)
      return emitError("unknown type name '" + dre->spelling + "'");
    if (std::holds_alternative<VarDeclOp>(*lookup))
      return emitError("'" + dre->spelling + "' names a value, not a type");
    auto attr = dyn_cast<SymbolConstantAttr>(
        std::get<Scope::MetaParameterValue>(*lookup).getAttr());
    if (!attr || !isa<MLIRTypeType>(attr.getType()))
      return emitError("'" + dre->spelling + "' names a value, not a type");

    // TODO: Handle type parameters!
    return RefType::get(attr.getSymbol(),
                        ParamBindArrayAttr::get(parser.getContext(), {}));
  }

  return emitError("FIXME: Unsupported type kind!");
}

//===----------------------------------------------------------------------===//
// LitParser
//===----------------------------------------------------------------------===//

namespace {
/// Declaration bodies are parsed after all the signatures at the current
/// level of the file are parsed.  This keeps track of
struct DeferredDeclBodyToParse {
  /// This is the scope for the declaration, which also contains the
  /// declaration itself.
  RCRef<Scope> declScope;

  /// This is where to start lexing the body from.
  LitLexerCursor lexerCursor;

  /// This is the indentation level of the decl.
  size_t indentLevel;

  /// This is the location of each input parameter.
  std::vector<Location> inputParamLocs;
};
} // namespace

/// This class provides the implementation details of the concrete Lit Grammar.
struct LitParser : public LitParserBase {
  LitParser(LitLexer &lexer, SharedParserState *sharedParserState)
      : LitParserBase(lexer, sharedParserState) {}

  ParseResult parseFile(ModuleOp module);

  void finalizeScopeDecl();
  void performNameBinding();

  const RCRef<Scope> &getCurrentScope() const { return currentScope; }

  // Expressions.
  // TODO: Move expression emission elsewhere!

  /// Emit the specified expression tree to MLIR in the current context.
  MLIRValueRep emitExpr(ExprNode *node) {
    EmitterState state(*this, *currentScope);
    return node->emit(state);
  }

  Value emitExprAsValue(ExprNode *node) {
    EmitterState state(*this, *currentScope);
    return node->emit(state).getAsValue(translateLocation(node->getLoc()),
                                        state.builder);
  }

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
  ParseResult parseIfStmt(size_t curIndent);
  ParseResult parseDefStmt(size_t curIndent);
  void parseDefBody(LITFuncOp defDecl, size_t defIndent,
                    ArrayRef<Location> inputParamLocs);
  ParseResult parseStructStmt(size_t curIndent);

  // Simple statements.
  ParseResult parseVarDeclStmt();
  ParseResult parseReturnStmt();
  ParseResult parseAssignmentStmt(ExprParser &exprParser, ExprNode *lhs,
                                  SMLoc equalsLoc);

private:
  /// This is the current context that we're parsing into.
  RCRef<Scope> currentScope;

  /// These are deferred declarations that need parsing, which are processed
  /// after other things in a scope have been resolved.
  std::vector<DeferredDeclBodyToParse> deferredDecls;
};

/// file ::= statements
ParseResult LitParser::parseFile(ModuleOp module) {
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
/// Once its body is fully parsed, we loop back around to parse the bodies
/// of any nested scopes (e.g. nested functions) that are encountered while
/// parsing this scope.  This ensures that the forward references between
/// peer declarations are handled correctly, for example in mutually
/// recursive functions and code like this:
///
///   def foo():
///     def bar():
///       print(x)
///     x = 42
///     bar()
///   foo()
void LitParser::finalizeScopeDecl() {
  // If we have any expressions that need second pass name binding, do it now.
  NameBindingContext(*this, *currentScope).doNameBinding();

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
  if (auto indent = getToken().getIndentation()) {
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
/// compound_stmt ::= if_stmt
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
  case LitToken::kw_if:
    return parseIfStmt(curIndent);
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
///               | var_decl_stmt
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
  case LitToken::kw_if:
  case LitToken::kw_def:
  case LitToken::kw_struct:
    emitError() << "'" << getToken().getSpelling()
                << "' statement must be on its own line";
    return parseStmt(0, stmtContext);

  case LitToken::kw_pass:
    // pass_stmt ::= "pass"
    consumeToken(LitToken::kw_pass);
    return success();
  case LitToken::kw_var:
    return parseVarDeclStmt();
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
  ExprParser exprParser(*this);
  ExprNode *expr = exprParser.parseExpression();

  // If the expression was followed by a `=` then we have an assignment.  If not
  // then we have an expression_stmt.
  SMLoc equalsLoc;
  if (consumeIf(LitToken::equal, &equalsLoc))
    return parseAssignmentStmt(exprParser, expr, equalsLoc);

  // Materialize the expression statement in our current scope but discard the
  // result on the floor.
  (void)emitExpr(expr);
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
  // Use this builder to place any VarDeclOps. In Python there is only one
  // scope per function and all variables belong to that scope, so builders
  // should reflect that.
  auto funcBodyBuilder = currentScope->getDeclBuilder();

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

    // TODO(types): Add types instead of hard coding to index type!
    auto declType = POP::PointerType::get(builder.getIndexType());

    // TODO: This will emit it on first use, which can be in weird places (e.g.
    // inside of the branch of an if statement).  We want to use dataflow
    // analysis to do definitive analysis of the accesses to the declaration. We
    // could just emit all these in the entry to the enclosing function/module
    // to maintain SSA.
    auto varDecl = funcBodyBuilder.create<VarDeclOp>(
        translateLocation(lhs->getLoc()), declType,
        funcBodyBuilder.getStringAttr(dre->spelling));
    currentScope->addToScope(dre->spelling, varDecl,
                             sharedParserState->errorOccurred);
    lvalue = varDecl;
  }

  ExprNode *rhs = exprParser.parseExpression();

  // Materialize the expression statement in our current scope.
  // TODO: Should pass in contextual type if known from previous declaration.
  auto rhsValue = emitExprAsValue(rhs);

  // If IR generation failed, return success since we have a fine parse.
  if (!lvalue || !rhsValue)
    return success();

  // TODO(types): this is incorrect for index types, need to coerce to the
  // destination type.
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
struct ParsedMetaSignature {
  SmallVector<ParamDeclAttr> inputParameters;
  std::vector<Location> inputParamLocs;

  ParseResult parseOptionalMetaSignature(LitParserBase &p) {
    if (!p.consumeIf(LitToken::l_square) || p.consumeIf(LitToken::r_square))
      return success();

    auto parseMetaParameter = [&]() -> ParseResult {
      inputParamLocs.push_back(p.getTokenLocation());

      StringAttr name;
      if (p.parseIdentifier(name, "expected parameter name")) {
        // TODO: Scan ahead for better recovery.
        return failure();
      }

      Type paramType = IndexType::get(p.getContext());
      if (p.consumeIf(LitToken::colon)) {
        ExprParser exprParser(p);
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

/// if_stmt ::=  "if" assignment_expression ":" suite
///             ("elif" assignment_expression ":" suite)*
///             ["else" ":" suite]
ParseResult LitParser::parseIfStmt(size_t curIndent) {
  OpBuilder &builder = currentScope->getBuilder();
  Location ifLoc = translateLocation(consumeToken(LitToken::kw_if).getLoc());
  auto one = builder.create<index::ConstantOp>(ifLoc, 1);

  // Parse the condition expression of the If statement and create a comparison
  // with the current builder.
  // The caller should be sure to have the correct builder in currentScope to
  // build the conditional expression in the desired place.
  auto parseCondition = [&](index::CmpOp &cmpOp) -> ParseResult {
    // TODO: add type checking: the condition should be bool
    ExprParser exprParser(*this);
    ExprNode *condExp = exprParser.parseExpression();
    Location loc = translateLocation(condExp->getLoc());
    Value cond = emitExprAsValue(condExp);
    if (!cond)
      return failure();
    auto cmpBuilder = currentScope->getBuilder();
    cmpOp = cmpBuilder.create<index::CmpOp>(loc, index::IndexCmpPredicate::EQ,
                                            cond, one);
    if (parseToken(LitToken::colon, "expected ':' after expression"))
      return failure();
    return success();
  };

  index::CmpOp cmp;
  if (failed(parseCondition(cmp)))
    return failure();

  auto ifOp = builder.create<scf::IfOp>(ifLoc, cmp, /*withElse=*/true);
  {
    SaveAndRestore<OpBuilder> builderSaver(builder, ifOp.getThenBodyBuilder());
    if (failed(parseSuite(curIndent, StmtContext::normal)))
      return failure();
  }
  scf::IfOp lastIfOp = ifOp;
  while (getToken().is(LitToken::kw_elif)) {
    Location elifLoc =
        translateLocation(consumeToken(LitToken::kw_elif).getLoc());
    auto elseBuilder = lastIfOp.getElseBodyBuilder();
    index::CmpOp elifCmp;
    {
      SaveAndRestore<OpBuilder> builderSaver(currentScope->getBuilder(),
                                             elseBuilder);
      if (failed(parseCondition(elifCmp)))
        return failure();
    }
    lastIfOp =
        elseBuilder.create<scf::IfOp>(elifLoc, elifCmp, /*withElse=*/true);
    SaveAndRestore<OpBuilder> builderSaver(currentScope->getBuilder(),
                                           lastIfOp.getThenBodyBuilder());
    if (failed(parseSuite(curIndent, StmtContext::normal)))
      return failure();
  }
  if (getToken().is(LitToken::kw_else)) {
    consumeToken(LitToken::kw_else);
    if (parseToken(LitToken::colon, "expected ':' after else"))
      return failure();
    SaveAndRestore<OpBuilder> builderSaver(currentScope->getBuilder(),
                                           lastIfOp.getElseBodyBuilder());
    if (failed(parseSuite(curIndent, StmtContext::normal)))
      return failure();
  }
  return success();
}

namespace {
struct ParsedParam {
  /// If this parameter has a type specifier or default value, these indicates
  /// where the expressions may be re-parsed from.
  SMLoc loc;
  StringAttr name;
  Optional<LitLexerCursor> typeCursor;
  Optional<LitLexerCursor> initValueCursor;

  // TODO: Implement support for variadic parameter markers:
  // Python's parameter grammar embeds checking for `/` and `*` and `**` into
  // the grammar, we can just check for it using ad-hoc logic for simplicity,
  // according to the following rules:
  //   1) Only one /, *, and ** parameter may exist in the parameter list.
  //   2) They are specified in that order.
  //   3) These do not permit default arguments.
  ParseResult parse(LitParserBase &p) {
    loc = p.getToken().getLoc();

    if (p.parseIdentifier(name, "expected parameter name"))
      // TODO: Scan ahead for better recovery.
      return failure();

    if (p.consumeIf(LitToken::colon)) {
      if (ExprParser::parseOverExpression(p, typeCursor))
        return failure();
    }
    if (p.consumeIf(LitToken::equal)) {
      if (ExprParser::parseOverExpression(p, initValueCursor))
        return failure();
    }
    return success();
  };
};
} // namespace

namespace {
/// funcdef ::=  [decorators] "def" identifier [meta_signature]
///              "(" [value_param_list] ")" ["->" expression] ":" suite
///
/// value_param_list  ::= value_parameter ("," value_parameter)*
/// value_parameter   ::= value_parammarker identifier_opt_type ["=" expression]
/// value_parammarker ::= "/" | "*" | "**"
///
struct ParsedDefSignature {
  StringAttr name;
  ParsedMetaSignature metaSignature;
  SmallVector<ParsedParam> params;
  Optional<LitLexerCursor> resultTypeCursor;

  ParseResult parse(LitParserBase &p) {
    // TODO: Add support for decorators.
    if (p.parseToken(LitToken::kw_def, "expected 'def' declaration") ||
        p.parseIdentifier(name, "expected function name") ||
        metaSignature.parseOptionalMetaSignature(p) ||
        p.parseToken(LitToken::l_paren, "expected '(' for parameter list"))
      return failure();

    if (!p.consumeIf(LitToken::r_paren)) {
      if (p.parseCommaSeparatedList(
              [&]() { return params.emplace_back(ParsedParam()).parse(p); }) ||
          p.parseToken(LitToken::r_paren, "expected ')' for parameter list"))
        return failure();
    }

    // Parse the result type if present.

    // TODO: This will be one difference between a def and fn: no result type on
    // a def should default to returning a (default initialized) Object, whereas
    // a fn can return void.  We can provide a guaranteed optimization to remove
    // it though.
    if (p.consumeIf(LitToken::minus_greater)) {
      if (ExprParser::parseOverExpression(p, resultTypeCursor))
        return failure();
    }

    return p.parseToken(LitToken::colon, "expected ':' in function definition");
  }
};
} // namespace

ParseResult LitParser::parseDefStmt(size_t curIndent) {
  Location loc = getTokenLocation();
  ParsedDefSignature info;
  if (info.parse(*this))
    return failure();

  // We have parsed the signature but skipped over the actual types, we use
  // unresolved types for now.
  SmallVector<Location> paramLocs;
  SmallVector<StringAttr> paramNames;
  // TODO(types): Replace index with unresolved types here.
  SmallVector<Type> paramTypes(info.params.size(),
                               IndexType::get(getContext()));
  for (const auto &param : info.params) {
    paramLocs.push_back(translateLocation(param.loc));
    paramNames.push_back(param.name);
    // TODO: add support for default parameter expressions.
    if (param.initValueCursor)
      emitError(param.initValueCursor->getLoc(getLexer()),
                "TODO: No default values yet");
  }

  SmallVector<Type> resultTypes;
  if (info.resultTypeCursor)
    resultTypes.push_back(IndexType::get(getContext()));

  auto builder = currentScope->getBuilder();
  auto functionType = builder.getFunctionType(paramTypes, resultTypes);
  auto linkage = builder.getAttr<LinkageAttr>(Linkage::Public);

  // TODO: Should have nicer builder.
  auto newFunc = builder.create<LITFuncOp>(
      loc, info.name, StringArrayAttr::get(getContext(), paramNames),
      TypeAttr::get(functionType), linkage,
      ParamDeclArrayAttr::get(getContext(), info.metaSignature.inputParameters),
      TypeArrayAttr::get(getContext(), {}),
      ConstraintArrayAttr::get(getContext(), {}), FlatSymbolRefAttr());
  auto bodyBlock = new Block();
  bodyBlock->addArguments(paramTypes, paramLocs);
  newFunc.getRegion().push_back(bodyBlock);

  auto newFuncRefAttr = SymbolConstantAttr::get(
      FlatSymbolRefAttr::get(info.name), newFunc.getSignature());

  currentScope->addToScope(info.name,
                           Scope::MetaParameterValue{newFuncRefAttr, loc},
                           sharedParserState->errorOccurred);

  // We cannot parse the current body without having parsed other declarations
  // at the current level, so we defer parsing it.  Remember that we need to
  // do so.
  deferredDecls.push_back({RCRef<Scope>::create(newFunc, currentScope.copy()),
                           lexer.getCursor(), curIndent,
                           std::move(info.metaSignature.inputParamLocs)});

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

namespace {
/// var_decl_stmt ::= "var" identifier ":" expression ["=" expression]
///                 | "var" identifier "=" expression [TODO]
///
struct ParsedVarDecl {
  SMLoc loc;
  StringAttr name;
  Optional<LitLexerCursor> typeCursor;
  Optional<LitLexerCursor> initValueCursor;

  ParseResult parse(LitParserBase &p) {
    loc = p.getToken().getLoc();
    if (p.parseToken(LitToken::kw_var, "expected 'var' declaration") ||
        p.parseIdentifier(name, "expected name for 'var' declaration") ||
        p.parseToken(LitToken::colon, "var declaration requires a type") ||
        ExprParser::parseOverExpression(p, typeCursor))
      return failure();

    if (p.consumeIf(LitToken::equal)) {
      if (ExprParser::parseOverExpression(p, initValueCursor))
        return failure();
    }
    return success();
  }
};
} // namespace

ParseResult LitParser::parseVarDeclStmt() {
  LitLexerCursor declCursor = getLexer().getCursor();
  ParsedVarDecl info;
  if (info.parse(*this))
    return failure();

  // TODO: add support for default parameter expressions.
  if (info.initValueCursor)
    emitError(info.loc, "var initializers not supported yet");

  auto builder = currentScope->getDeclBuilder();

  // If we are in a function, emit a variable declaration, if we are in a
  // struct, emit a field declaration.  Both have the same IR representation.
  auto varType = POP::PointerType::get(UnresolvedType::get(getContext()));
  auto varDecl = builder.create<VarDeclOp>(translateLocation(info.loc), varType,
                                           info.name);
  currentScope->addToScope(info.name, varDecl,
                           sharedParserState->errorOccurred);
  currentScope->addExprToNameBind(varDecl, declCursor);
  return success();
}

void NameBindingContext::nameBind(VarDeclOp op, LitLexerCursor cursor) {
  // Move the lexer to point to the start of the declaration so we can reparse.
  cursor.restore(parser.getLexer());

  // Reparse the var-decl signature again.  We know the initial parse succeeded.
  ParsedVarDecl info;
  (void)info.parse(parser);

  // Parse the type if present.
  if (info.typeCursor)
    op.getResult().setType(
        POP::PointerType::get(resolveType(info.typeCursor.value())));

  if (info.initValueCursor)
    emitError(info.initValueCursor->getLoc(parser.getLexer()),
              "var initializers not supported yet");
}

/// return_stmt ::= "return" [expression_list]
ParseResult LitParser::parseReturnStmt() {
  auto loc = consumeToken(LitToken::kw_return).getLoc();

  SmallVector<Value> operandValues;

  // If there is an expression list present, parse it.
  if (!getToken().getIndentation().has_value()) {
    ExprParser exprParser(*this);
    SmallVector<ExprNode *> operandExprs;
    exprParser.parseExpressionList(operandExprs);

    // Materialize the expression values into our current scope.
    // TODO: Should pass in contextual type from return value.
    for (auto expr : operandExprs) {
      auto value = emitExprAsValue(expr);
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

/// structdef ::=
///   [decorators] "struct" identifier [meta_signature] ":" suite
///
ParseResult LitParser::parseStructStmt(size_t curIndent) {
  auto loc = getTokenLocation();

  // TODO: Add support for decorators.
  consumeToken(LitToken::kw_struct);

  StringAttr nameAttr;
  ParsedMetaSignature metaSignature;
  if (parseIdentifier(nameAttr, "expected struct name") ||
      metaSignature.parseOptionalMetaSignature(*this) ||
      parseToken(LitToken::colon, "expected ':' in function definition"))
    return failure();

  auto builder = currentScope->getBuilder();
  // TODO: Should have nicer builder.
  auto newStruct = builder.create<LITStructDeclOp>(
      loc, nameAttr,
      ParamDeclArrayAttr::get(getContext(), metaSignature.inputParameters),
      TypeArrayAttr::get(getContext(), {}));
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

  // If we have any expressions that need second pass name binding, do it now.
  NameBindingContext(*this, *currentScope).doNameBinding();
  return success();
}

void NameBindingContext::nameBind(LITStructDeclOp op, LitLexerCursor cursor) {
  op->emitError("TODO: name binding not implemented yet");
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
                       KGENDialect, scf::SCFDialect>();

  // This is the result module we are parsing into.
  mlir::OwningOpRef<ModuleOp> module(ModuleOp::create(
      FileLineColLoc::get(context, sourceBuf->getBufferIdentifier(), /*line=*/0,
                          /*column=*/0)));

  SharedParserState sharedState(context);

  // Parse the file.
  LitLexer lexer(sourceMgr, context);
  if (LitParser(lexer, &sharedState).parseFile(*module))
    return nullptr;

  // Make sure the parse module has no other structural problems detected by
  // the verifier.
  auto verificationTimer = ts.nest("Verify module");
  if (failed(verify(*module)))
    return {};
  return module;
}
