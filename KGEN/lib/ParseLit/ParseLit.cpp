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

#include "LitDecls.h"
#include "LitExprNodes.h"
#include "LitScope.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "LitSharedState.h"
#include "Support/IndexDialect/IndexAttrs.h"
#include "Support/IndexDialect/IndexDialect.h"
#include "Support/IndexDialect/IndexOps.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/Timing.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace M;
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
void Scope::addToScope(StringRef name, ScopeValue newValue,
                       LitSharedState &sharedState) {
  Optional<Scope::ScopeValue> &entry = decls[name];
  if (!entry) {
    entry = newValue;
    return;
  }

  auto diag = emitError(getLocationFrom(newValue), "invalid redefinition of \"")
              << name << '"';
  diag.attachNote(getLocationFrom(entry.value())) << "previous definition here";
  sharedState.errorOccurred = true;

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
// LitParser
//===----------------------------------------------------------------------===//

/// This class provides the implementation details of the concrete Lightning
/// grammar.
struct LitParser : public LitParserBase {
  LitParser(LitLexer &lexer, Scope &scope)
      : LitParserBase(lexer), scope(scope), builder(scope.getDeclBuilder()) {}

  ParseResult parseFile(ModuleOp module);

  const Scope &getScope() const { return scope; }

  // Expressions.
  // TODO: Move expression emission elsewhere!

  /// Emit the specified expression tree to MLIR in the current context.
  MLIRValueRep emitExpr(ExprNode *node) {
    EmitterState state(*this, scope, builder);
    return node->emit(state);
  }

  Value emitExprAsValue(ExprNode *node) {
    EmitterState state(*this, scope, builder);
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
  void parseDefBody(LITFuncOp defDecl);
  ParseResult parseStructStmt(size_t curIndent);
  void parseStructBody(LITStructDeclOp structDecl);

  // Simple statements.
  ParseResult parseVarDeclStmt();
  ParseResult parseReturnStmt();
  ParseResult parseAssignmentStmt(ExprParser &exprParser, ExprNode *lhs,
                                  SMLoc equalsLoc);

private:
  /// This is declaration scope that we're parsing into.
  Scope &scope;

  /// This is the builder that we are constructing IR into.
  OpBuilder builder;
};

/// file ::= statements
ParseResult LitParser::parseFile(ModuleOp module) {
  // We fail either if we have a non-recoverable parse error, or if we emitted
  // an error and then recovered.  In either case, the IR will not be valid and
  // the caller should not verify it.
  if (parseStmts(/*indent=*/0, StmtContext::normal))
    return failure();

  if (getSharedState().errorOccurred)
    return failure();

  return success();
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

  // Use this builder to place any VarDeclOps. In Python there is only one
  // scope per function and all variables belong to that scope, so builders
  // should reflect that.
  auto funcBodyBuilder = scope.getDeclBuilder();

  // Look up the name being assigned to if it already exists.
  Value lvalue;
  if (Optional<Scope::ScopeValue> decl =
          scope.lookupInCurrentScope(dre->spelling)) {
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
    scope.addToScope(dre->spelling, varDecl, getSharedState());
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
  SmallVector<ParamDeclAttr> inputDecls;
  std::vector<Location> inputLocs;

  ParseResult parseOptionalMetaSignature(LitParserBase &p) {
    if (!p.consumeIf(LitToken::l_square) || p.consumeIf(LitToken::r_square))
      return success();

    auto parseMetaParameter = [&]() -> ParseResult {
      inputLocs.push_back(p.getTokenLocation());

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
      inputDecls.push_back(ParamDeclAttr::get(name, paramType));
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
  Location ifLoc = translateLocation(consumeToken(LitToken::kw_if).getLoc());
  auto one = builder.create<index::ConstantOp>(ifLoc, 1);

  // Parse the condition expression of the If statement and create a comparison
  // with the current builder.
  // The caller should be sure to have the correct builder in scope to
  // build the conditional expression in the desired place.
  auto parseCondition = [&](index::CmpOp &cmpOp) -> ParseResult {
    // TODO: add type checking: the condition should be bool
    ExprParser exprParser(*this);
    ExprNode *condExp = exprParser.parseExpression();
    Location loc = translateLocation(condExp->getLoc());
    Value cond = emitExprAsValue(condExp);
    if (!cond)
      return failure();
    cmpOp = builder.create<index::CmpOp>(loc, index::IndexCmpPredicate::EQ,
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
      SaveAndRestore<OpBuilder> builderSaver(builder, elseBuilder);
      if (failed(parseCondition(elifCmp)))
        return failure();
    }
    lastIfOp =
        elseBuilder.create<scf::IfOp>(elifLoc, elifCmp, /*withElse=*/true);
    SaveAndRestore<OpBuilder> builderSaver(builder,
                                           lastIfOp.getThenBodyBuilder());
    if (failed(parseSuite(curIndent, StmtContext::normal)))
      return failure();
  }
  if (getToken().is(LitToken::kw_else)) {
    consumeToken(LitToken::kw_else);
    if (parseToken(LitToken::colon, "expected ':' after else"))
      return failure();
    SaveAndRestore<OpBuilder> builderSaver(builder,
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
  LitLexerCursor declCursor = getLexer().getCursor();
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

  auto functionType = builder.getFunctionType(paramTypes, resultTypes);
  auto linkage = builder.getAttr<LinkageAttr>(Linkage::Public);

  // TODO: Should have nicer builder.
  auto funcDecl = builder.create<LITFuncOp>(
      loc, info.name, StringArrayAttr::get(getContext(), paramNames),
      TypeAttr::get(functionType), linkage,
      ParamDeclArrayAttr::get(getContext(), info.metaSignature.inputDecls),
      TypeArrayAttr::get(getContext(), {}),
      ConstraintArrayAttr::get(getContext(), {}), FlatSymbolRefAttr());
  auto bodyBlock = new Block();
  bodyBlock->addArguments(paramTypes, paramLocs);
  funcDecl.getRegion().push_back(bodyBlock);

  auto funcDeclRefAttr = SymbolConstantAttr::get(
      FlatSymbolRefAttr::get(info.name), funcDecl.getSignature());

  scope.addToScope(info.name, Scope::MetaParameterValue{funcDeclRefAttr, loc},
                   getSharedState());

  // We cannot parse the current body without having parsed other declarations
  // at the current level, so we defer parsing it.  Remember that we need to
  // do so.
  getDeclResolver().addDecl(funcDecl, &scope, declCursor);

  // Skip the body of this definition: go to a token the starts a line at the
  // same indent level (or less) as the current definition.
  skipUntilIndentation(curIndent);
  return success();
}

/// Parse a deferred 'def' body.
void LitParser::parseDefBody(LITFuncOp defDecl) {
  size_t defIndent = getToken().getIndentation().value_or(0);

  ParsedDefSignature info;
  (void)info.parse(*this); // We know this will succeed parsing.

  // Add the meta parameters to the symbol table.
  for (auto [param, loc] :
       llvm::zip(defDecl.getParamDecls(), info.metaSignature.inputLocs)) {
    auto value = ParamDeclRefAttr::get(param.getName(), param.getType());
    scope.addToScope(param.getName(), Scope::MetaParameterValue{value, loc},
                     getSharedState());
  }

  // Set up the body of the def, creating declarations for the value parameters
  // and adding them to the symbol table.
  for (auto [arg, name] : llvm::zip(defDecl.getBody()->getArguments(),
                                    defDecl.getValueParamNames())) {
    // Create a mutable var.decl that references to the name can load from.
    // TODO: This is the wrong default, reconsider this for 'fn's when we have
    // a notion of immutability.
    auto type = POP::PointerType::get(arg.getType());
    auto varDecl = builder.create<VarDeclOp>(arg.getLoc(), type, name);
    scope.addToScope(name, varDecl, getSharedState());
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
      OpBuilder::atBlockEnd(bodyBlock).create<ReturnOp>(
          defDecl->getLoc(), ArrayRef<TypedAttr>(), ArrayRef<Value>());
    } else if (!getSharedState().errorOccurred) {
      Location endLoc =
          bodyBlock->empty() ? defDecl.getLoc() : bodyBlock->back().getLoc();
      emitError(endLoc, "return expected at end of 'def' with results");
    }
  }
}

void DeclResolver::resolveBody(LITFuncOp op, Scope &scope) {
  LitLexer lexer(sharedState, scope.getCursor());
  LitParser parser(lexer, scope);
  parser.parseDefBody(op);
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

  auto builder = scope.getDeclBuilder();

  // If we are in a function, emit a variable declaration, if we are in a
  // struct, emit a field declaration.  Both have the same IR representation.
  auto varType = POP::PointerType::get(UnresolvedType::get(getContext()));
  auto varDecl = builder.create<VarDeclOp>(translateLocation(info.loc), varType,
                                           info.name);
  scope.addToScope(info.name, varDecl, getSharedState());

  // Remember that we parsed this declaration so we can finish type checking it
  // when it gets referenced.
  getDeclResolver().addDecl(varDecl, &scope, declCursor);
  return success();
}

void DeclResolver::resolveSignature(VarDeclOp op, Scope &scope) {
  // Set up a lexer to point to the start of the declaration so we can reparse.
  LitLexer lexer(sharedState, scope.getCursor());
  LitParser parser(lexer, scope);

  // Reparse the var-decl signature again.  We know the initial parse succeeded.
  ParsedVarDecl info;
  (void)info.parse(parser);

  // Parse the type if present.
  if (info.typeCursor)
    op.getResult().setType(POP::PointerType::get(
        resolveType(info.typeCursor.value(), scope, parser)));

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
  LITFuncOp decl = dyn_cast<LITFuncOp>(scope.getDecl());
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
  builder.create<ReturnOp>(translateLocation(loc), ArrayRef<TypedAttr>(),
                           operandValues);
  return success();
}

/// structdef ::=
///   [decorators] "struct" identifier [meta_signature] ":" suite
///
ParseResult LitParser::parseStructStmt(size_t curIndent) {
  auto loc = getTokenLocation();
  auto declCursor = getLexer().getCursor();

  // TODO: Add support for decorators.
  consumeToken(LitToken::kw_struct);

  StringAttr nameAttr;
  ParsedMetaSignature metaSignature;
  if (parseIdentifier(nameAttr, "expected struct name") ||
      metaSignature.parseOptionalMetaSignature(*this) ||
      parseToken(LitToken::colon, "expected ':' in function definition"))
    return failure();

  // TODO: Should have nicer builder.
  auto newStruct = builder.create<LITStructDeclOp>(
      loc, nameAttr,
      ParamDeclArrayAttr::get(getContext(), metaSignature.inputDecls),
      TypeArrayAttr::get(getContext(), {}));
  newStruct.getRegion().push_back(new Block());

  auto newRefAttr = SymbolConstantAttr::get(FlatSymbolRefAttr::get(nameAttr),
                                            builder.getType<MLIRTypeType>());

  scope.addToScope(nameAttr, Scope::MetaParameterValue{newRefAttr, loc},
                   getSharedState());

  // Remember that we parsed this declaration so we can finish type checking it
  // when it gets referenced.
  getDeclResolver().addDecl(newStruct, &scope, declCursor);

  // Skip the body of this definition: go to a token the starts a line at the
  // same indent level (or less) as the current definition.
  skipUntilIndentation(curIndent);
  return success();
}

void LitParser::parseStructBody(LITStructDeclOp structDecl) {
  size_t structIndent = getToken().getIndentation().value_or(0);

  // TODO: Add support for decorators.
  consumeToken(LitToken::kw_struct);

  StringAttr nameAttr;
  ParsedMetaSignature metaSignature;
  if (parseIdentifier(nameAttr, "expected struct name") ||
      metaSignature.parseOptionalMetaSignature(*this) ||
      parseToken(LitToken::colon, "expected ':' in function definition"))
    return;

  // Add the meta parameters to the symbol table.
  for (auto [param, loc] :
       llvm::zip(structDecl.getParamDecls(), metaSignature.inputLocs)) {
    auto value = ParamDeclRefAttr::get(param.getName(), param.getType());
    scope.addToScope(param.getName(), Scope::MetaParameterValue{value, loc},
                     getSharedState());
  }

  (void)parseSuite(structIndent, StmtContext::structBody);
}

void DeclResolver::resolveBody(LITStructDeclOp op, Scope &scope) {
  LitLexer lexer(sharedState, scope.getCursor());
  LitParser parser(lexer, scope);
  parser.parseStructBody(op);
}

//===----------------------------------------------------------------------===//
// LitSharedState
//===----------------------------------------------------------------------===//

/// Get the name of the main buffer so we can rapidly build Location objects
/// on demand.
static StringAttr getMainBufferNameIdentifier(const SourceMgr &sourceMgr,
                                              MLIRContext *context) {
  auto mainBuffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  StringRef bufferName = mainBuffer->getBufferIdentifier();
  if (bufferName.empty())
    bufferName = "<unknown>";
  return StringAttr::get(context, bufferName);
}

LitSharedState::LitSharedState(llvm::SourceMgr &sourceMgr, MLIRContext *context)
    : sourceMgr(sourceMgr), context(context),
      declResolver(std::make_unique<DeclResolver>(*this)),
      bufferNameIdentifier(getMainBufferNameIdentifier(sourceMgr, context)) {}

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

  LitSharedState sharedState(sourceMgr, context);
  LitLexer lexer(sharedState);

  // The outermost scope contains the __builtins__ function definitions.
  // TODO: Add these:
  // https://docs.python.org/3/library/functions.html#built-in-funcs
  // https://docs.python.org/3/reference/executionmodel.html#naming-and-binding
  Scope &builtinsScope =
      sharedState.declResolver->addDecl(*module, nullptr, lexer.getCursor());

  // Create the module scope which will contain all things we parse.  These
  // shadow the builtins module during name lookup.
  Scope &fileScope = sharedState.declResolver->addDecl(*module, &builtinsScope,
                                                       lexer.getCursor());

  // Parse the file.
  if (LitParser(lexer, fileScope).parseFile(*module))
    return nullptr;

  // With the top-level of the file parsed, we can now go ahead and resolve all
  // of the deferred declarations.
  sharedState.declResolver->resolveAll();

  if (sharedState.errorOccurred)
    return nullptr;

  // Make sure the parse module has no other structural problems detected by
  // the verifier.
  auto verificationTimer = ts.nest("Verify module");
  if (failed(verify(*module)))
    return {};
  return module;
}
