//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements basic statement parsing.
//
//===----------------------------------------------------------------------===//

#include "LitDecls.h"
#include "LitExprNodes.h"
#include "LitParserBase.h"
#include "LitScope.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/IndexDialect/IndexAttrs.h"
#include "Support/IndexDialect/IndexOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace M::KGEN::LIT;
using namespace M::KGEN;
using namespace M;
namespace scf = mlir::scf;

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
// LitStmtParser
//===----------------------------------------------------------------------===//

/// This class provides the implementation details of the concrete Lightning
/// grammar.
namespace {
struct LitStmtParser : public LitParserBase {
  LitStmtParser(LitLexer &lexer, Scope &scope)
      : LitParserBase(lexer), scope(scope), builder(scope.getDeclEndBuilder()) {
  }

  ParseResult parseFile(ModuleOp module);

  const Scope &getScope() const { return scope; }
  OpBuilder &getBuilder() { return builder; }

  // Expression emission.

  /// Emit the specified expression tree to MLIR in the current context.
  MLIRValueRep emitExpr(ExprNode *node) {
    EmitterState state(*this, scope, builder);
    return node->emit(state);
  }

  Value emitExprAsValue(ExprNode *node) {
    EmitterState state(*this, scope, builder);
    return state.emitAsValue(node);
  }

  ParseResult parseSuite(ssize_t curIndent);
  ParseResult parseStmts(size_t minIndent);
  ParseResult parseStmt(size_t curIndent);
  ParseResult parseSimpleStmt();

  // Compound statements.
  ParseResult parseIfStmt(size_t curIndent);
  ParseResult parseWhileStmt(size_t curIndent);

  // Simple statements.
  ParseResult parseReturnStmt();
  ParseResult parseAssignmentStmt(ExprNode *lhs, SMLoc equalsLoc);

  // Declarations.
  ParseResult parseDefStmt(size_t curIndent);
  ParseResult parseStructStmt(size_t curIndent);
  ParseResult parseVarDeclStmt();

  /// Type parsing.
  ParseResult parseType(Type &result) {
    return LitParserBase::parseType(result, scope);
  }

private:
  /// This is declaration scope that we're parsing into.
  Scope &scope;

  /// This is the builder that we are constructing IR into.
  OpBuilder builder;
};
} // namespace

/// Parse a suite, which is either a series of comma separated simple_stmt's on
/// one line, or an indented block of statements. curIndent is the containing
/// statement's indentation level.
///
/// suite     ::=  [stmt_list NEWLINE] | NEWLINE INDENT statement+ DEDENT
/// stmt_list ::=  simple_stmt (";" simple_stmt)* [";"]
ParseResult LitStmtParser::parseSuite(ssize_t curIndent) {
  // Ignore empty body at end of file: a `pass` is not required.
  if (getToken().is(LitToken::eof))
    return success();

  // If there is a newline, then parse a list of statements.
  if (auto indent = getToken().getIndentation()) {
    // If the current token is less indented that the source of the suite,
    // then the body is empty.  We don't require a pass.
    if (ssize_t(indent.value()) <= curIndent)
      return success();
    return parseStmts(curIndent + 1);
  }

  // Otherwise, parse a stmt_list.
  do {
    if (parseSimpleStmt())
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
ParseResult LitStmtParser::parseStmts(size_t minIndent) {
  while (getToken().isNot(LitToken::eof)) {
    auto indent = getToken().getIndentation();
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
/// compound_stmt ::= if_stmt
///                 | while_stmt
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
ParseResult LitStmtParser::parseStmt(size_t curIndent) {
  // Handle compound stmts here and chain to simple statements to handle the
  // whole "statement" production.
  switch (getToken().getKind()) {
  case LitToken::kw_if:
    return parseIfStmt(curIndent);
  case LitToken::kw_while:
    return parseWhileStmt(curIndent);
  case LitToken::kw_def:
    return parseDefStmt(curIndent);
  case LitToken::kw_struct:
    // We don't support structs in structs (yet?).
    if (isa<LITStructDeclOp>(scope.getDecl()))
      emitError("nested struct not supported here");
    return parseStructStmt(curIndent);

  // NOTE: When adding new cases here, make sure to add them to parseSimpleStmt
  // as well for error recovery.
  default:
    // Otherwise must be a simple statement.
    return parseSimpleStmt();
  }
}

//===----------------------------------------------------------------------===//
// Simple statements.
//===----------------------------------------------------------------------===//

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
ParseResult LitStmtParser::parseSimpleStmt() {
  switch (getToken().getKind()) {
  case LitToken::kw_if:
  case LitToken::kw_while:
  case LitToken::kw_def:
  case LitToken::kw_struct:
    emitError() << "'" << getToken().getSpelling()
                << "' statement must be on its own line";
    return parseStmt(0);

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
  if (isa<LITStructDeclOp>(scope.getDecl()))
    emitError("invalid expression in this context");

  // expression_stmt ::= starred_expression
  // assignment_stmt ::=
  //                 (target_list "=")+ (starred_expression | yield_expression)
  ExprNode *expr = nullptr;
  if (parseExpression(expr))
    return failure();

  // If the expression was followed by a `=` then we have an assignment.  If not
  // then we have an expression_stmt.
  SMLoc equalsLoc;
  if (consumeIf(LitToken::equal, &equalsLoc))
    return parseAssignmentStmt(expr, equalsLoc);

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
ParseResult LitStmtParser::parseAssignmentStmt(ExprNode *lhs, SMLoc equalsLoc) {
  // Finish parsing the assignment.
  ExprNode *rhs = nullptr;
  if (parseExpression(rhs))
    return failure();

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

  // Materialize the expression statement in our current scope.
  auto rhsValue = emitExprAsValue(rhs);
  // If IR generation failed, return success since we have a fine parse.
  if (!rhsValue)
    return success();

  // Look up the name being assigned to if it already exists.
  auto nameAttr = builder.getStringAttr(dre->spelling);
  Value lvalue;
  if (Optional<Scope::NameEntry> decl = scope.lookupInCurrentScope(nameAttr)) {
    // Don't allow reassigning to functions and other constant parameters.
    VarDeclOp var;
    if (std::holds_alternative<Scope *>(decl.value())) {
      Scope &varScope = *std::get<Scope *>(decl.value());
      var = dyn_cast<VarDeclOp>(varScope.getDecl());
      // Make sure the destination is resolved, an error will be diagnosed so
      // we can just return parse success.
      if (failed(getDeclResolver().resolve(
              varScope, DeclResolvedness::signatureResolved, lhs->getLoc())))
        return success();
    }

    if (var) {
      lvalue = var.getResult();
    } else {
      emitError(lhs->getLoc(), "this declaration isn't reassignable");
      return success();
    }
  } else {
    // Otherwise, introduce a new lit.var.decl node whose type matches the
    // initializer expression.
    //
    // TODO(autopromotions): turn infinite integers into concrete ones as
    // needed.
    auto declType = POP::PointerType::get(rhsValue.getType());

    // Use this builder to place any VarDeclOps. In Python there is only one
    // scope per function and all variables belong to that scope, so builders
    // should reflect that.
    auto varDecl = scope.getDeclBuilder().create<VarDeclOp>(
        translateLocation(lhs->getLoc()), declType, nameAttr);
    getDeclResolver().addFullyResolvedDecl(varDecl, &scope);
    lvalue = varDecl;
  }

  // Check to see if the destination type and the source type are compatible.
  auto destEltType =
      cast<POP::PointerType>(lvalue.getType()).getResolvedElementType();
  // TODO: Implement implicit conversions.
  if (destEltType && destEltType != rhsValue.getType()) {
    emitError(rhs->getLoc(), "cannot convert value of type ")
        << rhsValue.getType() << " to " << destEltType;
    return success();
  }

  // If everything worked out, store the resultant value into the lvalue for the
  // destination.  If things didn't work, just drop this on the floor.
  builder.create<POP::StoreOp>(translateLocation(equalsLoc), rhsValue, lvalue,
                               /*alignment*/ None);

  return success();
}

/// return_stmt ::= "return" [expression_list]
ParseResult LitStmtParser::parseReturnStmt() {
  auto loc = consumeToken(LitToken::kw_return).getLoc();

  SmallVector<Value> operandValues;

  // If there is an expression list present, parse it.
  if (!getToken().getIndentation().has_value()) {
    SmallVector<ExprNode *> operandExprs;
    if (parseExpressionList(operandExprs))
      return failure();

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

//===----------------------------------------------------------------------===//
// Compound statements.
//===----------------------------------------------------------------------===//

static ParseResult emitExprAsCondition(ExprNode *condExp, Value &condValue,
                                       LitStmtParser &parser) {
  // TODO(types): add type checking: the condition should be bool.
  // TODO(parameters): If the condition is a meta constant, don't emit dead code
  // to test it.
  Value cond = parser.emitExprAsValue(condExp);
  if (!cond)
    return failure();

  // TODO(types): we only support 'index' values as a hack right now.
  if (!cond.getType().isIndex())
    return parser.emitError(condExp->getLoc(), "value of type ")
           << cond.getType() << " isn't convertible to Bool";

  auto &builder = parser.getBuilder();
  auto one = builder.create<index::ConstantOp>(cond.getLoc(), 1);
  condValue = builder.create<index::CmpOp>(
      cond.getLoc(), index::IndexCmpPredicate::EQ, cond, one);
  return success();
}

/// while_stmt ::=  "while" assignment_expression ":" suite
///                 ["else" ":" suite]
ParseResult LitStmtParser::parseWhileStmt(size_t curIndent) {
  Location whileLoc =
      translateLocation(consumeToken(LitToken::kw_while).getLoc());

  ExprNode *condExp = nullptr;
  if (parseExpression(condExp) ||
      parseToken(LitToken::colon, "expected ':' after expression"))
    return failure();

  // We will be moving the builder into sub-regions that are created, make sure
  // we end up after it when this is done.
  SaveAndRestore<OpBuilder> builderSaver(builder);

  auto whileOp = builder.create<scf::WhileOp>(whileLoc, ArrayRef<Type>(),
                                              ArrayRef<Value>());
  Block *before = builder.createBlock(&whileOp.getBefore());
  Block *after = builder.createBlock(&whileOp.getAfter());

  // Create the condition block.
  builder = OpBuilder::atBlockEnd(before);

  Value condVal;
  if (emitExprAsCondition(condExp, condVal, *this))
    return success(); // IRGen error already emitted; parse succeeded!

  builder.create<scf::ConditionOp>(whileLoc, condVal, ArrayRef<Value>());

  // Create the after block.
  builder = OpBuilder::atBlockEnd(after);
  if (failed(parseSuite(curIndent)))
    return failure();
  builder.create<scf::YieldOp>(whileLoc);

  // If there is an else block, emit it after the while op.
  if (getToken().getIndentation().has_value() &&
      getToken().getIndentation().value() >= curIndent &&
      consumeIf(LitToken::kw_else)) {
    builder.setInsertionPointAfter(whileOp);
    if (parseToken(LitToken::colon, "expected ':' after else") ||
        parseSuite(curIndent))
      return failure();
  }
  return success();
}

/// if_stmt ::=  "if" assignment_expression ":" suite
///             ("elif" assignment_expression ":" suite)*
///             ["else" ":" suite]
ParseResult LitStmtParser::parseIfStmt(size_t curIndent) {
  Location ifLoc = translateLocation(consumeToken(LitToken::kw_if).getLoc());

  // We will be moving the builder into sub-regions that are created, make sure
  // we end up after it when this is done.
  SaveAndRestore<OpBuilder> builderSaver(builder);

  ExprNode *condExp = nullptr;
  Value cond;
  if (parseExpression(condExp) ||
      parseToken(LitToken::colon, "expected ':' after 'if' expression"))
    return failure();

  if (emitExprAsCondition(condExp, cond, *this))
    return success();

  // Create the 'if' and parse the body into its "then" region.
  auto ifOp = builder.create<scf::IfOp>(ifLoc, cond, /*withElse=*/true);
  builder = ifOp.getThenBodyBuilder();
  if (failed(parseSuite(curIndent)))
    return failure();

  while (getToken().is(LitToken::kw_elif) &&
         getToken().getIndentation().has_value() &&
         getToken().getIndentation().value() >= curIndent) {
    Location elifLoc =
        translateLocation(consumeToken(LitToken::kw_elif).getLoc());
    if (parseExpression(condExp) ||
        parseToken(LitToken::colon, "expected ':' after 'elif' expression"))
      return failure();

    builder = ifOp.getElseBodyBuilder();
    if (emitExprAsCondition(condExp, cond, *this))
      return success();
    ifOp = builder.create<scf::IfOp>(elifLoc, cond, /*withElse=*/true);
    builder = ifOp.getThenBodyBuilder();
    if (failed(parseSuite(curIndent)))
      return failure();
  }

  if (getToken().getIndentation().has_value() &&
      getToken().getIndentation().value() >= curIndent &&
      consumeIf(LitToken::kw_else)) {
    if (parseToken(LitToken::colon, "expected ':' after else"))
      return failure();
    builder = ifOp.getElseBodyBuilder();
    if (failed(parseSuite(curIndent)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Definition statements
//===----------------------------------------------------------------------===//

ParseResult LitStmtParser::parseDefStmt(size_t curIndent) {
  Location loc = getTokenLocation();

  // TODO: Add support for decorators.
  StringAttr name;
  consumeToken(LitToken::kw_def);
  if (parseIdentifier(name, "expected function name"))
    return failure();

  auto functionType =
      builder.getFunctionType(ArrayRef<Type>(), ArrayRef<Type>());

  // TODO: Should have nicer builder.
  auto funcDecl = builder.create<LITFuncOp>(
      loc, name, StringArrayAttr::get(getContext(), {}),
      TypeAttr::get(functionType),
      builder.getAttr<LinkageAttr>(Linkage::Public),
      ParamDeclArrayAttr::get(getContext(), {}),
      TypeArrayAttr::get(getContext(), {}),
      ConstraintArrayAttr::get(getContext(), {}), FlatSymbolRefAttr());
  funcDecl.getRegion().push_back(new Block());

  // We cannot parse the current body without having parsed other declarations
  // at the current level, so we defer parsing it.
  getDeclResolver().addDecl(funcDecl, &scope, getLexer().getCursor(),
                            curIndent);

  // Skip the body of this definition: go to a token the starts a line at the
  // same indent level (or less) as the current definition.
  skipUntilIndentation(curIndent);
  return success();
}

ParseResult LitStmtParser::parseVarDeclStmt() {
  ssize_t indent = getToken().getIndentation().value_or(-size_t(1));
  auto loc = getTokenLocation();
  consumeToken(LitToken::kw_var);
  StringAttr name;
  if (parseIdentifier(name, "expected name for 'var' declaration"))
    return failure();

  auto builder = scope.getDeclBuilder();
  auto varType = POP::PointerType::get(UnresolvedType::get(getContext()));
  auto varDecl = builder.create<VarDeclOp>(loc, varType, name);

  // Remember that we parsed this declaration so we can finish type checking it
  // when it gets referenced.
  getDeclResolver().addDecl(varDecl, &scope, getLexer().getCursor(), indent);

  // Skip the body of this definition: go to a token the starts a line at the
  // same indent level (or less) as the current definition.
  skipUntilIndentation(indent, /*stopOnSemicolon=*/true);
  return success();
}

ParseResult LitStmtParser::parseStructStmt(size_t curIndent) {
  auto loc = getTokenLocation();

  // TODO: Add support for decorators.
  consumeToken(LitToken::kw_struct);

  StringAttr nameAttr;
  if (parseIdentifier(nameAttr, "expected struct name"))
    return failure();

  // TODO: Should have nicer builder.
  auto newStruct = builder.create<LITStructDeclOp>(
      loc, nameAttr, ParamDeclArrayAttr::get(getContext(), {}),
      TypeArrayAttr::get(getContext(), {}));
  newStruct.getRegion().push_back(new Block());

  // Remember that we parsed this declaration so we can finish type checking it
  // when it gets referenced.
  getDeclResolver().addDecl(newStruct, &scope, getLexer().getCursor(),
                            curIndent);

  // Skip the body of this definition: go to a token the starts a line at the
  // same indent level (or less) as the current definition.
  skipUntilIndentation(curIndent);
  return success();
}

//===----------------------------------------------------------------------===//
// Entry point to this file
//===----------------------------------------------------------------------===//

/// Parse a 'suite' production into the declaration specified by `Scope`.
/// This is the main entrypoint to this file.
ParseResult LitParserBase::parseSuite(Scope &scope, LitLexer &lexer) {
  return LitStmtParser(lexer, scope).parseSuite(scope.getIndentation());
}
