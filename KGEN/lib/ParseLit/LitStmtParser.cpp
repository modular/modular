//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements basic statement parsing.
//
//===----------------------------------------------------------------------===//

#include "LitASTDecl.h"
#include "LitDecls.h"
#include "LitExprs.h"
#include "LitParserBase.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace M::KGEN::LIT;
using namespace M::KGEN;
using namespace M;
namespace scf = mlir::scf;

//===----------------------------------------------------------------------===//
// LitStmtParser
//===----------------------------------------------------------------------===//

/// This class provides the implementation details of the concrete Lightning
/// grammar.
namespace {
struct LitStmtParser : public LitParserBase {
  LitStmtParser(LitLexer &lexer, ASTDecl &containingDecl)
      : LitParserBase(lexer), containingDecl(containingDecl),
        builder(containingDecl.getDeclEndBuilder()) {

    // Create the varDeclCursor with an arbitrary op.  We delete it on
    // destruction of this statement parser.
    varDeclCursor = builder.create<mlir::index::ConstantOp>(
        containingDecl.getLoc(), 1234567);
  }

  ~LitStmtParser() {
    // The varDeclCursor operation is no longer needed.
    varDeclCursor->erase();
  }

  ParseResult parseFile(ModuleOp module);

  const ASTDecl &getDecl() const { return containingDecl; }
  OpBuilder &getBuilder() { return builder; }

  // Expression emission.

  ExprEmitter getExprEmitter() {
    return ExprEmitter(getSharedState(), containingDecl, builder,
                       varDeclCursor);
  }

  ParseResult parseSuite(ssize_t curIndent);
  ParseResult parseStmts(size_t minIndent);
  ParseResult parseStmt(bool isSimpleStmt, size_t curIndent);

  // Compound statements.
  ParseResult parseIfStmt(size_t curIndent);
  ParseResult parseWhileStmt(size_t curIndent);

  // Simple statements.
  ParseResult parseReturnStmt(size_t returnIndent);
  ParseResult parseAssignmentStmt(ExprNode *lhs, SMLoc equalsLoc,
                                  size_t stmtIndent);

  // Declarations.
  ParseResult parseDefStmt(ArrayRef<ExprNode *> decorators, size_t curIndent);
  ParseResult parseStructStmt(ArrayRef<ExprNode *> decorators,
                              size_t curIndent);
  ParseResult parseVarDeclStmt(ArrayRef<ExprNode *> decorators,
                               size_t stmtIndent);

private:
  /// This is declaration / scope that we're parsing into.
  ASTDecl &containingDecl;

  /// This is the builder that we are constructing IR into.
  OpBuilder builder;

  /// This is the operation we should install VarDecl's ahead of.  This ensures
  /// they are emitted ahead of anything else in the region for the decl, and
  /// in decls with multiple regions (e.g. function bodies with if statements)
  /// it ensures the decl dominates the whole body.
  Operation *varDeclCursor;
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
    if (parseStmt(/*isSimpleStmt=*/true, /*NoWrapping=*/~size_t(0)))
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

    if (parseStmt(/*isSimpleStmt=*/false, indent.value()))
      return failure();
  }
  return success();
}

/// When `isSimpleStmt` is true, this parses the simple_stmt production,
/// otherwise it parses the broader `statment` production that includes compound
/// statements.
///
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
///
ParseResult LitStmtParser::parseStmt(bool isSimpleStmt, size_t stmtIndent) {
  // This lambda is used to generate an error when a compound statement is used
  // in a scenario that expects simple statements.
  auto rejectSimpleStmt = [&]() {
    if (isSimpleStmt)
      emitError() << "'" << getToken().getSpelling()
                  << "' statement must be on its own line";
  };

  switch (getToken().getKind()) {
    //===------------------------------------------------------------------===//
    // Compound statements.
    //===------------------------------------------------------------------===//
  case LitToken::kw_if:
    rejectSimpleStmt(); // Not a simple_stmt.
    return parseIfStmt(stmtIndent);
  case LitToken::kw_while:
    rejectSimpleStmt(); // Not a simple_stmt.
    return parseWhileStmt(stmtIndent);
  case LitToken::kw_def:
    rejectSimpleStmt(); // Not a simple_stmt.
    return parseDefStmt(/*decorators=*/{}, stmtIndent);
  case LitToken::kw_struct:
    rejectSimpleStmt(); // Not a simple_stmt.
    return parseStructStmt(/*decorators=*/{}, stmtIndent);

  case LitToken::at: {
    SmallVector<ExprNode *> decorators;
    consumeToken(LitToken::at);
    do {
      if (parseExpression(decorators.emplace_back(), stmtIndent))
        return failure();
    } while (consumeIf(LitToken::at));

    switch (getToken().getKind()) {
    case LitToken::kw_def:
      rejectSimpleStmt(); // Not a simple_stmt.
      return parseDefStmt(decorators, stmtIndent);
    case LitToken::kw_struct:
      rejectSimpleStmt(); // Not a simple_stmt.
      return parseStructStmt(decorators, stmtIndent);
    case LitToken::kw_var:
      return parseVarDeclStmt(decorators, stmtIndent);

    default:
      return emitError("unknown decorated statement");
    }
  }

    //===------------------------------------------------------------------===//
    // Simple statements.
    //===------------------------------------------------------------------===//
  case LitToken::kw_pass:
    // pass_stmt ::= "pass"
    consumeToken(LitToken::kw_pass);
    return success();
  case LitToken::kw_var:
    return parseVarDeclStmt(/*decorators=*/{}, stmtIndent);
  case LitToken::kw_return:
    return parseReturnStmt(stmtIndent);
  default:
    break;
  }

  // Otherwise, we must have a statement that starts with the expression
  // grammar.
  if (isa<LITStructDeclOp>(containingDecl))
    emitError("invalid expression in this context");

  // expression_stmt ::= starred_expression
  // assignment_stmt ::=
  //                 (target_list "=")+ (starred_expression |
  //                 yield_expression)
  ExprNode *expr = nullptr;
  if (parseExpression(expr, stmtIndent))
    return failure();

  // If the expression was followed by a `=` then we have an assignment.  If
  // not then we have an expression_stmt.
  SMLoc equalsLoc;
  if (consumeIf(LitToken::equal, &equalsLoc))
    return parseAssignmentStmt(expr, equalsLoc, stmtIndent);

  // Materialize the expression statement in our current scope but discard the
  // result on the floor.  Note that this does not materialize an LValue, but
  // does evaluate side effects.
  ExprEmitter state(getSharedState(), containingDecl, builder, nullptr);
  (void)expr->emitIR(state);
  return success();
}

//===----------------------------------------------------------------------===//
// Simple statements.
//===----------------------------------------------------------------------===//

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
ParseResult LitStmtParser::parseAssignmentStmt(ExprNode *lhs, SMLoc equalsLoc,
                                               size_t stmtIndent) {
  // Finish parsing the assignment.
  ExprNode *rhs = nullptr;
  if (parseExpression(rhs, stmtIndent))
    return failure();

  // Materialize the expression statement.
  auto rhsValue = getExprEmitter().emitDRValue(rhs);
  if (!rhsValue)
    return success(); // Parse succeeded.

  // Resolve LHS expression into an lvalue that we can store into.
  LValue lValue = getExprEmitter().emitLValue(lhs, rhsValue.getType(),
                                              "cannot assign to expression");
  if (!lValue)
    return success(); // Parse succeeded.

  // Check to see if the destination type and the source type are compatible.
  auto destEltType =
      cast<POP::PointerType>(lValue.getType()).getResolvedElementType();
  // TODO: Implement implicit conversions.
  if (destEltType != rhsValue.getType()) {
    emitError(rhs->getLoc(), "cannot convert value of type ")
        << rhsValue.getType() << " to " << destEltType;
    return success();
  }

  // If everything worked out, store the resultant value into the lvalue for the
  // destination.  If things didn't work, just drop this on the floor.
  builder.create<POP::StoreOp>(translateLocation(equalsLoc), rhsValue, lValue,
                               /*alignment*/ None);

  return success();
}

/// return_stmt ::= "return" [expression_list]
ParseResult LitStmtParser::parseReturnStmt(size_t returnIndent) {
  auto loc = consumeToken(LitToken::kw_return).getLoc();

  SmallVector<Value> operandValues;

  // If there is an expression list present, parse it.
  SmallVector<ExprNode *> operandExprs;
  if (!getToken().getIndentation().has_value()) {
    if (parseExpressionList(operandExprs, returnIndent))
      return failure();
  } else {
    // If there was no returned value, then default to "return None".  This
    // allows type inference to uniformly support all the things that the None
    // literal coerces to (e.g. an Optional type).
    operandExprs.push_back(getNoneExpr(loc));
  }

  // Materialize the expression values into IR.
  for (auto expr : operandExprs) {
    auto value = getExprEmitter().emitDRValue(expr);
    if (!value)
      return failure();
    operandValues.push_back(value);
  }

  // We don't support formation of tuples / multiple result values yet.
  if (operandValues.size() > 1) {
    emitError(loc, "tuple return not supported yet");
    return success();
  }

  // Check the result values match expected types.
  LITFuncOp decl = dyn_cast<LITFuncOp>(containingDecl);
  if (!decl) {
    emitError(loc, "cannot return from this context");
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
  // TODO(parameters): If the condition is a meta value, don't emit dead code
  // to test it.
  Value cond = parser.getExprEmitter().emitDRValue(condExp);
  if (!cond)
    return failure();

  // TODO(types): we only support 'index' values as a hack right now.
  if (!cond.getType().isIndex())
    return parser.emitError(condExp->getLoc(), "value of type ")
           << cond.getType() << " isn't convertible to Bool";

  auto &builder = parser.getBuilder();
  auto one = builder.create<mlir::index::ConstantOp>(cond.getLoc(), 1);
  condValue = builder.create<mlir::index::CmpOp>(
      cond.getLoc(), mlir::index::IndexCmpPredicate::EQ, cond, one);
  return success();
}

/// while_stmt ::=  "while" assignment_expression ":" suite
///                 ["else" ":" suite]
ParseResult LitStmtParser::parseWhileStmt(size_t curIndent) {
  Location whileLoc =
      translateLocation(consumeToken(LitToken::kw_while).getLoc());

  ExprNode *condExp = nullptr;
  if (parseExpression(condExp, None) ||
      parseToken(LitToken::colon, "expected ':' after expression"))
    return failure();

  // We will be moving the builder into sub-regions that are created, make sure
  // we end up after it when this is done.
  llvm::SaveAndRestore<OpBuilder> builderSaver(builder);

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
  llvm::SaveAndRestore<OpBuilder> builderSaver(builder);

  ExprNode *condExp = nullptr;
  Value cond;
  if (parseExpression(condExp, None) ||
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
    if (parseExpression(condExp, None) ||
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

ParseResult LitStmtParser::parseDefStmt(ArrayRef<ExprNode *> decorators,
                                        size_t curIndent) {
  Location loc = getTokenLocation();

  // TODO: Add support for decorators.
  StringAttr name;
  consumeToken(LitToken::kw_def);
  if (parseIdentifier(name, "expected function name"))
    return failure();

  auto funcDecl = builder.create<LITFuncOp>(loc, name);
  funcDecl.getRegion().push_back(new Block());

  // Process any decorators we will eventually want when they come up.
  if (!decorators.empty())
    emitError(decorators[0]->getLoc(), "no def decorators supported yet");

  // Skip the body of this definition: go to a token the starts a line at the
  // same indent level (or less) as the current definition.
  auto startCursor = getLexer().getCursor();
  skipUntilIndentation(curIndent);

  getDeclResolver().addDecl(funcDecl, &containingDecl, startCursor,
                            getLexer().getCursor(), curIndent);
  return success();
}

ParseResult LitStmtParser::parseVarDeclStmt(ArrayRef<ExprNode *> decorators,
                                            size_t stmtIndent) {
  auto loc = getTokenLocation();
  consumeToken(LitToken::kw_var);
  StringAttr name;
  if (parseIdentifier(name, "expected name for 'var' declaration"))
    return failure();

  auto varType = POP::PointerType::get(UnresolvedType::get(getContext()));
  auto varDecl = OpBuilder(varDeclCursor).create<VarDeclOp>(loc, varType, name);

  // Process any decorators we will eventually want when they come up.
  if (!decorators.empty())
    emitError(decorators[0]->getLoc(), "no var decorators supported yet");

  // Skip the body of this definition: go to a token the starts a line at the
  // same indent level (or less) as the current definition.
  auto startCursor = getLexer().getCursor();
  skipUntilIndentation(stmtIndent, /*stopOnSemicolon=*/true);

  // Remember that we parsed this declaration so we can finish type checking it
  // when it gets referenced.
  getDeclResolver().addDecl(varDecl, &containingDecl, startCursor,
                            getLexer().getCursor(), stmtIndent);

  return success();
}

ParseResult LitStmtParser::parseStructStmt(ArrayRef<ExprNode *> decorators,
                                           size_t curIndent) {
  // We don't support structs in structs (yet?).
  if (isa<LITStructDeclOp>(containingDecl))
    emitError("nested struct not supported here");

  auto loc = getTokenLocation();

  // TODO: Add support for decorators.
  consumeToken(LitToken::kw_struct);

  StringAttr nameAttr;
  if (parseIdentifier(nameAttr, "expected struct name"))
    return failure();

  auto newStruct = builder.create<LITStructDeclOp>(loc, nameAttr);

  // Process any decorators we will eventually want when they come up.
  if (!decorators.empty())
    emitError(decorators[0]->getLoc(), "no struct decorators supported yet");

  // Skip the body of this definition: go to a token the starts a line at the
  // same indent level (or less) as the current definition.
  auto startCursor = getLexer().getCursor();
  skipUntilIndentation(curIndent);

  // Remember that we parsed this declaration so we can finish type checking it
  // when it gets referenced.
  getDeclResolver().addDecl(newStruct, &containingDecl, startCursor,
                            getLexer().getCursor(), curIndent);

  return success();
}

//===----------------------------------------------------------------------===//
// Entry point to this file
//===----------------------------------------------------------------------===//

/// Parse a 'suite' production into the declaration specified by `ASTDecl`.
/// This is the main entrypoint to this file.
ParseResult LitParserBase::parseSuite(ASTDecl &containingDecl,
                                      LitLexer &lexer) {
  return LitStmtParser(lexer, containingDecl)
      .parseSuite(containingDecl.getIndentation());
}
