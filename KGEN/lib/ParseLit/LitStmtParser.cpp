//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements basic statement parsing.
//
//===----------------------------------------------------------------------===//

#include "ASTDecl.h"
#include "LitDecls.h"
#include "LitExprNodes.h"
#include "LitLexer.h"
#include "LitParserBase.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/HLCFDialect/HLCFOps.h"
#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/SaveAndRestore.h"
#include <filesystem>

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
  ParseResult parseBreakOrContinueStmt(LitToken::Kind kind, StringRef name,
                                       StringRef opName);

  // Declarations.
  ParseResult parseIncludeHack();
  ParseResult parseDefFnStmt(ArrayRef<ExprNode *> decorators, size_t curIndent);
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
  case LitToken::kw_fn:
    rejectSimpleStmt(); // Not a simple_stmt.
    return parseDefFnStmt(/*decorators=*/{}, stmtIndent);
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
    case LitToken::kw_fn:
      rejectSimpleStmt(); // Not a simple_stmt.
      return parseDefFnStmt(decorators, stmtIndent);
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
  case LitToken::kw___include:
    return parseIncludeHack();

  case LitToken::kw_pass:
    // pass_stmt ::= "pass"
    consumeToken(LitToken::kw_pass);
    return success();
  case LitToken::kw_var:
    return parseVarDeclStmt(/*decorators=*/{}, stmtIndent);
  case LitToken::kw_return:
    return parseReturnStmt(stmtIndent);
  case LitToken::kw_continue:
    return parseBreakOrContinueStmt(LitToken::kw_continue, "continue",
                                    HLCF::ContinueOp::getOperationName());
  case LitToken::kw_break:
    return parseBreakOrContinueStmt(LitToken::kw_break, "break",
                                    HLCF::BreakOp::getOperationName());
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

  // If this variable is being declared in a `def` definition, then we allow
  // implicit declarations of variables.  In `fn` and top level, we do not.
  ASTType lhsContextualType;
  if (getDecl().isDef)
    lhsContextualType = rhsValue.type;

  // Resolve LHS expression into an lvalue that we can store into.
  ASTTypeAnd<LValue> lValue = getExprEmitter().emitLValue(
      lhs, lhsContextualType, "cannot assign to immutable expression");
  if (!lValue)
    return success(); // Parse succeeded.

  // Check to see if the destination type and the source type are compatible.
  // TODO: Implement implicit conversions.
  if (!lValue.type.isEqualCanon(rhsValue.type)) {
    emitError(rhs->getLoc(), "cannot convert value of type ")
        << rhsValue.type << " to " << lValue.type;
    return success();
  }

  // If everything worked out, store the resultant value into the lvalue for the
  // destination.  If things didn't work, just drop this on the floor.
  builder.create<POP::StoreOp>(translateLocation(equalsLoc), rhsValue.ir,
                               lValue.ir, /*alignment*/ None);

  return success();
}

/// return_stmt ::= "return" [expression_list]
ParseResult LitStmtParser::parseReturnStmt(size_t returnIndent) {
  auto loc = consumeToken(LitToken::kw_return).getLoc();

  SmallVector<Value> operandValues;
  SmallVector<ASTType> operandTypes;

  // If there is an expression list present, parse it.
  SmallVector<ExprNode *> operandExprs;
  if (!getToken().getIndentation().has_value()) {
    // TODO use hadTrailingSep to return a singleton tuple ex. `return 1,`
    if (parseExpressionList(operandExprs, returnIndent,
                            /*hadTrailingSep=*/nullptr))
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
    operandValues.push_back(value.ir);
    operandTypes.push_back(value.type);
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
        << operandTypes[0] << " but 'def' expected "
        << containingDecl.getResolvedType();
    return success();
  }

  if (isa<LITFuncOp>(builder.getInsertionBlock()->getParentOp())) {
    // TODO: Support result parameters.
    builder.create<ReturnOp>(translateLocation(loc), ArrayRef<TypedAttr>(),
                             operandValues);
  } else {
    builder.create<HLCF::ReturnOp>(translateLocation(loc), operandValues);
  }
  // Split the block here. Subsequent statements are dead code.
  builder.setInsertionPointToStart(
      builder.getInsertionBlock()->splitBlock(builder.getInsertionPoint()));
  return success();
}

/// break_stmt ::= "break"
/// continue_stmt ::= "continue"
ParseResult LitStmtParser::parseBreakOrContinueStmt(LitToken::Kind kind,
                                                    StringRef name,
                                                    StringRef opName) {
  llvm::SMLoc loc = consumeToken(kind).getLoc();
  Block *block = builder.getInsertionBlock();

  // Ensure the break statement is being parsed within a loop context.
  if (!isa<HLCF::LoopOp>(block->getParentOp()) &&
      !block->getParentOp()->getParentOfType<HLCF::LoopOp>()) {
    emitError(loc, "'" + name + "' not inside a loop");
    return success();
  }

  // Split the block at the insertion point. Any subsequent statements are dead
  // code. Let region DCE handle it.
  Block *after = block->splitBlock(builder.getInsertionPoint());
  builder.setInsertionPointToEnd(block);
  OperationState state(translateLocation(loc), opName);
  builder.create(state);
  builder.setInsertionPointToStart(after);
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
  ASTTypeAnd<DRValue> cond = parser.getExprEmitter().emitDRValue(condExp);
  if (!cond)
    return failure();

  // TODO(types): we only support 'index' values as a hack right now.
  if (!cond.ir.getType().isIndex())
    return parser.emitError(condExp->getLoc(), "value of type ")
           << cond.type << " isn't convertible to Bool";

  auto &builder = parser.getBuilder();
  auto loc = cond.ir.getLoc();
  auto one = builder.create<mlir::index::ConstantOp>(loc, 1);
  condValue = builder.create<mlir::index::CmpOp>(
      loc, mlir::index::IndexCmpPredicate::EQ, cond.ir, one);
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

  auto loopOp = builder.create<HLCF::LoopOp>(whileLoc);
  Block *body = builder.createBlock(&loopOp.getBody());
  builder = OpBuilder::atBlockEnd(body);

  Value condVal;
  if (emitExprAsCondition(condExp, condVal, *this))
    return success(); // IRGen error already emitted; parse succeeded!

  // Generate the while condition check.
  auto condOp = builder.create<HLCF::IfOp>(whileLoc, condVal);
  builder.createBlock(&condOp.getThenRegion());
  builder.create<HLCF::YieldOp>(whileLoc);
  Block *exit = builder.createBlock(&condOp.getElseRegion());
  builder.create<HLCF::BreakOp>(whileLoc);

  // Create the body.
  builder.setInsertionPointAfter(condOp);
  if (failed(parseSuite(curIndent)))
    return failure();
  builder.create<HLCF::ContinueOp>(whileLoc);

  // The 'else' block is executed only when the condition check fails.
  if (getToken().getIndentation().has_value() &&
      getToken().getIndentation().value() >= curIndent &&
      consumeIf(LitToken::kw_else)) {
    builder.setInsertionPointToStart(exit);
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
  auto ifOp = builder.create<HLCF::IfOp>(ifLoc, cond);
  builder.createBlock(&ifOp.getThenRegion());
  if (failed(parseSuite(curIndent)))
    return failure();
  builder.create<HLCF::YieldOp>(ifLoc);

  while (getToken().is(LitToken::kw_elif) &&
         getToken().getIndentation().has_value() &&
         getToken().getIndentation().value() >= curIndent) {
    Location elifLoc =
        translateLocation(consumeToken(LitToken::kw_elif).getLoc());
    if (parseExpression(condExp, None) ||
        parseToken(LitToken::colon, "expected ':' after 'elif' expression"))
      return failure();

    builder.createBlock(&ifOp.getElseRegion());
    if (emitExprAsCondition(condExp, cond, *this))
      return success();
    ifOp = builder.create<HLCF::IfOp>(elifLoc, cond);
    builder.create<HLCF::YieldOp>(elifLoc);
    builder.createBlock(&ifOp.getThenRegion());
    if (failed(parseSuite(curIndent)))
      return failure();
    builder.create<HLCF::YieldOp>(ifLoc);
  }

  builder.createBlock(&ifOp.getElseRegion());
  if (getToken().getIndentation().has_value() &&
      getToken().getIndentation().value() >= curIndent &&
      consumeIf(LitToken::kw_else)) {
    if (parseToken(LitToken::colon, "expected ':' after else"))
      return failure();
    if (failed(parseSuite(curIndent)))
      return failure();
  }
  builder.create<HLCF::YieldOp>(ifLoc);
  return success();
}

/// Parse the __include "somePath" directive.
ParseResult LitStmtParser::parseIncludeHack() {
  SMLoc includeLoc = consumeToken(LitToken::kw___include).getLoc();
  StringRef path = getTokenSpelling();
  if (parseToken(LitToken::string, "expected path in __include directive"))
    return failure();

  // Strip off the ""'s.
  path = path.drop_front().drop_back();

  llvm::SourceMgr &sourceMgr = getSharedState().sourceMgr;

  // Resolve the absolute filename of the target.
  std::string absolutePath;
  if (std::filesystem::path(path.str()).is_absolute()) {
    absolutePath = path.str();
  } else {
    // Resolve relative paths w.r.t. the including file.
    const llvm::MemoryBuffer *includerBuffer = sourceMgr.getMemoryBuffer(
        sourceMgr.FindBufferContainingLoc(includeLoc));
    assert(includerBuffer && "Must be in a source buffer");
    auto includerPath =
        std::filesystem::path(includerBuffer->getBufferIdentifier().str());
    absolutePath = includerPath.replace_filename(path.str());
  }

  // Ask SourceMgr to open the file in question.
  std::string fullPath;
  unsigned fileID =
      sourceMgr.AddIncludeFile(absolutePath, includeLoc, fullPath);
  if (fileID == 0) {
    emitError(includeLoc, "could not find file '") << path << "'";
    return success(); // Parse success, semantic failure.
  }

  // Now that we have a MemoryBuffer, we can lex it, and therefore parse it.
  // do so.
  const llvm::MemoryBuffer *includerBuffer = sourceMgr.getMemoryBuffer(fileID);
  LitLexer lexer(getSharedState(), includerBuffer);
  return LitParserBase::parseSuite(containingDecl, lexer);
}

//===----------------------------------------------------------------------===//
// Definition statements
//===----------------------------------------------------------------------===//

namespace {
struct FnAttributes {
  /// This is set to true by @staticmethod.
  bool isStatic = false;
  // This is set to true by @interface.
  bool isInterface = false;
  // This is set by @implementedInterface(x).
  FlatSymbolRefAttr implementedInterface;

  void processDecorator(ExprNode *decorator, LitStmtParser &parser);
};
} // namespace

// Process a function decorator.
void FnAttributes::processDecorator(ExprNode *decorator,
                                    LitStmtParser &parser) {
  if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
    if (declRef->spelling == "staticmethod")
      isStatic = true;
    else if (declRef->spelling == "interface")
      isInterface = true;
    else
      parser.emitError(decorator->getLoc(), "unsupported decorator: ")
          << declRef->spelling;
    return;
  }

  // `x()` forms.
  if (auto callNode = dyn_cast<CallNode>(decorator)) {
    auto declRef = dyn_cast<DeclRefNode>(callNode->callee);
    if (!declRef || declRef->spelling != "implements") {
      parser.emitError(decorator->getLoc(), "unsupported decorator");
      return;
    }
    if (callNode->args.size() != 1 ||
        !isa<DeclRefNode>(callNode->args.front())) {
      parser.emitError(
          decorator->getLoc(),
          "@implements decorator must specify one interface by name");
      return;
    }
    if (implementedInterface)
      parser.emitError(decorator->getLoc(),
                       "only one @implements decorator is allowed");
    StringRef interfaceName =
        cast<DeclRefNode>(callNode->args.front())->spelling;
    implementedInterface =
        FlatSymbolRefAttr::get(parser.getContext(), interfaceName);
    return;
  }

  parser.emitError(decorator->getLoc(), "unsupported decorator");
}

ParseResult LitStmtParser::parseDefFnStmt(ArrayRef<ExprNode *> decorators,
                                          size_t curIndent) {
  // isDef is true when introduced by the 'def' keywords instead of 'fn'.
  bool isDef = getToken().is(LitToken::kw_def);
  Location loc = getTokenLocation();
  consumeToken();

  StringAttr name;
  if (parseIdentifier(name, "expected function name"))
    return failure();

  // Process any decorators we will eventually want when they come up.
  FnAttributes attrs;
  for (ExprNode *decorator : decorators)
    attrs.processDecorator(decorator, *this);

  // Is this a method?
  bool isMethod = false;
  StringAttr baseName = name; // Save the unmangled name.
  if (auto structDecl = dyn_cast<LITStructDeclOp>(containingDecl)) {
    std::string mangledName =
        (Twine(structDecl.getSymName()) + "::" + name.getValue()).str();
    name = StringAttr::get(getContext(), mangledName);
    isMethod = true;
  }

  if (attrs.isStatic && !isMethod) {
    emitError(loc, "only methods on structs may be declared static");
    attrs.isStatic = false;
  }

  Operation *litDecl;
  if (attrs.isInterface) {
    litDecl = builder.create<GeneratorInterfaceOp>(loc, name);
    if (isMethod)
      emitError(loc, "interfaces cannot be nested inside a struct");
  } else {
    auto funcDecl = builder.create<LITFuncOp>(loc, name);
    if (attrs.implementedInterface)
      funcDecl.setImplementsAttr(attrs.implementedInterface);
    if (attrs.isStatic)
      funcDecl.setIsStaticAttr(mlir::UnitAttr::get(getContext()));
    litDecl = funcDecl;
  }

  // Skip the body of this definition: go to a token the starts a line at the
  // same indent level (or less) as the current definition.
  auto startCursor = getLexer().getCursor();
  skipUntilIndentation(curIndent);

  auto &decl =
      getDeclResolver().addDecl(litDecl, baseName, &containingDecl, startCursor,
                                getLexer().getCursor(), curIndent);

  // Remember if this was declared as a 'def' or 'fn' because this affects
  // certain downstream behavior.
  decl.isDef = isDef;

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
  getDeclResolver().addDecl(varDecl, name, &containingDecl, startCursor,
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
  getDeclResolver().addDecl(newStruct, nameAttr, &containingDecl, startCursor,
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
  if (failed(LitStmtParser(lexer, containingDecl)
                 .parseSuite(containingDecl.getIndentation())))
    return failure();
  // Run region DCE to remove dead code.
  mlir::IRRewriter rewriter(containingDecl.getContext());
  containingDecl.getIfOperation()->walk([&](Operation *op) {
    (void)mlir::eraseUnreachableBlocks(rewriter, op->getRegions());
  });
  return success();
}
