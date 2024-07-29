//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MojoUtils.h"
#include "Signatures.h"
#include "Traits.h"

#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ClosureEmitter.h"
#include "KGEN/MojoParser/DLValues.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/ExprEmitter.h"
#include "KGEN/MojoParser/ExprNodes.h"
#include "KGEN/MojoParser/ParserBase.h"
#include "KGEN/MojoParser/StructEmitter.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MOGGPreElab/MOGGDecorators.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/Filesystem/Paths.h"

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

/// Parse an expression and immediately resolve it to a type.  This returns
/// failure on parse error.
static ParseResult parseType(ParserBase &p, ASTType &result, ASTDecl &declScope,
                             std::optional<size_t> stmtIndent) {
  ExprNode *expr = nullptr;
  if (p.parseExpression(expr, stmtIndent))
    return failure();

  ExprEmitter emitter(p.shared, declScope, EC_Type);
  result = emitter.emitExprType(expr);
  if (!result)
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// Decorator Support
//===----------------------------------------------------------------------===//

static void
rejectDecorators(ArrayRef<std::pair<ExprNode *, LexerCursor>> decoratorExprs,
                 ASTDecl &decl, SharedState &shared) {
  if (decoratorExprs.empty())
    return;

  shared.emitError(decoratorExprs[0].first->getLoc(),
                   "decorators not supported on this statement")
      << SourceRange(decoratorExprs.front().first->getRangeStart(),
                     decoratorExprs.back().first->getRangeEnd());
}

namespace {
/// Decorators attached to a declaration may be "signature" decorators, "body"
/// decorators, compiler decorators, or dynamic decorators.
///
/// - Signature decorators are applied during the resolution of the signature of
///   a declaration before it is name bound.
/// - Body decorators are applied after the body of the declaration is fully
///   resolved.
/// - Compiler decorators (TODO) are applied at some stage in the Mojo
///   compilation pipeline.
/// - Dynamic decorators (TODO) are applied at the object at runtime.
///
/// This is the base class for handling decorators on declarations. Signature
/// decorators are processed first and then leftover decorators are persisted
/// until body resolution is complete via the SharedState.
class Decorators : public SharedStateUser {
public:
  /// Create a class to handle decorators for a decl. If `signatureOnly` is set,
  /// the class will reject any decorator not processed during signature
  /// resolution.
  Decorators(ASTDecl &decl, SharedState &shared, bool signatureOnly = false)
      : SharedStateUser(shared), decl(decl), signatureOnly(signatureOnly) {}

  /// Handle the `@deprecated` decorator for all decls.
  LogicalResult handleDeprecated(ExprNode *expr);

  /// Process signature decorators on the declaration using the provided
  /// functor. The functor should return success if the decorator was processed
  /// as a signature decorator.
  void applySignatureDecorators(
      ArrayRef<std::pair<ExprNode *, LexerCursor>> decoratorExprs,
      function_ref<LogicalResult(ExprNode *)> process = [](ExprNode *) {
        return failure();
      });

  /// Process body decorators on the declaration using the provided functor.
  /// The functor should return success if the decorator was processed as a
  /// signature decorator. Any leftover decorators are emitted and deferred to
  /// the operation.
  void applyBodyDecorators(function_ref<LogicalResult(ExprNode *)> process);

private:
  /// The declaration this class is applying decorators to.
  ASTDecl &decl;
  /// Whether only signature decorators are allowed.
  bool signatureOnly;
};
} // namespace

LogicalResult Decorators::handleDeprecated(ExprNode *expr) {
  // Detect expression `deprecated` and complain that a warning message should
  // be explicitly specified.
  if (auto declRef = dyn_cast<DeclRefNode>(expr);
      declRef && declRef->spelling == "deprecated") {
    shared.emitError(expr->getLoc(), "@deprecated requires a warning message")
        << FixIt::insertAfterToken(expr->getRange().getEnd(),
                                   "(\"insert deprecation message here\")",
                                   shared.diags);
    return success();
  }

  // Detect expression `deprecated("some string")`.
  auto callNode = dyn_cast<CallNode>(expr);
  if (!callNode)
    return failure();
  auto declRef = dyn_cast<DeclRefNode>(callNode->callee);
  if (!declRef || declRef->spelling != "deprecated" ||
      callNode->operands.size() != 1 ||
      !callNode->operands.front().isPositional())
    return failure();
  auto strExpr = dyn_cast<StringLiteralNode>(callNode->operands.front().expr);
  if (!strExpr)
    return failure();
  cast<ASTDeclInterface>(decl).setDeprecationWarningAttr(
      StringAttr::get(getContext(), strExpr->getValue()));
  return success();
}

void Decorators::applySignatureDecorators(
    ArrayRef<std::pair<ExprNode *, LexerCursor>> decoratorExprs,
    function_ref<LogicalResult(ExprNode *)> process) {
  // Process decorators in the order they are seen. Stop at the first decorator
  // that needs to be deferred.
  while (true) {
    // Return if we are out of decorators.
    if (decoratorExprs.empty())
      return;
    ExprNode *decorator = decoratorExprs.front().first;
    if (succeeded(handleDeprecated(decorator)) ||
        succeeded(process(decorator))) {
      decoratorExprs = decoratorExprs.drop_front();
      continue;
    }
    break;
  }
  // Ensure that there are no other signature decorators afterwards. This is
  // an error.
  SmallVector<ExprNode *> bodyDecorators;
  bodyDecorators.push_back(decoratorExprs.front().first);
  for (auto [i, decorator] :
       llvm::enumerate(llvm::make_first_range(decoratorExprs.drop_front()))) {
    if (failed(process(decorator))) {
      bodyDecorators.push_back(decorator);
      continue;
    }
    // If the decorator applies, we have an error.
    InflightDiag diag =
        emitError(decorator->getLoc(),
                  "signature decorator cannot come after body decorator")
        << decorator->getRange();
    ExprNode *bodyDecorator = decoratorExprs[i].first;
    diag.attachNote(bodyDecorator->getLoc())
        << "previous body decorator applied here" << bodyDecorator->getRange();
    break;
  }

  if (!bodyDecorators.empty() && signatureOnly) {
    shared.emitError(bodyDecorators.front()->getLoc(),
                     "unsupported decorator on this statement")
        << SourceRange(bodyDecorators.front()->getRangeStart(),
                       bodyDecorators.back()->getRangeEnd());
    return;
  }

  // Defer the rest of the decorators through the shared state.
  decl.setBodyDecorators(bodyDecorators, shared);
}

void Decorators::applyBodyDecorators(
    function_ref<LogicalResult(ExprNode *)> process) {
  // Don't run decorators if the declaration is invalid.
  if (decl.isErroneous())
    return;

  ArrayRef<ExprNode *> decoratorExprs = decl.getBodyDecorators(shared);
  while (true) {
    // If there are no decorators left, just exit.
    if (decoratorExprs.empty())
      return;
    if (failed(process(decoratorExprs.front())))
      break;
    decoratorExprs = decoratorExprs.drop_front();
  }

  // Emit the expressions and persist the resulting PValue into the IR. For now,
  // assume that all decorators are "compiler" decorators.
  // TODO: Emit an attempt to call the decorator value.
  SmallVector<TypedAttr> decoPValues;
  decoPValues.reserve(decoratorExprs.size());
  ExprEmitter emitter(shared, decl, EC_Decorator);
  for (auto [i, decorator] : llvm::enumerate(decoratorExprs)) {
    // Make sure we don't have another body decorator.
    if (failed(process(decorator))) {
      if (PValue decoVal = emitter.emitExprPValue(decorator, EC_Decorator))
        decoPValues.push_back(decoVal);
      continue;
    }
    // If the decorator applies, we have an error.
    InflightDiag diag =
        emitError(decorator->getLoc(),
                  "body decorator cannot come after compiler decorator")
        << decorator->getRange();
    ExprNode *bodyDecorator = decoratorExprs[i - 1];
    diag.attachNote(bodyDecorator->getLoc())
        << "previous compiler decorator applied here"
        << bodyDecorator->getRange();
    break;
  }

  cast<ASTDeclInterface>(decl).setDecoratorsAttr(
      DecoratorsAttr::get(getContext(), decoPValues));
}

//===----------------------------------------------------------------------===//
// Function Decl implementation
//===----------------------------------------------------------------------===//

static constexpr const StringLiteral kMainSymbolName = "main";

/// Apply `@export` to an exportable declaration and register it with the shared
/// state to ensure no duplicate exports.
static void applyExport(SMLoc loc, SharedState &shared, ASTDecl &decl,
                        StringRef unmangledName, StringRef aliasName,
                        ExportInterface itf, bool isCExport = false) {
  // Handle the unique case of main. We implicitly export main, so this is
  // simply checking that the user didn't try to export it as something else.
  if (aliasName == kMainSymbolName) {
    if (unmangledName != kMainSymbolName)
      shared.emitError(loc, "only 'main' can be exported as 'main'");
    if (!isa<LIT::FuncOp>(decl))
      shared.emitError(loc, "exported 'main' must be a function");
    return;
  }
  if (unmangledName == kMainSymbolName) {
    shared.emitError(loc, "'main' can only be exported as 'main'");
    return;
  }

  llvm::TypeSwitch<ASTDecl &, void>(decl).Case<LIT::FuncOp, GlobalVarDeclOp>(
      [aliasName](auto op) { op.setLinkageName(aliasName); });
  if (isCExport)
    itf.setCExported();
  else
    itf.setExported();

  shared.declResolver->registerAndCheckExport(aliasName, loc);
}

/// Apply `@export("linkageName")` to an exportable declaration and register it
/// with the shared state to ensure no duplicate exports.
static void applyExport(SMLoc loc, SharedState &shared, ASTDecl &decl,
                        StringRef unmangledName, const CallNode &node,
                        ExportInterface itf) {
  ArrayRef<Operand> operands = node.operands;
  if (operands.empty() || operands.size() > 2) {
    shared.emitError(node.getLoc(), "@export requires 1 or 2 arguments");
    return;
  }

  std::optional<std::string> exportABI;
  std::optional<std::string> aliasName;
  for (const Operand &operand : operands) {
    auto strNode = dyn_cast<StringLiteralNode>(operand.expr);
    if (strNode && operand.isKeyword() && operand.name == "ABI") {
      exportABI = strNode->getValue();
      if (*exportABI != "C") {
        shared.emitError(operand.getLoc(),
                         "only \"C\" ABI is supported at the moment");
        return;
      }
    } else if (strNode && operand.isPositional()) {
      aliasName = strNode->getValue();
    } else {
      shared.emitError(node.getLoc(),
                       "@export requires a string specifying the "
                       "name of the exported symbol");
      return;
    }
  }

  if (exportABI && aliasName && !isCIdentifier(*aliasName)) {
    shared.emitError(loc, *aliasName) << " is not a valid C identifier";
    return;
  }
  applyExport(loc, shared, decl, unmangledName,
              aliasName ? StringRef(*aliasName) : unmangledName, itf,
              exportABI.has_value());
}

/// Now that all the structural properties are determined, perform any special
/// checks over the declaration based on its name.  This happens after
/// decorator processing because that is how defs work in Python.
///
/// If this function detects a problem, it marks the decl as erroneous and
/// resets the SpecialFunctionInfo.
static void verifyFunctionNameBinding(ASTDecl &decl, StringAttr name,
                                      const TypeCheckedFnSignature &tcSignature,
                                      SpecialFunctionInfo &fnInfo) {
  LIT::FuncOp funcOp = cast<LIT::FuncOp>(decl);

  ArrayRef<ParsedArgument> parsedArgs = tcSignature.argList.parsedArgs;
  ArrayRef<Type> argTypes = tcSignature.argTypes;
  auto &shared = tcSignature.paramList.shared;

  // On any semantic error we mark the declaration erroneous - so references to
  // it don't type check, and we clear our special function information.  This
  // reduces cascade errors.
  auto emitErrorLoc = [&](SMLoc loc,
                          const Twine &message = Twine()) -> InflightDiag {
    fnInfo = SpecialFunctionInfo();
    decl.setErroneous();
    return shared.emitError(loc, message);
  };
  auto emitError = [&](const Twine &message = Twine()) -> InflightDiag {
    fnInfo = SpecialFunctionInfo();
    decl.setErroneous();
    return shared.emitError(funcOp.getLoc(), message);
  };

  // If the argument list has a inout result or inout error, ignore it for type
  // checking purposes.
  while (!parsedArgs.empty() && parsedArgs.back().convention ==
                                    ParsedArgument::kConventionByRefResult) {
    parsedArgs = parsedArgs.drop_back();
    argTypes = argTypes.drop_back();
  }

  // If this definition is a struct/class member, compute the self type.
  ASTType selfType;
  constexpr size_t kSelfArgNo = 0;
  if (ASTDecl *parent = decl.getParentDecl();
      parent && isa<StructDeclOp, TraitDeclOp>(*parent)) {
    // The parent decl must be fully resolved in order to resolve any of its
    // members.
    assert(parent->resolvedness == DeclResolvedness::fully);
    selfType = parent->getTypeDeclSelf();
  }

  // Check any special function information.

  // Check that the 'self' argument of a method was specified correctly.
  if (selfType && !funcOp.getIsStatic()) {
    // Implement this as a lambda so we can early exit with 'return'.
    auto checkSelf = [&]() {
      ASTType selfArgType = argTypes[kSelfArgNo];
      const ParsedArgument &selfArg = parsedArgs[kSelfArgNo];

      // Don't check broken args, becaue we don't want redundant diagnostics.
      if (selfArg.isErroneous)
        return;

      // It ok if it exactly matches (typically with a specific convention).
      if (selfType.isEqualCanon(selfArgType))
        return;

      // It is ok if the self type has different parameters than the
      // declaration, this is a form of conditional conformance.
      if (selfType.getWithoutParameters(shared).isEqualCanon(
              selfArgType.getWithoutParameters(shared)))
        return;

      // Otherwise, this is an unrecognized self type. If this is a trait, the
      // explicit self type is very hard to specify in mojo, so we suggest to
      // use 'Self' instead.
      auto diag = emitErrorLoc(selfArg.loc, "'self' argument must have type ");
      if (isa<TraitDeclOp>(*decl.getParentDecl()))
        diag << "'Self' in trait method declaration";
      else
        diag << selfType;
      diag << ", but actually has type " << ASTType(argTypes[kSelfArgNo]);
      selfArg.isErroneous = true;
      if (selfArg.typeExpr)
        diag << selfArg.typeExpr->getRange();
    };

    if (argTypes.empty()) {
      // TODO('def' allows unused arguments): We can/should relax this for
      // 'def' declarations in the future, they should be able to implicit
      // ignore arguments like Python does.
      emitError("self argument must be present in instance method");
    } else {
      checkSelf();
    }
  }

  // Verify the argument count lines up.
  if (fnInfo.kind != SpecialFunctionKind::kNormal) {
    size_t numActualArgs = parsedArgs.size();
    size_t numMin = fnInfo.minNumArguments;
    ssize_t numMax = fnInfo.maxNumArguments;
    if (numMin == size_t(numMax) && numActualArgs != numMin) {
      emitError() << name << " requires " << numMin << " operand"
                  << plural(numMin);
    } else if (numActualArgs < numMin) {
      emitError() << name << " requires at least " << numMin << " operand"
                  << plural(numMin);
    } else if (numMax != -1 && numActualArgs > size_t(numMax)) {
      emitError() << name << " requires at most " << size_t(numMax)
                  << " operand" << plural(numMax);
    }
  }

  // Check other invariants based on method flags.
  if (fnInfo.isInstMethod()) {
    if (!selfType) {
      emitError() << name << " must be a method";
    } else if (funcOp.getIsStatic()) {
      if (!(fnInfo.flags & SpecialFunctionInfo::kImplicitlyStaticMethod))
        emitError("special method may not be a static method");
    } else if (fnInfo.requiresOwnedSelfInstMethod() &&
               parsedArgs[kSelfArgNo].convention !=
                   ParsedArgument::kConventionOwned) {
      emitErrorLoc(parsedArgs[kSelfArgNo].loc, "self argument must be 'owned'")
          << FixIt::insertBeforeToken(parsedArgs[kSelfArgNo].loc, "owned ");
    }
  }

  // Get the user-declared result type, which might be a memory-only type.
  ASTType declaredResultType = tcSignature.resultType;

  // Some functions like __new__ require a Self result type.
  if (fnInfo.flags & SpecialFunctionInfo::kSelfResult &&
      !declaredResultType.isEqualCanon(selfType))
    emitError() << name << " result type must be " << selfType;

  // If the function is required to return None, verify that.
  if (fnInfo.hasNoneResult() && !declaredResultType.isNoneType())
    emitError() << name << " result type must be elided (or None)";

  // Reject special functions declared as throwing when that is invalid.
  if (tcSignature.argList.effects.isThrows() &&
      fnInfo.flags & SpecialFunctionInfo::kCannotRaise) {
    // Specialize the error if raising is implicit because it was defined as a
    // def.
    if (funcOp.isDef()) {
      emitError() << "cannot define " << name
                  << " as 'def'; 'def' implicitly raises"
                  << FixIt::replaceToken(decl.getLoc(), "fn");
    } else {
      emitError() << name << " cannot be declared as raising an exception";
    }
  }

  // Diagnose common errors and handle other special cases.
  switch (fnInfo.kind) {
  default:
    break;
  case SpecialFunctionKind::kNew:
    emitError("'__new__' is not supported on structs; use '__init__' instead");
    break;
  case SpecialFunctionKind::kMLIRI1:
    if (!declaredResultType.mlirType.isSignlessInteger(1))
      emitError() << name << " result type must be __mlir_type.i1";
    break;
  case SpecialFunctionKind::kInit:
  case SpecialFunctionKind::kCopyInit:
  case SpecialFunctionKind::kMoveInit: {
    // The first/self argument is syntactically declared as a by-ref argument,
    // but we need to change it to InitSelf since it is not initialized coming
    // in.
    assert(!parsedArgs.empty() && "arg count already checked above");
    SMLoc selfArgLoc = parsedArgs[0].loc;

    // __init__ methods must take their self argument 'inout' syntactically.
    if (parsedArgs[0].convention != ParsedArgument::kConventionInitSelfResult) {
      auto diag = emitErrorLoc(selfArgLoc, "'self' in struct ")
                  << name << " must be passed 'inout'";
      if (parsedArgs[0].convention == ParsedArgument::kConventionUnspec)
        diag << FixIt::insertBeforeToken(selfArgLoc, "inout ");
    }

    if (fnInfo.kind == SpecialFunctionKind::kCopyInit) {
      if (parsedArgs[1].convention != ParsedArgument::kConventionBorrowed)
        emitErrorLoc(parsedArgs[1].loc,
                     "existing value argument must be passed as borrowed");
    } else if (fnInfo.kind == SpecialFunctionKind::kMoveInit) {
      if (parsedArgs[1].convention != ParsedArgument::kConventionOwned)
        emitErrorLoc(parsedArgs[1].loc,
                     "existing value argument must be passed as owned");
    }
    break;
  }
  }

  // If we have a special function kind and didn't have any errors with it,
  // remember which kind it is.
  if (fnInfo.kind != SpecialFunctionKind::kNormal)
    funcOp.setSpecialFnKind(uint8_t(fnInfo.kind));
}

namespace {
struct FnDecorators : public SharedStateUser {
  FnDecorators(ASTDecl &decl, ASTDecl &sigDecl, SharedState &shared,
               StringRef baseName, TypeCheckedFnSignature &tcSignature)
      : SharedStateUser(shared), decl(decl), sigDecl(sigDecl),
        funcOp(cast<LIT::FuncOp>(decl)), baseName(baseName),
        tcSignature(tcSignature) {}

  /// Apply a function signature decorator.
  LogicalResult apply(ExprNode *decorator);

private:
  void applyStaticMethod(const DeclRefNode &node);
  void applyCopyOrMoveCapture(const CallNode &node, bool isMove,
                              StringRef decorator);
  void applyLLVMMetadata(const CallNode &node);
  void applyNamedResult(const CallNode &node);

  ASTDecl &decl;
  ASTDecl &sigDecl;
  LIT::FuncOp funcOp;
  StringRef baseName;
  TypeCheckedFnSignature &tcSignature;
};
} // namespace

LogicalResult FnDecorators::apply(ExprNode *decorator) {
  // Process all the decorators we know about.
  if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
    if (declRef->spelling == "export")
      applyExport(decorator->getLoc(), shared, decl, baseName, baseName,
                  funcOp);
    else if (declRef->spelling == "staticmethod")
      applyStaticMethod(*declRef);
    else if (declRef->spelling == "always_inline")
      funcOp.setInlineLevel(InlineLevel::Always);
    else if (declRef->spelling == "no_inline")
      funcOp.setInlineLevel(InlineLevel::Never);
    else if (declRef->spelling == "parameter")
      tcSignature.argList.effects.setCapturing();
    else
      return failure();
    return success();
  }

  // `x()` forms.
  if (auto callNode = dyn_cast<CallNode>(decorator)) {
    if (auto declRef = dyn_cast<DeclRefNode>(callNode->callee)) {
      // @always_inline("nodebug")
      if (declRef->spelling == "always_inline" &&
          callNode->operands.size() == 1 &&
          callNode->operands[0].isPositionalStringLiteral("nodebug"))
        funcOp.setInlineLevel(InlineLevel::AlwaysNoDebug);
      else if (declRef->spelling == "export")
        applyExport(decorator->getLoc(), shared, decl, baseName, *callNode,
                    funcOp);
      else if (declRef->spelling == "__move_capture")
        applyCopyOrMoveCapture(*callNode, /*isMove=*/true, declRef->spelling);
      else if (declRef->spelling == "__copy_capture")
        applyCopyOrMoveCapture(*callNode, /*isMove=*/false, declRef->spelling);
      else if (declRef->spelling == "__llvm_metadata")
        applyLLVMMetadata(*callNode);
      else if (declRef->spelling == "__named_result")
        applyNamedResult(*callNode);
      else
        return failure();
      return success();
    }
  }
  return failure();
}

void FnDecorators::applyStaticMethod(const DeclRefNode &node) {
  // This decorator only applies to methods of structs and traits.
  if (!isa<StructDeclOp, TraitDeclOp>(*decl.getParentDecl())) {
    emitError(node.getLoc(), "only methods on structs may be declared static");
    return;
  }
  funcOp.setIsStatic(true);
}

void FnDecorators::applyCopyOrMoveCapture(const CallNode &node, bool isMove,
                                          StringRef decoratorSpelling) {
  // HACK(#16110): Need to implement proper capture list syntax rather than rely
  // on a special decorator.
  for (const Operand &operand : node.operands) {
    auto *declRef = dyn_cast<DeclRefNode>(operand.expr);
    if (!declRef) {
      emitError(operand.getLoc(), "'@")
          << decoratorSpelling << "' expected a declaration";
      continue;
    }
    LookupResult lookup = shared.lookupAndResolveDecl(
        declRef->spelling, declRef->getLoc(), *decl.getParentDecl(),
        /*searchParentScopes=*/true);
    if (lookup.isErroneous())
      continue;

    ArrayRef<ASTDecl *> decls = lookup.getIfSuccess();
    if (decls.empty()) {
      emitError(declRef->getLoc(), "cannot capture unknown value '")
          << declRef->spelling << "'";
      continue;
    }
    if (decls.size() != 1) {
      emitError(declRef->getLoc(), "cannot capture overloaded value '")
          << declRef->spelling << "'";
      continue;
    }

    // Emit an immutable copy of the captured declaration.
    LIT::FuncOp parentOp = funcOp->getParentOfType<LIT::FuncOp>();
    if (!parentOp) {
      emitError(declRef->getLoc(), "'@")
          << decoratorSpelling
          << "' decorator only applies to nested functions";
      return;
    }

    ExprEmitter emitter(shared, *decl.getParentDecl(), OpBuilder(funcOp));
    RValue captureRVal;
    if (!isMove) {
      // For a copy capture, just emit the value reference as an RValue, which
      // will make sure to copy it.
      captureRVal = emitter.emitExprRValue(declRef, EC_Capture);
      if (!captureRVal)
        return;
      // HACK: This only has the intended effect of "immortalizing" a
      // register-passable value by creating an SRValue.
      if (!captureRVal.getType().isTrivial(node.getLoc(), shared)) {
        emitError(node.getLoc(), "TODO: @__copy_capture only works as intended "
                                 "with trivial register-passable types");
      }
    } else {
      // For a move capture, we emit this with an implicit transfer.
      // HACK(#16110): This transfers ownership without an explicit `^` from
      // the user, because we don't have capture list syntax.
      UnaryOpNode transfer(ExprNode::kTransfer, declRef->getLoc(), declRef);
      captureRVal = emitter.emitExprRValue(&transfer, EC_Capture);
    }

    // We can only capture dynamic values so materialize param expressions.
    if (auto pval = captureRVal.getIfPValue()) {
      if (pval.getType().isRegisterPassable(decl.getLoc(), shared))
        captureRVal = emitter.emitSRValue({captureRVal, declRef}, EC_Capture);
      else
        captureRVal = emitter.emitMRValue({captureRVal, declRef}, EC_Capture);
    }
    if (!captureRVal)
      return;

    // How is this transfering the RValue into the closure?
    DeclIRValue resultVal;
    if (auto srVal = captureRVal.getIfSRValue())
      resultVal = srVal;
    else {
      assert(captureRVal.getIfMRValue() && "Unknown RValue kind");
      resultVal = captureRVal.getIfMRValue();
    }

    // Bind the name in the scope so further references don't look like
    // reference captures.
    // FIXME: It would be cleaner to have an explicit representation of this,
    // e.g. an op that produces a ref to the value in the capture list.  Instead
    // we are still forming references outside the closure for things that are
    // copied and moved into the closure.
    emitter.getDeclResolver().addFullyResolvedDecl(resultVal, declRef->spelling,
                                                   sigDecl.getLoc(), &sigDecl);

    // Both move and copy captures are the same here - a move capture just does
    // the transfer above to generate its RValue.
    shared.addCaptureToScope(decl, decls.front(),
                             Capture(captureRVal, Capture::kCopy));
  }
}

void FnDecorators::applyLLVMMetadata(const CallNode &node) {
  NamedAttrList attrs;
  ExprEmitter emitter(shared, sigDecl, EC_Decorator);
  for (Operand value : node.operands) {
    if (!value.name) {
      emitError(value.getLoc(), "LLVM metadata requires a name");
      continue;
    }
    if (PValue attr = emitter.emitExprPValue(value.expr, EC_Decorator))
      attrs.append(value.name, attr);
  }
  funcOp.setLLVMMetadataAttr(attrs.getDictionary(getContext()));
}

void FnDecorators::applyNamedResult(const CallNode &node) {
  DeclRefNode *dre;
  if (node.operands.size() != 1 ||
      !(dre = dyn_cast<DeclRefNode>(node.operands.front().expr))) {
    emitError(node.getLoc(), "`@__named_result` expected an identifier");
    return;
  }
  MutableArrayRef<ParsedArgument> args = tcSignature.argList.parsedArgs;
  if (args.empty() ||
      args.back().kgenConvention != ArgConvention::ByRefResult) {
    // TODO: We should make this decorator force the function to have a
    // `byref_result` instead, even for regpassable types.
    emitError(decl.getLoc())
        << "named results can only be used on functions with in-memory "
           "results, result type "
        << tcSignature.resultType << " is register-passable";
    return;
  }
  auto name = StringAttr::get(getContext(), dre->spelling);
  funcOp.setNamedResultAttr(name);
  args.back().name = name;
}

/// Process an extensibility decorator by generating additional trait binding
/// information about each argument and result type.
static void processExtensibilityDecorator(SharedState &shared, ASTDecl &decl,
                                          const ExprNode *decorator) {
  StringRef spelling;
  if (auto callNode = dyn_cast<CallNode>(decorator))
    if (auto declRef = dyn_cast<DeclRefNode>(callNode->callee))
      spelling = declRef->spelling;
  if (spelling.empty())
    return;

  using namespace MOGGPreElab::Decorators;
  if (!llvm::is_contained(
          {REGISTER_KERNEL, REGISTER_OVERRIDE, REGISTER_PUBLIC_OVERRIDE},
          spelling))
    return;

  // For each argument and result type, generate the set of explicit trait
  // conformances and witness tables. We need to dig out the declared argument
  // and result types.
  auto func = cast<LIT::FuncOp>(decl);
  if (!isa<FileModuleOp>(func->getParentOp())) {
    shared.emitError(decl.getLoc(), "@")
        << spelling << " is only supported on top-level functions";
    return;
  }

  LITSignatureType sig = func.getFullSignature();
  ArrayRef<Type> sigArgTypes = func.getFunctionType().getInputs();
  ASTType resultType = func.getUserResultType();
  // Reduce `sigArgTypes` to just the array of declared arguments.
  if (sig.hasMemoryOnlyResult())
    sigArgTypes = sigArgTypes.drop_back();
  // Drop the error slot if there is one.
  if (sig.isThrows())
    sigArgTypes = sigArgTypes.drop_back();

  // Extract the declared argument types.
  SmallVector<ASTType> argTypes;
  for (auto [idx, argType] : llvm::enumerate(sigArgTypes)) {
    Type type = argType;
    ArgConvention conv = sig.getArgConvention(idx);
    // Handle vararg kinds.
    if (sig.isPosVarArg(idx)) {
      auto variadic = cast<VariadicType>(type);
      type = variadic.getElementType();
      conv = variadic.getConvention();
    } else if (sig.isKwVarArg(idx)) {
      // Don't need to unpack anything. We treat the whole dictionary as the
      // value type.
    } else if (sig.isPackVarArg(idx)) {
      // For variadic packs, we don't have a type instance but we have the
      // metatype.
      Type metatype =
          ASTType(type).getVariadicPackInfo().getVariadicElementType();
      type = ParamRefType::get(UnknownAttr::get(metatype));
      conv = ArgConvention::BorrowedInReg;
    }
    if (SignatureType::hasAddress(conv))
      type = ASTType(type).getReferenceElementType();
    argTypes.push_back(type);
  }

  ExprEmitter emitter(shared, decl, EC_Type);
  auto generateConformancesImpl = [&](ASTType type, Location loc) {
    SyntheticNode node(shared.diags.convertLocToSMLoc(loc));
    ASTType metatype = type.getMetaType();
    // If this is already a trait type, then we know the value is going to be
    // resolved to a type constant.
    SmallVector<TypedAttr> conformances;
    if (auto trait = dyn_cast_or_null<TraitType>(metatype)) {
      conformances.push_back(PValue(type));
      return conformances;
    }
    // If this is some MLIR type, generate the default trait conformances.
    if (!metatype || isa<TypeType>(metatype)) {
      for (StringRef traitName : {"AnyType", "Copyable", "Movable"}) {
        auto traitDecl = cast_or_null<TraitDeclOp>(
            shared.lookupBuiltinTrait(traitName, &decl, node.getLoc()));
        if (!traitDecl)
          continue;
        TraitType trait = traitDecl.bindReference();
        if (PValue result =
                emitter.bindMLIRTypeToTrait({PValue(type), node}, trait))
          conformances.push_back(result);
      }
      return conformances;
    }
    // Otherwise, generate bindings for each explicit conformance.
    assert(isa<AnyStructType>(metatype));
    auto structDecl = cast<StructDeclOp>(metatype.getDecl(shared));
    for (TypeLineageAttr parentAttr : structDecl.getParentTypes()) {
      auto trait = cast<TraitType>(parentAttr.getType());
      if (PValue result = emitter.emitMetaTypeToTraitConversion(
              {PValue(type), node}, trait))
        conformances.push_back(result);
    }
    return conformances;
  };
  auto generateConformances = [&](ASTType type, Location loc) {
    return ParameterExprArrayAttr::get(loc.getContext(),
                                       generateConformancesImpl(type, loc));
  };

  SmallVector<Attribute> argConformances;
  Attribute resConformances = generateConformances(resultType, func.getLoc());
  for (auto [idx, argType] : llvm::enumerate(argTypes)) {
    argConformances.push_back(
        generateConformances(argType, func.getArgument(idx).getLoc()));
  }

  NamedAttrList attrs = func->getAttrDictionary();
  attrs.set(MOGGPreElab::MOGG_ARGUMENT_CONFORMANCES,
            ArrayAttr::get(shared.getContext(), argConformances));
  attrs.set(MOGGPreElab::MOGG_RESULT_CONFORMANCES, resConformances);
  func->setAttrs(attrs.getDictionary(shared.getContext()));
}

/// Given the lexical context of a function, return true if the default bit
/// for the function is capturing.
static bool isCapturingByDefault(LIT::FuncOp funcOp, StructDeclOp parent,
                                 ArrayRef<ParamDeclAttr> paramDecls) {
  // Any function that contains a capturing closure as a parameter is itself
  // capturing, include parent struct parameters.
  mlir::AttrTypeWalker walker;
  walker.addWalk([](SignatureType sig) {
    if (sig.isCapturing())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return llvm::any_of(
      llvm::concat<const ParamDeclAttr>(paramDecls, parent ? parent.getParams()
                                                           : std::nullopt),
      [&](ParamDeclAttr decl) { return walker.walk(decl).wasInterrupted(); });
}

std::pair<SmallVector<ParamDeclRefAttr>, LITSignatureType>
DeclResolver::createSelfContainedSignature(LITSignatureType original) {
  // Collect the subset of referenced parameters. Use a set vector to keep the
  // order deterministic.
  llvm::SetVector<ParamDeclRefAttr, SmallVector<ParamDeclRefAttr>> capturedRefs;
  original.walk([&](ParamDeclRefAttr ref) { capturedRefs.insert(ref); });

  SmallVector<ParamDeclRefAttr> captured = capturedRefs.takeVector();
  // Unbind the N capture parameters, creating a new signature with N new input
  // parameters prepended.
  // TODO: what if we capture a variadic?
  SmallVector<bool> variadicMask(captured.size(), false);
  auto unbound = LITSignatureType::prependParams(
      original,
      llvm::map_to_vector(
          captured,
          [](ParamDeclRefAttr ref) { return ParamDeclAttr::get(ref); }),
      variadicMask);
  return {std::move(captured), unbound};
}

static MLValue emitClosureInstance(SharedState &shared, ASTDecl &nestedFnDecl,
                                   SMLoc loc) {
  LIT::FuncOp nestedFn = cast<LIT::FuncOp>(nestedFnDecl);
  StringAttr fnName = nestedFn.getSourceNameAttr();
  Location mlirLoc = shared.translateLocation(loc);
  if (shared.diBuilder)
    mlirLoc = shared.diBuilder->createScopedLoc(mlirLoc);

  // Save the insertion point before closure creation since closure creation
  // nukes the nested function.
  ImplicitLocOpBuilder builder(mlirLoc, shared.getContext());
  builder.setInsertionPointAfter(nestedFn);
  OpBuilder::InsertPoint insertPoint = builder.saveInsertionPoint();
  ASTDecl *moduleDecl = nestedFnDecl.getNearestDeclOfType<FileModuleOp>();

  auto [capturedRefs, wrapperSig] =
      DeclResolver::createSelfContainedSignature(nestedFn.getSignature());
  if (!wrapperSig)
    return {};
  StructDeclOp closureWrapper =
      shared.getOrCreateClosureWrapper(loc, wrapperSig, moduleDecl);
  if (!closureWrapper)
    return {};

  // In order to emit a closure instance, we need the captures and in order to
  // compute the captures we need to resolve the body.
  if (failed(shared.declResolver->resolveFully(nestedFnDecl, loc)))
    return {};
  // Find all parameter captures in the function body.
  ParameterCollector::Analysis collectorCache;
  ParameterUseDefGraph graph(nestedFn.getBodyRegion());
  graph.calculate(collectorCache);
  SmallVector<ParamDeclRefAttr> paramCaptures =
      graph.usesFromAbove.takeVector();

  // Don't capture lifetime parameters, they carry no state, and are often
  // implicit lifetimes of captured references, which aren't used in the body
  // anyway.
  paramCaptures.erase(
      llvm::remove_if(paramCaptures,
                      [&](auto p) { return isa<LifetimeType>(p.getType()); }),
      paramCaptures.end());

  // Create an instance of the closure implementation in the parent function
  // right after the nested function definition.
  ClosureEmitter emitter(*moduleDecl, shared);
  StructDeclOp closureImpl =
      emitter.replaceNestedFunctionWithClosureImplStructDecl(
          loc, nestedFnDecl, paramCaptures, wrapperSig);

  emitter.createWrapperInitWithImpl(closureWrapper, closureImpl, loc);

  builder.restoreInsertionPoint(insertPoint);

  ExprEmitter exprEmitter(shared, *nestedFnDecl.getParentDecl(), builder);
  SyntheticNode node(loc);

  // Pass all the captured values into the initializer.  In the case of a move
  // capture, this will be an RValue for the thing captured, transfering to the
  // owned argument in the initializer.
  CallOperands closureImplInitArgs;
  for (auto &[_, capture] : shared.getCaptureRangeInScope(nestedFnDecl))
    closureImplInitArgs.add({capture.getValue(), node});

  // Create Closure Impl type by adding captured parameters to the ClosureImpl
  // DeclType.
  ValueDest closureDest(EC_Closure);
  Type closureImplType = closureImpl.bindReference(llvm::map_to_vector(
      paramCaptures, [](ParamDeclRefAttr ref) -> TypedAttr { return ref; }));

  CValue value = exprEmitter.emitConstructorCall(
      ASTType(closureImplType), std::move(closureImplInitArgs), node,
      CallSyntax::kTypeCall, closureDest, /*allowImplicitConversion=*/false);

  // Emit the Closure Wrapper instance.
  VarDeclOp var = exprEmitter.emitVarDecl(
      fnName, UnresolvedType::get(shared.getContext()),
      exprEmitter.translateLocation(loc), VarDeclKind::Var);
  ValueDest closureWrapperDest(var, EC_VarInit);

  CallOperands closureWrapperInitArgs;
  closureWrapperInitArgs.add({value, node});

  // Create the ClosureWrapper type by binding parent parameters to the
  // ClosureWrapper type.
  // TODO: Handle partial binding.
  LIT::StructType closureWrapperType =
      closureWrapper.bindReference(llvm::map_to_vector(
          capturedRefs, [](ParamDeclRefAttr ref) -> TypedAttr { return ref; }));

  exprEmitter.emitConstructorCall(ASTType(closureWrapperType),
                                  std::move(closureWrapperInitArgs), node,
                                  CallSyntax::kTypeCall, closureWrapperDest,
                                  /*allowImplicitConversion=*/false);
  return MLValue(var);
}

/// funcdef   ::=  [decorators] def_or_fn identifier [param_signature]
///                "(" [argument_list] ")" ["->" expression] ":" suite
/// def_or_fn ::= "def" | "fn"
///
LogicalResult DeclResolver::resolveSignature(LIT::FuncOp funcOp, Lexer &lexer,
                                             ASTDecl &decl) {
  ParserBase p(shared, lexer);
  auto decoratorExprs = p.parseDecorators(decl);
  assert(p.getToken().isAny(Token::kw_async, Token::kw_def, Token::kw_fn) &&
         "not a function definition?");
  bool isAsync = p.consumeIf(Token::kw_async);
  bool isDef = p.getToken().is(Token::kw_def);
  p.consumeToken();

  StringAttr baseName;
  SMLoc identifierLoc;
  if (p.parseIdentifier(baseName, "expected function name", &identifierLoc))
    return failure();

  // The function signature is a self-contained scope where the input and result
  // parameters of the function are visible by all types.  We must use a
  // temporary declaration here (with an empty name) because we don't want
  // references to the function itself to resolve to a fully-resolved decl, but
  // we need a fully-resolved decl for incremental lookups within the scope to
  // work out.
  ASTDecl &sigDecl = addFullyResolvedDecl(funcOp.getOperation(), StringAttr(),
                                          decl.getLoc(), decl.getParentDecl());

  // Parse declared meta parameters and add them to the current scope.
  ParsedParamList parsedParamList;

  // Add the parameters to the symbol table, and resolve their types.  We
  // add all of these after generic signature parsing so types used in the
  // signature list resolve to enclosing scopes, and we add them before the
  // value signature list so the types and parameters can resolve to the bound
  // values.
  if (parsedParamList.parseOptionalParameters(p, ArgListKind::kParamList))
    return failure();
  TypeCheckedParamList paramList(parsedParamList.params, sigDecl, shared);

  // Parse the function signature next.
  ParsedArgumentList fnSignature;
  // Set up the known effects.
  if (isAsync)
    fnSignature.effects.setAsync(true);
  if (isDef)
    fnSignature.effects.setThrows();

  // Parse the argument list next if present.
  if (fnSignature.parseArgumentListAndEffects(p, ArgListKind::kArgList))
    return failure();

  // Parse the result type if present.
  ExprNode *resultTypeExpr = nullptr;
  ExprNode *resultRefLifetimeExpr = nullptr;
  SMLoc resultLoc = p.getToken().getLoc();
  if (p.consumeIf(Token::minus_greater)) {
    // Parse a result reference if present.
    (void)p.parseRefSpecifier(resultRefLifetimeExpr);

    // Parse the result type expression.
    // If this result parsing fails, then we just continue on as if none was
    // specified.
    (void)p.parseExpression(resultTypeExpr);
  }

  // Emit the argument and result types.
  SpecialFunctionInfo fnInfo = SpecialFunctionInfo::get(baseName);

  TypeCheckedFnSignature tcSignature(paramList, fnSignature, resultTypeExpr,
                                     resultRefLifetimeExpr, resultLoc, isDef,
                                     &decl, fnInfo);

  // If any of the arguments had an error or if the result type is a type check
  // error, then we won't allow forming a reference to this function.
  if (isa<TypeCheckErrorType>(tcSignature.resultType.mlirType) ||
      llvm::any_of(fnSignature.parsedArgs,
                   [](ParsedArgument &arg) { return arg.isErroneous; }))
    decl.setErroneous();

  auto structDecl = dyn_cast<StructDeclOp>(decl.getParentDecl());
  if (isCapturingByDefault(funcOp, structDecl, paramList.paramDeclAttrs))
    fnSignature.effects.setCapturing();

  // Now that we have figured out the lexical structure, allow decorators to
  // take a crack at the signature.
  FnDecorators fnDecorators(decl, sigDecl, shared, baseName, tcSignature);
  Decorators(decl, shared)
      .applySignatureDecorators(decoratorExprs, [&](ExprNode *decorator) {
        return fnDecorators.apply(decorator);
      });

  // Propagate errors and the parsed decls in the signature.
  decl.takeDecls(sigDecl);

  // Now that all the structural properties are determined, perform any
  // name-binding specific checks over the declaration.  This happens after
  // decorator processing because that is how defs work in Python.  This also
  // fills in any implicitly declared types.
  verifyFunctionNameBinding(decl, baseName, tcSignature, fnInfo);

  // Now that we've processed the signature, bail if we had a missing colon.
  if (p.parseToken(Token::colon, "expected ':' in function definition"))
    return failure();

  // Finally now that the full signature has been resolved, build our IR.

  // Handle argument effects and build the ASTDecls for the arguments.
  OpBuilder builder = decl.getDeclEndBuilder();
  NamedAttrList attrs = funcOp->getAttrDictionary();

  // Compute the signature of the function.
  LITSignatureType signature = tcSignature.getLITSignatureType();
  if (!signature)
    return failure();

  // The implicitLifetimeDecls don't affect the signature, but they do get
  // prepended onto the paramDecls list.
  ParamDeclArrayAttr paramsArrayAttr;
  if (tcSignature.implicitLifetimeDecls.empty()) {
    paramsArrayAttr =
        builder.getAttr<ParamDeclArrayAttr>(paramList.paramDeclAttrs);
  } else {
    SmallVector<ParamDeclAttr> mergedParams;
    llvm::append_range(mergedParams, tcSignature.implicitLifetimeDecls);
    llvm::append_range(mergedParams, paramList.paramDeclAttrs);
    paramsArrayAttr = builder.getAttr<ParamDeclArrayAttr>(mergedParams);
  }

  attrs.set(funcOp.getParamsAttrName(), paramsArrayAttr);
  attrs.set(funcOp.getFunctionTypeAttrName(),
            TypeAttr::get(tcSignature.getFunctionType()));

  // Now that the FunctionType is set to the pretty type that includes implicit
  // lifetimes, we strip off the named lifetime decl references and replace them
  // with indices.
  signature = signature.replaceImplicitLifetimesWithIndexes(
      tcSignature.implicitLifetimeDecls);
  attrs.set(funcOp.getSignatureAttrName(), TypeAttr::get(signature));

  // Set the symbol to the mangled name and check for redefinition.
  attrs.set(funcOp.getSymNameAttrName(),
            getMangledName(baseName, *decl.getParentDecl(), signature));
  attrs.set(funcOp.getSourceNameAttrName(), baseName);

  // Remove the temporary "sym_namex" attribute set up in FuncOp::build, see
  // that method for an explanation.
  attrs.erase("sym_namex");

  // Bulk update the attributes.
  funcOp->setAttrs(attrs.getDictionary(funcOp.getContext()));

  // Set the symbol and notice if we are redeclaring something.
  if (Operation *existing = finalizeFuncSignature(funcOp, decl)) {
    const char *errorMessage = nullptr;
    auto existingFunc = cast<LIT::FuncOp>(existing);

    // We need to compare the (name erased) user result types, since memory-only
    // types may result in `!kgen.none` in the mlir signature result.
    auto resTy = ASTType(signature).getSignatureUserResultType();
    auto existingResTy =
        ASTType(existingFunc.getSignature()).getSignatureUserResultType();
    if (!resTy.isEqualCanon(existingResTy))
      errorMessage = " cannot overload on return type only";
    else
      errorMessage = " with identical signature";

    // On redefinition this is an overload of the same name.
    if (errorMessage) {
      auto diag = p.emitError(funcOp.getLoc(), "redefinition of function ")
                  << baseName << errorMessage;
      diag.attachNote(existing->getLoc()) << "previous definition here";
      decl.setErroneous();
    }
  }

  // If have a main function, fn main(), export it automatically.
  if (!structDecl && baseName == kMainSymbolName)
    getDeclResolver().exportMain(decl);

  // Generate a debug subprogram for this function.
  shared.setLocationDebugScope(funcOp);

  for (auto [parsedArg, bbArg] :
       llvm::zip(fnSignature.parsedArgs, funcOp.getBody()->getArguments()))
    bbArg.setLoc(shared.diags.translateLocation(parsedArg.loc));

  // Upon fully resolving a nonparametric closure, immediately materialize it
  // as a runtime value. It cannot be used as a parameter.
  if (funcOp->getParentOfType<LIT::FuncOp>()) {
    if (!signature.isCapturing()) {
      // Fully resolve the body so we can swap the IR value of the decl. Later
      // on, we will need this to determine the capture signature.
      decl.resolvedness = DeclResolvedness::fully;
      if (failed(resolveBody(funcOp, lexer, decl)))
        return failure();

      // If the function doesn't actually capture anything, don't demote it to
      // a runtime value.
      if (signature.isEscaping() ||
          !shared.getCaptureRangeInScope(decl).empty()) {
        if (!paramList.paramDeclAttrs.empty())
          return emitError(funcOp.getLoc(),
                           "TODO: closures cannot have parameters");

        // Emit closure structures necessary for instantiating an escaping
        // closure
        funcOp.setSignature(
            signature.getWithFnEffects(signature.getFnEffects().setEscaping()));
        MLValue instance = emitClosureInstance(shared, decl, decl.getLoc());
        if (!instance)
          return failure();
        decl.irValue = instance;
      } else {
        funcOp.setParamDeclAttr(
            ParamDeclAttr::get(funcOp.getSymNameAttr(), signature));
        funcOp.removeSymNameAttr();
      }
    } else {
      funcOp.setParamDeclAttr(
          ParamDeclAttr::get(funcOp.getSymNameAttr(), signature));
      funcOp.removeSymNameAttr();
    }
  }

  shared.notifyListenerOnFunctionDecl(decl, identifierLoc);
  return success();
}

/// Given a value of !kgen.variadic<..> construct a VariadicList and return
/// the variable declaration holding it.
static VarDeclOp makeVarArgWrapper(SRValue argValue, StringAttr argName,
                                   ASTDecl &parentDecl, ExprEmitter &emitter,
                                   SMLoc loc) {

  // Determine if this is VariadicList or VariadicListMem, and get it.
  auto variadicType = cast<VariadicType>(argValue.getType());
  ASTType variadicEltType = variadicType.getElementType();
  auto refType = dyn_cast<RefType>(variadicEltType);
  ASTType varListType =
      emitter.shared.getBuiltinVariadicListType(parentDecl, loc, (bool)refType);
  if (varListType.isTypeCheckErrorType())
    return {};

  // If this is a variadic of in-memory values that might not have lifetimes,
  // forbid taking the lifetime of the values.
  if (refType && ASTType(refType.getElementType())
                     .mightBeRegisterPassable(loc, emitter.shared)) {
    auto newRefType = refType.getWithLifetime(
        InvalidRefLifetimeAttr::get(refType.isMutable()));
    argValue = emitter.builder->create<RebindOp>(
        emitter.translateLocation(loc),
        variadicType.getWithElementType(newRefType), argValue);
  }

  // Emit a VarDeclOp: VaridicListMem needs a lifetime for its self accesses.
  // This also provides a user name for the argument.
  auto mlirLoc = emitter.translateLocation(loc);
  VarDeclOp varDecl =
      emitter.emitVarDecl(argName, UnresolvedType::get(emitter.getContext()),
                          mlirLoc, VarDeclKind::Arg);

  // Create an instance of the VariadicList, passing in the !kgen.variadic.  The
  // type checker will deduce all the parameters.
  ValueDest ctorDest(varDecl, EC_VarArgArgument);
  CallOperands operands;

  // Expr to provide location information.
  SyntheticNode srcLocNode(loc);
  operands.add({argValue, &srcLocNode});
  CValue ctorResult =
      emitter.emitConstructorCall(varListType, std::move(operands), &srcLocNode,
                                  CallSyntax::kTypeCall, ctorDest);
  if (!ctorResult) {
    ctorDest.resetForError();
    return {};
  }
  return varDecl;
}

ParseResult DeclResolver::resolveBody(LIT::FuncOp funcOp, Lexer &lexer,
                                      ASTDecl &decl) {
  // Push the debug scope for this function if necessary so that nested
  // operations have proper debug info.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(funcOp.getLocScope());

  // Set up information about value arguments.
  Block *bodyBlock = funcOp.getBody();
  ExprEmitter emitter(shared, decl, OpBuilder::atBlockEnd(bodyBlock));

  LITSignatureType funcSignature = funcOp.getSignature();

  // Set up the body of the fn/def, creating declarations for the value
  // parameters and adding them to the symbol table.
  for (auto [argIdxX, bbArg, convention] :
       llvm::enumerate(funcOp.getBody()->getArguments(),
                       funcSignature.getArgConventions())) {
    size_t argIdx = argIdxX;

    StringAttr argName = funcSignature.getArgName(argIdx);
    // Don't bind byref-result, it is handled specially by 'return'.
    if (SignatureType::isResultSlot(convention))
      continue;

    // Figure out which decl corresponds to this argument so we can finish it.
    ArrayRef<ASTDecl *> argDeclList = decl.lookupInCurrentScope(argName);
    assert(argDeclList.size() == 1 &&
           "Argument should be added by signature resolution");
    ASTDecl &argDecl = *argDeclList[0];

    // The argDecl is already set up with a basic representation when the
    // function signature was type checked.  We have to hack it a bit for
    // variadics and other cases that aren't modeled right.
    // TODO: Move variadics to be formed on the caller side not the callee side.

    // This function sets the argument decl to be fully resolved with the
    // specified IR representation.
    auto setDecl = [&](DeclIRValue value) {
      argDecl.setIRValue(std::move(value));
      shared.notifyListenerOnArgumentDecl(argDecl, argName, argDecl.getLoc());
    };

    // VarArg arguments are projected into a VariadicList.
    if (funcSignature.isPosVarArg(argIdx)) {
      auto declOp =
          makeVarArgWrapper(bbArg, argName, decl, emitter, argDecl.getLoc());
      if (!declOp)
        return failure();
      declOp.setArgShadowIndex(bbArg.getArgNumber());
      setDecl(DeclIRValue(declOp));
      continue;
    }

    // If this is an owned argument in a register, we project it into a vardecl
    // so that it is mutable in the callee.
    if (convention == ArgConvention::OwnedInReg) {
      VarDeclOp declOp = emitter.makeArgLValueVarSlot(SRValue(bbArg), argName,
                                                      argDecl.getLoc());
      if (declOp) {
        declOp.setArgShadowIndex(bbArg.getArgNumber());
        setDecl(MLValue(declOp));
      } else {
        argDecl.setErroneous();
      }
      continue;
    }

    // Ref convention works with registers and def functions without any funny
    // business.
    if (convention == ArgConvention::Ref) {
      // TODO: Merge MBValue and MLValue.
      if (convention == ArgConvention::BorrowedInMem)
        setDecl(MBValue(bbArg));
      else
        setDecl(MLValue(bbArg));
      continue;
    }

    // Borrowed arguments in 'def's get a special wrapper that allows them to be
    // mutable.
    auto setBorrowedDecl = [&](auto argBValue) {
      // Don't bother 'fn' arguments.
      if (!funcOp.isDef())
        return setDecl(argBValue);

      // Insert the def argument wrapper to make it lazily mutable on demand.
      setDecl(RCRef<DefArgumentWrapperDLValue>::create(
          &argDecl, argBValue, argBValue.getRValueType(), argIdx));
    };

    // If this is an MValue argument whose underlying type could be a register
    // type (e.g. because it is generic) then we cannot allow arbitrary user
    // defined references to bind to the argument.  These arguments will be
    // lowered late (after elaboration) by argument convention lowering to be
    // direct register passes, so any references will be invalid.
    //
    // To handle this, we cast the value to a marker lifetime which cannot be
    // bound to Reference.
    if (SignatureType::hasAddress(convention)) {
      auto argRefType = cast<RefType>(bbArg.getType());
      if (ASTType(argRefType.getElementType())
              .mightBeRegisterPassable(argDecl.getLoc(), shared)) {
        // Cast away our lifetime since the body can't use it.
        auto expectedType = argRefType.getWithLifetime(
            InvalidRefLifetimeAttr::get(argRefType.isMutable()));
        Value castedArg = emitter.builder->create<RebindOp>(
            emitter.translateLocation(argDecl.getLoc()), expectedType, bbArg);
        if (convention != ArgConvention::BorrowedInMem) {
          setDecl(MLValue(castedArg)); // owned or inout
          continue;
        }

        // Otherwise normal MBValue argument.
        setBorrowedDecl(MBValue(castedArg));
        continue;
      }

      if (convention == ArgConvention::BorrowedInMem)
        setBorrowedDecl(MBValue(bbArg)); // borrowed
      else
        setDecl(MLValue(bbArg)); // owned or inout
      continue;
    }

    // Otherwise, this is a borrowed register value.
    assert(convention == ArgConvention::BorrowedInReg);
    setBorrowedDecl(SBValue(bbArg));
  }

  // If the function has a named result slot, bind it here.
  if (StringAttr namedResult = funcOp.getNamedResultAttr()) {
    assert(funcSignature.hasMemoryOnlyResult() && "already checked");
    Value result = funcOp.getArguments().back();
    addFullyResolvedDecl(MLValue(result), namedResult, decl.getLoc(), &decl);
  }

  Block *body = funcOp.getBody();

  Operation *lastOpIterBefore =
      body->empty() ? nullptr : &body->getOperations().back();

  // With all the argument declarations set up, we can resolve the body of the
  // function.
  if (ParserBase(shared, lexer).parseSuite(decl))
    return failure();

  // If this decl or a parent is erroneous, return before emitting.  There is no
  // point to emitting after errors, and we might trip assertions because
  // erroneous decls don't respect invariants.
  if (decl.isErroneous() || decl.getParentDecl()->isErroneous())
    return success();

  // Function body is empty if the body block is empty or the last operation in
  // the block is still the same as it was before parseSuite.
  bool emptyBody =
      body->empty() || (lastOpIterBefore == &body->getOperations().back());

  // Emit a default "return None" if the function returns nothing, and add an
  // endop terminator.

  if (emptyBody && isa<TraitDeclOp>(*decl.getParentDecl())) {
    // Wipe out the body which may already contain some compiler generated
    // operations for handling argLValueVarSlot.
    body->walk([&](LIT::VarDeclOp op) {
      // Remove the value from parent's declsInScope first before destroying the
      // value.
      auto iter = decl.declsInScope.find(op.getNameAttr());
      if (iter != decl.declsInScope.end())
        iter->second.clear();
    });

    // Clear out any decls in the scope that reference IR in the body.
    for (auto &[name, decls] : decl.getDeclsInScope()) {
      for (ASTDecl *decl : decls) {
        TypeSwitch<DeclIRValue>(decl->getIRValue())
            .Case<SRValue, SBValue, MBValue, MRValue, MLValue>(
                [&](Value value) {
                  if (!isa<mlir::BlockArgument>(value))
                    decl->setIRValue(nullptr);
                });
      }
    }

    body->clear();
    // Don't append anything to an empty function if this is a trait function.
  } else {
    StructEmitter(shared).appendDefaultReturnAndEndOp(decl);
  }

  // Now that the body of the function is parsed, run any body decorators.
  Decorators(decl, shared).applyBodyDecorators([&](ExprNode *decorator) {
    processExtensibilityDecorator(shared, decl, decorator);
    return failure();
  });

  return success();
}

DefArgumentWrapperDLValue::DefArgumentWrapperDLValue(ASTDecl *argDecl,
                                                     BValue argRef,
                                                     ASTType eltType,
                                                     size_t argIndex)
    : BaseDLValue(eltType), argDecl(argDecl), argRef(argRef),
      argIndex(argIndex) {}

/// If this is a def argument shadow, resolve it to the incoming immutable
/// borrowed value without forming a local copy.  Otherwise return null.
MBValue DefArgumentWrapperDLValue::emitMBValueFromDefArgument(
    ExprEmitter &emitter) const {
  return argRef.getIfMBValue();
}

MBValue StoredAttributeRefDLValue::emitMBValueFromDefArgument(
    ExprEmitter &emitter) const {
  auto baseRef = baseVal.ir->emitMBValueFromDefArgument(emitter);
  if (!baseRef)
    return {};

  auto fieldRef = emitter.builder->create<RefStructGEROp>(
      expr->getLocation(emitter), baseRef, cast<StructFieldOp>(fieldOp));
  return MBValue(fieldRef);
}

void DefArgumentWrapperDLValue::print(raw_ostream &os) const {
  os << "def argument wrapper of type " << elementType;
}

// This hook is called before an argument is passed inout.
LValue
DefArgumentWrapperDLValue::prepareForInoutAccess(SMLoc loc,
                                                 ExprEmitter &emitter) const {
  // Okay, if the def argument is mutated, we need to snap into action and
  // lazily build a shadow in the function entry.
  auto func = cast<FuncOp>(argDecl->getParentDecl());
  ExprEmitter entryEmitter(emitter.shared, *argDecl->getParentDecl(),
                           OpBuilder::atBlockBegin(func.getBody()));
  StringAttr argName = func.getSignature().getArgName(argIndex);

  // Create the shadow box and copy the argument into it.  This will emit an
  // error at the specified location if the underlying type isn't copyable.
  VarDeclOp declOp = entryEmitter.makeArgLValueVarSlot(argRef, argName, loc);

  // Emission can fail when the type is non-copyable.
  if (!declOp) {
    argDecl->setErroneous();
    return LValue();
  }

  declOp.setArgShadowIndex(argIndex);

  // Update the representation so we don't do this again.
  argDecl->setIRValue(MLValue(declOp));
  return MLValue(declOp);
}

CValue DefArgumentWrapperDLValue::emitLoad(ValueDest &dest,
                                           ExprEmitter &emitter) const {
  // Loads of the def argument wrapper are simple enough.
  SyntheticNode expr(argDecl->getLoc());
  return emitter.emitCResult(argRef, &expr, dest);
}

void DefArgumentWrapperDLValue::emitStore(ASTExprAnd<CValue> value,
                                          ExprEmitter &emitter) const {
  // Okay, if the def argument is mutated, we need to snap into action and
  // lazily build a shadow in the function entry.
  LValue newVal = prepareForInoutAccess(value.expr->getLoc(), emitter);
  if (!newVal)
    return;

  // Ok, now emit a normal store.
  emitter.emitStoreToLValue(value, newVal, ExprContext::EC_Assignment);
}

//===----------------------------------------------------------------------===//
// Module Decl implementation
//===----------------------------------------------------------------------===//

ParseResult DeclResolver::resolveBody(LIT::FileModuleOp op, Lexer &lexer,
                                      ASTDecl &decl) {
  // Push a scope for the file of this module.
  DebugInfo::DIBuilder::ScopeGuard fileGuard;
  if (shared.diBuilder) {
    auto &sourceMgr = shared.getSourceMgr();
    int fileId = sourceMgr.FindBufferContainingLoc(lexer.getToken().getLoc());
    if (fileId) {
      StringRef filename =
          sourceMgr.getMemoryBuffer(fileId)->getBufferIdentifier();
      fileGuard = shared.diBuilder->pushFile(filename, "/");
    }
  }

  return ParserBase(shared, lexer).parseSuite(decl);
}

//===----------------------------------------------------------------------===//
// Package Decl implementation
//===----------------------------------------------------------------------===//

static bool isModuleOrPackagePath(const std::filesystem::path &path) {
  // Handle source files.
  if (path.extension() == ".mojo" || path.extension() == ".🔥")
    return true;
  // Handle source packages.
  return Filesystem::isMojoSourcePackagePath(path);
}

ParseResult DeclResolver::resolveBody(LIT::PackageOp op, ASTDecl &decl) {
  // A source package corresponds to a directory, resolving the body requires
  // iterating the filesystem directory and importing the corresponding
  // children.

  // Grab the directory that this package is defined in.
  std::optional<std::string> directoryStr = shared.getModuleSourcePath(decl);
  if (!directoryStr)
    return emitError(op.getLoc(), "unable to locate package directory");

  std::error_code ec;
  std::filesystem::path directory(*directoryStr);
  if (!std::filesystem::is_directory(directory, ec) || ec)
    return emitError(op.getLoc(), "unable to locate package directory");

  // Iterate the directory and import nested modules.
  OpBuilder builder = decl.getDeclEndBuilder();
  SmallVector<std::string> nestedModules;
  for (const auto &entry : std::filesystem::directory_iterator(directory, ec)) {
    if (ec || !isModuleOrPackagePath(entry.path()))
      continue;
    nestedModules.emplace_back(
        entry.path().filename().replace_extension().generic_string());
  }

  // Sort the nested modules to ensure that we get a deterministic filesystem
  // ordering across the different platforms.
  llvm::stable_sort(nestedModules);

  // Create an unresolved relative import for each nested module. That way we
  // only need to actually pull anything in from the filesystem if it gets
  // referenced.
  for (StringRef name : nestedModules) {
    StringAttr importName = builder.getStringAttr("." + name);
    StringAttr boundName = builder.getStringAttr(name);
    auto importDecl = builder.create<LIT::UnresolvedImportOp>(
        op->getLoc(), importName, boundName, /*declName=*/StringAttr(),
        /*importNameLoc=*/LocationAttr(),
        /*destNameLoc=*/LocationAttr());
    getDeclResolver().addDecl(importDecl, decl.loc, boundName, &decl,
                              LexerCursor(), LexerCursor(), /*indentation=*/-1);

    // Create an alias for the unmangled module name to allow for simplified
    // indexing into this module.
    boundName = builder.getStringAttr(name);
    importDecl = builder.create<LIT::UnresolvedImportOp>(
        op->getLoc(), importName, boundName, /*declName=*/StringAttr(),
        /*importNameLoc=*/LocationAttr(),
        /*declNameLoc=*/LocationAttr());
    getDeclResolver().addDecl(importDecl, decl.loc, boundName, &decl,
                              LexerCursor(), LexerCursor(), /*indentation=*/-1);
  }

  // Create a full wildcard import from the __init__, as the symbols defined
  // there are visible from the package.
  StringAttr importModule = builder.getStringAttr(".__init__");
  builder.create<UnresolvedWildcardImportOp>(op->getLoc(), importModule,
                                             /*fullImport=*/true);
  decl.addUnresolvedWildCardImport(importModule, /*isFullImport=*/true,
                                   decl.loc);

  // Resolve the body of the __init__ within the package, and inherit some
  // attributes from it if they are present.
  LookupResult initResult =
      shared.lookupAndResolveDecl("__init__", decl.loc, decl,
                                  /*searchParentScopes=*/false);
  if (initResult.isSuccess()) {
    ASTDecl &initDecl = *initResult.getIfSuccess().front();
    if (failed(resolveFully(initDecl, decl.loc)))
      return failure();
    if (auto initDeclOp = dyn_cast<ASTDeclInterface>(initDecl)) {
      // Inherit the docstring from the __init__ if it is present.
      if (auto docstring = initDeclOp.getDocStringAttr())
        op.setDocStringAttr(docstring);
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GlobalVarDecl implementation
//===----------------------------------------------------------------------===//

LogicalResult DeclResolver::resolveSignature(GlobalVarDeclOp op, Lexer &lexer,
                                             ASTDecl &decl) {
  ParserBase p(shared, lexer);
  SmallVector<std::pair<ExprNode *, LexerCursor>> decoratorExprs =
      p.parseDecorators(decl);

  // Re-parse the preamble. The syntax should have been checked already.
  if (!p.consumeIf(Token::kw_var)) {
    return shared.emitError(
        decl.getLoc(), "internal error: should be checked by statement parser");
  }
  StringAttr name;
  SMLoc identifierLoc;
  if (p.parseIdentifier(name,
                        "internal error: should be checked by statement parser",
                        &identifierLoc))
    return failure();

  // Parse the type if present.
  ASTType parsedType;
  ExprEmitter emitter(shared, *decl.getParentDecl(), EC_VarInit);
  if (p.consumeIf(Token::colon)) {
    ExprNode *typeExpr = nullptr;
    if (p.parseExpression(typeExpr, decl.getIndentation()))
      return failure();
    parsedType = emitter.emitExprType(typeExpr);
    if (!parsedType)
      return failure();
  }

  // Global variables require an initializer.
  ExprNode *initExpr = nullptr;
  if (p.parseToken(Token::equal, "expected '=' in global variable") ||
      p.parseVarInitExpression(initExpr, decl.getIndentation()))
    return failure();

  // Emit the initializer into an initializer function. If we have a type, then
  // emit directly into the LValue. Otherwise emit into the global to infer its
  // type.
  if (parsedType)
    op.setType(parsedType);
  // If we don't, we emit into the varOp itself, because this will infer the
  // type of the varOp from the initializer expression.
  ValueDest dest(op, EC_VarInit);

  op.getCtor().push_back(new Block);
  emitter.builder = OpBuilder::atBlockBegin(&op.getCtor().front());
  if (!emitter.emitExpr(initExpr, dest))
    return failure();

  assert(!isa<UnresolvedType>(op.getType()) &&
         "RValue emission should have inferred var type");

  // Emit the destructor call, if present, into the destructor function.
  op.getDtor().push_back(new Block());
  if (shared.typeHasMember(ASTType(op.getType()), "__del__",
                           initExpr->getLoc())) {
    emitter.builder = OpBuilder::atBlockBegin(&op.getDtor().front());
    MRValue owned(emitter.builder->create<GlobalVarRefOp>(op.getLoc(), op));
    ValueDest dest(EC_Destructor);
    (void)emitter.emitNamedMethodCall("__del__",
                                      CallOperands({{owned, initExpr}}), dest,
                                      CallSyntax::kDestructor, initExpr);
  }

  // Run signature decorators, if any.
  auto processDecorator = [&](ExprNode *decorator) -> LogicalResult {
    if (auto ref = dyn_cast<DeclRefNode>(decorator);
        ref && ref->spelling == "export") {
      applyExport(ref->getLoc(), shared, decl, name, name, op);
      return success();
    }
    if (auto call = dyn_cast<CallNode>(decorator)) {
      if (auto ref = dyn_cast<DeclRefNode>(call->callee);
          ref && ref->spelling == "export") {
        applyExport(call->getLoc(), shared, decl, name, *call, op);
        return success();
      }
    }
    return failure();
  };
  Decorators(decl, shared)
      .applySignatureDecorators(decoratorExprs, processDecorator);

  shared.notifyListenerOnVariableDecl(decl, identifierLoc);
  return success();
}

ParseResult DeclResolver::resolveBody(GlobalVarDeclOp op, Lexer &lexer,
                                      ASTDecl &decl) {
  Decorators(decl, shared).applyBodyDecorators([](ExprNode *decorator) {
    return failure();
  });
  return success();
}

//===----------------------------------------------------------------------===//
// Alias Decl implementation
//===----------------------------------------------------------------------===//

/// alias_decl_stmt ::= "alias" identifier ":" expression ["=" expression]
///                   | "alias" identifier "=" expression
///
LogicalResult DeclResolver::resolveSignature(AliasDeclOp aliasDeclOp,
                                             Lexer &lexer, ASTDecl &decl) {
  ParserBase p(shared, lexer);
  auto decoratorExprs = p.parseDecorators(decl);
  Decorators(decl, shared, /*signatureOnly=*/true)
      .applySignatureDecorators(decoratorExprs);

  // Parse the type if present.
  SMLoc identifierLoc;
  if (p.parseToken(Token::kw_alias, "internal error: checked by stmt parser") ||
      p.parseIdentifier("internal error: checked by stmt parser",
                        &identifierLoc))
    return failure();

  ASTType type;
  if (p.consumeIf(Token::colon)) {
    if (parseType(p, type, *decl.getParentDecl(), decl.getIndentation()))
      return failure();
  }
  if (p.parseToken(Token::equal, "expected '=' in alias declaration"))
    return failure();

  // Otherwise this is a normal `alias` declaration with an initializer.
  ExprNode *initExpr = nullptr;
  if (p.parseExpression(initExpr, decl.getIndentation()))
    return failure();

  ASTDecl &parentDecl = *decl.getParentDecl();
  ExprEmitter emitter(shared, parentDecl, EC_AliasValue);

  // Emit the value and convert to the expected type if we know it.
  auto rhsValue = emitter.emitExprPValue(initExpr, EC_AliasValue, type);
  if (!rhsValue)
    return failure();

  // If we had no declared type (`alias x = 42`), infer the type from the
  // initializer.
  if (!type)
    type = rhsValue.getType();

  // Remember the value, and update the type from UnresolvedType.
  NamedAttrList attrs = aliasDeclOp->getAttrDictionary();
  attrs.set(aliasDeclOp.getValueAttrName(), rhsValue.get());
  attrs.set(aliasDeclOp.getParamDeclAttrName(),
            ParamDeclAttr::get(aliasDeclOp.getName(), type));
  aliasDeclOp->setAttrs(attrs.getDictionary(decl.getContext()));

  // Process the doc string of the alias.
  p.parseDocString(decl);

  shared.notifyListenerOnAliasDecl(decl, identifierLoc);
  return success();
}

ParseResult DeclResolver::resolveBody(AliasDeclOp op, Lexer &lexer,
                                      ASTDecl &decl) {
  return success();
}

//===----------------------------------------------------------------------===//
// Struct Decl implementation
//===----------------------------------------------------------------------===//

/// For a struct or trait declaration, parse an optional list of parent types to
/// inherit from.
static ParseResult
parseOptionalParentList(ParserBase &p, ASTDecl &declScope, StringRef declName,
                        SmallVectorImpl<TypeLineageAttr> &parentTypes,
                        SharedState &shared) {
  if (!p.consumeIf(Token::l_paren) || p.consumeIf(Token::r_paren))
    return success();

  // Resolve the traits such that there are no duplicates.
  llvm::MapVector<Type, std::pair<TypeLineageAttr, SMLoc>> parentTypeSet;
  auto parseParent = [&]() -> ParseResult {
    ASTType type;
    SMLoc loc;
    if (p.getLocation(loc) ||
        parseType(p, type, declScope, declScope.getIndentation()))
      return failure();

    // Reject inheriting from types we don't support yet.
    if (!isa<TraitType>(type)) {
      if (isa<LIT::StructType>(type)) {
        p.emitError(loc)
            << "TODO: inheriting from other structs is not implemented";
      } else if (isa<ParamRefType>(type)) {
        p.emitError(loc) << "TODO: inheriting from a parameter expression is "
                            "not implemented";
      } else {
        p.emitError(loc) << "don't know how to inherit from this type";
      }
      declScope.setErroneous();
      return success();
    }

    auto it = parentTypeSet.insert({type, {TypeLineageAttr::get(type), loc}});
    if (!it.second) {
      // If the user explicitly inherited a trait that is already provided
      // elsewhere, provide a warning.
      auto [cur, curLoc] = it.first->second;
      InflightDiag diag = shared.emitWarning(loc, "'")
                          << declName << "' already inherits from "
                          << ASTType(type);
      if (cur.getInheritedFrom().empty()) {
        diag.attachNote(curLoc) << "previously inherited here";
      } else {
        diag.attachNote(curLoc)
            << "inherited through " << ASTType(cur.getInheritedFrom().back())
            << " here";
      }
    }
    // Successively flatten the parent list so we always have all the parents
    // available to check.
    // TODO: Encode an "inherited from" here, to make diagnostics nice.
    ASTDecl &traitDecl = shared.declResolver->getDeclForTypeSymbol(
        cast<TraitType>(type).getSymbol());
    for (TypeLineageAttr inherited :
         cast<TraitDeclOp>(traitDecl).getParentTypes()) {
      if (auto it = parentTypeSet.find(inherited.getType());
          it != parentTypeSet.end())
        continue;
      SmallVector<Type> lineage = llvm::to_vector(inherited.getInheritedFrom());
      lineage.push_back(type);
      Type parent = inherited.getType();
      parentTypeSet.insert(
          {parent, {TypeLineageAttr::get(parent, lineage), loc}});
    }
    return success();
  };
  if (p.parseCommaSeparatedList(parseParent, Token::r_paren) ||
      p.parseToken(Token::r_paren, "expected ')' for parameter list"))
    return failure();
  for (auto [type, _] : llvm::make_second_range(parentTypeSet))
    parentTypes.push_back(type);
  return success();
}

/// Process a decorator that is resolved at the signature phase of resolution
/// and return success, otherwise failure if it is handled later.
static LogicalResult processStructSignatureDecorator(ExprNode *decorator,
                                                     StructDeclOp structOp,
                                                     SharedState &shared,
                                                     ASTDecl &structDecl) {
  if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
    if (declRef->spelling == "register_passable") {
      structOp.setConvention(TypeConvention::RegisterPassable);
      return success();
    }
    // @value
    if (declRef->spelling == "value") {
      // During signature resolution, add the `Copyable` and `Movable` traits.
      if (ASTDecl *decl = shared.lookupBuiltinTrait(
              "Copyable", structDecl.getParentDecl(), decorator->getLoc()))
        StructEmitter::addTraitParent(structOp, decl);
      if (ASTDecl *decl = shared.lookupBuiltinTrait(
              "Movable", structDecl.getParentDecl(), decorator->getLoc()))
        StructEmitter::addTraitParent(structOp, decl);
      // Fallthrough the decorator to body resolution.
      return failure();
    }
  }

  // `x()` forms.
  if (auto callNode = dyn_cast<CallNode>(decorator)) {
    if (auto declRef = dyn_cast<DeclRefNode>(callNode->callee)) {
      // @register_passable("trivial")
      if (declRef->spelling == "register_passable" &&
          callNode->operands.size() == 1 &&
          callNode->operands[0].isPositionalStringLiteral("trivial")) {
        structOp.setConvention(TypeConvention::RegisterPassableTrivial);
        return success();
      }

      // @nonmaterializable(TargetType)
      if (declRef->spelling == "nonmaterializable" &&
          callNode->operands.size() == 1) {
        if (auto drn = dyn_cast<DeclRefNode>(callNode->operands[0].expr)) {
          ASTDecl *parentDecl = structDecl.getParentDecl();
          ExprEmitter emitter(shared, *parentDecl, EC_Type);
          if (ASTType t = emitter.emitExprType(drn)) {
            structOp.setNonmaterializableTargetAttr(TypeAttr::get(t.mlirType));
            return success();
          }
        }
      }
    }
  }
  // Not handled in signature phase.
  return failure();
}

/// Silence internal verifier errors when constructing types from the parser. We
/// don't want to show these to the user.
static auto silenceErrors(MLIRContext *ctx) {
  return [ctx] {
    InFlightDiagnostic diag = mlir::emitError(UnknownLoc::get(ctx));
    diag.abandon();
    return diag;
  };
}

/// structdef ::=
///   [decorators] "struct" identifier [param_signature] ":" suite
///
LogicalResult DeclResolver::resolveSignature(StructDeclOp structOp,
                                             Lexer &lexer, ASTDecl &decl) {
  ParserBase p(shared, lexer);
  auto decoratorExprs = p.parseDecorators(decl);

  // The struct signature is a self-contained scope where the input and result
  // parameters of the function are visible by all types.  We must use a
  // temporary declaration here (with an empty name) because we don't want
  // references to the function itself to resolve to a fully-resolved decl, but
  // we need a fully-resolved decl for incremental lookups within the scope to
  // work out.
  ASTDecl &sigDecl = addFullyResolvedDecl(structOp.getOperation(), StringAttr(),
                                          decl.getLoc(), decl.getParentDecl());

  ParsedParamList parsedParams;
  SmallVector<TypeLineageAttr> parentTypes;

  SMLoc identifierLoc;
  if (p.parseToken(Token::kw_struct,
                   "internal error: checked by stmt parser") ||
      p.parseIdentifier("internal error: checked by stmt parser",
                        &identifierLoc) ||
      parsedParams.parseOptionalParameters(p, ArgListKind::kParamList) ||
      parseOptionalParentList(p, sigDecl, structOp.getSymName(), parentTypes,
                              shared) ||
      p.parseToken(Token::colon, "expected ':' in struct definition") ||
      decl.isErroneous())
    return failure();

  TypeCheckedParamList paramSignature(parsedParams.params, sigDecl, shared);

  // Propagate signature errors and decls.
  decl.takeDecls(sigDecl);

  auto paramsArrayAttr =
      ParamDeclArrayAttr::get(getContext(), paramSignature.paramDeclAttrs);
  auto sig = TypeSignatureType::remapToSignature(
      silenceErrors(getContext()), paramsArrayAttr,
      paramSignature.getParamListAttr());
  if (!sig)
    return failure();
  structOp.setParamsAttr(paramsArrayAttr);
  structOp.setSignature(sig);
  structOp.setParentTypes(parentTypes);

  // Make every nominal type inherit from `AnyType`.
  if (ASTDecl *traitDecl = shared.lookupBuiltinTrait(
          "AnyType", decl.getParentDecl(), decl.getLoc()))
    StructEmitter::addTraitParent(structOp, traitDecl);

  // This is a struct, so we can use 'computeSelfTypeForStruct' to figure out
  // the self type.
  decl.setTypeDeclSelf(ASTDecl::computeSelfTypeForStruct(structOp));

  // Structs are memory-only unless they opt-in to being passed in registers.
  structOp.setConvention(TypeConvention::MemoryOnly);

  // Now that we have the basic struct set up, process signature decorators.
  Decorators(decl, shared)
      .applySignatureDecorators(decoratorExprs, [&](ExprNode *decorator) {
        return processStructSignatureDecorator(decorator, structOp, shared,
                                               decl);
      });

  // Always generate SourceName for structs (even on non-debug builds).
  structOp.setSourceNameAttr(shared.getSourceName(structOp));

  shared.notifyListenerOnStructDecl(decl, identifierLoc);
  return success();
}

/// Look up the __del__ destructor for the specified `type` which is needed
/// for the specified declaration (typically a var or argument declaration).
/// This returns the destructor if successful, diagnoses an error if not, and
/// returns null if there is no defined destructor.
static SymbolConstantAttr lookupDestructor(ASTDecl &structDecl,
                                           SharedState &shared) {
  auto dels = shared.lookupAndResolveDecl(
      "__del__", structDecl.getLoc(), structDecl, /*searchParentScopes=*/false);
  ArrayRef<ASTDecl *> entries = dels.getIfSuccess();
  // If there are no __del__ methods, return null.  This is valid.
  if (entries.empty())
    return {};
  if (entries.size() != 1) {
    auto diag = shared.emitError(structDecl.getLoc(),
                                 "invalid overloaded '__del__' method");
    for (auto candidate : entries)
      diag.attachNote(candidate->getLoc()) << "candidate declared here";
    return {};
  }
  ASTDecl &delDecl = *entries[0];
  LIT::FuncOp func = dyn_cast<LIT::FuncOp>(delDecl);
  if (!func) {
    shared.emitError(delDecl.getLoc(), "'__del__' must be a method");
    return {};
  }
  return func.getBoundSymbolRef();
}

/// Look up a special method impl for the specified `type` when there is exactly
/// one implementation (not overloaded).  This returns the method if successful,
/// and returns null if there is none.
static SymbolConstantAttr lookupSpecialMethod(ASTDecl &structDecl,
                                              SharedState &shared,
                                              SpecialFunctionKind specialKind) {
  const char *name = SpecialFunctionInfo::get(specialKind).name;
  LookupResult inits = shared.lookupAndResolveDecl(
      name, structDecl.getLoc(), structDecl, /*searchParentScopes=*/false);

  for (ASTDecl *candidate : inits.getIfSuccess()) {
    LIT::FuncOp func = dyn_cast<LIT::FuncOp>(candidate);
    if (func && func.getSpecialFunctionKind() == specialKind)
      return func.getBoundSymbolRef();
  }
  return {};
}

namespace {
struct StructBodyDecorators : public SharedStateUser {
  StructBodyDecorators(
      StructDeclOp structOp, ASTDecl &structDecl, DeclResolver &resolver,
      ArrayRef<std::pair<StructFieldOp, ASTDecl *>> structFields)
      : SharedStateUser(resolver.shared), structOp(structOp),
        structDecl(structDecl), structFields(structFields) {}

  LogicalResult processDecorator(ExprNode *decorator, LIT::FuncOp moveFunc,
                                 LIT::FuncOp copyFunc);

private:
  /// Process the @value body decorator on structs.  This synthesizes the
  /// memberwise init, copy ctor and move ctor if requested.
  void processValueDecorator(SMLoc decoratorLoc, LIT::FuncOp moveFunc,
                             LIT::FuncOp copyFunc);

  /// Get a constant symbol to a method, and return null if it is missing or
  /// something went wrong.
  /// Provide optionally a callback for the case where the method is missing.
  SymbolConstantAttr
  getSymbolForMethod(StringRef methodName, ExprNode *decorator,
                     function_ref<void()> callbackOnMissing = nullptr);

  /// Process the @op_implementation body decorator on structs.
  /// It adds a new operation in the IR that link the new op name with the
  /// relevant struct methods.
  void processOpImplDecorator(ExprNode *decorator, StringRef opName);

  StructDeclOp structOp;
  ASTDecl &structDecl;
  ArrayRef<std::pair<StructFieldOp, ASTDecl *>> structFields;
};
} // namespace

/// Synthesize the `__copyinit__` and `__moveinit__` stubs for `@value`
/// decorated structs early to ensure their movability and copyability
/// requirements are satisfied.
static std::pair<LIT::FuncOp, LIT::FuncOp>
preprocessValueDecorator(SharedState &shared, ASTDecl &structDecl) {
  auto declOp = cast<StructDeclOp>(structDecl);
  for (ExprNode *decorator : structDecl.getBodyDecorators(shared)) {
    if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
      if (declRef->spelling == "value") {
        std::optional<ValueInfo> info =
            ValueInfo::createValueInfo(structDecl, shared);
        if (!info)
          break;
        StructEmitter emitter(shared);
        LIT::FuncOp moveFunc, copyFunc;
        if (!declOp.isRegisterPassable() && !info->hasMove()) {
          moveFunc = emitter.synthesizeEmptyMoveInit(structDecl);
          moveFunc.setInlineLevel(InlineLevel::AlwaysNoDebug);
        }
        if (!declOp.isRegisterPassableTrivial() && !info->hasCopy()) {
          copyFunc = emitter.synthesizeEmptyCopyInit(structDecl);
          copyFunc.setInlineLevel(InlineLevel::AlwaysNoDebug);
        }
        return {moveFunc, copyFunc};
      }
    }
  }
  return {nullptr, nullptr};
}

void StructBodyDecorators::processValueDecorator(SMLoc decoratorLoc,
                                                 LIT::FuncOp moveFunc,
                                                 LIT::FuncOp copyFunc) {
  // Check to see the classification of the fields, the result type will be
  // copyable/movable iff all the fields are.
  bool isCopyable = true, isMovable = true;
  for (auto [fieldOp, fieldDecl] : structFields) {
    ASTType fieldType(fieldOp.getType());
    isCopyable &= fieldType.isCopyable(fieldDecl->getLoc(), shared);
    isMovable &= fieldType.isMovable(fieldDecl->getLoc(), shared);

    // If this field is neither copyable or movable, then we cannot do
    // anything in this decorator.
    if (!isCopyable && !isMovable) {
      auto diag =
          emitError(decoratorLoc, "'@value' cannot synthesize members: ")
          << fieldOp.getNameAttr() << " has non-copyable, non-movable type "
          << fieldType;
      diag.attachNote(fieldDecl->getLoc())
          << fieldOp.getNameAttr() << " declared here";
      structDecl.setErroneous();
      return;
    }
  }

  StructEmitter structEmitter(shared);
  StructDeclOp declOp = dyn_cast<StructDeclOp>(structDecl);
  std::optional<GeneratedStubs> stubs =
      structEmitter.addMissingValueMemberStubsToStruct(
          structDecl, /*generateFieldwiseInit=*/true);
  if (!stubs) {
    emitError(decoratorLoc, "'@value' cannot synthesize members of struct '")
        << declOp.getSymName() << "'";
    structDecl.setErroneous();
    return;
  }
  stubs->moveCtr = moveFunc;
  stubs->copyCtr = copyFunc;

  if (LIT::FuncOp copyCtr = stubs->copyCtr) {
    SymbolConstantAttr ref = copyCtr.getBoundSymbolRef();
    ASTDecl *copyCtrDecl =
        getDeclResolver().getDeclForFuncSymbol(ref.getSymbol());
    if (failed(structEmitter.populateMoveCopy(*copyCtrDecl, /*isMove=*/false)))
      shared.deleteDecl(*copyCtrDecl);
    else
      declOp.setCopyInitAttr(ref);
  }
  if (LIT::FuncOp moveCtr = stubs->moveCtr) {
    SymbolConstantAttr ref = moveCtr.getBoundSymbolRef();
    ASTDecl *moveCtrDecl =
        getDeclResolver().getDeclForFuncSymbol(ref.getSymbol());
    if (failed(structEmitter.populateMoveCopy(*moveCtrDecl, /*isMove=*/true)))
      shared.deleteDecl(*moveCtrDecl);
    else
      declOp.setMoveInitAttr(ref);
  }
}

SymbolConstantAttr StructBodyDecorators::getSymbolForMethod(
    StringRef methodName, ExprNode *decorator,
    function_ref<void()> callbackOnMissing) {
  // Get the possibly overloaded method.
  TypeCheckScopeInfo scopeInfo{structDecl, false, shared};
  auto methods = OverloadSet::lookup(
      scopeInfo, structDecl.getTypeDeclSelf(), methodName, decorator,
      CallSyntax::kMethodCallSynthetic, callbackOnMissing);

  // Case where we did not find the `impl` method or an error occured.
  if (!methods)
    return {};

  // Emit the constant symbol.
  auto methodsUValue = OverloadSetUValue::create(std::move(methods));
  ExprEmitter emitter(shared, structDecl, {});
  PValue value =
      emitter.emitPValue({methodsUValue, decorator}, ExprContext::EC_Decorator);
  if (!value)
    return {};

  return cast<SymbolConstantAttr>(value.get());
}

void StructBodyDecorators::processOpImplDecorator(ExprNode *decorator,
                                                  StringRef opName) {
  SMLoc decoratorLoc = decorator->getRangeStart();
  auto noImplMethodError = [this, decoratorLoc]() {
    emitError(decoratorLoc) << "struct annotated with '@op_implementation' "
                            << "should define an `impl` method";
  };

  auto implSym = getSymbolForMethod("impl", decorator, noImplMethodError);
  if (!implSym)
    return;

  auto canonicalizeSym = getSymbolForMethod("canonicalize", decorator);

  // Add the op implementation, and return an error if the op already had an
  // implementation.
  (void)shared.addCustomOpImpl(
      CustomOpImplAttr::get(opName, implSym, canonicalizeSym), decoratorLoc);
  return;
}

LogicalResult StructBodyDecorators::processDecorator(ExprNode *decorator,
                                                     LIT::FuncOp moveFunc,
                                                     LIT::FuncOp copyFunc) {
  // @value decorator
  if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
    if (declRef->spelling == "value") {
      processValueDecorator(decorator->getRangeStart(), moveFunc, copyFunc);
      return success();
    }
    if (declRef->spelling == "op_implementation") {
      emitError(decorator->getLoc())
          << "@op_implementation expects a string literal argument";
      structDecl.setErroneous();
      return success();
    }
    return failure();
  }

  // @op_implementation decorator
  if (auto callNode = dyn_cast<CallNode>(decorator)) {
    auto declRef = dyn_cast<DeclRefNode>(callNode->callee);
    if (!declRef || declRef->spelling != "op_implementation" ||
        callNode->operands.size() != 1 ||
        !callNode->operands.front().isPositional())
      return failure();
    auto strExpr = dyn_cast<StringLiteralNode>(callNode->operands.front().expr);
    if (!strExpr) {
      emitError(decorator->getLoc())
          << "@op_implementation expects a string literal argument";
      structDecl.setErroneous();
      return success();
    }
    processOpImplDecorator(decorator, strExpr->getValue());
    return success();
  }
  return failure();
}

/// Process the @register_passable decorator on structs.  This finalizes
/// semantic checks.
static void processRegisterPassableDecorator(
    StructDeclOp structOp, ASTDecl &structDecl,
    ArrayRef<std::pair<StructFieldOp, ASTDecl *>> structFields,
    DeclResolver &resolver, TypeConvention structPassability) {

  bool isTrivial = structPassability == TypeConvention::RegisterPassableTrivial;
  for (auto [fieldOp, fieldDecl] : structFields) {
    ASTType fieldType = fieldOp.getType();

    // Register-passable structs may only contain register-passable stored
    // values.
    // TODO(traits): We need to type constrain mlirtype parameters to being
    // register-only types to support things like this correctly:
    //  struct P[T: mlirtype]:
    //    var storage : T

    // If the field is at least as register-passable as the container then
    // we're happy.
    if (fieldType.getRegisterPassability(fieldDecl->getLoc(), resolver.shared) <
        structPassability) {
      StringRef trivialSuffix;
      if (isTrivial)
        trivialSuffix = "(\"trivial\")";

      auto diag = resolver.emitError(structOp.getLoc())
                  << "all members of '@register_passable" << trivialSuffix
                  << "' struct must themselves be '@register_passable"
                  << trivialSuffix << "'";
      diag.attachNote(fieldDecl->getLoc())
          << fieldOp.getNameAttr() << " declared with type " << fieldType;

      // We cannot support IRGen'ing references to this type, since it will
      // break invariant about being register passable without being composed
      // of such types.
      fieldDecl->getParentDecl()->setErroneous();
      return;
    }
  }
}

//===----------------------------------------------------------------------===//
// Trait Conformance Checking

/// Check conformance for struct that implements traits.
static LogicalResult verifyExplicitConformances(ASTDecl &structDecl,
                                                SharedState &shared) {
  bool hadErrors = false;
  auto structDeclOp = cast<StructDeclOp>(structDecl);
  for (TypeLineageAttr parent : structDeclOp.getParentTypes()) {
    std::optional<InflightDiag> diag;
    hadErrors |= failed(verifyConformance(structDecl, parent, shared, diag));
  }

  return success(!hadErrors);
}

ParseResult DeclResolver::resolveBody(StructDeclOp structOp, Lexer &lexer,
                                      ASTDecl &structDecl) {
  // Push the debug scope for this struct if necessary so that nested operations
  // have proper debug info.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(structOp.getLocScope());

  if (ParserBase(shared, lexer).parseSuite(structDecl))
    return failure();

  // Check to see if there is a destructor and install it into the StructDeclOp
  // if so.
  if (auto dtorAttr = lookupDestructor(structDecl, shared)) {
    // Check to see if we have an explicitly declared destructor.
    structOp.setDestructorAttr(dtorAttr);
  } else if (structDecl.getTypeDeclSelf() &&
             !structOp.isRegisterPassableTrivial() &&
             structDecl
                 .lookupInCurrentScope(StringAttr::get(getContext(), "__del__"))
                 .empty()) {
    structOp.setDestructorAttr(StructEmitter(shared)
                                   .synthesizeEmptyDtor(structDecl)
                                   .getBoundSymbolRef());
  }

  // Look up move and copy constructors and record them.
  if (!structOp.isRegisterPassable()) {
    if (auto copyInitAttr = lookupSpecialMethod(structDecl, shared,
                                                SpecialFunctionKind::kCopyInit))
      structOp.setCopyInitAttr(copyInitAttr);
    if (auto moveInitAttr = lookupSpecialMethod(structDecl, shared,
                                                SpecialFunctionKind::kMoveInit))
      structOp.setMoveInitAttr(moveInitAttr);
  }

  // If the struct is decorated with `@value`, make sure to synthesize the copy
  // and move constructors before the field types are signature resolved to
  // ensure that the Copyable and Movable trait requirements are satisfied.
  // FIXME: The order of decorator resolution here is a bit gross.
  auto [moveFunc, copyFunc] = preprocessValueDecorator(shared, structDecl);

  // This collects all the resolved struct fields. Now that the body is
  // completely resolved, check the declared fields for extra invariants.
  bool hasBadField = false;
  SmallVector<std::pair<StructFieldOp, ASTDecl *>> structFields;
  for (StructFieldOp field : structOp.getFieldDecls()) {
    // Make sure the field is signature resolved so we can get its type.
    auto fieldEntries = structDecl.lookupInCurrentScope(field.getNameAttr());
    assert(fieldEntries.size() == 1 && "field decls cannot be overloaded");
    ASTDecl &fieldASTDecl = *fieldEntries[0];
    if (failed(resolveSignature(fieldASTDecl, fieldASTDecl.getLoc()))) {
      hasBadField = true;
      continue;
    }
    structFields.push_back({field, &fieldASTDecl});
  }

  // If the struct is @register_passable, check invariants imposed by it before
  // checking other decorators.  This ensures that we reject invalid
  // register_passable types before processing them.
  if (structOp.isRegisterPassable()) {
    // TODO: Split trivial and register_passable apart.
    processRegisterPassableDecorator(structOp, structDecl, structFields, *this,
                                     structOp.getConvention());
  }

  // If any of the fields are bad, we do not process decorators since they
  // assume that the struct body if valid.
  if (hasBadField && !structDecl.getBodyDecorators(shared).empty()) {
    structDecl.setErroneous();
    return failure();
  }

  // If there are any body decorators, resolve them now.
  StructBodyDecorators structDecorators(structOp, structDecl, *this,
                                        structFields);
  Decorators(structDecl, shared)
      .applyBodyDecorators([&, moveFunc = moveFunc,
                            copyFunc = copyFunc](ExprNode *decorator) {
        return structDecorators.processDecorator(decorator, moveFunc, copyFunc);
      });

  if (structDecl.isErroneous())
    return success();

  // Finally, verify conformance of inherited traits.
  return verifyExplicitConformances(structDecl, shared);
}

//===----------------------------------------------------------------------===//
// StructFieldDecl implementation
//===----------------------------------------------------------------------===//

/// struct_field_decl_stmt ::= "var" identifier ":" expression
/// TODO: Support default values?
///
LogicalResult DeclResolver::resolveSignature(StructFieldOp fieldOp,
                                             Lexer &lexer, ASTDecl &decl) {
  ParserBase p(shared, lexer);
  auto decoratorExprs = p.parseDecorators(decl);

  ASTType type;
  SMLoc identifierLoc;
  // Parse the type if present.
  p.consumeToken(); // let or var.
  if (p.parseIdentifier("internal error: checked by stmt parser",
                        &identifierLoc) ||
      p.parseToken(Token::colon, "struct field declaration must have a type") ||
      parseType(p, type, *decl.getParentDecl(), decl.getIndentation()))
    return failure();

  if (auto traitType = dyn_cast<LIT::TraitType>(type.mlirType)) {
    emitError(decl.getLoc(), "TODO: dynamic traits not supported yet, please "
                             "use a compile time generic instead of ")
        << traitType;
    return failure();
  }

  fieldOp.setType(type);
  rejectDecorators(decoratorExprs, decl, shared);
  shared.notifyListenerOnStructFieldDecl(decl, identifierLoc);
  return success();
}

ParseResult DeclResolver::resolveBody(StructFieldOp op, Lexer &lexer,
                                      ASTDecl &decl) {
  return success();
}

//===----------------------------------------------------------------------===//
// Trait Decl implementation
//===----------------------------------------------------------------------===//

LogicalResult DeclResolver::resolveSignature(TraitDeclOp traitOp, Lexer &lexer,
                                             ASTDecl &decl) {
  ParserBase p(shared, lexer);
  auto decoratorExprs = p.parseDecorators(decl);
  Decorators(decl, shared, /*signatureOnly=*/true)
      .applySignatureDecorators(decoratorExprs);

  SMLoc identifierLoc;
  if (p.parseToken(Token::kw_trait, "internal error: checked by stmt parser") ||
      p.parseIdentifier("internal error: checked by trait parser",
                        &identifierLoc))
    return failure();

  if (p.consumeIf(Token::l_square)) {
    // If the current token is on a new line, report the error on the end of
    // the previous line, this is probably where the punctuation was omitted.
    auto diagLoc = p.getTokenLocOrEndOfPreviousLineIfOnNewLine();
    // Report the error.
    emitError(diagLoc,
              "TODO: trait declarations do not support parameters yet");
    return failure();
  }
  SmallVector<TypeLineageAttr> parentTypes;
  if (parseOptionalParentList(p, *decl.getParentDecl(), traitOp.getSymName(),
                              parentTypes, shared))
    return failure();

  if (p.parseToken(Token::colon, "expected ':' in trait definition"))
    return failure();

  // Make every trait inherit from `AnyType`, except itself.
  if (parentTypes.empty() && traitOp.getSymName() != "AnyType") {
    if (ASTDecl *anyTypeDecl = shared.lookupBuiltinTrait(
            "AnyType", decl.getParentDecl(), decl.getLoc())) {
      TraitType anyType = cast<TraitDeclOp>(anyTypeDecl).bindReference();
      parentTypes.push_back(TypeLineageAttr::get(anyType));
    }
  }

  // Insert the implicit trait parameter:
  // - T: a value of this trait type - the struct conforming to this trait.
  TraitType traitType = traitOp.bindReference();
  auto actualType = ParamDeclAttr::get(decl.mangleParamName("T"), traitType);

  MLIRContext *ctx = getContext();
  auto paramArray = ParamDeclArrayAttr::get(ctx, {actualType});
  auto paramListAttr =
      PogListAttr::get(ctx, StringAttr::get(ctx), PassingKind::Implicit);
  auto sig = TypeSignatureType::remapToSignature(silenceErrors(ctx), paramArray,
                                                 paramListAttr);
  if (!sig)
    return failure();
  traitOp.setParams(paramArray);
  traitOp.setSignature(sig);
  traitOp.setParentTypes(parentTypes);

  decl.setTypeDeclSelf(ASTDecl::computeSelfTypeForTrait(traitOp));

  shared.notifyListenerOnTraitDecl(decl, identifierLoc);

  return success();
}

namespace {
/// This replaces one attribute with another without respect to its original
/// type.  TODO: Is there a better way to do this?
struct AttrReplacer : public ParameterReplacer<AttrReplacer> {
  TypedAttr oldAttrValue, newAttrValue;

  AttrReplacer(TypedAttr oldAttrValue, TypedAttr newAttrValue)
      : oldAttrValue(oldAttrValue), newAttrValue(newAttrValue) {}

  template <typename T>
  std::conditional_t<std::is_base_of_v<Type, T>, Type, Attribute>
  doReplace(T value, size_t depth) {
    if (auto result = tryReplace(value, depth))
      return result;

    if constexpr (std::is_base_of_v<Type, T>)
      if (isa<ParameterScopeTypeInterface>(value))
        ++depth;

    SmallVector<Attribute, 16> newAttrs;
    SmallVector<Type, 16> newTypes;
    bool changed = false;
    auto walkFn = [&](auto value, SmallVectorImpl<decltype(value)> &values) {
      auto newValue = this->replaceImpl(value, depth);
      changed |= newValue != value;
      values.push_back(newValue);
    };
    value.walkImmediateSubElements(
        [&](Attribute attr) { walkFn(attr, newAttrs); },
        [&](Type type) { walkFn(type, newTypes); });
    if (!changed)
      return value;
    return value.replaceImmediateSubElements(newAttrs, newTypes);
  }

  // CRTP methods.
  Attribute tryReplace(Attribute attr, size_t depth) {
    if (attr == oldAttrValue)
      return newAttrValue;
    return {};
  }
  Type tryReplace(Type, size_t) { return {}; }
};
} // end anonymous namespace

/// Update the types for a method pulled from a trait base to a derived trait,
/// so they refer to the correct self type.
static void replaceTraitMethodSelfTypes(LIT::FuncOp func,
                                        TypedAttr parentSelfType,
                                        TypedAttr traitSelfType) {
  assert(isa<ParamDeclRefAttr>(parentSelfType) &&
         isa<ParamDeclRefAttr>(traitSelfType));
  AttrReplacer replacer(parentSelfType, traitSelfType);

  // Update functionType, signature, and block argument types.
  func.setSignature(replacer.replace(func.getSignature()));
  func.setFunctionType(replacer.replace(func.getFunctionType()));
  for (auto arg : func.getBody()->getArguments())
    arg.setType(replacer.replace(arg.getType()));
}

ParseResult DeclResolver::resolveBody(TraitDeclOp traitOp, Lexer &lexer,
                                      ASTDecl &traitDecl) {
  // Push the debug scope for this trait if necessary so that nested operations
  // have proper debug info.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(traitOp.getLocScope());

  if (ParserBase(shared, lexer).parseSuite(traitDecl))
    return failure();

  // Resolve functions in the body here so that we can diagnose them.
  // Deduplicate inherited functions based on their name and mangled name, since
  // the mangled name contains the required information for distinguishing
  // overload candidates.
  DenseSet<std::pair<StringAttr, StringAttr>> existingFns;
  for (auto &[name, decls] : traitDecl.declsInScope) {
    if (decls.empty() || !isa<LIT::FuncOp>(decls.front()))
      continue;
    for (ASTDecl *decl : decls) {
      auto func = cast<LIT::FuncOp>(*decl);
      if (failed(resolveFully(*decl, decl->getLoc())))
        return failure();

      existingFns.insert({name, func.getSymNameAttr()});
      if (!func.getBody()->empty()) {
        shared.emitError(decl->getLoc(),
                         "unexpected function body in trait function "
                         "declaration, use `...` or `pass`");
      }
      auto b = ImplicitLocOpBuilder::atBlockEnd(func.getLoc(), func.getBody());
      b.create<TraitFuncOp>();
    }
  }

  // Get our Self type, which will be a reference to the T parameter on this
  // trait.
  ASTType traitSelfType = traitDecl.getTypeDeclSelf();

  // Now just pull in the functions in the bodies of all parents.
  Block &body = *traitOp.getBody();
  for (TypeLineageAttr parent : traitOp.getParentTypes()) {
    ASTDecl &parentDecl =
        getDeclForTypeSymbol(cast<TraitType>(parent.getType()).getSymbol());
    if (failed(resolveFully(parentDecl, traitDecl.getLoc())))
      continue;

    ASTType parentSelfType = parentDecl.getTypeDeclSelf();

    // Inherit function members, which we can override without worry because
    // they are all just declarations.
    for (auto &[name, decls] : parentDecl.getDeclsInScope()) {
      if (decls.empty() || !isa<LIT::FuncOp>(decls.front()))
        continue;
      for (ASTDecl *decl : decls) {
        if (failed(resolveFully(*decl, traitDecl.getLoc())))
          continue;
        auto func = cast<LIT::FuncOp>(decl);
        // Ensure that a function with the same name and signature hasn't
        // already been declared.
        if (!existingFns.insert({name, func.getSymNameAttr()}).second)
          continue;
        func = func.clone();

        // We copied down the function from a base trait to a derived trait,
        // and its type (e.g. self arguments, but not limited to them) will
        // refer to the T parameter from the base trait.  That will have a
        // metatype from the base trait which we need to update to our correct
        // Self type.
        replaceTraitMethodSelfTypes(func, PValue(parentSelfType).get(),
                                    PValue(traitSelfType).get());

        // Mark the function as inherited so that conformance checking won't
        // give duplicate errors if it is not provided.
        func.setIsInherited(true);
        body.push_back(func);
        ASTDecl &clonedDecl =
            addFullyResolvedDecl(&*func, name, decl->getLoc(), &traitDecl);
        finalizeFuncSignature(func, clonedDecl);
      }
    }
  }

  if (SymbolConstantAttr dtor = lookupDestructor(traitDecl, shared))
    traitOp.setDtorSig(dtor.getType());

  return success();
}

//===----------------------------------------------------------------------===//
// UnresolvedImport Decl implementation
//===----------------------------------------------------------------------===//

ParseResult DeclResolver::resolveSignature(LIT::UnresolvedImportOp op,
                                           ASTDecl &decl) {
  PackageOp packageOp = op->getParentOfType<PackageOp>();

  // Grab the location of the import name if present.
  SMLoc importNameLoc =
      shared.diags.convertLocToSMLoc(op.getImportNameLocAttr());
  if (!importNameLoc.isValid())
    importNameLoc = decl.getLoc();

  // Check if we are importing a specific decl within the module, or the
  // module itself.
  if (auto declName = op.getDeclNameAttr()) {
    SMLoc declNameLoc = shared.diags.convertLocToSMLoc(op.getDeclNameLocAttr());
    if (!declNameLoc.isValid())
      declNameLoc = decl.getLoc();

    return getDeclResolver().importDeclFromModule(
        *decl.getParentDecl(), packageOp, op.getModuleNameAttr(), declName,
        op.getImportNameAttr(), decl.getLoc(), declNameLoc, importNameLoc);
  }
  return getDeclResolver().importModule(
      *decl.getParentDecl(), packageOp, op.getModuleNameAttr(),
      op.getImportNameAttr(), decl.getLoc(), importNameLoc);
}
