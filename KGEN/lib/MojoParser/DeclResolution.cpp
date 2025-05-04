//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/DeclResolver.h"

#include "ClosureEmitter.h"
#include "DLValues.h"
#include "ExprEmitter.h"
#include "ExprNodes.h"
#include "KGEN/MOGGPreElab/MOGGPreElabHelpers.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "MojoUtils.h"
#include "ParserBase.h"
#include "Signatures.h"
#include "StructEmitter.h"
#include "Traits.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MOGGPreElab/MOGGPreElabDecorators.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/Filesystem/Paths.h"

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Regex.h"
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

  ExprEmitter emitter(declScope, EC_Type);
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
/// Decorators attached to a declaration may be "signature" decorators or "body"
/// decorators.
///
/// - Signature decorators are applied during the resolution of the signature of
///   a declaration before it is name bound.
/// - Body decorators are applied after the body of the declaration is fully
///   resolved.
///
/// This is the base class for handling decorators on declarations. Signature
/// decorators are processed first and then leftover decorators are persisted
/// until body resolution is complete via the SharedState.
class Decorators : public SharedStateUser {
public:
  /// Create a class to handle decorators for a decl. If `signatureOnly` is set,
  /// the class will reject any decorator not processed during signature
  /// resolution.
  Decorators(ASTDecl &decl, bool signatureOnly = false)
      : SharedStateUser(decl.getShared()), decl(decl),
        signatureOnly(signatureOnly) {}

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
  /// Validate compiler decorators that are allowed to propagate.
  LogicalResult validateCompilerDecorator(TypedAttr attr);

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
  decl.setBodyDecorators(bodyDecorators);
}

// Helper function to extract symbol name from a TypedAttr
static std::optional<StringRef> extractDecoratorName(TypedAttr attr) {
  // Helper lambda to extract name from a symbol reference
  auto extractFromSymbolRef = [](SymbolRefAttr ref) -> StringRef {
    StringRef name = ref.getLeafReference().getValue();
    return name.substr(0, name.find_first_of("(["));
  };

  if (auto cst = dyn_cast<SymbolConstantAttr>(attr))
    return extractFromSymbolRef(cst.getSymbol());

  if (auto call = dyn_cast<ParamOperatorAttr>(attr)) {
    // Only process if it's an Apply operator with at least one operand
    if (call.getOpcode() != POC::Apply || call.getOperands().empty())
      return std::nullopt;

    if (auto firstOp = dyn_cast<SymbolConstantAttr>(call.getOperands().front()))
      return extractFromSymbolRef(firstOp.getSymbol());
  }

  return std::nullopt;
}

LogicalResult Decorators::validateCompilerDecorator(TypedAttr attr) {
  constexpr StringRef plainDre[] = {
      "doc_private",
      "lldb_formatter_wrapping_type",

      MOGGPreElab::Decorators::REGISTER_MOGG_INTRINSIC,
      MOGGPreElab::Decorators::REGISTER_INTERNAL_FUNCTION,
      "enforce_io_param",

      "register",
      "elementwise",
      "view_kernel",
      "mutable",
  };

  auto symbolName = extractDecoratorName(attr);
  if (!symbolName)
    return failure();

  if (auto call = dyn_cast<ParamOperatorAttr>(attr)) {
    return success(call.getOpcode() == POC::Apply &&
                   llvm::is_contained(plainDre, *symbolName) &&
                   call.getOperands().size() <= 3);
  }

  return success(llvm::is_contained(plainDre, *symbolName));
}

void Decorators::applyBodyDecorators(
    function_ref<LogicalResult(ExprNode *)> process) {
  // Don't run decorators if the declaration is invalid.
  if (decl.isErroneous())
    return;

  ArrayRef<ExprNode *> decoratorExprs = decl.getBodyDecorators();
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
  ExprEmitter emitter(decl, EC_Decorator);
  for (auto [i, decorator] : llvm::enumerate(decoratorExprs)) {
    // Make sure we don't have another body decorator.
    if (failed(process(decorator))) {
      if (PValue decoVal = emitter.emitExprPValue(decorator, EC_Decorator)) {
        if (failed(validateCompilerDecorator(decoVal))) {
          emitError(decorator->getLoc(), "unsupported compiler decorator")
              << decorator->getRange();
        }
        decoPValues.push_back(decoVal);
      }
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
static void applyExport(SMLoc loc, ASTDecl &decl, StringRef unmangledName,
                        StringRef aliasName, ExportInterface itf,
                        bool isCExport = false) {
  auto &shared = decl.getShared();
  // Handle the unique case of main. We implicitly export main, so this is
  // simply checking that the user didn't try to export it as something else.
  if (aliasName == kMainSymbolName) {
    if (unmangledName != kMainSymbolName)
      shared.emitError(loc, "only 'main' can be exported as 'main'");
    if (!isa<FnOp>(decl))
      shared.emitError(loc, "exported 'main' must be a function");
    return;
  }
  if (unmangledName == kMainSymbolName) {
    shared.emitError(loc, "'main' can only be exported as 'main'");
    return;
  }

  llvm::TypeSwitch<ASTDecl &, void>(decl).Case<FnOp, GlobalVarDeclOp>(
      [aliasName](auto op) { op.setLinkageName(aliasName); });
  if (isCExport)
    itf.setCExported();
  else
    itf.setExported();

  shared.declResolver->registerAndCheckExport(aliasName, loc);
}

/// Apply `@export("linkageName")` to an exportable declaration and register it
/// with the shared state to ensure no duplicate exports.
static void applyExport(SMLoc loc, ASTDecl &decl, StringRef unmangledName,
                        const CallNode &node, ExportInterface itf) {
  auto &shared = decl.getShared();
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
  applyExport(loc, decl, unmangledName,
              aliasName ? StringRef(*aliasName) : unmangledName, itf,
              exportABI.has_value());
}

namespace {
struct FnSigDecorators : public SharedStateUser {
  FnSigDecorators(ASTDecl &decl, ASTDecl &sigDecl, SharedState &shared,
                  StringRef baseName, TypeCheckedFnSignature &tcSignature)
      : SharedStateUser(shared), decl(decl), sigDecl(sigDecl),
        funcOp(cast<FnOp>(decl)), baseName(baseName), tcSignature(tcSignature) {
  }

  /// Apply a function signature decorator.
  LogicalResult applyOne(ExprNode *decorator);
  /// Finalize application of all signature decorators.
  void finalize();

  static LogicalResult checkAlwaysInlineBuiltin(FnOp funcBody,
                                                SharedState &shared);

private:
  void applyStaticMethod(const DeclRefNode &node);
  void applyImplicitDecorator(const DeclRefNode &node);
  void applyCopyOrMoveCapture(const CallNode &node, bool isMove,
                              StringRef decorator);

  ArrayAttr getLLVMMetadataArray(ArrayRef<Operand> operands);
  void applyLLVMMetadata(const CallNode &node);

  /// Register an LLVM arg metadata in the internal list to avoid churning mlir
  /// attributes as these arg metadata decorators are parsed. Must call finalize
  /// to actually apply metadata onto the function.
  void registerLLVMArgMetadata(const CallNode &node);

  ASTDecl &decl;
  ASTDecl &sigDecl;
  FnOp funcOp;
  StringRef baseName;
  TypeCheckedFnSignature &tcSignature;

  /// The working list of LLVMArgMetadata. Either empty, or initialized to a
  /// list with the same length as the total number of function arguments on
  /// first use.
  SmallVector<Attribute> llvmArgMetadata;

  /// The working vector of the LLVMMetadata.
  SmallVector<Attribute> llvmMetadata;
};
} // namespace

/// This function verifies @always_inline("builtin") functions after the body of
/// the function has been parsed.
LogicalResult FnSigDecorators::checkAlwaysInlineBuiltin(FnOp fnOp,
                                                        SharedState &shared) {
  // To see if this is foldable, synthesize a bunch of argument values that we
  // can cram into the function and see if it balks.
  SmallVector<TypedAttr> operands;

  // Figure out the callee.  We synthesize a bound reference to the callee
  // making up nonsense parameter bindings.
  ParameterEvaluator evaluator;
  SmallVector<TypedAttr> params;
  for (auto paramDecl : fnOp.collectAllParams(/*implOrigins*/ false)) {
    params.push_back(
        UnknownAttr::get(evaluator.getReboundType(paramDecl.getType())));
    evaluator.setParameterValue(paramDecl, params.back());
  }
  auto paramValueArray = ParameterExprArrayAttr::get(fnOp.getContext(), params);
  operands.push_back(fnOp.getBoundReference(paramValueArray));

  for (auto arg : fnOp.getBody()->getArguments())
    operands.push_back(
        UnknownAttr::get(evaluator.getReboundType(arg.getType())));

  if (shared.foldInlineBuiltinFunction(operands, fnOp.getLoc(), true))
    return success();
  return failure();
}

LogicalResult FnSigDecorators::applyOne(ExprNode *decorator) {
  // Process all the decorators we know about.
  if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
    if (declRef->spelling == "export")
      applyExport(decorator->getLoc(), decl, baseName, baseName, funcOp);
    else if (declRef->spelling == "staticmethod")
      applyStaticMethod(*declRef);
    else if (declRef->spelling == "always_inline")
      funcOp.setInlineLevel(InlineLevel::Always);
    else if (declRef->spelling == "no_inline")
      funcOp.setInlineLevel(InlineLevel::Never);
    else if (declRef->spelling == "parameter")
      tcSignature.argList.effects.setCapturing();
    else if (declRef->spelling == "__unsafe_disable_nested_origin_exclusivity")
      tcSignature.isNestedOriginExclusivityCheckingDisabled = true;
    else if (declRef->spelling == "implicit")
      applyImplicitDecorator(*declRef);
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
      else if (declRef->spelling == "always_inline" &&
               callNode->operands.size() == 1 &&
               callNode->operands[0].isPositionalStringLiteral("builtin"))
        funcOp.setInlineLevel(InlineLevel::AlwaysBuiltin);
      else if (declRef->spelling == "export")
        applyExport(decorator->getLoc(), decl, baseName, *callNode, funcOp);
      else if (declRef->spelling == "__move_capture")
        applyCopyOrMoveCapture(*callNode, /*isMove=*/true, declRef->spelling);
      else if (declRef->spelling == "__copy_capture")
        applyCopyOrMoveCapture(*callNode, /*isMove=*/false, declRef->spelling);
      else if (declRef->spelling == "__llvm_metadata")
        applyLLVMMetadata(*callNode);
      else if (declRef->spelling == "__llvm_arg_metadata")
        registerLLVMArgMetadata(*callNode);
      else
        return failure();
      return success();
    }
  }
  return failure();
}

void FnSigDecorators::applyStaticMethod(const DeclRefNode &node) {
  // This decorator only applies to methods of structs and traits.
  if (!decl.tryGetMethodParentDecl()) {
    emitError(node.getLoc(), "only methods on structs may be declared static");
    return;
  }
  funcOp.setIsStatic(true);
}

void FnSigDecorators::applyImplicitDecorator(const DeclRefNode &node) {
  if (SpecialFunctionInfo::get(baseName).kind != SpecialFunctionKind::kInit) {
    emitError(node.getLoc())
        << "'@implicit' may only be applied to '__init__' methods";
    return;
  }

  ArrayRef<ParsedArgument> args = tcSignature.argList.parsedArgs;

  // Drop any error and result slots, default arguments and variadics.
  // Things like `__init__(out x, x: T, y : T = 42).
  // Allow `__init__(out x, x: Int = 4)` which has a default.
  while (1) {
    if (args.empty())
      break;

    auto &lastArg = args.back();
    if (lastArg.convention == ParsedArgument::kConventionByRefResult) {
      args = args.drop_back();
      continue;
    }

    // Drop defaults and varargs so long as they aren't the last argument.
    if (args.size() > 1 &&
        (lastArg.initExpr ||                  // arg has a default.
         lastArg.vararg != VarArgKind::None)) // vararg lists can be empty
      args = args.drop_back();
    else
      break;
  }

  if (args.empty()) {
    emitError(node.getLoc())
        << "'@implicit' requires an argument to convert from";
    return;
  }

  // We must have a positional argument to take the new value.
  if (args.size() != 1 ||
      (args[0].kwArgHandling != KWArgHandling::kPositionalOnly &&
       args[0].kwArgHandling != KWArgHandling::kPositionalOrKeyword)) {
    emitError(node.getLoc())
        << "'@implicit' initializers must accept a single argument value";
    return;
  }
  funcOp.setIsImplicitConversion(true);
}

void FnSigDecorators::applyCopyOrMoveCapture(const CallNode &node, bool isMove,
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
    FnOp parentOp = funcOp->getParentOfType<FnOp>();
    if (!parentOp) {
      emitError(declRef->getLoc(), "'@")
          << decoratorSpelling
          << "' decorator only applies to nested functions";
      return;
    }

    ExprEmitter emitter(*decl.getParentDecl(), OpBuilder(funcOp));
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

    // How is this transferring the RValue into the closure?
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

/// Return AliasDeclOp that corresponds to the value's name by looking at all
/// aliases within parent scopes up to FileModule. Return nullopt if not found.
/// Emit error if cannot resolve import op or declaration with the value.name is
/// not an alias.
static std::optional<AliasDeclOp> getLLVMMetadataNameAlias(SharedState &shared,
                                                           ASTDecl &funcDecl,
                                                           StringAttr name) {
  ASTDecl *parent = &funcDecl;
  // Analyze all parent scopes of the function in order to find closed
  // declaration with the value.name. Fully resolve that declaration if needed.
  do {
    parent = parent->getParentDecl();
    if (!parent)
      return {};

    ArrayRef<ASTDecl *> nameDecls = parent->lookupInCurrentScope(name);
    // Not interesting scope. Keep looking up for the declaration with
    // value.name.
    if (nameDecls.empty())
      continue;

    if (isa<UnresolvedImportOp>(nameDecls.back())) {
      if (failed(shared.getDeclResolver().resolveBody(*nameDecls.back(),
                                                      funcDecl.getLoc()))) {
        shared.emitError(funcDecl.getLoc(), "cannot resolve alias '")
            << name << "' used in '@__llvm_metadata'";
        return {};
      }
    }
    if (auto aliasOp = dyn_cast<AliasDeclOp>(nameDecls.back()))
      return aliasOp;

    shared.emitError(funcDecl.getLoc(), "name '")
        << name << "' cannot be used in '@__llvm_metadata'";
    return {};
  } while (!isa<FileModuleOp>(parent));
  return {};
}

ArrayAttr FnSigDecorators::getLLVMMetadataArray(ArrayRef<Operand> operands) {
  ExprEmitter emitter(sigDecl, EC_Decorator);
  SmallVector<Attribute> metadata;
  for (Operand value : operands) {
    StringAttr metadataName;
    ExprNode *metadataValue;
    // Handle the case of only a metadata name, with no value associated.
    if (value.passKind == Operand::PassKind::kPositional) {
      auto declRef = dyn_cast<DeclRefNode>(value.expr);
      if (!declRef) {
        emitError(value.getLoc(), "Expected LLVM metadata name");
        continue;
      }
      metadataName = StringAttr::get(getContext(), declRef->spelling);
      metadataValue = nullptr;
    } else {
      if (!value.name) {
        emitError(value.getLoc(), "LLVM metadata requires a name");
        continue;
      }
      metadataName = value.name;
      metadataValue = value.expr;
    }

    // It might be possible that name comes from alias, therefore need to
    // analyze all module's aliases to see if alias's value needs to be used.
    if (std::optional<AliasDeclOp> aliasOp =
            getLLVMMetadataNameAlias(shared, sigDecl, metadataName))
      metadata.push_back(*aliasOp->getValue());
    else
      metadata.push_back(metadataName);

    if (metadataValue) {
      if (PValue attr = emitter.emitExprPValue(value.expr, EC_Decorator))
        metadata.push_back(attr);
    } else {
      // Store unit attr as value.
      metadata.push_back(UnitAttr::get(getContext()));
    }
  }
  return ArrayAttr::get(getContext(), metadata);
}

void FnSigDecorators::applyLLVMMetadata(const CallNode &node) {
  // Ignore empty metadata list.
  if (node.operands.empty())
    return;
  ArrayAttr metadata = getLLVMMetadataArray(node.operands);
  llvmMetadata.append(metadata.begin(), metadata.end());
}

void FnSigDecorators::registerLLVMArgMetadata(const CallNode &node) {
  if (node.operands.empty()) {
    emitError(node.getLoc(), "LLVM arg metadata requires an argument name");
    return;
  }

  Operand targetArg = node.operands[0];
  auto declRef = dyn_cast<DeclRefNode>(targetArg.expr);
  // We expect the first operand to be "positional", i.e. it should just be a
  // standalone name.
  if (targetArg.passKind != Operand::PassKind::kPositional || !declRef) {
    emitError(targetArg.getLoc(),
              "First argument of LLVM arg metadata must be an argument name");
    return;
  }

  // Ignore empty metadata list.
  if (node.operands.size() == 1)
    return;

  // Find argument number corresponding to this arg name.
  int64_t argIdx = -1;
  for (auto [index, arg] : llvm::enumerate(tcSignature.argList.parsedArgs)) {
    if (arg.name.getValue() == declRef->spelling) {
      argIdx = index;
      break;
    }
  }

  if (argIdx < 0) {
    emitError(targetArg.getLoc(), "No argument named ") << declRef->spelling;
    return;
  }

  // First time setting arg metadata, initialize with array of empty attributes.
  if (llvmArgMetadata.empty())
    llvmArgMetadata.insert(llvmArgMetadata.begin(),
                           tcSignature.argList.parsedArgs.size(),
                           ArrayAttr::get(getContext(), {}));

  llvmArgMetadata[argIdx] = getLLVMMetadataArray(node.operands.drop_front());
}

void FnSigDecorators::finalize() {
  if (!llvmArgMetadata.empty())
    funcOp.setLLVMArgMetadataArrayAttr(
        ArrayAttr::get(getContext(), llvmArgMetadata));

  if (!llvmMetadata.empty()) {
    // NOTE: @llvm_metadata are processed and added in reverse order
    funcOp.setLLVMMetadataArrayAttr(ArrayAttr::get(getContext(), llvmMetadata));
  }
}

static void processFunctionConformances(FnOp func, SharedState &shared,
                                        ASTDecl &decl) {
  FnTypeGeneratorType sig = func.getFullSignature();
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
      Type metatype = ASTType(type)
                          .getReferenceElementType()
                          .getVariadicPackInfo(shared)
                          .getVariadicElementType();
      type = ParamType::get(UnknownAttr::get(metatype));
      conv = ArgConvention::ReadReg;
    }
    if (hasAddress(conv))
      type = ASTType(type).getReferenceElementType();
    argTypes.push_back(type);
  }

  bool allVanillaKernelArgs = llvm::all_of(argTypes, [](ASTType astType) {
    if (auto structTy = dyn_cast<LIT::StructType>(astType.mlirType)) {
      return MOGGPreElab::isDPSTensor(structTy) ||
             MOGGPreElab::isMojoDeviceContextPtr(structTy);
    }
    return false;
  });

  // We don't need to attach the conformance attrs if we have a kernel working
  // purely with tensors
  if (allVanillaKernelArgs && resultType.isNoneType())
    return;

  ExprEmitter emitter(decl, EC_Type);
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
      for (StringRef traitName :
           {"UnknownDestructibility", "AnyType", "Copyable", "Movable"}) {
        auto traitDecl = cast_or_null<TraitDeclOp>(
            shared.lookupBuiltinTrait(traitName, &decl, node.getLoc()));
        if (!traitDecl)
          continue;
        TraitType trait = traitDecl.bindReference();
        if (PValue result =
                emitter.emitPValue({PValue(type), node}, EC_Trait, trait))
          conformances.push_back(result);
      }
      return conformances;
    }
    // Otherwise, generate bindings for each explicit conformance.
    assert(isa<StructMetaType>(metatype));
    auto structDecl = cast<StructDeclOp>(type.getDecl(shared));
    for (SymbolRefAttr parent : structDecl.getCanonicalTrait().getSymbols()) {
      auto trait = TraitType::get(parent);
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

  if (spelling != MOGGPreElab::Decorators::REGISTER_INTERNAL_FUNCTION)
    return;

  auto func = cast<FnOp>(decl);
  if (!(isa<FileModuleOp>(func->getParentOp()) || func.getIsStatic())) {
    shared.emitError(decl.getLoc(), "@")
        << spelling << " is only supported on top-level or static functions";
    return;
  }

  processFunctionConformances(func, shared, decl);
}

/// Given the lexical context of a function, return true if the default bit
/// for the function is capturing.
static bool isCapturingByDefault(FnOp funcOp, StructDeclOp parent,
                                 ArrayRef<ParamDeclAttr> paramDecls) {
  // Any function that contains a capturing closure as a parameter is itself
  // capturing, include parent struct parameters.
  mlir::AttrTypeWalker walker;
  walker.addWalk([](FuncType sig) {
    if (sig.isCapturing())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return llvm::any_of(
      llvm::concat<const ParamDeclAttr>(paramDecls, parent ? parent.getParams()
                                                           : std::nullopt),
      [&](ParamDeclAttr decl) { return walker.walk(decl).wasInterrupted(); });
}

std::pair<SmallVector<ParamDeclRefAttr>, FnTypeGeneratorType>
DeclResolver::createSelfContainedSignature(FnTypeGeneratorType original) {
  // Collect the subset of referenced parameters. Use a set vector to keep the
  // order deterministic.
  llvm::SmallSetVector<ParamDeclRefAttr, 4> capturedRefs;
  original.walk([&](ParamDeclRefAttr ref) { capturedRefs.insert(ref); });

  SmallVector<ParamDeclRefAttr> captured = capturedRefs.takeVector();
  // Unbind the N capture parameters, creating a FuncType with N new input
  // parameters prepended.
  // TODO: what if we capture a variadic?
  SmallVector<bool> variadicMask(captured.size(), false);
  auto unbound = FnTypeGeneratorType::prependParams(
      original,
      llvm::map_to_vector(
          captured,
          [](ParamDeclRefAttr ref) { return ParamDeclAttr::get(ref); }),
      variadicMask);
  return {std::move(captured), unbound};
}

static MLValue emitClosureInstance(ArrayRef<Capture> captures,
                                   ArrayRef<ParamDeclRefAttr> paramCaptures,
                                   ASTDecl &nestedFnDecl, SharedState &shared) {
  FnOp nestedFn = cast<FnOp>(nestedFnDecl);
  StringAttr fnName = nestedFn.getSourceNameAttr();
  SMLoc loc = nestedFnDecl.getLoc();
  Location mlirLoc = shared.translateLocation(loc);
  if (shared.diBuilder)
    mlirLoc = shared.diBuilder->createScopedLoc(mlirLoc);

  // Save the insertion point before closure creation since closure creation
  // nukes the nested function.
  ImplicitLocOpBuilder builder(mlirLoc, shared.getContext());
  builder.setInsertionPointAfter(nestedFn);
  OpBuilder::InsertPoint insertPoint = builder.saveInsertionPoint();
  ASTDecl *moduleDecl = nestedFnDecl.getNearestDeclOfType<FileModuleOp>();

  auto [capturedRefs, wrapperSig] = DeclResolver::createSelfContainedSignature(
      nestedFn.getFuncTypeGenerator());
  if (!wrapperSig)
    return {};
  StructDeclOp closureWrapper =
      shared.getOrCreateClosureWrapper(loc, wrapperSig, moduleDecl);
  if (!closureWrapper)
    return {};

  // Create an instance of the closure implementation in the parent function
  // right after the nested function definition.
  ClosureEmitter emitter(*moduleDecl, shared);
  StructDeclOp closureImpl =
      emitter.replaceNestedFunctionWithClosureImplStructDecl(
          captures, paramCaptures, nestedFnDecl, wrapperSig);

  emitter.createWrapperInitWithImpl(closureWrapper, closureImpl, loc);

  builder.restoreInsertionPoint(insertPoint);

  ExprEmitter exprEmitter(*nestedFnDecl.getParentDecl(), builder);
  SyntheticNode node(loc);

  // Pass all the captured values into the initializer.  In the case of a move
  // capture, this will be an RValue for the thing captured, transferring to the
  // owned argument in the initializer.
  CallOperands closureImplInitArgs;
  for (const Capture &capture : captures)
    closureImplInitArgs.add({capture.getValue(), node});

  // Create Closure Impl type by adding captured parameters to the ClosureImpl
  // DeclType.
  ValueDest closureDest(EC_Closure);
  Type closureImplType = closureImpl.bindReference(llvm::map_to_vector(
      paramCaptures, [](ParamDeclRefAttr ref) -> TypedAttr { return ref; }));

  CValue value = exprEmitter.emitConstructorCall(
      ASTType(closureImplType), std::move(closureImplInitArgs), node,
      CallSyntax::kTypeCall, closureDest);
  if (!value)
    return {};

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
                                  CallSyntax::kTypeCall, closureWrapperDest);
  return MLValue(var);
}

/// funcdef   ::=  [decorators] def_or_fn identifier [param_signature]
///                "(" [argument_list] ")" ["->" expression] ":" suite
/// def_or_fn ::= "def" | "fn"
///
LogicalResult DeclResolver::resolveSignature(FnOp funcOp, Lexer &lexer,
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

  // Parse declared parameters and add them to the current scope.
  ParsedParamList parsedParamList;

  // Add the parameters to the symbol table, and resolve their types.  We
  // add all of these after generic signature parsing so types used in the
  // signature list resolve to enclosing scopes, and we add them before the
  // value signature list so the types and parameters can resolve to the bound
  // values.
  if (parsedParamList.parseParametersIfPresent(p, ArgListKind::kParamList))
    return failure();
  TypeCheckedParamList paramList(parsedParamList.params, sigDecl);

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
  fnSignature.parseResultIfPresent(p);

  // Emit the argument and result types.
  SpecialFunctionInfo fnInfo = SpecialFunctionInfo::get(baseName);
  TypeCheckedFnSignature tcSignature(paramList, fnSignature,
                                     /*captureOrigins=*/nullptr, &decl, fnInfo);

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
  FnSigDecorators fnDecorators(decl, sigDecl, shared, baseName, tcSignature);
  Decorators(decl).applySignatureDecorators(
      decoratorExprs,
      [&](ExprNode *decorator) { return fnDecorators.applyOne(decorator); });
  fnDecorators.finalize();

  // Propagate errors and the parsed decls in the signature.
  decl.takeDecls(sigDecl);

  // Now that all the structural properties are determined, perform any
  // name-binding specific checks over the declaration.  This happens after
  // decorator processing because that is how defs work in Python.  This also
  // fills in any implicitly declared types.
  tcSignature.verifyFunctionNameBinding(decl, baseName, fnInfo);

  // Now that we've processed the signature, bail if we had a missing colon.
  if (p.parseToken(Token::colon, "expected ':' in function definition"))
    return failure();

  // Finally now that the full signature has been resolved, build our IR.

  // Handle argument effects and build the ASTDecls for the arguments.
  OpBuilder builder = decl.getDeclEndBuilder();
  NamedAttrList attrs = funcOp->getAttrDictionary();

  // Compute the signature of the function.
  FnTypeGeneratorType signature = tcSignature.getFnTypeGeneratorType();
  if (!signature)
    return failure();

  // The implicitOriginDecls don't affect the signature, but they do get
  // prepended onto the paramDecls list.
  ParamDeclArrayAttr paramsArrayAttr;
  if (tcSignature.implicitOriginDecls.empty()) {
    paramsArrayAttr =
        builder.getAttr<ParamDeclArrayAttr>(paramList.paramDeclAttrs);
  } else {
    SmallVector<ParamDeclAttr> mergedParams;
    llvm::append_range(mergedParams, paramList.paramDeclAttrs);
    llvm::append_range(mergedParams, tcSignature.implicitOriginDecls);
    paramsArrayAttr = builder.getAttr<ParamDeclArrayAttr>(mergedParams);
  }

  attrs.set(funcOp.getParamsAttrName(), paramsArrayAttr);
  attrs.set(funcOp.getFunctionTypeAttrName(),
            TypeAttr::get(tcSignature.getFunctionType()));

  // Now that the FunctionType is set to the pretty type that includes implicit
  // origins, we strip off the named origin decl references and replace them
  // with indices.
  signature = signature.replaceImplicitOriginsWithIndexes(
      tcSignature.implicitOriginDecls);
  attrs.set(funcOp.getFuncTypeGeneratorAttrName(), TypeAttr::get(signature));

  // Set the symbol to the mangled name and check for redefinition.
  attrs.set(funcOp.getSymNameAttrName(),
            getMangledName(baseName, *decl.getParentDecl(), signature));
  attrs.set(funcOp.getSourceNameAttrName(), baseName);

  // Set the result name binding if specified.
  if (StringAttr resultName = tcSignature.argList.resultArg.name)
    attrs.set(funcOp.getNamedResultAttrName(), resultName);

  // Remove the temporary "sym_namex" attribute set up in
  // StmtParser::parseDefFnStmt, see that method for an explanation.
  attrs.erase("sym_namex");

  // Bulk update the attributes.
  funcOp->setAttrs(attrs.getDictionary(funcOp.getContext()));

  // Set the symbol and notice if we are redeclaring something.
  if (Operation *existing = finalizeFuncSignature(funcOp, decl)) {
    const char *errorMessage = nullptr;
    auto existingFunc = cast<FnOp>(existing);

    // We need to compare the (name erased) user result types, since memory-only
    // types may result in `!kgen.none` in the mlir signature result.
    auto resTy = ASTType(signature.getUserResultType());

    // Loop through the args and check if any are keyword-only while overloading
    // the name of the argument.
    bool overloadedKeywordArgName = false;
    auto existingArgs =
        existingFunc.getFuncTypeGenerator().getArgListAttrs().getPogs();
    for (auto [arg, existingArg] :
         llvm::zip(tcSignature.argList.parsedArgs, existingArgs)) {
      if ((arg.kwArgHandling == KWArgHandling::kKeywordOnly ||
           existingArg.getPassingKind() == PassingKind::KwOnly) &&
          arg.name != existingArg.getName()) {
        overloadedKeywordArgName = true;
        break;
      }
    }

    auto existingResTy =
        ASTType(existingFunc.getFuncTypeGenerator().getUserResultType());

    if (!resTy.isEqualCanon(existingResTy))
      errorMessage = " cannot overload on return type only";
    else if (!overloadedKeywordArgName)
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

  auto notify = llvm::make_scope_exit(
      [&] { shared.notifyListenerOnFunctionDecl(decl, identifierLoc); });

  // Upon fully resolving a nonparametric closure, immediately materialize it as
  // a runtime value. It cannot be used as a parameter.
  if (!funcOp->getParentOfType<FnOp>())
    return success();

  // Fully resolve the body so we can swap the IR value of the decl. Later on,
  // we will need this to determine the capture signature.
  decl.resolvedness = DeclResolvedness::body;
  if (failed(resolveBody(funcOp, lexer, decl)))
    return failure();

  // Find all parameter captures in the function body.
  ParameterCollector::Analysis collectorCache;
  ParameterUseDefGraph graph(funcOp.getBodyRegion());
  graph.calculate(collectorCache);

  // Get captured parameters that cross with captured values.
  ParameterCollector collector(collectorCache);
  SmallVector<Capture> captures;
  SmallVector<ParamDeclRefAttr> capturedUses;
  for (auto &[_, capture] : shared.getCaptureRangeInScope(decl)) {
    captures.push_back(capture);
    bool unused = false;
    collector.collectUsesFromType(capture.getValue().getType(), capturedUses,
                                  unused);
  }
  for (ParamDeclRefAttr use : capturedUses)
    graph.usesFromAbove.insert(use);

  SmallVector<ParamDeclRefAttr> paramCaptures =
      graph.usesFromAbove.takeVector();

  // If this is a `@parameter` closure, attach the capture origins.
  if (signature.isCapturing()) {
    SmallVector<Type> captureTypes;
    for (const Capture &cap : captures)
      captureTypes.push_back(cap.getValue().getType());
    for (ParamDeclRefAttr param : paramCaptures)
      captureTypes.push_back(param.getType());

    SmallVector<TypedAttr> origins =
        shared.cachedOriginFinder.findOriginsIn(captureTypes);
    signature = signature.getWithBody(signature.getBody().getWithMetadata(
        signature.getFnMetadata().addCaptureOrigins(
            OriginSetAttr::get(getContext(), origins))));
    funcOp.setFuncTypeGenerator(signature);

    funcOp.setParamDeclAttr(
        ParamDeclAttr::get(funcOp.getSymNameAttr(), signature));
    funcOp.removeSymNameAttr();
    return success();
  }

  // If the function doesn't actually capture anything, don't demote it to a
  // runtime value.
  if (!signature.isEscaping() && captures.empty()) {
    funcOp.setParamDeclAttr(
        ParamDeclAttr::get(funcOp.getSymNameAttr(), signature));
    funcOp.removeSymNameAttr();
    return success();
  }

  if (!paramList.paramDeclAttrs.empty())
    return emitError(funcOp.getLoc(), "TODO: closures cannot have parameters");

  // Emit closure structures necessary for instantiating an escaping closure
  signature = signature.getWithBody(signature.getBody().getWithFnEffects(
      signature.getFnEffects().setEscaping()));
  funcOp.setFuncTypeGenerator(signature);
  MLValue instance = emitClosureInstance(captures, paramCaptures, decl, shared);
  if (!instance)
    return failure();
  decl.setIRValue(instance);

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
  auto varListStruct =
      dyn_cast_if_present<StructDeclOp>(varListType.getDecl(emitter.shared));
  if (!varListStruct) {
    emitter.emitError(loc, "malformed VariadicListInMem");
    return {};
  }

  // Bind the "is_owned" parameter, start by filling the parameter list with ?.
  if (refType) {
    SmallVector<TypedAttr> typeParams;
    ParameterEvaluator evaluator;
    for (Type type : varListStruct.getSignature().getParamTypes()) {
      typeParams.push_back(UnboundAttr::get(evaluator.getReboundType(type)));
      evaluator.addInputValue(typeParams.back());
    }

    // The last parameter is the "is_owned" parameter.
    // Emit the "is_owned" parameter.
    auto isOwnedAttr =
        BoolAttr::get(emitter.getContext(),
                      variadicType.getConvention() == ArgConvention::OwnedMem);
    SyntheticNode locExpr(loc);
    PValue isOwnedVal = // Convert to Bool.
        emitter.emitPValue({isOwnedAttr, &locExpr}, EC_Type,
                           typeParams.back().getType());
    if (!isOwnedVal)
      return {};
    typeParams.back() = isOwnedVal;

    varListType = varListStruct.bindReference(typeParams);
    assert(varListType && "Failed to bind type params");
  }

  // Emit a VarDeclOp: VaridicListMem needs a origin for its self accesses.
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

ParseResult DeclResolver::resolveBody(FnOp funcOp, Lexer &lexer,
                                      ASTDecl &decl) {
  Block &body = *funcOp.getBody();
  auto endFn = cast<EndFnOp>(body.front());

  // Push the debug scope for this function if necessary so that nested
  // operations have proper debug info.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder) {
    diScopeGuard = shared.diBuilder->pushScopeGuard(funcOp.getLocScope());

    // Reset the location on the endfn to correct debug scope.
    endFn->setLoc(shared.translateLocation(decl.getLoc()));
  }
  // About to parse the body.
  endFn.setUnresolved(false);

  // If this is a method in a trait, we only allow a "..."
  if (isa<TraitDeclOp>(*decl.getParentDecl())) {
    // Skip any docstring's that might be present.
    ParserBase p(shared, lexer);
    p.parseDocString(decl);

    // If we see an ellipsis, the function member is well formed: don't emit
    // arguments or any other setup logic.
    if (p.consumeIf(Token::dot_dot_dot) || p.consumeIf(Token::kw_pass)) {
      body.front().erase(); // Remove the lit.endfn op to replace it.
      OpBuilder::atBlockEnd(&body).create<UnreachableOp>(funcOp.getLoc());
      return success();
    }

    // Otherwise, must be a default implementation.  Parse it and then emit an
    // error later.
  }

  // Set up information about value arguments, emitting before the lit.endfn.
  ExprEmitter emitter(decl, OpBuilder(&body.front()));

  // Set up the body of the fn/def, creating declarations for the value
  // parameters and adding them to the symbol table.
  FnTypeGeneratorType funcSignature = funcOp.getFuncTypeGenerator();
  for (auto [argIdxX, bbArg, convention] :
       llvm::enumerate(funcOp.getBody()->getArguments(),
                       funcSignature.getArgConventions())) {
    size_t argIdx = argIdxX;

    StringAttr argName = funcSignature.getArgName(argIdx);

    // Figure out which decl corresponds to this argument so we can finish it.
    ArrayRef<ASTDecl *> argDeclList = decl.lookupInCurrentScope(argName);

    // Don't bind anonymous result slots, they don't have a decl.
    if (argDeclList.empty() && isResultSlot(convention))
      continue;

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

    // Ref convention works with registers and def functions without any funny
    // business.
    if (convention == ArgConvention::Ref ||
        convention == ArgConvention::MutRef ||
        convention == ArgConvention::OwnedMem ||
        convention == ArgConvention::Mut ||
        convention == ArgConvention::ByRefResult) {
      setDecl(CValue::getMValueForRef(bbArg));
      continue;
    }

    CValue argValue;
    if (convention == ArgConvention::ReadMem)
      argValue = MBValue(bbArg); // borrowed
    else {
      assert(convention == ArgConvention::ReadReg);
      // borrowed_in_reg is used for @register_passable("trivial") types, where
      // borrowed vs owned doesn't matter so we use SRValue.
      argValue = SRValue(bbArg);
    }

    if (!funcOp.isDef()) { // Don't bother 'fn' arguments.
      setDecl(argValue);
    } else {
      // Borrowed arguments in 'def's get a special wrapper that allows them to
      // be made lazily mutable on demand.
      setDecl(DLValue(RCRef<DefArgumentWrapperDLValue>::create(
          &argDecl, argValue, argValue.getRValueType(), argIdx)));
    }
  }

  // If we had a named result in a register, create a var decl to hold the
  // temporary and register it for name lookup.
  if (!funcSignature.hasMemoryOnlyResult() && funcOp.getNamedResultAttr()) {
    // Emit a VarDeclOp for the temporary within the function.  This makes it
    // assignable etc.
    // This also provides a user name for the argument.
    StringAttr resultName = funcOp.getNamedResultAttr();
    // If this is the 'out' argument of an initializer, we use a special
    // VarDeclKind so CheckLifetimes knows the whole object bit is live on
    // input.
    auto kind = funcOp.getSpecialFunctionInfo().isInitializer()
                    ? VarDeclKind::InitOutArg
                    : VarDeclKind::Arg;
    VarDeclOp varDecl = emitter.emitVarDecl(
        resultName, funcOp.getUserResultType(), funcOp.getLoc(), kind);
    ASTDecl &argDecl = addFullyResolvedDecl(MLValue(varDecl), resultName,
                                            decl.getLoc(), &decl);
    shared.notifyListenerOnArgumentDecl(argDecl, resultName, argDecl.getLoc());
  }

  // With all the argument declarations set up, we can resolve the body of the
  // function.
  if (ParserBase(shared, lexer).parseSuite(decl))
    return failure();

  // If this decl or a parent is erroneous, return before emitting.  There is no
  // point to emitting after errors, and we might trip assertions because
  // erroneous decls don't respect invariants.
  if (decl.isErroneous() || decl.getParentDecl()->isErroneous())
    return success();

  // Determine whether we need an implicit return at the end of the function.
  // An implicit return is generated for functions that return None.
  bool needDefaultReturn = false;
  if (ASTType(funcOp.getUserResultType()).isNoneType() ||
      funcOp.getNamedResultAttr())
    needDefaultReturn = true;

  // We can elide the boilerplate if we can trivially the user already has a
  // return. This won't catch cases where an 'if' has two returns in the bodies
  // etc but is enough to avoid generating IR noise.
  if (emitter.builder->getInsertionPoint() != body.begin()) {
    if (isa<LIT::ReturnOp, LIT::RaiseOp>(
            std::prev(emitter.builder->getInsertionPoint())))
      needDefaultReturn = false;
  }

  // Emit a default "return None" if the function returns nothing.
  if (needDefaultReturn)
    emitter.emitNormalReturn(funcOp.getLoc(), Value(), /*emitEndFunc=*/false);

  // Now that the body of the function is parsed, run any body decorators.
  Decorators(decl).applyBodyDecorators([&](ExprNode *decorator) {
    processExtensibilityDecorator(shared, decl, decorator);
    return failure();
  });

  // If this function is @always_inline("builtin"), check that its body obeys
  // the right invariants.
  if (funcOp.getInlineLevel() == InlineLevel::AlwaysBuiltin) {
    if (failed(FnSigDecorators::checkAlwaysInlineBuiltin(funcOp, shared)))
      funcOp.setInlineLevel(InlineLevel::AlwaysNoDebug);
  }

  // We don't support default implementations for trait methods yet. Reject and
  // recover cleanly.
  if (isa<TraitDeclOp>(*decl.getParentDecl())) {
    shared.emitError(decl.getLoc(),
                     "unexpected function body in trait function "
                     "declaration, use `...`");
    return success();
  }

  auto declOp = dyn_cast<LIT::StructDeclOp>(funcOp->getParentOp());
  if (!declOp)
    return success();

  for (auto decorator : declOp.getDecorators()) {
    if (extractDecoratorName(decorator) == "register" &&
        MOGGPreElab::fnNeedsConformances(funcOp))
      processFunctionConformances(funcOp, shared, decl);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Module Decl implementation
//===----------------------------------------------------------------------===//

ParseResult DeclResolver::resolveBody(LIT::FileModuleOp op, Lexer &lexer,
                                      ASTDecl &decl) {
  // Push a scope for the file of this module.
  DebugInfo::DIBuilder::ScopeGuard fileGuard;
  if (shared.diBuilder) {
    FileLineColLoc loc =
        shared.diags.translateLocation(lexer.getToken().getLoc());
    if (loc)
      fileGuard = shared.diBuilder->pushFile(loc.getFilename().getValue());
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
    if (failed(resolveBody(initDecl, decl.loc)))
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
  auto decoratorExprs = p.parseDecorators(decl);
  rejectDecorators(decoratorExprs, decl, shared);

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
  ExprEmitter emitter(*decl.getParentDecl(), EC_VarInit);
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

  shared.notifyListenerOnVariableDecl(decl, identifierLoc);
  return success();
}

ParseResult DeclResolver::resolveBody(GlobalVarDeclOp op, Lexer &lexer,
                                      ASTDecl &decl) {
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
  Decorators(decl, /*signatureOnly=*/true)
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

  ASTDecl &parentDecl = *decl.getParentDecl();

  NamedAttrList attrs = aliasDeclOp->getAttrDictionary();
  if (p.consumeIf(Token::equal)) {
    // Then this is a normal `alias` declaration with an initializer.
    ExprNode *initExpr = nullptr;
    if (p.parseExpression(initExpr, decl.getIndentation()))
      return failure();

    if (isa<LIT::TraitDeclOp>(parentDecl)) {
      p.emitError(identifierLoc) << "associated alias declarations in a trait "
                                    "shouldn't have an initializer";
      // Don't return; continue parsing as if it has no name, so that references
      // to the name will resolve.
    } else {
      ExprEmitter emitter(parentDecl, EC_AliasValue);

      // Emit the value and convert to the expected type if we know it.
      auto rhsValue = emitter.emitExprPValue(initExpr, EC_AliasValue, type);
      if (!rhsValue)
        return failure();

      // If we had no declared type (`alias x = 42`), infer the type from the
      // initializer.
      if (!type)
        type = rhsValue.getType();

      // Remember the value
      attrs.set(aliasDeclOp.getValueAttrName(), rhsValue.get());
    }
  } else {
    if (!isa<LIT::TraitDeclOp>(parentDecl)) {
      // Disallow this, because it would create diamond inheritance problems.
      p.emitError(identifierLoc)
          << "only traits may contain an alias without an initializer";
      return failure();
    }

    if (!type) {
      p.emitError(identifierLoc)
          << "alias without initial value must have a type";
      return failure();
    }
  }
  // Update the type from UnresolvedType
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

/// For a struct or trait declaration, parse an optional list of parent traits
/// to inherit from. `inheritedFrom` is a map from each inherited symbol to the
/// first symbol that explicitly inherits from it.
static ParseResult
parseOptionalInheritanceList(ParserBase &p, ASTDecl &declScope, ASTDecl &decl,
                             StringRef declName, SharedState &shared) {
  if (!p.consumeIf(Token::l_paren) || p.consumeIf(Token::r_paren))
    return success();

  DenseMap<SymbolRefAttr, std::pair<SymbolRefAttr, SMLoc>> *inheritedFrom =
      decl.getTraitConformanceLineage(/*createIfMissing=*/true);

  auto parseParent = [&]() -> ParseResult {
    ASTType type;
    SMLoc loc;
    if (p.getLocation(loc) ||
        parseType(p, type, declScope, declScope.getIndentation()))
      return failure();

    // Reject inheriting from types we don't support yet.
    auto traitType = dyn_cast<TraitType>(type);
    if (!traitType) {
      if (isa<LIT::StructType>(type)) {
        p.emitError(loc) << "inheriting from structs is not allowed";
      } else if (isa<ParamType>(type)) {
        p.emitError(loc)
            << "inheriting from a parameter expression is not allowed";
      } else {
        p.emitError(loc) << "don't know how to inherit from this type";
      }
      declScope.setErroneous();
      return success();
    }

    auto symbols = traitType.getSymbols();

    // If the user explicitly inherited a trait that is already provided
    // elsewhere, provide a warning.
    if (symbols.size() == 1) {
      auto symbol = symbols.front();
      auto [it, inserted] =
          inheritedFrom->try_emplace(symbol, std::make_pair(symbol, loc));
      if (!inserted) {
        auto [cur, curLoc] = it->second;
        InflightDiag diag = shared.emitWarning(loc, "'")
                            << declName << "' already inherits from "
                            << ASTType(TraitType::get(symbol));
        if (cur == symbol)
          diag.attachNote(curLoc) << "previously inherited here";
        else
          diag.attachNote(curLoc) << "inherited through "
                                  << ASTType(TraitType::get(cur)) << " here";
      }
    }

    // Successively flatten the parent list so we always have all the parents
    // available to check.
    // TODO: Encode an "inherited from" here, to make diagnostics nice.
    for (SymbolRefAttr symbol : symbols) {
      ASTDecl &traitDecl = shared.declResolver->getDeclForTypeSymbol(symbol);
      TraitType canonicalParent =
          cast<TraitDeclOp>(traitDecl).getCanonicalTrait();
      for (SymbolRefAttr parent : canonicalParent.getSymbols())
        inheritedFrom->try_emplace(parent, std::make_pair(symbol, loc));
    }
    return success();
  };
  if (p.parseCommaSeparatedList(parseParent, Token::r_paren) ||
      p.parseToken(Token::r_paren, "expected ')' for parameter list"))
    return failure();
  return success();
}

bool isTrivialRegisterPassable(CallNode *callNode) {
  if (auto declRef = dyn_cast<DeclRefNode>(callNode->callee)) {
    if (declRef->spelling == "register_passable" &&
        callNode->operands.size() == 1 &&
        callNode->operands[0].isPositionalStringLiteral("trivial")) {
      return true;
    }
  }
  return false;
}

static LogicalResult processTraitSignatureDecorator(ExprNode *decorator,
                                                    TraitDeclOp traitOp,
                                                    SharedState &shared,
                                                    ASTDecl &traitDecl) {
  if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
    if (declRef->spelling == "register_passable") {
      traitOp.setConvention(TypeConvention::RegisterPassable);
      return success();
    }
    // We don't process @explicit_destroy here, we do it in resolveSignature.
  }
  if (auto callNode = dyn_cast<CallNode>(decorator)) {
    if (isTrivialRegisterPassable(callNode)) {
      traitOp.setConvention(TypeConvention::RegisterPassableTrivial);
      return success();
    }
  }
  return failure();
}

/// Process a decorator that is resolved at the signature phase of resolution
/// and return success, otherwise failure if it is handled later.
static LogicalResult
processStructSignatureDecorator(ExprNode *decorator, StructDeclOp structOp,
                                SharedState &shared, ASTDecl &structDecl,
                                SmallVectorImpl<SymbolRefAttr> &traits) {
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
        traits.push_back(decl->getSymbolRef());
      if (ASTDecl *decl = shared.lookupBuiltinTrait(
              "Movable", structDecl.getParentDecl(), decorator->getLoc()))
        traits.push_back(decl->getSymbolRef());
      // Fallthrough the decorator to body resolution.
      return failure();
    }
    // We don't process @explicit_destroy here, we do it in resolveSignature.
  }

  // `x()` forms.
  if (auto callNode = dyn_cast<CallNode>(decorator)) {
    if (auto declRef = dyn_cast<DeclRefNode>(callNode->callee)) {
      // @register_passable("trivial")
      if (isTrivialRegisterPassable(callNode)) {
        structOp.setConvention(TypeConvention::RegisterPassableTrivial);
        return success();
      }

      // @nonmaterializable(TargetType)
      if (declRef->spelling == "nonmaterializable" &&
          callNode->operands.size() == 1) {
        if (auto drn = dyn_cast<DeclRefNode>(callNode->operands[0].expr)) {
          ASTDecl *parentDecl = structDecl.getParentDecl();
          ExprEmitter emitter(*parentDecl, EC_Type);
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

  SMLoc identifierLoc;
  if (p.parseToken(Token::kw_struct,
                   "internal error: checked by stmt parser") ||
      p.parseIdentifier("internal error: checked by stmt parser",
                        &identifierLoc) ||
      parsedParams.parseParametersIfPresent(p, ArgListKind::kParamList) ||
      parseOptionalInheritanceList(p, sigDecl, decl, structOp.getSymName(),
                                   shared) ||
      p.parseToken(Token::colon, "expected ':' in struct definition") ||
      decl.isErroneous())
    return failure();

  TypeCheckedParamList paramSignature(parsedParams.params, sigDecl);

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

  SmallVector<SymbolRefAttr> parentTraits;
  if (auto *inheritedFrom = decl.getTraitConformanceLineage())
    for (auto [symbol, _] : *inheritedFrom)
      parentTraits.push_back(symbol);

  // Make every nominal struct type inherit from `UnknownDestructibility`.
  if (ASTDecl *traitDecl = shared.lookupBuiltinTrait(
          "UnknownDestructibility", decl.getParentDecl(), decl.getLoc()))
    parentTraits.push_back(traitDecl->getSymbolRef());

  // This is a struct, so we can use 'computeSelfTypeForStruct' to figure out
  // the self type.
  decl.setTypeDeclSelf(ASTDecl::computeSelfTypeForStruct(structOp));

  // Structs are memory-only unless they opt-in to being passed in registers.
  structOp.setConvention(TypeConvention::MemoryOnly);

  // Now that we have the basic struct set up, process signature decorators.
  Decorators(decl).applySignatureDecorators(
      decoratorExprs, [&](ExprNode *decorator) {
        return processStructSignatureDecorator(decorator, structOp, shared,
                                               decl, parentTraits);
      });
  std::string linearTypeErrorMsg;
  for (auto decoratorExpr : decoratorExprs) {
    if (auto *declRefNode = dyn_cast<DeclRefNode>(decoratorExpr.first)) {
      // TODO(MOCO-1468): Remove this, always require argument to
      // @explicit_destroy.
      if (declRefNode->spelling == "explicit_destroy") {
        linearTypeErrorMsg =
            "Unhandled explicit_destroy type " + structOp.getDeclName().str();
      }
    } else if (auto *callNode = dyn_cast<CallNode>(decoratorExpr.first)) {
      if (auto declRef = dyn_cast<DeclRefNode>(callNode->callee)) {
        if (declRef->spelling == "explicit_destroy") {
          // TODO(MOCO-1468): Remove this, always require argument to
          // @explicit_destroy.
          if (callNode->operands.size() == 0) {
            linearTypeErrorMsg = "Unhandled explicit_destroy type " +
                                 structOp.getDeclName().str();
          } else {
            auto strExpr =
                dyn_cast<StringLiteralNode>(callNode->operands.front().expr);
            // TODO(MOCO-1468): Error message here.
            if (!strExpr)
              return failure();
            linearTypeErrorMsg = strExpr->getValue();
          }
        }
      }
    }
  }
  // TODO(MOCO-1468): Remove else; always require argument to @explicit_destroy.
  if (!linearTypeErrorMsg.empty()) {
    structOp.setLinearTypeErrorMsg(
        std::make_optional(llvm::StringRef(linearTypeErrorMsg)));
  } else {
    if (ASTDecl *implicitlyDestructibleDecl = shared.lookupBuiltinTrait(
            "AnyType", decl.getParentDecl(), decl.getLoc())) {
      parentTraits.push_back(implicitlyDestructibleDecl->getSymbolRef());
    }
  }

  structOp.setCanonicalTrait(getCanonicalTrait(parentTraits));

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
  FnOp func = dyn_cast<FnOp>(delDecl);
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
                                              SpecialFunctionKind specialKind) {
  const char *name = SpecialFunctionInfo::get(specialKind).name;
  LookupResult inits = structDecl.getShared().lookupAndResolveDecl(
      name, structDecl.getLoc(), structDecl, /*searchParentScopes=*/false);

  for (ASTDecl *candidate : inits.getIfSuccess()) {
    FnOp func = dyn_cast<FnOp>(candidate);
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

  LogicalResult processDecorator(ExprNode *decorator, StructDeclOp structOp,
                                 FnOp moveFunc, FnOp copyFunc);

private:
  /// Process the @value body decorator on structs.  This synthesizes the
  /// memberwise init, copy ctor and move ctor if requested.
  void processValueDecorator(SMLoc decoratorLoc, FnOp moveFunc, FnOp copyFunc);

  /// Get a constant symbol to a method, and return null if it is missing or
  /// something went wrong.
  /// Provide optionally a callback for the case where the method is missing.
  SymbolConstantAttr
  getSymbolForMethod(StringRef methodName, ExprNode *decorator,
                     function_ref<void()> callbackOnMissing = nullptr);

  StructDeclOp structOp;
  ASTDecl &structDecl;
  ArrayRef<std::pair<StructFieldOp, ASTDecl *>> structFields;
};
} // namespace

/// Synthesize the `__copyinit__` and `__moveinit__` stubs for `@value`
/// decorated structs early to ensure their movability and copyability
/// requirements are satisfied.
static std::pair<FnOp, FnOp> preprocessValueDecorator(ASTDecl &structDecl) {
  auto declOp = cast<StructDeclOp>(structDecl);
  for (ExprNode *decorator : structDecl.getBodyDecorators()) {
    if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
      if (declRef->spelling == "value") {
        std::optional<ValueInfo> info = ValueInfo::createValueInfo(structDecl);
        if (!info)
          break;
        StructEmitter emitter(structDecl.getShared());
        FnOp moveFunc, copyFunc;
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
                                                 FnOp moveFunc, FnOp copyFunc) {
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

  if (FnOp copyCtr = stubs->copyCtr) {
    SymbolConstantAttr ref = copyCtr.getBoundSymbolRef();
    ASTDecl *copyCtrDecl =
        getDeclResolver().getDeclForFuncSymbol(ref.getSymbol());
    if (failed(structEmitter.populateMoveCopy(*copyCtrDecl, /*isMove=*/false)))
      shared.deleteDecl(*copyCtrDecl);
    else
      declOp.setCopyInitAttr(ref);
  }
  if (FnOp moveCtr = stubs->moveCtr) {
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
  auto methods = OverloadSet::lookup(
      structDecl, structDecl.getTypeDeclSelf(), methodName, decorator,
      CallSyntax::kMethodCallSynthetic, callbackOnMissing);

  // Case where we did not find the `impl` method or an error occured.
  if (!methods)
    return {};

  // Emit the constant symbol.
  auto methodsUValue = OverloadSetUValue::create(std::move(methods));
  ExprEmitter emitter(structDecl, {});
  PValue value =
      emitter.emitPValue({methodsUValue, decorator}, ExprContext::EC_Decorator);
  if (!value)
    return {};

  return cast<SymbolConstantAttr>(value.get());
}

LogicalResult StructBodyDecorators::processDecorator(ExprNode *decorator,
                                                     StructDeclOp structOp,
                                                     FnOp moveFunc,
                                                     FnOp copyFunc) {
  // @value decorator
  if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
    if (declRef->spelling == "value") {
      processValueDecorator(decorator->getRangeStart(), moveFunc, copyFunc);
      return success();
    }
    if (declRef->spelling == "explicit_destroy") {
      return success();
    }
  }
  if (auto callNode = dyn_cast<CallNode>(decorator)) {
    if (auto declRef = dyn_cast<DeclRefNode>(callNode->callee)) {
      if (declRef->spelling == "explicit_destroy") {
        return success();
      }
    }
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

ParseResult DeclResolver::resolveBody(StructDeclOp structOp, Lexer &lexer,
                                      ASTDecl &structDecl) {
  // Push the debug scope for this struct if necessary so that nested operations
  // have proper debug info.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(structOp.getLocScope());

  if (ParserBase(shared, lexer).parseSuite(structDecl))
    return failure();

  // TODO(MOCO-1468): Pull this out into a helper.
  bool implicitlyDestructible = false;
  for (SymbolRefAttr symbol : structOp.getCanonicalTrait().getSymbols()) {
    ASTDecl &parentDecl = getDeclForTypeSymbol(symbol);
    if (auto parentTrait = dyn_cast<TraitDeclOp>(parentDecl)) {
      if (parentTrait.getSymName() == "AnyType") {
        implicitlyDestructible = true;
        break;
      }
    }
  }

  // Check to see if there is a destructor and install it into the StructDeclOp
  // if so.
  if (auto dtorAttr = lookupDestructor(structDecl, shared)) {
    // Check to see if we have an explicitly declared destructor.
    structOp.setDestructorAttr(dtorAttr);
  } else if (structDecl.getTypeDeclSelf() && implicitlyDestructible &&
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
    if (auto copyInitAttr =
            lookupSpecialMethod(structDecl, SpecialFunctionKind::kCopyInit))
      structOp.setCopyInitAttr(copyInitAttr);
    if (auto moveInitAttr =
            lookupSpecialMethod(structDecl, SpecialFunctionKind::kMoveInit))
      structOp.setMoveInitAttr(moveInitAttr);
  }

  // If the struct is decorated with `@value`, make sure to synthesize the copy
  // and move constructors before the field types are signature resolved to
  // ensure that the Copyable and Movable trait requirements are satisfied.
  // FIXME: The order of decorator resolution here is a bit gross.
  auto [moveFunc, copyFunc] = preprocessValueDecorator(structDecl);

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
  if (hasBadField && !structDecl.getBodyDecorators().empty()) {
    structDecl.setErroneous();
    return failure();
  }

  // If there are any body decorators, resolve them now.
  StructBodyDecorators structDecorators(structOp, structDecl, *this,
                                        structFields);
  Decorators(structDecl)
      .applyBodyDecorators(
          [&, moveFunc = moveFunc, copyFunc = copyFunc](ExprNode *decorator) {
            return structDecorators.processDecorator(decorator, structOp,
                                                     moveFunc, copyFunc);
          });

  if (structDecl.isErroneous())
    return success();

  // Finally, emit empty conformance tables.
  ImplicitLocOpBuilder b = ImplicitLocOpBuilder::atBlockEnd(
      structOp.getLoc(), &structOp.getFields().front());
  for (SymbolRefAttr parent : structOp.getCanonicalTrait().getSymbols()) {
    StringAttr name = b.getStringAttr(getFlattenedSymbolName(parent));
    ConformanceOp witnessTable = b.create<ConformanceOp>(name, parent);
    witnessTable.getBody().push_back(new Block());
    ASTDecl &decl = addDecl(witnessTable, structDecl.getLoc(), name,
                            &structDecl, {}, {}, -1);
    decl.resolvedness = DeclResolvedness::signature;
    // Conformances are always created as signature-resolved because there's no
    // less-resolved state for it (see CALROC for more).
  }
  return success();
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
  Decorators(decl).applySignatureDecorators(
      decoratorExprs, [&](ExprNode *decorator) {
        return processTraitSignatureDecorator(decorator, traitOp, shared, decl);
      });

  // TODO(MOCO-1468): Pull this out into a common helper.
  ArrayRef<ExprNode *> bodyDecorators = decl.getBodyDecorators();
  for (auto decorator : bodyDecorators) {
    if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
      if (declRef->spelling == "explicit_destroy") {
        continue;
      }
    }
    if (auto callNode = dyn_cast<CallNode>(decorator)) {
      if (auto declRef = dyn_cast<DeclRefNode>(callNode->callee)) {
        if (declRef->spelling == "explicit_destroy") {
          continue;
        }
      }
    }
    emitError(bodyDecorators.front()->getLoc(), "unrecognized body decorators ")
        << SourceRange(bodyDecorators.front()->getRangeStart(),
                       bodyDecorators.back()->getRangeEnd());
  }

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

  // Map from each symbol to the first symbol that explicitly inherits from it.
  if (parseOptionalInheritanceList(p, *decl.getParentDecl(), decl,
                                   traitOp.getSymName(), shared))
    return failure();
  SmallVector<SymbolRefAttr> parentTraits;
  if (auto *inheritedFrom = decl.getTraitConformanceLineage())
    for (auto [symbol, _] : *inheritedFrom)
      parentTraits.push_back(symbol);

  if (p.parseToken(Token::colon, "expected ':' in trait definition"))
    return failure();

  // TODO(MOCO-1468): Remove this, put an @explicit_destroy on
  // UnknownDestructibility's definition.
  if (traitOp.getSymName() == "UnknownDestructibility") {
    // TODO(MOCO-1468): Remove this, specify it in the code.
    traitOp.setLinearTypeErrorMsg(std::make_optional(llvm::StringRef(
        "Unhandled explicit_destroy type UnknownDestructibility")));
  }

  // Make every trait inherit from `UnknownDestructibility`, except itself.
  if (parentTraits.empty() &&
      traitOp.getSymName() != "UnknownDestructibility") {
    if (ASTDecl *unknownDestructibilityDecl = shared.lookupBuiltinTrait(
            "UnknownDestructibility", decl.getParentDecl(), decl.getLoc())) {
      parentTraits.push_back(unknownDestructibilityDecl->getSymbolRef());
    }
  }

  std::string linearTypeErrorMsg;
  for (auto decoratorExpr : decoratorExprs) {
    if (auto *declRefNode = dyn_cast<DeclRefNode>(decoratorExpr.first)) {
      // TODO(MOCO-1468): Remove this, always require argument to
      // @explicit_destroy.
      if (declRefNode->spelling == "explicit_destroy") {
        linearTypeErrorMsg =
            "Unhandled explicit_destroy type " + traitOp.getDeclName().str();
      }
    } else if (auto *callNode = dyn_cast<CallNode>(decoratorExpr.first)) {
      if (auto declRef = dyn_cast<DeclRefNode>(callNode->callee)) {
        if (declRef->spelling == "explicit_destroy") {
          // TODO(MOCO-1468): Remove this, always require argument to
          // @explicit_destroy.
          if (callNode->operands.size() == 0) {
            linearTypeErrorMsg = "Unhandled explicit_destroy type " +
                                 traitOp.getDeclName().str();
          } else {
            auto strExpr =
                dyn_cast<StringLiteralNode>(callNode->operands.front().expr);
            // TODO(MOCO-1468): Error message here.
            if (!strExpr)
              return failure();
            linearTypeErrorMsg = strExpr->getValue();
          }
        }
      }
    }
  }
  // TODO(MOCO-1468): Remove else; always require argument to @explicit_destroy.
  if (!linearTypeErrorMsg.empty()) {
    traitOp.setLinearTypeErrorMsg(
        std::make_optional(llvm::StringRef(linearTypeErrorMsg)));
  } else {
    // Make every trait inherit from `AnyType`, except itself and
    // UnknownDestructibility.
    if (traitOp.getSymName() != "AnyType" &&
        traitOp.getSymName() != "UnknownDestructibility") {
      if (ASTDecl *anyTypeDecl = shared.lookupBuiltinTrait(
              "AnyType", decl.getParentDecl(), decl.getLoc())) {
        parentTraits.push_back(anyTypeDecl->getSymbolRef());
      }
    }
  }

  // Insert the implicit trait parameter:
  // - _Self: a value of this trait type - the struct conforming to this trait.
  auto actualType = ParamDeclAttr::get(decl.mangleParamName("_Self"),
                                       traitOp.bindReference());

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

  // Add the trait itself to its canonical trait list.
  parentTraits.push_back(getFullyResolvedSymbolRef(traitOp));
  TraitType canonTrait = getCanonicalTrait(parentTraits);
  traitOp.setCanonicalTrait(canonTrait);

  decl.setTypeDeclSelf(ASTDecl::computeSelfTypeForTrait(traitOp));

  shared.notifyListenerOnTraitDecl(decl, identifierLoc);

  return success();
}

namespace {
/// This replaces one attribute with another without respect to its original
/// type.  TODO: Is there a better way to do this?
struct AttrReplacer : public IndexParameterReplacer<AttrReplacer> {
  TypedAttr oldAttrValue, newAttrValue;

  AttrReplacer(TypedAttr oldAttrValue, TypedAttr newAttrValue)
      : oldAttrValue(oldAttrValue), newAttrValue(newAttrValue) {}

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
static void replaceTraitMethodSelfTypes(FnOp func, TypedAttr parentSelfType,
                                        TypedAttr traitSelfType) {
  assert(isa<ParamDeclRefAttr>(parentSelfType) &&
         isa<ParamDeclRefAttr>(traitSelfType));
  AttrReplacer replacer(parentSelfType, traitSelfType);

  // Update functionType, signature, and block argument types.
  func.setFuncTypeGenerator(replacer.replace(func.getFuncTypeGenerator()));
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
  for (auto &[name, decls] : traitDecl.getDeclsInScope()) {
    if (decls.empty() || !isa<FnOp>(decls.front()))
      continue;
    for (ASTDecl *decl : decls) {
      auto func = cast<FnOp>(*decl);
      if (failed(resolveBody(*decl, decl->getLoc())))
        return failure();

      existingFns.insert({name, func.getSymNameAttr()});
    }
  }

  // Get our Self type, which will be a reference to the T parameter on this
  // trait.
  ASTType traitSelfType = traitDecl.getTypeDeclSelf();

  // Now just pull in the functions in the bodies of all parents.
  Block &body = *traitOp.getBody();
  for (SymbolRefAttr parent : traitOp.getCanonicalTrait().getSymbols()) {
    ASTDecl &parentDecl = getDeclForTypeSymbol(parent);
    if (failed(resolveBody(parentDecl, traitDecl.getLoc())))
      continue;

    ASTType parentSelfType = parentDecl.getTypeDeclSelf();

    // Inherit function members, which we can override without worry because
    // they are all just declarations.
    for (auto &[name, decls] : parentDecl.getDeclsInScope()) {
      if (decls.empty() || !isa<FnOp>(decls.front()))
        continue;
      for (ASTDecl *decl : decls) {
        if (failed(resolveBody(*decl, traitDecl.getLoc())))
          continue;
        auto func = cast<FnOp>(decl);
        if (func.getInheritedFrom())
          continue;
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
        func.setInheritedFromAttr(parent);
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

//===----------------------------------------------------------------------===//
// Trait Composition Decl implementation
//===----------------------------------------------------------------------===//

ParseResult DeclResolver::resolveSignature(TraitType traitType,
                                           ASTDecl &traitDecl) {
  // There is no signature to resolve for a trait composition.
  return success();
}

ParseResult DeclResolver::resolveBody(TraitType traitType, ASTDecl &traitDecl) {
  // Synthetic Trait Composition ASTDecl (STCASTD):
  // A trait composition decl is modeled as an "anonymous child trait" that
  // inherits from each trait in the composition. The differences are that:
  // - There is no physical TraitDeclOp in the IR for the trait composition.
  //   The ASTDecl's irValue is a TraitType (instead of a TraitDeclOp).
  // - Its child decls are "weak links" to the existing child decls of its
  //   parent traits. No new child ASTDecls or child Ops are created during this
  //   body resolution. As a result, the child methods' self parameter reference
  //   `_Self` still have the parent trait's type instead of the composition's.

  // Deduplicate member aliases if they have identical types. Otherwise, keep
  // all mergeable types in the list. They will each be checked during
  // conformance checking.
  DenseMap<StringAttr, Type> existingAliases;
  // Functions are deduplicated by filtering out all inherited functions.

  for (SymbolRefAttr symbol : traitType.getSymbols()) {
    ASTDecl &parentDecl = getDeclForTypeSymbol(symbol);
    if (failed(resolveBody(parentDecl, traitDecl.getLoc())))
      return failure();

    // Inherit members from the parent.
    for (auto &[name, decls] : parentDecl.getDeclsInScope()) {
      for (ASTDecl *decl : decls) {
        if (failed(resolveBody(*decl, traitDecl.getLoc())))
          return failure();

        if (auto fn = dyn_cast<FnOp>(decl)) {
          if (fn.getInheritedFrom())
            continue;
        } else if (auto alias = dyn_cast<AliasDeclOp>(decl)) {
          // Check if the type is mergeable with the existing alias type.
          if (auto it = existingAliases.find(name);
              it != existingAliases.end()) {
            Type existingType = it->second;
            Type newType = alias.getType();
            if (existingType == newType)
              continue;

            TraitType existingTrait = dyn_cast<TraitType>(existingType);
            TraitType newTrait = dyn_cast<TraitType>(newType);
            if (!existingTrait || !newTrait)
              return emitError(
                         traitDecl.getLoc(),
                         "trait composition has conflicting alias types for '")
                     << alias.getDeclName().getValue() << "'";
            // No need to update existingAliases since we don't care about the
            // specific trait type.
          } else {
            existingAliases[name] = alias.getType();
          }
        } else {
          // If the decl is not a function or alias, it is an error.
          return emitError(traitDecl.getLoc(),
                           "unexpected decl in trait composition")
                     .attachNote(decl->getLoc())
                 << " declared here";
        }

        attachDeclToTraitCompositionDecl(&traitDecl, decl, name);
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// WitnessTable Decl implementation
//===----------------------------------------------------------------------===//

ParseResult DeclResolver::resolveBody(ConformanceOp op, ASTDecl &decl) {
  // Verify conformance explicitly.
  std::optional<InflightDiag> diag;
  WitnessTable witnesses;
  if (failed(verifyConformance(*decl.getParentDecl(), op.getTraitRefAttr(),
                               diag, witnesses)))
    return failure();
  ImplicitLocOpBuilder b =
      ImplicitLocOpBuilder::atBlockEnd(op.getLoc(), &op.getBody().front());
  for (auto &[name, value] : witnesses)
    b.create<WitnessOp>(name, value);
  return success();
}
