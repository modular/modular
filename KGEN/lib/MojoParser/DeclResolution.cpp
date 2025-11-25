//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/DeclResolver.h"

#include "ClosureEmitter.h"
#include "DLValues.h"
#include "ExprNodes.h"
#include "IREmitter.h"
#include "KGEN/MOGGPreElab/MOGGPreElabHelpers.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "MojoUtils.h"
#include "ParserBase.h"
#include "ParserEvaluationContext.h"
#include "Signatures.h"
#include "StructEmitter.h"
#include "Traits.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MOGGPreElab/MOGGPreElabDecorators.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/ToolCommon/CompilationOptions.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/Filesystem/Paths.h"

#include "KGEN/LITDialect/LITUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

/// If the given ASTDecl represents an extension, return the ASTDecl for its
/// target struct. If the given ASTDecl represents a struct, return the struct
/// itself. Returns nullptr if this is neither a struct nor an extension.
static ASTDecl *getStructOrTargetStruct(ASTDecl &decl,
                                        DeclResolver &declResolver) {
  if (auto extensionOp =
          dyn_cast_or_null<ExtensionDeclOp>(decl.getIfOperation())) {
    auto targetStructRefMaybe = extensionOp.getTargetStruct();
    if (targetStructRefMaybe)
      return &declResolver.getDeclForTypeSymbol(*targetStructRefMaybe);
  } else if (isa_and_nonnull<StructDeclOp>(decl.getIfOperation())) {
    return &decl;
  }
  return nullptr;
}

/// Parse an expression and immediately resolve it to a type.  This returns
/// failure on parse error.
static ParseResult parseType(ParserBase &p, ASTType &result, ASTDecl &declScope,
                             std::optional<size_t> stmtIndent) {
  ExprNode *expr = nullptr;
  if (p.parseExpression(expr, stmtIndent))
    return failure();

  IREmitter emitter(declScope, EC_Type);
  result = emitter.emitExprType(expr);
  if (!result)
    return failure();

  return success();
}

static LogicalResult resolveDefaultedOpFromTrait(DeclResolver &resolver,
                                                 Operation *defaultedOp,
                                                 ASTDecl *structDecl) {
  auto traitFnDecl = defaultedOp->getParentOfType<TraitDeclOp>();

  auto traitSymbolRef = getFullyResolvedSymbolRef(traitFnDecl);
  auto conformanceSymName = getFlattenedSymbolName(traitSymbolRef);
  auto conformanceDecl = structDecl->lookupInCurrentScope(conformanceSymName);

  return resolver.resolveBody(*conformanceDecl.front(),
                              conformanceDecl.front()->getLoc());
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
  LogicalResult handleDeprecated(ExprNode *expr, ASTDecl &decl);

  /// Process signature decorators on the declaration using the provided
  /// functor. The functor should return success if the decorator was processed
  /// as a signature decorator. Any leftover decorators are emitted and deferred
  /// as body decorators.
  void applySignatureDecorators(
      ArrayRef<std::pair<ExprNode *, LexerCursor>> decoratorExprs,
      function_ref<LogicalResult(ExprNode *)> process = [](ExprNode *) {
        return failure();
      });

  /// Process body decorators on the declaration using the provided functor.
  /// The functor should return success if the decorator was processed as a
  /// body decorator. Any leftover decorators are emitted and set on the
  /// operation.
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

LogicalResult Decorators::handleDeprecated(ExprNode *expr, ASTDecl &decl) {
  // Detect expression `deprecated` and complain that a warning message should
  // be explicitly specified.
  if (auto declRef = dyn_cast<DeclRefNode>(expr);
      declRef && declRef->spelling == "deprecated") {
    shared.emitError(expr->getLoc(), "@deprecated requires a warning message "
                                     "or a replacement symbol (with 'use')")
        << FixIt::insertAfterToken(expr->getRange().getEnd(),
                                   "(\"insert deprecation message here\")",
                                   shared.diags);
    return success();
  }

  auto callNode = dyn_cast<CallNode>(expr);
  if (!callNode)
    return failure();
  auto declRef = dyn_cast<DeclRefNode>(callNode->callee);
  if (!declRef || declRef->spelling != "deprecated")
    return failure();
  if (callNode->operands.size() != 1) {
    shared.emitError(expr->getLoc(),
                     "@deprecated accepts either a warning message or a "
                     "replacement symbol (with 'use')");
    return success();
  }

  auto &arg = callNode->operands.front();
  // Handle a positional string, or a keyword argument reason=
  if (arg.isPositional() || (arg.isKeyword() && arg.name == "reason")) {
    auto strExpr = dyn_cast<StringLiteralNode>(arg.expr);
    if (!strExpr)
      return failure();

    cast<ASTDeclInterface>(decl.getIfOperation())
        .setDeprecationWarningAttr(
            StringAttr::get(getContext(), strExpr->getValue()));

    return success();
  }
  // Handle a use= argument
  else if (arg.isKeyword() && arg.name == "use") {
    auto target = dyn_cast<DeclRefNode>(arg.expr);
    if (!target) {
      shared.emitError(arg.expr->getLoc(), "'use' must reference a symbol");
      return failure();
    }

    LookupResult lookup = shared.lookupAndResolveDecl(
        target->spelling, target->getLoc(), *decl.getParentDecl(),
        /*searchParentScopes=*/true);
    if (lookup.isErroneous())
      return failure();

    ArrayRef<ASTDecl *> decls = lookup.getIfSuccess();
    if (decls.empty()) {
      shared.emitError(target->getLoc(), "cannot reference unknown value '")
          << target->spelling << "'";
      return failure();
    }

    std::string sourceName;
    if (auto sym = dyn_cast<mlir::SymbolOpInterface>(decl.getIfOperation())) {
      sourceName = sym.getName();
    } else if (auto fn = dyn_cast<FnOp>(decl.getIfOperation())) {
      sourceName = fn.getSourceName() ? fn.getSourceName()->str()
                                      : "<anonymous function>";
    } else if (auto alias = dyn_cast<AliasDeclOp>(decl.getIfOperation())) {
      sourceName = demangleParameterName(alias.getParamDecl().getName());
    } else {
      assert(false && "unhandled case");
      sourceName = "<unhandled case>";
    }

    cast<ASTDeclInterface>(decl.getIfOperation())
        .setDeprecationWarningAttr(StringAttr::get(
            getContext(),
            llvm::formatv("'{0}' is deprecated, use '{1}' instead", sourceName,
                          target->spelling)));

    return success();
  } else {
    emitError(expr->getLoc(), "deprecated must specify either a message or a "
                              "symbol (with the 'use' argument)");
  }

  return failure();
}

void Decorators::applySignatureDecorators(
    ArrayRef<std::pair<ExprNode *, LexerCursor>> decoratorExprs,
    function_ref<LogicalResult(ExprNode *)> process) {
  // Process decorators in the order they are seen. Collect body decorators to
  // be deferred.
  SmallVector<ExprNode *> bodyDecorators;
  for (auto &[decorator, _] : decoratorExprs) {
    if (succeeded(handleDeprecated(decorator, decl)) ||
        succeeded(process(decorator)))
      continue;
    bodyDecorators.push_back(decorator);
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

  SmallVector<ExprNode *> exprDecorators;
  for (auto decorator : decl.getBodyDecorators())
    if (failed(process(decorator)))
      exprDecorators.push_back(decorator);

  // Emit the expressions and persist the resulting PValue into the IR.
  // TODO: Emit an attempt to call the decorator value.
  SmallVector<TypedAttr> decoPValues;
  decoPValues.reserve(exprDecorators.size());
  IREmitter emitter(decl, EC_Decorator);
  for (auto *decorator : exprDecorators) {
    if (PValue decoVal = emitter.emitExprPValue(decorator, EC_Decorator)) {
      if (failed(validateCompilerDecorator(decoVal))) {
        emitError(decorator->getLoc(), "unsupported compiler decorator")
            << decorator->getRange();
      }
      decoPValues.push_back(decoVal);
    }
  }

  cast<ASTDeclInterface>(decl.getIfOperation())
      .setDecoratorsAttr(DecoratorsAttr::get(getContext(), decoPValues));
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
    if (!isa<FnOp>(decl.getIfOperation()))
      shared.emitError(loc, "exported 'main' must be a function");
    return;
  }
  if (unmangledName == kMainSymbolName) {
    shared.emitError(loc, "'main' can only be exported as 'main'");
    return;
  }

  llvm::TypeSwitch<Operation *, void>(decl.getIfOperation())
      .Case([aliasName](FnOp op) { op.setLinkageName(aliasName); });
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
        funcOp(cast_or_null<FnOp>(decl.getIfOperation())), baseName(baseName),
        tcSignature(tcSignature) {}

  /// Apply a function signature decorator.
  LogicalResult applyOne(ExprNode *decorator);
  /// Finalize application of all signature decorators.
  void finalize();

  static LogicalResult checkAlwaysInlineBuiltin(FnOp funcBody,
                                                SharedState &shared);

private:
  void applyImplicitDecorator(SMLoc decoratorLoc, const CallNode *callNode);
  void applyCopyOrMoveCapture(SMLoc decoratorLoc, const CallNode *callNode,
                              bool isMove, StringRef decorator);
  void applyExtern(SMLoc decoratorLoc, const CallNode *node);
  void applyAlwaysInline(const CallNode *node);
  void applyLLVMMetadata(SMLoc decoratorLoc, const CallNode *node);

  void applyArgumentless(StringRef spelling, const CallNode *callNode,
                         function_ref<void()> applyImpl);

  ArrayAttr getLLVMMetadataArray(ArrayRef<Operand> operands);

  /// Register an LLVM arg metadata in the internal list to avoid churning mlir
  /// attributes as these arg metadata decorators are parsed. Must call finalize
  /// to actually apply metadata onto the function.
  void applyLLVMArgMetadata(SMLoc decoratorLoc, const CallNode *node);

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
  ParserParameterEvaluator evaluator(shared);
  SmallVector<TypedAttr> params;
  for (auto paramDecl : fnOp.collectAllParams(/*implOrigins*/ false)) {
    params.push_back(
        UnknownAttr::get(evaluator.getReboundType(paramDecl.getType())));
    evaluator.setDeclBinding(paramDecl, params.back());
  }
  auto paramValueArray = ParameterExprArrayAttr::get(fnOp.getContext(), params);
  operands.push_back(
      fnOp.getBoundReference(shared.getEvaluationContext(), paramValueArray));

  for (auto arg : fnOp.getBody()->getArguments())
    operands.push_back(
        UnknownAttr::get(evaluator.getReboundType(arg.getType())));

  if (shared.foldInlineBuiltinFunction(operands, fnOp.getLoc(), true))
    return success();
  return failure();
}

LogicalResult FnSigDecorators::applyOne(ExprNode *decorator) {
  const DeclRefNode *declRef = dyn_cast<DeclRefNode>(decorator);
  const CallNode *callNode = nullptr;
  if (!declRef) {
    callNode = dyn_cast<CallNode>(decorator);
    if (callNode)
      declRef = dyn_cast<DeclRefNode>(callNode->callee);

    if (!callNode || !declRef) {
      emitError(decorator->getLoc(), "invalid expression in decorator");
      decl.setErroneous();
      return failure();
    }
  }

  StringRef spelling = declRef->spelling;
  if (spelling == "export") {
    // TODO: improve this
    if (callNode)
      applyExport(decorator->getLoc(), decl, baseName, *callNode, funcOp);
    else
      applyExport(decorator->getLoc(), decl, baseName, baseName, funcOp);
  } else if (spelling == "staticmethod") {
    applyArgumentless(spelling, callNode, [&]() {
      if (!decl.tryGetMethodParentDecl()) {
        emitError(declRef->getLoc(),
                  "only methods on structs may be declared static");
      }
    });
    // We set the staticmethod flag even on errors, since the user intention is
    // clear, and this will suppress errors about missing self arguments.
    funcOp.setIsStatic(true);
  } else if (spelling == "always_inline") {
    applyAlwaysInline(callNode);
  } else if (spelling == "no_inline") {
    applyArgumentless(spelling, callNode,
                      [&]() { funcOp.setInlineLevel(InlineLevel::Never); });
  } else if (spelling == "parameter") {
    applyArgumentless(spelling, callNode,
                      [&]() { tcSignature.argList.effects.setCapturing(); });
  } else if (spelling == "__unsafe_disable_nested_origin_exclusivity") {
    applyArgumentless(spelling, callNode, [&]() {
      tcSignature.isNestedOriginExclusivityCheckingDisabled = true;
    });
  } else if (spelling == "implicit") {
    applyImplicitDecorator(decorator->getLoc(), callNode);
  } else if (spelling == "extern") {
    applyExtern(decorator->getLoc(), callNode);
  } else if (spelling == "__move_capture") {
    applyCopyOrMoveCapture(decorator->getLoc(), callNode, /*isMove=*/true,
                           spelling);
  } else if (spelling == "__copy_capture") {
    applyCopyOrMoveCapture(decorator->getLoc(), callNode, /*isMove=*/false,
                           spelling);
  } else if (spelling == "__llvm_metadata") {
    applyLLVMMetadata(decorator->getLoc(), callNode);
  } else if (spelling == "__llvm_arg_metadata") {
    applyLLVMArgMetadata(decorator->getLoc(), callNode);
  } else {
    return failure();
  }

  return success();
}

void FnSigDecorators::applyImplicitDecorator(SMLoc decoratorLoc,
                                             const CallNode *callNode) {
  size_t numOperands = callNode ? callNode->operands.size() : 0;
  if (numOperands > 1) {
    emitError(callNode->getLoc())
        << "'@implicit' may not have more than 1 operand, got " << numOperands;
    return;
  }

  ImplicitConversionKind conversionKind = ImplicitConversionKind::Implicit;
  if (numOperands == 1) {
    const Operand &operand = callNode->operands[0];

    auto *boolExpr = dyn_cast<BoolLiteralNode>(operand.expr);
    if (!boolExpr || !operand.isKeyword() || operand.name != "deprecated") {
      emitError(callNode->getLoc())
          << "'@implicit' may only have a keyword argument 'deprecated' with "
             "literal boolean value";
      return;
    }
    if (boolExpr->value)
      conversionKind = ImplicitConversionKind::Deprecated;
  }

  if (SpecialFunctionInfo::get(baseName).kind != SpecialFunctionKind::kInit) {
    emitError(decoratorLoc)
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
    if (args.size() > 1 && (lastArg.initExpr || // arg has a default.
                                                // vararg lists can be empty
                            lastArg.variadicKind != VariadicKind::None))
      args = args.drop_back();
    else
      break;
  }

  // We must have a positional argument to take the new value.
  if (args.size() != 1 ||
      (args[0].kwArgHandling != KWArgHandling::kPositionalOnly &&
       args[0].kwArgHandling != KWArgHandling::kPositionalOrKeyword)) {
    emitError(decl.getLoc()) << "'@implicit' initializers must accept a single "
                                "positional argument value";
    return;
  }
  funcOp.setImplicitConversion(conversionKind);
}

void FnSigDecorators::applyCopyOrMoveCapture(SMLoc decoratorLoc,
                                             const CallNode *callNode,
                                             bool isMove,
                                             StringRef decoratorSpelling) {
  if (!callNode || callNode->operands.empty()) {
    emitError(decoratorLoc, "'@")
        << decoratorSpelling << "' must have arguments";
    return;
  }

  const CallNode &node = *callNode;
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

    IREmitter emitter(*decl.getParentDecl(), OpBuilder(funcOp));
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
                             Capture(captureRVal,
                                     CaptureConvention::kConventionCopy,
                                     declRef->spelling));
  }
}

void FnSigDecorators::applyExtern(SMLoc decoratorLoc,
                                  const CallNode *callNode) {
  size_t numOperands = callNode ? callNode->operands.size() : 0;
  if (numOperands != 1) {
    emitError(decoratorLoc, "'@extern' requires 1 argument");
    return;
  }

  Operand operand = callNode->operands[0];
  auto strNode = dyn_cast<StringLiteralNode>(operand.expr);
  if (!strNode || !operand.isPositional()) {
    emitError(operand.getLoc(), "'@extern' requires a string literal argument");
    return;
  }
  std::string libName = strNode->getValue();
  funcOp.setLinkageName(libName);

  if (!funcOp.getInputParams().empty()) {
    // TODO: Can this even happen?
    emitError(callNode->getLoc(),
              "'@extern' cannot be applied to a function with parameters");
    return;
  }

  if (decl.getParentDecl() && llvm::isa_and_nonnull<TraitDeclOp, StructDeclOp>(
                                  decl.getParentDecl()->getIfOperation())) {
    emitError(callNode->getLoc(), "'@extern' cannot be applied to a method");
    return;
  }

  funcOp.setExternal(true);
}

void FnSigDecorators::applyAlwaysInline(const CallNode *callNode) {
  size_t numOperands = callNode ? callNode->operands.size() : 0;
  if (numOperands == 0) {
    // `@always_inline` and `@always_inline()` are both allowed.
    funcOp.setInlineLevel(InlineLevel::Always);
    return;
  }

  if (numOperands > 1) {
    emitError(callNode->getLoc())
        << "'@always_inline' may not have more than 1 operand, got "
        << numOperands;
    return;
  }

  const Operand &operand = callNode->operands[0];
  if (operand.isPositionalStringLiteral("nodebug")) {
    funcOp.setInlineLevel(InlineLevel::AlwaysNoDebug);
  } else if (operand.isPositionalStringLiteral("builtin")) {
    funcOp.setInlineLevel(InlineLevel::AlwaysBuiltin);
  } else {
    emitError(callNode->getLoc())
        << "'@always_inline' operand must be \"nodebug\" or \"builtin\"";
  }
}

void FnSigDecorators::applyArgumentless(StringRef spelling,
                                        const CallNode *callNode,
                                        function_ref<void()> applyImpl) {
  if (!callNode)
    return applyImpl();
  emitError(callNode->getLoc()) << "'@" << spelling << "' cannot have arguments"
                                << FixIt::remove(callNode->getRange());
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

    if (isa_and_nonnull<UnresolvedImportOp>(
            nameDecls.back()->getIfOperation())) {
      if (failed(shared.getDeclResolver().resolveBody(*nameDecls.back(),
                                                      funcDecl.getLoc()))) {
        shared.emitError(funcDecl.getLoc(), "cannot resolve comptime value '")
            << name << "' used in '@__llvm_metadata'";
        return {};
      }
    }
    if (auto aliasOp =
            dyn_cast_or_null<AliasDeclOp>(nameDecls.back()->getIfOperation()))
      return aliasOp;

    shared.emitError(funcDecl.getLoc(), "name '")
        << name << "' cannot be used in '@__llvm_metadata'";
    return {};
  } while (!isa_and_nonnull<FileModuleOp>(parent->getIfOperation()));
  return {};
}

ArrayAttr FnSigDecorators::getLLVMMetadataArray(ArrayRef<Operand> operands) {
  IREmitter emitter(sigDecl, EC_Decorator);
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

void FnSigDecorators::applyLLVMMetadata(SMLoc decoratorLoc,
                                        const CallNode *node) {
  size_t numOperands = node ? node->operands.size() : 0;
  if (numOperands == 0) {
    emitError(decoratorLoc, "'@__llvm_metadata' requires operands");
    return;
  }

  ArrayAttr metadata = getLLVMMetadataArray(node->operands);
  llvmMetadata.append(metadata.begin(), metadata.end());
}

void FnSigDecorators::applyLLVMArgMetadata(SMLoc decoratorLoc,
                                           const CallNode *node) {
  size_t numOperands = node ? node->operands.size() : 0;
  if (numOperands == 0) {
    emitError(decoratorLoc, "'@__llvm_arg_metadata' requires operands");
    return;
  }

  Operand targetArg = node->operands[0];
  auto declRef = dyn_cast<DeclRefNode>(targetArg.expr);
  // We expect the first operand to be "positional", i.e. it should just be a
  // standalone name.
  if (targetArg.passKind != Operand::PassKind::kPositional || !declRef) {
    emitError(
        targetArg.getLoc(),
        "First argument of '@__llvm_arg_metadata' must be an argument name");
    return;
  }

  // Ignore empty metadata list.
  if (numOperands == 1)
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
    emitError(
        targetArg.getLoc(),
        "Function decorated by '@__llvm_arg_metadata' has no argument named '")
        << declRef->spelling << "'";
    return;
  }

  // First time setting arg metadata, initialize with array of empty attributes.
  if (llvmArgMetadata.empty())
    llvmArgMetadata.insert(llvmArgMetadata.begin(),
                           tcSignature.argList.parsedArgs.size(),
                           ArrayAttr::get(getContext(), {}));

  llvmArgMetadata[argIdx] = getLLVMMetadataArray(node->operands.drop_front());
}

void FnSigDecorators::finalize() {
  if (funcOp.isExternal()) {
    if (funcOp.getInlineLevel() != InlineLevel::Never &&
        funcOp.getInlineLevel() != InlineLevel::Automatic) {
      emitError(funcOp.getLoc(), "extern functions cannot be inlined");
      return;
    }
    funcOp.setInlineLevel(InlineLevel::Never);
  }

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
      auto variadic = sugarCast<VariadicType>(type);
      type = variadic.getElementType();
      conv = sig.getPosVarArgConvention(idx);
    } else if (sig.isKwVarArg(idx)) {
      // Don't need to unpack anything. We treat the whole dictionary as the
      // value type.
    } else if (sig.isPack(idx)) {
      // For variadic packs, we don't have a type instance but we have the
      // metatype.
      Type metatype = ASTType(type)
                          .getReferenceElementType()
                          .getVariadicPackInfo(shared)
                          .getVariadicElementType();
      type = ParamType::get(UnknownAttr::get(metatype));
      conv = ArgConvention::ReadReg;
    }
    type = RefType::stripRefConvention(type, conv);
    argTypes.push_back(type);
  }

  bool allVanillaKernelArgs = llvm::all_of(argTypes, [](ASTType astType) {
    if (auto structTy = sugarDynCast<LIT::StructType>(astType.mlirType)) {
      return MOGGPreElab::isDPSTensor(structTy) ||
             MOGGPreElab::isMojoDeviceContextPtr(structTy);
    }
    return false;
  });

  // We don't need to attach the conformance attrs if we have a kernel working
  // purely with tensors
  if (allVanillaKernelArgs && resultType.isNoneType())
    return;

  IREmitter emitter(decl, EC_Type);
  auto generateValueWitnesses = [&](ASTType type,
                                    Location loc) -> DictionaryAttr {
    SMLoc smloc = shared.diags.convertLocToSMLoc(loc);
    NamedAttrList methodsDict;
    // These are the trait methods that MOGG is interested in.
    for (auto [traitName, entryName] :
         SmallVector<std::pair<StringRef, StringRef>>{
             {"AnyType", "__del__"}, {"Movable", "__moveinit__"}}) {
      auto traitDecl = shared.lookupBuiltinTrait(traitName, &decl, smloc);
      if (!traitDecl)
        continue;
      auto traitDeclOp = cast_or_null<TraitDeclOp>(traitDecl->getIfOperation());
      if (!traitDeclOp)
        continue;
      TraitType trait = traitDeclOp.bindReference();
      FailureOr<TypedAttr> entry = getUniqueWitnessForTypeIfConforms(
          shared, type, trait, entryName, smloc);
      // If failed, an error will have been emitted. If empty, the type does not
      // conform to the trait. In either case, move on to the next trait method.
      if (failed(entry) || !*entry)
        continue;
      methodsDict.set(entryName, *entry);
    }
    return DictionaryAttr::get(shared.getContext(), methodsDict);
  };

  SmallVector<Attribute> argConformances;
  Attribute resConformances = generateValueWitnesses(resultType, func.getLoc());
  for (auto [idx, argType] : llvm::enumerate(argTypes)) {
    argConformances.push_back(
        generateValueWitnesses(argType, func.getArgument(idx).getLoc()));
  }

  NamedAttrList attrs = func->getAttrDictionary();
  attrs.set(MOGGPreElab::MOGG_ARGUMENT_VALUE_WITNESSES,
            ArrayAttr::get(shared.getContext(), argConformances));
  attrs.set(MOGGPreElab::MOGG_RESULT_VALUE_WITNESSES, resConformances);
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

  auto func = cast_or_null<FnOp>(decl.getIfOperation());
  if (!(isa<FileModuleOp>(func->getParentOp()) || func.getIsStatic())) {
    shared.emitError(decl.getLoc(), "@")
        << spelling << " is only supported on top-level or static functions";
    return;
  }

  processFunctionConformances(func, shared, decl);
}

/// Given the lexical context of a function, return true if the default bit
/// for the function is capturing.
static bool
isCapturingByDefault(SharedState &shared, FnOp funcOp, TraitType canonicalTrait,
                     std::optional<ArrayRef<ParamDeclAttr>> parentDecls,
                     ArrayRef<ParamDeclAttr> paramDecls) {
  // Any function that contains a capturing closure as a parameter is itself
  // capturing, include parent struct parameters.
  mlir::AttrTypeWalker walker;
  walker.addWalk([](FuncType sig) {
    if (sig.isCapturing())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  // Temporary solution to supporting capturing parametric closures inside
  // unified closures: propagate capturing effect with unified effect.
  walker.addWalk([&](SymbolRefAttr symbol) {
    auto traitDecl = shared.declResolver->getDeclForTypeSymbolIfExists(symbol);
    if (!traitDecl)
      return WalkResult::advance();
    TraitDeclOp traitDeclOp =
        dyn_cast_if_present<TraitDeclOp>(traitDecl->getIfOperation());
    if (traitDeclOp && traitDeclOp.getDefinesClosure())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  bool isInterrupted = false;
  if (canonicalTrait)
    isInterrupted = walker.walk(canonicalTrait).wasInterrupted();
  return isInterrupted ||
         llvm::any_of(llvm::concat<const ParamDeclAttr>(
                          paramDecls, parentDecls
                                          ? parentDecls.value()
                                          : SmallVector<ParamDeclAttr>()),
                      [&](ParamDeclAttr decl) {
                        return walker.walk(decl).wasInterrupted();
                      });
}

std::pair<SmallVector<ParamDeclRefAttr>, FnTypeGeneratorType>
DeclResolver::createSelfContainedSignature(FnTypeGeneratorType original) {
  // Collect the subset of referenced parameters. Use a set vector to keep the
  // order deterministic.
  llvm::SmallSetVector<ParamDeclRefAttr, 4> capturedRefs;
  getCanonicalType(original).walk(
      [&](ParamDeclRefAttr ref) { capturedRefs.insert(ref); });

  SmallVector<ParamDeclRefAttr> captured = capturedRefs.takeVector();
  // Unbind the N capture parameters, creating a FuncType with N new input
  // parameters prepended.
  // TODO: what if we capture a variadic?
  SmallVector<VariadicKind> variadicKinds(captured.size(), VariadicKind::None);
  auto unbound = FnTypeGeneratorType::prependParams(
      original,
      llvm::map_to_vector(
          captured,
          [](ParamDeclRefAttr ref) { return ParamDeclAttr::get(ref); }),
      variadicKinds);
  return {std::move(captured), unbound};
}

static bool allCopyable(ArrayRef<Capture> captures, SharedState &shared,
                        SMLoc loc) {
  for (const Capture &capture : captures) {
    switch (capture.getCaptureConvention()) {
    case CaptureConvention::kConventionCopy:
    case CaptureConvention::kConventionTrivialCopy:
    case CaptureConvention::kConventionRead:
    case CaptureConvention::kConventionMut:
      continue;
    default:
      if (!capture.getValue().getRValueType().isCopyable(loc, shared, true))
        return false;
    }
  }
  return true;
}

static TypeConvention getTypeConvention(ArrayRef<Capture> captures,
                                        SharedState &shared, SMLoc loc) {
  TypeConvention convention = TypeConvention::RegisterPassableTrivial;
  for (const Capture &capture : captures) {
    switch (convention) {
    case TypeConvention::RegisterPassableTrivial: {
      if (capture.getValue().getRValueType().isTrivial(loc, shared))
        break;
      if (capture.getValue().getRValueType().isRegisterPassable(loc, shared)) {
        convention = TypeConvention::RegisterPassable;
        break;
      }
      return TypeConvention::MemoryOnly;
    }
    case TypeConvention::RegisterPassable: {
      if (capture.getValue().getRValueType().isRegisterPassable(loc, shared))
        break;
      return TypeConvention::MemoryOnly;
    }
    default:
      break;
    }
  }
  return convention;
}

static MLValue emitUnifiedClosureInstance(ArrayRef<Capture> captures,
                                          ASTDecl &nestedFnDecl,
                                          SharedState &shared) {
  FnOp nestedFn = cast<FnOp>(nestedFnDecl.getIfOperation());
  SMLoc loc = nestedFnDecl.getLoc();
  Location mlirLoc = shared.translateLocation(loc);
  if (shared.diBuilder)
    mlirLoc = shared.diBuilder->createScopedLoc(mlirLoc);

  ASTDecl *moduleDecl = nestedFnDecl.getNearestDeclOfType<FileModuleOp>();
  auto [capturedRefs, wrapperSig] = DeclResolver::createSelfContainedSignature(
      nestedFn.getFuncTypeGenerator());
  ASTDecl *closureTrait = shared.getOrCreateClosureTrait(
      loc, *moduleDecl, wrapperSig, nestedFn.getInlineLevel());
  bool isCopyable = allCopyable(captures, shared, loc);
  TypeConvention convention = getTypeConvention(captures, shared, loc);
  if (!wrapperSig.isRegisterPassable())
    convention = TypeConvention::MemoryOnly;
  ASTDecl *closureWrapper = shared.getOrCreateUnifiedClosureWrapper(
      loc, wrapperSig, moduleDecl, nestedFn.getInlineLevel(), isCopyable,
      convention);

  ClosureEmitter &emitter = shared.getClosureEmitter();
  Value wrapperInstance = emitter.emitClosureOp(
      *moduleDecl, nestedFnDecl, captures,
      cast<StructDeclOp>(closureWrapper->getIfOperation()),
      cast<TraitDeclOp>(closureTrait->getIfOperation()), mlirLoc, isCopyable);

  nestedFnDecl.getIfOperation()->erase();
  nestedFnDecl.setIRValue(nullptr);
  shared.deleteDecl(nestedFnDecl);
  return MLValue(wrapperInstance);
}

static MLValue emitClosureInstance(ArrayRef<Capture> captures,
                                   ArrayRef<ParamDeclRefAttr> paramCaptures,
                                   ASTDecl &nestedFnDecl, SharedState &shared) {
  FnOp nestedFn = cast<FnOp>(nestedFnDecl.getIfOperation());
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
  ClosureEmitter &emitter = shared.getClosureEmitter();
  StructDeclOp closureImpl =
      emitter.replaceNestedFunctionWithClosureImplStructDecl(
          *moduleDecl, captures, paramCaptures, nestedFnDecl, wrapperSig);
  if (!closureImpl)
    return {};

  emitter.createWrapperInitWithImpl(*moduleDecl, closureWrapper, closureImpl,
                                    loc);

  builder.restoreInsertionPoint(insertPoint);

  IREmitter exprEmitter(*nestedFnDecl.getParentDecl(), builder);
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

/// Make a copy or cast mutability if needed in the parent scope so that the
/// semantics of the closure body are upheld. For example, consider the
/// following:
///
/// fn toy(read byCopy:String, prefix:String):
///   fn myclosure(prefix: String) unified {var byCopy} -> String:
///      byCopy += "v2" // LINE A
///      return prefix
///   takeIt(myclosure, prefix)
///
/// Note the mutation on line A. If we do not make a copy in the parser, the
/// parser will complain that you cannot bind a read only value to the mutable
/// reference argument of __iadd__ because it will map the byCopy value on line
/// A to the byCopy value passed into the function with argument convention read
/// . The parser is wrong about this because the user expressed that he wants to
/// make a mutable copy of the immutable reference and mutate the copy. We
/// fulfill the user's request but not until after checklifetimes, where we emit
/// the copy and replace usages of the original value with that copy. I will
/// introduce an op to avoid the extra copy (MOCO 2291). Until then, we just
/// emit an extra copy/move.
static LogicalResult createCaptureValues(ParserBase &p, ASTDecl &sigDecl,
                                         ParsedCaptureList &captureSignature,
                                         ASTDecl &decl) {
  FnOp funcOp = cast<FnOp>(decl.getIfOperation());
  IREmitter emitter(*decl.getParentDecl(), OpBuilder(funcOp));
  bool didFail = false;
  for (auto [name, capture, location] : captureSignature.parsedCaptures) {
    if (!ClosureEmitter::addCaptureValue(decl, location, name, capture, emitter,
                                         &sigDecl))
      didFail = true;
  }
  return didFail ? failure() : success();
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
  if (p.parseIdentifier(baseName, "expected function name", &identifierLoc,
                        /*allowKeyword=*/true))
    return failure();

  ASTDecl *parentDecl = decl.getParentDecl();
  // The function signature is a self-contained scope where the input and result
  // parameters of the function are visible by all types.  We must use a
  // temporary declaration here (with an empty name) because we don't want
  // references to the function itself to resolve to a fully-resolved decl, but
  // we need a fully-resolved decl for incremental lookups within the scope to
  // work out.
  ASTDecl &sigDecl = addFullyResolvedDecl(funcOp.getOperation(), StringAttr(),
                                          decl.getLoc(), parentDecl);

  // If this is a struct method, inherit parameter defined in the struct so that
  // we reject
  //
  // struct S[param: Int]:
  //     fn method[param: Int](self): pass
  if (auto structOp =
          dyn_cast_or_null<StructDeclOp>(parentDecl->getIfOperation())) {
    for (auto pog : structOp.getSignature().getParamListAttrs().getPogs()) {
      StringRef paramName = pog.getName().getValue();
      ArrayRef<ASTDecl *> paramDecls =
          parentDecl->lookupInCurrentScope(paramName);
      // Must be autoparams, they aren't explicitly declared by the user so
      // can't be looked up by their names, nor should they lead to name
      // conflict.
      if (paramDecls.empty())
        continue;

      // If we found it, it must be a parameter declaration.
      assert(paramDecls.size() == 1 &&
             "expected exactly one parameter declaration");
      ASTDecl *paramDecl = paramDecls.front();
      addFullyResolvedDecl(paramDecl->irValue, paramName, paramDecl->getLoc(),
                           &sigDecl);
    }
  }

  // Parse declared parameters and add them to the current scope.
  ParsedParamList parsedParamList;

  // Add the parameters to the symbol table, and resolve their types.  We
  // add all of these after generic signature parsing so types used in the
  // signature list resolve to enclosing scopes, and we add them before the
  // value signature list so the types and parameters can resolve to the bound
  // values.
  if (parsedParamList.parseParametersIfPresent(p, ArgListKind::kParamList))
    return failure();

  if (!parsedParamList.params.empty() && baseName == "__call__" &&
      dyn_cast_or_null<StructDeclOp>(parentDecl->getIfOperation())) {
    auto getItems = parentDecl->lookupInCurrentScope("__getitem__");
    auto setItems = parentDecl->lookupInCurrentScope("__setitem__");
    auto getAttrs = parentDecl->lookupInCurrentScope("__getattr__");
    if (!getItems.empty() || !setItems.empty() || !getAttrs.empty()) {
      auto diag = p.emitWarning(
          funcOp->getLoc(),
          llvm::formatv(
              "parametric '__call__' method cannot be "
              "called directly because '{}' defines '__getitem__', "
              "'__setitem__', or '__getattr__'; consider using a different "
              "name for this method",
              parentDecl->getNameIfOperation()));

      for (const auto &decl : getItems)
        diag.attachNote(decl->getLoc()) << "__getitem__ defined here";

      for (const auto &decl : setItems)
        diag.attachNote(decl->getLoc()) << "__setitem__ defined here";

      for (const auto &decl : getAttrs)
        diag.attachNote(decl->getLoc()) << "__getattr__ defined here";
    }
  }

  ParsedArgumentList fnSignature;
  // Set up the known effects.
  if (isAsync) {
    fnSignature.effects.setAsync(true);

    if (funcOp.isDefaultedTraitFn()) {
      // TODO(MOCO-2287): Support async defaulted trait methods
      shared.emitError(funcOp.getLoc())
          << "async defaulted trait methods are not supported yet";
      return failure();
    }
  }
  if (isDef)
    fnSignature.effects.setThrows();

  // Parse the argument list next if present.
  bool isMoveInitOrDel = // TODO(25.7): Remove this, it's a hack for migration.
      baseName.strref() == "__moveinit__" || baseName.strref() == "__del__";
  if (fnSignature.parseArgumentListAndEffects(p, ArgListKind::kArgList,
                                              isMoveInitOrDel))
    return failure();

  // TODO: effects parsing must be moved after captures parsing.
  // A capture list must be specified for every unified closure.
  ParsedCaptureList captureSignature;
  if (fnSignature.effects.isUnified() && captureSignature.parseCaptureList(p))
    return failure();

  // Emit copies/casts for captures. Otherwise the incorrect lifetime rules will
  // be applied to the values in the closure.
  if (fnSignature.effects.isUnified()) {
    if (!funcOp->getParentOfType<FnOp>()) {
      p.emitError(funcOp.getLoc(),
                  "unified effect is only applicable on nested functions");
      return failure();
    }
    if (captureSignature.captureAllByConvention.has_value()) {
      shared.setDefaultCaptureForScope(
          decl, *captureSignature.captureAllByConvention);
    } else if (failed(createCaptureValues(p, sigDecl, captureSignature, decl)))
      return failure();
  }

  // Parse the result type if present.
  fnSignature.parseResultIfPresent(p);

  // Parse the constraints if present.
  if (failed(fnSignature.parseConstraintsIfPresent(p)))
    return failure();
  std::optional<TypeCheckedParamList> paramListOrError =
      TypeCheckedParamList::create(parsedParamList, sigDecl);
  if (!paramListOrError.has_value())
    return failure();
  TypeCheckedParamList &paramList = *paramListOrError;

  // Emit the argument and result types.
  SpecialFunctionInfo fnInfo = SpecialFunctionInfo::get(baseName);
  TypeCheckedFnSignature tcSignature(paramList, fnSignature,
                                     /*captureOrigins=*/nullptr, &decl, fnInfo);

  // If any of the arguments had an error or if the result type is a type check
  // error, then we won't allow forming a reference to this function.
  if (sugarIsa<TypeCheckErrorType>(tcSignature.resultType.mlirType) ||
      llvm::any_of(fnSignature.parsedArgs,
                   [](ParsedArgument &arg) { return arg.isErroneous; }))
    decl.setErroneous();

  TraitType traitType;
  std::optional<ArrayRef<ParamDeclAttr>> parentParams;
  ASTDecl *structDecl = getStructOrTargetStruct(*decl.getParentDecl(), *this);
  StructDeclOp structOp = nullptr;
  if (structDecl)
    structOp = dyn_cast_or_null<StructDeclOp>(structDecl->getIfOperation());
  if (structOp) {
    traitType = structOp.getCanonicalTrait();
    parentParams = structOp.getParams();
  } else if (auto traitDecl = dyn_cast_or_null<TraitDeclOp>(
                 decl.getParentDecl()->getIfOperation())) {
    traitType = traitDecl.getCanonicalTrait();
  }
  if (isCapturingByDefault(shared, funcOp, traitType, parentParams,
                           paramList.paramDeclAttrs))
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

  decl.insertKnownAssumptions(tcSignature.fnConstraints);

  /// configure FnOp

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
            shared.declResolver->getMangledName(baseName, *decl.getParentDecl(),
                                                signature));
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

    if (!overloadedKeywordArgName) {
      auto existingResTy =
          ASTType(existingFunc.getFuncTypeGenerator().getUserResultType());
      if (!resTy.isEqualCanon(existingResTy))
        errorMessage = " cannot overload on return type only";
      else
        errorMessage = " with identical signature";
    }

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

  // Because our IR has type sugar, we can capture the same parameter from
  // the sugared and canonical version of the same type.  Remove one of the
  // versions from the captured uses.
  SmallPtrSet<StringAttr, 8> capturedParamNames;
  llvm::erase_if(paramCaptures, [&](ParamDeclRefAttr use) {
    return !capturedParamNames.insert(use.getName()).second;
  });

  // If this is a `@parameter` closure, attach the capture origins.
  if (signature.isCapturing() && !signature.isUnified()) {
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

  if (signature.isUnified()) {
    MLValue instance = emitUnifiedClosureInstance(captures, decl, shared);
    if (!instance)
      return failure();
    decl.setIRValue(instance);
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
                                   ASTDecl &parentDecl, IREmitter &emitter,
                                   SMLoc loc, ArgConvention convention) {

  // Determine if this is VariadicList or VariadicListMem, and get it.
  auto variadicType = sugarCast<VariadicType>(argValue.getType());
  ASTType variadicEltType = variadicType.getElementType();
  auto refType = sugarDynCast<RefType>(variadicEltType);
  ASTType varListType =
      emitter.shared.getBuiltinVariadicListType(parentDecl, loc, (bool)refType);
  if (varListType.isTypeCheckErrorType())
    return {};
  ASTDecl *varListStructDecl = varListType.getDecl(emitter.shared);
  if (!varListStructDecl) {
    emitter.emitError(loc, "malformed VariadicListInMem");
    return {};
  }
  auto varListStruct =
      dyn_cast_if_present<StructDeclOp>(varListStructDecl->getIfOperation());
  if (!varListStruct) {
    emitter.emitError(loc, "malformed VariadicListInMem");
    return {};
  }

  // Bind the "is_owned" parameter, start by filling the parameter list with ?.
  if (refType) {
    assert(varListStruct.getSignature().getParamTypes().size() == 4);
    SmallVector<TypedAttr> typeParams(4);
    ParserParameterEvaluator evaluator(emitter.shared);
    Type boolType = varListStruct.getSignature().getParamTypes()[0];
    Type eltType = varListStruct.getSignature().getParamTypes()[1];
    Type originType = varListStruct.getSignature().getParamTypes()[2];

    // The first parameter is the "elt_is_mutable" parameter.
    // Emit the "is_mutable" parameter
    auto makeBoolAttr = [&](bool value) -> PValue {
      auto boolAttr = BoolAttr::get(emitter.getContext(), value);
      SyntheticNode locMutableExpr(loc);
      return emitter.emitPValue({boolAttr, &locMutableExpr}, EC_Type, boolType);
    };

    auto isMut = makeBoolAttr(convention == ArgConvention::OwnedMem ||
                              convention == ArgConvention::DeinitMem ||
                              convention == ArgConvention::Mut);
    evaluator.appendIndexBinding(isMut);
    typeParams[0] = isMut.get();

    SyntheticNode locElExpr(loc);
    auto eltTypeAttr = emitter.emitPValue(
        {refType.getElementType(), &locElExpr}, EC_Type, eltType);
    if (!eltTypeAttr)
      return {};
    evaluator.appendIndexBinding(eltTypeAttr);
    typeParams[1] = eltTypeAttr.get();

    SyntheticNode locOriginExpr(loc);
    auto reboundOriginType = evaluator.getReboundType(originType);
    auto origin = emitter.emitPValue({refType.getOrigin(), &locOriginExpr},
                                     EC_Type, reboundOriginType);
    evaluator.appendIndexBinding(origin);
    typeParams[2] = origin.get();

    auto isOwned = makeBoolAttr(convention == ArgConvention::OwnedMem ||
                                convention == ArgConvention::DeinitMem);
    evaluator.appendIndexBinding(isOwned);
    typeParams[3] = isOwned.get();

    varListType = varListStruct.bindReference(typeParams);
    assert(varListType && "Failed to bind type params");
  }

  // Emit a VarDeclOp: VariadicListMem needs a origin for its self accesses.
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
    ctorDest.resetForError(emitter);
    return {};
  }
  return varDecl;
}

LogicalResult DeclResolver::resolveSyntheticBody(FnOp fn, ASTDecl &decl) {
  // TODO: Sink this to when the body is actually resolved.
  decl.resolvedness = DeclResolvedness::body;

  StructEmitter gen(*decl.getParentDecl());

  if (fn.getInheritedFrom())
    return gen.populateDefaultedTraitFunction(decl);

  switch (fn.getSpecialFunctionKind()) {
  default:
    // Matching by name is a bit gross, but we don't have general synthesized
    // decls so it should be robust.
    assert(fn.getSymName()->starts_with("copy(") &&
           "unknown synthetic function to synthesize");
    gen.populateExplicitCopy(decl);
    return success();
  case SpecialFunctionKind::kMoveInit:
    (void)gen.populateMoveCopy(decl, /*isMove*/ true);
    return success();
  case SpecialFunctionKind::kCopyInit:
    (void)gen.populateMoveCopy(decl, /*isMove*/ false);
    return success();
  }
}

ParseResult DeclResolver::resolveBody(FnOp funcOp, Lexer &lexer,
                                      ASTDecl &decl) {
  // TODO: Sink this to when the body is actually resolved.
  decl.resolvedness = DeclResolvedness::body;

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

  // If this an extern function, we only allow a "..." as the body. If it's a
  // trait method this must mean it's not defaulted so we can early exit the
  // function here as well.
  if (isa_and_nonnull<TraitDeclOp>(decl.getParentDecl()->getIfOperation()) ||
      funcOp.isExternal()) {
    // Skip any docstring's that might be present.
    ParserBase p(shared, lexer);
    p.parseDocString(decl);

    // If we see an ellipsis, the function member is well formed: don't emit
    // arguments or any other setup logic.
    if (p.consumeIf(Token::dot_dot_dot)) {
      body.front().erase(); // Remove the lit.endfn op to replace it.
      auto builder = OpBuilder::atBlockEnd(&body);
      UnreachableOp::create(builder, funcOp.getLoc());
      return success();
    }

    // Otherwise, must be a trait method with default implementation.

    // If a defaulted trait method should return a value but 'pass' is used,
    // emit an error. The user likely meant to use '...', so suggest that.
    if (auto tok = p.getToken();
        funcOp.isDefaultedTraitFn() && tok.is(Token::kw_pass)) {
      if (!ASTType(funcOp.getUserResultType()).isNoneType()) {
        InflightDiag diag = shared.emitError(
            tok.getLoc(), "trait method has results but default implementation "
                          "returns no value; did you mean '...'?");
        diag.attachNote(funcOp.getLoc())
            << "in '" << funcOp.getDeclName().getValue() << "', declared here";
        diag.addFixIt(FixIt::replaceToken(tok.getLoc(), "..."));
        return failure();
      }
    }
  }

  // Set up information about value arguments, emitting before the lit.endfn.
  IREmitter emitter(decl, OpBuilder(&body.front()));

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
          makeVarArgWrapper(bbArg, argName, decl, emitter, argDecl.getLoc(),
                            funcSignature.getPosVarArgConvention(argIdx));
      if (!declOp)
        return failure();
      declOp.setArgShadowIndex(bbArg.getArgNumber());
      setDecl(DeclIRValue(declOp));
      continue;
    }

    CValue argValue;
    if (convention == ArgConvention::ReadMem) {
      setDecl(MBValue(bbArg)); // borrowed
      continue;
    }
    if (convention == ArgConvention::ReadReg) {
      // borrowed_in_reg is used for @register_passable("trivial") types, where
      // borrowed vs owned doesn't matter so we use SRValue.
      setDecl(SRValue(bbArg));
      continue;
    }

    // Ref convention works with registers and def functions without any funny
    // business.
    setDecl(CValue::getMValueForRef(bbArg));
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

  if (funcOp.isExternal()) {
    shared.emitError(decl.getLoc(),
                     "unexpected function body in extern function "
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
  // TODO: Sink this to when the body is actually resolved.
  decl.resolvedness = DeclResolvedness::body;

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
  // TODO: Sink this to when the body is actually resolved.
  decl.resolvedness = DeclResolvedness::body;

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
    auto importDecl = LIT::UnresolvedImportOp::create(
        builder, op->getLoc(), importName, boundName, /*declName=*/StringAttr(),
        /*importNameLoc=*/LocationAttr(),
        /*destNameLoc=*/LocationAttr());
    getDeclResolver().addDecl(importDecl, decl.loc, boundName, &decl,
                              LexerCursor(), LexerCursor(), /*indentation=*/-1);

    // Create an alias for the unmangled module name to allow for simplified
    // indexing into this module.
    boundName = builder.getStringAttr(name);
    importDecl = LIT::UnresolvedImportOp::create(
        builder, op->getLoc(), importName, boundName, /*declName=*/StringAttr(),
        /*importNameLoc=*/LocationAttr(),
        /*declNameLoc=*/LocationAttr());
    getDeclResolver().addDecl(importDecl, decl.loc, boundName, &decl,
                              LexerCursor(), LexerCursor(), /*indentation=*/-1);
  }

  // Create a full wildcard import from the __init__, as the symbols defined
  // there are visible from the package.
  StringAttr importModule = builder.getStringAttr(".__init__");
  UnresolvedWildcardImportOp::create(builder, op->getLoc(), importModule,
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
    if (auto initDeclOp =
            dyn_cast_or_null<ASTDeclInterface>(initDecl.getIfOperation())) {
      // Inherit the docstring from the __init__ if it is present.
      if (auto docstring = initDeclOp.getDocStringAttr())
        op.setDocStringAttr(docstring);
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Alias Decl implementation
//===----------------------------------------------------------------------===//

/// alias_decl_stmt ::=
///   | "alias" identifier [param_signature] ":" expression ["=" expression]
///   | "alias" identifier [param_signature] "=" expression
///
LogicalResult DeclResolver::resolveSignature(AliasDeclOp aliasDeclOp,
                                             Lexer &lexer, ASTDecl &decl) {
  ParserBase p(shared, lexer);

  // Parse the decorators for alias declarations but only process them if they
  // are outside function bodies. This is because decorators inside function
  // bodies are not allowed, and this prevents redundant (and potentially
  // confusing) error messages.
  auto decoratorExprs = p.parseDecorators(decl);
  if (!isa_and_nonnull<FnOp>(decl.getParentDecl()->getIfOperation())) {
    Decorators(decl, /*signatureOnly=*/true)
        .applySignatureDecorators(decoratorExprs);
  }

  // Parse the type if present. Accept either 'alias' or 'comptime' keyword.
  SMLoc identifierLoc;
  if (p.getToken().isNot(Token::kw_alias, Token::kw_comptime)) {
    p.emitError(p.getToken().getLoc(),
                "internal error: checked by stmt parser");
    return failure();
  }
  p.consumeToken(); // Consume either kw_alias or kw_comptime

  if (p.parseIdentifier("internal error: checked by stmt parser",
                        &identifierLoc))
    return failure();

  // Parse the param signature if present.
  ParsedParamList parsedParams;
  if (parsedParams.parseParametersIfPresent(p, ArgListKind::kParamList))
    return failure();

  // The alias signature is a self-contained scope where the input parameters of
  // the alias are visible by all types.  We must use a temporary declaration
  // here (with an empty name) because we don't want references to the alias
  // itself to resolve to a fully-resolved decl, but we need a fully-resolved
  // decl for incremental lookups within the scope to work out.
  ASTDecl &sigDecl =
      addFullyResolvedDecl(aliasDeclOp.getOperation(), StringAttr(),
                           decl.getLoc(), decl.getParentDecl());

  std::optional<TypeCheckedParamList> paramSignatureOrError =
      TypeCheckedParamList::create(parsedParams, sigDecl);
  if (!paramSignatureOrError.has_value())
    return failure();
  TypeCheckedParamList &paramSignature = *paramSignatureOrError;

  ASTType type;
  if (p.consumeIf(Token::colon)) {
    if (parseType(p, type, sigDecl, decl.getIndentation()))
      return failure();
  }

  // If there are input parameters, the actual type of the alias is a generator
  // type. Parameterize the type with the input parameters.
  // The type of the alias is a standalone type, so it needs to reference its
  // input parameters by index refs (IRAIDAI), not name refs. This remapper
  // handles converting the name refs to index refs.
  IndexRefRemapper remapper(paramSignature.paramDeclAttrs, {});
  auto parameterizeType = [&](ASTType type) -> ASTType {
    if (paramSignature.paramDeclAttrs.empty())
      return type;

    SmallVector<Type> inputParamTypes;
    for (ParamDeclAttr param : paramSignature.paramDeclAttrs)
      inputParamTypes.push_back(remapper.replace(param.getType()));

    return GeneratorType::get(
        inputParamTypes, remapper.replace(type.mlirType),
        remapper.replace(paramSignature.getParamListAttr()));
  };

  ASTDecl &parentDecl = *decl.getParentDecl();

  NamedAttrList attrs = aliasDeclOp->getAttrDictionary();
  if (p.consumeIf(Token::equal)) {
    // Then this is a normal `alias` declaration with an initializer.
    ExprNode *initExpr = nullptr;
    if (p.parseVarInitExpression(initExpr, decl.getIndentation()))
      return failure();

    IREmitter emitter(sigDecl, EC_AliasValue);

    // Emit the value and convert to the expected type if we know it.
    PValue rhsValue = emitter.emitExprPValue(initExpr, EC_AliasValue, type);
    if (!rhsValue)
      return failure();

    // If we had no declared type (`alias x = 42`), infer the type from the
    // initializer.
    if (!type)
      type = rhsValue.getType();

    // If there are input parameters, we need to emit a value generator attr.
    type = parameterizeType(type);
    if (!paramSignature.paramDeclAttrs.empty()) {
      TypedAttr remappedBody = remapper.replace(rhsValue.get());
      auto genTp = cast<GeneratorType>(type);
      rhsValue = cast<TypedAttr>(GeneratorAttr::get(
          genTp.getInputParamTypes(), remappedBody, genTp.getMetadata()));
    }
    // Remember the value
    attrs.set(aliasDeclOp.getValueAttrName(), rhsValue.get());
  } else {
    if (!isa_and_nonnull<LIT::TraitDeclOp>(parentDecl.getIfOperation())) {
      // Disallow this, because it would create diamond inheritance problems.
      p.emitError(identifierLoc)
          << "only traits may contain a comptime member without an initializer";
      return failure();
    }

    if (!type) {
      p.emitError(identifierLoc)
          << "comptime value without an intitializer must have a type";
      return failure();
    }

    type = parameterizeType(type);
  }

  // Propagate signature errors and decls.
  decl.takeDecls(sigDecl);

  // Update the type from UnresolvedType
  attrs.set(aliasDeclOp.getParamDeclAttrName(),
            ParamDeclAttr::get(aliasDeclOp.getName(), type));
  aliasDeclOp->setAttrs(attrs.getDictionary(decl.getContext()));

  // Process the doc string of the alias.
  p.parseDocString(decl);

  if (auto parentTraitRef = dyn_cast_if_present<SymbolRefAttr>(
          aliasDeclOp->getAttr("parentTraitRef"))) {
    // Cleanup after ourselves.
    aliasDeclOp->removeAttr("parentTraitRef");

    // This can happen since since the signature resolution branch of the
    // overall 'resolve' function in DeclResolver doesn't guard against the
    // input decl being erroneous. Rather than add that check there for this
    // singular exceptional case catch it now.
    if (decl.isErroneous())
      return failure();

    ASTDecl &traitDecl = *decl.getParentDecl();
    auto name = demangleParameterName(*decl.getNameIfOperation());

    ASTDecl &parentTraitDecl = getDeclForTypeSymbol(parentTraitRef);

    auto decls = parentTraitDecl.lookupInCurrentScope(name);
    assert(decls.size() == 1 && "Expected to find exactly one decl");
    auto parentAliasDeclOp =
        cast<AliasDeclOp>(*decls.front()->getIfOperation());

    if (failed(resolveSignature(*decls.front(), decls.front()->getLoc())))
      return failure();

    SyntheticNode synthNode(traitDecl.getLoc());
    auto overrideAliasType = aliasDeclOp.getType();
    // Conjure a fake value here that we can hand to
    // canImplicitlyConvertToType.
    // TODO: Make a version of canImplicitlyConvertToType that can take
    // two types directly.
    // TODO: Be able to do this with canZeroCostConvert since we don't
    // want to call implicit constructors here.
    auto overrideAliasParamValue =
        PValue(ParamDeclRefAttr::get(aliasDeclOp.getParamDecl()));
    if (!IREmitter::canImplicitlyConvertToType(
            {overrideAliasParamValue, synthNode},
            parentAliasDeclOp.getParamDecl().getType(), traitDecl)) {
      auto diag = emitError(aliasDeclOp->getLoc(), "invalid redefinition of '")
                  << name << "': cannot convert " << ASTType(overrideAliasType)
                  << " to parent trait's member's type "
                  << ASTType(parentAliasDeclOp.getParamDecl().getType());
      diag.attachNote(parentAliasDeclOp->getLoc())
          << "parent trait's member defined here";
      return failure();
    }
  }

  shared.notifyListenerOnAliasDecl(decl, identifierLoc);
  return success();
}

ParseResult DeclResolver::resolveBody(AliasDeclOp op, Lexer &lexer,
                                      ASTDecl &decl) {
  // TODO: Sink this to when the body is actually resolved.
  decl.resolvedness = DeclResolvedness::body;
  return success();
}

//===----------------------------------------------------------------------===//
// Struct Decl implementation
//===----------------------------------------------------------------------===//

/// For a struct or trait declaration, parse an optional list of parent traits
/// to inherit from. `immediateParents` will be populated with the smallest set
/// of equivalent parent trait decls.
static ParseResult
parseOptionalInheritanceList(ParserBase &p, ASTDecl &declScope, ASTDecl &decl,
                             StringRef declName, SharedState &shared,
                             DenseSet<SymbolRefAttr> &immediateParents) {
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
    auto traitType = sugarDynCast<TraitType>(type);
    if (!traitType) {
      if (sugarIsa<LIT::StructType>(type)) {
        p.emitError(loc) << "inheriting from structs is not allowed";
      } else if (sugarIsa<ParamType>(type)) {
        p.emitError(loc)
            << "inheriting from a parameter expression is not allowed";
      } else {
        p.emitError(loc) << "don't know how to inherit from this type";
      }
      if (!traitType) {
        declScope.setErroneous();
        return success();
      }
    }

    // Successively flatten the parent list so we always have all the parents
    // available to check.
    // TODO: Encode an "inherited from" here, to make diagnostics nice.
    for (SymbolRefAttr symbol : traitType.getSymbols()) {
      // If this symbol is already a parent, skip it.
      if (inheritedFrom->contains(symbol))
        continue;
      ASTDecl &traitDecl = shared.declResolver->getDeclForTypeSymbol(symbol);
      TraitType canonicalParent =
          cast_or_null<TraitDeclOp>(traitDecl.getIfOperation())
              .getCanonicalTrait();
      for (SymbolRefAttr parent : canonicalParent.getSymbols()) {
        inheritedFrom->try_emplace(parent, std::make_pair(symbol, loc));
        // Any immediate parent that is actually a parent of this `symbol` is no
        // longer an immediate parent.
        immediateParents.erase(parent);
      }
      // Insert this `symbol` as an immediate parent. This must happen after the
      // loop, because this symbol itself is part of `canonicalParent` too.
      immediateParents.insert(symbol);
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
      // RP types implicitly conforms to Movable
      if (ASTDecl *decl = shared.lookupBuiltinTrait(
              "Movable", structDecl.getParentDecl(), decorator->getLoc()))
        traits.push_back(decl->getSymbolRef());
      return success();
    }
    // We don't process @explicit_destroy here, we do it in resolveSignature.
  }

  // `x()` forms.
  if (auto callNode = dyn_cast<CallNode>(decorator)) {
    if (auto declRef = dyn_cast<DeclRefNode>(callNode->callee)) {
      // @register_passable("trivial")
      if (isTrivialRegisterPassable(callNode)) {
        structOp.setConvention(TypeConvention::RegisterPassableTrivial);
        if (ASTDecl *decl = shared.lookupBuiltinTrait(
                "ImplicitlyCopyable", structDecl.getParentDecl(),
                decorator->getLoc()))
          traits.push_back(decl->getSymbolRef());
        if (ASTDecl *decl = shared.lookupBuiltinTrait(
                "Movable", structDecl.getParentDecl(), decorator->getLoc()))
          traits.push_back(decl->getSymbolRef());
        return success();
      }

      // @nonmaterializable(TargetType)
      if (declRef->spelling == "nonmaterializable" &&
          callNode->operands.size() == 1) {
        if (auto drn = dyn_cast<DeclRefNode>(callNode->operands[0].expr)) {
          ASTDecl *parentDecl = structDecl.getParentDecl();
          IREmitter emitter(*parentDecl, EC_Type);
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
  DenseSet<SymbolRefAttr> immediateParents; // unused.
  SMLoc identifierLoc;
  if (p.parseToken(Token::kw_struct,
                   "internal error: checked by stmt parser") ||
      p.parseIdentifier("internal error: checked by stmt parser",
                        &identifierLoc) ||
      parsedParams.parseParametersIfPresent(p, ArgListKind::kParamList) ||
      parseOptionalInheritanceList(p, sigDecl, decl, structOp.getSymName(),
                                   shared, immediateParents) ||
      p.parseToken(Token::colon, "expected ':' in struct definition") ||
      decl.isErroneous())
    return failure();

  std::optional<TypeCheckedParamList> paramSignatureOrError =
      TypeCheckedParamList::create(parsedParams, sigDecl);
  if (!paramSignatureOrError.has_value())
    return failure();
  TypeCheckedParamList &paramSignature = *paramSignatureOrError;

  // Propagate signature errors and decls.
  decl.takeDecls(sigDecl);

  auto paramsArrayAttr =
      ParamDeclArrayAttr::get(getContext(), paramSignature.paramDeclAttrs);
  auto sig = TypeSignatureType::remapToSignature(
      silenceErrors(getContext()), paramsArrayAttr,
      paramSignature.getParamListAttr());
  assert(sig && "could not remap signature");
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
/// The second return element is the symbol this destructor is inherited from,
/// or null if it is self-declared.
static std::pair<SymbolConstantAttr, std::optional<SymbolRefAttr>>
lookupDestructor(ASTDecl &structDecl, SharedState &shared) {
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
  FnOp func = dyn_cast_or_null<FnOp>(delDecl.getIfOperation());
  if (!func) {
    shared.emitError(delDecl.getLoc(), "'__del__' must be a method");
    return {};
  }
  return {func.getBoundSymbolRef(shared.getEvaluationContext()),
          func.getInheritedFrom()};
}

/// Look up a special method impl for the specified `type` when there is exactly
/// one implementation (not overloaded).  This returns the method if successful,
/// and returns null if there is none.
static SymbolConstantAttr lookupSpecialMethod(ASTDecl &structDecl,
                                              StringRef name,
                                              SpecialFunctionKind specialKind) {
  LookupResult inits = structDecl.getShared().lookupAndResolveDecl(
      name, structDecl.getLoc(), structDecl, /*searchParentScopes=*/false);

  for (ASTDecl *candidate : inits.getIfSuccess()) {
    FnOp func = dyn_cast_or_null<FnOp>(candidate->getIfOperation());
    if (func && func.getSpecialFunctionKind() == specialKind)
      return func.getBoundSymbolRef(
          structDecl.getShared().getEvaluationContext());
  }
  return {};
}

namespace {
struct StructDecorators : public SharedStateUser {
  StructDecorators(StructDeclOp structOp, ASTDecl &structDecl,
                   DeclResolver &resolver)
      : SharedStateUser(resolver.shared), structOp(structOp),
        structDecl(structDecl) {}

  LogicalResult processBodyDecorator(ExprNode *decorator);

private:
  /// Process the @fieldwise_init body decorator on structs.
  void processFieldwiseInitDecorator(SMLoc decoratorLoc, bool isImplicit);

  StructDeclOp structOp;
  ASTDecl &structDecl;
};
} // namespace

/// Look at the initializers of the specified struct to see if there is already
/// a fieldwise init.  If so, return it, otherwise return null.
static FnOp findFieldwiseInit(ASTDecl &structDecl) {
  auto &shared = structDecl.getShared();

  LookupResult inits =
      shared.lookupAndResolveDecl("__init__", structDecl.getLoc(), structDecl,
                                  /*searchParentScopes=*/false);
  if (inits.isErroneous())
    return {};

  auto structOp = cast_or_null<StructDeclOp>(structDecl.getIfOperation());
  unsigned numFields = std::distance(structOp.getFieldDecls().begin(),
                                     structOp.getFieldDecls().end());
  for (ASTDecl *declaration : inits.getIfSuccess()) {
    auto func = dyn_cast_or_null<FnOp>(declaration->getIfOperation());
    if (!func)
      continue;
    auto signature = func.getFuncTypeGenerator();
    ArrayRef<Type> inputTypes = signature.getArguments();
    ArrayRef<ArgConvention> convs = signature.getArgConventions();
    // Ignore the result slot and error result.
    while (!convs.empty() && isResultSlot(convs.back())) {
      inputTypes = inputTypes.drop_back();
      convs = convs.drop_back();
    }
    // TODO: Handle default arguments.
    if (inputTypes.size() != numFields)
      continue;
    // Skip any kind of var-args.
    if (signature.getBody().getMetadata().hasAnyVarArg())
      continue;

    bool isMatch = true;
    for (auto [type, conv, field] :
         llvm::zip(inputTypes, convs, structOp.getFieldDecls())) {
      // Strip the pointer type if present.
      ASTType argType = type;
      // Fieldwise initializers must have read/owned conventions. ref etc
      // are lit.ref's mechanically but these are invisible the to the caller.
      if (hasImplicitOrigin(conv)) {
        if (conv != ArgConvention::ReadMem && conv != ArgConvention::OwnedMem &&
            conv != ArgConvention::DeinitMem) {
          isMatch = false;
          break;
        }
        argType = ASTType(argType).getReferenceElementType();
      }

      if (!argType.isEqualCanon(field.getType())) {
        isMatch = false;
        break;
      }
    }
    if (isMatch)
      return func;
  }
  return {};
}

/// Process the @fieldwise_init body decorator on structs. 'isRequired'
/// indicates whether it is an error to already have a fieldwise init.
void StructDecorators::processFieldwiseInitDecorator(SMLoc decoratorLoc,
                                                     bool isImplicit) {
  // Don't add one if we already have one.
  if (FnOp init = findFieldwiseInit(structDecl)) {
    auto diag =
        emitError(decoratorLoc, "'")
        << cast_or_null<StructDeclOp>(structDecl.getIfOperation()).getSymName()
        << "' has an explicitly declared fieldwise initializer";
    diag.attachNote(init.getLoc()) << "initializer declared here";
    return;
  }

  // Generate the fieldwise init.
  StructEmitter structEmitter(structDecl);
  auto fn = structEmitter.synthesizeFieldwiseInit();
  if (!fn)
    return;

  // If "implicit", check for validity and set the bit.
  if (isImplicit) {
    auto fieldsRange =
        cast_or_null<StructDeclOp>(structDecl.getIfOperation()).getFieldDecls();
    if (std::distance(fieldsRange.begin(), fieldsRange.end()) != 1) {
      emitError(decoratorLoc,
                "@fieldwise_init(\"implicit\") is only valid on structs "
                "with a single field");
      return;
    }
    fn.setImplicitConversion(ImplicitConversionKind::Implicit);
  }
}

LogicalResult StructDecorators::processBodyDecorator(ExprNode *decorator) {
  if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
    if (declRef->spelling == "fieldwise_init") {
      processFieldwiseInitDecorator(decorator->getRangeStart(),
                                    /*implicit*/ false);
      return success();
    }
    if (declRef->spelling == "value") {
      // TODO(Mojo 25.7): remove this entirely.
      shared.emitError(declRef->getLoc(),
                       "'@value' has been removed, please use "
                       "'@fieldwise_init' and explicit "
                       "`Copyable` and `Movable` conformances instead");
      return success();
    }
    if (declRef->spelling == "explicit_destroy")
      return success();
  }
  if (auto callNode = dyn_cast<CallNode>(decorator)) {
    if (auto declRef = dyn_cast<DeclRefNode>(callNode->callee)) {
      if (declRef->spelling == "explicit_destroy")
        return success();

      if (declRef->spelling == "fieldwise_init") {
        if (callNode->operands.size() != 1 ||
            !isa<StringLiteralNode>(callNode->operands.front().expr) ||
            cast<StringLiteralNode>(callNode->operands.front().expr)
                    ->getValue() != "implicit")
          emitError(decorator->getRangeStart(),
                    "@fieldwise_init only allows an \"implicit\" argument");
        processFieldwiseInitDecorator(decorator->getRangeStart(),
                                      /*implicit*/ true);
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
  auto conformsToTrait = [&](StringRef traitName) {
    ASTDecl *traitDecl = shared.lookupBuiltinTrait(
        traitName, structDecl.getParentDecl(), structDecl.getLoc());
    if (!traitDecl)
      return false;
    auto trait = dyn_cast_or_null<TraitDeclOp>(traitDecl->getIfOperation());
    if (!trait)
      return false;
    return structDecl.doesNominalTypeConformTo(trait.bindReference());
  };

  // If the type lacks a __sp_fn__is_trivial member, synthesize it to
  // unresolved.
  auto synthesizeTrivialFlagIfNeeded = [&](StringRef spFnName) {
    std::string trivialDelTag = (spFnName + "is_trivial").str();
    if (!shared.typeHasMember(structDecl, trivialDelTag, structDecl.getLoc()))
      StructEmitter(structDecl).synthesizeUnresolvedAlias(trivialDelTag);
  };

  // Push the debug scope for this struct if necessary so that nested operations
  // have proper debug info.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(structOp.getLocScope());

  // Parse the body of the struct, which will give us all the methods and
  // fields, but without resolving their signatures or bodies.
  if (ParserBase(shared, lexer).parseSuite(structDecl))
    return failure();

  // At this point, we have to mark the struct as body resolved... for very
  // unfortunate reasons. The issue is that we need nested declarations (e.g.
  // struct fields) to be able to do unqualified name lookups from within the
  // struct body:
  //
  //    struct S:
  //       var x : Int  # Must look up 'Int', it isn't a param of 'S'.
  //
  // To support this, we mark the struct as body resolved at this point, even
  // though we don't even know what all the decls are within it - we're about to
  // synthesize new members etc, which means it definitely isn't body resolved
  // here.  This is a phase ordering and a modeling problem with ASTDecl - we
  // could add a new resolvedness level for this (between signature and body
  // resolved indicating that we can name lookup through it?).
  structDecl.resolvedness = DeclResolvedness::body;

  // This collects all the resolved struct fields. Now that the body is
  // parsed we can check the declared fields for extra invariants.
  bool hasBadField = false;
  bool hasNonTrivialDestructor = false;
  SmallVector<std::pair<StructFieldOp, ASTDecl *>> structFields;

  // Iterate over all the parsed decls.  in general these won't be signature
  // resolved, and we don't want to resolve functions.  We do need to resolve
  // struct fields signatures to understand their type.
  for (std::pair<StringAttr, TinyPtrVector<ASTDecl *>> decls :
       structDecl.getDeclsInScope()) {
    for (ASTDecl *decl : decls.second) {
      auto fieldOp = dyn_cast_or_null<StructFieldOp>(decl->getIfOperation());
      if (!fieldOp)
        continue;

      if (failed(resolveSignature(*decl, decl->getLoc()))) {
        hasBadField = true;
        continue;
      }
      if (ASTType(fieldOp.getType())
              .hasNontrivialDestructor(decl->getLoc(), shared))
        hasNonTrivialDestructor = true;
      structFields.push_back({fieldOp, decl});
    }
  }

  // Determine if there is an explicit conformance to AnyType.
  bool implicitlyDestructible = conformsToTrait("AnyType");

  // Check to see if there is a destructor and install it into the StructDeclOp
  // if so.
  bool synthesizedDtor = false;
  if (auto dtorAttr = lookupDestructor(structDecl, shared).first) {
    // Check to see if we have an explicitly declared destructor.
    structOp.setDestructorAttr(dtorAttr);
  } else if (implicitlyDestructible) {
    synthesizedDtor = true;
    (void)StructEmitter(structDecl).synthesizeEmptyDtor();
  }

  // If the structure conforms to "AnyType", we populate the trivial flag.
  if (implicitlyDestructible)
    synthesizeTrivialFlagIfNeeded("__del__");

  // If the struct conforms to well-known traits but doesn't have explicit
  // implementations of the corresponding methods, add signatures for them.
  // These can all be synthesized without resolving the members.
  if (conformsToTrait("Movable")) {
    if (!shared.typeHasMember(structDecl, "__moveinit__", structDecl.getLoc()))
      StructEmitter(structDecl).synthesizeEmptyMoveOrCopyInit(/*isMove=*/true);
    synthesizeTrivialFlagIfNeeded("__moveinit__");
  }
  if (conformsToTrait("Copyable")) {
    // TODO: this should synthesize a keyword only copy argument:
    // __copyinit__(out self, *, copy=others)
    if (!shared.typeHasMember(structDecl, "__copyinit__", structDecl.getLoc()))
      StructEmitter(structDecl).synthesizeEmptyMoveOrCopyInit(/*isMove=*/false);
    // NOTE: We don't need to synthesize copy() here, there should be a default
    // implementation.
    synthesizeTrivialFlagIfNeeded("__copyinit__");
  }

  // If we synthesized a destructor but the fields are all trivial, just drop
  // the destructor so CheckLifetimes doesn't need to worry about emitting calls
  // to it.
  if (synthesizedDtor && !hasNonTrivialDestructor)
    structOp.setDestructorAttr({});

  // If the struct is @register_passable, check invariants imposed by it before
  // checking other decorators.  This ensures that we reject invalid
  // register_passable types before processing them.
  if (structOp.isRegisterPassable())
    processRegisterPassableDecorator(structOp, structDecl, structFields, *this,
                                     structOp.getConvention());

  // Look up move and copy constructors and record them if declared.
  if (auto copyInitAttr = lookupSpecialMethod(structDecl, "__copyinit__",
                                              SpecialFunctionKind::kCopyInit))
    structOp.setCopyInitAttr(copyInitAttr);
  if (auto moveInitAttr = lookupSpecialMethod(structDecl, "__moveinit__",
                                              SpecialFunctionKind::kMoveInit))
    structOp.setMoveInitAttr(moveInitAttr);

  // If any of the fields are bad, we do not process decorators since they
  // assume that the struct body if valid.
  if (hasBadField && !structDecl.getBodyDecorators().empty()) {
    structDecl.setErroneous();
    return failure();
  }

  // If there are any body decorators, resolve them now.
  StructDecorators structDecorators(structOp, structDecl, *this);
  Decorators(structDecl).applyBodyDecorators([&](ExprNode *decorator) {
    return structDecorators.processBodyDecorator(decorator);
  });

  if (structDecl.isErroneous())
    return success();

  // Finally, emit empty conformance tables.
  ImplicitLocOpBuilder b = ImplicitLocOpBuilder::atBlockEnd(
      structOp.getLoc(), &structOp.getFields().front());
  for (SymbolRefAttr parent : structOp.getCanonicalTrait().getSymbols()) {
    StringAttr name = b.getStringAttr(getFlattenedSymbolName(parent));
    ASTDecl &parentDecl = getDeclForTypeSymbol(parent);
    SymbolRefArrayAttr immediateParents =
        cast_or_null<TraitDeclOp>(parentDecl.getIfOperation())
            .getImmediateParentsAttr();
    ConformanceOp witnessTable =
        ConformanceOp::create(b, name, parent, immediateParents);
    witnessTable.getBody().push_back(new Block());
    ASTDecl &decl = addDecl(witnessTable, structDecl.getLoc(), name,
                            &structDecl, {}, {}, -1);
    decl.resolvedness = DeclResolvedness::signature;
    // Conformances are always created as signature-resolved because there's no
    // less-resolved state for it (see CALROC for more).

    // Make sure the trait decl has been body resolved so we can check if
    // any methods provide implementations.
    if (failed(resolveBody(parentDecl, parentDecl.getLoc()))) {
      structDecl.setErroneous();
      return failure();
    }

    auto isInherited = [&](auto nestedOp, ASTDecl &parentDecl) {
      if (nestedOp.getInheritedFrom())
        return true;

      // inheritedFrom is set by signature resolution -- for inherited trait
      // methods the decl itself might contain a reference to the lit.fn op from
      // the parent this checks for that.
      auto parentTraitOp = cast<TraitDeclOp>(nestedOp->getParentOp());
      return getFullyResolvedSymbolRef(parentTraitOp) !=
             parentDecl.getSymbolRef();
    };

    auto isDefaulted = [](auto nestedOp) {
      if constexpr (std::is_same_v<FnOp, decltype(nestedOp)>)
        return nestedOp.isDefaultedTraitFn();
      else if constexpr (std::is_same_v<AliasDeclOp, decltype(nestedOp)>)
        return nestedOp.isDefaultedAssociatedAlias();
      return false;
    };

    auto insertDefaultDecl = [&](auto newOp, StringAttr childName,
                                 ASTDecl *childDecl) -> LogicalResult {
      if (!isDefaulted(newOp) || isInherited(newOp, parentDecl))
        return success();

      if constexpr (std::is_same_v<AliasDeclOp, decltype(newOp)>) {
        // Since we do not allow alias to be overloaded, we can at
        // most insert one defaulted alias per name into the
        // struct (if there is no user-provided one already) or it
        // will lead to redefinition error. If there are multiple
        // defaulted alias with the same name, raise an error.
        for (ASTDecl *existingDecl :
             structDecl.lookupInCurrentScope(childName)) {
          Operation *op = existingDecl->getIfOperation();
          if (auto existingAlias = dyn_cast_or_null<AliasDeclOp>(op);
              existingAlias && existingAlias.isDefaultedAssociatedAlias()) {

            SymbolRefAttr currentTraitRef = getFullyResolvedSymbolRef(
                existingAlias->getParentOfType<TraitDeclOp>());

            StringRef currentTraitName = currentTraitRef.getLeafReference();
            StringRef otherTraitName =
                childDecl->getParentDecl()->getSymbolRef().getLeafReference();

            // There are multiple default associated aliases with
            // the same name. Raise an error.
            auto diag =
                shared.emitError(structDecl.getLoc())
                << "trait member '"
                << demangleParameterName(existingAlias.getDeclName().getValue())
                << "' has conflicting default implementations in "
                << otherTraitName << " and " << currentTraitName
                << ", you must implement it manually";

            diag.attachNote(existingDecl->getLoc())
                << "original default implementation from trait "
                << currentTraitName << " here";

            diag.attachNote(newOp.getLoc())
                << "conflicting implementation from trait " << otherTraitName
                << " here";

            structDecl.setErroneous();
            return failure();
          }
          // This is a user provided alias, which shadows the
          // default value.
          return success();
        }
      }

      // Create a decl corresponding to the trait method we're inheriting.
      //
      // NOTE: this decl points to the lit.fn op in the actual trait so we now
      // have two decls pointing to the same lit.fn op.
      //
      // Ideally we'd create a stub lit.fn op in the struct with it's
      // inheritedFrom attribute pointing to the symbol ref attr of the trait
      // method, but since symbols are only created at signature resolution
      // time for lit.fn ops that's not an option (and attempting to signature
      // resolve trait methods at this point tends to cause cycles so is not
      // an option).
      //
      // Stashing the trait's lit.fn op here gives us an easy way to refer
      // back to it, signature resolving this struct's decl will actually
      // create the lit.fn op in the struct.decl op's body.
      auto &decl = shared.getDeclResolver().addDecl(
          newOp, childDecl->getLoc(), childName, &structDecl, LexerCursor(),
          LexerCursor(), -1);
      decl.resolvedness = DeclResolvedness::unparsed;
      return success();
    };

    StructEmitter emitter(structDecl);
    SmallVector<std::pair<FnOp, ASTDecl *>> nonEmptyTraitFns;
    for (auto &[childName, childDecls] : parentDecl.getDeclsInScope()) {
      for (ASTDecl *childDecl : childDecls) {
        if (auto nestedOp = childDecl->getIfOperation()) {
          LogicalResult result =
              TypeSwitch<Operation &, LogicalResult>(*nestedOp)
                  .Case<FnOp, AliasDeclOp>([&](auto nestedOp) {
                    return insertDefaultDecl(nestedOp, childName, childDecl);
                  })
                  .Default(LogicalResult::success());

          if (failed(result))
            return failure();
        }
      }
    }
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

  if (sugarIsa<TraitType>(type)) {
    emitError(decl.getLoc()) << "dynamic traits not supported yet, please "
                                "use a compile time generic instead of "
                             << type;
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
LogicalResult
DeclResolver::addSelfTypeToTrait(TraitDeclOp traitOp, ASTDecl &decl,
                                 SmallVector<SymbolRefAttr> &parentTraits,
                                 DenseSet<SymbolRefAttr> &immediateParents) {
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

  // Add the immediate parents to the trait.
  SmallVector<SymbolRefAttr> immediateParentsVec(immediateParents.begin(),
                                                 immediateParents.end());
  sortAndDeduplicateSymbols(immediateParentsVec);
  traitOp.setImmediateParents(
      SymbolRefArrayAttr::get(ctx, immediateParentsVec));

  decl.setTypeDeclSelf(ASTDecl::computeSelfTypeForTrait(traitOp));
  return success();
}

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
  DenseSet<SymbolRefAttr> immediateParents;
  if (parseOptionalInheritanceList(p, *decl.getParentDecl(), decl,
                                   traitOp.getSymName(), shared,
                                   immediateParents))
    return failure();
  SmallVector<SymbolRefAttr> parentTraits;
  bool definesClosure = traitOp.getDefinesClosure();
  if (auto *inheritedFrom = decl.getTraitConformanceLineage()) {
    for (auto [symbol, _] : *inheritedFrom) {
      parentTraits.push_back(symbol);
      if (definesClosure)
        continue;
      ASTDecl &type = getDeclForTypeSymbol(symbol);
      if (auto traitDecl =
              dyn_cast_if_present<TraitDeclOp>(type.getIfOperation()))
        if (traitDecl.getDefinesClosure())
          definesClosure = true;
    }
  }
  traitOp.setDefinesClosure(definesClosure);

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
      // No need to add UnknownDestructibility to immediateParents, since it
      // has an empty requirements table.
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
        // Update immediateParents only if it is empty, otherwise some other
        // parent trait will have already added it.
        if (immediateParents.empty())
          immediateParents.insert(anyTypeDecl->getSymbolRef());
      }
    }
  }

  // Insert the implicit trait parameter:
  // - _Self: a value of this trait type - the struct conforming to this trait.
  if (failed(addSelfTypeToTrait(traitOp, decl, parentTraits, immediateParents)))
    return failure();

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

/// Update the types for a method pulled from a trait base to a derived trait,
/// so they refer to the correct self type.
static void replaceTraitAliasSelfTypes(AliasDeclOp alias,
                                       TypedAttr parentSelfType,
                                       TypedAttr traitSelfType) {
  assert(isa<ParamDeclRefAttr>(parentSelfType) &&
         isa<ParamDeclRefAttr>(traitSelfType));
  AttrReplacer replacer(parentSelfType, traitSelfType);
  alias.setParamDeclAttr(
      ParamDeclAttr::get(alias.getParamDecl().getName(),
                         // Get updated type with new Self.
                         replacer.replace(alias.getParamDecl().getType())));
}

void DeclResolver::addParentDeclsToTrait(TraitDeclOp traitOp,
                                         ASTDecl &traitDecl) {

  // Since we lazily resolve nested decls the inheritedFrom attribute may or may
  // not already be set. In cases where that attribute isn't set the decl will
  // have a different parent trait decl op than the passed in op.
  auto isInherited = [&](auto nestedOp, ASTDecl &parentDecl) {
    if (nestedOp.getInheritedFrom())
      return true;

    auto parentTraitOp = cast<TraitDeclOp>(nestedOp->getParentOp());
    return getFullyResolvedSymbolRef(parentTraitOp) !=
           parentDecl.getSymbolRef();
  };

  // Now just pull in the functions in the bodies of all parents.
  for (SymbolRefAttr parentOrSelf : traitOp.getCanonicalTrait().getSymbols()) {
    ASTDecl &parentOrSelfDecl = getDeclForTypeSymbol(parentOrSelf);
    if (&parentOrSelfDecl == &traitDecl)
      continue;
    auto &parentDecl = parentOrSelfDecl;

    if (failed(resolveBody(parentDecl, traitDecl.getLoc())))
      continue;

    // Inherit function members, which we can override without worry because
    // they are all just declarations.
    for (auto &[name, declsInParent] : parentDecl.getDeclsInScope()) {
      if (declsInParent.empty())
        continue;
      if (isa_and_nonnull<FnOp>(declsInParent.front()->getIfOperation())) {
        for (ASTDecl *decl : declsInParent) {
          auto func = cast<FnOp>(decl->getIfOperation());

          if (isInherited(func, parentDecl))
            continue;

          addDecl(func, decl->getLoc(), name, &traitDecl, LexerCursor(),
                  LexerCursor(), -1);
        }
      } else if (auto parentAliasDecl = dyn_cast<AliasDeclOp>(
                     declsInParent.front()->getIfOperation())) {
        assert(declsInParent.size() == 1 &&
               "Can't have two aliases with same name.");
        auto &declInParent = *declsInParent.front();

        if (isInherited(parentAliasDecl, parentDecl))
          continue;

        ArrayRef<ASTDecl *> overrides = traitDecl.lookupInCurrentScope(name);
        // If there's no overrides, then we need to copy the alias decl from the
        // parent trait into this one.
        if (overrides.size() == 0) {
          // Add a synthetic decl that points to the parent trait's alias decl
          // op
          addDecl(declInParent.getIfOperation(), declInParent.getLoc(), name,
                  &traitDecl, LexerCursor(), LexerCursor(), -1);
        } else {

          // Theoretically there should be at most one override, since
          // duplicates aren't even added to the trait's ASTDecl entries.
          assert(overrides.size() == 1);

          auto override = overrides.front();
          auto overrideAliasDecl =
              dyn_cast_or_null<AliasDeclOp>(override->getIfOperation());
          if (!overrideAliasDecl) {
            auto diag =
                emitError(override->getLoc(), "invalid redefinition of ")
                << name;
            diag.attachNote(overrideAliasDecl->getLoc())
                << "cannot overload with this non-comptime definition";
            continue;
          }

          // This check is necessary since an alias mau be defined multiple
          // times in a trait's inheritance tree. If this branch is true then
          // that means that the current trait didn't define an alias of 'name'
          // and ad already created a decl pointing to one of the parent trait's
          // aliases.
          if (isInherited(overrideAliasDecl, traitDecl))
            continue;

          // Store a SymbolRefAttr pointing to the parent trait of the alias
          // we're currently overriding.
          //
          // This allows us to lookup the parent trait and its alias whenever
          // the override alias gets signature resolved and ensures that it's
          // valid (the types of the aliases implicitly convert).
          override->getIfOperation()->setAttr("parentTraitRef",
                                              parentDecl.getSymbolRef());
        }
      }
    }
  }

  auto [dtor, inheritedFrom] = lookupDestructor(traitDecl, shared);
  if (dtor) {
    std::string traitName = getFlattenedSymbolName(
        inheritedFrom.value_or(traitDecl.getSymbolRef()));
    // No need to fold here since the typeValue is always non-concrete.
    auto getWitnessAttr = GetWitnessAttr::get(
        PValue(traitDecl.getTypeDeclSelf()),
        StringAttr::get(traitDecl.getContext(), traitName),
        dtor.getSymbol().getLeafReference(), dtor.getType());
    traitOp.setDtorWitnessAttr(getWitnessAttr);
  }
}

ParseResult DeclResolver::resolveBody(TraitDeclOp traitOp, Lexer &lexer,
                                      ASTDecl &traitDecl) {
  // TODO: Sink this to when the body is actually resolved.
  traitDecl.resolvedness = DeclResolvedness::body;

  // Push the debug scope for this trait if necessary so that nested operations
  // have proper debug info.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(traitOp.getLocScope());

  if (ParserBase(shared, lexer).parseSuite(traitDecl))
    return failure();

  addParentDeclsToTrait(traitOp, traitDecl);

  return success();
}

/// Handles signature resolving inherited function decls in traits. In such
/// cases the passed in ASTDecl will be a child of the actual trait we're
/// working on, while the function op it contains is actually from the parent
/// trait we're inheriting from.
///
/// This logic was originally invoked during trait body resolution -- in an
/// effort to make the resolution of child declarations of traits lazier we've
/// moved it here.
///
/// The majority of the logic is largely the same as the less lazy version
/// except for some of the initial op and decl lookups.
LogicalResult
DeclResolver::resolveSyntheticSignature(FnOp inheritedFnOp,
                                        ASTDecl &childTraitFnDecl) {
  assert(isa<TraitDeclOp>(inheritedFnOp->getParentOp()) &&
         "Expected synthetic function decl's parent to be a trait");

  auto childTraitDecl = childTraitFnDecl.getParentDecl();

  // This covers the case of trait -> struct default method inheritance.
  if (inheritedFnOp.isDefaultedTraitFn() &&
      isa_and_nonnull<StructDeclOp>(childTraitDecl->getIfOperation()))
    return resolveDefaultedOpFromTrait(*this, inheritedFnOp, childTraitDecl);

  // This is the actual child trait of the decl.
  TraitDeclOp childTraitDeclOp =
      cast<TraitDeclOp>(childTraitDecl->getIfOperation());

  // And this is the parent trait of the function we're inheriting from.
  TraitDeclOp parentTraitDeclOp =
      cast<TraitDeclOp>(inheritedFnOp->getParentOp());

  SymbolRefAttr parentTraitRef = getFullyResolvedSymbolRef(parentTraitDeclOp);

  ASTDecl &parentTraitDecl = getDeclForTypeSymbol(parentTraitRef);
  auto functionName =
      dyn_cast<ASTDeclInterface>(inheritedFnOp.getOperation()).getDeclName();

  auto parentOverloadDecls = parentTraitDecl.lookupInCurrentScope(functionName);

  ASTDecl *inheritedFnDecl = nullptr;
  for (auto &overloadDecl : parentOverloadDecls) {
    if (inheritedFnOp.getOperation() == overloadDecl->getIfOperation()) {
      inheritedFnDecl = overloadDecl;
      if (failed(resolveSignature(*overloadDecl, overloadDecl->getLoc())))
        return failure();
    }
  }

  assert(inheritedFnDecl &&
         "Couldn't find the decl for inheritedFnOp in the parent trait.");

  auto parentFnSymName = inheritedFnOp.getSymNameAttr();

  DenseSet<StringAttr> existingFns;
  auto childFnDecls = childTraitDecl->lookupInCurrentScope(functionName);

  bool markDisabled = false;
  // Signature resolve all corresponding overloads in the child trait decl.
  for (auto &childOverload : childFnDecls) {
    auto actualParentTraitRef = getFullyResolvedSymbolRef(
        cast<TraitDeclOp>(childOverload->getIfOperation()->getParentOp()));

    // Skip processing any inherited members to avoid cycles.
    if (actualParentTraitRef != getFullyResolvedSymbolRef(childTraitDeclOp))
      continue;

    if (failed(resolveSignature(*childOverload, childOverload->getLoc())))
      return failure();

    auto childFnSymName =
        cast<FnOp>(childOverload->getIfOperation()).getSymNameAttr();

    // We've found that the child trait implements an overload with equivalent
    // signature. At this point we don't really care about this decl anymore.
    //
    // In such cases we'd really like to be able to just delete the decl we had
    // created at this point since nothing will ever actually make use of it (as
    // the child already has a definition).
    //
    // Unfortunately this isn't really possible without making it easy to hit UB
    // specifically around iterator invalidation.
    //
    // A very common sort of pattern across the parser is:
    //
    // for (auto& [name, decls] : scope.getDeclsInScope()) {
    //   for (auto& decl : decls)
    //     resolveSignature(decl, decl.getLoc());
    // }
    //
    // For decls such as the one we're currently dealing with this code would
    // bottom out here in this function and to properly remove childTraitFnDecl
    // from its parent scope we'd have to reach into one of the sub entries of
    // ASTDecl::declsInScope which in turn would cause issues with the second
    // for loop in the example above.
    if (parentFnSymName == childFnSymName) {
      markDisabled = true;
      break;
    }
  }

  // We need to make sure that the decl for the function we're inheriting is now
  // fully resolved.
  if (failed(resolveBody(*inheritedFnDecl, inheritedFnDecl->getLoc())))
    return failure();

  auto parentTraitSelfType = parentTraitDecl.getTypeDeclSelf();
  auto childTraitSelfType = childTraitDecl->getTypeDeclSelf();

  // Clone the function over but leave an empty body.
  //
  // This is necessary to avoid errors around type mismatches between trait self
  // types, to make this concrete consider:
  //
  //
  // trait Foo:
  //   fn foo(self) -> Int:
  //     ...
  //
  // trait Bar(Foo):
  //   fn bar(self) -> Int:
  //     return self.foo() * 2
  //
  // trait Baz(Bar):
  //   ...
  //
  // If we just naively cloned the full body of Bar.bar into Baz the lit.call to
  // foo would be expecting an argument of type Bar rather than Baz.
  //
  // Since we're only ever dealing with inherited trait methods in this function
  // and structs get to see a flat list of all their parent trait methods we'll
  // still be able to appropriately pick up the parent trait method with the
  // actual defaulted implementation.
  auto clonedFunc = inheritedFnOp.cloneWithoutRegions();

  {
    Block *entryBlock = clonedFunc.addEntryBlock();
    auto builder = OpBuilder::atBlockEnd(entryBlock);
    UnreachableOp::create(builder, clonedFunc.getLoc());
  }

  // In this case the child trait has an override for a method defined in the
  // parent trait. In these sorts of cases we need to 'deactivate' the decl we
  // had created earlier we do this by creating an empty body for the function
  // and marking it so that overload resolution will never pick it.
  if (markDisabled) {
    // We set this property and use it during overload resolution to skip
    // declarations like this.
    clonedFunc.setDisabled(true);

    // Append parent trait name to the function name
    auto parentTraitName = parentTraitDeclOp.getSymNameAttr();
    auto sourceName = clonedFunc.getSymNameAttr();
    // We need to make sure we won't have a symbol that will conflict with the
    // child's override.
    auto newName = StringAttr::get(clonedFunc->getContext(),
                                   parentTraitName.getValue() +
                                       "::" + sourceName.getValue());
    clonedFunc.setSymNameAttr(newName);
  }

  replaceTraitMethodSelfTypes(clonedFunc, PValue(parentTraitSelfType).get(),
                              PValue(childTraitSelfType).get());
  clonedFunc.setInheritedFromAttr(parentTraitRef);

  childTraitDeclOp.getBody()->push_back(clonedFunc);
  childTraitFnDecl.setIRValue(clonedFunc.getOperation());
  childTraitFnDecl.resolvedness = DeclResolvedness::body;

  // Clear the function body and replace with just kgen.unreachable
  // since we don't need to preserve the actual implementation
  clonedFunc.getBody()->clear();

  {
    auto builder = OpBuilder::atBlockEnd(clonedFunc.getBody());
    UnreachableOp::create(builder, clonedFunc.getLoc());
  }

  return success();
}

/// Handles signature resolving inherited alias decls in traits. In such cases
/// the passed in ASTDecl will be a child of the actual trait we're working on,
/// while the alias.decl op it contains is actually from the parent trait we're
/// inheriting from.
///
/// This logic was originally invoked during trait body resolution -- in an
/// effort to make the resolution of child declarations of traits lazier we've
/// moved it here.
///
/// The majority of the logic is largely the same as the less lazy version
/// except for some of the initial op and decl lookups.
LogicalResult
DeclResolver::resolveSyntheticSignature(AliasDeclOp inheritedAliasOp,
                                        ASTDecl &childTraitAliasDecl) {
  auto getFnIsTrivialKind = [](StringRef trivialTagName) {
    // Matching by name is a bit gross, but we don't have general synthesized
    // decls so it should be robust.
    if (trivialTagName == "__del__is_trivial")
      return SpecialFunctionKind::kDel;
    if (trivialTagName == "__moveinit__is_trivial")
      return SpecialFunctionKind::kMoveInit;
    if (trivialTagName == "__copyinit__is_trivial")
      return SpecialFunctionKind::kCopyInit;

    return SpecialFunctionKind::kNormal;
  };

  SpecialFunctionKind spFn = getFnIsTrivialKind(inheritedAliasOp.getDeclName());
  if (spFn != SpecialFunctionKind::kNormal) {
    StructEmitter gen(*childTraitAliasDecl.getParentDecl());
    TypedAttr isTrivial = gen.populateSpecialFnIsTrivial(
        getFnIsTrivialKind(inheritedAliasOp.getDeclName().strref()));

    if (isTrivial) {
      inheritedAliasOp.setParamDeclAttr(ParamDeclAttr::get(
          inheritedAliasOp.getParamDecl().getName(), isTrivial.getType()));
      inheritedAliasOp.setValueAttr(isTrivial);
    } else {
      // Something went wrong while resolving fields.
      childTraitAliasDecl.setErroneous();
    }
    childTraitAliasDecl.resolvedness = DeclResolvedness::body;
    return success();
  }

  assert(isa<TraitDeclOp>(inheritedAliasOp->getParentOp()) &&
         "Expected synthetic alias decl's parent to be a trait");

  ASTDecl *childTraitDecl = childTraitAliasDecl.getParentDecl();
  // This covers the case of trait -> struct default associated alias.
  if (inheritedAliasOp.isDefaultedAssociatedAlias() &&
      isa_and_nonnull<StructDeclOp>(childTraitDecl->getIfOperation()))
    return resolveDefaultedOpFromTrait(*this, inheritedAliasOp, childTraitDecl);

  // This is the actual child trait of the decl.
  TraitDeclOp childTraitDeclOp =
      cast<TraitDeclOp>(childTraitDecl->getIfOperation());

  // And this is the parent trait of the alias decl we're inheriting from.
  TraitDeclOp parentTraitDeclOp =
      cast<TraitDeclOp>(inheritedAliasOp->getParentOp());

  Block &childTraitBody = *childTraitDeclOp.getBody();
  SymbolRefAttr parentTraitRef = getFullyResolvedSymbolRef(parentTraitDeclOp);
  ASTDecl &parentTraitDecl = getDeclForTypeSymbol(parentTraitRef);

  // Since alias decls don't implement SymbolOpInterface we need to do a
  // lookup by source name.
  auto aliasName =
      demangleParameterName(inheritedAliasOp.getDeclName().getValue());

  auto parentAliasDecls = parentTraitDecl.lookupInCurrentScope(aliasName);
  auto &inheritedAliasDecl = *parentAliasDecls.front();

  assert(parentAliasDecls.size() == 1 &&
         isa_and_present<AliasDeclOp>(inheritedAliasDecl.getIfOperation()) &&
         "Expected to find exactly one comptime decl op");

  // Make sure to resolve the actual decl that holds inheritedAliasOp before we
  // proceed.
  if (failed(resolveBody(inheritedAliasDecl, inheritedAliasDecl.getLoc())))
    return failure();

  auto childTraitSelfType =
      childTraitAliasDecl.getParentDecl()->getTypeDeclSelf();
  auto parentTraitSelfType = parentTraitDecl.getTypeDeclSelf();

  auto clonedAliasDecl = inheritedAliasOp.clone();

  replaceTraitAliasSelfTypes(clonedAliasDecl, PValue(parentTraitSelfType).get(),
                             PValue(childTraitSelfType).get());

  // Mark the alias as inherited so that conformance checking won't
  // give duplicate errors if it is not provided.
  clonedAliasDecl.setInheritedFromAttr(parentTraitRef);
  childTraitBody.push_back(clonedAliasDecl);

  childTraitAliasDecl.setIRValue(clonedAliasDecl);
  childTraitAliasDecl.resolvedness = DeclResolvedness::body;
  // We don't need to call something like finalizeFuncSignature for
  // aliases because we can't have multiple aliases with the same name
  // (there's no such thing as alias overloading).

  return success();
}

//===----------------------------------------------------------------------===//
// Extension implementation
//===----------------------------------------------------------------------===//

LogicalResult DeclResolver::resolveSignature(ExtensionDeclOp extensionDeclOp,
                                             Lexer &lexer, ASTDecl &decl) {
  ParserBase p(shared, lexer);

  SMLoc identifierLoc;
  StringAttr structNameAttr;
  if (p.parseToken(Token::kw___extension,
                   "internal error: checked by stmt parser") ||
      p.parseIdentifier(structNameAttr,
                        "internal error: checked by extension parser",
                        &identifierLoc))
    return failure();

  ASTDecl *parentDecl = decl.getParentDecl();
  assert(parentDecl && "Extension has no parent decl");

  // Look up all declarations with the same name. We need to look up ALL because
  // if we just look for the first, we'll find the extension itself, which has
  // the same name.
  // Note we aren't resolving the struct here, just finding it.
  // TODO(MOCO-522): Arcana references about how the extension has a unique
  // generated name, but is known to the parent ASTDecl (and therefore to the
  // rest of the world) as the struct's name.
  // TODO(MOCO-522): Consider requiring the extension to import the exact struct
  // rather than being able to import an intermediate extension. If we have that
  // restriction, then we can:
  // - Make the struct (e.g. Spaceship) known by two names:
  //   - "Spaceship" (as before)
  //   - "struct:Spaceship"
  //   The latter would let us do a single lookupAndResolveDecl here instead of
  //   the more expensive lookupAllDeclsWithName.
  // TODO(MOCO-522): Consider modifying the import system to automatically
  // import a target struct when we import an extension.
  // TODO(MOCO-522): Update the conflict test and simplify this to
  // lookupAndResolveDecl call, now that we've upgraded extension names.
  StringRef structName = structNameAttr.getValue();
  LookupAllResult lookupResult = shared.lookupAllDeclsWithName(
      structName, identifierLoc, *parentDecl, /*resolve=*/false);
  ArrayRef<ASTDecl *> foundDecls = lookupResult.getIfSuccess();
  // Find the actual struct declaration among all the found declarations.
  StructDeclOp structDeclOp = nullptr;
  ASTDecl *structAstDecl = nullptr;
  for (ASTDecl *decl : foundDecls) {
    if (auto foundStructDeclOp =
            dyn_cast_or_null<StructDeclOp>(decl->getIfOperation())) {
      structDeclOp = foundStructDeclOp;
      structAstDecl = decl;
      break;
    }
  }
  if (!structDeclOp) {
    return emitError(identifierLoc, "can't find a struct named '")
           << structName << "'";
  }
  if (failed(resolve(*structAstDecl, DeclResolvedness::signature,
                     identifierLoc))) {
    return failure();
  }

  SymbolRefAttr targetStructAttr = structAstDecl->getSymbolRef();
  extensionDeclOp.setTargetStructAttr(targetStructAttr);

  // This is an extension, but all the methods should think they're inside a
  // struct, so let's use 'computeSelfTypeForStruct' on the structDeclOp to
  // figure out the self type they can use.
  decl.setTypeDeclSelf(ASTDecl::computeSelfTypeForStruct(structDeclOp));

  // Use the parent scope to resolve the traits in the inheritance list.
  // TODO(MOCO-522): This might need to change once we have parametric traits,
  // we might want to resolve from the extension's scope at that point.
  DenseSet<SymbolRefAttr> immediateParents;
  if (failed(parseOptionalInheritanceList(p, *parentDecl, decl,
                                          extensionDeclOp.getSymName(), shared,
                                          immediateParents)))
    return failure();

  // Store the immediate parent traits in the extension
  SmallVector<SymbolRefAttr> immediateParentsVec(immediateParents.begin(),
                                                 immediateParents.end());
  sortAndDeduplicateSymbols(immediateParentsVec);
  extensionDeclOp.setImmediateParents(
      SymbolRefArrayAttr::get(getContext(), immediateParentsVec));

  // Compute canonicalTrait for the extension (flattened trait hierarchy)
  if (!immediateParentsVec.empty()) {
    SmallVector<SymbolRefAttr> canonicalSymbols(immediateParentsVec);
    TraitType canonicalTrait = getCanonicalTrait(canonicalSymbols);
    extensionDeclOp.setCanonicalTrait(canonicalTrait);
  }

  shared.notifyListenerOnTraitDecl(decl, identifierLoc);

  if (p.consumeIf(Token::l_square)) {
    // If the current token is on a new line, report the error on the end of
    // the previous line, this is probably where the punctuation was omitted.
    auto diagLoc = p.getTokenLocOrEndOfPreviousLineIfOnNewLine();
    // Report the error.
    auto diag = emitError(
        diagLoc, "cannot specify parameter declarations on extensions");

    diag.attachNote(structAstDecl->getLoc())
        << "extension already assumes these parameter declarations";
    return failure();
  }

  if (p.parseToken(Token::colon, "expected ':' in extension definition"))
    return failure();

  return success();
}

ParseResult DeclResolver::resolveBody(ExtensionDeclOp extensionDeclOp,
                                      Lexer &lexer, ASTDecl &extensionDecl) {
  SymbolRefAttr structSymbolRef = extensionDeclOp.getTargetStruct().value();

  ASTDecl &structAstDecl = getDeclForTypeSymbol(structSymbolRef);
  StructDeclOp structDeclOp =
      cast<StructDeclOp>(structAstDecl.getIfOperation());

  // Copy struct param decls into the extension, so extension methods and
  // aliases can reference the struct's param decls.
  for (ParamDeclAttr param : structDeclOp.getParams()) {
    StringAttr demangledName =
        StringAttr::get(getContext(), demangleParameterName(param.getName()));
    addFullyResolvedDecl(PValue(ParamDeclRefAttr::get(param)), demangledName,
                         extensionDecl.getLoc(), &extensionDecl);
  }
  // Set extension's parameters to match target struct. This is to make it so
  // the verifier can see the param-decls and be aware of them when it's
  // verifying their param-refs.
  // TODO(MOCO-522): Definitely need arcana docs here.
  // TODO(MOCO-522): Possibly-related problem: there might be parts of the
  // compiler that assume a method is contained by a *struct* specifically, and
  // that could cause problems when the method introduces its own param decls,
  // see https://github.com/modularml/modular/pull/69012.
  if (!structDeclOp.getParams().empty()) {
    extensionDeclOp.setParamsAttr(structDeclOp.getParamsAttr());
    extensionDeclOp.setSignature(structDeclOp.getSignature());
  }

  // Push the struct's debug scope for this extension if necessary so that
  // nested operations have proper debug info.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(structDeclOp.getLocScope());

  if (ParserBase(shared, lexer).parseSuite(extensionDecl))
    return failure();

  // Now check for conflicts; things in the extension shouldnt already be in the
  // struct, unless theyre both methods because overloading is fine.
  if (extensionDecl.declsInScope && structAstDecl.declsInScope) {
    for (auto &[declName, extensionMemberDecls] : *extensionDecl.declsInScope) {
      ASTDecl *firstExtensionMemberDecl = extensionMemberDecls.front();

      // Skip parameter declarations, they are intentionally inherited from the
      // struct, the extension has an ASTDecl for every param declaration from
      // the target struct. If this is a ParamDeclRefAttr, skip it.
      if (CValue cval = firstExtensionMemberDecl->getIfIRValue())
        if (PValue pval = cval.getIfPValue())
          if (isa<ParamDeclRefAttr>(pval.get()))
            continue;

      bool isExtensionMethod =
          isa_and_nonnull<FnOp>(firstExtensionMemberDecl->getIfOperation());
      auto it = structAstDecl.declsInScope->find(declName);
      if (it == structAstDecl.declsInScope->end()) {
        // If there's nothing in the struct with this name, no conflict, done.
        continue;
      }
      ASTDecl *firstStructMemberDecl = it->second.front();
      bool isStructMethod =
          isa_and_nonnull<FnOp>(firstStructMemberDecl->getIfOperation());

      if (isExtensionMethod && isStructMethod) {
        // Method overloading is okay, done.
        continue;
      }

      // Show an error for each conflicting member in the extension decl, and
      // mark it erroneous.
      for (ASTDecl *extensionMemberDecl : extensionMemberDecls) {
        auto diag =
            emitError(extensionMemberDecl->getLoc(), "invalid redefinition of ")
            << declName;
        diag.attachNote(firstStructMemberDecl->getLoc())
            << "extension " << (isExtensionMethod ? "method" : "declaration")
            << " conflicts with struct "
            << (isStructMethod ? "method" : "declaration");
        extensionMemberDecl->setErroneous();
      }
      return failure();
    }
  }

  // Generate conformance tables for the extension's traits that the struct
  // doesn't already have. Use set difference to avoid duplicate conformances
  // between struct and extension. Extensions might have no canonical trait.
  if (extensionDeclOp.getCanonicalTrait()) {
    Block *extensionBody = extensionDeclOp.getBody();
    ImplicitLocOpBuilder b = ImplicitLocOpBuilder::atBlockEnd(
        extensionDeclOp.getLoc(), extensionBody);

    // Get the target struct's existing conformances
    SmallVector<SymbolRefAttr> structConformances(
        structDeclOp.getCanonicalTrait().getSymbols().begin(),
        structDeclOp.getCanonicalTrait().getSymbols().end());

    // Compute set difference: extension traits - struct traits
    SmallVector<SymbolRefAttr> extensionOnlyTraits;
    for (SymbolRefAttr extensionTrait :
         extensionDeclOp.getCanonicalTrait()->getSymbols()) {
      if (!llvm::is_contained(structConformances, extensionTrait))
        extensionOnlyTraits.push_back(extensionTrait);
    }

    // Create conformances only for extension-specific traits
    for (SymbolRefAttr parent : extensionOnlyTraits) {
      StringAttr name = b.getStringAttr(getFlattenedSymbolName(parent));
      ASTDecl &parentDecl = getDeclForTypeSymbol(parent);
      SymbolRefArrayAttr immediateParents =
          cast_or_null<TraitDeclOp>(parentDecl.getIfOperation())
              .getImmediateParentsAttr();
      ConformanceOp witnessTable =
          ConformanceOp::create(b, name, parent, immediateParents);
      witnessTable.getBody().push_back(new Block());
      ASTDecl &decl = addDecl(witnessTable, extensionDecl.getLoc(), name,
                              &extensionDecl, {}, {}, -1);
      decl.resolvedness = DeclResolvedness::signature;
      // Conformances are always created as signature-resolved because there's
      // no less-resolved state for it (see CALROC for more).

      // Extension conformance verification follows the same pattern as structs
      // and is handled in verifyAndBuildConformance() during ConformanceOp body
      // resolution. The trait body will be resolved there.
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// UnresolvedImport Decl implementation
//===----------------------------------------------------------------------===//

ParseResult DeclResolver::resolveSignature(LIT::UnresolvedImportOp op,
                                           ASTDecl &decl, bool resolveTarget) {
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
        op.getImportNameAttr(), decl.getLoc(), declNameLoc, importNameLoc,
        resolveTarget);
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
  // TODO: Sink this to when the body is actually resolved.
  traitDecl.resolvedness = DeclResolvedness::body;

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

        if (auto fn = dyn_cast_or_null<FnOp>(decl->getIfOperation())) {
          if (fn.getInheritedFrom())
            continue;
        } else if (auto alias =
                       dyn_cast_or_null<AliasDeclOp>(decl->getIfOperation())) {
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
              return emitError(traitDecl.getLoc(),
                               "trait composition has conflicting types for '")
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
  // TODO: Sink this to when the body is actually resolved.
  decl.resolvedness = DeclResolvedness::body;
  // Verify conformance explicitly.
  std::optional<MojoInflightDiag> diag;

  // For extension conformances, we need to pass the target struct, not the
  // extension
  ASTDecl *declToVerify = getStructOrTargetStruct(*decl.getParentDecl(), *this);
  assert(declToVerify &&
         "ConformanceOps are only created inside structs or extensions");

  if (failed(verifyAndBuildConformance(*declToVerify, op.getTraitRefAttr(),
                                       diag, op)))
    return failure();

  return success();
}
