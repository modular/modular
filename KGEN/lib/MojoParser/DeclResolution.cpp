//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/CallEmission.h"
#include "KGEN/MojoParser/ClosureEmitter.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/ExprEmitter.h"
#include "KGEN/MojoParser/ExprNode.h"
#include "KGEN/MojoParser/ExprNodes.h"
#include "KGEN/MojoParser/ParserBase.h"
#include "KGEN/MojoParser/StructEmitter.h"
#include "KGEN/POPDialect/POPOps.h"
#include "ParsedArgument.h"
#include "Support/Compiler/OperationUtils.h"
#include "Utils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
#include "llvm/ADT/TypeSwitch.h"
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
struct Decorators : public SharedStateUser {
  Decorators(ASTDecl &decl, SharedState &shared)
      : SharedStateUser(shared), decl(decl) {}

  /// Process signature decorators on the declaration using the provided
  /// functor. The functor should return success if the decorator was processed
  /// as a signature decorator.
  void applySignatureDecorators(
      ArrayRef<std::pair<ExprNode *, LexerCursor>> decoratorExprs,
      function_ref<LogicalResult(ExprNode *)> process);

  /// Process body decorators on the declaration using the provided functor.
  /// The functor should return success if the decorator was processed as a
  /// signature decorator. Any leftover decorators are emitted and deferred to
  /// the operation.
  void applyBodyDecorators(function_ref<LogicalResult(ExprNode *)> process);

  /// The declaration this class is applying decorators to.
  ASTDecl &decl;
};
} // namespace

void Decorators::applySignatureDecorators(
    ArrayRef<std::pair<ExprNode *, LexerCursor>> decoratorExprs,
    function_ref<LogicalResult(ExprNode *)> process) {
  // Process decorators in the order they are seen. Stop at the first decorator
  // that needs to be deferred.
  while (true) {
    // Return if we are out of decorators.
    if (decoratorExprs.empty())
      return;
    if (failed(process(decoratorExprs.front().first)))
      break;
    decoratorExprs = decoratorExprs.drop_front();
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
  // Defer the rest of the decorators through the shared state.
  decl.setBodyDecorators(bodyDecorators, shared);
}

void Decorators::applyBodyDecorators(
    function_ref<LogicalResult(ExprNode *)> process) {
  // Don't run decorators if the declaration is invalid.
  if (decl.hasReferenceError)
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

  TypeSwitch<ASTDecl &, void>(decl)
      .Case<LIT::FuncOp, StructDeclOp, GlobalVarDeclOp>([&](auto op) {
        op.setDecoratorsAttr(DecoratorsAttr::get(op.getContext(), decoPValues));
      });
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
    auto strNode = dyn_cast<StringLiteralNode>(operand.value);
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

/// Now that all the structural properties are determined, perform any
/// name-binding specific checks over the declaration.  This happens after
/// decorator processing because that is how defs work in Python.  This also
/// fills in any implicitly declared types, performs name mangling, and sets up
/// the signature correctly.
///
/// This allows magic behavior (like __new__ being static, checking of method
/// self requirements and enforcement of other invariants.
///
/// This returns failure (after emitting an error) when a type checking problem
/// is detected.
static void
verifyFunctionNameBinding(ASTDecl &decl, LIT::FuncOp funcOp, StringAttr name,
                          SmallVector<ParsedArgument> &args,
                          ArrayRef<Type> argTypes, ASTType &resultType,
                          const FnEffects &effects, SharedState &shared,
                          SpecialFunctionInfo fnInfo) {
  // On any semantic error we mark the declaration erroneous - so references to
  // it don't type check, and we clear our special function information.  This
  // reduces cascade errors.
  auto emitErrorLoc = [&](SMLoc loc,
                          const Twine &message = Twine()) -> InflightDiag {
    fnInfo = SpecialFunctionInfo();
    decl.hasReferenceError = true;
    return shared.emitError(loc, message);
  };
  auto emitError = [&](const Twine &message = Twine()) -> InflightDiag {
    fnInfo = SpecialFunctionInfo();
    decl.hasReferenceError = true;
    return shared.emitError(funcOp.getLoc(), message);
  };

  // This is true if the declared result type is modeled as the first argument
  // because it is returned in memory.
  bool hasMemoryResult =
      !args.empty() &&
      args[0].convention == ParsedArgument::kConventionInOutResult;

  // If this definition is a struct/class member, compute the self type.
  ASTType selfType;
  ssize_t selfArgNumber = -1;
  if (ASTDecl *parent = decl.getParentDecl();
      parent && isa<StructDeclOp, TraitDeclOp>(*parent)) {
    // The parent decl must be fully resolved in order to resolve any of its
    // members.
    assert(parent->resolvedness == DeclResolvedness::fully);
    selfType = parent->getSelfType();
    // If there is an in-memory result, self is passed as arg #1 otherwise #0.
    selfArgNumber = hasMemoryResult ? 1 : 0;
  }

  // __*init__ methods are weird - for memory-only results we define
  // init in convention Python style, but for @register_passable values, we
  // return it.  We handle this by mapping them to different enumerators so
  // things downstream have stronger invariants.
  if ((fnInfo.kind == SpecialFunctionKind::kInit ||
       fnInfo.kind == SpecialFunctionKind::kCopyInit ||
       fnInfo.kind == SpecialFunctionKind::kMoveInit) &&
      selfType && ASTType(selfType).isRegisterPassable(decl.getLoc(), shared)) {
    if (fnInfo.kind == SpecialFunctionKind::kCopyInit)
      fnInfo = SpecialFunctionInfo::get(SpecialFunctionKind::kCopyInitReg);
    else if (fnInfo.kind == SpecialFunctionKind::kInit)
      fnInfo = SpecialFunctionInfo::get(SpecialFunctionKind::kInitReg);
    else {
      assert(fnInfo.kind == SpecialFunctionKind::kMoveInit);
      emitError() << name
                  << " is not supported for @register_passable types, they "
                     "are always movable by copying a register";
    }
  }

  // Check any special function information.

  // __new__ and similar methods are implicitly static.
  if (fnInfo.flags & SpecialFunctionInfo::kImplicitlyStaticMethod)
    funcOp.setIsStatic(true);

  // Check that the 'self' argument of a method was specified correctly.
  if (selfType && !funcOp.getIsStatic()) {
    if (selfArgNumber >= ssize_t(argTypes.size())) {
      // TODO('def' allows unused arguments): We can/should relax this for
      // 'def' declarations in the future, they should be able to implicit
      // ignore arguments like Python does.
      emitError("self argument must be present in instance method");
    } else if (!ASTType(argTypes[selfArgNumber]).isEqualCanon(selfType)) {
      auto diag = emitErrorLoc(args[selfArgNumber].loc,
                               "'self' argument must have type ")
                  << selfType << " but actually has type "
                  << ASTType(argTypes[selfArgNumber]);
      if (args[selfArgNumber].typeExpr)
        diag << args[selfArgNumber].typeExpr->getRange();
    }
  }

  if (funcOp.getIsStatic() && !selfType) {
    emitError("only methods on structs may be declared static");
    funcOp.setIsStatic(false);
  }

  // Verify the argument count lines up.
  if (fnInfo.kind != SpecialFunctionKind::kNormal) {
    size_t numActualArgs = args.size() - std::max(selfArgNumber, ssize_t(0));
    size_t numMin = fnInfo.minNumArguments;
    ssize_t numMax = fnInfo.maxNumArguments;
    if (numMin == size_t(numMax) && numActualArgs != numMin) {
      emitError("special function ")
          << name << " must have " << numMin << " operand" << plural(numMin);
    } else if (numActualArgs < numMin) {
      emitError("special function ") << name << " must have at least " << numMin
                                     << " operand" << plural(numMin);
    } else if (numMax != -1 && numActualArgs > size_t(numMax)) {
      emitError("special function ")
          << name << " must have at most " << size_t(numMax) << " operand"
          << plural(numMax);
    }
  }

  // Check other invariants based on method flags.
  if (fnInfo.isInstMethod()) {
    if (!selfType) {
      emitError("special function must be a method");
    } else if (funcOp.getIsStatic()) {
      if (!(fnInfo.flags & SpecialFunctionInfo::kImplicitlyStaticMethod))
        emitError("special method may not be a static method");
    } else if (fnInfo.requiresOwnedSelfInstMethod()) {
      if (args[selfArgNumber].convention != ParsedArgument::kConventionOwned) {
        emitErrorLoc(args[selfArgNumber].loc, "self argument must be 'owned'")
            << FixIt::insertBeforeToken(args[selfArgNumber].loc, "owned ");
        args[selfArgNumber].convention = ParsedArgument::kConventionOwned;
      }
    }
  }

  ASTType declaredResultType =
      hasMemoryResult ? ASTType(argTypes[0]) : resultType;

  // Some functions like __new__ require a Self result type.
  if (fnInfo.flags & SpecialFunctionInfo::kSelfResult &&
      !declaredResultType.isEqualCanon(selfType))
    emitError() << name << " result type must be " << selfType;

  // If the function is required to return None, verify that.
  if (fnInfo.hasNoneResult() && !declaredResultType.isNoneType()) {
    emitError() << name << " result type must be elided (or None)";
    resultType = shared.getNoneType();
  }

  // Reject special functions declared as throwing when that is invalid.
  if (effects.isThrows() && fnInfo.flags & SpecialFunctionInfo::kCannotRaise) {
    // Specialize the error if raising is implicit because it was defined as a
    // def.
    if (funcOp.getIsDef()) {
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
    if (!resultType.mlirType.isSignlessInteger(1))
      emitError() << name << " result type must be __mlir_type.i1";
    break;
  case SpecialFunctionKind::kCopyInitReg:
    // Check that these are defined correctly.
    if (args[0].convention != ParsedArgument::kConventionBorrowed)
      emitErrorLoc(args[selfArgNumber].loc,
                   "self argument cannot be passed by reference");
    break;
  case SpecialFunctionKind::kInit:
  case SpecialFunctionKind::kCopyInit:
  case SpecialFunctionKind::kMoveInit: {
    // The first/self argument is syntactically declared as a by-ref argument,
    // but we need to change it to InitSelf since it is not initialized coming
    // in.
    assert(!args.empty() && "arg count already checked above");
    SMLoc selfArgLoc = args[0].loc;
    // __init__ methods must take their self argument by-ref syntactically.
    if (args[0].convention != ParsedArgument::kConventionInOut) {
      auto diag = emitErrorLoc(selfArgLoc, "'self' in struct ")
                  << name << " must be passed as mutable reference";
      if (args[0].convention == ParsedArgument::kConventionUnspec)
        diag << FixIt::insertBeforeToken(selfArgLoc, "inout ");
    }

    // Regardless force it to init_self so recovery follows the fix-it.
    args[0].convention = ParsedArgument::kConventionInitSelfResult;
    // We also force the passing kind of self to positional-only.
    args[0].kwArgHandling = ParsedArgument::KWArgHandling::kPositionalOnly;

    if (fnInfo.kind == SpecialFunctionKind::kCopyInit) {
      if (args[1].convention != ParsedArgument::kConventionBorrowed)
        emitErrorLoc(args[1].loc,
                     "existing value argument must be passed as borrowed");
    } else if (fnInfo.kind == SpecialFunctionKind::kMoveInit) {
      if (args[1].convention != ParsedArgument::kConventionOwned)
        emitErrorLoc(args[1].loc,
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
               StringRef baseName)
      : SharedStateUser(shared), decl(decl), sigDecl(sigDecl),
        funcOp(cast<LIT::FuncOp>(decl)), baseName(baseName) {}

  /// Apply a function signature decorator.
  LogicalResult apply(ExprNode *decorator, FnEffects &effects);

private:
  void applyAdaptive(const DeclRefNode &node);
  void applyMoveCapture(const CallNode &node);
  void applyLLVMMetadata(const CallNode &node);

  ASTDecl &decl;
  ASTDecl &sigDecl;
  LIT::FuncOp funcOp;
  StringRef baseName;
};
} // namespace

LogicalResult FnDecorators::apply(ExprNode *decorator, FnEffects &effects) {
  // Process all the decorators we know about.
  if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
    if (declRef->spelling == "export")
      applyExport(decorator->getLoc(), shared, decl, baseName, baseName,
                  funcOp);
    else if (declRef->spelling == "staticmethod")
      funcOp.setIsStatic(true);
    else if (declRef->spelling == "always_inline")
      funcOp.setInlineLevel(InlineLevel::Always);
    else if (declRef->spelling == "no_inline")
      funcOp.setInlineLevel(InlineLevel::Never);
    else if (declRef->spelling == "adaptive")
      applyAdaptive(*declRef);
    else if (declRef->spelling == "parameter")
      funcOp.setIsParametric(true);
    else if (declRef->spelling == "noncapturing")
      effects.setCapturing(false);
    else if (declRef->spelling == "closure")
      effects.setCapturing();
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
        applyMoveCapture(*callNode);
      else if (declRef->spelling == "__llvm_metadata")
        applyLLVMMetadata(*callNode);
      else
        return failure();
      return success();
    }
  }
  return failure();
}

void FnDecorators::applyAdaptive(const DeclRefNode &node) {
  if (funcOp.getIsAdaptive())
    emitError(node.getLoc(), "only one '@adaptive' decorator is allowed")
        << node.getRange();

  funcOp.setIsAdaptive(true);
}

void FnDecorators::applyMoveCapture(const CallNode &node) {
  // HACK(#16110): Need to implement proper capture list syntax rather than rely
  // on a special decorator.
  for (const Operand &operand : node.operands) {
    auto *declRef = dyn_cast<DeclRefNode>(operand.value);
    if (!declRef) {
      emitError(operand.getLoc(), "'@__move_capture' expected a declaration");
      continue;
    }

    LookupResult lookup = shared.lookupAndResolveDecl(
        declRef->spelling, declRef->getLoc(), *decl.getParentDecl(),
        /*searcInParentScopes=*/true);
    if (ArrayRef<ASTDecl *> decls = lookup.getIfSuccess(); !decls.empty()) {
      ExprEmitter emitter(shared, decl, EC_CaptureCopy);
      ValueDest dest(EC_CaptureCopy);
      std::optional<Capture> capture;
      if (emitter.emitDeclReference(declRef->spelling, decls, declRef, dest,
                                    capture) &&
          capture) {
        shared.addCaptureToScope(decl, decls.front(),
                                 Capture(capture->getValue(), /*isMove=*/true));
        continue;
      }
    }
    emitError(declRef->getLoc(), "cannot capture '")
        << declRef->spelling << "'";
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
    if (PValue attr = emitter.emitExprPValue(value.value, EC_Decorator))
      attrs.append(value.name, attr);
  }
  funcOp.setLLVMMetadataAttr(attrs.getDictionary(getContext()));
}

/// Given the lexical context of a function, return true if the default bit
/// for the function is capturing.
/// FIXME: The language modeling here is a mess. It needs more thought.
static bool isCapturingByDefault(LIT::FuncOp funcOp, StructDeclOp parent,
                                 ArrayRef<ParamDeclAttr> inputParamDecls,
                                 ArrayRef<ParamDeclAttr> resultParamDecls) {
  // Nested functions are capturing by default.
  if (funcOp->getParentOfType<LIT::FuncOp>())
    return true;
  // Any function that contains a capturing closure as a parameter is itself
  // capturing, include parent struct parameters.
  mlir::AttrTypeWalker walker;
  walker.addWalk([](SignatureType sig) {
    if (sig.isCapturing())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return llvm::any_of(
      llvm::concat<const ParamDeclAttr>(inputParamDecls, resultParamDecls,
                                        parent ? parent.getInputParams()
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
  auto unbound = LITSignatureType::prependParams(
      original, llvm::map_to_vector(captured, [](ParamDeclRefAttr ref) {
        return ParamDeclAttr::get(ref);
      }));
  return {std::move(captured), unbound};
}

static XRValue emitClosureInstance(SignatureType closureSignature,
                                   SharedState &shared, ASTDecl &nestedFnDecl,
                                   SMLoc loc) {
  LIT::FuncOp nestedFn = cast<LIT::FuncOp>(nestedFnDecl);
  auto parentFn = nestedFn->getParentOfType<LIT::FuncOp>();
  assert(parentFn && "expected nested function to have a parent FuncOp");

  // Save the insertion point before closure creation since closure creation
  // nukes the nested function.
  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockEnd(parentFn.getLoc(), parentFn.getBody());
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

  // Create an instance of the closure implementation in the parent function
  // right after the nested function definition.
  ClosureEmitter emitter(*moduleDecl, shared);
  StructDeclOp closureImpl =
      emitter.replaceNestedFunctionWithClosureImplStructDecl(
          loc, nestedFnDecl, paramCaptures, wrapperSig);

  // Map the closure wrapper captures to the impl captures.
  SmallDenseMap<unsigned, unsigned> fromImplToWrapperParameterMap;
  emitter.createWrapperInitWithImpl(closureWrapper, closureImpl,
                                    fromImplToWrapperParameterMap, loc);

  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (DebugInfo::DIScopeAttr spAttr = parentFn.getLocScope())
    diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);
  builder.restoreInsertionPoint(insertPoint);

  ExprEmitter exprEmitter(shared, *nestedFnDecl.getParentDecl(), builder);
  SyntheticNode node(loc);

  // Create a copy of the captured value.
  auto captureIteratorRange = shared.getCaptureRangeInScope(nestedFnDecl);
  SmallVector<ASTExprAnd<AnyValue>> closureImplInitArgs;
  for (auto &[_, capture] : captureIteratorRange) {
    AnyValue arg = capture.getValue();
    if (capture.isMoveCapture()) {
      // HACK(#16110): This transfers ownership without an explicit `^` from the
      // user, because we don't have capture list syntax.
      UnaryOpNode transfer(ExprNode::kTransfer, loc, &node);
      ValueDest dest(EC_CaptureCopy);
      arg = transfer.emitTransfer(arg, dest, exprEmitter);
    }
    closureImplInitArgs.push_back({arg, &node});
  }

  ValueDest closureDest;

  // Create Closure Impl type by adding captured parameters to the ClosureImpl
  // DeclType.
  Type closureImplType = closureImpl.bindReference(llvm::map_to_vector(
      paramCaptures, [](ParamDeclRefAttr ref) -> TypedAttr { return ref; }));
  CValue value = exprEmitter.emitConstructorCall(
      ASTType(closureImplType), closureImplInitArgs, &node,
      CallSyntax::kTypeCall, closureDest, /*allowImplicitConversion=*/false);
  // Emit the Closure Wrapper instance.
  ValueDest closureWrapperDest;
  SmallVector<ASTExprAnd<AnyValue>> closureWrapperInitArgs;
  closureWrapperInitArgs.push_back({value, &node});

  // Create the ClosureWrapper type by binding parent parameters to the
  // ClosureWrapper type.
  // TODO: Handle partial binding.
  DeclRefType closureWrapperType =
      closureWrapper.bindReference(llvm::map_to_vector(
          capturedRefs, [](ParamDeclRefAttr ref) -> TypedAttr { return ref; }));
  CValue closureWrapperInstance = exprEmitter.emitConstructorCall(
      ASTType(closureWrapperType), closureWrapperInitArgs, &node,
      CallSyntax::kTypeCall, closureWrapperDest,
      /*allowImplicitConversion=*/false);

  if (!closureWrapperInstance)
    return {};
  assert(closureWrapperInstance.getIfXRValue());
  return closureWrapperInstance.getIfXRValue();
}

PassingKind ParsedArgument::mapToPassingKind(KWArgHandling handling) {
  switch (handling) {
  case ParsedArgument::KWArgHandling::kPositionalOnly:
    return PassingKind::PosOnly;
  case ParsedArgument::KWArgHandling::kKeywordOnly:
    return PassingKind::KwOnly;
  case ParsedArgument::KWArgHandling::kPositionalOrKeyword:
    return PassingKind::PosOrKw;
  }
  llvm_unreachable("unhandled ParsedArgument::KWArgHandling");
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
  FnEffects effects;
  effects.setAsync(p.consumeIf(Token::kw_async));
  if (p.getToken().is(Token::kw_def))
    effects.setThrows();
  p.consumeToken();

  StringAttr baseName;
  SMLoc identifierLoc;
  if (p.parseIdentifier(baseName, "expected function name", &identifierLoc))
    return failure();

  // The function signature is a self-contained scope where the input and result
  // parameters of the function are visible by all types.
  ASTDecl &sigDecl = addFullyResolvedDecl(nullptr, StringAttr(), decl.getLoc(),
                                          decl.getParentDecl());

  // Add meta parameters from an enclosing declaration to the symbol table.
  // These are /in/ our current scope because we do not want name conflicts with
  // them and they are instance (not type-level) values.
  // TODO: Generalize this to support nested structs and functions.
  bool paramVarArg = false;
  auto structDecl = dyn_cast<StructDeclOp>(decl.getParentDecl());
  if (structDecl) {
    for (ParamDeclAttr param : structDecl.getInputParams()) {
      auto paramRef = ParamDeclRefAttr::get(param);
      addFullyResolvedDecl(PValue(paramRef), param.getName(), decl.getLoc(),
                           &sigDecl);
    }
    paramVarArg = structDecl.getSignature().getParamVarArg();
  }

  // Parse declared meta parameters and add them to the current scope.
  SmallVector<ParamDeclAttr> inputParamDecls, resultParamDecls;
  SmallVector<ParsedArgument> args;
  SmallVector<StringAttr> paramNames;
  SmallVector<PassingKind> paramPassingKinds;
  SmallVector<TypedAttr> paramDefaults;

  // Add the meta parameters to the symbol table, and resolve their types.  We
  // add all of these after generic signature parsing so types used in the
  // signature list resolve to enclosing scopes, and we add them before the
  // value signature list so the types and parameters can resolve to the bound
  // values.
  if (impl::parseOptionalParameterSignature(
          p, sigDecl, inputParamDecls, resultParamDecls, paramNames,
          paramPassingKinds, paramDefaults, paramVarArg))
    return failure();

  // Parse the argument list next if present.
  if (ParsedArgument::parseAndResolveParenthesizedArgumentList(
          p, args, ParsedArgument::ArgListKind::kArgList, effects))
    return failure();

  if (paramVarArg)
    effects.setParamVarArgs();

  // This doesn't support the capturing effect, reject it.
  if (effects.isCapturing()) {
    emitError(decl.getLoc(),
              "'capturing' effect not supported on this declaration");
    effects.setCapturing(false);
  }

  // Parse the result type if present.
  ExprNode *resultTypeExpr = nullptr;
  SMLoc resultLoc = p.getToken().getLoc();
  if (p.consumeIf(Token::minus_greater)) {
    if (p.parseExpression(resultTypeExpr))
      return failure();
  }

  // Check for a missing colon now, but don't yet bail out. We want to be able
  // to diagnose errors in the signature before we bail out, users often haven't
  // finished writing the signature when they hit the missing colon.
  ParseResult missingColon =
      p.parseToken(Token::colon, "expected ':' in function definition");

  // Emit the argument and result types.
  SmallVector<Type> argTypes;
  SmallVector<TypedAttr> argDefaults;
  auto reportError = [&] {
    decl.hasReferenceError = true;
    return success();
  };
  SpecialFunctionInfo fnInfo = SpecialFunctionInfo::get(baseName);

  // Process signature decorators in the same scope as signature resolution.
  auto processSignature = [&] {
    if (isCapturingByDefault(funcOp, structDecl, inputParamDecls,
                             resultParamDecls) &&
        !effects.isEscaping())
      effects.setCapturing();

    // Now that we have figured out the lexical structure, allow decorators to
    // take a crack at the signature.
    FnDecorators fnDecorators(decl, sigDecl, shared, baseName);
    Decorators(decl, shared)
        .applySignatureDecorators(decoratorExprs, [&](ExprNode *decorator) {
          return fnDecorators.apply(decorator, effects);
        });
  };

  ExprEmitter typeEmitter(shared, sigDecl, EC_Type);
  ASTType resultType = ParsedArgument::emitFunctionArgumentsAndResults(
      reportError, typeEmitter, paramNames, paramPassingKinds, inputParamDecls,
      resultTypeExpr, effects, args, argTypes, argDefaults, funcOp.getIsDef(),
      resultLoc, &decl, fnInfo, processSignature);
  if (!resultType)
    return failure();

  // Propagate errors and the parsed decls in the signature.
  decl.takeDecls(sigDecl);

  // Now that all the structural properties are determined, perform any
  // name-binding specific checks over the declaration.  This happens after
  // decorator processing because that is how defs work in Python.  This also
  // fills in any implicitly declared types.
  verifyFunctionNameBinding(decl, funcOp, baseName, args, argTypes, resultType,
                            effects, shared, fnInfo);

  // Now that all the types and signature information have been resolved,
  // compute the final MLIR types and KGEN conventions.  This also introduces
  // implicit lifetime parameters for borrows/inout/owned arguments.
  SmallVector<ParamDeclAttr> implicitLifetimeDecls;
  computeArgumentConventions(shared, args, argTypes, implicitLifetimeDecls,
                             decl);

  // Now that we've processed the signature, bail if we had a missing colon.
  if (missingColon)
    return failure();

  // Finally now that the full signature has been resolved, build our IR.

  // First, handle function effects. If the function raises, it implicitly gets
  // a variant result type.
  if (effects.isThrows()) {
    ASTType errorType =
        shared.getBuiltinErrorType(*decl.getParentDecl(), decl.getLoc());
    if (errorType.isTypeCheckErrorType())
      decl.hasReferenceError = true;

    resultType = VariantType::get({errorType, resultType},
                                  AnyRegTypeType::get(getContext()));
  }

  // Handle argument effects and build the ASTDecls for the arguments.
  SmallVector<Location> argLocs;
  SmallVector<StringAttr> argNames;
  SmallVector<PassingKind> argPassingKinds;
  SmallVector<ValueInputConvention> inputConventions;
  for (const ParsedArgument &arg : args) {
    argLocs.push_back(shared.diags.translateLocation(arg.loc));
    argPassingKinds.emplace_back(
        ParsedArgument::mapToPassingKind(arg.kwArgHandling));
    argNames.push_back(arg.name);
    inputConventions.push_back(arg.kgenConvention);
  }

  OpBuilder builder = decl.getDeclEndBuilder();
  NamedAttrList attrs = funcOp->getAttrDictionary();
  auto inputParamsAttr = builder.getAttr<ParamDeclArrayAttr>(inputParamDecls);
  auto resultParamsAttr = builder.getAttr<ParamDeclArrayAttr>(resultParamDecls);

  // Compute the signature of the function.
  FunctionType functionType =
      builder.getFunctionType(argTypes, {resultType.mlirType});
  auto metadata =
      FnMetadataAttr::get(builder.getContext(), argNames, argPassingKinds,
                          paramNames, paramPassingKinds, argDefaults,
                          paramDefaults, implicitLifetimeDecls.size());
  LITSignatureType signature = SignatureType::remapToSignature(
      inputParamsAttr, resultParamsAttr, functionType, inputConventions,
      effects, metadata, silenceErrors(getContext()));
  if (!signature)
    return failure();

  // FIXME(#26008): Async functions with memory-only do not work.
  if (signature.isAsync() && signature.hasMemoryOnlyResult()) {
    return emitError(
        resultTypeExpr ? resultTypeExpr->getLoc() : decl.getLoc(),
        "TODO: async functions do not support memory-only results yet");
  }

  // The implicitLifetimeDecls don't affect the signature, but they do get
  // prepended onto the inputParamDecls list.
  if (!implicitLifetimeDecls.empty()) {
    SmallVector<ParamDeclAttr> mergedInputParams(implicitLifetimeDecls.begin(),
                                                 implicitLifetimeDecls.end());
    mergedInputParams.append(inputParamDecls.begin(), inputParamDecls.end());
    inputParamsAttr = builder.getAttr<ParamDeclArrayAttr>(mergedInputParams);
  }

  attrs.set(funcOp.getInputParamsAttrName(), inputParamsAttr);
  attrs.set(funcOp.getResultParamsAttrName(), resultParamsAttr);
  attrs.set(funcOp.getFunctionTypeAttrName(), TypeAttr::get(functionType));

  // Now that the FunctionType is set to the pretty type that includes implicit
  // lifetimes, we strip off the named lifetime decl references and replace them
  // with indices.
  signature =
      signature.replaceImplicitLifetimesWithIndexes(implicitLifetimeDecls);
  attrs.set(funcOp.getSignatureAttrName(), TypeAttr::get(signature));

  // Set the symbol to the mangled name and check for redefinition.
  attrs.set(funcOp.getSymNameAttrName(), getMangledName(baseName, signature));
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
    if (!resTy.isEqualCanon(existingResTy)) {
      errorMessage = " cannot overload on return type only";
    } else if (!existingFunc.getIsAdaptive()) {
      // If the results match, we only error if the function is not adaptive.
      errorMessage = " with identical signature";
    }

    // On redefinition this is an overload of the same name.
    if (errorMessage) {
      auto diag = p.emitError(funcOp.getLoc(), "redefinition of function ")
                  << baseName << errorMessage;
      diag.attachNote(existing->getLoc()) << "previous definition here";
      decl.hasReferenceError = true;
    }
  }

  // If have a main function, fn main(), export it automatically.
  if (!structDecl && baseName == kMainSymbolName)
    getDeclResolver().exportMain(decl);

  // Generate a debug subprogram for this function.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  shared.setLocationDebugScope(diScopeGuard, funcOp);

  // If this is a nested function, set its parameter declaration. It will be
  // referenced via parameter references instead of symbol references.
  if (funcOp->getParentOfType<LIT::FuncOp>())
    funcOp.setParamDeclAttr(
        ParamDeclAttr::get(funcOp.getSymNameAttr(), signature));

  funcOp.getBody()->addArguments(argTypes, argLocs);

  if (!funcOp->getParentOfType<FuncOp>() ||
      (!signature.isCapturing() && !signature.isEscaping()))
    funcOp.setIsParametric(true);

  // Upon fully resolving a nonparametric closure, immediately materialize it
  // as a runtime value. It cannot be used as a parameter.
  if (!funcOp.getIsParametric()) {
    // Fully resolve the body so we can swap the IR value of the decl. Later on,
    // we will need this to determine the capture signature.
    decl.resolvedness = DeclResolvedness::fully;
    if (failed(resolveBody(funcOp, lexer, decl)))
      return failure();

    // If the function doesn't actually capture anything, don't demote it to a
    // runtime value.
    bool hasCapture = false;
    mlir::visitUsedValuesDefinedAbove(funcOp.getBodyRegion(),
                                      [&](OpOperand *) { hasCapture = true; });
    if (hasCapture || signature.isEscaping()) {
      if (funcOp.getIsAdaptive()) {
        decl.hasReferenceError = true;
        return emitError(
            funcOp.getLoc(),
            "nonparametric capturing closure cannot be marked @adaptive");
      }
      if (!signature.isEscaping() &&
          (!inputParamDecls.empty() || !resultParamDecls.empty())) {
        return emitError(funcOp.getLoc(),
                         "nonparametric capturing closure cannot have input or "
                         "result parameters");
      }

      OpBuilder b(funcOp.getContext());
      b.setInsertionPointAfter(funcOp);
      auto parent = funcOp->getParentOfType<LIT::FuncOp>();
      if (!parent)
        return failure();
      // Emit Closure structures necessary for instantiating an escaping
      // closure.
      if (signature.isEscaping()) {
        if (!inputParamDecls.empty() || !resultParamDecls.empty())
          return emitError(
              funcOp.getLoc(),
              "escaping closures cannot have input or result parameters yet");
        if (auto closure =
                emitClosureInstance(signature, shared, decl, decl.getLoc())) {
          decl.irValue = XRValue(closure);
        } else
          return failure();
      } else {
        decl.irValue = SBValue(b.create<CreateClosureOp>(
            parent.getLoc(), signature,
            ParamDeclRefAttr::get(*funcOp.getParamDecl()), ValueRange()));
      }
    }
  }

  shared.notifyListenerOnFunctionDecl(decl, identifierLoc);
  return success();
}

static LetRegDeclOp makeVarArgLValueVarSlot(const CValue &argValue,
                                            StringAttr argName,
                                            ASTDecl &parentDecl,
                                            OpBuilder &builder, SMLoc loc,
                                            SharedState &shared) {
  Location mloc = shared.translateLocation(loc);
  ASTType varListType = shared.getBuiltinVariadicListInstantiation(
      parentDecl, loc, argValue.getRValueType().getVariadicElementType());

  // Emit the initializer expression into the slot.
  ExprEmitter emitter(shared, parentDecl, builder);

  // Expr to provide location information.
  DeclRefNode srcExpr(StringRef(loc.getPointer(), argName.size()));
  SRValue val = emitter.emitSRValue({argValue, &srcExpr}, EC_DefArgumentShadow,
                                    varListType);
  return builder.create<LetRegDeclOp>(mloc, argName, val);
}

/// Create a mutable VarDecl for a function argument that captures its value.
/// argValue specifies the argument with the correct valuetype.
static VarLetDeclOp makeArgLValueVarSlot(const CValue &argValue,
                                         StringAttr argName,
                                         ASTDecl &parentDecl,
                                         OpBuilder &builder, SMLoc loc,
                                         SharedState &shared) {
  Location mloc = shared.translateLocation(loc);

  // Emit the initializer expression into the slot.
  ExprEmitter emitter(shared, parentDecl, builder);
  VarLetDeclOp varDecl =
      emitter.emitVarLetDecl(argName, argValue.getRValueType(), mloc,
                             VarLetDeclKind::Implicit, /*isSynthetic=*/false);

  // Expr to provide location information.
  DeclRefNode srcExpr(StringRef(loc.getPointer(), argName.size()));
  ValueDest dest(XLValue(varDecl), EC_DefArgumentShadow);
  if (!emitter.emitBValue({argValue, &srcExpr}, dest))
    dest.resetForError();

  return varDecl;
};

/// This adds a default return (lit.return of None, potentially converted
/// to a variant) and emits a EndFuncOp.
static void appendDefaultReturnAndEndOp(LIT::FuncOp func, ASTDecl &funcDecl,
                                        SharedState &shared) {
  Block &body = *func.getBody();
  auto b = ImplicitLocOpBuilder::atBlockEnd(func.getLoc(), &body);

  auto makeNoneReturn = [&] {
    // The function returns none.
    Value retVal = b.create<ParamConstantOp>(shared.getNoneAttr());

    // Wrap the result value if necessary.
    if (func.isThrows())
      retVal = b.create<VariantCreateOp>(func.getMLIRResultType(), retVal, 1);
    ExprEmitter::emitNormalReturn(b, retVal, funcDecl);
  };

  // If the function returns None, insert a "return None".
  ASTType normalResult = func.getUserResultType();
  if (normalResult.isNoneType() &&
      // No default return needed if we ended in a return.
      (body.empty() || !isa<LIT::ReturnOp>(body.back()))) {
    makeNoneReturn();
  } else if (func.getIsDef() && func.getSignature().hasMemoryOnlyResult()) {
    // If this `def` returns an object but is missing a return, insert one
    // automatically.
    auto objType = shared.lookupObjectType(funcDecl.getLoc(), funcDecl);
    if (objType &&
        objType.isEqualCanon(
            cast<RefType>(func.getArgument(0).getType()).getElementType())) {
      // Emit `object()` into the memory type return slot.
      ExprEmitter emitter(shared, funcDecl, EC_ReturnValue);
      emitter.builder = b;
      ValueDest resultDest(XLValue(func.getArgument(0)), EC_ReturnValue);
      // Create a dummy node to pass down.
      SyntheticNode locExpr(funcDecl.getLoc());
      CValue result = emitter.emitConstructorCall(
          objType, {}, &locExpr, CallSyntax::kImplicitConvert, resultDest);
      if (!result || !emitter.emitResult(result, &locExpr, resultDest))
        resultDest.resetForError();
      else
        makeNoneReturn();
    }
  }

  // Insert the default end terminator.
  b.create<LIT::EndFuncOp>();
}

ParseResult DeclResolver::resolveBody(LIT::FuncOp funcOp, Lexer &lexer,
                                      ASTDecl &decl) {
  // Push the debug scope for this function if necessary so that nested
  // operations have proper debug info.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (DebugInfo::DIScopeAttr spAttr = funcOp.getLocScope())
    diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

  // Set up information about value arguments.
  Block *bodyBlock = funcOp.getBody();
  auto builder = OpBuilder::atBlockEnd(bodyBlock);

  LITSignatureType funcSignature = funcOp.getSignature();

  // Set up the body of the fn/def, creating declarations for the value
  // parameters and adding them to the symbol table.
  for (auto [argName, bbArg, convention] :
       llvm::zip(funcSignature.getArgNames(), funcOp.getBody()->getArguments(),
                 funcSignature.getInputConventions())) {
    // Don't bind byref-result, it is handled specially by 'return'.
    if (convention == ValueInputConvention::ByRefResult)
      continue;

    // Figure out which decl corresponds to this argument so we can finish it.
    ArrayRef<ASTDecl *> argDeclList = decl.lookupInCurrentScope(argName);
    assert(argDeclList.size() == 1 &&
           "Argument should be added by signature resolution");
    ASTDecl &argDecl = *argDeclList[0];

    // This function sets the argument decl to be fully resolved with the
    // specified IR representation.
    auto setDecl = [&](DeclIRValue value) -> LogicalResult {
      argDecl.setIRValue(value);
      argDecl.resolvedness = DeclResolvedness::fully;
      if (auto rv = argDecl.getIfRValue()) {
        if (rv.getType().isTypeCheckErrorType())
          argDecl.hasReferenceError = true;
      } else if (auto lv = argDecl.getIfMLValue()) {
        if (lv.getRValueType().isTypeCheckErrorType())
          argDecl.hasReferenceError = true;
      } else if (auto lv = argDecl.getIfXLValue()) {
        if (lv.getRValueType().isTypeCheckErrorType())
          argDecl.hasReferenceError = true;
      } else if (auto bv = argDecl.getIfBValue()) {
        if (bv.getRValueType().isTypeCheckErrorType())
          argDecl.hasReferenceError = true;
      }
      shared.notifyListenerOnArgumentDecl(argDecl, argDecl.getLoc());
      return success();
    };

    shared.buildArgDebugInfo(builder, bbArg, argName);

    // VarArg arguments are projected into a VariadicList.
    if (funcSignature.isVarArg(bbArg.getArgNumber())) {
      auto declOp = makeVarArgLValueVarSlot(SRValue(bbArg), argName, decl,
                                            builder, argDecl.getLoc(), shared);
      if (failed(setDecl(DeclIRValue(declOp))))
        return failure();
      continue;
    }

    // PackVarArg arguments are always treated as their kgen.pack type
    // by-value right now.  TODO(literals): Project to a tuple like thing.
    if (isa<PackType>(bbArg.getType())) {
      if (failed(setDecl(SRValue(bbArg))))
        return failure();
      continue;
    }

    DeclIRValue argIRValue;
    switch (convention) {
    // Arguments passed by-reference can be directly used.
    case ValueInputConvention::ByRef:
      argIRValue = MLValue(bbArg);
      break;
    case ValueInputConvention::InitSelf:
    case ValueInputConvention::ByRefResult:
    case ValueInputConvention::OwnedInMem:
      // OwnedInMem passes ownership of the argument into the callee so we
      // can directly mutate it if we want to.
      argIRValue = XLValue(bbArg);
      break;
    case ValueInputConvention::OwnedInReg:
      argIRValue = makeArgLValueVarSlot(SRValue(bbArg), argName, decl, builder,
                                        argDecl.getLoc(), shared);
      break;

    case ValueInputConvention::BorrowedInReg:
      argIRValue = SBValue(bbArg);
      break;

    case ValueInputConvention::BorrowedInMem:
      argIRValue = MBValue(bbArg);
      break;
    case ValueInputConvention::None:
      llvm_unreachable("none convention not permitted in lit");
    }

    // Ok, now that we've figured out the IR representation of the ASTDecl,
    // install it.
    if (failed(setDecl(argIRValue)))
      return failure();
  }

  Block *body = funcOp.getBody();

  Operation *lastOpIterBefore =
      body->empty() ? nullptr : &body->getOperations().back();

  // With all the argument declarations set up, we can resolve the body of the
  // function.
  if (ParserBase(shared, lexer).parseSuite(decl))
    return failure();

  // Function body is empty if the body block is empty or the last operation in
  // the block is still the same as it was before parseSuite.
  bool emptyBody =
      body->empty() || (lastOpIterBefore == &body->getOperations().back());

  auto loc = funcOp.getLoc();

  // Create a placeholder result bind op if the function has result parameters.
  ArrayRef<ParamDeclAttr> resultParams = funcOp.getResultParams();
  if (!resultParams.empty()) {
    SmallVector<TypedAttr> placeholders;
    for (ParamDeclAttr decl : resultParams)
      placeholders.push_back(UnknownAttr::get(decl.getType()));
    builder.create<ParamResultBindOp>(loc, placeholders);
  }

  // Emit a default "return None" if the function returns nothing, and add an
  // endop terminator.

  if (emptyBody && isa<TraitDeclOp>(*decl.getParentDecl())) {
    // Wipe out the body which may already contain some compiler generated
    // operations for handling argLValueVarSlot.
    body->walk([&](LIT::VarLetDeclOp op) {
      // Remove the value from parent's declsInScope first before destroying the
      // value.
      auto iter = decl.declsInScope.find(op.getNameAttr());
      if (iter != decl.declsInScope.end())
        iter->second.clear();
    });

    body->clear();
    // Don't append anything to an empty function if this is a trait function.
  } else {
    appendDefaultReturnAndEndOp(funcOp, decl, shared);
  }

  // Now that the body of the function is parsed, run any body decorators.
  Decorators(decl, shared).applyBodyDecorators([](ExprNode *decorator) {
    return failure();
  });

  // Check that any alias forward declarations have been completed.
  if (!shared.diags.isErrorEmitted()) {
    bodyBlock->walk([&](AliasForwardDeclOp aliasFwdDeclOp) {
      // If the location for the resultParam was never set then this forward
      // declaration was never defined.
      if (!aliasFwdDeclOp.getResultParamLoc().has_value()) {
        emitError(aliasFwdDeclOp.getLoc(), "alias ")
            << aliasFwdDeclOp.getNameAttr()
            << " was never defined by a result parameter";
      }
    });
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
    if (ec || !SharedState::isModuleOrPackagePath(entry.path()))
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
    StringAttr boundName = builder.getStringAttr("$" + name);
    auto importDecl = builder.create<LIT::UnresolvedImportOp>(
        op->getLoc(), importName, boundName, /*declName=*/StringAttr(),
        /*importNameLoc=*/mlir::LocationAttr(),
        /*destNameLoc=*/mlir::LocationAttr());
    getDeclResolver().addDecl(importDecl, decl.loc, boundName, &decl,
                              LexerCursor(), LexerCursor(), /*indentation=*/-1);

    // Create an alias for the unmangled module name to allow for simplified
    // indexing into this module.
    boundName = builder.getStringAttr(name);
    importDecl = builder.create<LIT::UnresolvedImportOp>(
        op->getLoc(), importName, boundName, /*declName=*/StringAttr(),
        /*importNameLoc=*/mlir::LocationAttr(),
        /*declNameLoc=*/mlir::LocationAttr());
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
// VarLetDecl implementation
//===----------------------------------------------------------------------===//

ParseResult DeclResolver::resolveBody(LetRegDeclOp op, Lexer &lexer,
                                      ASTDecl &decl) {
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
  if (!p.consumeIf(Token::kw_var) && !p.consumeIf(Token::kw_let)) {
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
      p.parseVarLetInitExpression(initExpr, decl.getIndentation()))
    return failure();

  // Emit the initializer into an initializer function. If we have a type, then
  // emit directly into the LValue. Otherwise emit into the global to infer its
  // type.
  ValueDest dest;
  ExprContext exprContext = op.getIsVar() ? EC_VarInit : EC_LetInit;
  if (parsedType) {
    op.setType(parsedType);
    DLValue result(
        RCRef<GlobalDLValue>::create(op, parsedType, initExpr->getLoc()));
    dest = ValueDest(result, exprContext);
  } else {
    // If we don't, we emit into the varOp itself, because this will infer the
    // type of the varOp from the initializer expression.
    dest = ValueDest(op, exprContext);
  }

  op.getCtor().push_back(new Block);
  emitter.builder = OpBuilder::atBlockBegin(&op.getCtor().front());
  if (!emitter.emitExpr(initExpr, dest))
    return failure();

  assert(!isa<UnresolvedType>(op.getType()) &&
         "RValue emission should have inferred var type");

  // Emit the destructor call, if present, into the destructor function.
  op.getDtor().push_back(new Block);
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
  SMLoc identifierLoc;

  // Parse the type if present.
  if (p.parseToken(Token::kw_alias, "internal error: checked by stmt parser") ||
      p.parseIdentifier("internal error: checked by stmt parser",
                        &identifierLoc))
    return failure();

  ASTType type;
  if (p.consumeIf(Token::colon)) {
    if (parseType(p, type, *decl.getParentDecl(), decl.getIndentation()))
      return failure();
  }

  // Handle the case where there is no initializer.
  if (!p.consumeIf(Token::equal)) {
    // If there was neither a type or initializer, reject the var.
    if (!type) {
      p.emitError(aliasDeclOp.getLoc(),
                  "declaration must have either a type or an initializer");
      return failure();
    }

    // `alias x: Int` is a forward declaration of a return parameter from a
    // function call, so it must occur in a function.
    if (!aliasDeclOp->getParentOfType<LIT::FuncOp>()) {
      p.emitError(aliasDeclOp.getLoc(),
                  "parameter results may only be declared in a function");
      return failure();
    }

    // Ok, things seem set up right, replace the ParamDeclOp with the right
    // operation that will allow us to track things.
    OpBuilder builder(aliasDeclOp);
    Operation *forwardDecl = builder.create<AliasForwardDeclOp>(
        aliasDeclOp.getLoc(), aliasDeclOp.getName(), TypeAttr::get(type),
        mlir::LocationAttr(), DocStringAttr());
    decl.setIRValue(forwardDecl);

    // Remove the paramDeclOp from the IR, since we ended up changing our mind
    // about how to represent this.
    aliasDeclOp->erase();

    // The check that the alias was specified is handled when the function body
    // has been fully resolved.
    rejectDecorators(decoratorExprs, decl, shared);

    // Process the doc string of the alias.
    p.parseDocString(decl);

    shared.notifyListenerOnAliasDecl(decl, identifierLoc);
    return success();
  }

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
  rejectDecorators(decoratorExprs, decl, shared);

  // Process the doc string of the alias.
  p.parseDocString(decl);

  shared.notifyListenerOnAliasDecl(decl, identifierLoc);
  return success();
}

ParseResult DeclResolver::resolveBody(AliasDeclOp op, Lexer &lexer,
                                      ASTDecl &decl) {
  return success();
}

ParseResult DeclResolver::resolveBody(AliasForwardDeclOp aliasFwdDeclOp,
                                      Lexer &lexer, ASTDecl &decl) {
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
      if (isa<DeclRefType>(type)) {
        p.emitError(loc)
            << "TODO: inheriting from other structs is not implemented";
      } else if (isa<ParamRefType>(type)) {
        p.emitError(loc) << "TODO: inheriting from a parameter expression is "
                            "not implemented";
      } else {
        p.emitError(loc) << "don't know how to inherit from this type";
      }
      declScope.hasReferenceError = true;
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
          callNode->operands.size() == 1)
        if (auto drn = dyn_cast<DeclRefNode>(callNode->operands[0].value))
          if (auto parentDecl = structDecl.getParentDecl()) {
            ExprEmitter emitter(shared, *parentDecl, EC_Type);
            if (ASTType t = emitter.emitExprType(drn)) {
              structOp.setNonmaterializableTargetAttr(
                  TypeAttr::get(t.mlirType));
              return success();
            }
          }
    }
  }
  // Not handled in signature phase.
  return failure();
}

/// structdef ::=
///   [decorators] "struct" identifier [param_signature] ":" suite
///
LogicalResult DeclResolver::resolveSignature(StructDeclOp structOp,
                                             Lexer &lexer, ASTDecl &decl) {
  ParserBase p(shared, lexer);
  auto decoratorExprs = p.parseDecorators(decl);

  // The signature of a struct is a self-contained decl where types can
  // reference the struct parameters.
  ASTDecl &sigDecl = addFullyResolvedDecl(nullptr, StringAttr(), decl.getLoc(),
                                          decl.getParentDecl());

  SmallVector<ParamDeclAttr> inputParamDecls, resultParamDecls;
  SmallVector<StringAttr> paramNames;
  SmallVector<PassingKind> paramPassingKinds;
  SmallVector<TypedAttr> paramDefaults;
  SmallVector<TypeLineageAttr> parentTypes;

  bool paramVarArgs = false;
  SMLoc identifierLoc;
  if (p.parseToken(Token::kw_struct,
                   "internal error: checked by stmt parser") ||
      p.parseIdentifier("internal error: checked by stmt parser",
                        &identifierLoc) ||
      impl::parseOptionalParameterSignature(
          p, sigDecl, inputParamDecls, resultParamDecls, paramNames,
          paramPassingKinds, paramDefaults, paramVarArgs) ||
      parseOptionalParentList(p, sigDecl, structOp.getSymName(), parentTypes,
                              shared) ||
      p.parseToken(Token::colon, "expected ':' in struct definition") ||
      decl.hasReferenceError)
    return failure();

  // Propagate signature errors and decls.
  decl.takeDecls(sigDecl);

  auto inputParams = ParamDeclArrayAttr::get(getContext(), inputParamDecls);
  structOp.setInputParamsAttr(inputParams);
  auto sig = TypeSignatureType::remapToSignature(
      silenceErrors(getContext()), inputParams, paramNames, paramPassingKinds,
      paramDefaults, paramVarArgs);
  if (!sig)
    return failure();
  structOp.setSignature(sig);
  structOp.setParentTypes(parentTypes);

  // Reject result parameters.
  if (!resultParamDecls.empty())
    emitError(decl.getLoc(),
              "struct declarations do not support result parameters");

  // This is a struct, so we can use 'computeSelfTypeForStruct' to figure out
  // the self type.
  decl.setSelfType(ASTDecl::computeSelfTypeForStruct(structOp));

  // Structs are memory-only unless they opt-in to being passed in registers.
  structOp.setConvention(TypeConvention::MemoryOnly);

  // Now that we have the basic struct set up, process signature decorators.
  Decorators(decl, shared)
      .applySignatureDecorators(decoratorExprs, [&](ExprNode *decorator) {
        return processStructSignatureDecorator(decorator, structOp, shared,
                                               decl);
      });

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

/// Look up a __copyinit__/__moveinit__  impl for the specified `type`.  This
/// returns the method if successful, and returns null if there is none.
static SymbolConstantAttr lookupCopyMoveInit(ASTDecl &structDecl,
                                             SharedState &shared,
                                             SpecialFunctionKind specialKind) {
  const char *name = SpecialFunctionInfo::get(specialKind).name;
  LookupResult inits = shared.lookupAndResolveDecl(
      name, structDecl.getLoc(), structDecl, /*searchParentScopes=*/false);
  ArrayRef<ASTDecl *> entries = inits.getIfSuccess();
  for (ASTDecl *candidate : entries) {
    LIT::FuncOp func = dyn_cast<LIT::FuncOp>(candidate);
    if (func && func.getSpecialFunctionKind() == specialKind)
      return func.getBoundSymbolRef();
  }
  return {};
}

/// Given a struct that has no explicitly defined __del__ member, define a new
/// one with an empty body.  This allows the CheckLifetimes pass to insert field
/// dels as needed, and makes sure that anything that refers to this struct
/// properly runs its destructor.
static SymbolConstantAttr synthesizeEmptyDtor(SharedState &shared,
                                              StructDeclOp structOp,
                                              ASTDecl &structDecl,
                                              DeclResolver &resolver) {
  auto builder = ImplicitLocOpBuilder::atBlockEnd(
      structOp.getLoc(), &structOp.getFields().front());

  // Figure out the type of the 'self' argument.  It is the struct's `Self`
  // type for register passable things, or indirect for a memory-only type.
  ASTType selfType = structDecl.getSelfType();
  // The argument is always owned.
  ValueInputConvention convention = ValueInputConvention::OwnedInReg;
  if (!selfType.isRegisterPassable(structDecl.getLoc(), resolver.shared)) {
    selfType = selfType.getRefForArgument("self", /*isMut*/ true);
    convention = ValueInputConvention::OwnedInMem;
  }

  StringAttr selfName = builder.getStringAttr("self");

  // Create the FuncOp and ASTDecl for the method.
  StructEmitter emitter(shared);
  auto [funcOp, funcDecl] = emitter.synthesizeMethodInStruct(
      "__del__", selfType.mlirType, convention, selfName, PassingKind::PosOnly,
      shared.getNoneType(), structDecl, SpecialFunctionKind::kDel);

  // Set up the body.
  Block *body = funcOp.getBody();
  BlockArgument arg = body->getArgument(0);

  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (DebugInfo::DIScopeAttr spAttr = funcOp.getLocScope())
    diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

  // We need to make a var box + store for register_passable values since that
  // is what lifetime tracking expects.  It does not track the individual fields
  // of register passable values since they cannot be transfered and cannot be
  // lit.ownership.mark_destroyed.
  if (convention == ValueInputConvention::OwnedInReg) {
    builder.setInsertionPointToStart(body);
    (void)makeArgLValueVarSlot(SRValue(arg), selfName, funcDecl, builder,
                               structDecl.getLoc(), resolver.shared);
  }

  // Finish off the function with a return + lit.endfunc.
  appendDefaultReturnAndEndOp(funcOp, funcDecl, resolver.shared);

  return funcOp.getBoundSymbolRef();
}

struct StructBodyDecorators : public SharedStateUser {
  StructBodyDecorators(
      StructDeclOp structOp, ASTDecl &structDecl, DeclResolver &resolver,
      ArrayRef<std::pair<StructFieldOp, ASTDecl *>> structFields)
      : SharedStateUser(resolver.shared), structOp(structOp),
        structDecl(structDecl), resolver(resolver), structFields(structFields) {
  }

  LogicalResult processDecorator(ExprNode *decorator);

private:
  void processValueDecorator(SMLoc decoratorLoc);
  void processRegisterPassableDecorator(bool isTrivial);

  StructDeclOp structOp;
  ASTDecl &structDecl;
  DeclResolver &resolver;
  ArrayRef<std::pair<StructFieldOp, ASTDecl *>> structFields;
};

/// Process the @value body decorator on structs.  This synthesizes the
/// memberwise init, copy ctor and move ctor if requested.
void StructBodyDecorators::processValueDecorator(SMLoc decoratorLoc) {
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
    return;
  }
  if (LIT::FuncOp copyCtr = stubs->getCopyConstructor()) {
    SymbolConstantAttr ref = copyCtr.getBoundSymbolRef();
    ASTDecl *copyCtrDecl =
        getDeclResolver().getDeclForFuncSymbol(ref.getSymbol());
    if (failed(structEmitter.populateMoveCopy(*copyCtrDecl, /*isMove=*/false)))
      copyCtr.erase();
    else
      declOp.setCopyInitAttr(ref);
  }
  if (LIT::FuncOp moveCtr = stubs->getMoveConstructor()) {
    SymbolConstantAttr ref = moveCtr.getBoundSymbolRef();
    ASTDecl *moveCtrDecl =
        getDeclResolver().getDeclForFuncSymbol(ref.getSymbol());
    if (failed(structEmitter.populateMoveCopy(*moveCtrDecl, /*isMove=*/true)))
      moveCtr.erase();
    else
      declOp.setMoveInitAttr(ref);
  }
}

LogicalResult StructBodyDecorators::processDecorator(ExprNode *decorator) {
  if (auto declRef = dyn_cast<DeclRefNode>(decorator)) {
    if (declRef->spelling == "value") {
      processValueDecorator(decorator->getRangeStart());
      return success();
    }
    return failure();
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
      fieldDecl->getParentDecl()->hasReferenceError = true;
      return;
    }
  }

  // Trivial types may not have __copyinit__ or __del__ members.
  if (isTrivial) {
    auto rejectMemberIfPresent = [&](StringRef name) {
      auto members = structDecl.lookupInCurrentScope(name);
      if (!members.empty())
        resolver.emitError(members[0]->getLoc())
            << "'@register_passable(\"trivial\")' types may not have a '"
            << name << "' method";
    };

    rejectMemberIfPresent("__copyinit__");
    rejectMemberIfPresent("__del__");
  }
}

//===----------------------------------------------------------------------===//
// Trait Conformance Checking

/// Get specialized signature of a trait function with a struct (who implements
/// the trait) type. Also return parameter bindings for specializing the
/// expected struct method with the current struct type.
static std::pair<LITSignatureType, InputParamBindings>
getTraitFunctionSignature(ExprEmitter &emitter, LIT::FuncOp traitFn,
                          ASTType structSelfType) {
  LITSignatureType signature = traitFn.getFullSignature();
  SmallVector<TypedAttr> params;
  ArrayRef<Type> inputParamTypes = signature.getInputParamTypes();

  // Add trait's MT replacement.
  // FIXME(generics): We aren't propagating metatypes into pointer types, so
  // just pass a generic metatype here.
  auto anyRegTypeType = AnyRegTypeType::get(traitFn.getContext());
  params.push_back(TypeConstantAttr::get(anyRegTypeType, anyRegTypeType));
  // Add trait's T replacement.
  params.push_back(TypeConstantAttr::get(structSelfType, anyRegTypeType));
  ParameterEvaluator evaluator(params);
  auto bindings = InputParamBindings::getForDeclaredType(
      structSelfType.getMetaType(), emitter);
  for (Type type : inputParamTypes.drop_front(2)) {
    params.push_back(UnboundAttr::get(type));
    evaluator.addInputValue(params.back());
    bindings.addPrechecked(params.back());
  }

  return {signature.getSpecializedSignature(
              params,
              []() -> InFlightDiagnostic {
                assert(false && "getSpecializedSignature should not fail here");
                return {};
              }),
          std::move(bindings)};
}

/// Given the signature of a trait function, which assumes that the self type is
/// memory-only, compute the equivalent signature as if the self type is
/// register-passable.
static LITSignatureType getRegisterPassableSignature(LITSignatureType traitSig,
                                                     ASTType selfType,
                                                     bool trivial) {
  // This function does two things: if the self type is in the result slot, it
  // moves it to the return, mindful of error handling, and if it is found in
  // any arguments, it is taken out of memory as appropriate.
  SmallVector<Type> argTypes;
  SmallVector<ValueInputConvention> conventions;
  bool replacedResult = false;
  Type resultType = traitSig.getResultType();
  FnEffects fnEffects = traitSig.getFnEffects();
  size_t numImplicitLifetimeDecls = traitSig.getNumImplicitLifetimeDecls();

  for (auto [type, conv] :
       llvm::zip(traitSig.getValueInputs(), traitSig.getInputConventions())) {
    // Check for a `Self`-type result.
    if (conv == ValueInputConvention::ByRefResult ||
        conv == ValueInputConvention::InitSelf) {
      if (ASTType(type).getReferenceElementType().mlirType != selfType) {
        argTypes.push_back(type);
        conventions.push_back(conv);
        continue;
      }

      // We'll be dropping the reference, so we'll drop the implicit lifetime.
      --numImplicitLifetimeDecls;

      replacedResult = true;
      // Make sure to set the `ownedresult` bit if the type is not trivial.
      if (!trivial)
        fnEffects.setOwnedRegisterResult();
      // Move the self type into the result.
      if (!traitSig.isThrows()) {
        // Just overwrite the none result type.
        resultType = selfType;
        continue;
      }
      // For a throwing function, we need to insert the type into the variant.
      // The error type is the first type.
      auto variant = cast<VariantType>(resultType);
      resultType = VariantType::get(
          {ParamRefType::get(variant.getTypes().front()), selfType},
          AnyRegTypeType::get(variant.getContext()));
      continue;
    }

    // Check for a `Self`-type argument. It would always be in-memory.
    if (conv == ValueInputConvention::OwnedInMem ||
        conv == ValueInputConvention::BorrowedInMem) {
      if (ASTType(type).getReferenceElementType().mlirType != selfType) {
        argTypes.push_back(type);
        conventions.push_back(conv);
        continue;
      }

      // We'll be dropping the reference, so we'll drop the implicit lifetime.
      if (conv == ValueInputConvention::OwnedInMem)
        --numImplicitLifetimeDecls;
      else
        assert(!isa<RefType>(type) && "TODO: more references");

      // Unwrap the pointer type and update the convention.
      argTypes.push_back(selfType);
      conventions.push_back(conv == ValueInputConvention::OwnedInMem
                                ? ValueInputConvention::OwnedInReg
                                : ValueInputConvention::BorrowedInReg);
      continue;
    }
    argTypes.push_back(type);
    conventions.push_back(conv);
  }

  return SignatureType::get(
      FunctionType::get(traitSig.getContext(), argTypes, resultType),
      traitSig.getInputParamTypes(), traitSig.getResultParamTypes(),
      conventions, fnEffects,
      FnMetadataAttr::get(
          traitSig.getContext(),
          traitSig.getArgNames().drop_front(replacedResult),
          traitSig.getArgPassingKinds().drop_front(replacedResult),
          traitSig.getParamNames(), traitSig.getParamPassingKinds(),
          traitSig.getDefaultArguments(), traitSig.getDefaultParameters(),
          numImplicitLifetimeDecls));
}

/// Synthesize a single stub for a register-passable type to meet a conformance
/// requirement for a trait. Trait function prototypes assume memory-only
/// conventions for the trait self type, but register-passable types will
/// implement the opposite. Synthesize thunks that match the required signatures
/// by the trait.
static void synthesizeRegisterTraitStub(ASTDecl &structDecl,
                                        SharedState &shared, StringAttr name,
                                        TypedAttr callee,
                                        LITSignatureType memSig) {
  // Synthesize input and result parameter decls.
  SmallVector<ParamDeclAttr> inputParamDecls, resultParamDecls;
  Builder b(shared.getContext());
  for (auto [i, type, name] :
       llvm::enumerate(memSig.getInputParamTypes(), memSig.getParamNames())) {
    // The input parameter names are derived from the decl name.
    inputParamDecls.push_back(ParamDeclAttr::get(
        name.empty() ? b.getStringAttr("i" + Twine(i)) : name, type));
  }
  for (auto [i, type] : llvm::enumerate(memSig.getResultParamTypes())) {
    resultParamDecls.push_back(
        ParamDeclAttr::get(b.getStringAttr("o" + Twine(i)), type));
  }

  // createFunction will generate a function for us, but wants a set of input
  // types that have symbolic names for the argument lifetimes, not indexed
  // ones.
  SmallVector<Type> inputTypes;
  for (auto [conv, type, argName] :
       llvm::zip(memSig.getInputConventions(), memSig.getValueInputs(),
                 memSig.getArgNames())) {
    if (!isArgumentPassedWithImplicitLifetime(conv)) {
      inputTypes.push_back(type);
      continue;
    }

    // The input convention will have an index type, so replace it with a
    // ParamDeclRefAttr to a named lifetime, so createFunction can do it's
    // thing.
  }

  // Synthesize the method inside the struct.
  auto [thunk, decl] = StructEmitter(shared).synthesizeMethodInStruct(
      name, inputParamDecls, memSig.getParamPassingKinds(),
      memSig.getValueInputs(), memSig.getInputConventions(),
      memSig.getArgNames(), memSig.getArgPassingKinds(), memSig.getResultType(),
      structDecl, SpecialFunctionInfo::getKind(name), memSig.getFnEffects(),
      resultParamDecls, "`thunk_");
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (DebugInfo::DIScopeAttr spAttr = thunk.getLocScope())
    diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

  // Always inline the thunk. The calling convention conversion overhead is
  // guaranteed to be optimized away.
  thunk.setInlineLevel(InlineLevel::Always);

  // Now prepare to emit the call to the register-passable method.
  ExprEmitter emitter(shared, structDecl, EC_Trait);
  emitter.builder = OpBuilder::atBlockBegin(thunk.getBody());

  // The callee is partially bound, containing only its parent struct
  // parameters. Bind the rest of them here.
  SmallVector<TypedAttr> bindSigInputs{callee};
  for (ParamDeclAttr inputParam : inputParamDecls)
    bindSigInputs.push_back(ParamDeclRefAttr::get(inputParam));
  callee = ParamOperatorAttr::get(POC::BindSignature, bindSigInputs);

  // Treat the `init_self` argument like a result slot.
  bool hasResultSlot =
      memSig.hasMemoryOnlyResult() || memSig.hasInitSelfResult();

  // Construct the call operands from the function block arguments. Ensure
  // keyword-only arguments are specified accordingly.
  SyntheticNode node(structDecl.getLoc());
  SmallVector<FuncOperand> posOperands;
  SmallDenseMap<StringAttr, FuncOperand> kwOperands;
  for (auto [arg, kind, conv, name] : llvm::drop_begin(
           llvm::zip(thunk.getArguments(), memSig.getArgPassingKinds(),
                     memSig.getInputConventions(), memSig.getArgNames()),
           hasResultSlot)) {
    AnyValue value;
    switch (conv) {
    case ValueInputConvention::ByRef:
      value = MLValue(arg);
      break;
    case ValueInputConvention::OwnedInMem:
      value = XRValue(arg);
      break;
    case ValueInputConvention::OwnedInReg:
      value = SRValue(arg);
      break;
    case ValueInputConvention::BorrowedInReg:
      value = SBValue(arg);
      break;
    case ValueInputConvention::BorrowedInMem:
      value = MBValue(arg);
      break;
    default:
      llvm_unreachable("unexpected input convention");
    }
    if (kind == PassingKind::KwOnly)
      kwOperands.insert({name, {value, &node}});
    else
      posOperands.push_back({value, &node});
  }

  // Declare result parameters for the call, if required.
  SmallVector<ParamDeclAttr> callResultParams;
  for (auto [i, type] : llvm::enumerate(memSig.getResultParamTypes())) {
    callResultParams.push_back(
        ParamDeclAttr::get(b.getStringAttr("r" + Twine(i)), type));
  }

  // Allocate the value dest for the call. Set the value dest to the result
  // slot, if there is one, otherwise provide the expected rvalue type.
  ValueDest dest(EC_Trait);
  if (hasResultSlot)
    dest = ValueDest(XLValue(thunk.getArgument(0)), EC_Trait);

  CValue callResult = emitter.emitCallUnchecked(
      PValue(callee), CallOperands(posOperands, &kwOperands), callResultParams,
      dest, &node);

  // Bind result parameters, if there are any.
  ImplicitLocOpBuilder builder(shared.translateLocation(structDecl.getLoc()),
                               *emitter.builder);
  if (!callResultParams.empty()) {
    SmallVector<TypedAttr> resultRefs;
    for (ParamDeclAttr resultDecl : callResultParams)
      resultRefs.push_back(ParamDeclRefAttr::get(resultDecl));
    builder.create<LIT::ParamReturnOp>(
        ParameterExprArrayAttr::get(b.getContext(), resultRefs));
  }

  // If the callee is async, then await the result.
  if (memSig.isAsync()) {
    ValueDest dest(EC_Trait);
    callResult =
        emitter.emitNamedMethodCall("__await__", FuncOperand{callResult, &node},
                                    dest, CallSyntax::kMethodCall, &node);
    if (!callResult)
      return;
  }

  // Emit the function return. It's just a none return if the function has a
  // result slot.
  // FIXME: handle async
  Value retVal;
  if (hasResultSlot) {
    retVal =
        builder.create<ParamConstantOp>(KGEN::NoneAttr::get(b.getContext()));
  } else {
    retVal = emitter.emitSRValue({callResult, &node}, EC_Trait);
  }
  if (memSig.isThrows()) {
    retVal = builder.create<VariantCreateOp>(memSig.getResultType(), retVal,
                                             /*index=*/1);
  }
  builder.create<KGEN::ReturnOp>(retVal);
}

/// Synthesize stubs for register-passable types to meet conformance
/// requirements for a trait.
static void synthesizeRegisterTraitStubs(
    ASTDecl &structDecl, SharedState &shared,
    ArrayRef<std::pair<std::pair<StringAttr, TypedAttr>, LITSignatureType>>
        stubs) {
  for (auto [key, sig] : stubs) {
    auto [name, callee] = key;
    // If no rewrite is necessary, skip this function.
    if (callee.getType() == sig)
      continue;
    synthesizeRegisterTraitStub(structDecl, shared, name, callee, sig);
  }
}

/// Allow synthesizing default implementations of certain special functions.
static void synthesizeSpecialFunction(ASTDecl &structDecl, SharedState &shared,
                                      SpecialFunctionKind kind) {
  StructEmitter gen(shared);
  auto selfRefType =
      structDecl.getSelfType().getRefForArgument("self", /*isMut=*/true);
  auto empty = StringAttr::get(shared.getContext());

  // Synthesize the required special method. Importantly, don't mark the struct
  // as actually having this method so that destructors et al. are not
  // needlessly emitted.
  LIT::FuncOp func;
  if (kind == SpecialFunctionKind::kDel) {
    // Synthesize an empty destructor. Don't do anything special, because we
    // want check lifetimes to insert a call to the real destructor here, if it
    // has one.
    auto [dtor, decl] = gen.synthesizeMethodInStruct(
        "__del__", selfRefType, ValueInputConvention::OwnedInMem, empty,
        PassingKind::PosOnly, shared.getNoneType(), structDecl, kind,
        FnEffects(), "`thunk_");
    func = dtor;
  } else {
    // Determine the name and argument conventions of the function.
    ValueInputConvention otherConv;
    switch (kind) {
    case SpecialFunctionKind::kCopyInit:
      otherConv = ValueInputConvention::BorrowedInMem;
      break;
    case SpecialFunctionKind::kMoveInit:
      otherConv = ValueInputConvention::OwnedInMem;
      break;
    default:
      llvm_unreachable("unexpected special function kind to synthesize");
    }
    StringRef name = SpecialFunctionInfo::get(kind).name;
    Type existingType;
    if (kind == SpecialFunctionKind::kMoveInit)
      existingType = structDecl.getSelfType().getRefForArgument("existing",
                                                                /*isMut=*/true);
    else // FIXME: Eliminate when 'borrowed' moves.
      existingType = PointerType::get(structDecl.getSelfType());

    auto [ctor, decl] = gen.synthesizeMethodInStruct(
        name, {selfRefType, existingType},
        {ValueInputConvention::InitSelf, otherConv}, {empty, empty},
        {PassingKind::PosOnly, PassingKind::PosOnly}, shared.getNoneType(),
        structDecl, kind, FnEffects(), "`thunk_");
    func = ctor;
    // In every case, the implementation is a load+store.
    auto b = ImplicitLocOpBuilder::atBlockBegin(func.getLoc(), func.getBody());
    Value value;
    if (kind == SpecialFunctionKind::kMoveInit)
      value = b.create<LIT::LoadConsumeOp>(func.getArgument(1));
    else
      value = b.create<POP::LoadOp>(func.getArgument(1));
    b.create<RefStoreOp>(value, func.getArgument(0));
  }
  func.setInlineLevel(InlineLevel::AlwaysNoDebug);
  auto b = ImplicitLocOpBuilder::atBlockEnd(func.getLoc(), func.getBody());
  b.create<KGEN::ReturnOp>(
      Value(b.create<ParamConstantOp>(NoneAttr::get(b.getContext()))));
}

/// Check conformance for struct that implements traits.
static LogicalResult verifyConformance(ASTDecl &structDecl,
                                       SharedState &shared) {
  auto structDeclOp = cast<StructDeclOp>(structDecl);
  bool rpTrivial = structDeclOp.isRegisterPassableTrivial();
  bool regPassable = structDeclOp.isRegisterPassable();
  bool hadErrors = false;
  SyntheticNode node(structDecl.getLoc());
  ExprEmitter emitter(shared, structDecl, EC_Trait);
  ASTType selfType = structDecl.getSelfType();

  // For register-passable types, this is the set of stubs that need to be
  // synthesized for calling convention conversion. This maps a function name
  // and symbol reference to the required memory-only signature.
  llvm::MapVector<std::pair<StringAttr, TypedAttr>, LITSignatureType> regStubs;

  // These are the special methods that need to be synthesized.
  SmallVector<SpecialFunctionKind> specialFns;

  for (TypeLineageAttr parent : structDeclOp.getParentTypes()) {
    auto trait = dyn_cast<TraitType>(parent.getType());
    if (!trait)
      continue;
    ASTDecl &traitDecl =
        emitter.getDeclResolver().getDeclForTypeSymbol(trait.getSymbol());

    // Make sure to fully resolve the trait first.
    if (failed(shared.declResolver->resolveFully(traitDecl,
                                                 structDecl.getLoc()))) {
      hadErrors = true;
      continue;
    }

    bool allMatchFound = true;
    // Prepare an error. It will be abandoned if the check succeeds.
    StringRef traitName = cast<TraitDeclOp>(traitDecl).getSymName();
    InflightDiag diag = shared.emitError(structDecl.getLoc(), "struct ")
                        << selfType
                        << " does not implement all requirements for '"
                        << traitName << "'";

    for (auto &[name, decls] : traitDecl.getDeclsInScope()) {
      for (ASTDecl *decl : decls) {
        auto traitFn = dyn_cast<LIT::FuncOp>(*decl);
        // Skip any children that aren't methods or are inherited. This could be
        // an alias.
        if (!traitFn || traitFn.getIsInherited())
          continue;

        ArrayRef<ASTDecl *> decls = structDecl.lookupInCurrentScope(name);
        if (decls.empty() || !isa<LIT::FuncOp>(decls.front())) {
          if (canSynthesizeIfMissing(name, /*rpTrivial=*/rpTrivial,
                                     /*regPassable=*/regPassable, specialFns))
            continue;
          diag.attachNote(traitFn.getLoc())
              << "required function '" + name.str() + "' is not implemented";
          allMatchFound = false;
          break;
        }

        // Signature resolve the found decls first, so they can be checked.
        for (ASTDecl *decl : decls) {
          if (failed(shared.declResolver->resolve(
                  *decl, DeclResolvedness::signature, structDecl.getLoc())))
            hadErrors = true;
        }

        auto [newSignature, bindings] =
            getTraitFunctionSignature(emitter, traitFn, selfType);
        // Match against the transformed calling convention if the struct is
        // register-passable.
        LITSignatureType traitSignature = newSignature;
        if (regPassable) {
          newSignature =
              getRegisterPassableSignature(newSignature, selfType, rpTrivial);
        }

        OverloadSet ov(name, decls, std::move(bindings), &node,
                       CallSyntax::kMethodCall);
        PValue result = ov.filterOverloadSetForValueType(
            newSignature, emitter, [&](SMLoc loc) -> InflightDiag & {
              return diag.attachNote(decl->getLoc());
            });
        if (!result)
          allMatchFound = false;
        if (regPassable)
          regStubs.insert({{name, result.get()}, traitSignature});
      }
    }
    if (allMatchFound) {
      diag.abandon();
    } else {
      diag.attachNote(traitDecl.getLoc())
          << "trait '" << traitName << "' declared here";
      if (!parent.getInheritedFrom().empty()) {
        ASTDecl &parentDecl = emitter.getDeclResolver().getDeclForTypeSymbol(
            cast<TraitType>(parent.getInheritedFrom().front()).getSymbol());
        diag.attachNote(parentDecl.getLoc())
            << "inherited through '" << *parentDecl.getNameIfOperation()
            << "' here";
      }
      hadErrors = true;
    }
  }

  if (hadErrors)
    return failure();
  if (regPassable)
    synthesizeRegisterTraitStubs(structDecl, shared, regStubs.takeVector());
  for (SpecialFunctionKind kind : specialFns)
    synthesizeSpecialFunction(structDecl, shared, kind);
  return success();
}

ParseResult DeclResolver::resolveBody(StructDeclOp structOp, Lexer &lexer,
                                      ASTDecl &structDecl) {
  // Push the debug scope for this struct if necessary so that nested operations
  // have proper debug info.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (DebugInfo::DIScopeAttr spAttr = structOp.getLocScope())
    diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

  if (ParserBase(shared, lexer).parseSuite(structDecl))
    return failure();

  // Track whether any field needs destruction, if so, we need a __del__
  // method.
  bool needsDtorForFields = false;

  /// This collects all the resolved struct fields.
  SmallVector<std::pair<StructFieldOp, ASTDecl *>> structFields;

  // Now that the body is completely resolved, check the declared fields for
  // extra invariants.
  for (StructFieldOp field : structOp.getFieldDecls()) {
    // Make sure the field is signature resolved so we can get its type.
    auto fieldEntries = structDecl.lookupInCurrentScope(field.getNameAttr());
    assert(fieldEntries.size() == 1 && "field decls cannot be overloaded");
    ASTDecl &fieldASTDecl = *fieldEntries[0];
    if (failed(resolveSignature(fieldASTDecl, fieldASTDecl.getLoc())))
      continue;

    // If any field of this struct has a destructor, then the struct needs
    // one.
    needsDtorForFields |=
        ASTType(field.getType()).hasDestructor(fieldASTDecl.getLoc(), shared);

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

  // If there are any body decorators, resolve them now.
  StructBodyDecorators structDecorators(structOp, structDecl, *this,
                                        structFields);
  Decorators(structDecl, shared).applyBodyDecorators([&](ExprNode *decorator) {
    return structDecorators.processDecorator(decorator);
  });

  if (structDecl.hasReferenceError)
    return success();

  // Now that the struct body has been resolved, check to see if there is a
  // destructor and install it into the StructDeclOp if so.
  if (auto dtorAttr = lookupDestructor(structDecl, shared)) {
    // Check to see if we have an explicitly declared destructor.
    structOp.setDestructorAttr(dtorAttr);
  } else if (needsDtorForFields) {
    // If one of the fields needs to be destroyed, then we synthesize an empty
    // del function so that lifetime checking can handle field destruction.
    structOp.setDestructorAttr(
        synthesizeEmptyDtor(shared, structOp, structDecl, *this));
  }
  // Look up move and copy constructors and record them.
  if (!structOp.isRegisterPassable()) {
    if (auto copyInitAttr = lookupCopyMoveInit(structDecl, shared,
                                               SpecialFunctionKind::kCopyInit))
      structOp.setCopyInitAttr(copyInitAttr);
    if (auto moveInitAttr = lookupCopyMoveInit(structDecl, shared,
                                               SpecialFunctionKind::kMoveInit))
      structOp.setMoveInitAttr(moveInitAttr);
  }

  // Finally, verify conformance of inherited traits.
  return verifyConformance(structDecl, shared);
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

  // Make every trait inherit from Destructable, except itself.
  if (parentTypes.empty() && traitOp.getSymName() != "Destructable") {
    LookupResult lookup = shared.lookupAndResolveDecl(
        "Destructable", decl.getLoc(), *decl.getParentDecl(),
        /*searchParentScopes=*/true);
    if (!lookup.isFailure() && !lookup.getIfSuccess().empty()) {
      parentTypes.push_back(TypeLineageAttr::get(
          cast<TraitDeclOp>(lookup.getIfSuccess().front()).bindReference()));
    } else {
      emitError(decl.getLoc(), "could not find builtin 'Destructable' trait");
    }
  }

  // Insert the implicit trait parameters:
  // - MT: an AnyRegTypeType which points to the struct that implements this
  // trait.
  // - T: a ParamRef to MT which is the type of MT.
  // TODO: build AnyType instead
  auto mt = ParamDeclAttr::get("MT", AnyRegTypeType::get(decl.getContext()));
  auto mtRef = ParamDeclAttr::get(
      "T", KGEN::ParamRefType::get(KGEN::ParamDeclRefAttr::get(mt)));

  auto inputParams = ParamDeclArrayAttr::get(getContext(), {mt, mtRef});
  traitOp.setInputParams(inputParams);
  SmallVector<StringAttr> paramNames{StringAttr::get(decl.getContext(), ""),
                                     StringAttr::get(decl.getContext(), "")};
  SmallVector<PassingKind> paramPassingKinds{PassingKind::Implicit,
                                             PassingKind::Implicit};
  SmallVector<TypedAttr> paramDefaults;
  auto sig = TypeSignatureType::remapToSignature(
      silenceErrors(getContext()), inputParams, paramNames, paramPassingKinds,
      paramDefaults, false);
  if (!sig)
    return failure();
  traitOp.setSignature(sig);
  traitOp.setParentTypes(parentTypes);

  decl.setSelfType(ASTDecl::computeSelfTypeForTrait(traitOp));

  shared.notifyListenerOnTraitDecl(decl, identifierLoc);

  return success();
}

ParseResult DeclResolver::resolveBody(TraitDeclOp traitOp, Lexer &lexer,
                                      ASTDecl &traitDecl) {
  // Push the debug scope for this trait if necessary so that nested operations
  // have proper debug info.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (DebugInfo::DIScopeAttr spAttr = traitOp.getLocScope())
    diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

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

  // Now just pull in the functions in the bodies of all parents.
  Block &body = *traitOp.getBody();
  for (TypeLineageAttr parent : traitOp.getParentTypes()) {
    ASTDecl &parentDecl =
        getDeclForTypeSymbol(cast<TraitType>(parent.getType()).getSymbol());
    if (failed(resolveFully(parentDecl, traitDecl.getLoc())))
      continue;

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
        // Mark the function as inherited so that conformance checking won't
        // give duplicate errors if it is not provided.
        func.setIsInherited(true);
        body.push_back(func);
        finalizeFuncSignature(func, traitDecl);
        addFullyResolvedDecl(&*func, name, decl->getLoc(), &traitDecl);
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
