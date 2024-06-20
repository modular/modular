//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Traits.h"
#include "MojoUtils.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/CallEmission.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/ExprEmitter.h"
#include "KGEN/MojoParser/ExprNodes.h"
#include "KGEN/MojoParser/ParserParamEvaluator.h"
#include "KGEN/MojoParser/SharedState.h"
#include "KGEN/MojoParser/StructEmitter.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

/// Get specialized signature of a trait function with a struct (who implements
/// the trait) type. Also return parameter bindings for specializing the
/// expected struct method with the current struct type.
static std::pair<LITSignatureType, ParamBindings>
getTraitFunctionSignature(ExprEmitter &emitter, LIT::FuncOp traitFn,
                          ASTType structSelfType, TraitType trait) {
  LITSignatureType signature = traitFn.getFullSignature();
  SmallVector<TypedAttr> params;
  ArrayRef<Type> paramTypes = signature.getParamTypes();

  // Add trait's T replacement.
  params.push_back(TypeConstantAttr::get(structSelfType, trait));
  ParserParamEvaluator evaluator(emitter.getDeclResolver(), params);
  auto bindings =
      ParamBindings::getForDeclaredType(emitter.getScopeInfo(), structSelfType);
  // Leave the rest alone.
  for (Type type : paramTypes.drop_front()) {
    params.push_back(UnboundAttr::get(type));
    evaluator.addInputValue(params.back());
    bindings.addPrechecked(params.back());
  }

  return {signature.getSpecializedSignature(params), std::move(bindings)};
}

/// Given the signature of a trait function, which assumes that the self type is
/// memory-only, compute the equivalent signature as if the self type is
/// register-passable.
///
/// If isRegInit is true, then we need to transform the expected InitSelf to
/// a register result form.  This is to support the deprecated register-result
/// forms of __init__/__copyinit__.
/// TODO: Remove these special initializer forms.
static LITSignatureType getRegisterPassableSignature(LITSignatureType traitSig,
                                                     ASTType selfType,
                                                     bool isRegInit) {
  // This function does two things: if the self type is in the result slot, it
  // moves it to the return, mindful of error handling, and if it is found in
  // any arguments, it is taken out of memory as appropriate.
  SmallVector<Type> argTypes;
  SmallVector<ArgConvention> conventions;
  bool replacedResult = false;
  Type resultType = traitSig.getResultType();
  FnEffects fnEffects = traitSig.getFnEffects();
  size_t numImplicitLifetimeDecls = traitSig.getNumImplicitLifetimeDecls();

  for (auto [type, conv] :
       llvm::zip(traitSig.getArguments(), traitSig.getArgConventions())) {
    // Check for a `Self`-type result.
    if (conv == ArgConvention::ByRefResult ||
        // Rewrite InitSelf if  the type implements init in the deprecated way.
        // TODO: Remove this support.
        (conv == ArgConvention::InitSelf && isRegInit)) {
      // Don't modify a inout result of an unrelated type. If the function
      // raises, then the result is always returned through memory.
      if (ASTType(type).getReferenceElementType().mlirType != selfType ||
          traitSig.isThrows()) {
        argTypes.push_back(type);
        conventions.push_back(conv);
        continue;
      }

      // We'll be dropping the reference, so we'll drop the implicit lifetime.
      --numImplicitLifetimeDecls;

      replacedResult = true;
      // Move the self type into the result.
      resultType = selfType;
      continue;
    }

    // Check for a `Self`-type argument. It would always be in-memory.
    if (conv == ArgConvention::OwnedInMem ||
        conv == ArgConvention::BorrowedInMem) {
      if (ASTType(type).getReferenceElementType().mlirType != selfType) {
        argTypes.push_back(type);
        conventions.push_back(conv);
        continue;
      }

      // We'll be dropping the reference, so we'll drop the implicit lifetime.
      --numImplicitLifetimeDecls;

      // Unwrap the pointer type and update the convention.
      argTypes.push_back(selfType);
      conventions.push_back(conv == ArgConvention::OwnedInMem
                                ? ArgConvention::OwnedInReg
                                : ArgConvention::BorrowedInReg);
      continue;
    }
    argTypes.push_back(type);
    conventions.push_back(conv);
  }

  PogListAttr oldArgListAttrs = traitSig.getArgListAttrs();
  ArrayRef<PogMetadataAttr> pogs = oldArgListAttrs.getPogs();
  if (replacedResult) {
    pogs = pogs.drop_front(traitSig.hasInitSelfArg())
               .drop_back(traitSig.hasMemoryOnlyResult());
  }

  auto metadata = FnMetadataAttr::get(oldArgListAttrs.cloneWith(pogs),
                                      traitSig.getParamListAttrs(),
                                      numImplicitLifetimeDecls);
  return SignatureType::get(
      FunctionType::get(traitSig.getContext(), argTypes, resultType),
      traitSig.getParamTypes(), traitSig.getResultParamTypes(), conventions,
      fnEffects, metadata);
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
  SmallVector<ParamDeclAttr> paramDecls;
  SmallVector<TypedAttr> paramValues;
  ParameterEvaluator evaluator;
  Builder b(shared.getContext());
  for (auto [idx, type] : llvm::enumerate(memSig.getParamTypes())) {
    StringAttr name = memSig.getParamName(idx);
    // The parameter names are derived from the decl name.
    paramDecls.push_back(ParamDeclAttr::get(
        name.empty() ? b.getStringAttr("i" + Twine(idx)) : name,
        evaluator.getReboundType(type)));
    paramValues.push_back(ParamDeclRefAttr::get(paramDecls.back()));
    evaluator.addInputValue(paramValues.back());
  }

  // Synthesize the method inside the struct.
  PogListAttr argListAttr = memSig.getArgListAttrs();
  auto [thunk, _] = StructEmitter(shared).synthesizeMethodInStruct(
      name, paramDecls, memSig.getParamListAttrs(), memSig.getArguments(),
      memSig.getArgConventions(), argListAttr, memSig.getResultType(),
      structDecl, SpecialFunctionInfo::getKind(name), memSig.getFnEffects(),
      "_thunk");
  if (!thunk)
    return;
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(thunk.getLocScope());

  // Always inline the thunk. The calling convention conversion overhead is
  // guaranteed to be optimized away.
  thunk.setInlineLevel(InlineLevel::AlwaysNoDebug);

  // Now prepare to emit the call to the register-passable method.
  ExprEmitter emitter(shared, structDecl, EC_Trait);
  emitter.builder = OpBuilder::atBlockBegin(thunk.getBody());

  // The callee is partially bound, containing only its parent struct
  // parameters. Bind the rest of them here.
  SmallVector<TypedAttr> bindSigInputs{callee};
  llvm::append_range(bindSigInputs, paramValues);
  callee = ParamOperatorAttr::get(POC::BindSignature, bindSigInputs);

  SignatureType calleeSig = cast<LITSignatureType>(callee.getType());

  // Construct the call operands from the function block arguments. Ensure
  // keyword-only arguments are specified accordingly.
  SyntheticNode node(structDecl.getLoc());
  SmallVector<FuncOperand> posOperands;
  KeywordOperands kwOperands;
  bool hasLegacyInitSelfArg = false;
  for (auto [arg, conv, pogAttr] :
       llvm::zip(thunk.getArguments(), memSig.getArgConventions(),
                 argListAttr.getPogs())) {
    AnyValue value;
    switch (conv) {
    case ArgConvention::InitSelf:
      // If the implementation takes the same InitSelf argument then pass it.
      // If not, this must be the deprecated '-> Self' forms of init/copyinit.
      if (calleeSig.hasInitSelfArg()) {
        value = MLValue(arg);
        break;
      }
      // TODO: remove this old forms.
      hasLegacyInitSelfArg = true;
      continue;

    case ArgConvention::ByRefResult:
    case ArgConvention::ByRefError:
      continue; // Ignore this, it will be assigned to later.

    case ArgConvention::InOut:
      value = MLValue(arg);
      break;
    case ArgConvention::OwnedInMem:
      value = MRValue(arg);
      break;
    case ArgConvention::OwnedInReg:
      value = SRValue(arg);
      break;
    case ArgConvention::BorrowedInReg:
      value = SBValue(arg);
      break;
    case ArgConvention::BorrowedInMem:
      value = MBValue(arg);
      break;
    default:
      llvm_unreachable("unexpected input convention");
    }
    if (pogAttr.getPassingKind() == PassingKind::KwOnly)
      kwOperands.insert({pogAttr.getName(), {value, node}});
    else
      posOperands.push_back({value, node});
  }

  // Allocate the value dest for the call. Set the value dest to the result
  // slot, if there is one, otherwise provide the expected rvalue type.
  ValueDest dest(EC_Trait);
  bool hasRegisterResult = false;
  if (memSig.isAsync()) {
    // An async call returns a coroutine we have to await.
  } else if (memSig.hasMemoryOnlyResult()) {
    dest = ValueDest(MLValue(thunk.getArguments().back()), EC_Trait);
  } else if (memSig.hasInitSelfArg()) {
    if (hasLegacyInitSelfArg)
      dest = ValueDest(MLValue(thunk.getArgument(0)), EC_Trait);
    // If both the caller and callee take initself, we initialize it directly
    // above and need to return none.
  } else {
    hasRegisterResult = true;
  }

  CValue callResult = emitter.emitIndirectCall(
      PValue(callee), CallOperands(posOperands, &kwOperands), dest, node);
  if (!callResult)
    return;

  // If the callee is async, we got a coroutine. Now await it into the result.
  if (memSig.isAsync()) {
    ValueDest dest(MLValue(thunk.getArguments().back()), EC_Trait);
    if (!emitter.emitNamedMethodCall("__await__", FuncOperand{callResult, node},
                                     dest, CallSyntax::kMethodCall, node))
      return;
  }

  // Emit the function return. It's just a none return if the function has a
  // result slot.
  // FIXME: handle async
  ImplicitLocOpBuilder builder(shared.translateLocation(structDecl.getLoc()),
                               *emitter.builder);
  Value retVal;
  if (hasRegisterResult)
    retVal = emitter.emitSRValue({callResult, node}, EC_Trait);
  else if (memSig.isThrows())
    retVal = builder.create<ParamConstantOp>(builder.getBoolAttr(false));
  else
    retVal = builder.create<ParamConstantOp>(shared.getNoneAttr());
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
      structDecl.getTypeDeclSelf().getRefForArgument("self", /*isMut=*/true);
  MLIRContext *ctx = shared.getContext();
  auto empty = StringAttr::get(ctx);

  // Synthesize the required special method. Importantly, don't mark the struct
  // as actually having this method so that destructors et al. are not
  // needlessly emitted.
  LIT::FuncOp func;
  if (kind == SpecialFunctionKind::kDel) {
    // Synthesize an empty destructor. Don't do anything special, because we
    // want check lifetimes to insert a call to the real destructor here, if it
    // has one.
    auto [dtor, _] = gen.synthesizeMethodInStruct(
        "__del__", selfRefType, ArgConvention::OwnedInMem,
        PogListAttr::get(ctx, {empty}, {PassingKind::PosOnly}),
        shared.getNoneType(), structDecl, kind, FnEffects(), "_thunk");
    if (!dtor)
      return;
    func = dtor;
  } else {
    // Determine the name and argument conventions of the function.
    ArgConvention existingConv;
    switch (kind) {
    case SpecialFunctionKind::kCopyInit:
      existingConv = ArgConvention::BorrowedInMem;
      break;
    case SpecialFunctionKind::kMoveInit:
      existingConv = ArgConvention::OwnedInMem;
      break;
    default:
      llvm_unreachable("unexpected special function kind to synthesize");
    }
    StringRef name = SpecialFunctionInfo::get(kind).name;
    Type existingType;
    bool isMut = existingConv == ArgConvention::OwnedInMem;
    existingType =
        structDecl.getTypeDeclSelf().getRefForArgument("existing", isMut);
    auto [ctor, _] = gen.synthesizeMethodInStruct(
        name, {selfRefType, existingType},
        {ArgConvention::InitSelf, existingConv},
        PogListAttr::get(ctx, {empty, empty},
                         {PassingKind::PosOnly, PassingKind::PosOnly}),
        shared.getNoneType(), structDecl, kind, FnEffects(), "_thunk");
    if (!ctor)
      return;
    func = ctor;
    // In every case, the implementation is a load+store.
    auto b = ImplicitLocOpBuilder::atBlockBegin(func.getLoc(), func.getBody());
    Value value;
    if (kind == SpecialFunctionKind::kMoveInit)
      value = b.create<LIT::LoadConsumeOp>(func.getArgument(1));
    else
      value = b.create<RefLoadOp>(func.getArgument(1));
    b.create<RefStoreOp>(value, func.getArgument(0));
  }
  func.setInlineLevel(InlineLevel::AlwaysNoDebug);
  auto b = ImplicitLocOpBuilder::atBlockEnd(func.getLoc(), func.getBody());
  b.create<KGEN::ReturnOp>(
      Value(b.create<ParamConstantOp>(b.getAttr<NoneAttr>())));
}

LogicalResult LIT::verifyConformance(ASTDecl &structDecl,
                                     TypeLineageAttr parent,
                                     SharedState &shared,
                                     std::optional<InflightDiag> &diag) {
  auto trait = dyn_cast<TraitType>(parent.getType());
  if (!trait)
    return success();

  auto structDeclOp = cast<StructDeclOp>(structDecl);
  bool rpTrivial = structDeclOp.isRegisterPassableTrivial();
  bool regPassable = structDeclOp.isRegisterPassable();
  bool hadErrors = false;
  SyntheticNode node(structDecl.getLoc());
  ExprEmitter emitter(shared, structDecl, EC_Trait);
  ASTType selfType = structDecl.getTypeDeclSelf();

  // For register-passable types, this is the set of stubs that need to be
  // synthesized for calling convention conversion. This maps a function name
  // and symbol reference to the required memory-only signature.
  llvm::MapVector<std::pair<StringAttr, TypedAttr>, LITSignatureType> regStubs;

  // These are the special methods that need to be synthesized.
  SmallVector<SpecialFunctionKind> specialFns;

  ASTDecl &traitDecl =
      emitter.getDeclResolver().getDeclForTypeSymbol(trait.getSymbol());

  // Make sure to fully resolve the trait first.
  if (failed(shared.declResolver->resolveFully(traitDecl, structDecl.getLoc())))
    return failure();

  bool allMatchFound = true;
  // Prepare an error. It will be abandoned if the check succeeds.
  StringRef traitName = cast<TraitDeclOp>(traitDecl).getSymName();
  diag = shared.emitError(structDecl.getLoc(), "struct ")
         << selfType << " does not implement all requirements for '"
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
        if (canSynthesizeIfMissing(name, rpTrivial, regPassable)) {
          specialFns.push_back(SpecialFunctionInfo::getKind(name));
          continue;
        }
        diag->attachNote(decl->getLoc())
            << "required function '" + name.str() + "' is not implemented";
        allMatchFound = false;
        break;
      }

      // Signature resolve the found decls first, so they can be checked.
      bool isRegInit = false;
      for (ASTDecl *decl : decls) {
        if (failed(shared.declResolver->resolve(
                *decl, DeclResolvedness::signature, structDecl.getLoc()))) {
          hadErrors = true;
          continue;
        }

        // If this type implements with the deprecated kInitReg or
        // kCopyInitReg convention then we'll have to transform the InitSelf
        // argument for a match.
        // TODO: Remove these special initializer forms.
        if (regPassable) {
          auto specialKind = cast<LIT::FuncOp>(*decl).getSpecialFunctionKind();
          if (specialKind == SpecialFunctionKind::kInitReg ||
              specialKind == SpecialFunctionKind::kCopyInitReg ||
              specialKind == SpecialFunctionKind::kMoveInitReg)
            isRegInit = true;
        }
      }

      auto [newSignature, bindings] =
          getTraitFunctionSignature(emitter, traitFn, selfType, trait);
      // Match against the transformed calling convention if the struct is
      // register-passable.
      LITSignatureType traitSignature = newSignature;
      if (regPassable) {
        newSignature =
            getRegisterPassableSignature(newSignature, selfType, isRegInit);
      }

      // Omit errors for certain special functions where the parser will
      // specifically verify their signatures if present.
      bool emitError = !llvm::is_contained({SpecialFunctionKind::kMoveInit,
                                            SpecialFunctionKind::kCopyInit,
                                            SpecialFunctionKind::kDel},
                                           SpecialFunctionInfo::getKind(name));

      OverloadSet ov(name, decls, std::move(bindings), node,
                     CallSyntax::kMethodCallSynthetic);
      PValue result = ov.filterOverloadSetForValueType(
          newSignature, emitError
                            ? function_ref<InflightDiag &(SMLoc)>(
                                  [&](SMLoc loc) -> InflightDiag & {
                                    return diag->attachNote(decl->getLoc());
                                  })
                            : nullptr);
      if (!result && emitError)
        allMatchFound = false;
      if (regPassable && result)
        regStubs.insert({{name, result.get()}, traitSignature});
    }
  }
  if (allMatchFound) {
    diag->abandon();
    diag.reset();
  } else {
    diag->attachNote(traitDecl.getLoc())
        << "trait '" << traitName << "' declared here";
    if (!parent.getInheritedFrom().empty()) {
      ASTDecl &parentDecl = emitter.getDeclResolver().getDeclForTypeSymbol(
          cast<TraitType>(parent.getInheritedFrom().front()).getSymbol());
      diag->attachNote(parentDecl.getLoc())
          << "inherited through '" << *parentDecl.getNameIfOperation()
          << "' here";
    }
    hadErrors = true;
  }

  if (hadErrors)
    return failure();
  if (regPassable)
    synthesizeRegisterTraitStubs(structDecl, shared, regStubs.takeVector());
  for (SpecialFunctionKind kind : specialFns)
    synthesizeSpecialFunction(structDecl, shared, kind);
  return success();
}

/// Given a decl for a struct or trait type, return true if this type conforms
/// to the specified trait type.  On failure, this may set 'diag' to an inflight
/// diagnostic that explains why this doesn't conform.  It can be reported or
/// abandoned based on the client's needs.
bool ASTDecl::doesNominalTypeConformsTo(TraitType trait,
                                        std::optional<InflightDiag> &diag,
                                        SharedState &shared) {
  assert((::isa<StructDeclOp, TraitDeclOp>(*this)) && "Invalid decl kind");

  if (failed(shared.declResolver->resolveFully(*this, getLoc())))
    return false; // Error emitted.

  ArrayRef<TypeLineageAttr> parentTypes;
  auto structOp = dyn_cast<StructDeclOp>(*this);
  if (structOp)
    parentTypes = structOp.getParentTypes();
  else
    parentTypes = cast<TraitDeclOp>(*this).getParentTypes();

  // Check if the type explicitly conforms to the trait.
  if (llvm::find_if(parentTypes, [trait](TypeLineageAttr type) {
        return type.getType() == trait;
      }) != parentTypes.end())
    return true;

  // Check to see if this is already literally this trait.
  ASTDecl *traitDecl = ASTType(trait).getDecl(shared);
  if (!traitDecl)
    return false; // Erroneous.

  // Self conformance.
  if (traitDecl == this)
    return true;

  // Only structs can implicitly conform to traits.
  if (!structOp)
    return false;

  // Check if the type *implicitly* conforms to the trait.
  SmallVector<TypeLineageAttr> newParentTypes =
      llvm::to_vector(structOp.getParentTypes());
  unsigned curNumParents = newParentTypes.size();
  StructEmitter::appendTraits(newParentTypes, traitDecl);
  for (TypeLineageAttr newParent :
       llvm::drop_begin(newParentTypes, curNumParents))
    if (failed(verifyConformance(*this, newParent, shared, diag)))
      return false;

  // If we succeeded, remember this so we don't check again.
  structOp.setParentTypes(newParentTypes);
  return true;
}
