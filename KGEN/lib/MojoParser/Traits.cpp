//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Traits.h"
#include "MojoUtils.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/CallEmission.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/ExprEmitter.h"
#include "KGEN/MojoParser/ExprNodes.h"
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
                          ASTType structSelfType) {
  LITSignatureType signature = traitFn.getFullSignature();
  SmallVector<TypedAttr> params;
  ArrayRef<Type> paramTypes = signature.getParamTypes();

  // Add trait's MT replacement.
  // FIXME(generics): We aren't propagating metatypes into pointer types, so
  // just pass a generic metatype here.
  auto anyRegTypeType = TypeType::get(traitFn.getContext());
  params.push_back(TypeConstantAttr::get(anyRegTypeType, anyRegTypeType));
  // Add trait's T replacement.
  params.push_back(TypeConstantAttr::get(structSelfType, anyRegTypeType));
  ParameterEvaluator evaluator(params);
  auto bindings = ParamBindings::getForDeclaredType(
      emitter.declScope, emitter.shared, structSelfType.getMetaType());
  for (Type type : paramTypes.drop_front(2)) {
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
                                                     bool trivial,
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
      // Don't modify a byref result of an unrelated type.
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
      resultType = VariantType::get({*variant.getTypes().begin(), selfType});

      // The result is always owned because it includes a variant containing an
      // error.
      fnEffects.setOwnedRegisterResult();
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

  PogsAttr oldArgListAttrs = traitSig.getArgListAttrs();
  auto oldArgNames = oldArgListAttrs.getNames();
  auto oldPassingKinds = oldArgListAttrs.getPassingKinds();
  if (replacedResult) {
    if (traitSig.hasInitSelfArg()) {
      oldArgNames = oldArgNames.drop_front();
      oldPassingKinds = oldPassingKinds.drop_front();
    } else if (traitSig.hasMemoryOnlyResult()) {
      oldArgNames = oldArgNames.drop_back();
      oldPassingKinds = oldPassingKinds.drop_back();
    }
  }

  PogsAttr newArgListAttrs =
      oldArgListAttrs.cloneWith(oldArgNames, oldPassingKinds);
  auto metadata = FnMetadataAttr::get(
      newArgListAttrs, traitSig.getParamListAttrs(), numImplicitLifetimeDecls);
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
  Builder b(shared.getContext());
  for (auto [i, type, name] :
       llvm::enumerate(memSig.getParamTypes(), memSig.getParamNames())) {
    // The parameter names are derived from the decl name.
    paramDecls.push_back(ParamDeclAttr::get(
        name.empty() ? b.getStringAttr("i" + Twine(i)) : name, type));
  }

  // Synthesize the method inside the struct.
  auto [thunk, _] = StructEmitter(shared).synthesizeMethodInStruct(
      name, paramDecls, memSig.getParamListAttrs(), memSig.getArguments(),
      memSig.getArgConventions(), memSig.getArgListAttrs(),
      memSig.getResultType(), structDecl, SpecialFunctionInfo::getKind(name),
      memSig.getFnEffects(), "_thunk", /*ifMissing=*/true);
  if (!thunk)
    return;
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (DebugInfo::DIScopeAttr spAttr = thunk.getLocScope())
    diScopeGuard = shared.diBuilder->pushScopeGuard(spAttr);

  // Always inline the thunk. The calling convention conversion overhead is
  // guaranteed to be optimized away.
  thunk.setInlineLevel(InlineLevel::AlwaysNoDebug);

  // Now prepare to emit the call to the register-passable method.
  ExprEmitter emitter(shared, structDecl, EC_Trait);
  emitter.builder = OpBuilder::atBlockBegin(thunk.getBody());

  // The callee is partially bound, containing only its parent struct
  // parameters. Bind the rest of them here.
  SmallVector<TypedAttr> bindSigInputs{callee};
  for (ParamDeclAttr param : paramDecls)
    bindSigInputs.push_back(ParamDeclRefAttr::get(param));
  callee = ParamOperatorAttr::get(POC::BindSignature, bindSigInputs);

  SignatureType calleeSig = cast<LITSignatureType>(callee.getType());

  // Construct the call operands from the function block arguments. Ensure
  // keyword-only arguments are specified accordingly.
  SyntheticNode node(structDecl.getLoc());
  SmallVector<FuncOperand> posOperands;
  SmallDenseMap<StringAttr, FuncOperand> kwOperands;
  bool hasLegacyInitSelfArg = false;
  for (auto [arg, kind, conv, name] :
       llvm::zip(thunk.getArguments(), memSig.getArgPassingKinds(),
                 memSig.getArgConventions(), memSig.getArgNames())) {
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
      continue; // Ignore this, it will be assigned to later.

    case ArgConvention::ByRef:
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
    if (kind == PassingKind::KwOnly)
      kwOperands.insert({name, {value, node}});
    else
      posOperands.push_back({value, node});
  }

  // Allocate the value dest for the call. Set the value dest to the result
  // slot, if there is one, otherwise provide the expected rvalue type.
  ValueDest dest(EC_Trait);
  bool hasRegisterResult = false;
  if (memSig.hasMemoryOnlyResult())
    dest = ValueDest(MLValue(thunk.getArguments().back()), EC_Trait);
  else if (memSig.hasInitSelfArg() && hasLegacyInitSelfArg)
    dest = ValueDest(MLValue(thunk.getArgument(0)), EC_Trait);
  else
    hasRegisterResult = true;

  CValue callResult = emitter.emitCallUnchecked(
      PValue(callee), CallOperands(posOperands, &kwOperands), dest, node);
  if (!callResult)
    return;

  // If the callee is async, then await the result.
  if (memSig.isAsync()) {
    ValueDest dest(EC_Trait);
    callResult =
        emitter.emitNamedMethodCall("__await__", FuncOperand{callResult, node},
                                    dest, CallSyntax::kMethodCall, node);
    if (!callResult)
      return;
  }

  // Emit the function return. It's just a none return if the function has a
  // result slot.
  // FIXME: handle async
  ImplicitLocOpBuilder builder(shared.translateLocation(structDecl.getLoc()),
                               *emitter.builder);
  Value retVal;
  if (hasRegisterResult) {
    retVal = emitter.emitSRValue({callResult, node}, EC_Trait);
  } else {
    retVal =
        builder.create<ParamConstantOp>(KGEN::NoneAttr::get(b.getContext()));
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
    auto [dtor, _] = gen.synthesizeMethodInStruct(
        "__del__", selfRefType, ArgConvention::OwnedInMem,
        PogsAttr::get(shared.getContext(), empty, PassingKind::PosOnly),
        shared.getNoneType(), structDecl, kind, FnEffects(), "_thunk",
        /*ifMissing=*/true);
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
        structDecl.getSelfType().getRefForArgument("existing", isMut);
    auto [ctor, _] = gen.synthesizeMethodInStruct(
        name, {selfRefType, existingType},
        {ArgConvention::InitSelf, existingConv},
        PogsAttr::get(shared.getContext(), {empty, empty},
                      {PassingKind::PosOnly, PassingKind::PosOnly}),
        shared.getNoneType(), structDecl, kind, FnEffects(), "_thunk",
        /*ifMissing=*/true);
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
      Value(b.create<ParamConstantOp>(NoneAttr::get(b.getContext()))));
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
  ASTType selfType = structDecl.getSelfType();

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
          getTraitFunctionSignature(emitter, traitFn, selfType);
      // Match against the transformed calling convention if the struct is
      // register-passable.
      LITSignatureType traitSignature = newSignature;
      if (regPassable) {
        newSignature = getRegisterPassableSignature(newSignature, selfType,
                                                    rpTrivial, isRegInit);
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
