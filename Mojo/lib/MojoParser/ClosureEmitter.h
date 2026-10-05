//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Closure Emission.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_CLOSUREEMITTER_H
#define KGEN_MOJOPARSER_CLOSUREEMITTER_H

#include "ExprNodes.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/MojoParser/SharedState.h"
#include "StructEmitter.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"

namespace M::KGEN::LIT {
// Naming schemes used for emitting closure related-struct.
inline constexpr llvm::StringLiteral kClosurePrefix = "closure$";
inline constexpr llvm::StringLiteral kClosureInflatedPrefix = "inflated$";
inline constexpr llvm::StringLiteral kClosureExtensionPrefix = "extension$";
inline constexpr llvm::StringLiteral kClosureDeviceTypeSuffix =
    "::__device_type";

class ClosureEmitter : public FunctionEmitter {
public:
  ClosureEmitter(SharedState &shared);

  /// Return true if \p type provably conforms to \p traitDecl.
  static bool provenConformsToTrait(ASTType type, ASTDecl *traitDecl,
                                    SharedState &shared,
                                    ArrayRef<ConstraintAttr> callerAssumptions);

  struct PromotedClosureSelfArg {
    Type type;
    ArgConvention convention;
  };

  /// Move `nestedFnDecl` into `storageStructDecl` as a method and wire its
  /// captured values to the storage struct fields. Returns the method decl.
  ASTDecl *
  liftClosureIntoMethod(ASTDecl &nestedFnDecl, ASTDecl &storageStructDecl,
                        PromotedClosureSelfArg selfArg,
                        ArrayRef<StructDefFieldAttr> concreteFieldDecls,
                        ArrayRef<Value> concreteFieldCaptures,
                        ArrayRef<CaptureConvention> captureConventions,
                        ArrayRef<Type> selfBoundFieldTypes, Location location);

  /// Promote a closure decl into `targetParent`. nullptr → nearest FileModuleOp
  /// (thin/stateless promotions).
  ASTDecl *
  promoteClosure(ASTDecl &nestedFnDecl,
                 ArrayRef<ParamDeclAttr> prependedParams = {},
                 std::optional<PromotedClosureSelfArg> selfArg = std::nullopt,
                 TriBool capturingOverride = TriBool::unknown(),
                 ASTDecl *targetParent = nullptr);

  /// Adapter overload for callsites that currently hold ParamDeclRefAttr.
  ASTDecl *
  promoteClosure(ASTDecl &nestedFnDecl,
                 ArrayRef<ParamDeclRefAttr> prependedParamRefs,
                 std::optional<PromotedClosureSelfArg> selfArg = std::nullopt,
                 TriBool capturingOverride = TriBool::unknown(),
                 ASTDecl *targetParent = nullptr);

  Value emitClosure(ASTDecl &moduleDecl, ASTDecl &nestedFnDecl,
                    ArrayRef<Capture> captures, Location location,
                    bool isCopyable, ArrayRef<ParamDeclRefAttr> paramCaptures);
  static ASTDecl *addCaptureValue(SharedState &shared, ASTDecl &closure,
                                  StringRef name, SMLoc location);

  static ASTDecl *addCaptureValue(ASTDecl &closure, SMLoc location,
                                  StringRef name, CaptureConvention capture,
                                  IREmitter &emitter,
                                  ASTDecl *signatureDecl = nullptr);
  LIT::StructType getInflatedClosureForFnSymbol(IREmitter &emitter, SMLoc loc,
                                                PValue fnPValue);
  bool isInflatedClosureForFnSymbol(PValue fnSymbol, LIT::StructType wrapper);

  /// Extend a source closure type constrained by one closure trait to a
  /// structurally compatible target closure trait, without changing the
  /// source's physical type, the function assume that the compatibility between
  /// `srcClosureInst` and `tgtClosureInst` are tested. Returns the extension
  /// struct type value, with the parameters the bridging thunk captured bound
  /// to the values they hold at the conversion site.
  PValue createParamClosureExtensionType(IREmitter &emitter,
                                         ASTExprAnd<CValue> srcTypeVal,
                                         TraitSymbolAttr srcClosureInst,
                                         TraitSymbolAttr tgtClosureInst);

  /// One parent trait that a synthesized closure conforms to, named by symbol.
  struct ClosureParent {
    /// Resolves \p symbol's requirement up front; \p traitFnName is
    /// empty for marker traits, which have none.
    ClosureParent(SharedState &shared, TraitSymbolAttr symbol,
                  StringRef traitFnName, ClosureMethod closureMethod);

    ClosureParent(TraitSymbolAttr symbol, FnTypeGeneratorType traitFnSig,
                  StringAttr witnessName, ClosureMethod closureMethod)
        : symbol(symbol), closureMethod(closureMethod),
          witnessName(witnessName), signature(traitFnSig) {}

    TraitSymbolAttr getSymbol() const { return symbol; }
    StringAttr getWitnessName() const { return witnessName; }

    SymbolRefAttr getSymbolRef() const { return symbol.getSymbol(); }
    StringAttr getFlattenedName() const { return symbol.getFlattenedName(); }

    bool isEmpty() const { return closureMethod == ClosureMethod::NONE; }
    ClosureMethod getClosureMethod() const { return closureMethod; }

    TraitDeclOp getTrait(SharedState &shared) const;

    FnTypeGeneratorType getSignature() const { return signature; }

    InlineLevel getInlineLevel() const { return inlineLevel; }

  private:
    /// Symbol of the parent trait; its declaration is looked up from this.
    TraitSymbolAttr symbol;
    /// closure method tag corresponding to the method this parent represents.
    ClosureMethod closureMethod;
    /// The name used for building conformance table.
    StringAttr witnessName;
    /// The requirement function signature, null for marker trait.
    FnTypeGeneratorType signature;
    /// The requirement's inline level, which the synthesized witness inherits.
    InlineLevel inlineLevel = InlineLevel::Automatic;
  };

  /// Bundles the IR artifacts produced by liftClosure.
  struct Closure {
    ASTDecl *structDecl;         ///< The closure storage struct.
    ASTDecl *promotedCallMethod; ///< The storage struct's `__call__` method.
    TypedAttr typeAttr;          ///< Bound closure storage struct type.
  };

private:
  MLIRContext *ctx;

  // Cached attributes and types.
  StringAttr selfName, copyName;

  /// Augment a synthesized struct with trivial lifecycle traits.
  void addTrivialClosureLifecycle(ASTDecl &structDecl,
                                  const ClosureParent &callParent);

  /// Construct the closure struct, lift the nested function into a method, and
  /// emit witness tables for all closure parents.
  Closure liftClosure(ASTDecl &moduleDecl, SMLoc smLoc,
                      SmallVector<ClosureParent> &closureParents,
                      SymbolRefAttr parentSymbolRef,
                      SmallVector<StructDefFieldAttr> &&fieldDecls,
                      SmallVector<Value> &&fieldCaptures,
                      SmallVector<CaptureConvention> &&fieldCaptureConventions,
                      SmallVector<ParamDeclAttr> &&allStructParams,
                      SmallVector<TypedAttr> &&structParamBindings,
                      StringAttr name, TypeConvention typeConvention,
                      SmallVector<Type> &&deviceCaptureFieldTypes,
                      bool capturesEncodable, ASTDecl &nestedFnDecl);

  /// Given the signature of a trait function, specialize it and add it to the
  /// struct as \p fnName.
  /// Returns
  /// (a) the new FnOp,
  /// (b) the parameters of the function minus the origins and remapped to
  /// reference struct parameters instead of indices
  /// (c) the result of the function, remapped to reference the struct
  /// parameters instead of indices.
  std::tuple<FnOp, ArrayRef<ParamDeclAttr>, Type>
  pushBackTraitFunctionImpl(FnTypeGeneratorType traitFnSignature,
                            ASTDecl &structDecl, bool synthetic,
                            StringAttr fnName, SpecialFunctionKind specialFnID,
                            InlineLevel inlineLevel);
  struct DevicePassablePopulators {
    llvm::function_ref<FailureOr<SymbolConstantAttr>(FnOp)> isConvertible;
    llvm::function_ref<FailureOr<SymbolConstantAttr>(FnOp)> isEncodable;
    llvm::function_ref<FailureOr<SymbolConstantAttr>(FnOp)> toDeviceType;
    llvm::function_ref<FailureOr<SymbolConstantAttr>(FnOp)> typeName;
    llvm::function_ref<TypedAttr()> deviceType;
  };
  /// Add DevicePassable conformance using callbacks that populate each
  /// interface member and return its witness.
  void
  addConformanceToDevicePassable(ASTDecl &structDecl,
                                 const DevicePassablePopulators &populators);
  /// Add DevicePassable conformance to closure storage (__storage), whose
  /// device_type reflects the device representations of its captures.
  void
  addStorageConformanceToDevicePassable(ASTDecl &structDecl,
                                        ArrayRef<Type> deviceCaptureFieldTypes,
                                        StringRef name);

  /// Look up a prelude trait used as a closure parent.
  ClosureParent getBuiltinParent(StringRef traitName, StringRef traitFnName,
                                 ClosureMethod closureMethod);

  /// We need to lazily resolve this builtin traits, since at the time closure
  /// emitter is constructed, they are not registered.
  ClosureParent getAnyParent() {
    if (anyParent.has_value())
      return *anyParent;
    return getBuiltinParent("AnyType", "", ClosureMethod::NONE);
  }
  ClosureParent getMoveParent() {
    if (moveParent.has_value())
      return *moveParent;
    return getBuiltinParent("Movable", "__init__", ClosureMethod::MOVE);
  }
  ClosureParent getDeinitableParent() {
    if (deinitableParent.has_value())
      return *deinitableParent;
    return getBuiltinParent("Deinitable", "__deinit__", ClosureMethod::DEL);
  }
  ClosureParent getRegisterPassableParent() {
    if (registerPassableParent.has_value())
      return *registerPassableParent;
    return getBuiltinParent("RegisterPassable", "", ClosureMethod::NONE);
  }
  ClosureParent getTrivialRegisterTypeParent() {
    if (trivialRegisterTypeParent.has_value())
      return *trivialRegisterTypeParent;
    return getBuiltinParent("TrivialRegisterPassable", "", ClosureMethod::NONE);
  }
  ClosureParent getCopyParent() {
    if (copyParent.has_value())
      return *copyParent;
    return getBuiltinParent("Copyable", "__init__", ClosureMethod::COPY);
  }
  ClosureParent getImplicitlyCopyableParent() {
    if (implicitlyCopyableParent.has_value())
      return *implicitlyCopyableParent;
    return getBuiltinParent("ImplicitlyCopyable", "", ClosureMethod::NONE);
  }

  /// AnyType is the base metatype for all types.
  std::optional<ClosureParent> anyParent;
  /// Movable trait is a parent of all closures.
  std::optional<ClosureParent> moveParent;
  /// Deinitable trait is a parent of all closures.
  std::optional<ClosureParent> deinitableParent;
  /// RegisterPassable marks the type as register passable.
  std::optional<ClosureParent> registerPassableParent;
  /// TrivialRegisterPassable marks the state as trivially register passable.
  std::optional<ClosureParent> trivialRegisterTypeParent;
  /// Copy trait is a parent of some closures.
  std::optional<ClosureParent> copyParent;
  /// ImplicitlyCopyable trait is a parent of some closures. It has no defining
  /// methods.
  std::optional<ClosureParent> implicitlyCopyableParent;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_CLOSUREEMITTER_H
