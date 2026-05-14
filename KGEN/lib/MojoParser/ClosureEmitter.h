//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Closure Emission.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_CLOSUREEMITTER_H
#define KGEN_MOJOPARSER_CLOSUREEMITTER_H

#include "ExprNodes.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/SharedState.h"
#include "StructEmitter.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"

namespace M::KGEN::LIT {
class TypeCheckedFnSignature;
class TypeCheckedParamList;
struct AuxiliaryParameters;
using AliasSubstitutions = llvm::MapVector<mlir::StringAttr, TypedAttr>;
struct AdapteeParts {
  AliasSubstitutions aliasSubstitutions;
  DenseMap<StringAttr, TypedAttr> adapteeTypeMap;
  SmallVector<TypedAttr> fnLevelBindings;
  // flag to store when the callee returns in-register but the trait signature
  // expects a memory-only result.
  bool needsResultConversion = false;
};

/// Information about a closure parameter's external reference that needs
/// a where clause constraint. Contains the closure parameter name and the
/// alias representing the external reference.
struct ClosureExternalRef {
  /// The closure-typed parameter (e.g., "C" in `C: def(T) -> T`)
  ParamDeclAttr closureParam;
  /// The alias in the closure trait representing the external param reference
  AliasDeclOp aliasOp;
};

class ClosureEmitter : public FunctionEmitter {
public:
  ClosureEmitter(SharedState &shared);

  /// Collect external parameter references from closure-typed parameters.
  ///
  /// For each parameter constrained by a closure trait, this examines the
  /// alias ops in the trait body. These aliases represent external parameter
  /// references from the outer scope where the closure was defined.
  ///
  /// Example:
  ///   def useIt[T: Coord, C: def(T) -> T](impl: C, arg: T)
  ///
  /// The closure trait for `C` will have an alias `T` in its body. This
  /// function collects that alias along with the closure param `C`.
  /// The caller can then generate: `where _type_is_eq_parse_time[T, C.T]()`
  ///
  /// \param closureParam The closure-typed parameter to examine
  /// \param refs Output vector for collected external references
  void collectClosureExternalRefs(ParamDeclAttr closureParam,
                                  SmallVectorImpl<ClosureExternalRef> &refs);

  /// Iterate over closure traits in a TraitType and invoke a callback for each.
  void processClosureTraits(TraitType traitType,
                            std::function<void(TraitDeclOp)> const &callback);

  /// Return true if \p type is a compiler-synthesized closure type.
  static bool isClosureType(SharedState &shared, Type type);

  /// Generate a Parametric Closure Wrapper Struct, a struct that contains a
  /// parametric field. Both the field and the struct must conform to the
  /// associated closure trait characterized by the signature of the closure.
  ASTDecl *createClosureTrait(ASTDecl &moduleDecl,
                              FnTypeGeneratorType signatureType,
                              FnTypeGeneratorType key,
                              unsigned numPrependedCaptures,
                              SMLoc nestedFunctionOrTypeLocation);

  /// Promote a stateless closure decl to a top-level function decl.
  ASTDecl *
  promoteStatelessClosure(ASTDecl &nestedFnDecl,
                          ArrayRef<ParamDeclRefAttr> paramCaptures = {});

  Value emitClosureOp(ASTDecl &moduleDecl, ASTDecl &nestedFnDecl,
                      ArrayRef<Capture> captures, TraitDeclOp trait,
                      Location location, bool isCopyable,
                      FnTypeGeneratorType closureSig,
                      ArrayRef<ParamDeclRefAttr> paramCaptures);
  static ASTDecl *addCaptureValue(SharedState &shared, ASTDecl &closure,
                                  StringRef name, SMLoc location);

  static ASTDecl *addCaptureValue(ASTDecl &closure, SMLoc location,
                                  StringRef name, CaptureConvention capture,
                                  IREmitter &emitter,
                                  ASTDecl *signatureDecl = nullptr);
  /// Maps a raw closure signature to the canonical key and captured-parameter
  /// count used for closure-trait uniquing and name synthesis.
  static std::pair<FnTypeGeneratorType, unsigned>
  getClosureTraitKey(FnTypeGeneratorType rawSignature);
  ASTDecl *getOrCreateClosureTrait(FnTypeGeneratorType key,
                                   llvm::function_ref<ASTDecl *()> creation);
  /// Given a name and a trait decl, generate a struct that conforms to the
  /// trait and has a single field that also conforms to that same trait. For
  /// example, if the trait is:
  /// trait MyTrait:
  ///    def doSomething(self, x:Int):
  ///       ...
  /// This method will generate a struct:
  /// struct MyTraitWrapper[T: MyTraitWrapper](MyTraitWrapper):
  ///    var field: T
  ///    def doSomething(self, x:Int):
  ///       self.field.doSomething(x)
  /// This is useful in the context of emitting closures because in the case of
  /// closures "T" is an abstract type (ClosureType). Wrapping the closure
  /// instance in a struct renders it eligible to be handled properly by our
  /// check-lifetimes pass.
  ASTDecl *createStructWrapper(ASTDecl &moduleDecl, StringRef name,
                               ASTDecl &traitDecl, SMLoc location,
                               TypeConvention typeConvention, bool isCopyable,
                               bool isStateless, FnTypeGeneratorType sig = {});

  /// Given a trait decl and a function signature, generate a struct that can
  /// wrap a function pointer to be used as a closure.
  ASTDecl *createFnStructWrapper(ASTDecl &moduleDecl, ASTDecl &traitDecl,
                                 FnTypeGeneratorType signatureType,
                                 SMLoc location);
  Type getConcreteClosureWrapperTypeForFnSymbol(ASTDecl &declScope, SMLoc loc,
                                                PValue fnPValue);

private:
  MLIRContext *ctx;

  // Cached attributes and types.
  StringAttr selfName, copyName;

  /// Underlying implementation of `augmentWitnessTablesToConformTo` and
  /// `isCompatibleWith`.
  LogicalResult checkStructCompatibility(ASTType structType, ASTDecl *traitDecl,
                                         bool emitRebind);

  /// Synthesize an adaptor function that rebinds the closure wrapper's
  /// __call__ from auxiliary parameters to trait aliases, then add the
  /// conformance witness table.
  void buildCallAdaptorAndAddWitness(StructDeclOp structDeclOp,
                                     ASTDecl &structDecl,
                                     TraitDeclOp traitDeclOp, FnOp traitCallFn,
                                     FnOp structCallFn,
                                     const AdapteeParts &adapteeParts);

public:
  /// If the wrapper conforms to a trait that is compatible with the desired
  /// trait, emit a rebind. For example, suppose we have a parameter P with a
  /// closure metatype defined by `def(x:Int) -> Int`. We should be able to bind
  /// a struct wrapper type W to P if W conforms to the trait `def(z:Int) ->
  /// Int`. This will require a rebind though because of the differences in
  /// argument names.
  LogicalResult augmentWitnessTablesToConformTo(ASTType structType,
                                                ASTDecl *closureTrait);

  /// Checks if the wrapper struct type conforms to a trait that is compatible
  /// with the desired trait.
  LogicalResult isCompatibleWith(ASTType structType, ASTDecl *traitDecl);

  struct ClosureParent {
    ClosureParent(StringRef name, StringRef fnName, ClosureMethod closureMethod)
        : traitName(name), traitFnName(fnName), closureMethod(closureMethod) {}
    ClosureParent(TraitDeclOp trait, FnOp definingOp,
                  ClosureMethod closureMethod)
        : traitFnName(definingOp ? *definingOp.getSourceName() : ""),
          trait(trait), definingFn(definingOp), closureMethod(closureMethod) {}
    TraitDeclOp getTrait(ASTDecl &moduleDecl);
    FnOp getDefiningOp(ASTDecl &moduleDecl);
    SymbolRefAttr getSymbolRef(ASTDecl &moduleDecl);
    StringRef getDefiningOpName() const { return traitFnName; }
    StringAttr getFullSymbolName(ASTDecl &moduleDecl);
    bool isEmpty() const { return closureMethod == ClosureMethod::NONE; }
    ClosureMethod getClosureMethod() const { return closureMethod; }

  private:
    StringRef traitName;
    StringRef traitFnName;
    /// The parent definition
    TraitDeclOp trait;
    /// all closure parents have a single defining function.
    FnOp definingFn;
    /// symbol of the trait.
    SymbolRefAttr sym;
    /// full symbol name as string
    StringAttr fullSymbolName;
    /// closure method tag corresponding to the method this parent represents.
    ClosureMethod closureMethod;
  };

  TraitType getWrapperTraitType(ASTDecl &traitDecl, ASTDecl &moduleDecl,
                                bool isCopyable, TypeConvention typeConvention);

  /// Append a deterministic suffix encoding the conformance traits present in
  /// \p wrapperTraitType.
  void enumerateWrapperTraits(SmallVectorImpl<char> &out,
                              TraitType wrapperTraitType, ASTDecl &moduleDecl);

  /// This is `isEqualCanon` with one relaxation: parameters
  /// in the leading "before-`+`" region of the pog list (i.e.
  /// `PassingKind::Inferred`) are not user-bindable, so their names are
  /// arbitrary disambiguators and may differ.
  static bool isTypeRebindableTo(FuncTypeGeneratorType from,
                                 FuncTypeGeneratorType to);

private:
  /// Given a name, a list of builtin parent traits (like "Movable" for
  /// example), a location, and a populate method, return a trait declaration
  /// that inherits from the parent and contains the methods added to the
  /// function list populated by the populate method.
  std::pair<TraitDeclOp, ASTDecl *> createTraitOp(
      ASTDecl &moduleDecl, StringAttr name, SmallVector<ClosureParent> &parents,
      SMLoc nestedFunctionOrTypeLocation,
      llvm::function_ref<
          void(ASTDecl &traitDecl,
               DenseSet<std::pair<StringAttr, StringAttr>> &functions)>
          populateTrait);
  /// Generate a witness table for a closure op.
  TypedAttr addWitnessTablesToClosure(
      ASTDecl &moduleDecl, SMLoc smLoc, FnOp parent, ClosureType closureType,
      SmallVector<ClosureParent> &closureParents, SymbolRefAttr parentSymbolRef,
      llvm::MapVector<StringAttr, Type> const &aliases);

  /// Given a trait function, specialize it and add it to the struct.
  /// Returns
  /// (a) the new FnOp,
  /// (b) the parameters of the function minus the origins and remapped to
  /// reference struct parameters instead of indices
  /// (c) the result of the function, remapped to reference the struct
  /// parameters instead of indices.
  std::tuple<FnOp, ArrayRef<ParamDeclAttr>, Type>
  pushBackTraitFunctionImpl(FnOp traitFnOp, ASTDecl &structDecl,
                            bool synthetic = true, StringAttr customName = {});
  /// Given the wrapper struct, add to the conformance table to enable the
  /// closure to be used with kernel functions
  void addConformanceToDevicePassable(ASTDecl &structDecl,
                                      StructFieldOp devicePassedField,
                                      ParamDeclAttr impl,
                                      ParamDeclAttr originSet);
  /// Validate copy/move/del symbols and build a MemSymbolTripleAttr.
  /// Emits errors if required symbols are missing for the given convention.
  MemSymbolTripleAttr
  validateAndBuildTriple(TypedAttr copy, TypedAttr move, TypedAttr del,
                         CaptureConvention convention, const Capture &capture,
                         UnitAttr &isMove, ASTDecl &nestedFnDecl);

  /// Build a MemSymbolTripleAttr for capturing a concrete StructType value.
  /// Returns {triple, highestConvention}. Returns {nullptr, Unspecified} on
  /// error.
  std::pair<MemSymbolTripleAttr, TypeConvention>
  buildStructCaptureInfo(StructType structType, const Capture &capture,
                         CaptureConvention convention,
                         TypeConvention requestedConvention, UnitAttr &isMove,
                         ASTDecl &nestedFnDecl);

  /// Build a MemSymbolTripleAttr for capturing a generic ParamType value.
  /// Uses GetWitnessAttr to reference copy/move/del from the trait constraint.
  /// Returns {nullptr, Unspecified} on error.
  std::pair<MemSymbolTripleAttr, TypeConvention>
  buildParamCaptureInfo(ParamType paramType, const Capture &capture,
                        CaptureConvention convention,
                        TypeConvention requestedConvention, UnitAttr &isMove,
                        ASTDecl &nestedFnDecl, ASTDecl &moduleDecl);

  /// AnyType is the base metatype for all types.
  ClosureParent anyParent;
  /// Movable trait is a parent of all closures. Cache its defining op.
  ClosureParent moveParent;
  /// ImplicitlyDestructible trait is a parent of all closures. Cache its
  /// defining op.
  ClosureParent implicitlyDestructibleParent;
  /// RegisterPassable marks the type as register passable.
  ClosureParent registerPassableParent;
  /// TrivialRegisterPassable marks the state as trivially register passable.
  ClosureParent trivialRegisterTypeParent;
  /// Copy trait is a parent of some closures. Cache its defining op.
  ClosureParent copyParent;
  /// ImplicitlyCopyable trait is a parent of some closures. It has no defining
  /// methods.
  ClosureParent implicitlyCopyableParent;
  /// Closure traits live in the top level module. This cache guards against
  /// emitting duplicates.
  DenseMap<Type, ASTDecl *> closureTraitCache;

  /// Mapping from each known parent trait's SymbolRefAttr to
  /// a fixed ordinal used for readable mangling
  std::optional<DenseMap<SymbolRefAttr, unsigned>> parentOrdinals;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_CLOSUREEMITTER_H
