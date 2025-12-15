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

/// Top level types are the types of the Closure Wrapper function pointer
/// fields.
struct TopLevelTypes {
  Type callFuncFieldType;
  Type copyFuncFieldType;
  Type delFuncFieldType;
};

class ClosureEmitter : public FunctionEmitter {
public:
  ClosureEmitter(SharedState &shared);

  /// Generate a Closure Wrapper Struct, a struct that contains an opaque
  /// pointer to the underlying Closure Implementation instance.
  StructDeclOp
  createClosureWrapperStructDecl(ASTDecl &moduleDecl, StringAttr name,
                                 FnTypeGeneratorType signatureType,
                                 SMLoc nestedFunctionOrTypeLocation);

  /// Generate a Parametric Closure Wrapper Struct, a struct that contains a
  /// parametric field. Both the field and the struct must conform to the
  /// associated closure trait characterized by the signature of the closure.
  ASTDecl *createClosureTrait(ASTDecl &moduleDecl, StringAttr name,
                              FnTypeGeneratorType signatureType,
                              SMLoc nestedFunctionOrTypeLocation,
                              InlineLevel inlineLevel);

  /// Generate a Closure Implementation Struct, a struct that contains the
  /// capture list.
  StructDeclOp replaceNestedFunctionWithClosureImplStructDecl(
      ASTDecl &moduleDecl, ArrayRef<Capture> captures,
      ArrayRef<ParamDeclRefAttr> paramCaptures, ASTDecl &nestedfnDecl,
      FnTypeGeneratorType wrapperSigGen);

  /// Generate an initializer on the ClosureWrapper that accepts a ClosureImpl
  /// instance.
  FnOp createWrapperInitWithImpl(ASTDecl &moduleDecl,
                                 StructDeclOp closureWrapper,
                                 StructDeclOp closureImpl, SMLoc location);
  Value emitClosureOp(ASTDecl &moduleDecl, ASTDecl &nestedFnDecl,
                      ArrayRef<Capture> captures, StructDeclOp wrapper,
                      TraitDeclOp trait, Location location, bool isCopyable);
  static ASTDecl *addCaptureValue(SharedState &shared, ASTDecl &closure,
                                  StringRef name, SMLoc location);

  static ASTDecl *addCaptureValue(ASTDecl &closure, SMLoc location,
                                  StringRef name, CaptureConvention capture,
                                  IREmitter &emitter,
                                  ASTDecl *signatureDecl = nullptr);
  ASTDecl *getOrCreateClosureTrait(FnTypeGeneratorType key,
                                   llvm::function_ref<ASTDecl *()> creation);
  /// Given a name and a trait decl, generate a struct that conforms to the
  /// trait and has a single field that also conforms to that same trait. For
  /// example, if the trait is:
  /// trait MyTrait:
  ///    fn doSomething(self, x:Int):
  ///       ...
  /// This method will generate a struct:
  /// struct MyTraitWrapper[T: MyTraitWrapper](MyTraitWrapper):
  ///    var field: T
  ///    fn doSomething(self, x:Int):
  ///       self.field.doSomething(x)
  /// This is useful in the context of emitting closures because in the case of
  /// closures "T" is an abstract type (ClosureType). Wrapping the closure
  /// instance in a struct renders it eligible to be handled properly by our
  /// check-lifetimes pass.
  ASTDecl *createStructWrapper(ASTDecl &moduleDecl, StringRef name,
                               ASTDecl &traitDecl, SMLoc location,
                               TypeConvention typeConvention, bool isCopyable,
                               bool isStateless);

private:
  MLIRContext *ctx;

  // Cached attributes and types.
  StringAttr selfName, otherName, ptrToImplName, dtorFieldAttr;
  StringAttr copyFieldAttr, callFieldAttr, callMethodAttr;
  PointerType opaquePtrType;

  /// Given a closure wrapper, collect the top level function types.
  TopLevelTypes collectTopLevelFunctionTypes(StructDeclOp closureWrapper);

  /// Synthesize the constructor for a closure wrapper struct from a bare
  /// function pointer of the same function signature.
  void synthesizeWrapperFnPtrCtor(ASTDecl &decl, ASTType selfType,
                                  FnTypeGeneratorType sig);

  /// Underlying implementation of `augmentWitnessTablesToConformTo` and
  /// `isCompatibleWith`.
  LogicalResult checkStructCompatibility(ASTType structType, ASTDecl *traitDecl,
                                         bool emitRebind);

public:
  /// If the wrapper conforms to a trait that is compatible with the desired
  /// trait, emit a rebind. For example, suppose we have a parameter P with a
  /// closure metatype defined by `fn(x:Int) -> Int`. We should be able to bind
  /// a struct wrapper type W to P if W conforms to the trait `fn(z:Int) ->
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
                                bool isCopyable, TypeConvention typeConvention,
                                bool isStateless);

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
  TypedAttr
  addWitnessTablesToClosure(ASTDecl &moduleDecl, SMLoc smLoc, FnOp parent,
                            ClosureType closureType,
                            SmallVector<ClosureParent> &closureParents,
                            SymbolRefAttr parentSymbolRef);

  /// Given a trait function, specialize it and add it to the struct.
  /// Returns
  /// (a) the new FnOp,
  /// (b) the parameters of the function minus the origins and remapped to
  /// reference struct parameters instead of indices
  /// (c) the result of the function, remapped to reference the struct
  /// parameters instead of indices.
  std::tuple<FnOp, ArrayRef<ParamDeclAttr>, Type>
  pushBackTraitFunctionImpl(FnOp traitFnOp, ASTDecl &structDecl);
  /// Given the wrapper struct, add to the conformance table to enable the
  /// closure to be used with kernel functions
  void addConformanceToDevicePassable(ASTDecl &structDecl,
                                      StructFieldOp devicePassedField,
                                      ParamDeclAttr impl,
                                      ParamDeclAttr originSet);
  void addConformanceToExtern(ASTDecl &moduleDecl, ASTDecl &structDecl,
                              FuncTypeGeneratorType originalSignature);
  /// UnknownDestructibility is the base metatype for all types.
  ClosureParent unknownDestructibility;
  /// Movable trait is a parent of all closures. Cache its defining op.
  ClosureParent moveParent;
  /// Anytype trait is a parent of all closures. Cache its defining op.
  ClosureParent anyParent;
  /// Copy trait is a parent of some closures. Cache its defining op.
  ClosureParent copyParent;
  /// ImplicitlyCopyable trait is a parent of some closures. It has no defining
  /// methods.
  ClosureParent implicitlyCopyableParent;
  /// Closure traits live in the top level module. This cache guards against
  /// emitting duplicates.
  DenseMap<Type, ASTDecl *> closureTraitCache;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_CLOSUREEMITTER_H
