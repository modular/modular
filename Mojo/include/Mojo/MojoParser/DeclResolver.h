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
// Declaration parsing and name binding logic.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_DECLRESOLVER_H
#define KGEN_MOJOPARSER_DECLRESOLVER_H

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/LITDialect/SpecialFunctions.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/IRValues.h"
#include "Mojo/MojoParser/Lexer.h"
#include "Mojo/MojoParser/SharedState.h"
#include "Support/RCRef.h"

namespace M::KGEN {
class ParamDeclAttr;
class StructGeneratorOp;
enum class PassingKind : uint32_t;
} // namespace M::KGEN

namespace M::KGEN::LIT {
class AliasDeclOp;
struct UnresolvedWildcardImport;
class FileModuleOp;
class FnOp;
class PackageOp;
class ParserBase;
class SharedState;
class UnresolvedImportOp;
class ImportOp;
class EnumCaseOp;
class StructDeclOp;
class StructFieldOp;
class TraitDeclOp;
class ExtensionDeclOp;
struct ParsedArgument;
struct LambdaNode;
class IREmitter;
class ExprDest;
class BaseDLValue;

//===----------------------------------------------------------------------===//
// DeclResolver
//===----------------------------------------------------------------------===//

class DeclResolver : public SharedStateUser {
public:
  DeclResolver(SharedState &state);
  ~DeclResolver();

  //===--------------------------------------------------------------------===//
  // Decl Constructors
  //===--------------------------------------------------------------------===//

  /// Add a new declaration that needs to be resolved.
  ASTDecl &addDecl(Operation *operation, llvm::SMLoc loc, StringAttr baseName,
                   ASTDecl *parentDecl, LexerCursor cursor,
                   LexerCursor endCursor, ssize_t indentation);
  IRASTDecl &addDecl(CValue value, llvm::SMLoc loc, StringAttr baseName,
                     ASTDecl *parentDecl, LexerCursor cursor,
                     LexerCursor endCursor, ssize_t indentation);

  /// Add a declaration that was loaded from bytecode.
  ASTDecl &addBytecodeDecl(Operation *op, StringAttr baseName,
                           ASTDecl *parentDecl, DeclResolvedness resolvedness);

  /// Add a declaration that is already fully resolved.
  ASTDecl &addFullyResolvedDecl(Operation *operation, StringAttr baseName,
                                llvm::SMLoc loc, ASTDecl *parentDecl);
  IRASTDecl &addFullyResolvedDecl(CValue value, StringAttr baseName,
                                  llvm::SMLoc loc, ASTDecl *parentDecl);

  /// Create an anonymous scope for the parameters and arguments of a function
  /// type. The scope is fully resolved and is not installed in the parent's
  /// name table.
  FunctionTypeASTDecl &createFunctionTypeDecl(llvm::SMLoc loc,
                                              ASTDecl &parentDecl);

  /// Add a declaration that represents an erroneous declaration. The generated
  /// decl is treated as fully resolved, and in an error state. If `unlisted`
  /// is set, the decl is not registered in the parent's name table.
  ASTDecl &addErroneousDecl(StringRef baseName, llvm::SMLoc loc,
                            ASTDecl *parentDecl, bool unlisted = false);

  /// Create a witness declaration and install it in the trait's scope.
  WitnessASTDecl &
  createWitnessDecl(WitnessASTDecl::UnresolvedDecls unresolvedDecls,
                    TraitSymbolAttr traitSymbol, llvm::SMLoc loc,
                    ASTDecl &traitDecl, StringAttr name);

  /// Attach a declaration to its parent's name table.  For use with
  /// `makeUnlistedDecl`.
  void attachDeclToParentNameTable(ASTDecl *decl, StringAttr name);

  /// Remove an unreachable declaration and its owned children.
  void eraseDecl(ASTDecl &decl);

  /// Remove a set of unreachable declarations and their owned children. O(N)
  /// for an N-decl module, regardless of how many decls are removed.
  void eraseDecls(llvm::ArrayRef<ASTDecl *> decls);

  /// Replace a resolved declaration with a new declaration for \p value.
  ///
  /// Use this when resolving a declaration produces a value with a different
  /// representation from the declaration's original IR. For example,
  /// promoting a nested function with bound parameters moves its `FnOp` to a
  /// new `DefASTDecl`; the source binding must then refer to the bound function
  /// literal rather than leave a `DefASTDecl` holding a value.
  ///
  /// The replacement retains the old declaration's resolution state and every
  /// name-table entry, including aliases, and takes ownership of any children
  /// not already transferred elsewhere. The old declaration is retired, so
  /// callers must not access it afterwards.
  ASTDecl &replaceDecl(ASTDecl &decl, CValue value);

  /// Register a decl's symbol in the MLIR symbol table and declForTypeSymbol
  /// map, without adding it to the parent's declsInScope. This allows the
  /// decl to be found via ModuleType::getDecl() but not via name lookup.
  void registerDeclSymbol(ASTDecl *decl);

  // Adds decl to its parent ASTDecl under the name aliasName.
  // This assumes you've already used attachDeclToParentNameTable; use this
  // when you want it known to its parent as an *additional* name.
  // For example, an extension named "extension:Spaceship_3" might want to be
  // known to its parent as both "extension:Spaceship" (to help lookups which
  // look for all extensions for a given type) and also "extension:" (to make
  // importing all extensions easier).
  // TODO(MOCO-522): Centralize into arcana doc.
  void aliasDeclInParent(ASTDecl *decl, StringAttr aliasName);

  /// Register a newly added extension decl under its parent's aggregate
  /// "extension:" name.
  void registerExtensionDecl(ASTDecl &extensionDecl);

  /// Attach a declaration to a trait composition's decl. This does not modify
  /// any existing parent-child relationships of the `childDecl`. It merely adds
  /// it to the trait composition's declsInScope map.
  void attachDeclToTraitCompositionDecl(ASTDecl *traitDecl,
                                        TraitSymbolAttr witnessFor,
                                        SmallVector<ASTDecl *> &&childDecls,
                                        StringAttr name);

public:
  //===--------------------------------------------------------------------===//
  // Import Resolution
  //===--------------------------------------------------------------------===//

  /// Add a pre-existing set of declarations as children of the specified
  /// context, using the provided alias name (which may differ from that of the
  /// decl).
  void aliasDecls(ArrayRef<ASTDecl *> decls, StringAttr name,
                  llvm::SMLoc aliasLoc, ASTDecl &context);
  /// Try to add a pre-existing set of declarations as children of the specified
  /// context, using the provided alias name (which may differ from that of the
  /// decl). Does not error on failure, but returns a failure result.
  LogicalResult tryAliasDecls(ArrayRef<ASTDecl *> decls, StringAttr name,
                              llvm::SMLoc aliasLoc, ASTDecl &context);
  /// Add a pre-existing set of declarations imported from the given module, as
  /// children of the specified context, using the provided alias name (which
  /// may differ from that of the decl).
  LogicalResult aliasImportDecls(ArrayRef<ASTDecl *> decls, StringAttr name,
                                 StringAttr declName, ImportPathAttr moduleName,
                                 llvm::SMLoc aliasLoc, ASTDecl &context,
                                 bool allowMultipleWithSameName);

private:
  /// Allocate a declaration without adding it to a parent name table.
  ASTDecl &createUnlistedDecl(Operation *operation, llvm::SMLoc loc,
                              ASTDecl *parentDecl, LexerCursor cursor,
                              LexerCursor endCursor, ssize_t indentation);
  IRASTDecl &createUnlistedDecl(CValue value, llvm::SMLoc loc,
                                ASTDecl *parentDecl, LexerCursor cursor,
                                LexerCursor endCursor, ssize_t indentation);

  /// Add a pre-existing set of declarations, which may optionally be imported
  /// from a given module, as children of the specified context, using the
  /// provided alias name (which may differ from that of the decl).
  LogicalResult aliasDeclsImpl(ArrayRef<ASTDecl *> decls, StringAttr name,
                               llvm::SMLoc aliasLoc, ASTDecl &context,
                               bool emitDiagnostics = true,
                               ImportPathAttr moduleName = ImportPathAttr(),
                               StringAttr declNameInModule = StringAttr(),
                               bool allowMultipleWithSameName = false);

  /// Check whether \p incoming decls can coexist in scope with \p existing
  /// decls under the same naming rules as local declarations
  /// (attachDeclToParentNameTable):
  ///   - Functions form overload sets; multiple FnOps with the same name are
  ///     allowed.
  ///   - A struct and its extensions may share a name (single namespace).
  ///   - Everything else conflicts: fn+non-fn, non-fn+fn, non-fn+non-fn.
  ///
  /// Returns failure() if a conflict is found, regardless of emitDiagnostics.
  /// emitDiagnostics only gates whether an error is reported to the user.
  LogicalResult checkImportNamingConflict(ArrayRef<ASTDecl *> incoming,
                                          ArrayRef<ASTDecl *> existing,
                                          StringAttr name, llvm::SMLoc aliasLoc,
                                          bool emitDiagnostics);

public:
  /// Create a resolved ImportOp in \p dest's scope. Only nested ImportOps in
  /// the region gate access to child modules of \p modulePath.
  ImportASTDecl &createImportOp(ASTDecl &dest, mlir::OpBuilder &builder,
                                StringAttr name, ImportPathAttr modulePath,
                                mlir::Location loc);

  /// Import the provided decl from the given module decl, into the provided
  /// destination.
  /// resolveTarget determines whether we resolve the ultimate decl as well.
  LogicalResult importDeclFromModule(ASTDecl &dest, PackageOp currentPackage,
                                     ImportPathAttr moduleName,
                                     StringAttr sourceName, StringAttr destName,
                                     SMLoc loc, SMLoc sourceNameLoc,
                                     SMLoc destNameLoc,
                                     bool resolveTarget = true);
  /// Import decls from the given module into the provided destination context
  /// using a wild-card import. Only non-internal decls (those whose names don't
  /// start with an `_`) are imported.
  LogicalResult importWildcardDeclsFromModule(
      ASTDecl &context, const UnresolvedWildcardImport &unresolvedImport,
      StringAttr onlyName = {});

  //===--------------------------------------------------------------------===//
  // Decl Resolution
  //===--------------------------------------------------------------------===//

  /// Resolve the specified declaration to at least the specified level of
  /// resolution, performing incremental type checking as appropriate.
  LogicalResult resolve(ASTDecl &decl, DeclResolvedness howResolved,
                        llvm::SMLoc loc);
  LogicalResult resolveSignature(ASTDecl &decl, llvm::SMLoc loc) {
    return resolve(decl, DeclResolvedness::signature, loc);
  }
  LogicalResult resolveBody(ASTDecl &decl, llvm::SMLoc loc) {
    return resolve(decl, DeclResolvedness::body, loc);
  }

  /// Body-resolve the specified declaration and all declarations contained
  /// within it.
  void resolveAllWithin(ASTDecl &decl);

  /// Walk the standard library's package tree looking for the package whose
  /// public surface (its `__init__`'s declarations and re-exports) includes
  /// `name`. Returns the dotted package path to import (e.g. "std.memory") so a
  /// diagnostic can suggest the missing import, or nullopt when there is no
  /// match, the name is private, or the match is a genuine ambiguity (packages
  /// on diverging branches expose it; a single ancestor->descendant chain is
  /// resolved to its shortest, most public path). Supported only for a prebuilt
  /// (bytecode) `std`; see the definition for why.
  std::optional<std::string> findUniqueStdlibImportFor(StringRef name);

  /// Build the closure value for a `lambda` expression at emit time, by
  /// resolving the synthetic anonymous `def` it desugars to. Null on error.
  AnyValue resolveAnonymousClosure(const LambdaNode *node, IREmitter &emitter,
                                   ExprDest &dest);

  /// DeclResolution is an inherently recursive process - this return the
  /// current declaration that is being worked on.
  ASTDecl *getDeclCurrentlyProcessing() const {
    if (declsCurrentlyProcessing.stack.empty())
      return nullptr;
    return declsCurrentlyProcessing.stack.back();
  }

  /// Return the declaration context to use when pretty-printing parameter
  /// references in diagnostics.
  ASTDecl *getDiagnosticDeclContext() const {
    return diagnosticDeclContext ? diagnosticDeclContext
                                 : getDeclCurrentlyProcessing();
  }

  bool isAlreadyProcessing(ASTDecl &decl) const {
    return declsCurrentlyProcessing.map.contains(&decl);
  }

  //===--------------------------------------------------------------------===//
  // Top-Level Decl Resolution
  //===--------------------------------------------------------------------===//

  /// Resolve all of the declarations that are defined within or referenced by
  /// the given container `decl`. Erases any from-import placeholders the
  /// resolution superseded with a real ImportOp.
  void resolveAllReferencedFrom(ASTDecl &decl);

  /// Erase placeholder ops for decls left unparsed (never referenced).
  void eraseUnparsedDecls();

  /// Resolve the pending wildcard imports in the scope.
  LogicalResult resolveAllWildcardImports(ASTDecl &scope);

  /// Expand `scope`'s pending wildcard imports for `name` alone, leaving them
  /// pending so later names can still be found through them. Wildcards already
  /// searched for `name` are skipped.
  void expandWildcardsForName(ASTDecl &scope, StringAttr name,
                              bool stopOnFirstHit = false);

  /// If `decl` is a file module, pull the extensions its wildcard imports
  /// provide into its scope, so conformance checks against its traits and
  /// structs see them. Called once the decl's body is resolved.
  void completeModuleScope(ASTDecl &decl);

  /// Import every extension visible in `module`'s scope into `dest`, as part
  /// of importing from `module`.
  LogicalResult importModuleExtensions(ASTDecl &module, ASTDecl &dest,
                                       ImportPathAttr moduleName,
                                       SMLoc sourceLoc, SMLoc destLoc);

  ArrayRef<ASTDecl *> getParsedDeclList() const { return parsedDeclList; }

  //===--------------------------------------------------------------------===//
  // Symbol-ASTDecl Mapping
  //===--------------------------------------------------------------------===//

  /// Given the symbol for a declaration, return the ASTDecl that corresponds to
  /// it.  This doesn't allow null symbols, so it always succeeds.
  ASTDecl &getDeclForTypeSymbol(SymbolRefAttr symbol) const;
  ASTDecl *getDeclForTypeSymbolIfExists(SymbolRefAttr symbol) const;
  DefASTDecl *getDeclForFuncSymbol(SymbolRefAttr attr) const;

  /// This registers the finalized function with the DeclResolver after its
  /// signature has been resolved and its mangled name is available.  This
  /// returns an existing function if there is a redefinition problem.
  Operation *finalizeFuncSignature(FnOp funcOp, DefASTDecl &decl);

  /// Return the synthetic declaration for the given canonical trait
  /// composition, creating it if needed.
  ASTDecl *getTraitCompositionDecl(TraitType trait);

  //===--------------------------------------------------------------------===//
  // Export Handling
  //===--------------------------------------------------------------------===//

  void registerAndCheckExport(StringRef aliasName, SMLoc loc);
  void exportMain(DefASTDecl &funcDecl);

  //===--------------------------------------------------------------------===//
  // Decl Helpers
  //===--------------------------------------------------------------------===//

  /// Create a name from a signature by appending argument types into the name.
  static StringAttr getMangledName(StringAttr baseName, ASTDecl &container,
                                   FnTypeGeneratorType signatureGen);

  /// Given a signature type that may contain references to parameter
  /// declarations in a parent context, isolate it by creating a signature with
  /// no external references by inserting an parameter for every captured
  /// parameter declaration. Return the captured parameter references.
  static std::pair<SmallVector<ParamDeclRefAttr>, FnTypeGeneratorType>
  createSelfContainedSignature(FnTypeGeneratorType original);

  /// Given a trait type, return its canonical form (cached).
  TraitType getCanonicalTrait(TraitType trait);
  TraitType getCanonicalTrait(TraitSymbolAttr trait) {
    return getCanonicalTrait(TraitType::get(trait));
  }

  /// Given a list of symbols, canonicalize the list and return the canonical
  /// trait type.
  TraitType getCanonicalTrait(
      SmallVectorImpl<TraitSymbolAttr> &symbols,
      const DenseMap<TraitSymbolAttr, ConstraintAttr> &constraintMap =
          DenseMap<TraitSymbolAttr, ConstraintAttr>());

  /// Define the self parameter of the trait and add inheritance attributes. For
  /// example, given:
  ///
  /// "trait @vanilla", {@Movable}, {@AnyType}
  ///
  /// this function modifies "vanilla" to the following:
  ///
  /// trait @vanilla<?, *"_Self`": !lit.trait<@vanilla>>(!lit.trait<@AnyType,
  /// @Movable>) attributes {immediateParents = #M<symbols[@AnyType]>}
  LogicalResult addSelfTypeToTrait(TraitDeclOp traitOp, TraitASTDecl &decl,
                                   SmallVector<TraitSymbolAttr> &parentTraits,
                                   DenseSet<TraitSymbolAttr> &immediateParents,
                                   ArrayRef<ParamDeclAttr> parameters = {},
                                   ArrayRef<PassingKind> passingKinds = {});

  /// Remove the Decl registered under the oldName and re-register it under a
  /// new name.
  LogicalResult replaceNameAssociatedWithParameter(StringAttr oldName,
                                                   StringAttr newName,
                                                   ASTDecl &scope);

  /// This struct is used to update `diagnosticDeclContext`.
  /// This is used to ensure that error messages complaining about inferred
  /// parameters and types correctly refer to ParamDeclRefAttr's in the correct
  /// declaration.
  struct DiagnosticDeclContextChanger {
    DiagnosticDeclContextChanger(ASTDecl *declToUse);
    ~DiagnosticDeclContextChanger();

  private:
    DeclResolver *resolver = nullptr;
    ASTDecl *prevDiagnosticDeclContext = nullptr;
  };

private:
  friend class ASTDecl;
  friend class AliasASTDecl;
  friend class DefASTDecl;
  friend class ExtensionASTDecl;
  friend class StructASTDecl;
  friend class StructFieldASTDecl;
  friend class TraitASTDecl;
  friend class WitnessASTDecl;

  /// Iteratively resolve all parsed decls that were referenced outside the
  /// main container (typically stdlib/library declarations). Called by
  /// resolveAllReferencedFrom when no errors have been emitted.
  void resolveReferencedDecls();

  /// Erase from-import placeholders superseded by a real ImportOp. Runs at
  /// the end of every resolveAllReferencedFrom: the superseding gate is what
  /// name lookup uses, so the placeholders are dead in every parsing mode.
  void eraseDeadImportPlaceholders();

  ParseResult resolveSignature(TraitType traitType, ASTDecl &decl);
  ParseResult resolveBody(TraitType traitType, ASTDecl &decl);

  /// This map tracks the ASTDecl for every MLIR type declaration with a symbol.
  /// This does not include functions, only things that may be referred to by a
  /// StructType: StructTypes, aliases, etc.
  DenseMap<SymbolRefAttr, ASTDecl *> declForTypeSymbol;

  /// This map tracks the ASTDecl for every FnOp, allowing clients to map
  /// from MLIR symbol references to their body and AST information.  This is
  /// populated during signature resolution, since the symbol will be mangled.
  DenseMap<SymbolRefAttr, DefASTDecl *> declForFuncSymbol;

  /// This map tracks the synthetic, unlisted ASTDecls for trait compositions.
  /// Their IRValue is the canonical trait type-value. For details, see STCASTD.
  DenseMap<TraitType, ASTDecl *> canonicalTraitCompositionDecls;

  /// This map caches trait canonicalization.
  DenseMap<TraitType, TraitType> traitCanonicalizationCache;

  /// This map tracks the exported function names and their locations so that
  /// we can check if they are unique.
  /// Note: these StringRef keys cannot dangle because they point to the parsed
  /// source buffer, we don't need to use StringMap here.
  llvm::StringMap<SMLoc> exportedSymbolNames;

  /// This array holds all of the parsed declarations in a deterministic order.
  std::vector<ASTDecl *> parsedDeclList;

  /// Removed declarations retained until the resolver is destroyed so active
  /// resolution frames can unwind safely.
  std::vector<ASTDecl *> retiredDeclList;

  /// From-import placeholders (`from a.b import c`) that resolved to a
  /// submodule and were superseded by an ImportOp.
  llvm::SmallPtrSet<mlir::Operation *, 8> deadImportPlaceholders;

  /// Name binding is an recursive process in the general case.  This keeps
  /// track of the declarations currently being name bound so we can diagnose
  /// cyclic dependencies.
  struct CurrentProcessingSet {
    DenseMap<ASTDecl *, llvm::SMLoc> map;
    std::vector<ASTDecl *> stack;

    /// Insert the specified declaration into this set. If it already exists,
    /// return failure and leave the container alone.
    LogicalResult insert(ASTDecl *decl, llvm::SMLoc loc) {
      auto [it, inserted] = map.insert({decl, loc});
      if (!inserted)
        return failure();
      stack.push_back(decl);
      return success();
    }

    void pop() {
      map.erase(stack.back());
      stack.pop_back();
    }
  };
  CurrentProcessingSet declsCurrentlyProcessing;

  /// The declaration context used when pretty-printing parameter references in
  /// diagnostics.
  ASTDecl *diagnosticDeclContext = nullptr;

  /// Monotonic counter used to give each `lambda`'s synthetic anonymous `def` a
  /// unique name (see resolveAnonymousClosure).
  unsigned anonymousClosureCounter = 0;

  /// Allow access to private fields.
  friend SharedState;

  DeclResolver(const DeclResolver &) = delete;
  DeclResolver &operator=(const DeclResolver &) = delete;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_DECLRESOLVER_H
