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
// AST representation for a declaration.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_ASTDECL_H
#define KGEN_MOJOPARSER_ASTDECL_H

#include "Mojo/MojoParser/Constraints.h"
#include "Mojo/MojoParser/Lexer.h"
#include "Mojo/MojoParser/SharedState.h"

#include "Mojo/LITDialect/LITOps.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TinyPtrVector.h"
#include <string>

namespace M::KGEN::LIT {
class DocString;
class TypeScopeASTDecl;

struct UnresolvedWildcardImport {
  ImportPathAttr moduleName;
  SMLoc importLoc;
  /// Whether or not the wildcard import has been superseded by a later one (in
  /// source order) or has been drained through resolution.
  bool isSuperseded = false;

  /// The names that this wildcard has already been searched for.
  std::unique_ptr<llvm::StringSet<>> searchedNames = nullptr;

  /// Mark this wildcard as searched for `name`, returning false if it already
  /// was.
  bool markSearched(StringRef name);
};

/// This is the AST representation (as opposed to the MLIR representation) of a
/// declaration in a program.  These maintain type checking and other
/// information that is irrelevant by the time the parser has created a complete
/// and correct IR for a program.  Declarations often have other declarations
/// nested inside of them, forming "scopes" and supporting name lookup.
///
/// Declarations in Mojo work the same way as in Python: scopes are nested
/// and are defined when a builtin, module, class/struct, or function definition
/// is introduced.  Mojo (like Python) allows forward references to values
/// before they are defined, so the parser works in multiple phases where it
/// notices a declaration but does not parse its body until it is demanded.
class ASTDecl {
public:
  enum class Kind : uint8_t {
    Base,
    IR,
    OtherOp,
    Var,
    Alias,
    Conformance,
    Extension,
    Def,
    FnSignature,
    EnumCase,
    Import,
    TopLevelModule,
    FileModule,
    Package,
    Struct,
    StructField,
    Trait,
    Witness,
  };

  virtual ~ASTDecl();

  Kind getKind() const { return static_cast<Kind>(kind); }

  /// Resolve this source declaration's signature.
  virtual LogicalResult resolveSignature(Lexer &lexer);

  /// Resolve this declaration's signature when it has no source to parse.
  virtual LogicalResult resolveSyntheticSignature();

  /// Resolve this source declaration's body.
  virtual LogicalResult resolveBody(Lexer &lexer);

  /// Resolve this declaration's body when it has no source to parse.
  virtual LogicalResult resolveSyntheticBody();

  /// ASTDecl's always have a notion of where they came from.
  SharedState &getShared() const { return shared; }
  MLIRContext *getContext() const { return shared.getContext(); }

  /// Return tooling information for this declaration.
  ASTType getToolingType();
  std::optional<StringRef> getName();
  std::optional<StringRef> getDeprecationWarning();

  virtual CValue getIfIRValue() const { return {}; }

  /// Returns this function argument's number, or -1 for a non-argument.
  virtual ssize_t getArgNumber() const { return -1; }

  /// Returns this parameter's number, or -1 for a non-parameter.
  virtual ssize_t getParamNumber() const { return -1; }

  /// Returns this argument's ABI convention. Asserts if this is not an
  /// argument declaration.
  ArgConvention getArgConvention() const;

  /// Returns whether this argument appears in the user-visible signature.  This
  /// filters out implicitly declared error slots and result slots, but keeps
  /// user-declared "out" argument slots.
  bool isUserArgument() const;

  /// Return the operation backing this decl, or null if it does not have one.
  /// Callers that know the decl kind should downcast and use the typed
  /// getOp() accessors instead.
  virtual Operation *getIfOperation() const { return nullptr; }

  /// Return true if this decl is callable: a `FnOp`, or a witness for one.
  bool isCallableDecl() const;

  /// Return the full signature of this callable decl. `contextualType` is the
  /// type the decl is referenced through (e.g. the type of `x` in `x.foo`).
  FnTypeGeneratorType getDeclFullSignature(ASTType contextualType = {}) const;

  /// Return true if this decl is a static method.
  bool isStaticMethodDecl() const;

  /// Return how this decl may be used as an implicit conversion.
  ImplicitConversionKind getDeclImplicitConversionKind() const;

  /// Get the user-printable name of the declaration if it has one.
  ///  This removes any mangling (e.g. for parameters).
  std::optional<StringRef> getUserNameIfOperation() const;

  /// If the IRValue is a concrete type, return it as an ASTType.
  ASTType getIfTypeValue() const;

  LLVM_ATTRIBUTE_ALWAYS_INLINE LLVM_ATTRIBUTE_NODEBUG llvm::SMLoc
  getLoc() const {
    return loc;
  }
  void setLoc(llvm::SMLoc newLoc) { loc = newLoc; }
  LLVM_ATTRIBUTE_ALWAYS_INLINE LLVM_ATTRIBUTE_NODEBUG ASTDecl *
  getParentDecl() const {
    return parentDecl;
  }

  /// Get the nearest declaration of the given AST declaration class. This can
  /// return itself, a parent declaration, or null if no matching declaration
  /// is found.
  template <typename DeclT>
  DeclT *getNearestDeclOfType() {
    for (ASTDecl *cur = this; cur; cur = cur->getParentDecl())
      if (auto *decl = dyn_cast<DeclT>(cur))
        return decl;
    return nullptr;
  }

  /// Return the indentation of the introducer token or -1 if it wasn't on the
  /// start of line.
  ssize_t getIndentation() const { return indentation; }

  /// This cursor holds the location the parser should resume for the next phase
  /// of resolution.  For example, after initial scanning of a 'def', this will
  /// be on the def token.  After processing the signature, this will be after
  /// the colon.
  LexerCursor &getCursor() { return cursor; }

  /// Set the parse cursor and matching end-cursor together. Used when a decl's
  /// source is materialized after the decl is created.
  void setParseCursor(const LexerCursor &cursor, const LexerCursor &endCursor) {
    this->cursor = cursor;
    this->endCursorState = endCursor.getState();
  }

  /// Return true if the end of the speculatively scanned decl matches the
  /// specified cursor.
  bool isMatchingEndCursor(const LexerCursor &cursor) const {
    return endCursorState == cursor.getState();
  }

  /// Return the SymbolRefAttr for a declaration, including all scoping that may
  /// be needed, making it unique for every declaration.  This returns null for
  /// named values that do not have a declaration.
  SymbolRefAttr getSymbolRef() const;

  /// Add an unresolved wild card import into this scope.
  void addUnresolvedWildcardImport(UnresolvedWildcardImport unresolvedImport);

  /// Record that `name` was imported with @stable(recursive=True) into this
  /// scope, suppressing stability warnings for that name and its members.
  void addRecursivelyStableName(mlir::StringAttr name);

  /// Return true if `name` is covered by a @stable(recursive=True) import
  /// override in this scope or any enclosing scope.
  bool hasRecursivelyStableName(mlir::StringAttr name) const;

  /// Return true if `typeDecl`'s user-visible name is covered by a
  /// @stable(recursive=True) import override in this scope. Returns false if
  /// `typeDecl` is null or has no user-visible name.
  bool hasRecursivelyStableType(const ASTDecl *typeDecl) const;

  /// Record that `name` was imported with @__doc_inline into this scope. Doc
  /// generation documents such a name here, using the target's declaration.
  void addDocInlineName(mlir::StringAttr name);

  /// Return true if `name` was imported into this scope with @__doc_inline.
  bool isDocInlineName(mlir::StringAttr name) const;

  /// Return the doc string for this decl, or nullptr if there isn't one.
  DocStringAttr getDocString() const;

  /// Return the parsed `DocString` for this decl if available.
  std::optional<DocString> getParsedDocString() const;

  /// Return the docstring summary, or an empty string when absent.
  std::string getDocumentationSummary() const;

  /// Return the formatted docstring description, or an empty string when
  /// absent.
  std::string getDocumentationDescription() const;

  /// Given a decl for a struct or trait type, check if this type conforms
  /// to the specified trait type. Returns a 3-state result:
  /// - `yes` if the type definitely conforms
  /// - `no` if the type definitely does not conform
  /// - `unknown` if conformance depends on constraints that cannot be
  ///   evaluated statically (e.g., generic parameters)
  ///
  /// If concreteType is provided, its parameter bindings are used to evaluate
  /// conditional trait conformances. If callerAssumptions is non-empty, those
  /// where-clause assumptions are used to prove unfoldable constraints.
  TriBool doesNominalTypeConformTo(TraitType trait, ASTType concreteType,
                                   ArrayRef<ConstraintAttr> callerAssumptions);

  /// Same, additionally collecting the problematic constraints so a diagnosing
  /// caller can name them.
  ConstraintResult doesNominalTypeConformToWithDetails(
      TraitType trait, ASTType concreteType,
      ArrayRef<ConstraintAttr> callerAssumptions);

  /// Find all extensions in this scope that target a specific struct.
  /// If filterTrait is provided, only returns extensions that implement that
  /// specific trait.
  void findExtensionsInScopeForStruct(
      SymbolRefAttr targetStruct, llvm::SmallPtrSetImpl<ASTDecl *> &results,
      std::optional<TraitSymbolAttr> filterTrait = std::nullopt);

  /// Collect all declarations that could contribute to a type's namespace.
  /// Examples:
  /// - If there's a struct Spaceship with two extensions, this will return all
  ///   three of those declarations.
  /// - If there's a trait Sporkable with two extensions, this will return all
  ///   three of those declarations.
  /// This is useful for collecting declarations in which we can search for
  /// methods for a given type, e.g. `my_ship.some_method()`, since the method
  /// might be in any of those declarations.
  llvm::SmallVector<ASTDecl *, 4> collectTypeAndExtensions(ASTType type,
                                                           llvm::SMLoc callLoc);

  /// Whether this decl was loaded from bytecode.
  bool isLoadedFromBytecode() const { return loadedFromBytecode; }

  /// Whether this decl has been erased from the module and retired.
  bool isRetired() const { return retired; }

  //===--------------------------------------------------------------------===//
  // Name lookup
  //===--------------------------------------------------------------------===//

  /// Look up a name in this declaration's scope only: return null on failure.
  ArrayRef<ASTDecl *> lookupInCurrentScope(StringAttr name) const;
  ArrayRef<ASTDecl *> lookupInCurrentScope(StringRef name) const;

  /// Perform a lookup in this declaration's scope and all parent scopes,
  /// returning the nearest target or empty if nothing is found.
  ArrayRef<ASTDecl *> lookup(StringAttr name) const {
    const ASTDecl *curScope = this;
    while (curScope) {
      ArrayRef<ASTDecl *> result = curScope->lookupInCurrentScope(name);
      if (!result.empty())
        return result;
      curScope = curScope->getParentDecl();
    }
    return {};
  }

  /// Return an iterable set of declarations in this scope.  For lookups, use
  /// lookup or lookupInCurrentScope.
  ArrayRef<std::pair<StringAttr, TinyPtrVector<ASTDecl *>>>
  getDeclsInScope() const {
    if (declsInScope)
      return ArrayRef(declsInScope->begin(), declsInScope->end());
    return {};
  }

  /// Given a reference to a parameter, look at this declaration and enclosing
  /// scopes to find the ASTDecl that defines it (e.g. the enclosing function,
  /// struct or trait).  This can return null if not found.
  std::tuple<const ASTDecl *, ArrayRef<ParamDeclAttr>, size_t>
  lookupParamReference(ParamDeclRefAttr paramRef) const;

  //===--------------------------------------------------------------------===//
  // Other State management.
  //===--------------------------------------------------------------------===//

  /// Indicate that the decl has reference errors.
  void setErroneous();
  /// Return true if the decl has reference errors.
  bool isErroneous() const { return hasReferenceError; }

  /// Return any decorators that need to be processed as part of body resolution
  /// phase for a decl.
  ArrayRef<ExprNode *> getBodyDecorators() const;

  /// During signature resolution, this is called with any decorators that need
  /// to persist until body resolution.
  void setBodyDecorators(ArrayRef<ExprNode *> decorators);

  /// Check if the given name collides with an existing user declared parameter
  /// name in the scope, and if so, uniquely mangle it by postpending a backtick
  /// ("`"), scope depth, and a unique ID.
  StringAttr mangleUserDefinedParamName(StringAttr name);

  /// Create a unique parameter name by postpending a backtick ("`"), scope
  /// depth, and a unique ID.
  StringAttr mangleParamName(const Twine &name);

  /// Move the children decls of `src` (along with their constraints) into this
  /// decl. This is useful when a temporary decl needs to be created for parsing
  /// subexpressions but whose children will be inherited later by a decl being
  /// resolved.
  void takeDecls(ASTDecl &src);

  /// Append the children decls of `src` into this scope (updating their
  /// `parentDecl`), without replacing existing names already in this scope.
  /// Used when promoting temporary pattern-binding scopes into a case scope.
  void mergeDeclsFrom(ASTDecl &src);

  /// Anonymous origins, closure impl structs, and potentially other names are
  /// uniqued to avoid collisions. This returns an ID that is unique to this
  /// ASTDecl instance and help generate such names.
  unsigned getNextUniqueID() { return counter++; }

  /// Mark this ASTDecl as an explicit parameter scope so that parameter
  /// mangling can be applied for in-flight types.
  void setExplicitParamScope() { isExplicitParamScope = true; }
  bool getIsExplicitParamScope() const { return isExplicitParamScope; }

  /// Get the set of assumptions for this decl and all its parent decls.
  void getKnownAssumptionsIncludingParents(
      SmallVectorImpl<ConstraintAttr> &assumptions) const;

  /// If scope is non-null, returns its assumptions (including parents);
  /// otherwise returns an empty vector.
  static llvm::SmallVector<ConstraintAttr>
  getAssumptionsFromScope(const ASTDecl *scope);

  /// Insert a set of assumptions into this decl.
  void insertKnownAssumptions(ArrayRef<ConstraintAttr> assumptions);

  /// Dump the underlying IR value.
  void dump() const;

  /// The pending wildcard imports into this scope.
  MutableArrayRef<UnresolvedWildcardImport> getUnresolvedWildcardImports() {
    if (!unresolvedWildcardImports)
      return {};
    return *unresolvedWildcardImports;
  }

private:
  friend class M::BumpAllocator;
  friend class DeclResolver;
  friend class ModuleLoader;
  friend class SharedState;
  ASTDecl(SharedState &shared, llvm::SMLoc loc, ASTDecl *parentDecl,
          LexerCursor cursor, LexerCursor endCursor, ssize_t indentation);
  ASTDecl(const ASTDecl &) = delete;
  ASTDecl &operator=(const ASTDecl &) = delete;

protected:
  ASTDecl(Kind declKind, SharedState &shared, llvm::SMLoc loc,
          ASTDecl *parentDecl, LexerCursor cursor, LexerCursor endCursor,
          ssize_t indentation);

  /// The Mojo shared state this decl is associated with.
  SharedState &shared;

  /// This is the source location of the declaration, used for diagnostics and
  /// debug information.
  llvm::SMLoc loc;

  /// The parent scope that should continue name lookup.
  ASTDecl *parentDecl;

  /// This is the cursor that points to the next part of declaration to continue
  /// parsing as the declaration is progressively resolved.
  LexerCursor cursor;

  /// This is the lexer cursor state for the first token /after/ the
  /// declaration.  This is used to make sure that bits of a declaration are not
  /// skipped in the early parse and not processes in the later parse.
  const char *endCursorState;

  /// This is the indentation level of the introducer keyword, useful for
  /// parsing the body of the declaration.  If the declaration was not at the
  /// start of a line or this is the top level module, then this is set to -1.
  ssize_t indentation;

  /// The kind of the ASTDecl subclass, tested with isa/dyn_cast etc.
  const uint8_t kind;

public:
  /// This keeps track of what level of type checking this declaration has been
  /// through.  It is maintained by DeclResolver.
  DeclResolvedness resolvedness : 3; // Starts as DeclResolvedness::unparsed

private:
  /// When a bytecode decl depends on a source decl's children, we have to parse
  /// the signatures of all the children to register them in the symbol table.
  /// Cache this process using a flag on the decl.
  bool referencedFromBytecode : 1;

  /// This is set to true when an error is detected and reported about this
  /// declaration that could cause references to it to cause spurious downstream
  /// errors.  For example, "var x : SomeUndeclaredType" will cause errors for
  /// every reference to 'x' because the type will be bogus.
  bool hasReferenceError : 1;

  /// This is set to true if there is an entry for body-decorators in a
  /// backing hashtable.  Clients should use "getBodyDecorators().
  bool hasBodyDecorators : 1;

  /// This is set to true when the declaration was loaded from bytecode, not
  /// parsed from a textual source file. These declarations behave differently
  /// than source decls, and e.g., do not resolve in the same way as source
  /// decls.
  bool loadedFromBytecode : 1;

  /// True if this ASTDecl explicitly opts in to being treated as a parameter
  /// scope by `mangleParamName` even though its IR value isn't a
  /// `DeclInterface` op
  bool isExplicitParamScope : 1;

  /// True once this decl has been erased from the module and moved to
  /// retiredDeclList. Set by DeclResolver so a liveness check is O(1) instead
  /// of scanning parsedDeclList.
  bool retired : 1;

  /// The counter to allow the generation of unique IDs for this ASTDecl.
  unsigned counter = 0;

  /// These are the declarations defined within this scope.  This is lazily
  /// allocated the first time something is added, because it the vast majority
  /// of decls (leaves in the tree) don't need it.  (5-12% need it).
  ///
  /// This includes "owned" child decls whose `parentDecl` points back to this
  /// decl, as well as "inherited" child decls whose `parentDecl` points to
  /// other decls (as is the case for trait composition decls, see STCASTD).
  ///
  /// It also includes entries that are imported from other scopes. When an
  /// UnresolvedImportOp is resolved, it will be replaced with pointers to
  /// ASTDecls that are *owned* by other ASTDecls.
  ///
  /// It also sometimes includes the same ASTDecl twice, under different names.
  /// For example, an `__extension MyStruct` will be known to its parent ASTDecl
  /// as "extension:MyStruct" as well as "extension:" (the latter name is shared
  /// by all extensions).
  /// TODO(MOCO-2674): Have a review for this "extension:" choice, in case
  /// others have concerns about it.
  ///
  /// TODO(MOCO-522): Arcana docs on all the above, and on how importing works
  /// in general.
  ///
  /// TODO: Properly model inherited vs. owned decls.
  using DeclInScopeType = llvm::MapVector<StringAttr, TinyPtrVector<ASTDecl *>>;
  std::unique_ptr<DeclInScopeType> declsInScope;

  /// A set of modules with unresolved wildcard imports into this decl. This is
  /// lazily initialized because it is very rarely needed (~0.6% of all decls).
  /// This list only ever grows; UnresolvedWildcardImports are instead flagged
  /// as superseded once resolved or once a later import of the same module
  /// replaces it.
  using UnresolvedWildcardImportsType =
      llvm::SmallVector<UnresolvedWildcardImport>;
  std::unique_ptr<UnresolvedWildcardImportsType> unresolvedWildcardImports;

  /// Lazily-allocated set of import names that were imported with
  /// @stable(recursive=True) into this scope.  These names suppress stability
  /// warnings for the named binding and all member accesses through it.
  std::unique_ptr<llvm::DenseSet<mlir::StringAttr>> recursivelyStableNames;

  /// Lazily-allocated set of import names brought into this scope with
  /// @__doc_inline.
  std::unique_ptr<llvm::DenseSet<mlir::StringAttr>> docInlineNames;

  /// This is the set of constraints that can be assumed to be true inside
  /// this declaration. It may reference parameter declarations from this
  /// ASTDecl or its parent ASTDecls.
  std::unique_ptr<llvm::SetVector<ConstraintAttr>> knownAssumptions;
};

/// An AST declaration backed by an MLIR operation through shared storage.
///
/// Function signatures keep their `FnOp` in `DefASTDecl` so function types can
/// share their declaration-list base without pretending to have an operation.
class OpASTDecl : public ASTDecl {
public:
  static bool classof(const ASTDecl *decl) {
    switch (decl->getKind()) {
    case Kind::OtherOp:
    case Kind::Var:
    case Kind::Alias:
    case Kind::Conformance:
    case Kind::Extension:
    case Kind::EnumCase:
    case Kind::Import:
    case Kind::TopLevelModule:
    case Kind::FileModule:
    case Kind::Package:
    case Kind::Struct:
    case Kind::StructField:
    case Kind::Trait:
      return true;
    default:
      return false;
    }
  }

  Operation *getIfOperation() const final { return operation; }
  Operation &getOperation() const {
    assert(operation && "OpASTDecl must have an operation");
    return *operation;
  }

  /// Return the builder at the end of the operation's sole region.
  OpBuilder getDeclEndBuilder() {
    assert(getOperation().getNumRegions() == 1 &&
           "can't get builder for this OpASTDecl");
    return OpBuilder::atBlockEnd(&getOperation().getRegion(0).front());
  }

protected:
  OpASTDecl(Kind kind, SharedState &shared, Operation &operation,
            llvm::SMLoc loc, ASTDecl *parentDecl, LexerCursor cursor,
            LexerCursor endCursor, ssize_t indentation);

  void setOperation(Operation &newOperation) { operation = &newOperation; }

private:
  Operation *operation;
};

/// An AST declaration backed by a compile-time or runtime IR value.
class IRASTDecl final : public ASTDecl {
public:
  static bool classof(const ASTDecl *decl) {
    return decl->getKind() == Kind::IR;
  }

  CValue getIfIRValue() const override { return value; }

  /// Returns this function argument's number, or -1 for a non-argument.
  ssize_t getArgNumber() const override {
    if (isParameter)
      return -1;
    return argOrParamNumber;
  }

  /// Returns this parameter's number, or -1 for a non-parameter.
  ssize_t getParamNumber() const override {
    if (!isParameter)
      return -1;
    return argOrParamNumber;
  }

  void setArgOrParamNumber(ssize_t argOrParamNumber, bool isParameter) {
    this->argOrParamNumber = argOrParamNumber;
    this->isParameter = isParameter;
  }
  StringAttr getCapturedName() const { return capturedName; }
  void setCapturedName(StringAttr newCapturedName) {
    assert(newCapturedName && "captured declarations require a name");
    capturedName = newCapturedName;
  }
  void setValue(CValue newValue) { value = std::move(newValue); }

private:
  friend class M::BumpAllocator;
  friend class DeclResolver;

  IRASTDecl(SharedState &shared, CValue value, llvm::SMLoc loc,
            ASTDecl *parentDecl, LexerCursor cursor, LexerCursor endCursor,
            ssize_t indentation);

  CValue value;
  ssize_t argOrParamNumber;
  bool isParameter;
  StringAttr capturedName;
};

/// An AST declaration backed by an operation without dedicated declaration
/// semantics. This is intended for the long tail of operations that are
/// uncommonly used but are represented by an Operation.
class OtherOpASTDecl final : public OpASTDecl {
public:
  static bool classof(const ASTDecl *decl) {
    return decl->getKind() == Kind::OtherOp;
  }

  Operation &getOp() const { return getOperation(); }
  ParamAssertOp getIfParamAssertOp() const {
    return dyn_cast<ParamAssertOp>(getOp());
  }
  UnboundRegionOp getIfUnboundRegionOp() const {
    return dyn_cast<UnboundRegionOp>(getOp());
  }
  bool isParamAssert() const { return bool(getIfParamAssertOp()); }
  bool isUnboundRegion() const { return bool(getIfUnboundRegionOp()); }

private:
  friend class M::BumpAllocator;
  friend class DeclResolver;

  OtherOpASTDecl(SharedState &shared, Operation &op, llvm::SMLoc loc,
                 ASTDecl *parentDecl, LexerCursor cursor, LexerCursor endCursor,
                 ssize_t indentation);
};

/// The AST representation of a variable declaration.
class VarASTDecl final : public OpASTDecl {
public:
  static bool classof(const ASTDecl *decl) {
    return decl->getKind() == Kind::Var;
  }

  VarDeclOp getOp() const { return cast<VarDeclOp>(getOperation()); }
  ssize_t getArgNumber() const override { return argNumber; }
  bool isArgument() const {
    VarDeclKind kind = getOp().getKind();
    return argNumber >= 0 || kind == VarDeclKind::Arg ||
           kind == VarDeclKind::InitOutArg;
  }
  void setArgNumber(ssize_t newArgNumber) { argNumber = newArgNumber; }
  CValue getIfIRValue() const override {
    VarDeclKind kind = getOp().getKind();
    if (kind == VarDeclKind::Arg || kind == VarDeclKind::InitOutArg)
      return MLValue(getOp());
    return {};
  }

private:
  friend class M::BumpAllocator;
  friend class DeclResolver;

  VarASTDecl(SharedState &shared, VarDeclOp varOp, llvm::SMLoc loc,
             ASTDecl *parentDecl, LexerCursor cursor, LexerCursor endCursor,
             ssize_t indentation);

  ssize_t argNumber;
};

/// Base class for top-level, file, and package module declarations.
class ModuleASTDecl : public OpASTDecl {
public:
  static bool classof(const ASTDecl *decl) {
    return decl->getKind() >= Kind::TopLevelModule &&
           decl->getKind() <= Kind::Package;
  }

  LogicalResult resolveSignature(Lexer &lexer) override;
  LogicalResult resolveSyntheticSignature() override;
  LogicalResult resolveBody(Lexer &lexer) override;
  LogicalResult resolveSyntheticBody() override;

  /// Return the module operation backing this decl. Subclasses shadow this
  /// with a typed version.
  Operation *getOp() const { return &getOperation(); }

  /// Return the absolute import path for this module, or null if it has no
  /// symbol.
  ImportPathAttr getAbsoluteModuleName() const;

protected:
  ModuleASTDecl(Kind kind, SharedState &shared, Operation &moduleOp,
                llvm::SMLoc loc, ASTDecl *parentDecl, LexerCursor cursor,
                LexerCursor endCursor, ssize_t indentation);
};

/// The AST representation of the outer MLIR module.
class TopLevelModuleASTDecl final : public ModuleASTDecl {
public:
  static bool classof(const ASTDecl *decl) {
    return decl->getKind() == Kind::TopLevelModule;
  }

  ModuleOp getOp() const { return cast<ModuleOp>(getOperation()); }

private:
  friend class M::BumpAllocator;
  friend class DeclResolver;

  TopLevelModuleASTDecl(SharedState &shared, ModuleOp moduleOp, llvm::SMLoc loc,
                        ASTDecl *parentDecl, LexerCursor cursor,
                        LexerCursor endCursor, ssize_t indentation);
};

/// The AST representation of a source file module.
class FileModuleASTDecl final : public ModuleASTDecl {
public:
  static bool classof(const ASTDecl *decl) {
    return decl->getKind() == Kind::FileModule;
  }

  FileModuleOp getOp() const { return cast<FileModuleOp>(getOperation()); }
  LogicalResult resolveBody(Lexer &lexer) override;

private:
  friend class M::BumpAllocator;
  friend class DeclResolver;

  FileModuleASTDecl(SharedState &shared, FileModuleOp moduleOp, llvm::SMLoc loc,
                    ASTDecl *parentDecl, LexerCursor cursor,
                    LexerCursor endCursor, ssize_t indentation);
};

/// The AST representation of a source package.
class PackageASTDecl final : public ModuleASTDecl {
public:
  static bool classof(const ASTDecl *decl) {
    return decl->getKind() == Kind::Package;
  }

  PackageOp getOp() const { return cast<PackageOp>(getOperation()); }
  /// Resolve and return the package's __init__ module, if present.
  FailureOr<ModuleASTDecl *> resolveInitModule(llvm::SMLoc loc);
  LogicalResult resolveBody(Lexer &lexer) override;
  LogicalResult resolveSyntheticBody() override;

private:
  friend class M::BumpAllocator;
  friend class DeclResolver;

  LogicalResult resolvePackage();

  PackageASTDecl(SharedState &shared, PackageOp packageOp, llvm::SMLoc loc,
                 ASTDecl *parentDecl, LexerCursor cursor, LexerCursor endCursor,
                 ssize_t indentation);
};

/// Base class for declarations that own a function signature: this is
/// DefASTNode and FunctionTypeASTNode.
class FnSignatureASTDecl : public ASTDecl {
public:
  static bool classof(const ASTDecl *decl) {
    return decl->getKind() == Kind::Def || decl->getKind() == Kind::FnSignature;
  }

  virtual FnTypeGeneratorType getSignature() const = 0;

  ArrayRef<ASTDecl *> getParamDecls() const { return paramDecls; }
  ArrayRef<ASTDecl *> getArgDecls() const { return argDecls; }

  /// Return a list of user-declared argument declarations that we'd want to
  /// show in document generation, for example.  This filters out implicitly
  /// declared error slots and result slots, but keeps user-declared "out"
  /// argument slots.
  SmallVector<ASTDecl *> getUserArgDecls() const;

protected:
  FnSignatureASTDecl(Kind kind, SharedState &shared, llvm::SMLoc loc,
                     ASTDecl *parentDecl, LexerCursor cursor,
                     LexerCursor endCursor, ssize_t indentation);

  /// Copies parameter declarations into persistent AST storage.
  void setParamDecls(ArrayRef<ASTDecl *> decls);

  /// Copies argument declarations into persistent AST storage.
  void setArgDecls(ArrayRef<ASTDecl *> decls);

private:
  ArrayRef<ASTDecl *> paramDecls;
  ArrayRef<ASTDecl *> argDecls;
};

/// The AST representation of a function declaration.
class DefASTDecl final : public FnSignatureASTDecl {
public:
  static bool classof(const ASTDecl *decl) {
    return decl->getKind() == Kind::Def;
  }

  FnOp getOp() const {
    assert(operation && "DefASTDecl must have an operation");
    return operation;
  }
  void setOp(FnOp op) { operation = op; }
  Operation *getIfOperation() const override {
    return operation ? const_cast<FnOp &>(operation).getOperation() : nullptr;
  }

  FnTypeGeneratorType getSignature() const override {
    return getOp().getFuncTypeGenerator();
  }

  LogicalResult resolveSignature(Lexer &lexer) override;
  LogicalResult resolveSyntheticSignature() override;
  LogicalResult resolveBody(Lexer &lexer) override;
  LogicalResult resolveSyntheticBody() override;

  /// If this is a method of a struct, trait, or extension, return its parent.
  TypeScopeASTDecl *tryGetMethodParentDecl() const;

  /// Create unlisted declarations for the parameters and arguments represented
  /// only by this function's IR signature.
  void createIRSignatureDecls(DeclResolver &resolver);

private:
  friend class M::BumpAllocator;
  friend class DeclResolver;
  friend class ClosureEmitter;

  DefASTDecl(SharedState &shared, FnOp fnOp, llvm::SMLoc loc,
             ASTDecl *parentDecl, LexerCursor cursor, LexerCursor endCursor,
             ssize_t indentation);

  FnOp operation;
};

/// The AST representation of an anonymous function type.
class FunctionTypeASTDecl final : public FnSignatureASTDecl {
public:
  static bool classof(const ASTDecl *decl) {
    return decl->getKind() == Kind::FnSignature;
  }

  FnTypeGeneratorType getSignature() const override { return signature; }

  /// Stores the completed signature and copies its declarations into
  /// persistent AST storage.
  void setSignature(FnTypeGeneratorType signature, ArrayRef<ASTDecl *> params,
                    ArrayRef<ASTDecl *> args);

private:
  friend class M::BumpAllocator;
  friend class DeclResolver;

  FunctionTypeASTDecl(SharedState &shared, llvm::SMLoc loc, ASTDecl *parentDecl,
                      LexerCursor cursor, LexerCursor endCursor,
                      ssize_t indentation);

  FnTypeGeneratorType signature;
};

/// An AST declaration that provides a `Self` type to its members.
class TypeScopeASTDecl : public OpASTDecl {
public:
  static bool classof(const ASTDecl *decl) {
    return decl->getKind() == Kind::Extension ||
           decl->getKind() == Kind::Struct || decl->getKind() == Kind::Trait;
  }

  /// Return the `Self` type in this declaration's member scope.
  ASTType getSelfType() const {
    assert(resolvedness != DeclResolvedness::unparsed &&
           "signature must be resolved to get a resolved type");
    return selfType;
  }
  void setSelfType(ASTType type) {
    assert(type && "cannot set a null self type");
    selfType = type;
  }

  /// Get the map from each conformed trait to the first symbol that explicitly
  /// inherits from it.
  DenseMap<TraitSymbolAttr, std::pair<TraitSymbolAttr, SMLoc>> *
  getTraitConformanceLineage(bool createIfMissing = false);
  LogicalResult resolveSyntheticBody() override { return success(); }

protected:
  TypeScopeASTDecl(Kind kind, SharedState &shared, Operation &op,
                   llvm::SMLoc loc, ASTDecl *parentDecl, LexerCursor cursor,
                   LexerCursor endCursor, ssize_t indentation);

private:
  ASTType selfType;

  using TraitConformanceLineageType =
      DenseMap<TraitSymbolAttr, std::pair<TraitSymbolAttr, SMLoc>>;
  std::unique_ptr<TraitConformanceLineageType> traitConformanceLineage;
};

/// The common AST representation for struct and trait declarations.
class TypeASTDecl : public TypeScopeASTDecl {
public:
  static bool classof(const ASTDecl *decl) {
    return decl->getKind() == Kind::Struct || decl->getKind() == Kind::Trait;
  }

protected:
  TypeASTDecl(Kind kind, SharedState &shared, Operation &op, llvm::SMLoc loc,
              ASTDecl *parentDecl, LexerCursor cursor, LexerCursor endCursor,
              ssize_t indentation);
};

/// The AST representation of an extension declaration.
class ExtensionASTDecl final : public TypeScopeASTDecl {
public:
  static bool classof(const ASTDecl *decl) {
    return decl->getKind() == Kind::Extension;
  }

  ExtensionDeclOp getOp() const {
    return cast<ExtensionDeclOp>(getOperation());
  }
  LogicalResult resolveSignature(Lexer &lexer) override;
  LogicalResult resolveBody(Lexer &lexer) override;

private:
  friend class M::BumpAllocator;
  friend class DeclResolver;

  ExtensionASTDecl(SharedState &shared, ExtensionDeclOp extensionOp,
                   llvm::SMLoc loc, ASTDecl *parentDecl, LexerCursor cursor,
                   LexerCursor endCursor, ssize_t indentation);
};

/// The AST representation of a struct declaration.
class StructASTDecl final : public TypeASTDecl {
public:
  static bool classof(const ASTDecl *decl) {
    return decl->getKind() == Kind::Struct;
  }

  StructDeclOp getOp() const { return cast<StructDeclOp>(getOperation()); }
  LogicalResult resolveSignature(Lexer &lexer) override;
  LogicalResult resolveBody(Lexer &lexer) override;

  /// Compute the `Self` type represented by a struct declaration operation.
  static Type computeSelfType(StructDeclOp structOp);

  /// Return this struct's parameter declarations in signature order.
  ArrayRef<ASTDecl *> getParamDecls() const { return paramDecls; }

  /// Return the parameter declaration with the given internal name, if any.
  ASTDecl *getParamByName(StringAttr name) const;

private:
  friend class M::BumpAllocator;
  friend class DeclResolver;
  friend class SharedState;

  /// Copies parameter declarations into persistent AST storage.
  void setParamDecls(ArrayRef<ASTDecl *> decls);

  StructASTDecl(SharedState &shared, StructDeclOp structOp, llvm::SMLoc loc,
                ASTDecl *parentDecl, LexerCursor cursor, LexerCursor endCursor,
                ssize_t indentation);

  ArrayRef<ASTDecl *> paramDecls;
};

/// The AST representation of a trait declaration.
class TraitASTDecl final : public TypeASTDecl {
public:
  static bool classof(const ASTDecl *decl) {
    return decl->getKind() == Kind::Trait;
  }

  TraitDeclOp getOp() const { return cast<TraitDeclOp>(getOperation()); }
  LogicalResult resolveSignature(Lexer &lexer) override;
  LogicalResult resolveBody(Lexer &lexer) override;

  /// Compute the `Self` type represented by a trait declaration operation.
  static Type computeSelfType(TraitDeclOp traitOp);

private:
  friend class M::BumpAllocator;
  friend class DeclResolver;

  TraitASTDecl(SharedState &shared, TraitDeclOp traitOp, llvm::SMLoc loc,
               ASTDecl *parentDecl, LexerCursor cursor, LexerCursor endCursor,
               ssize_t indentation);
};

/// An AST declaration backed by an alias declaration operation.
class AliasASTDecl final : public OpASTDecl {
public:
  static bool classof(const ASTDecl *decl) {
    return decl->getKind() == Kind::Alias;
  }

  /// Return the alias operation represented by this declaration.
  AliasDeclOp getOp() const { return cast<AliasDeclOp>(getOperation()); }
  void setOp(AliasDeclOp op) { setOperation(*op.getOperation()); }
  LogicalResult resolveSignature(Lexer &lexer) override;
  LogicalResult resolveSyntheticSignature() override;
  LogicalResult resolveBody(Lexer &lexer) override;
  LogicalResult resolveSyntheticBody() override { return success(); }

private:
  friend class M::BumpAllocator;
  friend class DeclResolver;

  AliasASTDecl(SharedState &shared, AliasDeclOp aliasOp, llvm::SMLoc loc,
               ASTDecl *parentDecl, LexerCursor cursor, LexerCursor endCursor,
               ssize_t indentation);
};

/// The AST representation of a trait conformance declaration.
class ConformanceASTDecl final : public OpASTDecl {
public:
  static bool classof(const ASTDecl *decl) {
    return decl->getKind() == Kind::Conformance;
  }

  ConformanceOp getOp() const { return cast<ConformanceOp>(getOperation()); }
  StructASTDecl &getStructDecl() const;
  LogicalResult resolveBody(Lexer &lexer) override;
  LogicalResult resolveSyntheticBody() override;

private:
  friend class M::BumpAllocator;
  friend class DeclResolver;

  LogicalResult resolveConformance();

  ConformanceASTDecl(SharedState &shared, ConformanceOp conformanceOp,
                     llvm::SMLoc loc, ASTDecl *parentDecl, LexerCursor cursor,
                     LexerCursor endCursor, ssize_t indentation);
};

/// The AST representation of an import declaration.
class ImportASTDecl final : public OpASTDecl {
public:
  static bool classof(const ASTDecl *decl) {
    return decl->getKind() == Kind::Import;
  }

  ImportOp getIfResolvedOp() const {
    return dyn_cast_if_present<ImportOp>(getIfOperation());
  }
  UnresolvedImportOp getIfUnresolvedOp() const {
    return dyn_cast_if_present<UnresolvedImportOp>(getIfOperation());
  }
  bool isResolved() const { return bool(getIfResolvedOp()); }
  bool isUnresolved() const { return bool(getIfUnresolvedOp()); }
  void setResolvedOp(ImportOp op) { setOperation(*op.getOperation()); }

  LogicalResult resolveSignature(Lexer &lexer) override;
  LogicalResult resolveSignature(bool resolveTarget);
  LogicalResult resolveSyntheticSignature() override;
  LogicalResult resolveBody(Lexer &lexer) override;
  LogicalResult resolveSyntheticBody() override;

private:
  friend class M::BumpAllocator;
  friend class DeclResolver;

  ImportASTDecl(SharedState &shared, Operation &importOp, llvm::SMLoc loc,
                ASTDecl *parentDecl, LexerCursor cursor, LexerCursor endCursor,
                ssize_t indentation);
};

/// The AST representation of an enum case declaration.
class EnumCaseASTDecl final : public OpASTDecl {
public:
  static bool classof(const ASTDecl *decl) {
    return decl->getKind() == Kind::EnumCase;
  }

  EnumCaseOp getOp() const { return cast<EnumCaseOp>(getOperation()); }
  StructASTDecl &getEnumDecl() const {
    return cast<StructASTDecl>(*getParentDecl());
  }
  LogicalResult
  resolveAndCheckRedefinition(EnumCaseASTDecl *previousDefinition);
  LogicalResult resolveSignature(Lexer &lexer) override;
  LogicalResult resolveBody(Lexer &lexer) override;
  LogicalResult resolveSyntheticBody() override { return success(); }

private:
  friend class M::BumpAllocator;
  friend class DeclResolver;

  EnumCaseASTDecl(SharedState &shared, EnumCaseOp caseOp, llvm::SMLoc loc,
                  ASTDecl *parentDecl, LexerCursor cursor,
                  LexerCursor endCursor, ssize_t indentation);
};

/// The AST representation of a stored struct field declaration.
class StructFieldASTDecl final : public OpASTDecl {
public:
  static bool classof(const ASTDecl *decl) {
    return decl->getKind() == Kind::StructField;
  }

  StructFieldOp getOp() const { return cast<StructFieldOp>(getOperation()); }
  StructASTDecl &getStructDecl() const {
    return cast<StructASTDecl>(*getParentDecl());
  }
  LogicalResult resolveSignature(Lexer &lexer) override;
  LogicalResult resolveBody(Lexer &lexer) override;
  LogicalResult resolveSyntheticBody() override { return success(); }

private:
  friend class M::BumpAllocator;
  friend class DeclResolver;

  StructFieldASTDecl(SharedState &shared, StructFieldOp fieldOp,
                     llvm::SMLoc loc, ASTDecl *parentDecl, LexerCursor cursor,
                     LexerCursor endCursor, ssize_t indentation);
};

/// A synthesized declaration for a member inherited through trait composition.
/// It tracks the contributing declarations until they are resolved into a
/// single witness type.
class WitnessASTDecl final : public ASTDecl {
public:
  /// The declarations merged into this witness before signature resolution.
  using UnresolvedDecls = SmallVector<ASTDecl *>;

  /// The resolved witness entry.
  struct ResolvedType {
    StringAttr witnessName;
    Type witnessType;

    /// Additional function information, meaningless for non-function members.
    ImplicitConversionKind implicitConversion = ImplicitConversionKind::None;
    bool isStaticMethod = false;
  };

  static bool classof(const ASTDecl *decl) {
    return decl->getKind() == Kind::Witness;
  }

  UnresolvedDecls getDecls() const { return cast<UnresolvedDecls>(storage); }
  ResolvedType getWitnessEntry() const { return cast<ResolvedType>(storage); }
  LogicalResult resolveSignature(Lexer &lexer) override;
  LogicalResult resolveSyntheticSignature() override;
  LogicalResult resolveBody(Lexer &lexer) override;
  LogicalResult resolveSyntheticBody() override;

  /// Holds contributing declarations before resolution and the merged type
  /// afterward.
  SmartVariant<ResolvedType, UnresolvedDecls> storage;

  /// The trait symbol witnessed by this declaration.
  TraitSymbolAttr traitSymbol;

private:
  friend class M::BumpAllocator;
  friend class DeclResolver;

  WitnessASTDecl(SharedState &shared, UnresolvedDecls unresolvedDecls,
                 TraitSymbolAttr traitSymbol, llvm::SMLoc loc,
                 ASTDecl *parentDecl, LexerCursor cursor, LexerCursor endCursor,
                 ssize_t indentation);
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_ASTDECL_H
