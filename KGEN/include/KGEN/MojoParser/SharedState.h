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
// This file provides logic that is shared across all the subsystems in the Mojo
// parser.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_SHAREDSTATE_H
#define KGEN_MOJOPARSER_SHAREDSTATE_H

#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/OriginTrackable.h"
#include "KGEN/MojoParser/IRValues.h"
#include "KGEN/MojoParser/MojoDiags.h"

#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "Support/ErrorOr.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/PrettyStackTrace.h"

namespace M::KGEN {
class CompilationOptions;
class NoneAttr;
class ParamDeclAttr;
class ParamDeclRefAttr;
class FuncTypeGeneratorType;
class SymbolConstantAttr;
} // namespace M::KGEN

namespace M::KGEN::LIT {
class ASTDecl;
class ASTType;
class ClosureEmitter;
class DeclResolver;
class ExprNode;
struct Operand;
class FileModuleOp;
class FnOp;
class FnTypeGeneratorType;
class IREmitter;
class LexerCursor;
class LookupResult;
class LookupAllResult;
class PackageOp;
class ParserListener;
class StructDeclOp;
class TraitType;
class CallOperands;
class ParserEvaluationContext;
struct ParserConfig;
class CachedOriginFinder;
class TraitDeclOp;
enum class CallSyntax : uint8_t;
enum class CaptureConvention : uint8_t;

/// Capture represents a nested function value whose declaration is in the
/// parent function.
///
/// In the case of a __move_capture/__copy_capture, the 'value' of the capture
/// is an RValue defined in parent function, which is transferred into the
/// closure struct.
///
/// If the case of a captured reference, this an LValue for a 'var', a BValue
/// for a borrowed argument reference, etc.
class Capture {
public:
  Capture(CValue value, CaptureConvention kind, StringRef spelling)
      : value(value), kind(kind), spelling(spelling) {}
  CValue getValue() const { return value; }
  bool isCopy() const;
  bool isRef() const;
  /// The name of the capture
  StringRef getSpelling() const { return spelling; }
  CaptureConvention getCaptureConvention() const { return kind; }
  void setCaptureConvention(CaptureConvention convention) { kind = convention; }

private:
  CValue value;
  CaptureConvention kind;
  /// Store the name of the capture so we can emit meaningful error messages.
  StringRef spelling;
};

/// Store a list of parameter captures per closure type.
using ClosureParamCapture = std::pair<StringAttr, Type>;
using ClosureParamCaptures =
    DenseMap<StringAttr, SmallVector<ClosureParamCapture>>;

/// This enum indicates how much parsing and type checking has been done on
/// this declaration.
enum class DeclResolvedness : uint8_t {
  /// This declaration hasn't been parsed outside of its identifier being
  /// processed.  We don't know anything about its arguments, generic
  /// signature, etc.
  unparsed,

  /// This declaration has had its signature parsed and type checked, so we know
  /// what parameters and metaparameters it might take, but its body hasn't been
  /// processed.
  signature,

  /// This declaration has been fully type checked, including its body.  Any
  /// declarations within the body may not be fully resolved though.
  body,
};

/// This is state shared across multiple different instances of Parser
/// which are always shared across them.
class SharedState {
public:
  SharedState(llvm::SourceMgr &sourceMgr, ParserConfig &config);
  ~SharedState();

  MojoDiags diags; // Contains SourceMgr and MLIRContext pointers.
  const CompilationOptions &options;

  std::unique_ptr<DeclResolver> declResolver;
  std::unique_ptr<ClosureEmitter> closureEmitter;
  std::unique_ptr<DebugInfo::DIBuilder> diBuilder;
  ParserListener *parserListener;

  const mlir::StringAttr bufferNameIdentifier;

  /// Interned "extension:" name. Every scope that contains any extension also
  /// registers its extensions under this aggregate name (see the
  /// ExtensionDeclOp handling in SharedState::addDeclsForOp and the
  /// module-import paths in DeclResolver), so it serves as a cheap marker to
  /// skip extension lookups entirely in scopes that have none.
  const mlir::StringAttr extensionsScopeMarker;

  /// This is used to efficiently walk MLIR types to find embedded origins.
  CachedOriginFinder cachedOriginFinder;

  /// Find all ParamDeclRefAttr's in side the type at the current scope.
  void collectParamRefsInType(Type type,
                              SmallVectorImpl<ParamDeclRefAttr> &uses);

  llvm::SourceMgr &getSourceMgr() const { return diags.sourceMgr; }
  MLIRContext *getContext() const { return diags.context; }
  DeclResolver &getDeclResolver() const { return *declResolver; }
  ClosureEmitter &getClosureEmitter() const { return *closureEmitter; }

  /// Returns if we should diagnose missing doc strings.
  bool shouldDiagnoseMissingDocStrings() const;

  /// Get the library base path for documentation generation.
  StringRef getDocsBasePath() const { return docsBasePath; }

  /// Initialize the shared state for the given top-level decl.
  void initialize(ASTDecl &topLevelDecl);

  /// Return the top-level decl where modules can created in. This can only be
  /// used after the SharedState has been initialized.
  ASTDecl &getTopLevelDecl();

  /// This is the AST type that corresponds to TypeCheckErrorType.
  ASTType getTypeCheckErrorType() const;

  /// This is the decl for the builtin '!kgen.none' type.
  ASTType getNoneType() const;

  /// This returns a NoneAttr.
  NoneAttr getNoneAttr() const;

  /// Emit an error.
  MojoInflightDiag emitError(Location loc, const Twine &message = {});
  MojoInflightDiag emitError(llvm::SMLoc loc, const Twine &message = {});

  /// Emit a warning.
  MojoInflightDiag emitWarning(Location loc, const Twine &message = {});
  MojoInflightDiag emitWarning(llvm::SMLoc loc, const Twine &message = {});

  /// Inflate a lightweight SMLoc into an MLIR Location object for addition
  /// into the IR.
  Location translateLocation(llvm::SMLoc loc) const;
  FileLineColLoc createLocation(StringRef filename, unsigned line,
                                unsigned column);

  /// Allocate an expression node into the persistent bump pointer allocator.
  template <typename T, typename... Args>
  T *allocPersistent(Args &&...args) {
    void *node = persistentAllocator.Allocate(sizeof(T), llvm::Align::Of<T>());
    return new (node) T(std::forward<Args>(args)...);
  }

  /// memcpy the specified ArrayRef into the persistent allocator and return a
  /// pointer to the new data.  This cannot be used with things that have
  /// non-trivial copyctors/dtors because the expression allocator does run
  /// destructors.
  template <typename T>
  ArrayRef<T> getPersistentCopy(ArrayRef<T> elements) {
    if (elements.empty())
      return elements;

    size_t dataSize = sizeof(T) * elements.size();
    T *result = static_cast<T *>(
        persistentAllocator.Allocate(dataSize, llvm::Align::Of<T>()));
    memcpy((void *)result, elements.data(), dataSize);
    return ArrayRef<T>(result, elements.size());
  }

  /// memcpy the specified string data into the persistent allocator.
  StringRef getPersistentCopy(StringRef str) {
    auto result = getPersistentCopy(ArrayRef<char>(str.data(), str.size()));
    return StringRef(result.data(), result.size());
  }

  /// Lookup an operation inside the symbol table of the container decl.
  Operation *lookupSymbolIn(ASTDecl *container, StringAttr name);
  template <typename OpT>
  OpT lookupSymbolIn(ASTDecl *container, StringAttr name) {
    return dyn_cast_or_null<OpT>(lookupSymbolIn(container, name));
  }

  /// Set the symbol for the specified declaration (known to be an operation)
  /// into the MLIR symbol table for its container.  If the symbol is already
  /// declared in the same MLIR scope, then return the conflicting operation.
  Operation *uniquifyNameAndAddToParentSymbolTable(Operation *declOp);

  /// Shared state maintains an MLIR Block and deallocates it when the parser is
  /// torn down.  This can be used to allocate BlockArgument's that may or may
  /// not get used in the future.
  Block &getArgumentOwningBlock();

  /// Delete this decl and the operation associated with it. Handles all the
  /// related bookkeeping.
  void deleteDecl(ASTDecl &decl);

  //===--------------------------------------------------------------------===//
  // Name Lookup

  /// Return true if the specified type has a declared member with the specified
  /// name.
  bool typeHasMember(ASTType type, StringRef name, llvm::SMLoc loc);
  bool typeHasMember(ASTDecl &type, StringRef name, llvm::SMLoc loc);

  /// Perform a name lookup in the current scope and return the named
  /// declaration as a LookupResult.  If `searchParentScopes` is true, parent
  /// scopes are searched as well, as in unqualified name lookup.
  /// This will resolve any unresolved imports along the way.
  /// resolveTarget determines whether we resolve the ultimate declaration too.
  LookupResult lookupAndResolveDecl(StringRef name, llvm::SMLoc loc,
                                    ASTDecl &scope, bool searchParentScopes,
                                    bool resolveTarget = true);

  /// Perform a name lookup for a member in the specified type.
  /// This will resolve any unresolved imports along the way.
  /// resolveTarget determines whether we resolve the ultimate declaration too.
  LookupResult lookupAndResolveDecl(StringRef name, llvm::SMLoc loc,
                                    ASTType scope, bool searchParentScopes,
                                    bool resolveTarget = true);

  /// Perform a name lookup that collects ALL matching declarations instead of
  /// stopping at the first non-import match. This is useful for finding both
  /// the original struct and all extensions with the same name.
  LookupAllResult lookupAllDeclsWithName(StringRef name, llvm::SMLoc loc,
                                         ASTDecl &scope, bool resolve);

  /// Return the standard-library package op, but only when `std` was loaded
  /// from a prebuilt bytecode package (the form shipped to users). Returns null
  /// for a source-built `std` or before `std` is loaded. The missing-import
  /// suggestion (`DeclResolver::findUniqueStdlibImportFor`) uses this to walk
  /// the package tree.
  PackageOp getPrecompiledStdlibPackage();

  /// Materialize a lazily-loaded op from the precompiled standard-library
  /// bytecode package, making its nested ops visible. No-op if the op is
  /// already materialized or `std` is not a bytecode package.
  void materializePrecompiledStdlibOp(Operation *op);

  /// Given a parameter expression call to a function marked
  /// @always_inline("builtin"), scan the function to form an inlined parameter
  /// expression representation of the function given the specified argument
  /// values, then return the resultant expression.  If the function cannot be
  /// handled as a builtin, emit an error (when isError is true) and return
  /// null.
  TypedAttr foldInlineBuiltinFunction(ArrayRef<TypedAttr> operands,
                                      Location callLoc, bool isError);

  //===--------------------------------------------------------------------===//
  // Module Resolution

  /// A parsed import path: module-path components plus the number of leading
  /// dots of a relative import (0 for an absolute import).
  struct ImportPath {
    ImportPath() = default;
    ImportPath(std::initializer_list<StringRef> components,
               unsigned relativeLevel = 0)
        : components(components), relativeLevel(relativeLevel) {}
    ImportPath(ArrayRef<StringRef> components, unsigned relativeLevel = 0)
        : components(components.begin(), components.end()),
          relativeLevel(relativeLevel) {}

    /// Render the path as a Mojo ASTDecl name: the joined dotted-string form
    /// used for decl and import-op names. It is ambiguous for components that
    /// themselves contain periods (e.g. escaped identifiers) and is not a
    /// source-faithful rendering for diagnostics.
    std::string toDottedString() const {
      return std::string(relativeLevel, '.') + llvm::join(components, ".");
    }

    /// Convert to/from the unambiguous `#lit.import_path` attribute form,
    /// which import ops should prefer over the dotted-string forms above.
    ImportPathAttr toAttr(MLIRContext *ctx) const {
      return ImportPathAttr::get(ctx, relativeLevel, components);
    }
    static ImportPath fromAttr(ImportPathAttr attr) {
      ImportPath path;
      path.relativeLevel = attr.getRelativeLevel();
      for (StringAttr component : attr.getComponents())
        path.components.push_back(component.getValue());
      return path;
    }

    SmallVector<StringRef, 4> components;
    unsigned relativeLevel = 0;
  };

  /// Import the specified module or package, returning the decl. Always returns
  /// a valid decl, even if a corresponding module or package could not be
  /// found.
  ASTDecl &importModule(const ImportPath &path, PackageOp currentPackage,
                        llvm::SMLoc loc);

  /// Return true if the package has a nested module with the given name in the
  /// module-state cache. The package body must be resolved before calling this
  /// for the result to be accurate.
  bool hasNestedModule(PackageOp packageOp, StringRef name) const;

  /// Try to import a direct submodule with the given name, returning the
  /// submodule's decl, or null if no such submodule exists.
  ASTDecl *tryImportSubModule(ASTDecl &parent, StringRef name, llvm::SMLoc loc);

  /// Return the decls of every nested module/package currently materialized in
  /// the package's module-state cache. These are navigable but unlisted (not
  /// in the package's importable scope).
  SmallVector<ASTDecl *> getNestedModuleDecls(PackageOp packageOp) const;

  /// Scan a source package's directory and register a child decl for every
  /// sibling module/sub-package - a deferred `FileModuleOp` for each `.mojo`
  /// file and a (already-deferred) `PackageOp` for each sub-package. The
  /// children are unlisted (in the module-state cache + package IR, never the
  /// package's importable scope) and their bodies/files are opened lazily.
  void registerSourcePackageChildren(ASTDecl &packageDecl);

  /// Open and lex the source file of a deferred module, wiring up its parse
  /// cursor so its body can be resolved. The loc is the location that triggered
  /// materialization (e.g. the import that first referenced the module).
  LogicalResult materializeDeferredModule(ASTDecl &decl, llvm::SMLoc loc);

  /// Create a new module with the given name, location, and body.
  ASTDecl &createModule(StringRef moduleName,
                        const llvm::MemoryBuffer *moduleBuffer,
                        FileLineColLoc loc);

  /// Create a new package with the given path and desired name.
  ASTDecl &createPackage(StringRef path, StringRef name);

  /// Create a new package from the path to the given binary package.
  ASTDecl &createBinaryPackage(StringRef path, StringRef name);

  /// Return the source path for the given module decl, or nullopt if the decl
  /// doesn't have a source path.
  std::optional<std::string> getModuleSourcePath(ASTDecl &module);

  /// Resolve a declaration that originated from bytecode to the given
  /// resolvedness.
  LogicalResult resolveDeclFromBytecode(ASTDecl &decl,
                                        DeclResolvedness resolvedness);

  /// Function used to look up and resolve a decl with the given mangled name.
  ASTDecl *lookupAndResolveMangledDecl(StringAttr leafRef, SMLoc loc,
                                       ASTDecl &container,
                                       DeclResolvedness howResolved);

  /// When injecting a decl reference as a symbol reference into the IR, we have
  /// to make sure that the decl is at least signature resolved. This function
  /// takes a type and ensures that any referenced decls are resolved.
  LogicalResult resolveDeclReferencesIn(SMLoc loc, Type type);

  /// Look up the ASTDecl for a function symbol, loading it from a bytecode
  /// package on demand if it has not been resolved yet. Decls from bytecode are
  /// resolved lazily, so a symbol may not be registered until first referenced.
  /// Returns null if resolution fails.
  ASTDecl *resolveAndGetFuncDecl(SymbolRefAttr symbol, SMLoc loc);

  /// Finalize any imported bytecode modules. This should be called after all
  /// decls have been resolved, as this will erase bytecode operations attached
  /// to decls that have not been resolved.
  LogicalResult finalizeImportedBytecodeModules();

  /// Get the list of files included while processing all modules.
  ArrayRef<std::string> getIncludedFiles() const;

  /// Traverse the directories available for importing modules and packages,
  /// calling the given callback for each directory found.
  void
  traverseImportDirectories(unsigned importBufferFileId,
                            function_ref<WalkResult(StringRef)> callback) const;

  struct ModuleSpec {
    /// Mojo import kinds. Enumerator order defines resolution priority: a
    /// lower value outranks a higher one when several candidates share a stem.
    enum class Kind {
      SourcePackage,
      Precompiled,
      SourceModule,
      SourceDir,
    };

    /// Importable name: whole filename for directories, stem for files
    std::string name;
    std::filesystem::path path;
    Kind kind;

    /// For a SourceDir resolved from the import path: the dotted chain of
    /// directory names (including `name`) relative to the import roots. A
    /// plain directory is a namespace whose one name may span several roots, so
    /// submodules resolve against these components under every import root, not
    /// against `path`, which records only the first root's portion. Empty for
    /// every other kind, and for directories nested inside a source package,
    /// which stay single-homed and resolve through `path`.
    ///
    /// For example, given `-I one -I two` and
    ///
    ///   one/foo/bar/baz.mojo
    ///   two/foo/bar/qux.mojo
    ///
    /// the name `foo.bar` is one namespace spanning both roots. Its spec is
    ///
    ///   { name = "bar", path = "one/foo/bar", kind = SourceDir,
    ///     namespaceComponents = ["foo", "bar"] }
    ///
    /// and resolving `foo.bar.qux` searches `<root>/foo/bar` under every
    /// root, finding two/foo/bar/qux.mojo even though `path` names the
    /// first root's directory. The closed candidate baz.mojo gets
    ///
    ///   { name = "baz", path = "one/foo/bar/baz.mojo", kind = SourceModule,
    ///     namespaceComponents = [] }
    std::vector<std::string> namespaceComponents = {};

    bool isNamespace() const {
      return kind == Kind::SourceDir && !namespaceComponents.empty();
    }

    /// Return the module classification of a given path, optionally matching a
    /// specific name. Returns std::nullopt if not a (matching) ModuleSpec.
    static std::optional<ModuleSpec> classify(const std::filesystem::path &path,
                                              llvm::StringRef moduleName = "");

    /// Return true if the import candidate (kind, path) takes precedence over
    /// the other. Higher-priority kinds win; between candidates of
    /// the same kind (e.g. directories foo and foo.bar, which share the stem
    /// 'foo'), the lexicographically smaller filename wins so the result does
    /// not depend on the platform's directory iteration order.
    bool takesImportPrecedence(ModuleSpec other) {
      if (kind != other.kind)
        return kind < other.kind;
      return path.filename() < other.path.filename();
    }

    bool isPrecompiled() const { return kind == ModuleSpec::Kind::Precompiled; }

    bool isSourcePackage() const {
      return kind == ModuleSpec::Kind::SourcePackage;
    }

    bool isSourcePackageLike() const {
      return kind == ModuleSpec::Kind::SourcePackage ||
             kind == ModuleSpec::Kind::SourceDir;
    }

    bool isSourceModule() const {
      return kind == ModuleSpec::Kind::SourceModule;
    }

    /// The canonical form of `path`, with symlinks resolved: the identity of
    /// the entity this spec names.
    std::string canonicalPath() const;
  };

  /// Resolve the absolute path for a given module name. Returns nullopt if the
  /// module cannot be found.
  std::optional<ModuleSpec> resolveModulePath(StringRef moduleName,
                                              llvm::SMLoc includeLoc);

  /// Resolve the absolute path for a given module name within the provided
  /// directory. Returns nullopt if the module cannot be found.
  ///
  /// \p isInsideSourcePackage is true for submodule search inside a source
  /// package. In that mode, any `.mojoc` candidate is ignored.
  std::optional<SharedState::ModuleSpec>
  resolveModulePath(StringRef moduleName, StringRef includeDir,
                    bool ignorePrebuilt, bool isInsideSourcePackage);

  /// Collect the portion directories of the namespace described by
  /// `parentSpec`: for each import directory visible from
  /// `importBufferFileId`, descend the spec's namespace components as plain
  /// directories. Portions are returned in traversal order, deduplicated.
  ///
  /// A "portion" is one root's directory contributing to the namespace. For the
  /// example on `namespaceComponents`, the portions of `foo.bar` are
  ///
  ///   [ one/foo/bar, two/foo/bar ]
  ///
  /// The search path always contains the same directories for every
  /// import site, and portions are recomputed from it at each resolution
  /// step, so a root where `foo` is missing, or is claimed by a source
  /// package or module file (a closed candidate), simply contributes no
  /// portion.
  SmallVector<std::string>
  collectNamespacePortions(const ModuleSpec &parentSpec,
                           unsigned importBufferFileId);

  /// Resolve the name as a submodule of the namespace described by
  /// `parentSpec`, searching every portion. Returns every distinct thing the
  /// name could be: closed (non-directory) candidates win over directory
  /// candidates, which merge into a single nested-namespace spec rather than
  /// competing. More than one element therefore means the import is
  /// ambiguous; several portions provide a closed candidate.
  SmallVector<ModuleSpec>
  resolveNamespaceSubModule(StringRef moduleName, const ModuleSpec &parentSpec,
                            unsigned importBufferFileId);

  void registerWrapperBuffer(unsigned bufferId, StringRef wrappedSourcePath);
  std::optional<StringRef> getWrappedSourcePath(unsigned bufferId) const;

  //===--------------------------------------------------------------------===//
  // Debug Info

  /// Generate a debug subprogram for this function and set it in its location.
  void setLocationDebugScope(FnOp funcOp);
  /// Get the debug source name for a symbol.
  DebugInfo::SourceNameAttr getSourceName(mlir::SymbolOpInterface op);

  //===--------------------------------------------------------------------===//
  // Listener Interface

  /// Notify the parser listener, if present, of a parsed alias decl.
  void notifyListenerOnAliasDecl(ASTDecl &decl, SMLoc identifierLoc);

  /// Notify the parser listener, if present, of a parsed argument decl.
  void notifyListenerOnArgumentDecl(ASTDecl &decl, StringRef argName,
                                    SMLoc identifierLoc);

  /// Notify the parser listener, if present, of a parsed function.
  void notifyListenerOnFunctionDecl(ASTDecl &decl, SMLoc identifierLoc);

  /// Notify the parser listener that an import is currently being resolved.
  void notifyListenerOnImport(SMLoc importLoc);

  /// Notify the parser listener, if present, that an import of a module within
  /// the given package is currently being resolved. `getPackageDecl` is a
  /// function called to get the package decl if the listener needs it.
  void notifyListenerOnImport(SMLoc importLoc,
                              function_ref<ASTDecl &()> getPackageDecl);

  /// Notify the parser listener, if present, that a member within the given
  /// decl is being looked up. `searchParentScopes` is true if the lookup is not
  /// restricted to just the given decl.
  void notifyListenerOnMemberLookup(ASTDecl &decl, SMLoc lookupLoc,
                                    bool searchParentScopes = false);
  /// Notify the parser listener, if present, that a member within the given
  /// decl is being looked up. `getDeclFn` is a function called to get the decl
  /// if the listener needs it. `searchParentScopes` is true if the lookup is
  /// not restricted to just the given decl.
  void notifyListenerOnMemberLookup(SMLoc lookupLoc,
                                    function_ref<ASTDecl &()> getDeclFn,
                                    bool searchParentScopes = false);

  /// Notify the parser listener, if present, that a new `module` decl has been
  /// created by the parser.
  void notifyListenerOnModuleDecl(ASTDecl &decl, SMLoc identifierLoc);

  /// Notify the parser listener, if present, that a new import of the form
  /// `from Module [as Alias]` has been resolved by the parser. The location
  /// corresponds to the start of the module path (not to its optional alias);
  /// the listener is notified once per path component.
  void notifyListenerOnModuleImport(ASTDecl &decl, const ImportPath &modulePath,
                                    SMLoc loc);

  /// Notify the parser listener, if present, of a parsed function or struct
  /// parameter.
  void notifyListenerOnParameterDecl(ASTDecl &decl, SMLoc identifierLoc);

  /// Notify the parser listener, if present, that a new `struct` declaration
  /// has been resolved by the parser.
  void notifyListenerOnStructDecl(ASTDecl &decl, SMLoc identifierLoc);

  /// Notify the parser listener, if present, that a new `struct field`
  /// declaration has been resolved by the parser.
  void notifyListenerOnStructFieldDecl(ASTDecl &decl, SMLoc identifierLoc);

  /// Notify the parser listener, if present, that a new `trait` declaration
  /// has been resolved by the parser.
  void notifyListenerOnTraitDecl(ASTDecl &decl, SMLoc identifierLoc);

  /// Notify the parser listener, if present, that a new `let` or `var`
  /// declaration has been resolved by the parser.
  void notifyListenerOnVariableDecl(ASTDecl &decl, SMLoc identifierLoc);

  /// Notify the parser listener, if present, that a new reference has been
  /// resolved by the parser, i.e. its declarations are known.
  void notifyListenerOnRef(ArrayRef<ASTDecl *> decls, StringRef spelling,
                           SMLoc loc);
  void notifyListenerOnRef(ArrayRef<ASTDecl *> decls, StringRef spelling,
                           SourceRange range);

  /// Notify the parser listener, if present, that a new reference from an
  /// expression has been resolved.
  void notifyListenerOnRef(ArrayRef<ASTDecl *> decls, StringRef spelling,
                           const ExprNode *expr);
  void notifyListenerOnRef(ArrayRef<ASTDecl *> decls, StringRef spelling,
                           const ExprNode *expr, CallSyntax syntax);

  /// Notify the parser listener, if present, that a call is being resolved with
  /// the given operands.
  void notifyListenerOnCall(ArrayRef<ASTDecl *> decls, SMLoc rparenLoc,
                            CallSyntax syntax,
                            const CallOperands &callOperands);

  /// Notify the listener, if present, that parameter operands are being bound
  /// to one of the given decls.
  void notifyListenerOnParameterBinding(ArrayRef<ASTDecl *> decls,
                                        llvm::SMLoc rsquareLoc,
                                        ArrayRef<Operand> operands);

  //===--------------------------------------------------------------------===//
  // Builtin Module

  /// Return true if the parser has builtins available.
  bool hasBuiltinModule() const;

  /// Lookup a builtin trait like `AnyType`, `Copyable`, `Movable` etc.  On
  /// error this returns null but does not print an error.
  ASTDecl *lookupBuiltinTrait(StringRef traitName, SMLoc loc);

  /// A handy version of lookupBuiltinTrait, which returns the trait type.
  TraitType lookupBuiltinTraitType(StringRef traitName, SMLoc loc);

  /// Lookup the specified name, and check that it is a non-parameterized type.
  /// This emits a diagnostic on error and returns null, or returns the ASTDecl
  /// of the type on success.
  ASTDecl *lookupNamedTypeDecl(StringRef name, ASTDecl &context,
                               llvm::SMLoc loc);

  /// Lookup the specified name in builtin.prelude when builtin is enabled. If
  /// builtin is disabled, search from the provided `context`. The function
  /// check whether it is a non-parameterized type. This emits a diagnostic on
  /// error and returns null, or returns the type on success.
  ASTType lookupBuiltinType(StringRef name, ASTDecl &context, llvm::SMLoc loc);

  /// Get a builtin type, or emit an error and return TypeCheckErrorType if
  /// invalid. These never return null.
  ASTDecl *getBuiltinCoroutineType(llvm::SMLoc loc);
  ASTDecl *getBuiltinDevicePassableTrait(llvm::SMLoc loc);
  ASTDecl *getBuiltinRaisingCoroutineType(llvm::SMLoc loc);
  ASTType getStandardCollectionType(llvm::SMLoc loc, StringRef name);
  ASTType getBuiltinSliceType(llvm::SMLoc loc, StringRef name);
  ASTType getBuiltinStubsMLIRType(llvm::SMLoc loc);

  /// Lookup a builtin special function overload set.
  ArrayRef<ASTDecl *> getBuiltinFunction(ASTDecl &context,
                                         const ImportPath &modulePath,
                                         StringRef fnName, llvm::SMLoc loc);
  ArrayRef<ASTDecl *> getBuiltinFunction(ASTDecl &module, StringRef fnName,
                                         llvm::SMLoc loc);

  struct Impl;
  Impl &getImpl() const { return *impl; }

  /// Given a signature [Int](y:Int) -> Int for example, return the trait. If
  /// there is not a trait already generated, the compiler will generate the
  /// following:
  ///  trait Closure_Int_yInt_Int(Movable, AnyType):
  ///      def __call__(mut self, y: Int) -> Int:
  ///         ...
  ASTDecl *getOrCreateClosureTrait(SMLoc loc, ASTDecl &moduleDecl,
                                   FnTypeGeneratorType sig);

  /// Get or create the universal closure trait, we don't care about the
  /// signatures and we can adapt the trait to any signature.
  ASTDecl *getUniversalParametricClosureTrait();

  bool isUniversalParametricClosureTrait(ASTDecl *trait) {
    return trait == getUniversalParametricClosureTrait();
  }

  /// Get or create a struct that defines conformance of targetTrait in terms of
  /// sourceTrait.
  ASTDecl *getOrCreateExtension(SMLoc loc, TraitDeclOp sourceTrait,
                                TraitDeclOp targetTrait, ASTType sourceMetaType,
                                ASTDecl *moduleDecl);
  /// Function used to create a thunk. This API is limited intentionally to
  /// ensure that the creation is transaction. This is important to retain
  /// invariants with packaging.
  using CreateThunkFn = FnOp (*)(Attribute, ASTDecl &moduleDecl, SMLoc useLoc);

  /// This gets a function conversion thunk between the two provided function
  /// types within the provided module, or creates one if needed.
  FnOp getOrCreateFunctionThunk(Attribute key, CreateThunkFn create,
                                SMLoc useLoc);

  /// Given a scope that refers to a nested function, return the set of captured
  /// values. The name of the capture is paired with the metadata.
  const llvm::MapVector<StringRef, Capture> &
  getCaptureRangeInScope(ASTDecl &scope);

  /// Given a nested function, a capture value, and the corresponding capture
  /// ASTDecl, store the capture associated with the nested function.
  void addCaptureToScope(ASTDecl &scope, ASTDecl *captureDecl, Capture capture);
  CaptureConvention defaultCaptureConventionInScope(ASTDecl &scope);
  /// If a capture has already been registered in this scope it means the
  /// capture instance in the parent has already been generated.
  bool captureInstanceExistsInScope(ASTDecl &scope, StringRef spelling);
  /// Override the default capture convention for captures in this scope.
  void setDefaultCaptureForScope(ASTDecl &scope,
                                 CaptureConvention defaultConvention);

  /// Return the captured parameters map for all closures defined in the
  /// function represented by \p op. Returns nullptr if no captures have been
  /// registered for this op.
  ClosureParamCaptures *getClosureParamCapturesForOp(Operation *op);

  /// Look up the captures registered for the closure named \p closureName as
  /// visible from \p startOp.
  ArrayRef<ClosureParamCapture>
  lookupClosureCaptureFromOp(Operation *startOp, StringAttr closureName);
  /// Set the captured parameters map for a given function.
  void setClosureParamCaptures(ASTDecl &functionDecl,
                               ClosureParamCaptures closureParamCaptures);

  /// Add an entry to the captured closures map of the given function.
  void addClosureParamCaptures(ASTDecl &functionDecl, StringAttr closureName,
                               SmallVector<ClosureParamCapture> captures);

  /// These two methods are used to memoize whether a type is implicitly
  /// convertible to another type, which includes overload resolution etc.
  std::optional<bool> getCachedImplicitConvertibility(ASTType from, ASTType to);
  void cacheImplicitConvertibility(ASTType from, ASTType to,
                                   bool isConvertible);

  /// These two methods memoize ASTDecl::doesNominalTypeConformTo for the
  /// common, assumption-free case, keyed by (type decl, required trait,
  /// concrete type). Only definitive (yes/no) results are cached; `unknown` is
  /// phase-dependent and never stored. `conforms` is the `yes`/`no` result as a
  /// bool.
  std::optional<bool> getCachedNominalConformance(const ASTDecl *decl,
                                                  TraitType trait,
                                                  ASTType concreteType);
  void cacheNominalConformance(const ASTDecl *decl, TraitType trait,
                               ASTType concreteType, bool conforms);

  /// Get the attribute evaluation context.
  ParserEvaluationContext &getEvaluationContext();

  /// Create a ParameterEvaluator configured with the parser's evaluation
  /// context.
  ParameterEvaluator getParameterEvaluator();
  ParameterEvaluator getParameterEvaluator(ArrayRef<ParamDeclAttr> paramDecls,
                                           ArrayRef<TypedAttr> paramValues);

private:
  /// The physical entity a module or package is read out of, shared by every
  /// binding of it.
  struct ModuleOrigin;

  /// The internal state of an imported module or package.
  struct ModuleState;

  /// Add magic things to the builtins decl when parsing starts.
  void addBuiltinTypes(ASTDecl &builtinsDecl);

  /// Look up a builtin type; either by finding it in the cache or resolving it
  /// from a module and caching the result.
  ASTType getCachedBuiltinType(const ImportPath &path, StringRef name,
                               llvm::SMLoc loc);
  /// Look up a builtin type decl; either by finding it in the cache or
  /// resolving it from a module and caching the result.
  ASTDecl *getCachedBuiltinTypeDecl(const ImportPath &path, StringRef name,
                                    llvm::SMLoc loc);

  /// Import the specified module or package, returning the module state.
  /// Always returns a valid module state, even if the module could not be
  /// found. `isImplicit` marks a package pulled in by the compiler rather than
  /// a user `import`.
  ModuleState &importModuleState(const ImportPath &path, ASTDecl *context,
                                 llvm::SMLoc loc, bool isImplicit = false);

  /// Look up a module by its name, in the specified parent scope, in the module
  /// cache. Returns nullptr on a miss. On a hit:
  ///   * Prevent module self-imports
  ///   * Memoizes the import loc on first resolution
  ModuleState *lookupModuleCache(StringRef name, ASTDecl *parentDecl,
                                 ModuleState *parentState, llvm::SMLoc loc,
                                 llvm::SMLoc identifierLoc, bool emitErrors);

  /// Import the specified module or package nested within the given parent
  /// decl, returning the module state. Always returns a valid module state,
  /// even if the module could not be found.
  ModuleState &importSubModuleState(StringRef name, ASTDecl *parentDecl,
                                    llvm::SMLoc loc, llvm::SMLoc identifierLoc);

  ModuleState *importSubModuleStateImpl(StringRef name, ASTDecl *parentDecl,
                                        llvm::SMLoc loc,
                                        llvm::SMLoc identifierLoc,
                                        bool emitErrors);

  /// Open (and lex-prepare) the module source file at the given path within the
  /// source manager, reusing an already-open buffer if present. Returns null
  /// without emitting a diagnostic if the file cannot be opened.
  const llvm::MemoryBuffer *openModuleFile(StringRef path, llvm::SMLoc loc);

  /// Import the specified module or package, which contains `.` indexing,
  /// returning the module state. Always returns a valid module state, even if
  /// the module could not be found.
  ModuleState &importRelativeModuleState(const ImportPath &path,
                                         ASTDecl *parentDecl, llvm::SMLoc loc);

  /// Returns the ModuleOrigin that the spec names, shared by every binding of
  /// the same entity.
  ///
  /// The bound name is in the given parent decl's scope. An origin can carry
  /// one symbol path only, since re-anchoring an artefact rewrites its
  /// references to a single path. Binding one entity under two names is a
  /// dual-mount error.
  ///
  /// Null is a success: a namespace spans several import roots, so names no
  /// single entity.
  ErrorOr<ModuleOrigin *> getOrCreateModuleOrigin(const ModuleSpec &spec,
                                                  StringRef boundName,
                                                  ASTDecl &parentDecl);

  /// Shared core of createModuleState and createDeferredModuleState: create the
  /// `FileModuleOp` + unlisted decl + nested module state. The caller supplies
  /// the parse cursor (valid for an already-open module, invalid for a deferred
  /// one) and is responsible for importing builtins (eagerly, or at
  /// materialization for a deferred module).
  ModuleState &createFileModuleState(StringAttr declName,
                                     ModuleState &parentState,
                                     FileLineColLoc loc, llvm::SMLoc declLoc,
                                     LexerCursor cursor, LexerCursor endCursor,
                                     const ModuleSpec &spec);

  /// Create a new module state with the given name, location, and body.
  ModuleState &createModuleState(StringAttr declName,
                                 const llvm::MemoryBuffer *moduleBuffer,
                                 ModuleState &parentState, FileLineColLoc loc,
                                 const ModuleSpec &spec);

  /// Create a module state for a source module whose file has not been opened.
  ModuleState &createDeferredModuleState(ModuleSpec moduleSpec,
                                         ModuleState &parentState);

  /// Create a new module state for a package with the given spec, location,
  /// and body. The importLoc, if valid, is the location of the `import` that
  /// pulled the package in. The spec's kind must be a SourcePackage or
  /// SourceDir.
  ModuleState &createPackageState(ModuleSpec moduleSpec,
                                  ModuleState &parentState, SMLoc importLoc);

  /// Create a new module state for a binary package with the given spec.
  ModuleState &createBinaryPackageState(SMLoc loc, const ModuleSpec &spec,
                                        ModuleState &parentState);

  /// Returns the dotted module path of `target` below `root`, or an empty
  /// string when `target` is not nested below `root`.
  static std::string moduleMountPath(const ModuleState &root,
                                     const ModuleState &target);

  /// Create an error module state and emit the given error message. If
  /// `unlisted` is set, the erroneous decl is not registered in
  /// `errorContext`'s name table. A non-empty `note` is attached to the error.
  ModuleState &createErrorModuleState(SMLoc loc, StringAttr name,
                                      ASTDecl &errorContext,
                                      const Twine &errorMsg,
                                      bool unlisted = false,
                                      const Twine &note = {});

  /// Implicitly import the builtin modules into the given module decl.
  void importBuiltinModules(ASTDecl &moduleDecl);

  /// This is used for memory that lives as long as the global parser does.
  llvm::BumpPtrAllocator persistentAllocator;

  /// A flag indicating if prebuilt packages should not be considered during
  /// parsing.
  bool disablePrebuiltPackages = false;

  /// If true, auto-import the builtin package.
  bool useBuiltinModule = true;

  /// Base library path prefix for generated documentation links.
  std::string docsBasePath;

  std::unique_ptr<Impl> impl;
};

/// This class is intended to be used as a convenience base class for subsystems
/// that want to have access to various SharedState functionality in a
/// convenient way.
class SharedStateUser {
public:
  SharedStateUser(SharedState &shared) : shared(shared) {}

  /// This reference provides direct access to SharedState for anything
  /// fancy.
  SharedState &shared;

  // Convenience forwarding functions used pervasively through the frontend.

  MLIRContext *getContext() const { return shared.getContext(); }
  llvm::SourceMgr &getSourceMgr() const { return shared.getSourceMgr(); }
  DeclResolver &getDeclResolver() const { return shared.getDeclResolver(); }

  mlir::Location translateLocation(SMLoc loc) {
    return shared.translateLocation(loc);
  }

  /// Emit an error.
  MojoInflightDiag emitError(Location loc, const Twine &message = {}) const {
    return shared.emitError(loc, message);
  }
  MojoInflightDiag emitError(llvm::SMLoc loc, const Twine &message = {}) const {
    return shared.emitError(loc, message);
  }

  /// Emit a warning.
  MojoInflightDiag emitWarning(Location loc, const Twine &message = {}) const {
    return shared.emitWarning(loc, message);
  }
  MojoInflightDiag emitWarning(llvm::SMLoc loc,
                               const Twine &message = {}) const {
    return shared.emitWarning(loc, message);
  }
  // Allow assignments in derived classes.
  void operator=(const SharedStateUser &other) {
    assert(&other.shared == &shared);
  }
};

/// This is the result of lookupDecl.
class LookupResult {
  enum Kind {
    kSuccess,   //<- Lookup succeeded and result is non-null.
    kFailure,   //<- Lookup failed to find something of this name.
    kErroneous, //<- Lookup found an error, but it is already diagnosed.
  } kind;

  /// This is non-empty when we find something: in the case of a failure, we
  /// found entities that we can't use, e.g. we found things in our local scope
  /// that are not "self." qualified.  This points to the symbol entry in an
  /// ASTDecl, so the pointer is stable.
  ArrayRef<ASTDecl *> decls;
  LookupResult(Kind kind, ArrayRef<ASTDecl *> decls)
      : kind(kind), decls(decls) {}

public:
  static LookupResult getSuccess(ArrayRef<ASTDecl *> decls) {
    assert(!decls.empty() && "cannot form successful lookup without decls");
    return {kSuccess, decls};
  }
  /// Failure means that lookup failed, but they can still have decls
  /// attached for diagnostic purposes.
  static LookupResult getFailure(ArrayRef<ASTDecl *> decls) {
    return {kFailure, decls};
  }
  static LookupResult getErroneous() { return {kErroneous, {}}; }

  /// Return decls only if lookup was a success, because failures can
  /// also store decls for diagnostic purposes.
  ArrayRef<ASTDecl *> getIfSuccess() const {
    return isSuccess() ? decls : ArrayRef<ASTDecl *>{};
  }
  /// Return decls from a failed lookup, for diagnostic purposes.
  ArrayRef<ASTDecl *> getIfFailure() const {
    return isFailure() ? decls : ArrayRef<ASTDecl *>{};
  }
  bool isSuccess() const { return kind == kSuccess; }
  bool isFailure() const { return kind == kFailure; }
  bool isErroneous() const { return kind == kErroneous; }
  void setToFailure() {
    kind = kFailure;
    decls = {};
  }
};

/// This is the result of lookupAllDeclsWithName that collects all matching
/// declarations instead of stopping at the first non-import match.
class LookupAllResult {
  enum Kind {
    kSuccess,   //<- Lookup succeeded and result is non-empty.
    kFailure,   //<- Lookup failed to find something of this name.
    kErroneous, //<- Lookup found an error, but it is already diagnosed.
  } kind;

  /// This contains all matching declarations found during lookup.
  /// Unlike LookupResult, this owns the storage to allow collecting
  /// declarations from multiple scopes.
  std::vector<ASTDecl *> decls;
  LookupAllResult(Kind kind, std::vector<ASTDecl *> decls)
      : kind(kind), decls(std::move(decls)) {}

public:
  static LookupAllResult getSuccess(std::vector<ASTDecl *> decls) {
    assert(!decls.empty() && "cannot form successful lookup without decls");
    return {kSuccess, std::move(decls)};
  }
  /// Failure means that lookup failed, but they can still have decls
  /// attached for diagnostic purposes.
  static LookupAllResult getFailure(std::vector<ASTDecl *> decls) {
    return {kFailure, std::move(decls)};
  }
  static LookupAllResult getErroneous() { return {kErroneous, {}}; }

  /// Return decls only if lookup was a success, because failures can
  /// also store decls for diagnostic purposes.
  ArrayRef<ASTDecl *> getIfSuccess() const {
    if (isSuccess())
      return decls;
    else
      return {};
  }
  /// Return decls from a failed lookup, for diagnostic purposes.
  ArrayRef<ASTDecl *> getIfFailure() const {
    if (isFailure())
      return decls;
    else
      return {};
  }
  bool isSuccess() const { return kind == kSuccess; }
  bool isFailure() const { return kind == kFailure; }
  bool isErroneous() const { return kind == kErroneous; }
};

/// If a crash happens while this object is live, it will print out the message
/// and format contextual information for the code at the specified source
/// location. This makes it easier to debug parser/typechecker/IR emission
/// related bugs.
class CrashReporter : public llvm::PrettyStackTraceEntry {
public:
  // When 'loc' is null, this reporter doesn't do anything.
  CrashReporter(SMLoc loc, const char *message, SharedState &shared)
      : loc(loc), message(message), shared(shared) {}

  /// print - Emit information about this stack frame to OS.
  virtual void print(raw_ostream &os) const override;

public:
  SMLoc loc;
  const char *message;
  SharedState &shared;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_SHAREDSTATE_H
