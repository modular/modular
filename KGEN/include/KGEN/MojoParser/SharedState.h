//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the base class for Mojo parsers that is common between
// expression and statement parsing.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_SHAREDSTATE_H
#define KGEN_MOJOPARSER_SHAREDSTATE_H

#include "KGEN/MojoParser/IRValues.h"
#include "Support/Compiler/Diags.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "mlir/IR/BuiltinOps.h"
#include <filesystem>

namespace M::KGEN {
class CompilationOptions;
class NoneAttr;
class ParamDeclAttr;
class SignatureType;
} // namespace M::KGEN

namespace M::LLCL {
class Runtime;
} // namespace M::LLCL

namespace M::KGEN::LIT {
class ASTDecl;
class ASTType;
class DeclResolver;
class ExprNode;
struct Operand;
class FileModuleOp;
class FuncOp;
class LookupResult;
class PackageOp;
class ParserListener;
class StructDeclOp;
struct CallOperands;
struct ParserConfig;
enum class CallSyntax : uint8_t;

/// Capture represents a nested function value whose declaration is in the
/// parent function.
class Capture {
public:
  Capture(CValue value, bool isMove = false) : value(value), isMove(isMove) {}
  CValue getValue() const { return value; }
  bool isMoveCapture() const { return isMove; }

  /// Get the underlying MLIR value.
  Value getMlirValue() const;

private:
  /// The value of the capture.
  CValue value;
  /// True if the capture is by move, otherwise it is by copy.
  bool isMove;
};

/// This enum indicates how much parsing and type checking has been done on
/// this declaration.
enum class DeclResolvedness : int8_t {
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
  fully
};

/// This is state shared across multiple different instances of Parser
/// which are always shared across them.
class SharedState {
public:
  SharedState(llvm::SourceMgr &sourceMgr, ParserConfig &config);
  ~SharedState();

  Diags diags; // Contains SourceMgr and MLIRContext pointers.
  const CompilationOptions &options;

  std::unique_ptr<DeclResolver> declResolver;
  std::unique_ptr<DebugInfo::DIBuilder> diBuilder;
  ParserListener *parserListener;
  LLCL::Runtime &runtime;

  const mlir::StringAttr bufferNameIdentifier;

  llvm::SourceMgr &getSourceMgr() const { return diags.sourceMgr; }
  MLIRContext *getContext() const { return diags.context; }

  /// Returns if we should emitting warnings for missing doc strings.
  bool shouldWarnMissingDocStrings() const;

  /// Returns if we should generate lifetimes using !lit.ref<>.
  bool useExperimentalLifetimes() const;

  /// Initialize the shared state for the given top-level decl.
  void initialize(ASTDecl &topLevelDecl);

  /// Return the top-level decl where modules can created in. This can only be
  /// used after the SharedState has been initialized.
  ASTDecl &getTopLevelDecl();

  /// This is the AST type that corresponds to TypeCheckErrorType.
  ASTType getTypeCheckErrorType() const;

  /// This is the decl for the builtin '!kgen.none' type.
  ASTType getNoneType() const;

  /// This is the decl for the builtin '!lit.lifetime' type, aka Lifetime.
  ASTType getLifetimeType() const;

  /// This returns a NoneAttr.
  NoneAttr getNoneAttr() const;

  /// Get a mangled parameter name based on the source location.
  StringAttr getMangledParameterName(const Twine &baseName, SMLoc loc);

  /// Return a mangled version of the given module name. This is used to avoid
  /// conflicts with symbols that are actually visible.
  StringAttr getMangledModuleName(MLIRContext *ctx, StringRef moduleName);

  /// Emit an error.
  InflightDiag emitError(Location loc, const Twine &message = {});
  InflightDiag emitError(llvm::SMLoc loc, const Twine &message = {});

  /// Emit a warning.
  InflightDiag emitWarning(Location loc, const Twine &message = {});
  InflightDiag emitWarning(llvm::SMLoc loc, const Twine &message = {});

  /// Inflate a lightweight SMLoc into an MLIR Location object for addition
  /// into the IR.
  Location translateLocation(llvm::SMLoc loc) const;

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
    memcpy(result, elements.data(), dataSize);
    return ArrayRef<T>(result, elements.size());
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
  Operation *setResolvedDeclSymbol(Operation *declOp);

  //===--------------------------------------------------------------------===//
  // Name Lookup

  /// Return true if the specified type has a declared member with the specified
  /// name.
  bool typeHasMember(ASTType type, StringRef name, llvm::SMLoc loc);
  bool typeHasMember(ASTDecl &type, StringRef name, llvm::SMLoc loc);

  /// Perform a name lookup in the current scope and return the named
  /// declaration as a LookupResult.  If `searchParentScopes` is true, parent
  /// scopes are searched as well, as in unqualified name lookup.
  LookupResult lookupAndResolveDecl(StringRef name, llvm::SMLoc loc,
                                    ASTDecl &scope, bool searchParentScopes);

  /// Perform a name lookup for a member in the specified type.
  LookupResult lookupAndResolveDecl(StringRef name, llvm::SMLoc loc,
                                    ASTType scope, bool searchParentScopes);

  /// Lookup the specified name, and check that it is a non-parameterized type.
  /// This emits a diagnostic on error and returns null, or returns the type on
  /// success.
  ASTType lookupNonparameterizedNamedType(StringRef name, llvm::SMLoc loc,
                                          ASTDecl &context);

  /// Lookup the `object` type in the specified context and return it if found,
  /// otherwise emit an error and return null.
  ASTType lookupObjectType(llvm::SMLoc loc, ASTDecl &context);

  /// Lookup the `AnyType` trait in the specified context and return it if
  /// found, otherwise emit an error and return null.
  ASTDecl *lookupAnyTypeTrait(llvm::SMLoc loc, ASTDecl *context);

  /// Lookup the `Copyable` trait in the specified context and return it if
  /// found, otherwise emit an error and return null.
  ASTDecl *lookupCopyableTrait(llvm::SMLoc loc, ASTDecl *context);

  /// Lookup the `Movable` trait in the specified context and return it if
  /// found, otherwise emit an error and return null.
  ASTDecl *lookupMovableTrait(llvm::SMLoc loc, ASTDecl *context);

  //===--------------------------------------------------------------------===//
  // Module Resolution

  /// Import the specified module or package, returning the decl. Always returns
  /// a valid decl, even if a corresponding module or package could not be
  /// found.
  ASTDecl &importModule(StringRef name, PackageOp currentPackage,
                        llvm::SMLoc loc);

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

  /// Returns true if the given local path corresponds with a module or a
  /// package.
  static bool isModuleOrPackagePath(const std::filesystem::path &path);

  /// Cache the state of any modules that we parsed.
  void cacheParsedModules();

  /// Resolve a declaration that originated from bytecode to the given
  /// resolvedness.
  LogicalResult resolveDeclFromBytecode(ASTDecl &decl,
                                        DeclResolvedness resolvedness);

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

  //===--------------------------------------------------------------------===//
  // Debug Info

  /// Builds the debug info for a block argument if needed.
  void buildArgDebugInfo(OpBuilder &builder, BlockArgument arg, StringRef name);
  /// Generate a debug subprogram for this function and set it in its location.
  void setLocationDebugScope(DebugInfo::DIBuilder::ScopeGuard &diScopeGuard,
                             LIT::FuncOp funcOp);
  /// Get the debug source name for a function.
  DebugInfo::SourceNameAttr getSourceName(LIT::FuncOp func);

  //===--------------------------------------------------------------------===//
  // Listener Interface

  /// Notify the parser listener, if present, of a parsed alias decl.
  void notifyListenerOnAliasDecl(ASTDecl &decl, SMLoc identifierLoc);

  /// Notify the parser listener, if present, of a parsed alias decl.
  void notifyListenerOnArgumentDecl(ASTDecl &decl, SMLoc identifierLoc);

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
  /// `from Module [as Alias]` has been resolved by the parser. The provided
  /// location and spelling correspond to the module name and not to its
  /// optional alias.
  void notifyListenerOnModuleImport(ASTDecl &decl, StringRef spelling,
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
                            const CallOperands &callOperands);

  /// Notify the listener, if present, that parameter operands are being bound
  /// to one of the given decls.
  void notifyListenerOnParameterBinding(ArrayRef<ASTDecl *> decls,
                                        llvm::SMLoc rsquareLoc,
                                        ArrayRef<Operand> operands);

  //===--------------------------------------------------------------------===//
  // Builtin Module

  /// Get a builtin type, or emit an error and return TypeCheckErrorType if
  /// invalid. These never return null.
  ASTType getBuiltinBoolType(ASTDecl &context, llvm::SMLoc loc);
  ASTType getBuiltinTupleType(ASTDecl &context, llvm::SMLoc loc);
  ASTType getBuiltinErrorType(ASTDecl &context, llvm::SMLoc loc);
  ASTType getBuiltinIntType(ASTDecl &context, llvm::SMLoc loc);
  ASTType getBuiltinIntLiteralType(ASTDecl &context, llvm::SMLoc loc);
  ASTType getBuiltinStringLiteralType(ASTDecl &context, llvm::SMLoc loc);
  ASTType getBuiltinSliceType(ASTDecl &context, llvm::SMLoc loc);
  ASTType getBuiltinListLiteralType(ASTDecl &context, llvm::SMLoc loc);
  ASTType getBuiltinVariadicListType(ASTDecl &context, llvm::SMLoc loc,
                                     bool inMem);
  ASTType getBuiltinDoubleType(ASTDecl &context, llvm::SMLoc loc);
  ASTType getBuiltinCoroutineType(ASTDecl &context, llvm::SMLoc loc);
  ASTType getBuiltinRaisingCoroutineType(ASTDecl &context, llvm::SMLoc loc);

  /// This returns an instance of Tuple[...] with the specified element types
  /// installed.
  ASTType getBuiltinTupleInstantion(ASTDecl &context, llvm::SMLoc loc,
                                    ArrayRef<Type> elements);

  struct Impl;
  Impl &getImpl() const { return *impl; }

  /// Emitters invoke this method to get a closure declaration.
  StructDeclOp getOrCreateClosureWrapper(SMLoc loc, SignatureType sig,
                                         ASTDecl *moduleDecl);

  /// Given a scope that refers to a nested function, return the set of captured
  /// values in the form of a range: the begin and end iterators of the capture
  /// list.
  const llvm::MapVector<ASTDecl *, Capture> &
  getCaptureRangeInScope(ASTDecl &scope);

  /// Given a nested function, a capture value, and the corresponding capture
  /// ASTDecl, store the capture associated with the nested function.
  void addCaptureToScope(ASTDecl &scope, ASTDecl *captureDecl, Capture capture);

private:
  /// The internal state of an imported module or package.
  struct ModuleState;

  /// Add magic things to the builtins decl when parsing starts.
  void addBuiltinTypes(ASTDecl &builtinsDecl);

  /// Import the specified module or package, returning the module state.
  /// Always returns a valid module state, even if the module could not be
  /// found.
  ModuleState &importModuleState(StringRef name, ASTDecl *context,
                                 llvm::SMLoc loc);

  /// Import the specified module or package nested within the given parent
  /// decl, returning the module state. Always returns a valid module state,
  /// even if the module could not be found.
  ModuleState &importSubModuleState(StringRef name, ASTDecl *parentDecl,
                                    llvm::SMLoc loc);

  /// Import the specified module or package, which contains `.` indexing,
  /// returning the module state. Always returns a valid module state, even if
  /// the module could not be found.
  ModuleState &importRelativeModuleState(StringRef name, ASTDecl *parentDecl,
                                         llvm::SMLoc loc);

  /// Create a new module state with the given name, location, and body.
  ModuleState &createModuleState(StringAttr declName, StringAttr mangledName,
                                 const llvm::MemoryBuffer *moduleBuffer,
                                 ModuleState &parentState, FileLineColLoc loc,
                                 bool enableCaching);

  /// Create a new module state for a package with the given name, location,
  /// and body.
  ModuleState &createPackageState(StringAttr declName, StringAttr mangledName,
                                  StringRef packagePath,
                                  ModuleState &parentState, FileLineColLoc loc);

  /// Create a new module state for a binary package with the given name.
  ModuleState &createBinaryPackageState(SMLoc loc, StringAttr declName,
                                        StringAttr mangledName,
                                        StringRef packagePath,
                                        ModuleState &parentState);

  /// Create an error module state with the given mangled name, and emit the
  /// given error message.
  ModuleState &createErrorModuleState(SMLoc loc, StringAttr mangledName,
                                      ASTDecl &errorContext,
                                      const Twine &errorMsg);

  /// Resolve the dependencies of the given module.
  void resolveModuleDependencies(ModuleState &module, ASTDecl *parentDecl,
                                 StringRef moduleBuffer);

  /// Attempt to get a cached version of the given modules. If loading from
  /// the cache fails, the modules will be processed as normal.
  void loadModulesFromCache(MutableArrayRef<ModuleState *> moduleStates);

  /// Implicitly import the builtin modules into the given module decl.
  void importBuiltinModules(ASTDecl &moduleDecl);

  /// This is used for memory that lives as long as the global parser does.
  llvm::BumpPtrAllocator persistentAllocator;

  /// A flag indicating if we're currently parsing the standard library.
  bool parsingStandardLibrary = false;

  /// If true, auto-import the builtin package.
  bool useBuiltinModule = true;

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
  DeclResolver &getDeclResolver() const { return *shared.declResolver; }

  mlir::Location translateLocation(SMLoc loc) {
    return shared.translateLocation(loc);
  }

  /// Emit an error.
  InflightDiag emitError(Location loc, const Twine &message = {}) {
    return shared.emitError(loc, message);
  }
  InflightDiag emitError(llvm::SMLoc loc, const Twine &message = {}) {
    return shared.emitError(loc, message);
  }

  /// Emit a warning.
  InflightDiag emitWarning(Location loc, const Twine &message = {}) {
    return shared.emitWarning(loc, message);
  }
  InflightDiag emitWarning(llvm::SMLoc loc, const Twine &message = {}) {
    return shared.emitWarning(loc, message);
  }
};

/// This is the result of lookupDecl.
class LookupResult {
  enum Kind {
    kSuccess,   //<- Lookup succeeded and result is non-null.
    kFailure,   //<- Lookup failed to find something of this name.
    kErroneous, //<- Lookup found an error, but it is already diagnosed.
  } kind;

  /// This is non-empty when the Kind is kSuccess.  This points to the symbol
  /// entry in an ASTDecl, so the pointer is stable.
  ArrayRef<ASTDecl *> decls;
  LookupResult(Kind kind, ArrayRef<ASTDecl *> decls)
      : kind(kind), decls(decls) {}

public:
  static LookupResult getSuccess(ArrayRef<ASTDecl *> decls) {
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
} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_SHAREDSTATE_H
