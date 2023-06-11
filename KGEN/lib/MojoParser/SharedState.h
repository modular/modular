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

#ifndef SHARED_STATE_H
#define SHARED_STATE_H

#include "Diags.h"

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
class DominanceInfo;
} // namespace mlir

namespace M {
struct MojoParserConfig;
} // namespace M

namespace M::DebugInfo {
class DIBuilder;
} // namespace M::DebugInfo

namespace M::KGEN {
class CompilationOptions;
class ParamDeclAttr;
} // namespace M::KGEN

namespace M::LLCL {
class Runtime;
} // namespace M::LLCL

namespace M::KGEN::LIT {
class DeclResolver;
class ASTDecl;
class ASTType;
class LookupResult;
class NoneAttr;

/// Given a number, return one string if the number is 1, otherwise return the
/// other.  This is typically used to generate an "s" suffix, but can also be
/// used for things like `plural(count, "was", "were")`.
inline const char *plural(size_t value, const char *one = "",
                          const char *other = "s") {
  return value == 1 ? one : other;
}

/// This is state shared across multiple different instances of Parser
/// which are always shared across them.
class SharedState {
public:
  SharedState(llvm::SourceMgr &sourceMgr, MojoParserConfig &config,
              bool enableCaching = true);
  ~SharedState();

  Diags diags; // Contains SourceMgr and MLIRContext pointers.
  const CompilationOptions &options;

  std::unique_ptr<DeclResolver> declResolver;
  std::unique_ptr<DebugInfo::DIBuilder> diBuilder;
  LLCL::Runtime &runtime;

  const mlir::StringAttr bufferNameIdentifier;

  llvm::SourceMgr &getSourceMgr() const { return diags.sourceMgr; }
  MLIRContext *getContext() const { return diags.context; }

  /// Returns if we should validate doc strings.
  bool shouldValidateDocStrings() const;

  /// Initialize the shared state for the given top-level decl.
  void initialize(ASTDecl &topLevelDecl);

  /// This is the AST type that corresponds to TypeCheckErrorType.
  ASTType getTypeCheckErrorType() const;

  /// This is the decl for the builtin 'kgen.none' type.
  ASTType getNoneType() const;

  /// This returns a NoneAttr.
  NoneAttr getNoneAttr() const;

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

  /// Set the symbol for the specified declaration (known to be an operation)
  /// into the MLIR symbol table for its container.  If the symbol is already
  /// declared in the same MLIR scope, then return the conflicting operation.
  Operation *setResolvedDeclSymbol(Operation *declOp);

  /// Get the shared operation dominance analysis.
  mlir::DominanceInfo &getDomInfo();

  //===--------------------------------------------------------------------===//
  // Name Lookup

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

  //===--------------------------------------------------------------------===//
  // Module Resolution

  /// Import the specified module, returning the module decl. Always returns a
  /// valid decl, even if the module could not be found.
  ASTDecl &importModule(StringRef moduleName, llvm::SMLoc loc);

  /// Create a new module with the given name, location, and body.
  ASTDecl &createModule(StringRef moduleName,
                        const llvm::MemoryBuffer *moduleBuffer,
                        FileLineColLoc loc);

  /// Cache the state of any modules that we parsed.
  void cacheParsedModules();

  /// Get the list of files included while processing all modules.
  ArrayRef<std::string> getIncludedFiles() const;

  //===--------------------------------------------------------------------===//
  // Builtin Module

  /// The name of the builtin Bool module.
  static constexpr StringLiteral kBuiltinBoolModuleName = "Bool";
  /// The name of the builtin Tuple module.
  static constexpr StringLiteral kBuiltinTupleModuleName = "Tuple";
  /// The name of the builtin Error module.
  static constexpr StringLiteral kBuiltinErrorModuleName = "Error";
  /// The name of the builtin Int module.
  static constexpr StringLiteral kBuiltinIntModuleName = "Int";
  /// The name of the builtin aliases module.
  static constexpr StringLiteral kBuiltinTypeAliasesModuleName = "TypeAliases";
  /// The name of the builtin string module.
  static constexpr StringLiteral kBuiltinStringModuleName = "StringLiteral";
  /// The name of the builtin string ref module.
  static constexpr StringLiteral kBuiltinStringRefModuleName = "StringRef";
  /// The name of the builtin slice module.
  static constexpr StringLiteral kBuiltinSliceModuleName = "BuiltinSlice";
  /// The name of the builtin list module.
  static constexpr StringLiteral kBuiltinListModuleName = "BuiltinList";
  /// The name of the builtin FloatLiteral module.
  static constexpr StringLiteral kBuiltinDoubleModuleName = "FloatLiteral";

  /// All the builtin modules.
  /// FIXME: We need a better way to include all the builtin modules. Perhaps
  /// a proper Prolog module, but wildcard imports don't play nice togther.
  static constexpr StringLiteral kBuiltinModuleNames[] = {
      kBuiltinBoolModuleName,        kBuiltinTupleModuleName,
      kBuiltinErrorModuleName,       kBuiltinIntModuleName,
      kBuiltinTypeAliasesModuleName, kBuiltinStringModuleName,
      kBuiltinSliceModuleName,       kBuiltinListModuleName,
      kBuiltinDoubleModuleName,      kBuiltinStringRefModuleName};

  /// Get a builtin type, or emit an error and return null if invalid.
  ASTType getBuiltinBoolType(llvm::SMLoc loc);
  ASTType getBuiltinTupleType(llvm::SMLoc loc);
  ASTType getBuiltinErrorType(llvm::SMLoc loc);
  ASTType getBuiltinIntType(llvm::SMLoc loc);
  ASTType getBuiltinStringLiteralType(llvm::SMLoc loc);
  ASTType getBuiltinSliceType(llvm::SMLoc loc);
  ASTType getBuiltinListLiteralType(llvm::SMLoc loc);
  ASTType getBuiltinDoubleType(llvm::SMLoc loc);

  /// This returns an instance of Tuple[...] with the specified element types
  /// installed.
  ASTType getBuiltinTupleInstantion(llvm::SMLoc loc, ArrayRef<Type> elements);

  struct Impl;
  Impl &getImpl() const { return *impl; }

private:
  /// The internal state of an imported module.
  struct ModuleState;

  /// Add magic things to the builtins decl when parsing starts.
  void addBuiltinTypes(ASTDecl &builtinsDecl);

  /// Import the specified module, returning the module state. Always returns a
  /// valid module state, even if the module could not be found.
  ModuleState &importModuleState(StringRef moduleName, llvm::SMLoc loc);

  /// Create a new module state with the given name, location, and body.
  ModuleState &createModuleState(StringRef moduleName,
                                 const llvm::MemoryBuffer *moduleBuffer,
                                 FileLineColLoc loc);

  /// Resolve the dependencies of the given module.
  void resolveModuleDependencies(ModuleState &module, StringRef moduleBuffer);

  /// Attempt to get a cached version of the given modules. If loading from the
  /// cache fails, the modules will be processed as normal.
  void loadModulesFromCache(MutableArrayRef<ModuleState *> moduleStates);

  /// This is used for memory that lives as long as the global parser does.
  llvm::BumpPtrAllocator persistentAllocator;

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
  static LookupResult getFailure() { return {kFailure, {}}; }
  static LookupResult getErroneous() { return {kErroneous, {}}; }

  ArrayRef<ASTDecl *> getIfSuccess() const { return decls; }
  bool isSuccess() const { return kind == kSuccess; }
  bool isFailure() const { return kind == kFailure; }
  bool isErroneous() const { return kind == kErroneous; }
};

} // namespace M::KGEN::LIT

#endif // SHARED_STATE_H
