//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the base class for Lit file parsers that is common between
// expression and statement parsing.
//
//===----------------------------------------------------------------------===//

#ifndef LIT_SHARED_STATE_H
#define LIT_SHARED_STATE_H

#include "LitDiags.h"

#include "mlir/IR/BuiltinOps.h"

namespace M::DebugInfo {
class DIBuilder;
} // namespace M::DebugInfo

namespace M::KGEN {
class CompilationOptions;
class ParamDeclAttr;
}

namespace M::KGEN::LIT {
class DeclResolver;
class ASTDecl;
class ASTType;
class MValue;
class LookupResult;

/// Given a number, return one string if the number is 1, otherwise return the
/// other.  This is typically used to generate an "s" suffix, but can also be
/// used for things like `plural(count, "was", "were")`.
inline const char *plural(size_t value, const char *one = "",
                          const char *other = "s") {
  return value == 1 ? one : other;
}

/// This is state shared across multiple different instances of LitParser
/// which are always shared across them.
class LitSharedState {
public:
  LitSharedState(llvm::SourceMgr &sourceMgr, MLIRContext *context,
                 const CompilationOptions &options, bool useMLIRDiagnostics);
  ~LitSharedState();

  LitDiags diags; // Contains SourceMgr and MLIRContext pointers.
  const CompilationOptions &options;

  std::unique_ptr<DeclResolver> declResolver;
  std::unique_ptr<DebugInfo::DIBuilder> diBuilder;

  const mlir::StringAttr bufferNameIdentifier;

  llvm::SourceMgr &getSourceMgr() const { return diags.sourceMgr; }
  MLIRContext *getContext() const { return diags.context; }

  /// Initialize the shared state for the given top-level decl.
  void initialize(ASTDecl &topLevelDecl);

  /// This is the AST type that corresponds to TypeCheckErrorType.
  ASTType getTypeCheckErrorType() const;

  /// This is the decl for the builtin 'kgen.none' type.
  ASTType getNoneType() const;

  /// Emit an error.
  LitDiagnostic emitError(Location loc, const Twine &message = {});
  LitDiagnostic emitError(llvm::SMLoc loc, const Twine &message = {});

  /// Emit a warning.
  LitDiagnostic emitWarning(Location loc, const Twine &message = {});
  LitDiagnostic emitWarning(llvm::SMLoc loc, const Twine &message = {});

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

  /// Lookup the `Error` type in the specified context and return it if found,
  /// otherwise emit an error and return null.
  ASTType lookupErrorType(llvm::SMLoc loc, ASTDecl &context);

  //===--------------------------------------------------------------------===//
  // Module Resolution

  /// Import the specified module, returning the module decl. Always returns a
  /// valid decl, even if the module could not be found.
  ASTDecl &importModule(StringRef moduleName, llvm::SMLoc loc);

  /// Create a new module with the given name, location, and body.
  ASTDecl &createModule(StringRef moduleName,
                        const llvm::MemoryBuffer *moduleBuffer,
                        FileLineColLoc loc);

  /// Get the list of files included while processing all modules.
  ArrayRef<std::string> getIncludedFiles() const;

private:
  /// Add magic things to the builtins decl when parsing starts.
  void addBuiltinTypes(ASTDecl &builtinsDecl);

  /// This is used for memory that lives as long as the global parser does.
  llvm::BumpPtrAllocator persistentAllocator;

  class Impl;
  std::unique_ptr<Impl> impl;
};

/// This class is intended to be used as a convenience base class for subsystems
/// that want to have access to various LitSharedState functionality in a
/// convenient way.
class LitSharedStateUser {
public:
  LitSharedStateUser(LitSharedState &shared) : shared(shared) {}

  /// This reference provides direct access to LitSharedState for anything
  /// fancy.
  LitSharedState &shared;

  // Convenience forwarding functions used pervasively through the frontend.

  MLIRContext *getContext() const { return shared.getContext(); }
  llvm::SourceMgr &getSourceMgr() const { return shared.getSourceMgr(); }
  DeclResolver &getDeclResolver() const { return *shared.declResolver; }

  mlir::Location translateLocation(SMLoc loc) {
    return shared.translateLocation(loc);
  }

  /// Emit an error.
  LitDiagnostic emitError(Location loc, const Twine &message = {}) {
    return shared.emitError(loc, message);
  }
  LitDiagnostic emitError(llvm::SMLoc loc, const Twine &message = {}) {
    return shared.emitError(loc, message);
  }

  /// Emit a warning.
  LitDiagnostic emitWarning(Location loc, const Twine &message = {}) {
    return shared.emitWarning(loc, message);
  }
  LitDiagnostic emitWarning(llvm::SMLoc loc, const Twine &message = {}) {
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
  bool isFailure() const { return kind == kFailure; }
  bool isErroneous() const { return kind == kErroneous; }
};

} // namespace M::KGEN::LIT

#endif // LIT_SHARED_STATE_H
