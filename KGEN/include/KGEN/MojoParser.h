//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_H
#define KGEN_MOJOPARSER_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/Support/MemoryBuffer.h"
#include <filesystem>
#include <string>

namespace llvm {
class SMDiagnostic;
class SourceMgr;
} // namespace llvm
namespace mlir {
class TimingScope;
class Type;
} // namespace mlir

namespace M {
namespace KGEN {
class CompilationOptions;
namespace LIT {
class PackageOp;
class SharedState;
} // namespace LIT
} // namespace KGEN
namespace LLCL {
class Runtime;
} // namespace LLCL

class DeclView;
class MojoParserListener;
class MojoASTDeclRef;
class MojoASTTypeRef;

/// This class provides the various configurations used to parse a .mojo file.
struct MojoParserConfig {
  MojoParserConfig(MLIRContext *context, LLCL::Runtime &runtime,
                   const KGEN::CompilationOptions &options)
      : context(context), runtime(runtime), options(options) {}

  /// This enum defines different levels of caching acceptible for the parser.
  enum CachingLevel {
    /// No caching is allowed.
    kCacheNone,

    /// Caching is allowed just for imported modules, main/input/root modules
    /// are not cached.
    kCacheImports,

    /// Caching is allowed for all modules.
    kCacheAll,
  };

  /// The MLIR context to use when parsing the file.
  MLIRContext *context;

  /// The runtime to use when parsing the file.
  LLCL::Runtime &runtime;

  /// The compilation options to use when parsing the file.
  const KGEN::CompilationOptions &options;

  /// When true, this prints diagnostics through MLIR (so MLIR features like
  /// -verify-diagnostics may be used). When false, this prints them through
  /// SourceMgr to get ranges and fixit hints.
  bool useMLIRDiagnostics = false;

  /// If true, this will process and validate the doc strings in the file.
  bool validateDocStrings = false;

  /// If true, use !lit.ref representation for full lifetimes support in Mojo.
  bool experimentalLifetimes = false;

  /// The level of module caching enabled in the parser.
  CachingLevel moduleCachingLevel = kCacheAll;

  /// An optional listener that is used to inspect certain events of the parser.
  /// For simplicity it is a single item, but it could evolve into a list of
  /// listeners.
  MojoParserListener *parserListener = nullptr;

  /// Maximum number of notes to print per compiler error or warning.
  int maxNotesPerDiagnostic = 10;
};

//===----------------------------------------------------------------------===//
// MojoParserListener
//===----------------------------------------------------------------------===//

/// This class provides an interface for other language tools like LSP servers
/// to inspect specific events in the parser.
class MojoParserListener {
public:
  virtual ~MojoParserListener() = default;

  /// Notify the listener that a new `alias` declaration has been resolved by
  /// the parser.
  virtual void onAliasDecl(MojoASTDeclRef declRef, llvm::SMLoc identifierLoc);

  /// Notify the listener that a new `function argument` declaration has been
  /// resolved by the parser.
  ///
  /// It is guaranteed that this listener has been notified of its parent
  /// function decl before this call.
  virtual void onArgumentDecl(MojoASTDeclRef declRef,
                              llvm::SMLoc identifierLoc);

  /// Notify the listener that a new `def` or `fn` function declaration has been
  /// resolved by the parser. This includes struct methods and closures.
  virtual void onFunctionDecl(MojoASTDeclRef declRef,
                              llvm::SMLoc identifierLoc);

  /// Notify the listener that an import is currently being resolved.
  virtual void onImport(llvm::SMLoc importLoc);

  /// Notify the listener that an import of a module within the given package is
  /// currently being resolved.
  virtual void onImport(MojoASTDeclRef packageDecl, llvm::SMLoc importLoc);

  /// Notify the listener that a member within the given decl is being looked
  /// up.
  virtual void onMemberLookup(MojoASTDeclRef decl, llvm::SMLoc lookupLoc);

  /// Notify the listener that a new `module` decl has been created by the
  /// parser.
  virtual void onModuleDecl(MojoASTDeclRef declRef, llvm::SMLoc identifierLoc);

  /// Notify the listener that a new import of the form `from Module [as Alias]`
  /// has been resolved by the parser. The provided location and spelling
  /// correspond to the module name and not to its optional alias.
  virtual void onModuleImport(MojoASTDeclRef declRef, StringRef spelling,
                              llvm::SMLoc loc);

  /// Notify the listener that a new `struct` declaration has been resolved by
  /// the parser.
  virtual void onStructDecl(MojoASTDeclRef declRef, llvm::SMLoc identifierLoc);

  /// Notify the listener that a new `struct field` declaration has been
  /// resolved by the parser.
  virtual void onStructFieldDecl(MojoASTDeclRef declRef,
                                 llvm::SMLoc identifierLoc);

  /// Notify the listener that a new `let` or `var` declaration has been
  /// resolved by the parser.
  virtual void onVariableDecl(MojoASTDeclRef declRef,
                              llvm::SMLoc identifierLoc);

  /// Notify the listener that a new reference has been resolved by the parser,
  /// i.e. its declaration is known.
  virtual void onRef(MojoASTDeclRef declRef, StringRef spelling,
                     llvm::SMLoc loc);
};

//===----------------------------------------------------------------------===//
// MojoParserREPLListener
//===----------------------------------------------------------------------===//

/// This class provides a listener for interacting with the parser for REPL
/// like expressions. It contains various hooks to allow for customizing the
/// behavior of the parser.
class MojoParserREPLListener {
public:
  virtual ~MojoParserREPLListener() = default;

  //===------------------------------------------------------------------===//
  // Notifications

  /// The following methods are called by the parser to notify the listener of
  /// various events during parsing. These can be useful for logging,
  /// debugging, etc.

  /// Notify the listener that the parser has wrapped the input expression
  /// into code capable of being parsed. `wrappedExpr` is the fully wrapped
  /// expression.
  virtual void notifyWrappedExpr(StringRef wrappedExpr) = 0;

  /// Notify the listener that the parser applied fixes the original input
  /// expression.
  virtual void notifyFixedExpr(StringRef fixedExpr) = 0;

  /// Notify the listener that the given set of diagnostics were emitted while
  /// parsing the wrapped expression.
  virtual void notifyDiagnostics(ArrayRef<llvm::SMDiagnostic> diagnostics) = 0;

  //===------------------------------------------------------------------===//
  // Queries

  /// The following methods are called by the parser to query the listener for
  /// various information. These can be useful for customizing the behavior of
  /// the parser.

  /// Query the listener to see if a variable with the given name and type
  /// should be persisted. If this returns true, the variable will be appended
  /// to the list of fields within the struct passed to the expression
  /// function.
  virtual bool shouldPersistVariable(StringRef name, Type type) = 0;
};

//===----------------------------------------------------------------------===//
// MojoParserContext
//===----------------------------------------------------------------------===//

/// This class provides a context for parsing and interacting with Mojo
/// modules.
class MojoParserContext {
public:
  MojoParserContext(llvm::SourceMgr &sourceMgr, MojoParserConfig &config);
  ~MojoParserContext();

  /// Return the current module being parsed.
  ModuleOp getModule();

  /// Return the source manager used by the parser.
  llvm::SourceMgr &getSourceMgr();

  /// Return the full list of directories considered for module lookup from
  /// the given file.
  std::vector<std::string> getModuleSearchDirectories(unsigned fileId);

  /// Return the compilation options used by the parser.
  const KGEN::CompilationOptions &getCompilationOptions();

  /// Parse a SourceMgr file given its id as a module.
  ///
  /// In the case of success, the decl corresponding to the module is returned.
  /// In the case of an error, a null decl is returned.
  MojoASTDeclRef parseFile(unsigned int fileId);

  //===--------------------------------------------------------------------===//
  // REPL

  /// The following methods provide functionality for interacting with the
  /// parser context from REPL like environments.

  /// The following methods allow for interacting with the parser for REPL
  /// like expressions, i.e., in environments like Jupyter notebooks. `exprId`
  /// is a unique identifier for the expression being parsed, and is used as the
  /// generated module name. `exprText` is the expression to parse.
  /// `replExprFnName` is the name of the function to use for wrapping the
  /// expression. `replVariables` is a list of pre-existing variables to make
  /// available to the expression function, these variables should be used as
  /// `Pointer[Pointer[]]` fields within a struct that is passed by reference to
  /// the expression function. For example, given the following expression:
  ///
  ///   print(a)
  ///
  /// Where `a` is a pre-existing repl variable with type `Int`, the
  /// expression wrapper will effectively emulate the following:
  ///
  ///   struct ReplContext:
  ///     var a: Pointer[Pointer[Int]]
  ///
  ///   fn replExprFn(context&: ReplContext):
  ///      print(context.a.load().load())
  ///
  /// In the case of success, the decl corresponding to the expr function is
  /// returned. In the case of an error, a null decl is returned.
  MojoASTDeclRef
  parseREPLExpresion(MojoParserREPLListener &listener, StringRef exprId,
                     StringRef exprText, StringRef replExprFnName,
                     ArrayRef<std::pair<StringRef, Type>> replVariables);

  /// Remove the previously parsed REPL expression. This allows for removing an
  /// erroneous expression when it is only detected as invalid after it has been
  /// parsed.
  void removeLastREPLExpression();

  /// Get the declaration that defined an AST type.
  MojoASTDeclRef getDecl(MojoASTTypeRef type);

protected:
  /// A struct representing the internal state of the parser.
  struct Impl;

  /// The internal state of the parser.
  std::unique_ptr<Impl> impl;
};

//===----------------------------------------------------------------------===//
// Driver Entry Points
//===----------------------------------------------------------------------===//

/// Returns true if the given file path corresponds to a mojo package.
bool isMojoSourcePackagePath(const std::filesystem::path &path);

/// Parse a single .mojo file and return the MLIR module for it.
///
/// If `includedFiles` is provided, it is set to the list of included files when
/// parsing imports.
OwningOpRef<ModuleOp>
importMojoFile(llvm::SourceMgr &sourceMgr, MojoParserConfig &config,
               mlir::TimingScope &ts,
               SmallVectorImpl<std::string> *includedFiles = nullptr);

/// Parse a single mojo package at the given path and return the full context
/// MLIR module, and the corresponding PackageOp for it.
///
/// If `includedFiles` is provided, it is set to the list of included files when
/// parsing imports.
std::pair<OwningOpRef<ModuleOp>, KGEN::LIT::PackageOp>
importMojoPackage(StringRef path, StringRef packageName,
                  llvm::SourceMgr &sourceMgr, MojoParserConfig &config,
                  mlir::TimingScope &ts,
                  SmallVectorImpl<std::string> *includedFiles = nullptr);
} // namespace M

#endif // KGEN_MOJOPARSER_H
