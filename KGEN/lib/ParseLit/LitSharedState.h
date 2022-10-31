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

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace llvm {
class SourceMgr;
}

namespace M::KGEN {
class ParamBindArrayAttr;
}

namespace M::KGEN::LIT {
class DeclResolver;
class ASTDecl;

/// This is state shared across multiple different instances of LitParser
/// which are always shared across them.
class LitSharedState {
public:
  LitSharedState(llvm::SourceMgr &sourceMgr, MLIRContext *context);
  ~LitSharedState();

  llvm::SourceMgr &sourceMgr;
  MLIRContext *const context;
  std::unique_ptr<DeclResolver> declResolver;

  const mlir::StringAttr bufferNameIdentifier;

  /// This is the AST type that corresponds to TypeCheckErrorType.
  ASTDecl *typeCheckErrorTypeDecl = nullptr;

  /// This is the decl for the builtin 'index' type.
  ASTDecl *indexDecl = nullptr;
  /// This is the decl for the builtin 'kgen.none' type.
  ASTDecl *noneDecl = nullptr;

  /// This is the decl for the builtin POP::PointerType type.
  ASTDecl *pointerDecl = nullptr;
  /// This is the decl for the builtin lit.object type.
  ASTDecl *objectDecl = nullptr;

  /// This is set to true if an error occurred at any point processing the file.
  bool errorOccurred = false;

  /// This is used for memory that lives as long as the global parser does.
  llvm::BumpPtrAllocator persistentAllocator;

  /// Inflate a lightweight SMLoc into an MLIR Location object for addition
  /// into the IR.
  Location translateLocation(llvm::SMLoc loc);
};

/// This type represents an AST type for a value or declaration, which is a
/// pointer to the DeclAST that defines it as well as any bound parameters.
///
/// The bound parameters may themselves refer to other parameters in the
/// enclosing scope, e.g. in the case of `SomeType[size*42]`.
///
class ASTType {
public:
  ASTDecl *getDecl() const { return decl; }
  ParamBindArrayAttr getParamValues() const;

  ASTType() : decl(nullptr) {}
  ASTType(ASTDecl *decl);
  ASTType(ASTDecl *decl, ParamBindArrayAttr attrs);

  bool operator!() const { return decl == nullptr; }
  operator bool() const { return decl != nullptr; }

private:
  ASTDecl *decl;
  Attribute paramValues; // This is always a ParamBindArrayAttr.
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
  signatureResolved,

  /// This declaration has been fully type checked, including its body.  Any
  /// declarations within the body may not be fully resolved though.
  fullyResolved
};

/// This keeps track of specific kinds of "magic" declarations that do not have
/// a standard AST representation.
enum class MagicDeclKind {
  // This is not a magic declaration, process it as normal.
  kNormal,
  // This type is produced when an error is detected to simplify clients.
  kTypeCheckErrorType,
  // This is the __builtin.mlirtype.builtin.index type.
  kIndexType,
  // This is the __builtin.mlirtype.lit.none type.
  kNoneType,
  // This is a POP::PointerType type.
  kPointerType,
  // This is the lit.object type.
  kObjectType,
};

} // namespace M::KGEN::LIT

#endif // LIT_SHARED_STATE_H
