//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the main entrypoints for the Mojo parser.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/ASTDeclRef.h"
#include "ASTDecl.h"
#include "ASTType.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/ASTDeclView.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// MojoASTDeclRef
//===----------------------------------------------------------------------===//

/// Unwrap a raw ASTDecl pointer.
static ASTDecl *unwrapMojoASTDecl(void *declImpl) {
  assert(declImpl && "expected valid MojoASTDeclRef impl");
  return reinterpret_cast<ASTDecl *>(declImpl);
}

Operation *MojoASTDeclRef::getIfOperation() const {
  return unwrapMojoASTDecl(impl)->getIfOperation();
}

MojoASTTypeRef MojoASTDeclRef::getType() const {
  return TypeSwitch<ASTDecl &, MojoASTTypeRef>(*unwrapMojoASTDecl(impl))
      .Case<VarLetDeclOp, LetRegDeclOp>(
          [&](auto op) { return MojoASTTypeRef(op.getType()); })
      .Default({});
}

std::optional<StringAttr> MojoASTDeclRef::getMangledName() const {
  return TypeSwitch<ASTDecl &, std::optional<StringAttr>>(
             *unwrapMojoASTDecl(impl))
      .Case<FileModuleOp, FuncOp, LetRegDeclOp, StructDeclOp, StructFieldOp,
            VarLetDeclOp>([&](auto op) { return op.getNameAttr(); })
      .Case<AliasDeclOp>([&](AliasDeclOp op) { return op.getName(); })
      .Default({});
}

std::optional<StringRef> MojoASTDeclRef::getName() const {

  return TypeSwitch<ASTDecl &, std::optional<StringRef>>(
             *unwrapMojoASTDecl(impl))
      .Case<LetRegDeclOp, StructDeclOp, StructFieldOp, VarLetDeclOp>(
          [](auto op) { return op.getName(); })
      .Case<FuncOp>([](FuncOp op) {
        // We remove the parameter section and argument section from the symbol
        // name to keep only the identifier.
        StringRef mangled = op.getName();
        return mangled.substr(0, mangled.find_first_of("(["));
      })
      .Case<FileModuleOp>([](FileModuleOp op) {
        // We remove the trailing $.
        StringRef fullName = op.getName();
        fullName.consume_front("$");
        return fullName;
      })
      .Case<AliasDeclOp>([](AliasDeclOp op) {
        return demangleParameterName(op.getParamDecl().getName());
      })
      .Default({});
}

llvm::SMLoc MojoASTDeclRef::getLoc() const {
  return unwrapMojoASTDecl(impl)->getLoc();
}

std::unique_ptr<DeclView> MojoASTDeclRef::getView() const {
  return TypeSwitch<ASTDecl &, std::unique_ptr<DeclView>>(
             *unwrapMojoASTDecl(impl))
      .Case<AliasDeclOp>([&](auto op) {
        return std::unique_ptr<AliasDeclView>(new AliasDeclView(*this));
      })
      .Case<FileModuleOp>([&](auto op) {
        return std::unique_ptr<ModuleDeclView>(new ModuleDeclView(*this));
      })
      .Case<FuncOp>([&](auto op) {
        return std::unique_ptr<FunctionDeclView>(new FunctionDeclView(*this));
      })
      .Case<StructDeclOp>([&](auto op) {
        return std::unique_ptr<StructDeclView>(new StructDeclView(*this));
      })
      .Case<StructFieldOp>([&](auto op) {
        return std::unique_ptr<StructFieldDeclView>(
            new StructFieldDeclView(*this));
      })
      .Case<LetRegDeclOp, VarLetDeclOp>([&](auto op) {
        return std::unique_ptr<VariableDeclView>(new VariableDeclView(*this));
      })
      .Default({});
}

//===----------------------------------------------------------------------===//
// MojoASTTypeRef
//===----------------------------------------------------------------------===//

/// Unwrap a raw ASTDecl pointer.
static ASTType unwrapMojoASTType(void *declImpl) {
  assert(declImpl && "expected valid MojoASTDeclRef impl");
  return ASTType(Type::getFromOpaquePointer(declImpl));
}

MojoASTTypeRef::MojoASTTypeRef(const mlir::Type &type)
    : MojoASTTypeRef(const_cast<void *>(type.getAsOpaquePointer())) {}

MojoASTDeclRef MojoASTTypeRef::getDecl(SharedState &sharedState) {
  return MojoASTDeclRef(unwrapMojoASTType(impl).getDecl(sharedState));
}

std::string MojoASTTypeRef::getAsString() const {
  return unwrapMojoASTType(impl).getAsString(/*forDiag=*/true);
}

MojoASTTypeRef MojoASTTypeRef::getPointerElementType() const {
  return unwrapMojoASTType(impl).getPointerElementType().mlirType;
}

Type MojoASTTypeRef::getMLIRType() const {
  return Type::getFromOpaquePointer(impl);
}
