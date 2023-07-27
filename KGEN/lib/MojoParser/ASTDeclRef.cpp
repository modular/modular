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

/// If this decl corresponds to a not owned function argument, return its
/// corresponding BlockArgument. Otherwise, return null.
static BlockArgument getIfNotOwnedFunctionArgument(MojoASTDeclRef declRef) {
  return TypeSwitch<DeclIRValue, BlockArgument>(
             (*unwrapMojoASTDecl(declRef.getAsVoidPointer())).getIRValue())
      .Case<SBValue, SRValue, SLValue, MBValue>([&](auto val) -> BlockArgument {
        if (auto bbArg = dyn_cast<BlockArgument>(Value(val)))
          if (isa<LIT::FuncOp>(bbArg.getOwner()->getParentOp()))
            return bbArg;
        return {};
      })
      .Default({});
}

/// Return the defining Op from the IR encapsulated by this decl. It might be
/// null.
static Operation *getDefiningOpFromIR(MojoASTDeclRef declRef) {
  ASTDecl &astDecl = *unwrapMojoASTDecl(declRef.getAsVoidPointer());
  return TypeSwitch<DeclIRValue, Operation *>(astDecl.getIRValue())
      .Case<SBValue, SRValue, SLValue, MBValue>(
          [&](auto val) -> Operation * { return Value(val).getDefiningOp(); })
      .Default((Operation *)nullptr);
}

Operation *MojoASTDeclRef::getIfOperation() const {
  return unwrapMojoASTDecl(impl)->getIfOperation();
}

MojoASTTypeRef MojoASTDeclRef::getType() const {
  return TypeSwitch<ASTDecl &, MojoASTTypeRef>(*unwrapMojoASTDecl(impl))
      .Case<GlobalVarDeclOp, LetRegDeclOp, VarLetDeclOp>(
          [&](auto op) { return MojoASTTypeRef(op.getType()); })
      .Default({});
}

std::optional<StringAttr> MojoASTDeclRef::getMangledName() const {
  auto getFromOp = [](Operation *op) -> std::optional<StringAttr> {
    if (!op)
      return std::nullopt;
    return cast<ASTDeclInterface>(*op).getDeclName();
  };

  // We first try to get the name from the operation. Then we try to match the
  // decl with a function argument. Finally, as a last resort, we extract the
  // defining Op from the IR to fetch the name.
  ASTDecl &decl = *unwrapMojoASTDecl(impl);

  if (auto name = getFromOp(decl.getIfOperation()))
    return name;

  if (BlockArgument bbArg = getIfNotOwnedFunctionArgument(*this)) {
    auto func = cast<FuncOp>(*decl.getParentDecl());
    return func.getValueParamNames()[bbArg.getArgNumber()];
  }

  return getFromOp(getDefiningOpFromIR(*this));
}

std::optional<StringRef> MojoASTDeclRef::getName() const {
  auto getFromOp = [](Operation *op) -> std::optional<StringRef> {
    if (!op)
      return std::nullopt;
    return TypeSwitch<Operation &, std::optional<StringRef>>(*op)
        .Case<GlobalVarDeclOp, LetRegDeclOp, StructDeclOp, StructFieldOp,
              VarLetDeclOp>([](auto op) { return op.getName(); })
        .Case([](FuncOp op) {
          // FIXME(#18029): We should use MangledSymbol::demangle instead of
          // doing this.

          // We remove the parameter section and argument section from the
          // symbol name to keep only the identifier.
          StringRef mangled = op.getName();
          return mangled.substr(0, mangled.find_first_of("(["));
        })
        .Case([](FileModuleOp op) {
          // We remove the leading $.
          StringRef fullName = op.getName();
          fullName.consume_front("$");
          return fullName;
        })
        .Case([](AliasDeclOp op) {
          return demangleParameterName(op.getParamDecl().getName());
        })
        .Default([](Operation &op) {
          return cast<ASTDeclInterface>(op).getDeclName();
        });
  };

  // We first try to get the name from the operation. Then we try to match the
  // decl with a function argument. Finally, as a last resort, we extract the
  // defining Op from the IR to fetch the name.
  ASTDecl &decl = *unwrapMojoASTDecl(impl);

  if (auto name = getFromOp(decl.getIfOperation()))
    return name;

  if (BlockArgument bbArg = getIfNotOwnedFunctionArgument(*this)) {
    auto func = cast<FuncOp>(*decl.getParentDecl());
    return func.getValueParamNames()[bbArg.getArgNumber()];
  }

  return getFromOp(getDefiningOpFromIR(*this));
}

llvm::SMLoc MojoASTDeclRef::getLoc() const {
  return unwrapMojoASTDecl(impl)->getLoc();
}

MojoASTDeclRef MojoASTDeclRef::getParentDecl() const {
  return MojoASTDeclRef(unwrapMojoASTDecl(impl)->getParentDecl());
}

std::unique_ptr<DeclView> MojoASTDeclRef::getView() const {
  ASTDecl &astDecl = *unwrapMojoASTDecl(impl);

  if (isa<AliasDeclOp>(astDecl))
    return std::unique_ptr<AliasDeclView>(new AliasDeclView(*this));
  if (isa<FileModuleOp>(astDecl))
    return std::unique_ptr<ModuleDeclView>(new ModuleDeclView(*this));
  if (isa<FuncOp>(astDecl))
    return std::unique_ptr<FunctionDeclView>(new FunctionDeclView(*this));
  if (isa<StructDeclOp>(astDecl))
    return std::unique_ptr<StructDeclView>(new StructDeclView(*this));
  if (isa<StructFieldOp>(astDecl))
    return std::unique_ptr<StructFieldDeclView>(new StructFieldDeclView(*this));
  if (isa<GlobalVarDeclOp, LetRegDeclOp, VarLetDeclOp>(astDecl))
    return std::unique_ptr<VariableDeclView>(new VariableDeclView(*this));

  // After failing to match with regular Ops, we then inspect the IR to identify
  // if this decl is an argument.
  if (BlockArgument bbArg = getIfNotOwnedFunctionArgument(*this)) {
    // The parent FunctionDeclView is the one who owns the docstring of this
    // argument, so it's easier just to contruct that view and extract the
    // argument from it.
    MojoASTDeclRef parentDecl = getParentDecl();
    auto functionView = cast<FunctionDeclView>(parentDecl.getView());
    // As the function decl view doesn't store by-ref arguments, we need to
    // adjust the arg index accordingly.
    size_t index = bbArg.getArgNumber();
    auto funcOp = cast<FuncOp>(parentDecl.getIfOperation());
    if (funcOp.getSignature().getInputConvention(0) ==
        KGEN::ValueInputConvention::ByRefResult)
      --index;
    return std::make_unique<ArgumentDeclView>(functionView->getArgs()[index]);
  }
  // FIXME(#17974): Owned arguments are resolved as VarLet decls, and currently
  // it is not possible to recover their original BlockArguments, so we can't
  // generate a proper View for this kind of decl.
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Children

MojoASTDeclRef::ChildEntry MojoASTDeclRef::ChildIterator::operator*() const {
  ASTDecl *decl = unwrapMojoASTDecl(const_cast<void *>(getBase()));
  auto it = std::next(decl->getDeclsInScope().begin(), getIndex());
  ArrayRef<ASTDecl *> decls = it->second;
  ArrayRef<void *> rawDecls(reinterpret_cast<void *const *>(decls.data()),
                            decls.size());
  return ChildEntry(it->first, rawDecls);
}

MojoASTDeclRef::ChildIterator::ChildIterator(MojoASTDeclRef decl, size_t index)
    : llvm::indexed_accessor_iterator<ChildIterator, const void *, ChildEntry,
                                      ChildEntry, ChildEntry>(
          decl.getAsVoidPointer(), index) {}

llvm::iterator_range<MojoASTDeclRef::ChildIterator>
MojoASTDeclRef::getChildren() const {
  ASTDecl *decl = unwrapMojoASTDecl(impl);
  return llvm::make_range(ChildIterator(*this, 0),
                          ChildIterator(*this, decl->getDeclsInScope().size()));
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
