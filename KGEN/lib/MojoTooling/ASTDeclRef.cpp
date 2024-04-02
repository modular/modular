//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the main entrypoints for the Mojo parser.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoTooling/ASTDeclRef.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ASTType.h"
#include "KGEN/MojoTooling/ASTDeclView.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// MojoASTDeclRef
//===----------------------------------------------------------------------===//

/// If this decl corresponds to a not owned function argument, return its
/// corresponding BlockArgument. Otherwise, return null.
static BlockArgument getIfNotOwnedFunctionArgument(MojoASTDeclRef declRef) {
  return TypeSwitch<DeclIRValue, BlockArgument>(declRef->getIRValue())
      .Case<SBValue, SRValue, MBValue, MLValue>([&](Value val) {
        // Look through rebinds of arguments, which may happen for certain
        // argument conventions.
        if (auto rebind = val.getDefiningOp<RebindOp>())
          val = rebind.getInput();

        if (auto bbArg = dyn_cast<BlockArgument>(Value(val)))
          if (isa<LIT::FuncOp>(bbArg.getOwner()->getParentOp()))
            return bbArg;
        return BlockArgument();
      })
      .Default({});
}

static ParamDeclRefAttr getIfParameter(MojoASTDeclRef declRef) {
  if (auto val = dyn_cast_if_present<PValue>(declRef->getIRValue())) {
    if (auto paramRef = dyn_cast<ParamDeclRefAttr>(val.get()))
      return paramRef;
  }
  return {};
}

/// Return the defining Op from the IR encapsulated by this decl. It might be
/// null.
static Operation *getDefiningOpFromIR(MojoASTDeclRef declRef) {
  return TypeSwitch<DeclIRValue, Operation *>(declRef->getIRValue())
      .Case<SBValue, SRValue, MBValue, MLValue>(
          [&](auto val) -> Operation * { return Value(val).getDefiningOp(); })
      .Default((Operation *)nullptr);
}

Operation *MojoASTDeclRef::getIfOperation() const {
  return decl->getIfOperation();
}

MojoASTTypeRef MojoASTDeclRef::getType() const {
  return TypeSwitch<ASTDecl &, MojoASTTypeRef>(*decl)
      .Case<GlobalVarDeclOp, VarDeclOp>(
          [&](auto op) { return MojoASTTypeRef(op.getType()); })
      .Case([&](LIT::FuncOp op) { return op.getFullSignature(); })
      .Case([&](StructDeclOp op) { return decl->computeSelfTypeForStruct(op); })
      .Case([&](TraitDeclOp op) { return decl->computeSelfTypeForTrait(op); })
      .Default({});
}

std::optional<StringRef> MojoASTDeclRef::getName() const {
  auto getFromOp = [&](Operation *op) -> std::optional<StringRef> {
    if (!op)
      return std::nullopt;
    return TypeSwitch<Operation &, std::optional<StringRef>>(*op)
        .Case<GlobalVarDeclOp, StructDeclOp, StructFieldOp, VarDeclOp>(
            [](auto op) { return op.getName(); })
        .Case([&](LIT::FuncOp op) { return op.getSourceName(); })
        .Case<FileModuleOp, PackageOp>([](auto op) { return op.getSymName(); })
        .Case([](AliasDeclOp op) {
          return demangleParameterName(op.getParamDecl().getName());
        })
        .Case([](ASTDeclInterface op) { return op.getDeclName(); })
        .Default({});
  };

  // We first try to get the name from the operation. Then we try to match the
  // decl with a function argument. Finally, as a last resort, we extract the
  // defining Op from the IR to fetch the name.
  if (auto name = getFromOp(decl->getIfOperation()))
    return name;

  if (BlockArgument bbArg = getIfNotOwnedFunctionArgument(*this)) {
    auto func = cast<LIT::FuncOp>(*decl->getParentDecl());
    ArrayRef<StringAttr> argNames = func.getSignature().getArgNames();
    if (size_t argNumber = bbArg.getArgNumber(); argNumber < argNames.size())
      return argNames[argNumber];
    return std::nullopt;
  }

  if (auto paramRef = getIfParameter(*this))
    return demangleIfNeeded(paramRef).getName().getValue();

  return getFromOp(getDefiningOpFromIR(*this));
}

llvm::SMLoc MojoASTDeclRef::getLoc() const { return decl->getLoc(); }

MojoASTDeclRef MojoASTDeclRef::getParentDecl() const {
  return MojoASTDeclRef(decl->getParentDecl());
}

/// Create an Argument decl view for the given decl and argument index.
std::unique_ptr<ArgumentDeclView> createArgumentDeclView(MojoASTDeclRef declRef,
                                                         unsigned arg) {
  // The parent FunctionDeclView is the one who owns the docstring of this
  // argument, so it's easier just to contruct that view and extract the
  // argument from it.
  MojoASTDeclRef parentDecl = declRef.getParentDecl();
  auto functionView =
      llvm::unique_dyn_cast_or_null<FunctionDeclView>(parentDecl.getView());
  if (!functionView)
    return nullptr;
  return std::make_unique<ArgumentDeclView>(functionView->getArguments()[arg]);
}

std::unique_ptr<DeclView> MojoASTDeclRef::getView() const {
  if (isa<AliasDeclOp>(*decl))
    return std::unique_ptr<AliasDeclView>(new AliasDeclView(*this));
  if (isa<FileModuleOp>(*decl))
    return std::unique_ptr<ModuleDeclView>(new ModuleDeclView(*this));
  if (isa<LIT::FuncOp>(*decl))
    return std::unique_ptr<FunctionDeclView>(new FunctionDeclView(*this));
  if (isa<StructDeclOp>(*decl))
    return std::unique_ptr<StructDeclView>(new StructDeclView(*this));
  if (isa<StructFieldOp>(*decl))
    return std::unique_ptr<StructFieldDeclView>(new StructFieldDeclView(*this));
  if (auto varDecl = dyn_cast<VarDeclOp>(*decl)) {
    // Handle the case of an argument materialized in a variable.
    if (varDecl.getKind() == VarDeclKind::Arg) {
      auto parentFn = varDecl->getParentOfType<LIT::FuncOp>();
      auto argNames = parentFn.getSignature().getArgNames();
      auto argIt = llvm::find(argNames, varDecl.getNameAttr());
      if (argIt != argNames.end())
        return createArgumentDeclView(*this, argIt - argNames.begin());
    }
    // Otherwise, this is a regular variable.
    return std::unique_ptr<VariableDeclView>(new VariableDeclView(*this));
  }
  if (isa<GlobalVarDeclOp>(*decl))
    return std::unique_ptr<VariableDeclView>(new VariableDeclView(*this));
  if (isa<PackageOp>(*decl))
    return std::unique_ptr<PackageDeclView>(new PackageDeclView(*this));
  if (isa<TraitDeclOp>(*decl))
    return std::unique_ptr<TraitDeclView>(new TraitDeclView(*this));

  // After failing to match with regular Ops, we then inspect the IR to
  // identify if this decl is an argument.
  if (BlockArgument bbArg = getIfNotOwnedFunctionArgument(*this))
    return createArgumentDeclView(*this, bbArg.getArgNumber());

  // Now we inspect the IR checking for a parameter.
  if (ParamDeclRefAttr param = getIfParameter(*this)) {
    auto name = demangleIfNeeded(param).getName().getValue();
    // The parent FunctionDeclView or StructDeclView is the one who owns the
    // docstring of this parameter, so it's easier to construct that view and
    // extract the parameter from it.
    auto getParamViewFromParent =
        [&](auto &parentView) -> std::unique_ptr<DeclView> {
      for (const ParameterDeclView &paramView : parentView->parameters)
        if (paramView.getName() == name)
          return std::make_unique<ParameterDeclView>(paramView);
      return nullptr;
    };

    return TypeSwitch<DeclView *, std::unique_ptr<DeclView>>(
               getParentDecl().getView().get())
        .Case<FunctionDeclView, StructDeclView>(getParamViewFromParent)
        .Default({nullptr});
  }

  return nullptr;
}

std::optional<DeclViewKind> MojoASTDeclRef::getApproximateViewKind() const {
  if (isa<AliasDeclOp>(*decl))
    return DeclViewKind::DK_AliasDeclView;
  if (isa<FileModuleOp>(*decl))
    return DeclViewKind::DK_ModuleDeclView;
  if (isa<LIT::FuncOp>(*decl))
    return DeclViewKind::DK_FunctionDeclView;
  if (isa<StructDeclOp>(*decl))
    return DeclViewKind::DK_StructDeclView;
  if (isa<StructFieldOp>(*decl))
    return DeclViewKind::DK_StructFieldDeclView;
  if (auto varDecl = dyn_cast<VarDeclOp>(*decl)) {
    if (varDecl.getKind() == VarDeclKind::Arg)
      return DeclViewKind::DK_ArgumentDeclView;
    return DeclViewKind::DK_VariableDeclView;
  }
  if (isa<GlobalVarDeclOp>(*decl))
    return DeclViewKind::DK_VariableDeclView;
  if (isa<PackageOp>(*decl))
    return DeclViewKind::DK_PackageDeclView;
  if (isa<TraitDeclOp>(*decl))
    return DeclViewKind::DK_TraitDeclView;

  // After failing to match with regular Ops, we then inspect the IR to identify
  // if this decl is an argument.
  if (getIfNotOwnedFunctionArgument(*this))
    return DeclViewKind::DK_ArgumentDeclView;

  // Now we inspect the IR checking for a parameter.
  if (getIfParameter(*this))
    return DeclViewKind::DK_ParameterDeclView;
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Children

MojoASTDeclRef::ChildEntry MojoASTDeclRef::ChildIterator::operator*() const {
  auto it = std::next(getBase()->getDeclsInScope().begin(), getIndex());
  return ChildEntry(it->first, it->second);
}

MojoASTDeclRef::ChildIterator::ChildIterator(MojoASTDeclRef decl, size_t index)
    : llvm::indexed_accessor_iterator<ChildIterator, ASTDecl *, ChildEntry,
                                      ChildEntry, ChildEntry>(decl.decl,
                                                              index) {}

llvm::iterator_range<MojoASTDeclRef::ChildIterator>
MojoASTDeclRef::getChildren() const {
  return llvm::make_range(ChildIterator(*this, 0),
                          ChildIterator(*this, decl->getDeclsInScope().size()));
}

//===----------------------------------------------------------------------===//
// MojoASTTypeRef
//===----------------------------------------------------------------------===//

MojoASTDeclRef MojoASTTypeRef::getDecl(SharedState &sharedState) {
  return MojoASTDeclRef(type.getDecl(sharedState));
}

std::string MojoASTTypeRef::getAsString() const {
  return type.getAsString(/*forDiag=*/true);
}

/// If the current type is a reference, return the type of the pointee. This
/// aborts if the current type isn't a reference.
MojoASTTypeRef MojoASTTypeRef::getReferenceElementType() const {
  return type.getReferenceElementType();
}

Type MojoASTTypeRef::getMLIRType() const { return type.mlirType; }
