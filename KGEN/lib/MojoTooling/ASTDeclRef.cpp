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
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/ASTType.h"
#include "KGEN/MojoParser/DLValues.h"
#include "KGEN/MojoTooling/ASTDeclView.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

//===----------------------------------------------------------------------===//
// MojoASTDeclRef
//===----------------------------------------------------------------------===//

/// Return the signature type contained by this decl (e.g. if it's a function),
/// or null otherwise.
static LITSignatureType getSignatureFromDecl(ASTDecl *decl) {
  if (!decl)
    return nullptr;
  if (auto func = dyn_cast<LIT::FuncOp>(*decl))
    return func.getSignature();
  if (auto pValue = decl->getIfIRValue().getIfPValue())
    return dyn_cast_or_null<LITSignatureType>(pValue.getIfTypeValue().mlirType);
  return nullptr;
}

/// Return the index of the argument that corresponds to the given decl.
static std::optional<size_t> getDeclArgIndex(ASTDecl &decl, BlockArgument arg) {
  // If this is a normal argument, we can just return the argument number.
  if (arg.getParentRegion())
    return arg.getArgNumber();
  // Otherwise, we need to inspect the children of the parent decl. The parser
  // uses a shared block for all dangling arguments, so we need to find the
  // correct one manually.
  size_t argIndex = 0;
  for (auto [name, decls] : decl.getParentDecl()->getDeclsInScope()) {
    if (decls.size() != 1)
      continue;

    // Check if the decl is the one we're looking for.
    if (auto cv = decls.front()->getIfIRValue()) {
      // Ignore parameters for our indexing.
      if (decls.front()->getIfIRValue().getIfPValue())
        continue;

      if (cv.getMlirValue() == arg)
        return argIndex;
      ++argIndex;
    }
  }
  return std::nullopt;
}

/// If this decl corresponds to a not owned function argument, return its
/// corresponding BlockArgument. Otherwise, return null.
static BlockArgument getIfNotOwnedFunctionArgument(MojoASTDeclRef declRef) {
  Value val = declRef->getIfIRValue().getMlirValue();
  if (!val)
    return {};

  // Look through rebinds of arguments, which may happen for certain
  // argument conventions.
  if (auto rebind = val.getDefiningOp<RebindOp>())
    val = rebind.getInput();

  // Check if this is a block argument of a function.
  if (auto bbArg = dyn_cast<BlockArgument>(val)) {
    if (isa_and_nonnull<LIT::FuncOp>(bbArg.getOwner()->getParentOp()))
      return bbArg;
    // If this is a block without a proper owner, this is generally a
    // block argument for a function signature. These are detached from
    // normal IR.
    if (!bbArg.getOwner()->getParentOp())
      return bbArg;
  }

  return {};
}

static ParamDeclRefAttr getIfParameter(MojoASTDeclRef declRef) {
  if (auto val = declRef->getIfIRValue().getIfPValue()) {
    if (auto paramRef = dyn_cast<ParamDeclRefAttr>(val.get()))
      return paramRef;
  }
  return {};
}

/// Return the defining Op from the IR encapsulated by this decl. It might be
/// null.
static Operation *getDefiningOpFromIR(MojoASTDeclRef declRef) {
  if (Value val = declRef->getIfIRValue().getMlirValue())
    return val.getDefiningOp();
  return nullptr;
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
        .Case([](LIT::FuncOp op) { return op.getSourceName(); })
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
    LITSignatureType signature = getSignatureFromDecl(decl->getParentDecl());
    if (!signature)
      return std::nullopt;
    std::optional<size_t> argNumber = getDeclArgIndex(*decl, bbArg);
    if (argNumber && *argNumber < signature.getNumArguments())
      return signature.getArgName(*argNumber);
    return std::nullopt;
  }

  if (auto paramRef = getIfParameter(*this))
    return demangleIfNeeded(paramRef).getName().getValue();

  return getFromOp(getDefiningOpFromIR(*this));
}

std::optional<StringRef> MojoASTDeclRef::getDeprecationWarning() const {
  if (auto declItf = dyn_cast<ASTDeclInterface>(decl->getIfOperation()))
    if (StringAttr attr = declItf.getDeprecationWarningAttr())
      return attr.getValue();
  return {};
}

llvm::SMLoc MojoASTDeclRef::getLoc() const { return decl->getLoc(); }

MojoASTDeclRef MojoASTDeclRef::getParentDecl() const {
  return MojoASTDeclRef(decl->getParentDecl());
}

/// Create an Argument decl view for the given decl and argument index.
static std::unique_ptr<ArgumentDeclView>
createArgumentDeclView(MojoASTDeclRef declRef, unsigned arg) {
  // The parent FunctionDeclView is the one who owns the docstring of this
  // argument, so it's easier just to construct that view and extract the
  // argument from it.
  MojoASTDeclRef parentDecl = declRef->getParentDecl();
  auto functionView =
      llvm::unique_dyn_cast_or_null<FunctionDeclView>(parentDecl.getView());
  if (!functionView || functionView->getArguments().size() <= arg)
    return nullptr;
  return std::make_unique<ArgumentDeclView>(functionView->getArguments()[arg]);
}

/// Helper method for `getViewImpl` that either returns a DeclViewKind or a new
/// DeclView instance depending on the ResultType.
template <typename ResultType, typename DeclViewT, typename... DeclArgs>
ResultType MojoASTDeclRef::createDeclView(DeclArgs &&...declArgs) const {
  if constexpr (std::is_same_v<ResultType, ApproximateDeclViewKind>)
    return DeclViewT::getKindStatic();
  else
    return std::unique_ptr<DeclViewT>(
        new DeclViewT(std::forward<DeclArgs>(declArgs)...));
}

/// Common implementation for `getView` and `getApproximateViewKind`.
///
/// If parametrized with `DeclViewInstance`, it will return a DeclView, which
/// can be an expensive operation for entities like function arguments, but it
/// is guaranteed to be correct. This is considered the correct but slow path.
///
/// If parametrized with `ApproximateDeclViewKind`, it will return a
/// `DeclViewKind` by doing only cheap lookups. In general, expensive or
/// unbounded iterations are disallowed in this variant and only executed in
/// the `DeclViewInstance` case. This is considered the approximate but fast
/// path, which is used by interactive tools like the LSP.
template <typename ResultType>
ResultType MojoASTDeclRef::getViewImpl() const {
  static_assert(
      std::is_same_v<ResultType, ApproximateDeclViewKind> ||
          std::is_same_v<ResultType, DeclViewInstance>,
      "Only ApproximateDeclViewKind or DeclViewInstance are valid parameters.");

  constexpr bool isApproximateResult =
      std::is_same_v<ResultType, ApproximateDeclViewKind>;

  if (isa<AliasDeclOp>(*decl))
    return createDeclView<ResultType, AliasDeclView>(*this);

  if (isa<LIT::FuncOp>(*decl))
    return createDeclView<ResultType, FunctionDeclView>(*this);

  // If the decl corresponds to a signature, synthesize a function view for
  // it.
  if (auto signature = getSignatureFromDecl(decl))
    return createDeclView<ResultType, FunctionDeclView>(*this, signature);

  if (isa<FileModuleOp>(*decl))
    return createDeclView<ResultType, ModuleDeclView>(*this);

  if (isa<StructDeclOp>(*decl))
    return createDeclView<ResultType, StructDeclView>(*this);

  if (isa<StructFieldOp>(*decl))
    return createDeclView<ResultType, StructFieldDeclView>(*this);

  if (auto varDecl = dyn_cast<VarDeclOp>(*decl)) {
    // Handle the case of an argument materialized in a variable.
    if (varDecl.getKind() == VarDeclKind::Arg) {
      if constexpr (isApproximateResult) {
        return DeclViewKind::DK_ArgumentDeclView;
      } else {
        auto parentFn = varDecl->getParentOfType<LIT::FuncOp>();
        for (auto [idx, pogAttr] : llvm::enumerate(
                 parentFn.getSignature().getArgListAttrs().getPogs()))
          if (pogAttr.getName() == varDecl.getNameAttr())
            return createArgumentDeclView(*this, idx);
      }
    }
    // Otherwise, this is a regular variable.
    return createDeclView<ResultType, VariableDeclView>(*this);
  }

  if (isa<GlobalVarDeclOp>(*decl))
    return createDeclView<ResultType, VariableDeclView>(*this);

  if (isa<PackageOp>(*decl))
    return createDeclView<ResultType, PackageDeclView>(*this);

  if (isa<TraitDeclOp>(*decl))
    return createDeclView<ResultType, TraitDeclView>(*this);

  // After failing to match with regular Ops, we then inspect the IR to
  // identify if this decl is an argument.
  if (BlockArgument bbArg = getIfNotOwnedFunctionArgument(*this)) {
    if constexpr (isApproximateResult) {
      return DeclViewKind::DK_ArgumentDeclView;
    } else {
      if (std::optional<size_t> argIdx = getDeclArgIndex(*decl, bbArg))
        return createArgumentDeclView(*this, *argIdx);
      return nullptr;
    }
  }

  // Handle def argument shadows, the parser produces these as
  // DefArgumentWrapperDLValue so we need to dig through them to find the
  // underlying BlockArgument for the function.
  if (auto lvalue = decl->getIfIRValue().getIfLValue()) {
    // Unresolved to mutable.
    if (auto dlvalue = lvalue.getIfDLValue()) {
      if (dlvalue->isDefArgument()) {
        auto &defArgDLVal = ((DefArgumentWrapperDLValue &)*dlvalue);
        if constexpr (isApproximateResult)
          return DeclViewKind::DK_ArgumentDeclView;
        else
          return createArgumentDeclView(*this, defArgDLVal.argIndex);
      }
    }
    // Resolved to mutable.
    if (auto mlValue = lvalue.getIfMLValue()) {
      if (auto var = mlValue.getDefiningOp<VarDeclOp>()) {
        if (var.getArgShadowIndex().has_value()) {
          if constexpr (isApproximateResult)
            return DeclViewKind::DK_ArgumentDeclView;
          else
            return createArgumentDeclView(*this,
                                          var.getArgShadowIndex().value());
        }
      }
    }
  }

  // Now we inspect the IR checking for a parameter.
  if (ParamDeclRefAttr param = getIfParameter(*this)) {
    if constexpr (isApproximateResult) {
      return DeclViewKind::DK_ParameterDeclView;
    } else {
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
      std::unique_ptr<DeclView> parentView = getParentDecl().getView();
      if (!parentView)
        return nullptr;
      return TypeSwitch<DeclView *, std::unique_ptr<DeclView>>(&*parentView)
          .Case<FunctionDeclView, StructDeclView>(getParamViewFromParent)
          .Default({nullptr});
    }
  }

  return {};
}

std::unique_ptr<DeclView> MojoASTDeclRef::getView() const {
  return getViewImpl<DeclViewInstance>();
}

std::optional<DeclViewKind> MojoASTDeclRef::getApproximateViewKind() const {
  return getViewImpl<ApproximateDeclViewKind>();
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
