//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the implementation of the LitSharedState class.
//
//===----------------------------------------------------------------------===//

#include "LitSharedState.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/LITDialect/LITOps.h"
#include "LitASTDecl.h"
#include "LitDecls.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/SourceMgr.h"
#include <set>

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

using llvm::SMLoc;
using llvm::SourceMgr;

struct AttributeVectorComparison {
  bool operator()(const std::vector<ParamBindAttr> &lhs,
                  const std::vector<ParamBindAttr> &rhs) const {
    if (lhs.size() != rhs.size())
      return lhs.size() < rhs.size();
    for (size_t i = 0, e = lhs.size(); i != e; ++i) {
      if (lhs[i] != rhs[i])
        return lhs[i].getAsOpaquePointer() < rhs[i].getAsOpaquePointer();
    }
    return false;
  }
};

class LitSharedState::Impl {
public:
  DenseMap<std::pair<ASTDecl *, const ParamBindAttr *>, ASTTypeStorage *>
      uniquedASTTypes;

  // TODO(complile time): This is horribly inefficient.
  // Switch to StorageUniquer or something else?
  std::set<std::vector<ParamBindAttr>, AttributeVectorComparison> uniquedParams;
};

/// Get the name of the main buffer so we can rapidly build Location objects
/// on demand.
static StringAttr getMainBufferNameIdentifier(const SourceMgr &sourceMgr,
                                              MLIRContext *context) {
  auto mainBuffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  StringRef bufferName = mainBuffer->getBufferIdentifier();
  if (bufferName.empty())
    bufferName = "<unknown>";
  return StringAttr::get(context, bufferName);
}

LitSharedState::LitSharedState(llvm::SourceMgr &sourceMgr, MLIRContext *context)
    : sourceMgr(sourceMgr), context(context),
      declResolver(std::make_unique<DeclResolver>(*this)),
      bufferNameIdentifier(getMainBufferNameIdentifier(sourceMgr, context)),
      impl(std::make_unique<Impl>()) {}

LitSharedState::~LitSharedState() { declResolver.reset(); }

/// Encode the specified source location information into a Location object
/// for attachment to the IR or error reporting.
Location LitSharedState::translateLocation(SMLoc loc) {
  unsigned mainFileID = sourceMgr.getMainFileID();
  auto lineAndColumn = sourceMgr.getLineAndColumn(loc, mainFileID);
  return FileLineColLoc::get(bufferNameIdentifier, lineAndColumn.first,
                             lineAndColumn.second);
}

ASTType LitSharedState::getTypeCheckErrorType() const {
  return typeCheckErrorTypeDecl->getResolvedType();
}

ASTType LitSharedState::getIndexType() const {
  return indexDecl->getResolvedType();
}

ASTType LitSharedState::getNoneType() const {
  return noneDecl->getResolvedType();
}

// FIXME: This isn't correctly parameterized.
ASTType LitSharedState::getPointerType() const {
  return pointerDecl->getResolvedType();
}

ASTType LitSharedState::getObjectType() const {
  return objectDecl->getResolvedType();
}

// FIXME: This isn't correctly parameterized; we need variadics.
ASTType LitSharedState::getSignatureType() const {
  return signatureDecl->getResolvedType();
}

//===----------------------------------------------------------------------===//
// ASTType
//===----------------------------------------------------------------------===//

/// Convert this type to a human readable string representation so it can be
/// printed out for diagnostics.
std::string ASTType::getAsString() const {
  if (!pointer || !getDecl())
    return "<<NULL ASTTYPE>>";

  std::string result;
  llvm::raw_string_ostream os(result);
  os << "'";

  if (auto typeDecl = dyn_cast<LITStructDeclOp>(*getDecl())) {
    // TODO: Could include name scope information.
    os << typeDecl.getName();
  } else if (getDecl()->isMagic()) {
    switch (getDecl()->magicKind) {
    case MagicDeclKind::kNormal:
      llvm_unreachable("not a magic declaration?");
    case MagicDeclKind::kIndexType:
      os << "!builtin.index";
      break;
    case MagicDeclKind::kNoneType:
      os << "!lit.none";
      break;
    case MagicDeclKind::kTypeCheckErrorType:
      os << "<<TypeCheckError>>";
      break;
    case MagicDeclKind::kPointerType:
      os << "<<Pointer>>";
      break;
    case MagicDeclKind::kSignatureType:
      os << "<<FnSignature>>";
      break;
    }
  } else {
    // TODO: Add "aka" information when we have "type defs".
    os << "<<unknown ASTType>>";
  }

  ArrayRef<ParamBindAttr> params = getParamValues();
  if (!params.empty()) {
    os << '[';
    llvm::interleaveComma(params, os, [&](ParamBindAttr bind) {
      // TODO: This isn't really right, but will work enough for now.
      printParamValue(bind.getValue(), os);
    });
    os << ']';
  }

  os << '\'';
  return os.str();
}

mlir::Diagnostic &M::KGEN::LIT::operator<<(mlir::Diagnostic &diag,
                                           ASTType type) {
  return diag << type.getAsString();
}

/// Print to standard error with newline after it, for use in a debugger.
void ASTType::dump() const { llvm::errs() << getAsString() << '\n'; }

ArrayRef<ParamBindAttr>
LitSharedState::getUniquedParams(ArrayRef<ParamBindAttr> params) {
  // std::set produces stable pointers.
  return *impl->uniquedParams.insert(params.vec()).first;
}

/// Get a uniqued and pointer sized reference to an ASTType.
ASTType LitSharedState::getASTType(ASTDecl *decl,
                                   ArrayRef<ParamBindAttr> params) {
  if (!decl)
    return ASTType();
  params = getUniquedParams(params);
  auto &entry = impl->uniquedASTTypes[{decl, params.data()}];
  if (entry)
    return ASTType(entry);

  // Ok, the entry hasn't been established, make it now.
  return entry = allocPersistent<ASTTypeStorage>(*decl, params);
}
