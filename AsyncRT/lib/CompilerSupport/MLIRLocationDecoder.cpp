//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/CompilerSupport/MLIRLocationDecoder.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"

using namespace M;
using namespace LLCL;

EncodedLocation MLIRLocationDecoder::getEncodedLocation(mlir::Location loc) {
  return {(intptr_t)loc.getAsOpaquePointer(),
          RCRef<MLIRLocationDecoder>::create()};
}

DecodedLocation MLIRLocationDecoder::decode(const EncodedLocation &loc) const {
  auto mlirLoc = mlir::Location::getFromOpaquePointer((void *)loc.getData());
  if (auto fileLineColLoc = dyn_cast<mlir::FileLineColLoc>(mlirLoc))
    return DecodedLocation{fileLineColLoc.getFilename().str(),
                           (int)fileLineColLoc.getLine(),
                           (int)fileLineColLoc.getColumn()};

  std::string locStr;
  llvm::raw_string_ostream stream(locStr);
  stream << mlirLoc;
  return DecodedLocation{stream.str()};
}

/// Implement the LocationDecoder hook for addRef.
void MLIRLocationDecoder::addRef() const {
  RCRef<ReferenceCounted<MLIRLocationDecoder>>::lowLevelAddRef(
      const_cast<MLIRLocationDecoder *>(this));
}

/// Implement the LocationDecoder hook for dropRef.
void MLIRLocationDecoder::dropRef() const {
  RCRef<ReferenceCounted<MLIRLocationDecoder>>::lowLevelDropRef(
      const_cast<MLIRLocationDecoder *>(this));
}

EncodedDiagnostic LLCL::getMLIRDiagnostic(Error e, mlir::Location loc) {
  return {std::move(e), MLIRLocationDecoder::getEncodedLocation(loc)};
}
