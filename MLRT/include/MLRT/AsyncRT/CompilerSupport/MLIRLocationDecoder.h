//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_ASYNCRT_SUPPORT_MLIRLOCATIONDECODER_H
#define MLRT_ASYNCRT_SUPPORT_MLIRLOCATIONDECODER_H

#include "MLRT/AsyncRT/Support/Diagnostic.h"
#include "MLRT/AsyncRT/Support/Location.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/ReferenceCounted.h"
#include <string>

namespace M::AsyncRT {

/// This class implements LocationDecoder and reports the MLIR location as
/// file/line/column when possible, otherwise just reports the printed location.
class MLIRLocationDecoder final : public ReferenceCounted<MLIRLocationDecoder>,
                                  public LocationDecoder {
public:
  MLIRLocationDecoder() = default;

  static EncodedLocation getEncodedLocation(mlir::Location loc);

  /// Implement the LocationDecoder hooks - the EncodedLocation contains a
  /// pointer that can be decoded with the context into a full mlir::Location.
  DecodedLocation decode(const EncodedLocation &loc) const override;
  void addRef() const override;
  void dropRef() const override;
};

/// Given an Error and an mlir::Location, we can create an EncodedDiagnostic.
EncodedDiagnostic getMLIRDiagnostic(Error e, mlir::Location loc);

} // namespace M::AsyncRT

#endif // MLRT_ASYNCRT_SUPPORT_MLIRLOCATIONDECODER_H
