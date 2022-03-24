//===- LLCL/Runtime/Diagnostic.h ------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Diagnostics are combinations of an error message + location information.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_RUNTIME_DIAGNOSTIC_H
#define LLCL_RUNTIME_DIAGNOSTIC_H

#include "LLCL/Runtime/Location.h"
#include "Support/Error.h"

namespace LLCL {

/// This is a combination of an `Error` message with an encoded location.  It is
/// relatively efficient to pass around, but its location must be decoded before
/// it can be interpreted.
class EncodedDiagnostic {
public:
  M::Error message;
  EncodedLocation location;

  EncodedDiagnostic(M::Error message, EncodedLocation location)
      : message(std::move(message)), location(std::move(location)) {}
  EncodedDiagnostic(EncodedDiagnostic &&) = default;

  Runtime &getRuntime() const { return location.getRuntime(); }

  DecodedLocation decodeLocation() const { return location.decode(); }
};

} // namespace LLCL

#endif // LLCL_RUNTIME_DIAGNOSTIC_H
