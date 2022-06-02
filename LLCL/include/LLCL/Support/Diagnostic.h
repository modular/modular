//===- LLCL/Support/Diagnostic.h ------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Diagnostics are combinations of an error message + location information.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_SUPPORT_DIAGNOSTIC_H
#define LLCL_SUPPORT_DIAGNOSTIC_H

#include "LLCL/Support/Location.h"
#include "Support/Error.h"

namespace LLCL {

/// This is a combination of an `Error` message with an encoded location.  It is
/// relatively efficient to pass around, but its location must be decoded before
/// it can be interpreted.
class EncodedDiagnostic {
public:
  EncodedDiagnostic(M::Error message, EncodedLocation location)
      : message(std::move(message)), location(std::move(location)) {}
  EncodedDiagnostic(EncodedDiagnostic &&) = default;

  /// Access the message in the diagnostic.
  const M::Error &getMessage() const { return message; }
  M::Error &getMessage() { return message; }

  /// Access the location in the diagnostic.
  const EncodedLocation &getLocation() const { return location; }
  EncodedLocation &getLocation() { return location; }

  /// Decode the compressed location into a `DecodedLocation` for rendering.
  DecodedLocation decodeLocation() const { return location.decode(); }

private:
  M::Error message;
  EncodedLocation location;
};

} // namespace LLCL

#endif // LLCL_SUPPORT_DIAGNOSTIC_H
