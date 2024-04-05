//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOBUILD_BSPSERVER_H
#define KGEN_MOJOBUILD_BSPSERVER_H

#include "KGEN/MojoBuild/Protocol.h"
#include "mlir/Tools/lsp-server-support/Transport.h"

namespace M::Build {

/// Implements generic requests and responses for a build server, loosely
/// following the specification defined here:
/// https://build-server-protocol.github.io/docs/specification
///
/// This class does not (and should not) implement logic specifically related to
/// Mojo and building Mojo projects.
class BSPServer {
public:
  /// Handles the `build/initialize` request.
  void onBuildInitialize(const InitializeBuildParams &params,
                         mlir::lsp::Callback<InitializeBuildResult> callback);
};
} // namespace M::Build

#endif
