//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_UNITTESTS_MOJO_LSP_SERVER_SUPPORT_H
#define KGEN_UNITTESTS_MOJO_LSP_SERVER_SUPPORT_H

#include "../tools/mojo-lsp-test-client/LSPBatchClient.h"

namespace lsp = mlir::lsp;

namespace M {

/// Create a test client that asserts the execution doesn't fail and also dumps
/// the contents of server IO files upon errors.
LSPBatchClient createTestClient();

} // namespace M

#endif // KGEN_UNITTESTS_MOJO_LSP_SERVER_SUPPORT_H
