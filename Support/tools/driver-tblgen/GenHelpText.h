//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TOOLS_DRIVERTBLGEN_GENHELPTEXT_H
#define SUPPORT_TOOLS_DRIVERTBLGEN_GENHELPTEXT_H

namespace M {

class BackendRegistry;

/// Registers the "gen-help-text" backend.
void registerGenHelpTextBackend(BackendRegistry &registry);
} // namespace M

#endif // SUPPORT_TOOLS_DRIVERTBLGEN_GENHELPTEXT_H
