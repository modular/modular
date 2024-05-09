//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines the API for interacting with the Mojo LLDB plugin.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOLLDB_PLUGIN_H
#define KGEN_MOJOLLDB_PLUGIN_H

#include "Support/Context.h"
#include "Support/SymbolExport.h"

namespace M::KGEN {

/// Set the context to use inside the LLDB plugin.  This should be set before
/// the LLDB plugin initializes.
MODULAR_VISIBILITY_EXPORT void setLLDBPluginContext(ContextRef ctx);

} // namespace M::KGEN

#endif // KGEN_MOJOLLDB_PLUGIN_H
